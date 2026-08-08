"""
set_reid_model.py

Arquitectura de matching set-to-set para re-identificar una detección nueva
("query", DOS vistas del mismo objeto -- tight + wide, ver más abajo)
contra una galería de tracks perdidos (cada uno con hasta n vistas
históricas). Reemplaza a TrackQueryReIDModule (reid_track_query_module.py)
-- ataca directo el desbalance match/no-match que se observó ahí (dominado
por queries de relleno) y el pooling temprano del historial de cada track
(que puede estar favoreciendo memorización -- ver history_jitter_ratio en
reid_dataloader.py).

Diferencias de diseño respecto a TrackQueryReIDModule:
  - La query es un pequeño SET de 2 vistas del mismo objeto (recorte
    ajustado + recorte amplio, ver _load_query_views en reid_dataloader.py)
    en vez de una fusión propia con "memoria" -- los daños de este dataset
    no son suficientemente discriminantes por apariencia local sola, así
    que se le da contexto de pieza vía una segunda vista, procesada con el
    MISMO principio que el historial del banco: no colapsar antes de
    comparar (ver TrackReIDModel).
  - Las vistas del historial de cada track candidato (y las de la query)
    NO se funden en un solo vector antes de la cross-attention real -- se
    mantienen separadas hasta ese punto, para no perder granularidad
    por-vista. Solo se poolean (vía attention pooling aprendido) para un
    `track_summary` pensado para indexado barato, no para la decisión final.
  - El match/no-match y la identidad del track se resuelven con un SOLO
    softmax de (k+1) clases (k candidatos + "no match"), entrenado con una
    sola cross-entropy -- no dos losses separadas con pesos a mano.

Piezas del archivo:
  - AttentionPool: el mecanismo de pooling por atención (Pooling by
    Multihead Attention), reusado en dos puntos distintos del pipeline
    (resumen del banco, resumen de la query post-cross-attention) --
    mismo código, instancias/pesos independientes en cada uso.
  - IntraTrackEncoder / CrossAttentionMatcher / MatchScorer / TrackReIDModel:
    la arquitectura pura, trabaja sobre embeddings ya calculados (ver
    `if __name__ == "__main__"` para un ejemplo con tensores dummy, sin
    encoder ni dataloader real).
  - SetTrackReIDModule: wrapper end-to-end que agrega el encoder visual
    (DINOv2Encoder, reusado de reid_track_query_module.py) -- toma píxeles
    crudos (o embeddings ya cacheados, ver PrecomputedEmbeddingCache en
    reid_dataloader.py) y devuelve los logits (k+1). Es lo que conecta con
    ReIDEpisodeDataset (reid_dataloader.py) para entrenar de verdad -- ver
    train_track_query_reid.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid_track_query_module import DINOv2Encoder


class AttentionPool(nn.Module):
    """
    Pooling by Multihead Attention (PMA, Set Transformer -- Lee et al.
    2019): un query aprendible atiende sobre un conjunto de vistas y las
    resume en UN vector -- no mean/max pooling, para poder pesar más las
    vistas más "representativas" en vez de promediarlas todas por igual.
    Implementado a mano (no nn.MultiheadAttention) para poder sumar un bias
    de calidad directamente sobre los scores, antes del softmax.

    Reusado en dos puntos del pipeline con el MISMO mecanismo pero
    instancias/pesos independientes (son tareas de pooling distintas, ver
    TrackReIDModel):
      - resumir las n vistas del historial de cada track candidato en un
        `track_summary` (ver IntraTrackEncoder), ANTES de la cross-attention.
      - resumir las 2 vistas de la query (ya refinadas por la
        cross-attention contra la galería) en un solo `q_new`, DESPUÉS de
        la cross-attention.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = embed_dim ** -0.5

    def forward(
        self, items: torch.Tensor, items_mask: torch.Tensor = None, extra_score_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        items: (M, n, D) -- M = elementos aplanados (ej. B*k para el banco, B para la query)
        items_mask: opcional, (M, n) bool, True = item válido. None = todos
            válidos (ej. las 2 vistas de la query, siempre ambas reales).
        extra_score_bias: opcional, (M, n) -- se suma al score de atención
            antes del softmax (ver quality_score en IntraTrackEncoder).

        Returns:
            pooled: (M, D)
        """
        M, n, D = items.shape
        if items_mask is None:
            items_mask = torch.ones(M, n, dtype=torch.bool, device=items.device)

        # filas sin NINGÚN item válido (ej. slot de banco con padding): un
        # key_padding_mask 100% True puede devolver NaN de la atención --
        # se descarta después vía bank_mask en MatchScorer, pero no debe
        # contaminar el resto del batch.
        safe_mask = items_mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True

        query = self.pool_query.expand(M, 1, D)
        keys = self.key_proj(items)
        values = self.value_proj(items)
        scores = torch.bmm(query, keys.transpose(1, 2)) * self.scale  # (M, 1, n)
        scores = scores.masked_fill(~safe_mask.unsqueeze(1), float("-inf"))
        if extra_score_bias is not None:
            scores = scores + extra_score_bias.unsqueeze(1).masked_fill(~safe_mask.unsqueeze(1), 0.0)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        pooled = torch.bmm(weights, values).squeeze(1)  # (M, D)

        # residual con la media (solo items válidos) como base estable,
        # mismo criterio que MultiScaleCropFusion/TrackHistoryFusion en
        # reid_track_query_module.py.
        mean_valid = (items * safe_mask.unsqueeze(-1)).sum(dim=1) / safe_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.norm(pooled + mean_valid)


class IntraTrackEncoder(nn.Module):
    """
    Paso 1: contextualiza las n vistas de cada track candidato ENTRE SÍ
    (self-attention, permutation-invariant -- no asume orden temporal
    fuerte entre vistas) y además resume el track en un solo vector vía
    AttentionPool.

    `views_contextualized` (salida SIN poolear) es lo que alimenta después
    a CrossAttentionMatcher -- ahí es donde realmente importa no perder
    granularidad por vista. `track_summary` (la salida pooleada) es solo
    para indexado barato (filtrar candidatos por similitud coseno cuando la
    galería es grande) -- no participa de la decisión de matching final.

    Reusado también para contextualizar las 2 vistas de la query entre sí
    (ver TrackReIDModel) -- tratándola como "un track de 1 candidato x 2
    vistas" -- misma instancia/pesos, sin cambios de interfaz: el módulo ya
    soporta número variable de vistas con masking.

    `views_mask`: True donde la vista es real (no padding) -- tracks con
    menos de n vistas válidas se manejan con attention masking, no con
    padding silencioso (un vector de ceros que igual entraría a la
    atención como si fuera una vista real).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPool(embed_dim, dropout=dropout)

    def forward(
        self, views: torch.Tensor, views_mask: torch.Tensor, quality_score: torch.Tensor = None,
    ):
        """
        views: (B, k, n, D)
        views_mask: (B, k, n) bool, True = vista válida
        quality_score: opcional, (B, k, n) -- score de calidad/confianza
            (blur, oclusión, confidence de detección) por vista. Si se
            pasa, se SUMA como bias aditivo al score de atención del
            pooling (más calidad -> más peso) -- no reemplaza la atención
            aprendida, la modula.

        Returns:
            views_contextualized: (B, k, n, D)
            track_summary: (B, k, D)
        """
        B, k, n, D = views.shape
        flat_views = views.reshape(B * k, n, D)
        flat_mask = views_mask.reshape(B * k, n)

        safe_mask = flat_mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True
        contextualized = self.self_attn(flat_views, src_key_padding_mask=~safe_mask)

        flat_quality = quality_score.reshape(B * k, n) if quality_score is not None else None
        track_summary = self.pool(contextualized, flat_mask, extra_score_bias=flat_quality)

        return contextualized.reshape(B, k, n, D), track_summary.reshape(B, k, D)


class CrossAttentionMatcher(nn.Module):
    """
    Paso 2: cross-attention entre la detección nueva y la galería.
    DIRECCIÓN FIJA, no invertir:
        Q  = las L vistas de la detección nueva, ya contextualizadas entre
             sí (L=2: tight + wide, ver TrackReIDModel) -- lo que estamos
             tratando de identificar.
        K,V = views_contextualized de la galería (el conocimiento
             acumulado contra el que comparamos), aplanando (k, n) en una
             sola dimensión de memoria (k*n).

    Por qué esta dirección y no la inversa (Q=galería, K,V=detección
    nueva): con la detección nueva como K,V (longitud L=2, chica y fija),
    la atención no tendría casi nada real entre qué elegir -- el softmax
    sobre pocos elementos fijos no discrimina bien (mismo problema que se
    discutió para MemoryFusion en reid_track_query_module.py). Con la
    galería como K,V (longitud k*n, variable y grande) sí hay una elección
    real: sobre qué vista de qué track candidato debe apoyarse cada vista
    de la detección nueva.

    Los pesos de atención devueltos son SOLO para logging/debug/
    interpretabilidad (ver a qué vistas de qué track atendió más cada
    vista de la query) -- la decisión de matching real la toma MatchScorer
    con un scorer aprendido sobre `track_summary`, NO la masa de atención
    agregada por track acá. Un umbral sobre esto no tendría señal de
    gradiente para los casos sin match real (ningún término de loss lo
    tocaría), y el nivel "de base" del máximo depende de cuántas vistas
    válidas haya (k*n varía por episodio) -- no es comparable entre casos
    con un corte fijo.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, query_views: torch.Tensor, views_contextualized: torch.Tensor, views_mask: torch.Tensor,
    ):
        """
        query_views: (B, L, D) -- las L vistas de la detección nueva (L=2:
            tight + wide), ya contextualizadas entre sí por IntraTrackEncoder
        views_contextualized: (B, k, n, D) -- salida de IntraTrackEncoder
        views_mask: (B, k, n) bool, True = vista válida

        Returns:
            q_new: (B, L, D) -- cada vista de la query, refinada con
                contexto de la galería (se poolea a 1 vector aparte, ver
                TrackReIDModel)
            attn_weights: (B, L, k, n) -- SOLO debug/interpretabilidad
        """
        B, k, n, D = views_contextualized.shape
        L = query_views.shape[1]
        memory = views_contextualized.reshape(B, k * n, D)
        memory_mask = views_mask.reshape(B, k * n)

        # mismo guard de fila-100%-inválida que en IntraTrackEncoder --
        # acá aplicaría solo si TODO el banco está vacío en este episodio,
        # caso degenerado pero igual conviene no romper con NaN.
        safe_mask = memory_mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True

        attn_out, attn_weights = self.attn(
            query=query_views, key=memory, value=memory,
            key_padding_mask=~safe_mask,
            need_weights=True, average_attn_weights=True,
        )
        q_new = self.norm(attn_out + query_views)
        return q_new, attn_weights.reshape(B, L, k, n)


class MatchScorer(nn.Module):
    """
    Paso 3: la decisión de matching real (no la atención). Scorer bilineal
    + MLP aprendido (no dot product crudo -- un dot product asume que
    `q_new` y `track_summary` ya viven en el mismo espacio geométrico sin
    ninguna transformación, que no hay por qué esperar) sobre
    [q_new, track_summary_i] por cada track candidato i, más una columna
    aprendida "no match / objeto nuevo" -- sin esta clase el sistema
    fuerza siempre una asignación a algún candidato, aunque ninguno sea en
    realidad el correcto.

    Entrenar con cross-entropy estándar sobre el índice del candidato
    correcto -- no con triplet loss sobre embeddings sueltos: eso optimiza
    distancias relativas entre pares, esto optimiza directamente la
    decisión de k clases que de verdad importa.

    El candidato "new" (objeto genuinamente nuevo, sin match en el banco)
    NO tiene columna propia acá -- es un candidato más dentro de
    `track_summary`, agregado por TrackReIDModel vía un embedding
    aprendible que participa en el mismo cross-attention y en el mismo
    bilinear que los candidatos reales (ver TrackReIDModel._append_new_slot).
    Antes existía un `no_match_logit` como parámetro libre desconectado de
    la query -- el cross-entropy podía satisfacer los casos NEW_TRACK
    ajustando solo ese escalar, sin que el gradiente pasara por el
    bilinear/atención, diluyendo la señal de identidad en esa fracción del
    batch. Con "new" como candidato homogéneo, un solo mecanismo aprende
    ambas decisiones.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, q_new: torch.Tensor, track_summary: torch.Tensor, bank_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        q_new: (B, 1, D)
        track_summary: (B, k, D) -- incluye el candidato "new" (ver
            TrackReIDModel._append_new_slot), no hace falta tratarlo aparte.
        bank_mask: opcional, (B, k) bool -- True = candidato válido (banco
            con padding, menos de k candidatos reales en este episodio; el
            slot "new" siempre es válido). Los logits de slots inválidos se
            enmascaran a -inf antes del softmax/cross-entropy.

        Returns:
            logits: (B, k) -- una columna por candidato de track_summary
        """
        B, k, D = track_summary.shape
        q_expanded = q_new.expand(-1, k, -1)  # (B, k, D)
        scores = self.mlp(self.bilinear(q_expanded, track_summary)).squeeze(-1)  # (B, k)

        if bank_mask is not None:
            scores = scores.masked_fill(~bank_mask, float("-inf"))

        return scores


class TrackReIDModel(nn.Module):
    """
    Combina IntraTrackEncoder (paso 1) + CrossAttentionMatcher (paso 2) +
    MatchScorer (paso 3). El encoder visual (DINOv2/DINOv3) vive AFUERA --
    este módulo recibe embeddings ya calculados (ver SetTrackReIDModule,
    que agrega el encoder encima de esta arquitectura pura).

    La query llega como `query_views` (B, 2, D) -- tight + wide, ver
    _load_query_views en reid_dataloader.py -- y se procesa con el MISMO
    principio que el historial del banco ("no colapsar antes de comparar"):
      1. Se pasa por `self.intra_track` (la MISMA instancia que contextualiza
         las n vistas de cada track candidato -- tratando a la query como
         "1 track de 2 vistas", el módulo ya soporta esto sin cambios de
         interfaz) para que sus 2 vistas se informen entre sí antes de
         comparar contra la galería. Se descarta el `track_summary` que
         devuelve ese paso -- para la query el pooleo pasa DESPUÉS de la
         cross-attention, no antes (ver paso 3).
      2. Cross-attention contra la galería (`self.matcher`) -- Q=las 2
         vistas ya contextualizadas, K,V=galería. Devuelve 2 salidas
         (una por vista de la query), todavía sin poolear.
      3. `self.query_pool` (una AttentionPool PROPIA, pesos independientes
         de la del banco -- ver AttentionPool) resume esas 2 salidas en un
         solo `q_new` antes de pasarlo al scorer.

    El candidato "new" (objeto genuinamente nuevo, sin match real en el
    banco) es un embedding aprendible (`self.new_embedding`) que se agrega
    como un track MÁS de la galería, con una sola "vista" válida -- el
    propio embedding, sin observaciones reales -- ANTES de `self.intra_track`
    (ver `_append_new_slot`). Así pasa por el MISMO cross-attention contra
    la query y el MISMO bilinear de `self.scorer` que los candidatos reales,
    en vez de vivir como un logit aparte desconectado de la query (ver
    docstring de MatchScorer sobre por qué el diseño anterior diluía la
    señal de identidad en los casos NEW_TRACK). Siempre se agrega al final
    del banco, así el índice k (0-indexed, justo después de los k
    candidatos reales) sigue siendo "no match" -- mismo target que ya usa
    reid_dataloader.py, sin tocarlo.

    forward(): modo entrenamiento -- devuelve logits (B, k+1) listos para
    F.cross_entropy contra el índice del track correcto (o k = "new").

    predict(): modo inferencia -- devuelve match_id (o None si "new")
    por elemento del batch, más los pesos de atención para debug/
    interpretabilidad (ver CrossAttentionMatcher).
    """

    def __init__(
        self, embed_dim: int, num_heads: int = 4, intra_track_layers: int = 1, dropout: float = 0.1,
    ):
        super().__init__()
        self.intra_track = IntraTrackEncoder(
            embed_dim, num_heads=num_heads, num_layers=intra_track_layers, dropout=dropout
        )
        self.matcher = CrossAttentionMatcher(embed_dim, num_heads=num_heads, dropout=dropout)
        self.query_pool = AttentionPool(embed_dim, dropout=dropout)
        self.scorer = MatchScorer(embed_dim, dropout=dropout)
        # candidato "new" -- ver docstring de la clase. Escala 0.02 igual
        # que pool_query en AttentionPool (mismo criterio de init chica).
        self.new_embedding = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def _append_new_slot(
        self,
        gallery_views: torch.Tensor,
        views_mask: torch.Tensor,
        quality_score: torch.Tensor = None,
        bank_mask: torch.Tensor = None,
    ):
        """Agrega `self.new_embedding` como el track k (0-indexed) de la
        galería, con una sola vista válida -- las demás posiciones de `n`
        quedan enmascaradas, igual que un track real corto (ver
        IntraTrackEncoder/AttentionPool, que ya manejan menos de n vistas
        válidas vía masking). `bank_mask` para este slot es siempre True --
        a diferencia de los k reales, nunca es padding."""
        B, k, n, D = gallery_views.shape
        new_track = torch.zeros(B, 1, n, D, dtype=gallery_views.dtype, device=gallery_views.device)
        new_track[:, :, 0, :] = self.new_embedding.expand(B, 1, D)
        new_mask = torch.zeros(B, 1, n, dtype=torch.bool, device=views_mask.device)
        new_mask[:, :, 0] = True

        gallery_views = torch.cat([gallery_views, new_track], dim=1)
        views_mask = torch.cat([views_mask, new_mask], dim=1)
        if quality_score is not None:
            new_quality = torch.zeros(B, 1, n, dtype=quality_score.dtype, device=quality_score.device)
            quality_score = torch.cat([quality_score, new_quality], dim=1)
        if bank_mask is not None:
            new_bank_mask = torch.ones(B, 1, dtype=torch.bool, device=bank_mask.device)
            bank_mask = torch.cat([bank_mask, new_bank_mask], dim=1)
        return gallery_views, views_mask, quality_score, bank_mask

    def forward(
        self,
        query_views: torch.Tensor,
        gallery_views: torch.Tensor,
        views_mask: torch.Tensor,
        quality_score: torch.Tensor = None,
        bank_mask: torch.Tensor = None,
    ) -> dict:
        """
        query_views: (B, 2, D) -- tight + wide de la detección nueva
        gallery_views: (B, k, n, D) -- SIN el candidato "new", se agrega
            adentro (ver _append_new_slot)
        views_mask: (B, k, n) bool
        quality_score: opcional (B, k, n)
        bank_mask: opcional (B, k) bool
        """
        B, L, D = query_views.shape
        query_mask = torch.ones(B, 1, L, dtype=torch.bool, device=query_views.device)
        q_contextualized, _ = self.intra_track(query_views.unsqueeze(1), query_mask)  # reusa el mismo módulo del banco
        q_contextualized = q_contextualized.squeeze(1)  # (B, L, D) -- descartamos el track_summary, no aplica acá

        gallery_views, views_mask, quality_score, bank_mask = self._append_new_slot(
            gallery_views, views_mask, quality_score, bank_mask
        )
        views_contextualized, track_summary = self.intra_track(gallery_views, views_mask, quality_score)

        q_new_per_view, attn_weights = self.matcher(q_contextualized, views_contextualized, views_mask)  # (B, L, D)
        q_new = self.query_pool(q_new_per_view).unsqueeze(1)  # (B, 1, D) -- pooling propio, no promedio simple

        logits = self.scorer(q_new, track_summary, bank_mask)
        return {
            "logits": logits,              # (B, k+1) -- última columna = candidato "new"
            "attn_weights": attn_weights,  # (B, L, k+1, n) -- solo debug
            "track_summary": track_summary,  # (B, k+1, D) -- indexado barato aparte
        }

    @torch.no_grad()
    def predict(
        self,
        query_views: torch.Tensor,
        gallery_views: torch.Tensor,
        views_mask: torch.Tensor,
        quality_score: torch.Tensor = None,
        bank_mask: torch.Tensor = None,
        track_ids: list = None,
    ):
        """track_ids (opcional, len k): mapea el índice ganador a un id real
        de track del banco; sin esto, devuelve el índice crudo (0..k-1) o
        None si "new". `gallery_views` acá es el banco SIN el candidato
        "new" -- self.forward lo agrega adentro (ver _append_new_slot), por
        eso `k = gallery_views.shape[1]` (el tamaño real, previo a agregarlo)
        sigue siendo el índice correcto de "new" en `out["logits"]`."""
        out = self.forward(query_views, gallery_views, views_mask, quality_score, bank_mask)
        pred = out["logits"].argmax(dim=-1)  # (B,), k = "new"
        k = gallery_views.shape[1]
        match_ids = []
        for p in pred.tolist():
            if p == k:
                match_ids.append(None)
            elif track_ids is not None:
                match_ids.append(track_ids[p])
            else:
                match_ids.append(p)
        return match_ids, out["attn_weights"]


class SetTrackReIDModule(nn.Module):
    """
    Wrapper end-to-end: encoder visual (DINOv2Encoder, reusado de
    reid_track_query_module.py -- funciona igual para DINOv2 y DINOv3, ver
    docstring de esa clase) + TrackReIDModel (IntraTrackEncoder ->
    CrossAttentionMatcher -> MatchScorer). Reemplaza a TrackQueryReIDModule.

    proj_dim: proyecta el embedding crudo del encoder (ej. 768 para
    dinov2-base, 384 para dinov2-small) a esta dimensión ANTES de
    TrackReIDModel -- reduce parámetros entrenables y memoria de GPU (los
    modelos "base"/"vitb16" sin proyección llegan a ~465M parámetros
    entrenables y no entran en una GPU de 8GB, ver conversación) sin tocar
    el encoder (que sigue congelado con su dimensión nativa). `None` o `0`
    desactivan la proyección -- se usa la dimensión nativa del encoder
    directamente.
    """

    def __init__(
        self,
        num_heads: int = 4,
        intra_track_layers: int = 1,
        dropout: float = 0.1,
        proj_dim: int = None,
        dinov2_model_name: str = "facebook/dinov2-base",
        dinov2_checkpoint_path: str = None,
        dinov2_embedding_dim: int = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = DINOv2Encoder(
            model_name=dinov2_model_name,
            checkpoint_path=dinov2_checkpoint_path,
            embedding_dim=dinov2_embedding_dim,
            freeze=freeze_encoder,
        )
        encoder_dim = self.encoder.embed_dim
        self.proj = nn.Linear(encoder_dim, proj_dim) if proj_dim else None
        model_dim = proj_dim or encoder_dim
        self.model = TrackReIDModel(
            embed_dim=model_dim, num_heads=num_heads, intra_track_layers=intra_track_layers, dropout=dropout,
        )

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        # images: (..., C, H, W) -> (..., encoder_dim) -> (..., proj_dim) si hay proyección
        leading_shape = images.shape[:-3]
        C, H, W = images.shape[-3:]
        flat = images.reshape(-1, C, H, W)
        visual = self.encoder(flat).reshape(*leading_shape, -1)
        return self.proj(visual) if self.proj is not None else visual

    def forward(
        self,
        query_views: torch.Tensor,
        bank_images: torch.Tensor,
        bank_obs_mask: torch.Tensor,
        bank_mask: torch.Tensor,
        precomputed: bool = False,
    ) -> dict:
        """
        query_views: (B, 2, C, H, W) -- tight + wide, o (B, 2, encoder_dim) si precomputed=True
        bank_images: (B, N, K, C, H, W), o (B, N, K, encoder_dim) si precomputed=True
        bank_obs_mask: (B, N, K) bool
        bank_mask: (B, N) bool
        """
        if precomputed:
            query_embed = self.proj(query_views) if self.proj is not None else query_views
            gallery_views = self.proj(bank_images) if self.proj is not None else bank_images
        else:
            query_embed = self._encode(query_views)     # (B, 2, D)
            gallery_views = self._encode(bank_images)    # (B, N, K, D)
        return self.model(query_embed, gallery_views, bank_obs_mask, bank_mask=bank_mask)


if __name__ == "__main__":
    # Ejemplo dummy end-to-end -- solo shapes + backward, sin dataloader ni
    # training loop real (ver docstring del módulo). D=768 = dim real de
    # dinov2-base/dinov3-vitb16 (ver DINOv2Encoder en reid_track_query_module.py).
    torch.manual_seed(0)
    B, k, n, D = 2, 5, 5, 768
    L = 2  # tight + wide, ver _load_query_views en reid_dataloader.py

    model = TrackReIDModel(embed_dim=D, num_heads=4, intra_track_layers=1, dropout=0.1)

    query_views = torch.randn(B, L, D)
    gallery_views = torch.randn(B, k, n, D)
    views_mask = torch.ones(B, k, n, dtype=torch.bool)
    views_mask[:, -1, 3:] = False  # el último track candidato tiene solo 3/5 vistas válidas
    bank_mask = torch.ones(B, k, dtype=torch.bool)
    quality_score = torch.rand(B, k, n)

    out = model(query_views, gallery_views, views_mask, quality_score=quality_score, bank_mask=bank_mask)
    print("logits:", out["logits"].shape)               # (B, k+1)
    print("attn_weights:", out["attn_weights"].shape)   # (B, L, k+1, n)
    print("track_summary:", out["track_summary"].shape)  # (B, k+1, D)

    # target de ejemplo: batch 0 matchea el track índice 2, batch 1 es "new" (índice k).
    target = torch.tensor([2, k])
    loss = F.cross_entropy(out["logits"], target)
    print("loss:", loss.item())
    loss.backward()
    print("backward OK, sin NaN:", not torch.isnan(loss).item())

    match_ids, attn = model.predict(
        query_views, gallery_views, views_mask, quality_score=quality_score, bank_mask=bank_mask,
        track_ids=["trackA", "trackB", "trackC", "trackD", "trackE"],
    )
    print("match_ids (inferencia):", match_ids)
