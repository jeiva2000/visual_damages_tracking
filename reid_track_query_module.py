"""
Módulo de ReID basado en Track-Queries con Cross-Attention contra un banco
de tracks perdidos (lost tracks bank), pensado para integrarse como paso
posterior al matching de ByteTrack/BoT-SORT cuando se detecta un ID nuevo.

Arquitectura por bloque:
    queries (detecciones nuevas)
        -> self-attention entre queries (para resolver competencia por el mismo slot)
        -> cross-attention (Q=queries, K,V=banco de tracks perdidos + encoding temporal/espacial)
        -> clasificador match/no-match por query
        -> (si match) identidad = argmax de los pesos de atención

Notas de diseño (resumen de la discusión):
- No se usa positional encoding secuencial clásico: tanto el banco como las
  queries son conjuntos, no secuencias. El orden de inserción es arbitrario.
- Se inyecta encoding espacial/de parte del vehículo, tanto en queries como
  en banco.
- NO se inyecta encoding temporal (frames_since_lost) en K/V del banco:
  hacerlo sesga el dot-product de cross-attention hacia tracks recientes,
  penalizando el caso real donde el match correcto es un track perdido
  hace mucho tiempo. La similitud que decide identidad debe ser puramente
  visual + espacial. (Queda pendiente evaluar si reintroducirla más adelante
  solo como feature auxiliar del clasificador match/no-match, concatenada,
  no sumada a K/V.)
- El match/no-match se resuelve con un clasificador MLP separado, no con un
  slot "null" en K/V, porque desacopla "similitud a cada candidato" de
  "existe o no un candidato válido".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DINOv2Encoder(nn.Module):
    """
    Envuelve un backbone DINOv2 de HuggingFace (`transformers.AutoModel`) para
    producir un embedding visual (CLS token) por crop de imagen -- misma vía
    de carga que dinov2_reid_finetune.py (ReIDModel.backbone) y
    dinov2_reid_analysis.py (load_dinov2), para poder reutilizar directamente
    un checkpoint fine-tuneado en este módulo.

    model_name: cualquier checkpoint de HF compatible con AutoModel, ej.
        "facebook/dinov2-small" -> hidden_size 384
        "facebook/dinov2-base"  -> hidden_size 768
        "facebook/dinov2-large" -> hidden_size 1024
        "facebook/dinov2-giant" -> hidden_size 1536

    checkpoint_path (opcional): state_dict de ReIDModel de
    dinov2_reid_finetune.py (claves "backbone.*" + "projection.*"). Si se
    pasa, se sobreescriben los pesos del backbone con los fine-tuneados para
    daños/vehículos y se reconstruye también su head de proyección
    entrenado -- en ese caso `embedding_dim` debe coincidir con el usado al
    entrenar ese checkpoint, y pasa a ser el embed_dim final (no el
    hidden_size crudo del backbone).

    Las imágenes de entrada deben venir ya normalizadas (ImageNet mean/std)
    y con tamaño múltiplo del patch size del backbone (14 en DINOv2, 16 en
    DINOv3) -- 224x224 (DEFAULT_TRANSFORM en reid_dataloader.py) es múltiplo
    de ambos, así que no hace falta ajustar el preprocesado al cambiar de
    encoder. Compatible con cualquier checkpoint de HF cuyo AutoModel
    devuelva last_hidden_state con el CLS token en la posición 0 -- vale
    para DINOv2 y para las variantes ViT de DINOv3 (que además traen 4
    "register tokens" en las posiciones 1-4, pero eso no afecta la posición
    0). Las variantes ConvNeXt de DINOv3 NO son compatibles tal cual (no son
    transformers, no exponen last_hidden_state con esa forma).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        checkpoint_path: str = None,
        embedding_dim: int = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.projection = None
        if checkpoint_path is not None:
            if embedding_dim is None:
                raise ValueError(
                    "Si pasas checkpoint_path, debes especificar embedding_dim "
                    "(el usado al entrenar ese checkpoint en dinov2_reid_finetune.py)."
                )
            state = torch.load(checkpoint_path, map_location="cpu")
            backbone_state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
            if not backbone_state:
                raise ValueError(
                    f"{checkpoint_path} no tiene claves 'backbone.*' "
                    f"(¿no es un checkpoint de dinov2_reid_finetune.py?)"
                )
            self.backbone.load_state_dict(backbone_state, strict=True)

            projection_state = {k[len("projection."):]: v for k, v in state.items() if k.startswith("projection.")}
            if not projection_state:
                raise ValueError(f"{checkpoint_path} no tiene claves 'projection.*'")
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
            self.projection.load_state_dict(projection_state, strict=True)

        self.embed_dim = embedding_dim if self.projection is not None else hidden_size

        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            if self.projection is not None:
                for p in self.projection.parameters():
                    p.requires_grad = False
                self.projection.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # mantener backbone/projection en eval aunque el módulo esté en train()
            self.backbone.eval()
            if self.projection is not None:
                self.projection.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W) -> (B, embed_dim)
        def _run(imgs):
            cls_emb = self.backbone(pixel_values=imgs).last_hidden_state[:, 0, :]
            if self.projection is not None:
                cls_emb = self.projection(cls_emb)
            return cls_emb

        if self.freeze:
            with torch.no_grad():
                return _run(images)
        return _run(images)


# NOTA: la idea original de encoding espacial (parte del vehículo + bbox
# absoluto, sumado a los tokens de query/banco ya fusionados) sigue afuera --
# se probó primero el módulo con solo similitud visual pura. La clase
# SpatialEncoding que sí se implementó (abajo) resuelve un problema distinto
# y más acotado: dentro de MultiScaleCropFusion, para desambiguar qué daño
# es "el de referencia" cuando el crop de contexto amplio muestra varios.
# Si más adelante se quiere agregar además class de parte + bbox absoluto a
# nivel de token fusionado, es una capa aparte, no reemplaza a esta.


class SpatialEncoding(nn.Module):
    """
    Codifica la posición relativa del daño de referencia dentro de un crop
    -- (cx, cy, w, h) normalizados en [0,1], ver _bbox_relative_to_crop en
    reid_dataloader.py -- en un embedding sumable al token visual de esa
    escala.

    Por qué hace falta: el crop de contexto amplio (más padding) puede
    mostrar OTROS daños de rebote, no solo el de referencia -- la apariencia
    visual sola no distingue cuál es "el" daño de esta observación. Esta
    caja sí lo distingue, porque siempre corresponde exactamente al daño de
    referencia (el crop se generó expandiendo desde SU bbox): en el crop
    ajustado la señal es casi trivial (el daño ocupa casi todo el cuadro,
    centrado), el aporte real está en las escalas con más contexto, donde
    el centro se corre si el crop se clippeó contra el borde de la imagen.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        # boxes: (..., 4) -> (..., embed_dim)
        return self.mlp(boxes)


class MultiScaleCropFusion(nn.Module):
    """
    Fusiona los embeddings de S recortes de la misma detección, tomados con
    padding decreciente (ej. crop 0 = amplio, con contexto de la
    pieza/vehículo alrededor, crop S-1 = ajustado al daño) -- ver
    DEFAULT_PADDING_FACTORS en reid_dataloader.py.

    Por qué NO mean-pooling: promediar diluye la señal fina del recorte
    ajustado (que es la más discriminativa para diferenciar dos daños
    parecidos) con la señal más genérica de los recortes de contexto. Un
    promedio fijo no puede decidir, por instancia, cuánto contexto hace
    falta.

    Por qué attention pooling: un vector de consulta aprendible hace
    cross-attention sobre los S embeddings de escala y aprende a ponderar
    cada uno según el caso -- para un daño con textura muy característica,
    puede apoyarse más en el crop ajustado; para un daño ambiguo, puede
    darle más peso al contexto amplio.

    scale_embedding: a diferencia de la posición en el banco (que es
    arbitraria y por eso no se codifica), aquí el índice de escala SÍ es
    información real y fija (0 siempre es el más amplio, S-1 el más
    ajustado), así que se suma un embedding aprendido por posición de escala.

    spatial_encoding: además del índice de escala, se suma la posición
    relativa del daño DENTRO de esa escala (ver SpatialEncoding arriba) --
    pero SOLO en la escala 0 (amplia): es la única vista donde puede
    aparecer más de un daño y la escala por sí sola no dice cuál es el de
    referencia. En las escalas más ajustadas el crop se genera expandiendo
    desde el mismo bbox, así que el daño cae casi siempre centrado -- el
    bbox relativo ahí es prácticamente constante por escala, redundante con
    lo que ya aporta `scale_embedding`, así que no vale la pena gastarle
    parámetros.
    """

    def __init__(self, embed_dim: int, num_scales: int = 5, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_scales = num_scales
        self.scale_embedding = nn.Embedding(num_scales, embed_dim)
        self.spatial_encoding = SpatialEncoding(embed_dim)
        # máscara fija: solo la escala 0 (amplia) usa spatial_encoding (ver
        # docstring de la clase); se registra como buffer para no
        # recrearla en cada forward y para que siga al módulo entre devices.
        spatial_mask = torch.zeros(num_scales, 1)
        spatial_mask[0] = 1.0
        self.register_buffer("spatial_mask", spatial_mask)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, crop_embeddings: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        # crop_embeddings: (M, num_scales, embed_dim), M = detecciones aplanadas (B*Q o B*N)
        # boxes: (M, num_scales, 4), caja del daño relativa a cada crop
        M = crop_embeddings.shape[0]
        scale_idx = torch.arange(self.num_scales, device=crop_embeddings.device)
        tokens = (
            crop_embeddings
            + self.scale_embedding(scale_idx).unsqueeze(0)              # (M, num_scales, embed_dim)
            + self.spatial_encoding(boxes) * self.spatial_mask.unsqueeze(0)  # solo escala 0
        )

        query = self.pool_query.expand(M, -1, -1)  # (M, 1, embed_dim)
        fused, _ = self.attn(query=query, key=tokens, value=tokens, need_weights=False)
        fused = fused.squeeze(1)  # (M, embed_dim)

        # residual con la media simple como base estable, por si la atención
        # colapsa al inicio del entrenamiento (evita gradientes muertos)
        fused = self.norm(fused + crop_embeddings.mean(dim=1))
        return fused  # (M, embed_dim)


class TrackHistoryFusion(nn.Module):
    """
    Fusiona hasta K observaciones DISTINTAS del historial de un track del
    banco (no escalas espaciales de una sola foto -- ver
    MultiScaleCropFusion para eso) en un solo embedding, vía attention
    pooling con un query aprendible -- mismo mecanismo que
    MultiScaleCropFusion pero sin scale_embedding ni spatial_encoding: acá
    el índice dentro de K no tiene un significado fijo como "amplio/
    ajustado", son simplemente instantes distintos del track repartidos en
    el tiempo (ver _load_track_history_crops en reid_dataloader.py), así
    que no hay nada fijo que codificar por posición.

    Por qué attention pooling: igual que en MultiScaleCropFusion, un
    promedio fijo no puede decidir por track cuánto pesar cada observación
    (ej. una vista más clara del daño vs. una más ocluida).

    obs_mask: tracks con historial más corto que K tienen slots de padding
    -- se enmascaran como key_padding_mask, igual que bank_mask/query_mask
    en el resto del módulo.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, obs_embeddings: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        # obs_embeddings: (M, K, embed_dim), M = tracks del banco aplanados (B*N)
        # obs_mask: (M, K) bool, True donde hay observación real (no padding)
        M = obs_embeddings.shape[0]
        # evita filas con obs_mask 100% False (slots de padding del banco,
        # ver bank_mask en reid_dataloader.py -- esos tracks no existen y se
        # descartan más adelante vía bank_mask en TrackQueryReIDModule.forward):
        # un key_padding_mask todo-True puede devolver NaN de MultiheadAttention,
        # que luego contaminaría el resto del batch aunque el resultado de esa
        # fila se ignore después.
        safe_mask = obs_mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True

        query = self.pool_query.expand(M, -1, -1)  # (M, 1, embed_dim)
        fused, _ = self.attn(
            query=query, key=obs_embeddings, value=obs_embeddings,
            key_padding_mask=~safe_mask, need_weights=False,
        )
        fused = fused.squeeze(1)  # (M, embed_dim)

        # residual con la media (solo sobre slots válidos) como base estable
        n_valid = safe_mask.sum(dim=1, keepdim=True).clamp_min(1)  # (M, 1)
        mean = (obs_embeddings * safe_mask.unsqueeze(-1)).sum(dim=1) / n_valid
        fused = self.norm(fused + mean)
        return fused  # (M, embed_dim)


class MemoryFusion(nn.Module):
    """
    Segunda etapa de fusión: el embedding local de una detección (salida de
    MultiScaleCropFusion, ya resume las 3 escalas multi-crop) hace
    cross-attention contra la "memoria" -- el frame COMPLETO de esa misma
    observación con el bbox del daño dibujado encima (ver _load_memory_image
    en reid_dataloader.py), encodeado por DINOv2 en un solo embedding.

    Por qué: el embedding local (recortes) solo ve la zona alrededor del
    daño, aun con contexto amplio. La memoria completa la foto entera --
    resto del vehículo, otros daños, objetos de referencia -- para que el
    token final combine detalle local + escena completa en el momento de
    la detección, no solo lo que entra en los recortes.

    local_embed hace de QUERY y memory_embed de K,V (un solo token): con un
    solo K,V la atención no "elige" entre candidatos, funciona como una
    modulación/inyección de la memoria sobre el embedding local (similar en
    espíritu al residual de MultiScaleCropFusion, pero cruzando dos fuentes
    distintas en vez de promediar escalas de la misma fuente).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, local_embed: torch.Tensor, memory_embed: torch.Tensor) -> torch.Tensor:
        # local_embed, memory_embed: (M, embed_dim) -> (M, embed_dim)
        query = local_embed.unsqueeze(1)     # (M, 1, embed_dim)
        kv = memory_embed.unsqueeze(1)       # (M, 1, embed_dim)
        attn_out, _ = self.attn(query=query, key=kv, value=kv, need_weights=False)
        attn_out = attn_out.squeeze(1)       # (M, embed_dim)
        return self.norm(attn_out + local_embed)


class TrackQueryReIDModule(nn.Module):
    """
    El encoder visual (DINOv2) vive DENTRO de este módulo (self.encoder).
    Args esperados en forward:
      query_images:   (B, Q, S, C, H, W)  S recortes multi-escala (mismo
                                           bbox, distinto padding -- ver
                                           DEFAULT_PADDING_FACTORS en
                                           reid_dataloader.py) por cada
                                           detección nueva, normalizados
                                           (ImageNet mean/std)
      query_boxes:    (B, Q, S, 4)   caja del daño (cx,cy,w,h normalizados)
                                      relativa a CADA uno de esos S crops --
                                      ver _bbox_relative_to_crop en
                                      reid_dataloader.py
      query_memory:   (B, Q, C, H, W)  frame COMPLETO de esa misma detección
                                        con el bbox del daño dibujado -- ver
                                        _load_memory_image en
                                        reid_dataloader.py
      query_mask:     (B, Q)        bool, True donde hay query válida (padding)

      bank_images:    (B, N, K, C, H, W)  hasta K observaciones DISTINTAS
                                           del historial de cada track del
                                           banco, repartidas en el tiempo
                                           (no escalas espaciales de una
                                           sola foto -- ver
                                           _load_track_history_crops en
                                           reid_dataloader.py), cada una un
                                           recorte ajustado (sin padding
                                           espacial, eso se omite por ahora)
      bank_obs_mask:  (B, N, K)      bool, True donde el track tiene una
                                      observación real en ese slot de
                                      historial (tracks con menos de K
                                      observaciones se enmascaran)
      bank_memory:    (B, N, C, H, W)  frame completo con el bbox dibujado
                                        de la observación MÁS RECIENTE del
                                        track, mismo esquema que query_memory
      bank_mask:      (B, N)         bool, True donde hay track válido en el
                                      banco (padding a nivel de slot)

      Los S embeddings de escala de cada query se generan con DINOv2 y se
      funden en UN embedding LOCAL vía MultiScaleCropFusion (attention
      pooling con query aprendible, no mean-pooling -- ver docstring de la
      clase) -- así ese embedding puede pesar distinto "daño ajustado" vs.
      "contexto de pieza/vehículo" según el caso, en vez de comprometerse a
      un solo padding de bbox o a un promedio fijo. query_boxes le da a esa
      fusión la posición exacta del daño de referencia dentro de cada
      escala (ver SpatialEncoding), necesaria porque el crop de contexto
      amplio puede mostrar más de un daño. Los K embeddings de historial de
      cada slot del banco se funden análogamente vía TrackHistoryFusion
      (mismo attention pooling, sin scale_embedding/spatial_encoding -- ver
      docstring de esa clase).

      Ese embedding local (query o banco) hace luego cross-attention (vía
      MemoryFusion) contra el embedding de *_memory (frame completo, un
      solo K,V) -- así el token final que entra al self/cross-attention
      grande combina el detalle local con el contexto de la escena completa
      en el momento de esa observación.

      Nota: a nivel de token ya fusionado todavía solo se usa similitud
      visual pura (sin part/bbox absoluto ni frames_since_lost). Ver notas
      de diseño arriba.

      precomputed: bool, default False. Si True, *_images/*_memory ya vienen
      encodeados por DINOv2 (ver PrecomputedEmbeddingCache en
      reid_dataloader.py: (B,Q,S,encoder_dim)/(B,Q,encoder_dim) en vez de
      con canal C,H,W) y se salta self._encode() -- para iterar rápido en
      experimentos sin correr DINOv2 en cada época (solo válido con encoder
      congelado y sin aumentación).

    Returns dict con:
      match_logit:      (B, Q)      logit binario match/no-match por query
      attn_weights:     (B, Q, N)   pesos de cross-attention (para argmax de identidad)
      pred_identity:    (B, Q)      índice argmax dentro del banco (válido solo si match)
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_self_attn_layers: int = 2,
        dropout: float = 0.1,
        num_scales: int = 5,
        fusion_num_heads: int = 4,
        encoder: nn.Module = None,
        encoder_embed_dim: int = None,
        dinov2_model_name: str = "facebook/dinov2-base",
        dinov2_checkpoint_path: str = None,
        dinov2_embedding_dim: int = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.dim = dim

        # encoder visual: si no se pasa uno propio, carga DINOv2 vía HuggingFace
        # (transformers.AutoModel, ver DINOv2Encoder). Pasa `encoder=` solo para
        # inyectar un módulo ya construido (tests, o un backbone que ya vive en
        # otro objeto) -- debe implementar forward(images) -> (B, encoder_embed_dim).
        if encoder is not None:
            if encoder_embed_dim is None:
                raise ValueError("Si pasas `encoder=`, debes especificar `encoder_embed_dim`.")
            self.encoder = encoder
            encoder_dim = encoder_embed_dim
        else:
            self.encoder = DINOv2Encoder(
                model_name=dinov2_model_name,
                checkpoint_path=dinov2_checkpoint_path,
                embedding_dim=dinov2_embedding_dim,
                freeze=freeze_encoder,
            )
            encoder_dim = self.encoder.embed_dim

        # funde los S embeddings de escala de CADA query (una sola detección,
        # mismo bbox con distinto padding) en un solo embedding local.
        self.scale_fusion = MultiScaleCropFusion(
            embed_dim=encoder_dim, num_scales=num_scales, num_heads=fusion_num_heads, dropout=dropout
        )
        # funde los K embeddings de historial de CADA track del banco
        # (observaciones distintas en el tiempo, no escalas espaciales) en un
        # solo embedding local -- ver TrackHistoryFusion.
        self.track_history_fusion = TrackHistoryFusion(
            embed_dim=encoder_dim, num_heads=fusion_num_heads, dropout=dropout
        )

        # cross-attention del embedding local (query) contra la memoria --
        # frame completo con el bbox dibujado, un solo K,V -- compartido
        # entre query y banco por la misma razón que scale_fusion.
        self.memory_fusion = MemoryFusion(
            embed_dim=encoder_dim, num_heads=fusion_num_heads, dropout=dropout
        )

        # proyección del embed_dim del encoder al dim interno del transformer
        self.query_in_proj = nn.Linear(encoder_dim, dim)
        self.bank_in_proj = nn.Linear(encoder_dim, dim)

        # NOTA: encoding de parte del vehículo + bbox absoluto a nivel de token
        # fusionado sigue afuera (ver nota de diseño arriba). El SpatialEncoding
        # que sí está activo vive dentro de MultiScaleCropFusion (self.scale_fusion).

        # self-attention entre queries: resuelve competencia por el mismo track
        self_attn_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.query_self_attn = nn.TransformerEncoder(
            self_attn_layer, num_layers=num_self_attn_layers
        )

        # cross-attention: Q = queries, K,V = banco
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(dim)

        # clasificador match/no-match
        # input: [query_original, cross_attn_output, max_attn_weight, entropy]
        classifier_in_dim = dim * 2 + 2
        self.match_classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )

    def _encode(self, images):
        # images: (..., C, H, W) -> (..., encoder_dim), preservando las dims iniciales
        leading_shape = images.shape[:-3]
        C, H, W = images.shape[-3:]
        flat_images = images.reshape(-1, C, H, W)
        visual = self.encoder(flat_images)               # (prod(leading_shape), encoder_dim)
        return visual.reshape(*leading_shape, -1)

    def _build_query_tokens(self, query_images, query_boxes, query_memory, precomputed=False):
        # query_images: (B, Q, S, C, H, W) o, si precomputed=True, ya
        # (B, Q, S, encoder_dim) -- ver PrecomputedEmbeddingCache en
        # reid_dataloader.py. query_memory: (B, Q, C, H, W) o (B, Q, encoder_dim).
        # Encodea los S recortes multi-escala con DINOv2 (salvo que ya vengan
        # encodeados) y los funde en un embedding LOCAL por query
        # (MultiScaleCropFusion espera (M, S, encoder_dim) / (M, S, 4), así
        # que se aplana B*Q antes de fusionar y se deshace después). Ese
        # embedding local hace luego cross-attention contra el embedding de
        # la memoria (frame completo) vía MemoryFusion.
        B, Q, S = query_images.shape[:3]
        visual = query_images if precomputed else self._encode(query_images)  # (B, Q, S, encoder_dim)
        local_fused = self.scale_fusion(visual.reshape(B * Q, S, -1), query_boxes.reshape(B * Q, S, -1))

        memory_visual = query_memory if precomputed else self._encode(query_memory)  # (B, Q, encoder_dim)
        fused = self.memory_fusion(local_fused, memory_visual.reshape(B * Q, -1))  # (B*Q, encoder_dim)
        fused = fused.reshape(B, Q, -1)
        return self.query_in_proj(fused)

    def _build_bank_tokens(self, bank_images, bank_obs_mask, bank_memory, precomputed=False):
        # bank_images: (B, N, K, C, H, W) o (B, N, K, encoder_dim) si
        # precomputed=True -- K observaciones del historial del track, no
        # escalas espaciales (ver TrackHistoryFusion). bank_obs_mask: (B, N, K)
        # bool. bank_memory: (B, N, C, H, W) o (B, N, encoder_dim) -> (B, N, dim).
        # Mismo esquema que _build_query_tokens, aplanando B*N para la fusión.
        B, N, K = bank_images.shape[:3]
        visual = bank_images if precomputed else self._encode(bank_images)  # (B, N, K, encoder_dim)
        local_fused = self.track_history_fusion(visual.reshape(B * N, K, -1), bank_obs_mask.reshape(B * N, K))

        memory_visual = bank_memory if precomputed else self._encode(bank_memory)  # (B, N, encoder_dim)
        fused = self.memory_fusion(local_fused, memory_visual.reshape(B * N, -1))  # (B*N, encoder_dim)
        fused = fused.reshape(B, N, -1)
        return self.bank_in_proj(fused)

    def forward(
        self,
        query_images, query_boxes, query_memory, query_mask,
        bank_images, bank_obs_mask, bank_memory, bank_mask,
        precomputed: bool = False,
    ):
        B, Q = query_images.shape[0], query_images.shape[1]
        N = bank_images.shape[1]

        query_tokens_raw = self._build_query_tokens(query_images, query_boxes, query_memory, precomputed=precomputed)
        bank_tokens = self._build_bank_tokens(bank_images, bank_obs_mask, bank_memory, precomputed=precomputed)

        # self-attention entre queries (padding mask: True = ignorar)
        query_key_padding_mask = ~query_mask
        query_tokens = self.query_self_attn(
            query_tokens_raw, src_key_padding_mask=query_key_padding_mask
        )

        # cross-attention Q=queries, K,V=banco
        bank_key_padding_mask = ~bank_mask
        attn_out, attn_weights = self.cross_attn(
            query=query_tokens,
            key=bank_tokens,
            value=bank_tokens,
            key_padding_mask=bank_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,  # (B, Q, N) promediado sobre heads
        )
        attn_out = self.cross_attn_norm(attn_out + query_tokens)

        # señales auxiliares para el clasificador: max weight y entropía
        eps = 1e-8
        max_weight = attn_weights.max(dim=-1, keepdim=True).values  # (B, Q, 1)
        entropy = -(attn_weights.clamp_min(eps) * attn_weights.clamp_min(eps).log()).sum(
            dim=-1, keepdim=True
        )  # (B, Q, 1)

        classifier_input = torch.cat(
            [query_tokens_raw, attn_out, max_weight, entropy], dim=-1
        )
        match_logit = self.match_classifier(classifier_input).squeeze(-1)  # (B, Q)

        # identidad predicha: argmax de attn_weights, enmascarando slots inválidos del banco
        masked_weights = attn_weights.masked_fill(~bank_mask.unsqueeze(1), float("-inf"))
        pred_identity = masked_weights.argmax(dim=-1)  # (B, Q)

        return {
            "match_logit": match_logit,
            "attn_weights": attn_weights,
            "pred_identity": pred_identity,
        }


def varifocal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Varifocal Loss (Zhang et al., 2021), adaptada a etiquetas binarias duras
    (0/1) en vez de scores de calidad continuos (IoU-aware):

      - target == 1 (match verdadero): término estándar de BCE, sin down-weighting.
        Queremos gradiente completo en los pocos positivos que sí tenemos.
      - target == 0 (no-match): weight = alpha * p^gamma, con p = sigmoid(logit)
        DETACHED (sin gradiente a través del peso, como en el paper original).
        Esto reduce el peso de negativos fáciles (p bajo, ya bien clasificados)
        y concentra el gradiente en negativos difíciles (p alto pese a ser
        no-match) — mismo espíritu que focal loss pero asimétrico entre clases,
        que es justo el desbalance que tenemos (la mayoría de queries nuevas
        de ByteTrack son genuinamente 'no match').

    logits, targets: mismo shape, ya aplanados/filtrados por máscara de padding.
    Si vienen vacíos (0 elementos -- ej. un batch sin ninguna query válida),
    devuelve 0.0 en vez de NaN (`.mean()`/`.sum()` de un tensor vacío rompe).
    """
    if logits.numel() == 0:
        return logits.new_zeros(0) if reduction == "none" else logits.sum()

    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    p_detached = p.detach()
    weight = targets + (1.0 - targets) * alpha * p_detached.pow(gamma)

    loss = weight * ce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def compute_losses(
    outputs: dict,
    match_labels: torch.Tensor,      # (B, Q) float 0/1, 1 = hay match verdadero
    identity_labels: torch.Tensor,   # (B, Q) long, índice correcto en el banco (ignorado si no hay match)
    query_mask: torch.Tensor,        # (B, Q) bool
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    w_match: float = 1.0,
    w_identity: float = 1.0,
):
    """
    Dos losses:
      - Varifocal loss para match/no-match (reemplaza a BCE ponderado; maneja
        el desbalance de clases de forma más suave que pos_weight fijo, ver
        `varifocal_loss` arriba).
      - CrossEntropy sobre attn_weights para la identidad correcta, SOLO en
        queries donde match_labels == 1.

    `w_match`/`w_identity` pesan la suma final (`total_loss`) -- por defecto
    1:1, pero conviene poder ajustarlos si una de las dos domina el
    gradiente en la práctica (mismo patrón que w_triplet/w_id/w_consistency
    en dinov2_reid_finetune.py).
    """
    flat_mask = query_mask.reshape(-1)

    match_logit = outputs["match_logit"].reshape(-1)[flat_mask]
    match_labels_flat = match_labels.reshape(-1)[flat_mask]
    match_loss = varifocal_loss(
        match_logit, match_labels_flat, alpha=vfl_alpha, gamma=vfl_gamma
    )

    attn_weights = outputs["attn_weights"]  # (B, Q, N)
    B, Q, N = attn_weights.shape
    attn_flat = attn_weights.reshape(B * Q, N)[flat_mask]
    identity_flat = identity_labels.reshape(-1)[flat_mask]
    is_match_flat = match_labels_flat.bool()

    if is_match_flat.any():
        # NLL sobre los pesos de atención (ya están normalizados por softmax interno)
        log_attn = torch.log(attn_flat.clamp_min(1e-8))
        identity_loss = F.nll_loss(
            log_attn[is_match_flat], identity_flat[is_match_flat]
        )
    else:
        identity_loss = torch.tensor(0.0, device=match_logit.device)

    total_loss = w_match * match_loss + w_identity * identity_loss
    return {
        "total_loss": total_loss,
        "match_loss": match_loss,
        "identity_loss": identity_loss,
    }


if __name__ == "__main__":
    # smoke test rápido con formas dummy.
    #
    # NOTA: en este entorno de pruebas no hay acceso de red para descargar el
    # DINOv2 real vía HuggingFace, así que inyectamos un backbone dummy
    # (mismo shape de salida) usando el argumento `encoder=` para validar
    # que el flujo imagen -> encoder -> fusión multi-escala -> proyección
    # -> transformer funciona.
    #
    # En tu workstation/HPC con acceso a internet, simplemente omite
    # `encoder=` y `encoder_embed_dim=` y usa:
    #   TrackQueryReIDModule(dim=256, dinov2_model_name="facebook/dinov2-base")
    # o pasa el checkpoint fine-tuneado de dinov2_reid_finetune.py:
    #   TrackQueryReIDModule(dim=256, dinov2_checkpoint_path="reid_finetuned/best_model.pt",
    #                        dinov2_embedding_dim=256)

    torch.manual_seed(0)
    B, Q, N, S = 2, 5, 10, 5  # S = escalas por crop (ver DEFAULT_PADDING_FACTORS en reid_dataloader.py)
    K = 4  # observaciones de historial por track del banco (ver TrackHistoryFusion)
    C, H, W = 3, 224, 224
    dummy_embed_dim = 768

    class _DummyBackbone(nn.Module):
        """Sustituto de DINOv2 solo para probar el flujo end-to-end sin red."""
        def __init__(self, embed_dim):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Conv2d(C, embed_dim, kernel_size=1)

        def forward(self, x):
            return self.proj(self.pool(x)).flatten(1)

    module = TrackQueryReIDModule(
        dim=256,
        num_heads=8,
        num_scales=S,
        encoder=_DummyBackbone(dummy_embed_dim),
        encoder_embed_dim=dummy_embed_dim,
        freeze_encoder=False,  # dummy, no tiene sentido congelarlo
    )

    query_images = torch.randn(B, Q, S, C, H, W)
    query_boxes = torch.rand(B, Q, S, 4)
    query_memory = torch.randn(B, Q, C, H, W)
    query_mask = torch.ones(B, Q, dtype=torch.bool)

    bank_images = torch.randn(B, N, K, C, H, W)
    bank_obs_mask = torch.ones(B, N, K, dtype=torch.bool)
    bank_memory = torch.randn(B, N, C, H, W)
    bank_mask = torch.ones(B, N, dtype=torch.bool)

    out = module(query_images, query_boxes, query_memory, query_mask,
                 bank_images, bank_obs_mask, bank_memory, bank_mask)
    print("match_logit:", out["match_logit"].shape)
    print("attn_weights:", out["attn_weights"].shape)
    print("pred_identity:", out["pred_identity"].shape)

    match_labels = torch.randint(0, 2, (B, Q)).float()
    identity_labels = torch.randint(0, N, (B, Q))
    losses = compute_losses(out, match_labels, identity_labels, query_mask, vfl_alpha=0.75, vfl_gamma=2.0)
    print(losses)

