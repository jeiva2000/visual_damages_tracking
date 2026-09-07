"""
Sampler jerárquico P x K x N (secuencias x tracks x vistas) + batch-hard
triplet mining (Hermans et al., "In Defense of the Triplet Loss for Person
Re-Identification") + reporte de composición de batch para debug.

Motivación: el dataloader de episodios actual (ReIDEpisodeDataset,
reid_dataloader.py) arma cada episodio de forma independiente y generativa
(cada __getitem__ sortea su propia identidad de referencia + distractores,
sin usar el índice `idx` que le pasaría un Sampler). Ya mitiga en parte el
riesgo de que el modelo aprenda shortcuts a nivel de video (iluminación,
color del vehículo) en vez de apariencia real del daño, vía
`hard_distractor_ratio`: una fracción configurable de los distractores de
CADA episodio son del mismo video que la identidad de referencia
(intra-secuencia, el negativo difícil real). Pero no da control EXPLÍCITO
sobre cuántas secuencias/tracks/vistas distintos entran a un batch, ni un
mecanismo para verificar esa composición -- que es lo que pide este módulo.

Este es un componente NUEVO y AUTÓNOMO, pensado para un esquema de
entrenamiento por métrica (embeddings + distancia, estilo Hermans et al.)
-- un paradigma distinto del banco/query de clasificación (k+1)-way que usa
hoy TrackReIDModel/SetTrackReIDModule (ver set_reid_model.py), que no opera
sobre batches planos de instancias sino sobre episodios completos
(banco + query). Por eso -- y por instrucción explícita de verificarlo
"antes de conectarlo al dataset real" -- se prueba acá con un dataset
dummy, SIN conectarlo todavía a ReIDEpisodeDataset/train_track_query_reid.py.
Conectarlo al entrenamiento real requiere antes decidir CÓMO convive con el
entrenamiento por episodios existente (¿lo reemplaza? ¿loss auxiliar en
paralelo sobre los embeddings intermedios, ej. track_summary/q_new de
TrackReIDModel?) -- queda pendiente de decisión, no asumido acá.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from reid_dataloader import DamageTrackIndex, DEFAULT_TRANSFORM, TrackObservation, _expand_bbox


class FlatTrackViewDataset(Dataset):
    """Aplana un DamageTrackIndex a nivel de OBSERVACIÓN individual (una
    "vista" = un TrackObservation) -- la unidad mínima que necesita
    HierarchicalPKSampler para samplear P secuencias x K tracks x N vistas
    por batch. Reusa DamageTrackIndex/IdentityTrack tal cual (no duplica el
    indexado por video que ya existe ahí, ver DamageTrackIndex.by_video en
    reid_dataloader.py) -- solo agrega el indexado por TRACK dentro de cada
    secuencia, que no existía.

    `by_sequence`: video_id -> track_id (identity_id) -> lista de índices
    GLOBALES en `self.items` -- la estructura que consume el sampler.
    """

    def __init__(self, track_index: DamageTrackIndex, transform=None, query_wide_factor: Optional[float] = None):
        self.transform = transform or DEFAULT_TRANSFORM
        # si se pasa, cada item también trae un recorte AMPLIO (factor
        # query_wide_factor) de la MISMA observación además del ajustado --
        # ver train_linear_reid_head.py, que promedia ambos por observación
        # para entrenar sobre el mismo tipo de embedding "query" (ajustada +
        # amplia) que usa _load_query_views en reid_dataloader.py, en vez de
        # solo el recorte ajustado aislado.
        self.query_wide_factor = query_wide_factor

        self.items: List[Tuple[str, str, TrackObservation]] = []  # (video_id, identity_id, obs)
        by_sequence: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for track in track_index.tracks:
            for obs in track.observations:
                gidx = len(self.items)
                self.items.append((track.video_id, track.identity_id, obs))
                by_sequence[track.video_id][track.identity_id].append(gidx)
        self.by_sequence: Dict[str, Dict[str, List[int]]] = {
            seq: dict(tracks) for seq, tracks in by_sequence.items()
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        """Recorte ajustado (factor=1.0) de la observación -- suficiente
        para un encoder tipo DINOv2 sin las 2/3 escalas del banco/query de
        ReIDEpisodeDataset (acá cada item YA es una sola vista/instancia).
        Si se pasó `query_wide_factor`, además incluye `image_wide` (mismo
        criterio de recorte que la vista amplia de la query real).
        Reemplazar `transform`/subclasear para otro criterio de recorte."""
        video_id, track_id, obs = self.items[idx]
        from PIL import Image
        image = Image.open(obs.image_path).convert("RGB")
        image_t = self.transform(image.crop(_expand_bbox(obs.bbox, 1.0, image.size)))
        item = {"image": image_t, "sequence_id": video_id, "track_id": track_id}
        if self.query_wide_factor is not None:
            wide_t = self.transform(image.crop(_expand_bbox(obs.bbox, self.query_wide_factor, image.size)))
            item["image_wide"] = wide_t
        return item


def collate_flat_views(batch: List[dict]) -> dict:
    """Apila `image` (y `image_wide` si está presente); `sequence_id`/
    `track_id` quedan como listas de strings (no tensorizables) -- ver
    report_batch_composition y batch_hard_triplet_loss, que las consumen
    directamente así."""
    out = {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "sequence_id": [b["sequence_id"] for b in batch],
        "track_id": [b["track_id"] for b in batch],
    }
    if "image_wide" in batch[0]:
        out["image_wide"] = torch.stack([b["image_wide"] for b in batch], dim=0)
    return out


class HierarchicalPKSampler(Sampler[List[int]]):
    """BatchSampler de 3 niveles sobre un FlatTrackViewDataset: cada batch =
    P secuencias x K tracks/secuencia x N vistas/track (batch_size máximo =
    P*K*N -- ver más abajo cuándo puede salir más chico).

    Nivel secuencia (P): SIN reemplazo dentro de una época -- se agotan las
    secuencias disponibles antes de repetir ninguna (mismo criterio que un
    BatchSampler estándar: si el total de secuencias utilizables no es
    múltiplo de P, el resto de la última tanda se descarta, como
    drop_last=True).

    Nivel track (K): sin reemplazo dentro de cada secuencia si hay >= K
    tracks disponibles ahí. Si hay menos:
      - `allow_replacement_within_sequence=True` (default): completa
        sampleando CON reemplazo dentro de esa misma secuencia (algún
        track se repite en el batch). La secuencia entra igual al pool de
        secuencias utilizables de la época.
      - `allow_replacement_within_sequence=False`: esa secuencia queda
        FUERA del pool de secuencias utilizables (nunca se elige) -- ver
        `len()`, que ya cuenta solo las utilizables.

    Nivel vista (N): mismo criterio, pero sobre las observaciones de UN
    track y controlado por `allow_replacement_within_track`. A diferencia
    del nivel secuencia, este filtro NO se puede aplicar por adelantado (no
    se sabe qué K tracks tocan hasta samplearlos), así que con
    `allow_replacement_within_track=False` un track corto simplemente se
    OMITE del batch en formación -- el batch puede salir con menos de
    P*K*N vistas en ese caso (documentado, no es un bug: `collate_flat_views`
    soporta cualquier tamaño de batch).

    `seed`: si se fija, la baraja de secuencias/tracks/vistas es
    reproducible entre épocas (útil para debug/tests); si no, usa el
    estado global de `random`, igual que el resto del dataloader del
    proyecto (ver ReIDEpisodeDataset.seed en reid_dataloader.py).
    """

    def __init__(
        self,
        dataset: FlatTrackViewDataset,
        P: int,
        K: int,
        N: int,
        allow_replacement_within_sequence: bool = True,
        allow_replacement_within_track: bool = True,
        seed: Optional[int] = None,
    ):
        if P < 1 or K < 1 or N < 1:
            raise ValueError(f"P, K, N deben ser >= 1 (recibido P={P}, K={K}, N={N})")
        self.dataset = dataset
        self.P, self.K, self.N = P, K, N
        self.allow_replacement_within_sequence = allow_replacement_within_sequence
        self.allow_replacement_within_track = allow_replacement_within_track
        self.seed = seed
        self._epoch = 0

        self._usable_sequences = [
            seq for seq, tracks in dataset.by_sequence.items()
            if allow_replacement_within_sequence or len(tracks) >= K
        ]
        if len(self._usable_sequences) < P:
            raise ValueError(
                f"Solo hay {len(self._usable_sequences)} secuencia(s) utilizable(s) "
                f"({'todas' if allow_replacement_within_sequence else f'con >= {K} tracks'}) "
                f"pero se pidió P={P} por batch."
            )

    def _rng(self) -> random.Random:
        if self.seed is None:
            return random  # estado global -- mismo criterio que ReIDEpisodeDataset sin seed
        return random.Random(self.seed + self._epoch)

    def __len__(self) -> int:
        return len(self._usable_sequences) // self.P

    def _sample_track_ids(self, rng, tracks_dict: Dict[str, List[int]]) -> Optional[List[str]]:
        track_ids = list(tracks_dict.keys())
        if len(track_ids) >= self.K:
            return rng.sample(track_ids, self.K)
        if self.allow_replacement_within_sequence:
            return [rng.choice(track_ids) for _ in range(self.K)]
        return None  # no debería pasar: _usable_sequences ya filtró esto

    def _sample_view_indices(self, rng, global_idxs: List[int]) -> Optional[List[int]]:
        if len(global_idxs) >= self.N:
            return rng.sample(global_idxs, self.N)
        if self.allow_replacement_within_track:
            return [rng.choice(global_idxs) for _ in range(self.N)]
        return None  # track descartado para este batch (ver docstring de la clase)

    def __iter__(self):
        rng = self._rng()
        self._epoch += 1

        sequences = list(self._usable_sequences)
        rng.shuffle(sequences)
        n_batches = len(sequences) // self.P

        for b in range(n_batches):
            batch_indices: List[int] = []
            for seq in sequences[b * self.P: (b + 1) * self.P]:
                tracks_dict = self.dataset.by_sequence[seq]
                track_ids = self._sample_track_ids(rng, tracks_dict)
                if track_ids is None:
                    continue
                for track_id in track_ids:
                    view_idxs = self._sample_view_indices(rng, tracks_dict[track_id])
                    if view_idxs is None:
                        continue
                    batch_indices.extend(view_idxs)
            yield batch_indices


class TrackAnchorPool:
    """Adaptador liviano de DamageTrackIndex a la estructura `by_sequence`
    que espera HierarchicalPKSampler (video_id -> track_id -> [índices
    globales]), pero a granularidad de TRACK COMPLETO -- no de vista
    individual como FlatTrackViewDataset. Pensado para controlar, vía
    HierarchicalPKSampler(pool, ..., N=1), qué track ancla cada episodio de
    un batch de ReIDEpisodeDataset (ver `pk_driven` en reid_dataloader.py):
    acá no importa QUÉ vista se elige (eso lo decide ReIDEpisodeDataset
    internamente -- historial del banco, split real/sintético, query), solo
    QUÉ track.

    Clave de la integración: `self.tracks` es el MISMO objeto (mismo orden)
    que `track_index.tracks`, así que el índice global que el sampler
    asigna a un track coincide exactamente con su posición en
    `track_index.tracks` -- `ReIDEpisodeDataset.__getitem__(idx)` con
    `pk_driven=True` hace justamente `track_index.tracks[idx]`. Si se
    reconstruye o reordena `track_index` después de crear el pool, esa
    correspondencia se rompe -- usar `build_track_pk_sampler` (más abajo)
    para no tener que llevar la cuenta a mano."""

    def __init__(self, track_index: DamageTrackIndex):
        self.tracks: List = track_index.tracks
        by_sequence: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for i, track in enumerate(self.tracks):
            by_sequence[track.video_id][track.identity_id].append(i)
        self.by_sequence: Dict[str, Dict[str, List[int]]] = {
            seq: dict(tracks) for seq, tracks in by_sequence.items()
        }

    def __len__(self) -> int:
        return len(self.tracks)


def build_track_pk_sampler(
    track_index: DamageTrackIndex,
    P: int,
    K: int,
    allow_replacement_within_sequence: bool = True,
    seed: Optional[int] = None,
) -> HierarchicalPKSampler:
    """Arma el HierarchicalPKSampler que controla, por batch de
    ReIDEpisodeDataset, qué P secuencias x K tracks anclan sus episodios
    (ver `pk_driven` en reid_dataloader.py) -- reemplaza el
    sample_random_identity() completamente al azar por composición
    controlada, sin tocar el resto de la construcción del episodio (banco
    de distractores vía hard_distractor_ratio, casos
    REAL_REID/SYNTHETIC_SPLIT/NEW_TRACK). `N=1` fijo: acá "vista" no
    aplica -- cada track del batch ancla exactamente un episodio.
    `allow_replacement_within_track` no tiene efecto con N=1 (una lista de
    un solo elemento siempre "alcanza").

    Usar el `track_index` YA CONSTRUIDO (build_track_index en
    train_track_query_reid.py) y pasar el sampler resultante como
    `batch_sampler` de un DataLoader sobre un ReIDEpisodeDataset(
    pk_driven=True) construido con ESE MISMO `track_index` -- ver
    docstring de TrackAnchorPool sobre por qué debe ser el mismo objeto."""
    pool = TrackAnchorPool(track_index)
    return HierarchicalPKSampler(
        pool, P=P, K=K, N=1,
        allow_replacement_within_sequence=allow_replacement_within_sequence,
        allow_replacement_within_track=True,  # sin efecto con N=1
        seed=seed,
    )


def report_batch_composition(sequence_ids: Sequence[str], track_ids: Sequence[str]) -> dict:
    """Diagnóstico de un batch ya generado (ver HierarchicalPKSampler):
    cuántas secuencias distintas, cuántos tracks distintos por secuencia, y
    para cada ancla del batch, cuántos negativos (distinta track_id) son
    intra-secuencia (mismo sequence_id) vs inter-secuencia (distinto
    sequence_id) -- y cuántos positivos (misma track_id) hay disponibles.
    Toma directamente `sequence_id`/`track_id` tal como los devuelve
    collate_flat_views (o cualquier batch real una vez conectado) -- no
    depende del dataset ni de índices globales."""
    sequence_ids = list(sequence_ids)
    track_ids = list(track_ids)
    if len(sequence_ids) != len(track_ids):
        raise ValueError("sequence_ids y track_ids deben tener la misma longitud")

    tracks_per_sequence: Dict[str, set] = defaultdict(set)
    for seq_id, track_id in zip(sequence_ids, track_ids):
        tracks_per_sequence[seq_id].add(track_id)

    per_anchor = []
    for a in range(len(track_ids)):
        n_pos = n_intra_neg = n_inter_neg = 0
        for b in range(len(track_ids)):
            if a == b:
                continue
            if track_ids[b] == track_ids[a]:
                n_pos += 1
            elif sequence_ids[b] == sequence_ids[a]:
                n_intra_neg += 1
            else:
                n_inter_neg += 1
        per_anchor.append({"positives": n_pos, "intra_seq_negatives": n_intra_neg, "inter_seq_negatives": n_inter_neg})

    n = len(per_anchor)
    return {
        "batch_size": n,
        "n_sequences": len(tracks_per_sequence),
        "tracks_per_sequence": {seq: len(tracks) for seq, tracks in tracks_per_sequence.items()},
        "per_anchor": per_anchor,
        "avg_positives": sum(a["positives"] for a in per_anchor) / n if n else 0.0,
        "avg_intra_seq_negatives": sum(a["intra_seq_negatives"] for a in per_anchor) / n if n else 0.0,
        "avg_inter_seq_negatives": sum(a["inter_seq_negatives"] for a in per_anchor) / n if n else 0.0,
    }


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    track_ids: Sequence[str],
    sequence_ids: Optional[Sequence[str]] = None,
    margin: float = 0.3,
    intra_seq_bonus: float = 0.0,
) -> torch.Tensor:
    """Batch-hard triplet mining (Hermans et al. 2017): para cada ancla del
    batch, mina DENTRO del batch actual (no de todo el dataset) el positivo
    más lejano (misma track_id, mayor distancia) y el negativo más cercano
    (distinta track_id, menor distancia), y aplica el margen sobre esas dos
    distancias. `margin > 0`: margen duro clásico, relu(margin + d_ap -
    d_an). `margin <= 0`: margen suave (softplus(d_ap - d_an)), sin
    hiperparámetro extra, tal como proponen Hermans et al. como alternativa.

    Como HierarchicalPKSampler ya garantiza negativos intra-secuencia en el
    batch (ver report_batch_composition), el mining por distancia pura ya
    los prioriza naturalmente cuando son los más difíciles de verdad -- sin
    necesitar lógica adicional. `intra_seq_bonus` (default 0.0 = mining
    puro por distancia, comportamiento estándar de Hermans et al.) es la
    válvula opcional pedida para el caso en que eso no alcance en la
    práctica: si > 0, resta ese valor a la distancia de los negativos
    INTRA-secuencia (misma sequence_id, distinta track_id) antes de elegir
    el más cercano -- sesga la selección hacia ellos aunque no sean
    todavía los más cercanos en el espacio de embeddings. Requiere pasar
    `sequence_ids` si se usa.

    Devuelve el escalar promediado sobre las anclas con al menos un
    positivo Y un negativo válidos en el batch (con el sampler, todas las
    anclas deberían tenerlos salvo K=1 o P*K=1)."""
    if intra_seq_bonus > 0 and sequence_ids is None:
        raise ValueError("intra_seq_bonus > 0 requiere pasar sequence_ids")

    B = embeddings.shape[0]
    track_ids = list(track_ids)
    dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

    same_track = torch.tensor(
        [[track_ids[i] == track_ids[j] for j in range(B)] for i in range(B)],
        dtype=torch.bool, device=embeddings.device,
    )
    self_mask = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    pos_mask = same_track & ~self_mask
    neg_mask = ~same_track

    valid_anchor = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid_anchor.any():
        raise RuntimeError(
            "Ningún ancla del batch tiene a la vez un positivo (misma track_id) y un "
            "negativo (distinta track_id) -- revisa K (>=2 para tener positivos) y "
            "P*K (>=2 identidades distintas para tener negativos)."
        )

    d_ap = dist.masked_fill(~pos_mask, float("-inf")).max(dim=1).values

    mining_dist = dist
    if intra_seq_bonus > 0:
        sequence_ids = list(sequence_ids)
        same_seq = torch.tensor(
            [[sequence_ids[i] == sequence_ids[j] for j in range(B)] for i in range(B)],
            dtype=torch.bool, device=embeddings.device,
        )
        intra_neg_mask = neg_mask & same_seq
        mining_dist = dist - intra_seq_bonus * intra_neg_mask.float()
    d_an = mining_dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values

    per_anchor_loss = (
        F.relu(margin + d_ap - d_an) if margin > 0 else F.softplus(d_ap - d_an)
    )
    return per_anchor_loss[valid_anchor].mean()


if __name__ == "__main__":
    # Smoke test con índices sintéticos (sin imágenes reales en disco, ver
    # _DummyFlatTrackViewDataset) -- confirma la composición del batch y
    # que la loss corre sin errores, ANTES de conectar nada al dataset
    # real (ver docstring del módulo).
    from reid_dataloader import IdentityTrack

    class _DummyFlatTrackViewDataset(FlatTrackViewDataset):
        def __init__(self, track_index: DamageTrackIndex):
            self.items = []
            by_sequence = defaultdict(lambda: defaultdict(list))
            for track in track_index.tracks:
                for obs in track.observations:
                    gidx = len(self.items)
                    self.items.append((track.video_id, track.identity_id, obs))
                    by_sequence[track.video_id][track.identity_id].append(gidx)
            self.by_sequence = {seq: dict(tracks) for seq, tracks in by_sequence.items()}

        def __getitem__(self, idx):
            video_id, track_id, _ = self.items[idx]
            return {"image": torch.randn(3, 8, 8), "sequence_id": video_id, "track_id": track_id}

    def _make_index(n_videos, tracks_per_video, obs_per_track_fn):
        tracks = []
        for v in range(n_videos):
            for i in range(tracks_per_video(v) if callable(tracks_per_video) else tracks_per_video):
                n_obs = obs_per_track_fn(v, i) if callable(obs_per_track_fn) else obs_per_track_fn
                obs_list = [
                    TrackObservation(frame_idx=f, image_path="dummy.jpg", bbox=(0, 0, 100, 100), part_label=f % 4)
                    for f in range(n_obs)
                ]
                tracks.append(IdentityTrack(identity_id=f"v{v}_id{i}", video_id=f"video_{v}", observations=obs_list))
        return DamageTrackIndex(tracks)

    from torch.utils.data import DataLoader

    print("=== Caso 1: composición homogénea (5 videos x 6 tracks x 10 obs), P=3 K=3 N=4 ===")
    index = _make_index(n_videos=5, tracks_per_video=6, obs_per_track_fn=10)
    dataset = _DummyFlatTrackViewDataset(index)
    P, K, N = 3, 3, 4
    sampler = HierarchicalPKSampler(dataset, P=P, K=K, N=N, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_flat_views)
    print(f"[INFO] {len(sampler)} batches/época (5 secuencias // P={P}), batch_size esperado = P*K*N = {P*K*N}")

    batch = next(iter(loader))
    print(f"image {tuple(batch['image'].shape)}")
    report = report_batch_composition(batch["sequence_id"], batch["track_id"])
    print(f"batch_size real={report['batch_size']}  n_sequences={report['n_sequences']}")
    print("tracks_per_sequence:", report["tracks_per_sequence"])
    print(f"avg positivos/ancla={report['avg_positives']:.2f}  "
          f"avg intra-seq negativos/ancla={report['avg_intra_seq_negatives']:.2f}  "
          f"avg inter-seq negativos/ancla={report['avg_inter_seq_negatives']:.2f}")
    assert report["batch_size"] == P * K * N
    assert report["n_sequences"] == P
    assert all(v == K for v in report["tracks_per_sequence"].values())
    assert all(a["positives"] == N - 1 for a in report["per_anchor"])
    assert all(a["intra_seq_negatives"] == (K - 1) * N for a in report["per_anchor"])
    assert all(a["inter_seq_negatives"] == (P - 1) * K * N for a in report["per_anchor"])
    print("[OK] composición P x K x N exacta, como se esperaba con datos homogéneos")

    embeddings = torch.randn(batch["image"].shape[0], 16)
    loss = batch_hard_triplet_loss(embeddings, batch["track_id"], batch["sequence_id"])
    loss_biased = batch_hard_triplet_loss(embeddings, batch["track_id"], batch["sequence_id"], intra_seq_bonus=0.5)
    assert torch.isfinite(loss) and torch.isfinite(loss_biased)
    print(f"batch-hard triplet loss (embeddings random, solo shape/no-NaN): {loss.item():.4f}")
    print(f"batch-hard triplet loss (intra_seq_bonus=0.5): {loss_biased.item():.4f}")

    print()
    print("=== Caso 2: una secuencia con pocos tracks (K pedido=3, disponibles=2) ===")
    scarce_tracks = [
        IdentityTrack(
            identity_id=f"scarce_id{i}", video_id="video_scarce",
            observations=[
                TrackObservation(frame_idx=f, image_path="dummy.jpg", bbox=(0, 0, 100, 100), part_label=0)
                for f in range(10)
            ],
        )
        for i in range(2)  # solo 2 tracks, < K=3
    ]
    rich_index = _make_index(n_videos=4, tracks_per_video=6, obs_per_track_fn=10)
    combined_index = DamageTrackIndex(rich_index.tracks + scarce_tracks)
    dataset2 = _DummyFlatTrackViewDataset(combined_index)

    sampler_norepl = HierarchicalPKSampler(dataset2, P=3, K=3, N=4, allow_replacement_within_sequence=False, seed=0)
    print(f"[INFO] allow_replacement_within_sequence=False -> {len(sampler_norepl._usable_sequences)}/5 secuencias "
          f"utilizables (video_scarce, con 2 tracks < K=3, queda afuera)")
    assert "video_scarce" not in sampler_norepl._usable_sequences

    sampler_repl = HierarchicalPKSampler(dataset2, P=3, K=3, N=4, allow_replacement_within_sequence=True, seed=0)
    print(f"[INFO] allow_replacement_within_sequence=True  -> {len(sampler_repl._usable_sequences)}/5 secuencias "
          f"utilizables (video_scarce entra, completando K con tracks repetidos)")
    assert "video_scarce" in sampler_repl._usable_sequences

    print()
    print("=== Caso 3: un track corto (N pedido=4, disponibles=2) dentro de una secuencia normal ===")
    short_track_tracks = [
        IdentityTrack(
            identity_id="short_id", video_id="video_shorttrack",
            observations=[
                TrackObservation(frame_idx=f, image_path="dummy.jpg", bbox=(0, 0, 100, 100), part_label=0)
                for f in range(2)  # solo 2 vistas, < N=4
            ],
        )
    ] + [
        IdentityTrack(
            identity_id=f"normal_id{i}", video_id="video_shorttrack",
            observations=[
                TrackObservation(frame_idx=f, image_path="dummy.jpg", bbox=(0, 0, 100, 100), part_label=0)
                for f in range(10)
            ],
        )
        for i in range(5)
    ]
    dataset3 = _DummyFlatTrackViewDataset(DamageTrackIndex(short_track_tracks))
    rng_probe = random.Random(0)
    short_global_idxs = dataset3.by_sequence["video_shorttrack"]["short_id"]

    sampler3_no = HierarchicalPKSampler(dataset3, P=1, K=1, N=4, allow_replacement_within_track=False, seed=0)
    view_idxs_no = sampler3_no._sample_view_indices(rng_probe, short_global_idxs)
    assert view_idxs_no is None
    print("[INFO] allow_replacement_within_track=False -> track corto se omite del batch (None)")

    sampler3_yes = HierarchicalPKSampler(dataset3, P=1, K=1, N=4, allow_replacement_within_track=True, seed=0)
    view_idxs_yes = sampler3_yes._sample_view_indices(rng_probe, short_global_idxs)
    assert view_idxs_yes is not None and len(view_idxs_yes) == 4
    print(f"[INFO] allow_replacement_within_track=True  -> completa con reemplazo: {view_idxs_yes}")

    print()
    print("=== Caso 4: TrackAnchorPool/build_track_pk_sampler -- integración con ReIDEpisodeDataset ===")
    # (ver pk_driven en reid_dataloader.py): acá NO se pasa por
    # FlatTrackViewDataset/collate_flat_views -- el sampler itera
    # directamente y se resuelve cada índice global contra
    # pool.tracks[i] (== track_index.tracks[i], la invariante que explota
    # ReIDEpisodeDataset.pk_driven), sin necesidad de imágenes ni de un
    # segundo Dataset/DataLoader.
    index4 = _make_index(n_videos=4, tracks_per_video=5, obs_per_track_fn=10)
    pool = TrackAnchorPool(index4)
    P4, K4 = 3, 2
    pk_sampler = build_track_pk_sampler(index4, P=P4, K=K4, seed=0)
    print(f"[INFO] {len(pk_sampler)} batches/época (4 secuencias // P={P4}), "
          f"batch_size esperado = P*K = {P4 * K4}")

    pk_batch = next(iter(pk_sampler))
    resolved = [(pool.tracks[i].video_id, pool.tracks[i].identity_id) for i in pk_batch]
    seq_ids, track_ids = zip(*resolved)
    pk_report = report_batch_composition(seq_ids, track_ids)
    print(f"batch_size real={pk_report['batch_size']}  n_sequences={pk_report['n_sequences']}")
    print("tracks_per_sequence:", pk_report["tracks_per_sequence"])
    assert pk_report["batch_size"] == P4 * K4
    assert pk_report["n_sequences"] == P4
    assert all(v == K4 for v in pk_report["tracks_per_sequence"].values())
    assert len(set(pk_batch)) == len(pk_batch)  # sin reemplazo (4 secuencias con 5 tracks c/u, alcanza sin repetir)
    print("[OK] TrackAnchorPool/build_track_pk_sampler resuelven índices únicos con la composición P x K esperada")

    print()
    print("[OK] smoke test completado sin errores")
