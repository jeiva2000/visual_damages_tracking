"""
Dataloader para entrenar TrackReIDModel (ver set_reid_model.py).

Idea central: los tracks que ByteTrack/BoT-SORT ya generan (o mejor, tracks
con identidad GT independiente, si la tienes vía anotación) son la fuente de
verdad. De cada identidad se pueden extraer N observaciones (crops) a lo
largo del video. Un "episodio" de entrenamiento = un banco (galería) de
tracks perdidos, cada uno con hasta `num_track_history` vistas de su
historial, + UNA sola query -- la detección nueva, con una única vista.

Problema que resuelve este archivo: los casos reales de ReID (un track se
pierde por oclusión y reaparece más tarde con la MISMA identidad) son
escasos frente a los casos triviales (track nuevo sin relación con nada
anterior). Sin corrección, el dataset natural underrepresenta justo el caso
que nos interesa optimizar. Se usan tres mecanismos para compensarlo:

  1. REAL_REID: cuando el track tiene una brecha real en el tiempo
     (oclusión/salida de cuadro) en los datos, se usa tal cual: segmento
     antes de la brecha -> banco, segmento después -> query positiva.
  2. SYNTHETIC_SPLIT: cuando un track es continuo (sin brecha real) pero
     suficientemente largo, se corta artificialmente en un punto aleatorio,
     simulando una pérdida. Esto multiplica los positivos disponibles sin
     depender de que ocurran oclusiones reales en el video.
  3. Ratios de sampling configurables (`case_probs`) que se aplican por
     episodio, INDEPENDIENTES de la frecuencia natural en los datos. Es
     decir: el sampler decide de antemano qué proporción de la próxima
     query será positiva/negativa-dura/negativa-nueva, y busca en el índice
     un caso que cumpla esa condición, en vez de simplemente iterar el
     dataset en su distribución natural.

Los distractores del banco (negativos) se seleccionan con dos niveles de
dificultad:
  - hard: misma parte del vehículo (part_label) o mismo video/ventana
    temporal, identidad distinta -> fuerza al modelo a discriminar más allá
    de "se parece" y "está cerca en el tiempo".
  - easy: identidades de otros videos, para dar también negativos triviales
    y no sobre-optimizar solo para el caso difícil.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Para TRAIN (nunca para val/test, ver train_track_query_reid.py): el
# encoder congelado + sin ninguna aumentación hace que la parte entrenable
# memorice rápido detalles de iluminación/color específicos de los videos
# de train. Deliberadamente NO se toca geometría (sin flip ni resized-crop
# aleatorio): las 3 escalas ya son fijas y con semántica propia (amplio /
# intermedio / ajustado, ver scale_embedding), y flipear invalidaría el
# bbox relativo ya calculado (SpatialEncoding) si no se ajusta en conjunto.
# ColorJitter no toca coordenadas, así que es seguro de aplicar sin más.
AUGMENTED_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Factores de padding para los recortes multi-escala. Orden: amplio -> intermedio
# -> ajustado. La idea es que el primer recorte (contexto fuerte) le dé al
# modelo información de UBICACIÓN dentro del vehículo (qué pieza, qué zona),
# y el último recorte (ajustado) le dé el DETALLE fino del daño en sí.
# 3 escalas en vez de 5: menos pasadas por el encoder por detección (3x en
# vez de 5x), relevante para entrenar en la RTX 2080 local.
DEFAULT_PADDING_FACTORS = [6.0, 1.6, 1.0]  # [amplio, intermedio, ajustado]

# Tamaño mínimo absoluto (en píxeles, sobre la imagen original) para el
# recorte de contexto amplio. Un factor puramente relativo (ej. 6x el bbox)
# no sirve de nada si el daño es pequeño: 6x un bbox de 20x20px sigue siendo
# minúsculo y no alcanza a mostrar ningún elemento que ancle la posición
# (borde de puerta, faro, manija). Este mínimo absoluto fuerza a que el
# crop de contexto capture suficiente superficie del vehículo sin importar
# qué tan chico sea el daño original.
MIN_CONTEXT_CROP_SIZE_PX = 400


def _expand_bbox(
    bbox: Tuple[float, float, float, float],
    factor: float,
    image_size: Tuple[int, int],
    min_size_px: float = 0.0,
):
    """Expande un bbox (x1,y1,x2,y2) desde su centro por `factor`, garantizando
    además un tamaño mínimo absoluto `min_size_px` (útil para el crop de
    contexto amplio en daños pequeños), clippeado a los límites de la imagen."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) / 2 * factor, min_size_px / 2)
    half_h = max((y2 - y1) / 2 * factor, min_size_px / 2)
    img_w, img_h = image_size
    new_x1 = max(0, cx - half_w)
    new_y1 = max(0, cy - half_h)
    new_x2 = min(img_w, cx + half_w)
    new_y2 = min(img_h, cy + half_h)
    return (new_x1, new_y1, new_x2, new_y2)


def _spread_indices(length: int, k: int, jitter_ratio: float = 0.0) -> List[int]:
    """Hasta `k` índices de un track de `length` observaciones, repartidos lo
    más uniforme posible en el tiempo (siempre incluye el primero y el
    último si length >= k) -- para tomar observaciones DISTINTAS del
    historial del track en vez de solo las últimas. Si length <= k,
    devuelve todos los índices disponibles (el resto se enmascara).

    `jitter_ratio` (default 0.0 = determinista, comportamiento actual): con
    valor > 0, cada índice objetivo se corre un poco al azar (hasta
    `jitter_ratio * espaciado_entre_puntos`) -- así el MISMO track no
    muestra siempre exactamente las mismas observaciones entre episodios.
    Sin esto, con pocos tracks, el modelo puede memorizar "estas 4 imágenes
    exactas = este id" en vez de aprender similitud visual genérica."""
    if length <= k:
        return list(range(length))
    if k == 1:
        idx = length - 1
        if jitter_ratio > 0:
            idx = max(0, idx - random.randint(0, round(jitter_ratio * length)))
        return [idx]

    spacing = (length - 1) / (k - 1)
    radius = round(jitter_ratio * spacing)
    idxs = set()
    for i in range(k):
        target = i * spacing
        if radius > 0:
            target += random.randint(-radius, radius)
        idxs.add(max(0, min(length - 1, round(target))))
    return sorted(idxs)


class PrecomputedEmbeddingCache:
    """Embeddings de DINOv2 ya calculados para cada observación de un
    DamageTrackIndex -- ver `precompute-cache` en train_track_query_reid.py.

    Solo tiene sentido mientras el encoder esté CONGELADO: si se descongela
    (fine-tuning), los embeddings cambian en cada paso y el cache queda
    inválido de inmediato. Tampoco es compatible con aumentación (ColorJitter
    cambia los píxeles en cada llamada, así que ya no habría un embedding
    único por observación que cachear) -- por eso el cache siempre se
    construye con el transform SIN aumentar, aunque el split sea train.
    Pensado para iterar rápido en experimentos (comparar hiperparámetros,
    criterio de selección de checkpoint, backbone fine-tuneado) sin correr
    DINOv2 en cada época -- no para la corrida final "de verdad", que sí
    quiere aumentación en train.

    Clave: `(image_path, bbox)` -- identifica una observación de forma
    única independientemente de a qué track/episodio pertenezca. Valor:
    tensor `(num_scales, encoder_dim)` -- los `num_scales` embeddings de
    escala de esa observación (el diseño actual -- ver set_reid_model.py --
    no usa memoria/frame completo, solo matching directo entre vistas)."""

    def __init__(self, embeddings: Dict[Tuple[str, Tuple[float, float, float, float]], torch.Tensor]):
        self.embeddings = embeddings

    def get(self, obs: "TrackObservation") -> torch.Tensor:
        key = (obs.image_path, tuple(obs.bbox))
        cached = self.embeddings.get(key)
        if cached is None:
            raise KeyError(
                f"No hay embedding cacheado para {key} -- ¿el cache corresponde a este "
                f"mismo split/dataset, o se reconstruyó el índice después de generarlo?"
            )
        return cached  # (S, encoder_dim)

    def save(self, path: str):
        torch.save({"embeddings": self.embeddings}, path)

    @classmethod
    def load(cls, path: str) -> "PrecomputedEmbeddingCache":
        data = torch.load(path, map_location="cpu")
        return cls(data["embeddings"])


class QueryCase(Enum):
    REAL_REID = "real_reid"            # track con brecha real, misma identidad reaparece
    SYNTHETIC_SPLIT = "synthetic_split"  # track continuo cortado artificialmente
    NEW_TRACK = "new_track"            # identidad genuinamente ausente del banco


@dataclass
class TrackObservation:
    frame_idx: int
    image_path: str          # ruta al frame completo
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) en píxeles
    part_label: int


@dataclass
class IdentityTrack:
    identity_id: str
    video_id: str
    observations: List[TrackObservation] = field(default_factory=list)  # ordenadas por frame_idx

    def find_real_gaps(self, min_gap_frames: int = 3) -> List[int]:
        """Índices donde hay una brecha temporal real (oclusión) >= min_gap_frames.
        Depende de que quien construye el índice ya haya excluido las
        observaciones ocluidas (visibility<=0) en vez de conservarlas -- el
        hueco resultante en frame_idx es la señal (ver build_track_index en
        train_track_query_reid.py, que filtra por `visibility` del gt.txt)."""
        gaps = []
        for i in range(1, len(self.observations)):
            delta = self.observations[i].frame_idx - self.observations[i - 1].frame_idx
            if delta >= min_gap_frames:
                gaps.append(i)
        return gaps


class DamageTrackIndex:
    """Índice de todas las identidades disponibles, organizado por video, para
    hacer sampling eficiente de distractores por parte/video/proximidad."""

    def __init__(self, tracks: List[IdentityTrack]):
        self.tracks = tracks
        self.by_video: Dict[str, List[IdentityTrack]] = {}
        self.by_part: Dict[int, List[IdentityTrack]] = {}
        for t in tracks:
            self.by_video.setdefault(t.video_id, []).append(t)
            parts = {obs.part_label for obs in t.observations}
            for p in parts:
                self.by_part.setdefault(p, []).append(t)

    def sample_hard_distractors(self, exclude_identity: IdentityTrack, k: int) -> List[IdentityTrack]:
        """Prioriza distractores del mismo video y, si comparten part_label
        con `exclude_identity`, mejor (más difícil)."""
        same_video = [
            t for t in self.by_video.get(exclude_identity.video_id, [])
            if t.identity_id != exclude_identity.identity_id
        ]
        random.shuffle(same_video)
        return same_video[:k]

    def sample_easy_distractors(self, exclude_identity: IdentityTrack, k: int) -> List[IdentityTrack]:
        """Distractores de otros videos: negativos triviales, para no
        sobre-optimizar solo el caso difícil."""
        other_video_tracks = [t for t in self.tracks if t.video_id != exclude_identity.video_id]
        random.shuffle(other_video_tracks)
        return other_video_tracks[:k]

    def sample_random_identity(self, exclude: Optional[IdentityTrack] = None) -> IdentityTrack:
        candidates = self.tracks if exclude is None else [
            t for t in self.tracks if t.identity_id != exclude.identity_id
        ]
        return random.choice(candidates)


class ReIDEpisodeDataset(Dataset):
    """
    Cada __getitem__ genera UN episodio:
      - un banco (galería) de hasta `max_bank_size` tracks candidatos, cada
        uno con hasta `num_track_history` vistas de su historial
      - UNA sola query -- la detección nueva, con una única vista (no tiene
        historial propio todavía) -- ver set_reid_model.py
      - `target`: índice del track candidato correcto (0..max_bank_size-1)
        si la query reaparece, o `max_bank_size` si es genuinamente nueva
        (clase "no match")

    `episodes_per_epoch` es un número virtual (no depende del tamaño real
    de datos): el sampling es generativo, así que puedes fijar cuántos
    episodios quieres ver por epoch.
    """

    def __init__(
        self,
        track_index: DamageTrackIndex,
        episodes_per_epoch: int = 5000,
        max_bank_size: int = 16,
        num_track_history: int = 4,
        history_jitter_ratio: float = 0.0,
        case_probs: Optional[Dict[QueryCase, float]] = None,
        hard_distractor_ratio: float = 0.6,
        min_track_len_for_split: int = 6,
        min_gap_frames: int = 3,
        synthetic_gap_max: int = 0,
        query_wide_factor: float = 2.5,
        padding_factors: Optional[List[float]] = None,
        transform=None,
        return_debug_info: bool = False,
        return_video_ids: bool = False,
        force_reid: bool = False,
        seed: Optional[int] = None,
        embedding_cache: Optional[PrecomputedEmbeddingCache] = None,
    ):
        self.track_index = track_index
        self.episodes_per_epoch = episodes_per_epoch
        self.max_bank_size = max_bank_size
        # cuántas observaciones distintas del historial de un track (no solo
        # la última) se usan para armar su slot del banco -- ver
        # _load_track_history_crops. Por ahora, PRUEBA: sin padding espacial
        # (cada observación aporta un solo recorte ajustado, factor=1.0);
        # eso se decide aparte de esto.
        self.num_track_history = num_track_history
        # 0.0 (default) = selección determinista de las num_track_history
        # observaciones (siempre las mismas para el mismo track, ver
        # _spread_indices). >0: jitter aleatorio para que el mismo track no
        # muestre siempre exactamente las mismas imágenes entre episodios --
        # evita memorizar "estas imágenes exactas = este id" con pocos tracks.
        self.history_jitter_ratio = history_jitter_ratio
        # factor de expansión del bbox para la vista AMPLIA de la query (ver
        # _load_query_views) -- contexto de pieza (líneas de panel, bordes,
        # paso de rueda) para desambiguar daños de apariencia local parecida
        # en distintas ubicaciones del vehículo (no hay piezas etiquetadas
        # para resolver esto con clasificación, así que se agrega vía una
        # segunda vista más amplia en vez de una señal categórica). Puede
        # necesitar tuning según el tamaño típico del daño vs. la pieza.
        self.query_wide_factor = query_wide_factor
        self.padding_factors = padding_factors or DEFAULT_PADDING_FACTORS
        self.num_scales = len(self.padding_factors)
        self.case_probs = case_probs or {
            QueryCase.REAL_REID: 0.2,
            QueryCase.SYNTHETIC_SPLIT: 0.4,
            QueryCase.NEW_TRACK: 0.4,
        }
        self.hard_distractor_ratio = hard_distractor_ratio
        self.min_track_len_for_split = min_track_len_for_split
        self.min_gap_frames = min_gap_frames
        # 0 (default) = comportamiento actual, corte en un solo punto (before/
        # after quedan en frames consecutivos, casi sin diferencia de
        # apariencia). > 0: además de cortar, se descarta una ventana de
        # 1..synthetic_gap_max observaciones alrededor del corte -- simula la
        # ESTRUCTURA de una oclusión real (un hueco en el tiempo) para los
        # tracks que no tienen ninguna brecha real, ver _try_synthetic_split.
        self.synthetic_gap_max = synthetic_gap_max
        self.transform = transform or DEFAULT_TRANSFORM
        # solo para inspección visual (ver train_track_query_reid.py): agrega
        # al batch la ruta+bbox absoluto (sin normalizar) de la observación
        # que generó la query principal, para poder mostrar el frame completo
        # sin recortar -- no se usa para entrenar, así que apagado por defecto.
        self.return_debug_info = return_debug_info
        # si se fija, cada episodio agrega `bank_video_ids` (video_id de cada
        # slot del banco, "" para el padding) y `query_video_id` -- solo para
        # las métricas de confusión intra/inter-secuencia de validación (ver
        # _intra_inter_confusion en train_track_query_reid.py), no para
        # entrenar. Requiere el collate_fn dedicado (ver
        # collate_episodes_with_video_ids), porque son strings, no tensores.
        self.return_video_ids = return_video_ids
        # para el dataset de TEST (ver run_test en train_track_query_reid.py):
        # fuerza a que la query principal SIEMPRE sea un caso real de
        # reidentificación (REAL_REID o SYNTHETIC_SPLIT) -- nunca NEW_TRACK --
        # resampleando identidad de referencia hasta encontrar una que
        # soporte una brecha real o un corte sintético. Train/val NO usan
        # esto: ahí interesa la mezcla natural de casos (incluye NEW_TRACK).
        self.force_reid = force_reid
        # si se fija, cada __getitem__(idx) resetea random.seed(seed + idx)
        # ANTES de samplear nada -- así el episodio `idx` es siempre el mismo,
        # epoch tras epoch. Sin esto (default None, lo que usa train), cada
        # __getitem__ sortea con el estado global de `random`, así que un
        # DataLoader que reconstruye el iterador cada epoch (como hace
        # train_track_query_reid.py) genera episodios DISTINTOS cada vez --
        # perfecto para train (más diversidad), pero hace que val/test no
        # sean un benchmark estable entre epochs/corridas si no se fija.
        self.seed = seed
        # ver PrecomputedEmbeddingCache -- si se pasa, _load_track_history_crops
        # y _load_query_views devuelven embeddings ya calculados en vez de
        # abrir la imagen/recortar/pasar por DINOv2. Solo válido con encoder
        # congelado y sin aumentación (self.transform se ignora en ese caso).
        self.embedding_cache = embedding_cache

    def __len__(self):
        return self.episodes_per_epoch

    def _load_multiscale_crops(self, obs: TrackObservation) -> torch.Tensor:
        """Genera los `num_scales` recortes, orden amplio -> ajustado, y los
        apila (S,C,H,W). Al primer recorte (contexto amplio) se le fuerza
        un tamaño mínimo absoluto para que ubique la posición en el
        vehículo incluso si el daño original es pequeño. Usado SOLO para
        construir el PrecomputedEmbeddingCache (ver run_precompute_cache en
        train_track_query_reid.py) -- durante el entrenamiento normal, el
        historial del banco se arma vía _load_track_history_crops, que solo
        pasa por acá indirectamente a través del cache ya construido."""
        image = Image.open(obs.image_path).convert("RGB")
        img_size = image.size  # (W, H) en PIL
        crops = []
        for i, factor in enumerate(self.padding_factors):
            min_size = MIN_CONTEXT_CROP_SIZE_PX if i == 0 else 0.0
            expanded_bbox = _expand_bbox(obs.bbox, factor, img_size, min_size_px=min_size)
            crops.append(self.transform(image.crop(expanded_bbox)))
        return torch.stack(crops, dim=0)  # (S,C,H,W)

    def _load_query_views(self, obs: TrackObservation) -> torch.Tensor:
        """La detección nueva -- DOS vistas del mismo objeto (no una sola):
          - AJUSTADA (factor=1.0): textura fina del daño, misma receta que
            la última posición del historial del banco.
          - AMPLIA (factor=query_wide_factor): contexto de pieza (líneas de
            panel, bordes, paso de rueda). Sin dibujar nada encima en
            ninguna de las dos.
        Motivo: los daños de este dataset (manchas, rayones, líneas) no son
        suficientemente discriminantes por apariencia local sola -- sin
        contexto de en qué parte del vehículo están, daños parecidos en
        ubicaciones distintas pueden confundirse en el matching. No hay
        piezas etiquetadas para resolver esto con clasificación (y un
        segmentador de piezas metería su propio error), así que se agrega
        contexto vía esta segunda vista en vez de una señal categórica.

        Ambas vistas se procesan con el mismo encoder compartido que el
        resto (banco incluido) y se fusionan entre sí vía IntraTrackEncoder
        -- el mismo módulo que fusiona las vistas del banco -- antes de la
        cross-attention contra la galería (ver TrackReIDModel en
        set_reid_model.py).

        _expand_bbox ya clippea contra los bordes de la imagen -- si el
        daño está cerca del borde, el crop amplio queda recortado de forma
        asimétrica (no centrado en el bbox); no se resuelve acá.

        Con `embedding_cache` activo, la vista amplia queda atada a la
        escala "intermedia" ya precalculada (self.padding_factors[1], NO
        necesariamente == query_wide_factor) -- el cache solo tiene
        `num_scales` embeddings fijos por observación, no puede generar uno
        nuevo con un factor arbitrario bajo demanda."""
        if self.embedding_cache is not None:
            scale_embeds = self.embedding_cache.get(obs)
            wide_idx = 1 if len(scale_embeds) > 1 else 0
            return torch.stack([scale_embeds[-1], scale_embeds[wide_idx]], dim=0)

        image = Image.open(obs.image_path).convert("RGB")
        tight_box = _expand_bbox(obs.bbox, 1.0, image.size)
        wide_box = _expand_bbox(obs.bbox, self.query_wide_factor, image.size)
        query_tight = self.transform(image.crop(tight_box))
        query_wide = self.transform(image.crop(wide_box))
        return torch.stack([query_tight, query_wide], dim=0)  # (2, C, H, W)

    def _sample_case(self) -> QueryCase:
        cases, weights = zip(*self.case_probs.items())
        return random.choices(cases, weights=weights, k=1)[0]

    def _load_track_history_crops(
        self, observations: List[TrackObservation]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hasta `num_track_history` observaciones de ESTE track, repartidas
        uniformemente en el tiempo (ver _spread_indices) -- no solo las
        últimas -- para aprovechar en producción el historial completo que
        ya se tiene de un track perdido. Padding espacial por posición:
          - primera (más antigua de las seleccionadas): recorte AMPLIO fijo
            (mismo factor + mínimo absoluto que la escala 0 de
            _load_multiscale_crops) -- para ubicar el daño en el contexto
            del vehículo (sin esto, con todo ajustado, no hay forma de
            saber en qué parte del vehículo/fondo está si es sutil).
          - última (más reciente): recorte AJUSTADO (factor=1.0), acotado
            al daño puro.
          - las intermedias (si num_track_history > 2): factor de padding
            ALEATORIO por observación (distinto cada vez) para no repetir
            prácticamente el mismo recorte en 2-3 posiciones consecutivas.
        Si el track tiene menos observaciones que num_track_history, el
        resto del slot se enmascara (mismo patrón que bank_mask/query_mask).

        Con `embedding_cache` activo, reusa los embeddings de escala ya
        precalculados de _load_multiscale_crops (amplio/ajustado en los
        extremos; en las intermedias, al no haber píxeles para recalcular
        un factor random, se usa la escala intermedia fija) -- no hace
        falta re-encodear."""
        idxs = _spread_indices(len(observations), self.num_track_history, jitter_ratio=self.history_jitter_ratio)
        crops = []
        for pos, i in enumerate(idxs):
            obs = observations[i]
            is_first = pos == 0
            is_last = pos == len(idxs) - 1
            if self.embedding_cache is not None:
                scale_embeds = self.embedding_cache.get(obs)
                if is_first:
                    crops.append(scale_embeds[0])
                elif is_last:
                    crops.append(scale_embeds[-1])
                else:
                    crops.append(scale_embeds[len(scale_embeds) // 2])
                continue
            image = Image.open(obs.image_path).convert("RGB")
            if is_first:
                crop_box = _expand_bbox(obs.bbox, self.padding_factors[0], image.size, min_size_px=MIN_CONTEXT_CROP_SIZE_PX)
            elif is_last:
                crop_box = _expand_bbox(obs.bbox, 1.0, image.size)
            else:
                crop_box = _expand_bbox(obs.bbox, random.uniform(1.5, 3.5), image.size)
            crops.append(self.transform(image.crop(crop_box)))

        obs_mask = torch.zeros(self.num_track_history, dtype=torch.bool)
        obs_mask[: len(crops)] = True
        while len(crops) < self.num_track_history:
            crops.append(torch.zeros_like(crops[0]))
        return torch.stack(crops, dim=0), obs_mask  # (K,C,H,W) o (K,encoder_dim); (K,)


    def _try_real_reid(self, identity: IdentityTrack):
        gaps = identity.find_real_gaps(self.min_gap_frames)
        if not gaps:
            return None
        split_idx = random.choice(gaps)
        before, after = identity.observations[:split_idx], identity.observations[split_idx:]
        if not before or not after:
            return None
        return before, after

    def _try_synthetic_split(self, identity: IdentityTrack):
        """Corta un track SIN brecha real en un punto artificial para generar
        un positivo (ver docstring del módulo). Con `synthetic_gap_max=0`
        (default), `before`/`after` quedan en frames consecutivos -- casi sin
        diferencia de apariencia, un positivo "fácil". Con
        `synthetic_gap_max>0`, además descarta una ventana de 1..
        synthetic_gap_max observaciones alrededor del corte, simulando la
        estructura de una oclusión (un hueco en el tiempo) aunque no haya
        pasado de verdad -- un positivo más parecido al caso real."""
        n = len(identity.observations)
        if n < self.min_track_len_for_split:
            return None
        split_idx = random.randint(2, n - 2)
        if self.synthetic_gap_max <= 0:
            before, after = identity.observations[:split_idx], identity.observations[split_idx:]
            return before, after

        gap = random.randint(1, self.synthetic_gap_max)
        before_end = max(1, split_idx - gap // 2)
        after_start = min(n - 1, split_idx + (gap - gap // 2))
        if after_start <= before_end:
            after_start = before_end + 1
        if after_start >= n:
            return None
        before, after = identity.observations[:before_end], identity.observations[after_start:]
        if not before or not after:
            return None
        return before, after

    def _sample_case_reid_only(self) -> QueryCase:
        # ignora el peso de NEW_TRACK del case_probs del usuario -- para
        # force_reid no tiene sentido samplearlo, ver _sample_reid_split.
        cases = [QueryCase.REAL_REID, QueryCase.SYNTHETIC_SPLIT]
        weights = [self.case_probs.get(c, 0.0) for c in cases]
        if sum(weights) == 0:
            weights = [0.5, 0.5]
        return random.choices(cases, weights=weights, k=1)[0]

    def _sample_reid_split(self, max_tries: int = 50):
        """Solo para force_reid=True: garantiza que la query principal sea
        SIEMPRE un caso real de reidentificación -- resamplea identidad de
        referencia hasta encontrar una con una brecha real o suficiente
        longitud para un corte sintético. A diferencia del camino normal
        (__getitem__), nunca cae a NEW_TRACK."""
        for _ in range(max_tries):
            reference_identity = self.track_index.sample_random_identity()
            case = self._sample_case_reid_only()
            split = None
            if case == QueryCase.REAL_REID:
                split = self._try_real_reid(reference_identity)
                if split is None:
                    case = QueryCase.SYNTHETIC_SPLIT
            if case == QueryCase.SYNTHETIC_SPLIT:
                split = self._try_synthetic_split(reference_identity)
            if split is not None:
                return case, reference_identity, split
        raise RuntimeError(
            f"force_reid=True: no se encontró ningún track con suficiente longitud/brecha "
            f"para reidentificación pura tras {max_tries} intentos -- revisa min_track_len_for_split "
            f"o si el índice tiene tracks demasiado cortos."
        )

    def __getitem__(self, idx: int):
        if self.seed is not None:
            random.seed(self.seed + idx)

        bank_crops: List[torch.Tensor] = []
        bank_obs_masks: List[torch.Tensor] = []
        bank_video_ids: List[str] = []
        true_identity_bank_pos: Optional[int] = None

        if self.force_reid:
            case, reference_identity, split = self._sample_reid_split()
        else:
            case = self._sample_case()
            reference_identity = self.track_index.sample_random_identity()

            split = None
            if case == QueryCase.REAL_REID:
                split = self._try_real_reid(reference_identity)
                if split is None:  # no había brecha real disponible, cae a split sintético
                    case = QueryCase.SYNTHETIC_SPLIT
            if case == QueryCase.SYNTHETIC_SPLIT:
                split = self._try_synthetic_split(reference_identity)
                if split is None:  # track muy corto, cae a caso NEW_TRACK
                    case = QueryCase.NEW_TRACK

        if case in (QueryCase.REAL_REID, QueryCase.SYNTHETIC_SPLIT):
            before, after = split
            crops, obs_mask = self._load_track_history_crops(before)
            bank_crops.append(crops)
            bank_obs_masks.append(obs_mask)
            bank_video_ids.append(reference_identity.video_id)
            true_identity_bank_pos = 0
            query_obs = random.choice(after)
        else:
            # NEW_TRACK: la identidad de referencia NO entra al banco; su
            # primera observación es la query "genuinamente nueva" (no-match).
            query_obs = reference_identity.observations[0]
            true_identity_bank_pos = None  # no hay match verdadero

        query_views = self._load_query_views(query_obs)
        # el query siempre viene de una observación de reference_identity (ya
        # sea `after` en REAL_REID/SYNTHETIC_SPLIT, o su primera observación
        # en NEW_TRACK) -- mismo video en ambos casos.
        query_video_id = reference_identity.video_id

        # completar banco con distractores hasta max_bank_size
        remaining_slots = self.max_bank_size - len(bank_crops)
        n_hard = int(remaining_slots * self.hard_distractor_ratio)
        n_easy = remaining_slots - n_hard

        hard_distractors = self.track_index.sample_hard_distractors(reference_identity, n_hard)
        easy_distractors = self.track_index.sample_easy_distractors(reference_identity, n_easy)
        for distractor in hard_distractors + easy_distractors:
            crops, obs_mask = self._load_track_history_crops(distractor.observations)
            bank_crops.append(crops)
            bank_obs_masks.append(obs_mask)
            bank_video_ids.append(distractor.video_id)

        # barajar el orden de los slots ya poblados (match real + distractores):
        # sin esto, el track que matchea queda SIEMPRE en la posición 0 (se
        # agrega antes que los distractores) y el modelo puede aprender el
        # sesgo de posición en vez de identidad real. bank_mask sigue siendo
        # un prefijo contiguo -- el shuffle es solo entre los slots válidos,
        # el padding se agrega después, sin tocar.
        n_valid = len(bank_crops)
        order = list(range(n_valid))
        random.shuffle(order)
        bank_crops = [bank_crops[i] for i in order]
        bank_obs_masks = [bank_obs_masks[i] for i in order]
        bank_video_ids = [bank_video_ids[i] for i in order]
        if true_identity_bank_pos is not None:
            true_identity_bank_pos = order.index(true_identity_bank_pos)

        bank_mask = torch.zeros(self.max_bank_size, dtype=torch.bool)
        bank_mask[: len(bank_crops)] = True
        while len(bank_crops) < self.max_bank_size:
            bank_crops.append(torch.zeros_like(bank_crops[0]))
            bank_obs_masks.append(torch.zeros_like(bank_obs_masks[0]))
            bank_video_ids.append("")
        bank_images = torch.stack(bank_crops, dim=0)  # (max_bank_size, K, C, H, W)
        bank_obs_mask_t = torch.stack(bank_obs_masks, dim=0)  # (max_bank_size, K)

        # target: índice del track candidato correcto (0..max_bank_size-1)
        # si la query reaparece, o max_bank_size ("no match") si es
        # genuinamente nueva -- ver MatchScorer en set_reid_model.py, que
        # siempre agrega esta clase extra al final de sus logits.
        target = true_identity_bank_pos if true_identity_bank_pos is not None else self.max_bank_size

        result = {
            "bank_images": bank_images,
            "bank_obs_mask": bank_obs_mask_t,
            "bank_mask": bank_mask,
            "query_views": query_views,
            "target": torch.tensor(target, dtype=torch.long),
        }
        if self.return_debug_info:
            # ruta + bbox absoluto (sin normalizar) de la observación de la
            # query -- no es un tensor, usar collate_episodes_debug.
            result["query0_meta"] = (query_obs.image_path, query_obs.bbox)
        if self.return_video_ids:
            # no son tensores -- usar collate_episodes_with_video_ids.
            result["bank_video_ids"] = bank_video_ids
            result["query_video_id"] = query_video_id
        return result


def collate_episodes(batch: List[dict]) -> dict:
    """Todos los episodios ya tienen shapes fijas (max_bank_size,
    num_track_history), así que un stack directo basta."""
    return {
        key: torch.stack([item[key] for item in batch], dim=0)
        for key in batch[0].keys()
    }


def collate_episodes_with_video_ids(batch: List[dict]) -> dict:
    """Como collate_episodes, pero conserva `bank_video_ids`/`query_video_id`
    (ver return_video_ids en ReIDEpisodeDataset) como listas simples en vez
    de apilarlas -- son strings, no tensores. Pensado para val/test, donde
    se usan para las métricas de confusión intra/inter-secuencia (ver
    _intra_inter_confusion en train_track_query_reid.py); train no lo
    necesita, así que sigue usando collate_episodes."""
    list_keys = {"bank_video_ids", "query_video_id"}
    return {
        key: [item[key] for item in batch] if key in list_keys
        else torch.stack([item[key] for item in batch], dim=0)
        for key in batch[0].keys()
    }


def collate_episodes_debug(batch: List[dict]) -> dict:
    """Como collate_episodes, pero conserva `query0_meta` (ver
    return_debug_info en ReIDEpisodeDataset) como lista simple en vez de
    apilarla -- no es un tensor (ruta + bbox absoluto), así que no tiene
    sentido stackearla. Solo para inspección visual, no para entrenar."""
    return {
        key: [item[key] for item in batch] if key == "query0_meta"
        else torch.stack([item[key] for item in batch], dim=0)
        for key in batch[0].keys()
    }


if __name__ == "__main__":
    # Smoke test con un índice sintético (sin imágenes reales en disco).
    # Reemplaza _load_multiscale_crops temporalmente con un dummy para validar shapes.

    class _DummyReIDEpisodeDataset(ReIDEpisodeDataset):
        def _load_multiscale_crops(self, obs):
            return torch.randn(self.num_scales, 3, 224, 224)

        def _load_track_history_crops(self, observations):
            idxs = _spread_indices(len(observations), self.num_track_history)
            obs_mask = torch.zeros(self.num_track_history, dtype=torch.bool)
            obs_mask[: len(idxs)] = True
            return torch.randn(self.num_track_history, 3, 224, 224), obs_mask

        def _load_query_views(self, obs):
            return torch.randn(2, 3, 224, 224)

    # construir identidades falsas: 5 videos x 6 identidades x 10 observaciones
    fake_tracks = []
    for v in range(5):
        for i in range(6):
            obs_list = [
                TrackObservation(
                    frame_idx=f, image_path="dummy.jpg", bbox=(0, 0, 100, 100), part_label=f % 4
                )
                for f in range(10)
            ]
            fake_tracks.append(IdentityTrack(identity_id=f"v{v}_id{i}", video_id=f"video_{v}", observations=obs_list))

    index = DamageTrackIndex(fake_tracks)
    dataset = _DummyReIDEpisodeDataset(index, episodes_per_epoch=20, max_bank_size=8)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_episodes)

    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape, v.dtype)
