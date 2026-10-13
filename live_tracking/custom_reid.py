"""
custom_reid.py
--------------
ReID de apariencia "custom" (DINOv2 fine-tuneado) para reemplazar el ReID
nativo (osnet) de los trackers de boxmot que lo soportan (BotSort, DeepOcSort,
BoostTrack). Portado desde damages_tracking/botsort_boxmot_reid_eval.py y
damages_tracking/dinov2_reid_finetune.py -- misma arquitectura y checkpoint,
para que los resultados sean comparables a los de ese script.

DinoV2ReID.get_features(boxes, img) implementa la misma interfaz que el
wrapper de ReID nativo en main_track.py::_load_reid(), asi que el resto del
pipeline (constructores de tracker, el cache de embeddings) no distingue
entre ambas fuentes.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.ops import roi_align
from transformers import AutoModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224  # divisible por 14 (patch size de DINOv2)


def crop_with_margin(frame, bbox, margin_ratio):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin_ratio, bh * margin_ratio
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class ReIDModel(nn.Module):
    """Misma arquitectura que dinov2_reid_finetune.py -- necesaria para poder
    cargar los pesos del checkpoint fine-tuneado (backbone DINOv2 + proyeccion
    lineal + batchnorm). num_classes/unfrozen_blocks no importan en inferencia,
    solo tienen que dar las formas de capa correctas para el state_dict."""

    def __init__(self, model_name, embedding_dim, num_classes=1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, pixel_values, return_logits=False):
        out = self.backbone(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0, :]
        emb = self.projection(cls_token)
        emb = F.normalize(emb, dim=-1)
        if return_logits:
            return emb, self.classifier(emb)
        return emb


class DinoV2ReID:
    def __init__(self, checkpoint, model_name, embedding_dim, margin_ratio, device):
        self.device = device
        self.margin_ratio = margin_ratio
        self.embedding_dim = embedding_dim
        self.model = ReIDModel(model_name, embedding_dim).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        # el classifier del checkpoint tiene un tamano (num_classes de train)
        # que no coincide con el num_classes=1 de aca; strict=False no evita
        # errores de tamano (solo de llaves faltantes/extra), asi que se
        # descarta a mano -- igual que en botsort_boxmot_reid_eval.py.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        real_missing = [m for m in missing if not m.startswith("classifier")]
        real_unexpected = [u for u in unexpected if not u.startswith("classifier")]
        if real_missing or real_unexpected:
            print(f"[WARN] DinoV2ReID: mismatch al cargar pesos (fuera del classifier): "
                  f"missing={real_missing} unexpected={real_unexpected}")
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def get_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        crops = []
        for row in boxes:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            crop = crop_with_margin(img, (x1, y1, x2, y2), self.margin_ratio)
            if crop is None or crop.size == 0:
                crop = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            crops.append(crop)

        tensors = [self.transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops]
        batch = torch.stack(tensors).to(self.device)
        emb = self.model(batch, return_logits=False)
        return emb.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_crops(self, crops: list) -> np.ndarray:
        """Embebe una lista de recortes YA HECHOS (imagenes chicas
        independientes, una por entrada) en un solo forward pass batcheado.
        A diferencia de `get_features(boxes, img)`, que asume que TODAS las
        cajas vienen de la MISMA imagen (recorta `img` internamente por
        cada box), aca cada crop puede venir de un frame/momento distinto --
        util para el banco de reid-stitch cuando se guardan varias
        instancias historicas de una identidad perdida (--reid-stitch-bank-
        samples > 1), no solo el ultimo recorte."""
        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        tensors = [self.transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops]
        batch = torch.stack(tensors).to(self.device)
        emb = self.model(batch, return_logits=False)
        return emb.cpu().numpy().astype(np.float32)


# ============================================================================
# P4: ReID sin backbone propio -- ROI-Align sobre el feature map interno del
# detector (backbone.res_layers.2), capturado con un forward hook durante el
# forward pass que YA corre main_track.py para detectar. Portado desde
# damages_tracking/p4_reid_finetune.py -- misma capa/checkpoint, para que sea
# comparable a ese script. A diferencia de DinoV2ReID, get_features(boxes, img)
# ignora los pixeles de `img` (solo usa su shape para escalar las cajas al
# tamano del feature map) -- la apariencia real viene del feature map
# capturado por el hook, no de un crop.
# ============================================================================

P4_HOOK_LAYER = "backbone.res_layers.2"
P4_ROI_OUTPUT_SIZE = (7, 7)
P4_DIM = 1024  # dim del embedding ROI-Align sobre backbone.res_layers.2


def extract_p4(feature_map, boxes_xyxy, frame_wh):
    W, H = frame_wh
    _, _, Hf, Wf = feature_map.shape
    scaled = [[x1 * Wf / W, y1 * Hf / H, x2 * Wf / W, y2 * Hf / H] for x1, y1, x2, y2 in boxes_xyxy]
    rois = torch.tensor(scaled, dtype=torch.float32, device=feature_map.device)
    bidx = torch.zeros((rois.shape[0], 1), device=feature_map.device)
    pooled = roi_align(feature_map, torch.cat([bidx, rois], dim=1), output_size=P4_ROI_OUTPUT_SIZE)
    return F.normalize(pooled.mean(dim=[2, 3]), p=2, dim=1)  # [N, P4_DIM]


class P4ReIDModel(nn.Module):
    """Misma arquitectura que p4_reid_finetune.py -- el 'backbone' (deteccion +
    ROI-Align) ya corrio en el forward del detector; esto es solo la
    proyeccion + clasificador sobre el embedding P4."""

    def __init__(self, embedding_dim, num_classes=1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(P4_DIM, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, p4_embeds, return_logits=False):
        emb = self.projection(p4_embeds)
        emb = F.normalize(emb, dim=-1)
        if return_logits:
            return emb, self.classifier(emb)
        return emb


class P4ReID:
    def __init__(self, checkpoint, embedding_dim, device):
        self.device = device
        self.embedding_dim = embedding_dim
        self.model = P4ReIDModel(embedding_dim).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        real_missing = [m for m in missing if not m.startswith("classifier")]
        real_unexpected = [u for u in unexpected if not u.startswith("classifier")]
        if real_missing or real_unexpected:
            print(f"[WARN] P4ReID: mismatch al cargar pesos (fuera del classifier): "
                  f"missing={real_missing} unexpected={real_unexpected}")
        self.model.eval()
        self._feat = {}  # poblado por el hook via attach(), ver _hook_fn

    def attach(self, detector_model: nn.Module) -> None:
        """Registra el forward hook sobre P4_HOOK_LAYER del detector real que
        va a correr las inferencias de esta secuencia -- debe llamarse UNA vez
        por instancia de detector (nunca funciona con --onnx-file, un
        InferenceSession no expone activaciones intermedias)."""
        for name, mod in detector_model.named_modules():
            if name == P4_HOOK_LAYER:
                mod.register_forward_hook(self._hook_fn)
                return
        raise ValueError(f"P4ReID: capa '{P4_HOOK_LAYER}' no encontrada en el detector.")

    def _hook_fn(self, module, input, output):
        self._feat["map"] = output

    @torch.no_grad()
    def get_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        feat = self._feat.get("map")
        if feat is None:
            raise RuntimeError(
                "P4ReID: no hay feature map capturado todavia -- attach() debe correr "
                "sobre el detector, y el detector debe haber corrido (en vivo, no desde "
                "det_cache) para el frame actual antes de llamar a get_features()."
            )
        h, w = img.shape[:2]
        p4_embeds = extract_p4(feat, boxes[:, :4].tolist(), (w, h))
        emb = self.model(p4_embeds.to(self.device), return_logits=False)
        return emb.cpu().numpy().astype(np.float32)


# ============================================================================
# C4 puro: MISMO hook/capa que P4ReID (backbone.res_layers.2), pero SIN
# ninguna proyeccion/clasificador entrenado -- el embedding es el crudo
# ROI-Align, L2-normalizado. Reproduce la metodologia EXACTA validada en el
# analisis de k-fold (analyze_dinov2_nn_baseline.py --encoder p4
# --kfold-skip-training con --embedding-cache activo -- el modo que corrieron
# TODAS las corridas "C4 puro"/"C4 + cabeza" de esta sesion, ver
# reid_dataloader.py::_load_query_views/_load_track_history_crops en su rama
# de cache): 3 escalas de padding [6.0, 1.6, 1.0] (amplio/intermedio/
# ajustado, DEFAULT_PADDING_FACTORS), query = promedio ajustado+intermedio,
# banco = promedio de hasta N observaciones repartidas en el tiempo (mas
# antigua->amplia, mas reciente->ajustada, intermedias->intermedia).
#
# Pensada SOLO para --reid-stitch (galeria externa de perdidos, ver
# reid_gallery.py) -- no para pasarse como reid_model= nativo de BotSort/
# DeepOcSort (esos esperan un embedding por caja por frame, no la logica de
# query-2-vistas/banco-N-muestras de aca).
# ============================================================================

C4_PADDING_FACTORS = [6.0, 1.6, 1.0]  # [amplio, intermedio, ajustado] -- igual que DEFAULT_PADDING_FACTORS en reid_dataloader.py
C4_MIN_CONTEXT_CROP_SIZE_PX = 400  # igual que MIN_CONTEXT_CROP_SIZE_PX en reid_dataloader.py


def _expand_bbox_c4(bbox, factor, frame_wh, min_size_px=0.0):
    """Copia de reid_dataloader.py::_expand_bbox -- duplicada aca (funcion
    pura sin dependencias) porque eval_tracking/ y visual_damages_tracking/
    son directorios hermanos, sin import directo entre si."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    W, H = frame_wh
    half_w = max((x2 - x1) / 2 * factor, min_size_px / 2)
    half_h = max((y2 - y1) / 2 * factor, min_size_px / 2)
    return (max(0, cx - half_w), max(0, cy - half_h), min(W, cx + half_w), min(H, cy + half_h))


def _spread_indices_c4(length: int, k: int):
    """Copia de reid_dataloader.py::_spread_indices con jitter_ratio=0
    (deterministico) -- no hace falta jitter en inferencia en vivo, cada
    track solo se entierra una vez."""
    if length <= k:
        return list(range(length))
    if k == 1:
        return [length - 1]
    spacing = (length - 1) / (k - 1)
    idxs = {max(0, min(length - 1, round(i * spacing))) for i in range(k)}
    return sorted(idxs)


def extract_c4_scales(feature_map, boxes_xyxy, frame_wh, padding_factors=C4_PADDING_FACTORS):
    """ROI-Align sobre `feature_map` para cada caja de `boxes_xyxy`, en cada
    una de `padding_factors` -- mismo criterio que precompute_p4_roi_cache.py
    (expand_bbox + escalar a coords del feature map + roi_align + avg-pool +
    L2-normalize). Devuelve (N, S, P4_DIM) con S=len(padding_factors), orden
    [amplio, intermedio, ajustado]."""
    W, H = frame_wh
    _, _, Hf, Wf = feature_map.shape
    rois = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        for i, factor in enumerate(padding_factors):
            min_size = C4_MIN_CONTEXT_CROP_SIZE_PX if i == 0 else 0.0
            ex1, ey1, ex2, ey2 = _expand_bbox_c4((x1, y1, x2, y2), factor, (W, H), min_size_px=min_size)
            rois.append([ex1 * Wf / W, ey1 * Hf / H, ex2 * Wf / W, ey2 * Hf / H])
    rois_t = torch.tensor(rois, dtype=torch.float32, device=feature_map.device)
    bidx = torch.zeros((rois_t.shape[0], 1), device=feature_map.device)
    pooled = roi_align(feature_map, torch.cat([bidx, rois_t], dim=1), output_size=P4_ROI_OUTPUT_SIZE)
    pooled = F.normalize(pooled.mean(dim=[2, 3]), p=2, dim=1)  # (N*S, P4_DIM)
    S = len(padding_factors)
    return pooled.view(len(boxes_xyxy), S, -1)


class C4PuroReID:
    """Ver docstring de la seccion arriba. `num_track_history` (default 4)
    replica el mismo parametro del analisis de k-fold (ReIDEpisodeDataset).

    Uso esperado desde main_track.py (--reid-stitch --reid-source c4_puro):
      - get_features(boxes, img): embedding de QUERY (objeto recien nacido)
        para comparar contra la galeria -- promedio ajustado+intermedio del
        frame actual.
      - get_scale_features(boxes, img): las 3 escalas SIN promediar, para
        que el caller vaya acumulando el historial de CADA track vivo en
        CADA frame (una lista de (3,P4_DIM) por raw_id).
      - pool_history(history_rows): cuando un track se pierde, el caller le
        pasa la lista completa acumulada -- devuelve el embedding de banco
        final (ya promediado) para reid_gallery.bury().
    """

    def __init__(self, num_track_history: int = 4, device: str = "cpu",
                 query_single_view: bool = False, bank_single_view: bool = False):
        self.num_track_history = num_track_history
        self.device = device
        self.embedding_dim = P4_DIM
        self._feat = {}
        # Ablacion factorial single-view vs multi-view (M10/M3), 2 ejes
        # independientes:
        #   query_single_view=True -> get_features() devuelve SOLO la escala
        #     ajustada (sin promediar con la intermedia).
        #   bank_single_view=True  -> pool_history() ignora el historial
        #     completo, se queda con la escala ajustada de la observacion
        #     MAS RECIENTE unicamente (sin contexto amplio, sin repartir en
        #     el tiempo).
        # Los 4 cruces (multi/multi = default = comportamiento previo,
        # single/single, multi/single, single/multi) permiten aislar si el
        # aporte viene del lado del banco, de la query, de ambos, o de
        # ninguno -- a diferencia del viejo `single_crop` (ahora dividido en
        # estos 2 flags) que los prendia/apagaba juntos.
        self.query_single_view = query_single_view
        self.bank_single_view = bank_single_view

    def attach(self, detector_model: nn.Module) -> None:
        """Identico a P4ReID.attach() -- mismo hook/capa, ver esa docstring."""
        for name, mod in detector_model.named_modules():
            if name == P4_HOOK_LAYER:
                mod.register_forward_hook(self._hook_fn)
                return
        raise ValueError(f"C4PuroReID: capa '{P4_HOOK_LAYER}' no encontrada en el detector.")

    def _hook_fn(self, module, input, output):
        self._feat["map"] = output

    def _require_feat(self):
        feat = self._feat.get("map")
        if feat is None:
            raise RuntimeError(
                "C4PuroReID: no hay feature map capturado todavia -- attach() debe correr "
                "sobre el detector, y el detector debe haber corrido en vivo (no desde "
                "det_cache) para el frame actual antes de llamar a get_features()/get_scale_features()."
            )
        return feat

    @torch.no_grad()
    def get_scale_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        """(N, 3, P4_DIM), orden [amplio, intermedio, ajustado] -- SIN
        promediar, para acumular el historial de un track vivo (ver
        pool_history)."""
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, len(C4_PADDING_FACTORS), self.embedding_dim), dtype=np.float32)
        feat = self._require_feat()
        h, w = img.shape[:2]
        scales = extract_c4_scales(feat, boxes[:, :4].tolist(), (w, h))
        return scales.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def get_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        """(N, P4_DIM) -- embedding de QUERY: promedio de escala ajustada +
        intermedia del frame actual, igual que _load_query_views en su modo
        de cache (nunca usa la escala amplia -- esa es solo para el banco).
        Con self.query_single_view=True: SOLO la escala ajustada (sin
        promediar) -- ablacion factorial single-view vs multi-view."""
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        scales = self.get_scale_features(boxes, img)  # (N, 3, D) = [amplio, intermedio, ajustado]
        if self.query_single_view:
            return scales[:, 2, :]
        return scales[:, 1:3, :].mean(axis=1)

    def pool_history(self, history_rows):
        """`history_rows`: lista de (3, P4_DIM) en orden CRONOLOGICO (una por
        frame que el track estuvo vivo, ver get_scale_features). Devuelve
        (P4_DIM,) -- el embedding de banco final para esa identidad al
        perderse (None si la lista esta vacia).

        Con self.bank_single_view=True (ablacion factorial single-view vs
        multi-view, ver M10/M3): ignora el historial completo y
        `num_track_history` -- se queda con la escala ajustada de la
        observacion MAS RECIENTE unicamente, el equivalente a un solo
        recorte por track (sin contexto amplio, sin repartir en el tiempo).

        Default (bank_single_view=False): selecciona hasta
        `num_track_history` posiciones repartidas en el tiempo
        (_spread_indices_c4): la mas antigua aporta su escala amplia, la
        mas reciente su escala ajustada, las intermedias su escala
        intermedia -- promediadas en un solo vector, igual que
        _load_track_history_crops en su modo de cache."""
        if not history_rows:
            return None
        if self.bank_single_view:
            return history_rows[-1][2]  # ultima observacion, escala ajustada
        idxs = _spread_indices_c4(len(history_rows), self.num_track_history)
        selected = []
        for pos, i in enumerate(idxs):
            scale_row = history_rows[i]  # (3, D) = [amplio, intermedio, ajustado]
            if pos == 0:
                selected.append(scale_row[0])
            elif pos == len(idxs) - 1:
                selected.append(scale_row[2])
            else:
                selected.append(scale_row[1])
        return np.mean(np.stack(selected, axis=0), axis=0)


# ============================================================================
# P4-FPN puro: IDENTICO a C4PuroReID (mismo ROI-Align de 3 escalas, misma
# logica de query/banco) pero hookeando encoder.fpn_blocks.0 -- la fusion FPN
# REAL (C5+C4), 384-dim -- en vez de backbone.res_layers.2 (que es C4 crudo,
# 1024-dim, pese al nombre historico "P4" usado en gran parte de esta sesion).
# Ver LAYER_SPECS en p4_backbone_encoder.py para la distincion completa. A
# diferencia de C4, esta capa vive DENTRO del encoder -- el hook solo dispara
# si el forward completo del detector (backbone+encoder) corre, pero eso ya
# pasa siempre en main_track.py (necesita el encoder+decoder para las cajas),
# asi que no hace falta ningun cambio adicional en needs_live_p4.
# ============================================================================

P4_FPN_HOOK_LAYER = "encoder.fpn_blocks.0"
P4_FPN_DIM = 384


class P4FpnPuroReID(C4PuroReID):
    """Ver docstring de C4PuroReID -- misma interfaz e implementacion,
    unicamente cambia la capa hookeada y la dimension del embedding."""

    def __init__(self, num_track_history: int = 4, device: str = "cpu",
                 query_single_view: bool = False, bank_single_view: bool = False):
        super().__init__(num_track_history=num_track_history, device=device,
                          query_single_view=query_single_view, bank_single_view=bank_single_view)
        self.embedding_dim = P4_FPN_DIM

    def attach(self, detector_model: nn.Module) -> None:
        for name, mod in detector_model.named_modules():
            if name == P4_FPN_HOOK_LAYER:
                mod.register_forward_hook(self._hook_fn)
                return
        raise ValueError(f"P4FpnPuroReID: capa '{P4_FPN_HOOK_LAYER}' no encontrada en el detector.")


# ============================================================================
# DINOv2 puro: mismo criterio de 3 escalas / query-2-vistas / banco-N-muestras
# que C4PuroReID/P4FpnPuroReID, pero SIN backbone propio del detector -- usa
# DINOv2 crudo (facebook/dinov2-base por default, CLS token, SIN proyeccion
# entrenada) sobre recortes de PIXELES reales, no ROI-Align sobre un feature
# map. Reproduce el modo "DINOv2 puro" del k-fold (reid_track_query_module.py
# ::DINOv2Encoder con checkpoint_path=None, embedding_dim=None -- ver
# reid_dataloader.py::_load_multiscale_crops/_load_track_history_crops).
#
# A diferencia de C4/P4 (donde ROI-Align sobre un feature map ya calculado es
# practicamente gratis, asi que las 3 escalas se recalculan en TODOS los
# frames), pasar cada recorte por DINOv2 es caro -- por eso esta clase
# acumula los RECORTES DE PIXELES (no embeddings) en cada frame, y solo corre
# el backbone una vez, en batch, al nacer un track (query) o al perderlo
# (pool_history) -- mismo principio de costo que el patron ya usado por
# --reid-source dinov2 nativo + --reid-stitch (last_crop/crop_reservoir en
# main_track.py), pero con 3 escalas por posicion en vez de un solo recorte.
# ============================================================================

DINOV2_PADDING_FACTORS = [6.0, 1.6, 1.0]  # [amplio, intermedio, ajustado] -- igual que C4_PADDING_FACTORS
DINOV2_MIN_CONTEXT_CROP_SIZE_PX = 400
DINOV2_IMG_SIZE = 224


class DinoV2PuroReID:
    """Uso esperado desde main_track.py (--reid-stitch --reid-source dinov2_puro):
      - get_scale_crops(boxes, img): recortes SIN encodear (N, 3, H, W, 3)
        uint8 RGB, para que el caller acumule el historial de cada track vivo
        en cada frame (una lista de (3,H,W,3) por raw_id).
      - get_features(boxes, img): embedding de QUERY (objeto recien nacido)
        -- promedio ajustado+intermedio del frame actual, YA encodeado.
      - pool_history(history_rows): cuando un track se pierde, el caller le
        pasa la lista completa de recortes acumulados -- devuelve el
        embedding de banco final (ya encodeado y promediado).
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", num_track_history: int = 4,
                 device: str = "cpu", query_single_view: bool = False, bank_single_view: bool = False):
        self.device = device
        self.num_track_history = num_track_history
        # ver C4PuroReID.query_single_view/bank_single_view -- misma ablacion factorial M10/M3
        self.query_single_view = query_single_view
        self.bank_single_view = bank_single_view
        self.model = AutoModel.from_pretrained(model_name).to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        self.transform = T.Compose([
            T.Resize((DINOV2_IMG_SIZE, DINOV2_IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def get_scale_crops(self, boxes: np.ndarray, img: np.ndarray) -> list:
        """Lista (una por caja) de arrays (3, DINOV2_IMG_SIZE, DINOV2_IMG_SIZE, 3)
        uint8 RGB, escalas [amplio, intermedio, ajustado] -- SIN encodear
        (ver docstring de la clase: encodear se difiere a
        get_features()/pool_history() para no correr DINOv2 en cada frame)."""
        if boxes is None or len(boxes) == 0:
            return []
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = []
        for row in boxes:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            scales = []
            for i, factor in enumerate(DINOV2_PADDING_FACTORS):
                min_size = DINOV2_MIN_CONTEXT_CROP_SIZE_PX if i == 0 else 0.0
                ex1, ey1, ex2, ey2 = _expand_bbox_c4((x1, y1, x2, y2), factor, (w, h), min_size_px=min_size)
                crop = rgb[int(ey1):int(ey2), int(ex1):int(ex2)]
                if crop.size == 0:
                    crop = np.zeros((DINOV2_IMG_SIZE, DINOV2_IMG_SIZE, 3), dtype=np.uint8)
                else:
                    crop = cv2.resize(crop, (DINOV2_IMG_SIZE, DINOV2_IMG_SIZE))
                scales.append(crop)
            out.append(np.stack(scales, axis=0))
        return out

    @torch.no_grad()
    def _encode_crops(self, crops: list) -> np.ndarray:
        """(N, embedding_dim) -- CLS token crudo, L2-normalizado."""
        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        tensors = [self.transform(Image.fromarray(c)) for c in crops]
        batch = torch.stack(tensors).to(self.device)
        out = self.model(pixel_values=batch)
        cls = out.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, dim=-1)
        return cls.cpu().numpy().astype(np.float32)

    def get_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Query: promedio de escala ajustada + intermedia del frame actual,
        igual que C4PuroReID.get_features. Con self.query_single_view=True:
        SOLO la escala ajustada."""
        scale_crops = self.get_scale_crops(boxes, img)
        if not scale_crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if self.query_single_view:
            flat = [scales[2] for scales in scale_crops]  # solo ajustada
            return self._encode_crops(flat)
        flat = [c for scales in scale_crops for c in scales[1:3]]  # [intermedio, ajustado] por caja
        embs = self._encode_crops(flat)
        embs = embs.reshape(len(scale_crops), 2, self.embedding_dim)
        return embs.mean(axis=1)

    def pool_history(self, history_rows):
        """`history_rows`: lista de (3,H,W,3) uint8 en orden CRONOLOGICO (una
        por frame que el track estuvo vivo, ver get_scale_crops). Selecciona
        hasta `num_track_history` posiciones repartidas en el tiempo
        (_spread_indices_c4): la mas antigua aporta su escala amplia, la mas
        reciente su escala ajustada, las intermedias su escala intermedia --
        se encodean TODAS en un solo forward batcheado y se promedian,
        exactamente igual que C4PuroReID.pool_history.

        Con self.bank_single_view=True (ablacion factorial single-view vs
        multi-view): ignora el historial completo -- encodea SOLO la escala
        ajustada de la observacion mas reciente."""
        if not history_rows:
            return None
        if self.bank_single_view:
            embs = self._encode_crops([history_rows[-1][2]])
            return embs[0]
        idxs = _spread_indices_c4(len(history_rows), self.num_track_history)
        selected_crops = []
        for pos, i in enumerate(idxs):
            scale_row = history_rows[i]  # (3, H, W, 3) = [amplio, intermedio, ajustado]
            if pos == 0:
                selected_crops.append(scale_row[0])
            elif pos == len(idxs) - 1:
                selected_crops.append(scale_row[2])
            else:
                selected_crops.append(scale_row[1])
        embs = self._encode_crops(selected_crops)
        return embs.mean(axis=0)


# ============================================================================
# GenericPuroReID: mismo protocolo multi-escala/banco-con-historial que
# C4PuroReID/P4FpnPuroReID/DinoV2PuroReID, pero envolviendo CUALQUIER modelo
# de boxmot.reid.core.reid.ReID (OSNet, CLIP-ReID, etc.) -- agregado para la
# ablacion "single-crop vs multi-crop" pedida junto con el review M3/M10:
# --reid-source native (el default de main_track.py) solo guarda UN recorte
# (el mas reciente) al perder el track, sin el truco de 3 escalas repartidas
# en el tiempo que SI reciben C4/P4-FPN/DINOv2 puro. Esta clase le da a un
# ReID real (entrenado) el MISMO protocolo, para que la comparacion "trained
# ReID vs. reusar el detector" no este confundida con "single-crop vs.
# multi-crop bank".
#
# A diferencia de C4PuroReID (ROI-Align sobre un feature map ya calculado,
# practicamente gratis) esta clase SI paga un forward real del modelo por
# cada escala -- no hay atajo posible sin re-entrenar el ReID para aceptar
# un feature map compartido, asi que el costo por evento es mas alto (ver
# discusion de Section~sec:computational_cost, misma logica que DINOv2 puro).
# ============================================================================

class GenericPuroReID:
    """Uso esperado desde main_track.py (--reid-stitch --reid-source native_puro
    --native-reid-weights <peso>.pt):
      - get_features(boxes, img): embedding de QUERY -- promedio ajustado+
        intermedio del frame actual, igual criterio que C4PuroReID.
      - get_scale_features(boxes, img): las 3 escalas SIN promediar, para
        que el caller acumule el historial de cada track vivo en cada frame.
      - pool_history(history_rows): al perder el track, pool_history recibe
        la lista completa acumulada y devuelve el embedding de banco final.
    """

    def __init__(self, weights, num_track_history: int = 4, device: str = "cpu", half: bool = False):
        from boxmot.reid.core.reid import ReID
        reid_device = "0" if device == "cuda" else "cpu"
        # half=True (forward en fp16): reduce a la mitad la memoria del
        # modelo CLIP-VehicleID -- necesario para algunos videos con
        # DeepOC-SORT (que ademas carga su propio embebedor nativo en la
        # misma GPU de 8GB) que dan CUDA OOM en fp32, reproducible en el
        # mismo frame incluso sin ninguna otra carga en la GPU.
        self._reid = ReID(weights=weights, device=reid_device, half=half)
        if not hasattr(self._reid, "get_features"):
            def _get_features(boxes, img):
                return self._reid(img, boxes=boxes)
            self._reid.get_features = _get_features
        self.num_track_history = num_track_history
        self.device = device
        self.embedding_dim = None  # se completa en el primer forward (depende del modelo)

    def get_scale_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        """(N, 3, dim), orden [amplio, intermedio, ajustado] -- 3 forwards
        reales del modelo (uno por escala), a diferencia de C4PuroReID donde
        las 3 escalas comparten un unico feature map ya calculado."""
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, len(C4_PADDING_FACTORS), self.embedding_dim or 1), dtype=np.float32)
        h, w = img.shape[:2]
        all_boxes = []
        for b in boxes[:, :4].tolist():
            for i, factor in enumerate(C4_PADDING_FACTORS):
                min_size = C4_MIN_CONTEXT_CROP_SIZE_PX if i == 0 else 0.0
                all_boxes.append(_expand_bbox_c4(tuple(b), factor, (w, h), min_size_px=min_size))
        boxes_arr = np.asarray(all_boxes, dtype=np.float32)
        embeds = np.asarray(self._reid.get_features(boxes_arr, img), dtype=np.float32)
        self.embedding_dim = embeds.shape[-1]
        S = len(C4_PADDING_FACTORS)
        return embeds.reshape(len(boxes), S, -1)

    def get_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        """(N, dim) -- embedding de QUERY: promedio de escala ajustada +
        intermedia del frame actual, igual criterio que C4PuroReID.get_features."""
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, self.embedding_dim or 1), dtype=np.float32)
        scales = self.get_scale_features(boxes, img)  # (N, 3, D) = [amplio, intermedio, ajustado]
        return scales[:, 1:3, :].mean(axis=1)

    def pool_history(self, history_rows):
        """Identico a C4PuroReID.pool_history -- selecciona hasta
        `num_track_history` posiciones repartidas en el tiempo
        (_spread_indices_c4) y promedia."""
        if not history_rows:
            return None
        idxs = _spread_indices_c4(len(history_rows), self.num_track_history)
        selected = []
        for pos, i in enumerate(idxs):
            scale_row = history_rows[i]  # (3, D) = [amplio, intermedio, ajustado]
            if pos == 0:
                selected.append(scale_row[0])
            elif pos == len(idxs) - 1:
                selected.append(scale_row[2])
            else:
                selected.append(scale_row[1])
        return np.mean(np.stack(selected, axis=0), axis=0)
