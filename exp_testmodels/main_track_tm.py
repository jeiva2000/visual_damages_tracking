"""
track_onnx_boxmot.py
--------------------
RT-DETR (ONNX) + ByteTrack usando la librería boxmot.

boxmot desacopla completamente el detector del tracker, lo que permite
usar cualquier tracker (ByteTrack, BoTrack, StrongSORT, DeepSORT...)
con cualquier detector externo.

Instalación:
    pip install boxmot

Uso:
    # Desde carpeta de frames (ONNX)
    python track_onnx_boxmot.py \\
        --onnx-file  modelo.onnx \\
        --frames     /ruta/frames \\
        --output     resultado.mp4

    # Desde carpeta de frames (checkpoint .pth de RT-DETR/RTDETRv2)
    python track_onnx_boxmot.py \\
        --pth-file   best.pth \\
        --config     configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml \\
        --frames     /ruta/frames \\
        --output     resultado.mp4

    # Desde vídeo
    python track_onnx_boxmot.py \\
        --onnx-file  modelo.onnx \\
        --video      entrada.mp4 \\
        --output     resultado.mp4 \\
        [--tracker    bytetrack]   \\   ← bytetrack | botrack | strongsort | ocsort | deepocsort
        [--fps        30]          \\
        [--conf       0.25]        \\
        [--imgsz      640]         \\
        [--device     cuda]

Formato de entrada al tracker (boxmot estándar):
    np.array([[x1, y1, x2, y2, score, cls_id], ...])  float32
"""

# =============================================================================
# COPIA EXPERIMENTAL de main_track.py (md5 c034f905) para el barrido de
# detectores de /mnt/Disk2/test_models (exp_testmodels/). NO editar
# main_track.py en su lugar. Unicos cambios, controlados por variables de
# entorno (sin ellas se comporta igual que el original):
#   TM_LETTERBOX=1  preprocesado de infer_onnx_rtv4.py (D-FINE/RT-DETRv4):
#                   resize con aspecto + padding centrado a imgsz,
#                   orig_target_sizes=[[imgsz, imgsz]], y cajas reajustadas
#                   al tamaño original (adjust_boxes).
#   TM_SCORE_IDX=k  salida del ONNX usada como score (default 2 = 'scores';
#                   3 = 4a salida de los modelos quality, la que usa
#                   infer_onnx_q.py).
#   TM_ALT_CACHE=1  si el ONNX tiene 4 salidas, guarda ADEMAS un det-cache
#                   con la otra variante de score en <det_cache_dir>_ALT/
#                   (mismo nombre de archivo, mismo model_key) para correr la
#                   segunda variante sin repetir la inferencia.
# =============================================================================
import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Paleta de colores por track_id
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    ( 56, 182, 255), (255, 112,  56), ( 56, 255, 112), (255,  56, 182),
    (112,  56, 255), (255, 230,  56), ( 56, 230, 255), (230, 255,  56),
    (182, 255,  56), ( 56, 255, 230), (255,  56, 112), ( 56, 112, 255),
    (255, 182,  56), ( 56, 255, 182), (182,  56, 255), (230,  56, 255),
]

def track_color(tid: int) -> tuple:
    return _PALETTE[int(tid) % len(_PALETTE)]


# Umbral de score usado SOLO para decidir que cajas crudas reciben un embedding
# ReID cacheable (ver "Cache de embeddings ReID" en run()). Bien por debajo de
# cualquier --conf real que se sweepee en este pipeline (el grid mas bajo hoy
# arranca en 0.35), para que el cache sirva para cualquier --conf sin recalcular.
REID_CACHE_MIN_SCORE = 0.1


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Runtime
# ──────────────────────────────────────────────────────────────────────────────

def build_session(onnx_path: str, device: str) -> ort.InferenceSession:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"[INFO] ONNXRuntime: {sess.get_providers()[0]}")
    # Detectar formato del modelo
    input_names = [i.name for i in sess.get_inputs()]
    sess._is_ultralytics = "orig_target_sizes" not in input_names
    if sess._is_ultralytics:
        print("[INFO] Formato: Ultralytics (single input, cxcywh normalizado)")
    else:
        print("[INFO] Formato: RT-DETR custom (images + orig_target_sizes)")
    return sess


class TorchSession:
    """Envoltorio mínimo para exponer un modelo RT-DETR (.pth) con la misma
    interfaz mínima que build_session() (atributo _is_ultralytics)."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._is_ultralytics = False


def build_torch_session(config_path: str, pth_path: str, device: str) -> TorchSession:
    """Carga un checkpoint de entrenamiento RT-DETR/RTDETRv2 (.pth) reconstruyendo
    la arquitectura desde su YAML de config, igual que tools/export_onnx.py del
    repo RT-DETR (RTDETRv2/rtdetrv2_pytorch)."""
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.core import YAMLConfig

    # El checkpoint sobrescribe el backbone justo después: evitamos que YAMLConfig
    # descargue los pesos ImageNet pretrained (PResNet/HGNetv2) del backbone.
    cfg = YAMLConfig(
        config_path, resume=pth_path,
        PResNet={'pretrained': False}, HGNetv2={'pretrained': False},
    )

    checkpoint = torch.load(pth_path, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class DeployModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    dev = torch.device('cuda' if device == 'cuda' else 'cpu')
    model = DeployModel().to(dev).eval()
    print(f"[INFO] PyTorch (RT-DETR): config={config_path}  checkpoint={pth_path}  device={dev}")
    return TorchSession(model, dev)


_TM_LETTERBOX = os.environ.get("TM_LETTERBOX", "0") == "1"
_TM_SCORE_IDX = int(os.environ.get("TM_SCORE_IDX", "2"))
_TM_ALT_CACHE = os.environ.get("TM_ALT_CACHE", "0") == "1"
_TM_LB = None        # (ratio, pad_w, pad_h, imgsz) del ultimo frame letterbox
_TM_LAST_ALT = None  # scores de la variante alternativa del ultimo frame


def _tm_letterbox(pil, size):
    """resize_with_aspect_ratio de infer_onnx_rtv4.py."""
    ow, oh = pil.size
    ratio = min(size / ow, size / oh)
    nw, nh = int(ow * ratio), int(oh * ratio)
    im = pil.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    pw, ph = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im, (pw, ph))
    return canvas, ratio, pw, ph


def preprocess(img_bgr: np.ndarray, imgsz: int):
    global _TM_LB
    h, w = img_bgr.shape[:2]
    pil  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if _TM_LETTERBOX:
        canvas, ratio, pw, ph = _tm_letterbox(pil, imgsz)
        _TM_LB = (ratio, pw, ph, imgsz)
        return T.ToTensor()(canvas).unsqueeze(0).numpy(), (w, h)
    pil_resized = pil.resize((imgsz, imgsz))
    im_data = T.ToTensor()(pil_resized).unsqueeze(0).numpy()
    return im_data, (w, h)


def run_inference(sess, im_data, orig_size):
    w, h = orig_size

    if isinstance(sess, TorchSession):
        images = torch.from_numpy(im_data).to(sess.device)
        sizes = torch.tensor([[w, h]], device=sess.device)
        with torch.no_grad():
            labels, boxes, scores = sess.model(images, sizes)
        return labels[0].cpu().numpy(), boxes[0].cpu().numpy(), scores[0].cpu().numpy()

    if sess._is_ultralytics:
        # Output: [1, 300, 6] → [cx, cy, w, h, score, class_id] normalizado
        outputs = sess.run(None, {"images": im_data})
        preds = outputs[0][0]  # (300, 6)
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        scores = preds[:, 4]
        labels = preds[:, 5].astype(np.int64)
        # Convertir cxcywh normalizado → xyxy absoluto
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        return labels, boxes, scores
    else:
        # Output: labels, boxes (xyxy abs), scores [, queries]
        global _TM_LAST_ALT
        sizes = [[_TM_LB[3], _TM_LB[3]]] if _TM_LETTERBOX else [[w, h]]
        outputs = sess.run(None, {
            "images": im_data,
            "orig_target_sizes": sizes,
        })
        sc = [np.asarray(outputs[2][0]).reshape(-1)]
        if len(outputs) > 3 and np.asarray(outputs[3]).size == np.asarray(outputs[2]).size:
            sc.append(np.asarray(outputs[3][0]).reshape(-1))
        scores = sc[_TM_SCORE_IDX - 2]
        _TM_LAST_ALT = sc[1 if _TM_SCORE_IDX == 2 else 0] if len(sc) > 1 else None
        labels, boxes = outputs[0][0], np.array(outputs[1][0], dtype=np.float32)
        if _TM_LETTERBOX:
            ratio, pw, ph, _ = _TM_LB
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pw) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - ph) / ratio
        return labels, boxes, scores


# ──────────────────────────────────────────────────────────────────────────────
# Dibujado
# ──────────────────────────────────────────────────────────────────────────────

def draw_tracks(frame: np.ndarray, tracks: np.ndarray, reid_events: Optional[dict] = None) -> np.ndarray:
    """
    tracks: array (N, 7) → [x1, y1, x2, y2, track_id, score, cls_id]
    formato de salida estándar de boxmot.

    `reid_events` (opcional, ver --draw-reid-events): {track_id: {"status":
    "ok"|"fail", ...}} -- si el id de una fila esta ahi, se dibuja ademas un
    marco grueso verde ("REID OK", matcheo contra la galeria) o rojo ("REID
    FAIL", intento con candidatos reales que no supero el umbral) sobre la
    caja normal, para ver a ojo en el video donde ocurre cada intento.
    """
    out = frame.copy()
    if tracks is None or len(tracks) == 0:
        return out

    for row in tracks:
        x1, y1, x2, y2 = [int(v) for v in row[:4]]
        tid    = int(row[4])
        score  = float(row[5])
        cls_id = int(row[6])

        color = track_color(tid)
        label = f"ID{tid} c{cls_id} {score:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1, th + 6)
        cv2.rectangle(out, (x1, ly-th-bl-4), (x1+tw+4, ly), color, -1)
        cv2.putText(out, label, (x1+2, ly-bl-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        event = reid_events.get(tid) if reid_events else None
        if event is not None:
            is_ok = event["status"] == "ok"
            ecolor = (0, 255, 0) if is_ok else (0, 0, 255)
            etext = "REID OK" if is_ok else "REID FAIL"
            cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), ecolor, 4)
            (ew, eh), ebl = cv2.getTextSize(etext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ey1 = min(frame.shape[0] - eh - ebl - 4, y2 + 4)
            cv2.rectangle(out, (x1 - 3, ey1), (x1 - 3 + ew + 8, ey1 + eh + ebl + 4), ecolor, -1)
            cv2.putText(out, etext, (x1 + 1, ey1 + eh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fuente de frames
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def frame_source(args: argparse.Namespace):
    """Generador de (frame_bgr). Devuelve también el total y la resolución."""
    if args.frames:
        paths = sorted(
            [p for p in Path(args.frames).iterdir()
             if p.suffix.lower() in SUPPORTED_EXTS],
            key=lambda p: p.stem,
        )
        total = len(paths)
        def gen():
            for p in paths:
                img = cv2.imread(str(p))
                if img is not None:
                    yield img
        return gen(), total

    else:
        cap   = cv2.VideoCapture(args.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        def gen():
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    break
                yield frame
        return gen(), total


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def tracks_to_mot(tracks: np.ndarray, frame_id: int, cls_offset: int = 0) -> list:
    """Convierte tracks boxmot (N,7) → líneas MOT format.
    MOT: frame,id,x,y,w,h,score,-1,-1,-1
    """
    lines = []
    if tracks is None or len(tracks) == 0:
        return lines
    for row in tracks:
        x1, y1, x2, y2 = row[:4]
        tid   = int(row[4]) + cls_offset
        score = float(row[5])
        w, h  = x2 - x1, y2 - y1
        lines.append(
            f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
            f"{score:.6f},-1,-1,-1\n")
    return lines


def _cjm_tracks_to_mot(online_targets, frame_id: int, cls_offset: int = 0) -> list:
    """Convierte salida de cjm_byte_track → líneas MOT."""
    lines = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = int(t.track_id) + cls_offset
        x, y, w, h = map(float, tlwh)
        score = float(getattr(t, "score", 1.0))
        lines.append(
            f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
            f"{score:.6f},-1,-1,-1\n")
    return lines


def _cjm_tracks_to_draw(online_targets, cls_offset: int = 0) -> np.ndarray:
    """Convierte salida cjm_byte_track → array (N,7) para draw_tracks."""
    if not online_targets:
        return np.empty((0, 7))
    rows = []
    for t in online_targets:
        x, y, w, h = map(float, t.tlwh)
        tid = int(t.track_id) + cls_offset
        score = float(getattr(t, "score", 1.0))
        rows.append([x, y, x + w, y + h, tid, score, 0])
    return np.array(rows, dtype=np.float32)


_REID_MODEL_PATH = Path("osnet_x0_25_msmt17.pt")


def _single_view_flags(args: argparse.Namespace):
    """(query_single_view, bank_single_view) -- ablacion factorial single-view
    vs multi-view (M10/M3). --reid-stitch-single-crop (flag viejo, dejado por
    compatibilidad con corridas ya lanzadas) prende los dos ejes juntos;
    --reid-stitch-query-single-view/--reid-stitch-bank-single-view los
    controlan por separado para aislar de que lado viene el efecto."""
    legacy = getattr(args, "reid_stitch_single_crop", False)
    q = legacy or getattr(args, "reid_stitch_query_single_view", False)
    b = legacy or getattr(args, "reid_stitch_bank_single_view", False)
    return q, b


def load_reid(args: argparse.Namespace, detector_model=None):
    """Carga la instancia de ReID (nativa u custom) segun --reid-source.
    Expuesta a nivel de modulo (no anidada en run()) para que run_eval.py
    pueda cargarla UNA sola vez por combinacion de hiperparametros y
    reusarla entre las N secuencias del split -- ver run(..., reid=...).
    Recargar el modelo por cada secuencia era el cuello de botella real de
    --reid-stitch/--reid-source dinov2 con trackers livianos como ByteTrack.

    --reid-source p4 es la excepcion: NO se puede compartir entre secuencias
    (el hook de P4ReID.attach() queda pegado a una instancia concreta del
    detector, y esa instancia se reconstruye por secuencia en run() -- ver
    _build_sess()). detector_model solo se pasa (y solo hace falta) para p4;
    debe ser el nn.Module real del detector (TorchSession.model.model), nunca
    un InferenceSession de ONNX."""
    if args.reid_source == "dinov2":
        from custom_reid import DinoV2ReID
        reid_device = 'cuda' if args.reid_device == 'cuda' else 'cpu'
        return DinoV2ReID(
            checkpoint=args.reid_checkpoint,
            model_name=args.reid_model,
            embedding_dim=args.reid_embedding_dim,
            margin_ratio=args.reid_margin_ratio,
            device=reid_device,
        )
    if args.reid_source == "p4":
        from custom_reid import P4ReID
        reid_device = 'cuda' if args.reid_device == 'cuda' else 'cpu'
        reid = P4ReID(checkpoint=args.reid_checkpoint, embedding_dim=args.reid_embedding_dim,
                      device=reid_device)
        # detector_model es None cuando reid_cache YA cubre toda la secuencia
        # (needs_live_p4=False en run()) -- en ese caso get_features() nunca
        # se llama, asi que no hace falta el hook. Si SI hiciera falta y no
        # hay detector_model, get_features() lo va a detectar solo (RuntimeError
        # claro por falta de feature map) en vez de fallar aca.
        if detector_model is not None:
            reid.attach(detector_model)
        return reid
    if args.reid_source == "c4_puro":
        from custom_reid import C4PuroReID
        reid_device = 'cuda' if args.reid_device == 'cuda' else 'cpu'
        q_single, b_single = _single_view_flags(args)
        reid = C4PuroReID(num_track_history=args.reid_stitch_num_track_history, device=reid_device,
                           query_single_view=q_single, bank_single_view=b_single)
        if detector_model is not None:
            reid.attach(detector_model)
        return reid
    if args.reid_source == "p4fpn_puro":
        from custom_reid import P4FpnPuroReID
        reid_device = 'cuda' if args.reid_device == 'cuda' else 'cpu'
        q_single, b_single = _single_view_flags(args)
        reid = P4FpnPuroReID(num_track_history=args.reid_stitch_num_track_history, device=reid_device,
                              query_single_view=q_single, bank_single_view=b_single)
        if detector_model is not None:
            reid.attach(detector_model)
        return reid
    if args.reid_source == "dinov2_puro":
        from custom_reid import DinoV2PuroReID
        reid_device = 'cuda' if args.reid_device == 'cuda' else 'cpu'
        q_single, b_single = _single_view_flags(args)
        return DinoV2PuroReID(model_name=args.reid_model,
                               num_track_history=args.reid_stitch_num_track_history,
                               device=reid_device,
                               query_single_view=q_single, bank_single_view=b_single)
    if args.reid_source == "native_puro":
        from custom_reid import GenericPuroReID
        reid_device = 'cuda' if args.reid_device == 'cuda' else 'cpu'
        native_weights = getattr(args, "native_reid_weights", None) or str(_REID_MODEL_PATH)
        return GenericPuroReID(weights=native_weights,
                                num_track_history=args.reid_stitch_num_track_history,
                                device=reid_device,
                                half=getattr(args, "native_reid_half", False))
    from boxmot.reid.core.reid import ReID
    reid_device = '0' if args.reid_device == 'cuda' else 'cpu'
    native_weights = Path(getattr(args, "native_reid_weights", None) or _REID_MODEL_PATH)
    reid = ReID(weights=native_weights, device=reid_device, half=False)
    # BotSORT v19 espera get_features(boxes, img) → wrapper
    if not hasattr(reid, 'get_features'):
        def _get_features(boxes, img):
            return reid(img, boxes=boxes)
        reid.get_features = _get_features
    return reid


def reid_key_for(args: argparse.Namespace) -> str:
    if args.reid_source == "dinov2":
        return f"dinov2:{os.path.basename(args.reid_checkpoint)}:{args.reid_model}"
    if args.reid_source == "p4":
        return f"p4:{os.path.basename(args.reid_checkpoint)}"
    q_single, b_single = _single_view_flags(args)
    view_suffix = f":q{int(q_single)}b{int(b_single)}" if (q_single or b_single) else ""
    if args.reid_source == "c4_puro":
        return f"c4_puro{view_suffix}"
    if args.reid_source == "p4fpn_puro":
        return f"p4fpn_puro{view_suffix}"
    if args.reid_source == "dinov2_puro":
        return f"dinov2_puro:{args.reid_model}{view_suffix}"
    if args.reid_source == "native_puro":
        native_weights = getattr(args, "native_reid_weights", None) or str(_REID_MODEL_PATH)
        return f"native_puro:{os.path.basename(native_weights)}"
    native_weights = getattr(args, "native_reid_weights", None) or str(_REID_MODEL_PATH)
    return f"native:{os.path.basename(native_weights)}"


def _reset_track_id_counter(tracker_module_path: str) -> None:
    """BaseTrack._count (ver basetrack.py de boxmot) es un atributo de CLASE
    compartido por TODAS las instancias del proceso -- desde que run_eval.py
    corre todas las secuencias de un split EN EL MISMO PROCESO (import
    directo de main_track.run(), ya no un subproceso por secuencia), sin
    este reset los ids de ByteTrack/BotSort siguen la numeración acumulada
    de TODAS las secuencias anteriores de la corrida (confirmado: ids en
    6400+ para la secuencia #30 de una corrida native_reid_bytetrack_alldata,
    vs. 1-43 para BotSort/OcSort/DeepOcSort en la misma secuencia -- BotSort
    tiene el mismo bug pero crea muchos menos tracks por secuencia, así que
    tarda más en notarse). No afecta métricas ni la clasificación
    reidentified/new_id_born/id_swap (esa lógica compara ids solo DENTRO de
    la misma secuencia), solo hace ilegibles los ids en video/logs. Mismo
    criterio que infer_track_reid_gaps.py::_reset_track_id_counter. Envuelto
    en try/except porque la ruta interna del módulo puede cambiar entre
    versiones de boxmot -- si falla, seguimos con los ids acumulados en vez
    de romper la corrida."""
    try:
        import importlib
        basetrack = importlib.import_module(tracker_module_path)
        basetrack.BaseTrack.clear_count()
    except Exception as e:
        print(f"[WARN] No se pudo resetear el contador de ids de {tracker_module_path} ({e}) -- los ids van a quedar acumulados entre secuencias.")


def tracker_needs_reid(args: argparse.Namespace) -> bool:
    t = args.tracker.lower()
    if t in ("cjm_bytetrack", "bytetrack", "ocsort"):
        return False
    if t == "botsort":
        return not args.no_native_reid
    if t == "boosttrack":
        return args.with_reid
    return True  # deepocsort, strongsort, occluboost


def run(args: argparse.Namespace, reid=None):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda solicitado pero CUDA no está disponible; usando cpu.")
        args.device = "cpu"
    if args.reid_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --reid-device cuda solicitado pero CUDA no está disponible; usando cpu.")
        args.reid_device = "cpu"

    use_cjm = args.tracker.lower() == "cjm_bytetrack"

    needs_reid = tracker_needs_reid(args)
    needs_reid_stitch = args.reid_stitch

    # ── Cache de detecciones crudas del detector ────────────────────────────
    # El umbral --conf se aplica DESPUÉS de la inferencia (ver process()), así
    # que las detecciones crudas no dependen de --conf/--tracker/--track-thresh/
    # --match-thresh/--max-age. Cachearlas permite reusar la misma inferencia
    # del detector entre distintas combinaciones de un sweep de hiperparámetros.
    # (Reubicado ANTES de cargar reid/construir el tracker: --reid-source p4
    # necesita saber cache_hit/reid_cache_hit para decidir si hace falta
    # construir la sesion del detector, ver mas abajo.)
    det_cache_path = Path(args.det_cache) if args.det_cache else None
    model_key = os.path.basename(args.pth_file or args.onnx_file)
    cached_boxes = cached_scores = cached_labels = None
    cache_hit = False

    if det_cache_path is not None and det_cache_path.is_file():
        cache = np.load(det_cache_path, allow_pickle=True)
        if str(cache["model_key"]) == model_key and int(cache["imgsz"]) == args.imgsz:
            cached_boxes  = cache["boxes"]
            cached_scores = cache["scores"]
            cached_labels = cache["labels"]
            cache_hit = True
            print(f"[INFO] Det-cache: reusando {det_cache_path} ({len(cached_boxes)} frames), "
                  f"no se carga el detector.")
        else:
            print(f"[WARN] Det-cache {det_cache_path} no coincide (modelo/imgsz distintos); recalculando.")

    # ── Cache de embeddings ReID ─────────────────────────────────────────────
    # El embedding de apariencia de una caja no depende de --conf/--tracker/
    # lambda_*/dlo_boost_coef/etc, solo de la imagen+caja y del modelo de ReID
    # -- mismo razonamiento que el cache de detecciones. IMPORTANTE: NO se
    # calculan embeddings sobre las cajas crudas del detector sin filtrar --
    # RT-DETR devuelve ~300 queries por frame (la inmensa mayoria basura de
    # score bajisimo) y correr ReID sobre las 300 es ~20-100x mas caro que
    # sobre las que realmente importan. Se cachea con un umbral bajo de
    # seguridad (REID_CACHE_MIN_SCORE), bien por debajo de cualquier --conf
    # real que se sweepee, para que el cache sirva para cualquier --conf de la
    # grilla sin recalcular, sin pagar el costo de las 300 cajas crudas.
    reid_cache_path = Path(args.reid_cache) if args.reid_cache else None
    reid_key = reid_key_for(args)
    cached_embs = None
    cached_emb_thresh = None
    reid_cache_hit = False

    if needs_reid and reid_cache_path is not None and reid_cache_path.is_file():
        rc = np.load(reid_cache_path, allow_pickle=True)
        if str(rc["reid_key"]) != reid_key:
            print(f"[WARN] Reid-cache {reid_cache_path} no coincide (modelo ReID distinto); recalculando.")
        elif float(rc["min_score"]) > args.conf:
            print(f"[WARN] Reid-cache {reid_cache_path} calculado con min_score="
                  f"{float(rc['min_score'])} > --conf={args.conf}; recalculando.")
        else:
            cached_embs = rc["embs"]
            cached_emb_thresh = float(rc["min_score"])
            reid_cache_hit = True
            print(f"[INFO] Reid-cache: reusando {reid_cache_path} ({len(cached_embs)} frames, "
                  f"min_score={cached_emb_thresh}), no se recalculan embeddings ReID.")

    new_embs = []
    emb_thresh = min(args.conf, REID_CACHE_MIN_SCORE) if needs_reid else None

    def _build_sess():
        if args.pth_file:
            return build_torch_session(args.config, args.pth_file, args.device)
        return build_session(args.onnx_file, args.device)

    # --reid-source p4 no tiene backbone propio: necesita el nn.Module real del
    # detector para pegarle un forward hook (ver custom_reid.py::P4ReID.attach),
    # y ese hook solo dispara si el detector corre en vivo. Si det_cache tiene
    # hit pero reid_cache NO, hay que forzar la sesion igual -- se ignoran las
    # cajas de esa corrida en vivo (se usan las de det_cache, ver process() mas
    # abajo), solo importa que dispare el hook para poder calcular embeddings
    # frescos este frame. Si reid_cache YA cubre toda la secuencia (ReID interno
    # de BotSort/etc), ni siquiera esto hace falta. --reid-stitch con p4 (capa
    # externa) es distinto: sus embeddings NUNCA se cachean (igual que con
    # dinov2, ver reid_gallery.py) -- necesita el hook disparando en TODOS los
    # frames, sin importar reid_cache_hit.
    # c4_puro y p4fpn_puro tienen la MISMA necesidad que p4 (hook sobre una
    # capa del detector, aunque p4fpn_puro hookea encoder.fpn_blocks.0 en vez
    # de backbone.res_layers.2 -- ver custom_reid.py::P4FpnPuroReID). dinov2_puro
    # NO entra aca: no tiene hook, trabaja sobre pixeles crudos igual que
    # 'dinov2' nativo, asi que puede usar det_cache normalmente.
    needs_live_p4 = args.reid_source in ("p4", "c4_puro", "p4fpn_puro") and (
        needs_reid_stitch or (needs_reid and not reid_cache_hit)
    )
    sess = None if (cache_hit and not needs_live_p4) else _build_sess()
    new_boxes, new_scores, new_labels = [], [], []
    new_alt_scores = []

    # Instancia unica de ReID: se le pasa al tracker (para que with_reid quede
    # habilitado -- BoostTrack/BotSort exigen reid_model no-None para eso) y se
    # reusa aca mismo para calcular los embeddings que se cachean mas abajo. En
    # cache-hit nunca se le llama get_features(), pero igual hace falta la
    # instancia para satisfacer ese chequeo del tracker. --reid-stitch tambien
    # necesita esta instancia (para trackers sin ReID interno, es la UNICA
    # fuente de apariencia). Si el caller (run_eval.py) ya cargo una instancia
    # compartida entre secuencias, se reusa esa en vez de recargar -- --reid-
    # source p4 nunca llega con una instancia compartida (ver run_eval.py):
    # necesita el sess de ESTA secuencia especifica para el hook.
    if reid is None:
        if needs_reid or needs_reid_stitch:
            detector_model = (
                sess.model.model if (sess is not None and isinstance(sess, TorchSession)) else None
            )
            reid = load_reid(args, detector_model=detector_model)
        else:
            reid = None

    reid_gallery = None
    _crop_with_margin = None
    if needs_reid_stitch:
        from reid_gallery import ReidGallery
        from custom_reid import crop_with_margin as _crop_with_margin
        reid_gallery = ReidGallery(cosine_thresh=args.reid_stitch_cosine_thresh,
                                    max_gap=args.reid_stitch_max_gap,
                                    debug=args.reid_stitch_debug,
                                    viz_dir=args.reid_stitch_viz_dir,
                                    viz_top_k=args.reid_stitch_viz_top_k)

    if use_cjm:
        from cjm_byte_track.byte_tracker import BYTETracker

        def _make_tracker():
            return BYTETracker(
                track_thresh=args.track_thresh,
                track_buffer=args.max_age,
                match_thresh=args.match_thresh,
                frame_rate=int(args.fps),
            )
    else:
        from boxmot.trackers import (
            ByteTrack, BotSort, StrongSort, OcSort, DeepOcSort, OccluBoost, BoostTrack
        )
        TRACKERS = {
            "bytetrack":   ByteTrack,
            "botsort":     BotSort,
            "strongsort":  StrongSort,
            "ocsort":      OcSort,
            "deepocsort":  DeepOcSort,
            "occluboost":  OccluBoost,
            "boosttrack":  BoostTrack,
        }
        tracker_cls = TRACKERS.get(args.tracker.lower())
        if tracker_cls is None:
            raise ValueError(f"Tracker desconocido: {args.tracker}. Opciones: {list(TRACKERS)}")

        def _make_tracker():
            if args.tracker.lower() == "bytetrack":
                _reset_track_id_counter("boxmot.trackers.bbox.bytetrack.basetrack")
                return tracker_cls(
                    track_thresh=args.track_thresh,
                    match_thresh=args.match_thresh,
                    track_buffer=args.max_age,
                    frame_rate=int(args.fps),
                )
            elif args.tracker.lower() == "ocsort":
                # OcSort NO comparte la API de ByteTrack -- antes se le pasaban
                # track_thresh/match_thresh/track_buffer/frame_rate por error
                # (bug: BaseTracker.__init__ los descarta en silencio via
                # **kwargs sin avisar, asi que toda corrida previa de ocsort
                # corrio siempre con los defaults de la libreria, ignorando
                # cualquier flag). Params reales de OcSort: min_conf/delta_t/
                # inertia/use_byte/Q_xy_scaling/Q_s_scaling (propios) +
                # det_thresh/max_age/min_hits/iou_threshold/asso_func (via
                # BaseTracker). use_byte=False fijo: --conf ya filtra las
                # detecciones de baja confianza aguas arriba (ver process()
                # mas abajo), asi que la 2da etapa estilo BYTE nunca tendria
                # nada de baja confianza que rescatar.
                return tracker_cls(
                    det_thresh=args.conf,
                    max_age=args.max_age,
                    min_hits=args.min_hits,
                    iou_threshold=args.iou_threshold,
                    asso_func=args.asso_func,
                    delta_t=args.delta_t,
                    inertia=args.inertia,
                    use_byte=False,
                )
            elif args.tracker.lower() == "botsort":
                _reset_track_id_counter("boxmot.trackers.bbox.botsort.basetrack")
                use_reid = not args.no_native_reid
                # get_cmc_method() solo desactiva CMC con None real, no con el
                # string "none" -- lo normalizamos aca para poder barrer
                # ['ecc', 'sof', 'none'] desde la CLI (ver --cmc-sweep en sweep.py).
                cmc_method = None if args.cmc_method.lower() == "none" else args.cmc_method
                return tracker_cls(
                    reid_model=reid,
                    with_reid=use_reid,
                    track_high_thresh=args.track_thresh,
                    match_thresh=args.match_thresh,
                    track_buffer=args.max_age,
                    frame_rate=int(args.fps),
                    appearance_thresh=args.appearance_thresh,
                    proximity_thresh=args.proximity_thresh,
                    new_track_thresh=args.new_track_thresh,
                    cmc_method=cmc_method,
                    fuse_first_associate=args.fuse_first_associate,
                )
            elif args.tracker.lower() == "deepocsort":
                # w_association_emb/alpha_fixed_emb/aw_param: params propios de
                # DeepOcSort. embedding_off=False/aw_off=False/cmc_off=False
                # FIJOS en codigo (no son flags) -- ReID + fusion adaptativa +
                # CMC SIEMPRE activos, nunca se sweepean (decision explicita:
                # siempre queremos DeepOcSort con ReID activo). det_thresh
                # atado a --conf para consistencia con el resto del pipeline;
                # max_age/min_hits/iou_threshold/asso_func via BaseTracker.
                return tracker_cls(
                    reid_model=reid,
                    embedding_off=False,
                    aw_off=False,
                    cmc_off=False,
                    w_association_emb=args.w_association_emb,
                    alpha_fixed_emb=args.alpha_fixed_emb,
                    aw_param=args.aw_param,
                    det_thresh=args.conf,
                    max_age=args.max_age,
                    min_hits=args.min_hits,
                    iou_threshold=args.iou_threshold,
                    asso_func=args.asso_func,
                    delta_t=args.delta_t,
                    inertia=args.inertia,
                )
            elif args.tracker.lower() == "boosttrack":
                # lambda_iou/lambda_mhd/lambda_shape: pesos del costo de asociacion
                # combinado (IoU + Mahalanobis + forma). dlo_boost_coef: cuanto se
                # sube el score de detecciones debiles cercanas a un track existente,
                # antes de aplicar det_thresh (el "boost" que le da nombre al tracker).
                # with_reid=False por default en BoostTrack (al reves que BotSort) --
                # --with-reid lo prende explicitamente. max_age via BaseTracker.
                cmc_method = None if args.cmc_method.lower() == "none" else args.cmc_method
                return tracker_cls(
                    reid_model=reid,
                    with_reid=args.with_reid,
                    lambda_iou=args.lambda_iou,
                    lambda_mhd=args.lambda_mhd,
                    lambda_shape=args.lambda_shape,
                    dlo_boost_coef=args.dlo_boost_coef,
                    cmc_method=cmc_method,
                    max_age=args.max_age,
                )
            else:
                return tracker_cls(
                    reid_model=reid,
                    device=args.reid_device,
                    fp16=False,
                )

    tracker = _make_tracker()

    frames, total  = frame_source(args)

    first = next(frames)
    h, w  = first.shape[:2]

    if cache_hit and len(cached_boxes) != total:
        print(f"[WARN] Det-cache {det_cache_path} tiene {len(cached_boxes)} frames pero la "
              f"secuencia tiene {total}; recalculando.")
        cache_hit = False
        cached_boxes = cached_scores = cached_labels = None
        sess = _build_sess()

    if reid_cache_hit and len(cached_embs) != total:
        print(f"[WARN] Reid-cache {reid_cache_path} tiene {len(cached_embs)} frames pero la "
              f"secuencia tiene {total}; recalculando.")
        reid_cache_hit = False
        cached_embs = None

    output_path = Path(args.output)
    no_video = str(output_path) == '/dev/null'
    if not no_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps, (w, h),
        )
    else:
        writer = None
    print(f"[INFO] Tracker : {args.tracker}")
    if not no_video:
        print(f"[INFO] Salida  : {output_path}  |  {w}×{h} @ {args.fps} fps")

    mot_lines = []
    save_mot = args.mot_output is not None

    # Estadisticas de reidentificacion NATIVA del tracker (sin ningun modulo de
    # re-id externo): un track_id "recuperado" es uno que ya se habia visto
    # antes en esta secuencia pero NO estaba activo en el frame inmediatamente
    # anterior -- o sea, el tracker (ByteTrack via Kalman+IoU, o BotSort via su
    # propio embedding de apariencia) le devolvio la identidad original tras
    # una interrupcion, en vez de crearle un id nuevo. "Nuevo" es la primera
    # vez que ese track_id aparece en la secuencia.
    id_stats = {"seen": set(), "prev_active": set(), "new": 0, "recovered": 0}

    # Estado de --reid-stitch (reidentificacion externa por apariencia, ver
    # reid_gallery.py): raw_to_final mapea el id que puso el tracker al id
    # final ya fusionado. last_crop guarda el ultimo recorte de pixeles de
    # cada track vivo (barato, sin ReID) por si se pierde y hay que enterrar
    # su huella en la galeria -- funciona para dinov2 (puede re-embeber un
    # crop cualquiera despues). p4 NO puede: solo tiene feature map de un
    # forward pass en vivo, nunca de un crop suelto de un frame pasado -- para
    # p4 se usa last_embed en cambio, calculando (barato, ROI-Align sobre el
    # feature map que YA existe este frame) y guardando el embedding de cada
    # track vivo EN CADA frame, listo para enterrar sin volver a tocar pixeles.
    raw_to_final = {}
    # last_crop: cuando --reid-stitch-bank-samples=1 (default), un solo
    # recorte por raw_id (el mas reciente, comportamiento de siempre).
    # crop_reservoir/crop_seen_count: cuando --reid-stitch-bank-samples>1,
    # reservoir sampling (Algorithm R) de hasta N recortes por raw_id a lo
    # largo de TODA su vida activa -- da una muestra representativa del
    # aspecto del track en distintos momentos/angulos, no solo el ultimo.
    last_crop = {}
    crop_reservoir = {}
    crop_seen_count = {}
    last_embed = {}
    p4_stitch = needs_reid_stitch and args.reid_source == "p4"
    # c4_puro: a diferencia de p4_stitch (que solo guarda el ULTIMO embedding,
    # sobreescrito cada frame), acumula las 3 escalas de CADA frame en una
    # lista por raw_id -- necesario para reproducir pool_history() (banco =
    # promedio de num_track_history frames repartidos en el tiempo, no solo
    # el mas reciente), ver custom_reid.py::C4PuroReID.
    c4_stitch = needs_reid_stitch and args.reid_source == "c4_puro"
    track_history_c4 = {}
    # p4fpn_puro: identico a c4_stitch (mismo hook/ROI-Align, ver
    # custom_reid.py::P4FpnPuroReID), solo cambia la capa hookeada.
    p4fpn_stitch = needs_reid_stitch and args.reid_source == "p4fpn_puro"
    track_history_p4fpn = {}
    # dinov2_puro: misma logica de acumular-y-diferir que c4_stitch/p4fpn_stitch,
    # pero acumulando RECORTES DE PIXELES (no embeddings, caros de calcular)
    # -- ver custom_reid.py::DinoV2PuroReID.
    dinov2_puro_stitch = needs_reid_stitch and args.reid_source == "dinov2_puro"
    track_history_dinov2_puro = {}
    # native_puro: mismo criterio de acumular-y-diferir que c4_stitch/
    # p4fpn_stitch (embeddings, no crudos como dinov2_puro, ya que
    # GenericPuroReID no necesita diferir un forward caro por frame -- lo
    # paga igual en cada llamada, ver custom_reid.py::GenericPuroReID) --
    # agregado para la ablacion single-crop vs multi-crop de un ReID real
    # (OSNet/CLIP-ReID) via --native-reid-weights.
    native_puro_stitch = needs_reid_stitch and args.reid_source == "native_puro"
    track_history_native_puro = {}
    bank_samples = max(1, getattr(args, "reid_stitch_bank_samples", 1))
    # final_id -> {"status": "ok"|"fail", "until": frame_id} -- resaltado en
    # el video de intentos de reid-stitch (ver draw_tracks). "ok" = un id
    # nuevo matcheo contra la galeria este frame; "fail" = lo intento (el
    # banco tenia al menos un candidato) pero no supero el umbral. Se
    # mantiene visible `reid_event_hold_frames` cuadros despues del intento
    # para que sea visible al mirar el video a velocidad normal (un solo
    # frame de highlight es casi imperceptible).
    reid_events = {}

    def process(img: np.ndarray, frame_id: int) -> np.ndarray:
        idx = frame_id - 1
        if cache_hit:
            boxes, scores, labels = cached_boxes[idx], cached_scores[idx], cached_labels[idx]
            if needs_live_p4:
                # det_cache ya tiene las cajas de este frame, pero P4ReID
                # necesita que el hook dispare EN ESTE frame para tener el
                # feature map con el que calcular embeddings frescos -- se
                # descartan las cajas/scores de esta corrida en vivo (mismo
                # modelo/imgsz que las cacheadas, por construccion de
                # det_cache), solo importa el efecto secundario del hook.
                im_data, orig_size = preprocess(img, args.imgsz)
                run_inference(sess, im_data, orig_size)
        else:
            im_data, orig_size = preprocess(img, args.imgsz)
            labels, boxes, scores = run_inference(sess, im_data, orig_size)
            if det_cache_path is not None:
                new_boxes.append(boxes)
                new_scores.append(scores)
                new_labels.append(labels)
                if _TM_ALT_CACHE and _TM_LAST_ALT is not None:
                    new_alt_scores.append(_TM_LAST_ALT)

        mask = scores >= args.conf

        embs_f = None
        if needs_reid:
            # thresh: el umbral con el que se calcularon/cachean los embeddings
            # (mas bajo que --conf, ver REID_CACHE_MIN_SCORE). emb_mask siempre
            # incluye a mask (mask ⊆ emb_mask), asi que sub_mask recorta los
            # embeddings ya calculados al subconjunto que ademas pasa --conf.
            thresh = cached_emb_thresh if reid_cache_hit else emb_thresh
            emb_mask = scores >= thresh
            if reid_cache_hit:
                embs_c = cached_embs[idx]
            else:
                boxes_c = boxes[emb_mask].astype(np.float32)
                embs_c = reid.get_features(boxes_c, img) if len(boxes_c) else np.empty((0, 1), dtype=np.float32)
                if reid_cache_path is not None:
                    new_embs.append(embs_c)
            sub_mask = mask[emb_mask]
            embs_f = embs_c[sub_mask]

        boxes_f = boxes[mask].astype(np.float32)
        scores_f = scores[mask].astype(np.float32)
        labels_f = labels[mask].astype(np.float32)

        img_h, img_w = img.shape[:2]
        img_info = (img_h, img_w)
        img_size = (img_h, img_w)

        if use_cjm:
            if len(boxes_f) == 0:
                dets = np.zeros((0, 5), dtype=np.float32)
            else:
                dets = np.column_stack([boxes_f, scores_f])
            online_targets = tracker.update(dets, img_info, img_size)
            tracks = _cjm_tracks_to_draw(online_targets)
            # Con --reid-stitch, el id puede remapearse mas abajo -- el .txt
            # se escribe al final via tracks_to_mot() (sobre tracks, ya
            # remapeado), no aca sobre online_targets (ids crudos).
            if save_mot and not needs_reid_stitch:
                mot_lines.extend(_cjm_tracks_to_mot(online_targets, frame_id))
        else:
            if len(boxes_f) == 0:
                dets = np.empty((0, 6), dtype=np.float32)
            else:
                dets = np.column_stack([boxes_f, scores_f, labels_f])
            tracks = tracker.update(dets, img, embs=embs_f)
            if save_mot and not needs_reid_stitch:
                mot_lines.extend(tracks_to_mot(tracks, frame_id))

        current_ids = {int(row[4]) for row in tracks} if tracks is not None and len(tracks) else set()

        if needs_reid_stitch:
            prev_active_raw = id_stats["prev_active"]

            # OJO: la deteccion de "muerte" (mas abajo) tiene que correr
            # SIEMPRE que needs_reid_stitch, no solo cuando este frame tiene
            # cajas -- antes estaba anidada dentro de `and len(tracks)`, y
            # eso hacia que un frame CUALQUIERA sin ninguna deteccion (0
            # cajas, de cualquier id) se saltee la comparacion entera. Como
            # `id_stats["prev_active"]` se pisa con el set (vacio) de ESE
            # frame sin condicion (mas abajo, fuera de este bloque), la
            # proxima vez que aparecen cajas ya no queda memoria de que un
            # id acababa de desaparecer -- esa identidad nunca se enterraba,
            # quedando invisible para el reid PARA SIEMPRE (confirmado con
            # un caso real: un daño que desaparecio justo antes de un frame
            # con cero detecciones nunca entro al banco; al reaparecer mas
            # tarde, el reid solo tenia candidatos de OTROS daños en el
            # banco y matcheo mal, con confianza alta, contra uno de ellos).
            if len(tracks):
                # Nacimiento: un raw_id nunca visto se compara contra la
                # galeria de perdidos ANTES de aceptarlo como id genuinamente
                # nuevo.
                newborn_idx = [i for i, row in enumerate(tracks) if int(row[4]) not in raw_to_final]
                if newborn_idx:
                    newborn_boxes = np.stack([tracks[i][:4] for i in newborn_idx]).astype(np.float32)
                    newborn_embeds = reid.get_features(newborn_boxes, img)
                    norms = np.linalg.norm(newborn_embeds, axis=1, keepdims=True) + 1e-9
                    newborn_raw_ids = [int(tracks[i][4]) for i in newborn_idx]
                    newborn_crops = None
                    if args.reid_stitch_viz_dir and _crop_with_margin is not None:
                        newborn_crops = [_crop_with_margin(img, tuple(b), args.reid_margin_ratio)
                                          for b in newborn_boxes]
                    bank_size_before = reid_gallery.bank_size
                    # raw_ids: tagea el nombre del archivo de viz con el id resultante (matched
                    # final_id, o el propio raw_id si no matcheo) -- sin esto, build_reid_error_gallery
                    # (dinov2_reid_finetune.py) solo podia indexar por numero de frame, y perdia
                    # comparaciones cuando 2+ nacimientos caian en el MISMO frame.
                    matches = reid_gallery.match_newborns(newborn_embeds / norms, crops=newborn_crops,
                                                           frame_id=frame_id, raw_ids=newborn_raw_ids)
                    for local_i, i in enumerate(newborn_idx):
                        raw_id = int(tracks[i][4])
                        matched = matches[local_i]
                        raw_to_final[raw_id] = matched if matched is not None else raw_id
                        if args.draw_reid_events:
                            if matched is not None:
                                reid_events[matched] = {"status": "ok",
                                                         "until": frame_id + args.reid_event_hold_frames}
                            elif bank_size_before > 0:
                                # intento genuino (habia con que comparar) que no
                                # supero el umbral -- distinto de un nacimiento
                                # con la galeria vacia, que no es un "intento" real
                                reid_events[raw_id] = {"status": "fail",
                                                        "until": frame_id + args.reid_event_hold_frames}

                if p4_stitch:
                    # p4 no puede reembeber un recorte guardado despues -- se
                    # calcula (barato: ROI-Align sobre el feature map que YA
                    # dispararon las lineas de arriba, no un forward pass nuevo)
                    # y se guarda el embedding de CADA track vivo este frame, ANTES
                    # de remapear la columna de id, sobreescribiendo el de frames
                    # anteriores. Si se pierde mas adelante, ya esta listo.
                    all_boxes = np.stack([row[:4] for row in tracks]).astype(np.float32)
                    all_embeds = reid.get_features(all_boxes, img)
                    for row, embed in zip(tracks, all_embeds):
                        last_embed[int(row[4])] = embed
                elif c4_stitch:
                    # Igual idea que p4_stitch (ROI-Align sobre el feature map
                    # YA calculado este frame, sin forward pass nuevo), pero
                    # ACUMULA las 3 escalas de CADA frame en vez de sobreescribir
                    # -- pool_history() necesita el historial completo, no solo
                    # el ultimo frame, para poder repartir num_track_history
                    # observaciones a lo largo de TODA la vida del track (ver
                    # custom_reid.py::C4PuroReID).
                    all_boxes = np.stack([row[:4] for row in tracks]).astype(np.float32)
                    all_scale_embeds = reid.get_scale_features(all_boxes, img)  # (N, 3, D)
                    for row, scale_row in zip(tracks, all_scale_embeds):
                        track_history_c4.setdefault(int(row[4]), []).append(scale_row)
                elif p4fpn_stitch:
                    # Identico a c4_stitch -- ver comentario arriba.
                    all_boxes = np.stack([row[:4] for row in tracks]).astype(np.float32)
                    all_scale_embeds = reid.get_scale_features(all_boxes, img)  # (N, 3, D)
                    for row, scale_row in zip(tracks, all_scale_embeds):
                        track_history_p4fpn.setdefault(int(row[4]), []).append(scale_row)
                elif dinov2_puro_stitch:
                    # A diferencia de c4_stitch/p4fpn_stitch (ROI-Align gratis
                    # sobre un feature map que ya existe), recortar pixeles es
                    # barato pero ENCODEAR con DINOv2 no lo es -- se acumulan
                    # los recortes SIN encodear, y solo se corre el backbone en
                    # pool_history() al perder el track (ver custom_reid.py::
                    # DinoV2PuroReID).
                    all_boxes = np.stack([row[:4] for row in tracks]).astype(np.float32)
                    scale_crops = reid.get_scale_crops(all_boxes, img)  # lista de (3,H,W,3)
                    for row, crop_row in zip(tracks, scale_crops):
                        track_history_dinov2_puro.setdefault(int(row[4]), []).append(crop_row)
                elif native_puro_stitch:
                    # GenericPuroReID: sin feature map compartido que reusar
                    # (a diferencia de c4_stitch/p4fpn_stitch), asi que esto
                    # SI paga 3 forwards reales del modelo por frame -- igual
                    # costo que get_features() en cualquier otro momento, ver
                    # custom_reid.py::GenericPuroReID.
                    all_boxes = np.stack([row[:4] for row in tracks]).astype(np.float32)
                    all_scale_embeds = reid.get_scale_features(all_boxes, img)  # (N, 3, D)
                    for row, scale_row in zip(tracks, all_scale_embeds):
                        track_history_native_puro.setdefault(int(row[4]), []).append(scale_row)
                else:
                    # Recorte barato (sin ReID) de cada track vivo este frame, por
                    # si se pierde en un frame futuro -- se guarda con el raw_id,
                    # ANTES de remapear la columna de id.
                    for row in tracks:
                        crop = _crop_with_margin(img, tuple(row[:4]), args.reid_margin_ratio)
                        if crop is None or crop.size == 0:
                            continue
                        raw_id = int(row[4])
                        last_crop[raw_id] = crop
                        if bank_samples > 1:
                            # reservoir sampling (Algorithm R): mantiene hasta
                            # `bank_samples` recortes representativos de TODA
                            # la vida del track vista hasta ahora, sin saber
                            # de antemano cuanto va a durar.
                            seen = crop_seen_count.get(raw_id, 0) + 1
                            crop_seen_count[raw_id] = seen
                            reservoir = crop_reservoir.setdefault(raw_id, [])
                            if len(reservoir) < bank_samples:
                                reservoir.append(crop)
                            else:
                                j = random.randint(0, seen - 1)
                                if j < bank_samples:
                                    reservoir[j] = crop

                tracks = tracks.copy()
                for i, row in enumerate(tracks):
                    tracks[i, 4] = raw_to_final[int(row[4])]

                # Dos filas de este frame pueden resolver al mismo final_id (ej.
                # el propio tracker recupera un id nativo por su cuenta justo
                # cuando la galeria fusiona OTRO id nativo a esa misma identidad)
                # -- TrackEval rechaza el frame entero si eso pasa. Nos quedamos
                # con la de mayor score.
                best_idx = {}
                for i, row in enumerate(tracks):
                    fid = int(row[4])
                    b = best_idx.get(fid)
                    if b is None or tracks[i, 5] > tracks[b, 5]:
                        best_idx[fid] = i
                if len(best_idx) < len(tracks):
                    tracks = tracks[sorted(best_idx.values())]

            # Muerte: un raw_id activo el frame anterior que ya no aparece
            # entierra su ultima huella conocida en la galeria -- el embedding
            # ya calculado en su ultimo frame vivo (p4) o re-embebiendo su
            # ultimo recorte de pixeles guardado (dinov2). Corre SIEMPRE
            # (aunque este frame no tenga tracks), ver comentario arriba.
            for raw_id in (prev_active_raw - current_ids):
                crop = None
                if p4_stitch:
                    embeds = last_embed.pop(raw_id, None)
                    if embeds is None:
                        continue
                elif c4_stitch:
                    history = track_history_c4.pop(raw_id, None)
                    embeds = reid.pool_history(history) if history else None
                    if embeds is None:
                        continue
                elif p4fpn_stitch:
                    history = track_history_p4fpn.pop(raw_id, None)
                    embeds = reid.pool_history(history) if history else None
                    if embeds is None:
                        continue
                elif dinov2_puro_stitch:
                    history = track_history_dinov2_puro.pop(raw_id, None)
                    embeds = reid.pool_history(history) if history else None
                    if embeds is None:
                        continue
                elif native_puro_stitch:
                    history = track_history_native_puro.pop(raw_id, None)
                    embeds = reid.pool_history(history) if history else None
                    if embeds is None:
                        continue
                elif bank_samples > 1:
                    # varias instancias de la vida del track (reservoir
                    # sampling, ver arriba) -- se embeben TODAS de una en un
                    # solo forward pass batcheado (mas barato que una llamada
                    # por crop). `crop` pasa a ser el reservoir COMPLETO (no
                    # solo el mas reciente): reid_gallery.bury() ahora guarda
                    # todas las muestras, asi que la visualizacion puede
                    # mostrar un filmstrip con las `bank_samples` reales, no
                    # un unico recorte por identidad.
                    crops_multi = crop_reservoir.pop(raw_id, None)
                    crop_seen_count.pop(raw_id, None)
                    last_crop.pop(raw_id, None)
                    if not crops_multi:
                        continue
                    embeds = reid.embed_crops(crops_multi)
                    crop = crops_multi
                else:
                    crop = last_crop.pop(raw_id, None)
                    if crop is None or crop.size == 0:
                        continue
                    h_c, w_c = crop.shape[:2]
                    embeds = reid.get_features(np.array([[0, 0, w_c, h_c]], dtype=np.float32), crop)
                reid_gallery.bury(raw_to_final.get(raw_id, raw_id), embeds,
                                   crops=crop, frame_id=frame_id)
            reid_gallery.age_and_prune(frame_id=frame_id)
            if (dinov2_puro_stitch or native_puro_stitch) and frame_id % 50 == 0:
                # dinov2_puro/native_puro pagan forwards reales del modelo (a
                # diferencia de c4_stitch/p4fpn_stitch, que solo reusan el
                # feature map ya calculado via ROI-Align) -- el allocator de
                # CUDA retiene bloques ya liberados en cache en vez de
                # devolverlos, asi que en videos largos con muchos eventos de
                # nacimiento/perdida el pico de memoria "en uso" puede crecer
                # hasta dejar muy poco margen libre para el siguiente forward
                # batcheado grande (visto en la practica: CUDA OOM con
                # native_puro/CLIP-VehicleID en DeepOC-SORT + w-association-emb,
                # que ya carga su propio modelo de apariencia ademas de este).
                torch.cuda.empty_cache()

            if save_mot:
                mot_lines.extend(tracks_to_mot(tracks, frame_id))

        for tid in current_ids:
            if tid not in id_stats["seen"]:
                id_stats["seen"].add(tid)
                id_stats["new"] += 1
            elif tid not in id_stats["prev_active"]:
                id_stats["recovered"] += 1
        id_stats["prev_active"] = current_ids

        active_reid_events = None
        if args.draw_reid_events and reid_events:
            for k in [k for k, v in reid_events.items() if frame_id > v["until"]]:
                del reid_events[k]
            active_reid_events = reid_events
        return draw_tracks(img, tracks, reid_events=active_reid_events)

    if writer:
        writer.write(process(first, 1))
    else:
        process(first, 1)

    for i, frame in enumerate(tqdm(frames, total=total - 1, desc="Tracking", unit="fr")):
        if writer:
            writer.write(process(frame, i + 2))
        else:
            process(frame, i + 2)

    # % sobre EPISODIOS (cada vez que un track_id arranca una racha de
    # aparicion, sea la primera vez o tras un hueco), no sobre ids unicos --
    # un mismo id puede recuperarse mas de una vez en la secuencia.
    recovered = id_stats["recovered"]
    new = id_stats["new"]
    total_episodes = new + recovered
    pct = 100 * recovered / total_episodes if total_episodes else 0.0
    print(f"[INFO] Reidentificacion nativa ({args.tracker}): {recovered}/{total_episodes} "
          f"episodios recuperados tras una interrupcion ({pct:.1f}%), {new} primeras apariciones "
          f"({len(id_stats['seen'])} ids unicos en total)")

    if writer:
        writer.release()
        print(f"[OK] Vídeo guardado en: {output_path.resolve()}")

    if save_mot:
        mot_path = Path(args.mot_output)
        mot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mot_path, 'w') as f:
            f.writelines(mot_lines)
        print(f"[OK] MOT annotations: {mot_path.resolve()}  ({len(mot_lines)} lines)")

    if det_cache_path is not None and not cache_hit:
        det_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            det_cache_path,
            boxes=np.array(new_boxes, dtype=object),
            scores=np.array(new_scores, dtype=object),
            labels=np.array(new_labels, dtype=object),
            model_key=model_key,
            imgsz=args.imgsz,
        )
        print(f"[OK] Det-cache guardada: {det_cache_path} ({len(new_boxes)} frames)")
        if _TM_ALT_CACHE and len(new_alt_scores) == len(new_boxes) and new_boxes:
            alt_path = Path(str(det_cache_path.parent) + "_ALT") / det_cache_path.name
            alt_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                alt_path,
                boxes=np.array(new_boxes, dtype=object),
                scores=np.array(new_alt_scores, dtype=object),
                labels=np.array(new_labels, dtype=object),
                model_key=model_key,
                imgsz=args.imgsz,
            )
            print(f"[OK] Det-cache ALT guardada: {alt_path}")

    if needs_reid and reid_cache_path is not None and not reid_cache_hit:
        reid_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            reid_cache_path,
            embs=np.array(new_embs, dtype=object),
            reid_key=reid_key,
            min_score=emb_thresh,
        )
        print(f"[OK] Reid-cache guardada: {reid_cache_path} ({len(new_embs)} frames, min_score={emb_thresh})")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RT-DETR ONNX + boxmot tracker → vídeo con IDs persistentes."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", type=str, help="Carpeta de frames (jpg/png/…)")
    src.add_argument("--video",  type=str, help="Fichero de vídeo de entrada")

    p.add_argument("--onnx-file",    type=str,   default=None,
                   help="Modelo ONNX (alternativa a --pth-file/--config)")
    p.add_argument("--pth-file",     type=str,   default=None,
                   help="Checkpoint de entrenamiento RT-DETR (.pth). Requiere --config")
    p.add_argument("--config",       type=str,   default=None,
                   help="YAML de configuración RT-DETR usado para entrenar --pth-file")
    p.add_argument("--output",       type=str,   default="tracked.mp4")
    p.add_argument("--mot-output",   type=str,   default=None,
                   help="Ruta .txt para exportar anotaciones MOT (TrackEval compatible)")
    p.add_argument("--det-cache",    type=str,   default=None,
                   help="Ruta .npz para cachear las detecciones crudas del detector. "
                        "Si ya existe (y coincide modelo/imgsz), se reusa y se salta la inferencia del detector.")
    p.add_argument("--reid-cache",   type=str,   default=None,
                   help="Ruta .npz para cachear los embeddings ReID de cada caja cruda "
                        "(solo si el tracker usa ReID). Si ya existe (y coincide el modelo "
                        "de ReID), se reusa y se saltan los forward pass de ReID.")
    p.add_argument("--tracker",      type=str,   default="bytetrack",
                   choices=["bytetrack", "cjm_bytetrack", "botsort", "strongsort", "ocsort", "deepocsort", "occluboost", "boosttrack"],
                   help="Tracker a usar (default: bytetrack)")
    p.add_argument("--fps",          type=float, default=30.0)
    p.add_argument("--conf",         type=float, default=0.4,
                   help="Umbral de confianza mínimo para pasar dets al tracker (default: 0.4)")
    p.add_argument("--imgsz",        type=int,   default=640)
    p.add_argument("--device",       type=str,   default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--reid-device", type=str,   default="cpu", choices=["cpu", "cuda"],
                   help="[BotSORT/StrongSORT/DeepOCSort/OccluBoost] Device para el modelo ReID "
                        "(default: cpu; los lotes pequeños de recortes no suelen beneficiarse de GPU)")
    p.add_argument("--reid-source",  type=str,   default="native",
                   choices=["native", "dinov2", "p4", "c4_puro", "p4fpn_puro", "dinov2_puro", "native_puro"],
                   help="Fuente del ReID para trackers que lo soportan (BotSort/DeepOcSort/"
                        "BoostTrack): 'native' usa osnet (default), 'dinov2' reemplaza el ReID "
                        "interno del tracker por un embedder DINOv2 fine-tuneado propio "
                        "(requiere --reid-checkpoint), 'p4' usa ROI-Align sobre el feature map "
                        "interno del detector (backbone.res_layers.2) en vez de un backbone de "
                        "imagen propio -- requiere --reid-checkpoint y --pth-file (NO funciona "
                        "con --onnx-file, un InferenceSession no expone activaciones intermedias). "
                        "'c4_puro'/'p4fpn_puro'/'dinov2_puro' -- SOLO junto con --reid-stitch, nunca "
                        "como reid_model= nativo de un tracker -- SIN ningun checkpoint/proyeccion "
                        "entrenada, reproduciendo la metodologia EXACTA validada en el analisis de "
                        "k-fold (query=promedio ajustado+intermedio del frame actual, banco=promedio "
                        "de --reid-stitch-num-track-history frames repartidos en el tiempo): "
                        "'c4_puro' usa backbone.res_layers.2 (C4 crudo, 1024-dim, ROI-Align, requiere "
                        "--pth-file); 'p4fpn_puro' usa encoder.fpn_blocks.0 (P4 REAL -- fusion FPN "
                        "C5+C4, 384-dim, ROI-Align, requiere --pth-file); 'dinov2_puro' usa DINOv2 "
                        "crudo (--reid-model, default facebook/dinov2-small, CLS token sobre recortes "
                        "de pixeles, sin proyeccion) -- ver custom_reid.py::C4PuroReID/"
                        "P4FpnPuroReID/DinoV2PuroReID.")
    p.add_argument("--reid-checkpoint", type=str, default=None,
                   help="[--reid-source dinov2/p4] Checkpoint .pt del modelo fine-tuneado "
                        "(ver damages_tracking/dinov2_reid_finetune.py / p4_reid_finetune.py)")
    p.add_argument("--native-reid-weights", type=str, default=None,
                   help="[--reid-source native] Nombre/ruta del peso preentrenado a usar en vez "
                        "del default de boxmot (osnet_x0_25_msmt17.pt) -- cualquiera de "
                        "boxmot.reid.core.config.TRAINED_URLS (p.ej. osnet_x1_0_msmt17.pt, "
                        "resnet50_msmt17.pt, clip_veri.pt, clip_vehicleid.pt); boxmot lo descarga "
                        "solo si no existe localmente. None = comportamiento previo sin cambios.")
    p.add_argument("--native-reid-half", action="store_true",
                   help="[--reid-source native_puro] Corre el modelo (ej. CLIP-VehicleID) en fp16 "
                        "en vez de fp32 -- reduce a la mitad su huella de memoria GPU. Default False "
                        "(comportamiento previo sin cambios); usar si aparece CUDA OOM con "
                        "DeepOC-SORT (que carga su propio embebedor nativo ademas de este).")
    p.add_argument("--reid-model",   type=str,   default="facebook/dinov2-small",
                   help="[--reid-source dinov2] Nombre HF del backbone -- debe coincidir con el "
                        "usado para entrenar --reid-checkpoint (default: facebook/dinov2-small)")
    p.add_argument("--reid-embedding-dim", type=int, default=256,
                   help="[--reid-source dinov2/p4] Debe coincidir con el usado en el entrenamiento "
                        "de --reid-checkpoint (default: 256)")
    p.add_argument("--reid-margin-ratio", type=float, default=0.2,
                   help="[--reid-source dinov2, --reid-stitch] Margen proporcional agregado a "
                        "cada caja antes de recortar (default: 0.2)")

    # Reidentificacion externa por apariencia (ver reid_gallery.py) -- para
    # trackers SIN ReID interno (ByteTrack, cjm_bytetrack, OcSort), o como
    # capa extra sobre cualquier otro. Usa el ReID de --reid-source/--reid-
    # checkpoint (arriba) solo en dos momentos por segmento de trayectoria:
    # nacimiento y muerte de un id, no en cada frame.
    p.add_argument("--reid-stitch", action="store_true",
                   help="Activa la reidentificacion externa por apariencia (nacimiento/muerte "
                        "de ids, ver reid_gallery.py). Off por default.")
    p.add_argument("--reid-stitch-cosine-thresh", type=float, default=0.5,
                   help="[--reid-stitch] Similitud coseno minima para fusionar un id nuevo con "
                        "uno perdido (default: 0.5)")
    p.add_argument("--reid-stitch-max-gap", type=int, default=200,
                   help="[--reid-stitch] Frames maximos que una identidad perdida permanece en "
                        "la galeria antes de descartarse (default: 200)")
    p.add_argument("--reid-stitch-bank-samples", type=int, default=1,
                   help="[--reid-stitch, dinov2 (no aplica a p4/c4_puro)] Cuantas instancias "
                        "historicas de un track guardar al enterrarlo (muestreadas con reservoir "
                        "sampling a lo largo de TODA su vida activa, no solo el ultimo frame). Con "
                        "el default (1) el comportamiento es igual que antes (un solo embedding, "
                        "el ultimo recorte). Con valores mayores (ej. 10), match_newborns compara "
                        "contra el PROMEDIO de similitud de las N instancias por candidato -- util "
                        "cuando la apariencia de un daño cambia bastante segun el angulo/frame en "
                        "que se vio.")
    p.add_argument("--reid-stitch-num-track-history", type=int, default=4,
                   help="[--reid-stitch, --reid-source c4_puro/p4fpn_puro/dinov2_puro SOLAMENTE] Cuantos frames "
                        "repartidos a lo largo de la vida de un track se usan para armar su "
                        "embedding de banco al perderse (ver C4PuroReID.pool_history en "
                        "custom_reid.py) -- mismo parametro/default que num_track_history en el "
                        "analisis de k-fold (reid_dataloader.py), para que sea directamente "
                        "comparable.")
    p.add_argument("--reid-stitch-single-crop", action="store_true",
                   help="[--reid-stitch, --reid-source c4_puro/p4fpn_puro/dinov2_puro SOLAMENTE] "
                        "ALIAS de compatibilidad = --reid-stitch-query-single-view + "
                        "--reid-stitch-bank-single-view juntos (ver esos dos para el detalle). "
                        "Mantenido para no romper corridas ya lanzadas con este flag unico; para "
                        "la ablacion factorial (aislar banco de query) usa los dos flags por "
                        "separado en vez de este.")
    p.add_argument("--reid-stitch-query-single-view", action="store_true",
                   help="[--reid-stitch, --reid-source c4_puro/p4fpn_puro/dinov2_puro SOLAMENTE] "
                        "Ablacion factorial single-view vs multi-view (M10/M3), eje QUERY: "
                        "get_features() usa SOLO la escala ajustada del frame actual (no promedia "
                        "con la intermedia). El banco (pool_history) no se toca -- para aislar el "
                        "aporte de la query, combinar con --reid-stitch-bank-single-view=False "
                        "(default). Default False = comportamiento previo sin cambios.")
    p.add_argument("--reid-stitch-bank-single-view", action="store_true",
                   help="[--reid-stitch, --reid-source c4_puro/p4fpn_puro/dinov2_puro SOLAMENTE] "
                        "Ablacion factorial single-view vs multi-view (M10/M3), eje BANCO: "
                        "pool_history() ignora --reid-stitch-num-track-history y las 3 escalas -- "
                        "usa SOLO la escala ajustada de la observacion mas reciente al morir el "
                        "track (sin contexto amplio, sin historial repartido en el tiempo). La "
                        "query no se toca -- para aislar el aporte del banco, combinar con "
                        "--reid-stitch-query-single-view=False (default). Default False = "
                        "comportamiento previo sin cambios.")
    p.add_argument("--reid-stitch-debug", action="store_true",
                   help="[--reid-stitch] Imprime cada bury/prune/match (con similitud coseno) del "
                        "banco de reid externo -- para verificar por que un caso especifico matchea o no.")
    p.add_argument("--reid-stitch-viz-dir", type=str, default=None,
                   help="[--reid-stitch] Si se pasa, guarda ahi una imagen por cada nacimiento que "
                        "matchea contra la galeria: el recorte nuevo + los --reid-stitch-viz-top-k "
                        "candidatos comparados (con su similitud coseno, verde el que matcheo).")
    p.add_argument("--reid-stitch-viz-top-k", type=int, default=5,
                   help="[--reid-stitch-viz-dir] Cuantos candidatos del banco mostrar por comparacion "
                        "(default: 5)")
    p.add_argument("--draw-reid-events", action="store_true",
                   help="[--reid-stitch, --output video] Resalta en el video de salida cada intento de "
                        "reidentificacion: marco verde + 'REID OK' cuando un id nuevo matchea contra la "
                        "galeria, marco rojo + 'REID FAIL' cuando lo intenta con candidatos reales en el "
                        "banco pero no supera el umbral (no se marca un nacimiento con la galeria vacia, "
                        "eso no es un intento real). El marco se mantiene --reid-event-hold-frames cuadros "
                        "para que sea visible mirando el video a velocidad normal.")
    p.add_argument("--reid-event-hold-frames", type=int, default=20,
                   help="[--draw-reid-events] Cuantos frames mantener visible el marco de un evento de "
                        "reid (default: 20, ~0.66s a 30fps)")

    # Parámetros ByteTrack (ignorados si se usa otro tracker)
    p.add_argument("--track-thresh", type=float, default=0.5,
                   help="[ByteTrack] Umbral alto primera ronda (default: 0.5)")
    p.add_argument("--match-thresh", type=float, default=0.7,
                   help="[ByteTrack] IoU mínimo para match (default: 0.7)")
    p.add_argument("--max-age",      type=int,   default=30,
                   help="[ByteTrack] Frames máximos en lost (default: 30)")

    # Parámetros BotSort (ignorados si se usa otro tracker)
    p.add_argument("--appearance-thresh", type=float, default=0.25,
                   help="[BotSort] Umbral de similitud de apariencia (ReID) para aceptar "
                        "un match (default: 0.25)")
    p.add_argument("--proximity-thresh", type=float, default=0.5,
                   help="[BotSort] Gate de IoU que habilita el uso de apariencia en la "
                        "asociación (default: 0.5)")
    p.add_argument("--new-track-thresh", type=float, default=0.6,
                   help="[BotSort] Umbral de score para crear un track nuevo (default: 0.6)")
    p.add_argument("--cmc-method", type=str, default="ecc",
                   help="[BotSort] Método de Camera Motion Compensation (default: ecc, "
                        "igual que el default de boxmot)")
    p.add_argument("--no-native-reid", action="store_true",
                   help="[BotSort] Desactiva el ReID nativo (osnet) para medir un BotSort "
                        "\"puro\" (solo Kalman+IoU) como baseline")
    p.add_argument("--fuse-first-associate", action="store_true",
                   help="[BotSort] Fusiona apariencia en el primer paso de asociación "
                        "(default: False, igual que boxmot)")

    # Parámetros DeepOcSort (ignorados si se usa otro tracker)
    p.add_argument("--w-association-emb", type=float, default=0.5,
                   help="[DeepOcSort] Peso de la distancia de apariencia (ReID) en el costo "
                        "de asociación total (default: 0.5)")
    p.add_argument("--alpha-fixed-emb", type=float, default=0.95,
                   help="[DeepOcSort] Tasa fija de actualización EMA del embedding de cada "
                        "track (default: 0.95)")
    p.add_argument("--aw-param", type=float, default=0.5,
                   help="[DeepOcSort] Peso de la fusión adaptativa (adaptive weighting) entre "
                        "apariencia y movimiento (default: 0.5). aw_off queda fijo en False en "
                        "código -- la fusión adaptativa siempre está activa, no se sweepea")
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="[OcSort/DeepOcSort] Gate de IoU para asociación, vía BaseTracker "
                        "(default: 0.3)")

    # Parámetros OcSort/DeepOcSort compartidos (via BaseTracker o propios de
    # cada clase, ver docstring de OcSort/DeepOcSort en boxmot)
    p.add_argument("--min-hits", type=int, default=3,
                   help="[OcSort/DeepOcSort] Hits mínimos antes de confirmar un track "
                        "(default: 3, default de BaseTracker)")
    p.add_argument("--delta-t", type=int, default=3,
                   help="[OcSort/DeepOcSort] Ventana temporal para estimar velocidad/dirección "
                        "(default: 3)")
    p.add_argument("--inertia", type=float, default=0.2,
                   help="[OcSort/DeepOcSort] Peso del término de dirección de velocidad en el "
                        "costo de asociación (default: 0.2)")
    p.add_argument("--asso-func", type=str, default="iou",
                   choices=["iou", "giou", "diou", "ciou", "hmiou"],
                   help="[OcSort/DeepOcSort] Métrica de distancia usada para la asociación "
                        "(default: iou)")

    # Parámetros BoostTrack (ignorados si se usa otro tracker; --cmc-method ya
    # definido arriba, mismo nombre/semántica que en BotSort)
    p.add_argument("--lambda-iou", type=float, default=0.5,
                   help="[BoostTrack] Peso de IoU en el costo de asociación combinado "
                        "(default: 0.5)")
    p.add_argument("--lambda-mhd", type=float, default=0.25,
                   help="[BoostTrack] Peso de la distancia de Mahalanobis en el costo de "
                        "asociación combinado (default: 0.25)")
    p.add_argument("--lambda-shape", type=float, default=0.25,
                   help="[BoostTrack] Peso de la similitud de forma en el costo de "
                        "asociación combinado (default: 0.25)")
    p.add_argument("--dlo-boost-coef", type=float, default=0.65,
                   help="[BoostTrack] Cuán agresivo es el boost de confianza a detecciones "
                        "débiles cercanas a un track existente (default: 0.65)")
    p.add_argument("--with-reid", action="store_true",
                   help="[BoostTrack] Activa el ReID nativo (osnet), apagado por default "
                        "en este tracker (al revés que BotSort)")

    args = p.parse_args()
    if bool(args.onnx_file) == bool(args.pth_file):
        p.error("especifica exactamente uno de --onnx-file o --pth-file")
    if args.pth_file and not args.config:
        p.error("--pth-file requiere --config")
    if args.reid_source == "dinov2" and not args.reid_checkpoint:
        p.error("--reid-source dinov2 requiere --reid-checkpoint")
    if args.reid_source == "p4":
        if not args.reid_checkpoint:
            p.error("--reid-source p4 requiere --reid-checkpoint")
        if not args.pth_file:
            p.error("--reid-source p4 requiere --pth-file (no funciona con --onnx-file, "
                     "un InferenceSession no expone activaciones intermedias para el hook)")
    if args.reid_source == "c4_puro":
        if not args.pth_file:
            p.error("--reid-source c4_puro requiere --pth-file (no funciona con --onnx-file, "
                     "un InferenceSession no expone activaciones intermedias para el hook)")
        if not args.reid_stitch:
            p.error("--reid-source c4_puro solo tiene sentido junto con --reid-stitch (no hay "
                     "checkpoint/proyeccion propia para usarlo como reid_model= nativo de un tracker)")
    if args.reid_source == "p4fpn_puro":
        if not args.pth_file:
            p.error("--reid-source p4fpn_puro requiere --pth-file (no funciona con --onnx-file, "
                     "un InferenceSession no expone activaciones intermedias para el hook)")
        if not args.reid_stitch:
            p.error("--reid-source p4fpn_puro solo tiene sentido junto con --reid-stitch (no hay "
                     "checkpoint/proyeccion propia para usarlo como reid_model= nativo de un tracker)")
    if args.reid_source == "dinov2_puro" and not args.reid_stitch:
        p.error("--reid-source dinov2_puro solo tiene sentido junto con --reid-stitch (no hay "
                 "checkpoint/proyeccion propia para usarlo como reid_model= nativo de un tracker)")
    if args.reid_source == "native_puro" and not args.reid_stitch:
        p.error("--reid-source native_puro solo tiene sentido junto con --reid-stitch (protocolo "
                 "multi-escala/banco-con-historial -- para usarlo como reid_model= nativo de un "
                 "tracker sin --reid-stitch, usa --reid-source native en su lugar)")
    return args


if __name__ == "__main__":
    run(parse_args())


# ──────────────────────────────────────────────────────────────────────────────
# GT-as-predictions sanity check
# ──────────────────────────────────────────────────────────────────────────────

def gt_as_predictions(gt_dir: str, output_dir: str, split: str = "test") -> str:
    """
    Copia los GT como predicciones para verificar que el pipeline de evaluación
    es correcto (todas las métricas deben dar 1).

    Estructura esperada de gt_dir:
        gt_dir/{split}/{seq}/gt/gt.txt

    Genera:
        output_dir/gt_sanity/{seq}.txt

    Retorna la ruta al directorio con los txts generados.
    """
    split_dir = Path(gt_dir) / split
    if not split_dir.exists():
        split_dir = Path(gt_dir)

    out_dir = Path(output_dir) / "gt_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if not sequences:
        raise ValueError(f"No se encontraron secuencias en {split_dir}")

    for seq in sequences:
        gt_file = split_dir / seq / "gt" / "gt.txt"
        if not gt_file.exists():
            print(f"[WARN] GT no encontrado: {gt_file}")
            continue

        out_file = out_dir / f"{seq}.txt"
        lines_out = []
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame_id, tid = parts[0], parts[1]
                x, y, w, h = parts[2], parts[3], parts[4], parts[5]
                lines_out.append(f"{frame_id},{tid},{x},{y},{w},{h},1,-1,-1,-1\n")

        with open(out_file, "w") as f:
            f.writelines(lines_out)
        print(f"[GT-sanity] {seq}: {len(lines_out)} lineas → {out_file}")

    return str(out_dir)