"""
infer_track_reid_gaps.py

Corre un detector (RT-DETRv2/v4 ONNX, ver infers/) + un tracker (ByteTrack o
BotSort, via boxmot) sobre un split MOT-style del dataset (mismo layout que
train_track_query_reid.py: secuencia/gt/gt.txt + secuencia/img1/), y detecta
cuando el tracker FALLA en reidentificar: un track de GT tiene una brecha
temporal real (oclusión -- ver IdentityTrack.find_real_gaps en
reid_dataloader.py) pero el id predicho DESPUÉS de la brecha es un id RECIÉN
NACIDO (nunca visto antes en la secuencia) en vez de continuar el id
predicho de ANTES de la brecha -- es decir, el tracker no reidentificó, creó
una identidad nueva cuando debería haber reusado la vieja.

MÉTODO
------
Por cada frame, empareja las cajas predichas por el tracker contra las cajas
de GT vía IoU (asignación greedy 1-a-1, umbral --iou-match-thresh) para
saber qué id predicho le tocó a cada id de GT en ese frame. Con eso, para
cada brecha real de cada IdentityTrack de GT, compara el id predicho justo
ANTES de la brecha contra el id predicho justo DESPUÉS:

    - iguales                                  -> reidentified (bien)
    - distintos, y el de después es un id       -> new_id_born (EL FALLO
      recién nacido en ese frame                   que pide esta corrida)
    - distintos, pero el de después ya existía  -> id_swap (confusión con
      antes (de otro track activo)                 otro track, categoría
                                                     aparte, se reporta
                                                     igual pero no es el foco)
    - GT sin caja predicha emparejada en          -> unevaluated (miss del
      cualquiera de los dos lados                   detector, no del tracker)

Guarda, por cada `new_id_born`, los crops de la última observación antes de
la brecha (la identidad "perdida") y la primera después (el "nacimiento" que
debió haber sido esa misma identidad) -- mismo criterio visual que
lost_id_crops/new_id_crops en damages_tracking_v2/infer_boxmot.py, pero acá
validado contra GT real en vez de la propia heurística de "lost_stracks" del
tracker.

USO
---
python infer_track_reid_gaps.py \
    --detector rtv2 --onnx path/to/rtdetrv2.onnx \
    --tracker bytetrack --split test \
    --outdir reid_gap_report

python infer_track_reid_gaps.py \
    --detector rtv4 --onnx path/to/rtdetrv4.onnx \
    --tracker botsort --reid-weights osnet_x1_0_msmt17.pt \
    --split test --outdir reid_gap_report_botsort

--tracker botsort --no-native-reid mide un BotSort puramente motion+IoU
(sin apariencia), para comparar contra ByteTrack en igualdad de condiciones.

DEPENDENCIAS
------------
pip install "boxmot==18.0.0" lap opencv-python onnxruntime torch torchvision pillow

OJO con la versión de boxmot -- >=19 reescribió la API pública (ya no expone
ByteTrack/BotSort directamente en boxmot.trackers, sino una clase facade
BoxMOT en boxmot.pipeline) y `update()` deja de devolver un np.ndarray con
las columnas documentadas más abajo. build_tracker_factory/run_sequence acá
están escritos y verificados contra boxmot==18.0.0 (última versión con la
API "clásica" -- BotSort(reid_weights=..., device=..., half=..., with_reid=...),
ByteTrack(), tracker.update(dets, img) -> np.ndarray (x1,y1,x2,y2,id,conf,cls,det_ind),
tracker.lost_stracks). Con boxmot>=19 este script rompe.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from train_track_query_reid import (
    DATA_ROOTS,
    build_track_index,
    discover_sequences,
    resolve_sequence_dirs,
)

import infers.infer_onnx_rtv2 as rtv2
import infers.infer_onnx_rtv4 as rtv4


# --------------------------------------------------------------------------- #
# Detector: dispatch rtv2/rtv4 -> función que corre un solo frame (PIL) y
# devuelve (labels, boxes, scores) SIN threshold -- el threshold lo aplica
# el caller (--det-thresh), igual para ambos detectores.
# --------------------------------------------------------------------------- #

def build_detector(detector_kind: str, onnx_path: str, provider: str):
    import onnxruntime as ort
    providers = [provider] if provider else ort.get_available_providers()
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"[INFO] Detector {detector_kind} cargado ({onnx_path}), device={ort.get_device()}")

    if detector_kind == "rtv2":
        transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

        def infer_fn(im_pil: Image.Image):
            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None]
            im_data = transforms(im_pil)[None]
            labels, boxes, scores = sess.run(
                output_names=None,
                input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
            )
            return labels[0], boxes[0], scores[0]
        return infer_fn

    if detector_kind == "rtv4":
        transforms = T.Compose([T.ToTensor()])

        def infer_fn(im_pil: Image.Image):
            return rtv4.process_image(sess, transforms, im_pil)
        return infer_fn

    raise ValueError(f"--detector desconocido: {detector_kind}")


# --------------------------------------------------------------------------- #
# Tracker: dispatch bytetrack/botsort -> factory que arma una instancia
# NUEVA por secuencia (el estado de un tracker no debe cruzar secuencias).
#
# API de boxmot==18.0.0 (fijada en requirements -- ver nota en el docstring
# del módulo): BotSort.__init__ recibe `reid_weights` (Path a los pesos, NO
# un modelo ya cargado) y arma su propio backend `ReID` internamente SOLO si
# with_reid=True (ver boxmot/trackers/botsort/botsort.py) -- por eso, a
# diferencia de versiones más nuevas de boxmot (>=19, que separan la carga
# del backend de ReID vía boxmot.reid.core.ReID + reid_model=), acá NO hace
# falta instanciar el backend de ReID a mano, alcanza con pasarle la ruta.
# `update()` devuelve un np.ndarray CRUDO (no un wrapper con `.xyxy`/`.id`),
# columnas (x1,y1,x2,y2,id,conf,cls,det_ind) -- ver run_sequence, que indexa
# por columna en vez de usar esos atributos.
# --------------------------------------------------------------------------- #

def build_tracker_factory(tracker_kind: str, device: str, with_native_reid: bool, reid_weights: str):
    if tracker_kind == "bytetrack":
        from boxmot.trackers import ByteTrack
        return lambda: ByteTrack()

    if tracker_kind == "botsort":
        from boxmot.trackers import BotSort
        if with_native_reid:
            print(f"[INFO] ReID nativo de BotSort: {reid_weights}")
        else:
            print("[INFO] BotSort SIN ReID nativo (--no-native-reid) -- solo motion+IoU")

        def factory():
            return BotSort(
                reid_weights=Path(reid_weights), device=device, half=(device != "cpu"),
                with_reid=with_native_reid,
            )
        return factory

    raise ValueError(f"--tracker desconocido: {tracker_kind}")


# --------------------------------------------------------------------------- #
# IoU + asignación greedy 1-a-1 (GT <-> predicciones del tracker), por frame
# --------------------------------------------------------------------------- #

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4), ambas xyxy. Devuelve (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def greedy_match(iou: np.ndarray, thresh: float) -> List[Tuple[int, int]]:
    """Empareja filas (GT) con columnas (predicciones) 1-a-1, mayor IoU
    primero, descartando pares por debajo de `thresh`. Suficiente para esto
    (no hace falta asignación húngara óptima -- el volumen por frame es
    chico y lo que importa es la identidad, no la calidad exacta del
    emparejamiento marginal)."""
    pairs = []
    if iou.size == 0:
        return pairs
    candidates = [(iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1]) if iou[i, j] >= thresh]
    candidates.sort(key=lambda t: -t[0])
    used_i, used_j = set(), set()
    for _, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------- #
# Una secuencia completa: detección + tracking frame a frame, matching contra
# GT, y clasificación de cada brecha real
# --------------------------------------------------------------------------- #

def _index_frames(img_dir: str) -> Dict[int, str]:
    """Mismo criterio que _index_frames en train_track_query_reid.py --
    NO asume padding de ceros en el nombre del archivo, así que el orden
    cronológico correcto es por el int parseado, no por sorted() de string."""
    frame_files: Dict[int, str] = {}
    for fname in os.listdir(img_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        try:
            idx = int(stem)
        except ValueError:
            continue
        frame_files[idx] = os.path.abspath(os.path.join(img_dir, fname))
    return frame_files


def save_crop(image_path: str, bbox: Tuple[float, float, float, float], out_path: str) -> bool:
    img = cv2.imread(image_path)
    if img is None:
        return False
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return False
    cv2.imwrite(out_path, img[y1:y2, x1:x2])
    return True


def run_sequence(
    seq_dir: str, detector_fn, tracker_factory, det_thresh: float, iou_match_thresh: float,
    min_gap_frames: int, gt_conf_thresh: float, gt_class_id: int, gt_visibility_thresh: float,
    crops_dir: str,
) -> List[dict]:
    seq_name = os.path.basename(seq_dir.rstrip("/"))
    img_dir = os.path.join(seq_dir, "img1")
    frame_files = _index_frames(img_dir)
    frame_ids_sorted = sorted(frame_files)
    if not frame_ids_sorted:
        print(f"[WARN] {seq_name}: sin frames en img1/, se omite")
        return []

    index = build_track_index(
        seq_dir, gt_conf_thresh=gt_conf_thresh, gt_class_id=gt_class_id,
        gt_visibility_thresh=gt_visibility_thresh, min_observations=2, sequence_dirs=[seq_dir],
    )

    # --- detección + tracking, un frame a la vez, en orden cronológico ---
    tracker = tracker_factory()
    pred_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # frame -> (ids, xyxy)
    first_seen_frame: Dict[int, int] = {}  # pred_id -> primer frame en que aparece en la secuencia

    for fidx in frame_ids_sorted:
        img_bgr = cv2.imread(frame_files[fidx])
        if img_bgr is None:
            pred_by_frame[fidx] = (np.empty(0, dtype=int), np.empty((0, 4), dtype=np.float32))
            continue
        im_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        labels, boxes, scores = detector_fn(im_pil)
        keep = scores > det_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if len(boxes) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.concatenate(
                [np.asarray(boxes, dtype=np.float32),
                 np.asarray(scores, dtype=np.float32)[:, None],
                 np.asarray(labels, dtype=np.float32)[:, None]],
                axis=1,
            )
        tracks = tracker.update(dets, img_bgr)
        # tracks: np.ndarray CRUDO, columnas (x1,y1,x2,y2,id,conf,cls,det_ind)
        # -- ver nota de versión de boxmot en build_tracker_factory.
        if len(tracks):
            ids = tracks[:, 4].astype(int)
            xyxy = np.asarray(tracks[:, :4], dtype=np.float32)
        else:
            ids = np.empty(0, dtype=int)
            xyxy = np.empty((0, 4), dtype=np.float32)
        pred_by_frame[fidx] = (ids, xyxy)
        for pid in ids.tolist():
            if pid not in first_seen_frame:
                first_seen_frame[pid] = fidx

    # --- matching GT <-> predicciones, por frame (joint, no track-por-track,
    #     para que 2 GT boxes del mismo frame no puedan reclamar la misma
    #     predicción) ---
    gt_by_frame: Dict[int, List[Tuple[object, int]]] = {}
    for track in index.tracks:
        for oi, obs in enumerate(track.observations):
            gt_by_frame.setdefault(obs.frame_idx, []).append((track, oi))

    pred_id_for_obs: Dict[Tuple[str, int], Optional[int]] = {}
    for fidx, entries in gt_by_frame.items():
        pred_ids, pred_xyxy = pred_by_frame.get(fidx, (np.empty(0, dtype=int), np.empty((0, 4), dtype=np.float32)))
        gt_boxes = np.array([entries_track.observations[oi].bbox for entries_track, oi in entries], dtype=np.float32)
        iou = iou_matrix(gt_boxes, pred_xyxy)
        pairs = greedy_match(iou, iou_match_thresh)
        matched = dict(pairs)
        for gi, (track, oi) in enumerate(entries):
            pred_id_for_obs[(track.identity_id, oi)] = int(pred_ids[matched[gi]]) if gi in matched else None

    # --- clasificación de cada brecha real ---
    results = []
    for track in index.tracks:
        gaps = track.find_real_gaps(min_gap_frames)
        for gi in gaps:
            before_obs, after_obs = track.observations[gi - 1], track.observations[gi]
            pred_before = pred_id_for_obs.get((track.identity_id, gi - 1))
            pred_after = pred_id_for_obs.get((track.identity_id, gi))

            if pred_before is None or pred_after is None:
                outcome = "unevaluated"
            elif pred_before == pred_after:
                outcome = "reidentified"
            elif first_seen_frame.get(pred_after) == after_obs.frame_idx:
                outcome = "new_id_born"
            else:
                outcome = "id_swap"

            results.append({
                "sequence": seq_name, "gt_identity_id": track.identity_id,
                "frame_before": before_obs.frame_idx, "frame_after": after_obs.frame_idx,
                "gap_len": after_obs.frame_idx - before_obs.frame_idx,
                "pred_id_before": pred_before, "pred_id_after": pred_after,
                "outcome": outcome,
            })

            if outcome == "new_id_born":
                base = f"{seq_name}_{track.identity_id.split('__')[-1]}_gap{gi}"
                save_crop(before_obs.image_path, before_obs.bbox,
                          os.path.join(crops_dir, f"{base}_lost_f{before_obs.frame_idx}.jpg"))
                save_crop(after_obs.image_path, after_obs.bbox,
                          os.path.join(crops_dir, f"{base}_newborn_f{after_obs.frame_idx}.jpg"))

    n_new_born = sum(1 for r in results if r["outcome"] == "new_id_born")
    print(f"[OK] {seq_name}: {len(results)} brechas reales, {n_new_born} new_id_born")
    return results


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", choices=["rtv2", "rtv4"], required=True)
    p.add_argument("--onnx", required=True, help="Checkpoint ONNX del detector.")
    p.add_argument("--provider", default="CUDAExecutionProvider",
                    help="Execution provider de onnxruntime (default: CUDAExecutionProvider).")
    p.add_argument("--det-thresh", type=float, default=0.4,
                    help="Umbral de score mínimo para pasarle una detección al tracker.")

    p.add_argument("--tracker", choices=["bytetrack", "botsort"], required=True)
    p.add_argument("--no-native-reid", action="store_true",
                    help="[botsort] Desactiva el ReID nativo (osnet) -- mide un BotSort puramente "
                         "motion+IoU, en igualdad de condiciones con ByteTrack.")
    p.add_argument("--reid-weights", default="osnet_x1_0_msmt17.pt",
                    help="[botsort, salvo --no-native-reid] Pesos del backend de ReID nativo de boxmot.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    p.add_argument("--split", choices=list(DATA_ROOTS), default="test")
    p.add_argument("--data-root", default=None,
                    help="Si se pasa, ignora --split y usa este path directo (bypassea SPLIT_OVERRIDES, "
                         "ver resolve_sequence_dirs en train_track_query_reid.py).")
    p.add_argument("--gt-conf-thresh", type=float, default=0.0)
    p.add_argument("--gt-class-id", type=int, default=-1)
    p.add_argument("--gt-visibility-thresh", type=float, default=0.0)
    p.add_argument("--min-gap-frames", type=int, default=3,
                    help="Brecha temporal mínima (frames) en el GT para contar como reaparición real -- "
                         "mismo default que ReIDEpisodeDataset (ver reid_dataloader.py).")
    p.add_argument("--iou-match-thresh", type=float, default=0.5,
                    help="IoU mínimo para emparejar una caja predicha con una caja de GT en el mismo frame.")

    p.add_argument("--max-sequences", type=int, default=None,
                    help="Corta a las primeras N secuencias del split -- útil para una corrida rápida de prueba.")
    p.add_argument("--outdir", default="reid_gap_report")
    return p.parse_args()


def run_reid_gap_report(
    detector: str, onnx_path: str, provider: str, det_thresh: float,
    tracker: str, no_native_reid: bool, reid_weights: str, device: str,
    split: str, data_root: Optional[str],
    gt_conf_thresh: float, gt_class_id: int, gt_visibility_thresh: float,
    min_gap_frames: int, iou_match_thresh: float, max_sequences: Optional[int],
    outdir: str, force: bool = False,
) -> str:
    """Cuerpo real de la corrida completa -- separado de main()/parse_args()
    para poder llamarse EN PROCESO desde train_track_query_reid.py (ver
    --reid-gap-report en el subcomando train), no solo desde la CLI de este
    script. Devuelve la ruta al reid_gap_report.json.

    Idempotente por diseño (`force=False`): si `outdir/reid_gap_report.json`
    ya existe, no vuelve a correr detector+tracker sobre todo el split (caro)
    -- pensado para el caso de uso real de train_track_query_reid.py, donde
    esto se ejecuta ANTES de cada corrida de entrenamiento y no tiene sentido
    repetirlo si el split/detector/tracker no cambiaron."""
    report_path = os.path.join(outdir, "reid_gap_report.json")
    if os.path.exists(report_path) and not force:
        print(f"[INFO] {report_path} ya existe, se omite la corrida (force=True para reprocesar)")
        return report_path

    data_root, seq_dirs = resolve_sequence_dirs(data_root, split)
    if seq_dirs is None:
        seq_dirs = discover_sequences(data_root)
    if max_sequences is not None:
        seq_dirs = seq_dirs[:max_sequences]
    print(f"[INFO] {len(seq_dirs)} secuencias a procesar (root={data_root})")

    detector_fn = build_detector(detector, onnx_path, provider)
    tracker_factory = build_tracker_factory(
        tracker, device, with_native_reid=(not no_native_reid), reid_weights=reid_weights,
    )

    os.makedirs(outdir, exist_ok=True)
    crops_dir = os.path.join(outdir, "new_id_born_crops")
    os.makedirs(crops_dir, exist_ok=True)

    all_results = []
    for seq_dir in seq_dirs:
        print(f"[INFO] Procesando secuencia {os.path.basename(seq_dir.rstrip('/'))}")
        all_results.extend(run_sequence(
            seq_dir, detector_fn, tracker_factory, det_thresh, iou_match_thresh,
            min_gap_frames, gt_conf_thresh, gt_class_id, gt_visibility_thresh,
            crops_dir,
        ))

    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    counts = Counter(r["outcome"] for r in all_results)
    total = len(all_results)
    print(f"\n[RESUMEN] {total} brechas reales de GT evaluadas ({tracker}"
          f"{'+osnet' if (tracker == 'botsort' and not no_native_reid) else ''}):")
    for outcome in ("reidentified", "new_id_born", "id_swap", "unevaluated"):
        n = counts.get(outcome, 0)
        pct = 100 * n / total if total else 0.0
        print(f"  {outcome:15s}: {n:5d} ({pct:5.1f}%)")

    evaluable = total - counts.get("unevaluated", 0)
    if evaluable:
        fail_rate = counts.get("new_id_born", 0) / evaluable
        print(f"\n[FALLO DE REID] {counts.get('new_id_born', 0)}/{evaluable} brechas evaluables "
              f"({100 * fail_rate:.1f}%): el tracker nació un id nuevo en vez de reidentificar.")
    print(f"[OK] Reporte guardado en {report_path}, crops de fallos en {crops_dir}")
    return report_path


def main():
    args = parse_args()
    run_reid_gap_report(
        detector=args.detector, onnx_path=args.onnx, provider=args.provider, det_thresh=args.det_thresh,
        tracker=args.tracker, no_native_reid=args.no_native_reid, reid_weights=args.reid_weights,
        device=args.device, split=args.split, data_root=args.data_root,
        gt_conf_thresh=args.gt_conf_thresh, gt_class_id=args.gt_class_id,
        gt_visibility_thresh=args.gt_visibility_thresh, min_gap_frames=args.min_gap_frames,
        iou_match_thresh=args.iou_match_thresh, max_sequences=args.max_sequences, outdir=args.outdir,
        force=True,  # la CLI standalone siempre corre -- el cache es para el caller de train_track_query_reid.py
    )


if __name__ == "__main__":
    main()
