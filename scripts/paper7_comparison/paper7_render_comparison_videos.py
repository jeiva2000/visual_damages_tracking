"""
paper7_render_comparison_videos.py

Genera videos .mp4 (cajas GT en rojo + cajas predichas en verde, con su ID)
para comparar cualitativamente [7] (Perez-Zarate et al., EAAI 2026,
\\cite{perezzarate2026rtdetrvd}) contra nuestra mejor configuracion
(C4 + DeepOC-SORT) sobre las mismas 5 secuencias de test usadas en
tab:comparison_paper7 -- sin reentrenar ni recorrer nada, solo lee los
.txt MOT ya generados por ambas corridas y los frames crudos del dataset.

Fuentes (confirmadas contra el log real, HOTA 51.907~=51.91 / IDF1 68.04
coinciden con la fila "Ours, C4 + DeepOC-SORT" de la tabla):
  - [7]:    external_paper7/mot_rowg/<seq>.txt
  - Ours:   eval_tracking/c4_puro_live_deepocsort_tvdt5seq/txt/<seq>.txt
  - Frames: datasets/dataset_split_2/train/<seq>/img1/*.jpg
  - GT:     datasets/dataset_split_2/mot_gt/train/<seq>/gt/gt.txt

Uso (en el servidor, dentro del env rtdetr):
    python3 scripts/paper7_comparison/paper7_render_comparison_videos.py --outdir paper7_comparison_videos
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2

SEQUENCES = [
    "aveo",
    "VID_20260118_222031",
    "VID_20260118_222136",
    "VID_20260118_222249",
    "VID_20260118_222318",
]

DATASET_ROOT = "/mnt/Disk2/datasets/dataset_split_2"
PAPER7_MOT_DIR = (
    "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking/"
    "external_paper7/mot_rowg"
)
OURS_MOT_DIR = (
    "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/"
    "c4_puro_live_deepocsort_tvdt5seq/txt"
)
SOURCES = {
    "paper7": PAPER7_MOT_DIR,
    "ours_c4_deepocsort": OURS_MOT_DIR,
}


def _load_mot(path: str) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            by_frame[frame].append((tid, x, y, x + w, y + h))
    return by_frame


def _draw(img_bgr, gt_boxes, pred_boxes):
    canvas = img_bgr.copy()
    for tid, x1, y1, x2, y2 in gt_boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f"gt:{tid}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    for tid, x1, y1, x2, y2 in pred_boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(canvas, f"id:{tid}", (x1, min(canvas.shape[0] - 4, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    return canvas


def render_sequence(seq_name: str, mot_path: str, out_path: str, fps: float, draw_gt: bool):
    img_dir = os.path.join(DATASET_ROOT, "train", seq_name, "img1")
    gt_path = os.path.join(DATASET_ROOT, "mot_gt", "train", seq_name, "gt", "gt.txt")

    pred_by_frame = _load_mot(mot_path)
    gt_by_frame = _load_mot(gt_path) if draw_gt else {}

    frame_files = sorted(
        (int(os.path.splitext(fn)[0]), os.path.join(img_dir, fn))
        for fn in os.listdir(img_dir) if fn.lower().endswith((".jpg", ".png"))
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = None
    for fidx, fpath in frame_files:
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            continue
        if writer is None:
            h, w = img_bgr.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        annotated = _draw(img_bgr, gt_by_frame.get(fidx, []), pred_by_frame.get(fidx, []))
        writer.write(annotated)
    if writer is not None:
        writer.release()
        print(f"[OK] {out_path}")
    else:
        print(f"[WARN] {seq_name}: no se encontraron frames en {img_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="paper7_comparison_videos")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-gt", action="store_true", help="No dibujar cajas GT en rojo.")
    args = p.parse_args()

    for seq_name in SEQUENCES:
        for source_name, mot_dir in SOURCES.items():
            mot_path = os.path.join(mot_dir, f"{seq_name}.txt")
            if not os.path.isfile(mot_path):
                print(f"[SKIP] no existe {mot_path}")
                continue
            out_path = os.path.join(args.outdir, source_name, f"{seq_name}.mp4")
            render_sequence(seq_name, mot_path, out_path, args.fps, draw_gt=not args.no_gt)


if __name__ == "__main__":
    main()
