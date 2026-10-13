"""
paper7_required_reid_events.py

Adaptacion de infer_track_reid_gaps.py::run_sequence para construir
required_reid_events.json (y, de paso, el desglose de brechas reales estilo
Tabla V) a partir de la salida MOT YA CALCULADA de [7] (Perez-Zarate et al.,
EAAI 2026, \\cite{perezzarate2026rtdetrvd}), en vez de correr un detector+
tracker propio en vivo -- respondiendo M4 del checklist de revision con el
enfoque que propuso el usuario: identificar los casos donde el transformer
entrenado de [7] falla en reidentificar, y (en un paso posterior, con
analyze_dinov2_nn_baseline.py --tracker-failures-json/--required-reid-events)
probar si NUESTRA senal de apariencia (C4, ya calibrada en 0.9352) los
hubiera resuelto.

Unica diferencia real con infer_track_reid_gaps.py::run_sequence: el paso de
"deteccion + tracking, un frame a la vez" (lineas 486-538 de ese archivo) se
reemplaza por parsear un archivo MOT ya existente. Todo lo demas (matching
GT<->prediccion por IoU, _count_required_reidentifications, clasificacion de
huecos reales) se reusa TAL CUAL, importado sin modificar, de
infer_track_reid_gaps.py.

IMPORTANTE (ver tracker_failure_dataset.py): los crops usados para calcular
embeddings en el paso posterior salen del GT (bbox real), no de las cajas
predichas por [7] -- por eso este script no necesita las cajas de [7] para
nada mas que decidir CUANDO cambio el id predicho, no para las imagenes que
se comparan despues.

Uso:
    python3 paper7_required_reid_events.py \\
        --paper7-mot-dir /path/a/custom_mot_2/TrackEval/data/trackers/mot_challenge/damages-test/<config>/data \\
        --sequences aveo VID_20260118_222031 VID_20260118_222136 VID_20260118_222249 VID_20260118_222318 \\
        --split train --outdir paper7_reid_gap_report

Luego:
    python3 analyze_dinov2_nn_baseline.py --split train \\
        --required-reid-events paper7_reid_gap_report/required_reid_events.json \\
        --encoder p4 --match-threshold 0.9352 --same-video-only
"""
import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from infer_track_reid_gaps import _count_required_reidentifications, greedy_match, iou_matrix
from train_track_query_reid import build_track_index, resolve_sequence_dirs


def _load_paper7_mot(mot_path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Parsea un .txt MOT-format (frame,id,x,y,w,h,conf,...) a
    {frame: (ids, xyxy)} -- mismo shape que pred_by_frame en
    infer_track_reid_gaps.py::run_sequence."""
    by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    with open(mot_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame, tid, x, y, w, h = (
                int(float(parts[0])), int(float(parts[1])),
                float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]),
            )
            by_frame.setdefault(frame, []).append((tid, x, y, x + w, y + h))

    pred_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for frame, rows in by_frame.items():
        ids = np.array([r[0] for r in rows], dtype=int)
        xyxy = np.array([r[1:] for r in rows], dtype=np.float32)
        pred_by_frame[frame] = (ids, xyxy)
    return pred_by_frame


def _first_seen_frame(pred_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, int]:
    first_seen: Dict[int, int] = {}
    for frame in sorted(pred_by_frame.keys()):
        ids, _ = pred_by_frame[frame]
        for pid in ids.tolist():
            if pid not in first_seen:
                first_seen[pid] = frame
    return first_seen


def process_sequence(
    seq_dir: str, mot_path: str, iou_match_thresh: float, min_gap_frames: int,
) -> Tuple[str, Dict[Tuple[str, int], Optional[int]], List[dict], object]:
    """Devuelve (seq_name, pred_id_for_obs, real_gap_results, index) para UNA
    secuencia -- index se arma aca mismo (sequence_dirs=[seq_dir]) para poder
    combinarlo despues con las demas secuencias sin recorrer el dataset
    entero."""
    seq_name = os.path.basename(seq_dir.rstrip("/"))
    index = build_track_index(None, sequence_dirs=[seq_dir])

    pred_by_frame = _load_paper7_mot(mot_path)
    first_seen_frame = _first_seen_frame(pred_by_frame)

    gt_by_frame: Dict[int, List[Tuple[object, int]]] = {}
    for track in index.tracks:
        for oi, obs in enumerate(track.observations):
            gt_by_frame.setdefault(obs.frame_idx, []).append((track, oi))

    pred_id_for_obs: Dict[Tuple[str, int], Optional[int]] = {}
    for fidx, entries in gt_by_frame.items():
        pred_ids, pred_xyxy = pred_by_frame.get(fidx, (np.empty(0, dtype=int), np.empty((0, 4), dtype=np.float32)))
        gt_boxes = np.array([t.observations[oi].bbox for t, oi in entries], dtype=np.float32)
        iou = iou_matrix(gt_boxes, pred_xyxy)
        pairs = greedy_match(iou, iou_match_thresh)
        matched = dict(pairs)
        for gi, (track, oi) in enumerate(entries):
            pred_id_for_obs[(track.identity_id, oi)] = int(pred_ids[matched[gi]]) if gi in matched else None

    # Clasificacion de huecos reales, estilo Tabla V (bonus, no estrictamente
    # necesario para required_reid_events, pero barato de calcular aca mismo
    # con lo que ya tenemos).
    real_gap_results = []
    for track in index.tracks:
        for gi in track.find_real_gaps(min_gap_frames):
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
            real_gap_results.append({
                "sequence": seq_name, "gt_identity_id": track.identity_id,
                "frame_before": before_obs.frame_idx, "frame_after": after_obs.frame_idx,
                "gap_len": after_obs.frame_idx - before_obs.frame_idx,
                "pred_id_before": pred_before, "pred_id_after": pred_after,
                "outcome": outcome,
            })

    return seq_name, pred_id_for_obs, real_gap_results, index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper7-mot-dir", required=True,
                   help="Carpeta con los .txt MOT de [7] (un archivo por secuencia, mismo nombre que la secuencia).")
    p.add_argument("--sequences", nargs="+", required=True)
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Split del dataset ACTUAL donde viven estas secuencias (ver seqinfo.ini) -- "
                        "solo se usa para resolve_sequence_dirs, las --sequences ya filtran a las 5 de interes.")
    p.add_argument("--iou-match-thresh", type=float, default=0.5)
    p.add_argument("--min-gap-frames", type=int, default=3)
    p.add_argument("--outdir", default="paper7_reid_gap_report")
    args = p.parse_args()

    data_root, _ = resolve_sequence_dirs(None, args.split)

    all_required_reid_events: List[dict] = []
    all_real_gap_results: List[dict] = []
    for seq_name in args.sequences:
        seq_dir = os.path.join(data_root, seq_name)
        mot_path = os.path.join(args.paper7_mot_dir, f"{seq_name}.txt")
        if not os.path.isfile(mot_path):
            raise FileNotFoundError(f"No se encontro {mot_path}")
        print(f"[INFO] Procesando {seq_name} (mot={mot_path})")
        _, pred_id_for_obs, real_gap_results, index = process_sequence(
            seq_dir, mot_path, args.iou_match_thresh, args.min_gap_frames,
        )
        n_required_reid, _required_reid_details, required_reid_events = _count_required_reidentifications(
            seq_name, index, pred_id_for_obs,
        )
        print(f"  required_reid: {n_required_reid}, huecos reales: {len(real_gap_results)}")
        all_required_reid_events.extend(required_reid_events)
        all_real_gap_results.extend(real_gap_results)

    os.makedirs(args.outdir, exist_ok=True)
    events_path = os.path.join(args.outdir, "required_reid_events.json")
    with open(events_path, "w") as f:
        json.dump(all_required_reid_events, f, indent=2)
    gaps_path = os.path.join(args.outdir, "reid_gap_report.json")
    with open(gaps_path, "w") as f:
        json.dump(all_real_gap_results, f, indent=2)

    print(f"\n[REQUIRED-REID TOTAL] {len(all_required_reid_events)} eventos en {events_path}")

    counts = Counter(r["outcome"] for r in all_real_gap_results)
    total = len(all_real_gap_results)
    print(f"\n[RESUMEN estilo Tabla V] {total} brechas reales de GT evaluadas ([7]):")
    for outcome in ("reidentified", "new_id_born", "id_swap", "unevaluated"):
        n = counts.get(outcome, 0)
        pct = 100 * n / total if total else 0.0
        print(f"  {outcome:15s}: {n:5d} ({pct:5.1f}%)")
    print(f"[OK] Reporte de huecos reales guardado en {gaps_path}")


if __name__ == "__main__":
    main()
