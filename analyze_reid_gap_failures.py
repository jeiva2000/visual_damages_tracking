"""
analyze_reid_gap_failures.py

Retoma la investigación pausada sobre las corridas EN VIVO end-to-end de
Corrida 15 (README.md) -- `main_track.py` vía `run_eval.py --reid-stitch`,
72 secuencias, salidas en `eval_tracking/{c4_puro,p4fpn_puro,dinov2_puro}_
live_{tracker}_alldata/txt/*.txt` -- las que respaldan
`paper/live_validation/table_live_c4_vs_native_alldata.tex` y
`table_live_all_methods_idf1_idsw_alldata.tex`.

PREGUNTA
--------
Para el tracker+método elegido (pensado para el mejor combo real de esas
tablas, ej. OC-SORT+C4 u OC-SORT+P4-FPN, ambos casi empatados en IDF1/IDSW),
de cada brecha real de GT donde el resultado fue `new_id_born` (el tracker
creó un id nuevo en vez de reidentificar), ¿cuánto es por el parámetro de
poda del banco (`--reid-stitch-max-gap`, default 200 -- ver
`reid_gallery.py::age_and_prune`, que borra una identidad perdida antes de
que `match_newborns` tenga la oportunidad de compararla) y cuánto es un
fallo genuino de similitud (la identidad seguía en el banco pero el coseno
nunca superó el umbral calibrado)?

MÉTODO (dos pasos, el segundo opcional vía --relaunch)
-------------------------------------------------------
1. Clasifica cada brecha real de GT (misma lógica que `run_sequence()` en
   `infer_track_reid_gaps.py` -- matching IoU greedy por frame contra las
   predicciones YA CALCULADAS de la corrida en vivo, en vez de correr
   detector+tracker de nuevo) en reidentified/new_id_born/id_swap/
   unevaluated, y separa los new_id_born en gap_len > --max-gap (la
   identidad fue podada del banco ANTES de que el reid pudiera intentar
   nada -- "el reid no se lanzó") vs gap_len <= --max-gap (la identidad
   seguía viva en el banco -- el reid sí se lanzó y no matcheó).

2. Con --relaunch: para las secuencias con al menos un caso gap>max_gap,
   vuelve a correr `main_track.py` (mismos hiperparámetros de tracker que
   `eval_tracking/run_puro_live_alldata.sh`, la corrida real) pero con
   `--reid-stitch-max-gap` prácticamente ilimitado -- así el banco nunca
   poda esas identidades y `match_newborns` sí llega a comparar. Si en esa
   corrida sin límite el caso pasa a `reidentified`, confirma que la poda
   era la única causa; si sigue fallando, es un fallo genuino de similitud
   (o de otra índole) que el gap no explica.

USO
---
Paso 1 (rápido, solo lee texto, no necesita GPU):
    python analyze_reid_gap_failures.py \
        --data-root /path/to/eval_tracking/all_dataset_view/all \
        --live-run-dir /path/to/eval_tracking/c4_puro_live_ocsort_alldata \
        --tracker ocsort --reid-source c4_puro \
        --outdir gap_cause_ocsort_c4

Paso 2 (relanza el reid sobre las secuencias afectadas, necesita GPU + el
mismo checkpoint/config que se usó en la corrida original):
    python analyze_reid_gap_failures.py \
        --data-root /path/to/eval_tracking/all_dataset_view/all \
        --live-run-dir /path/to/eval_tracking/c4_puro_live_ocsort_alldata \
        --tracker ocsort --reid-source c4_puro \
        --outdir gap_cause_ocsort_c4 --relaunch \
        --eval-tracking-dir /path/to/eval_tracking \
        --pth-file /path/to/rtdetrv2_v3_qcs_videos.pth \
        --config /path/to/rtdetrv2_r101vd_6x_coco.yml

Corre en el mismo entorno/máquina donde vive `eval_tracking/` (el servidor
remoto) -- este script importa `train_track_query_reid`/`infer_track_reid_
gaps` de este repo y, con --relaunch, invoca `main_track.py` de
`eval_tracking/` como subproceso.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'reid_lib'))  # library modules

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from train_track_query_reid import build_track_index
from infer_track_reid_gaps import iou_matrix, greedy_match

# Hiperparámetros por tracker, copiados TAL CUAL de la corrida real
# (`eval_tracking/run_puro_live_alldata.sh`) -- si ese script cambia, hay
# que sincronizar esto a mano para que el --relaunch sea comparable.
TRACKER_HPARAMS: Dict[str, List[str]] = {
    "bytetrack": ["--track-thresh", "0.2", "--match-thresh", "0.9", "--max-age", "30"],
    "botsort": ["--track-thresh", "0.2", "--new-track-thresh", "0.2", "--match-thresh", "0.8",
                "--max-age", "30", "--proximity-thresh", "0.5", "--appearance-thresh", "0.4",
                "--cmc-method", "ecc", "--no-native-reid"],
    "ocsort": ["--max-age", "60", "--min-hits", "5", "--iou-threshold", "0.3",
               "--delta-t", "5", "--inertia", "0.4", "--asso-func", "iou"],
    "deepocsort": ["--max-age", "30", "--min-hits", "5", "--iou-threshold", "0.4",
                   "--w-association-emb", "0.7", "--alpha-fixed-emb", "0.99",
                   "--aw-param", "0.5", "--delta-t", "2", "--inertia", "0.4", "--asso-func", "iou"],
}
# Umbrales calibrados, también de run_puro_live_alldata.sh.
REID_SOURCE_THRESH: Dict[str, float] = {
    "c4_puro": 0.9352, "p4fpn_puro": 0.7988, "dinov2_puro": 0.6703,
}
# Mayor que cualquier secuencia real del dataset (clips cortos, muy por
# debajo de esto) -- con este valor, ReidGallery.age_and_prune() nunca
# borra una identidad por edad, así que match_newborns() SIEMPRE llega a
# comparar contra todo lo que sigue "perdido". Es la forma de "lanzar" la
# comparación de reid que el banco, con el --reid-stitch-max-gap real
# (default 200), nunca llegó a intentar.
UNCAPPED_MAX_GAP = 1_000_000


def read_mot_predictions(txt_path: str) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    """frame -> [(pred_id, (x1,y1,x2,y2)), ...]. Mismo formato que escribe
    tracks_to_mot() en main_track.py (frame,id,x,y,w,h,conf,-1,-1,-1)."""
    by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    if not os.path.isfile(txt_path):
        return by_frame
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            pid = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            by_frame.setdefault(frame, []).append((pid, (x, y, x + w, y + h)))
    return by_frame


def first_seen_by_pred_id(pred_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]) -> Dict[int, int]:
    first_seen: Dict[int, int] = {}
    for frame in sorted(pred_by_frame):
        for pid, _ in pred_by_frame[frame]:
            if pid not in first_seen:
                first_seen[pid] = frame
    return first_seen


def classify_sequence_gaps(
    seq_dir: str, pred_txt_path: str, min_gap_frames: int, iou_match_thresh: float,
    gt_conf_thresh: float = 0.0, gt_class_id: int = -1, gt_visibility_thresh: float = 0.0,
) -> List[dict]:
    """Misma clasificación reidentified/new_id_born/id_swap/unevaluated que
    `run_sequence()` en infer_track_reid_gaps.py, pero leyendo predicciones
    YA CALCULADAS (de una corrida en vivo con --reid-stitch) en vez de
    correr detector+tracker de nuevo."""
    seq_name = os.path.basename(seq_dir.rstrip("/"))
    index = build_track_index(
        seq_dir, gt_conf_thresh=gt_conf_thresh, gt_class_id=gt_class_id,
        gt_visibility_thresh=gt_visibility_thresh, min_observations=2, sequence_dirs=[seq_dir],
    )
    pred_by_frame = read_mot_predictions(pred_txt_path)
    first_seen_frame = first_seen_by_pred_id(pred_by_frame)

    gt_by_frame: Dict[int, List[Tuple[object, int]]] = {}
    for track in index.tracks:
        for oi, obs in enumerate(track.observations):
            gt_by_frame.setdefault(obs.frame_idx, []).append((track, oi))

    pred_id_for_obs: Dict[Tuple[str, int], Optional[int]] = {}
    for fidx, entries in gt_by_frame.items():
        preds = pred_by_frame.get(fidx, [])
        pred_ids = np.array([p[0] for p in preds], dtype=int)
        pred_xyxy = (np.array([p[1] for p in preds], dtype=np.float32)
                     if preds else np.empty((0, 4), dtype=np.float32))
        gt_boxes = np.array([t.observations[oi].bbox for t, oi in entries], dtype=np.float32)
        iou = iou_matrix(gt_boxes, pred_xyxy)
        pairs = greedy_match(iou, iou_match_thresh)
        matched = dict(pairs)
        for gi, (track, oi) in enumerate(entries):
            pred_id_for_obs[(track.identity_id, oi)] = int(pred_ids[matched[gi]]) if gi in matched else None

    results = []
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

            results.append({
                "sequence": seq_name, "gt_identity_id": track.identity_id,
                "frame_before": before_obs.frame_idx, "frame_after": after_obs.frame_idx,
                "gap_len": after_obs.frame_idx - before_obs.frame_idx,
                "pred_id_before": pred_before, "pred_id_after": pred_after,
                "outcome": outcome,
            })
    return results


def relaunch_sequence(
    main_track_py: str, data_root: str, seq: str, tracker: str, reid_source: str,
    pth_file: str, config: str, conf: float, imgsz: int, fps: float, device: str,
    out_txt: str, log_path: str,
) -> int:
    img_dir = os.path.join(data_root, seq, "img1")
    cmd = [
        sys.executable, main_track_py,
        "--frames", img_dir, "--pth-file", pth_file, "--config", config,
        "--output", os.devnull, "--mot-output", out_txt,
        "--tracker", tracker, "--conf", str(conf), "--imgsz", str(imgsz),
        "--fps", str(fps), "--device", device, "--reid-device", device,
        "--reid-stitch", "--reid-source", reid_source,
        "--reid-stitch-cosine-thresh", str(REID_SOURCE_THRESH[reid_source]),
        "--reid-stitch-max-gap", str(UNCAPPED_MAX_GAP),
    ] + TRACKER_HPARAMS[tracker]
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return r.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="eval_tracking/all_dataset_view/all -- una carpeta por secuencia con gt/gt.txt + img1/.")
    p.add_argument("--live-run-dir", required=True,
                   help="p.ej. eval_tracking/c4_puro_live_ocsort_alldata (contiene txt/<seq>.txt).")
    p.add_argument("--tracker", required=True, choices=list(TRACKER_HPARAMS))
    p.add_argument("--reid-source", required=True, choices=list(REID_SOURCE_THRESH))
    p.add_argument("--max-gap", type=int, default=200,
                   help="--reid-stitch-max-gap usado en la corrida original a analizar (default de main_track.py, "
                        "no se pasó explícito en run_puro_live_alldata.sh).")
    p.add_argument("--min-gap-frames", type=int, default=3,
                   help="Mismo default que ReIDEpisodeDataset/infer_track_reid_gaps.py.")
    p.add_argument("--iou-match-thresh", type=float, default=0.5)
    p.add_argument("--outdir", default="reid_gap_cause_report")

    p.add_argument("--relaunch", action="store_true",
                   help="Re-corre main_track.py con --reid-stitch-max-gap sin límite sobre las secuencias con "
                        "fallos gap>--max-gap, para comprobar si el reid las habría reidentificado.")
    p.add_argument("--eval-tracking-dir", default=None, help="Carpeta con main_track.py (requerido con --relaunch).")
    p.add_argument("--pth-file", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    if args.relaunch and not (args.eval_tracking_dir and args.pth_file and args.config):
        raise SystemExit("--relaunch requiere --eval-tracking-dir, --pth-file y --config")

    sequences = sorted(
        n for n in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, n))
    )
    txt_dir = os.path.join(args.live_run_dir, "txt")
    os.makedirs(args.outdir, exist_ok=True)

    all_results = []
    for seq in sequences:
        pred_txt = os.path.join(txt_dir, f"{seq}.txt")
        if not os.path.isfile(pred_txt):
            print(f"[WARN] sin predicciones para {seq} en {txt_dir}, se omite")
            continue
        all_results.extend(classify_sequence_gaps(
            os.path.join(args.data_root, seq), pred_txt, args.min_gap_frames, args.iou_match_thresh,
        ))

    with open(os.path.join(args.outdir, "gap_outcomes.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    counts = Counter(r["outcome"] for r in all_results)
    total = len(all_results)
    evaluable = total - counts.get("unevaluated", 0)
    print(f"\n[RESUMEN] {args.tracker} + {args.reid_source}: {total} brechas reales de GT ({evaluable} evaluables)")
    for outcome in ("reidentified", "new_id_born", "id_swap", "unevaluated"):
        n = counts.get(outcome, 0)
        print(f"  {outcome:15s}: {n:5d} ({100 * n / total if total else 0:.1f}%)")

    new_born = [r for r in all_results if r["outcome"] == "new_id_born"]
    long_gap = [r for r in new_born if r["gap_len"] > args.max_gap]
    short_gap = [r for r in new_born if r["gap_len"] <= args.max_gap]

    if new_born:
        print(f"\n[CAUSA DE FALLO] de {len(new_born)} new_id_born:")
        print(f"  gap_len > {args.max_gap} (podado del banco -- el reid NUNCA se lanzó): "
              f"{len(long_gap)} ({100 * len(long_gap) / len(new_born):.1f}%)")
        print(f"  gap_len <= {args.max_gap} (seguía en el banco -- el reid se lanzó y no matcheó): "
              f"{len(short_gap)} ({100 * len(short_gap) / len(new_born):.1f}%)")
    with open(os.path.join(args.outdir, "long_gap_failures.json"), "w") as f:
        json.dump(long_gap, f, indent=2)

    if not args.relaunch:
        if long_gap:
            print(f"\n[SIGUIENTE PASO] {len(long_gap)} casos guardados en long_gap_failures.json -- "
                  f"correr de nuevo con --relaunch (+ --eval-tracking-dir/--pth-file/--config) para "
                  f"comprobar si el reid los habría reidentificado sin el límite de --reid-stitch-max-gap.")
        return
    if not long_gap:
        print("\n[OK] No hay casos gap>max_gap que relanzar.")
        return

    affected_seqs = sorted({r["sequence"] for r in long_gap})
    print(f"\n[RELAUNCH] Re-corriendo {args.tracker}+{args.reid_source} con "
          f"--reid-stitch-max-gap {UNCAPPED_MAX_GAP} sobre {len(affected_seqs)} secuencias afectadas...")

    main_track_py = os.path.join(args.eval_tracking_dir, "main_track.py")
    uncapped_dir = os.path.join(args.outdir, "uncapped_txt")
    os.makedirs(uncapped_dir, exist_ok=True)

    for seq in affected_seqs:
        out_txt = os.path.join(uncapped_dir, f"{seq}.txt")
        log_path = os.path.join(uncapped_dir, f"{seq}.log")
        print(f"  [RUN] {seq}")
        rc = relaunch_sequence(
            main_track_py, args.data_root, seq, args.tracker, args.reid_source,
            args.pth_file, args.config, args.conf, args.imgsz, args.fps, args.device,
            out_txt, log_path,
        )
        if rc != 0:
            print(f"  [ERROR] {seq}: revisar {log_path}")

    uncapped_results = []
    for seq in affected_seqs:
        out_txt = os.path.join(uncapped_dir, f"{seq}.txt")
        if not os.path.isfile(out_txt):
            continue
        uncapped_results.extend(classify_sequence_gaps(
            os.path.join(args.data_root, seq), out_txt, args.min_gap_frames, args.iou_match_thresh,
        ))
    uncapped_by_key = {
        (r["sequence"], r["gt_identity_id"], r["frame_before"], r["frame_after"]): r["outcome"]
        for r in uncapped_results
    }

    confirmed_pruning, still_fails = [], []
    for r in long_gap:
        key = (r["sequence"], r["gt_identity_id"], r["frame_before"], r["frame_after"])
        entry = {**r, "outcome_uncapped": uncapped_by_key.get(key, "not_found")}
        (confirmed_pruning if entry["outcome_uncapped"] == "reidentified" else still_fails).append(entry)

    with open(os.path.join(args.outdir, "relaunch_verdicts.json"), "w") as f:
        json.dump({"confirmed_gap_pruning": confirmed_pruning, "genuine_or_other_failure": still_fails}, f, indent=2)

    n = len(long_gap)
    print(f"\n[VEREDICTO] de los {n} casos con gap > {args.max_gap}:")
    print(f"  {len(confirmed_pruning)} ({100 * len(confirmed_pruning) / n:.1f}%) SÍ se hubieran reidentificado "
          f"sin el límite de --reid-stitch-max-gap -- confirmado, la poda del banco es la causa.")
    print(f"  {len(still_fails)} ({100 * len(still_fails) / n:.1f}%) siguen fallando incluso con el banco "
          f"sin límite -- fallo genuino de similitud (o de otra causa), no explicado por el gap.")

    total_new_born = len(new_born)
    print(f"\n[CONCLUSIÓN GLOBAL] de {total_new_born} new_id_born totales: "
          f"{len(short_gap)} ({100 * len(short_gap) / total_new_born:.1f}%) ya tenían la identidad en el banco "
          f"y fallaron por similitud; {len(confirmed_pruning)} "
          f"({100 * len(confirmed_pruning) / total_new_born:.1f}%) fallaron SOLO por la poda del banco "
          f"(--reid-stitch-max-gap); {len(still_fails)} ({100 * len(still_fails) / total_new_born:.1f}%) "
          f"fallan de verdad incluso con el banco sin límite de edad.")


if __name__ == "__main__":
    main()
