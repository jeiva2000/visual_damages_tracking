"""
EXPERIMENTO AISLADO -- métricas de la validación EN VIVO excluyendo las 10
secuencias que probablemente vio el detector actual en su entrenamiento.

No re-ejecuta detector ni trackers: re-pasa TrackEval (misma llamada que
eval_tracking/run_eval.py::run_trackeval) sobre las predicciones ya guardadas
(<dir>/txt) y el GT ya preparado (<dir>/gt_prepared/all), con un seqmap
filtrado. Dos pasadas por carpeta:
  - all72  : las 72 secuencias -- CONTROL, debe reproducir el summary original.
  - excl62 : sin las 10 secuencias vistas.
Las salidas van a eval_tracking/exp_exclude_seen/live/<carpeta>/<pasada>/
(--OUTPUT_FOLDER), nunca dentro de las carpetas originales, que solo se leen.

Las 10 secuencias (frames de default_train_combine_video.json identificados
por md5 contra dataset_split_2, ver memoria project_novideo_detector_experiment):
"""
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ET = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking"
OUT = os.path.join(ET, "exp_exclude_seen", "live")
SEEN = ["touaregmedina", "classiccar", "video6", "chevroletazul", "video2",
        "video4", "4x4verde", "chevroletdorado", "video3", "video5"]
METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]


def live_dirs():
    for d in sorted(os.listdir(ET)):
        p = os.path.join(ET, d)
        if d.endswith("_alldata") and os.path.isdir(os.path.join(p, "txt")) and os.path.isdir(os.path.join(p, "gt_prepared", "all")):
            yield d


def sequences_of(d):
    with open(os.path.join(ET, d, "seqmap.txt")) as f:
        return [l.strip() for l in f if l.strip() and l.strip() != "name"]


def run_one(d, tag, seqs):
    out = os.path.join(OUT, d, tag)
    os.makedirs(out, exist_ok=True)
    seqmap = os.path.join(out, "seqmap.txt")
    with open(seqmap, "w") as f:
        f.write("name\n" + "".join(f"{s}\n" for s in seqs))
    # Misma configuración que run_eval.py::run_trackeval, pero vía la API de
    # TrackEval (run_mot_challenge.py convierte --OUTPUT_FOLDER en lista).
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "trackeval_api.py"),
           os.path.join(ET, d, "gt_prepared", "all"), seqmap, os.path.join(ET, d, "txt"), out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(out, "trackeval.log"), "w") as f:
        f.write(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        return d, tag, None
    return d, tag, read_summary(os.path.join(out, "pedestrian_summary.txt"))


def read_summary(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        head, vals = f.readline().split(), f.readline().split()
    return {k: float(v) for k, v in zip(head, vals)}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for d in live_dirs():
        seqs = sequences_of(d)
        jobs.append((d, "all72", seqs))
        jobs.append((d, "excl62", [s for s in seqs if s not in SEEN]))
    print(f"{len(jobs)} evaluaciones TrackEval ({len(jobs)//2} carpetas)")
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "3"))) as ex:
        results = list(ex.map(lambda j: run_one(*j), jobs))

    rows = []
    for d, tag, m in results:
        orig = read_summary(os.path.join(ET, d, "txt", "pedestrian_summary.txt"))
        row = {"dir": d, "pasada": tag}
        for k in METRICS:
            row[k] = None if m is None else m.get(k)
            row[f"{k}_orig"] = None if orig is None else orig.get(k)
        rows.append(row)
    path = os.path.join(OUT, "live_metrics_all72_vs_excl62.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    bad = [r["dir"] for r in rows if r["pasada"] == "all72" and (r["HOTA"] is None or r["HOTA_orig"] is None or abs(r["HOTA"] - r["HOTA_orig"]) > 1e-3)]
    print(f"[OK] {path}")
    print("CONTROL all72 == original:", "TODOS OK" if not bad else f"NO coinciden: {bad}")


if __name__ == "__main__":
    main()
