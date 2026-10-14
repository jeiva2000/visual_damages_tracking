"""
EXPERIMENTO AISLADO -- como recompute_live_trackeval.py pero excluyendo ADEMAS
video1 (val del entrenamiento con vídeo del detector actual: eligió el
checkpoint) -> 61 secuencias. Misma llamada TrackEval (trackeval_api.py) sobre
los txt ya guardados; salidas en exp_exclude_seen/live/<carpeta>/excl61/.
"""
import csv, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_live_trackeval import ET, OUT, SEEN, live_dirs, sequences_of, run_one, read_summary, METRICS

EXCL = set(SEEN) | {"video1"}


def main():
    jobs = [(d, "excl61", [s for s in sequences_of(d) if s not in EXCL]) for d in live_dirs()]
    assert all(len(j[2]) == 61 for j in jobs), {j[0]: len(j[2]) for j in jobs}
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "3"))) as ex:
        res = list(ex.map(lambda j: run_one(*j), jobs))
    rows = []
    for d, tag, m in res:
        e62 = read_summary(os.path.join(OUT, d, "excl62", "pedestrian_summary.txt"))
        row = {"dir": d}
        for k in METRICS:
            row[f"{k}_excl62"] = None if e62 is None else e62.get(k)
            row[f"{k}_excl61"] = None if m is None else m.get(k)
        rows.append(row)
    path = os.path.join(OUT, "live_metrics_excl62_vs_excl61.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"[OK] {path}")


if __name__ == "__main__":
    main()
