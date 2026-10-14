"""
EXPERIMENTO AISLADO -- tab:ttl_sensitivity excluyendo las 10 secuencias que
probablemente vio el detector actual.

No re-ejecuta el replay: re-pasa TrackEval (trackeval_api.py, misma config que
run_eval.py::run_trackeval) sobre las salidas ya guardadas de
replay_ttl_ablation.py (eval_tracking/ttl_replay_<tracker>_sweep/maxgap<N>/txt
+ gt_prepared/pool), con un seqmap filtrado. Dos pasadas:
  - all64  : las 64 secuencias del pool -- CONTROL, debe reproducir el summary original.
  - excl54 : sin las 10 secuencias vistas.
Salidas en eval_tracking/exp_exclude_seen/ttl/<tracker>/maxgap<N>/<pasada>/.
"""
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_live_trackeval import ET, SEEN, read_summary  # noqa: E402

import subprocess  # noqa: E402

OUT = os.path.join(ET, "exp_exclude_seen", "ttl")
TRACKERS = ["bytetrack", "botsort", "ocsort", "deepocsort"]
MAX_GAPS = [25, 50, 100, 200, 400, 800]
API = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trackeval_api.py")


def run_one(tr, mg, tag, seqs):
    src = os.path.join(ET, f"ttl_replay_{tr}_sweep", f"maxgap{mg}")
    out = os.path.join(OUT, tr, f"maxgap{mg}", tag)
    os.makedirs(out, exist_ok=True)
    seqmap = os.path.join(out, "seqmap.txt")
    with open(seqmap, "w") as f:
        f.write("name\n" + "".join(f"{s}\n" for s in seqs))
    r = subprocess.run([sys.executable, API, os.path.join(src, "gt_prepared", "pool"), seqmap,
                        os.path.join(src, "txt"), out], capture_output=True, text=True)
    with open(os.path.join(out, "trackeval.log"), "w") as f:
        f.write(r.stdout + "\n" + r.stderr)
    m = read_summary(os.path.join(out, "pedestrian_summary.txt")) if r.returncode == 0 else None
    return tr, mg, tag, m, read_summary(os.path.join(src, "txt", "pedestrian_summary.txt"))


def main():
    jobs = []
    for tr in TRACKERS:
        for mg in MAX_GAPS:
            with open(os.path.join(ET, f"ttl_replay_{tr}_sweep", f"maxgap{mg}", "seqmap.txt")) as f:
                seqs = [l.strip() for l in f if l.strip() and l.strip() != "name"]
            jobs.append((tr, mg, "all64", seqs))
            jobs.append((tr, mg, "excl54", [s for s in seqs if s not in SEEN]))
    print(f"{len(jobs)} evaluaciones TrackEval")
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "3"))) as ex:
        results = list(ex.map(lambda j: run_one(*j), jobs))
    rows = []
    for tr, mg, tag, m, orig in results:
        row = {"tracker": tr, "max_gap": mg, "pasada": tag}
        for k in ["HOTA", "IDF1", "IDSW"]:
            row[k] = None if m is None else m.get(k)
            row[f"{k}_orig"] = None if orig is None else orig.get(k)
        rows.append(row)
    path = os.path.join(OUT, "ttl_all64_vs_excl54.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    bad = [(r["tracker"], r["max_gap"]) for r in rows if r["pasada"] == "all64" and
           (r["IDF1"] is None or r["IDF1_orig"] is None or abs(r["IDF1"] - r["IDF1_orig"]) > 1e-3)]
    print(f"[OK] {path}")
    print("CONTROL all64 == original:", "TODOS OK" if not bad else f"NO coinciden: {bad}")
    open(os.path.join(ET, "exp_exclude_seen", "ttl_recompute.done"), "w").write("DONE\n")


if __name__ == "__main__":
    main()
