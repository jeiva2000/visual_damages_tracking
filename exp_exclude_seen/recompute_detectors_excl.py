"""TrackEval all72 vs excl62 sobre corridas nativas ByteTrack de distintos detectores (salidas en exp_exclude_seen/live_detectors/)."""
import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_live_trackeval import ET, SEEN, read_summary
API = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trackeval_api.py")
OUT = os.path.join(ET, "exp_exclude_seen", "live_detectors")
DIRS = {"actual": "native_reid_bytetrack_alldata",
        "rtdetr_damages": "exp_custommot2_detector/native_reid_bytetrack_alldata_CUSTOMMOT2",
        "v2_qcs": "exp_novideo2_detector/native_reid_bytetrack_alldata_NOVIDEO2",
        "v3_qcs_cardd": "exp_novideo_detector/native_reid_bytetrack_alldata_NOVIDEO",
        "noglass": "exp_noglass_detector/native_reid_bytetrack_alldata_NOGLASS",
        "rtdetrv4": "exp_testmodels/rtdetrv4", "rtdetrv4_epoch_11": "exp_testmodels/rtdetrv4_epoch_11"}
def run(name, d, tag):
    src = os.path.join(ET, d); out = os.path.join(OUT, name, tag); os.makedirs(out, exist_ok=True)
    seqs = [l.strip() for l in open(os.path.join(src, "seqmap.txt")) if l.strip() and l.strip() != "name"]
    if tag == "excl62": seqs = [s for s in seqs if s not in SEEN]
    sm = os.path.join(out, "seqmap.txt"); open(sm, "w").write("name\n" + "".join(s + "\n" for s in seqs))
    subprocess.run([sys.executable, API, os.path.join(src, "gt_prepared", "all"), sm, os.path.join(src, "txt"), out], capture_output=True)
    return name, tag, len(seqs), read_summary(os.path.join(out, "pedestrian_summary.txt"))
jobs = [(n, d, t) for n, d in DIRS.items() for t in ("all72", "excl62")]
with ThreadPoolExecutor(3) as ex: res = list(ex.map(lambda j: run(*j), jobs))
K = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "MOTA", "IDF1", "IDSW"]
for n, t, ns, m in res:
    print(f"{n:18s} {t:6s} n={ns:2d} " + (" ".join(f"{k}={m[k]:.2f}" for k in K) if m else "FALLO"))
