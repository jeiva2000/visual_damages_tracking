import os

C4_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/c4_puro_live_bytetrack_alldata/txt"
NATIVE_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/native_reid_bytetrack_alldata/txt"


def read_mot(p):
    by_id = {}
    with open(p) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0])); pid = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            by_id.setdefault(pid, []).append((frame, (x, y, x + w, y + h)))
    return by_id


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    return inter / (aarea + barea - inter + 1e-9)


cases = [
    ("video6", 4163, 574, 723),
    ("video_0vbcxRjnfa70hc8AVvt_11", 5368, 93, 242),
    ("video_JLWTKzWBPMgHBprokgv_11", 8177, 212, 361),
    ("video_a11dEYp7edDns6V6BPr_11", 9741, 9, 158),
    ("video_2gDBsa0MryOsdvPGg4o_11", 6148, 88, 236),
    ("video_UIjssjyZiHDJwUCcUj7_11", 9209, 289, 437),
    ("video_987C1eJaVmqH539Kryo_11", 7598, 191, 336),
    ("touaregmedina", 2273, 695, 834),
    ("video_3rntaZcqXY1h8ur5GcJ_11", 6565, 157, 295),
    ("video_0jGlrs0wJs3utJftEoD_11", 5040, 137, 274),
]

for seq, pid, fbefore, fafter in cases:
    c4_path = os.path.join(C4_DIR, seq + ".txt")
    native_path = os.path.join(NATIVE_DIR, seq + ".txt")
    if not (os.path.isfile(c4_path) and os.path.isfile(native_path)):
        print(seq, "MISSING FILE")
        continue
    by_id = read_mot(c4_path)
    native_by_id = read_mot(native_path)
    obs = by_id[pid]
    obs_dict = dict(obs)
    if fafter not in obs_dict:
        print(seq, pid, "frame_after no en obs")
        continue
    q_bbox = obs_dict[fafter]
    lost = 0
    for opid, oobs in by_id.items():
        if opid == pid:
            continue
        last_f, last_b = oobs[-1]
        if last_f < fafter and (fafter - last_f) <= 200:
            lost += 1
    native_by_frame = {}
    for opid, oobs in native_by_id.items():
        for f, b in oobs:
            native_by_frame.setdefault(f, []).append((opid, b))
    best = None
    best_iou = 0
    for npid, nbbox in native_by_frame.get(fafter, []):
        v = iou(nbbox, q_bbox)
        if v > best_iou:
            best_iou = v
            best = npid
    diverges = best != pid
    msg = "seq={} pid={} f={}->{} lost={} native_pid={} iou={:.2f} DIVERGE={}".format(
        seq, pid, fbefore, fafter, lost, best, best_iou, diverges
    )
    print(msg)
