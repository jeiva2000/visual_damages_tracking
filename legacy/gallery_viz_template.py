"""
Re-arma las 2 imagenes de galeria EN VIVO (success2 y newtrack) con la MISMA
plantilla que visualize_episode viene usando en todo este hilo: banco a la
izquierda (una fila por candidato, hasta 4 tiles de su historial + score),
query a la derecha (2 vistas: ajustada + amplia). No recalcula embeddings --
reusa las similitudes YA calculadas por gallery_from_live_preds.py/
gallery_success2.py (ver logs), solo cambia como se arma el batch/imagen.
"""
import os
import sys

sys.path.insert(0, "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking")
sys.path.insert(0, "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking")

import torch
from PIL import Image

from custom_reid import _spread_indices_c4, _expand_bbox_c4
from reid_dataloader import DEFAULT_TRANSFORM
from train_track_query_reid import visualize_episode

DATA_ROOT = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/all_dataset_view/all"
C4_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/c4_puro_live_bytetrack_alldata/txt"
OUT_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking/c4_proof_visuals"
THRESHOLD = 0.9352
MAX_BANK_SIZE = 16


def read_mot(txt_path):
    by_id = {}
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame = int(float(p[0])); pid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            by_id.setdefault(pid, []).append((frame, (x, y, x + w, y + h)))
    for pid in by_id:
        by_id[pid].sort()
    return by_id


def frame_img(seq, frame_idx):
    return Image.open(os.path.join(DATA_ROOT, seq, "img1", f"{frame_idx:06d}.jpg")).convert("RGB")


def crop_tensor(seq, frame_idx, bbox, factor):
    img = frame_img(seq, frame_idx)
    ex = _expand_bbox_c4(bbox, factor, img.size)
    crop = img.crop(ex)
    return DEFAULT_TRANSFORM(crop)


def build(seq, query_frame, query_bbox, candidates, target_pid, out_path):
    """candidates: lista de dicts {pid, sim, obs (lista completa (frame,bbox)
    de ESE candidato, ya recortada al momento correcto), is_true}."""
    candidates = sorted(candidates, key=lambda c: -c["sim"])
    N = len(candidates)

    bank_images = torch.zeros(MAX_BANK_SIZE, 4, 3, 224, 224)
    bank_mask = torch.zeros(MAX_BANK_SIZE, dtype=torch.bool)
    target = MAX_BANK_SIZE  # default: no match ("new"), ver S en el resto del pipeline
    sims_ordered = []

    for i, cand in enumerate(candidates):
        obs = cand["obs"]
        idxs = _spread_indices_c4(len(obs), 4)
        tiles = []
        for pos, oi in enumerate(idxs):
            fidx, bbox = obs[oi]
            factor = 6.0 if pos == 0 else (1.0 if pos == len(idxs) - 1 else 1.6)
            tiles.append(crop_tensor(seq, fidx, bbox, factor))
        while len(tiles) < 4:
            tiles.append(torch.zeros(3, 224, 224))
        bank_images[i] = torch.stack(tiles, dim=0)
        bank_mask[i] = True
        sims_ordered.append(cand["sim"])
        if cand["is_true"]:
            target = i

    query_tight = crop_tensor(seq, query_frame, query_bbox, 1.0)
    query_wide = crop_tensor(seq, query_frame, query_bbox, 6.0)
    query_views = torch.stack([query_tight, query_wide], dim=0)

    batch = {
        "bank_images": bank_images.unsqueeze(0),
        "bank_mask": bank_mask.unsqueeze(0),
        "query_views": query_views.unsqueeze(0),
        "target": torch.tensor([target], dtype=torch.long),
    }

    best_slot = int(torch.tensor(sims_ordered).argmax().item()) if sims_ordered else None
    best_sim = sims_ordered[best_slot] if best_slot is not None else -1.0
    pred_is_match = best_sim >= THRESHOLD
    pred_slot = best_slot if pred_is_match else MAX_BANK_SIZE

    fake_logits = torch.full((MAX_BANK_SIZE + 1,), float("-inf"))
    for i, s in enumerate(sims_ordered):
        fake_logits[i] = s
    fake_logits[MAX_BANK_SIZE] = THRESHOLD
    fake_logits = fake_logits / 0.05

    visualize_episode(batch, 0, out_path, pred=pred_slot, logits=fake_logits)
    print(f"[OK] {out_path}  target={target} pred={pred_slot} correcto={pred_slot == target}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- caso success2: video_1Vl7JUY8fBQhAph15I7_11, query pid=5861 f=181 ----
    seq1 = "video_1Vl7JUY8fBQhAph15I7_11"
    by_id1 = read_mot(os.path.join(C4_DIR, f"{seq1}.txt"))
    query_bbox1 = dict(by_id1[5861])[181]
    true_obs = [(f, b) for f, b in by_id1[5861] if f <= 31]
    cands1 = [
        {"pid": 5861, "sim": 0.9570, "obs": true_obs, "is_true": True},
        {"pid": 5864, "sim": 0.9116, "obs": by_id1[5864], "is_true": False},
        {"pid": 5866, "sim": 0.8149, "obs": by_id1[5866], "is_true": False},
        {"pid": 5862, "sim": 0.8111, "obs": by_id1[5862], "is_true": False},
        {"pid": 5890, "sim": 0.8032, "obs": by_id1[5890], "is_true": False},
    ]
    build(seq1, 181, query_bbox1, cands1, 5861, os.path.join(OUT_DIR, f"tpl_success_{seq1}.jpg"))

    # ---- caso newtrack: VID_20260118_222031, query pid=155 f=174 ----
    seq2 = "VID_20260118_222031"
    by_id2 = read_mot(os.path.join(C4_DIR, f"{seq2}.txt"))
    query_bbox2 = dict(by_id2[155])[174]
    sims2 = {131: 0.9022, 146: 0.8955, 113: 0.8943, 127: 0.8888, 79: 0.8841, 81: 0.8754,
             102: 0.8748, 96: 0.8616, 76: 0.8515, 77: 0.8511, 82: 0.8489, 78: 0.8118}
    cands2 = [{"pid": pid, "sim": sim, "obs": by_id2[pid], "is_true": False} for pid, sim in sims2.items()]
    build(seq2, 174, query_bbox2, cands2, None, os.path.join(OUT_DIR, f"tpl_newtrack_{seq2}.jpg"))


if __name__ == "__main__":
    main()
