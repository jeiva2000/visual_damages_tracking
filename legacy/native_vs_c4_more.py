"""
Genera mas ejemplos NATIVE vs +C4 (mecanismo query-bank), en un solo paso:
calcula embeddings REALES (C4PuroReID/custom_reid.py, mismo backbone/3-escalas
que la corrida real), arma la imagen con la plantilla visualize_episode
(banco izquierda, query derecha) y la compone al lado del panel NATIVE
(sin mecanismo). Reusa las predicciones YA CALCULADAS de
c4_puro_live_bytetrack_alldata / native_reid_bytetrack_alldata -- no vuelve
a correr el detector.
"""
import os
import sys

sys.path.insert(0, "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from custom_reid import C4PuroReID, _spread_indices_c4, _expand_bbox_c4
from p4_backbone_encoder import P4BackboneEncoder
from reid_dataloader import DEFAULT_TRANSFORM
from train_track_query_reid import visualize_episode

CFG = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml"
PTH = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/damages_tracking/models/rtdetrv2_v3_qcs_videos.pth"
DATA_ROOT = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/all_dataset_view/all"
C4_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/c4_puro_live_bytetrack_alldata/txt"
NATIVE_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/native_reid_bytetrack_alldata/txt"
OUT_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking/c4_proof_visuals"
THRESHOLD = 0.9352
MAX_GAP = 200
MAX_BANK_SIZE = 16
IMGSZ = 640
DEVICE = "cpu"

CASES = [
    ("video_0vbcxRjnfa70hc8AVvt_11", 5368, 93, 242),
    ("video_JLWTKzWBPMgHBprokgv_11", 8177, 212, 361),
    ("video_987C1eJaVmqH539Kryo_11", 7598, 191, 336),
]


def read_mot(txt_path):
    by_frame, by_id = {}, {}
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame = int(float(p[0])); pid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            bbox = (x, y, x + w, y + h)
            by_frame.setdefault(frame, {})[pid] = bbox
            by_id.setdefault(pid, []).append((frame, bbox))
    for pid in by_id:
        by_id[pid].sort()
    return by_frame, by_id


def preprocess(img_bgr, imgsz):
    from torchvision import transforms as T
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return T.ToTensor()(pil.resize((imgsz, imgsz))).unsqueeze(0)


class FrameCache:
    def __init__(self, model, c4reid):
        self.model = model
        self.c4reid = c4reid

    def get_scales(self, seq, frame_idx, boxes_xyxy):
        img_path = os.path.join(DATA_ROOT, seq, "img1", f"{frame_idx:06d}.jpg")
        img_bgr = cv2.imread(img_path)
        im_data = preprocess(img_bgr, IMGSZ)
        with torch.no_grad():
            self.model.backbone(im_data.to(DEVICE))
        boxes = np.array(boxes_xyxy, dtype=np.float32)
        return self.c4reid.get_scale_features(boxes, img_bgr)


def pool_history(scale_rows):
    idxs = _spread_indices_c4(len(scale_rows), 4)
    selected = []
    for pos, i in enumerate(idxs):
        row = scale_rows[i]
        selected.append(row[0] if pos == 0 else (row[2] if pos == len(idxs) - 1 else row[1]))
    return np.mean(np.stack(selected, axis=0), axis=0)


def crop_pil(seq, frame_idx, bbox, pad=0.3, size=260):
    img = Image.open(os.path.join(DATA_ROOT, seq, "img1", f"{frame_idx:06d}.jpg")).convert("RGB")
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1 -= w * pad; x2 += w * pad; y1 -= h * pad; y2 += h * pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    crop = img.crop((x1, y1, x2, y2))
    crop.thumbnail((size, size))
    return crop


def crop_tensor(seq, frame_idx, bbox, factor):
    img = Image.open(os.path.join(DATA_ROOT, seq, "img1", f"{frame_idx:06d}.jpg")).convert("RGB")
    ex = _expand_bbox_c4(bbox, factor, img.size)
    return DEFAULT_TRANSFORM(img.crop(ex))


def build_c4_template(model, cache, seq, frame_before, query_frame, query_pid, out_path):
    by_frame, by_id = read_mot(os.path.join(C4_DIR, f"{seq}.txt"))
    lost = {}
    for pid, obs in by_id.items():
        if pid == query_pid:
            continue
        last_f, _ = obs[-1]
        if last_f < query_frame and (query_frame - last_f) <= MAX_GAP:
            lost[pid] = obs
    # el candidato VERDADERO es el mismo pid final (el gap ya esta fusionado
    # en las predicciones guardadas) -- se agrega aparte, truncado a SOLO sus
    # observaciones ANTES del hueco real (frame_before), igual que lo veria
    # reid_gallery.bury() en el momento real, sin fuga de datos del futuro.
    true_obs = [(f, b) for f, b in by_id[query_pid] if f <= frame_before]
    if true_obs:
        lost[query_pid] = true_obs

    q_bbox = by_frame[query_frame][query_pid]
    q_scales = cache.get_scales(seq, query_frame, [q_bbox])[0]
    q_embed = q_scales[1:3].mean(axis=0)
    q_embed = q_embed / (np.linalg.norm(q_embed) + 1e-9)

    results = []
    for pid, obs in lost.items():
        idxs = _spread_indices_c4(len(obs), 4)
        scale_rows = [cache.get_scales(seq, obs[i][0], [obs[i][1]])[0] for i in idxs]
        bank_embed = pool_history(scale_rows)
        bank_embed = bank_embed / (np.linalg.norm(bank_embed) + 1e-9)
        sim = float(np.dot(q_embed, bank_embed))
        results.append({"pid": pid, "sim": sim, "obs": obs, "is_true": pid == query_pid})
    results.sort(key=lambda r: -r["sim"])

    print(f"\n[{seq}] query_pid={query_pid} frame={query_frame} threshold={THRESHOLD}")
    for r in results:
        print(f"  pid={r['pid']} sim={r['sim']:.4f} {'<-- TRUE MATCH' if r['is_true'] else ''}")

    N = len(results)
    bank_images = torch.zeros(MAX_BANK_SIZE, 4, 3, 224, 224)
    bank_mask = torch.zeros(MAX_BANK_SIZE, dtype=torch.bool)
    target = N
    for i, r in enumerate(results):
        obs = r["obs"]
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
        if r["is_true"]:
            target = i

    query_views = torch.stack([crop_tensor(seq, query_frame, q_bbox, 1.0), crop_tensor(seq, query_frame, q_bbox, 6.0)], dim=0)
    batch = {
        "bank_images": bank_images.unsqueeze(0), "bank_mask": bank_mask.unsqueeze(0),
        "query_views": query_views.unsqueeze(0), "target": torch.tensor([target], dtype=torch.long),
    }
    sims = [r["sim"] for r in results]
    best_slot = int(np.argmax(sims)) if sims else None
    best_sim = sims[best_slot] if best_slot is not None else -1.0
    pred_slot = best_slot if (best_slot is not None and best_sim >= THRESHOLD) else N

    fake_logits = torch.full((MAX_BANK_SIZE + 1,), float("-inf"))
    for i, s in enumerate(sims):
        fake_logits[i] = s
    fake_logits[MAX_BANK_SIZE] = THRESHOLD
    fake_logits = fake_logits / 0.05

    visualize_episode(batch, 0, out_path, pred=pred_slot, logits=fake_logits)
    print(f"[OK] {out_path}  correcto={pred_slot == target}")
    return q_bbox


def build_native_panel(seq, frame_before, frame_after, query_pid):
    native_by_frame, _ = read_mot(os.path.join(NATIVE_DIR, f"{seq}.txt"))
    c4_by_frame, _ = read_mot(os.path.join(C4_DIR, f"{seq}.txt"))
    q_bbox_before = c4_by_frame[frame_before][query_pid]
    q_bbox_after = c4_by_frame[frame_after][query_pid]

    def best_native(frame, bbox):
        best_pid, best_iou = None, 0
        for npid, nbbox in native_by_frame.get(frame, {}).items():
            ax1, ay1, ax2, ay2 = bbox; bx1, by1, bx2, by2 = nbbox
            ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            iou = inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-9)
            if iou > best_iou:
                best_iou, best_pid = iou, npid
        return best_pid, native_by_frame[frame][best_pid] if best_pid is not None else bbox

    id_before, bbox_before = best_native(frame_before, q_bbox_before)
    id_after, bbox_after = best_native(frame_after, q_bbox_after)

    c1 = crop_pil(seq, frame_before, bbox_before)
    c2 = crop_pil(seq, frame_after, bbox_after)
    W = c1.width + c2.width + 30
    H = max(c1.height, c2.height) + 70
    panel = Image.new("RGB", (W, H), (15, 15, 15))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), "NATIVE (sin mecanismo de reid)", fill=(255, 255, 255))
    panel.paste(c1, (0, 60))
    draw.text((5, 40), f"f={frame_before} id={id_before}", fill=(180, 180, 255))
    panel.paste(c2, (c1.width + 20, 60))
    failed = id_before != id_after
    color = (230, 60, 60) if failed else (0, 220, 0)
    status = "id NUEVO -> fallo" if failed else "mismo id -> ok"
    draw.text((c1.width + 25, 40), f"f={frame_after} id={id_after}  ({status})", fill=color)
    return panel


def compose(native_panel, c4_img_path, out_path, title):
    c4_img = Image.open(c4_img_path)
    W = native_panel.width + c4_img.width + 20
    H = max(native_panel.height, c4_img.height) + 30
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), title, fill=(255, 255, 0))
    canvas.paste(native_panel, (0, 30))
    canvas.paste(c4_img, (native_panel.width + 20, 30))
    canvas.save(out_path, quality=92)
    print(f"[OK] {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[INFO] cargando modelo...")
    p4enc = P4BackboneEncoder(CFG, PTH, layer="c4")
    p4enc.eval()
    for p in p4enc.parameters():
        p.requires_grad = False
    c4reid = C4PuroReID(num_track_history=4, device=DEVICE)
    c4reid.attach(p4enc.model)
    cache = FrameCache(p4enc.model, c4reid)

    for seq, pid, fbefore, fafter in CASES:
        tpl_path = os.path.join(OUT_DIR, f"tpl_more_{seq}.jpg")
        build_c4_template(p4enc, cache, seq, fbefore, fafter, pid, tpl_path)
        native_panel = build_native_panel(seq, fbefore, fafter, pid)
        compose(
            native_panel, tpl_path,
            os.path.join(OUT_DIR, f"compare_native_vs_c4_more_{seq}.jpg"),
            f"NATIVE vs +C4 -- {seq} (hueco real de {fafter - fbefore} frames)",
        )


if __name__ == "__main__":
    main()
