"""
Reconstruye, a partir de las predicciones YA CALCULADAS de la corrida real
c4_puro_live_bytetrack_alldata (la que respalda tab:live_c4_vs_native_alldata),
una comparacion estilo consulta-banco fiel a lo que reid_gallery.py vio en
el momento de decidir -- SIN volver a correr el detector completo (eso era
demasiado lento en CPU). Reusa directo custom_reid.py::C4PuroReID/
extract_c4_scales (mismo hook, misma logica de 3 escalas y seleccion de
banco) sobre el backbone (solo backbone, no encoder/decoder -- rapido).
"""
import os
import sys

sys.path.insert(0, "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from custom_reid import C4PuroReID, extract_c4_scales, _spread_indices_c4
from p4_backbone_encoder import P4BackboneEncoder

CFG = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml"
PTH = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/damages_tracking/models/rtdetrv2_v3_qcs_videos.pth"
DATA_ROOT = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/all_dataset_view/all"
C4_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/c4_puro_live_bytetrack_alldata/txt"
OUT_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking/c4_proof_visuals"
THRESHOLD = 0.9352
MAX_GAP = 200
IMGSZ = 640
DEVICE = "cpu"


def read_mot(txt_path):
    by_frame = {}
    by_id = {}
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0])); pid = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            bbox = (x, y, x + w, y + h)
            by_frame.setdefault(frame, []).append((pid, bbox))
            by_id.setdefault(pid, []).append((frame, bbox))
    for pid in by_id:
        by_id[pid].sort()
    return by_frame, by_id


def frame_path(seq, frame_idx):
    return os.path.join(DATA_ROOT, seq, "img1", f"{frame_idx:06d}.jpg")


def preprocess(img_bgr, imgsz):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_resized = pil.resize((imgsz, imgsz))
    from torchvision import transforms as T
    im_data = T.ToTensor()(pil_resized).unsqueeze(0)
    return im_data


class FrameCache:
    """Corre SOLO el backbone (no encoder/decoder) para cada frame pedido, una
    sola vez -- cachea el feature map por frame para no repetir el forward."""

    def __init__(self, model, c4reid):
        self.model = model
        self.c4reid = c4reid
        self._cache = {}

    def get_scales(self, seq, frame_idx, boxes_xyxy):
        img_path = frame_path(seq, frame_idx)
        img_bgr = cv2.imread(img_path)
        h, w = img_bgr.shape[:2]
        im_data = preprocess(img_bgr, IMGSZ)
        with torch.no_grad():
            self.model.backbone(im_data.to(DEVICE))
        boxes = np.array(boxes_xyxy, dtype=np.float32)
        return self.c4reid.get_scale_features(boxes, img_bgr)  # (N,3,D)


def pool_history(scale_rows):
    """scale_rows: lista de (3,D) en orden cronologico. Igual que
    C4PuroReID.pool_history pero sin instancia (lista ya armada)."""
    idxs = _spread_indices_c4(len(scale_rows), 4)
    selected = []
    for pos, i in enumerate(idxs):
        row = scale_rows[i]
        if pos == 0:
            selected.append(row[0])
        elif pos == len(idxs) - 1:
            selected.append(row[2])
        else:
            selected.append(row[1])
    return np.mean(np.stack(selected, axis=0), axis=0)


def crop_bbox(seq, frame_idx, bbox, pad=0.3):
    img = Image.open(frame_path(seq, frame_idx)).convert("RGB")
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1 -= w * pad; x2 += w * pad; y1 -= h * pad; y2 += h * pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    return img.crop((x1, y1, x2, y2))


def build_case(model, c4reid, cache, seq, query_frame, query_pid, tag):
    by_frame, by_id = read_mot(os.path.join(C4_DIR, f"{seq}.txt"))

    # candidatos "perdidos" en query_frame: su ultimo frame < query_frame,
    # y (query_frame - ultimo_frame) <= MAX_GAP -- igual criterio que
    # reid_gallery.age_and_prune.
    lost = []
    for pid, obs in by_id.items():
        if pid == query_pid:
            continue
        last_frame, last_bbox = obs[-1]
        if last_frame < query_frame and (query_frame - last_frame) <= MAX_GAP:
            lost.append(pid)

    # embedding de la query (frame actual, ajustado+intermedio)
    q_bbox = dict(by_frame[query_frame])[query_pid]
    q_scales = cache.get_scales(seq, query_frame, [q_bbox])[0]  # (3,D)
    q_embed = q_scales[1:3].mean(axis=0)
    q_embed = q_embed / (np.linalg.norm(q_embed) + 1e-9)

    results = []
    for pid in lost:
        obs = by_id[pid]
        idxs = _spread_indices_c4(len(obs), 4)
        scale_rows = []
        crops_for_viz = []
        for i in idxs:
            fidx, bbox = obs[i]
            scales = cache.get_scales(seq, fidx, [bbox])[0]
            scale_rows.append(scales)
        bank_embed = pool_history(scale_rows)
        bank_embed = bank_embed / (np.linalg.norm(bank_embed) + 1e-9)
        sim = float(np.dot(q_embed, bank_embed))
        results.append({"pid": pid, "sim": sim, "last_seen_frame": obs[-1][0], "last_seen_bbox": obs[-1][1],
                         "n_obs": len(obs)})

    results.sort(key=lambda r: -r["sim"])
    print(f"\n[{tag}] seq={seq} query_pid={query_pid} frame={query_frame}  threshold={THRESHOLD}")
    print(f"  bank_size={len(results)}")
    for r in results:
        print(f"  candidato pid={r['pid']}: sim={r['sim']:.4f} (>=umbral? {'SI' if r['sim']>=THRESHOLD else 'no'}) "
              f"ultimo_frame_visto={r['last_seen_frame']} n_obs={r['n_obs']}")
    if results:
        best = results[0]
        pred_match = best["sim"] >= THRESHOLD
        print(f"  [DECISION] mejor candidato pid={best['pid']} sim={best['sim']:.4f} -> "
              f"{'MATCH' if pred_match else 'NO MATCH'}")
    else:
        print("  [DECISION] banco vacio -> nace como id nuevo")

    # visualizacion: query + hasta 6 candidatos, cada uno con su similitud
    tile = 160
    n_show = min(6, len(results))
    canvas = Image.new("RGB", (tile * (n_show + 1) + 40, tile + 60), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    q_crop = crop_bbox(seq, query_frame, q_bbox).resize((tile, tile))
    canvas.paste(q_crop, (10, 40))
    draw.text((10, 10), f"QUERY pid={query_pid} f={query_frame}", fill=(255, 255, 0))
    for i, r in enumerate(results[:n_show]):
        x0 = 10 + tile + 20 + i * (tile + 10)
        crop = crop_bbox(seq, r["last_seen_frame"], r["last_seen_bbox"]).resize((tile, tile))
        canvas.paste(crop, (x0, 40))
        color = (0, 220, 0) if r["sim"] >= THRESHOLD else (220, 60, 60)
        draw.text((x0, 10), f"pid={r['pid']} sim={r['sim']:.3f}", fill=color)
    out_path = os.path.join(OUT_DIR, f"gallery_{tag}_{seq}.jpg")
    canvas.save(out_path, quality=92)
    print(f"[OK] {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[INFO] cargando modelo (solo backbone hace falta)...")
    p4enc = P4BackboneEncoder(CFG, PTH, layer="c4")
    p4enc.eval()
    for p in p4enc.parameters():
        p.requires_grad = False
    c4reid = C4PuroReID(num_track_history=4, device=DEVICE)
    c4reid.attach(p4enc.model)
    cache = FrameCache(p4enc.model, c4reid)

    # caso 1: exito -- native id_swap (5582->5594), c4 reidentifico (5582 se
    # mantiene). el "nacimiento" en el c4 real fue justo en frame 314-ish
    # (buscamos el primer frame de un id que empieza cerca de ahi con la
    # misma zona que 5582 -- en la practica el id resultante YA es 5582
    # porque el gallery lo fusiono, asi que reconstruimos la comparacion en
    # el momento exacto: query = primera vez que 5582 REAPARECE tras el hueco).
    by_frame, by_id = read_mot(os.path.join(C4_DIR, "video_12Kt2fvxMGbX4sPNK40_11.txt"))
    obs_5582 = by_id.get(5582, [])
    # buscamos el hueco real (>=90 frames, cerca del reportado 216->314)
    gap_frame = None
    for i in range(1, len(obs_5582)):
        if obs_5582[i][0] - obs_5582[i - 1][0] > 50:
            gap_frame = obs_5582[i][0]
            break
    print(f"[INFO] reaparicion de pid=5582 detectada en frame={gap_frame}")
    if gap_frame:
        build_case(p4enc, c4reid, cache, "video_12Kt2fvxMGbX4sPNK40_11", gap_frame, 5582, "success")

    # caso 2: nace nuevo (pid=155 en VID_20260118_222031, frame~174)
    build_case(p4enc, c4reid, cache, "VID_20260118_222031", 174, 155, "newtrack")


if __name__ == "__main__":
    main()
