import os
import sys

sys.path.insert(0, "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking")
from gallery_from_live_preds import (
    P4BackboneEncoder, C4PuroReID, FrameCache, read_mot, pool_history, crop_bbox,
    CFG, PTH, DEVICE, THRESHOLD, MAX_GAP, OUT_DIR, C4_DIR,
)
from custom_reid import _spread_indices_c4
from PIL import Image, ImageDraw
import numpy as np

SEQ = "video_1Vl7JUY8fBQhAph15I7_11"
TRUE_PID = 5861
GAP_BEFORE_FRAME = 31   # ultimo frame ANTES del hueco real (el banco solo debe ver esto)
QUERY_FRAME = 181        # primer frame DESPUES del hueco (reaparicion)


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

    by_frame, by_id = read_mot(os.path.join(C4_DIR, f"{SEQ}.txt"))

    # banco: pid=TRUE_PID pero SOLO con sus frames <= GAP_BEFORE_FRAME (asi
    # como lo veria reid_gallery.bury() en el momento real, antes de que el
    # hueco existiera) + otros pids genuinamente perdidos en QUERY_FRAME.
    candidates = {}
    true_obs = [(f, b) for f, b in by_id[TRUE_PID] if f <= GAP_BEFORE_FRAME]
    candidates[TRUE_PID] = true_obs
    for pid, obs in by_id.items():
        if pid == TRUE_PID:
            continue
        last_frame, last_bbox = obs[-1]
        if last_frame < QUERY_FRAME and (QUERY_FRAME - last_frame) <= MAX_GAP:
            candidates[pid] = obs

    q_bbox = dict(by_frame[QUERY_FRAME])[TRUE_PID]
    q_scales = cache.get_scales(SEQ, QUERY_FRAME, [q_bbox])[0]
    q_embed = q_scales[1:3].mean(axis=0)
    q_embed = q_embed / (np.linalg.norm(q_embed) + 1e-9)

    results = []
    for pid, obs in candidates.items():
        idxs = _spread_indices_c4(len(obs), 4)
        scale_rows = [cache.get_scales(SEQ, obs[i][0], [obs[i][1]])[0] for i in idxs]
        bank_embed = pool_history(scale_rows)
        bank_embed = bank_embed / (np.linalg.norm(bank_embed) + 1e-9)
        sim = float(np.dot(q_embed, bank_embed))
        results.append({"pid": pid, "sim": sim, "last_seen_frame": obs[-1][0], "last_seen_bbox": obs[-1][1],
                         "n_obs": len(obs), "is_true": pid == TRUE_PID})

    results.sort(key=lambda r: -r["sim"])
    print(f"\n[SUCCESS2] seq={SEQ} query_pid={TRUE_PID} frame={QUERY_FRAME}  threshold={THRESHOLD}")
    print(f"  bank_size={len(results)}")
    for r in results:
        tag = "  <- VERDADERO MATCH" if r["is_true"] else ""
        print(f"  candidato pid={r['pid']}: sim={r['sim']:.4f} (>=umbral? {'SI' if r['sim']>=THRESHOLD else 'no'}) "
              f"ultimo_frame_visto={r['last_seen_frame']} n_obs={r['n_obs']}{tag}")
    best = results[0]
    pred_match = best["sim"] >= THRESHOLD
    correcto = pred_match and best["is_true"]
    print(f"  [DECISION] mejor candidato pid={best['pid']} sim={best['sim']:.4f} -> "
          f"{'MATCH' if pred_match else 'NO MATCH'}  (correcto={correcto})")

    tile = 160
    n_show = min(6, len(results))
    canvas = Image.new("RGB", (tile * (n_show + 1) + 40, tile + 60), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    q_crop = crop_bbox(SEQ, QUERY_FRAME, q_bbox).resize((tile, tile))
    canvas.paste(q_crop, (10, 40))
    draw.text((10, 10), f"QUERY pid={TRUE_PID} f={QUERY_FRAME}", fill=(255, 255, 0))
    for i, r in enumerate(results[:n_show]):
        x0 = 10 + tile + 20 + i * (tile + 10)
        crop = crop_bbox(SEQ, r["last_seen_frame"], r["last_seen_bbox"]).resize((tile, tile))
        canvas.paste(crop, (x0, 40))
        color = (0, 220, 0) if r["sim"] >= THRESHOLD else (220, 60, 60)
        label = f"pid={r['pid']} sim={r['sim']:.3f}"
        if r["is_true"]:
            label += " [TRUE]"
        draw.text((x0, 10), label, fill=color)
    out_path = os.path.join(OUT_DIR, f"gallery_success2_{SEQ}.jpg")
    canvas.save(out_path, quality=92)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
