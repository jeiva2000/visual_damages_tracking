"""
Compone, para cada caso, un panel NATIVE (sin mecanismo -- solo el id que le
puso el tracker nativo, antes/despues) al lado del panel C4 YA GENERADO
(banco+query, plantilla visualize_episode, ver gallery_viz_template.py) --
para verlos lado a lado en una sola imagen.
"""
import os

from PIL import Image, ImageDraw, ImageFont

DATA_ROOT = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/all_dataset_view/all"
OUT_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking/c4_proof_visuals"
NATIVE_DIR = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking/native_reid_bytetrack_alldata/txt"


def read_mot(txt_path):
    by_frame = {}
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame = int(float(p[0])); pid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            by_frame.setdefault(frame, {})[pid] = (x, y, x + w, y + h)
    return by_frame


def crop_bbox(seq, frame_idx, bbox, pad=0.3, size=260):
    img = Image.open(os.path.join(DATA_ROOT, seq, "img1", f"{frame_idx:06d}.jpg")).convert("RGB")
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1 -= w * pad; x2 += w * pad; y1 -= h * pad; y2 += h * pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    crop = img.crop((x1, y1, x2, y2))
    crop.thumbnail((size, size))
    return crop


def build_native_panel(seq, frame_before, id_before, frame_after, id_after, failed):
    by_frame = read_mot(os.path.join(NATIVE_DIR, f"{seq}.txt"))
    bbox_before = by_frame[frame_before][id_before]
    c1 = crop_bbox(seq, frame_before, bbox_before)
    if frame_before == frame_after and id_before == id_after:
        # objeto genuinamente nuevo: un solo crop (el nacimiento), sin par antes/despues
        W = c1.width + 20
        H = c1.height + 70
        panel = Image.new("RGB", (W, H), (15, 15, 15))
        draw = ImageDraw.Draw(panel)
        draw.text((10, 8), "NATIVE (sin mecanismo de reid)", fill=(255, 255, 255))
        panel.paste(c1, (0, 60))
        draw.text((5, 40), f"f={frame_before} id={id_before} (nace nuevo, no hay nada que matchear)", fill=(0, 220, 0))
        return panel

    bbox_after = by_frame[frame_after][id_after]
    c2 = crop_bbox(seq, frame_after, bbox_after)
    W = c1.width + c2.width + 30
    H = max(c1.height, c2.height) + 70
    panel = Image.new("RGB", (W, H), (15, 15, 15))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), "NATIVE (sin mecanismo de reid)", fill=(255, 255, 255))
    panel.paste(c1, (0, 60))
    draw.text((5, 40), f"f={frame_before} id={id_before}", fill=(180, 180, 255))
    panel.paste(c2, (c1.width + 20, 60))
    color = (230, 60, 60) if failed else (0, 220, 0)
    status = "id NUEVO -> fallo" if failed else "mismo id -> ok"
    draw.text((c1.width + 25, 40), f"f={frame_after} id={id_after}  ({status})", fill=color)
    return panel


def compose(seq, native_panel, c4_img_path, out_path, title):
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
    # caso 1: exito -- native falla (id nuevo), c4 reidentifica
    seq1 = "video_1Vl7JUY8fBQhAph15I7_11"
    panel1 = build_native_panel(seq1, 31, 5861, 181, 5921, failed=True)
    compose(
        seq1, panel1,
        os.path.join(OUT_DIR, f"tpl_success_{seq1}.jpg"),
        os.path.join(OUT_DIR, f"compare_native_vs_c4_success_{seq1}.jpg"),
        f"NATIVE vs +C4 -- {seq1} (misma oclusion real, ~150 frames)",
    )

    # caso 2: objeto nuevo -- nativo y c4 COINCIDEN (nada que reidentificar,
    # ambos le ponen id nuevo correctamente) -- se incluye para contraste,
    # no es una falla de nativo.
    seq2 = "VID_20260118_222031"
    panel2 = build_native_panel(seq2, 174, 155, 174, 155, failed=False)
    compose(
        seq2, panel2,
        os.path.join(OUT_DIR, f"tpl_newtrack_{seq2}.jpg"),
        os.path.join(OUT_DIR, f"compare_native_vs_c4_newtrack_{seq2}.jpg"),
        f"NATIVE vs +C4 -- {seq2} (objeto genuinamente nuevo, ambos coinciden)",
    )


if __name__ == "__main__":
    main()
