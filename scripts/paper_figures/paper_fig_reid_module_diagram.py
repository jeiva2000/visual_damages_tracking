"""
Genera la figura-resumen del módulo de re-identificación del paper
(fig. de sec:frozen_association, `images/reid_module_diagram.pdf`) a partir
de un caso REAL de la validación en vivo, con la MISMA lógica que usa
`main_track.py --reid-stitch --reid-source c4_puro` (eval_tracking/
custom_reid.py::C4PuroReID + reid_gallery.py):

  - query  q   = media de las escalas ajustada (x1.0) e intermedia (x1.6)
                 de la detección que reaparece (C4PuroReID.get_features).
  - banco  b_j = media de hasta `num_track_history`=4 observaciones del
                 track perdido repartidas en el tiempo (_spread_indices_c4):
                 la más antigua a x6.0 (mínimo 400 px), las intermedias a
                 x1.6, la más reciente a x1.0 (C4PuroReID.pool_history).
  - s_ij       = coseno entre q y b_j (ambos L2-normalizados), MEDIDO -- no
                 ilustrativo.
  - a_j        = `frames_lost` de reid_gallery en el momento del match:
                 el track se entierra (bury) el frame siguiente a
                 desaparecer y age_and_prune lo envejece una vez por frame,
                 así que a_j = query_frame - último_frame - 1. Entran al
                 banco solo los tracks con a_j <= MAX_GAP.
  - historial  = main_track.py guarda el historial POR raw_id (el id que
                 pone el tracker), no por id final: si un id final ya había
                 sido fusionado antes por la galería, al perderse de nuevo
                 solo cuenta el tramo de su raw_id actual. El .txt de la
                 corrida solo tiene ids finales, así que el tramo se
                 reconstruye cortando en huecos > TRACK_BUFFER (más allá del
                 buffer el tracker ya no puede recuperar el raw_id por su
                 cuenta, ese tramo nuevo es necesariamente otro raw_id).
                 Huecos <= TRACK_BUFFER se asumen del mismo raw_id -- es la
                 única aproximación de la reconstrucción.

Reemplaza a los dos generadores ad-hoc previos (build_diagram.py con
matplotlib y build_reid_diagram_pptx.py), que vivían en scratchpads ya
borrados y recortaban las imágenes a mano desde el raster de
bank_query_diagram.pdf (tiles de 112 px) en vez de salir de los frames
originales. Los recortes que se dibujan acá son los MISMOS rectángulos
(_expand_bbox_c4, mismos factores y mínimo absoluto) sobre los que se hace
el ROI-Align del embedding -- lo que se ve es exactamente lo que se encodea.

Corre en el servidor (necesita el detector, los frames y los .txt de la
corrida en vivo). Salida: <out-prefix>.pdf/.png + <out-prefix>.json con
todo el detalle del caso (frames, cajas, similitudes) para trazabilidad +
<out-prefix>_assets/ con cada recorte en PNG a resolución nativa (rutas en
el .json). Con el .json y los assets, paper_fig_reid_module_diagram_pptx.py
arma la misma figura como .pptx editable (corre en local, sin modelo).

Caso por defecto: ByteTrack, video_Ve7DX8i8A6RX5h8OyFt_11, pid 9577 (GT
id 5, IoU >= 0.5 antes y después), hueco 299 -> 366 (67 frames > buffer, lo
resolvió el banco; el nativo pierde la identidad). Encontrado con una
búsqueda sobre las 72 secuencias de las corridas *_alldata.

Uso:
    python3 scripts/paper_figures/paper_fig_reid_module_diagram.py
    python3 scripts/paper_figures/paper_fig_reid_module_diagram.py --tracker botsort --seq <seq> \
        --true-pid <pid> --gap-before-frame <f> --query-frame <f>
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'reid_lib'))  # library modules
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'live_tracking'))  # custom_reid.py, reid_gallery.py
import argparse
import json
import os
import sys

sys.path.insert(0, "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw

from custom_reid import (
    C4_MIN_CONTEXT_CROP_SIZE_PX, C4_PADDING_FACTORS, C4PuroReID, _expand_bbox_c4, _spread_indices_c4,
)
from gallery_from_live_preds import CFG, PTH, DEVICE, FrameCache, frame_path, read_mot
from p4_backbone_encoder import P4BackboneEncoder

EVAL_TRACKING = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking"
THRESHOLD_C4 = 0.9352  # tau calibrado de C4 (sec:frozen_association)
MAX_GAP = 200          # T, presupuesto de tiempo del banco (eq:bank_evict)
NUM_TRACK_HISTORY = 4
TRACK_BUFFER = 30      # track_buffer de los 4 trackers en las corridas en vivo

# Paleta (misma que las versiones anteriores de la figura).
C_TEXT = "#2b2b2b"
C_MUTED = "#5b6770"
C_ARROW = "#7a7a7a"
C_QUERY = "#b8860b"
C_QUERY_TXT = "#8a6400"
C_BANK_BG = "#f3efe6"
C_BANK_BORDER = "#c9c2a8"
C_BOX_BG = "#eef1f4"
C_BOX_BORDER = "#8a949c"
C_MATCH = "#1e7a3c"
C_MATCH_BG = "#e6f3ea"
C_REJECT = "#a83232"
C_REJECT_BG = "#f7e6e6"
C_TAU_BG = "#fdf6e3"


# --------------------------------------------------------------------------- #
# Reconstrucción del caso (lógica de producción)
# --------------------------------------------------------------------------- #

def history_positions(obs):
    """(índices elegidos por _spread_indices_c4, factor de escala de cada uno)
    -- mismo criterio posición->escala que C4PuroReID.pool_history."""
    idxs = _spread_indices_c4(len(obs), NUM_TRACK_HISTORY)
    out = []
    for pos, i in enumerate(idxs):
        if pos == 0:
            scale_idx = 0
        elif pos == len(idxs) - 1:
            scale_idx = 2
        else:
            scale_idx = 1
        out.append((i, scale_idx))
    return out


def last_raw_segment(obs):
    """Último tramo de `obs` (lista cronológica (frame, bbox)) sin huecos
    > TRACK_BUFFER -- ver "historial" en el docstring del módulo."""
    start = 0
    for k in range(1, len(obs)):
        if obs[k][0] - obs[k - 1][0] > TRACK_BUFFER:
            start = k
    return obs[start:]


def l2n(v):
    return v / (np.linalg.norm(v) + 1e-9)


def build_case(args, cache):
    txt = os.path.join(EVAL_TRACKING, f"c4_puro_live_{args.tracker}_alldata", "txt", f"{args.seq}.txt")
    by_frame, by_id = read_mot(txt)

    # Banco en query_frame: el pid verdadero SOLO con sus observaciones
    # previas al hueco (así lo vio reid_gallery.bury() al perderse -- en el
    # .txt final ya aparece fusionado con su reaparición) + todos los pids
    # genuinamente perdidos con a_j <= MAX_GAP.
    candidates = {args.true_pid: last_raw_segment(
        [(f, b) for f, b in by_id[args.true_pid] if f <= args.gap_before_frame])}
    for pid, obs in by_id.items():
        if pid == args.true_pid:
            continue
        before = [(f, b) for f, b in obs if f < args.query_frame]
        if not before or len(before) < len(obs) and obs[len(before)][0] == args.query_frame:
            continue  # nunca visto antes, o activo en query_frame -> no está en el banco
        age = args.query_frame - before[-1][0] - 1
        # age == 0: desapareció justo en query_frame -- main_track.py lo
        # entierra DESPUÉS de match_newborns en ese mismo frame, así que
        # todavía no está en el banco cuando se decide el match.
        if 1 <= age <= MAX_GAP:
            candidates[pid] = last_raw_segment(before)

    q_bbox = dict(by_frame[args.query_frame])[args.true_pid]
    q_scales = cache.get_scales(args.seq, args.query_frame, [q_bbox])[0]  # (3,D) [x6.0, x1.6, x1.0]
    q_embed = l2n(q_scales[1:3].mean(axis=0))

    entries = []
    for pid, obs in candidates.items():
        positions = history_positions(obs)
        selected = []
        views = []
        for i, scale_idx in positions:
            fidx, bbox = obs[i]
            scales = cache.get_scales(args.seq, fidx, [bbox])[0]
            selected.append(scales[scale_idx])
            views.append({"frame": fidx, "bbox": list(bbox), "factor": C4_PADDING_FACTORS[scale_idx]})
        b_embed = l2n(np.mean(np.stack(selected, axis=0), axis=0))
        entries.append({
            "pid": pid,
            "is_true": pid == args.true_pid,
            "sim": float(np.dot(q_embed, b_embed)),
            "age": args.query_frame - obs[-1][0] - 1,
            "n_obs": len(obs),
            "views": views,
        })
    entries.sort(key=lambda e: -e["sim"])

    best = entries[0]
    return {
        "tracker": args.tracker,
        "seq": args.seq,
        "true_pid": args.true_pid,
        "gap_before_frame": args.gap_before_frame,
        "query_frame": args.query_frame,
        "query_bbox": list(q_bbox),
        "threshold": THRESHOLD_C4,
        "max_gap": MAX_GAP,
        "bank_size": len(entries),
        "best_pid": best["pid"],
        "best_sim": best["sim"],
        "pred_match": best["sim"] >= THRESHOLD_C4,
        "correct": best["sim"] >= THRESHOLD_C4 and best["is_true"],
        "entries": entries,
    }


# --------------------------------------------------------------------------- #
# Recortes para dibujar (mismos rectángulos que el ROI-Align)
# --------------------------------------------------------------------------- #

# Rectángulos a tapar SOLO en lo que se dibuja (matrículas) -- {frame: [(x1,y1,x2,y2), ...]},
# se llena desde --mask en main(). El embedding (FrameCache.get_scales) lee el
# frame original sin tocar, así que las similitudes no dependen de esto.
DISPLAY_MASKS = {}


def load_frame(seq, frame_idx):
    img = Image.open(frame_path(seq, frame_idx)).convert("RGB")
    if frame_idx in DISPLAY_MASKS:
        draw = ImageDraw.Draw(img)
        for rect in DISPLAY_MASKS[frame_idx]:
            draw.rectangle(rect, fill=(160, 160, 160))
    return img


def view_crop(seq, frame_idx, bbox, factor):
    img = load_frame(seq, frame_idx)
    min_size = C4_MIN_CONTEXT_CROP_SIZE_PX if factor == C4_PADDING_FACTORS[0] else 0.0
    box = _expand_bbox_c4(tuple(bbox), factor, img.size, min_size_px=min_size)
    return np.asarray(img.crop(tuple(int(round(v)) for v in box)))


def context_crop(seq, frame_idx, bbox, pad_ratio=2.5):
    """Recorte de contexto del frame completo alrededor de la detección, con
    la caja del detector dibujada -- solo para el panel "reappeared
    detection"; no interviene en el embedding."""
    img = load_frame(seq, frame_idx)
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad = max(w, h) * pad_ratio
    cx0, cy0 = max(0, x1 - pad), max(0, y1 - pad)
    cx1, cy1 = min(img.width, x2 + pad), min(img.height, y2 + pad)
    crop = np.asarray(img.crop((int(cx0), int(cy0), int(cx1), int(cy1))))
    return crop, (x1 - cx0, y1 - cy0, w, h)


# --------------------------------------------------------------------------- #
# Dibujo
# --------------------------------------------------------------------------- #

class Canvas:
    """Ejes en coordenadas de "diseño" (0..W x 0..H, origen abajo-izquierda)
    para poder ubicar cajas, imágenes y flechas en unidades fijas."""

    def __init__(self, width, height, scale=0.1):
        self.fig = plt.figure(figsize=(width * scale, height * scale))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")

    def image(self, arr, x, y, w, h, border=None, lw=1.2):
        self.ax.imshow(arr, extent=(x, x + w, y, y + h), aspect="auto", zorder=3, interpolation="lanczos")
        self.ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=border or "#9aa3aa",
                                    linewidth=lw, zorder=4))

    def box(self, x, y, w, h, fc, ec, text="", size=10, color=C_TEXT, weight="normal", lw=1.2, pad=0.6):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
                                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
        if text:
            self.text(x + w / 2, y + h / 2, text, size=size, color=color, weight=weight)

    def text(self, x, y, s, size=10, color=C_TEXT, weight="normal", ha="center", va="center", style="normal"):
        self.ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, style=style, zorder=5)

    def arrow(self, x0, y0, x1, y1, color=C_ARROW, lw=1.1, rad=0.0, zorder=3):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                          color=color, linewidth=lw, zorder=zorder,
                                          connectionstyle=f"arc3,rad={rad}"))

    def save(self, prefix):
        for ext in ("pdf", "png"):
            self.fig.savefig(f"{prefix}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(self.fig)


def select_rows(case, n_rows):
    """Las `n_rows` entradas de mayor similitud; si la verdadera no entra,
    reemplaza a la última (siempre se muestra)."""
    rows = case["entries"][:n_rows]
    if not any(e["is_true"] for e in rows):
        rows = rows[:-1] + [next(e for e in case["entries"] if e["is_true"])]
    return rows


def save_assets(case, out_prefix):
    """Guarda cada recorte (query + todas las vistas de todas las entradas
    del banco) como PNG en <out_prefix>_assets/ y anota la ruta relativa al
    .json en el propio `case` -- insumo de paper_fig_reid_module_diagram_pptx.py."""
    asset_dir = f"{out_prefix}_assets"
    os.makedirs(asset_dir, exist_ok=True)
    rel_dir = os.path.basename(asset_dir)
    seq, qf, q_bbox = case["seq"], case["query_frame"], case["query_bbox"]
    case["query_assets"] = {}
    for name, factor in (("tight", C4_PADDING_FACTORS[2]), ("intermediate", C4_PADDING_FACTORS[1])):
        fname = f"query_{name}.png"
        Image.fromarray(view_crop(seq, qf, q_bbox, factor)).save(os.path.join(asset_dir, fname))
        case["query_assets"][name] = {"path": f"{rel_dir}/{fname}", "factor": factor}
    for e in case["entries"]:
        for k, v in enumerate(e["views"]):
            fname = f"bank_{e['pid']}_{k}.png"
            Image.fromarray(view_crop(seq, v["frame"], v["bbox"], v["factor"])).save(os.path.join(asset_dir, fname))
            v["path"] = f"{rel_dir}/{fname}"


def factor_label(f):
    return f"×{f:.1f}"


def draw_figure(case, out_prefix, n_rows):
    W, H = 160.0, 60.0
    cv = Canvas(W, H)
    seq, qf, q_bbox = case["seq"], case["query_frame"], case["query_bbox"]

    # ---- 1. Query generation ------------------------------------------------
    cv.text(1, H - 2.5, "1. Query generation", size=12, weight="bold", color=C_MUTED, ha="left")

    ctx, (bx, by, bw, bh) = context_crop(seq, qf, q_bbox)
    ch, cw = ctx.shape[:2]
    det_w = 20.0
    det_h = det_w * ch / cw
    det_x, det_y = 1.0, H / 2 - det_h / 2 - 1
    cv.image(ctx, det_x, det_y, det_w, det_h)
    sx, sy = det_w / cw, det_h / ch
    cv.ax.add_patch(Rectangle((det_x + bx * sx, det_y + det_h - (by + bh) * sy), bw * sx, bh * sy,
                              fill=False, edgecolor=C_QUERY, linewidth=1.6, zorder=4))
    cv.text(det_x + det_w / 2, det_y - 2.2, "reappeared detection", size=9, color=C_MUTED, style="italic")

    QV = 11.0
    qv_x = 29.0
    q_views = [(C4_PADDING_FACTORS[2], "tight"), (C4_PADDING_FACTORS[1], "intermediate")]
    q_ys = [H / 2 + 4.5, H / 2 - 4.5 - QV - 3]
    for (factor, name), qy in zip(q_views, q_ys):
        cv.image(view_crop(seq, qf, q_bbox, factor), qv_x, qy, QV, QV, border=C_QUERY, lw=1.8)
        cv.text(qv_x + QV / 2, qy - 1.6, f"{name} ({factor_label(factor)})", size=9, color=C_QUERY_TXT,
                weight="bold")
        cv.arrow(det_x + det_w + 0.3, H / 2 - 1, qv_x - 0.4, qy + QV / 2)

    mean_x, mean_w, mean_h = 44.5, 6.5, 4.0
    mean_y = H / 2 - 1 - mean_h / 2
    cv.box(mean_x, mean_y, mean_w, mean_h, C_BOX_BG, C_BOX_BORDER, "mean", size=10)
    for qy in q_ys:
        cv.arrow(qv_x + QV + 0.4, qy + QV / 2, mean_x - 0.3, mean_y + mean_h / 2)

    q_cx, q_cy, q_r = 56.5, H / 2 - 1, 2.8
    cv.ax.add_patch(plt.Circle((q_cx, q_cy), q_r, facecolor="#e3eaf2", edgecolor="#3d4b59", linewidth=1.4, zorder=2))
    cv.text(q_cx, q_cy, "$q$", size=13, weight="bold")
    cv.arrow(mean_x + mean_w + 0.3, q_cy, q_cx - q_r - 0.2, q_cy)

    # ---- 2. Identity bank ---------------------------------------------------
    rows = select_rows(case, n_rows)

    bank_x, bank_w = 63.5, 50.0
    bank_y, bank_h = 3.0, H - 9.5
    cv.text(bank_x, H - 2.5, "2. Identity bank (recently-lost tracks)", size=12, weight="bold",
            color=C_MUTED, ha="left")
    cv.box(bank_x, bank_y, bank_w, bank_h, C_BANK_BG, C_BANK_BORDER, lw=1.2, pad=1.2)
    cv.text(bank_x + bank_w / 2, bank_y + bank_h - 2.0,
            f"entry $j$ evicted once $a_j > T = {MAX_GAP}$ frames", size=8.5, color="#6b5d2e", style="italic")

    TILE = 8.4
    GAP = 0.7
    row_pitch = (bank_h - 9.5) / n_rows
    row_x = bank_x + 3.0
    for r, e in enumerate(rows):
        row_top = bank_y + bank_h - 4.5 - r * row_pitch
        ty = row_top - TILE - 0.6
        is_best = e["pid"] == case["best_pid"]
        border = C_MATCH if (is_best and case["pred_match"]) else None
        for k, v in enumerate(e["views"]):
            tx = row_x + k * (TILE + GAP)
            cv.image(view_crop(seq, v["frame"], v["bbox"], v["factor"]), tx, ty, TILE, TILE,
                     border=border, lw=1.8 if border else 1.0)
            cv.text(tx + TILE / 2, ty - 1.2, factor_label(v["factor"]), size=7, color=C_MUTED)
        row_end = row_x + NUM_TRACK_HISTORY * (TILE + GAP)
        b_cx, b_cy = row_end + 2.6, ty + TILE / 2
        cv.ax.add_patch(plt.Circle((b_cx, b_cy), 1.9, facecolor="white",
                                   edgecolor=C_MATCH if border else "#6b6b6b", linewidth=1.2, zorder=2))
        cv.text(b_cx, b_cy, f"$b_{{{r + 1}}}$", size=9)
        cv.arrow(row_x + len(e["views"]) * (TILE + GAP) - GAP + 0.2, b_cy, b_cx - 2.0, b_cy, lw=0.8)
        s_color = C_MATCH if border else C_TEXT
        cv.text(b_cx + 3.0, b_cy + 1.3, f"$s = {e['sim']:.4f}$", size=8.5, color=s_color,
                weight="bold" if border else "normal", ha="left")
        cv.text(b_cx + 3.0, b_cy - 1.4, f"$a_j = {e['age']}$", size=8, color=C_MUTED, ha="left")
        cv.arrow(q_cx + q_r * 0.9, q_cy, row_x - 0.4, b_cy, color=C_MATCH if border else C_ARROW,
                 lw=1.4 if border else 0.9, rad=0.0)

    n_hidden = case["bank_size"] - len(rows)
    footer = "each row: up to 4 views spread over the track history, averaged into $b_j$"
    if n_hidden > 0:
        footer += f"  ($+{n_hidden}$ more entries not shown)"
    cv.text(bank_x + bank_w / 2, bank_y + 2.0, footer, size=7.5, color=C_MUTED, style="italic")

    # ---- 3. Re-identification decision --------------------------------------
    d_x = 117.0
    cv.text(d_x, H - 2.5, "3. Re-identification decision", size=12, weight="bold", color=C_MUTED, ha="left")
    bw_, bh_ = 17.0, 7.0
    best_y = H / 2 - bh_ / 2 - 1
    cv.box(d_x, best_y, bw_, bh_, "white", "#444444",
           f"best match (Hungarian):\n$s = {case['best_sim']:.4f}$", size=9)
    tau_x = d_x + bw_ + 4.0
    cv.box(tau_x, best_y, bw_, bh_, C_TAU_BG, "#444444",
           f"$s \\geq \\tau$ ?\n$\\tau = {THRESHOLD_C4}$", size=9.5)
    cv.arrow(d_x + bw_ + 0.3, best_y + bh_ / 2, tau_x - 0.3, best_y + bh_ / 2)

    matched = case["pred_match"]
    yes_y = best_y + bh_ + 5.0
    no_y = best_y - bh_ - 5.0
    cv.box(tau_x, yes_y, bw_, bh_, C_MATCH_BG, C_MATCH, "MATCH\nmerge track ID", size=9.5, color=C_MATCH,
           weight="bold", lw=1.8 if matched else 1.0)
    cv.box(tau_x, no_y, bw_, bh_, C_REJECT_BG, C_REJECT, "below $\\tau$:\nnew identity", size=9.5,
           color=C_REJECT, weight="bold", lw=1.0 if matched else 1.8)
    cv.arrow(tau_x + bw_ / 2, best_y + bh_ + 0.3, tau_x + bw_ / 2, yes_y - 0.3,
             color=C_MATCH, lw=1.8 if matched else 0.8)
    cv.text(tau_x + bw_ / 2 + 1.0, best_y + bh_ + 2.5, "yes", size=8.5, color=C_MATCH, ha="left", style="italic")
    cv.arrow(tau_x + bw_ / 2, best_y - 0.3, tau_x + bw_ / 2, no_y + bh_ + 0.3,
             color=C_REJECT, lw=0.8 if matched else 1.8)
    cv.text(tau_x + bw_ / 2 + 1.0, best_y - 2.5, "no", size=8.5, color=C_REJECT, ha="left", style="italic")

    # la flecha del mejor candidato sale del borde derecho del banco, a la
    # altura de su fila, hacia la caja "best match"
    best_row = next(r for r, e in enumerate(rows) if e["pid"] == case["best_pid"])
    b_cy_best = bank_y + bank_h - 4.5 - best_row * row_pitch - TILE / 2 - 0.6
    cv.arrow(bank_x + bank_w + 0.2, b_cy_best, d_x - 0.3, best_y + bh_ * 0.75,
             color=C_MATCH if matched else C_ARROW, lw=1.4)

    cv.save(out_prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort", "ocsort", "deepocsort"])
    parser.add_argument("--seq", default="video_Ve7DX8i8A6RX5h8OyFt_11")
    parser.add_argument("--true-pid", type=int, default=9577)
    parser.add_argument("--gap-before-frame", type=int, default=299,
                        help="último frame del pid verdadero ANTES del hueco (el banco solo ve hasta acá)")
    parser.add_argument("--query-frame", type=int, default=366, help="frame de la reaparición")
    parser.add_argument("--n-rows", type=int, default=3, help="entradas del banco a dibujar (la verdadera siempre)")
    parser.add_argument("--out-prefix", default="paper_figures/reid_module_diagram")
    parser.add_argument("--mask", action="append", default=None, metavar="FRAME:x1,y1,x2,y2",
                        help="rectángulo a tapar en gris (matrícula) solo al dibujar; repetible. "
                             "Default: ninguna.")
    args = parser.parse_args()

    for m in args.mask or []:
        frame_s, rect_s = m.split(":")
        DISPLAY_MASKS.setdefault(int(frame_s), []).append(tuple(float(v) for v in rect_s.split(",")))

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    print("[INFO] cargando backbone C4 (cpu)...")
    enc = P4BackboneEncoder(CFG, PTH, layer="c4").eval()
    for p in enc.parameters():
        p.requires_grad = False
    c4reid = C4PuroReID(num_track_history=NUM_TRACK_HISTORY, device=DEVICE)
    c4reid.attach(enc.model)
    cache = FrameCache(enc.model, c4reid)

    with torch.no_grad():
        case = build_case(args, cache)

    print(f"[CASE] {case['tracker']} {case['seq']} pid={case['true_pid']} "
          f"f{case['gap_before_frame']}->f{case['query_frame']} bank_size={case['bank_size']}")
    for e in case["entries"]:
        tag = "  <- VERDADERO" if e["is_true"] else ""
        print(f"  pid={e['pid']:>6} sim={e['sim']:.4f} a_j={e['age']:>3} n_obs={e['n_obs']}{tag}")
    print(f"[DECISION] best pid={case['best_pid']} sim={case['best_sim']:.4f} "
          f"match={case['pred_match']} correcto={case['correct']}")

    save_assets(case, args.out_prefix)
    case["rows_shown"] = [e["pid"] for e in select_rows(case, args.n_rows)]
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(case, f, indent=2)
    draw_figure(case, args.out_prefix, args.n_rows)
    print(f"[OK] {args.out_prefix}.pdf/.png/.json + {args.out_prefix}_assets/")


if __name__ == "__main__":
    main()
