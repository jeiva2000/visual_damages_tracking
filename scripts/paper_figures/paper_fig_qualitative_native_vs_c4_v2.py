"""Figura cualitativa de la introducción (fig:c4_example), versión 2: etiquetas de las cajas con UN SOLO tamaño.

Motivo. En la versión original las etiquetas de las cajas ("ID6435 c0 0.79", "REID OK") estaban incrustadas en
los píxeles de los frames renderizados, y cada panel es un recorte reescalado a 420 px con un factor distinto
(pad 1.3 a 2.2): el texto salía entre ~8 y ~28 px de alto. Aquí cada panel se recompone desde el frame original
del dataset (1280x720) con el MISMO recorte que la versión original, y las cajas y sus etiquetas se dibujan
sobre el panel ya recortado, con la misma fuente (Carlito Bold), tamaño y grosor en los 12 paneles.

Datos: las cajas, IDs y confianzas salen de las predicciones publicadas (native_reid_<tracker>_alldata y
c4_puro_live_<tracker>_alldata); los tres primeros casos reproducen exactamente las de la figura original. En el
caso de DeepOC-SORT las predicciones publicadas difieren algo de las de la figura original (misma secuencia,
mismos frames 283 -> 465 e IDs 13 -> 55; cambian cajas y confianza).

Uso:
    python3 scripts/paper_figures/paper_fig_qualitative_native_vs_c4_v2.py --preview          # solo los paneles + mosaico de revisión
    python3 scripts/paper_figures/paper_fig_qualitative_native_vs_c4_v2.py --export           # además PPTX/PDF/PNG (LibreOffice remoto)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

IN_DIR = "paper/qualitative_examples/fig1_v2_inputs"
OUT_DIR = "paper/qualitative_examples"
FONT_BOLD = "paper/training_methodology/_fonts/Carlito-Bold.ttf"
PANEL = 420
FONT_PX = 25          # tamaño ÚNICO de letra de las etiquetas, en píxeles del panel de 420 px
LINE_W, LINE_W_HI = 3, 6
# recorte de la versión original (build_figure_new_sequences.py::CASES)
CROP = {
    "bytetrack": dict(box_before=(374, 30, 282, 155), box_after=(119, 16, 1012, 249), pad=1.3),
    "botsort": dict(box_before=(729, 519, 562, 183), box_after=(555, 539, 734, 178), pad=1.5),
    "ocsort": dict(box_before=(533, 13, 447, 436), box_after=(556, 55, 442, 533), pad=1.3),
    "deepocsort": dict(box_before=(281, 350, 106, 63), box_after=(250, 246, 145, 97), pad=2.2),
}
# El run publicado de ByteTrack (18-ago) no reiniciaba el contador de IDs entre secuencias: los IDs de la
# secuencia de la figura van de 6409 a 6462. Se muestran numerados desde 1 dentro de la secuencia (6409 -> 1,
# 6435 -> 27), que es lo que da hoy el pipeline con el contador reiniciado (comprobado: misma caja y confianza).
ID_OFFSET = {"bytetrack": 6408}
# la identidad original lleva el MISMO color antes de la oclusión y tras recuperarla con C4 (verde); el ID nuevo del nativo, rojo
REJECT, MATCH = (214, 48, 49), (46, 160, 67)
BEFORE = MATCH
OTHER = [(30, 144, 255), (186, 85, 211), (0, 190, 200), (255, 105, 180), (140, 140, 140), (80, 80, 200)]


def crop_window(size, box, pad):
    """Misma lógica que build_figure_new_sequences.py::crop_square, devolviendo la ventana (l, t, r, b)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * pad
    left, top, right, bottom = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
    iw, ih = size
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > iw:
        left -= (right - iw)
        right = iw
    if bottom > ih:
        top -= (bottom - ih)
        bottom = ih
    left, top = max(0, left), max(0, top)
    return int(left), int(top), int(right), int(bottom)


def text_color(fill):
    """Texto oscuro sobre colores claros (naranja, cian...), blanco sobre el resto."""
    lum = (0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]) / 255
    return (30, 30, 30) if lum > 0.55 else "white"


def tag(draw, font, text, x, y, fill, below=False):
    """Etiqueta rellena; (x, y) = esquina superior izquierda de la caja."""
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l + 8, b - t + 8
    ty = y if below else y - h
    if ty < 0:
        ty = 0 if not below else y
    tx = min(max(x, 0), PANEL - w)
    draw.rectangle([tx, ty, tx + w, ty + h], fill=fill)
    draw.text((tx + 4 - l, ty + 4 - t), text, font=font, fill=text_color(fill))


def render_panel(frame_path, rows, target_id, target_color, window, font, highlight=False, id_offset=0):
    img = Image.open(frame_path).convert("RGB")
    l, t, r, b = window
    panel = img.crop(window).resize((PANEL, PANEL), Image.LANCZOS)
    sx, sy = PANEL / (r - l), PANEL / (b - t)
    d = ImageDraw.Draw(panel)
    drawn = []
    for k, row in enumerate(sorted(rows, key=lambda q: q["id"] == target_id)):   # el objetivo se dibuja el último
        x0, y0 = (row["x"] - l) * sx, (row["y"] - t) * sy
        x1, y1 = x0 + row["w"] * sx, y0 + row["h"] * sy
        if x1 < 0 or y1 < 0 or x0 > PANEL or y0 > PANEL:
            continue
        is_t = row["id"] == target_id
        color = target_color if is_t else OTHER[row["id"] % len(OTHER)]
        d.rectangle([x0, y0, x1, y1], outline=color, width=(LINE_W_HI if (is_t and highlight) else LINE_W))
        drawn.append((row, x0, y0, x1, y1, color, is_t))
    for row, x0, y0, x1, y1, color, is_t in drawn:
        if x0 >= -2 and y0 >= -2:   # solo si la esquina superior izquierda de la caja es visible
            tag(d, font, "ID%d c0 %.2f" % (row["id"] - id_offset, row["conf"]), x0, y0, color)
        if is_t and highlight:
            tag(d, font, "REID OK", x0, y1, MATCH, below=True)
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=IN_DIR)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "qualitative_native_vs_c4_new_sequences_v2_print.pptx"))
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--export-host", default=os.environ.get("EXPORT_HOST", "localhost"))
    ap.add_argument("--fonts-dir", default="paper/training_methodology/_fonts")
    args = ap.parse_args()
    font = ImageFont.truetype(FONT_BOLD, FONT_PX)
    cases = json.load(open(os.path.join(args.in_dir, "fig1_cases.json")))
    base_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                            os.path.splitext(os.path.basename(args.out))[0] + "_assets")
    os.makedirs(base_dir, exist_ok=True)
    panels = {}
    for r, c in enumerate(cases):
        cr = CROP[c["tr"]]
        fb, fa = (os.path.join(args.in_dir, "%s_%s.jpg" % (c["tr"], w)) for w in ("before", "after"))
        size = Image.open(fb).size
        wb, wa = crop_window(size, cr["box_before"], cr["pad"]), crop_window(size, cr["box_after"], cr["pad"])
        specs = [(fb, c["before_native"], c["idb"], BEFORE, wb, False),
                 (fa, c["after_native"], c["idn"], REJECT, wa, False),
                 (fa, c["after_c4"], c["idb"], MATCH, wa, True)]
        for col, (fp, rows, tid, colr, win, hi) in enumerate(specs):
            p = render_panel(fp, rows, tid, colr, win, font, hi, ID_OFFSET.get(c["tr"], 0))
            name = "panel_%d_%d.png" % (r, col)
            p.save(os.path.join(base_dir, name))
            panels[(r, col)] = name
    if args.preview or not args.export:
        mosaic = Image.new("RGB", (3 * PANEL + 8, 4 * PANEL + 12), "white")
        for (r, col), name in panels.items():
            mosaic.paste(Image.open(os.path.join(base_dir, name)), (col * (PANEL + 4), r * (PANEL + 4)))
        mp = os.path.join(base_dir, "preview_mosaic.png")
        mosaic.save(mp)
        print("[OK] paneles en", base_dir, "| mosaico:", mp)
    if args.export:
        import paper_fig_qualitative_native_vs_c4 as base
        from paper_fig_qualitative_native_vs_c4 import build
        for bc, c in zip(base.CASES, cases):
            off = ID_OFFSET.get(c["tr"], 0)
            bc["before"], bc["native"], bc["c4"] = ("ID%d" % (c["idb"] - off), "ID%d" % (c["idn"] - off), "ID%d" % (c["idb"] - off))
        from paper_fig_reid_module_diagram_pptx import export
        build(panels, base_dir, args.out)
        print("[OK]", args.out)
        export(args.out, args.export_host, args.fonts_dir)


if __name__ == "__main__":
    main()
