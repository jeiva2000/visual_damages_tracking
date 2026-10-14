"""
Arma la figura-resumen del módulo de re-identificación como .pptx editable,
con la maqueta de `paper/training_methodology/reid_module_diagram.pptx`
(misma diagramación en 3 bloques, fuente Calibri, paleta, grosores de línea,
conectores enganchados a las formas con punta triangular y sombra), pero con
el contenido del caso REAL que reconstruye paper_fig_reid_module_diagram.py:
recortes a resolución nativa, similitudes medidas y las escalas que usa de
verdad el pipeline C4 (query = x1.0 + x1.6; banco = hasta 4 vistas x6.0 /
x1.6 / x1.0).

Diferencias deliberadas con la maqueta de ejemplo (que tenía contenido
ilustrativo, ver paper_fig_reid_module_diagram.py):
  - la segunda vista de la query es la intermedia (x1.6), no "wide (x6.0)";
    su etiqueta va debajo de la imagen en vez de entre las flechas.
  - cada fila del banco muestra las vistas que entran de verdad a b_j (hasta
    4, no 3), con su escala debajo; el banco se ensancha para que entren y
    el bloque 3 se corre a la derecha lo mismo.
  - las similitudes son las medidas (4 decimales).
  - cada recorte lleva debajo (o encima) una tira de celdas de color que
    representa SU embedding (sin etiqueta en la query; e_{j,k} en el
    banco, notación de las ecuaciones); las tiras de una fila
    se agrupan en un marco punteado y se promedian en la tira b_j, que es
    la que se compara con q (eq:cosine_sim) y la que se conecta con la
    decisión. Los colores de las tiras son DECORATIVOS (semilla fija), no
    valores reales del embedding.
  - la fila con mejor similitud va abajo y es la que se conecta con la
    decisión, igual que en el ejemplo; si el caso fuese un no-match, el
    camino resaltado sería el de "new identity".

No necesita el modelo: corre en local a partir del .json y la carpeta
_assets/ que genera paper_fig_reid_module_diagram.py en el servidor.

Con --export además genera <out>.pdf y <out>.png recortados al contenido.
La conversión la hace LibreOffice en --export-host (el servidor: acá no hay
LibreOffice ni Calibri) con Carlito -- clon libre de Calibri con las mismas
métricas, que LibreOffice usa automáticamente en lugar de Calibri -- desde
--fonts-dir, dentro de un HOME temporal (no instala nada en el sistema).
Detalles que evitan resultados degradados, encontrados al exportar:
  - LibreOffice 6.4 ignora las opciones de exportación por línea de comandos
    y por defecto re-muestrea las imágenes a JPEG muy comprimido: se fuerzan
    ReduceImageResolution=false / UseLosslessCompression=true en el perfil
    de usuario del HOME temporal.
  - el recorte se hace en origen: una primera conversión de la diapositiva
    completa da la caja de tinta (en píxeles), y se guarda una copia del
    .pptx con el tamaño de diapositiva ajustado a esa caja; la segunda conversión ya
    sale sin márgenes (no hace falta ninguna librería de PDF).
El .pptx editable (<out>) conserva siempre el tamaño 13.33x7.5 del ejemplo.

Uso:
    python3 scripts/paper_figures/paper_fig_reid_module_diagram_pptx.py            # solo .pptx
    python3 scripts/paper_figures/paper_fig_reid_module_diagram_pptx.py --export   # + .pdf y .png
    python3 scripts/paper_figures/paper_fig_reid_module_diagram_pptx.py --export-only   # exporta el .pptx
        # tal como está en disco, sin regenerarlo (respeta ediciones a mano)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'reid_lib'))  # library modules
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import tempfile

from PIL import Image, ImageChops

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# Paleta y grosores tomados de reid_module_diagram.pptx.
INK = "222222"
MUTED = "666666"
BOX_BG, BOX_BORDER = "ECECEC", "999999"
BANK_BG, BANK_BORDER = "F3EEDD", "C9B87A"
Q_BG, Q_BORDER = "D9E4F5", "5A7CB8"
QUERY = "B8860B"
MATCH, MATCH_BG = "2E7D32", "E1F3E1"
REJECT, REJECT_BG = "B03A2E", "FBE3E1"
TILE_BORDER = "999999"
LW_TILE, LW_BOX, LW_BANK, LW_HI = 0.5, 0.75, 0.75, 1.25  # pt reales (maquetas a tamaño de impresión)
FONT = "Calibri"
# Rampa de azules (alrededor del borde de q) para las tiras de embedding decorativas.
EMB_RAMP = ["DCE6F4", "B5C9E8", "8BA8D8", "5A7CB8", "35578F"]
EMB_CELLS = 8



def rgb(hex_):
    return RGBColor.from_string(hex_)


def factor_label(f):
    return f"×{f:.1f}"


class Slide:
    def __init__(self, base_dir):
        self.prs = Presentation()
        self.prs.slide_width = Emu(12192000)   # 13.333 in, igual que el ejemplo
        self.prs.slide_height = Emu(6858000)   # 7.5 in
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.base_dir = base_dir

    # ---- formas ------------------------------------------------------------
    @staticmethod
    def _no_theme_shadow(shape):
        """python-pptx agrega un <p:style> con effectRef idx>0: PowerPoint lo
        anula con el <a:effectLst/> vacío de shadow.inherit=False, pero
        LibreOffice lo ignora y dibuja la sombra del tema -- se fuerza a 0."""
        shape.shadow.inherit = False
        ref = shape._element.find(".//" + qn("a:effectRef"))
        if ref is not None:
            ref.set("idx", "0")

    def _style_line(self, shape, color, width):
        shape.line.color.rgb = rgb(color)
        shape.line.width = Pt(width)

    def text(self, x, y, w, h, s, size, color=INK, bold=False, italic=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP):
        tb = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        self._fill_text(tb.text_frame, s, size, color, bold, italic, align, anchor, inset=0)
        return tb

    def _fill_text(self, tf, s, size, color, bold, italic, align, anchor, inset):
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Emu(inset * 2)
        tf.margin_top = tf.margin_bottom = Emu(inset)
        p = tf.paragraphs[0]
        p.alignment = align
        p.text = s  # "\v" -> <a:br/> dentro del mismo párrafo, como en el ejemplo
        for run in p.runs:
            f = run.font
            f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
            f.color.rgb = rgb(color)

    def rich_text(self, x, y, w, h, segments, size, color=MUTED, align=PP_ALIGN.CENTER):
        """`segments`: lista de (texto, subíndice?, cursiva?) -- para la
        notación v_adj, e_{j,k}, b_j sin depender de fuentes matemáticas."""
        tb = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
        p = tf.paragraphs[0]
        p.alignment = align
        for text, sub, italic in segments:
            run = p.add_run()
            run.text = text
            f = run.font
            f.name, f.size, f.italic = FONT, Pt(size), italic
            f.color.rgb = rgb(color)
            if sub:
                f._element.set("baseline", "-25000")
        return tb

    def strip(self, x, y, w, h, seed, border=Q_BORDER, lw=0.75):
        """Tira de EMB_CELLS celdas = un embedding (colores decorativos,
        determinísticos por `seed`). Agrupada; devuelve el marco exterior
        (para enganchar conectores)."""
        rng = random.Random(seed)
        grp = self.slide.shapes.add_group_shape()
        cw = w / EMB_CELLS
        for c in range(EMB_CELLS):
            cell = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + c * cw), Inches(y), Inches(cw), Inches(h))
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(rng.choice(EMB_RAMP))
            cell.line.fill.background()
            self._no_theme_shadow(cell)
        frame = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        frame.fill.background()
        self._style_line(frame, border, lw)
        self._no_theme_shadow(frame)
        return frame

    def box(self, x, y, w, h, fill, border, s="", size=11, color=INK, bold=False, italic=False,
            lw=LW_BOX, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
        sp = self.slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        sp.fill.solid()
        sp.fill.fore_color.rgb = rgb(fill)
        self._style_line(sp, border, lw)
        self._no_theme_shadow(sp)
        if s:
            self._fill_text(sp.text_frame, s, size, color, bold, italic, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE,
                            inset=25400)
        return sp

    def image(self, rel_path, x, y, w, h, border, lw):
        pic = self.slide.shapes.add_picture(os.path.join(self.base_dir, rel_path),
                                            Inches(x), Inches(y), Inches(w), Inches(h))
        self._style_line(pic, border, lw)
        return pic

    # ---- conectores ----------------------------------------------------------
    @staticmethod
    def cxn_point(shape, idx):
        """Coordenadas (EMU) del punto de conexión `idx` -- rect/roundRect:
        0 arriba, 1 izq, 2 abajo, 3 der; elipse: 8 puntos en sentido
        antihorario desde arriba (0 arriba, 2 izq, 4 abajo, 6 der, 7 arriba-der)."""
        x, y, w, h = shape.left, shape.top, shape.width, shape.height
        if shape.shape_type == 1 and shape.auto_shape_type == MSO_SHAPE.OVAL:
            ang = math.pi / 2 + idx * math.pi / 4  # 0 = arriba, sentido antihorario
            return int(x + w / 2 + w / 2 * math.cos(ang)), int(y + h / 2 - h / 2 * math.sin(ang))
        return {0: (x + w // 2, y), 1: (x, y + h // 2), 2: (x + w // 2, y + h), 3: (x + w, y + h // 2)}[idx]

    def arrow(self, start, end, color=MUTED, lw=LW_BOX):
        (s_shape, s_idx), (e_shape, e_idx) = start, end
        x0, y0 = self.cxn_point(s_shape, s_idx)
        x1, y1 = self.cxn_point(e_shape, e_idx)
        c = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(x0), Emu(y0), Emu(x1), Emu(y1))
        # enganche a las formas (el conector sigue a la forma al moverla en PowerPoint)
        cnv = c._element.find(qn("p:nvCxnSpPr")).find(qn("p:cNvCxnSpPr"))
        for tag, shp, idx in (("a:stCxn", s_shape, s_idx), ("a:endCxn", e_shape, e_idx)):
            el = etree.SubElement(cnv, qn(tag))
            el.set("id", str(shp.shape_id))
            el.set("idx", str(idx))
        self._style_line(c, color, lw)
        ln = c.line._get_or_add_ln()
        tail = etree.SubElement(ln, qn("a:tailEnd"))
        tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
        # sin sombra: a tamaño de impresión la sombra del ejemplo (pensada para
        # una diapositiva que se reduce ~0.4x) se ve como una línea doble. Hay
        # que anular también la del tema (effectRef del <p:style>), que
        # LibreOffice dibuja aunque no haya <a:effectLst>.
        c._element.spPr.append(etree.fromstring(
            '<a:effectLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>'))
        ref = c._element.find(".//" + qn("a:effectRef"))
        if ref is not None:
            ref.set("idx", "0")
        return c

    def save(self, path):
        self.prs.save(path)


def build(case, base_dir, out_path, crop=None, layout="decision", n_rows=3):
    """`layout`: "decision" (figura-resumen del módulo, sec:frozen_association)
    o "gallery" (construcción del banco y la query, fig:batch).
    `crop` (opcional, pulgadas): (x0, y0, x1, y1) -- desplaza todo y
    ajusta el tamaño de la diapositiva a esa caja (solo para exportar)."""
    sl = Slide(base_dir)
    if layout == "gallery":
        _draw_gallery(sl, case, n_rows)
    else:
        _draw_decision(sl, case)
    if crop is not None:
        x0, y0, x1, y1 = (Inches(v) for v in crop)
        for shape in sl.slide.shapes:
            shape.left, shape.top = shape.left - x0, shape.top - y0
        sl.prs.slide_width, sl.prs.slide_height = Emu(x1 - x0), Emu(y1 - y0)
    sl.save(out_path)


# Maquetas a TAMAÑO DE IMPRESIÓN (1:1): coordenadas en pulgadas reales sobre
# una diapositiva de PRINT_W de ancho, fuentes en pt reales. El paper usa
# elsarticle [preprint,12pt] -> \textwidth = 390 pt = 5.4 in (final,1p:
# 384 pt), y las figuras se incluyen a \linewidth, así que se imprimen ~1:1.
# Ningún texto baja de FS_MIN (Elsevier pide ~7-8 pt en la figura impresa).
PRINT_W = 5.40
FS_MIN, FS_BOX, FS_TITLE, FS_Q = 8, 8.5, 9.5, 10


def _draw_decision(sl, case):
    """Figura-resumen del módulo (sec:frozen_association), en dos pisos para
    que entre a 5.4 in con letra legible: arriba query + banco, abajo la
    decisión. Recoge los ajustes hechos a mano sobre la versión apaisada:
    títulos de bloque en una misma línea y alineados con su bloque, etiqueta
    "tight" separada de su imagen, y la fila central del banco alineada con
    q (flecha q -> fila central horizontal)."""
    sl.prs.slide_width, sl.prs.slide_height = Inches(PRINT_W + 0.2), Inches(3.4)
    matched = case["pred_match"]
    hi = MATCH if matched else REJECT
    TY = 0.05  # línea común de los títulos del piso superior

    # ---- 1. Query generation -------------------------------------------------
    sl.text(0.05, TY, 2.6, 0.18, "1. Query generation", FS_TITLE, bold=True)
    CY = 1.25          # eje horizontal del bloque (centro de la detección y de q)
    QW, QS_H = 0.60, 0.07
    QX = 0.92
    det = sl.box(0.05, CY - 0.25, 0.70, 0.50, BOX_BG, BOX_BORDER, "reappeared\vdetection", size=FS_MIN, lw=LW_BOX)
    qa = case["query_assets"]
    t_y = CY - 0.85
    sl.text(QX - 0.2, t_y - 0.19, QW + 0.4, 0.14, f"tight ({factor_label(qa['tight']['factor'])})", FS_MIN,
            color=QUERY, bold=True, align=PP_ALIGN.CENTER)
    tight = sl.image(qa["tight"]["path"], QX, t_y, QW, QW, QUERY, LW_HI)
    v_adj = sl.strip(QX, t_y + QW + 0.10, QW, QS_H, seed="v_adj", lw=LW_TILE)
    i_y = CY + 0.25
    v_int = sl.strip(QX, i_y - 0.10 - QS_H, QW, QS_H, seed="v_int", lw=LW_TILE)
    inter = sl.image(qa["intermediate"]["path"], QX, i_y, QW, QW, QUERY, LW_HI)
    sl.text(QX - 0.3, i_y + QW + 0.04, QW + 0.6, 0.14, f"intermediate ({factor_label(qa['intermediate']['factor'])})",
            FS_MIN, color=QUERY, bold=True, align=PP_ALIGN.CENTER)
    mean = sl.box(1.66, CY - 0.19, 0.70, 0.38, BOX_BG, BOX_BORDER, "mean\vembedding", size=FS_MIN, lw=LW_BOX)
    q = sl.box(2.44, CY - 0.17, 0.34, 0.34, Q_BG, Q_BORDER, "q", size=FS_Q, bold=True, italic=True,
               shape=MSO_SHAPE.OVAL, lw=LW_BOX)
    sl.arrow((det, 3), (tight, 1), lw=LW_BOX)
    sl.arrow((det, 3), (inter, 1), lw=LW_BOX)
    sl.arrow((tight, 2), (v_adj, 0), lw=LW_TILE)
    sl.arrow((inter, 0), (v_int, 2), lw=LW_TILE)
    sl.arrow((v_adj, 3), (mean, 1), lw=LW_BOX)
    sl.arrow((v_int, 3), (mean, 1), lw=LW_BOX)
    sl.arrow((mean, 3), (q, 2), lw=LW_BOX)

    # ---- 2. Identity bank ----------------------------------------------------
    TILE, PITCH, S_H, B_W = 0.38, 0.43, 0.055, 0.28
    BANK_X, BANK_Y, BANK_W = 2.88, 0.24, 2.48
    row_x = BANK_X + 0.10
    by_pid = {e["pid"]: e for e in case["entries"]}
    rows = sorted((by_pid[p] for p in case["rows_shown"]), key=lambda e: e["sim"])  # la mejor abajo
    n_views_max = max(len(e["views"]) for e in rows)
    ROW_PITCH = 0.62
    mid = (len(rows) - 1) / 2
    row_ys = [CY - TILE / 2 + (r - mid) * ROW_PITCH for r in range(len(rows))]  # fila central a la altura de q
    head_y = row_ys[0] - 0.17
    bank_h = row_ys[-1] + TILE + 0.04 + S_H + 0.08 - BANK_Y
    sl.text(BANK_X, TY, BANK_W, 0.18, "2. Identity bank (recently-lost tracks)", FS_TITLE, bold=True)
    bank = sl.box(BANK_X, BANK_Y, BANK_W, bank_h, BANK_BG, BANK_BORDER, lw=LW_BANK)
    bank.adjustments[0] = 0.08
    for k in range(n_views_max):  # escalas: una sola cabecera por columna
        f = rows[-1]["views"][k]["factor"] if k < len(rows[-1]["views"]) else None
        if f is not None:
            sl.text(row_x + k * PITCH - 0.03, head_y, TILE + 0.06, 0.14, factor_label(f), FS_MIN, color=MUTED,
                    align=PP_ALIGN.CENTER)
    b_x = row_x + (n_views_max - 1) * PITCH + TILE + 0.16
    q_idx = {1: [6], 2: [7, 5], 3: [7, 6, 5]}.get(len(rows), [6] * len(rows))
    best_b = None
    for r, (e, ry) in enumerate(zip(rows, row_ys)):
        is_best = e["pid"] == case["best_pid"]
        sy = ry + TILE + 0.04
        views_end = row_x + (len(e["views"]) - 1) * PITCH + TILE
        group = sl.box(row_x - 0.025, sy - 0.02, views_end - row_x + 0.05, S_H + 0.04, BANK_BG, MUTED, lw=0.5,
                       shape=MSO_SHAPE.RECTANGLE)
        group.line.dash_style = 4  # MSO_LINE.DASH
        first = None
        for k, v in enumerate(e["views"]):
            x = row_x + k * PITCH
            pic = sl.image(v["path"], x, ry, TILE, TILE, TILE_BORDER, LW_TILE)
            sl.strip(x, sy, TILE, S_H, seed=f"e_{e['pid']}_{k}", lw=LW_TILE)
            if k == 0:
                first = pic
        b = sl.strip(b_x, sy - 0.01, B_W, S_H + 0.02, seed=f"b_{e['pid']}",
                     border=hi if is_best else Q_BORDER, lw=LW_HI if is_best else LW_BOX)
        sl.arrow((group, 3), (b, 1), lw=LW_TILE)
        sl.text(b_x - 0.10, ry + TILE / 2 - 0.09, 0.62, 0.14, f"s = {e['sim']:.4f}", FS_MIN,
                color=hi if is_best else MUTED, bold=is_best, anchor=MSO_ANCHOR.MIDDLE)
        sl.arrow((q, q_idx[r]), (first, 1), color=hi if is_best else MUTED, lw=LW_HI if is_best else LW_BOX)
        if is_best:
            best_b = b

    # ---- 3. Re-identification decision (piso inferior) -----------------------
    # s >= tau centrado bajo la tira b_j del mejor candidato (flecha vertical
    # recta); las dos salidas a su izquierda, MATCH arriba y new identity
    # abajo, cada una con una flecha recta desde el lado izquierdo de tau.
    BW, BH = 0.85, 0.40
    OW, OH = 0.78, 0.38
    b_cx = best_b.left / 914400 + best_b.width / 914400 / 2
    tau_x = b_cx - BW / 2
    MATCH_Y = BANK_Y + bank_h + 0.08
    tau_y = MATCH_Y + OH / 2 + 0.10
    no_y = tau_y + BH / 2 + 0.10
    out_x = tau_x - 0.40 - OW
    tau = sl.box(tau_x, tau_y, BW, BH, BANK_BG, BANK_BORDER, f"s \u2265 \u03c4 ?\v\u03c4 = {case['threshold']}",
                 size=FS_BOX, lw=LW_BOX)
    yes = sl.box(out_x, MATCH_Y, OW, OH, MATCH_BG, MATCH, "MATCH:\vmerge ID", size=FS_BOX, color=MATCH,
                 bold=True, lw=LW_HI if matched else LW_BOX)
    no = sl.box(out_x, no_y, OW, OH, REJECT_BG, REJECT, "below \u03c4:\vnew identity", size=FS_BOX,
                color=REJECT, bold=True, lw=LW_BOX if matched else LW_HI)
    sl.arrow((best_b, 2), (tau, 0), color=hi, lw=LW_HI)
    sl.arrow((tau, 1), (yes, 3), color=MATCH, lw=LW_HI if matched else LW_BOX)
    sl.arrow((tau, 1), (no, 3), color=REJECT, lw=LW_BOX if matched else LW_HI)
    sl.text(out_x + OW + 0.05, MATCH_Y + 0.02, 0.3, 0.14, "yes", FS_MIN, color=MATCH, italic=True)
    sl.text(out_x + OW + 0.05, no_y + OH - 0.16, 0.3, 0.14, "no", FS_MIN, color=REJECT, italic=True)


def gallery_rows(case, n_rows):
    """Las `n_rows` entradas de mayor similitud (los candidatos más difíciles
    del banco real), con la verdadera siempre incluida -- mismo criterio que
    paper_fig_reid_module_diagram.py::select_rows --, ordenadas por su
    primera aparición en el video (orden natural del banco, sin delatar cuál
    es el match por la posición)."""
    rows = case["entries"][:n_rows]
    if not any(e["is_true"] for e in rows):
        rows = rows[:-1] + [next(e for e in case["entries"] if e["is_true"])]
    return sorted(rows, key=lambda e: e["views"][0]["frame"])


def _draw_gallery(sl, case, n_rows):
    """fig:batch -- cómo se construyen los dos conjuntos: cada fila es una
    entrada del banco (hasta 4 vistas repartidas en su historial, con su
    escala -- una cabecera por columna -- y su embedding, promediados en b_j)
    y a la derecha las 2 vistas de la query (solo las imágenes: su fusión en
    q se muestra en la figura "decision"). Sin similitudes ni decisión. La
    entrada verdadera se marca como MATCH."""
    rows = gallery_rows(case, n_rows)
    TILE, PITCH, S_H, B_W = 0.52, 0.58, 0.07, 0.32
    X0, ROW_PITCH, ROW_H = 0.05, 0.84, 0.76
    tile_x0 = X0 + 0.55
    n_views_max = max(len(e["views"]) for e in rows)
    b_x = tile_x0 + (n_views_max - 1) * PITCH + TILE + 0.16
    ROW_W = b_x + B_W + 0.10 - X0
    HEAD_Y = 0.30
    Y0 = HEAD_Y + 0.18
    sl.prs.slide_width = Inches(PRINT_W + 0.2)
    sl.prs.slide_height = Inches(Y0 + len(rows) * ROW_PITCH + 0.2)

    sl.text(X0, 0.05, ROW_W, 0.18, "Identity bank (recently-lost tracks)", FS_TITLE, bold=True)
    ref = max(rows, key=lambda e: len(e["views"]))
    for k, v in enumerate(ref["views"]):  # escalas: una sola cabecera por columna
        sl.text(tile_x0 + k * PITCH - 0.03, HEAD_Y, TILE + 0.06, 0.14, factor_label(v["factor"]), FS_MIN,
                color=MUTED, align=PP_ALIGN.CENTER)
    for r, e in enumerate(rows):
        ry = Y0 + r * ROW_PITCH
        is_true = e["is_true"]
        frame = sl.box(X0, ry, ROW_W, ROW_H, "FFFFFF", MATCH if is_true else BOX_BORDER,
                       lw=LW_HI if is_true else LW_TILE)
        frame.adjustments[0] = 0.10
        sl.text(X0 + 0.07, ry + ROW_H / 2 - 0.09, 0.45, 0.18, f"track {chr(ord('A') + r)}", FS_BOX,
                color=MATCH if is_true else INK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        ty = ry + 0.05
        sy = ty + TILE + 0.05
        views_end = tile_x0 + (len(e["views"]) - 1) * PITCH + TILE
        group = sl.box(tile_x0 - 0.025, sy - 0.02, views_end - tile_x0 + 0.05, S_H + 0.04, "FFFFFF", MUTED,
                       lw=0.5, shape=MSO_SHAPE.RECTANGLE)
        group.line.dash_style = 4  # MSO_LINE.DASH
        for k, v in enumerate(e["views"]):
            x = tile_x0 + k * PITCH
            sl.image(v["path"], x, ty, TILE, TILE, TILE_BORDER, LW_TILE)
            sl.strip(x, sy, TILE, S_H, seed=f"e_{e['pid']}_{k}", lw=LW_TILE)
        b = sl.strip(b_x, sy - 0.01, B_W, S_H + 0.02, seed=f"b_{e['pid']}",
                     border=MATCH if is_true else Q_BORDER, lw=LW_HI if is_true else LW_BOX)
        sl.arrow((group, 3), (b, 1), lw=LW_TILE)
        if is_true:
            sl.text(b_x - 0.25, ty + 0.02, B_W + 0.30, 0.15, "MATCH", FS_MIN, color=MATCH, bold=True,
                    align=PP_ALIGN.RIGHT)

    # ---- query --------------------------------------------------------------
    qa = case["query_assets"]
    QW = 0.95
    QX = X0 + ROW_W + 0.25 + (PRINT_W - (X0 + ROW_W + 0.25) - QW) / 2
    sl.text(X0 + ROW_W + 0.2, 0.05, PRINT_W - ROW_W - 0.15, 0.18, "Query (reappeared detection)", FS_TITLE,
            bold=True, align=PP_ALIGN.CENTER)
    bank_mid = Y0 + (len(rows) * ROW_PITCH - (ROW_PITCH - ROW_H)) / 2
    t_y, i_y = bank_mid - 0.20 - QW, bank_mid + 0.30
    sl.text(QX - 0.3, t_y - 0.18, QW + 0.6, 0.14, f"tight ({factor_label(qa['tight']['factor'])})", FS_MIN,
            color=QUERY, bold=True, align=PP_ALIGN.CENTER)
    sl.image(qa["tight"]["path"], QX, t_y, QW, QW, QUERY, LW_HI)
    sl.text(QX - 0.3, i_y - 0.18, QW + 0.6, 0.14, f"intermediate ({factor_label(qa['intermediate']['factor'])})",
            FS_MIN, color=QUERY, bold=True, align=PP_ALIGN.CENTER)
    sl.image(qa["intermediate"]["path"], QX, i_y, QW, QW, QUERY, LW_HI)


# --------------------------------------------------------------------------- #
# Exportación a PDF/PNG (LibreOffice remoto + Carlito)
# --------------------------------------------------------------------------- #

REMOTE_DIR = "/tmp/reid_fig_export"
LO_PDF_PREFS = {"ReduceImageResolution": "false", "UseLosslessCompression": "true",
                "Quality": "100", "EmbedStandardFonts": "true"}


def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _remote_setup(host, fonts_dir):
    _run(["ssh", host, f"mkdir -p {REMOTE_DIR}/home/.fonts {REMOTE_DIR}/out"])
    fonts = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.lower().endswith(".ttf")]
    if not fonts:
        raise SystemExit(f"[ERROR] no hay .ttf en {fonts_dir} (se espera Carlito)")
    _run(["scp", "-q", *fonts, f"{host}:{REMOTE_DIR}/home/.fonts/"])
    items = "".join(
        f'<item oor:path="/org.openoffice.Office.Common/Filter/PDF/Export"><prop oor:name="{k}" oor:op="fuse">'
        f"<value>{v}</value></prop></item>" for k, v in LO_PDF_PREFS.items())
    # el perfil se crea en la primera ejecución de soffice; después se le
    # inyectan las preferencias de exportación (idempotente)
    script = f"""
set -e
export HOME={REMOTE_DIR}/home
fc-cache -f $HOME/.fonts >/dev/null 2>&1 || true
REG=$HOME/.config/libreoffice/4/user/registrymodifications.xcu
[ -f $REG ] || timeout 120 soffice --headless --terminate_after_init >/dev/null 2>&1 || true
grep -q UseLosslessCompression $REG || python3 - "$REG" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
open(p, "w").write(s.replace("</oor:items>", {items!r} + "</oor:items>"))
PY
"""
    _run(["ssh", host, "bash -s"], input=script)


def _remote_convert(host, pptx_path, pdf_path):
    name = os.path.basename(pptx_path)
    _run(["scp", "-q", pptx_path, f"{host}:{REMOTE_DIR}/{name}"])
    _run(["ssh", host, f"cd {REMOTE_DIR} && rm -f out/* && HOME={REMOTE_DIR}/home timeout 180 "
                       f"soffice --headless --convert-to pdf --outdir out {name}"])
    _run(["scp", "-q", f"{host}:{REMOTE_DIR}/out/{os.path.splitext(name)[0]}.pdf", pdf_path])


def _render_png(pdf_path, png_path, dpi):
    _run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", pdf_path, os.path.splitext(png_path)[0]])


def crop_pptx(src_path, dst_path, crop):
    """Copia de `src_path` con todo desplazado y la diapositiva ajustada a
    `crop` (pulgadas) -- trabaja sobre el .pptx tal como está en disco, así
    que respeta las ediciones hechas a mano en PowerPoint/Keynote."""
    prs = Presentation(src_path)
    x0, y0, x1, y1 = (Inches(v) for v in crop)
    for shape in prs.slides[0].shapes:
        shape.left, shape.top = shape.left - x0, shape.top - y0
    prs.slide_width, prs.slide_height = Emu(x1 - x0), Emu(y1 - y0)
    prs.save(dst_path)


def export(out_path, host, fonts_dir, margin_in=0.03, dpi=300):
    """<out>.pdf y <out>.png a partir del .pptx `out_path` tal como está."""
    _remote_setup(host, fonts_dir)
    stem = os.path.splitext(out_path)[0]
    with tempfile.TemporaryDirectory() as tmp:
        # 1) diapositiva completa -> caja de tinta
        full_pptx = os.path.join(tmp, "full.pptx")
        shutil.copy(out_path, full_pptx)
        _remote_convert(host, full_pptx, os.path.join(tmp, "full.pdf"))
        probe_dpi = 150
        _render_png(os.path.join(tmp, "full.pdf"), os.path.join(tmp, "full.png"), probe_dpi)
        im = Image.open(os.path.join(tmp, "full.png")).convert("RGB")
        bx0, by0, bx1, by1 = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
        src = Presentation(out_path)
        W_in, H_in = src.slide_width / 914400, src.slide_height / 914400
        crop = (max(0.0, bx0 / probe_dpi - margin_in), max(0.0, by0 / probe_dpi - margin_in),
                min(W_in, bx1 / probe_dpi + margin_in), min(H_in, by1 / probe_dpi + margin_in))
        # 2) diapositiva ajustada al contenido -> PDF final + PNG
        tight_pptx = os.path.join(tmp, "tight.pptx")
        crop_pptx(out_path, tight_pptx, crop)
        _remote_convert(host, tight_pptx, f"{stem}.pdf")
    _render_png(f"{stem}.pdf", f"{stem}.png", dpi)
    print(f"[OK] {stem}.pdf  {stem}.png ({dpi} dpi)  recorte (in): {tuple(round(v, 2) for v in crop)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-json", default="paper/training_methodology/reid_module_diagram_measured.json")
    parser.add_argument("--layout", choices=["decision", "gallery"], default="decision",
                        help="decision: figura-resumen del módulo; gallery: construcción del banco/query (fig:batch)")
    parser.add_argument("--n-rows", type=int, default=3,
                        help="(gallery) entradas del banco a mostrar -- 3 = las mismas que la figura \"decision\"")
    parser.add_argument("--out", default=None,
                        help="default: paper/training_methodology/{reid_module_diagram,bank_query_diagram}_measured.pptx")
    parser.add_argument("--export", action="store_true", help="genera también <out>.pdf y <out>.png")
    parser.add_argument("--export-only", action="store_true",
                        help="NO regenera el .pptx: exporta <out> tal como está (p.ej. tras editarlo a mano)")
    parser.add_argument("--export-host", default=os.environ.get("EXPORT_HOST", "localhost"), help="host con LibreOffice (vía ssh)")
    parser.add_argument("--fonts-dir", default="paper/training_methodology/_fonts",
                        help="carpeta con los .ttf de Carlito")
    parser.add_argument("--dpi", type=int, default=300, help="resolución del .png")
    args = parser.parse_args()
    with open(args.case_json) as f:
        case = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(args.case_json))
    if args.out is None:
        name = "bank_query_diagram" if args.layout == "gallery" else "reid_module_diagram"
        args.out = os.path.join(base_dir, f"{name}_measured.pptx")
    if not args.export_only:
        build(case, base_dir, args.out, layout=args.layout, n_rows=args.n_rows)
        print(f"[OK] {args.out}")
    if args.export or args.export_only:
        export(args.out, args.export_host, args.fonts_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
