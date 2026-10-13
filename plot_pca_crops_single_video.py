"""
plot_pca_crops_single_video.py

Versión más intuitiva de plot_pca_single_video (plot_embedding_compression.py
-- la base de compression_pca_single_video.png en paper/embedding_analisis/).
El problema de la original (ver conversación): son puntos de color abstractos
sin contexto -- no se reconoce a simple vista que son recortes de daños de
vehículo, y la comparación de tamaño entre encoders (la idea central: C4
comprime más, nube más chica) depende de comparar el número "total variance
(trace)" en el título de 3 paneles separados, en letra chica.

Mejoras sobre el original, ejes compartidos igual que antes (para que
"nube más chica" se vea directamente, sin normalizar por panel):
  1. Recortes reales (mismo criterio TIGHT_FACTOR que el resto de scripts de
     esta familia), marco de color = identidad (nipy_spectral, igual que
     _single_video_colors) -- el fondo de puntos de color se mantiene para
     preservar la forma/densidad completa de la nube (35 identidades).
  2. Círculo que encierra la nube completa (radio = distancia máxima al
     centroide de TODOS los puntos, no solo los que llevan recorte) + el
     valor de "total variance (trace)" anotado en grande sobre el plot mismo
     -- hace evidente "esta nube es más chica" sin tener que leer el título.
  3. Con 35 identidades enteras los recortes se solapaban entre sí y la
     figura quedaba ilegible (ver conversación) -- ahora solo se pone recorte
     encima de una selección CURADA y chica de identidades (no un subsample
     al azar): los pares con errores reales de rank-1 en C4 encontrados por
     investigate_c4_video6_confusion.py (CONFUSABLE_TRACK_IDS_VIDEO6) + unas
     pocas de contraste sin errores conocidos (--n-contrast-identities). El
     resto de las 35 sigue en el fondo como punto de color, sin recorte.

Con --print-width W (pulgadas) dibuja la figura A TAMAÑO DE IMPRESIÓN (el
paper la incluye a linewidth = 5.4 in en elsarticle preprint): dimensiones
y fuentes reales (>= --font-size pt), recortes en la MISMA proporción
respecto del panel que la versión normal (mismos solapes), y guarda
pca_crops_single_video_print.pdf (texto vectorial) + .png (600 dpi). Mismos
datos, semilla y PCA -- solo cambia el dibujo. Sin --print-width, la salida
es la de siempre (21x7 in, que el paper reducía al 0.26: títulos a 3.1 pt,
ejes a 2.6 pt).

Uso (correr en el servidor, mismos caches que plot_embedding_compression.py):
    python3 plot_pca_crops_single_video.py --kfold-pool \
        --p4-cache p4_roi_cache_kfold_pool.pt --p4fpn-cache p4fpn_roi_cache_kfold_pool.pt \
        --output-dir compression_analysis_kfoldpool
"""
import argparse
import os
import random

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle

from embedding_analysis_dinov2_vs_p4 import TIGHT_FACTOR, _video_of, compute_cache_embeddings, compute_dinov2_embeddings, sample_identities
from plot_embedding_compression import _single_video_colors
from reid_dataloader import _expand_bbox
from train_track_query_reid import build_track_index, resolve_sequence_dirs

# track_ids de video6 con errores REALES de rank-1 en C4 (ver
# investigate_c4_video6_confusion.py: pares 25<->26, 11<->12, 28<->3, 10<->12,
# todos con n_errores_rank1>=1) -- mostrarlos acá conecta esta figura con esa
# investigación en vez de un subsample al azar sin motivo.
CONFUSABLE_TRACK_IDS_VIDEO6 = ["25", "26", "11", "12", "28", "3", "10"]


def _load_crop(obs, thumb_size):
    """Recorte AJUSTADO (factor=1.0, TIGHT_FACTOR) -- mismo criterio que el
    resto de scripts de esta familia (plot_embedding_crops_single_video.py,
    plot_embedding_crops_by_identity.py)."""
    image = Image.open(obs.image_path).convert("RGB")
    crop = image.crop(_expand_bbox(obs.bbox, TIGHT_FACTOR, image.size))
    crop.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
    return np.asarray(crop)


PRINT_TITLES = {"P4 (FPN C5+C4)": "P4-FPN (C5+C4)"}
PRINT_THUMB_PX = 160  # resolución de carga de cada recorte en modo impresión


def plot_pca_crops_single_video(sources, samples, identities, video, seed, crops_per_identity, n_contrast_identities, thumb_size, output_dir,
                                print_width=None, font_size=8.0, font_dir=None):
    """`print_width` (pulgadas, opcional): ver docstring del módulo. `font_dir`
    (opcional): carpeta con .ttf de Carlito (misma fuente que el resto de
    figuras del paper)."""
    printing = print_width is not None
    if printing:
        plt.rcParams.update({"pdf.fonttype": 42, "font.size": font_size, "axes.linewidth": 0.5,
                             "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                             "xtick.major.size": 2, "ytick.major.size": 2})
        if font_dir:
            from matplotlib import font_manager
            for f in sorted(os.listdir(font_dir)):
                if f.lower().endswith(".ttf"):
                    font_manager.fontManager.addfont(os.path.join(font_dir, f))
            plt.rcParams["font.family"] = "Carlito"
    mask, colors, unique_ids = _single_video_colors(identities, video)
    sub_samples = [s for s, keep in zip(samples, mask) if keep]
    sub_identities = [ident for ident, keep in zip(identities, mask) if keep]
    ident_color = dict(zip(sub_identities, colors))

    # Identidades a mostrar como recorte -- curadas, no un subsample al azar de las
    # 35: los pares confusos de C4 (CONFUSABLE_TRACK_IDS_VIDEO6, ver
    # investigate_c4_video6_confusion.py) + unas pocas de contraste (identidades
    # SIN errores conocidos, para comparar visualmente contra las confusas). El
    # resto de las 35 sigue en el fondo como punto de color (preserva la nube
    # completa), pero no se le pone recorte encima -- con 35 identidades enteras
    # los recortes se solapaban entre sí y la figura quedaba ilegible.
    rng = random.Random(seed)
    unique_sub_identities = sorted(set(sub_identities))
    confusable_present = [i for i in unique_sub_identities if i.split("__")[-1] in CONFUSABLE_TRACK_IDS_VIDEO6]
    remaining = [i for i in unique_sub_identities if i not in confusable_present]
    contrast_present = rng.sample(remaining, min(n_contrast_identities, len(remaining)))
    display_identities = set(confusable_present + contrast_present)
    print(f"[INFO] Identidades mostradas con recorte: {len(display_identities)} ({len(confusable_present)} confusas + {len(contrast_present)} de contraste) de {len(unique_sub_identities)} totales")
    # Máscara para el scatter de fondo -- SOLO estas identidades, no las 35
    # (el resto queda oculto; el círculo/trace sigue calculado sobre la nube
    # COMPLETA de video6, ver más abajo, para no invalidar esa comparación).
    display_mask = np.array([ident in display_identities for ident in sub_identities])
    display_colors = np.asarray(colors)[display_mask]

    by_identity = {}
    for i, ident in enumerate(sub_identities):
        if ident in display_identities:
            by_identity.setdefault(ident, []).append(i)
    display_idx = set()
    for idxs in by_identity.values():
        display_idx.update(rng.sample(idxs, min(crops_per_identity, len(idxs))))

    pca_results = []
    for name, embeds in sources:
        sub = embeds[mask]
        norm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
        n_components = min(50, norm.shape[0] - 1, norm.shape[1])
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(norm)
        total_var = float(norm.var(axis=0).sum())
        pca_results.append((name, proj, total_var))

    # ejes compartidos entre los 3 paneles -- la comparación de tamaño depende de esto
    # (ver plot_pca_compression/plot_pca_single_video, mismo criterio).
    all_xy = np.concatenate([proj[:, :2] for _, proj, _ in pca_results], axis=0)
    lim = float(np.abs(all_xy).max()) * 1.05

    if printing:
        n = len(pca_results)
        left_in, right_in, wspace_in = 0.40, 0.03, 0.08
        title_in, bottom_in = 0.20, 0.32
        panel_in = (print_width - left_in - right_in - (n - 1) * wspace_in) / n
        fig_h = title_in + panel_in + bottom_in
        fig = plt.figure(figsize=(print_width, fig_h))
        axes = [fig.add_axes([(left_in + i * (panel_in + wspace_in)) / print_width, bottom_in / fig_h,
                              panel_in / print_width, panel_in / fig_h]) for i in range(n)]
        # misma proporción recorte/panel que la versión normal (thumb_size px a
        # 150 dpi sobre un panel de ~7 in) -> mismos solapes
        thumb_pt = thumb_size / (7 * 72) * (panel_in * 72)
    else:
        fig, axes = plt.subplots(1, len(pca_results), figsize=(7 * len(pca_results), 7))
        if len(pca_results) == 1:
            axes = [axes]
    for ax, (name, proj, total_var) in zip(axes, pca_results):
        # Círculo que encierra TODA la nube (radio = distancia máxima al centroide,
        # sobre todos los puntos, no solo la submuestra con recorte) -- hace visible
        # de un vistazo qué tan comprimido es el espacio, sin leer el título.
        centroid = proj[:, :2].mean(axis=0)
        radius = float(np.linalg.norm(proj[:, :2] - centroid, axis=1).max())
        ax.add_patch(Circle(centroid, radius, fill=False, edgecolor="black", linewidth=0.8 if printing else 1.8,
                            linestyle="--", zorder=2, alpha=0.6))
        # Posición FIJA en coordenadas de ejes (no relativa al círculo) -- con el
        # círculo puede acercarse al borde superior del panel y chocar con el
        # título; anclar a la esquina evita el solape sin importar el radio.
        ax.text(
            0.03, 0.97, f"trace = {total_var:.4f}" if printing else f"total variance\n(trace) = {total_var:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=font_size if printing else 13, fontweight="bold",
            zorder=4, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1 if printing else 3),
        )

        ax.scatter(proj[display_mask, 0], proj[display_mask, 1], s=3 if printing else 16, alpha=0.55, c=display_colors,
                   edgecolors="none", zorder=1)
        for i in display_idx:
            ident = sub_identities[i]
            _, obs = sub_samples[i]
            if printing:
                crop = _load_crop(obs, PRINT_THUMB_PX)
                imagebox = OffsetImage(crop, zoom=thumb_pt / max(crop.shape[:2]))
            else:
                crop = _load_crop(obs, thumb_size)
                imagebox = OffsetImage(crop, zoom=1.0)
            ab = AnnotationBbox(
                imagebox, (proj[i, 0], proj[i, 1]), frameon=True, pad=0.05,
                bboxprops=dict(edgecolor=ident_color[ident], linewidth=0.7 if printing else 2.2), zorder=3,
            )
            ax.add_artist(ab)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        if printing:
            # paso uniforme "redondo" (~2 marcas por semieje) -- redondear posiciones
            # de linspace daba marcas desiguales (-0.5, -0.2, 0, 0.2, 0.5)
            step = min((0.1, 0.2, 0.25, 0.3, 0.5, 1.0), key=lambda st: abs(lim / st - 2.2))
            k = int(np.floor(lim / step + 1e-9))
            ticks = np.round(np.arange(-k, k + 1) * step, 2)
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.tick_params(labelsize=font_size, pad=1.5)
            ax.set_title(PRINT_TITLES.get(name, name), fontsize=font_size + 1, fontweight="bold", pad=3)
            ax.set_xlabel("PC1", fontsize=font_size, labelpad=1)
            if ax is axes[0]:
                ax.set_ylabel("PC2", fontsize=font_size, labelpad=1)
            else:
                ax.set_yticklabels([])
        else:
            ax.set_title(f"{name} -- PCA, single video ({video})\n{len(unique_ids)} identities")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
    if printing:
        for ext in ("pdf", "png"):
            path = os.path.join(output_dir, f"pca_crops_single_video_print.{ext}")
            fig.savefig(path, dpi=600)
            print(f"[OK] {path}")
        plt.close(fig)
        return
    fig.tight_layout()
    path = os.path.join(output_dir, "pca_crops_single_video.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--kfold-pool", action="store_true",
        help="Ignora --split -- usa el pool de 64 secuencias de --kfold-cv, mismo criterio que "
             "embedding_analysis_dinov2_vs_p4.py/plot_embedding_compression.py.",
    )
    parser.add_argument("--p4-cache", required=True, help="PrecomputedEmbeddingCache de C4 crudo (--layer c4).")
    parser.add_argument("--p4fpn-cache", default=None, help="[opcional] PrecomputedEmbeddingCache de P4 REAL (--layer p4fpn).")
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument("--num-identities", type=int, default=1000, help="Alto por default -- se cae solo al máximo disponible en el pool.")
    parser.add_argument("--obs-per-identity", type=int, default=15)
    parser.add_argument(
        "--crops-per-identity", type=int, default=4,
        help="Cuántos recortes reales mostrar por CADA identidad elegida (ver --n-contrast-identities) -- "
             "más de 1 para poder ver si los recortes de una MISMA identidad quedan agrupados entre sí.",
    )
    parser.add_argument(
        "--n-contrast-identities", type=int, default=4,
        help="Cuántas identidades SIN errores conocidos agregar como contraste visual, además de las "
             "confusas de CONFUSABLE_TRACK_IDS_VIDEO6 (ver investigate_c4_video6_confusion.py).",
    )
    parser.add_argument("--thumb-size", type=int, default=42, help="Tamaño (px, lado mayor) de cada thumbnail -- bajado de 55 a 42 ahora que hay varios recortes por identidad.")
    parser.add_argument("--print-width", type=float, default=None,
                        help="(pulgadas) dibuja a tamaño de impresión -- ver docstring. Paper: 5.4")
    parser.add_argument("--font-size", type=float, default=8.0, help="(con --print-width) tamaño mínimo de letra, pt")
    parser.add_argument("--font-dir", default=None, help="(con --print-width) carpeta con los .ttf de Carlito")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-wide-factor", type=float, default=9.0, help="Mismo default que embedding_analysis_dinov2_vs_p4.py.")
    parser.add_argument("--output-dir", default="compression_analysis")
    parser.add_argument(
        "--video", default=None,
        help="Video a usar -- default: el que más identidades aporta al pool muestreado "
             "(mismo criterio que --single-video-pca en plot_embedding_compression.py).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.kfold_pool:
        from kfold_utils import list_all_sequences
        seq_dirs = list_all_sequences()
        print(f"[INFO] --kfold-pool: {len(seq_dirs)} secuencias")
        index = build_track_index(None, sequence_dirs=seq_dirs)
    else:
        data_root, seq_dirs = resolve_sequence_dirs(None, args.split)
        index = build_track_index(data_root, sequence_dirs=seq_dirs)

    samples = sample_identities(index, args.num_identities, args.obs_per_identity, args.seed)
    identities = [ident for ident, _ in samples]
    print(f"[INFO] {len(samples)} observaciones de {len(set(identities))} identidades")

    dinov2_embeds = compute_dinov2_embeddings(samples, args.dinov2_model_name, args.device, args.query_wide_factor)
    c4_embeds = compute_cache_embeddings(samples, args.p4_cache)
    sources = [("DINOv2", dinov2_embeds), ("C4", c4_embeds)]
    if args.p4fpn_cache:
        sources.append(("P4 (FPN C5+C4)", compute_cache_embeddings(samples, args.p4fpn_cache)))

    if args.video:
        video = args.video
    else:
        by_video = {}
        for ident in identities:
            by_video.setdefault(_video_of(ident), []).append(ident)
        video = max(by_video, key=lambda v: len(set(by_video[v])))
    print(f"[INFO] video = {video}")

    plot_pca_crops_single_video(sources, samples, identities, video, args.seed, args.crops_per_identity, args.n_contrast_identities, args.thumb_size, args.output_dir,
                                print_width=args.print_width, font_size=args.font_size, font_dir=args.font_dir)


if __name__ == "__main__":
    main()
