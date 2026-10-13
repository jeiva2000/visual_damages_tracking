"""
plot_embedding_crops_by_identity.py

Versión con recortes reales de `plot_nonlinear_projections` (la función de
plot_embedding_compression.py que genera tsne_by_identity.png -- la base de
compression_tsne_by_identity_kfoldpool.png en paper/embedding_analisis/) --
mismo propósito que plot_embedding_crops_single_video.py (ver conversación:
ver los recortes reales en vez de puntos de color es más intuitivo para
juzgar agrupamiento a simple vista) pero acá sobre la submuestra MULTI-VIDEO
de --tsne-identities (default 20, concentrada en pocos videos -- ver
_pick_display_identities), no un solo video.

Mismo esquema de color que _pick_display_identities: paleta secuencial
distinta por VIDEO, tono distinto por IDENTIDAD dentro de ese video (mismo
tono = misma identidad = mismo video) -- el marco de cada recorte usa ese
color, así se puede leer agrupamiento por color Y por contenido visual del
recorte al mismo tiempo.

Igual que en plot_embedding_crops_single_video.py, mostrar TODAS las
observaciones de las identidades elegidas como thumbnail satura la figura
-- se subsamplea a --crops-per-identity recortes reales por identidad; el
resto de las observaciones de esas mismas identidades queda como punto gris
de fondo (preserva la estructura completa del t-SNE de esa submuestra).

Con --print-width W (pulgadas) dibuja la figura A TAMAÑO DE IMPRESIÓN (el
paper la incluye a linewidth = 5.4 in en elsarticle preprint): dimensiones
y fuentes reales (>= --font-size pt), leyenda fuera de los paneles, y guarda
tsne_by_identity_crops_print.pdf (texto vectorial) + .png (600 dpi). Mismos
datos, semilla y t-SNE que la versión normal -- solo cambia el dibujo. Sin
--print-width, la salida es la de siempre (19.5x6.5 in, que el paper
reducía al 0.28: títulos a 3.3 pt, números de identidad a 1.9 pt).

Uso (correr en el servidor, mismos caches que plot_embedding_compression.py):
    python3 plot_embedding_crops_by_identity.py --kfold-pool \
        --p4-cache p4_roi_cache_kfold_pool.pt --p4fpn-cache p4fpn_roi_cache_kfold_pool.pt \
        --output-dir compression_analysis_kfoldpool
"""
import argparse
import os
import random

import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from embedding_analysis_dinov2_vs_p4 import TIGHT_FACTOR, compute_cache_embeddings, compute_dinov2_embeddings, sample_identities
from plot_embedding_compression import _pick_display_identities
from reid_dataloader import _expand_bbox
from train_track_query_reid import build_track_index, resolve_sequence_dirs


def _load_crop(obs, thumb_size):
    """Recorte AJUSTADO (factor=1.0, TIGHT_FACTOR) -- el daño puro, igual
    criterio que plot_embedding_crops_single_video.py. `thumb_size`: lado
    mayor en píxeles."""
    image = Image.open(obs.image_path).convert("RGB")
    crop = image.crop(_expand_bbox(obs.bbox, TIGHT_FACTOR, image.size))
    crop.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
    return np.asarray(crop)


PRINT_TITLES = {"P4 (FPN C5+C4)": "P4-FPN (C5+C4)"}
PRINT_THUMB_PX = 160  # resolución de carga de cada recorte en modo impresión (se muestra a --thumb-pt)


def plot_crops_by_identity(sources, samples, identities, display_ids, color_map, video_legend, seed, crops_per_identity, thumb_size, output_dir,
                           print_width=None, font_size=8.0, thumb_pt=16.0, font_dir=None):
    """`print_width` (pulgadas, opcional): ver docstring del módulo. `font_dir`
    (opcional): carpeta con .ttf de Carlito, para usar la misma fuente que el
    resto de figuras del paper (clon métrico de Calibri, más estrecha que la
    DejaVu Sans por defecto)."""
    printing = print_width is not None
    if printing and font_dir:
        from matplotlib import font_manager
        for f in sorted(os.listdir(font_dir)):
            if f.lower().endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(font_dir, f))
        plt.rcParams["font.family"] = "Carlito"
    mask = np.array([ident in display_ids for ident in identities])
    sub_samples = [s for s, keep in zip(samples, mask) if keep]
    sub_identities = [ident for ident, keep in zip(identities, mask) if keep]

    # Subsample por identidad para mostrar como recorte real -- el resto de sus
    # observaciones queda como punto gris de fondo (ver docstring del módulo).
    rng = random.Random(seed)
    by_identity = {}
    for i, ident in enumerate(sub_identities):
        by_identity.setdefault(ident, []).append(i)
    display_idx = set()
    for idxs in by_identity.values():
        display_idx.update(rng.sample(idxs, min(crops_per_identity, len(idxs))))

    if printing:
        plt.rcParams.update({"pdf.fonttype": 42, "font.size": font_size})
        n = len(sources)
        wspace_in, side_in = 0.06, 0.02
        panel_in = (print_width - 2 * side_in - (n - 1) * wspace_in) / n
        title_in, legend_in = 0.22, 0.22
        fig_h = title_in + panel_in + legend_in
        fig = plt.figure(figsize=(print_width, fig_h))
        axes = [fig.add_axes([(side_in + i * (panel_in + wspace_in)) / print_width, legend_in / fig_h,
                              panel_in / print_width, panel_in / fig_h]) for i in range(n)]
    else:
        fig, axes = plt.subplots(1, len(sources), figsize=(6.5 * len(sources), 6.5))
        if len(sources) == 1:
            axes = [axes]
    for col, (ax, (name, embeds)) in enumerate(zip(axes, sources)):
        sub_embeds = embeds[mask]
        norm = sub_embeds / (np.linalg.norm(sub_embeds, axis=1, keepdims=True) + 1e-8)
        proj = TSNE(
            n_components=2, perplexity=min(30, len(sub_embeds) // 3), random_state=seed,
            init="pca", metric="cosine",
        ).fit_transform(norm)

        ax.scatter(proj[:, 0], proj[:, 1], s=2 if printing else 8, alpha=0.35, color="lightgray", edgecolors="none", zorder=1)
        labeled_identities = set()
        for i in display_idx:
            ident = sub_identities[i]
            _, obs = sub_samples[i]
            if printing:
                # se carga a PRINT_THUMB_PX y se muestra a thumb_pt puntos (OffsetImage
                # dibuja 1 px = 1 pt * zoom): nítido a 600 dpi
                crop = _load_crop(obs, PRINT_THUMB_PX)
                imagebox = OffsetImage(crop, zoom=thumb_pt / max(crop.shape[:2]))
            else:
                crop = _load_crop(obs, thumb_size)
                imagebox = OffsetImage(crop, zoom=1.0)
            ab = AnnotationBbox(
                imagebox, (proj[i, 0], proj[i, 1]), frameon=True, pad=0.05,
                bboxprops=dict(edgecolor=color_map[ident], linewidth=0.9 if printing else 2.2), zorder=2,
            )
            ax.add_artist(ab)
            # número de identidad real -- identity_id = f"{seq_name}__{track_id}"
            # (ver train_track_query_reid.py), track_id es lo que distingue
            # identidades DENTRO del mismo video (ver sample_identities). Solo UNA
            # vez por identidad (no en cada recorte repetido) -- con 2+ recortes de
            # la misma identidad muy cerca en el t-SNE, el label duplicado quedaba
            # ilegible, superpuesto sobre sí mismo.
            if ident in labeled_identities:
                continue
            labeled_identities.add(ident)
            track_id = ident.split("__")[-1]
            ax.annotate(
                track_id, xy=(proj[i, 0], proj[i, 1]), xycoords="data",
                xytext=(0, -((thumb_pt if printing else thumb_size) / 2 + (1.5 if printing else 7))),
                textcoords="offset points",
                ha="center", va="top", fontsize=font_size if printing else 7, fontweight="bold", color=color_map[ident],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.3 if printing else 0.5), zorder=3,
            )
        if printing:
            # margen interno: el recorte (thumb_pt) y su número se dibujan en puntos,
            # alrededor de su coordenada -- sin margen se salen del panel
            panel_pt = ax.get_window_extent().width * 72 / fig.dpi
            x0, x1 = proj[:, 0].min(), proj[:, 0].max()
            y0, y1 = proj[:, 1].min(), proj[:, 1].max()
            fx = (thumb_pt * 0.6) / panel_pt
            fy_top, fy_bot = (thumb_pt * 0.6) / panel_pt, (thumb_pt * 0.6 + font_size * 1.6) / panel_pt
            dx, dy = (x1 - x0), (y1 - y0)
            ax.set_xlim(x0 - dx * fx / (1 - 2 * fx), x1 + dx * fx / (1 - 2 * fx))
            ax.set_ylim(y0 - dy * fy_bot / (1 - fy_top - fy_bot), y1 + dy * fy_top / (1 - fy_top - fy_bot))
            ax.set_title(PRINT_TITLES.get(name, name), fontsize=font_size + 1, fontweight="bold", pad=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        else:
            ax.set_title(f"{name} -- t-SNE by identity")
            if col == 0:
                handles = [Patch(facecolor=color, label=video) for video, color in video_legend]
                ax.legend(handles=handles, title="same color = same video", fontsize=7, title_fontsize=8, loc="best")
        ax.set_xticks([]); ax.set_yticks([])
    if printing:
        handles = [Line2D([], [], linestyle="none", label="Same colour = same video:")]
        handles += [Patch(facecolor=color, label=video) for video, color in video_legend]
        fig.legend(handles=handles, fontsize=font_size, loc="lower center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.2, handletextpad=0.4, columnspacing=1.2,
                   borderaxespad=0.1)
        for ext, dpi in (("pdf", 600), ("png", 600)):
            path = os.path.join(output_dir, f"tsne_by_identity_crops_print.{ext}")
            fig.savefig(path, dpi=dpi)
            print(f"[OK] {path}")
        plt.close(fig)
        return
    fig.tight_layout()
    path = os.path.join(output_dir, "tsne_by_identity_crops.png")
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
    parser.add_argument("--tsne-identities", type=int, default=12, help="Cuántas identidades mostrar (reducido de 20 -- menos objetos, más legible).")
    parser.add_argument(
        "--crops-per-identity", type=int, default=2,
        help="Cuántos recortes reales mostrar por identidad (con 3 paneles side-by-side, mantener "
             "bajo para legibilidad) -- el resto queda como punto gris de fondo.",
    )
    parser.add_argument("--thumb-size", type=int, default=42, help="Tamaño (px, lado mayor) de cada thumbnail.")
    parser.add_argument("--print-width", type=float, default=None,
                        help="(pulgadas) dibuja a tamaño de impresión -- ver docstring. Paper: 5.4")
    parser.add_argument("--font-size", type=float, default=8.0, help="(con --print-width) tamaño mínimo de letra, pt")
    parser.add_argument("--thumb-pt", type=float, default=16.0, help="(con --print-width) lado de cada recorte, pt")
    parser.add_argument("--font-dir", default=None, help="(con --print-width) carpeta con los .ttf de Carlito")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-wide-factor", type=float, default=9.0, help="Mismo default que embedding_analysis_dinov2_vs_p4.py.")
    parser.add_argument("--output-dir", default="compression_analysis")
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

    display_ids, color_map, video_legend = _pick_display_identities(identities, args.tsne_identities, args.seed)
    plot_crops_by_identity(sources, samples, identities, display_ids, color_map, video_legend, args.seed, args.crops_per_identity, args.thumb_size, args.output_dir,
                           print_width=args.print_width, font_size=args.font_size, thumb_pt=args.thumb_pt,
                           font_dir=args.font_dir)


if __name__ == "__main__":
    main()
