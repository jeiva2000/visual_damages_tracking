"""
plot_embedding_crops_single_video.py

Extiende plot_nonlinear_single_video (plot_embedding_compression.py)
reemplazando los puntos de color por los RECORTES REALES del daño en cada
posición del t-SNE -- pensado para investigar el hallazgo sin resolver del
README (el video con más identidades del pool, ~35 IDs: el UMAP de C4
muestra clusters mezclados que el t-SNE del mismo video no deja ver tan
claro) viendo directamente qué identidades se confunden y por qué (recortes
visualmente parecidos, mismo tipo de daño/pieza, ángulo, etc.), en vez de
solo el punto de color.

Mismo video (`biggest_video` = el que más identidades aporta al pool -- ver
--single-video-pca en plot_embedding_compression.py), mismos 3 encoders
(DINOv2 crudo, C4 crudo, P4-FPN) y mismo t-SNE (perplexity, metric coseno,
init pca) que plot_nonlinear_single_video -- la única diferencia es la
representación visual:
  - Fondo: TODAS las observaciones del video como puntos grises chicos
    (preserva la estructura completa del t-SNE, igual que el fondo gris de
    plot_pca_compression).
  - Encima: --crops-per-identity recortes reales por identidad (el
    AJUSTADO, factor=1.0 -- el daño puro, el mismo TIGHT_FACTOR que ve el
    encoder, ver embedding_analysis_dinov2_vs_p4.py), con un marco de color
    por identidad (misma paleta nipy_spectral que _single_video_colors) para
    poder leer agrupamiento por color Y por contenido visual a la vez.
    Mostrar TODAS las ~15 obs/identidad como thumbnail satura la figura y la
    hace ilegible -- por eso se subsamplea por identidad.

Uso (correr en el servidor, requiere acceso al dataset + cache P4/P4-FPN ya
generado, igual que embedding_analysis_dinov2_vs_p4.py):
    python3 scripts/embedding_analysis/plot_embedding_crops_single_video.py --kfold-pool \
        --p4-cache p4_roi_cache_kfold_pool.pt --p4fpn-cache p4fpn_roi_cache_kfold_pool.pt \
        --output-dir compression_analysis_kfoldpool
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'reid_lib'))  # library modules
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

from embedding_analysis_dinov2_vs_p4 import TIGHT_FACTOR, _video_of, compute_cache_embeddings, compute_dinov2_embeddings, sample_identities
from plot_embedding_compression import _single_video_colors
from reid_dataloader import _expand_bbox
from train_track_query_reid import build_track_index, resolve_sequence_dirs


def _load_crop(obs, thumb_size):
    """Recorte AJUSTADO (factor=1.0, TIGHT_FACTOR) -- el daño puro, sin la
    vista amplia de contexto (esa se promedia adentro del embedding pero acá
    solo interesa mostrar de un vistazo qué daño es cada punto)."""
    image = Image.open(obs.image_path).convert("RGB")
    crop = image.crop(_expand_bbox(obs.bbox, TIGHT_FACTOR, image.size))
    crop.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
    return np.asarray(crop)


def plot_crops_single_video(sources, samples, identities, video, seed, crops_per_identity, thumb_size, output_dir):
    mask, colors, unique_ids = _single_video_colors(identities, video)
    sub_samples = [s for s, keep in zip(samples, mask) if keep]
    sub_identities = [ident for ident, keep in zip(identities, mask) if keep]
    ident_color = dict(zip(sub_identities, colors))  # _single_video_colors ya da 1 color/identidad

    # Subsample por identidad para mostrar como recorte real -- el resto queda
    # como punto gris de fondo (ver docstring del módulo).
    rng = random.Random(seed)
    by_identity = {}
    for i, ident in enumerate(sub_identities):
        by_identity.setdefault(ident, []).append(i)
    display_idx = set()
    for idxs in by_identity.values():
        display_idx.update(rng.sample(idxs, min(crops_per_identity, len(idxs))))

    for name, embeds in sources:
        sub_embeds = embeds[mask]
        norm = sub_embeds / (np.linalg.norm(sub_embeds, axis=1, keepdims=True) + 1e-8)
        proj = TSNE(
            n_components=2, perplexity=min(30, len(sub_embeds) // 3), random_state=seed,
            init="pca", metric="cosine",
        ).fit_transform(norm)

        fig, ax = plt.subplots(figsize=(11, 11))
        ax.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.35, color="lightgray", edgecolors="none", zorder=1)
        for i in display_idx:
            _, obs = sub_samples[i]
            crop = _load_crop(obs, thumb_size)
            imagebox = OffsetImage(crop, zoom=1.0)
            ab = AnnotationBbox(
                imagebox, (proj[i, 0], proj[i, 1]), frameon=True, pad=0.05,
                bboxprops=dict(edgecolor=ident_color[sub_identities[i]], linewidth=2.5),
            )
            ax.add_artist(ab)
        ax.set_title(
            f"{name} -- t-SNE, single video ({video}), recortes reales\n"
            f"{len(unique_ids)} identities, {len(display_idx)} recortes mostrados (de {len(sub_samples)} observaciones)"
        )
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        path = os.path.join(output_dir, f"tsne_crops_single_video_{safe_name}.png")
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
        help="Cuántos recortes reales mostrar por identidad -- el resto de sus observaciones "
             "queda como punto gris de fondo (mostrar todas satura la figura).",
    )
    parser.add_argument("--thumb-size", type=int, default=48, help="Tamaño (px, lado mayor) de cada thumbnail.")
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
    n_ids_video = len(set(ident for ident in identities if _video_of(ident) == video))
    print(f"[INFO] video = {video} ({n_ids_video} identidades)")

    plot_crops_single_video(sources, samples, identities, video, args.seed, args.crops_per_identity, args.thumb_size, args.output_dir)


if __name__ == "__main__":
    main()
