"""
plot_dendrogram_crops_single_video.py

Complementa plot_pca_crops_single_video.py (PCA + círculo de tamaño --
muestra que el espacio de C4 es más chico/comprimido) con una vista de
clustering jerárquico (dendrograma, scipy.cluster.hierarchy) sobre las MISMAS
identidades curadas de video6 (los pares con errores reales de rank-1 en C4,
ver investigate_c4_video6_confusion.py, + unas pocas de contraste) -- ver
conversación: el dendrograma muestra directamente A QUÉ ALTURA (distancia)
se fusiona cada par, que es una forma más directa de leer "C4 los confunde,
los otros no" que comparar el tamaño de una nube PCA.

Cada identidad se resume en UN vector (promedio de sus embeddings L2-
normalizados en TODAS sus observaciones de video6 -- el mismo criterio de
"track summary" que usa el matching real, ver `_embed_batch`/track_summary
en embedding_analysis_dinov2_vs_p4.py/analyze_dinov2_nn_baseline.py), no
por observación individual -- un dendrograma necesita una distancia por
identidad, así que se pierde la variación intra-identidad que sí se ve en
el PCA (ver conversación). Distancia = coseno, linkage = average.

Cada hoja del árbol lleva el recorte real de esa identidad (mismo criterio
TIGHT_FACTOR que el resto de scripts de esta familia) en vez de un número --
el MISMO recorte en los 3 paneles (DINOv2/C4/P4-FPN), elegido una sola vez
por identidad, para que sea reconocible de un panel a otro.

Eje Y (distancia coseno) COMPARTIDO entre los 3 paneles -- mismo motivo que
los ejes compartidos del PCA: si cada panel se autoescala a su propio
máximo, un árbol comprimido (C4) y uno disperso (DINOv2/P4-FPN) se ven
igual de "altos", ocultando justo la compresión que se quiere mostrar.

Uso (correr en el servidor, mismos caches que plot_pca_crops_single_video.py):
    python3 scripts/embedding_analysis/plot_dendrogram_crops_single_video.py --kfold-pool \
        --p4-cache p4_roi_cache_kfold_pool.pt --p4fpn-cache p4fpn_roi_cache_kfold_pool.pt \
        --output-dir compression_analysis_kfoldpool
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import argparse
import os
import random

import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from embedding_analysis_dinov2_vs_p4 import TIGHT_FACTOR, _video_of, compute_cache_embeddings, compute_dinov2_embeddings, sample_identities
from plot_embedding_compression import _single_video_colors
from plot_pca_crops_single_video import CONFUSABLE_TRACK_IDS_VIDEO6
from reid_dataloader import _expand_bbox
from train_track_query_reid import build_track_index, resolve_sequence_dirs


def _load_crop(obs, thumb_size):
    """Recorte AJUSTADO (factor=1.0, TIGHT_FACTOR) -- mismo criterio que el
    resto de scripts de esta familia."""
    image = Image.open(obs.image_path).convert("RGB")
    crop = image.crop(_expand_bbox(obs.bbox, TIGHT_FACTOR, image.size))
    crop.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
    return np.asarray(crop)


def _select_display_identities(unique_sub_identities, seed, n_contrast_identities):
    """Mismo criterio de curación que plot_pca_crops_single_video.py -- los
    pares confusos de C4 (CONFUSABLE_TRACK_IDS_VIDEO6) + unas pocas de
    contraste al azar (reproducible por seed)."""
    rng = random.Random(seed)
    confusable_present = [i for i in unique_sub_identities if i.split("__")[-1] in CONFUSABLE_TRACK_IDS_VIDEO6]
    remaining = [i for i in unique_sub_identities if i not in confusable_present]
    contrast_present = rng.sample(remaining, min(n_contrast_identities, len(remaining)))
    return set(confusable_present + contrast_present)


def plot_dendrogram_crops_single_video(sources, samples, identities, video, seed, n_contrast_identities, thumb_size, output_dir):
    mask, colors, unique_ids = _single_video_colors(identities, video)
    sub_samples = [s for s, keep in zip(samples, mask) if keep]
    sub_identities = [ident for ident, keep in zip(identities, mask) if keep]
    ident_color = dict(zip(sub_identities, colors))

    unique_sub_identities = sorted(set(sub_identities))
    rng = random.Random(seed)
    display_identities = _select_display_identities(unique_sub_identities, seed, n_contrast_identities)
    print(f"[INFO] Identidades en el dendrograma: {len(display_identities)} de {len(unique_sub_identities)} totales")

    # UN recorte representativo por identidad -- el MISMO en los 3 paneles.
    obs_by_identity = {}
    for ident, obs in zip(sub_identities, [s[1] for s in sub_samples]):
        obs_by_identity.setdefault(ident, []).append(obs)
    representative_obs = {ident: rng.choice(obs_by_identity[ident]) for ident in display_identities}

    id_order = sorted(display_identities)
    track_labels = [ident.split("__")[-1] for ident in id_order]

    # Paso 1: calcular el linkage de los 3 encoders ANTES de dibujar, para poder
    # compartir la escala del eje Y (distancia coseno) entre paneles -- igual
    # criterio que los ejes compartidos del PCA (plot_pca_crops_single_video.py):
    # sin esto, cada panel se autoescala a su propio máximo y la comparación
    # "C4 fusiona a MENOR distancia absoluta" (la compresión) queda oculta,
    # porque un árbol comprimido y uno disperso se verían igual de "altos".
    linkages = []
    for name, embeds in sources:
        sub_embeds = embeds[mask]
        norm = sub_embeds / (np.linalg.norm(sub_embeds, axis=1, keepdims=True) + 1e-8)

        # Vector por identidad = promedio ("track summary") sobre TODAS sus
        # observaciones de video6, no solo la que se muestra como recorte.
        vectors = []
        for ident in id_order:
            idxs = [i for i, s_ident in enumerate(sub_identities) if s_ident == ident]
            mean_vec = norm[idxs].mean(axis=0)
            mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
            vectors.append(mean_vec)
        vectors = np.stack(vectors, axis=0)

        cos_sim = vectors @ vectors.T
        cos_dist = np.clip(1.0 - cos_sim, 0.0, None)
        np.fill_diagonal(cos_dist, 0.0)
        condensed = squareform(cos_dist, checks=False)
        Z = linkage(condensed, method="average")
        linkages.append((name, Z))

    y_max_shared = max((Z[:, 2].max() if len(Z) else 1.0) for _, Z in linkages)
    y_thumb = -0.12 * y_max_shared

    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 6.5))
    if len(sources) == 1:
        axes = [axes]
    for ax, (name, Z) in zip(axes, linkages):
        ddata = dendrogram(Z, labels=track_labels, ax=ax, color_threshold=0, above_threshold_color="black")
        leaves = ddata["leaves"]  # índices en id_order, en el orden izquierda-a-derecha del árbol
        # Posiciones REALES de las hojas (no asumir la convención 5,15,25... de scipy --
        # más robusto ante cambios de versión) -- se leen de los xticks que matplotlib
        # asignó al pasar `labels`, y recién después se los saca del eje.
        leaf_x = list(ax.get_xticks())
        ax.set_xticks([])

        for x, leaf_idx in zip(leaf_x, leaves):
            ident = id_order[leaf_idx]
            crop = _load_crop(representative_obs[ident], thumb_size)
            imagebox = OffsetImage(crop, zoom=1.0)
            ab = AnnotationBbox(
                imagebox, (x, y_thumb), frameon=True, pad=0.05, xycoords="data",
                bboxprops=dict(edgecolor=ident_color[ident], linewidth=2.2), zorder=3, annotation_clip=False,
            )
            ax.add_artist(ab)
            ax.annotate(
                track_labels[leaf_idx], xy=(x, y_thumb), xycoords="data",
                xytext=(0, -(thumb_size / 2 + 6)), textcoords="offset points",
                ha="center", va="top", fontsize=8, fontweight="bold", color=ident_color[ident],
                annotation_clip=False, zorder=4,
            )

        ax.set_ylim(y_thumb - 0.35 * y_max_shared, y_max_shared * 1.08)
        ax.set_title(f"{name} -- clustering jerárquico (video6)")
        ax.set_ylabel("distancia coseno (linkage=average)")
    fig.tight_layout()
    path = os.path.join(output_dir, "dendrogram_crops_single_video.png")
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
        "--n-contrast-identities", type=int, default=4,
        help="Cuántas identidades SIN errores conocidos agregar como contraste, además de las confusas "
             "de CONFUSABLE_TRACK_IDS_VIDEO6 (mismo criterio que plot_pca_crops_single_video.py).",
    )
    parser.add_argument("--thumb-size", type=int, default=55, help="Tamaño (px, lado mayor) de cada thumbnail.")
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

    plot_dendrogram_crops_single_video(sources, samples, identities, video, args.seed, args.n_contrast_identities, args.thumb_size, args.output_dir)


if __name__ == "__main__":
    main()
