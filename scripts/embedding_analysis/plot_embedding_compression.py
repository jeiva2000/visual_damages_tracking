"""
plot_embedding_compression.py

Complementa embedding_analysis_dinov2_vs_p4.py (que mide separabilidad
identidad a identidad) con una figura centrada en la FORMA GLOBAL del
espacio de embeddings de DINOv2/C4/P4-FPN -- qué tan comprimido/anisotrópico
es cada uno (ver conversación: el gap intra/inter chico de C4 se explica
porque su espacio es anisotrópico, casi todo apunta en una dirección
parecida, con o sin relación de identidad).

t-SNE (el que usa embedding_analysis_dinov2_vs_p4.py) NO sirve para esto:
por diseño SIEMPRE separa visualmente los clusters en el plot final, sin
importar qué tan cerca estén en el espacio original -- un espacio
comprimido y uno disperso pueden terminar viéndose igual de "separados" en
un t-SNE. Esta figura usa en cambio:
  - PCA 2D (proyección LINEAL, sobre embeddings L2-normalizados -- mismo
    espacio donde vive la similitud coseno -- con EJES COMPARTIDOS entre
    los 3 encoders) -- al no distorsionar distancias, un espacio comprimido
    se ve como una nube chica, uno disperso como una nube grande. TODOS los
    puntos se dibujan en gris de fondo (preserva la forma/tamaño real de la
    nube completa); encima se colorea por identidad solo una submuestra
    chica (--tsne-identities, mismas identidades/colores que los paneles
    t-SNE/UMAP) para no repetir colores de tab20 con las ~344 identidades
    completas.
  - Varianza ACUMULADA CRUDA (sin normalizar) de PCA, las 3 curvas
    superpuestas -- la medida cuantitativa de compresión en magnitud
    ABSOLUTA (el ratio normalizado NO sirve para esto -- ver docstring de
    plot_pca_compression, mide forma relativa, no tamaño).

Además, a pedido explícito (comparar igual contra proyecciones no
lineales pese a la limitación de t-SNE de arriba), genera t-SNE Y UMAP
coloreados por identidad sobre una submuestra chica (--tsne-identities)
para que los colores sigan siendo legibles -- lectura CUALITATIVA de si
las identidades siguen agrupándose localmente incluso dentro de un espacio
comprimido (para conectar con el silhouette), no una medida de compresión.

Uso:
    python3 scripts/embedding_analysis/plot_embedding_compression.py --kfold-pool --num-identities 1000 \
        --p4-cache p4_roi_cache_kfold_pool.pt --p4fpn-cache p4fpn_roi_cache_kfold_pool.pt \
        --output-dir compression_analysis
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import argparse
import os
import random

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from embedding_analysis_dinov2_vs_p4 import _video_of, compute_cache_embeddings, compute_dinov2_embeddings, sample_identities
from train_track_query_reid import build_track_index, resolve_sequence_dirs

# una paleta secuencial (matplotlib colormap) distinta por video -- ver _pick_display_identities.
_VIDEO_PALETTES = ["Greens", "Blues", "Reds", "Purples", "Oranges", "Greys", "YlOrBr", "PuBuGn"]


def _pick_display_identities(identities, n_identities_display, seed):
    """Submuestra chica de identidades + color por identidad, compartida
    entre paneles (PCA/t-SNE/UMAP) para que el mismo color signifique la
    misma identidad en toda la figura.

    Concentrada en unos pocos videos con VARIAS identidades cada uno (no un
    muestreo al azar sobre todo el pool) -- con un muestreo al azar, la
    gran mayoría de los 20 caen en videos donde son la única identidad
    mostrada (confirmado empíricamente: 14/20 en un run real), así que casi
    toda la separación visual entre colores termina siendo separación
    ENTRE video (el negativo fácil, ver intra_inter_similarities) en vez de
    DENTRO del mismo video (el caso relevante que estas figuras deberían
    ilustrar). Acá se ordenan los videos por cuántas identidades aportan a
    la muestra y se toma de a `max_per_video` de los que más tienen, hasta
    completar n_identities_display -- así casi todas las identidades
    mostradas compiten de verdad dentro de la misma galería (mismo video).

    El color en sí NO es un tab20 arbitrario por identidad -- eso no deja
    reconocer a simple vista qué identidades comparten video (justo lo que
    esta muestra quiere mostrar). En cambio, cada VIDEO se lleva una paleta
    secuencial propia (`_VIDEO_PALETTES`) y cada identidad DENTRO de ese
    video un tono distinto de esa paleta -- mismo tono de verde = mismo
    video, tonos de verde distintos = identidades distintas de ese video."""
    by_video = {}
    for ident in sorted(set(identities)):
        by_video.setdefault(_video_of(ident), []).append(ident)
    videos_sorted = sorted(by_video, key=lambda v: -len(by_video[v]))

    rng = random.Random(seed)
    max_per_video = max(2, -(-n_identities_display // 3))  # techo(n/3), al menos 2 -- reparte entre unos pocos videos, no uno solo
    display_ids, display_videos_order = [], []
    for video in videos_sorted:
        if len(display_ids) >= n_identities_display:
            break
        candidates = by_video[video]
        take = min(max_per_video, len(candidates), n_identities_display - len(display_ids))
        display_ids.extend(rng.sample(candidates, take))
        display_videos_order.append(video)
    display_ids = sorted(display_ids)

    color_map, video_legend = {}, []
    for i, video in enumerate(display_videos_order):
        ids_in_video = sorted(ident for ident in display_ids if _video_of(ident) == video)
        cmap = plt.get_cmap(_VIDEO_PALETTES[i % len(_VIDEO_PALETTES)])
        shades = np.linspace(0.35, 0.9, len(ids_in_video)) if len(ids_in_video) > 1 else np.array([0.6])
        for ident, shade in zip(ids_in_video, shades):
            color_map[ident] = cmap(shade)
        # nombre genérico ("Video 1", no el nombre real de archivo) -- para figuras del
        # paper, no para debug; el nombre real interno no aporta nada al lector.
        video_legend.append((f"Video {i + 1}", cmap(0.65)))

    return set(display_ids), color_map, video_legend


def plot_pca_compression(sources, identities, display_ids, color_map, video_legend, output_dir):
    """PCA 2D con ejes compartidos (tamaño ABSOLUTO de la nube) + varianza
    cruda acumulada (magnitud absoluta, NO ratio normalizado), las 3 curvas
    superpuestas.

    OJO -- `pca.explained_variance_ratio_` (usado en una versión anterior de
    esta función) es una medida de FORMA relativa, normalizada por la
    varianza TOTAL de cada encoder por separado -- dos nubes con tamaño
    absoluto muy distinto (una gigante, otra minúscula) pueden tener el
    mismo ratio si tienen la misma forma relativa. Eso es lo que pasó acá:
    con el ratio, C4 no se veía "comprimido" (de hecho necesitaba MÁS
    componentes que P4-FPN para el mismo % de SU PROPIA varianza), aunque
    el scatter de abajo muestre clarísimo que su nube absoluta es mucho más
    chica -- exactamente lo que explica que sus similitudes intra/inter
    salgan ambas altas (ver conversación). Por eso acá se usa
    `pca.explained_variance_` CRUDA (sin normalizar) y la varianza TOTAL
    (traza de la covarianza, `norm.var(axis=0).sum()`) como el número que sí
    es comparable en magnitud absoluta entre encoders.

    Colorea por identidad SOLO una submuestra chica (`display_ids`,
    superpuesta sobre el resto en gris) -- con las ~344 identidades
    completas los colores de tab20 se repiten y no aportan nada (mismo
    problema que tenía el t-SNE original con tantas identidades). El fondo
    gris con TODOS los puntos preserva la lectura de tamaño/forma de la
    nube completa; los puntos coloreados encima muestran que las
    identidades siguen agrupándose localmente incluso dentro de esa nube."""
    display_mask = np.array([ident in display_ids for ident in identities])
    display_colors = [color_map[ident] for ident in identities if ident in display_ids]

    pca_results = []
    for name, embeds in sources:
        norm = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
        n_components = min(100, norm.shape[0] - 1, norm.shape[1])
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(norm)
        total_var = float(norm.var(axis=0).sum())  # traza de la covarianza -- tamaño absoluto real, no acotado por n_components
        pca_results.append((name, proj, pca.explained_variance_, total_var))

    # ejes compartidos entre los 3 paneles -- calculados sobre PC1/PC2 de los 3 encoders juntos,
    # así una nube más chica/apretada se VE más chica, en vez de que cada panel se auto-escale
    # (el mismo problema por el que t-SNE no sirve acá).
    all_xy = np.concatenate([proj[:, :2] for _, proj, _, _ in pca_results], axis=0)
    lim = float(np.abs(all_xy).max()) * 1.05

    fig, axes = plt.subplots(1, len(pca_results), figsize=(6 * len(pca_results), 5.5))
    if len(pca_results) == 1:
        axes = [axes]
    for col, (ax, (name, proj, ev, total_var)) in enumerate(zip(axes, pca_results)):
        ax.scatter(proj[:, 0], proj[:, 1], s=6, alpha=0.15, color="lightgray", edgecolors="none", zorder=1)
        ax.scatter(proj[display_mask, 0], proj[display_mask, 1], s=18, alpha=0.9, c=display_colors, edgecolors="none", zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(
            f"{name} -- PCA (shared axes)\n"
            f"total variance (trace) = {total_var:.4f} -- color = {len(display_ids)} identities"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if col == 0:
            handles = [Patch(facecolor=color, label=video) for video, color in video_legend]
            ax.legend(handles=handles, title="same color = same video", fontsize=7, title_fontsize=8, loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, "pca_compression_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] {path}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, _, ev, total_var in pca_results:
        cum = np.cumsum(ev)
        ax.plot(np.arange(1, len(cum) + 1), cum, marker="o", markersize=3, label=f"{name} (total={total_var:.4f})")
    ax.set_xlabel("number of PCA components")
    ax.set_ylabel("cumulative variance (raw, unnormalized)")
    ax.set_title("Compression: absolute variance magnitude captured per component")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "pca_explained_variance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] {path}")


def _single_video_colors(identities, video):
    """Máscara + color por identidad (colormap continuo, no tab20 -- se
    queda corto con las ~35 identidades que puede tener el video más
    poblado) restringido a un único video -- compartido entre
    plot_pca_single_video y plot_nonlinear_single_video para que el mismo
    color signifique la misma identidad en ambas figuras."""
    mask = np.array([_video_of(ident) == video for ident in identities])
    sub_identities = [ident for ident in identities if _video_of(ident) == video]
    unique_ids = sorted(set(sub_identities))
    cmap = plt.get_cmap("nipy_spectral", len(unique_ids))
    color_map = {ident: cmap(i) for i, ident in enumerate(unique_ids)}
    colors = [color_map[ident] for ident in sub_identities]
    return mask, colors, unique_ids


def plot_pca_single_video(sources, identities, video, output_dir):
    """Mismo PCA de ejes compartidos que plot_pca_compression, pero
    restringido a un ÚNICO video -- la galería real y completa de
    producción (todos los vehículos que compiten de verdad en un mismo
    clip), en vez de la mezcla de las ~344 identidades de todo el pool.
    Sin fondo gris (acá TODO son observaciones de ese video, no hay "resto
    de la población" que mostrar aparte)."""
    mask, colors, unique_ids = _single_video_colors(identities, video)

    pca_results = []
    for name, embeds in sources:
        sub = embeds[mask]
        norm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
        n_components = min(50, norm.shape[0] - 1, norm.shape[1])
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(norm)
        total_var = float(norm.var(axis=0).sum())
        pca_results.append((name, proj, total_var))

    all_xy = np.concatenate([proj[:, :2] for _, proj, _ in pca_results], axis=0)
    lim = float(np.abs(all_xy).max()) * 1.05

    fig, axes = plt.subplots(1, len(pca_results), figsize=(6 * len(pca_results), 5.5))
    if len(pca_results) == 1:
        axes = [axes]
    for ax, (name, proj, total_var) in zip(axes, pca_results):
        ax.scatter(proj[:, 0], proj[:, 1], s=14, alpha=0.85, c=colors, edgecolors="none")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{name} -- PCA, single video ({video})\ntotal variance (trace) = {total_var:.4f} -- {len(unique_ids)} identities")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    path = os.path.join(output_dir, "pca_single_video.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] {path}")


def plot_nonlinear_single_video(sources, identities, video, seed, output_dir):
    """t-SNE y UMAP restringidos al MISMO único video que plot_pca_single_video
    (mismos colores por identidad) -- verificación cualitativa: ¿las
    identidades siguen agrupándose localmente incluso dentro de esta
    galería real, comprimida en el caso de C4? No mide compresión (para
    eso, plot_pca_single_video) -- t-SNE/UMAP siempre separan visualmente,
    ver docstring del módulo."""
    import umap

    mask, colors, unique_ids = _single_video_colors(identities, video)

    for method_name, fname in [("t-SNE", "tsne_single_video.png"), ("UMAP", "umap_single_video.png")]:
        fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 5.5))
        if len(sources) == 1:
            axes = [axes]
        for ax, (name, embeds) in zip(axes, sources):
            sub = embeds[mask]
            norm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
            if method_name == "t-SNE":
                proj = TSNE(
                    n_components=2, perplexity=min(30, len(sub) // 3), random_state=seed,
                    init="pca", metric="cosine",
                ).fit_transform(norm)
            else:
                proj = umap.UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(norm)
            ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=20, alpha=0.85, edgecolors="none")
            ax.set_title(f"{name} -- {method_name}, single video ({video})\n{len(unique_ids)} identities")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[OK] {path}")


def plot_nonlinear_projections(sources, identities, display_ids, color_map, video_legend, seed, output_dir):
    """t-SNE y UMAP coloreados por identidad, sobre la MISMA submuestra
    chica que plot_pca_compression (mismo color = misma identidad en toda
    la figura) -- lectura CUALITATIVA de agrupamiento local, no una medida
    de compresión (para eso, ver plot_pca_compression)."""
    import umap

    mask = np.array([ident in display_ids for ident in identities])
    sub_identities = [ident for ident in identities if ident in display_ids]
    colors = [color_map[ident] for ident in sub_identities]

    for method_name, fname in [("t-SNE", "tsne_by_identity.png"), ("UMAP", "umap_by_identity.png")]:
        fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 5.5))
        if len(sources) == 1:
            axes = [axes]
        for col, (ax, (name, embeds)) in enumerate(zip(axes, sources)):
            sub = embeds[mask]
            norm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
            if method_name == "t-SNE":
                proj = TSNE(
                    n_components=2, perplexity=min(30, len(sub) // 3), random_state=seed,
                    init="pca", metric="cosine",
                ).fit_transform(norm)
            else:
                proj = umap.UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(norm)
            ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=20, alpha=0.85, edgecolors="none")
            ax.set_title(f"{name} -- {method_name} by identity (n={len(display_ids)} identities)")
            if col == 0:
                handles = [Patch(facecolor=color, label=video) for video, color in video_legend]
                ax.legend(handles=handles, title="same color = same video", fontsize=7, title_fontsize=8, loc="best")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[OK] {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--kfold-pool", action="store_true",
        help="Ignora --split -- usa el pool de 64 secuencias de --kfold-cv (kfold_utils.list_all_sequences, "
             "mismo criterio que embedding_analysis_dinov2_vs_p4.py).",
    )
    parser.add_argument("--p4-cache", required=True, help="PrecomputedEmbeddingCache de C4 crudo (--layer c4, ver precompute_p4_roi_cache.py).")
    parser.add_argument("--p4fpn-cache", default=None, help="[opcional] PrecomputedEmbeddingCache de P4 REAL (--layer p4fpn).")
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument("--num-identities", type=int, default=1000, help="Alto por default -- para PCA/varianza conviene usar TODAS las identidades elegibles (se cae solo al máximo disponible).")
    parser.add_argument("--obs-per-identity", type=int, default=15)
    parser.add_argument("--tsne-identities", type=int, default=20, help="Cuántas identidades mostrar (coloreadas) en los paneles t-SNE/UMAP -- legibilidad de color.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-wide-factor", type=float, default=9.0, help="Mismo default que embedding_analysis_dinov2_vs_p4.py/analyze_dinov2_nn_baseline.py.")
    parser.add_argument("--output-dir", default="compression_analysis")
    parser.add_argument(
        "--single-video-pca", action="store_true",
        help="Si se pasa, además arma un PCA (ejes compartidos, sin fondo gris) restringido "
             "al video con más identidades del pool -- la galería real y completa de un solo clip, "
             "en vez de la mezcla de todo el pool (ver plot_pca_single_video).",
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

    display_ids, color_map, video_legend = _pick_display_identities(identities, args.tsne_identities, args.seed)
    plot_pca_compression(sources, identities, display_ids, color_map, video_legend, args.output_dir)
    plot_nonlinear_projections(sources, identities, display_ids, color_map, video_legend, args.seed, args.output_dir)

    if args.single_video_pca:
        by_video = {}
        for ident in identities:
            by_video.setdefault(_video_of(ident), []).append(ident)
        biggest_video = max(by_video, key=lambda v: len(set(by_video[v])))
        print(f"[INFO] --single-video-pca: video con más identidades = {biggest_video} ({len(set(by_video[biggest_video]))} identidades)")
        plot_pca_single_video(sources, identities, biggest_video, args.output_dir)
        plot_nonlinear_single_video(sources, identities, biggest_video, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
