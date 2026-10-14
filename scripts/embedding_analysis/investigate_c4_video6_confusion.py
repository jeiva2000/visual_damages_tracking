"""
investigate_c4_video6_confusion.py

Sigue al hallazgo visual de plot_embedding_crops_single_video.py: en C4,
video6 (35 identidades, 525 observaciones) muestra una zona con varios
recortes de colores distintos solapados/entremezclados en el t-SNE. Este
script cuantifica si eso es confusión REAL (el vecino más cercano de verdad
sale de otra identidad) o solo apretamiento visual propio de un espacio más
comprimido -- IMPORTANTE: C4 comprime más (nube más chica en PCA, ver
plot_pca_single_video/plot_pca_compression) pero eso NO implica menos
separabilidad -- rank-1 same-video de C4 en el pool completo ya salió
~0.989 (ver README), casi tan bueno como P4-FPN/DINOv2. Este script mide lo
mismo pero restringido a video6 y con detalle por-par-de-identidades, para
poder mirar los recortes reales de los pares más al límite.

Para cada observación de video6, calcula (mismo criterio de vecino más
cercano same-video que cmc_same_video en embedding_analysis_dinov2_vs_p4.py):
  - best_same_sim: similitud coseno del vecino MÁS parecido de SU MISMA identidad.
  - best_diff_sim / confuser_identity: similitud e identidad del vecino MÁS
    parecido de OTRA identidad (el "distractor" más peligroso).
  - margin = best_same_sim - best_diff_sim (positivo y grande = buen margen;
    negativo = el vecino más cercano real es de otra identidad, o sea rank-1
    FALLA para esa observación).
  - nn_correct: si el vecino más cercano ABSOLUTO (sin restringir a same-id)
    es de la identidad correcta -- esto es literalmente rank-1 same-video
    por observación.

Agrega por PAR de identidades (identity, confuser_identity) y reporta los
`--top-k` pares con margen promedio más bajo -- los candidatos reales a
"parecerse" en C4 -- junto con cuántas de esas observaciones tuvieron
nn_correct=False (error real de rank-1, no solo margen chico). Corre lo
mismo para DINOv2 y P4-FPN sobre las MISMAS observaciones para poder decir
si el par es confuso SOLO en C4 (compresión) o en los 3 encoders (contenido
visual genuinamente ambiguo).

Además genera una figura de comparación lado a lado (recortes reales, sin
proyección) de los identidades involucradas en los `--zoom-pairs` pares más
confusos -- para inspección visual directa.

Uso (correr en el servidor, mismos caches que plot_embedding_crops_single_video.py):
    python3 scripts/embedding_analysis/investigate_c4_video6_confusion.py --p4-cache p4_roi_cache_kfold_pool.pt \
        --p4fpn-cache p4fpn_roi_cache_kfold_pool.pt --output-dir compression_analysis_kfoldpool
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import argparse
import os
import random
from collections import defaultdict

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from embedding_analysis_dinov2_vs_p4 import TIGHT_FACTOR, _video_of, compute_cache_embeddings, compute_dinov2_embeddings, sample_identities
from reid_dataloader import _expand_bbox
from train_track_query_reid import build_track_index, resolve_sequence_dirs


def nearest_neighbor_records(embeds, identities):
    """Por observación: mejor same-id, mejor diff-id (con su identidad), margen
    y si el vecino más cercano ABSOLUTO (rank-1) es correcto. Restringido
    implícitamente a las observaciones que se le pasen (acá, ya filtradas a
    un solo video -- mismo criterio same-video que cmc_same_video)."""
    identities = np.asarray(identities)
    norm = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    sims = norm @ norm.T
    n = len(identities)
    np.fill_diagonal(sims, -np.inf)

    records = []
    for i in range(n):
        same_mask = identities == identities[i]
        same_mask[i] = False
        diff_mask = ~same_mask
        diff_mask[i] = False
        if not same_mask.any() or not diff_mask.any():
            continue
        row = sims[i]
        best_same_idx = int(np.argmax(np.where(same_mask, row, -np.inf)))
        best_diff_idx = int(np.argmax(np.where(diff_mask, row, -np.inf)))
        best_same_sim = float(row[best_same_idx])
        best_diff_sim = float(row[best_diff_idx])
        nn_idx = int(np.argmax(row))
        records.append({
            "i": i, "identity": identities[i],
            "best_same_sim": best_same_sim, "best_diff_sim": best_diff_sim,
            "confuser_identity": identities[best_diff_idx], "confuser_idx": best_diff_idx,
            "margin": best_same_sim - best_diff_sim,
            "nn_correct": bool(identities[nn_idx] == identities[i]),
        })
    return records


def aggregate_by_pair(records):
    """Agrupa por par NO ordenado (identity, confuser_identity) -- promedia
    margen y cuenta errores reales de rank-1 (nn_correct=False) en ese par."""
    by_pair = defaultdict(lambda: {"margins": [], "n_errors": 0, "n_obs": 0})
    for r in records:
        pair = tuple(sorted((r["identity"], r["confuser_identity"])))
        entry = by_pair[pair]
        entry["margins"].append(r["margin"])
        entry["n_obs"] += 1
        if not r["nn_correct"]:
            entry["n_errors"] += 1
    rows = []
    for pair, entry in by_pair.items():
        rows.append({
            "pair": pair, "mean_margin": float(np.mean(entry["margins"])),
            "min_margin": float(np.min(entry["margins"])),
            "n_obs": entry["n_obs"], "n_errors": entry["n_errors"],
        })
    rows.sort(key=lambda r: r["mean_margin"])
    return rows


def print_report(name, records, top_k):
    n_errors = sum(1 for r in records if not r["nn_correct"])
    rank1 = 1.0 - n_errors / len(records)
    print(f"\n[{name}] video6: rank-1 same-video = {rank1:.4f} ({n_errors}/{len(records)} errores) -- mean margin = {np.mean([r['margin'] for r in records]):.4f}")
    rows = aggregate_by_pair(records)
    print(f"[{name}] Top {top_k} pares MÁS confusos (margen promedio más bajo):")
    for row in rows[:top_k]:
        print(f"    {row['pair'][0]} <-> {row['pair'][1]}: mean_margin={row['mean_margin']:.4f} min_margin={row['min_margin']:.4f} n_obs={row['n_obs']} n_errores_rank1={row['n_errors']}")
    return rows


def _load_crop(obs, thumb_size):
    image = Image.open(obs.image_path).convert("RGB")
    crop = image.crop(_expand_bbox(obs.bbox, TIGHT_FACTOR, image.size))
    crop.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
    return np.asarray(crop)


def plot_pair_comparison(samples_video, identities_video, pair, output_dir, thumb_size, max_crops=6):
    id_a, id_b = pair
    crops_a = [obs for ident, obs in samples_video if ident == id_a][:max_crops]
    crops_b = [obs for ident, obs in samples_video if ident == id_b][:max_crops]
    n = max(len(crops_a), len(crops_b))
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for col in range(n):
        for row, crops, label in [(0, crops_a, id_a), (1, crops_b, id_b)]:
            ax = axes[row, col]
            ax.axis("off")
            if col < len(crops):
                ax.imshow(_load_crop(crops[col], thumb_size * 3))
            if col == 0:
                ax.set_title(label, fontsize=8, loc="left")
    fig.suptitle(f"C4 -- par más confuso en video6: {id_a}  vs  {id_b}", fontsize=11)
    fig.tight_layout()
    safe = f"{id_a}_vs_{id_b}".replace("::", "-").replace("/", "_")
    path = os.path.join(output_dir, f"confusable_pair_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--kfold-pool", action="store_true", default=True)
    parser.add_argument("--p4-cache", required=True)
    parser.add_argument("--p4fpn-cache", default=None)
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument("--num-identities", type=int, default=1000)
    parser.add_argument("--obs-per-identity", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0, help="Mismo seed que plot_embedding_crops_single_video.py -- reproduce el mismo pool/video6/observaciones.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-wide-factor", type=float, default=9.0)
    parser.add_argument("--output-dir", default="compression_analysis_kfoldpool")
    parser.add_argument("--video", default=None, help="Default: el que más identidades aporta al pool (mismo criterio que antes).")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--zoom-pairs", type=int, default=3, help="Cuántos de los pares más confusos en C4 graficar con recortes reales lado a lado.")
    parser.add_argument("--thumb-size", type=int, default=48)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.kfold_pool:
        from kfold_utils import list_all_sequences
        seq_dirs = list_all_sequences()
        index = build_track_index(None, sequence_dirs=seq_dirs)
    else:
        data_root, seq_dirs = resolve_sequence_dirs(None, args.split)
        index = build_track_index(data_root, sequence_dirs=seq_dirs)

    samples = sample_identities(index, args.num_identities, args.obs_per_identity, args.seed)
    identities = [ident for ident, _ in samples]
    print(f"[INFO] {len(samples)} observaciones de {len(set(identities))} identidades")

    if args.video:
        video = args.video
    else:
        by_video = {}
        for ident in identities:
            by_video.setdefault(_video_of(ident), []).append(ident)
        video = max(by_video, key=lambda v: len(set(by_video[v])))
    mask = np.array([_video_of(ident) == video for ident in identities])
    n_ids_video = len(set(ident for ident in identities if _video_of(ident) == video))
    print(f"[INFO] video = {video} ({n_ids_video} identidades, {mask.sum()} observaciones)")

    samples_video = [s for s, keep in zip(samples, mask) if keep]
    identities_video = [ident for ident, keep in zip(identities, mask) if keep]

    dinov2_embeds = compute_dinov2_embeddings(samples, args.dinov2_model_name, args.device, args.query_wide_factor)[mask]
    c4_embeds = compute_cache_embeddings(samples, args.p4_cache)[mask]
    sources = [("DINOv2", dinov2_embeds), ("C4", c4_embeds)]
    if args.p4fpn_cache:
        sources.append(("P4 (FPN C5+C4)", compute_cache_embeddings(samples, args.p4fpn_cache)[mask]))

    c4_rows = None
    for name, embeds in sources:
        records = nearest_neighbor_records(embeds, identities_video)
        rows = print_report(name, records, args.top_k)
        if name == "C4":
            c4_rows = rows

    print("\n[INFO] Recortes reales de los pares más confusos en C4 (para inspección visual):")
    for row in c4_rows[:args.zoom_pairs]:
        plot_pair_comparison(samples_video, identities_video, row["pair"], args.output_dir, args.thumb_size)


if __name__ == "__main__":
    main()
