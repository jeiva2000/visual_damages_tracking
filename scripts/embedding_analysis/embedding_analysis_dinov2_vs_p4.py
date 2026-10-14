"""
embedding_analysis_dinov2_vs_p4.py

Comparación visual + cuantitativa de la señal de identidad en embeddings
crudos (SIN ninguna cabeza entrenada) de DINOv2 (CLS) vs P4 (ROI-Align sobre
backbone.res_layers.2 de RT-DETRv2, ver p4_backbone_encoder.py) -- para la
sección "análisis de embeddings" del paper.

Metodología (misma familia que analyze_dinov2_layers.py/dinov2_reid_analysis.py
en ../damages_tracking, adaptada para comparar ambos encoders lado a lado con
el MISMO estilo de plot):
  - Muestra --num-identities identidades con >= --obs-per-identity
    observaciones cada una, de --split.
  - DINOv2: CLS crudo sobre DOS vistas -- AJUSTADA (factor=1.0) + AMPLIA
    (factor=--query-wide-factor) -- promediadas, se calcula en vivo
    (AutoModel). P4/C4: se leen del cache YA CALCULADO (PrecomputedEmbeddingCache,
    ver precompute_p4_roi_cache.py), promediando la escala ajustada
    (padding_factors[-1]=1.0) con la intermedia (padding_factors[1]=1.6),
    sin volver a correr el detector. En AMBOS casos es el MISMO
    promedio-de-dos-vistas que usa la query real del matching (ver
    `_load_query_views`/`_embed_batch`) -- comparar solo con el crop
    ajustado (como se hacía antes de este fix) evalúa una entrada distinta
    a la que el algoritmo real usa, porque el método real siempre agrega
    contexto de pieza además del daño recortado.
  - Por encoder: proyección t-SNE 2D coloreada por identidad (fila de
    arriba, lectura cualitativa) + panel de separabilidad (fila de abajo).
    El panel de abajo depende de --gallery-episodes:
      - --gallery-episodes > 0 (RECOMENDADO): panel best-of-galería -- NO
        mide pares al azar, replica el mecanismo real de decisión (ver
        `_calibrate_threshold` en analyze_dinov2_nn_baseline.py): arma
        episodios de galería (banco de hasta --max-bank-size candidatos,
        mezcla real/sintético/nuevo vía ReIDEpisodeDataset) y por cada uno
        guarda la similitud del MEJOR candidato (`best_sim`) junto con si
        la query tenía de verdad un match en ese banco (`is_match`).
        Reporta el umbral óptimo de accuracy (mismo criterio que
        producción) y el ROC-AUC de `best_sim` como separador de
        `is_match` -- la métrica de separabilidad que en verdad predice
        el resultado downstream.
      - --gallery-episodes 0 (default, fallback más barato): histograma de
        similitud coseno intra-identidad vs inter-identidad (gap +
        silhouette), con "inter" separado en mismo-video (relevante en
        producción, ver intra_inter_similarities) vs distinto-video
        (negativo trivial). OJO -- el gap NO predice separabilidad real:
        `best_sim | is_match=False` (best-of-galería) es un MÁXIMO sobre
        varios distractores, típicamente más alto que un par inter al
        azar, así que el gap intra/inter subestima sistemáticamente a
        encoders con inter alto pero buen ranking relativo (C4 es el
        ejemplo: gap chico, pero accuracy/AUC best-of-galería de los
        mejores de los tres -- ver conversación / sección resultados).
        Usar solo como diagnóstico preliminar barato, no como evidencia de
        separabilidad para el paper.
  - Siempre, además: curva CMC (rank-k accuracy, ver cmc_same_video) --
    métrica estándar de retrieval en ReID, complementaria al silhouette
    pero más directa: para cada observación, ¿aparece su identidad
    correcta entre los k vecinos más cercanos DEL MISMO VIDEO? rank-1 =
    "el vecino más cercano ya es correcto". A diferencia del gap, no pasa
    por ningún umbral ni por la escala absoluta de similitud -- mide
    directamente el ranking, que es como en verdad decide el matching real
    (argmax sobre la galería). Se guarda en `cmc_curve.png` (un panel, las
    3 curvas superpuestas) y en el summary.json (`cmc_same_video`).

Uso (correr en el servidor, requiere acceso al dataset + cache P4 ya generado):
    python3 scripts/embedding_analysis/embedding_analysis_dinov2_vs_p4.py --split val \
        --p4-cache p4_roi_cache_val.pt --output-dir embedding_analysis_val
    python3 scripts/embedding_analysis/embedding_analysis_dinov2_vs_p4.py --kfold-pool --gallery-episodes 300 \
        --p4-cache p4_roi_cache_kfold_pool.pt --p4fpn-cache p4fpn_roi_cache_kfold_pool.pt \
        --output-dir embedding_analysis_kfoldpool
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'reid_lib'))  # library modules
import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_dinov2_nn_baseline import _best_candidate, _build_loader, _embed_batch
from reid_dataloader import DEFAULT_TRANSFORM, PrecomputedEmbeddingCache, _expand_bbox
from train_track_query_reid import build_track_index, resolve_sequence_dirs

TIGHT_FACTOR = 1.0  # misma escala para DINOv2 (crop) y P4 (padding_factors[-1] del cache)


def sample_identities(index, num_identities, obs_per_identity, seed):
    """Clave de identidad = (video_id, identity_id) -- identity_id NO es único
    entre videos (ver IdentityTrack en reid_dataloader.py), así que usar solo
    identity_id conflaría identidades de videos distintos como si fueran la
    misma."""
    eligible = [t for t in index.tracks if len(t.observations) >= obs_per_identity]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = eligible[:num_identities]
    if len(chosen) < num_identities:
        print(f"[WARN] Solo {len(chosen)}/{num_identities} identidades tienen >= {obs_per_identity} observaciones.")
    samples = []
    for track in chosen:
        key = f"{track.video_id}::{track.identity_id}"
        obs_sorted = sorted(track.observations, key=lambda o: o.frame_idx)
        idxs = np.linspace(0, len(obs_sorted) - 1, obs_per_identity).astype(int)
        for i in sorted(set(idxs.tolist())):
            samples.append((key, obs_sorted[i]))
    return samples


@torch.no_grad()
def compute_dinov2_embeddings(samples, model_name, device, query_wide_factor, batch_size=32):
    """Dos vistas por observación -- AJUSTADA (factor=1.0, el daño puro) +
    AMPLIA (factor=query_wide_factor, contexto de pieza) -- promediadas
    DESPUÉS de encodear cada una, exactamente igual que `_load_query_views`
    + `_embed_batch` en el pipeline real (reid_dataloader.py/
    analyze_dinov2_nn_baseline.py). El matching real SIEMPRE ve la vista
    amplia además del recorte del daño; comparar solo con el crop ajustado
    (como hacía esta función antes de este fix) evalúa una entrada distinta
    a la que en verdad usa el algoritmo -- ver conversación."""
    from transformers import AutoModel
    print(f"[INFO] Cargando {model_name}...")
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Procesado por lotes de OBSERVACIONES (no de crops sueltos): con muestras
    # grandes (--num-identities alto), armar los 2*N crops en memoria de una
    # sola vez antes de batchear (como hacía una versión anterior de esta
    # función) hizo OOM en el servidor (15GB RAM) -- cada lote abre/recorta/
    # transforma solo `batch_size` observaciones (2*batch_size crops) por vez.
    outs = []
    for i in range(0, len(samples), batch_size):
        chunk_samples = samples[i:i + batch_size]
        tight_crops, wide_crops = [], []
        for _, obs in chunk_samples:
            image = Image.open(obs.image_path).convert("RGB")
            tight_crops.append(DEFAULT_TRANSFORM(image.crop(_expand_bbox(obs.bbox, TIGHT_FACTOR, image.size))))
            wide_crops.append(DEFAULT_TRANSFORM(image.crop(_expand_bbox(obs.bbox, query_wide_factor, image.size))))
        chunk_t = torch.stack(tight_crops + wide_crops, dim=0).to(device)  # (2b,C,H,W): [tight x b, wide x b]
        cls = model(pixel_values=chunk_t).last_hidden_state[:, 0, :].cpu().numpy()
        b = len(chunk_samples)
        outs.append((cls[:b] + cls[b:]) / 2.0)  # mean-pool tight+wide, igual que query_embeds real
    return np.concatenate(outs, axis=0)


def compute_cache_embeddings(samples, cache_path):
    """Sirve para cualquier PrecomputedEmbeddingCache (C4 crudo o P4 FPN real,
    ver --layer en precompute_p4_roi_cache.py). Promedia la escala AJUSTADA
    (factor 1.0, índice -1) con la escala INTERMEDIA (factor 1.6, índice 1)
    -- exactamente el mismo criterio de `_load_query_views` cuando corre con
    `embedding_cache` activo (la vista amplia con cache queda atada a
    padding_factors[1], NO al query_wide_factor real -- ver docstring ahí),
    para quedar consistente con compute_dinov2_embeddings (tight+wide
    promediados) en vez de usar solo el crop ajustado."""
    cache = PrecomputedEmbeddingCache.load(cache_path)
    embeds = []
    for _, obs in samples:
        scale_embeds = cache.get(obs)  # (S, D) -- padding_factors = [6.0, 1.6, 1.0]
        wide_idx = 1 if len(scale_embeds) > 1 else 0
        embeds.append(((scale_embeds[-1] + scale_embeds[wide_idx]) / 2.0).numpy())
    return np.stack(embeds, axis=0)


def _video_of(identity_key):
    """`identity_key` = f"{video_id}::{identity_id}" (ver sample_identities)."""
    return identity_key.split("::")[0]


def intra_inter_similarities(embeds, identities):
    """intra = mismo track (misma identidad). El resto de pares NO se agrupa
    todo junto como "inter" -- se separa en:
      - inter_same_video: identidad distinta, MISMO video. Es el caso
        relevante en producción: la galería de tracking vive dentro de un
        único video, nunca hace falta distinguir un vehículo del video A de
        uno del video B. Es también el caso que `DamageTrackIndex.
        sample_hard_distractors` (reid_dataloader.py) trata como "difícil".
      - inter_diff_video: identidad distinta, video distinto. Negativo
        trivial (`DamageTrackIndex.sample_easy_distractors`) -- casi siempre
        fácil de separar (fondo/luz/cámara ya lo delatan), y si se mezcla
        sin distinguir con inter_same_video infla el gap/silhouette respecto
        al caso que en verdad importa.
    """
    norm = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    sims = norm @ norm.T
    identities = np.asarray(identities)
    videos = np.asarray([_video_of(i) for i in identities])
    same_id = identities[:, None] == identities[None, :]
    same_video = videos[:, None] == videos[None, :]
    iu = np.triu_indices(len(identities), k=1)
    same_id_pair = same_id[iu]
    same_video_pair = same_video[iu]
    sim_pair = sims[iu]

    intra = sim_pair[same_id_pair]
    inter_same_video = sim_pair[~same_id_pair & same_video_pair]
    inter_diff_video = sim_pair[~same_id_pair & ~same_video_pair]
    return intra, inter_same_video, inter_diff_video


def silhouette_same_video(embeds, identities):
    """Silhouette restringido a comparar cada punto solo contra identidades
    del MISMO video -- promedia (ponderado por tamaño) el silhouette de
    sklearn calculado por separado dentro de cada grupo de video que aporte
    >=2 identidades distintas a la muestra (con una sola identidad no hay
    "otro cluster" del mismo video con el que comparar, ese video se excluye
    del promedio). Es el análogo, para silhouette, de restringir inter a
    inter_same_video en intra_inter_similarities."""
    identities = np.asarray(identities)
    videos = np.asarray([_video_of(i) for i in identities])
    scores, weights = [], []
    for video in np.unique(videos):
        mask = videos == video
        sub_ids = identities[mask]
        if len(set(sub_ids.tolist())) < 2:
            continue
        scores.append(silhouette_score(embeds[mask], sub_ids, metric="cosine"))
        weights.append(int(mask.sum()))
    if not scores:
        return float("nan")
    return float(np.average(scores, weights=weights))


def cmc_same_video(embeds, identities, max_k=20):
    """Rank-k accuracy / curva CMC (Cumulative Matching Characteristic) --
    la métrica estándar de retrieval en ReID, más directa/intuitiva que
    silhouette: para cada observación, se rankea por similitud coseno TODAS
    las demás observaciones del MISMO video (excluyéndose a sí misma -- ver
    intra_inter_similarities para por qué same-video es el caso relevante)
    y se chequea si alguna identidad correcta aparece entre las primeras k.
    rank-1 = "el vecino MÁS cercano ya es la identidad correcta" -- la
    pregunta concreta que hace este análisis, sin pasar por ningún umbral
    ni por la escala absoluta de similitud (a diferencia del gap), y sin la
    abstracción del silhouette (que promedia distancias intra/inter en vez
    de mirar el ranking real observación por observación).

    Devuelve (cmc, n_valid): `cmc` es un array (max_k,) con la accuracy
    acumulada en cada rank; `n_valid` es cuántas observaciones tenían
    alguna galería same-video no vacía (excluye video con una sola
    identidad Y un solo track, si lo hubiera -- no debería pasar con
    obs_per_identity>=2)."""
    identities = np.asarray(identities)
    videos = np.asarray([_video_of(i) for i in identities])
    norm = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    sims = norm @ norm.T
    n = len(identities)

    hits_at_k = np.zeros(max_k, dtype=np.float64)
    n_valid = 0
    for i in range(n):
        gallery_mask = videos == videos[i]
        gallery_mask[i] = False
        gallery_idx = np.where(gallery_mask)[0]
        if len(gallery_idx) == 0:
            continue
        n_valid += 1
        order = np.argsort(-sims[i, gallery_idx])
        ranked_correct = (identities[gallery_idx][order] == identities[i]).astype(np.float64)
        cum_hit = (np.cumsum(ranked_correct) > 0).astype(np.float64)
        k_cap = min(max_k, len(cum_hit))
        hits_at_k[:k_cap] += cum_hit[:k_cap]
        if k_cap < max_k:
            hits_at_k[k_cap:] += cum_hit[-1]  # galería más chica que max_k -- se mantiene el último valor alcanzado
    cmc = hits_at_k / max(n_valid, 1)
    return cmc, n_valid


def run_gallery_argmax_analysis(index, model, device, embedding_cache, num_episodes, seed, max_bank_size, num_track_history, query_wide_factor, batch_size=4):
    """Replica el mecanismo real de decisión (ver `_calibrate_threshold` en
    analyze_dinov2_nn_baseline.py, mismo criterio de accuracy binaria):
    arma `num_episodes` episodios de galería vía ReIDEpisodeDataset
    (force_reid=False -- mezcla natural de REAL_REID/SYNTHETIC_SPLIT/
    NEW_TRACK) y guarda, por episodio, la similitud del MEJOR candidato del
    banco (`best_sim`) y si la query tenía de verdad un match en ese banco
    (`is_match`). `model`/`embedding_cache`: pasar el encoder vivo (DINOv2,
    embedding_cache=None) O un cache precomputado (C4/P4-FPN, model=None) --
    `_embed_batch` ya distingue ambos casos por la forma del batch."""
    loader = _build_loader(
        index, num_episodes, max_bank_size, num_track_history, query_wide_factor,
        force_reid=False, seed=seed, batch_size=batch_size, embedding_cache=embedding_cache,
    )
    best_sims, is_match = [], []
    for batch in loader:
        track_summary, query_embeds, bank_mask, S = _embed_batch(model, batch, device, head=None)
        target = batch["target"]
        for b in range(query_embeds.shape[0]):
            _, _, _, best_sim = _best_candidate(query_embeds[b], track_summary[b], bank_mask[b])
            best_sims.append(best_sim)
            is_match.append(int(target[b].item()) < S)
    best_sims = np.array(best_sims)
    is_match = np.array(is_match)

    best_thr, best_acc = 0.0, -1.0
    for t in np.unique(best_sims):
        acc = ((best_sims >= t) == is_match).mean()
        if acc > best_acc:
            best_thr, best_acc = float(t), float(acc)
    auc = float(roc_auc_score(is_match, best_sims)) if len(set(is_match.tolist())) > 1 else float("nan")

    return {
        "best_sims": best_sims, "is_match": is_match,
        "match_mean": float(best_sims[is_match].mean()) if is_match.any() else float("nan"),
        "no_match_mean": float(best_sims[~is_match].mean()) if (~is_match).any() else float("nan"),
        "n_match": int(is_match.sum()), "n_no_match": int((~is_match).sum()),
        "threshold": best_thr, "accuracy_at_threshold": best_acc, "auc": auc,
    }


def plot_comparison(results, output_dir, results_argmax=None):
    """Fila de arriba: t-SNE por identidad (igual siempre, lectura
    cualitativa). Fila de abajo: DEPENDE de si se pasa `results_argmax`
    (ver run_gallery_argmax_analysis) --
      - Si se pasa (--gallery-episodes > 0): panel best-of-galería
        (best_sim separado por match/no-match, accuracy@umbral real y AUC)
        -- la métrica de separabilidad que en verdad predice el resultado
        downstream (ver conversación: el gap intra/inter NO lo predice,
        C4 sale con gap chico pero es el mejor o casi-mejor en accuracy/AUC).
      - Si no se pasa: fallback al histograma intra/inter viejo (gap +
        silhouette) -- se mantiene solo para correr el script sin pagar el
        costo extra de --gallery-episodes, no es la lectura recomendada
        de separabilidad."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 11))
    if len(results) == 1:
        axes = axes.reshape(2, 1)
    argmax_by_name = dict(results_argmax) if results_argmax is not None else None

    identities_all = results[0][1]["identities"]
    unique_ids = sorted(set(identities_all))
    cmap = plt.get_cmap("tab20", len(unique_ids))
    color_map = {ident: cmap(i) for i, ident in enumerate(unique_ids)}

    for col, (name, r) in enumerate(results):
        colors = [color_map[i] for i in r["identities"]]
        ax = axes[0, col]
        ax.scatter(r["tsne"][:, 0], r["tsne"][:, 1], c=colors, s=28, alpha=0.85, edgecolors="none")
        ax.set_title(f"{name} -- t-SNE by identity")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1, col]
        if argmax_by_name is not None:
            ra = argmax_by_name[name]
            match_sims = ra["best_sims"][ra["is_match"]]
            no_match_sims = ra["best_sims"][~ra["is_match"]]
            ax.hist(match_sims, bins=30, alpha=0.6, label=f"match (n={len(match_sims)})", color="tab:green", density=True)
            ax.hist(no_match_sims, bins=30, alpha=0.6, label=f"no-match (n={len(no_match_sims)})", color="tab:red", density=True)
            ax.axvline(ra["threshold"], color="black", linestyle="--", linewidth=1.5, label=f"umbral óptimo={ra['threshold']:.3f}")
            ax.set_title(f"{name} -- best-of-galería\nacc@umbral={ra['accuracy_at_threshold']:.3f} AUC={ra['auc']:.3f}")
            ax.set_xlabel("similitud coseno del mejor candidato")
            ax.legend(fontsize=8)
        else:
            ax.hist(r["intra"], bins=30, alpha=0.6, label=f"intra-ID (n={len(r['intra'])})", color="tab:green", density=True)
            ax.hist(r["inter_same_video"], bins=30, alpha=0.6, label=f"inter-ID, same video (n={len(r['inter_same_video'])})", color="tab:orange", density=True)
            ax.hist(r["inter_diff_video"], bins=30, alpha=0.6, label=f"inter-ID, diff video (n={len(r['inter_diff_video'])})", color="tab:red", density=True)
            ax.axvline(r["intra"].mean(), color="tab:green", linestyle="--", linewidth=1.5)
            if len(r["inter_same_video"]):
                ax.axvline(r["inter_same_video"].mean(), color="tab:orange", linestyle="--", linewidth=1.5)
            ax.axvline(r["inter_diff_video"].mean(), color="tab:red", linestyle="--", linewidth=1.5)
            gap_same_video = r["intra"].mean() - r["inter_same_video"].mean() if len(r["inter_same_video"]) else float("nan")
            ax.set_title(
                f"{name} -- cosine similarity\n"
                f"intra={r['intra'].mean():.3f} inter_same_video={r['inter_same_video'].mean() if len(r['inter_same_video']) else float('nan'):.3f}\n"
                f"gap(same video)={gap_same_video:.3f} silhouette(same video)={r['silhouette_same_video']:.3f}"
            )
            ax.set_xlabel("cosine similarity")
            ax.legend(fontsize=8)

    fig.tight_layout()
    combined_path = os.path.join(output_dir, "dinov2_vs_p4_comparison.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Combined figure saved to {combined_path}")


def plot_cmc_curve(cmc_results, output_dir):
    """Curva CMC (rank-k accuracy same-video), las 3 líneas superpuestas en
    un solo panel -- ver cmc_same_video."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, cmc, n_valid in cmc_results:
        ks = np.arange(1, len(cmc) + 1)
        ax.plot(ks, cmc, marker="o", markersize=3, label=f"{name} (rank-1={cmc[0]:.3f}, n={n_valid})")
    ax.set_xlabel("rank k")
    ax.set_ylabel("accuracy acumulada (same-video)")
    ax.set_title("CMC: ¿aparece la identidad correcta entre los k vecinos más cercanos?")
    # zoom al rango real de los datos -- con rank-1 ya arriba de 0.9 en los 3 encoders
    # (típico en ReID same-video, la galería real es chica), un eje 0-1 aplasta las
    # curvas y no deja ver la diferencia; -0.02/+0.02 de margen sobre el rango real.
    y_min = min(cmc.min() for _, cmc, _ in cmc_results)
    ax.set_ylim(max(0, y_min - 0.02), 1.005)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "cmc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] Curva CMC guardada en {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--kfold-pool", action="store_true",
        help="Ignora --split -- usa el pool de 64 secuencias de --kfold-cv (kfold_utils.list_all_sequences, "
             "train+test físicos, val EXCLUIDO a propósito -- ver docstring de kfold_utils.py). Mismo criterio "
             "que --kfold-pool en precompute_p4_roi_cache.py, así que los caches *_kfold_pool.pt son compatibles "
             "directo (mismos sequence_dirs -> mismos image_path -> mismas keys de cache).",
    )
    parser.add_argument("--p4-cache", required=True, help="PrecomputedEmbeddingCache de C4 crudo (--layer c4, ver precompute_p4_roi_cache.py).")
    parser.add_argument("--p4fpn-cache", default=None, help="[opcional] PrecomputedEmbeddingCache de P4 REAL (--layer p4fpn) -- agrega una 3ra columna a la comparación.")
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument("--num-identities", type=int, default=15)
    parser.add_argument("--obs-per-identity", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Default cpu a propósito -- no competir por GPU con otras corridas.")
    parser.add_argument("--output-dir", default="embedding_analysis")
    parser.add_argument("--cmc-max-k", type=int, default=20, help="Rank máximo de la curva CMC same-video (ver cmc_same_video).")
    parser.add_argument(
        "--gallery-episodes", type=int, default=0,
        help="Si > 0, corre además el análisis best-of-galería (ver run_gallery_argmax_analysis) -- "
             "mismo mecanismo que _calibrate_threshold en analyze_dinov2_nn_baseline.py, N episodios.",
    )
    parser.add_argument("--gallery-seed", type=int, default=123, help="Mismo default que --calibration-seed en analyze_dinov2_nn_baseline.py.")
    parser.add_argument("--max-bank-size", type=int, default=16, help="[--gallery-episodes] mismo default que analyze_dinov2_nn_baseline.py.")
    parser.add_argument("--num-track-history", type=int, default=4, help="[--gallery-episodes] mismo default que analyze_dinov2_nn_baseline.py.")
    parser.add_argument(
        "--query-wide-factor", type=float, default=9.0,
        help="Factor de expansión de la vista amplia de DINOv2 (ver compute_dinov2_embeddings) y, con "
             "--gallery-episodes, de la query real -- mismo default que analyze_dinov2_nn_baseline.py.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.kfold_pool:
        from kfold_utils import list_all_sequences
        seq_dirs = list_all_sequences()
        print(f"[INFO] --kfold-pool: {len(seq_dirs)} secuencias (train+test físicos, val excluido -- ver kfold_utils.py)")
        index = build_track_index(None, sequence_dirs=seq_dirs)
    else:
        data_root, seq_dirs = resolve_sequence_dirs(None, args.split)
        index = build_track_index(data_root, sequence_dirs=seq_dirs)
    samples = sample_identities(index, args.num_identities, args.obs_per_identity, args.seed)
    identities = [ident for ident, _ in samples]
    pool_label = "kfold_pool (64 seq)" if args.kfold_pool else args.split
    print(f"[INFO] {len(samples)} observaciones de {len(set(identities))} identidades (pool={pool_label})")

    videos_per_identity = {ident: _video_of(ident) for ident in set(identities)}
    ids_per_video = {}
    for ident, video in videos_per_identity.items():
        ids_per_video.setdefault(video, []).append(ident)
    n_videos_with_pair = sum(1 for ids in ids_per_video.values() if len(ids) >= 2)
    print(
        f"[INFO] {len(ids_per_video)} videos distintos entre las identidades muestreadas; "
        f"{n_videos_with_pair} aportan >=2 identidades (o sea, tienen pares inter_same_video). "
        f"Identidades/video: {sorted((len(v) for v in ids_per_video.values()), reverse=True)}"
    )
    if n_videos_with_pair == 0:
        print(
            "[WARN] Ningún video aporta 2+ identidades -- inter_same_video quedará vacío, "
            "el gap/silhouette 'same video' no se puede calcular con esta muestra. "
            "Subí --num-identities o --obs-per-identity, o revisá cuántos tracks elegibles "
            "tiene cada video en este split."
        )

    dinov2_embeds = compute_dinov2_embeddings(samples, args.dinov2_model_name, args.device, args.query_wide_factor)
    p4_embeds = compute_cache_embeddings(samples, args.p4_cache)

    sources = [("DINOv2", dinov2_embeds), ("C4", p4_embeds)]
    if args.p4fpn_cache:
        sources.append(("P4 (FPN C5+C4)", compute_cache_embeddings(samples, args.p4fpn_cache)))

    results = []
    cmc_results = []
    summary = {}
    for name, embeds in sources:
        intra, inter_same_video, inter_diff_video = intra_inter_similarities(embeds, identities)
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeds) // 3), random_state=args.seed, init="pca").fit_transform(embeds)
        sil_all = silhouette_score(embeds, identities, metric="cosine")
        sil_same_video = silhouette_same_video(embeds, identities)
        cmc, cmc_n_valid = cmc_same_video(embeds, identities, max_k=args.cmc_max_k)
        cmc_results.append((name, cmc, cmc_n_valid))
        results.append((name, {
            "tsne": tsne, "identities": identities, "intra": intra,
            "inter_same_video": inter_same_video, "inter_diff_video": inter_diff_video,
            "silhouette_same_video": sil_same_video,
        }))
        gap_same_video = float(intra.mean() - inter_same_video.mean()) if len(inter_same_video) else float("nan")
        gap_diff_video = float(intra.mean() - inter_diff_video.mean())
        summary[name] = {
            "intra_mean": float(intra.mean()),
            "inter_same_video_mean": float(inter_same_video.mean()) if len(inter_same_video) else None,
            "inter_diff_video_mean": float(inter_diff_video.mean()),
            "gap_same_video": gap_same_video,
            "gap_diff_video": gap_diff_video,
            "silhouette_same_video": sil_same_video,
            "silhouette_all_videos": float(sil_all),
            "cmc_same_video": {"rank1": float(cmc[0]), "rank5": float(cmc[min(4, len(cmc) - 1)]), "rank10": float(cmc[min(9, len(cmc) - 1)]), "n_valid": cmc_n_valid, "curve": cmc.tolist()},
        }
        print(
            f"[{name}] intra={intra.mean():.4f} "
            f"inter_same_video={inter_same_video.mean() if len(inter_same_video) else float('nan'):.4f} "
            f"inter_diff_video={inter_diff_video.mean():.4f} "
            f"gap_same_video={gap_same_video:.4f} gap_diff_video={gap_diff_video:.4f} "
            f"silhouette_same_video={sil_same_video:.4f} silhouette_all_videos={sil_all:.4f} "
            f"rank1={cmc[0]:.4f} rank5={cmc[min(4, len(cmc) - 1)]:.4f} rank10={cmc[min(9, len(cmc) - 1)]:.4f} (n={cmc_n_valid})"
        )
    plot_cmc_curve(cmc_results, args.output_dir)

    results_argmax = None
    if args.gallery_episodes > 0:
        print(f"[INFO] Corriendo análisis best-of-galería ({args.gallery_episodes} episodios, seed={args.gallery_seed})...")
        from transformers import AutoModel
        dinov2_model = AutoModel.from_pretrained(args.dinov2_model_name).to(args.device).eval()
        for p in dinov2_model.parameters():
            p.requires_grad = False

        gallery_sources = [("DINOv2", dinov2_model, None), ("C4", None, PrecomputedEmbeddingCache.load(args.p4_cache))]
        if args.p4fpn_cache:
            gallery_sources.append(("P4 (FPN C5+C4)", None, PrecomputedEmbeddingCache.load(args.p4fpn_cache)))

        results_argmax = []
        for name, model, cache in gallery_sources:
            r = run_gallery_argmax_analysis(
                index, model, args.device, cache, args.gallery_episodes, args.gallery_seed,
                args.max_bank_size, args.num_track_history, args.query_wide_factor,
            )
            results_argmax.append((name, r))
            summary.setdefault(name, {})["gallery_argmax"] = {
                "n_match": r["n_match"], "n_no_match": r["n_no_match"],
                "match_mean": r["match_mean"], "no_match_mean": r["no_match_mean"],
                "threshold": r["threshold"], "accuracy_at_threshold": r["accuracy_at_threshold"], "auc": r["auc"],
            }
            print(
                f"[{name}][gallery] n_match={r['n_match']} n_no_match={r['n_no_match']} "
                f"match_mean={r['match_mean']:.4f} no_match_mean={r['no_match_mean']:.4f} "
                f"umbral_optimo={r['threshold']:.4f} acc={r['accuracy_at_threshold']:.4f} auc={r['auc']:.4f}"
            )

    plot_comparison(results, args.output_dir, results_argmax=results_argmax)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Resumen numérico guardado en {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
