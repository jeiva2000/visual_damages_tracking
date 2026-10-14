"""
Diagnostico de anisotropia/"representation degeneration" sobre los
embeddings C4 crudos (backbone.res_layers.2, ROI-Align 3 escalas) ya
cacheados en p4_roi_cache_kfold_pool.pt -- ver conversacion: las
similitudes coseno quedan comprimidas cerca de 1.0 (0.85-0.99) aunque el
rank-1 discrimine bien (0.989). Hipotesis: hay una direccion "promedio"
compartida por (casi) todos los embeddings que domina la similitud cruda.

Mide:
  1. ||media|| de los vectores unitarios (0 = isotropico perfecto, 1 =
     todos apuntan exactamente igual).
  2. Espectro de PCA (autovalores de la covarianza) -> participation ratio
     (dimensionalidad efectiva) y % de varianza en las primeras k
     componentes.
  3. Similitud coseno par-a-par ANTES/DESPUES de restar la media global y
     re-normalizar ("all-but-the-top" liviano, Mu & Viswanath 2018),
     separado en pares MISMA identidad vs DISTINTA identidad -- para ver
     si la correccion "descomprime" los pares distintos sin destruir la
     separacion de los pares iguales.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'reid_lib'))  # library modules
import os
import random
import sys

import numpy as np
import torch

from kfold_utils import list_all_sequences
from reid_dataloader import PrecomputedEmbeddingCache
from train_track_query_reid import build_track_index

CACHE_PATH = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking/p4_roi_cache_kfold_pool.pt"
SCALE_IDX = 2  # 0=amplio, 1=intermedio, 2=ajustado (ver C4_PADDING_FACTORS)
SEED = 0
N_RANDOM_PAIRS = 30000


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("[INFO] cargando cache...")
    cache = PrecomputedEmbeddingCache.load(CACHE_PATH)
    print(f"[INFO] {len(cache.embeddings)} observaciones cacheadas")

    print("[INFO] cargando indice del pool de 64 secuencias...")
    seq_dirs = list_all_sequences()
    index = build_track_index(None, sequence_dirs=seq_dirs)

    vecs = []
    identity_ids = []
    video_ids = []
    n_missing = 0
    for track in index.tracks:
        for obs in track.observations:
            key = (obs.image_path, tuple(obs.bbox))
            cached = cache.embeddings.get(key)
            if cached is None:
                n_missing += 1
                continue
            vecs.append(cached[SCALE_IDX].numpy())
            identity_ids.append(track.identity_id)
            video_ids.append(track.video_id)
    print(f"[INFO] {len(vecs)} vectores usados ({n_missing} sin cache), dim={len(vecs[0])}")

    X = np.stack(vecs, axis=0).astype(np.float64)  # (N, D), ya unitarios
    N, D = X.shape
    ids = np.array(identity_ids)
    vids = np.array(video_ids)

    # ---- 1. norma de la media ----
    mu = X.mean(axis=0)
    mu_norm = np.linalg.norm(mu)
    print(f"\n[1] ||media de los vectores unitarios|| = {mu_norm:.4f}  (0=isotropico perfecto, 1=degenerado total)")
    print(f"    referencia: para N={N} vectores unitarios ALEATORIOS en D={D}, "
          f"||media|| esperado ~ 1/sqrt(N) = {1/np.sqrt(N):.4f}")

    # ---- 2. espectro PCA ----
    Xc = X - mu
    cov = (Xc.T @ Xc) / N  # (D, D)
    eigvals = np.linalg.eigvalsh(cov)[::-1]  # descendente
    eigvals = np.clip(eigvals, 0, None)
    total_var = eigvals.sum()
    participation_ratio = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    print(f"\n[2] Participation ratio (dimensionalidad efectiva) = {participation_ratio:.1f} / {D} "
          f"({100*participation_ratio/D:.1f}% del total)")
    for k in [1, 2, 5, 10, 20, 50, 100]:
        frac = eigvals[:k].sum() / total_var
        print(f"    varianza explicada por las primeras {k:3d} componentes: {100*frac:.1f}%")

    # ---- 3. similitud coseno par-a-par, antes/despues de restar la media ----
    # restrict_same_video=True -- SOLO pares del mismo video (la condicion
    # real de produccion, ver same_video_only=True en TrackerFailureReIDDataset)
    def pair_stats(Xnorm, label, restrict_same_video=False):
        rng = np.random.RandomState(SEED)
        n_draw = N_RANDOM_PAIRS * (20 if restrict_same_video else 1)  # mas draws, se descartan la mayoria (distinto video)
        idx_a = rng.randint(0, N, n_draw)
        idx_b = rng.randint(0, N, n_draw)
        valid = idx_a != idx_b
        if restrict_same_video:
            valid = valid & (vids[idx_a] == vids[idx_b])
        idx_a, idx_b = idx_a[valid], idx_b[valid]
        if restrict_same_video:
            idx_a, idx_b = idx_a[:N_RANDOM_PAIRS], idx_b[:N_RANDOM_PAIRS]
        same = ids[idx_a] == ids[idx_b]
        diff = ~same
        sims = (Xnorm[idx_a] * Xnorm[idx_b]).sum(axis=1)
        s_same = sims[same]
        s_diff = sims[diff]
        print(f"    [{label}] pares MISMA identidad (n={len(s_same)}): mean={s_same.mean():.4f} std={s_same.std():.4f}")
        print(f"    [{label}] pares DISTINTA identidad (n={len(s_diff)}): mean={s_diff.mean():.4f} std={s_diff.std():.4f}")
        print(f"    [{label}] separacion (mean_same - mean_diff) = {s_same.mean() - s_diff.mean():.4f}")
        return s_same, s_diff

    print("\n[3a] ANTES de corregir (coseno crudo) -- pares de TODO el dataset (incluye distinto video):")
    pair_stats(X, "crudo/global")

    print("\n[3a-bis] ANTES de corregir (coseno crudo) -- pares SOLO del MISMO video (la condicion real del banco):")
    pair_stats(X, "crudo/mismo-video", restrict_same_video=True)

    print("\n[3b] DESPUES de restar la media global y re-normalizar (all-but-the-top, D=0 -> solo centrar) -- global:")
    X_centered = X - mu
    X_centered_norm = X_centered / np.linalg.norm(X_centered, axis=1, keepdims=True)
    pair_stats(X_centered_norm, "centrado/global")

    print("\n[3b-bis] DESPUES de centrar -- SOLO mismo video:")
    pair_stats(X_centered_norm, "centrado/mismo-video", restrict_same_video=True)

    # ---- 3c. remover ademas la primera componente principal ----
    print("\n[3c] DESPUES de restar la media Y proyectar fuera de la componente principal top-1 -- SOLO mismo video:")
    eigvecs = np.linalg.eigh(cov)[1][:, ::-1]  # columnas = autovectores, orden descendente
    top1 = eigvecs[:, 0:1]  # (D, 1)
    X_no_top1 = X_centered - (X_centered @ top1) @ top1.T
    X_no_top1_norm = X_no_top1 / np.linalg.norm(X_no_top1, axis=1, keepdims=True)
    pair_stats(X_no_top1_norm, "centrado_sin_top1/mismo-video", restrict_same_video=True)


if __name__ == "__main__":
    main()
