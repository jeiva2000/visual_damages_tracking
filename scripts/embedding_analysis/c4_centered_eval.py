"""
"C4 + centrado": recalibra el umbral y reevalua los mismos casos estrictos
de la Tabla 2 (all_methods_same_cases_pct), pero restando la media global
de train (por fold, sin fuga) antes de comparar coseno -- ver conversacion
sobre anisotropia (isotropy_diagnostic.py). No reentrena nada, no toca la
cabeza -- es una correccion cerrada (centrar+renormalizar) sobre el mismo
embedding C4 crudo cacheado.

Reusa el MISMO fold_assignment.json y el MISMO reid_gap_report.json que ya
uso methods_on_strict_cases.py para "C4 puro" -- resultado 100% comparable
fila a fila.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'reid_lib'))  # library modules
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from analyze_dinov2_nn_baseline import _calibrate_threshold
from compare_kfold_cases import _predict_case
from kfold_utils import list_all_sequences
from reid_dataloader import PrecomputedEmbeddingCache
from train_track_query_reid import build_track_index

CACHE_PATH = "p4_roi_cache_kfold_pool.pt"
FOLD_ASSIGNMENT = "p4_roi_kfold_baseline/fold_assignment.json"
BASE_REIDGAP_DIR = "kfold_dinov2_raw"
TRACKERS = ["bytetrack", "botsort"]
DEVICE = "cpu"
MAX_BANK_SIZE = 16
NUM_TRACK_HISTORY = 4
QUERY_WIDE_FACTOR = 9.0
SEED = 0


class CenteredEmbeddingCache:
    """Misma interfaz que PrecomputedEmbeddingCache (.get(obs) -> (S,D)),
    pero resta `mu` (media de train de ESE fold, ver abajo) a cada escala
    antes de devolver -- _cosine_sim ya renormaliza al comparar, asi que no
    hace falta renormalizar aca."""

    def __init__(self, raw_cache: PrecomputedEmbeddingCache, mu: np.ndarray):
        self.raw_cache = raw_cache
        self.mu = torch.from_numpy(mu).float()

    def get(self, obs):
        return self.raw_cache.get(obs) - self.mu.unsqueeze(0)


def compute_mu(raw_cache, seq_dirs):
    index = build_track_index(None, sequence_dirs=seq_dirs)
    vecs = []
    for track in index.tracks:
        for obs in track.observations:
            key = (obs.image_path, tuple(obs.bbox))
            cached = raw_cache.embeddings.get(key)
            if cached is None:
                continue
            vecs.append(cached.numpy())  # (S, D)
    all_vecs = np.concatenate(vecs, axis=0)  # (N*S, D)
    return all_vecs.mean(axis=0)  # (D,)


def main():
    print("[INFO] cargando cache crudo...")
    raw_cache = PrecomputedEmbeddingCache.load(CACHE_PATH)

    with open(FOLD_ASSIGNMENT) as f:
        folds_raw = json.load(f)
    folds = {int(k): v for k, v in folds_raw.items()}
    k = len(folds)
    seq_lookup = {os.path.basename(s.rstrip("/")): s for s in list_all_sequences()}

    strict_cases_by_tracker = {}
    for tracker in TRACKERS:
        report_path = os.path.join(BASE_REIDGAP_DIR, f"reid_gap_{tracker}_kfold_all", "reid_gap_report.json")
        with open(report_path) as f:
            all_cases = json.load(f)
        strict_cases_by_tracker[tracker] = [c for c in all_cases if c["outcome"] != "unevaluated"]
        print(f"[INFO] {tracker}: {len(strict_cases_by_tracker[tracker])} casos estrictos")

    results = {t: {"n_correct": 0, "n_evaluated": 0, "n_skipped": 0} for t in TRACKERS}
    fold_thresholds = []

    for fold_idx in range(k):
        val_fold = (fold_idx + 1) % k
        test_seqs = folds[fold_idx]          # ya son rutas completas
        train_seq_dirs = [s for i, seqs in folds.items() if i not in (fold_idx, val_fold) for s in seqs]
        test_seq_names = set(os.path.basename(s.rstrip("/")) for s in test_seqs)

        print(f"\n[FOLD {fold_idx}] train={len(train_seq_dirs)} test={len(test_seqs)}")
        mu = compute_mu(raw_cache, train_seq_dirs)
        centered_cache = CenteredEmbeddingCache(raw_cache, mu)

        train_index = build_track_index(None, sequence_dirs=train_seq_dirs)
        cal_args = SimpleNamespace(
            num_calibration_episodes=300, max_bank_size=MAX_BANK_SIZE,
            num_track_history=NUM_TRACK_HISTORY, query_wide_factor=QUERY_WIDE_FACTOR,
            force_reid=False, calibration_seed=123, batch_size=4,
            embedding_cache=centered_cache, device=DEVICE,
        )
        threshold = _calibrate_threshold(None, train_index, cal_args, head=None)
        fold_thresholds.append(threshold)
        print(f"[FOLD {fold_idx}] umbral (centrado) = {threshold:.4f}")

        test_seq_dirs = [seq_lookup[n] for n in test_seq_names]
        test_index = build_track_index(None, sequence_dirs=test_seq_dirs)

        for tracker in TRACKERS:
            fold_cases = [c for c in strict_cases_by_tracker[tracker] if c["sequence"] in test_seq_names]
            for case in fold_cases:
                correct = _predict_case(
                    None, None, test_index, case, threshold, DEVICE,
                    MAX_BANK_SIZE, NUM_TRACK_HISTORY, QUERY_WIDE_FACTOR, SEED,
                    embedding_cache=centered_cache,
                )
                if correct is None:
                    results[tracker]["n_skipped"] += 1
                    continue
                results[tracker]["n_evaluated"] += 1
                if correct:
                    results[tracker]["n_correct"] += 1

    print(f"\n[UMBRALES] centrado, por fold: {[round(t,4) for t in fold_thresholds]}")
    print(f"[UMBRALES] media = {np.mean(fold_thresholds):.4f}  (referencia: crudo era ~0.9352)")

    print("\n[RESULTADO FINAL] C4 + centrado, mismos casos estrictos que la Tabla 2:")
    for tracker in TRACKERS:
        r = results[tracker]
        acc = r["n_correct"] / r["n_evaluated"] if r["n_evaluated"] else float("nan")
        print(f"  {tracker}: {r['n_correct']}/{r['n_evaluated']} = {100*acc:.1f}%  (saltados={r['n_skipped']})")

    with open("c4_centered_results.json", "w") as f:
        json.dump({"fold_thresholds": fold_thresholds, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
