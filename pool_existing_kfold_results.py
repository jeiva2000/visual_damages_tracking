"""
Script chico, standalone: reconstruye los conteos crudos (n_identity_correct,
n_match_correct) a partir de los kfold_results.json YA GENERADOS por las
corridas oficiales (identity_acc/match_acc * n_total, redondeado -- son
fracciones exactas de enteros, así que la reconstrucción es exacta salvo
error de redondeo de punto flotante), y calcula la accuracy POOLEADA
(cross-validated prediction pooling, ver analyze_dinov2_nn_baseline_pooled_kfold.py)
con intervalo de Wilson -- sin volver a correr nada, cero cómputo GPU.

Uso:
    python3 pool_existing_kfold_results.py p4_roi_kfold_baseline/kfold_results.json C4
    python3 pool_existing_kfold_results.py p4fpn_roi_kfold_baseline/kfold_results.json P4-FPN
    python3 pool_existing_kfold_results.py kfold_dinov2_raw/kfold_results.json DINOv2
"""
import json
import math
import sys


def wilson_interval(x, n, z=1.96):
    if not n:
        return float("nan"), float("nan"), float("nan")
    p = x / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return p, (center - half) / denom, (center + half) / denom


def pool(path, tracker="bytetrack"):
    fold_results = json.load(open(path))
    for label, key_getter in [
        ("synthetic_test", lambda f: f["synthetic"]),
        (tracker, lambda f: f["trackers"][tracker]),
    ]:
        n_id_correct = 0
        n_id_total = 0
        n_m_correct = 0
        n_m_total = 0
        for f in fold_results:
            m = key_getter(f)
            n_id_total += m["n_identity_total"]
            n_id_correct += round(m["identity_acc"] * m["n_identity_total"])
            if m.get("match_acc") is not None:
                n_m_total += m["n_total"]
                n_m_correct += round(m["match_acc"] * m["n_total"])
        id_p, id_lo, id_hi = wilson_interval(n_id_correct, n_id_total)
        m_p, m_lo, m_hi = wilson_interval(n_m_correct, n_m_total) if n_m_total else (float("nan"),) * 3
        print(
            f"  {label:<16s} identity_acc={id_p:.4f} [{id_lo:.4f},{id_hi:.4f}] (n={n_id_total})  "
            f"match_acc={m_p:.4f} [{m_lo:.4f},{m_hi:.4f}] (n={n_m_total})"
        )


if __name__ == "__main__":
    path, name = sys.argv[1], sys.argv[2]
    print(f"[{name}] {path}")
    pool(path)
