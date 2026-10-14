"""
Utilidades para re-particionar dataset_split_2 en k folds a nivel de
secuencia (nunca mezclar tracks de la misma secuencia entre folds), en vez
del train/val/test fijo -- ver train_track_query_reid.py:SPLIT_OVERRIDES,
que aplica solo cuando se resuelve por --split, no acá: `list_all_sequences`
junta las secuencias de los splits físicos train+test (NO val) y las trata
como un solo pool, ignorando SPLIT_OVERRIDES a propósito (estamos
re-particionando desde cero, no reproduciendo el split fijo).

`val` queda EXCLUIDO del pool a propósito: es el split que usa
eval_tracking/sweep.py para tunear los hiperparámetros de ByteTrack/BotSort
(track_thresh/match_thresh/track_buffer/etc.) -- si se incluyera en el pool,
esas secuencias terminarían como test de varios folds (confirmado
empíricamente: 4/6 folds en una corrida real), contaminando la medición de
"held-out test" con datos que el tracker ya vio durante su propio tuneo. El
tracker es un componente FIJO compartido por todas las configs de ReID que
se comparan en k-fold (no se retunea por fold), así que su set de tuneo
debe quedar fuera del pool por completo, no rotar dentro de él.

OJO: no alcanza con excluir el directorio físico DATA_ROOTS["val"] -- por
SPLIT_OVERRIDES, `resolve_sequence_dirs(None, "val")` puede resolver además
secuencias que viven FÍSICAMENTE en train/ o test/ pero quedaron
reasignadas LÓGICAMENTE a val (confirmado: video_SrS9HJyuASbyp3rpyqs_11
aparece en train/test pero se resuelve como val). Por eso se excluye por
NOMBRE contra el val ya resuelto (post-overrides), no por directorio de
origen."""
import json
import os
import random
from typing import Dict, List

from train_track_query_reid import DATA_ROOTS, build_track_index, discover_sequences, resolve_sequence_dirs


def list_all_sequences() -> List[str]:
    _, val_seqs = resolve_sequence_dirs(None, "val")
    val_names = {os.path.basename(s.rstrip("/")) for s in (val_seqs or [])}

    seqs: List[str] = []
    for root in DATA_ROOTS.values():
        for s in discover_sequences(root):
            if os.path.basename(s.rstrip("/")) not in val_names:
                seqs.append(s)
    return seqs


def sequence_has_real_gap(seq_dir: str, min_gap_frames: int = 3) -> bool:
    index = build_track_index(None, sequence_dirs=[seq_dir])
    return any(track.find_real_gaps(min_gap_frames) for track in index.tracks)


def assign_folds(seq_dirs: List[str], k: int, seed: int) -> Dict[int, List[str]]:
    """Separa seq_dirs en has_gap/no_gap (ver sequence_has_real_gap), baraja
    cada grupo por separado con random.Random(seed) y reparte round-robin en
    k folds -- así cada fold queda con una proporción similar de secuencias
    con reid real, en vez de que la casualidad del shuffle global concentre
    todas en un solo fold (como pasaba con el split fijo train/val, ver
    SPLIT_OVERRIDES en train_track_query_reid.py)."""
    if k < 2:
        raise ValueError(f"k debe ser >= 2, se pidió k={k}")

    has_gap = [s for s in seq_dirs if sequence_has_real_gap(s)]
    no_gap = [s for s in seq_dirs if s not in has_gap]

    rng = random.Random(seed)
    rng.shuffle(has_gap)
    rng.shuffle(no_gap)

    folds: Dict[int, List[str]] = {i: [] for i in range(k)}
    for i, s in enumerate(has_gap):
        folds[i % k].append(s)
    for i, s in enumerate(no_gap):
        folds[i % k].append(s)

    for fold_idx, fold_seqs in folds.items():
        if len(fold_seqs) < 2:
            raise ValueError(
                f"Fold {fold_idx} quedó con {len(fold_seqs)} secuencia(s) (< 2, mínimo de "
                f"HierarchicalPKSampler con P=2). Reducí --num-folds o revisá el dataset."
            )
    n_folds_without_gap = sum(1 for fold_seqs in folds.values() if not any(s in has_gap for s in fold_seqs))
    if n_folds_without_gap:
        print(
            f"[WARN] assign_folds: {n_folds_without_gap}/{k} folds sin ninguna secuencia con reid "
            f"real (has_gap={len(has_gap)} secuencias en total para repartir entre {k} folds)."
        )

    return folds


def save_fold_assignment(assignment: Dict[int, List[str]], path: str) -> None:
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in assignment.items()}, f, indent=2)


def load_fold_assignment(path: str) -> Dict[int, List[str]]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}
