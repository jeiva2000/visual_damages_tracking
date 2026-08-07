"""
train_track_query_reid.py

Entrenamiento de SetTrackReIDModule (set_reid_model.py) sobre episodios
generados por ReIDEpisodeDataset (reid_dataloader.py): matching set-to-set
de una detección nueva (query, una sola vista) contra una galería de tracks
perdidos (cada uno con varias vistas de su historial), con una clase extra
aprendida de "no match".

Construye el DamageTrackIndex desde un dataset con el mismo layout MOT-style
que usa el resto del repo (build_track_index.py / dinov2_reid_finetune.py):

    secuencia/gt/gt.txt   -> frame_id,track_id,x,y,w,h[,conf][,class][,?][,visibility]
    secuencia/img1/       -> frames ya extraídos, nombrados por número de frame

USO
---
python train_track_query_reid.py dump-batches \
    --split train \
    --outdir reid_batch_dump \
    --num-batches 3 --batch-size 2 --max-bank-size 8

python train_track_query_reid.py train --train-split train --val-split val

--split train/val/test resuelve la ruta contra DATA_ROOTS (mismo criterio
que build_track_index.py). Para apuntar a otro dataset (ej. uno de prueba
local), pasa --data_root explícito en su lugar.

DEPENDENCIAS
------------
pip install torch torchvision pillow
"""

import argparse
import json
import math
import os
import random
import shutil
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from reid_dataloader import (
    DamageTrackIndex,
    IdentityTrack,
    TrackObservation,
    ReIDEpisodeDataset,
    QueryCase,
    AUGMENTED_TRANSFORM,
    PrecomputedEmbeddingCache,
    collate_episodes,
    collate_episodes_with_video_ids,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from reid_track_query_module import DINOv2Encoder
from set_reid_model import SetTrackReIDModule
from hierarchical_pk_sampler import build_track_pk_sampler

# Misma convención que build_track_index.py: los splits del dataset viven en
# rutas fijas del servidor, así no hay que repetir la ruta completa a mano
# cada vez -- `--split train/val/test` alcanza. `--data_root` sigue
# disponible para apuntar a otro lado (ej. un dataset de prueba local).
DATA_ROOTS = {
    "train": "/mnt/Disk2/datasets/dataset_split_2/train",
    "val":   "/mnt/Disk2/datasets/dataset_split_2/val",
    "test":  "/mnt/Disk2/datasets/dataset_split_2/test",
}


# Reasignación LÓGICA de secuencias puntuales entre splits -- sin mover
# ninguna carpeta física de DATA_ROOTS (es la misma estructura que usan
# build_track_index.py/dinov2_reid_finetune.py, compartida con otros
# scripts del repo). Motivo: train tenía muy pocos tracks con una brecha
# temporal REAL (ver find_real_gaps/_try_real_reid en reid_dataloader.py --
# solo 23/56 secuencias con al menos uno), mientras val concentraba
# desproporcionadamente esos casos (las 8/8 secuencias de val tienen). Este
# swap le da a train una secuencia rica en reid real, y a val una de train
# (sin reid real) para no perder tamaño. Solo aplica cuando el split se
# resuelve por --split (ver discover_sequences_for_split) -- si se pasa
# --data_root explícito, no se toca nada de esto.
SPLIT_OVERRIDES: Dict[str, str] = {
    "video_3rntaZcqXY1h8ur5GcJ_11": "train",  # val -> train: 4 tracks, 4 con reid real (100%)
    "video_SrS9HJyuASbyp3rpyqs_11": "val",    # train -> val: 5 tracks, 0 con reid real (compensa tamaño)
}


def discover_sequences_for_split(split: str) -> List[str]:
    """Secuencias que EFECTIVAMENTE pertenecen a `split`, aplicando
    SPLIT_OVERRIDES -- reemplaza discover_sequences(DATA_ROOTS[split]) en
    los callers que resuelven por --split (no por --data_root explícito)."""
    own_root = DATA_ROOTS[split]
    seqs = [
        s for s in discover_sequences(own_root)
        if SPLIT_OVERRIDES.get(os.path.basename(s.rstrip("/")), split) == split
    ]
    for other_split, other_root in DATA_ROOTS.items():
        if other_split == split:
            continue
        for s in discover_sequences(other_root):
            if SPLIT_OVERRIDES.get(os.path.basename(s.rstrip("/"))) == split:
                seqs.append(s)
    return seqs


def resolve_sequence_dirs(data_root: Optional[str], split: Optional[str]) -> Tuple[str, Optional[List[str]]]:
    """Como resolve_data_root, pero además devuelve la lista de directorios
    de secuencia a usar. Si se pasa --data_root explícito, SPLIT_OVERRIDES
    no aplica (devuelve None -- build_track_index cae a discover_sequences
    normal sobre ese data_root). Si se resuelve por --split, aplica el
    override (ver discover_sequences_for_split)."""
    if data_root is not None:
        return data_root, None
    split = split or "train"
    return DATA_ROOTS[split], discover_sequences_for_split(split)


# --------------------------------------------------------------------------- #
# Construcción del DamageTrackIndex a partir de un dataset MOT-style
# (mismo formato/convenciones que build_track_index.py y dinov2_reid_finetune.py)
# --------------------------------------------------------------------------- #

def load_gt_mot(gt_path: str, conf_thresh: float, class_id: int, visibility_thresh: float = 0.0):
    """Parsea un gt.txt estilo MOT: frame_id,track_id,x,y,w,h[,conf][,class][,?][,visibility],
    bbox xywh top-left. OJO: en dataset_split_2 el gt.txt trae 10 columnas (hay
    un campo sin usar entre class y visibility), así que visibility es la
    ÚLTIMA columna (parts[9]), no parts[8] -- mismo criterio que
    dinov2_reid_finetune.py:load_gt_mot. visibility<=visibility_thresh se
    descarta siempre (visibility=0 = oclusión total)."""
    rows = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            visibility = float(parts[9]) if len(parts) > 9 else 1.0
            if conf < conf_thresh or w <= 0 or h <= 0:
                continue
            if class_id > 0 and cls != class_id:
                continue
            if visibility <= visibility_thresh:
                continue
            rows.append({
                "frame": frame_id, "id": track_id,
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                "part_label": cls,
            })
    return rows


def discover_sequences(data_root: str) -> List[str]:
    """Subcarpetas de data_root con gt/gt.txt + img1/, mismo layout que
    DATA_ROOTS en build_track_index.py."""
    seqs = []
    for name in sorted(os.listdir(data_root)):
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        if os.path.exists(os.path.join(seq_dir, "gt", "gt.txt")) and os.path.isdir(os.path.join(seq_dir, "img1")):
            seqs.append(seq_dir)
    return seqs


def _index_frames(img_dir: str) -> Dict[int, str]:
    """Mapea número de frame (nombre del archivo sin extensión) -> ruta
    absoluta, igual que en dinov2_reid_finetune.py / dinov2_reid_analysis.py."""
    frame_files: Dict[int, str] = {}
    for fname in os.listdir(img_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        try:
            idx = int(stem)
        except ValueError:
            continue
        frame_files[idx] = os.path.abspath(os.path.join(img_dir, fname))
    return frame_files


def build_track_index(
    data_root: str,
    gt_conf_thresh: float = 0.0,
    gt_class_id: int = -1,
    gt_visibility_thresh: float = 0.0,
    min_observations: int = 2,
    sequence_dirs: Optional[List[str]] = None,
) -> DamageTrackIndex:
    """Recorre todas las secuencias de `data_root` (o `sequence_dirs`, si se
    pasa -- ver resolve_sequence_dirs/SPLIT_OVERRIDES) y arma un IdentityTrack
    por (secuencia, track_id) del gt.txt, con una TrackObservation por fila
    (frame ya resuelto a ruta absoluta). part_label usa la columna `class`
    del gt.txt -- es la única señal categórica disponible hoy; si más
    adelante hay una anotación real de "parte del vehículo", reemplazar acá.
    Tracks con menos de `min_observations` filas se descartan (no aportan
    nada útil ni como positivo ni como distractor con historia)."""
    all_tracks: List[IdentityTrack] = []
    seq_dirs = sequence_dirs if sequence_dirs is not None else discover_sequences(data_root)
    if not seq_dirs:
        raise FileNotFoundError(f"No se encontraron secuencias válidas (gt/gt.txt + img1/) en {data_root}")

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir.rstrip("/"))
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        img_dir = os.path.join(seq_dir, "img1")

        rows = load_gt_mot(gt_path, gt_conf_thresh, gt_class_id, gt_visibility_thresh)
        if not rows:
            print(f"[WARN] {seq_name}: sin anotaciones tras los filtros, se omite")
            continue
        frame_files = _index_frames(img_dir)

        by_track: Dict[int, List[dict]] = {}
        for row in rows:
            by_track.setdefault(row["id"], []).append(row)

        n_tracks = 0
        for track_id, track_rows in by_track.items():
            track_rows.sort(key=lambda r: r["frame"])
            observations = []
            for row in track_rows:
                image_path = frame_files.get(row["frame"])
                if image_path is None:
                    continue
                observations.append(TrackObservation(
                    frame_idx=row["frame"],
                    image_path=image_path,
                    bbox=(row["x1"], row["y1"], row["x2"], row["y2"]),
                    part_label=row["part_label"],
                ))
            if len(observations) < min_observations:
                continue
            all_tracks.append(IdentityTrack(
                identity_id=f"{seq_name}__{track_id}",
                video_id=seq_name,
                observations=observations,
            ))
            n_tracks += 1
        print(f"[OK] {seq_name}: {n_tracks} tracks útiles")

    if not all_tracks:
        raise RuntimeError(f"Ningún track útil (>= {min_observations} observaciones) en {data_root}")
    print(f"[OK] Índice total: {len(all_tracks)} tracks de {len(seq_dirs)} secuencias")
    return DamageTrackIndex(all_tracks)


# --------------------------------------------------------------------------- #
# Visualización: por track del banco y por query, sus S recortes multi-escala
# --------------------------------------------------------------------------- #

def _denormalize_to_uint8(tensor: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype("uint8"))


def visualize_episode(
    batch: dict, episode_idx: int, out_path: str, tile_size: int = 112, label_width: int = 180, query_size: int = 224,
    pred: Optional[int] = None,
):
    """Panel izquierdo: una fila por cada track candidato válido de la
    galería, con hasta K observaciones de su historial (repartidas en el
    tiempo, ver _load_track_history_crops en reid_dataloader.py). Tiles
    negros = slot de historial sin observación real (track más corto que
    K, ver bank_obs_mask). Panel derecho: la QUERY -- la detección nueva,
    DOS vistas (tight + wide, ver _load_query_views en reid_dataloader.py).
    Ambos paneles vienen de los tensores que ya ve el modelo, sin volver a
    abrir la imagen original.

    `pred` (opcional, ver _save_fail_episodes/_save_correct_episodes): el
    índice que predijo el modelo (0..N-1 = matchea ese candidato, N = "no
    match") -- si se pasa, se marca con "<- PRED" el slot correspondiente,
    distinto de "<- MATCH" (la verdad, ver `target`), para ver de un
    vistazo si se equivocó de candidato o de match/no-match."""
    bank_images = batch["bank_images"][episode_idx]    # (N, K, C, H, W)
    bank_mask = batch["bank_mask"][episode_idx]         # (N,)
    query_views = batch["query_views"][episode_idx]     # (2, C, H, W) -- [tight, wide]
    target = int(batch["target"][episode_idx].item())

    N = bank_images.shape[0]
    K = bank_images.shape[1]
    valid_bank_idxs = bank_mask.nonzero(as_tuple=True)[0].tolist()
    matched_bank_idx = target if target < N else None
    pred_bank_idx = pred if (pred is not None and pred < N) else None

    rows = []  # (label, [K tiles de historial])
    for n in valid_bank_idxs:
        tags = []
        if n == matched_bank_idx:
            tags.append("MATCH")
        if n == pred_bank_idx:
            tags.append("PRED")
        tag = f"  <- {'+'.join(tags)}" if tags else ""
        rows.append((
            f"candidato {n}{tag}",
            [_denormalize_to_uint8(bank_images[n, k]).resize((tile_size, tile_size)) for k in range(K)],
        ))

    left_w = label_width + K * tile_size
    left_h = max(1, len(rows)) * tile_size
    label_bar_h = 22
    canvas_w = left_w + 2 * query_size
    canvas_h = max(left_h, label_bar_h + query_size)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for r, (label, tiles) in enumerate(rows):
        y = r * tile_size
        draw.rectangle([0, y, label_width, y + tile_size], fill=(20, 20, 20))
        draw.text((8, y + tile_size // 2 - 6), label, fill=(255, 255, 0))
        for c, tile in enumerate(tiles):
            canvas.paste(tile, (label_width + c * tile_size, y))

    query_label = f"query (Q): tight | wide -- GT: {'MATCH id=' + str(matched_bank_idx) if matched_bank_idx is not None else 'no match'}"
    if pred is not None:
        pred_desc = f"MATCH id={pred}" if pred < N else "no match"
        query_label += f"  |  PRED: {pred_desc}"
    draw.rectangle([left_w, 0, canvas_w, label_bar_h], fill=(20, 20, 20))
    draw.text((left_w + 8, 4), query_label, fill=(255, 255, 0))
    for v in range(query_views.shape[0]):  # 0=tight, 1=wide
        view_img = _denormalize_to_uint8(query_views[v]).resize((query_size, query_size))
        canvas.paste(view_img, (left_w + v * query_size, label_bar_h))

    canvas.save(out_path)


# --------------------------------------------------------------------------- #
# Subcomando: dump-batches
# --------------------------------------------------------------------------- #

def run_dump_batches(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root, sequence_dirs = resolve_sequence_dirs(args.data_root, args.split)
    track_index = build_track_index(
        data_root,
        gt_conf_thresh=args.gt_conf_thresh,
        gt_class_id=args.gt_class_id,
        gt_visibility_thresh=args.gt_visibility_thresh,
        sequence_dirs=sequence_dirs,
    )

    dataset_kwargs = dict(
        max_bank_size=args.max_bank_size,
        num_track_history=args.num_track_history,
        history_jitter_ratio=args.history_jitter_ratio,
        synthetic_gap_max=args.synthetic_gap_max,
        query_wide_factor=args.query_wide_factor,
    )
    if args.use_hierarchical_sampler:
        pk_sampler = build_track_pk_sampler(
            track_index, P=args.pk_sequences_per_batch, K=args.pk_tracks_per_sequence,
            allow_replacement_within_sequence=not args.pk_no_replacement,
        )
        effective_batch_size = args.pk_sequences_per_batch * args.pk_tracks_per_sequence
        print(f"[INFO] HierarchicalPKSampler activo: P={args.pk_sequences_per_batch} "
              f"K={args.pk_tracks_per_sequence} -> batch_size efectivo={effective_batch_size}, "
              f"{len(pk_sampler)} batches/época (--batch-size se ignora)")
        dataset = ReIDEpisodeDataset(
            track_index, episodes_per_epoch=len(pk_sampler) * effective_batch_size,
            pk_driven=True, **dataset_kwargs,
        )
        loader = DataLoader(dataset, batch_sampler=pk_sampler, collate_fn=collate_episodes)
    else:
        dataset = ReIDEpisodeDataset(
            track_index, episodes_per_epoch=args.num_batches * args.batch_size, **dataset_kwargs,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_episodes)

    os.makedirs(args.outdir, exist_ok=True)
    saved = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        for episode_idx in range(batch["bank_images"].shape[0]):
            out_path = os.path.join(args.outdir, f"batch{batch_idx}_ep{episode_idx}.jpg")
            visualize_episode(batch, episode_idx, out_path)
            saved += 1
    print(f"[OK] {saved} episodios guardados en {args.outdir}")


def dump_example_batches(loader: DataLoader, split_name: str, out_dir: str, max_episodes: Optional[int] = None) -> None:
    """Guarda una visualización (ver visualize_episode) de cada episodio del
    PRIMER batch de `loader`, para poder mirar a ojo -- al arrancar el
    entrenamiento -- qué le está llegando de verdad al modelo (bboxes,
    escalas, memoria, mezcla de match/no-match) antes de esperar epochs
    enteros."""
    batch = next(iter(loader))
    n_episodes = batch["bank_images"].shape[0]
    if max_episodes is not None:
        n_episodes = min(n_episodes, max_episodes)
    for episode_idx in range(n_episodes):
        out_path = os.path.join(out_dir, f"{split_name}_ep{episode_idx}.jpg")
        visualize_episode(batch, episode_idx, out_path)
    print(f"[OK] {n_episodes} episodios de ejemplo ({split_name}) guardados en {out_dir}")


def build_example_loader(index: "DamageTrackIndex", args, num_episodes: int, seed: Optional[int] = None) -> DataLoader:
    """Dataset/loader aparte SOLO para dump_example_batches: siempre con
    imágenes reales (sin embedding_cache -- si el train/val real usa cache,
    bank_images/query_views serían embeddings, no píxeles, y
    visualize_episode no podría dibujarlos)."""
    dataset = ReIDEpisodeDataset(
        index, episodes_per_epoch=num_episodes,
        max_bank_size=args.max_bank_size,
        num_track_history=args.num_track_history, history_jitter_ratio=args.history_jitter_ratio,
        synthetic_gap_max=args.synthetic_gap_max, query_wide_factor=args.query_wide_factor, seed=seed,
    )
    return DataLoader(dataset, batch_size=num_episodes, collate_fn=collate_episodes)


# --------------------------------------------------------------------------- #
# Subcomando: train
# --------------------------------------------------------------------------- #

def _seed_worker(worker_id):
    # sin esto, los workers de DataLoader (fork) heredan el mismo estado de
    # `random` del proceso padre y pueden generar episodios correlacionados
    # entre sí -- cada worker necesita su propia semilla.
    random.seed(torch.initial_seed() % (2 ** 32))


def build_module(args, device: torch.device) -> SetTrackReIDModule:
    module = SetTrackReIDModule(
        num_heads=args.num_heads,
        intra_track_layers=args.intra_track_layers,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
        dinov2_model_name=args.dinov2_model_name,
        dinov2_checkpoint_path=args.dinov2_checkpoint_path,
        dinov2_embedding_dim=args.dinov2_embedding_dim,
        freeze_encoder=not args.unfreeze_encoder,
    )
    return module.to(device)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    # bank_video_ids/query_video_id (ver collate_episodes_with_video_ids) son
    # listas de strings, no tensores -- .to(device) rompería con ellas.
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def _epoch_accuracies(logits: torch.Tensor, target: torch.Tensor, max_bank_size: int) -> Dict[str, float]:
    """acc: aciertos sobre las (max_bank_size + 1) clases (candidato
    correcto o "no match"). match_acc: acierto de la decisión binaria "hay
    match o no" (colapsa los candidatos en una sola clase "match").
    identity_acc: acierto de CUÁL candidato, SOLO sobre episodios donde de
    verdad hay match -- no tiene sentido medirlo en no-match."""
    pred = logits.argmax(dim=-1)
    acc = (pred == target).float().mean().item()

    pred_is_match = pred != max_bank_size
    true_is_match = target != max_bank_size
    match_acc = (pred_is_match == true_is_match).float().mean().item()

    if true_is_match.any():
        identity_acc = (pred[true_is_match] == target[true_is_match]).float().mean().item()
    else:
        identity_acc = float("nan")

    return {"acc": acc, "match_acc": match_acc, "identity_acc": identity_acc}


@torch.no_grad()
def _intra_inter_confusion(
    logits: torch.Tensor,
    target: torch.Tensor,
    bank_video_ids: List[List[str]],
    query_video_id: List[str],
    max_bank_size: int,
) -> Dict[str, float]:
    """Sobre las predicciones EQUIVOCADAS que además eligen un candidato
    (identidad incorrecta, o "match" cuando en realidad no lo había) --
    ¿el candidato mal elegido es del MISMO video que la query
    (intra-secuencia, el negativo difícil real, ver hard_distractor_ratio
    en reid_dataloader.py) o de OTRO video (inter-secuencia, el negativo
    fácil)? No tiene sentido aplicar esto al TRUE match: por construcción,
    la identidad correcta siempre vive en el mismo video que la query
    (mismo track reapareciendo), así que ahí la pregunta no aporta nada --
    lo interesante es dónde se concentra el ERROR. Si `inter_seq_error_rate`
    no cae a ~0 con el entrenamiento (o crece), es señal de que el modelo
    depende de shortcuts a nivel de video (iluminación/color) en vez de la
    apariencia real del daño -- debería casi no confundirse con negativos
    de otro video, que son el caso "fácil"."""
    pred = logits.argmax(dim=-1)
    wrong_with_pick = (pred != target) & (pred != max_bank_size)

    n_intra, n_inter = 0, 0
    for b in wrong_with_pick.nonzero(as_tuple=True)[0].tolist():
        picked = int(pred[b].item())
        if bank_video_ids[b][picked] == query_video_id[b]:
            n_intra += 1
        else:
            n_inter += 1

    total = n_intra + n_inter
    return {
        "n_wrong_picks": total,
        "n_intra": n_intra,
        "n_inter": n_inter,
        "intra_seq_error_rate": (n_intra / total) if total else float("nan"),
        "inter_seq_error_rate": (n_inter / total) if total else float("nan"),
    }


@torch.no_grad()
def _save_fail_episodes(
    batch: dict, logits: torch.Tensor, fails_dir: str, n_already_saved: int, max_fails: int
) -> int:
    """Guarda (ver visualize_episode) los episodios de `batch` donde la
    predicción se equivocó: match/no-match incorrecto, o correcto pero con
    candidato equivocado (solo aplica si de verdad matchea). Devuelve
    cuántos guardó (para no pasar de `max_fails` entre llamadas sucesivas)."""
    N = batch["bank_mask"].shape[1]
    pred = logits.argmax(dim=-1)
    target = batch["target"]
    pred_is_match = pred != N
    true_is_match = target != N
    wrong_match = pred_is_match != true_is_match
    wrong_identity = true_is_match & pred_is_match & (pred != target)

    saved = 0
    for episode_idx in wrong_match.nonzero(as_tuple=True)[0].tolist():
        if n_already_saved + saved >= max_fails:
            break
        out_path = os.path.join(fails_dir, f"fail{n_already_saved + saved:04d}_wrongmatch.jpg")
        visualize_episode(batch, episode_idx, out_path, pred=int(pred[episode_idx]))
        saved += 1
    for episode_idx in wrong_identity.nonzero(as_tuple=True)[0].tolist():
        if n_already_saved + saved >= max_fails:
            break
        out_path = os.path.join(fails_dir, f"fail{n_already_saved + saved:04d}_wrongidentity.jpg")
        visualize_episode(batch, episode_idx, out_path, pred=int(pred[episode_idx]))
        saved += 1
    return saved


@torch.no_grad()
def _save_correct_episodes(
    batch: dict, logits: torch.Tensor, correct_dir: str, n_already_saved: int, max_correct: int
) -> int:
    """Simétrico a _save_fail_episodes: guarda los episodios donde la
    predicción fue correcta, para poder ver también en qué casos acierta."""
    N = batch["bank_mask"].shape[1]
    pred = logits.argmax(dim=-1)
    target = batch["target"]
    is_correct = pred == target

    saved = 0
    for episode_idx in is_correct.nonzero(as_tuple=True)[0].tolist():
        if n_already_saved + saved >= max_correct:
            break
        tag = "match" if target[episode_idx].item() < N else "nomatch"
        out_path = os.path.join(correct_dir, f"correct{n_already_saved + saved:04d}_{tag}.jpg")
        visualize_episode(batch, episode_idx, out_path, pred=int(pred[episode_idx]))
        saved += 1
    return saved


def run_epoch(
    module: SetTrackReIDModule,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    desc: Optional[str] = None,
    precomputed: bool = False,
    fails_dir: Optional[str] = None,
    max_fails: int = 50,
    correct_dir: Optional[str] = None,
    max_correct: int = 50,
) -> Dict[str, float]:
    """Una pasada completa por `loader`. Si `optimizer` es None, corre en
    modo eval (val/test): sin backward, sin dropout. Devuelve promedios de
    loss y accuracy sobre todos los batches (identity_acc promedia solo los
    batches que tuvieron al menos un episodio con match, ver
    _epoch_accuracies). `desc` sobreescribe la etiqueta de la barra de
    progreso (default "train"/"val" según `optimizer`; útil para distinguir
    "test"). `precomputed` debe coincidir con si `loader` viene de un
    dataset con `embedding_cache` activo (ver PrecomputedEmbeddingCache) --
    si no coincide, el módulo intenta correr DINOv2 sobre tensores que ya
    son embeddings (o viceversa).

    `fails_dir`/`correct_dir`: si se pasan, guardan (ver visualize_episode)
    los episodios donde la predicción se equivocó/acertó respectivamente
    (match/no-match, y si matchea, identidad), hasta `max_fails`/
    `max_correct` en total cada uno."""
    is_train = optimizer is not None
    module.train(is_train)

    sums = {"total_loss": 0.0, "acc": 0.0, "match_acc": 0.0}
    n_batches = 0
    identity_acc_sum, n_identity_batches = 0.0, 0
    # solo se llenan si `loader` viene de un dataset con return_video_ids=True
    # (val/test, ver collate_episodes_with_video_ids) -- train no las tiene.
    # OJO: se acumulan CONTEOS crudos (no el ratio ya calculado por batch) --
    # con batch_size chico (6-8), la mayoría de los batches tienen 0-3 wrong
    # picks, así que promediar ratios por batch (cada uno casi binario:
    # 0/0.5/1.0) le da el mismo peso a un batch con 1 evento que a uno con 5,
    # generando oscilaciones enormes de época a época sin que el modelo haya
    # cambiado nada real. El ratio se calcula UNA vez al final, sobre el
    # total de la época -- eso sí es estadísticamente estable.
    n_intra_total, n_inter_total = 0, 0

    desc = desc or ("train" if is_train else "val")
    desc += f" epoch {epoch}" if epoch is not None else ""
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    n_fails_saved = 0
    n_correct_saved = 0

    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            batch = _batch_to_device(batch, device)
            out = module(
                batch["query_views"], batch["bank_images"], batch["bank_obs_mask"], batch["bank_mask"],
                precomputed=precomputed,
            )
            logits = out["logits"]
            loss = F.cross_entropy(logits, batch["target"])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if fails_dir is not None and n_fails_saved < max_fails:
                n_fails_saved += _save_fail_episodes(batch, logits, fails_dir, n_fails_saved, max_fails)
            if correct_dir is not None and n_correct_saved < max_correct:
                n_correct_saved += _save_correct_episodes(batch, logits, correct_dir, n_correct_saved, max_correct)

            accs = _epoch_accuracies(logits, batch["target"], batch["bank_mask"].shape[1])
            sums["total_loss"] += loss.item()
            sums["acc"] += accs["acc"]
            sums["match_acc"] += accs["match_acc"]
            if not math.isnan(accs["identity_acc"]):
                identity_acc_sum += accs["identity_acc"]
                n_identity_batches += 1
            n_batches += 1

            if "bank_video_ids" in batch:
                conf = _intra_inter_confusion(
                    logits, batch["target"], batch["bank_video_ids"], batch["query_video_id"],
                    batch["bank_mask"].shape[1],
                )
                n_intra_total += conf["n_intra"]
                n_inter_total += conf["n_inter"]

            pbar.set_postfix(
                loss=f"{sums['total_loss'] / n_batches:.4f}",
                acc=f"{sums['acc'] / n_batches:.3f}",
                match_acc=f"{sums['match_acc'] / n_batches:.3f}",
                id_acc=f"{(identity_acc_sum / n_identity_batches) if n_identity_batches else float('nan'):.3f}",
            )

    avg = {k: v / n_batches for k, v in sums.items()}
    avg["identity_acc"] = identity_acc_sum / n_identity_batches if n_identity_batches else float("nan")
    n_wrong_picks_total = n_intra_total + n_inter_total
    avg["intra_seq_error_rate"] = n_intra_total / n_wrong_picks_total if n_wrong_picks_total else float("nan")
    avg["inter_seq_error_rate"] = n_inter_total / n_wrong_picks_total if n_wrong_picks_total else float("nan")
    return avg


def _format_metrics(metrics: Dict[str, float]) -> str:
    base = (
        f"loss={metrics['total_loss']:.4f} acc={metrics['acc']:.4f} "
        f"match_acc={metrics['match_acc']:.4f} id_acc={metrics['identity_acc']:.4f}"
    )
    # solo train no las tiene (run_epoch las deja en nan si el batch no
    # traía bank_video_ids, ver collate_episodes_with_video_ids).
    if not math.isnan(metrics.get("intra_seq_error_rate", float("nan"))):
        base += (
            f" intra_err={metrics['intra_seq_error_rate']:.4f} "
            f"inter_err={metrics['inter_seq_error_rate']:.4f}"
        )
    return base


def run_test_eval(args, device: torch.device, module: SetTrackReIDModule) -> Optional[Dict[str, float]]:
    """Evaluación final sobre el split de test, con dos diferencias respecto
    a val: (1) carga el MEJOR checkpoint (best_model.pt), no los pesos
    actuales del módulo -- es el que mejor generalizó, no el último; (2) usa
    force_reid=True: cada episodio de test es un caso real de
    reidentificación (la query principal SIEMPRE reaparece), a diferencia
    de train/val que usan la mezcla natural de casos (incluye NEW_TRACK)."""
    test_root, test_seq_dirs = resolve_sequence_dirs(args.test_data_root, args.test_split)
    print(f"[INFO] test_root={test_root}")

    test_index = build_track_index(
        test_root, gt_conf_thresh=args.gt_conf_thresh, gt_class_id=args.gt_class_id,
        gt_visibility_thresh=args.gt_visibility_thresh, sequence_dirs=test_seq_dirs,
    )
    test_cache_path = getattr(args, "test_embedding_cache", None)
    test_cache = PrecomputedEmbeddingCache.load(test_cache_path) if test_cache_path else None
    if test_cache is not None:
        print(f"[INFO] Usando cache de embeddings para test: {test_cache_path}")
    test_dataset = ReIDEpisodeDataset(
        test_index, episodes_per_epoch=args.test_episodes,
        max_bank_size=args.max_bank_size,
        num_track_history=args.num_track_history, history_jitter_ratio=args.history_jitter_ratio,
        synthetic_gap_max=args.synthetic_gap_max, query_wide_factor=args.query_wide_factor,
        force_reid=True, seed=args.seed,  # mismos episodios entre corridas -- test comparable
        embedding_cache=test_cache,
        return_video_ids=True,  # métricas de confusión intra/inter-secuencia, ver _intra_inter_confusion
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=collate_episodes_with_video_ids,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
    )

    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    module.load_state_dict(ckpt["model"])
    print(f"[OK] Cargado {best_ckpt_path} (epoch {ckpt['epoch']}, "
          f"best_val_loss={ckpt['best_val_loss']:.4f}) para evaluar en test")

    test_metrics = run_epoch(
        module, test_loader, device, optimizer=None,
        desc="test", precomputed=(test_cache is not None),
    )
    print(f"[TEST] {_format_metrics(test_metrics)}")
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    return test_metrics


def run_train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_root, train_seq_dirs = resolve_sequence_dirs(args.train_data_root, args.train_split)
    val_root, val_seq_dirs = resolve_sequence_dirs(args.val_data_root, args.val_split)
    print(f"[INFO] train_root={train_root}")
    print(f"[INFO] val_root={val_root}")

    train_index = build_track_index(
        train_root, gt_conf_thresh=args.gt_conf_thresh, gt_class_id=args.gt_class_id,
        gt_visibility_thresh=args.gt_visibility_thresh, sequence_dirs=train_seq_dirs,
    )
    val_index = build_track_index(
        val_root, gt_conf_thresh=args.gt_conf_thresh, gt_class_id=args.gt_class_id,
        gt_visibility_thresh=args.gt_visibility_thresh, sequence_dirs=val_seq_dirs,
    )

    train_cache = PrecomputedEmbeddingCache.load(args.train_embedding_cache) if args.train_embedding_cache else None
    val_cache = PrecomputedEmbeddingCache.load(args.val_embedding_cache) if args.val_embedding_cache else None
    if train_cache is not None:
        print(f"[INFO] Usando cache de embeddings para train: {args.train_embedding_cache} "
              f"(sin aumentación -- ColorJitter no aplica sobre embeddings cacheados)")
    if val_cache is not None:
        print(f"[INFO] Usando cache de embeddings para val: {args.val_embedding_cache}")

    train_dataset_kwargs = dict(
        max_bank_size=args.max_bank_size,
        num_track_history=args.num_track_history, history_jitter_ratio=args.history_jitter_ratio,
        synthetic_gap_max=args.synthetic_gap_max, query_wide_factor=args.query_wide_factor,
        transform=AUGMENTED_TRANSFORM,  # ColorJitter, frena memorizar iluminación/color de train
        embedding_cache=train_cache,    # si está activo, transform arriba se ignora
    )
    if args.use_hierarchical_sampler:
        # controla qué P secuencias x K tracks anclan los episodios de cada
        # batch de train (ver build_track_pk_sampler en
        # hierarchical_pk_sampler.py, pk_driven en reid_dataloader.py) en
        # vez de sample_random_identity() al azar -- el resto del episodio
        # (banco de distractores, casos REAL_REID/SYNTHETIC_SPLIT/
        # NEW_TRACK) no cambia. batch_sampler reemplaza batch_size/shuffle
        # del DataLoader -- --batch-size se ignora acá.
        pk_sampler = build_track_pk_sampler(
            train_index, P=args.pk_sequences_per_batch, K=args.pk_tracks_per_sequence,
            allow_replacement_within_sequence=not args.pk_no_replacement,
        )
        effective_batch_size = args.pk_sequences_per_batch * args.pk_tracks_per_sequence
        print(f"[INFO] HierarchicalPKSampler activo: P={args.pk_sequences_per_batch} "
              f"K={args.pk_tracks_per_sequence} -> batch_size efectivo={effective_batch_size}, "
              f"{len(pk_sampler)} batches/época (--batch-size se ignora)")
        train_dataset = ReIDEpisodeDataset(
            train_index, episodes_per_epoch=len(pk_sampler) * effective_batch_size,
            pk_driven=True, **train_dataset_kwargs,
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=pk_sampler, collate_fn=collate_episodes,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            worker_init_fn=_seed_worker if args.num_workers > 0 else None,
        )
    else:
        train_dataset = ReIDEpisodeDataset(
            train_index, episodes_per_epoch=args.train_episodes_per_epoch, **train_dataset_kwargs,
        )
        train_loader = None  # se arma más abajo junto con val_loader, como antes

    val_dataset = ReIDEpisodeDataset(
        val_index, episodes_per_epoch=args.val_episodes,
        max_bank_size=args.max_bank_size,
        num_track_history=args.num_track_history, history_jitter_ratio=args.history_jitter_ratio,
        synthetic_gap_max=args.synthetic_gap_max, query_wide_factor=args.query_wide_factor,
        seed=args.seed,  # mismos episodios epoch a epoch -- val comparable de verdad
        embedding_cache=val_cache,
        return_video_ids=True,  # métricas de confusión intra/inter-secuencia, ver _intra_inter_confusion
    )

    # el sampling es generativo (cada __getitem__ arma un episodio nuevo al
    # azar si no se fija `seed`, sin depender del índice, salvo pk_driven --
    # ver arriba), así que `shuffle` no aporta nada acá. train y val usan
    # collate_fn distinto: val necesita bank_video_ids/query_video_id
    # (strings, no tensores) para las métricas de confusión
    # intra/inter-secuencia; train no las usa.
    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
    )
    if train_loader is None:  # no armado arriba -- caso sin HierarchicalPKSampler
        train_loader = DataLoader(train_dataset, collate_fn=collate_episodes, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, collate_fn=collate_episodes_with_video_ids, **common_loader_kwargs)

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.skip_batch_examples:
        examples_dir = os.path.join(args.output_dir, "batch_examples")
        os.makedirs(examples_dir, exist_ok=True)
        train_example_loader = build_example_loader(train_index, args, args.num_example_episodes)
        val_example_loader = build_example_loader(val_index, args, args.num_example_episodes, seed=args.seed)
        dump_example_batches(train_example_loader, "train", examples_dir)
        dump_example_batches(val_example_loader, "val", examples_dir)

    module = build_module(args, device)
    trainable_params = [p for p in module.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[INFO] {n_trainable:,} parámetros entrenables "
          f"(encoder {'des' if args.unfreeze_encoder else ''}congelado)")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=device)
        module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[OK] Resumido desde {args.resume_from} (epoch {start_epoch}, best_val_loss={best_val_loss:.4f})")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            module, train_loader, device, optimizer=optimizer,
            epoch=epoch, precomputed=(train_cache is not None),
        )
        val_metrics = run_epoch(
            module, val_loader, device, optimizer=None,
            epoch=epoch, precomputed=(val_cache is not None),
        )
        print(f"[epoch {epoch}] train: {_format_metrics(train_metrics)} | val: {_format_metrics(val_metrics)}")

        checkpoint = {
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "last_model.pt"))

        if val_metrics["total_loss"] < best_val_loss - args.min_delta:
            best_val_loss = val_metrics["total_loss"]
            epochs_without_improvement = 0
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            print(f"[OK] Nuevo mejor val_loss={best_val_loss:.4f} -> best_model.pt")

            fails_dir = os.path.join(args.output_dir, "fails_validation")
            correct_dir = os.path.join(args.output_dir, "correct_validation")
            for d in (fails_dir, correct_dir):
                if os.path.exists(d):
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
            run_epoch(
                module, val_loader, device, optimizer=None,
                desc="val-fails", precomputed=(val_cache is not None),
                fails_dir=fails_dir, correct_dir=correct_dir,
            )
        else:
            epochs_without_improvement += 1
            print(f"[INFO] Sin mejora hace {epochs_without_improvement}/{args.patience} epoch(s) "
                  f"(mejor val_loss={best_val_loss:.4f})")
            if epochs_without_improvement >= args.patience:
                print(f"[STOP] Early stopping: sin mejora en {args.patience} epochs consecutivos")
                break

    print("[OK] Entrenamiento terminado")

    if not os.path.exists(os.path.join(args.output_dir, "best_model.pt")):
        print("[WARN] No hay best_model.pt (¿el loop no llegó a mejorar ningún epoch?), se omite la evaluación de test")
        return
    try:
        run_test_eval(args, device, module)
    except FileNotFoundError as e:
        print(f"[WARN] No se pudo correr la evaluación de test (dataset no encontrado): {e}")


# --------------------------------------------------------------------------- #
# Subcomando: precompute-cache
# --------------------------------------------------------------------------- #

def run_precompute_cache(args):
    """Precalcula embeddings DINOv2 para TODAS las observaciones únicas de
    un split y los guarda en disco (PrecomputedEmbeddingCache). Solo válido
    mientras el encoder se use CONGELADO y SIN aumentación -- por eso el
    cache se construye siempre con el transform default (sin ColorJitter),
    aunque el split sea train: no tiene sentido cachear un embedding que
    cambia cada vez que se le pasa una imagen aumentada distinta. Pensado
    para iterar rápido en experimentos (comparar hiperparámetros, criterio
    de checkpoint, backbone fine-tuneado) sin correr DINOv2 en cada época --
    para la corrida final "de verdad" con aumentación en train, no se usa
    cache ahí."""
    device = torch.device(args.device)
    data_root, sequence_dirs = resolve_sequence_dirs(args.data_root, args.split)
    print(f"[INFO] data_root={data_root}")

    index = build_track_index(
        data_root, gt_conf_thresh=args.gt_conf_thresh, gt_class_id=args.gt_class_id,
        gt_visibility_thresh=args.gt_visibility_thresh, sequence_dirs=sequence_dirs,
    )

    encoder = DINOv2Encoder(
        model_name=args.dinov2_model_name, checkpoint_path=args.dinov2_checkpoint_path,
        embedding_dim=args.dinov2_embedding_dim, freeze=True,
    ).to(device).eval()

    # dataset "crudo" solo para reusar _load_multiscale_crops (sin
    # embedding_cache, con el transform default -- sin aumentar).
    crop_dataset = ReIDEpisodeDataset(index, episodes_per_epoch=1)

    observations, seen = [], set()
    for track in index.tracks:
        for obs in track.observations:
            key = (obs.image_path, tuple(obs.bbox))
            if key not in seen:
                seen.add(key)
                observations.append((key, obs))
    print(f"[INFO] {len(observations)} observaciones únicas a encodear")

    embeddings = {}
    for start in tqdm(range(0, len(observations), args.encode_batch_size), desc="precompute", dynamic_ncols=True):
        chunk = observations[start:start + args.encode_batch_size]
        crops_list = [crop_dataset._load_multiscale_crops(obs) for _, obs in chunk]  # cada (S,C,H,W)

        scale_flat = torch.cat(crops_list, dim=0).to(device)  # (chunk_size*S, C,H,W)
        with torch.no_grad():
            scale_feats = encoder(scale_flat).cpu()  # (chunk_size*S, encoder_dim)

        S = crops_list[0].shape[0]
        for i, (key, _) in enumerate(chunk):
            embeddings[key] = scale_feats[i * S:(i + 1) * S]  # (S, encoder_dim)

    cache = PrecomputedEmbeddingCache(embeddings)
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)
    cache.save(args.output_path)
    dim = next(iter(embeddings.values())).shape[-1]
    print(f"[OK] Cache guardado en {args.output_path} ({len(embeddings)} observaciones, encoder_dim={dim})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    dump_p = sub.add_parser("dump-batches", help="Genera batches de episodios y guarda una visualización en disco")
    dump_p.add_argument("--data_root", default=None,
                         help="Carpeta con secuencias (subcarpetas con gt/gt.txt e img1/). "
                              "Si se omite, se resuelve por --split contra DATA_ROOTS.")
    dump_p.add_argument("--split", choices=list(DATA_ROOTS), default="train",
                         help="Split del dataset (ignorado si se pasa --data_root explícito).")
    dump_p.add_argument("--outdir", default="reid_batch_dump")
    dump_p.add_argument("--num-batches", type=int, default=3)
    dump_p.add_argument("--batch-size", type=int, default=2)
    dump_p.add_argument("--max-bank-size", type=int, default=8)
    dump_p.add_argument("--num-track-history", type=int, default=4,
                         help="Observaciones del historial de cada track del banco (repartidas en el "
                              "tiempo, sin padding espacial -- ver TrackHistoryFusion) a usar por slot.")
    dump_p.add_argument("--history-jitter-ratio", type=float, default=0.0,
                         help="0.0 (default) = selección determinista del historial (siempre las mismas "
                              "observaciones para el mismo track). >0: jitter aleatorio para romper esa "
                              "huella fija -- ver _spread_indices en reid_dataloader.py.")
    dump_p.add_argument("--synthetic-gap-max", type=int, default=0,
                         help="0 (default) = corte sintético en un solo punto (sin gap real). >0: descarta "
                              "hasta esa cantidad de observaciones alrededor del corte, simulando una "
                              "oclusión -- ver _try_synthetic_split en reid_dataloader.py.")
    dump_p.add_argument("--query-wide-factor", type=float, default=2.5,
                         help="Factor de expansión del bbox para la vista AMPLIA de la query (contexto "
                              "de pieza) -- ver _load_query_views en reid_dataloader.py.")
    dump_p.add_argument("--gt-conf-thresh", type=float, default=0.0)
    dump_p.add_argument("--gt-class-id", type=int, default=-1)
    dump_p.add_argument("--gt-visibility-thresh", type=float, default=0.0)
    dump_p.add_argument("--seed", type=int, default=42)
    dump_p.add_argument("--use-hierarchical-sampler", action="store_true",
                         help="Inspeccionar la composición de un batch armado con HierarchicalPKSampler "
                              "(ver el flag homónimo en el subcomando train) antes de entrenar con él.")
    dump_p.add_argument("--pk-sequences-per-batch", type=int, default=3, help="P (solo con --use-hierarchical-sampler).")
    dump_p.add_argument("--pk-tracks-per-sequence", type=int, default=2, help="K (solo con --use-hierarchical-sampler).")
    dump_p.add_argument("--pk-no-replacement", action="store_true",
                         help="Ver el flag homónimo en el subcomando train.")

    train_p = sub.add_parser("train", help="Entrena SetTrackReIDModule con early stopping y checkpoint del mejor val")
    train_p.add_argument("--train-data-root", default=None)
    train_p.add_argument("--train-split", choices=list(DATA_ROOTS), default="train")
    train_p.add_argument("--val-data-root", default=None)
    train_p.add_argument("--val-split", choices=list(DATA_ROOTS), default="val")
    train_p.add_argument("--test-data-root", default=None)
    train_p.add_argument("--test-split", choices=list(DATA_ROOTS), default="test")
    train_p.add_argument("--gt-conf-thresh", type=float, default=0.0)
    train_p.add_argument("--gt-class-id", type=int, default=-1)
    train_p.add_argument("--gt-visibility-thresh", type=float, default=0.0)

    train_p.add_argument("--max-bank-size", type=int, default=16)
    train_p.add_argument("--num-track-history", type=int, default=4,
                          help="Observaciones del historial de cada track del banco (repartidas en el "
                               "tiempo, sin padding espacial -- ver TrackHistoryFusion) a usar por slot.")
    train_p.add_argument("--history-jitter-ratio", type=float, default=0.0,
                          help="0.0 (default) = selección determinista del historial (siempre las mismas "
                               "observaciones para el mismo track). >0: jitter aleatorio para romper esa "
                               "huella fija -- ver _spread_indices en reid_dataloader.py.")
    train_p.add_argument("--synthetic-gap-max", type=int, default=0,
                          help="0 (default) = corte sintético en un solo punto (sin gap real). >0: descarta "
                               "hasta esa cantidad de observaciones alrededor del corte, simulando una "
                               "oclusión -- ver _try_synthetic_split en reid_dataloader.py.")
    train_p.add_argument("--query-wide-factor", type=float, default=2.5,
                          help="Factor de expansión del bbox para la vista AMPLIA de la query (contexto "
                               "de pieza) -- ver _load_query_views en reid_dataloader.py.")
    train_p.add_argument("--train-episodes-per-epoch", type=int, default=2000)
    train_p.add_argument("--val-episodes", type=int, default=400)
    train_p.add_argument("--test-episodes", type=int, default=400,
                          help="Episodios de test al final del entrenamiento -- todos casos reales de "
                               "reidentificación (force_reid=True), no la mezcla natural de train/val.")
    train_p.add_argument("--batch-size", type=int, default=6)
    train_p.add_argument("--num-workers", type=int, default=4)

    train_p.add_argument("--use-hierarchical-sampler", action="store_true",
                          help="Controla qué P secuencias x K tracks anclan los episodios de cada batch "
                               "de TRAIN (ver HierarchicalPKSampler/TrackAnchorPool en "
                               "hierarchical_pk_sampler.py), en vez de sample_random_identity() "
                               "completamente al azar -- el resto del episodio (banco de distractores "
                               "vía hard_distractor_ratio, casos REAL_REID/SYNTHETIC_SPLIT/NEW_TRACK) no "
                               "cambia. Solo aplica a train -- val/test siguen igual (seed fijo, "
                               "comparabilidad entre epochs/corridas). Con esto activo, --batch-size se "
                               "ignora: el batch queda fijo en P*K (ver --pk-sequences-per-batch/"
                               "--pk-tracks-per-sequence).")
    train_p.add_argument("--pk-sequences-per-batch", type=int, default=3,
                          help="P: secuencias distintas por batch de train (solo con "
                               "--use-hierarchical-sampler).")
    train_p.add_argument("--pk-tracks-per-sequence", type=int, default=2,
                          help="K: tracks distintos por secuencia por batch de train (solo con "
                               "--use-hierarchical-sampler). batch_size efectivo = P*K.")
    train_p.add_argument("--pk-no-replacement", action="store_true",
                          help="Si se pasa, las secuencias con menos de K tracks quedan afuera del pool "
                               "de secuencias utilizables en vez de completar con tracks repetidos (ver "
                               "allow_replacement_within_sequence en HierarchicalPKSampler). Solo con "
                               "--use-hierarchical-sampler.")

    train_p.add_argument("--train-embedding-cache", default=None,
                          help="Cache de PrecomputedEmbeddingCache (ver subcomando precompute-cache) para train -- "
                               "salta DINOv2 en cada época, pero desactiva de facto la aumentación (ColorJitter no "
                               "aplica sobre embeddings). Solo para iterar rápido, no para la corrida final.")
    train_p.add_argument("--val-embedding-cache", default=None)
    train_p.add_argument("--test-embedding-cache", default=None)

    train_p.add_argument("--proj-dim", type=int, default=256,
                          help="Proyecta el embedding crudo del encoder (ej. 768 para dinov2-base, 384 "
                               "para dinov2-small) a esta dimensión antes de TrackReIDModel -- reduce "
                               "parámetros entrenables (pelear overfitting) y memoria de GPU sin tocar "
                               "el encoder. Default 256 (aplica incluso a encoders 'small', que ya son "
                               "384 nativos -- mantiene la dimensión interna del modelo consistente "
                               "entre encoders para comparar en igualdad de condiciones). Pasar "
                               "--proj-dim 0 para desactivar la proyección y usar la dimensión nativa.")
    train_p.add_argument("--num-heads", type=int, default=4)
    train_p.add_argument("--intra-track-layers", type=int, default=1,
                          help="Capas de self-attention en IntraTrackEncoder (contextualiza las vistas "
                               "del historial de cada track candidato entre sí, ver set_reid_model.py).")
    train_p.add_argument("--dropout", type=float, default=0.1)
    train_p.add_argument("--dinov2-model-name", default="facebook/dinov2-base")
    train_p.add_argument("--dinov2-checkpoint-path", default=None)
    train_p.add_argument("--dinov2-embedding-dim", type=int, default=None)
    train_p.add_argument("--unfreeze-encoder", action="store_true",
                          help="Sin esto el backbone DINOv2 queda congelado (default).")

    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--epochs", type=int, default=100)

    train_p.add_argument("--patience", type=int, default=10,
                          help="Epochs consecutivos sin mejora en val_loss antes de early-stop.")
    train_p.add_argument("--min-delta", type=float, default=1e-4,
                          help="Mejora mínima en val_loss para contar como progreso (evita resetear el patience por ruido).")
    train_p.add_argument("--output-dir", default="reid_track_query_runs")
    train_p.add_argument("--resume-from", default=None, help="Checkpoint (last_model.pt/best_model.pt) desde donde resumir.")
    train_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--num-example-episodes", type=int, default=4,
                          help="Episodios del primer batch de train/val a volcar como imagen en "
                               "<output-dir>/batch_examples/ al arrancar el entrenamiento (ver dump_example_batches).")
    train_p.add_argument("--skip-batch-examples", action="store_true",
                          help="No guardar los ejemplos de batches de train/val al arrancar.")

    precompute_p = sub.add_parser(
        "precompute-cache",
        help="Precalcula embeddings DINOv2 para un split completo (PrecomputedEmbeddingCache) -- "
             "solo válido con encoder congelado y sin aumentación, para iterar rápido en experimentos.",
    )
    precompute_p.add_argument("--data_root", default=None)
    precompute_p.add_argument("--split", choices=list(DATA_ROOTS), default="train")
    precompute_p.add_argument("--gt-conf-thresh", type=float, default=0.0)
    precompute_p.add_argument("--gt-class-id", type=int, default=-1)
    precompute_p.add_argument("--gt-visibility-thresh", type=float, default=0.0)
    precompute_p.add_argument("--dinov2-model-name", default="facebook/dinov2-base")
    precompute_p.add_argument("--dinov2-checkpoint-path", default=None)
    precompute_p.add_argument("--dinov2-embedding-dim", type=int, default=None)
    precompute_p.add_argument("--encode-batch-size", type=int, default=32,
                               help="Observaciones por batch (cada una aporta num_scales+1 imágenes al encoder).")
    precompute_p.add_argument("--output-path", required=True)
    precompute_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "dump-batches":
        run_dump_batches(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "precompute-cache":
        run_precompute_cache(args)


if __name__ == "__main__":
    main()
