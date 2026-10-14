"""
Antes de invertir en explorar P4 de RT-DETR, chequea si alguna capa
INTERMEDIA de DINOv2 (no solo la última, que es la que usa DINOv2Encoder
hoy -- ver reid_track_query_module.py:130, last_hidden_state[:, 0, :])
separa mejor los daños por identidad que el CLS final.

DINOv2 vía HuggingFace expone `output_hidden_states=True`: devuelve una
tupla con (embeddings + salida de cada bloque transformer), así que se
puede sacar el CLS token de cada capa sin tocar el modelo ni hacer cirugía
de grafo (a diferencia de RT-DETR/ONNX). También prueba, por capa, el
promedio de patch tokens (excluyendo CLS y, si aplica, register tokens de
DINOv3) como pooling alternativo al CLS solo.

Reusa el mismo dataset/transform/crop que usa el pipeline real
(reid_dataloader.py: build_track_index, _expand_bbox, DEFAULT_TRANSFORM)
para que el resultado sea directamente comparable con lo que ve el modelo
en producción.

Métrica de ranking por capa: gap = similitud_intra_identidad -
similitud_inter_identidad_mismo_video -- ese "mismo video, distinta
identidad" es exactamente el caso difícil detrás del intra_seq_error_rate
alto que se viene viendo en validación/test.

Uso:
    python3 scripts/embedding_analysis/analyze_dinov2_layers.py --dinov2-model-name facebook/dinov2-small --split train
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import argparse
import random

import numpy as np
import torch
from transformers import AutoModel

from reid_dataloader import DEFAULT_TRANSFORM, _expand_bbox
from train_track_query_reid import build_track_index, resolve_sequence_dirs
from PIL import Image


def _load_crops(split, num_identities, obs_per_identity, min_observations, padding_factor, seed):
    random.seed(seed)
    data_root, seq_dirs = resolve_sequence_dirs(None, split)
    index = build_track_index(data_root, sequence_dirs=seq_dirs, min_observations=min_observations)

    eligible = [t for t in index.tracks if len(t.observations) >= min_observations]
    if not eligible:
        raise SystemExit(f"No hay tracks con >= {min_observations} observaciones en split={split}")
    random.shuffle(eligible)
    chosen = eligible[:num_identities]
    print(f"[INFO] {len(chosen)} identidades elegidas de {len(eligible)} elegibles (split={split})")

    tensors, identities, videos = [], [], []
    for track in chosen:
        obs_sample = random.sample(track.observations, min(obs_per_identity, len(track.observations)))
        for obs in obs_sample:
            image = Image.open(obs.image_path).convert("RGB")
            crop_box = _expand_bbox(obs.bbox, padding_factor, image.size)
            crop = image.crop([int(v) for v in crop_box])
            tensors.append(DEFAULT_TRANSFORM(crop))
            identities.append(track.identity_id)
            videos.append(track.video_id)
        print(f"[OK] {track.identity_id} ({track.video_id}): {len(obs_sample)} crops")

    return torch.stack(tensors), identities, videos


@torch.no_grad()
def _extract_all_layers(model, batch, device, num_register_tokens, batch_size=16):
    """Devuelve dict {layer_idx: {'cls': (N,D), 'patch_mean': (N,D)}}."""
    model.eval()
    num_layers = model.config.num_hidden_layers
    per_layer = {l: {"cls": [], "patch_mean": []} for l in range(num_layers + 1)}

    for i in range(0, batch.shape[0], batch_size):
        chunk = batch[i:i + batch_size].to(device)
        out = model(pixel_values=chunk, output_hidden_states=True)
        for l, hs in enumerate(out.hidden_states):  # (B, tokens, D), embeddings + cada bloque
            cls = hs[:, 0, :]
            patch_tokens = hs[:, 1 + num_register_tokens:, :]
            patch_mean = patch_tokens.mean(dim=1)
            per_layer[l]["cls"].append(cls.cpu().numpy())
            per_layer[l]["patch_mean"].append(patch_mean.cpu().numpy())

    return {
        l: {k: np.concatenate(v, axis=0) for k, v in d.items()}
        for l, d in per_layer.items()
    }


def _cosine_sim_matrix(embs):
    norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    return norm @ norm.T


def _similarity_gap(embs, identities, videos):
    sim = _cosine_sim_matrix(embs)
    n = len(identities)
    intra, inter_same_video, inter_other_video = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if identities[i] == identities[j]:
                intra.append(sim[i, j])
            elif videos[i] == videos[j]:
                inter_same_video.append(sim[i, j])
            else:
                inter_other_video.append(sim[i, j])
    intra_mean = float(np.mean(intra)) if intra else float("nan")
    hard_mean = float(np.mean(inter_same_video)) if inter_same_video else float("nan")
    easy_mean = float(np.mean(inter_other_video)) if inter_other_video else float("nan")
    return intra_mean, hard_mean, easy_mean, intra_mean - hard_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-identities", type=int, default=15)
    parser.add_argument("--obs-per-identity", type=int, default=6)
    parser.add_argument("--min-observations", type=int, default=4)
    parser.add_argument("--padding-factor", type=float, default=1.6)
    parser.add_argument("--num-register-tokens", type=int, default=0, help="4 para variantes DINOv3 con register tokens, 0 para DINOv2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch, identities, videos = _load_crops(
        args.split, args.num_identities, args.obs_per_identity,
        args.min_observations, args.padding_factor, args.seed,
    )

    print(f"[INFO] Cargando {args.dinov2_model_name}...")
    model = AutoModel.from_pretrained(args.dinov2_model_name).to(args.device)

    per_layer = _extract_all_layers(model, batch, args.device, args.num_register_tokens, args.batch_size)

    print(f"\n{'layer':>6} {'pooling':>11} {'intra':>8} {'inter_hard':>11} {'inter_easy':>11} {'gap(intra-hard)':>16}")
    results = []
    for l, d in per_layer.items():
        for pooling in ("cls", "patch_mean"):
            intra, hard, easy, gap = _similarity_gap(d[pooling], identities, videos)
            print(f"{l:>6} {pooling:>11} {intra:>8.4f} {hard:>11.4f} {easy:>11.4f} {gap:>16.4f}")
            results.append((l, pooling, intra, hard, easy, gap))

    best = max(results, key=lambda r: r[5])
    print(
        f"\n[MEJOR] layer={best[0]} pooling={best[1]} gap={best[5]:.4f} "
        f"(vs. layer={model.config.num_hidden_layers} pooling=cls, que es lo que usa DINOv2Encoder hoy)"
    )
    last_cls = next(r for r in results if r[0] == model.config.num_hidden_layers and r[1] == "cls")
    print(f"[ACTUAL] layer={last_cls[0]} pooling=cls gap={last_cls[5]:.4f} (config actual del proyecto)")


if __name__ == "__main__":
    main()
