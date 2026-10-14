"""
precompute_paper7_c4_roi_cache.py

Copia de precompute_p4_roi_cache.py, adaptada para cachear embeddings desde
el detector RT-DETR (Ultralytics/HGNetV2) del trabajo previo del grupo [7]
(ver paper7_backbone_encoder.py para la eleccion del layer "C4-equivalente"
por stride) en vez de nuestro propio RT-DETRv2 -- MISMO formato de cache
(`PrecomputedEmbeddingCache`, clave `(image_path, bbox)`, mismos
`DEFAULT_PADDING_FACTORS`) para poder usarlo como reemplazo drop-in de
--embedding-cache en analyze_dinov2_nn_baseline.py sin tocar ese archivo.

Uso -- calibracion del umbral de coseno para esta fuente de embedding
(M4 del checklist de revision: nuestra estrategia de reid aplicada sobre la
salida de tracking del transformer entrenado de [7]):
    python3 scripts/paper7_comparison/precompute_paper7_c4_roi_cache.py --split train --output-path paper7_c4_roi_cache_train.pt

Luego:
    python3 analyze_dinov2_nn_baseline.py --split train --embedding-cache paper7_c4_roi_cache_train.pt \\
        --encoder dinov2 --num-calibration-episodes 300 --calibration-seed 123
    (--encoder dinov2 es un placeholder inerte: con el cache cubriendo TODAS
    las observaciones del split pedido, build_encoder(args) igual carga un
    modelo pero _embed_batch nunca le pide un forward real -- ver
    embedding_cache en ReIDEpisodeDataset, reid_dataloader.py.)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root
import argparse
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.ops import roi_align
from tqdm import tqdm

from paper7_backbone_encoder import Paper7ROIAlignEncoder
from reid_dataloader import DEFAULT_PADDING_FACTORS, MIN_CONTEXT_CROP_SIZE_PX, PrecomputedEmbeddingCache, _expand_bbox
from train_track_query_reid import build_track_index, resolve_sequence_dirs


def _collect_unique_observations_by_image(index):
    """Identico a precompute_p4_roi_cache.py::_collect_unique_observations_by_image."""
    by_image, seen = {}, set()
    for track in index.tracks:
        for obs in track.observations:
            key = (obs.image_path, tuple(obs.bbox))
            if key in seen:
                continue
            seen.add(key)
            by_image.setdefault(obs.image_path, []).append(obs)
    return by_image, len(seen)


def run_precompute_paper7_c4_roi_cache(args):
    device = torch.device(args.device)
    if args.kfold_pool:
        from kfold_utils import list_all_sequences
        sequence_dirs = list_all_sequences()
        print(f"[INFO] --kfold-pool: {len(sequence_dirs)} secuencias (las 3 splits físicos juntas, ver kfold_utils.py)")
        index = build_track_index(None, sequence_dirs=sequence_dirs)
    else:
        data_root, sequence_dirs = resolve_sequence_dirs(args.data_root, args.split)
        print(f"[INFO] data_root={data_root}")
        index = build_track_index(data_root, sequence_dirs=sequence_dirs)
    by_image, n_obs = _collect_unique_observations_by_image(index)
    image_paths = list(by_image.keys())
    print(f"[INFO] {n_obs} observaciones únicas en {len(image_paths)} frames únicos")

    print(f"[INFO] Cargando Paper7ROIAlignEncoder (checkpoint={args.paper7_checkpoint})...")
    model = Paper7ROIAlignEncoder(args.paper7_checkpoint).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    resize = args.detector_resize
    resize_transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])
    padding_factors = list(DEFAULT_PADDING_FACTORS)
    S = len(padding_factors)

    embeddings = {}
    t0 = time.time()
    for start in tqdm(range(0, len(image_paths), args.frame_batch_size), desc="precompute-paper7-c4-roi", dynamic_ncols=True):
        chunk_paths = image_paths[start:start + args.frame_batch_size]
        frame_tensors, orig_sizes = [], []
        for path in chunk_paths:
            image = Image.open(path).convert("RGB")
            orig_sizes.append(image.size)  # (W, H)
            frame_tensors.append(resize_transform(image))
        batch = torch.stack(frame_tensors, dim=0).to(device)  # (B,3,resize,resize)

        with torch.no_grad():
            feat = model.raw_feature_map(batch)  # (B,1024,Hf,Wf)
        _, _, Hf, Wf = feat.shape

        rois, keys_order = [], []
        for b, path in enumerate(chunk_paths):
            W, H = orig_sizes[b]
            for obs in by_image[path]:
                for i, factor in enumerate(padding_factors):
                    min_size = MIN_CONTEXT_CROP_SIZE_PX if i == 0 else 0.0
                    x1, y1, x2, y2 = _expand_bbox(obs.bbox, factor, (W, H), min_size_px=min_size)
                    rois.append([b, x1 * Wf / W, y1 * Hf / H, x2 * Wf / W, y2 * Hf / H])
                keys_order.append((obs.image_path, tuple(obs.bbox)))

        rois_t = torch.tensor(rois, dtype=torch.float32, device=device)
        with torch.no_grad():
            pooled = roi_align(feat, rois_t, output_size=(args.roi_output_size, args.roi_output_size))
            pooled = F.normalize(pooled.mean(dim=[2, 3]), p=2, dim=1).cpu()  # (n_rois, 1024)

        for i, key in enumerate(keys_order):
            embeddings[key] = pooled[i * S:(i + 1) * S]  # (S, 1024)

    elapsed = time.time() - t0
    print(
        f"[OK] {len(embeddings)} observaciones encodeadas en {elapsed:.1f}s "
        f"({len(image_paths)} frames, {elapsed / max(len(image_paths), 1):.4f}s/frame)"
    )

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)
    PrecomputedEmbeddingCache(embeddings).save(args.output_path)
    print(f"[OK] Cache guardado en {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--kfold-pool", action="store_true",
        help="Ignora --data-root/--split -- usa las 72 secuencias del pool de --kfold-cv "
             "(kfold_utils.list_all_sequences, los 3 splits físicos juntos) en vez de un split fijo.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--paper7-checkpoint", default="best.pt",
        help="Checkpoint .pt del detector RT-DETR/Ultralytics de [7] (best.pt, ver "
             "custom_mot_2 en el servidor curtea.cs.us.es).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detector-resize", type=int, default=640)
    parser.add_argument("--roi-output-size", type=int, default=7)
    parser.add_argument("--frame-batch-size", type=int, default=16, help="Frames únicos por forward del backbone (no observaciones).")
    args = parser.parse_args()
    run_precompute_paper7_c4_roi_cache(args)


if __name__ == "__main__":
    main()
