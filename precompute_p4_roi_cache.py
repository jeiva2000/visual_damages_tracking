"""
precompute_p4_roi_cache.py

Cachea embeddings P4 (ROI-Align sobre `backbone.res_layers.2` de RT-DETRv2,
ver p4_backbone_encoder.py) para TODAS las observaciones únicas de un split,
en el MISMO formato que `PrecomputedEmbeddingCache` (reid_dataloader.py) que
ya usa `run_precompute_cache` (train_track_query_reid.py) para DINOv2 --
misma clave `(image_path, bbox)`, mismo `num_scales` (DEFAULT_PADDING_FACTORS)
-- así que el cache resultante es un reemplazo drop-in para --embedding-cache
en analyze_dinov2_nn_baseline.py / train_linear_reid_head.py.

Diferencia clave con el cache de DINOv2 (y con P4BackboneEncoder, el proxy
rápido crop-based): ahí se corre el encoder UNA VEZ POR CADA CROP/escala de
CADA observación. Acá se corre el backbone UNA SOLA VEZ POR FRAME (a
resolución nativa de detección, --detector-resize) -- todas las
observaciones/escalas de ese frame comparten el mismo feature map vía
ROI-Align, igual que se usaría en producción (un solo forward del detector,
ya corrido para trackear, reutilizado para ReID). Esto es lo que permite
medir la ventaja de VELOCIDAD de P4, no solo la señal (ver conversación).

Uso:
    python3 precompute_p4_roi_cache.py --split val --output-path p4_roi_cache_val.pt
    python3 precompute_p4_roi_cache.py --split test --output-path p4_roi_cache_test.pt

Luego:
    python3 analyze_dinov2_nn_baseline.py --split val --embedding-cache p4_roi_cache_val.pt
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'reid_lib'))  # library modules
import argparse
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.ops import roi_align
from tqdm import tqdm

from p4_backbone_encoder import P4ROIAlignEncoder
from reid_dataloader import DEFAULT_PADDING_FACTORS, MIN_CONTEXT_CROP_SIZE_PX, PrecomputedEmbeddingCache, _expand_bbox
from train_track_query_reid import build_track_index, resolve_sequence_dirs


def _collect_unique_observations_by_image(index):
    """{image_path: [obs, ...]} sobre observaciones únicas por (image_path,bbox)
    -- mismo criterio de dedup que run_precompute_cache (train_track_query_reid.py),
    pero agrupado por frame en vez de aplanado, para poder correr el backbone
    una sola vez por frame."""
    by_image, seen = {}, set()
    for track in index.tracks:
        for obs in track.observations:
            key = (obs.image_path, tuple(obs.bbox))
            if key in seen:
                continue
            seen.add(key)
            by_image.setdefault(obs.image_path, []).append(obs)
    return by_image, len(seen)


def run_precompute_p4_roi_cache(args):
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

    print(f"[INFO] Cargando P4ROIAlignEncoder (layer={args.layer}, config={args.p4_config}, checkpoint={args.p4_checkpoint})...")
    model = P4ROIAlignEncoder(args.p4_config, args.p4_checkpoint, layer=args.layer).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    resize = args.detector_resize
    resize_transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])
    padding_factors = list(DEFAULT_PADDING_FACTORS)
    if args.wide_scale_factor is not None:
        padding_factors[0] = args.wide_scale_factor
        print(f"[INFO] --wide-scale-factor {args.wide_scale_factor}: usando padding_factors={padding_factors} "
              f"(ablacion M10 -- default seria {DEFAULT_PADDING_FACTORS})")
    S = len(padding_factors)

    embeddings = {}
    t0 = time.time()
    for start in tqdm(range(0, len(image_paths), args.frame_batch_size), desc="precompute-p4-roi", dynamic_ncols=True):
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
             "(kfold_utils.list_all_sequences, los 3 splits físicos juntos) en vez de un split fijo. "
             "Necesario para --kfold-cv --kfold-skip-training --encoder p4 en analyze_dinov2_nn_baseline.py.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--layer", choices=["c4", "c5", "p4fpn", "p3fpn"], default="c4",
        help="Qué capa de RT-DETRv2 usar (ver LAYER_SPECS en p4_backbone_encoder.py). 'c4' = lo "
             "llamado 'P4' en el resto de la sesión (backbone crudo, pre-encoder). 'p4fpn' = P4 "
             "REAL (fusión FPN C5+C4 del encoder).",
    )
    parser.add_argument("--p4-config", default="../configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml")
    parser.add_argument("--p4-checkpoint", default="../damages_tracking/models/rtdetrv2_v3_qcs_videos.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detector-resize", type=int, default=640, help="Resolución de entrada al detector (igual convención que p4_reid_finetune.py).")
    parser.add_argument("--roi-output-size", type=int, default=7, help="Igual que ROI_OUTPUT_SIZE en p4_reid_finetune.py.")
    parser.add_argument("--frame-batch-size", type=int, default=16, help="Frames únicos por forward del backbone (no observaciones).")
    parser.add_argument(
        "--wide-scale-factor", type=float, default=None,
        help="[Ablacion M10] Sobreescribe DEFAULT_PADDING_FACTORS[0] (la escala 'amplia'/vista mas "
             "antigua del historial del banco, default 6.0) con este valor -- intermedia (1.6) y "
             "ajustada (1.0) quedan fijas. None = comportamiento default sin cambios.",
    )
    args = parser.parse_args()
    run_precompute_p4_roi_cache(args)


if __name__ == "__main__":
    main()
