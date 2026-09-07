"""
Ad-hoc: genera imagenes (query + banco + heatmap de similitud/umbral) para
un puñado de casos ESTRICTOS reales de ByteTrack, usando C4 puro (sin cabeza,
sin embedding_cache -- encoder real corriendo sobre los crops) y el umbral
YA calibrado por fold (kfold_results.json de p4_roi_kfold_baseline), tal
cual la Tabla 2 (all_methods_same_cases_pct) evalua "C4". No reentrena nada.

Uso: python3 visualize_c4_proof.py
"""
import json
import os

import torch
from torch.utils.data import DataLoader

from analyze_dinov2_nn_baseline import _best_candidate, _embed_batch
from kfold_utils import list_all_sequences
from p4_backbone_encoder import P4BackboneEncoder
from reid_dataloader import collate_episodes_with_video_ids
from train_track_query_reid import build_track_index, visualize_episode
from tracker_failure_dataset import load_new_id_born_cases, TrackerFailureReIDDataset

TRACKER = "bytetrack"
CFG_DIR = "p4_roi_kfold_baseline"
REPORT = "kfold_dinov2_raw/reid_gap_bytetrack_kfold_all/reid_gap_report.json"
OUT_DIR = "c4_proof_visuals"
HEATMAP_TEMPERATURE = 0.05
DEVICE = "cpu"
N_EXAMPLES = 6


def fold_for_sequence(fold_results, seq_name):
    for fr in fold_results:
        if seq_name in fr["test_seqs"]:
            return fr
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[INFO] Cargando P4BackboneEncoder (C4, cpu)...")
    model = P4BackboneEncoder(
        "../configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
        "../damages_tracking/models/rtdetrv2_v3_qcs_videos.pth",
        layer="c4",
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    seq_lookup = {os.path.basename(s.rstrip("/")): s for s in list_all_sequences()}
    with open(os.path.join(CFG_DIR, "kfold_results.json")) as f:
        fold_results = json.load(f)

    all_cases = load_new_id_born_cases(REPORT)  # solo new_id_born = casos limpios, brecha real
    with open(REPORT) as f:
        raw = json.load(f)
    strict = [c for c in raw if c["outcome"] != "unevaluated"]
    # priorizamos new_id_born (mas ilustrativo: banco vs query tras oclusion real);
    # si no alcanzan N_EXAMPLES, completamos con el resto de los estrictos (id_swap incluido)
    ordered = all_cases + [c for c in strict if c not in all_cases]

    n_done = 0
    for case in ordered:
        if n_done >= N_EXAMPLES:
            break
        seq_name = case["sequence"]
        fr = fold_for_sequence(fold_results, seq_name)
        if fr is None:
            print(f"[WARN] {seq_name}: no encontrada en ningun fold, se omite")
            continue
        threshold = fr["threshold"]
        test_seq_dirs = [seq_lookup[n] for n in fr["test_seqs"]]
        index = build_track_index(None, sequence_dirs=test_seq_dirs)

        dataset = TrackerFailureReIDDataset(
            index, [case], same_video_only=True,
            max_bank_size=16, num_track_history=4, query_wide_factor=9.0,
            seed=0, return_video_ids=True,
        )
        if len(dataset) == 0:
            print(f"[WARN] {seq_name}: no se pudo mapear al indice, se omite")
            continue

        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_episodes_with_video_ids)
        batch = next(iter(loader))
        track_summary, query_embeds, bank_mask, S = _embed_batch(model, batch, DEVICE, head=None)
        target = int(batch["target"][0].item())
        valid_slots, sims, best_slot, best_sim = _best_candidate(query_embeds[0], track_summary[0], bank_mask[0])
        pred_is_match = best_sim >= threshold
        pred_slot = best_slot if pred_is_match else S

        fake_logits = torch.full((S + 1,), float("-inf"))
        fake_logits[torch.from_numpy(valid_slots)] = torch.from_numpy(sims).float()
        fake_logits[S] = threshold
        fake_logits = fake_logits / HEATMAP_TEMPERATURE

        safe_key = f"{seq_name}__{case['gt_identity_id']}__{case['frame_before']}_{case['frame_after']}".replace("/", "_")
        out_path = os.path.join(OUT_DIR, f"{safe_key}__c4_puro.jpg")
        visualize_episode(batch, 0, out_path, pred=pred_slot, logits=fake_logits)
        correcto = pred_slot == target
        print(
            f"[OK] {out_path} fold={fr['fold']} threshold={threshold:.4f} best_sim={best_sim:.4f} "
            f"best_slot={best_slot} target={target} pred={pred_slot} correcto={correcto} outcome={case['outcome']}"
        )
        n_done += 1

    print(f"\n[DONE] {n_done} imagenes generadas en {OUT_DIR}/")


if __name__ == "__main__":
    main()
