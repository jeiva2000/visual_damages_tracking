import json
import os

import torch

from analyze_dinov2_nn_baseline import _best_candidate, _embed_batch
from kfold_utils import list_all_sequences
from p4_backbone_encoder import P4BackboneEncoder
from reid_dataloader import ReIDEpisodeDataset, collate_episodes_with_video_ids
from train_track_query_reid import build_track_index, visualize_episode

CFG_DIR = "p4_roi_kfold_baseline"
FOLD_IDX = 0
DEVICE = "cpu"
OUT_DIR = "c4_proof_visuals"
SEEDS = range(1, 40)


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
    fr = next(r for r in fold_results if r["fold"] == FOLD_IDX)
    threshold = fr["threshold"]
    test_seq_dirs = [seq_lookup[n] for n in fr["test_seqs"]]
    index = build_track_index(None, sequence_dirs=test_seq_dirs)

    n_new_track_seen = 0
    for seed in SEEDS:
        dataset = ReIDEpisodeDataset(
            index, episodes_per_epoch=100, max_bank_size=16, num_track_history=4,
            query_wide_factor=9.0, force_reid=False, hard_distractor_ratio=1.0,
            synthetic_gap_max=0, seed=seed, return_video_ids=True,
        )
        for i in range(len(dataset)):
            try:
                item = dataset[i]
            except IndexError:
                continue
            S_item = item["bank_images"].shape[0]
            target = int(item["target"].item())
            if target != S_item:
                continue
            n_new_track_seen += 1

            batch = collate_episodes_with_video_ids([item])
            track_summary, query_embeds, bank_mask, S = _embed_batch(model, batch, DEVICE, head=None)
            valid_slots, sims, best_slot, best_sim = _best_candidate(query_embeds[0], track_summary[0], bank_mask[0])
            pred_is_match = best_sim >= threshold
            pred_slot = best_slot if pred_is_match else S

            if pred_slot != S:
                continue  # sigue siendo falso positivo, seguimos buscando uno correcto

            # encontrado: rechazo correcto (pred=no match, y de verdad no habia match)
            query_video_id = batch["query_video_id"][0]
            bank_video_ids = batch["bank_video_ids"][0]
            print(f"\n[ENCONTRADO] seed={seed} episodio#{i} (visto #{n_new_track_seen} de NEW_TRACK en total)")
            print(f"[QUERY video_id] {query_video_id}")
            print(f"[FOLD] {FOLD_IDX}  threshold={threshold:.4f}")
            print("[SIMILITUDES CRUDAS por candidato valido]")
            for slot, sim in zip(valid_slots.tolist(), sims.tolist()):
                supera = "SI" if sim >= threshold else "no"
                marca = "  <- ARGMAX (mayor sim, pero AUN ASI no supera umbral)" if slot == best_slot else ""
                same = "MISMO video" if bank_video_ids[slot] == query_video_id else f"OTRO video ({bank_video_ids[slot]})"
                print(f"  candidato {slot}: sim={sim:.4f}  (>=umbral? {supera})  [{same}]{marca}")
            print(f"\n[DECISION] best_sim={best_sim:.4f} < threshold={threshold:.4f} -> pred_is_match=False")
            print(f"[PRED FINAL] pred_slot={pred_slot} (='no match')  target={target}  correcto=True")

            fake_logits = torch.full((S + 1,), float("-inf"))
            fake_logits[torch.from_numpy(valid_slots)] = torch.from_numpy(sims).float()
            fake_logits[S] = threshold
            fake_logits = fake_logits / 0.05

            out_path = os.path.join(OUT_DIR, f"new_track_CORRECT_seed{seed}_ep{i}_fold{FOLD_IDX}__c4_puro.jpg")
            visualize_episode(batch, 0, out_path, pred=pred_slot, logits=fake_logits)
            print(f"[OK] {out_path}")
            print(f"\n[TOTAL] {n_new_track_seen} episodios NEW_TRACK vistos antes de encontrar uno correcto")
            return

    print(f"[WARN] no se encontro ningun rechazo correcto en {n_new_track_seen} episodios NEW_TRACK vistos, ampliar SEEDS")


if __name__ == "__main__":
    main()
