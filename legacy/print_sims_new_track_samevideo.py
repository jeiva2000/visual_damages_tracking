import json
import os

import torch
from torch.utils.data import DataLoader

from analyze_dinov2_nn_baseline import _best_candidate, _embed_batch
from kfold_utils import list_all_sequences
from p4_backbone_encoder import P4BackboneEncoder
from reid_dataloader import ReIDEpisodeDataset, collate_episodes_with_video_ids
from train_track_query_reid import build_track_index, visualize_episode

CFG_DIR = "p4_roi_kfold_baseline"
FOLD_IDX = 0
DEVICE = "cpu"
SEED = 7
OUT_DIR = "c4_proof_visuals"


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

    # hard_distractor_ratio=1.0 -- banco 100% mismo video (equivalente a
    # same_video_only=True de TrackerFailureReIDDataset), para que sea
    # directamente comparable con la metodologia real de la Tabla 2.
    dataset = ReIDEpisodeDataset(
        index, episodes_per_epoch=300, max_bank_size=16, num_track_history=4,
        query_wide_factor=9.0, force_reid=False, hard_distractor_ratio=1.0,
        synthetic_gap_max=0, seed=SEED, return_video_ids=True,
    )

    found = None
    n_errors = 0
    for i in range(len(dataset)):
        try:
            item = dataset[i]
        except IndexError:
            # bug real: con hard_distractor_ratio=1.0, si el video de la
            # query tiene 0 identidades adicionales, bank_crops queda vacio
            # y torch.zeros_like(bank_crops[0]) revienta -- se saltea.
            n_errors += 1
            continue
        S_item = item["bank_images"].shape[0]
        target = int(item["target"].item())
        if target == S_item:
            batch = collate_episodes_with_video_ids([item])
            found = (i, batch, S_item)
            break
    print(f"[INFO] {n_errors} episodios saltados por el bug de banco vacio")
    if found is None:
        print("[WARN] no se encontro ningun episodio NEW_TRACK, subi episodes_per_epoch")
        return

    i, batch, S = found
    track_summary, query_embeds, bank_mask, S2 = _embed_batch(model, batch, DEVICE, head=None)
    target = int(batch["target"][0].item())
    query_video_id = batch["query_video_id"][0]
    bank_video_ids = batch["bank_video_ids"][0]
    valid_slots, sims, best_slot, best_sim = _best_candidate(query_embeds[0], track_summary[0], bank_mask[0])
    pred_is_match = best_sim >= threshold
    pred_slot = best_slot if pred_is_match else S

    print(f"\n[EPISODIO #{i}] NEW_TRACK, hard_distractor_ratio=1.0 (banco 100% mismo video)")
    print(f"[QUERY video_id] {query_video_id}")
    print(f"[FOLD] {FOLD_IDX}  threshold={threshold:.4f}")
    print("[SIMILITUDES CRUDAS por candidato valido]")
    for slot, sim in zip(valid_slots.tolist(), sims.tolist()):
        supera = "SI" if sim >= threshold else "no"
        marca = "  <- ARGMAX (mayor sim)" if slot == best_slot else ""
        same = "MISMO video" if bank_video_ids[slot] == query_video_id else f"OTRO video ({bank_video_ids[slot]})"
        print(f"  candidato {slot}: sim={sim:.4f}  (>=umbral? {supera})  [{same}]{marca}")
    print(f"\n[DECISION] best_slot={best_slot} best_sim={best_sim:.4f} >= threshold={threshold:.4f} -> pred_is_match={pred_is_match}")
    print(f"[PRED FINAL] pred_slot={pred_slot} (S={S}='no match')  target={target}  correcto={pred_slot == target}")

    fake_logits = torch.full((S + 1,), float("-inf"))
    fake_logits[torch.from_numpy(valid_slots)] = torch.from_numpy(sims).float()
    fake_logits[S] = threshold
    fake_logits = fake_logits / 0.05

    out_path = os.path.join(OUT_DIR, f"new_track_samevideo_episode{i}_fold{FOLD_IDX}__c4_puro.jpg")
    visualize_episode(batch, 0, out_path, pred=pred_slot, logits=fake_logits)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
