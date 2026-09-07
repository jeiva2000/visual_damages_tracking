import json
import os

import torch

from analyze_dinov2_nn_baseline import _best_candidate, _build_loader, _embed_batch
from kfold_utils import list_all_sequences
from p4_backbone_encoder import P4BackboneEncoder
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

    # force_reid=False -- mezcla REAL_REID/SYNTHETIC_SPLIT/NEW_TRACK (case_probs
    # default incluye NEW_TRACK=0.4), a diferencia de TrackerFailureReIDDataset
    # (siempre hay match real). Iteramos episodios sueltos (batch_size=1) hasta
    # encontrar uno con target==S (query genuinamente nueva, fuera del banco).
    loader = _build_loader(
        index, episodes=200, max_bank_size=16, num_track_history=4,
        query_wide_factor=9.0, force_reid=False, seed=SEED, batch_size=1,
    )

    found = None
    for i, batch in enumerate(loader):
        S_batch = batch["bank_images"].shape[1]
        target = int(batch["target"][0].item())
        if target == S_batch:
            found = (i, batch, S_batch)
            break
    if found is None:
        print("[WARN] no se encontro ningun episodio NEW_TRACK en las primeras 200, subi --episodes")
        return

    i, batch, S = found
    video_id = batch.get("video_ids", [["?"]])[0] if "video_ids" in batch else "?"
    track_summary, query_embeds, bank_mask, S2 = _embed_batch(model, batch, DEVICE, head=None)
    target = int(batch["target"][0].item())
    valid_slots, sims, best_slot, best_sim = _best_candidate(query_embeds[0], track_summary[0], bank_mask[0])
    pred_is_match = best_sim >= threshold
    pred_slot = best_slot if pred_is_match else S

    print(f"\n[EPISODIO #{i}] NEW_TRACK real (target={target}=S, la identidad de la query no esta en el banco)")
    print(f"[FOLD] {FOLD_IDX}  threshold={threshold:.4f}")
    print("[SIMILITUDES CRUDAS por candidato valido]")
    for slot, sim in zip(valid_slots.tolist(), sims.tolist()):
        supera = "SI" if sim >= threshold else "no"
        marca = "  <- ARGMAX (mayor sim)" if slot == best_slot else ""
        print(f"  candidato {slot}: sim={sim:.4f}  (>=umbral? {supera}){marca}")
    print(f"\n[DECISION] best_slot={best_slot} best_sim={best_sim:.4f} >= threshold={threshold:.4f} -> pred_is_match={pred_is_match}")
    print(f"[PRED FINAL] pred_slot={pred_slot} (S={S}='no match')  target={target}  correcto={pred_slot == target}")

    fake_logits = torch.full((S + 1,), float("-inf"))
    fake_logits[torch.from_numpy(valid_slots)] = torch.from_numpy(sims).float()
    fake_logits[S] = threshold
    fake_logits = fake_logits / 0.05

    out_path = os.path.join(OUT_DIR, f"new_track_episode{i}_fold{FOLD_IDX}__c4_puro.jpg")
    visualize_episode(batch, 0, out_path, pred=pred_slot, logits=fake_logits)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
