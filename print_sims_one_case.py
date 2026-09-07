import json
import os

from torch.utils.data import DataLoader

from analyze_dinov2_nn_baseline import _best_candidate, _embed_batch
from kfold_utils import list_all_sequences
from p4_backbone_encoder import P4BackboneEncoder
from reid_dataloader import collate_episodes_with_video_ids
from train_track_query_reid import build_track_index
from tracker_failure_dataset import TrackerFailureReIDDataset

CFG_DIR = "p4_roi_kfold_baseline"
REPORT = "kfold_dinov2_raw/reid_gap_bytetrack_kfold_all/reid_gap_report.json"
DEVICE = "cpu"

TARGET_CASE = {
    "sequence": "video_Mpn2iv5EwC8BQM4Guf2_11",
    "gt_identity_id": "video_Mpn2iv5EwC8BQM4Guf2_11__2",
    "frame_before": 346, "frame_after": 423,
}


def fold_for_sequence(fold_results, seq_name):
    for fr in fold_results:
        if seq_name in fr["test_seqs"]:
            return fr
    return None


def main():
    with open(REPORT) as f:
        raw = json.load(f)
    case = next(
        c for c in raw
        if c["sequence"] == TARGET_CASE["sequence"]
        and c["gt_identity_id"] == TARGET_CASE["gt_identity_id"]
        and c["frame_before"] == TARGET_CASE["frame_before"]
        and c["frame_after"] == TARGET_CASE["frame_after"]
    )

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

    fr = fold_for_sequence(fold_results, case["sequence"])
    threshold = fr["threshold"]
    test_seq_dirs = [seq_lookup[n] for n in fr["test_seqs"]]
    index = build_track_index(None, sequence_dirs=test_seq_dirs)

    dataset = TrackerFailureReIDDataset(
        index, [case], same_video_only=True,
        max_bank_size=16, num_track_history=4, query_wide_factor=9.0,
        seed=0, return_video_ids=True,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_episodes_with_video_ids)
    batch = next(iter(loader))
    track_summary, query_embeds, bank_mask, S = _embed_batch(model, batch, DEVICE, head=None)
    target = int(batch["target"][0].item())
    valid_slots, sims, best_slot, best_sim = _best_candidate(query_embeds[0], track_summary[0], bank_mask[0])

    print(f"\n[CASO] {case['sequence']} id={case['gt_identity_id']} frames={case['frame_before']}->{case['frame_after']} outcome={case['outcome']}")
    print(f"[FOLD] {fr['fold']}  threshold={threshold:.4f}")
    print(f"[TARGET] slot={target} (S={S} = 'no match')")
    print("[SIMILITUDES CRUDAS por candidato valido]")
    for slot, sim in zip(valid_slots.tolist(), sims.tolist()):
        tags = []
        if slot == target:
            tags.append("MATCH (GT)")
        if slot == best_slot:
            tags.append("ARGMAX (mayor sim)")
        tag = f"  <- {' + '.join(tags)}" if tags else ""
        supera = "SI" if sim >= threshold else "no"
        print(f"  candidato {slot}: sim={sim:.4f}  (>=umbral? {supera}){tag}")
    print(f"\n[DECISION] best_slot={best_slot} best_sim={best_sim:.4f} >= threshold={threshold:.4f} -> pred_is_match={best_sim >= threshold}")
    pred_slot = best_slot if best_sim >= threshold else S
    print(f"[PRED FINAL] pred_slot={pred_slot}  target={target}  correcto={pred_slot == target}")


if __name__ == "__main__":
    main()
