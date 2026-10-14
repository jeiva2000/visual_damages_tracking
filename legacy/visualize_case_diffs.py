"""
Genera visualizaciones (query + banco + heatmap de similitud, ver
visualize_episode en train_track_query_reid.py) para los casos puntuales
donde dinov2_puro y v6style difieren (ver compare_kfold_cases.py) -- una
imagen por config por caso, para inspeccionar visualmente por qué un
modelo acierta y el otro no en ESE caso concreto. Reusa los checkpoints/
umbrales YA entrenados, sin reentrenar nada.

Uso:
    python3 visualize_case_diffs.py --comparison-json compare_dinov2_v6_bytetrack.json
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from analyze_dinov2_nn_baseline import _best_candidate, _embed_batch
from kfold_utils import list_all_sequences
from linear_reid_head import load_head
from reid_dataloader import collate_episodes_with_video_ids
from train_track_query_reid import build_track_index, visualize_episode
from tracker_failure_dataset import TrackerFailureReIDDataset, load_required_reid_cases

CONFIGS = {
    "dinov2_puro": {"dir": "kfold_dinov2_raw", "has_head": False},
    "v6style": {"dir": "kfold_v6style", "has_head": True},
}


def _load_fold_results(config_dir):
    with open(os.path.join(config_dir, "kfold_results.json")) as f:
        return json.load(f)


def _fold_for_sequence(fold_results, seq_name):
    for fr in fold_results:
        if seq_name in fr["test_seqs"]:
            return fr
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument("--max-bank-size", type=int, default=16)
    parser.add_argument("--num-track-history", type=int, default=4)
    parser.add_argument("--query-wide-factor", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heatmap-temperature", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--comparison-json", default="compare_dinov2_v6_bytetrack.json")
    parser.add_argument("--output-dir", default="case_diff_visuals")
    args = parser.parse_args()

    with open(args.comparison_json) as f:
        resultado = json.load(f)

    dinov2_falla_v6_ok = sorted(
        k for k in resultado["dinov2_puro"]
        if resultado["dinov2_puro"].get(k) is False and resultado["v6style"].get(k) is True
    )
    v6_falla_dinov2_ok = sorted(
        k for k in resultado["v6style"]
        if resultado["v6style"].get(k) is False and resultado["dinov2_puro"].get(k) is True
    )
    print(f"[INFO] {len(dinov2_falla_v6_ok)} casos dinov2_falla_v6_ok, {len(v6_falla_dinov2_ok)} casos v6_falla_dinov2_ok")

    events_path = os.path.join("kfold_v7_6fold", f"reid_gap_{args.tracker}_kfold_all", "required_reid_events.json")
    cases = load_required_reid_cases(events_path)
    cases_by_key = {
        f"{c['sequence']}__{c['gt_identity_id']}__{c['frame_before']}_{c['frame_after']}": c
        for c in cases
    }

    print(f"[INFO] Cargando {args.dinov2_model_name}...")
    model = AutoModel.from_pretrained(args.dinov2_model_name).to(args.device).eval()
    seq_lookup = {os.path.basename(s.rstrip("/")): s for s in list_all_sequences()}
    fold_results_by_config = {name: _load_fold_results(cfg["dir"]) for name, cfg in CONFIGS.items()}

    groups = {"dinov2_falla_v6_ok": dinov2_falla_v6_ok, "v6_falla_dinov2_ok": v6_falla_dinov2_ok}

    for group_name, keys in groups.items():
        out_dir = os.path.join(args.output_dir, group_name)
        os.makedirs(out_dir, exist_ok=True)
        for case_key in keys:
            case = cases_by_key[case_key]
            safe_key = case_key.replace("/", "_")
            for config_name, cfg in CONFIGS.items():
                fr = _fold_for_sequence(fold_results_by_config[config_name], case["sequence"])
                if fr is None:
                    print(f"[WARN] {case_key}: secuencia no encontrada en ningún fold de {config_name}, se omite")
                    continue
                threshold = fr["threshold"]
                test_seq_dirs = [seq_lookup[n] for n in fr["test_seqs"]]
                index = build_track_index(None, sequence_dirs=test_seq_dirs)

                head = None
                if cfg["has_head"]:
                    head_path = os.path.join(cfg["dir"], f"fold_{fr['fold']}", "best_head.pt")
                    head = load_head(head_path, args.device)

                dataset = TrackerFailureReIDDataset(
                    index, [case], same_video_only=True,
                    max_bank_size=args.max_bank_size, num_track_history=args.num_track_history,
                    query_wide_factor=args.query_wide_factor, seed=args.seed, return_video_ids=True,
                )
                if len(dataset) == 0:
                    print(f"[WARN] {case_key}/{config_name}: no se pudo mapear al índice, se omite")
                    continue
                loader = DataLoader(dataset, batch_size=1, collate_fn=collate_episodes_with_video_ids)
                batch = next(iter(loader))
                track_summary, query_embeds, bank_mask, S = _embed_batch(model, batch, args.device, head=head)
                target = int(batch["target"][0].item())
                valid_slots, sims, best_slot, best_sim = _best_candidate(query_embeds[0], track_summary[0], bank_mask[0])
                pred_is_match = best_sim >= threshold
                pred_slot = best_slot if pred_is_match else S

                fake_logits = torch.full((S + 1,), float("-inf"))
                fake_logits[torch.from_numpy(valid_slots)] = torch.from_numpy(sims).float()
                fake_logits[S] = threshold
                fake_logits = fake_logits / args.heatmap_temperature

                out_path = os.path.join(out_dir, f"{safe_key}__{config_name}.jpg")
                visualize_episode(batch, 0, out_path, pred=pred_slot, logits=fake_logits)
                print(f"[OK] {out_path} (target={target} pred={pred_slot} correcto={pred_slot == target})")


if __name__ == "__main__":
    main()
