"""
Compara, CASO POR CASO (fallos reales de un tracker, ya evaluados por la
validación cruzada), qué casos corrige una configuración que otra falla --
y viceversa. Reusa los checkpoints/umbrales YA ENTRENADOS de las 4 corridas
de k-fold (kfold_dinov2_raw, kfold_v6style, kfold_intermedio, kfold_v7_6fold),
sin reentrenar ni recalibrar nada -- cada caso se evalúa con el modelo y el
umbral del fold al que pertenece su secuencia de test (igual que hizo
run_kfold_cv originalmente).

Uso:
    python3 compare_kfold_cases.py --tracker bytetrack
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'reid_lib'))  # library modules
import argparse
import json
import os

import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from transformers import AutoModel

from analyze_dinov2_nn_baseline import _best_candidate, _embed_batch
from kfold_utils import list_all_sequences
from linear_reid_head import load_head
from reid_dataloader import PrecomputedEmbeddingCache, collate_episodes_with_video_ids
from train_track_query_reid import build_track_index
from tracker_failure_dataset import TrackerFailureReIDDataset, load_required_reid_cases

CONFIGS = {
    # head=None en las 2 primeras (--kfold-skip-training: nunca hubo head entrenada)
    "dinov2_puro": {"dir": "kfold_dinov2_raw", "has_head": False, "embedding_cache": None},
    "p4_roi": {"dir": "p4_roi_kfold_baseline", "has_head": False, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    "v6style": {"dir": "kfold_v6style", "has_head": True, "embedding_cache": None},
    "intermedio": {"dir": "kfold_intermedio", "has_head": True, "embedding_cache": None},
    "v7": {"dir": "kfold_v7_6fold", "has_head": True, "embedding_cache": None},
    "p4_v6head": {"dir": "p4_roi_kfold_head_v6style", "has_head": True, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    # --- P4-FPN real para bytetrack/botsort (agregado 2026-08-16) -- mismo
    # fold_assignment que dinov2_puro/p4_roi, cache p4fpn_roi_cache_kfold_pool.pt. ---
    "p4fpn_roi": {"dir": "p4fpn_roi_kfold_baseline", "has_head": False, "embedding_cache": "p4fpn_roi_cache_kfold_pool.pt"},
    "p4fpn_v6head": {"dir": "p4fpn_roi_kfold_head_v6style", "has_head": True, "embedding_cache": "p4fpn_roi_cache_kfold_pool.pt"},
    # --- OSNet puro (osnet_x0_25_msmt17, ReID real preentrenado en MSMT17 --
    # el mismo modelo que BotSort/DeepOcSort usan de forma nativa, ver
    # main_track.py::load_reid rama default) -- agregado en respuesta al
    # review M3: falta un ReID real (no autosupervisado generico) en la
    # comparacion. Cache generado por precompute_osnet_embedding_cache.py. ---
    "osnet_roi": {"dir": "osnet_puro_kfold_baseline", "has_head": False, "embedding_cache": "osnet_cache_kfold_pool.pt"},
    # --- CLIP-ReID / VehicleID (clip_vehicleid.pt -- ReID real entrenado
    # especificamente en VEHICULOS, no personas -- el mas cercano a nuestro
    # dominio disponible off-the-shelf en boxmot). Mismo cache/protocolo de
    # crop que osnet_roi, ver precompute_osnet_embedding_cache.py. ---
    "clipvehicleid_roi": {"dir": "clipvehicleid_puro_kfold_baseline", "has_head": False, "embedding_cache": "clipvehicleid_cache_kfold_pool.pt"},
    # --- ocsort/deepocsort (agregados 2026-08-15, ver --trackers ocsort
    # deepocsort en analyze_dinov2_nn_baseline.py) -- dirs propios, distintos
    # de los de arriba (bytetrack/botsort), aunque comparten el mismo
    # fold_assignment.json y el mismo pool de 64 secuencias. ---
    "dinov2_puro_oc": {"dir": "kfold_ocsort_deepocsort_baseline", "has_head": False, "embedding_cache": None},
    "c4_roi_oc": {"dir": "c4_roi_kfold_baseline_ocsort_deepocsort", "has_head": False, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    "c4_v6head_oc": {"dir": "c4_roi_kfold_head_v6style_ocsort_deepocsort", "has_head": True, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    "p4fpn_roi_oc": {"dir": "p4fpn_roi_kfold_baseline_ocsort_deepocsort", "has_head": False, "embedding_cache": "p4fpn_roi_cache_kfold_pool.pt"},
    "p4fpn_v6head_oc": {"dir": "p4fpn_roi_kfold_head_v6style_ocsort_deepocsort", "has_head": True, "embedding_cache": "p4fpn_roi_cache_kfold_pool.pt"},
    "osnet_roi_oc": {"dir": "osnet_puro_kfold_baseline_ocsort_deepocsort", "has_head": False, "embedding_cache": "osnet_cache_kfold_pool.pt"},
    "clipvehicleid_roi_oc": {"dir": "clipvehicleid_puro_kfold_baseline_ocsort_deepocsort", "has_head": False, "embedding_cache": "clipvehicleid_cache_kfold_pool.pt"},
    # --- Resto de cabezas entrenadas sobre C4 (H2 intermedio, H3 v7, H4
    # pure-triplet; H1 = p4_v6head / c4_v6head_oc arriba) -- mismos dirs cuyos
    # kfold_results.json dan los valores de tab:kfold_identity_acc. Para
    # McNemar de C4 puro vs cada cabeza sobre los mismos eventos. ---
    "p4_intermedio_head": {"dir": "p4_roi_kfold_head_intermedio", "has_head": True, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    "p4_v7head": {"dir": "p4_roi_kfold_head_v7", "has_head": True, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    "p4_puretriplet_head": {"dir": "p4_roi_kfold_head_pure_triplet", "has_head": True, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    "c4_intermedio_head_oc": {"dir": "c4_roi_kfold_head_intermedio_ocsort_deepocsort", "has_head": True, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    "c4_v7head_oc": {"dir": "c4_roi_kfold_head_v7_ocsort_deepocsort", "has_head": True, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    "c4_puretriplet_head_oc": {"dir": "c4_roi_kfold_head_pure_triplet_ocsort_deepocsort", "has_head": True, "embedding_cache": "p4_roi_cache_kfold_pool.pt"},
    # --- EXPERIMENTO AISLADO NOVIDEO (detector rtdetrv2_v3_qcs_cardd_NOVIDEO.pth,
    # entrenado solo con QCS v3 + CarDD, sin frames de video) -- ver
    # exp_novideo_detector/run_novideo_kfold.sh. Mismos eventos y folds que
    # p4_roi / c4_roi_oc / p4fpn_roi / p4fpn_roi_oc; solo cambian features y umbrales. ---
    "c4_novideo": {"dir": "exp_novideo_detector/c4_kfold_baseline_NOVIDEO", "has_head": False, "embedding_cache": "exp_novideo_detector/c4_roi_cache_kfold_pool_NOVIDEO.pt"},
    "c4_novideo_oc": {"dir": "exp_novideo_detector/c4_kfold_baseline_oc_NOVIDEO", "has_head": False, "embedding_cache": "exp_novideo_detector/c4_roi_cache_kfold_pool_NOVIDEO.pt"},
    "p4fpn_novideo": {"dir": "exp_novideo_detector/p4fpn_kfold_baseline_NOVIDEO", "has_head": False, "embedding_cache": "exp_novideo_detector/p4fpn_roi_cache_kfold_pool_NOVIDEO.pt"},
    "p4fpn_novideo_oc": {"dir": "exp_novideo_detector/p4fpn_kfold_baseline_oc_NOVIDEO", "has_head": False, "embedding_cache": "exp_novideo_detector/p4fpn_roi_cache_kfold_pool_NOVIDEO.pt"},
}


def _load_fold_results(config_dir):
    with open(os.path.join(config_dir, "kfold_results.json")) as f:
        return json.load(f)


def _fold_for_sequence(fold_results, seq_name):
    for fr in fold_results:
        if seq_name in fr["test_seqs"]:
            return fr
    return None


def _predict_case(
    model, head, index, case, threshold, device, max_bank_size, num_track_history, query_wide_factor, seed,
    embedding_cache=None,
):
    """(correct: bool, pred_correcto_pero_no_match: ...) para UN caso -- reusa
    TrackerFailureReIDDataset (same_video_only=True, igual que run_kfold_cv/
    _evaluate_tracker_on_cases) y _best_candidate. Devuelve None si el caso
    no se pudo mapear al índice (mismo motivo que los WARN ya vistos en
    infer_track_reid_gaps.py: required_reid sin brecha temporal real).

    `embedding_cache`, si se pasa (ver CONFIGS["p4_roi"]), hace que
    _embed_batch use los embeddings P4 ya calculados (ROI-Align) en vez de
    correr `model` -- `model` puede ser None en ese caso."""
    dataset = TrackerFailureReIDDataset(
        index, [case], same_video_only=True,
        max_bank_size=max_bank_size, num_track_history=num_track_history,
        query_wide_factor=query_wide_factor, seed=seed, return_video_ids=True,
        embedding_cache=embedding_cache,
    )
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_episodes_with_video_ids)
    batch = next(iter(loader))
    track_summary, query_embeds, bank_mask, S = _embed_batch(model, batch, device, head=head)
    target = int(batch["target"][0].item())
    _, _, best_slot, best_sim = _best_candidate(query_embeds[0], track_summary[0], bank_mask[0])
    pred_is_match = best_sim >= threshold
    pred_slot = best_slot if pred_is_match else S
    return pred_slot == target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort", "ocsort", "deepocsort"], default="bytetrack")
    parser.add_argument("--max-bank-size", type=int, default=16)
    parser.add_argument("--num-track-history", type=int, default=4)
    parser.add_argument("--query-wide-factor", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=None, help="Si se pasa, guarda el detalle completo por caso.")
    parser.add_argument(
        "--configs", nargs="+", default=["dinov2_puro", "v6style"], choices=list(CONFIGS),
        help="Configs a evaluar (ver CONFIGS) -- la PRIMERA es el baseline contra el que se comparan "
             "todas las demás (diffs + McNemar exacto). Default: comparación original de la Corrida 8.",
    )
    args = parser.parse_args()

    configs = {name: CONFIGS[name] for name in args.configs}
    baseline = args.configs[0]

    needs_dinov2 = any(cfg["embedding_cache"] is None for cfg in configs.values())
    model = None
    if needs_dinov2:
        print(f"[INFO] Cargando {args.dinov2_model_name}...")
        model = AutoModel.from_pretrained(args.dinov2_model_name).to(args.device).eval()

    embedding_caches = {}
    for cfg in configs.values():
        cache_path = cfg["embedding_cache"]
        if cache_path and cache_path not in embedding_caches:
            print(f"[INFO] Cargando embedding cache: {cache_path}")
            embedding_caches[cache_path] = PrecomputedEmbeddingCache.load(cache_path)

    seq_lookup = {os.path.basename(s.rstrip("/")): s for s in list_all_sequences()}

    # La fuente de los casos GLOBALES puede leerse de CUALQUIER config del set
    # (todas comparten el mismo reid_gap_{tracker}_kfold_all -- las que no lo
    # generaron directo lo symlinkean desde la que sí, ver run_overnight_resume.sh/
    # run_p4fpn_ocsort_deepocsort_pipeline.sh). Se usa el baseline (primer
    # --configs) en vez de hardcodear "v7" -- ese dir no existe para los
    # configs _oc (ocsort/deepocsort), agregados 2026-08-15.
    events_path = os.path.join(configs[baseline]["dir"], f"reid_gap_{args.tracker}_kfold_all", "required_reid_events.json")
    cases = load_required_reid_cases(events_path)
    print(f"[INFO] {len(cases)} casos globales ({args.tracker}, required_reid)")

    fold_results_by_config = {name: _load_fold_results(cfg["dir"]) for name, cfg in configs.items()}

    # resultado[config][case_key] = True/False/None (None = no mapeable en ese fold)
    resultado = {name: {} for name in configs}

    for name, cfg in configs.items():
        print(f"\n[INFO] ==== Config: {name} ====")
        cache = embedding_caches.get(cfg["embedding_cache"])
        model_for_config = None if cache is not None else model
        fold_results = fold_results_by_config[name]
        for fr in fold_results:
            fold_idx = fr["fold"]
            threshold = fr["threshold"]
            test_seq_names = set(fr["test_seqs"])
            fold_cases = [c for c in cases if c["sequence"] in test_seq_names]
            if not fold_cases:
                continue
            test_seq_dirs = [seq_lookup[n] for n in test_seq_names]
            test_index = build_track_index(None, sequence_dirs=test_seq_dirs)

            head = None
            if cfg["has_head"]:
                head_path = os.path.join(cfg["dir"], f"fold_{fold_idx}", "best_head.pt")
                head = load_head(head_path, args.device)

            for case in fold_cases:
                case_key = f"{case['sequence']}__{case['gt_identity_id']}__{case['frame_before']}_{case['frame_after']}"
                correct = _predict_case(
                    model_for_config, head, test_index, case, threshold, args.device,
                    args.max_bank_size, args.num_track_history, args.query_wide_factor, args.seed,
                    embedding_cache=cache,
                )
                resultado[name][case_key] = correct
            print(f"[INFO] fold {fold_idx}: {len(fold_cases)} casos evaluados")

    # -- comparaciones pareadas, todas contra `baseline` -- #
    all_keys = set()
    for name in configs:
        all_keys.update(resultado[name].keys())

    def _diff(a, b):
        """casos donde `a` falla (correct=False) y `b` acierta (correct=True)."""
        out = []
        for k in sorted(all_keys):
            ra, rb = resultado[a].get(k), resultado[b].get(k)
            if ra is False and rb is True:
                out.append(k)
        return out

    print("\n" + "=" * 70)
    for name in args.configs[1:]:
        for a, b in [(baseline, name), (name, baseline)]:
            d = _diff(a, b)
            print(f"\n[{a} FALLA, {b} ACIERTA] ({len(d)} casos)")
            for k in d:
                print(f"  {k}")

    for name in args.configs[1:]:
        n_common_keys = sum(
            1 for k in all_keys
            if resultado[baseline].get(k) is not None and resultado[name].get(k) is not None
        )
        n_both_correct = sum(
            1 for k in all_keys
            if resultado[baseline].get(k) is True and resultado[name].get(k) is True
        )
        n_both_wrong = sum(
            1 for k in all_keys
            if resultado[baseline].get(k) is False and resultado[name].get(k) is False
        )
        b_wins = len(_diff(baseline, name))  # baseline falla, name acierta
        a_wins = len(_diff(name, baseline))  # name falla, baseline acierta
        n_discordant = b_wins + a_wins
        # McNemar exacto = binomial de dos colas sobre los pares discordantes (b_wins vs a_wins), p=0.5.
        pvalue = binomtest(min(b_wins, a_wins), n_discordant, 0.5).pvalue if n_discordant > 0 else float("nan")
        print(
            f"\n[RESUMEN {baseline} vs {name}] casos comparables={n_common_keys}  "
            f"ambos_ok={n_both_correct}  ambos_mal={n_both_wrong}  "
            f"{baseline}_falla_{name}_ok={b_wins}  {name}_falla_{baseline}_ok={a_wins}  "
            f"McNemar exacto p={pvalue:.5f}"
        )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(resultado, f, indent=2)
        print(f"\n[OK] Detalle completo guardado en {args.output_json}")


if __name__ == "__main__":
    main()
