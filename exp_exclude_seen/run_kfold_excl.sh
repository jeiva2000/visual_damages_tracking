#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO -- k-fold (tab:kfold_identity_acc) SIN las 10 secuencias
# que probablemente vio el detector actual (etiqueta EXCL).
# -----------------------------------------------------------------------------
# Mismos comandos que las corridas vigentes (run_table7_rerun_postfix_fastonly.sh,
# run_dinov2_postfix_gpu.sh, run_clipvehicleid_kfold_baseline_*.sh), con:
#   - el MISMO fold_assignment de cada corrida, quitando esas 10 secuencias de
#     todos los folds -> los umbrales se RECALIBRAN sin ellas y sus eventos no
#     se evalúan;
#   - los MISMOS eventos (reid_gap_*_kfold_all copiados con -L: en las corridas
#     originales son enlaces simbólicos relativos);
#   - salidas en exp_exclude_seen/<corrida>_EXCL, nada pisa lo existente.
# Cabezas entrenadas (H1-H4): no incluidas (requieren re-entrenar por fold).
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
APP=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking
EXP="$APP/exp_exclude_seen"
ONNX="$APP/checkpoints/rtdetrv2_v3_qcs_videos.onnx"   # requerido por --kfold-cv; no se usa (eventos ya existen)
SEEN="touaregmedina classiccar video6 chevroletazul video2 video4 4x4verde chevroletdorado video3 video5"
cd "$APP"
mkdir -p "$EXP"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

run() {  # run <label> <src_dir> <trackers> <encoder> <cache|-> <device>
  local label="$1" src="$2" trackers="$3" enc="$4" cache="$5" dev="$6"
  local out="$EXP/$(basename "$src")_EXCL"
  if [ -f "$out/kfold_results.json" ]; then echo "[$(ts)] $label ya hecho"; return; fi
  echo "[$(ts)] ==== $label ===="
  mkdir -p "$out"
  # kfold_ocsort_deepocsort_baseline no guarda fold_assignment: su corrida original usó el de
  # kfold_dinov2_raw (idéntico a todos los demás, md5 599ff319)
  local fa_src="$src/fold_assignment.json"; [ -f "$fa_src" ] || fa_src=kfold_dinov2_raw/fold_assignment.json
  python3 - "$fa_src" "$out/fold_assignment.json" $SEEN <<'PY'
import json, os, sys
src, dst, seen = sys.argv[1], sys.argv[2], set(sys.argv[3:])
fa = json.load(open(src))
drop = lambda seqs: [s for s in seqs if os.path.basename(s.rstrip("/")) not in seen]
out = {k: drop(v) for k, v in fa.items()} if isinstance(fa, dict) else [drop(v) for v in fa]
json.dump(out, open(dst, "w"), indent=1)
n0 = sum(len(v) for v in (fa.values() if isinstance(fa, dict) else fa))
n1 = sum(len(v) for v in (out.values() if isinstance(out, dict) else out))
print(f"   fold_assignment: {n0} -> {n1} secuencias")
PY
  for t in $trackers; do
    [ -d "$out/reid_gap_${t}_kfold_all" ] || cp -rL "$src/reid_gap_${t}_kfold_all" "$out/"
  done
  local cache_arg=""; [ "$cache" != "-" ] && cache_arg="--embedding-cache $cache"
  python3 analyze_dinov2_nn_baseline.py --kfold-cv --kfold-skip-training --encoder "$enc" $cache_arg \
    --trackers $trackers \
    --reid-gap-onnx "$ONNX" --reid-gap-det-thresh 0.4 --reid-gap-tracker-params-json tracker_params.json \
    --fold-assignment-path "$out/fold_assignment.json" \
    --output-dir "$out" --device "$dev" > "$out/kfold.log" 2>&1 \
    || { echo "[$(ts)] ABORT $label"; tail -60 "$out/kfold.log"; exit 1; }
}

run "C4 (bt+bs)"        p4_roi_kfold_baseline                               "bytetrack botsort"  p4     p4_roi_cache_kfold_pool.pt         cpu
run "C4 (oc+doc)"       c4_roi_kfold_baseline_ocsort_deepocsort             "ocsort deepocsort"  p4     p4_roi_cache_kfold_pool.pt         cpu
run "P4-FPN (bt+bs)"    p4fpn_roi_kfold_baseline                            "bytetrack botsort"  p4fpn  p4fpn_roi_cache_kfold_pool.pt      cpu
run "P4-FPN (oc+doc)"   p4fpn_roi_kfold_baseline_ocsort_deepocsort          "ocsort deepocsort"  p4fpn  p4fpn_roi_cache_kfold_pool.pt      cpu
run "OSNet (bt+bs)"     osnet_puro_kfold_baseline                           "bytetrack botsort"  dinov2 osnet_cache_kfold_pool.pt          cpu
run "OSNet (oc+doc)"    osnet_puro_kfold_baseline_ocsort_deepocsort         "ocsort deepocsort"  dinov2 osnet_cache_kfold_pool.pt          cpu
run "CLIP (bt+bs)"      clipvehicleid_puro_kfold_baseline                   "bytetrack botsort"  dinov2 clipvehicleid_cache_kfold_pool.pt  cpu
run "CLIP (oc+doc)"     clipvehicleid_puro_kfold_baseline_ocsort_deepocsort "ocsort deepocsort"  dinov2 clipvehicleid_cache_kfold_pool.pt  cpu
run "DINOv2 (bt+bs)"    kfold_dinov2_raw                                    "bytetrack botsort"  dinov2 -                                  cuda
run "DINOv2 (oc+doc)"   kfold_ocsort_deepocsort_baseline                    "ocsort deepocsort"  dinov2 -                                  cuda
echo "[$(ts)] ==== LISTO k-fold EXCL ===="
echo DONE > "$EXP/kfold_EXCL.done"
