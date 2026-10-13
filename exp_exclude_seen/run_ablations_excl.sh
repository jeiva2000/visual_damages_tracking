#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO -- ablaciones k-fold SIN las 10 secuencias que
# probablemente vio el detector actual (etiqueta EXCL). Complementa
# run_kfold_excl.sh (tab:kfold_identity_acc) y se lanza cuando éste termina.
# -----------------------------------------------------------------------------
# Mismos comandos que las corridas vigentes del paper:
#   tab:gap_length_stratification   run_gaplength_ablation_cpu.sh
#   tab:threshold_sensitivity       run_threshold_c4_p4fpn_cpu.sh / run_threshold_sensitivity_ablation.sh
#   tab:wide_scale_sensitivity      run_m10_scale_fusion_postfix.sh  (6.0 = C4 de run_kfold_excl.sh)
#   tab:query_fusion_sensitivity    run_m10_fusion_postfix.sh        (mean = C4 de run_kfold_excl.sh)
#   tab:inverted_calibration_4trackers  analyze_dinov2_nn_baseline_inverted_calib.py
# con el fold_assignment compartido (md5 599ff319) sin esas 10 secuencias y los
# mismos eventos (copiados con -L). Salidas en exp_exclude_seen/abl/<corrida>_EXCL.
# Dos colas en paralelo: CPU (encoders con caché) y GPU (DINOv2, sin caché).
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
APP=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking
EXP="$APP/exp_exclude_seen"
ABL="$EXP/abl"
ONNX="$APP/checkpoints/rtdetrv2_v3_qcs_videos.onnx"
FA_EXCL="$EXP/p4_roi_kfold_baseline_EXCL/fold_assignment.json"   # 64 -> 54, generado por run_kfold_excl.sh
cd "$APP"
mkdir -p "$ABL"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# La cola CPU solo necesita los EXCL de C4 (bt+bs, oc+doc), ya hechos; la cola
# GPU (DINOv2) espera a que run_kfold_excl.sh termine para no competir con él.
for f in "$FA_EXCL" "$EXP/c4_roi_kfold_baseline_ocsort_deepocsort_EXCL/kfold_results.json"; do
  [ -f "$f" ] || { echo "falta $f"; exit 1; }
done

# Eventos: bytetrack/botsort de p4_roi_kfold_baseline, ocsort/deepocsort de
# c4_roi_kfold_baseline_ocsort_deepocsort (ya copiados con -L en *_EXCL).
EV_BS="$EXP/p4_roi_kfold_baseline_EXCL"
EV_OC="$EXP/c4_roi_kfold_baseline_ocsort_deepocsort_EXCL"

run() {  # run <outname> <script> <trackers> <device> <args...>
  local name="$1" script="$2" trackers="$3" dev="$4"; shift 4
  local out="$ABL/${name}_EXCL"
  if [ -f "$out/kfold_results.json" ]; then echo "[$(ts)] $name ya hecho"; return 0; fi
  echo "[$(ts)] ==== $name ===="
  mkdir -p "$out"
  cp "$FA_EXCL" "$out/fold_assignment.json"
  for t in $trackers; do
    case $t in bytetrack|botsort) src="$EV_BS";; *) src="$EV_OC";; esac
    [ -d "$out/reid_gap_${t}_kfold_all" ] || cp -rL "$src/reid_gap_${t}_kfold_all" "$out/"
  done
  python3 "$script" --kfold-cv --kfold-skip-training --trackers $trackers "$@" \
    --reid-gap-onnx "$ONNX" --reid-gap-det-thresh 0.4 --reid-gap-tracker-params-json tracker_params.json \
    --fold-assignment-path "$out/fold_assignment.json" \
    --output-dir "$out" --device "$dev" > "$out/kfold.log" 2>&1 \
    || { echo "[$(ts)] ABORT $name"; tail -40 "$out/kfold.log"; return 1; }
  grep -a -A4 'RESUMEN K-FOLD' "$out/kfold.log" | tail -4
}

C4="--encoder p4 --embedding-cache p4_roi_cache_kfold_pool.pt"
PF="--encoder p4fpn --embedding-cache p4fpn_roi_cache_kfold_pool.pt"
MAIN=analyze_dinov2_nn_baseline.py
INV=analyze_dinov2_nn_baseline_inverted_calib.py

cpu_queue() {
  for bin in lt3 3-8 9-32 gt32; do
    run "gaplen_${bin}_bt_bs"   $MAIN "bytetrack botsort"  cpu $C4 --gap-length-bin "$bin"
    run "gaplen_${bin}_oc_doc"  $MAIN "ocsort deepocsort"  cpu $C4 --gap-length-bin "$bin"
  done
  for s in 1 2 4 8; do
    run "wide${s}" $MAIN bytetrack cpu --encoder p4 --embedding-cache "c4_cache_wide${s}.pt"
  done
  for f in near wide max concat; do
    run "fusion_${f}" $MAIN bytetrack cpu $C4 --query-fusion "$f"
  done
  for t in 0.50 0.65 0.80 0.90 0.96 0.99; do run "thr_c4_t${t}"    $MAIN bytetrack cpu $C4 --match-threshold "$t"; done
  for t in 0.30 0.45 0.60 0.85 0.92 0.97; do run "thr_p4fpn_t${t}" $MAIN bytetrack cpu $PF --match-threshold "$t"; done
  run inverted_c4    $INV "bytetrack botsort ocsort deepocsort" cpu $C4
  run inverted_p4fpn $INV "bytetrack botsort ocsort deepocsort" cpu $PF
  echo "[$(ts)] cola CPU lista"; touch "$ABL/cpu.done"
}

gpu_queue() {
  until [ -f "$EXP/kfold_EXCL.done" ]; do sleep 60; done
  echo "[$(ts)] k-fold EXCL listo, arrancando cola GPU"
  for t in 0.20 0.35 0.50 0.75 0.85 0.95; do run "thr_dinov2_t${t}" $MAIN bytetrack cuda --encoder dinov2 --match-threshold "$t"; done
  run inverted_dinov2 $INV "bytetrack botsort ocsort deepocsort" cuda --encoder dinov2
  echo "[$(ts)] cola GPU lista"; touch "$ABL/gpu.done"
}

cpu_queue > "$ABL/cpu_queue.out" 2>&1 &
gpu_queue > "$ABL/gpu_queue.out" 2>&1 &
wait
echo "[$(ts)] ==== LISTO ablaciones EXCL ===="
echo DONE > "$ABL/ablations_EXCL.done"
