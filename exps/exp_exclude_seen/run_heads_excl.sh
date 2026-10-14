#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO -- cabezas entrenadas H1-H4 (tab:kfold_identity_acc) SIN
# las 10 secuencias que probablemente vio el detector actual (etiqueta EXCL).
# -----------------------------------------------------------------------------
# Mismos comandos que run_head_variants_postfix.sh (2026-09-04, las corridas
# que reproducen la tabla), con:
#   - fold_assignment y eventos de <base>_EXCL (64 -> 54 secuencias): cada
#     fold RE-ENTRENA la cabeza y recalibra el umbral sin esas 10 secuencias;
#   - salidas en exp_exclude_seen/<corrida>_EXCL, nada pisa lo existente.
#
# MODE=repro: misma corrida sobre el pool original de 64 (fold_assignment y
# eventos originales) en exp_exclude_seen/repro_heads/<corrida>, para
# comprobar que el código actual reproduce los valores publicados.
# VARIANTS / HGROUPS permiten limitar (p.ej. VARIANTS=v7 HGROUPS="c4_btbs").
# =============================================================================
set -o pipefail
export PYTHONHASHSEED=0   # el entrenamiento de cabezas no es determinista entre procesos sin esto (repro 2026-10-02: H3 C4 bt 0.9070 vs 0.8777 publicado)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
APP=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking
EXP="$APP/exp_exclude_seen"
ONNX="$APP/checkpoints/rtdetrv2_v3_qcs_videos.onnx"
MODE="${MODE:-excl}"
VARIANTS="${VARIANTS:-v6style intermedio v7 pure_triplet}"
HGROUPS="${HGROUPS:-c4_btbs c4_ocdoc p4fpn_btbs p4fpn_ocdoc}"
DEV="${DEV:-cuda}"
# HEAD_TORCH_SEED=N: fija la inicialización de la cabeza vía seeded_analyze.py
# (sin él no hay semilla de torch en el camino k-fold) y añade _s<N> a la salida.
SFX=""; PY=(python3 analyze_dinov2_nn_baseline.py)
if [ -n "${HEAD_TORCH_SEED:-}" ]; then
  export HEAD_TORCH_SEED; SFX="_s${HEAD_TORCH_SEED}"; PY=(python3 "$EXP/seeded_analyze.py")
fi
cd "$APP"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

variant_args() {
  case "$1" in
    v6style)      echo "--new-track-prob 0" ;;
    intermedio)   echo "--new-track-prob 0.25 --infonce-weight 0" ;;
    v7)           echo "" ;;
    pure_triplet) echo "--new-track-prob 0 --infonce-weight 0" ;;
  esac
}

run() {  # run <encoder> <cache> <trackers> <base> <outname> <variant>
  local enc="$1" cache="$2" trackers="$3" base="$4" outname="$5" v="$6"
  local src out
  if [ "$MODE" = repro ]; then
    src="$APP/$base"; out="$EXP/repro_heads/${outname}${SFX}"
  else
    src="$EXP/${base}_EXCL"; out="$EXP/${outname}_EXCL${SFX}"
  fi
  if [ -f "$out/kfold_results.json" ]; then echo "[$(ts)] $outname ya hecho"; return; fi
  [ -f "$src/fold_assignment.json" ] || { echo "[$(ts)] ABORT: falta $src/fold_assignment.json"; exit 1; }
  echo "[$(ts)] ==== $MODE $outname${SFX} ($trackers) ===="
  mkdir -p "$out"
  cp "$src/fold_assignment.json" "$out/fold_assignment.json"
  for t in $trackers; do
    [ -d "$out/reid_gap_${t}_kfold_all" ] || cp -rL "$src/reid_gap_${t}_kfold_all" "$out/"
  done
  "${PY[@]}" --kfold-cv --encoder "$enc" --embedding-cache "$cache" \
    --trackers $trackers $(variant_args "$v") \
    --reid-gap-onnx "$ONNX" --reid-gap-det-thresh 0.4 --reid-gap-tracker-params-json tracker_params.json \
    --fold-assignment-path "$out/fold_assignment.json" \
    --output-dir "$out" --device "$DEV" > "$out/kfold.log" 2>&1 \
    || { echo "[$(ts)] ABORT $outname"; tail -60 "$out/kfold.log"; exit 1; }
  echo "[$(ts)] $outname done"
}

for v in $VARIANTS; do
  for g in $HGROUPS; do
    case "$g" in
      c4_btbs)     run p4    p4_roi_cache_kfold_pool.pt    "bytetrack botsort" p4_roi_kfold_baseline                      "p4_roi_kfold_head_${v}"                      "$v" ;;
      c4_ocdoc)    run p4    p4_roi_cache_kfold_pool.pt    "ocsort deepocsort" c4_roi_kfold_baseline_ocsort_deepocsort    "c4_roi_kfold_head_${v}_ocsort_deepocsort"    "$v" ;;
      p4fpn_btbs)  run p4fpn p4fpn_roi_cache_kfold_pool.pt "bytetrack botsort" p4fpn_roi_kfold_baseline                   "p4fpn_roi_kfold_head_${v}"                   "$v" ;;
      p4fpn_ocdoc) run p4fpn p4fpn_roi_cache_kfold_pool.pt "ocsort deepocsort" p4fpn_roi_kfold_baseline_ocsort_deepocsort "p4fpn_roi_kfold_head_${v}_ocsort_deepocsort" "$v" ;;
    esac
  done
done
echo "[$(ts)] ==== LISTO heads $MODE ===="
[ "$MODE" = excl ] && echo DONE > "$EXP/heads_EXCL${SFX}.done"
