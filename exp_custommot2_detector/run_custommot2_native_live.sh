#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO (etiqueta CUSTOMMOT2) -- validación en vivo con el método
# NATIVO (sin reid externo) usando el detector rtdetr_damages.onnx (2025-09-19)
# copiado de teide:/data/backup/serperzar/custom_mot_2/rtdetr_damages.onnx (md5 564a0547)
# como model/rtdetr_damages_CUSTOMMOT2.onnx. Mismos hiperparámetros que la corrida
# vigente native_reid_bytetrack_alldata; cachés y salidas propios en
# eval_tracking/exp_custommot2_detector/, nada pisa lo existente.
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
ET=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking
cd "$ET"
OUT="$ET/exp_custommot2_detector"
mkdir -p "$OUT"
ONNX=$OUT/model/rtdetr_damages_CUSTOMMOT2.onnx
CACHES="--det-cache-dir $OUT/det_cache_CUSTOMMOT2 --reid-cache-dir $OUT/reid_cache_CUSTOMMOT2"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
run() {  # run <tag> <args...>
  local tag="$1"; shift
  if [ -f "$OUT/${tag}.ok" ]; then echo "[$(ts)] $tag ya hecho, se salta"; return; fi
  echo "[$(ts)] ==================== $tag ===================="
  python3 run_eval.py --onnx-file "$ONNX" --data-root "$ET/all_dataset_view" --gt-root "$ET/all_dataset_view_gt" \
    --split all --conf 0.4 --device cuda --no-convert-gt $CACHES \
    --output-dir "$OUT/$tag" "$@" > "$OUT/${tag}.log" 2>&1
  local status=$?
  echo "[$(ts)] $tag done, exit=$status"
  if [ "$status" -ne 0 ]; then echo "[$(ts)] ABORT: revisa $OUT/${tag}.log"; tail -60 "$OUT/${tag}.log"; exit 1; fi
  touch "$OUT/${tag}.ok"
}
run native_reid_bytetrack_alldata_CUSTOMMOT2 --tracker bytetrack --track-thresh 0.2 --match-thresh 0.9 --max-age 30
echo DONE > "$OUT/native_live_CUSTOMMOT2.done"
