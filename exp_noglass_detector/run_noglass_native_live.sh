#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO (etiqueta NOGLASS) -- validación en vivo con el método
# NATIVO (sin reid externo) usando el detector r101_1cat_noglass_nvt_oldcdd_dmg_001.
# Solo existe en .onnx (md5 017294f5, 4 copias idénticas en 172 y teide; el .pth
# no se encontró en ningún servidor). Mismos hiperparámetros que la corrida
# vigente native_reid_bytetrack_alldata; cachés y salidas propios en
# eval_tracking/exp_noglass_detector/, nada pisa lo existente.
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
ET=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking
cd "$ET"
OUT="$ET/exp_noglass_detector"
mkdir -p "$OUT"
ONNX=/mnt/Disk2/test_models/checkpoints/r101_1cat_noglass_nvt_oldcdd_dmg_001.onnx
CACHES="--det-cache-dir $OUT/det_cache_NOGLASS --reid-cache-dir $OUT/reid_cache_NOGLASS"
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
run native_reid_bytetrack_alldata_NOGLASS --tracker bytetrack --track-thresh 0.2 --match-thresh 0.9 --max-age 30
echo DONE > "$OUT/native_live_NOGLASS.done"
