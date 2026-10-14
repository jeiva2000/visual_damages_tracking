#!/usr/bin/env bash
# =============================================================================
# DIAGNÓSTICO (NOVIDEO): ¿la caída de detección del detector sin videos se debe
# en parte al umbral de confianza --conf 0.4, ajustado para el detector actual?
# Barre --conf con ByteTrack nativo reutilizando det_cache_NOVIDEO (guarda las
# 300 queries por frame con su score, antes del umbral; model_key coincide, no
# se recalcula nada con el detector). CPU, para no competir con la GPU.
# OJO: barrer sobre las 72 secuencias y elegir el mejor es mirar el test -- sirve
# como diagnóstico, no como protocolo; para un protocolo limpio, elegir en val.
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
ET=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking
cd "$ET"
SRC="$ET/exp_novideo_detector"
OUT="$SRC/sweep_conf"
mkdir -p "$OUT"
PTH=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/damages_tracking/models/rtdetrv2_v3_qcs_cardd_NOVIDEO.pth
CFG=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml
ts() { date '+%Y-%m-%d %H:%M:%S'; }
for c in ${CONFS:-0.3 0.5 0.6 0.7 0.8}; do
  tag="native_bytetrack_conf${c}_NOVIDEO"
  [ -f "$OUT/${tag}.ok" ] && { echo "[$(ts)] $tag ya hecho"; continue; }
  echo "[$(ts)] ==== $tag ===="
  python3 run_eval.py --pth-file "$PTH" --config "$CFG" --data-root "$ET/all_dataset_view" --gt-root "$ET/all_dataset_view_gt" \
    --split all --conf "$c" --device cpu --no-convert-gt \
    --det-cache-dir "$SRC/det_cache_NOVIDEO" --reid-cache-dir "$SRC/reid_cache_NOVIDEO" \
    --output-dir "$OUT/$tag" --tracker bytetrack --track-thresh 0.2 --match-thresh 0.9 --max-age 30 \
    > "$OUT/${tag}.log" 2>&1 || { echo "[$(ts)] ABORT $tag"; tail -40 "$OUT/${tag}.log"; exit 1; }
  touch "$OUT/${tag}.ok"
done
echo DONE > "$OUT/sweep.done"
