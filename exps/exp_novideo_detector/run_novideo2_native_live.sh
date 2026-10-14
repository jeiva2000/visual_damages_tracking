#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO -- validación EN VIVO, método NATIVO, con el segundo
# detector sin videos (etiqueta NOVIDEO2)
# -----------------------------------------------------------------------------
# Detector NOVIDEO2: rtdetrv2_v2_qcs_NOVIDEO2.pth (md5 5a28e31d...), copia de
#   teide:/home/serperzar/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/output/
#         rtdetrv2_r101vd_6x_coco_v2_qcs/best.pth   (2026-04-27, época 19)
#   entrenado SOLO con fotos de QCS: 1570 it/epoch x batch 8 =
#   coco_v2_qcs/instances_train2017.json (12562 imgs, sin CarDD, sin frames de
#   video). Su ONNX es test_models/checkpoints/rtdetrv2_v2_qcs.onnx (mismo md5).
#
# Mismo protocolo que run_novideo_live.sh (bloque nativo): 72 secuencias,
# 4 trackers, mismos hiperparámetros que run_native_reid_alldata.sh, backend
# --pth-file (igual que el nativo NOVIDEO), conf 0.4.
# CACHES PROPIOS (det_cache/reid_cache de run_eval.py no incluyen el modelo en
# la clave) y salidas *_NOVIDEO2 en exp_novideo2_detector/ (run_eval.py hace
# rmtree de --output-dir).
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
ET=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking
cd "$ET"
OUT="$ET/exp_novideo2_detector"
mkdir -p "$OUT"
PTH=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/damages_tracking/models/rtdetrv2_v2_qcs_NOVIDEO2.pth
CFG=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml
DATA_ROOT="$ET/all_dataset_view"
GT_ROOT="$ET/all_dataset_view_gt"
CACHES="--det-cache-dir $OUT/det_cache_NOVIDEO2 --reid-cache-dir $OUT/reid_cache_NOVIDEO2"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

run() {  # run <tag> <args...>
  local tag="$1"; shift
  if [ -f "$OUT/${tag}.ok" ]; then echo "[$(ts)] $tag ya hecho, se salta"; return; fi
  echo "[$(ts)] ==================== $tag ===================="
  python3 run_eval.py --pth-file "$PTH" --config "$CFG" --data-root "$DATA_ROOT" --gt-root "$GT_ROOT" \
    --split all --conf 0.4 --device cuda --reid-device cuda --no-convert-gt $CACHES \
    --output-dir "$OUT/$tag" "$@" > "$OUT/${tag}.log" 2>&1
  local status=$?
  echo "[$(ts)] $tag done, exit=$status"
  if [ "$status" -ne 0 ]; then echo "[$(ts)] ABORT: revisa $OUT/${tag}.log"; tail -60 "$OUT/${tag}.log"; exit 1; fi
  touch "$OUT/${tag}.ok"
}

run native_reid_bytetrack_alldata_NOVIDEO2  --tracker bytetrack --track-thresh 0.2 --match-thresh 0.9 --max-age 30
run native_reid_botsort_alldata_NOVIDEO2    --tracker botsort --track-thresh 0.2 --new-track-thresh 0.2 --match-thresh 0.8 --max-age 30 --proximity-thresh 0.5 --appearance-thresh 0.4 --cmc-method ecc
run native_reid_ocsort_alldata_NOVIDEO2     --tracker ocsort --max-age 60 --min-hits 5 --iou-threshold 0.3 --delta-t 5 --inertia 0.4 --asso-func iou
run native_reid_deepocsort_alldata_NOVIDEO2 --tracker deepocsort --max-age 30 --min-hits 5 --iou-threshold 0.4 --w-association-emb 0.7 --alpha-fixed-emb 0.99 --aw-param 0.5 --delta-t 2 --inertia 0.4 --asso-func iou

echo "[$(ts)] ==================== TODO LISTO (nativo en vivo NOVIDEO2) ===================="
echo DONE > "$OUT/native_live_NOVIDEO2.done"
