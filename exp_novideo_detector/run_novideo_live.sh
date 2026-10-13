#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO -- validación EN VIVO con el detector SIN VIDEOS (NOVIDEO)
# -----------------------------------------------------------------------------
# Repite la validación en vivo (72 secuencias, 4 trackers) de
# run_native_reid_alldata.sh + run_puro_live_alldata.sh con el detector
# rtdetrv2_v3_qcs_cardd_NOVIDEO.pth (solo QCS v3 + CarDD, nunca vio frames de
# video). Ver exp_novideo_detector/run_novideo_kfold.sh para la k-fold.
#
# Diferencias deliberadas con las corridas del detector actual:
#   - el nativo también usa --pth-file (el actual usaba el ONNX del detector
#     actual): nativo y C4 NOVIDEO quedan con el mismo backend, comparables.
#   - umbrales calibrados con el propio detector NOVIDEO en la k-fold
#     (C4 0.9294, P4-FPN 0.7840), mismo criterio que 0.9352/0.7988 del actual.
#   - CACHES PROPIOS: run_eval.py cachea detecciones y embeddings ReID con
#     clave = secuencia + carpeta de frames, SIN el modelo. Con los caches por
#     defecto (eval_tracking/det_cache, reid_cache) esta corrida reusaría en
#     silencio las detecciones del detector actual. Por eso --det-cache-dir y
#     --reid-cache-dir apuntan a carpetas NOVIDEO.
#   - run_eval.py hace rmtree de --output-dir: todas las salidas llevan sufijo
#     _NOVIDEO dentro de exp_novideo_detector/, nunca colisionan.
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
ET=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking
cd "$ET"
OUT="$ET/exp_novideo_detector"
mkdir -p "$OUT"
PTH=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/damages_tracking/models/rtdetrv2_v3_qcs_cardd_NOVIDEO.pth
CFG=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml
DATA_ROOT="$ET/all_dataset_view"
GT_ROOT="$ET/all_dataset_view_gt"
SPLIT=all
CACHES="--det-cache-dir $OUT/det_cache_NOVIDEO --reid-cache-dir $OUT/reid_cache_NOVIDEO"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

run() {  # run <tag> <args...>
  local tag="$1"; shift
  if [ -f "$OUT/${tag}.ok" ]; then echo "[$(ts)] $tag ya hecho, se salta"; return; fi
  echo "[$(ts)] ==================== $tag ===================="
  python3 run_eval.py --pth-file "$PTH" --config "$CFG" --data-root "$DATA_ROOT" --gt-root "$GT_ROOT" \
    --split $SPLIT --conf 0.4 --device cuda --reid-device cuda --no-convert-gt $CACHES \
    --output-dir "$OUT/$tag" "$@" > "$OUT/${tag}.log" 2>&1
  local status=$?
  echo "[$(ts)] $tag done, exit=$status"
  if [ "$status" -ne 0 ]; then echo "[$(ts)] ABORT: revisa $OUT/${tag}.log"; tail -60 "$OUT/${tag}.log"; exit 1; fi
  touch "$OUT/${tag}.ok"
}

BT="--tracker bytetrack --track-thresh 0.2 --match-thresh 0.9 --max-age 30"
BS="--tracker botsort --track-thresh 0.2 --new-track-thresh 0.2 --match-thresh 0.8 --max-age 30 --proximity-thresh 0.5 --appearance-thresh 0.4 --cmc-method ecc"
OC="--tracker ocsort --max-age 60 --min-hits 5 --iou-threshold 0.3 --delta-t 5 --inertia 0.4 --asso-func iou"
DO="--tracker deepocsort --max-age 30 --min-hits 5 --iou-threshold 0.4 --w-association-emb 0.7 --alpha-fixed-emb 0.99 --aw-param 0.5 --delta-t 2 --inertia 0.4 --asso-func iou"

# ---- 1) nativo (sin módulo) -- mismos flags que run_native_reid_alldata.sh ----
run native_reid_bytetrack_alldata_NOVIDEO  $BT
run native_reid_botsort_alldata_NOVIDEO    $BS
run native_reid_ocsort_alldata_NOVIDEO     $OC
run native_reid_deepocsort_alldata_NOVIDEO $DO

# ---- 2) C4 puro (umbral calibrado NOVIDEO) -- mismos flags que run_puro_live_alldata.sh ----
C4="--reid-stitch --reid-source c4_puro --reid-stitch-cosine-thresh 0.9294"
run c4_puro_live_bytetrack_alldata_NOVIDEO  $BT $C4
run c4_puro_live_botsort_alldata_NOVIDEO    $BS --no-native-reid $C4
run c4_puro_live_ocsort_alldata_NOVIDEO     $OC $C4
run c4_puro_live_deepocsort_alldata_NOVIDEO $DO $C4

# ---- 3) P4-FPN puro (umbral calibrado NOVIDEO) ----
P4="--reid-stitch --reid-source p4fpn_puro --reid-stitch-cosine-thresh 0.7840"
run p4fpn_puro_live_bytetrack_alldata_NOVIDEO  $BT $P4
run p4fpn_puro_live_botsort_alldata_NOVIDEO    $BS --no-native-reid $P4
run p4fpn_puro_live_ocsort_alldata_NOVIDEO     $OC $P4
run p4fpn_puro_live_deepocsort_alldata_NOVIDEO $DO $P4

echo "[$(ts)] ==================== TODO LISTO (validación en vivo NOVIDEO) ===================="
echo DONE > "$OUT/live_NOVIDEO.done"
