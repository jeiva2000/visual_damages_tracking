#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO -- detector SIN VIDEOS (etiqueta NOVIDEO)
# -----------------------------------------------------------------------------
# Pregunta: ¿las features C4 / P4-FPN de un detector que NUNCA vio frames de
# las secuencias de evaluación re-identifican igual que las del detector
# actual (rtdetrv2_v3_qcs_videos.pth)?
#
# Detector NOVIDEO: rtdetrv2_v3_qcs_cardd_NOVIDEO.pth (md5 db5e8f47...), copia de
#   teide:/home/serperzar/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/output/
#         rtdetrv2_r101vd_6x_coco_v3_qcs_cardd/best.pth
#   entrenado solo con QCS v3 + CarDD (default_train_cocov3_cardd.json,
#   16562 imgs: 2070 it/epoch x batch 8 coincide exactamente).
#
# Mismo protocolo que run_table7_rerun_postfix_fastonly.sh (C4/P4-FPN puro,
# --kfold-skip-training), con los MISMOS eventos y folds: se copian (lectura
# sobre el original) fold_assignment.json y reid_gap_*_kfold_all/ de las
# corridas actuales -- _generate_reid_gap_report es idempotente, así que NO se
# regeneran eventos con ningún detector. Solo cambian las features y los
# umbrales calibrados con ellas.
#
# Nada de lo que escribe este script pisa resultados del detector actual:
# todo va a exp_novideo_detector/ con sufijo NOVIDEO.
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
APP=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking
EXP="$APP/exp_novideo_detector"
CKPT=../damages_tracking/models/rtdetrv2_v3_qcs_cardd_NOVIDEO.pth
ONNX="$APP/checkpoints/rtdetrv2_v3_qcs_videos.onnx"   # requerido por --kfold-cv; NO se usa (eventos ya existen)
cd "$APP"
mkdir -p "$EXP"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ---- 1) caches de features con el detector NOVIDEO --------------------------
for layer in c4 p4fpn; do
  out="$EXP/${layer}_roi_cache_kfold_pool_NOVIDEO.pt"
  if [ -f "$out" ]; then echo "[$(ts)] cache $layer ya existe, se reutiliza: $out"; continue; fi
  echo "[$(ts)] ==== cache $layer (NOVIDEO) ===="
  python3 -u precompute_p4_roi_cache.py --kfold-pool --layer "$layer" \
    --p4-checkpoint "$CKPT" --output-path "$out" > "$EXP/precompute_${layer}_NOVIDEO.log" 2>&1 \
    || { echo "[$(ts)] ABORT cache $layer"; tail -40 "$EXP/precompute_${layer}_NOVIDEO.log"; exit 1; }
done

# ---- 2) k-fold (mismos eventos/folds que las corridas actuales) --------------
run_kfold() {
  local label="$1" src="$2" outdir="$3" trackers="$4" cache="$5" log="$6"
  echo "[$(ts)] ==== k-fold $label ===="
  mkdir -p "$outdir"
  cp -n "$src/fold_assignment.json" "$outdir/"
  for t in $trackers; do
    # en las corridas originales reid_gap_*_kfold_all son ENLACES SIMBOLICOS
    # relativos (../kfold_dinov2_raw/...): se copia el CONTENIDO real (-L) para
    # que el experimento tenga su propia copia; un enlace roto se reemplaza.
    [ -L "$outdir/reid_gap_${t}_kfold_all" ] && [ ! -e "$outdir/reid_gap_${t}_kfold_all" ] && rm "$outdir/reid_gap_${t}_kfold_all"
    [ -d "$outdir/reid_gap_${t}_kfold_all" ] || cp -rL "$src/reid_gap_${t}_kfold_all" "$outdir/"
  done
  python3 analyze_dinov2_nn_baseline.py --kfold-cv --kfold-skip-training --encoder p4 \
    --embedding-cache "$cache" --trackers $trackers \
    --reid-gap-onnx "$ONNX" --reid-gap-det-thresh 0.4 --reid-gap-tracker-params-json tracker_params.json \
    --fold-assignment-path "$outdir/fold_assignment.json" \
    --output-dir "$outdir" --device cpu > "$log" 2>&1 \
    || { echo "[$(ts)] ABORT $label"; tail -60 "$log"; exit 1; }
}
run_kfold "C4 NOVIDEO (bytetrack+botsort)"     p4_roi_kfold_baseline                       "$EXP/c4_kfold_baseline_NOVIDEO"        "bytetrack botsort"   "$EXP/c4_roi_cache_kfold_pool_NOVIDEO.pt"    "$EXP/kfold_c4_NOVIDEO.log"
run_kfold "C4 NOVIDEO (ocsort+deepocsort)"     c4_roi_kfold_baseline_ocsort_deepocsort     "$EXP/c4_kfold_baseline_oc_NOVIDEO"     "ocsort deepocsort"   "$EXP/c4_roi_cache_kfold_pool_NOVIDEO.pt"    "$EXP/kfold_c4_oc_NOVIDEO.log"
run_kfold "P4-FPN NOVIDEO (bytetrack+botsort)" p4fpn_roi_kfold_baseline                    "$EXP/p4fpn_kfold_baseline_NOVIDEO"     "bytetrack botsort"   "$EXP/p4fpn_roi_cache_kfold_pool_NOVIDEO.pt" "$EXP/kfold_p4fpn_NOVIDEO.log"
run_kfold "P4-FPN NOVIDEO (ocsort+deepocsort)" p4fpn_roi_kfold_baseline_ocsort_deepocsort  "$EXP/p4fpn_kfold_baseline_oc_NOVIDEO"  "ocsort deepocsort"   "$EXP/p4fpn_roi_cache_kfold_pool_NOVIDEO.pt" "$EXP/kfold_p4fpn_oc_NOVIDEO.log"

echo "[$(ts)] ==== LISTO: caches + k-fold NOVIDEO ===="
for f in kfold_c4_NOVIDEO.log kfold_c4_oc_NOVIDEO.log kfold_p4fpn_NOVIDEO.log kfold_p4fpn_oc_NOVIDEO.log; do
  echo "--- $f ---"; grep -a -A6 'RESUMEN K-FOLD' "$EXP/$f" | tail -6
done
echo DONE > "$EXP/kfold_NOVIDEO.done"
