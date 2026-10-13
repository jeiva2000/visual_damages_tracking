#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO (exp_testmodels) -- HOTA con ByteTrack NATIVO (sin reid
# externo) sobre las 72 secuencias de dataset_split_2 para cada detector ONNX
# de /mnt/Disk2/test_models/checkpoints. Mismos hiperparámetros que la corrida
# vigente native_reid_bytetrack_alldata (conf 0.4, track 0.2, match 0.9, max-age 30).
# Usa las COPIAS run_eval_tm.py / main_track_tm.py (ver su cabecera):
#   rtv2  : modelos de 3 salidas, preprocesado del pipeline (igual que el paper)
#   q     : modelos quality (4 salidas): score = 4a salida (infer_onnx_q.py),
#           y de paso guarda el det-cache con score = 'scores' (infer_onnx_rtv2.py)
#           para la variante q_s2, que se evalúa sin repetir inferencia
#   rtv4  : RT-DETRv4/D-FINE, preprocesado letterbox de infer_onnx_rtv4.py
# Ya evaluados en otras carpetas (no se repiten): rtdetrv2_v3_qcs_videos (actual),
# rtdetrv2_v2_qcs (exp_novideo2_detector), rtdetrv2_qcs_cardd_prueba = v3_qcs_cardd
# (exp_novideo_detector), r101_1cat_noglass_nvt_oldcdd_dmg_001 (exp_noglass_detector).
# Duplicados por md5 omitidos: rtdetr_dataset_v4_3_best (= v4_3_19).
# Paralelismo: JOBS (default 2) modelos a la vez. Reanudable (<tag>.ok).
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
ET=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking
OUT="$ET/exp_testmodels"
CK=/mnt/Disk2/test_models/checkpoints
cd "$ET"
mkdir -p "$OUT/logs"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_one() {  # run_one <tag> <onnx> <env...>
  local tag="$1" onnx="$2"; shift 2
  if [ -f "$OUT/logs/${tag}.ok" ]; then echo "[$(ts)] $tag ya hecho"; return 0; fi
  echo "[$(ts)] ==== $tag ===="
  env "$@" python3 run_eval_tm.py --onnx-file "$onnx" \
    --data-root "$ET/all_dataset_view" --gt-root "$ET/all_dataset_view_gt" \
    --split all --conf 0.4 --device cuda --no-convert-gt \
    --det-cache-dir "$OUT/det_cache_${tag}" --reid-cache-dir "$OUT/reid_cache_${tag}" \
    --output-dir "$OUT/$tag" --tracker bytetrack --track-thresh 0.2 --match-thresh 0.9 --max-age 30 \
    > "$OUT/logs/${tag}.log" 2>&1 \
    || { echo "[$(ts)] ABORT $tag"; tail -30 "$OUT/logs/${tag}.log"; return 1; }
  touch "$OUT/logs/${tag}.ok"
  echo "[$(ts)] $tag OK: $(sed -n 2p "$OUT/$tag/txt/pedestrian_summary.txt" | awk '{print "HOTA="$1" DetA="$2" AssA="$3}')"
}

job() {  # job <kind> <model>
  local kind="$1" m="$2" name="${2%.onnx}"
  case $kind in
    rtv2) run_one "${name}" "$CK/$m" ;;
    rtv4) run_one "${name}" "$CK/$m" TM_LETTERBOX=1 ;;
    q)    run_one "${name}__q" "$CK/$m" TM_SCORE_IDX=3 TM_ALT_CACHE=1 || return 1
          # variante score='scores' (infer_onnx_rtv2.py): reusa el det-cache ALT, sin inferencia
          [ -d "$OUT/det_cache_${name}__q_s2" ] || cp -r "$OUT/det_cache_${name}__q_ALT" "$OUT/det_cache_${name}__q_s2"
          run_one "${name}__q_s2" "$CK/$m" TM_SCORE_IDX=2 ;;
  esac
}
export -f run_one job ts
export ET OUT CK

cat > "$OUT/jobs.txt" <<'EOF'
rtv4 rtdetrv4.onnx
rtv4 rtdetrv4_epoch_11.onnx
rtv4 rtdetrv4_epoch_19.onnx
q rtdetr_quality_datasetv4_best.onnx
q rtdetr_quality_datasetv4_last.onnx
q rtdetr_quality_datasetv4_19.onnx
q rtdetr_quality_datasetv4_29.onnx
q rtdetr_quality_datasetv4_39.onnx
q rtdetrv2_qcs_quality_19.onnx
q prueba_quality_19.onnx
q prueba_quality_29.onnx
q rtdetr_quality_19_teide.onnx
rtv2 rtdetr_dataset_v4_best.onnx
rtv2 rtdetr_dataset_v4_4_best.onnx
rtv2 rtdetr_dataset_v4_19.onnx
rtv2 rtdetr_dataset_v4_29.onnx
rtv2 rtdetr_dataset_v4_1_19.onnx
rtv2 rtdetr_dataset_v4_1_29.onnx
rtv2 rtdetr_dataset_v4_2_19.onnx
rtv2 rtdetr_dataset_v4_2_29.onnx
rtv2 rtdetr_dataset_v4_3_19.onnx
rtv2 rtdetr_dataset_v4_3_29.onnx
rtv2 rtdetr_dataset_v4_4_9.onnx
rtv2 rtdetr_dataset_v4_4_19.onnx
rtv2 rtdetr_dataset_v4_4_29.onnx
rtv2 rtdetrv2_qcs_cardd_default.onnx
rtv2 rtdetrv2_qcs_cardd_default_19.onnx
rtv2 rtdetrv2_qcs_cardd_default_29.onnx
rtv2 rtdetrv2_qcs_cardd_filter.onnx
rtv2 rtdetrv2_qcs_cardd_filter_19.onnx
rtv2 rtdetrv2_qcs_cardd_filter_29.onnx
EOF
# rtdetr_quality_19_teide.onnx usa pesos externos: checkpoints/model.onnx.data

echo "[$(ts)] $(wc -l < "$OUT/jobs.txt") modelos, JOBS=${JOBS:-2} en paralelo"
xargs -P "${JOBS:-2}" -L 1 bash -c 'job "$0" "$1"' < "$OUT/jobs.txt"
echo "[$(ts)] ==== LISTO barrido test_models ===="
echo DONE > "$OUT/testmodels.done"
