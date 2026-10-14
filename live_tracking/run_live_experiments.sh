#!/usr/bin/env bash
# Live evaluation of the paper: native tracker, or tracker + external re-identification module.
# Flags copied from the scripts used for the reported runs (62 evaluation sequences are selected
# afterwards when scoring; see exps/exp_exclude_seen/). One process, sequential.
#
# Usage:  DETECTOR_PTH=... DETECTOR_CFG=... DATA_ROOT=... GT_ROOT=... \
#         ./run_live_experiments.sh <method> <tracker> [output_dir]
#   method:  native | c4 | p4fpn | dinov2 | osnet | clip
#   tracker: bytetrack | botsort | ocsort | deepocsort
set -eo pipefail
: "${DETECTOR_PTH:?}" "${DETECTOR_CFG:?}" "${DATA_ROOT:?}" "${GT_ROOT:?}"
METHOD="$1"; TRACKER="$2"; OUT="${3:-out/${METHOD}_${TRACKER}}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# similarity threshold tau per signal (calibrated on the k-fold, excluding the 10 detector-seen sequences)
case "$METHOD" in
  native) SRC="" ;;
  c4)     SRC="--reid-stitch --reid-source c4_puro --reid-stitch-cosine-thresh 0.9405" ;;
  p4fpn)  SRC="--reid-stitch --reid-source p4fpn_puro --reid-stitch-cosine-thresh 0.7981" ;;
  dinov2) SRC="--reid-stitch --reid-source dinov2_puro --reid-stitch-cosine-thresh 0.6919" ;;
  osnet)  SRC="--reid-stitch --reid-source native --reid-stitch-cosine-thresh 0.8000" ;;
  clip)   SRC="--reid-stitch --reid-source native_puro --native-reid-weights clip_vehicleid.pt --reid-stitch-cosine-thresh 0.7698" ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

# tracker hyperparameters (BoT-SORT: its internal appearance model is switched off when the module is attached)
case "$TRACKER" in
  bytetrack)  TRK="--track-thresh 0.2 --match-thresh 0.9 --max-age 30" ;;
  botsort)    TRK="--track-thresh 0.2 --new-track-thresh 0.2 --match-thresh 0.8 --max-age 30 --proximity-thresh 0.5 --appearance-thresh 0.4 --cmc-method ecc"
              [ "$METHOD" != native ] && TRK="$TRK --no-native-reid" ;;
  ocsort)     TRK="--max-age 60 --min-hits 5 --iou-threshold 0.3 --delta-t 5 --inertia 0.4 --asso-func iou" ;;
  deepocsort) TRK="--max-age 30 --min-hits 5 --iou-threshold 0.4 --w-association-emb 0.7 --alpha-fixed-emb 0.99 --aw-param 0.5 --delta-t 2 --inertia 0.4 --asso-func iou" ;;
  *) echo "unknown tracker $TRACKER"; exit 1 ;;
esac

python3 "$HERE/run_eval_nodedup.py" --pth-file "$DETECTOR_PTH" --config "$DETECTOR_CFG" \
  --data-root "$DATA_ROOT" --gt-root "$GT_ROOT" --split all \
  --tracker "$TRACKER" --conf 0.4 --device cuda --reid-device cuda --no-convert-gt \
  $SRC $TRK --output-dir "$OUT"
