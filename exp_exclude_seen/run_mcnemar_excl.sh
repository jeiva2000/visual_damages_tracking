#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO EXCL -- McNemar exacto C4 vs {P4-FPN, OSNet, CLIP-VehicleID}
# (y DINOv2 si INCLUDE_DINOV2=1) sobre las corridas k-fold sin las 10 secuencias
# vistas (configs *_excl de compare_kfold_cases.py). CPU (todas con caché).
# Salidas: exp_exclude_seen/mcnemar_excl/mcnemar_excl_<tracker>.{json,log}
# =============================================================================
set -o pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
APP=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking
OUT="$APP/exp_exclude_seen/mcnemar_excl"
cd "$APP"; mkdir -p "$OUT"
DEV=${DEV:-cpu}
for tr in bytetrack botsort ocsort deepocsort; do
  s=""; case $tr in ocsort|deepocsort) s="_oc";; esac
  cfgs="c4_excl$s p4fpn_excl$s osnet_excl$s clip_excl$s"
  tag=""; [ "${INCLUDE_DINOV2:-0}" = 1 ] && { cfgs="c4_excl$s dinov2_excl$s"; tag="_dinov2"; }
  f="$OUT/mcnemar_excl${tag}_${tr}"
  [ -f "$f.json" ] && { echo "$tr ya hecho"; continue; }
  echo "[$(date '+%F %T')] ==== $tr: $cfgs ===="
  nice python3 compare_kfold_cases.py --tracker "$tr" --configs $cfgs --device "$DEV" --output-json "$f.json" > "$f.log" 2>&1 \
    || { echo "ABORT $tr"; tail -30 "$f.log"; exit 1; }
  grep "RESUMEN" "$f.log"
done
echo DONE > "$OUT/mcnemar_excl${tag}.done"
