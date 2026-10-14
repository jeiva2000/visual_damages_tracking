#!/usr/bin/env bash
# =============================================================================
# EXPERIMENTO AISLADO EXCL -- McNemar exacto C4 vs C4+H1..H4, con las cabezas
# RE-ENTRENADAS sin las 10 secuencias vistas (run_heads_excl.sh; configs
# c4_h*_excl* de compare_kfold_cases.py). CPU (todas con caché).
# Salidas: exp_exclude_seen/mcnemar_excl/mcnemar_excl_heads_<tracker>.{json,log}
# =============================================================================
set -o pipefail
export PYTHONHASHSEED=0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
APP=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking
OUT="$APP/exp_exclude_seen/mcnemar_excl"
cd "$APP"; mkdir -p "$OUT"
DEV=${DEV:-cpu}
for tr in bytetrack botsort ocsort deepocsort; do
  s=""; case $tr in ocsort|deepocsort) s="_oc";; esac
  cfgs="c4_excl$s c4_h1_excl$s c4_h2_excl$s c4_h3_excl$s c4_h4_excl$s"
  f="$OUT/mcnemar_excl_heads_${tr}"
  [ -f "$f.json" ] && { echo "$tr ya hecho"; continue; }
  echo "[$(date '+%F %T')] ==== $tr: $cfgs ===="
  nice python3 compare_kfold_cases.py --tracker "$tr" --configs $cfgs --device "$DEV" --output-json "$f.json" > "$f.log" 2>&1 \
    || { echo "ABORT $tr"; tail -30 "$f.log"; exit 1; }
  grep "RESUMEN" "$f.log"
done
echo DONE > "$OUT/mcnemar_excl_heads.done"
