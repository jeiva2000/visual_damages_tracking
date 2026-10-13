#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate rtdetr
ET=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking; cd $ET
X=$ET/exp_exclude_seen/hard_cases/sims; mkdir -p $X
PTH=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/damages_tracking/models/rtdetrv2_v3_qcs_videos.pth
CFG=/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml
SEQS="video_75wAEktItLIYWrWOW7E_11 video_0vbcxRjnfa70hc8AVvt_11 video_0XWkyECiQbjitGSpVQK_11 video_Ve7DX8i8A6RX5h8OyFt_11 video_0jGlrs0wJs3utJftEoD_11 video_1Vl7JUY8fBQhAph15I7_11 video_rICjFXthyXrdrgr58oE_11 video_kYUj2rFivaSQMwnjjGp_11 video_Mpn2iv5EwC8BQM4Guf2_11"
for t in botsort bytetrack; do
 case $t in
  botsort) TA="--track-thresh 0.2 --new-track-thresh 0.2 --match-thresh 0.8 --max-age 30 --proximity-thresh 0.5 --appearance-thresh 0.4 --cmc-method ecc --no-native-reid";;
  bytetrack) TA="--track-thresh 0.2 --match-thresh 0.9 --max-age 30";;
 esac
 for s in $SEQS; do
  SIMLOG=$X/${t}_$s.jsonl python3 run_eval_viz.py --pth-file $PTH --config $CFG --data-root all_dataset_view --gt-root all_dataset_view_gt --split all \
   --sequences $s --tracker $t --conf 0.4 --device cuda --reid-device cuda --no-convert-gt \
   --reid-stitch --reid-source c4_puro --reid-stitch-cosine-thresh 0.9405 $TA \
    \
   --det-cache-dir exp_live_excl/det_cache_w1 --reid-cache-dir exp_live_excl/reid_cache_w1 \
   --output-dir $X/out_${t}_$s > $X/${t}_$s.log 2>&1; echo "$t $s rc=$?" >> $X/progress.txt
 done
done
touch $X/DONE
