#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=reid_track_query_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1             # 1x A40 en gpu04 (libre ahora mismo)
#SBATCH --nodelist=gpu04             # comprueba antes con: scontrol show node gpu04
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 13:00:00

set -euo pipefail

mkdir -p logs

# --- Parámetros de entrenamiento (ajusta a tu caso) ---
# Asume que este script se lanza con `sbatch` desde dentro del repo (mismo
# directorio que train_track_query_reid.py) -- SLURM usa por default el
# directorio desde el que se hizo el sbatch como working directory.
# Dataset de este cluster -- VDT_DATASET_BASE (ver train_track_query_reid.py)
# reemplaza la base hardcodeada del otro servidor SIN perder SPLIT_OVERRIDES
# (--train-data-root/--val-data-root explícitos sí lo perderían).
export VDT_DATASET_BASE=/home/serperzar/datasets/dataset_split_2
DINOV2_MODEL=${DINOV2_MODEL:-facebook/dinov2-small}
OUTPUT_DIR=${OUTPUT_DIR:-reid_track_query_run_${SLURM_JOB_ID:-run}}

# --- Entorno conda ---
# Este script NO compila nada de CUDA (train_track_query_reid.py es PyTorch +
# HuggingFace transformers puro, sin ops/kernels custom como RT-DETR/
# ultralytics) -- alcanza con el entorno ya armado, no hace falta
# module load GCC/CUDA ni nvcc.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr   # TODO: confirmar que este entorno existe en ESTE cluster con
                         # torch/torchvision/transformers/pillow/tqdm instalados

# --- Entrenamiento ---
python3 train_track_query_reid.py train \
    --train-split train --val-split val \
    --dropout 0.3 --weight-decay 1e-3 \
    --dinov2-model-name "$DINOV2_MODEL" \
    --output-dir "$OUTPUT_DIR"
