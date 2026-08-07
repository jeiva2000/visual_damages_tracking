#!/usr/bin/env bash
# Lanza dos sweeps secuenciales de train_track_query_reid.py sobre la misma
# GPU (uno a la vez):
#   1. Encoders (MODELS) -- compara backbones, con --query-wide-factor en su
#      default.
#   2. Factor de escala de la vista amplia de la query (QUERY_WIDE_FACTORS)
#      -- modelo FIJO (QUERY_WIDE_FACTOR_MODEL, el mas rapido) para no
#      multiplicar el tiempo total; una vez elegido el mejor factor, se
#      puede correr aparte contra los demas encoders si hace falta.
# Ambos sweeps comparten COMMON_ARGS para que la comparacion sea justa
# dentro de cada uno. Si una corrida falla (ej. DINOv3 sin HF_TOKEN
# exportado), se reporta al final y se sigue con la siguiente en vez de
# abortar todo.
#
# Uso:
#   export HF_TOKEN=hf_xxxx   # necesario para los modelos DINOv3 (gated)
#   ./run_encoder_sweep.sh
#
# Para que sobreviva un corte de SSH durante la noche, lanzalo con nohup o
# dentro de tmux/screen, ej.:
#   nohup ./run_encoder_sweep.sh > sweep.log 2>&1 &

set -u
set -o pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd /mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[WARN] HF_TOKEN no esta seteado -- los modelos DINOv3 (gated) van a fallar al cargar. DINOv2 no lo necesita."
fi

# Args comunes a TODAS las corridas (los dos sweeps) -- agregar/quitar aca.
COMMON_ARGS=(
  --train-split train
  --val-split val
  --dropout 0.3
  --weight-decay 1e-3
  # --synthetic-gap-max 80   # descomentar para activar el gap sintetico mas realista (ver conversacion)
)

FAILED=()

run_training() {
  # $1 = nombre corto para output-dir/log, resto = args extra de esta corrida
  local NAME="$1"; shift
  local OUTDIR="reid_track_query_run_${NAME}"
  local LOG="log_${NAME}.txt"

  echo "=============================================="
  echo "[RUN] ${NAME} -> output-dir=${OUTDIR}  (log: ${LOG})"
  echo "=============================================="

  python3 train_track_query_reid.py train \
    "${COMMON_ARGS[@]}" "$@" \
    --output-dir "${OUTDIR}" \
    2>&1 | tee "${LOG}"

  local status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ]; then
    echo "[FAIL] ${NAME} termino con codigo ${status} -- sigo con el siguiente"
    FAILED+=("${NAME}")
  else
    echo "[OK] ${NAME} termino bien"
  fi
}

# --- Sweep 1: encoders -- un modelo por linea, agregar/quitar los que quieras probar. ---
MODELS=(
  "facebook/dinov3-vits16-pretrain-lvd1689m"
  "facebook/dinov3-vitb16-pretrain-lvd1689m"
  "facebook/dinov2-small"
  "facebook/dinov2-base"
)

for MODEL in "${MODELS[@]}"; do
  run_training "${MODEL//\//_}" --dinov2-model-name "${MODEL}"
done

# --- Sweep 2: factor de escala de la vista AMPLIA de la query (contexto de
# pieza, ver _load_query_views en reid_dataloader.py) -- modelo fijo, mas
# rapido, para no multiplicar el tiempo total del sweep de encoders. Default
# actual del flag es 2.5 -- se incluye igual para tener el punto de
# comparacion dentro de esta misma tabla. ---
QUERY_WIDE_FACTOR_MODEL="facebook/dinov2-small"
QUERY_WIDE_FACTORS=(1.5 2.5 3.5 5.0)

for FACTOR in "${QUERY_WIDE_FACTORS[@]}"; do
  run_training "querywide_${FACTOR}" \
    --dinov2-model-name "${QUERY_WIDE_FACTOR_MODEL}" --query-wide-factor "${FACTOR}"
done

echo
echo "=============================================="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "[OK] Todas las corridas terminaron sin errores."
else
  echo "[WARN] Fallaron ${#FAILED[@]} corrida(s):"
  printf '  - %s\n' "${FAILED[@]}"
fi
