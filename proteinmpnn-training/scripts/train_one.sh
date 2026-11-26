#!/usr/bin/env bash
# Portable single-model launcher; run inside an allocated compute job.
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the prepared structure dataset}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the source ProteinMPNN project}"
: "${RUN_ROOT:?Set RUN_ROOT to a new run directory}"
PYTHON="${PYTHON:-python}"
MODEL_NAME="${1:-exp_085}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

case "$MODEL_NAME" in
  exp_063) SOURCE_CHECKPOINT="$SOURCE_ROOT/training/exp_proton_063/model_weights/epoch_last.pt" ;;
  exp_082) SOURCE_CHECKPOINT="$SOURCE_ROOT/training/exp_proton_082/model_weights/epoch375_step218666.pt" ;;
  exp_085) SOURCE_CHECKPOINT="$SOURCE_ROOT/training/exp_proton_085/model_weights/epoch365_step206190.pt" ;;
  *) echo "Choose exp_063, exp_082, or exp_085" >&2; exit 2 ;;
esac

test -f "$SOURCE_CHECKPOINT"
test -f "$DATA_DIR/list.csv"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

exec "$PYTHON" -u "$REPO_ROOT/training/train_ph_adapter_light_v1.py" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --model-name "$MODEL_NAME" \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_ROOT/${MODEL_NAME}_ph_v1" \
  --epochs "${EPOCHS:-100}" \
  --num-train-examples 6000 \
  --num-valid-examples 1200 \
  --min-train-examples 500 \
  --min-valid-examples 100 \
  --batch-size 800 \
  --max-protein-length 800 \
  --learning-rate 0.001 \
  --state-loss-weight 0.25 \
  --sensitivity-batches 16 \
  --mixed-precision
