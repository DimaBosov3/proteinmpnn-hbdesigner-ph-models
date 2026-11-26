#!/usr/bin/env bash
# Sequential use of one allocated GPU; no device probing or job submission.
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${RUN_ROOT:?Set RUN_ROOT}"
PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

for MODEL_NAME in exp_063 exp_082 exp_085; do
  bash "$REPO_ROOT/scripts/train_one.sh" "$MODEL_NAME"
done

"$PYTHON" "$REPO_ROOT/training/summarize_three_ph_adapters_v1.py" \
  --report "$RUN_ROOT/exp_063_ph_v1/report.json" \
  --report "$RUN_ROOT/exp_082_ph_v1/report.json" \
  --report "$RUN_ROOT/exp_085_ph_v1/report.json" \
  --output "$RUN_ROOT/summary.json"
