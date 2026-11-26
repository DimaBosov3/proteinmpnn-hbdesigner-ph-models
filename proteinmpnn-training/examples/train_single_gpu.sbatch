#!/usr/bin/env bash
#SBATCH --job-name=proteinmpnn-ph
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
: "${DATA_DIR:?Export DATA_DIR before submission}"
: "${SOURCE_ROOT:?Export SOURCE_ROOT before submission}"
: "${RUN_ROOT:?Export RUN_ROOT before submission}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
bash scripts/train_one.sh "${MODEL_NAME:-exp_085}"
