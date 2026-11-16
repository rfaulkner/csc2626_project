#!/bin/bash
#SBATCH --job-name=csc2626-sweep
#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%A_%a_%x.out
#SBATCH --error=slurm/output/%A_%a_%x.err
#SBATCH --array=0-79%20
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Week 4: Main ACT vs LANG sweep across tasks, variants, lambda, and 5 seeds.
# Default grid: 2 tasks * 2 variants * 4 lambdas * 5 seeds = 80 array tasks
# (each task runs LANG and ACT + random-gated controls on the same seed/BC ckpt).

set -euo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"
mkdir -p slurm/output results

export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# shellcheck disable=SC1091
source .venv/bin/activate

readonly TASKS=(stack sort)
readonly VARIANTS=(semantic exec)
readonly LAMS=(0.0 0.05 0.1 0.2)
readonly SEEDS=(0 1 2 3 4)
readonly CHANNELS="${CHANNELS:-bc lang act random_lang random_act}"
readonly BUDGET="${BUDGET:-100}"

idx="${SLURM_ARRAY_TASK_ID:-0}"
seed="${SEEDS[$((idx % 5))]}"
idx=$((idx / 5))
lam="${LAMS[$((idx % 4))]}"
idx=$((idx / 4))
variant="${VARIANTS[$((idx % 2))]}"
task="${TASKS[$((idx / 2))]}"

for channel in ${CHANNELS}; do
  # BC does not depend on lambda; run once at lam=0.0.
  if [[ "${channel}" == "bc" && "${lam}" != "0.0" ]]; then
    continue
  fi
  python run_iil.py \
    --task "${task}" \
    --variant "${variant}" \
    --channel "${channel}" \
    --budget "${BUDGET}" \
    --lam "${lam}" \
    --seed "${seed}"
done
