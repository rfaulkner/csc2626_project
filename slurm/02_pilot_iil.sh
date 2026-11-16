#!/bin/bash
#SBATCH --job-name=csc2626-pilot
#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%j_%x.out
#SBATCH --error=slurm/output/%j_%x.err
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Week 3: First matched-budget comparison of BC vs LANG vs ACT on one variant.

set -euo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"
mkdir -p slurm/output results

export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# shellcheck disable=SC1091
source .venv/bin/activate

readonly TASK="${TASK:-stack}"
readonly VARIANT="${VARIANT:-semantic}"
readonly BUDGET="${BUDGET:-100}"
readonly LAM="${LAM:-0.1}"
readonly SEED="${SEED:-0}"

for channel in bc lang act; do
  python run_iil.py \
    --task "${TASK}" \
    --variant "${VARIANT}" \
    --channel "${channel}" \
    --budget "${BUDGET}" \
    --lam "${LAM}" \
    --seed "${SEED}"
done

python plot_results.py --results_dir results --out_dir results/summary
