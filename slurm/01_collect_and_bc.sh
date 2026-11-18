#!/bin/bash
#SBATCH --job-name=csc2626-bc
#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%A_%a_%x.out
#SBATCH --error=slurm/output/%A_%a_%x.err
#SBATCH --array=0-3
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Weeks 1-2: Collect N=50 expert demos, train initial BC policies (pi_h, pi_l)
# across 5 seeds, check per-subgoal competence, and calibrate (tau_h, tau_l).

set -euo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"
mkdir -p slurm/output data checkpoints

export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# shellcheck disable=SC1091
source .venv/bin/activate

readonly TASKS=(stack stack sort sort)
readonly VARIANTS=(semantic exec semantic exec)
readonly IDX="${SLURM_ARRAY_TASK_ID:-0}"
readonly TASK="${TASKS[${IDX}]}"
readonly VARIANT="${VARIANTS[${IDX}]}"
readonly NUM_DEMOS="${NUM_DEMOS:-50}"
readonly SEEDS="${SEEDS:-0 1 2 3 4}"

demos_file="data/demos_${TASK}_${VARIANT}.pt"
if [[ ! -f "${demos_file}" ]]; then
  python collect_demos.py \
    --task "${TASK}" \
    --variant "${VARIANT}" \
    --num_demos "${NUM_DEMOS}" \
    --out "${demos_file}"
fi

for seed in ${SEEDS}; do
  python train_bc.py \
    --task "${TASK}" \
    --variant "${VARIANT}" \
    --demos "${demos_file}" \
    --out_dir "checkpoints/${TASK}_${VARIANT}_seed${seed}" \
    --skip_existing \
    --seed "${seed}"
done
