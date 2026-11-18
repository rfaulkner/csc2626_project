#!/bin/bash
#SBATCH --job-name=csc2626-w4-sweep
#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%A_%a_%x.out
#SBATCH --error=slurm/output/%A_%a_%x.err
#SBATCH --array=0-19
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Week 4: Main ACT vs LANG experiment matrix (2 tasks x 2 variants x 5 seeds = 20 jobs).
# Per (task, variant, seed), runs:
#   1. Matched supervision-cost budget (COST_BUDGET=100):
#      - lambda-invariant channels once (bc, lang, random_lang)
#      - lambda sweep (0.0, 0.05, 0.1, 0.2) for act, act_label, random_act
#   2. Matched per-query budget (QUERY_BUDGET=25 at lambda=0.0):
#      - bc, lang, act, act_label, random_lang, random_act

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
readonly SEEDS=(0 1 2 3 4)
readonly LAMS="${LAMS:-0.0 0.05 0.1 0.2}"
readonly COST_BUDGET="${COST_BUDGET:-100}"
readonly QUERY_BUDGET="${QUERY_BUDGET:-25}"

idx="${SLURM_ARRAY_TASK_ID:-0}"
seed="${SEEDS[$((idx % 5))]}"
idx=$((idx / 5))
variant="${VARIANTS[$((idx % 2))]}"
task="${TASKS[$((idx / 2))]}"

run_cond() {
  local ch="$1"
  local budget="$2"
  local lam="$3"
  python run_iil.py \
    --task "${task}" \
    --variant "${variant}" \
    --channel "${ch}" \
    --budget "${budget}" \
    --lam "${lam}" \
    --seed "${seed}" \
    --skip_existing
}

# 1. Matched cost budget across lambda sweep (plot_results.py broadcasts lam=0 baselines).
for ch in bc lang random_lang; do
  run_cond "${ch}" "${COST_BUDGET}" 0.0
done
for lam in ${LAMS}; do
  for ch in act act_label random_act; do
    run_cond "${ch}" "${COST_BUDGET}" "${lam}"
  done
done

# 2. Matched query-count budget (lam=0.0).
if [[ "${QUERY_BUDGET}" != "0" && "${QUERY_BUDGET}" != "${COST_BUDGET}" ]]; then
  for ch in bc lang act act_label random_lang random_act; do
    run_cond "${ch}" "${QUERY_BUDGET}" 0.0
  done
fi
