#!/bin/bash
# Submit the chained Week 1-4 Slurm pipeline (or run a fast local --smoke check):
#   bash slurm/launch_week4.sh           # submits 01 -> 03 -> 04 via sbatch
#   bash slurm/launch_week4.sh --smoke   # runs a miniature local Week 4 dry-run

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p slurm/output

if [[ "${1:-}" == "--smoke" ]]; then
  readonly PY="${PYTHON_BIN:-python}"
  readonly WORK="${SMOKE_DIR:-/tmp/csc2626_w4_smoke}"
  rm -rf "${WORK}" && mkdir -p "${WORK}"
  for variant in semantic exec; do
    "${PY}" collect_demos.py --task stack --variant "${variant}" --num_demos 4 \
      --out "${WORK}/demos_stack_${variant}.pt"
    for seed in 0 1; do
      "${PY}" train_bc.py --task stack --variant "${variant}" \
        --demos "${WORK}/demos_stack_${variant}.pt" \
        --out_dir "${WORK}/ckpt/stack_${variant}_seed${seed}" \
        --steps_h 120 --steps_l 160 --calib_episodes 2 --eval_episodes 4 \
        --text_mode hash --seed "${seed}"
      for ch in bc lang random_lang; do
        "${PY}" run_iil.py --task stack --variant "${variant}" --channel "${ch}" \
          --demos "${WORK}/demos_stack_${variant}.pt" \
          --bc_dir "${WORK}/ckpt/stack_${variant}_seed${seed}" \
          --out_dir "${WORK}/results/stack_${variant}/${ch}_lam0_B8/seed${seed}" \
          --budget 8 --lam 0.0 --rounds 2 --episodes_per_round 2 --eval_episodes 4 \
          --steps_h 100 --steps_l 120 --cf_samples_per_round 1 --text_mode hash --seed "${seed}"
      done
      for lam in 0.0 0.1; do
        for ch in act act_label; do
          "${PY}" run_iil.py --task stack --variant "${variant}" --channel "${ch}" \
            --demos "${WORK}/demos_stack_${variant}.pt" \
            --bc_dir "${WORK}/ckpt/stack_${variant}_seed${seed}" \
            --out_dir "${WORK}/results/stack_${variant}/${ch}_lam${lam}_B8/seed${seed}" \
            --budget 8 --lam "${lam}" --rounds 2 --episodes_per_round 2 --eval_episodes 4 \
            --steps_h 100 --steps_l 120 --cf_samples_per_round 1 --text_mode hash --seed "${seed}"
        done
      done
    done
  done
  "${PY}" plot_results.py --results_dir "${WORK}/results" --out_dir "${WORK}/summary"
  cat "${WORK}/summary/week4_report.md"
  exit 0
fi

job_bc="$(sbatch --parsable slurm/01_collect_and_bc.sh)"
job_sweep="$(sbatch --parsable --dependency="afterok:${job_bc}" slurm/03_sweep_act_lang.sh)"
job_plot="$(sbatch --parsable --dependency="afterok:${job_sweep}" slurm/04_summarize_week4.sh)"
echo "Submitted Week 4 pipeline: BC=${job_bc} -> SWEEP=${job_sweep} -> PLOT=${job_plot}"
