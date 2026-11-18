#!/bin/bash
#SBATCH --job-name=csc2626-w4-plot
#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%j_%x.out
#SBATCH --error=slurm/output/%j_%x.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Week 4 aggregation: compute IQM, 95% bootstrap CIs, AUC, Welch t-tests, & figures.

set -euo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"
mkdir -p slurm/output results/summary

# shellcheck disable=SC1091
source .venv/bin/activate

python plot_results.py --results_dir results --out_dir results/summary
head -n 40 results/summary/week4_report.md
