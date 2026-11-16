# Language or Action? Robot-Gated Interactive Imitation Learning

Codebase and Slurm scripts for Weeks 1–4 of the CSC2626 project comparing **language subgoal corrections (`LANG`)** and **action takeovers (`ACT`)** under a shared two-level uncertainty gate and matched supervision budget $B$ in ManiSkill3.

## Setup

On a Compute Canada / Killarney **login node** (installs `.venv`, patches module loading, and pre-caches ManiSkill3 assets + CLIP embeddings for offline compute nodes):

```bash
bash setup.sh
source .venv/bin/activate
```

Local setup with `uv`:

```bash
uv venv --python 3.11 .venv
uv pip install -p .venv -r requirements.txt
source .venv/bin/activate
```

## Structure (Weeks 1–4)

| File | Role |
| :--- | :--- |
| `envs.py` | **W1–2**: `OrderedStack-v0` (`stack`) and `ColorSort-v0` (`sort`) in `semantic` (permuted order + distractors) and `exec` (smaller cubes, wider poses, random pushes) variants. |
| `expert.py` | **W1**: `LanguageOracle` ($z_t^*$) and closed-loop `ScriptedPickPlaceExpert` with mid-episode release-and-regrasp recovery. |
| `models.py` | **W2**: Frozen CLIP `TextEncoder`, `HighLevelTransformer` $\pi_h(z \mid s, g, z_{\text{prev}})$, and FiLM-conditioned state `LowLevelDiffusionPolicy` $\pi_l(a \mid s, z)$. |
| `gate.py` | **W3**: Shared gate $q(s) = \mathbb{I}[u_h(s) > \tau_h \lor u_l(s) > \tau_l]$ (entropy + Diff-DAgger diffusion loss) and held-out rollout calibration. |
| `collect_demos.py` | **W1–2**: Collects $N$ scripted demonstrations segmented into $D_h$ and $D_l$. |
| `train_bc.py` | **W2–3**: Trains initial BC $(\pi_h, \pi_l)$, checks per-subgoal competence, and calibrates $(\tau_h, \tau_l)$. |
| `run_iil.py` | **W3–4**: Runs $R$-round budgeted IIL (`bc`, `lang`, `act`, `act_label`, `random_lang`, `random_act`) with $c_z = 1$, $c_a = 1 + \lambda k$, and counterfactual query precision. |
| `plot_results.py` | **W4**: Aggregates seeds (mean, IQM, 95% bootstrap CI, AUC, Welch's $t$-test) and plots success vs. supervision cost. |

## Quickstart (CLI)

```bash
# 1. Collect N=50 demos and pretrain + calibrate initial BC policy
python collect_demos.py --task stack --variant semantic --num_demos 50
python train_bc.py --task stack --variant semantic --seed 0

# 2. Run matched-budget interactive IL (B=100, R=5 rounds, M=20 eps/round)
python run_iil.py --task stack --variant semantic --channel bc   --budget 100 --lam 0.1 --seed 0
python run_iil.py --task stack --variant semantic --channel lang --budget 100 --lam 0.1 --seed 0
python run_iil.py --task stack --variant semantic --channel act  --budget 100 --lam 0.1 --seed 0

# 3. Aggregate metrics and plot curves -> results/summary/
python plot_results.py --results_dir results --out_dir results/summary
```

## Slurm Experiments (`slurm/`)

```bash
# Weeks 1–2: Demo collection + multi-seed BC + gate calibration across all 4 (task, variant) pairs
JOB_BC=$(sbatch --parsable slurm/01_collect_and_bc.sh)

# Week 3: Single-variant pilot comparison (BC vs LANG vs ACT)
sbatch --dependency=afterok:${JOB_BC} slurm/02_pilot_iil.sh

# Week 4: Full sweep (tasks x variants x lambdas x 5 seeds, with random-gate controls)
JOB_SWEEP=$(sbatch --parsable --dependency=afterok:${JOB_BC} slurm/03_sweep_act_lang.sh)
sbatch --dependency=afterok:${JOB_SWEEP} --wrap="source .venv/bin/activate && python plot_results.py"
```
