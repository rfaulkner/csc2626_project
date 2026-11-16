"""Aggregate multi-seed IIL metrics (IQM, 95% bootstrap CI, AUC, Welch t-test) & plot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def iqm(values: np.ndarray) -> float:
    if len(values) < 4:
        return float(np.mean(values))
    q25, q75 = np.percentile(values, [25, 75])
    mid = values[(values >= q25) & (values <= q75)]
    return float(np.mean(mid)) if len(mid) else float(np.mean(values))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) <= 1:
        v = float(values[0]) if len(values) else float("nan")
        return v, v
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def normalized_auc(costs: list[float], succs: list[float], budget: float) -> float:
    xs = np.minimum(np.array(costs, dtype=np.float64), budget)
    ys = np.array(succs, dtype=np.float64)
    if budget <= 0 or len(xs) < 2:
        return float(ys[-1])
    if xs[-1] < budget:
        xs = np.append(xs, budget)
        ys = np.append(ys, ys[-1])
    area = float(np.sum(0.5 * (ys[:-1] + ys[1:]) * np.diff(xs)))
    return area / budget


def load_runs(results_root: str) -> dict[tuple, dict[int, list[dict]]]:
    grouped: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(results_root, "**", "metrics.csv"), recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (
                    row["task"],
                    row["variant"],
                    row["channel"],
                    float(row["lam"]),
                    float(row["budget"]),
                )
                grouped[key][int(row["seed"])].append(row)
    for seeds in grouped.values():
        for s in seeds:
            seeds[s].sort(key=lambda r: int(r["round"]))
    return grouped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="results/summary")
    args = p.parse_args()

    grouped = load_runs(args.results_dir)
    if not grouped:
        print(f"No metrics.csv files found under {args.results_dir}.")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    summary_rows = []
    final_by_cond: dict[tuple, np.ndarray] = {}

    for (task, variant, channel, lam, budget), seed_runs in sorted(grouped.items()):
        finals, aucs, sems, execs, precs = [], [], [], [], []
        for rows in seed_runs.values():
            costs = [float(r["cum_cost"]) for r in rows]
            succs = [float(r["success"]) for r in rows]
            finals.append(succs[-1])
            aucs.append(normalized_auc(costs, succs, budget))
            sems.append(float(rows[-1]["semantic_fail_rate"]))
            execs.append(float(rows[-1]["exec_fail_rate"]))
            p_vals = [
                float(r["query_precision"])
                for r in rows[1:]
                if r["query_precision"] != "nan" and not np.isnan(float(r["query_precision"]))
            ]
            if p_vals:
                precs.append(float(np.mean(p_vals)))

        arr = np.array(finals, dtype=np.float64)
        final_by_cond[(task, variant, lam, budget, channel)] = arr
        ci_lo, ci_hi = bootstrap_ci(arr)
        summary_rows.append(
            {
                "task": task,
                "variant": variant,
                "channel": channel,
                "lam": lam,
                "budget": budget,
                "n_seeds": len(arr),
                "final_mean": round(float(arr.mean()), 4),
                "final_iqm": round(iqm(arr), 4),
                "ci95_lo": round(ci_lo, 4),
                "ci95_hi": round(ci_hi, 4),
                "auc_mean": round(float(np.mean(aucs)), 4),
                "semantic_fail": round(float(np.mean(sems)), 4),
                "exec_fail": round(float(np.mean(execs)), 4),
                "query_precision": round(float(np.mean(precs)), 4) if precs else float("nan"),
            }
        )

    out_csv = os.path.join(args.out_dir, "summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {out_csv} ({len(summary_rows)} condition summaries).")

    # Welch's t-test between LANG and ACT where both exist.
    settings = {(k[0], k[1], k[2], k[3]) for k in final_by_cond}
    for task, variant, lam, budget in sorted(settings):
        a = final_by_cond.get((task, variant, lam, budget, "lang"))
        b = final_by_cond.get((task, variant, lam, budget, "act"))
        if a is not None and b is not None and len(a) >= 2 and len(b) >= 2:
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            print(
                f"Welch t-test [{task}/{variant} lam={lam:g} B={budget:g}] "
                f"LANG({a.mean():.3f}) vs ACT({b.mean():.3f}): t={t_stat:.3f}, p={p_val:.4f}"
            )

    # Plot success vs cumulative supervision cost per (task, variant, lam).
    panels = sorted({(k[0], k[1], k[3], k[4]) for k in grouped})
    for task, variant, lam, budget in panels:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for (tk, vr, ch, lm, bg), seed_runs in sorted(grouped.items()):
            if (tk, vr, lm, bg) != (task, variant, lam, budget):
                continue
            n_rounds = min(len(r) for r in seed_runs.values())
            mean_c = np.mean(
                [[float(r[i]["cum_cost"]) for i in range(n_rounds)] for r in seed_runs.values()],
                axis=0,
            )
            succ_mat = np.array(
                [[float(r[i]["success"]) for i in range(n_rounds)] for r in seed_runs.values()]
            )
            mean_s = succ_mat.mean(axis=0)
            lo_s = np.array([bootstrap_ci(succ_mat[:, i])[0] for i in range(n_rounds)])
            hi_s = np.array([bootstrap_ci(succ_mat[:, i])[1] for i in range(n_rounds)])
            ax.plot(mean_c, mean_s, marker="o", label=ch.upper())
            if len(seed_runs) > 1:
                ax.fill_between(mean_c, lo_s, hi_s, alpha=0.2)
        ax.set_xlabel("Cumulative Supervision Cost")
        ax.set_ylabel("Eval Success Rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{task} ({variant}) — λ={lam:g}, B={budget:g}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, f"curve_{task}_{variant}_lam{lam:g}.png")
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
