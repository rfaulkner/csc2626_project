"""Aggregate Week 3-4 IIL runs (IQM, 95% CI, AUC, Welch t-test) & render plots."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

LAM_INVARIANT = ("bc", "lang", "random_lang")
CHANNEL_ORDER = ("bc", "lang", "act", "act_label", "random_lang", "random_act")


def iqm(values: np.ndarray) -> float:
    if len(values) < 4:
        return float(np.mean(values))
    q25, q75 = np.percentile(values, [25, 75])
    mid = values[(values >= q25) & (values <= q75)]
    return float(np.mean(mid)) if len(mid) else float(np.mean(values))


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
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
    return float(np.sum(0.5 * (ys[:-1] + ys[1:]) * np.diff(xs)) / budget)


def load_runs(results_root: str) -> dict[tuple, dict[int, list[dict]]]:
    grouped: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    pattern = os.path.join(results_root, "**", "metrics.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
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

    # Broadcast lambda-invariant runs (bc, lang, random_lang at lam=0) across
    # all lambdas evaluated for the same (task, variant, budget).
    lams_by_tvb: dict[tuple, set[float]] = defaultdict(set)
    for tk, vr, _, lm, bg in list(grouped.keys()):
        lams_by_tvb[(tk, vr, bg)].add(lm)
    for (tk, vr, bg), lams in lams_by_tvb.items():
        for ch in LAM_INVARIANT:
            src = grouped.get((tk, vr, ch, 0.0, bg))
            if not src:
                continue
            for lm in lams:
                if (tk, vr, ch, lm, bg) not in grouped:
                    grouped[(tk, vr, ch, lm, bg)] = src
    return grouped


def plot_cost_curves(grouped: dict, out_dir: str):
    panels = sorted({(k[0], k[1], k[3], k[4]) for k in grouped})
    for task, variant, lam, budget in panels:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for ch in CHANNEL_ORDER:
            seed_runs = grouped.get((task, variant, ch, lam, budget))
            if not seed_runs:
                continue
            n_rounds = min(len(r) for r in seed_runs.values())
            mean_c = np.mean(
                [[float(r[i]["cum_cost"]) for i in range(n_rounds)] for r in seed_runs.values()],
                axis=0,
            )
            if ch == "bc":
                mean_c = np.linspace(0.0, budget, n_rounds)
            succ_mat = np.array(
                [[float(r[i]["success"]) for i in range(n_rounds)] for r in seed_runs.values()]
            )
            mean_s = succ_mat.mean(axis=0)
            lo_s = np.array([bootstrap_ci(succ_mat[:, i])[0] for i in range(n_rounds)])
            hi_s = np.array([bootstrap_ci(succ_mat[:, i])[1] for i in range(n_rounds)])
            ls = "--" if ch.startswith("random") or ch == "bc" else "-"
            ax.plot(mean_c, mean_s, marker="o", linestyle=ls, label=ch.upper())
            if len(seed_runs) > 1 and ch != "bc":
                ax.fill_between(mean_c, lo_s, hi_s, alpha=0.18)
        ax.set_xlabel("Cumulative Supervision Cost")
        ax.set_ylabel("Eval Success Rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{task} ({variant}) — λ={lam:g}, B={budget:g}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"curve_{task}_{variant}_lam{lam:g}_B{budget:g}.png"),
            dpi=160,
        )
        plt.close(fig)


def plot_lambda_sweeps(summary_by_key: dict, out_dir: str):
    tb_pairs = sorted({(k[0], k[3]) for k in summary_by_key})
    for task, budget in tb_pairs:
        lams = sorted({k[2] for k in summary_by_key if k[0] == task and k[3] == budget})
        if len(lams) < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True)
        for ax, variant in zip(axes, ("semantic", "exec")):
            for ch in ("bc", "lang", "act", "act_label"):
                xs, ys, los, his = [], [], [], []
                for lm in lams:
                    row = summary_by_key.get((task, variant, lm, budget, ch))
                    if row:
                        xs.append(lm)
                        ys.append(row["final_iqm"])
                        los.append(row["ci95_lo"])
                        his.append(row["ci95_hi"])
                if xs:
                    ls = "--" if ch == "bc" else "-"
                    ax.plot(xs, ys, marker="o", linestyle=ls, label=ch.upper())
                    ax.fill_between(xs, los, his, alpha=0.15)
            ax.set_title(f"{task} — {variant}")
            ax.set_xlabel("Takeover Step Cost Weight λ")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Final Success (IQM)")
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"lambda_sweep_{task}_B{budget:g}.png"), dpi=160)
        plt.close(fig)


def plot_failure_breakdowns(summary_by_key: dict, out_dir: str):
    tlb = sorted({(k[0], k[2], k[3]) for k in summary_by_key})
    for task, lam, budget in tlb:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
        any_bars = False
        for ax, variant in zip(axes, ("semantic", "exec")):
            chans, sems, execs = [], [], []
            for ch in ("bc", "lang", "act", "act_label"):
                row = summary_by_key.get((task, variant, lam, budget, ch))
                if row:
                    chans.append(ch.upper())
                    sems.append(row["semantic_fail"])
                    execs.append(row["exec_fail"])
            if chans:
                any_bars = True
                x = np.arange(len(chans))
                ax.bar(x, sems, width=0.55, label="Semantic fail (z ≠ z*)")
                ax.bar(x, execs, width=0.55, bottom=sems, label="Execution fail (z = z*)")
                ax.set_xticks(x)
                ax.set_xticklabels(chans, fontsize=8)
            ax.set_title(f"{task} ({variant}) — λ={lam:g}")
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, axis="y", alpha=0.3)
        if any_bars:
            axes[0].set_ylabel("Failure Rate")
            axes[1].legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"failure_breakdown_{task}_lam{lam:g}_B{budget:g}.png"),
                dpi=160,
            )
        plt.close(fig)


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
    summary_by_key: dict[tuple, dict] = {}
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
        row = {
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
        summary_rows.append(row)
        summary_by_key[(task, variant, lam, budget, channel)] = row

    out_csv = os.path.join(args.out_dir, "summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Write concise Markdown report + Welch's t-tests for Week 4.
    md_lines = [
        "# Week 4 Experiment Summary (ACT vs LANG)",
        "",
        "| Task | Variant | Channel | λ | B | Seeds | Final IQM | Mean [95% CI] | AUC | Sem Fail | Exec Fail | Query Prec |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in summary_rows:
        md_lines.append(
            f"| {r['task']} | {r['variant']} | {r['channel'].upper()} | {r['lam']:g} | {r['budget']:g} "
            f"| {r['n_seeds']} | {r['final_iqm']:.3f} | {r['final_mean']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] "
            f"| {r['auc_mean']:.3f} | {r['semantic_fail']:.2f} | {r['exec_fail']:.2f} | {r['query_precision']:.2f} |"
        )

    settings = sorted({(k[0], k[1], k[2], k[3]) for k in final_by_cond})
    ttest_lines = []
    for task, variant, lam, budget in settings:
        a = final_by_cond.get((task, variant, lam, budget, "lang"))
        b = final_by_cond.get((task, variant, lam, budget, "act"))
        if a is not None and b is not None and len(a) >= 2 and len(b) >= 2:
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            line = (
                f"- **{task}/{variant}** (λ={lam:g}, B={budget:g}): "
                f"LANG={a.mean():.3f} vs ACT={b.mean():.3f} → t={t_stat:.3f}, p={p_val:.4f}"
            )
            ttest_lines.append(line)
            print(line)
    if ttest_lines:
        md_lines.extend(["", "## Welch's t-Tests (LANG vs ACT)", "", *ttest_lines])

    report_path = os.path.join(args.out_dir, "week4_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {out_csv} and {report_path} ({len(summary_rows)} rows).")

    plot_cost_curves(grouped, args.out_dir)
    plot_lambda_sweeps(summary_by_key, args.out_dir)
    plot_failure_breakdowns(summary_by_key, args.out_dir)


if __name__ == "__main__":
    main()
