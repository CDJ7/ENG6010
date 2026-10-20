"""
plot_results.py - turn the evaluation CSVs into the figures and the summary table.

Reads:   results/eval_results.csv, results/friction_sweep.csv, runs/*/training_log.csv
Writes:  results/success_rate_bar.png   (MAIN figure)
         results/gap_summary.png
         results/friction_sweep.png
         results/training_curves.png
         results/summary.csv            (also printed to the console)

Error bars: standard deviation across seeds when there are 2+ seeds per controller;
with a single seed, the binomial standard error sqrt(p(1-p)/n) over the episodes.
"""
import matplotlib
matplotlib.use("Agg")   # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

COLORS = {"fixed": "#D55E00", "randomized": "#0072B2"}   # colour-blind-safe orange / blue
LABELS = {"fixed": "Fixed (no randomization)", "randomized": "Domain randomized"}
NICE = {"nominal": "Nominal", "low_friction": "Low friction\n0.4x", "high_friction": "High friction\n1.6x",
        "heavy": "Heavy\n1.3x mass", "light": "Light\n0.7x mass", "weak_motors": "Weak motors\n0.6x",
        "pseudo_real": "Pseudo-real\n(0.6f, 1.25m, 0.75s)"}


def success_stats(df, group_cols):
    """Mean success over seeds and an error bar (std over seeds, or binomial SE with one seed)."""
    per_seed = df.groupby(group_cols + ["seed"])["success"].agg(["mean", "count"]).reset_index()
    out = per_seed.groupby(group_cols).agg(success_rate=("mean", "mean"), seed_std=("mean", "std"),
                                           n_seeds=("seed", "nunique"), n_eps=("count", "sum")).reset_index()
    binom = np.sqrt(out.success_rate * (1 - out.success_rate) / out.n_eps)
    out["err"] = np.where(out.n_seeds > 1, out.seed_std.fillna(0), binom)
    return out


def plot_main_bar(df):
    stats = success_stats(df, ["model", "condition"])
    conds = list(C.TEST_CONDITIONS)
    modes = [m for m in C.MODES if m in stats.model.unique()]
    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, mode in enumerate(modes):
        s = stats[stats.model == mode].set_index("condition").reindex(conds)
        pos = x + (k - (len(modes) - 1) / 2) * w
        bars = ax.bar(pos, 100 * s.success_rate, w, yerr=100 * s.err, capsize=3,
                      color=COLORS[mode], label=LABELS[mode])
        for b, v, e in zip(bars, s.success_rate, s.err):
            ax.text(b.get_x() + b.get_width() / 2, 100 * (v + e) + 1, f"{100 * v:.0f}%", ha="center",
                    va="bottom", fontsize=8, color=COLORS[mode], fontweight="bold")
    ax.set_xticks(x, [NICE.get(c, c) for c in conds], fontsize=9)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 125)
    n_seeds = stats.n_seeds.max()
    err_note = "std across seeds" if n_seeds > 1 else "binomial standard error, 1 seed"
    ax.set_title(f"Success rate on shifted physics (success = {C.SUCCESS_DISTANCE:.0f} m in "
                 f"{C.MAX_EPISODE_STEPS * C.CONTROL_DT:.0f} s without falling; error bars: {err_note})",
                 fontsize=10)
    ax.axvspan(-0.5, 0.5, color="0.92", zorder=-1)   # shade the training (nominal) condition
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.set_yticks(range(0, 101, 20))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "success_rate_bar.png", dpi=150)
    plt.close(fig)


def plot_gap(df):
    stats = success_stats(df, ["model", "condition"]).set_index(["model", "condition"])
    modes = [m for m in C.MODES if m in stats.index.get_level_values(0)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    w = 0.38
    for k, mode in enumerate(modes):
        vals = [100 * stats.loc[(mode, "nominal"), "success_rate"], 100 * stats.loc[(mode, "pseudo_real"), "success_rate"]]
        pos = x + (k - (len(modes) - 1) / 2) * w
        ax.bar(pos, vals, w, color=COLORS[mode], label=LABELS[mode])
        ax.annotate("", xy=(pos[1], vals[1]), xytext=(pos[0], vals[0]),
                    arrowprops=dict(arrowstyle="->", color=COLORS[mode], lw=1.5))
        ax.text(pos[1], vals[1] + 2, f"gap {vals[0] - vals[1]:.0f} pts", ha="center", fontsize=9,
                color=COLORS[mode], fontweight="bold")
    ax.set_xticks(x, ["Nominal sim\n(training physics)", "Pseudo-real robot\n(friction 0.6x, mass 1.25x, motor 0.75x)"])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Sim-to-'real' gap: success drop from nominal to pseudo-real", fontsize=11)
    ax.legend(loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "gap_summary.png", dpi=150)
    plt.close(fig)


def plot_sweep():
    path = C.RESULTS_DIR / "friction_sweep.csv"
    if not path.exists():
        print("friction_sweep.csv not found - skipping friction_sweep.png")
        return
    sw = pd.read_csv(path)
    stats = success_stats(sw, ["model", "friction_scale"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    lo, hi = C.RANDOMIZATION_RANGES["friction_scale"]
    ax.axvspan(lo, hi, color="0.9", zorder=-1, label=f"randomization range ({lo}-{hi}x)")
    for mode in [m for m in C.MODES if m in stats.model.unique()]:
        s = stats[stats.model == mode]
        ax.errorbar(s.friction_scale, 100 * s.success_rate, yerr=100 * s.err, marker="o", capsize=3,
                    color=COLORS[mode], label=LABELS[mode])
    ax.axvline(1.0, color="0.4", ls=":", lw=1)
    ax.text(1.0, 103, "nominal", ha="center", fontsize=8, color="0.3")
    ax.set_xlabel("Friction scale (x nominal)")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Success rate vs. ground friction", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "friction_sweep.png", dpi=150)
    plt.close(fig)


def plot_training_curves():
    logs = []
    for d in sorted(C.RUNS_DIR.glob("*_seed*")):
        f = d / C.TRAIN_LOG_FILE
        if f.exists() and f.stat().st_size > 0:
            log = pd.read_csv(f)
            mode, seed = d.name.rsplit("_seed", 1)
            log["model"], log["seed"] = mode, int(seed)
            logs.append(log)
    if not logs:
        print("No training logs found - skipping training_curves.png")
        return
    logs = pd.concat(logs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for mode in [m for m in C.MODES if m in logs.model.unique()]:
        lg = logs[logs.model == mode]
        for ax, col in zip(axes, ["mean_episode_reward", "mean_distance"]):
            g = lg.groupby("timesteps")[col]
            mean = g.mean().rolling(5, min_periods=1).mean()
            ax.plot(mean.index, mean.values, color=COLORS[mode], label=LABELS[mode])
            if lg.seed.nunique() > 1:
                std = g.std().rolling(5, min_periods=1).mean()
                ax.fill_between(mean.index, mean - std, mean + std, color=COLORS[mode], alpha=0.2)
    axes[0].set_ylabel("Mean episode reward (training env)")
    axes[1].set_ylabel("Mean distance per episode (m)")
    axes[1].axhline(C.SUCCESS_DISTANCE, color="0.4", ls=":", lw=1)
    axes[1].text(0, C.SUCCESS_DISTANCE + 0.1, "success distance", fontsize=8, color="0.3")
    for ax in axes:
        ax.set_xlabel("Timesteps")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)
    fig.suptitle("Training curves (each controller measured on its own training physics)", fontsize=11)
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def summary_table(df):
    stats = success_stats(df, ["model", "condition"])
    dist = df.groupby(["model", "condition"]).agg(mean_distance=("distance", "mean"),
                                                  fall_rate=("fell", "mean"),
                                                  mean_reward=("reward", "mean")).reset_index()
    table = stats.merge(dist, on=["model", "condition"])
    order = {c: i for i, c in enumerate(C.TEST_CONDITIONS)}
    table = table.sort_values(["condition", "model"], key=lambda s: s.map(order) if s.name == "condition" else s)
    table = table[["condition", "model", "success_rate", "err", "mean_distance", "fall_rate",
                   "mean_reward", "n_seeds", "n_eps"]]
    table.to_csv(C.RESULTS_DIR / "summary.csv", index=False, float_format="%.4f")

    pivot = table.pivot(index="condition", columns="model", values="success_rate").reindex(list(C.TEST_CONDITIONS))
    print("\nSuccess rate per test condition")
    print("-" * 52)
    print(f"{'condition':<16s}" + "".join(f"{m:>14s}" for m in pivot.columns))
    for cond, row in pivot.iterrows():
        print(f"{cond:<16s}" + "".join(f"{100 * v:13.1f}%" for v in row.values))
    if {"nominal", "pseudo_real"} <= set(pivot.index):
        gap = 100 * (pivot.loc["nominal"] - pivot.loc["pseudo_real"])
        print("-" * 52)
        print(f"{'gap (nom-real)':<16s}" + "".join(f"{v:12.1f}pt" for v in gap.values))
    print(f"\nFull table saved to {C.RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    path = C.RESULTS_DIR / "eval_results.csv"
    if not path.exists():
        raise SystemExit("results/eval_results.csv not found. Run evaluate.py first.")
    df = pd.read_csv(path)
    plot_main_bar(df)
    plot_gap(df)
    plot_sweep()
    plot_training_curves()
    summary_table(df)
    print("Plots written to", C.RESULTS_DIR)
