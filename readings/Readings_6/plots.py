"""
plots.py - every figure in results/ is made here (matplotlib, PNG).

Colours follow one fixed order per controller so a controller has the same
colour in every figure.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {
    "Random": "#8a8a8a",
    "Random-shooting MPC": "#eb6834",
    "CEM MPC": "#2a78d6",
    "Distilled policy": "#1baf7a",
}
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_wm_loss(histories, out_dir):
    """Train/val loss per epoch for each world-model training round (seed 0).

    histories: list of (label, history) pairs.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, (label, h) in enumerate(histories):
        c = SEED_COLORS[i % len(SEED_COLORS)]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=c, lw=2, label=f"{label} - train")
        ax.plot(ep, h["val_loss"], color=c, lw=2, ls="--", label=f"{label} - validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on normalised delta-state targets")
    ax.set_title("World-model training loss (seed 0)")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "world_model_loss.png")


def plot_prediction_error(curves, out_dir):
    """Open-loop multi-step error vs horizon; curves: list of (label, err dict)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (label, e) in enumerate(curves):
        c = SEED_COLORS[i % len(SEED_COLORS)]
        axes[0].plot(e["horizon"], e["rmse"], color=c, lw=2, marker="o", ms=4, label=label)
        axes[1].plot(e["horizon"], e["angle_err_deg"], color=c, lw=2, marker="o", ms=4, label=label)
    for ax in axes:
        ax.axvline(15, color="#555555", lw=1, ls=":")
        ax.set_xlabel("prediction horizon (steps, open loop)")
    axes[0].text(15.4, axes[0].get_ylim()[1] * 0.9, "planning horizon H = 15", fontsize=8, color="#555555")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("state RMSE  [cos, sin, theta_dot]")
    axes[1].set_ylabel("mean |angle error| (degrees)")
    axes[0].set_title("Compounding error of the learned model")
    axes[1].set_title("Angle error vs horizon")
    axes[0].legend(fontsize=8)
    return _save(fig, out_dir, "prediction_error_vs_horizon.png")


def plot_return_bars(summary, out_dir):
    names = list(summary)
    means = [summary[n]["return_mean"] for n in names]
    stds = [summary[n]["return_std"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
           color=[COLORS[n] for n in names], error_kw={"elinewidth": 1.2, "ecolor": "#333333"})
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m - s - 40, f"{m:.0f} ± {s:.0f}", ha="center", va="top", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylabel("episode return (higher is better, 0 = perfect)")
    n = summary[names[0]]["n_episodes"]
    ax.set_title(f"Pendulum-v1 return, mean ± std over {n} episodes")
    ax.set_ylim(min(m - s for m, s in zip(means, stds)) - 180, 0)
    ax.axhline(0, color="#333333", lw=0.8)
    return _save(fig, out_dir, "return_comparison.png")


def plot_decision_time(summary, out_dir):
    names = list(summary)
    t = [summary[n]["decision_time_ms_mean"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(names))
    ax.bar(x, t, width=0.6, color=[COLORS[n] for n in names])
    for xi, v in zip(x, t):
        ax.text(xi, v * 1.15, f"{v:.3f} ms" if v < 1 else f"{v:.1f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(min(t) / 3, max(t) * 4)
    ax.set_xticks(x, names)
    ax.set_ylabel("time per decision (ms, log scale)")
    ax.set_title("Decision time per control step (1 CPU thread)")
    return _save(fig, out_dir, "decision_time_comparison.png")


def plot_dagger(dagger_hists, cem_mean, random_mean, out_dir):
    """Policy return after BC (iteration 0) and each DAgger iteration, per seed."""
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for i, (seed, h) in enumerate(dagger_hists.items()):
        c = SEED_COLORS[i % len(SEED_COLORS)]
        ax.errorbar(h["iteration"], h["return_mean"], yerr=h["return_std"], color=c,
                    lw=2, marker="o", ms=6, capsize=3, label=f"distilled policy, seed {seed}")
    ax.axhline(cem_mean, color=COLORS["CEM MPC"], ls="--", lw=1.5, label="CEM MPC teacher (eval mean)")
    ax.axhline(random_mean, color=COLORS["Random"], ls=":", lw=1.5, label="random policy (eval mean)")
    its = list(dagger_hists.values())[0]["iteration"]
    ax.set_xticks(its, ["BC" if i == 0 else f"DAgger {i}" for i in its])
    ax.set_ylabel("episode return")
    ax.set_title("Distilling Mode-2 into Mode-1")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return _save(fig, out_dir, "distillation_dagger.png")


def plot_mbrl_rounds(rounds_by_seed, out_dir):
    """CEM return on the episodes collected in each model-based RL round."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, (seed, r) in enumerate(rounds_by_seed.items()):
        c = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(r["round"], r["return_mean"], color=c, lw=2, marker="o", ms=6, label=f"seed {seed}")
    rounds = list(rounds_by_seed.values())[0]["round"]
    ax.set_xticks(rounds, [f"model {k}\n({'random data only' if k == 0 else f'+ {k} round(s) CEM data'})"
                          for k in rounds], fontsize=8)
    ax.set_ylabel("CEM MPC return (same start states)")
    ax.set_title("Model-based RL loop: collect with CEM, retrain model")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "mbrl_rounds.png")


def plot_filmstrip(frames_by_controller, out_dir, n_cols=8):
    """Grid of rendered frames: one row per controller, time left to right."""
    names = list(frames_by_controller)
    fig, axes = plt.subplots(len(names), n_cols, figsize=(n_cols * 1.4, len(names) * 1.55))
    for r, name in enumerate(names):
        frames = frames_by_controller[name]
        idx = np.linspace(0, len(frames) - 1, n_cols).astype(int)
        for c, k in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(frames[k])
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"t = {k}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.suptitle("Swing-up from hanging down (seed 0 controllers)", fontsize=10)
    return _save(fig, out_dir, "swingup_filmstrip.png")
