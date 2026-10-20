"""
make_media.py - render pictures of the robot for slides / reports (no viewer needed).

    python make_media.py

Writes results/media/:
    robot.png                          the robot standing (nominal)
    filmstrip_<condition>.png          fixed (top) vs randomized (bottom), snapshots over time
    <model>_<condition>.gif            short animated clips
Uses offscreen rendering (mujoco.Renderer), so it also works when the live viewer cannot open.
"""
import imageio
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from envs.walker_env import WalkerEnv
from evaluate import find_runs, load_policy

MEDIA_DIR = C.RESULTS_DIR / "media"
SNAP_TIMES = [0.0, 0.3, 0.6, 1.0, 2.0, 4.0]          # seconds
GIF_SECONDS = 4.0


def rollout_frames(model, vecnorm, physics, seconds, seed=C.EVAL_SEED):
    """Run one episode and return (frames, info). Frames keep the last pose after a fall."""
    env = WalkerEnv(physics=physics)
    obs, _ = env.reset(seed=seed)
    frames, info = [env.render()], {"fell": False, "distance": 0.0}
    for _ in range(int(seconds / C.CONTROL_DT)):
        if not info["fell"]:
            action, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
            obs, _, term, _, info = env.step(action[0])
        frames.append(env.render())
    env.close()
    return frames, info


def filmstrip(policies, cond):
    physics = C.TEST_CONDITIONS[cond]
    fig, axes = plt.subplots(len(policies), len(SNAP_TIMES), figsize=(2.6 * len(SNAP_TIMES), 1.9 * len(policies)))
    for r, (mode, (model, vecnorm)) in enumerate(policies.items()):
        frames, info = rollout_frames(model, vecnorm, physics, max(SNAP_TIMES))
        for c, t in enumerate(SNAP_TIMES):
            ax = axes[r, c]
            ax.imshow(frames[int(round(t / C.CONTROL_DT))][90:470, 120:840])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {t:.1f} s", fontsize=9)
            if c == 0:
                ax.set_ylabel("Fixed" if mode == "fixed" else "Randomized", fontsize=10)
        status = "FELL" if info["fell"] else f"still walking, {info['distance']:.1f} m"
        axes[r, -1].text(1.03, 0.5, status, transform=axes[r, -1].transAxes, fontsize=9, va="center")
    fig.suptitle(f"{cond.replace('_', ' ').title()}: friction {physics['friction_scale']}x, "
                 f"mass {physics['mass_scale']}x, motor strength {physics['motor_strength_scale']}x", fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA_DIR / f"filmstrip_{cond}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(1)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    runs = {}
    for mode, seed, d in find_runs():
        if mode in C.MODES and mode not in runs:      # first seed of each main controller
            runs[mode] = load_policy(d)
    if set(runs) != set(C.MODES):
        raise SystemExit("Need trained fixed and randomized models in runs/.")

    env = WalkerEnv()
    env.reset(seed=0)
    imageio.imwrite(MEDIA_DIR / "robot.png", env.render())
    env.close()

    for cond in ["nominal", "low_friction", "weak_motors", "pseudo_real"]:
        filmstrip(runs, cond)
        print("filmstrip", cond)
    for cond in ["low_friction", "pseudo_real"]:
        for mode, (model, vecnorm) in runs.items():
            frames, _ = rollout_frames(model, vecnorm, C.TEST_CONDITIONS[cond], GIF_SECONDS)
            small = [f[60:500:2, 60:900:2] for f in frames[::2]]          # crop, half size, 25 fps
            imageio.mimsave(MEDIA_DIR / f"{mode}_{cond}.gif", small, duration=2 * C.CONTROL_DT * 1000, loop=0)
            print("gif", mode, cond)
    print("Media written to", MEDIA_DIR)
