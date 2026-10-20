"""
export_policy.py - export a trained policy to a numpy-only file for deployment (phase 2).

    python export_policy.py                       # every run in runs/
    python export_policy.py --run randomized_seed0

Writes runs/<run>/policy_export.npz with the network weights and the frozen observation
normalization, then checks that deploy/numpy_policy.py gives the SAME actions as
Stable-Baselines3 on 1,000 test observations.
"""
import argparse

import numpy as np
import torch

import config as C
from deploy.numpy_policy import NumpyPolicy
from envs.walker_env import WalkerEnv
from evaluate import find_runs, load_policy

EXPORT_FILE = "policy_export.npz"


def export(run_dir):
    model, vecnorm = load_policy(run_dir)
    layers = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    arrays = {f"W{i}": l.weight.detach().numpy().astype(np.float64) for i, l in enumerate(layers)}
    arrays.update({f"b{i}": l.bias.detach().numpy().astype(np.float64) for i, l in enumerate(layers)})
    arrays.update(
        n_hidden=len(layers),
        W_out=model.policy.action_net.weight.detach().numpy().astype(np.float64),
        b_out=model.policy.action_net.bias.detach().numpy().astype(np.float64),
        obs_mean=vecnorm.obs_rms.mean, obs_var=vecnorm.obs_rms.var,
        clip_obs=vecnorm.clip_obs, epsilon=vecnorm.epsilon,
    )
    out = run_dir / EXPORT_FILE
    np.savez(out, **arrays)

    # check: numpy policy == SB3 policy on real observations from the environment
    policy, env, worst = NumpyPolicy(out), WalkerEnv(randomize=True), 0.0
    obs, _ = env.reset(seed=0)
    for _ in range(1000):
        sb3, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
        mine = policy.act(obs)
        worst = max(worst, float(np.abs(sb3[0] - mine).max()))
        obs, _, term, trunc, _ = env.step(sb3[0])
        if term or trunc:
            obs, _ = env.reset()
    status = "OK" if worst < 1e-4 else "MISMATCH"
    print(f"{run_dir.name:>22s} -> {out.name}  max |numpy - SB3| = {worst:.2e}  [{status}]")
    return worst


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Export trained policies to numpy")
    parser.add_argument("--run", help="run folder name, e.g. randomized_seed0 (default: all)")
    args = parser.parse_args()
    runs = [d for _, _, d in find_runs() if args.run is None or d.name == args.run]
    if not runs:
        raise SystemExit("No matching trained runs in runs/.")
    for d in runs:
        export(d)
