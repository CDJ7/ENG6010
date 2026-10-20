"""
evaluate.py - test every trained controller on the shifted "pseudo-real" simulators.

For every model in runs/<mode>_seed<N>/ and every condition in config.TEST_CONDITIONS,
run EVAL_EPISODES episodes. Episode i always uses seed EVAL_SEED + i, so the fixed and
randomized controllers see exactly the same start states. VecNormalize statistics are
frozen (training=False) so evaluation never changes them.

Outputs:
    results/eval_results.csv     one row per episode (model, seed, condition, success, distance, reward)
    results/friction_sweep.csv   one row per episode of the friction sweep

Usage:
    python evaluate.py                 # full evaluation + friction sweep
    python evaluate.py --episodes 10   # faster, fewer episodes
    python evaluate.py --no-sweep      # skip the friction sweep
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config as C
from envs.walker_env import WalkerEnv


def find_runs():
    """Return a list of (mode, seed, run_dir) for every trained model."""
    runs = []
    for d in sorted(C.RUNS_DIR.glob("*_seed*")):
        if (d / C.MODEL_FILE).exists() and (d / C.VECNORM_FILE).exists():
            mode, seed = d.name.rsplit("_seed", 1)
            runs.append((mode, int(seed), d))
    return runs


def load_policy(run_dir):
    """Load the PPO model and its frozen VecNormalize statistics."""
    model = PPO.load(run_dir / C.MODEL_FILE, device="cpu")
    vecnorm = VecNormalize.load(str(run_dir / C.VECNORM_FILE), DummyVecEnv([lambda: WalkerEnv()]))
    vecnorm.training = False        # do NOT update running statistics during evaluation
    vecnorm.norm_reward = False
    return model, vecnorm


def run_episodes(model, vecnorm, physics: dict, n_episodes: int, seed0: int = C.EVAL_SEED):
    """Run n_episodes in lock-step (one batched policy call per step) and return per-episode results."""
    envs = [WalkerEnv(physics=physics) for _ in range(n_episodes)]
    obs = np.stack([env.reset(seed=seed0 + i)[0] for i, env in enumerate(envs)])
    done = np.zeros(n_episodes, dtype=bool)
    total_reward = np.zeros(n_episodes)
    results = [None] * n_episodes

    while not done.all():
        actions, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        for i, env in enumerate(envs):
            if done[i]:
                continue
            o, r, term, trunc, info = env.step(actions[i])
            obs[i] = o
            total_reward[i] += r
            if term or trunc:
                done[i] = True
                results[i] = dict(success=int(info["success"]), distance=info["distance"],
                                  reward=total_reward[i], fell=int(info["fell"]),
                                  steps=env.steps, episode=i)
    for env in envs:
        env.close()
    return results


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Evaluate trained controllers on shifted physics")
    parser.add_argument("--episodes", type=int, default=C.EVAL_EPISODES)
    parser.add_argument("--sweep-episodes", type=int, default=C.SWEEP_EPISODES)
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()

    runs = find_runs()
    if not runs:
        raise SystemExit("No trained models found in runs/. Run train.py first.")
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows, sweep_rows = [], []
    t0 = time.time()
    for mode, seed, run_dir in runs:
        model, vecnorm = load_policy(run_dir)
        for cond, physics in C.TEST_CONDITIONS.items():
            res = run_episodes(model, vecnorm, physics, args.episodes)
            for r in res:
                rows.append(dict(model=mode, seed=seed, condition=cond, **physics, **r))
            sr = np.mean([r["success"] for r in res])
            dist = np.mean([r["distance"] for r in res])
            print(f"{mode:>10s} seed{seed} | {cond:<14s} success {sr:6.1%}  mean distance {dist:5.2f} m")

        if not args.no_sweep:
            for fs in C.FRICTION_SWEEP:
                physics = dict(C.NOMINAL_PHYSICS, friction_scale=fs)
                res = run_episodes(model, vecnorm, physics, args.sweep_episodes)
                for r in res:
                    sweep_rows.append(dict(model=mode, seed=seed, friction_scale=fs, **r))
            print(f"{mode:>10s} seed{seed} | friction sweep done")

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "eval_results.csv", index=False)
    print(f"\nSaved {len(df)} episodes to {C.RESULTS_DIR / 'eval_results.csv'}")
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(C.RESULTS_DIR / "friction_sweep.csv", index=False)
        print(f"Saved friction sweep to {C.RESULTS_DIR / 'friction_sweep.csv'}")
    print(f"Evaluation took {(time.time() - t0) / 60:.1f} min")
