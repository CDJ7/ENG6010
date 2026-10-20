"""
train.py - train PPO walking controllers with and without domain randomization.

Examples (from the project folder, with the venv active):
    python train.py --quick                       # few-minute smoke test of both modes
    python train.py --mode fixed                  # fixed physics, default seeds
    python train.py --mode randomized --seed 0    # randomized physics, one seed
    python train.py --all                         # both modes x all DEFAULT_SEEDS
    python train.py --mode randomized_plus        # phase 2: + gains / latency / sensor noise

Each run is saved to runs/<mode>_seed<N>/ :
    model.zip, vecnormalize.pkl, training_log.csv
"""
import argparse
import csv
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

import config as C
from envs.walker_env import WalkerEnv


class TrainingLogger(BaseCallback):
    """After every PPO rollout, append the recent mean episode reward / length / distance to a CSV."""

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.t0 = time.time()

    def _on_training_start(self):
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "mean_episode_reward", "mean_episode_length",
                                    "mean_distance", "wall_time_s"])

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if len(buf) == 0:
            return
        rew = np.mean([e["r"] for e in buf])
        length = np.mean([e["l"] for e in buf])
        dist = np.mean([e.get("distance", np.nan) for e in buf])
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, f"{rew:.3f}", f"{length:.1f}",
                                    f"{dist:.3f}", f"{time.time() - self.t0:.1f}"])


def make_env(randomize: bool, extras: bool, seed: int, rank: int):
    def _init():
        env = WalkerEnv(randomize=randomize, extras=extras)
        env.reset(seed=seed * 1000 + rank)
        return env
    return _init


def train(mode: str, seed: int, timesteps: int) -> None:
    run_dir = C.RUNS_DIR / f"{mode}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    randomize = mode in ("randomized", "randomized_plus")
    extras = mode == "randomized_plus"        # phase 2: also gains / latency / sensor noise
    print(f"\n=== Training {mode} | seed {seed} | {timesteps:,} timesteps -> {run_dir} ===")

    torch.manual_seed(seed)
    np.random.seed(seed)
    venv = DummyVecEnv([make_env(randomize, extras, seed, i) for i in range(C.N_ENVS)])
    venv = VecMonitor(venv, info_keywords=("distance",))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=C.PPO_KWARGS["gamma"])

    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **C.PPO_KWARGS)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=TrainingLogger(run_dir / C.TRAIN_LOG_FILE),
                progress_bar=False)
    minutes = (time.time() - t0) / 60

    model.save(run_dir / C.MODEL_FILE)
    venv.save(str(run_dir / C.VECNORM_FILE))
    with open(run_dir / "train_info.txt", "w") as f:
        f.write(f"mode={mode}\nseed={seed}\ntimesteps={timesteps}\ntrain_minutes={minutes:.2f}\n")
    venv.close()
    print(f"Done: {mode} seed {seed} in {minutes:.1f} min")


if __name__ == "__main__":   # required on Windows
    torch.set_num_threads(1)  # small MLP: one thread is fastest and leaves cores free
    parser = argparse.ArgumentParser(description="Train PPO with or without domain randomization")
    parser.add_argument("--mode", choices=C.ALL_MODES,
                        help="fixed, randomized, or randomized_plus (phase 2: + gains/latency/noise)")
    parser.add_argument("--seed", type=int, help="single seed (default: all of config.DEFAULT_SEEDS)")
    parser.add_argument("--timesteps", type=int, default=C.TOTAL_TIMESTEPS)
    parser.add_argument("--all", action="store_true",
                        help="train fixed + randomized for every default seed")
    parser.add_argument("--quick", action="store_true", help="tiny smoke test (both modes, seed 0)")
    args = parser.parse_args()

    if args.quick:
        C.RUNS_DIR = C.ROOT_DIR / "runs_quick"   # never overwrite the real models
        modes, seeds, steps = C.MODES, [0], C.QUICK_TIMESTEPS
    else:
        modes = C.MODES if (args.all or args.mode is None) else [args.mode]
        seeds = C.DEFAULT_SEEDS if args.seed is None else [args.seed]
        steps = args.timesteps

    for m in modes:
        for s in seeds:
            train(m, s, steps)
