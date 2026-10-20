"""
numpy_policy.py - run an exported PPO policy with nothing but numpy.

On a real robot (phase 2) the onboard computer only needs this file, the .npz written by
export_policy.py, and numpy. Hwangbo et al. note that running their network took only
microseconds; this is the same idea: a few matrix multiplies per control step.
"""
from pathlib import Path

import numpy as np


class NumpyPolicy:
    def __init__(self, npz_path: Path):
        d = np.load(npz_path)
        self.obs_mean, self.obs_var = d["obs_mean"], d["obs_var"]
        self.clip_obs, self.eps = float(d["clip_obs"]), float(d["epsilon"])
        n = int(d["n_hidden"])
        self.hidden = [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)]
        self.W_out, self.b_out = d["W_out"], d["b_out"]

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Same formula as Stable-Baselines3 VecNormalize (frozen statistics)."""
        z = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.eps)
        return np.clip(z, -self.clip_obs, self.clip_obs)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action in [-1, 1] for one raw (un-normalized) observation."""
        h = self.normalize(np.asarray(obs, dtype=np.float64))
        for W, b in self.hidden:
            h = np.tanh(W @ h + b)
        return np.clip(self.W_out @ h + self.b_out, -1.0, 1.0)
