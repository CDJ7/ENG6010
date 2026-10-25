"""
utils.py - small helpers shared by every other file.

* symlog / symexp: the squashing transform from DreamerV3 (Hafner et al.,
  2023, Eq. 1). In the official repo they live in embodied/jax/nets.py.
* set_seed: fixes every random number generator so runs are reproducible.
* Stopwatch: measures how long a controller takes to choose an action.
* make_env: builds Pendulum-v1 (headless rendering when frames are needed).
"""

import os
import random
import time

import numpy as np
import torch

# Pygame (used by Gymnasium's renderer) needs a video driver; "dummy" lets it
# draw into off-screen images on a machine without a display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import gymnasium as gym  # noqa: E402


# ---------------------------------------------------------------------------
# DreamerV3: symlog / symexp
# ---------------------------------------------------------------------------
def symlog(x: torch.Tensor) -> torch.Tensor:
    """symlog(x) = sign(x) * log(1 + |x|).

    DreamerV3 applies this to inputs and regression targets so the network
    sees small numbers whatever the scale of the raw signal. It behaves like
    the identity near zero and like a log for large magnitudes, and it keeps
    the sign (unlike a plain log).
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """symexp(x) = sign(x) * (exp(|x|) - 1), the exact inverse of symlog."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
class Stopwatch:
    """Accumulates wall-clock time over many short calls.

    Used as `with sw: action = controller.act(obs)` so that only the time the
    controller spends *deciding* is counted, not the simulator step.
    """

    def __init__(self):
        self.times = []
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.times.append(time.perf_counter() - self._t0)
        return False

    @property
    def mean_ms(self) -> float:
        return 1000.0 * float(np.mean(self.times)) if self.times else 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def make_env(env_id: str = "Pendulum-v1", render: bool = False) -> gym.Env:
    """Create the Gymnasium environment (200-step episodes for Pendulum)."""
    return gym.make(env_id, render_mode="rgb_array" if render else None)


def to_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)
