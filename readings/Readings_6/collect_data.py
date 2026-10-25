"""
collect_data.py - run controllers in Pendulum-v1 and store what happened.

Every controller in this project (random, random shooting, CEM, distilled
policy) has the same two methods:

    controller.reset()      # called at the start of each episode
    controller.act(obs)     # returns a numpy action of shape (1,)

`rollout()` runs one episode with any controller and records transitions
(state, action, next_state, reward) together with the time each decision
took. `collect_random_dataset()` uses it to build the first training set for
the world model and saves it as a .npz file.
"""

import os

import numpy as np

from utils import Stopwatch, make_env


class RandomController:
    """Uniform random torques in [-2, 2]: the baseline and the data collector."""

    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs):
        return self.rng.uniform(-2.0, 2.0, size=(1,)).astype(np.float32)


def rollout(env, controller, seed, start_hanging_down=False, render=False):
    """Run one full episode (200 steps for Pendulum-v1).

    Args:
        env: a Gymnasium environment.
        controller: object with reset() and act(obs).
        seed: seed for env.reset, which fixes the random initial state.
        start_hanging_down: if True, overwrite the initial state with the
            pole hanging straight down at rest (used for the GIFs, so every
            controller faces the hardest swing-up).
        render: if True, also return the rgb frames.

    Returns:
        dict with obs, actions, next_obs, rewards (arrays), the episode
        return, the per-step decision times and optionally frames.
    """
    obs, _ = env.reset(seed=seed)
    if start_hanging_down:
        env.unwrapped.state = np.array([np.pi, 0.0])
        obs = env.unwrapped._get_obs()
    controller.reset()
    sw = Stopwatch()
    frames = [env.render()] if render else None
    O, A, O2, R = [], [], [], []
    done = False
    while not done:
        with sw:
            action = controller.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        O.append(obs); A.append(action); O2.append(next_obs); R.append(reward)
        obs = next_obs
        done = terminated or truncated
        if render:
            frames.append(env.render())
    out = dict(
        obs=np.asarray(O, np.float32),
        actions=np.asarray(A, np.float32),
        next_obs=np.asarray(O2, np.float32),
        rewards=np.asarray(R, np.float32),
        ret=float(np.sum(R)),
        decision_times=np.asarray(sw.times),
    )
    if render:
        out["frames"] = frames
    return out


def episodes_to_dataset(episodes):
    """Stack a list of rollout() results into one transition dataset."""
    return dict(
        obs=np.concatenate([e["obs"] for e in episodes]),
        actions=np.concatenate([e["actions"] for e in episodes]),
        next_obs=np.concatenate([e["next_obs"] for e in episodes]),
        rewards=np.concatenate([e["rewards"] for e in episodes]),
    )


def merge_datasets(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def collect_random_dataset(cfg, seed, n_episodes, path=None):
    """Collect `n_episodes` of random-action data, optionally saving a .npz.

    Episodes are kept whole (shape episodes x 200 x dim are recoverable from
    the flat arrays) so the same function can create held-out trajectories
    for the multi-step prediction test.
    """
    env = make_env(cfg.env_id)
    controller = RandomController(seed)
    episodes = [rollout(env, controller, seed=seed * 10_000 + i) for i in range(n_episodes)]
    env.close()
    data = episodes_to_dataset(episodes)
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **data)
    return data, episodes
