"""
planner.py - Mode-2 controllers: Model-Predictive Control in the learned model.

LeCun (2022) describes "Mode-2" perception-planning-action: the agent uses
its world model to imagine the outcome of a sequence of actions, scores the
imagined trajectory with the cost module, optimises the action sequence to
lower the cost, then executes only the *first* action and starts over at the
next time step. That is exactly Model-Predictive Control (MPC).

Two optimisers are implemented, both searching over open-loop action
sequences of length H = 15 (the same as DreamerV3's `imag_length: 15`):

* RandomShootingPlanner - sample N random sequences, keep the best one.
* CEMPlanner - the Cross-Entropy Method: repeatedly sample sequences from a
  Gaussian, keep the lowest-cost "elites", refit the Gaussian to them.

Both are deliberately expensive: each decision needs thousands of imagined
model steps. That cost is what Mode-1 distillation (policy.py) removes.
"""

import numpy as np
import torch

from cost import pendulum_cost
from utils import to_tensor

A_LOW, A_HIGH = -2.0, 2.0


@torch.no_grad()
def imagined_cost(model, obs, action_seqs):
    """Total cost of N action sequences, imagined from the same start state.

    Args:
        model: WorldModel.
        obs: (3,) current real observation.
        action_seqs: (N, H) torques.
    Returns:
        (N,) summed cost  sum_t cost(s_t, a_t)  +  cost(s_H, 0).

    # LeCun: the world model predicts s_{t+1}; the fixed intrinsic cost
    # scores each predicted state. No simulator call happens here.
    """
    N, H = action_seqs.shape
    s = obs.expand(N, -1)
    total = torch.zeros(N)
    for t in range(H):
        a = action_seqs[:, t:t + 1]
        total += pendulum_cost(s, a)
        s = model.predict(s, a)                   # imagine one step ahead
    total += pendulum_cost(s, torch.zeros(N, 1))  # terminal state cost
    return total


class CEMPlanner:
    """Cross-Entropy Method MPC (LeCun's Mode-2 controller).

    At every real time step:
      1. start from the previous plan shifted by one step (warm start);
      2. for `iterations` rounds: sample `population` sequences from
         N(mean, std), clip to the torque limits, imagine their cost in the
         world model, and refit mean/std to the `elites` cheapest ones;
      3. return mean[0] - only the first action is executed, then we replan.
    """

    name = "CEM MPC"

    def __init__(self, model, cfg, seed=0):
        self.model = model
        self.H = cfg.plan_horizon
        self.pop = cfg.cem_population
        self.n_elite = cfg.cem_elites
        self.iters = cfg.cem_iterations
        self.init_std = cfg.cem_init_std
        self.momentum = cfg.cem_momentum
        self.gen = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self):
        self.mean = torch.zeros(self.H)

    @torch.no_grad()
    def act(self, obs):
        obs = to_tensor(obs)
        mean = self.mean.clone()
        std = torch.full((self.H,), self.init_std)
        for _ in range(self.iters):
            noise = torch.randn(self.pop, self.H, generator=self.gen)
            samples = torch.clamp(mean + std * noise, A_LOW, A_HIGH)
            costs = imagined_cost(self.model, obs, samples)
            elites = samples[torch.topk(costs, self.n_elite, largest=False).indices]
            mean = self.momentum * mean + (1 - self.momentum) * elites.mean(0)
            std = self.momentum * std + (1 - self.momentum) * elites.std(0)
        # LeCun Mode-2 / MPC: execute only the first action of the plan ...
        action = mean[0].item()
        # ... and keep the rest of the plan as the warm start for the next step.
        self.mean = torch.cat([mean[1:], torch.zeros(1)])
        return np.array([action], dtype=np.float32)


class RandomShootingPlanner:
    """Random-shooting MPC baseline: best of N uniformly random sequences.

    It gets the same number of imagined rollouts per decision as CEM
    (population x iterations) but no iterative refinement.
    """

    name = "Random-shooting MPC"

    def __init__(self, model, cfg, seed=0):
        self.model = model
        self.H = cfg.plan_horizon
        self.n = cfg.rs_samples
        self.gen = torch.Generator().manual_seed(seed)

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs):
        obs = to_tensor(obs)
        samples = A_LOW + (A_HIGH - A_LOW) * torch.rand(self.n, self.H, generator=self.gen)
        costs = imagined_cost(self.model, obs, samples)
        best = samples[torch.argmin(costs)]
        return np.array([best[0].item()], dtype=np.float32)   # first action only, replan
