"""
policy.py - Mode-1: a fast reactive policy distilled from the Mode-2 planner.

LeCun (2022, Fig. 5) proposes that the slow, deliberate Mode-2 process
(planning with the world model) can be "compiled" into a fast Mode-1 policy
network that maps a state straight to an action. The planner acts as the
teacher; the policy imitates it. Once trained, the policy needs one small
forward pass per decision instead of thousands of imagined model steps.

The Diffusion World Model paper (Ding et al., 2024) makes the same practical
point from the RL side: imagined rollouts are used only while *training*, so
the deployed policy is fast.

Distillation here uses DAgger (Ross et al., 2011) rather than plain
behaviour cloning:

  iteration 0 (BC): run the CEM planner, record (state, planner action),
                    fit the policy by regression.
  iteration k >= 1: run the *current policy*, ask the planner what it would
                    have done in every state the policy visited, add those
                    labels to the dataset, refit.

Plain BC only ever sees states the expert visits. As soon as the student
makes a small mistake it drifts into states it has never been trained on and
the errors snowball (the same compounding-error story as for world models).
DAgger fixes this by labelling the student's own state distribution.
"""

import numpy as np
import torch
import torch.nn as nn

from collect_data import rollout
from utils import to_tensor

OBS_SCALE = torch.tensor([1.0, 1.0, 8.0])   # rough scale of [cos, sin, theta_dot]
MAX_TORQUE = 2.0


class Policy(nn.Module):
    """Small MLP: obs (3) -> torque in [-2, 2] via tanh."""

    def __init__(self, obs_dim=3, act_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        return MAX_TORQUE * torch.tanh(self.net(obs / OBS_SCALE))


class PolicyController:
    """Wraps a Policy with the same reset()/act() interface as the planners."""

    name = "Distilled policy"

    def __init__(self, policy):
        self.policy = policy.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs):
        # LeCun Mode-1: one forward pass, no world model, no search.
        return self.policy(to_tensor(obs).unsqueeze(0))[0].numpy()


def train_policy(states, actions, cfg, seed):
    """Behaviour cloning: minimise ||pi(s) - a_planner||^2 on the dataset."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    S, A = to_tensor(states), to_tensor(actions)
    pi = Policy(S.shape[1], A.shape[1], cfg.pi_hidden)
    opt = torch.optim.Adam(pi.parameters(), lr=cfg.pi_lr)
    for _ in range(cfg.pi_epochs):
        perm = torch.randperm(len(S), generator=g)
        for i in range(0, len(S), cfg.pi_batch_size):
            b = perm[i:i + cfg.pi_batch_size]
            loss = ((pi(S[b]) - A[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        final = ((pi(S) - A) ** 2).mean().item()
    return pi.eval(), final


def label_with_planner(planner, states):
    """Ask the Mode-2 planner which action it would take in each state.

    The states come from one trajectory in time order, so the planner's warm
    start (shifted previous plan) still makes sense from state to state.
    """
    planner.reset()
    return np.stack([planner.act(s) for s in states])


def evaluate_returns(env, controller, seeds):
    return [rollout(env, controller, seed=s)["ret"] for s in seeds]


def distill_with_dagger(planner, env, cfg, seed, verbose=True):
    """Compile the CEM planner into a policy with BC + DAgger.

    Returns:
        the final Policy and a history dict with, per iteration, the dataset
        size, the BC loss and the mean/std return of the policy on fixed
        evaluation episodes.
    """
    eval_seeds = [seed * 1000 + 900 + i for i in range(cfg.dagger_eval_episodes)]
    hist = {"iteration": [], "dataset_size": [], "bc_loss": [],
            "return_mean": [], "return_std": []}

    # Iteration 0: plain behaviour cloning on the planner's own trajectories.
    S, A = [], []
    for i in range(cfg.dagger_expert_episodes):
        ep = rollout(env, planner, seed=seed * 1000 + 500 + i)
        S.append(ep["obs"]); A.append(ep["actions"])
    S, A = np.concatenate(S), np.concatenate(A)

    for it in range(cfg.dagger_iterations + 1):
        if it > 0:
            # DAgger: run the *student*, label the states it visits with the teacher.
            for j in range(cfg.dagger_rollouts_per_iter):
                ep = rollout(env, PolicyController(pi), seed=seed * 1000 + 600 + 10 * it + j)
                S = np.concatenate([S, ep["obs"]])
                A = np.concatenate([A, label_with_planner(planner, ep["obs"])])
        pi, loss = train_policy(S, A, cfg, seed)
        rets = evaluate_returns(env, PolicyController(pi), eval_seeds)
        hist["iteration"].append(it)
        hist["dataset_size"].append(int(len(S)))
        hist["bc_loss"].append(loss)
        hist["return_mean"].append(float(np.mean(rets)))
        hist["return_std"].append(float(np.std(rets)))
        if verbose:
            tag = "BC" if it == 0 else f"DAgger {it}"
            print(f"    {tag:9s} | {len(S):5d} labelled states | loss {loss:.4f} "
                  f"| return {np.mean(rets):8.1f} +/- {np.std(rets):6.1f}")
    return pi, hist
