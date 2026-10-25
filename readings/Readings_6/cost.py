"""
cost.py - the fixed "intrinsic cost" used by the planner.

LeCun (2022) splits an autonomous agent into modules. One of them is the
*intrinsic cost*: a fixed, hard-wired function that says how "unhappy" the
agent is in a given state. It is not learned. The world model predicts what
will happen; the cost module scores it; the actor searches for actions that
make the predicted cost small.

For Pendulum-v1 we can use the environment's own reward formula, negated:

    cost(s, u) = theta^2 + 0.1 * theta_dot^2 + 0.001 * u^2

where theta is the pole angle (0 = upright, normalised to [-pi, pi]),
theta_dot the angular velocity and u the motor torque. The planner applies
this formula to states the *learned world model* predicts, never to states
from the simulator. That is what "planning inside the model" means.
"""

import torch


def pendulum_cost(obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Per-step cost for a batch of (predicted) observations and actions.

    Args:
        obs:    (..., 3) tensor of [cos(theta), sin(theta), theta_dot].
        action: (..., 1) tensor of torques.
    Returns:
        (...) tensor of costs, i.e. the negative Pendulum-v1 reward.
    """
    # atan2 recovers the angle in [-pi, pi] directly from (cos, sin), which
    # is exactly Pendulum's angle_normalize(theta).
    theta = torch.atan2(obs[..., 1], obs[..., 0])
    theta_dot = obs[..., 2]
    u = torch.clamp(action[..., 0], -2.0, 2.0)
    return theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * u ** 2
