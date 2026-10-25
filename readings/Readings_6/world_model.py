"""
world_model.py - the learned dynamics model ("world model").

The world model answers one question: "if I am in state s and apply torque
a, where will I be next?" It is a plain MLP

    f_phi(s, a) -> delta = s' - s

Predicting the *change* in state rather than s' itself is a standard trick
in model-based RL: the change is small and centred near zero, which is much
easier for a network to fit.

Connections to the readings
---------------------------
* LeCun (2022): this is the "world model" module of the agent. Mode-2
  reasoning uses it to imagine the consequences of candidate action
  sequences before acting.
* DreamerV3 (Hafner et al., 2023): DreamerV3 learns a world model from
  experience and trains its behaviour in imagination. Here we keep the same
  idea but with a deterministic MLP on low-dimensional states instead of an
  RSSM on images. The optional `use_symlog` switch applies DreamerV3's
  symlog transform to the regression targets.
* Diffusion World Model (Ding et al., 2024): a one-step model like this one
  is fed its own predictions when rolled out, so its errors compound with
  the horizon. `multi_step_error()` measures exactly that (their Fig. 1.1).
"""

import numpy as np
import torch
import torch.nn as nn

from utils import symexp, symlog, to_tensor

MAX_SPEED = 8.0  # Pendulum-v1 clips theta_dot to [-8, 8]


def mlp(in_dim, out_dim, hidden, layers, act=nn.SiLU):
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), act()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class WorldModel(nn.Module):
    """MLP dynamics model with built-in input/output normalisation."""

    def __init__(self, obs_dim=3, act_dim=1, hidden=200, layers=3, use_symlog=False):
        super().__init__()
        self.use_symlog = use_symlog
        self.net = mlp(obs_dim + act_dim, obs_dim, hidden, layers)
        # Normalisation statistics are buffers: saved with the model, never trained.
        self.register_buffer("in_mean", torch.zeros(obs_dim + act_dim))
        self.register_buffer("in_std", torch.ones(obs_dim + act_dim))
        self.register_buffer("out_mean", torch.zeros(obs_dim))
        self.register_buffer("out_std", torch.ones(obs_dim))

    # -- target transform -----------------------------------------------------
    def encode_target(self, delta):
        """Raw delta -> the space the network is trained in."""
        if self.use_symlog:            # DreamerV3: regress symlog(target)
            delta = symlog(delta)
        return (delta - self.out_mean) / self.out_std

    def decode_target(self, y):
        """Network output -> raw delta (exact inverse of encode_target)."""
        delta = y * self.out_std + self.out_mean
        if self.use_symlog:            # ... and map back with symexp
            delta = symexp(delta)
        return delta

    def fit_normalizer(self, obs, act, next_obs):
        x = torch.cat([obs, act], -1)
        self.in_mean.copy_(x.mean(0))
        self.in_std.copy_(x.std(0) + 1e-6)
        t = next_obs - obs
        if self.use_symlog:
            t = symlog(t)
        self.out_mean.copy_(t.mean(0))
        self.out_std.copy_(t.std(0) + 1e-6)

    # -- prediction -----------------------------------------------------------
    def forward(self, obs, act):
        """Normalised network output (used in the training loss)."""
        x = (torch.cat([obs, act], -1) - self.in_mean) / self.in_std
        return self.net(x)

    def predict(self, obs, act):
        """One imagined step: s_{t+1} = s_t + f(s_t, a_t), in raw units.

        Two pieces of prior knowledge about the state space are applied after
        the network's prediction: (cos, sin) is projected back onto the unit
        circle and theta_dot is clipped to the simulator's limit. They do not
        change *where* the model thinks the pendulum is, they only keep the
        prediction a valid Pendulum observation.
        """
        act = torch.clamp(act, -2.0, 2.0)
        nxt = obs + self.decode_target(self.forward(obs, act))
        cs = nxt[..., :2]
        cs = cs / torch.clamp(cs.norm(dim=-1, keepdim=True), min=1e-6)
        thdot = torch.clamp(nxt[..., 2:], -MAX_SPEED, MAX_SPEED)
        return torch.cat([cs, thdot], -1)


def train_world_model(data, cfg, seed, use_symlog=None, verbose=True):
    """Fit a fresh world model on a transition dataset.

    The dataset is split into train / validation (cfg.wm_val_fraction). The
    loss is the mean-squared error in the normalised target space.

    Returns:
        model (eval mode) and a history dict with per-epoch train/val loss.
    """
    use_symlog = cfg.use_symlog if use_symlog is None else use_symlog
    g = torch.Generator().manual_seed(seed)
    obs, act, nxt = (to_tensor(data[k]) for k in ("obs", "actions", "next_obs"))
    n = len(obs)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * cfg.wm_val_fraction)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    torch.manual_seed(seed)
    model = WorldModel(obs.shape[1], act.shape[1], cfg.wm_hidden, cfg.wm_layers, use_symlog)
    model.fit_normalizer(obs[tr_idx], act[tr_idx], nxt[tr_idx])
    target = model.encode_target(nxt - obs)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.wm_lr, weight_decay=cfg.wm_weight_decay)

    hist = {"train_loss": [], "val_loss": []}
    for epoch in range(cfg.wm_epochs):
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        losses = []
        for i in range(0, len(order), cfg.wm_batch_size):
            b = order[i:i + cfg.wm_batch_size]
            loss = ((model(obs[b], act[b]) - target[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item() * len(b))
        model.eval()
        with torch.no_grad():
            val = ((model(obs[val_idx], act[val_idx]) - target[val_idx]) ** 2).mean().item()
        hist["train_loss"].append(sum(losses) / len(order))
        hist["val_loss"].append(val)
    if verbose:
        print(f"    world model ({'symlog' if use_symlog else 'plain'}): "
              f"{n} transitions, final train {hist['train_loss'][-1]:.4f}, "
              f"val {hist['val_loss'][-1]:.4f}")
    return model, hist


@torch.no_grad()
def multi_step_error(model, episodes, max_h=30):
    """Open-loop prediction error versus horizon (DWM paper, Fig. 1.1).

    From every start time t of every held-out episode we feed the model the
    *true* first state and the *recorded* action sequence, and let it roll
    forward on its own predictions for max_h steps. Nothing from the real
    trajectory is fed back in, so any one-step error is carried along.

    Returns:
        dict with, for h = 1..max_h, the RMSE over the 3 state dimensions and
        the mean absolute angle error in degrees.
    """
    starts_o, act_seqs, true_seqs = [], [], []
    for ep in episodes:
        T = len(ep["obs"])
        for t in range(T - max_h):
            starts_o.append(ep["obs"][t])
            act_seqs.append(ep["actions"][t:t + max_h])
            true_seqs.append(ep["next_obs"][t:t + max_h])
    s = to_tensor(np.stack(starts_o))
    A = to_tensor(np.stack(act_seqs))          # (N, H, 1)
    Y = to_tensor(np.stack(true_seqs))         # (N, H, 3)
    rmse, ang = [], []
    for h in range(max_h):
        s = model.predict(s, A[:, h])
        err = s - Y[:, h]
        rmse.append(err.pow(2).mean().sqrt().item())
        d = torch.atan2(s[:, 1], s[:, 0]) - torch.atan2(Y[:, h, 1], Y[:, h, 0])
        d = torch.atan2(torch.sin(d), torch.cos(d))      # wrap to [-pi, pi]
        ang.append(torch.rad2deg(d.abs()).mean().item())
    return {"horizon": list(range(1, max_h + 1)), "rmse": rmse, "angle_err_deg": ang}
