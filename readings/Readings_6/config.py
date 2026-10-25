"""
config.py - every hyperparameter for the project in one place.

This mirrors how DreamerV3 keeps all of its settings in a single
`configs.yaml` file (danijar/dreamerv3). Changing an experiment means
changing numbers here, not hunting through the code.

`get_config(quick=True)` returns a much smaller version of the same
experiment that finishes in about a minute; it is only meant to check that
the pipeline runs, not to produce meaningful numbers.
"""

from dataclasses import dataclass, field, replace
from typing import List


@dataclass
class Config:
    # ------------------------------------------------------------------ general
    env_id: str = "Pendulum-v1"
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    torch_threads: int = 1          # 1 thread = fair, repeatable CPU timing
    results_dir: str = "results"
    data_dir: str = "data"

    # ---------------------------------------------------- 1. data collection
    n_random_episodes: int = 50     # 50 x 200 steps = 10,000 random transitions
    n_heldout_episodes: int = 10    # held-out random trajectories for step 3

    # ------------------------------------------------------- 2. world model
    wm_hidden: int = 128
    wm_layers: int = 3
    wm_lr: float = 1e-3
    wm_weight_decay: float = 1e-5
    wm_epochs: int = 60
    wm_batch_size: int = 256
    wm_val_fraction: float = 0.1
    use_symlog: bool = False        # DreamerV3 symlog on the delta targets
    compare_symlog: bool = True     # also train the other variant (seed 0) for the plots

    # --------------------------------------------- 3. world-model evaluation
    max_eval_horizon: int = 30      # open-loop prediction error for 1..30 steps

    # ------------------------------------------------- 4. Mode-2 planners
    plan_horizon: int = 15          # = DreamerV3 imag_length: 15
    cem_population: int = 300
    cem_elites: int = 30
    cem_iterations: int = 4
    cem_init_std: float = 1.0       # actions live in [-2, 2]
    cem_momentum: float = 0.1       # smoothing of the mean/std updates
    rs_samples: int = 1200          # random shooting gets the same budget (300 x 4)
    # Optional model-based RL loop: collect data with CEM, retrain, repeat.
    # Off by default: with 2 rounds it did not change CEM's return at all
    # (see results/ablation_mbrl/), so it was not worth the extra runtime.
    # Turn it back on with `python main.py --mbrl-rounds 2`.
    mbrl_rounds: int = 0
    mbrl_episodes_per_round: int = 3

    # ------------------------------------------- 5. Mode-1 policy (DAgger)
    pi_hidden: int = 64
    pi_lr: float = 1e-3
    pi_epochs: int = 150
    pi_batch_size: int = 256
    dagger_expert_episodes: int = 3     # iteration 0 = plain behaviour cloning
    dagger_iterations: int = 4          # DAgger iterations after BC
    dagger_rollouts_per_iter: int = 3
    dagger_eval_episodes: int = 5

    # --------------------------------------------------------- 6. evaluation
    n_eval_episodes: int = 5            # per controller, per seed

    # ------------------------------------------------------------ 7. media
    make_gifs: bool = True
    gif_frame_skip: int = 2


def get_config(quick: bool = False) -> Config:
    """Return the full configuration, or a tiny one for `--quick` test runs."""
    cfg = Config()
    if quick:
        cfg = replace(
            cfg,
            seeds=[0],
            results_dir="results_quick",   # never overwrite the real results
            data_dir="data_quick",
            n_random_episodes=10,
            n_heldout_episodes=3,
            wm_epochs=10,
            compare_symlog=False,
            cem_population=100,
            cem_elites=10,
            cem_iterations=3,
            rs_samples=300,
            mbrl_rounds=1,
            mbrl_episodes_per_round=1,
            pi_epochs=20,
            dagger_expert_episodes=1,
            dagger_iterations=1,
            dagger_rollouts_per_iter=1,
            dagger_eval_episodes=1,
            n_eval_episodes=2,
            make_gifs=False,
        )
    return cfg
