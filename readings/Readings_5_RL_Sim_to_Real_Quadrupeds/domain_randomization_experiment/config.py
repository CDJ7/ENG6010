"""
config.py - every setting for the Domain Randomization Experiment lives here.

ENGR 6010 (AI in Robotics), Readings 5.
Change a number here and every script (train / evaluate / plot / watch) picks it up.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (pathlib so they work the same on Windows and everywhere else)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
ROBOT_XML = ROOT_DIR / "envs" / "robot.xml"
RUNS_DIR = ROOT_DIR / "runs"            # trained models: runs/<mode>_seed<N>/
RESULTS_DIR = ROOT_DIR / "results"      # CSVs and plots
# The smoke test (quick_test.py, or DR_QUICK=1) writes to separate folders so it
# never overwrites the real trained models and results.
if os.environ.get("DR_QUICK") == "1":
    RUNS_DIR = ROOT_DIR / "runs_quick"
    RESULTS_DIR = ROOT_DIR / "results_quick"
VIDEOS_DIR = RESULTS_DIR / "videos"     # MP4s from watch.py --record

MODEL_FILE = "model.zip"                # PPO weights
VECNORM_FILE = "vecnormalize.pkl"       # VecNormalize running mean / std
TRAIN_LOG_FILE = "training_log.csv"     # reward curve written during training

# ---------------------------------------------------------------------------
# Simulation and control (position targets + PD, as in Hwangbo et al. 2019
# and most of the policies in the Gurram et al. 2024 survey)
# ---------------------------------------------------------------------------
PHYSICS_DT = 0.005          # MuJoCo timestep (s); must match robot.xml
FRAME_SKIP = 4              # physics substeps per policy step -> 50 Hz control
CONTROL_DT = PHYSICS_DT * FRAME_SKIP

# Joint order in the model: back hip, back knee, front hip, front knee
DEFAULT_POSE = [0.3, -0.6, 0.3, -0.6]   # standing joint angles (rad)
ACTION_SCALE = 0.6                       # action in [-1, 1] -> +/- 0.6 rad around the pose
KP = 40.0                                # PD proportional gain (Nm/rad)
KD = 1.0                                 # PD derivative gain (Nm*s/rad)
TORQUE_LIMIT = 12.0                      # nominal motor torque limit (Nm)

# ---------------------------------------------------------------------------
# Task and reward
# ---------------------------------------------------------------------------
TARGET_SPEED = 1.0          # m/s, forward speed the reward asks for
MAX_EPISODE_STEPS = 500     # 500 steps * 0.02 s = 10 s episodes
ALIVE_BONUS = 0.5
FORWARD_WEIGHT = 2.0        # reward = FORWARD_WEIGHT * min(vx, TARGET_SPEED)
TORQUE_PENALTY = 0.02       # times mean((tau / limit)^2)
TILT_PENALTY = 0.5          # times pitch^2
MIN_TORSO_HEIGHT = 0.18     # below this the robot has fallen (m)
MAX_TILT = 1.0              # |pitch| above this (rad, ~57 deg) counts as a fall
RESET_NOISE = 0.05          # uniform noise added to the initial joint angles / velocities

# ---------------------------------------------------------------------------
# Definition of success (explained in README.md)
# An episode is a success if the torso travels at least SUCCESS_DISTANCE metres
# forward within MAX_EPISODE_STEPS (10 s) without falling.
# ---------------------------------------------------------------------------
SUCCESS_DISTANCE = 5.0

# ---------------------------------------------------------------------------
# Domain randomization ranges (sampled uniformly at every episode reset)
# Follows the survey's list: friction, mass and motor strength.
# ---------------------------------------------------------------------------
RANDOMIZATION_RANGES = {
    "friction_scale": (0.5, 1.5),
    "mass_scale": (0.8, 1.2),
    "motor_strength_scale": (0.7, 1.1),
}

NOMINAL_PHYSICS = {"friction_scale": 1.0, "mass_scale": 1.0, "motor_strength_scale": 1.0}

# ---------------------------------------------------------------------------
# PPO training (identical for both conditions so randomization is the ONLY difference)
# ---------------------------------------------------------------------------
MODES = ["fixed", "randomized"]
DEFAULT_SEEDS = [0]                # one seed for now; use e.g. [0, 1, 2] for the full study
TOTAL_TIMESTEPS = 1_500_000
QUICK_TIMESTEPS = 20_000           # --quick smoke test
N_ENVS = 8                         # parallel copies of the env inside DummyVecEnv

PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=512,                   # per env -> 4096 samples per update
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), log_std_init=-0.5),
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_EPISODES = 50
EVAL_SEED = 12345                  # episode i uses seed EVAL_SEED + i for every model

# Test conditions: (friction, mass, motor strength). Several are deliberately
# OUTSIDE the training ranges to test generalization.
TEST_CONDITIONS = {
    "nominal":         dict(friction_scale=1.0, mass_scale=1.0,  motor_strength_scale=1.0),
    "low_friction":    dict(friction_scale=0.4, mass_scale=1.0,  motor_strength_scale=1.0),
    "high_friction":   dict(friction_scale=1.6, mass_scale=1.0,  motor_strength_scale=1.0),
    "heavy":           dict(friction_scale=1.0, mass_scale=1.3,  motor_strength_scale=1.0),
    "light":           dict(friction_scale=1.0, mass_scale=0.7,  motor_strength_scale=1.0),
    "weak_motors":     dict(friction_scale=1.0, mass_scale=1.0,  motor_strength_scale=0.6),
    "pseudo_real":     dict(friction_scale=0.6, mass_scale=1.25, motor_strength_scale=0.75),
}

# Friction sweep for friction_sweep.png
FRICTION_SWEEP = [0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.65, 1.8]
SWEEP_EPISODES = 30
