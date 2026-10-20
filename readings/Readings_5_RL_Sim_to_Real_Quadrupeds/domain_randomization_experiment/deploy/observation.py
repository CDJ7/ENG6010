"""
observation.py - build the policy observation from a robot state.

The state dict is what a robot can actually measure (IMU + joint encoders), so the same
function works for the MuJoCo robot and, in phase 2, a real one (Unitree LowState):
    base_height     torso height (m)             - from a state estimator on a real robot
    pitch           body pitch (rad)             - IMU
    joint_pos       joint angles (rad)           - joint encoders
    base_lin_vel    [vx, vz] (m/s)               - state estimator
    pitch_rate      body pitch rate (rad/s)      - IMU gyroscope
    joint_vel       joint velocities (rad/s)     - joint encoders
It must stay identical to WalkerEnv._get_obs (deploy_sim.py checks this).
"""
import numpy as np

import config as C


def build_observation(state: dict, prev_action: np.ndarray) -> np.ndarray:
    return np.concatenate([
        [state["base_height"], state["pitch"]],
        np.asarray(state["joint_pos"]) - np.asarray(C.DEFAULT_POSE),
        [state["base_lin_vel"][0], state["base_lin_vel"][1], state["pitch_rate"]],
        np.clip(state["joint_vel"], -20, 20) * 0.1,
        prev_action,
    ]).astype(np.float32)


def action_to_joint_targets(action: np.ndarray) -> np.ndarray:
    """Policy action in [-1, 1] -> joint position targets (rad)."""
    return np.asarray(C.DEFAULT_POSE) + C.ACTION_SCALE * np.clip(action, -1.0, 1.0)
