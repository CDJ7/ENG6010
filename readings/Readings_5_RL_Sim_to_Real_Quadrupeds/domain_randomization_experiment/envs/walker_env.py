"""
walker_env.py - custom Gymnasium environment for the planar bounding robot.

Control setup (follows Hwangbo et al. 2019 and the Gurram et al. 2024 survey):
    policy action a in [-1, 1]^4  ->  joint position targets
        q_target = DEFAULT_POSE + ACTION_SCALE * a
    a PD controller in Python runs at EVERY physics substep:
        tau = kp * (q_target - q) - kd * qdot
    and the torque is clipped to the motor limit. "Motor strength" scales both
    the PD output and the torque limit, the way the survey describes it.

Physics config: friction_scale, mass_scale and motor_strength_scale are applied
to the MuJoCo model at reset, always starting from the NOMINAL values stored at
construction, so the scaling never compounds from one episode to the next.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

import config as C


class WalkerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(round(1.0 / C.CONTROL_DT))}

    def __init__(self, physics: dict | None = None, randomize: bool = False,
                 render_mode: str | None = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(C.ROBOT_XML))
        self.data = mujoco.MjData(self.model)
        assert abs(self.model.opt.timestep - C.PHYSICS_DT) < 1e-9, "PHYSICS_DT must match robot.xml"

        self.randomize = randomize
        self.physics = dict(C.NOMINAL_PHYSICS if physics is None else physics)
        self.render_mode = render_mode
        self._renderer = None

        # --- store nominal values once, so scaling always starts from the originals ---
        self.nominal_friction = self.model.geom_friction.copy()
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_inertia = self.model.body_inertia.copy()

        # joint bookkeeping: 3 root DOF (x, z, pitch) + 4 actuated joints
        self.act_qpos_idx = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
                                      for i in range(self.model.nu)])
        self.act_qvel_idx = np.array([self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
                                      for i in range(self.model.nu)])
        self.default_pose = np.array(C.DEFAULT_POSE, dtype=np.float64)
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        # the rootz slide joint measures displacement from the torso's start height
        self.z0 = float(self.model.body_pos[self.torso_id][2])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = 2 + 4 + 3 + 4 + 4   # height, pitch, q, root vel, qdot, previous action
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.prev_action = np.zeros(self.model.nu)
        self.steps = 0
        self.start_x = 0.0
        self.motor_strength = 1.0

    # ------------------------------------------------------------------ physics
    def sample_physics(self) -> dict:
        """Draw one random physics config from the ranges in config.py."""
        return {k: float(self.np_random.uniform(lo, hi))
                for k, (lo, hi) in C.RANDOMIZATION_RANGES.items()}

    def apply_physics(self, physics: dict) -> None:
        """Scale friction, link masses / inertias and motor strength from the NOMINAL model."""
        m = self.model
        m.geom_friction[:] = self.nominal_friction
        m.geom_friction[:, 0] = self.nominal_friction[:, 0] * physics["friction_scale"]
        # Hwangbo et al. randomized inertial properties: scale mass AND inertia together
        m.body_mass[:] = self.nominal_mass * physics["mass_scale"]
        m.body_inertia[:] = self.nominal_inertia * physics["mass_scale"]
        self.motor_strength = physics["motor_strength_scale"]
        self.current_physics = dict(physics)

    # ------------------------------------------------------------------ helpers
    def _pd_torque(self, q_target: np.ndarray) -> np.ndarray:
        q = self.data.qpos[self.act_qpos_idx]
        qd = self.data.qvel[self.act_qvel_idx]
        tau = self.motor_strength * (C.KP * (q_target - q) - C.KD * qd)
        limit = self.motor_strength * C.TORQUE_LIMIT
        return np.clip(tau, -limit, limit)

    def _get_obs(self) -> np.ndarray:
        qpos, qvel = self.data.qpos, self.data.qvel
        return np.concatenate([
            [qpos[1] + self.z0, qpos[2]],       # torso height, pitch
            qpos[self.act_qpos_idx] - self.default_pose,
            qvel[0:3],                          # vx, vz, pitch rate
            np.clip(qvel[self.act_qvel_idx], -20, 20) * 0.1,
            self.prev_action,
        ]).astype(np.float32)

    def distance(self) -> float:
        return float(self.data.qpos[0] - self.start_x)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "physics" in options:
            self.physics = dict(options["physics"])
        physics = self.sample_physics() if self.randomize else self.physics
        self.apply_physics(physics)

        mujoco.mj_resetData(self.model, self.data)
        n = C.RESET_NOISE
        self.data.qpos[self.act_qpos_idx] = self.default_pose + self.np_random.uniform(-n, n, self.model.nu)
        self.data.qvel[:] = self.np_random.uniform(-n, n, self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self.prev_action = np.zeros(self.model.nu)
        self.steps = 0
        self.start_x = float(self.data.qpos[0])
        return self._get_obs(), {"physics": self.current_physics}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_target = self.default_pose + C.ACTION_SCALE * action

        x_before = self.data.qpos[0]
        torque_cost = 0.0
        for _ in range(C.FRAME_SKIP):           # PD runs at every physics substep
            tau = self._pd_torque(q_target)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            torque_cost += np.mean((tau / C.TORQUE_LIMIT) ** 2)
        torque_cost /= C.FRAME_SKIP
        self.steps += 1

        vx = (self.data.qpos[0] - x_before) / C.CONTROL_DT
        height, pitch = self.data.qpos[1] + self.z0, self.data.qpos[2]
        fell = bool(height < C.MIN_TORSO_HEIGHT or abs(pitch) > C.MAX_TILT)

        reward = (C.FORWARD_WEIGHT * min(vx, C.TARGET_SPEED)
                  + C.ALIVE_BONUS
                  - C.TORQUE_PENALTY * torque_cost
                  - C.TILT_PENALTY * pitch ** 2)

        self.prev_action = action
        terminated = fell
        truncated = self.steps >= C.MAX_EPISODE_STEPS
        dist = self.distance()
        info = {
            "distance": dist,
            "fell": fell,
            "success": bool(truncated and not fell and dist >= C.SUCCESS_DISTANCE),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 540, 960)
        self._renderer.update_scene(self.data, camera="track")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
