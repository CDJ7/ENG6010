"""
robot_interface.py - one interface, two robots.

The control loop in deploy_sim.py only talks to a RobotInterface:
    read_state()                      -> state dict (see observation.py)
    send_command(q_target, kp, kd)    joint position targets + PD gains
This is the same shape as Unitree's low-level API (unitree_sdk2_python): LowState gives
IMU + joint states, and every motor in LowCmd takes a target position q, kp and kd, with
the PD loop running on the robot's motor drivers. So the policy, the observation code and
the loop stay the same in phase 2 - only the RobotInterface implementation changes.

    SimRobot   MuJoCo implementation (fully working, used by deploy_sim.py)
    Go2Robot   TEMPLATE for a Unitree Go2 over unitree_sdk2_python (not runnable here)
"""
from abc import ABC, abstractmethod

import mujoco
import numpy as np

import config as C
from envs.walker_env import WalkerEnv


class RobotInterface(ABC):
    @abstractmethod
    def read_state(self) -> dict: ...

    @abstractmethod
    def send_command(self, q_target: np.ndarray, kp: float, kd: float) -> None: ...

    @abstractmethod
    def wait_for_next_cycle(self) -> None:
        """Advance / sleep one control period (CONTROL_DT = 20 ms)."""


class SimRobot(RobotInterface):
    """The MuJoCo robot. The PD loop runs 'on the robot' at every physics substep,
    like a motor driver tracking LowCmd (q, kp, kd)."""

    def __init__(self, physics: dict | None = None, seed: int = 0):
        self._env = WalkerEnv(physics=physics)       # reuses the model + physics scaling
        self._env.reset(seed=seed)
        self.model, self.data = self._env.model, self._env.data
        self.start_x = float(self.data.qpos[0])
        self._q_target = self.data.qpos[self._env.act_qpos_idx].copy()
        self._kp, self._kd = C.KP, C.KD

    def read_state(self) -> dict:
        e, qpos, qvel = self._env, self.data.qpos, self.data.qvel
        return dict(base_height=qpos[1] + e.z0, pitch=qpos[2],
                    joint_pos=qpos[e.act_qpos_idx].copy(), base_lin_vel=qvel[0:2].copy(),
                    pitch_rate=qvel[2], joint_vel=qvel[e.act_qvel_idx].copy())

    def send_command(self, q_target, kp, kd):
        self._q_target, self._kp, self._kd = np.asarray(q_target), kp, kd

    def wait_for_next_cycle(self):
        e = self._env
        for _ in range(C.FRAME_SKIP):
            q, qd = self.data.qpos[e.act_qpos_idx], self.data.qvel[e.act_qvel_idx]
            tau = e.motor_strength * (self._kp * e.pd_gain_scale * (self._q_target - q)
                                      - self._kd * e.pd_gain_scale * qd)
            limit = e.motor_strength * C.TORQUE_LIMIT
            self.data.ctrl[:] = np.clip(tau, -limit, limit)
            mujoco.mj_step(self.model, self.data)

    def distance(self) -> float:
        return float(self.data.qpos[0] - self.start_x)


class Go2Robot(RobotInterface):
    """TEMPLATE for phase 2 - shows where unitree_sdk2_python plugs in. Not tested here.

    A Go2 has 12 joints and moves in 3D, so phase 2 must first retrain the policy on a Go2
    model (e.g. unitree_go2 from MuJoCo Menagerie) with a matching state dict. The SDK calls
    follow the repo's low-level examples (go2_stand_example.py):
        ChannelFactoryInitialize(0, network_interface)
        sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(callback, 10)
        pub = ChannelPublisher("rt/lowcmd", LowCmd_);      pub.Init()
        cmd.motor_cmd[i].q / .dq / .kp / .kd / .tau ; cmd.crc = CRC().Crc(cmd); pub.Write(cmd)
    Sport mode must be switched off first (MotionSwitcherClient), and SAFETY comes first:
    start with the robot hanging, low gains, and an emergency stop.
    """

    def __init__(self, network_interface: str = "eth0"):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: F401
        except ImportError as e:
            raise RuntimeError("Go2Robot needs unitree_sdk2_python (phase 2, real hardware only)") from e
        raise NotImplementedError("Phase-2 template: fill in with the Go2 low-level example code")

    def read_state(self) -> dict:
        # LowState -> imu_state.rpy / gyroscope, motor_state[i].q / .dq; height and base
        # velocity come from a state estimator (e.g. leg odometry + IMU).
        raise NotImplementedError

    def send_command(self, q_target, kp, kd):
        # for i in range(12): cmd.motor_cmd[i].q = q_target[i]; .kp = kp; .kd = kd; .dq = .tau = 0
        raise NotImplementedError

    def wait_for_next_cycle(self):
        # sleep until the next 20 ms tick (the SDK's motor loop itself runs at 500 Hz)
        raise NotImplementedError
