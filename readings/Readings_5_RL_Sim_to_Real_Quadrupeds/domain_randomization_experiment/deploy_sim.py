"""
deploy_sim.py - run an exported policy through the robot-deployment loop (phase 2 demo).

    python export_policy.py                                   # once
    python deploy_sim.py --run randomized_seed0 --condition pseudo_real

This is the loop a real robot would run, at 50 Hz:
    state  = robot.read_state()                 (Unitree: LowState)
    obs    = build_observation(state, prev_action)
    action = policy.act(obs)                    (numpy only, no torch)
    robot.send_command(q_targets, kp, kd)       (Unitree: LowCmd q / kp / kd)
Here the robot is SimRobot (MuJoCo). In phase 2, swap in a real RobotInterface
(deploy/robot_interface.py: Go2Robot template) and nothing else changes.

It also checks that the loop reproduces the training environment exactly, so the
deployment code is known to match what the policy was trained on.
"""
import argparse

import numpy as np

import config as C
from deploy.numpy_policy import NumpyPolicy
from deploy.observation import action_to_joint_targets, build_observation
from deploy.robot_interface import SimRobot
from envs.walker_env import WalkerEnv
from export_policy import EXPORT_FILE


def run_deployment_loop(policy, robot, steps=C.MAX_EPISODE_STEPS):
    prev_action = np.zeros(4)
    for t in range(steps):
        state = robot.read_state()
        if state["base_height"] < C.MIN_TORSO_HEIGHT or abs(state["pitch"]) > C.MAX_TILT:
            return t, True                                    # fell -> e-stop
        action = policy.act(build_observation(state, prev_action))
        robot.send_command(action_to_joint_targets(action), C.KP, C.KD)
        robot.wait_for_next_cycle()
        prev_action = action
    return steps, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment-loop demo with an exported policy")
    parser.add_argument("--run", default="randomized_seed0")
    parser.add_argument("--condition", default="pseudo_real", choices=list(C.TEST_CONDITIONS))
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    npz = C.RUNS_DIR / args.run / EXPORT_FILE
    if not npz.exists():
        raise SystemExit(f"{npz} not found - run `python export_policy.py` first.")
    policy, physics = NumpyPolicy(npz), C.TEST_CONDITIONS[args.condition]

    # 1) consistency check: deployment loop vs. the gym environment, same start state
    robot, env = SimRobot(physics, seed=C.EVAL_SEED), WalkerEnv(physics=physics)
    obs, _ = env.reset(seed=C.EVAL_SEED)
    prev = np.zeros(4)
    for _ in range(100):
        o_dep = build_observation(robot.read_state(), prev)
        assert np.allclose(o_dep, obs, atol=1e-5), "deployment observation != training observation"
        a = policy.act(o_dep)
        robot.send_command(action_to_joint_targets(a), C.KP, C.KD)
        robot.wait_for_next_cycle()
        obs, _, term, _, _ = env.step(a)
        prev = a
        if term:
            break
    print("Consistency check passed: the deployment loop reproduces the training environment.")

    # 2) run episodes through the loop
    print(f"\n{args.run} on '{args.condition}' {physics}")
    wins = 0
    for ep in range(args.episodes):
        robot = SimRobot(physics, seed=C.EVAL_SEED + ep)
        steps, fell = run_deployment_loop(policy, robot)
        ok = (not fell) and robot.distance() >= C.SUCCESS_DISTANCE
        wins += ok
        print(f"  episode {ep}: {robot.distance():5.2f} m in {steps * C.CONTROL_DT:4.1f} s  "
              f"{'FELL' if fell else 'ok  '}  success={ok}")
    print(f"Success {wins}/{args.episodes}")
