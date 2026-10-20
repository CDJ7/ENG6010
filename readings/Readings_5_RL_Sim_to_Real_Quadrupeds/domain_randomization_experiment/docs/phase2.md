# Phase 2: Using This Simulation for the Next Project

Phase 1 (this folder) showed in simulation that domain randomization closes a sim-to-"real" gap.
This page explains how the same code base carries into a phase-2 project. The natural next step
from the Readings 5 sources is moving toward a **real quadruped (Unitree Go2)**:

- Hwangbo et al. show the full pipeline.
- The survey lists the tools.
- Kine2Go gives Go2 motion data.
- `unitree_sdk2_python` is the robot interface.

## What phase 1 already provides (all tested)

| Piece | File | What it gives phase 2 |
|-------|------|------------------------|
| Gym environment with PD position control | `envs/walker_env.py` | The same control setup as Hwangbo et al. and the Unitree `LowCmd` (q target + kp + kd) |
| Physics randomization | `config.RANDOMIZATION_RANGES` | Friction, mass + inertia, motor strength, re-sampled every reset and never compounding |
| **Extra sim-to-real effects** | `config.EXTRA_RANDOMIZATION_RANGES`, mode `randomized_plus` | PD-gain randomization, **actuator latency** (0 to 40 ms) and **sensor noise**, the effects Hwangbo et al. and the survey call out |
| Repeatable test suite | `evaluate.py`, `config.TEST_CONDITIONS`, `config.PROBE_CONDITIONS` | Fixed-seed episodes, so any new controller gets a fair comparison |
| Plots + summary table | `plot_results.py` | Works for any number of modes and seeds |
| **Policy export** | `export_policy.py`, `deploy/numpy_policy.py` | Runs the trained network with **numpy only** (no torch on the robot). Matches SB3 to within 1e-6 |
| **Robot interface + control loop** | `deploy/robot_interface.py`, `deploy/observation.py`, `deploy_sim.py` | `read_state()` → observation → policy → `send_command(q, kp, kd)` at 50 Hz. `SimRobot` is the working MuJoCo version. `Go2Robot` is the SDK2 template |
| Pictures and videos without a viewer | `make_media.py`, `watch.py --record` | Offscreen rendering for reports and slides |

`deploy_sim.py` checks that the deployment loop reproduces the training environment exactly.
This means the code a real robot would run is already known to match what the policy saw in
training. Only the `RobotInterface` implementation has to change.

## What the phase-2 probes show

`evaluate.py` also runs probe conditions that the main experiment did **not** randomize:

- 20 ms and 40 ms of actuator latency
- sensor noise
- PD gains at 0.7x
- the pseudo-real robot with latency and noise added

Measured success rates. Fixed and randomized are the mean over 3 seeds, and `randomized_plus` is
1 seed, with 50 episodes each:

| Probe | Fixed | Randomized | randomized_plus |
|-------|-------|------------|-----------------|
| latency 20 ms | 19% | 78% | 100% |
| latency 40 ms | 0% | 9% | 100% |
| sensor noise (std 0.05) | 20% | 71% | 98% |
| soft PD gains 0.7x | 100% | 100% | 100% |
| pseudo-real + 20 ms latency + noise | 0% | 3% | 86% |

The main lesson for phase 2 is that **randomization only protects against what you randomize.**
The randomized controller handles friction, mass and motor changes, but it collapses on latency it
never saw. The `randomized_plus` controller adds gain, latency and noise randomization and handles
every probe. That is the step phase 2 has to take before touching hardware. The trade-off: on
very weak motors (0.6x) it succeeded only 16% of the time, and it trained a bit more slowly.

```powershell
python train.py --mode randomized_plus --seed 0     # ~10-20 min
python evaluate.py
python plot_results.py
```

## A reward-design lesson from phase 1

I tried replacing the capped progress reward `2*min(vx, 1.0)` with a stricter speed-tracking
reward `2*(1.0 - |vx - 1.0|)`, which also penalizes going faster than 1 m/s. The policies did walk
at 1 m/s instead of about 1.3 m/s. However, the randomized controllers then **lost most of their
robustness**. On 20 test episodes for each of 2 seeds, weak motors dropped to 0% and 0%, and
pseudo-real dropped to 10% and 0%. With the capped reward, the same conditions were at 100%. So I
kept the capped reward for the experiment. For phase 2, this means that reward terms change what
gait the policy learns, and the gait decides how robust it is. Velocity-command tracking (roadmap
step 2) should be added together with randomization and re-tested on the probe conditions, not
assumed to be free.

## Roadmap

1. **Swap the robot model.** Point `config.ROBOT_XML` at a Go2 MJCF (for example `unitree_go2` from
   MuJoCo Menagerie). Then extend `walker_env.py` from planar to 3D:
   - 12 joints
   - a gravity vector or quaternion from the IMU instead of a single pitch angle
   - lateral and yaw velocities

   Training, evaluation, plotting, export and the deployment loop do not change.
2. **Velocity commands.** Add a commanded `(vx, vy, yaw rate)` to the observation and track it in the
   reward. This is what Hwangbo et al. trained, and it is the same shape as Unitree's
   `SportClient.Move(vx, vy, vyaw)`.
3. **Close the actuator gap.** Log `LowState` (joint q, dq, estimated torque) from a real Go2 and fit an
   actuator network, as Hwangbo et al. did. Plug it into `_pd_torque()` in place of the ideal PD + clip.
4. **Use Kine2Go.** Its 800 Go2 rollouts (joint kinematics, base state and actions at 60 Hz) can be:
   - reference motions for an imitation reward
   - data for behavioral cloning to pre-train the policy
   - a check on whether the simulated Go2's joint ranges and speeds look realistic
5. **Teacher-student / RMA** (from the survey). Train a teacher that sees the true physics config,
   which the env already knows in `current_physics`. Then train a student that only sees
   proprioception history.
6. **Deploy.** Export with `export_policy.py`, fill in `Go2Robot` with the SDK2 low-level calls,
   and run the unchanged loop from `deploy_sim.py`. Safety first:
   - Turn off sport mode.
   - Start with the robot hanging in a harness, with low gains.
   - Keep an emergency stop in reach.

## Honest limits to carry forward

- The phase-1 robot is planar. It cannot fall sideways and is not a Go2.
- The "real robot" is still a simulator. Only real hardware closes the loop.
- `Go2Robot` is an untested template, because there was no robot or SDK in this environment.
