# Readings 5: Reinforcement Learning and Sim-to-Real Transfer for Quadruped Robots

ENGR 6010 (AI in Robotics)

## What this reading was about

This reading looked at how robots with legs learn to walk using **reinforcement learning (RL)**
in simulation, and how those learned controllers are moved onto a **real robot**. The hard part
is the **"reality gap"**: a policy that works well in a simulator often fails on hardware,
because the simulator's physics (friction, mass, actuators, delays) never quite match the real world.

The assignment covered four sources, one from each category:

1. **Primary reading: Hwangbo et al. (2019), "Learning Agile and Dynamic Motor Skills for Legged Robots"**
   (*Science Robotics*). The ETH Zurich team trained walking, high-speed running and fall-recovery
   policies for the ANYmal quadruped entirely in simulation, then ran them on the real robot with
   no fine-tuning. Their key ideas:
   - They **randomized the robot's inertial properties** so the policy could not overfit to one model.
   - They trained an **"actuator network"** on real motor data, because the actuators were too hard
     to model with equations.
   - The policy outputs **joint position targets**, and a **PD controller** turns them into torques.

   A key result: policies trained on an *idealized* actuator model could not take a single step on
   the real robot.
2. **Review / survey: Gurram, Uttam & Ohol (2024), "Reinforcement Learning for Quadrupedal Locomotion:
   Current Advancements and Future Perspectives"** (arXiv). It gives an overview of the field and covers:
   - Sim-to-real techniques: **domain randomization** of friction, mass, motor strength and gains,
     curriculum learning, teacher-student (privileged) learning, and RMA adaptation modules.
   - Reward design and gait-bound vs. gait-free methods.
   - The fact that most recent work uses **PPO** on low-cost Unitree robots.
3. **Interactive web demos & open-source tools: Kine2Go dataset (MIMUW Robotics, Hugging Face).**
   It is an open-access dataset of **800 simulated rollouts** for the Unitree Go2:
   - It starts from **40 reference motion clips**: dog motion capture and synthetic dog clips from
     AI4Animation, horse clips, and Solo8 robot motions.
   - Each clip is retargeted to the Go2 and tracked by its own **PPO imitation policy** trained in
     the Genesis simulator.
   - Each clip has 20 perturbed rollouts, which log joint kinematics, base state and actions at 60 Hz.

   It shows the same simulation-plus-PPO pipeline as the other readings, run 40 times. Because the
   gait comes from reference motions (gait-bound), it contrasts with the self-discovered trot in the
   primary reading. Because it is all simulation data, it does not close the sim-to-real gap by itself.
4. **Official API documentation: Unitree SDK2 Python (`unitree_sdk2_python`, GitHub).** Unitree's
   official Python SDK over CycloneDDS. It covers the "deploy on the real robot" step:
   - high-level velocity commands
   - low-level `LowState` / `LowCmd` messages, with a position target plus kp and kd gains for each motor

## What's in this folder

| Path | What it is |
|------|------------|
| [`docs/Readings 5.docx`](docs/Readings%205.docx) | My Readings 5 document: summary and "what I learned" for each of the four sources |
| [`domain_randomization_experiment/`](domain_randomization_experiment/) | Code project I built to try the readings' main idea myself (see below) |
| [`domain_randomization_experiment/docs/writeup.md`](domain_randomization_experiment/docs/writeup.md) | "Program Explanation" and "Results" sections for the report |
| [`domain_randomization_experiment.zip`](domain_randomization_experiment.zip) | The same project zipped, ready to unzip and run on Windows |

## The code project: Domain Randomization Experiment

To see the readings' main sim-to-real idea in action, I built a small experiment in MuJoCo with
Stable-Baselines3 PPO:

- I train a planar two-legged "bounding" robot (a 2D slice of a quadruped) to walk twice:
  - once on **fixed** nominal physics;
  - once with **domain randomization**, where friction (0.5x to 1.5x), link mass (0.8x to 1.2x)
    and motor strength (0.7x to 1.1x) are re-sampled at every episode reset.
- As in Hwangbo et al. and the survey, the policy outputs **joint position targets**, and a PD
  controller produces the torques.
- I then test both controllers on **changed simulators** that stand in for "the real robot".

Results (1 seed each, 1.5 M timesteps, 50 test episodes per condition; success = 5 m in 10 s without falling):

| Test condition | Fixed | Randomized |
|----------------|-------|------------|
| Nominal | 100% | 100% |
| Low friction 0.4x | 0% | 100% |
| High friction 1.6x | 100% | 100% |
| Heavy 1.3x mass | 2% | 100% |
| Light 0.7x mass | 100% | 100% |
| Weak motors 0.6x | 0% | 100% |
| **Pseudo-real** (0.6x friction, 1.25x mass, 0.75x motors) | **0%** | **100%** |

The "sim-to-real" gap (nominal minus pseudo-real) shrank from **100 points to 0 points** with
randomization. The randomized controller trained somewhat more slowly but reached the same final reward.

![Success rate bar chart](domain_randomization_experiment/results/success_rate_bar.png)

See [`domain_randomization_experiment/README.md`](domain_randomization_experiment/README.md) for
step-by-step Windows / VS Code setup, how to run training, evaluation, plots and the viewer, and troubleshooting.

## References

- Gurram, M., Uttam, P. K., & Ohol, S. S. (2024). *Reinforcement learning for quadrupedal locomotion:
  Current advancements and future perspectives* (arXiv:2410.10438). https://doi.org/10.48550/arXiv.2410.10438
- Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Lee, J., Tsounis, V., Koltun, V., & Hutter, M. (2019).
  Learning agile and dynamic motor skills for legged robots. *Science Robotics, 4*(26), eaau5872.
  https://doi.org/10.1126/scirobotics.aau5872
- MIMUW Robotics. (2026). *Kine2Go: Kinematic dataset for the Unitree Go2 robot* [Data set]. Hugging Face.
  https://huggingface.co/datasets/MIMUW-Robotics/kine2go
- Unitree Robotics. (n.d.). *unitree_sdk2_python* [Computer software]. GitHub.
  https://github.com/unitreerobotics/unitree_sdk2_python
