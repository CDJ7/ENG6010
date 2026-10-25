# Program Explanation

**Project:** Mode-2 to Mode-1: Planning with a Learned World Model and Distilling It into a Fast Policy
**Course:** ENG 6010, Artificial Intelligence in Robotics, Readings 6

## Goal

All four readings this week revolve around one idea: an agent that has a model of how the world responds to its actions can *think before it acts*, but thinking is expensive. LeCun (2022) calls slow, deliberate, model-based reasoning **Mode-2** and fast, reflexive behaviour **Mode-1**, and he argues (his Figure 5) that the results of Mode-2 planning should be "compiled" into a Mode-1 policy so the agent does not have to plan every time. DreamerV3 (Hafner et al., 2023) and the Diffusion World Model paper (Ding et al., 2024) both learn a world model and use *imagined* rollouts from it, but only during training, so the policy that is finally deployed is a single cheap network.

My goal was to build the smallest complete version of that story I could run on a laptop CPU. The program:

1. learns a world model of a simple robot (a torque-controlled pendulum) from random interaction,
2. uses that model for Mode-2 control: a Cross-Entropy Method (CEM) planner run as Model-Predictive Control,
3. distils the planner into a small Mode-1 policy network with behaviour cloning and DAgger, and
4. measures how much performance the fast policy keeps and how much faster it is.

The whole pipeline runs with `python main.py`, uses fixed random seeds, and repeats everything for three seeds.

## The environment: Gymnasium `Pendulum-v1`

`Pendulum-v1` is a single rigid pole attached to a motor at one end. The pole starts at a random angle θ ∈ [−π, π] with a small random angular velocity θ̇ ∈ [−1, 1], and the task is to swing it up and balance it upright (θ = 0). The motor is too weak to lift the pole directly from the bottom, so the controller has to pump energy in by swinging back and forth first. That makes it a small but genuine planning problem.

- **State/observation:** a 3-vector `[cos θ, sin θ, θ̇]`. Using cos and sin avoids the jump from +π to −π. θ̇ is clipped to [−8, 8] rad/s.
- **Action:** a single continuous torque u ∈ [−2, 2] N·m.
- **Dynamics:** θ̈ = (3g / 2l)·sin θ + (3 / ml²)·u with g = 10, m = l = 1, integrated with a time step of 0.05 s.
- **Reward:** r = −(θ² + 0.1·θ̇² + 0.001·u²), with θ normalised to [−π, π]. The best possible reward is 0 (upright, still, no torque) and the worst is about −16.3 per step.
- **Episode:** 200 steps (10 simulated seconds), so returns range from about −3,250 to 0. Random torques typically score around −1,200, and a good swing-up controller scores roughly −120 to −200 depending on the start state.

## Step 1: Collecting data

The agent starts knowing nothing about the dynamics. `collect_data.py` runs 50 episodes of uniformly random torques (10,000 transitions) and stores each `(state, action, next_state, reward)` tuple in a compressed `.npz` file. I also collect 10 more random episodes from different start states and never train on them; they are held out to test the world model's multi-step predictions. Every controller in the project has the same `reset()` / `act(obs)` interface, so the same `rollout()` function is used for data collection, DAgger and the final evaluation. That function also times every `act()` call, which is how I measure decision time.

## Step 2: Training the world model

The world model (`world_model.py`) is a multilayer perceptron with three hidden layers of 128 SiLU units. It takes the current state and action, `(s, a)`, and predicts the **change** in state, Δs = s′ − s, rather than s′ itself. Over one 0.05 s step the state barely changes, so predicting the difference gives the network a small, zero-centred target that is much easier to fit. The inputs are standardised to zero mean and unit variance, and so are the targets. The normalisation statistics are stored inside the model as buffers, so the planner can call `model.predict(s, a)` in raw units.

Training uses Adam (learning rate 10⁻³, small weight decay) on a mean-squared-error loss for 60 epochs, with 10% of the transitions held out as a validation set. Both losses are recorded every epoch and plotted.

When the model is rolled forward on its own predictions, I apply two small pieces of prior knowledge after each step: (cos θ, sin θ) is re-projected onto the unit circle, and θ̇ is clipped to the simulator's ±8 limit. Neither changes the predicted angle; they only keep each prediction a valid pendulum state.

I also implemented DreamerV3's **symlog** transform as an option (`--symlog`, and a side-by-side comparison on seed 0). symlog(x) = sign(x)·ln(1 + |x|) compresses large targets while staying close to the identity near zero, and symexp is its exact inverse. With the flag on, the model regresses symlog(Δs) and predictions are mapped back with symexp. This mirrors `embodied/jax/nets.py` in the DreamerV3 repository. In DreamerV3 symlog matters because one set of hyperparameters has to handle rewards and observations of wildly different scales across domains.

### The cost function

LeCun's architecture has a separate **intrinsic cost** module: a fixed, hard-wired function that rates how bad a state is and is not learned. In my program this is `cost.py`, which contains the Pendulum reward formula with its sign flipped:

  cost(s, u) = θ² + 0.1·θ̇² + 0.001·u², with θ = atan2(sin θ, cos θ).

The key point is that the planner evaluates this function on states **predicted by the learned world model**, never on states from the simulator. The planner only "knows" the task through the cost and only "knows" the physics through the learned model.

## Step 3: Evaluating the world model and its compounding error

A one-step error of a fraction of a percent sounds negligible. The Diffusion World Model paper points out (their Figure 1.1) that the error grows with the horizon when a one-step model is fed its own outputs, because each prediction starts from a slightly wrong state. To measure this, `multi_step_error()` takes every start time in the held-out trajectories, gives the model the true start state and the recorded action sequence, and rolls it forward for 1 to 30 steps with no corrections from reality. At each horizon I record the RMSE over the three state dimensions and the mean absolute angle error in degrees. This plot tells me how far ahead the model can be trusted, which is why the planning horizon is short (15 steps, the same as DreamerV3's `imag_length: 15`).

## Step 4: Mode-2, planning with the Cross-Entropy Method (MPC)

The Mode-2 controller is in `planner.py`. At every real time step it solves a small optimisation problem inside the world model: find the 15-step torque sequence (a₀, …, a₁₄) that minimises the total imagined cost

  J = Σₜ cost(ŝₜ, aₜ) + cost(ŝ₁₅, 0), where ŝ₀ = current real state and ŝₜ₊₁ = model(ŝₜ, aₜ).

The Cross-Entropy Method searches for that sequence by keeping a Gaussian distribution over action sequences (one mean and one standard deviation per time step):

1. Sample 300 action sequences from the Gaussian and clip them to [−2, 2].
2. Roll all 300 forward through the world model as one batch and compute their imagined cost J.
3. Keep the 30 cheapest sequences (the "elites") and refit the mean and standard deviation to them, with a little smoothing.
4. Repeat for 4 iterations, then return the first element of the final mean.

Only that **first action** is sent to the real pendulum. At the next step the whole optimisation runs again from the new real state. Executing one action and then replanning is what makes this Model-Predictive Control, and it is exactly LeCun's description of Mode-2: the agent imagines the consequences of an action sequence, optimises it against the cost, executes the first action and starts over. Replanning also limits the damage from compounding error, because the model never has to be right more than 15 steps ahead, and every step starts again from a real observation. To save work, the planner is warm-started with the previous plan shifted by one step.

As a baseline I implemented **random shooting**: sample 1,200 uniformly random sequences (the same 300 × 4 budget as CEM), pick the cheapest, and execute its first action. This shows how much CEM's iterative refinement adds over blind sampling.

The cost of Mode-2 is obvious from the numbers: each CEM decision requires 4 × 300 × 16 = 19,200 imagined model steps (batched into 64 network evaluations).

I also implemented an optional **model-based RL loop**: run CEM, add the transitions it visits to the dataset, retrain the model, and repeat. It is the data-collection strategy DreamerV3 uses. In my first full run, two rounds of this loop left CEM's return unchanged (−198.1 → −198.0 → −198.2 on seed 0). For a pendulum, random data already covers the state space well, so I turned the loop off by default (`--mbrl-rounds 2` turns it back on). The numbers from that run are kept in `results/ablation_mbrl/`.

## Step 5: Mode-1, distilling the planner with behaviour cloning and DAgger

The Mode-1 policy (`policy.py`) is a small MLP with two hidden layers of 64 tanh units. It maps the 3-dimensional observation directly to a torque, with a final tanh scaled to [−2, 2]. It contains no world model and does no search: one forward pass per decision. This is the "compilation" LeCun describes in his Figure 5, where Mode-2 serves as the teacher for the Mode-1 actor.

The policy is trained by supervised regression onto the planner's actions, in two stages:

**Iteration 0, behaviour cloning (BC).** I run the CEM planner for 3 episodes, record every (state, planner action) pair, and fit the policy with a mean-squared-error loss. The weakness of plain BC is that the policy only sees states the expert visits. The expert almost never makes mistakes, so the data contains almost no examples of how to recover from one. Once the student makes a small error it drifts into unfamiliar states, makes bigger errors there, and the trajectory falls apart. This is the same compounding-error problem as in the world model, now in the policy.

**Iterations 1 to 4, DAgger (Dataset Aggregation).** In each iteration I let the *current policy* drive the pendulum for 3 episodes. For every state the policy visits, I ask the CEM planner which action it would have taken there, add those (state, planner action) pairs to the dataset, and retrain the policy on everything collected so far. The training data now covers the states the policy actually reaches, mistakes included, so the policy learns how to recover from them. After each iteration I evaluate the policy on 5 fixed episodes so the learning curve can be plotted.

This is the important link to the Diffusion World Model paper's argument. All the expensive imagination, planning and labelling happens *during training*. The deployed policy never touches the world model.

## Step 6: Evaluation

`evaluate.py` runs all four controllers (random, random-shooting MPC, CEM MPC and the distilled policy) on the same 5 unseen start states per seed. These start states are never used for training or DAgger. Over 3 seeds that is 15 episodes per controller. For each controller I report the mean ± standard deviation of the episode return and the mean wall-clock time of one `act()` call in milliseconds. PyTorch is pinned to a single CPU thread, so all controllers are timed under the same conditions. The program also renders one swing-up per controller from the hardest start (hanging straight down, at rest) as GIFs and a filmstrip, and writes every number to `results/summary.json`.

## How the parts map onto the readings

| Program part | Reading | Connection |
|---|---|---|
| `world_model.py` (Δs MLP) | LeCun; DreamerV3 | The agent's predictive model of the world, learned from its own experience |
| `cost.py` | LeCun | The fixed, non-learned *intrinsic cost* that defines the task |
| `planner.py` (CEM MPC) | LeCun (Mode-2) | Imagine, optimise the action sequence, execute the first action, replan |
| Horizon H = 15 | DreamerV3 (paper and `imag_length: 15`) | Short imagination keeps model error under control |
| `multi_step_error()` | Diffusion World Model, Fig. 1.1 | Compounding error of one-step models over long horizons |
| `utils.symlog/symexp`, `--symlog` | DreamerV3 paper and `nets.py` | Robust regression targets |
| `policy.py` (BC + DAgger) | LeCun Fig. 5; Diffusion World Model | Compile Mode-2 into a fast Mode-1 actor; imagination only at training time |
| `config.py` | DreamerV3 repo `configs.yaml` | All hyperparameters in one place |

The main simplifications, compared with the papers, are that my world model is a deterministic MLP on a 3-dimensional true state instead of a latent model learned from camera images (DreamerV3's RSSM or DWM's diffusion model), and that the "policy learning" is imitation of a planner rather than actor-critic training in imagination. The structure (learn a model, plan in it, compile the plans into a reactive policy) is the same.
