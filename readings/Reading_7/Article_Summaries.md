# Reading 7: What Each Article Does and What I Learned

**ENG 6010: Artificial Intelligence in Robotics**

This reading set contains four papers. Read in order, they move step by step from language down to robot motors. Each one answers the same question in a different way: **large language models (LLMs) are good at understanding what we want, but they cannot directly output the precise, high-frequency motor commands a robot needs. So what should sit in between?**

| # | Article | The bridge between language and action | What turns that bridge into motion |
|---|---|---|---|
| 1 | Gorilla | an API call | the tool or library being called |
| 2 | VoxPoser | a 3D value map of where to go and what to avoid | a motion planner run as MPC |
| 3 | Language to Rewards | a reward function | MuJoCo MPC, an optimiser |
| 4 | Kine2Go | motion data (states plus motor actions) | RL policies trained by imitation |

For each article I also built a small runnable demo of its core idea (in `demos/`) and an interactive slide that runs it live (`presentation/`). The demos are simplified and run on a laptop CPU without calling any LLM. Wherever the paper's LLM would produce output, the demo uses a hand-written example in the paper's format. Numbers from my demos come from `results/summary.json` (`python evaluate.py`); numbers from the papers are labelled as such.

---

## 1. Gorilla: Large Language Model Connected with Massive APIs

*Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez (UC Berkeley and Microsoft Research), 2023*

### What the article does

The authors want LLMs to use tools through API calls: for example, finding the right machine-learning model and writing the one line of code that loads it. Even GPT-4 is unreliable at this. It **hallucinates**, inventing models or repositories that do not exist, and it often picks the wrong library.

They make three contributions:

1. **APIBench**, a benchmark of 1,645 real API calls scraped from TorchHub (94), TensorFlow Hub (626) and HuggingFace (925). Each API is stored in a JSON format (domain, API call, arguments, description, performance, and so on). They used GPT-4 with "self-instruct" to write 10 realistic user requests per API, giving 16,450 instruction–API pairs.
2. **Gorilla**, a LLaMA-7B model fine-tuned on those pairs. It can run *zero-shot*, from the prompt alone, or with a **retriever** (BM25 or GPT-Index) that pastes the most relevant, up-to-date API documentation into the prompt. They also train it *retriever-aware*, with retrieved documentation included in the training prompts, so it learns to read the answer from that documentation.
3. **An evaluation metric based on code structure.** Generated code is parsed into an abstract syntax tree (AST) and checked against the database using the arguments that identify an API, e.g. `repo_or_dir` and `model` for `torch.hub.load`. A call that matches no API at all is a **hallucination**. A call to a real API that is the wrong one is an **error**.

### Key results (from the paper)

- **Zero-shot overall accuracy** (TorchHub / HuggingFace / TensorFlow Hub): Gorilla scored 59.1% / 71.7% / 83.8%, against GPT-4's 38.7% / 19.8% / 18.2%. Gorilla's hallucination rate was 7.0% / 11.0% / 5.4%, while GPT-4's was 36.6% / 37.2% / 78.7%.
- **A poor retriever can hurt.** With BM25 at test time, Gorilla's accuracy fell to 40.3% / 17.0% / 41.9%, because the wrong document misleads the model. With a perfect ("Oracle") retriever it reached 67.2% / 91.3% / 94.2%.
- **Retriever-aware training lets Gorilla follow documentation changes at test time.** For example, when a model's backbone was upgraded from ResNet-50 to ResNet-101, or when a model moved to a different repository, Gorilla output the updated call.
- **It can also handle constraints** such as "under 10M parameters and at least 70% ImageNet accuracy": with retrieval it matched the best model (GPT-3.5), and it had the highest constraint accuracy in the zero-shot setting.
- **A side observation:** GPT-3.5 hallucinated less than GPT-4. The authors suggest RLHF tuning may make a model more truthful.

### What I learned

- **Hallucination can be measured, not just complained about.** Defining it as "a call that matches nothing in the database" turns a vague problem into a number you can track and reduce. Parsing code into a tree and comparing the parts that matter, while ignoring optional arguments like `pretrained=True`, is a simple but powerful idea.
- **Retrieval is not automatically better.** I expected that giving a model more documentation would always help, but a weak retriever made Gorilla worse. My own demo shows why. My BM25 retriever ranked the right document first for 5 of my 6 example prompts, but for "less than 10M parameters and at least 70% accuracy" it picked DenseNet-121 instead of MobileNet V2. BM25 matches words, not numbers, so it cannot reason about constraints.
- **Training a model to *read* documentation is more robust than training it to *memorise* answers.** In my documentation-update test, a stand-in for a model fine-tuned without retrieval kept outputting the old `fcn_resnet50` call. Once the docs changed, that call matched nothing, so the AST check marked it as a hallucination. The retriever-aware stand-in read the updated doc and was correct.
- **For robotics,** this paper is the top layer of the stack: the LLM's job is to pick the right tool and call it correctly, and the tool does the real work.

---

## 2. VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models

*Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei (Stanford University and UIUC), CoRL 2023*

### What the article does

Most earlier LLM-for-robotics systems make the LLM call a fixed list of hand-designed skills ("motion primitives") such as `move_to(pose)` or `open_gripper()`. Those skills limit what the robot can do. VoxPoser avoids them.

- **The LLM writes Python code instead of skills.** Given an instruction such as *"Open the top drawer, and watch out for that vase!"*, GPT-4 writes code that:
  1. calls perception APIs: a vision-language pipeline using OWL-ViT to detect objects, Segment Anything to mask them, and XMem to track them;
  2. uses NumPy to fill **3D voxel maps** of the scene (100 × 100 × 100);
  3. puts high values where the robot should go (an **affordance map**, e.g. the drawer handle) and around what it should avoid (an **avoidance map**, e.g. the vase). Extra maps control rotation, gripper and velocity.
- **The maps are combined into a cost map.** A motion planner then searches it greedily for an end-effector path.
- **MPC executes the plan:** the robot executes only the first waypoint and replans at 5 Hz from the newest camera view. Nothing is trained, so it works zero-shot on new instructions and objects.
- **Optional dynamics learning:** for contact-heavy tasks such as opening a door, the zero-shot trajectory can serve as a starting point for learning a dynamics model.

### Key results (from the paper)

- **Real robot**, five everyday tasks: 88% success with a static scene and 70% with disturbances, against 24% and 0% for an LLM using primitives (a Code-as-Policies variant).
- **Simulation**, 13 tasks and 2,766 instructions: it generalised best to unseen instructions and unseen object attributes, e.g. 65% vs 17.5% (primitives) vs 0% (learned U-Net cost maps) on unseen object-interaction tasks.
- **Dynamics learning with the VoxPoser prior** (door / window / fridge):

  | | Door | Window | Fridge |
  |---|---|---|---|
  | Zero-shot | 6.7% | 3.3% | 18.3% |
  | Learned with the zero-shot trajectory as a prior | 88.3% | 80.0% | 91.7% |

  Learning with the prior took under 3 minutes of interaction; exploring without it went past the 12-hour limit.
- **Most real-world errors came from perception,** e.g. detecting object parts, not from the LLM.

### What I learned

- **The best interface for an LLM is one it is already good at.** LLMs are good at writing code and at common sense ("handles are grabbed", "stay away from fragile things"). They are bad at numbers like joint angles. VoxPoser asks the LLM for the first two and lets a planner handle the numbers.
- **Value maps are really potential fields:** attract to the goal, repel from obstacles. The paper itself notes this connection to classic path planning. In my 2-D version, adding the avoidance map raised clean arrivals (reaching the target without hitting the vase) from 43% to 93% over 300 random scenes, but greedy search still occasionally grazed the vase (7%). That is the known weakness of potential fields. I also built a "trap": a mug, vase and bowl arranged in a cup shape on the straight line to the handle. Greedy descent walks into the cup and has to scrape its way back out (219 steps), while A\* search over the same map goes around it in 77. VoxPoser gets away with a greedy planner because it replans constantly and its maps are usually simple.
- **Replanning is what makes it robust.** In my demo, when the drawer handle moved halfway through the motion, following the original plan succeeded only 6% of the time, while replanning every step (MPC) succeeded 88% of the time. That matches the paper's "with disturbances" results and is the same plan–act–replan loop as the CEM planner in my Readings 6 project.
- **Even a zero-shot system can speed up learning.** Using its trajectory as a starting point turned an impossible exploration problem (over 12 hours) into a 3-minute one.

---

## 3. Language to Rewards for Robotic Skill Synthesis

*Wenhao Yu, Nimrod Gileadi, Chuyuan Fu, Sean Kirmani, Kuang-Huei Lee, et al. (Google DeepMind), 2023*

### What the article does

This paper takes the next step down the stack. People usually describe the **outcome** they want ("stand up on your back legs"), not the motor torques. So the authors make the LLM write a **reward function** instead of actions or skill calls, and let an optimiser find the actions.

- **Reward Translator** (two GPT-4 stages):
  1. **Motion Descriptor:** expands the user's request into a structured description by filling in a template, e.g. *"torso pitch 90°, torso height 0.65 m, front feet lifted to 0.65 m"*.
  2. **Reward Coder:** turns that description into code that sets the weights and targets of predefined reward terms, e.g. `set_torso_rewards(height=0.65, pitch=np.deg2rad(90))`.
- **The reward has the form** R = −Σ wᵢ · nᵢ(rᵢ(s, a, ψᵢ)): weights wᵢ, residual terms rᵢ that are zero when a goal is met, and parameters ψᵢ such as target heights.
- **Motion Controller:** **MuJoCo MPC** optimises actions for the reward in real time, using iLQG for walking and Predictive Sampling for manipulation.
- **Interactive correction:** because the optimisation is real-time, a user can watch the robot and correct it in plain language. The paper teaches a quadruped to *moonwalk* in four instructions.

### Key results (from the paper)

- **17 tasks** (9 for a quadruped, 8 for a robot arm with a 20-joint hand): the method solved **90%** of tasks, against **50%** for Code as Policies with hand-designed primitive skills.
- **Removing the Motion Descriptor** (Reward Coder only) solved fewer tasks. For example, it often forgot to move the hand to the drawer handle before trying to open the drawer.
- **On a real robot arm,** with an added "sim-to-real" reward term to keep motions slow and steady, it pushed and grasped objects.
- **Limitations:** the templates are written by hand for each robot type, the LLM only adjusts a fixed set of reward terms, and rewards can't change over time.

### What I learned

- **Rewards are a better language for goals than motor commands or skills.** My pendulum version showed this clearly. With the same predictive-sampling optimiser, changing only the reward code solved all four *possible* instructions: swing up and balance, lean 20° and hold, spin clockwise, hang still. A fixed library of two hand-made skills solved only two, because no skill existed for "lean" or "spin". That mirrors the paper's 90% vs 50%.
- **An optimiser respects physics, even when the request doesn't.** I deliberately included an impossible instruction: "hold the pole horizontal". It needs 5 N·m of torque and the motor only gives 2. The optimiser did not fail randomly; it settled at **156.5° from upright**, which is the holdable pose closest to the goal (theory: 156.4°). This showed me why grounding matters: an LLM can ask for anything, but the optimiser only does what the physics allows.
- **Sampling-based MPC has a horizon sweet spot.** With 256 samples, my swing-up worked 100% of the time for horizons of 10–30 steps, but only 70% at 40 steps and 0% at 60 steps, because random sampling cannot search such a large space of plans. This connects to the 15-step imagination horizon I used in Readings 6.
- **Interactivity is what makes it practical.** Because the motion updates instantly, a person can steer the robot step by step, as in the moonwalk example and the live weight and target editing on my demo slide.

---

## 4. Kine2Go: Kinematic Dataset for the Unitree Go2 Robot with Diverse Gaits and Motions

*Władysław Pałucki, Paweł Siwak, Krzysztof Ciebiera, Marek Cygan (University of Warsaw), 2026*

### What the article does

At the bottom of the stack, a controller has to produce motor commands. RL can learn those, but without good reference data it tends to produce jerky, unnatural gaits, and hand-tuning rewards is tedious and easily "hacked". Humanoids have large motion datasets (AMASS, MoCapAct); quadrupeds did not. Kine2Go builds one for the Unitree Go2 in three stages:

1. **Kinematic retargeting:**
   - Motion capture from other bodies is mapped onto the Go2. The sources are a dog (AI4Animation), horses (VHDC), the Solo8 robot, and a wolf animation steered by hand.
   - Foot positions become targets, and **inverse kinematics** finds Go2 joint angles that reach them within the robot's workspace.
   - The result has no motor commands yet: it is "stop-motion frames".
2. **RL motion imitation:**
   - One PPO policy is trained per motion, in the style of Peng et al. (2020), in the Genesis GPU simulator: 8,192 parallel environments, about 2 billion steps and about 3 hours per motion.
   - The reward is a weighted sum of tracking terms (exponentials of squared errors): joint angles (weight 0.5), joint velocities (0.05), feet positions (0.2), root pose (0.15) and root velocity (0.1).
   - The policy sees its own state plus future reference frames (t+1, t+2, t+10, t+30).
3. **Rollouts and filtering:** every policy is run 20 times with random headings. States *and* motor actions are recorded, and falls and bad tracking are removed after watching the videos.

### Key results (from the paper)

- **800 trajectories from 40 policies:**

  | Source | Motions |
  |---|---|
  | AI4Animation dog | 15 |
  | Interactive wolf | 7 |
  | VHDC horses | 12 |
  | Solo8 robot | 6 |

  They cover walking, trotting, pacing, cantering, turning, strafing, crawling and path-following (circles, squares, figure-eights).
- **Each frame stores:** joint positions and velocities, base orientation and angular velocity, the policy's motor actions, and world-frame joint positions and velocities.
- **Intended use:** behaviour cloning, offline control, and especially **behavioural foundation models** (e.g. Meta Motivo, BFM-Zero). For evaluating those, the authors suggest Earth Mover's Distance rather than plain mean-squared error, because such models should copy a motion's *style*, not every frame.
- **Limitations:** almost no jumps (the one jump included is an accidental behaviour), no sitting, flat terrain only, and no foot-pressure data.

### What I learned

- **Motion capture alone is not training data for a robot.** It shows *what* the motion looks like but not *which motor commands produce it*. The RL imitation stage exists to fill in the actions, which turns animation data into something a controller can learn from.
- **Retargeting is not optional.** In my demo, copying a horse's foot positions directly onto the Go2 made **100%** of the targets unreachable, because a horse's feet are about 1.3 m below its hips and a Go2 leg is only about 0.43 m. For the dog it was 52–80%. Scaling first and then solving the inverse kinematics made **0%** unreachable. The Solo8 is already about Go2-sized, so it fit either way.
- **Reward design encodes priorities.** Implementing the paper's reward showed me that joint angles carry half the weight, while the feet term is the sharpest (scale 40). In my noisy-tracker test, the reward fell from 100% of its maximum to 84% at 5° of joint error and to 43% at 20°. A policy is pushed hard to keep its feet in the right places.
- **Diversity can come cheaply.** One policy run with 20 random headings already covers a wide area (like the paper's Figure 4b), which is how 40 motions become 800 trajectories.
- **This is the only paper of the four that trains with RL.** It shows that RL is still the tool of choice for the lowest, motor-level layer, as long as good reference data guides it.

---

## What I learned overall

1. **The key link is the bridge from words to motors, not one algorithm.** Each paper chooses a different interface (a tool call, a value map, a reward, a dataset) that turns high-level intent into something a reliable, grounded system can execute.
2. **Hand-designed skills are the common bottleneck.** VoxPoser (88% vs 24%) and Language to Rewards (90% vs 50%) both beat "LLM + primitives". My pendulum demo showed the same pattern (4 of 4 possible tasks vs 2 of 4).
3. **Every bridge must be grounded in reality:**
   - Gorilla grounds the LLM in real documentation,
   - VoxPoser in camera perception,
   - Language to Rewards in physics through the optimiser,
   - Kine2Go in the Go2's real joint limits and dynamics.
4. **Code is the natural interface for LLMs.** Three of the papers have the LLM write code, because that is something LLMs do well.
5. **MPC and RL do the low-level work.** VoxPoser and Language to Rewards use model-predictive control (the same plan–act–replan loop as my Readings 6 project), while Kine2Go uses RL on reference data. RL is one tool in the stack, not the thread that connects all four papers.

## References

- S. G. Patil, T. Zhang, X. Wang, J. E. Gonzalez. *Gorilla: Large Language Model Connected with Massive APIs.* arXiv:2305.15334, 2023.
- W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, L. Fei-Fei. *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models.* CoRL 2023, arXiv:2307.05973.
- W. Yu, N. Gileadi, C. Fu, S. Kirmani, K.-H. Lee, et al. *Language to Rewards for Robotic Skill Synthesis.* arXiv:2306.08647, 2023.
- W. Pałucki, P. Siwak, K. Ciebiera, M. Cygan. *Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions.* arXiv:2606.14433, 2026.
