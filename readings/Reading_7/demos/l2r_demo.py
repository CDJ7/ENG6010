"""
l2r_demo.py - Article 3: Language to Rewards for Robotic Skill Synthesis
(Yu et al., Google DeepMind, 2023).

The paper's pipeline:
    user instruction
      -> Motion Descriptor (LLM fills in a motion template)
      -> Reward Coder      (LLM writes code that sets reward terms)
      -> Motion Controller (MuJoCo MPC optimises actions for that reward)

with a reward of the form (paper Eq. 1)
    R(s, a) = - sum_i  w_i * n_i( r_i(s, a, psi_i) )
w_i = weight, r_i = residual (0 when the goal is met), psi_i = its parameters
(e.g. a target height), n = a norm (here the squared l2 norm).

This file reimplements the bottom two stages on the Gymnasium Pendulum-v1
physics (so it connects to Readings 6):
    RewardSpec / reward API   set_angle_reward(), set_speed_reward(), ...
    predictive_sampling()     MJPC's zeroth-order planner (Howell et al., 2022):
                              sample noisy versions of the current plan,
                              roll them out in the physics model, keep the best,
                              execute the first action, repeat.
    PRIMITIVES / CAP_PROGRAMS the Code-as-Policies baseline: the LLM may only
                              call a fixed library of hand-made skills.

The Motion Descriptor and Reward Coder outputs in INSTRUCTIONS are
hand-written examples in the paper's format (no LLM is called).
"""

import math
from dataclasses import dataclass, field

import numpy as np

G, M, L, MAX_SPEED, MAX_TORQUE, DT = 10.0, 1.0, 1.0, 8.0, 2.0, 0.05


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def step(th, thdot, u):
    """Exact Pendulum-v1 dynamics, vectorised (theta = 0 is upright,
    positive theta = pole tipped to the right)."""
    u = np.clip(u, -MAX_TORQUE, MAX_TORQUE)
    thdot = np.clip(thdot + (3 * G / (2 * L) * np.sin(th) + 3.0 / (M * L ** 2) * u) * DT, -MAX_SPEED, MAX_SPEED)
    return th + thdot * DT, thdot


# ---------------------------------------------------------------------------
# The reward interface the "Reward Coder" writes against
# ---------------------------------------------------------------------------
@dataclass
class RewardTerm:
    name: str
    target: float      # psi: the parameter the LLM chooses
    weight: float      # w: 0 disables the term
    unit: str


@dataclass
class RewardSpec:
    terms: dict = field(default_factory=dict)

    def reset_reward(self):
        self.terms = {"angle": RewardTerm("angle", 0.0, 0.0, "deg"),
                      "speed": RewardTerm("speed", 0.0, 0.0, "deg/s"),
                      "effort": RewardTerm("effort", 0.0, 0.0, "N*m")}

    # -- the API the LLM is told about (paper Fig. 2 "Reward API") -----------
    def set_angle_reward(self, target_deg, weight=1.0):
        """Pole angle target: 0 = straight up, positive = tipped right."""
        self.terms["angle"] = RewardTerm("angle", target_deg, weight, "deg")

    def set_speed_reward(self, target_deg_s, weight=0.1):
        """Angular speed target: positive = clockwise."""
        self.terms["speed"] = RewardTerm("speed", target_deg_s, weight, "deg/s")

    def set_effort_reward(self, weight=0.01):
        """Penalise motor torque (target is always 0)."""
        self.terms["effort"] = RewardTerm("effort", 0.0, weight, "N*m")

    def residuals(self, th, thdot, u):
        """r_i for each term (0 when that part of the goal is met)."""
        t = self.terms
        return {"angle": wrap(th - math.radians(t["angle"].target)),
                "speed": thdot - math.radians(t["speed"].target),
                "effort": u}

    def reward(self, th, thdot, u):
        """Paper Eq. 1 with n = squared l2 norm."""
        r = self.residuals(th, thdot, u)
        return -sum(self.terms[k].weight * r[k] ** 2 for k in r)


def reward_landscape(spec, n_theta=90, n_speed=60):
    """The reward of every (angle, speed) state with zero torque: the 'shape'
    the optimiser climbs. Angle in [-180, 180) deg, speed in [-8, 8] rad/s."""
    th = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    w = np.linspace(-MAX_SPEED, MAX_SPEED, n_speed)
    T, Wd = np.meshgrid(th, w)
    return spec.reward(T, Wd, np.zeros_like(T)), th, w


# The Motion Descriptor template, in the paper's style ([NUM: default] and
# {CHOICE: ...} slots that the LLM must fill in), adapted to the pendulum.
TEMPLATE = ["[start of description]",
            "The pole should point at [NUM: 0.0] degrees (0 = up, positive = right).",
            "The pole should {CHOICE: hold still, spin}.",
            "If spinning, it should spin at [NUM: 0.0] degrees/s (positive = clockwise).",
            "[optional] Use as little motor effort as possible.",
            "[end of description]"]

# Plain-language corrections (paper Sec. 4.5): each one edits the reward.
CORRECTIONS = [
    ("A bit further to the right.", "angle", "target", +10),
    ("Hold it more steadily.", "speed", "weight", +0.3),
    ("Use less energy.", "effort", "weight", +0.05),
    ("Spin faster.", "speed", "target", +120),
    ("Actually, lean the other way.", "angle", "flip", 0),
]


def apply_correction(spec, corr):
    """Apply one correction to the reward; return the changed line of code."""
    _, term, field, delta = corr
    t = spec.terms[term]
    if field == "flip":
        t.target = -t.target
    elif field == "target":
        t.target += delta
    else:
        t.weight = round(t.weight + delta, 3)
    if term == "angle":
        return f"set_angle_reward({t.target:g}, weight={t.weight:g})"
    if term == "speed":
        return f"set_speed_reward({t.target:g}, weight={t.weight:g})"
    return f"set_effort_reward(weight={t.weight:g})"


def run_reward_code(lines):
    """Execute Reward Coder output against the reward API (like the paper does)."""
    spec = RewardSpec()
    spec.reset_reward()
    env = {"np": np, "reset_reward": spec.reset_reward, "set_angle_reward": spec.set_angle_reward,
           "set_speed_reward": spec.set_speed_reward, "set_effort_reward": spec.set_effort_reward,
           "execute_plan": lambda seconds=10: None}
    exec("\n".join(lines), env)
    return spec


# ---------------------------------------------------------------------------
# Motion Controller: Predictive Sampling MPC (as used by MJPC)
# ---------------------------------------------------------------------------
class PredictiveSampling:
    """Keep a nominal action plan over the horizon. Each control step: sample
    noisy versions of it, roll every one out in the physics model, score it
    with the reward, keep the best (the nominal is a candidate too, so the plan
    never gets worse), execute its first action and shift it by one step."""

    def __init__(self, spec, horizon=30, samples=256, noise=1.0, seed=0, record=False):
        self.spec, self.H, self.N, self.noise = spec, horizon, samples, noise
        self.rng = np.random.default_rng(seed)
        self.plan = np.zeros(horizon)
        self.record = record            # keep every sampled rollout (for drawing)
        self.rollouts = None

    def act(self, th, thdot):
        knots = 6
        noise = self.rng.normal(0, self.noise, (self.N - 1, knots))
        # smooth noise: interpolate a few knots over the horizon
        xs = np.linspace(0, 1, knots)
        t = np.linspace(0, 1, self.H)
        smooth = np.stack([np.interp(t, xs, n) for n in noise])
        cand = np.clip(np.vstack([self.plan, self.plan + smooth]), -MAX_TORQUE, MAX_TORQUE)
        a, b = np.full(self.N, th), np.full(self.N, thdot)
        total = np.zeros(self.N)
        traj = [(a, b)] if self.record else None
        for k in range(self.H):
            total += self.spec.reward(a, b, cand[:, k])
            a, b = step(a, b, cand[:, k])
            if self.record:
                traj.append((a, b))
        i_best = int(np.argmax(total))
        if self.record:
            self.rollouts = dict(theta=np.stack([t[0] for t in traj], 1), thdot=np.stack([t[1] for t in traj], 1),
                                 score=total, best=i_best)
        best = cand[i_best]
        self.plan = np.append(best[1:], best[-1])
        return float(best[0])


# ---------------------------------------------------------------------------
# Code-as-Policies baseline: a fixed library of hand-made skills
# ---------------------------------------------------------------------------
def swing_up_and_balance(th, thdot):
    """Energy pumping until near the top, then a PD balance controller."""
    th = wrap(th)
    if math.cos(th) > 0.9:
        return float(np.clip(-(8.0 * th + 2.0 * thdot), -2, 2))
    energy = 0.5 * thdot ** 2 + 15.0 * math.cos(th)      # 15 = 3g/2l
    return float(np.clip(0.5 * (15.0 - energy) * thdot, -2, 2))


def stop_swinging(th, thdot):
    """Damp the motion so the pole comes to rest hanging down."""
    return float(np.clip(-2.0 * thdot, -2, 2))


PRIMITIVES = {"swing_up_and_balance": swing_up_and_balance, "stop_swinging": stop_swinging}


# ---------------------------------------------------------------------------
# Instructions with example LLM outputs (paper format)
# ---------------------------------------------------------------------------
def describe(angle=None, spin=None, effort=False):
    lines = ["[start of description]"]
    if angle is not None:
        lines.append(f"The pole should point at {angle:.0f} degrees (0 = straight up, positive = right).")
    lines.append("The pole should hold still." if spin is None
                 else f"The pole should spin at {spin:.0f} degrees/s (positive = clockwise).")
    if effort:
        lines.append("Use as little motor effort as possible.")
    lines.append("[end of description]")
    return lines


INSTRUCTIONS = [
    dict(text="Swing the pole up and balance it.",
         description=describe(angle=0),
         reward_code=["reset_reward()", "set_angle_reward(0, weight=1.0)",
                      "set_speed_reward(0, weight=0.1)", "set_effort_reward(weight=0.001)", "execute_plan(10)"],
         cap=["swing_up_and_balance()"], cap_fn="swing_up_and_balance",
         check=lambda th, w: (np.abs(np.degrees(wrap(th))) < 10).mean() > 0.9 and np.abs(w).mean() < 1.0),
    dict(text="Lean the pole 20 degrees to the right and hold it there.",
         description=describe(angle=20),
         reward_code=["reset_reward()", "set_angle_reward(20, weight=1.0)",
                      "set_speed_reward(0, weight=0.1)", "set_effort_reward(weight=0.001)", "execute_plan(10)"],
         cap=["# no primitive can hold a tilted pose;", "# closest available skill:", "swing_up_and_balance()"],
         cap_fn="swing_up_and_balance",
         check=lambda th, w: (np.abs(np.degrees(wrap(th)) - 20) < 7).mean() > 0.9),
    dict(text="Spin it clockwise, fast.",
         description=describe(spin=340),
         reward_code=["reset_reward()", "set_speed_reward(340, weight=1.0)",
                      "set_effort_reward(weight=0.0)", "execute_plan(10)"],
         cap=["# no 'spin' primitive in the library;", "# closest available skill:", "swing_up_and_balance()"],
         cap_fn="swing_up_and_balance",
         check=lambda th, w: w.mean() > 4.0),
    dict(text="Let it hang down and stay still.",
         description=describe(angle=180),
         reward_code=["reset_reward()", "set_angle_reward(180, weight=1.0)",
                      "set_speed_reward(0, weight=0.3)", "set_effort_reward(weight=0.001)", "execute_plan(10)"],
         cap=["stop_swinging()"], cap_fn="stop_swinging",
         check=lambda th, w: (np.abs(np.degrees(wrap(th - np.pi))) < 10).mean() > 0.9 and np.abs(w).mean() < 0.5),
    dict(text="Hold the pole horizontal, pointing right.",
         description=describe(angle=90),
         reward_code=["reset_reward()", "set_angle_reward(90, weight=1.0)",
                      "set_speed_reward(0, weight=0.1)", "set_effort_reward(weight=0.001)", "execute_plan(10)"],
         cap=["# no primitive can hold a tilted pose;", "# closest available skill:", "swing_up_and_balance()"],
         cap_fn="swing_up_and_balance",
         check=lambda th, w: (np.abs(np.degrees(wrap(th)) - 90) < 10).mean() > 0.9,
         note="Physically impossible: holding 90 deg needs 5 N*m but the motor gives 2 N*m. "
              "The largest lean that can be held is asin(6/15) = 23.6 deg."),
]


def rollout(instr, method, th0, thdot0, seconds=10.0, seed=0, spec=None, horizon=30):
    """Run one instruction with 'reward' (LLM reward + MPC) or 'cap' (primitives)."""
    spec = spec or run_reward_code(instr["reward_code"])
    ctrl = PredictiveSampling(spec, horizon=horizon, seed=seed) if method == "reward" else None
    th, w = th0, thdot0
    ths, ws = [], []
    for _ in range(int(seconds / DT)):
        u = ctrl.act(th, w) if ctrl else PRIMITIVES[instr["cap_fn"]](th, w)
        th, w = step(th, w, u)
        th, w = float(th), float(w)
        ths.append(th)
        ws.append(w)
    return np.array(ths), np.array(ws)


def evaluate(n_starts=10, seed=0):
    """Success of each instruction from random start states, judged on the
    last 3 seconds of a 10-second run."""
    rng = np.random.default_rng(seed)
    starts = [(rng.uniform(-np.pi, np.pi), rng.uniform(-1, 1)) for _ in range(n_starts)]
    out = []
    for i, instr in enumerate(INSTRUCTIONS):
        row = {"instruction": instr["text"]}
        for method in ("reward", "cap"):
            ok = 0
            for j, (th0, w0) in enumerate(starts):
                th, w = rollout(instr, method, th0, w0, seed=100 * i + j)
                ok += bool(instr["check"](th[-60:], w[-60:]))
            row[method] = ok / n_starts
        if instr["text"].startswith("Hold the pole horizontal"):
            th, w = rollout(instr, "reward", np.pi, 0.0, seed=7)
            row["achieved_lean_deg"] = float(np.degrees(np.abs(wrap(th[-60:]))).mean())
        out.append(row)
    return out


def horizon_sweep(horizons=(10, 20, 30, 40, 60), n_starts=10, seed=0):
    """Swing-up success vs planning horizon (256 samples per decision)."""
    rng = np.random.default_rng(seed)
    starts = [(rng.uniform(-np.pi, np.pi), rng.uniform(-1, 1)) for _ in range(n_starts)]
    instr = INSTRUCTIONS[0]
    res = {}
    for h in horizons:
        ok = 0
        for j, (th0, w0) in enumerate(starts):
            th, w = rollout(instr, "reward", th0, w0, seed=j, horizon=h)
            ok += bool(instr["check"](th[-60:], w[-60:]))
        res[h] = ok / n_starts
    return res


if __name__ == "__main__":
    import json
    import time
    t = time.time()
    print(json.dumps(evaluate(), indent=2))
    print("swing-up success vs horizon:", horizon_sweep())
    print(f"{time.time() - t:.1f} s")
