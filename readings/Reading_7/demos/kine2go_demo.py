"""
kine2go_demo.py - Article 4: Kine2Go (Pałucki et al., University of Warsaw, 2026).

Kine2Go builds a motion dataset for the Unitree Go2 quadruped in three stages:
  1. Kinematic retargeting - map motion capture from another body (dog,
     horse, the Solo8 robot, an animated wolf) onto the Go2: the foot
     positions become targets and inverse kinematics (IK) finds Go2 joint
     angles that reach them inside the robot's workspace.
  2. RL motion imitation - for every reference motion, train a PPO policy
     whose reward is a weighted sum of tracking terms (paper Eqs. 1-8).
  3. Rollouts + filtering - run each policy 20 times with random headings,
     record states AND motor actions, discard falls and bad tracking.
  Result: 40 policies x 20 rollouts = 800 trajectories.

This file reimplements the parts that can run on a laptop in a 2-D side
(sagittal) view, so each leg is a 2-link thigh + calf (the real Go2 also has
an abduction joint per leg, 12 joints in total; here 8):
    gait_reference()      a source animal's foot trajectories for a gait
    retarget()            scale to the Go2 and solve IK per frame
    ik_2link()/fk_2link() the leg geometry
    imitation_reward()    the paper's reward, Eqs. 1-8, with its weights/scales
    rollout_paths()       random-heading rollouts (paper Fig. 4b style)

Stage 2 (training PPO for ~3 GPU-hours per motion) is not reproduced: a
noisy tracker stands in for a trained policy so the reward can be explored.
Body sizes are rounded illustrative values, not official specifications.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Morphologies (metres, rounded / illustrative)
# ---------------------------------------------------------------------------
GO2 = dict(name="Unitree Go2", thigh=0.213, calf=0.213, body=0.38, stand=0.30)
SOURCES = {
    "dog":   dict(name="Dog (AI4Animation mocap)", thigh=0.25, calf=0.25, body=0.60, stand=0.42),
    "horse": dict(name="Horse (VHDC mocap)", thigh=0.70, calf=0.75, body=1.10, stand=1.30),
    "solo8": dict(name="Solo8 robot", thigh=0.16, calf=0.16, body=0.39, stand=0.24),
}
LEGS = ["FL", "FR", "RL", "RR"]

# Phase offsets (fraction of the gait cycle) and duty factor (time on ground)
GAITS = {
    "walk":   dict(offsets=dict(FL=0.25, FR=0.75, RL=0.0, RR=0.5), duty=0.75, stride=0.55, freq=1.2, lift=0.18),
    "trot":   dict(offsets=dict(FL=0.0, FR=0.5, RL=0.5, RR=0.0), duty=0.5, stride=0.70, freq=1.8, lift=0.22),
    "pace":   dict(offsets=dict(FL=0.0, FR=0.5, RL=0.0, RR=0.5), duty=0.5, stride=0.70, freq=1.8, lift=0.22),
    "canter": dict(offsets=dict(FL=0.0, FR=0.1, RL=0.5, RR=0.6), duty=0.4, stride=0.95, freq=2.2, lift=0.30),
}


# ---------------------------------------------------------------------------
# Leg geometry
# ---------------------------------------------------------------------------
def ik_2link(x, z, l1, l2):
    """Joint angles (hip, knee) that put the foot at (x, z) relative to the
    hip (x forward, z up, so z < 0 below the hip). Knee bends backwards, like
    every Go2 leg. Targets out of reach are pulled onto the workspace edge.
    Returns (hip, knee, was_clipped)."""
    d = math.hypot(x, z)
    reach_max, reach_min = 0.98 * (l1 + l2), abs(l1 - l2) + 0.05
    clipped = d > reach_max or d < reach_min
    if clipped:
        s = np.clip(d, reach_min, reach_max) / max(d, 1e-9)
        x, z, d = x * s, z * s, np.clip(d, reach_min, reach_max)
    cos_k = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    knee = math.acos(np.clip(cos_k, -1, 1))                 # 0 = straight leg
    # angle of the foot direction measured from straight down, forward positive;
    # the thigh is rotated backwards from it so the knee sits behind the leg
    foot_ang = math.atan2(x, -z)
    hip = foot_ang - math.atan2(l2 * math.sin(knee), l1 + l2 * math.cos(knee))
    return hip, knee, clipped


def fk_2link(hip, knee, l1, l2):
    """Knee and foot positions relative to the hip for given joint angles
    (hip: thigh angle from straight down, forward positive; knee: bend)."""
    kx, kz = l1 * math.sin(hip), -l1 * math.cos(hip)
    a = hip + knee
    return (kx, kz), (kx + l2 * math.sin(a), kz - l2 * math.cos(a))


# ---------------------------------------------------------------------------
# Stage 0: a source animal's reference motion
# ---------------------------------------------------------------------------
def foot_offset(phase, gait, src):
    """Foot position relative to its hip for one leg at a gait phase in [0, 1).
    Stance: the foot slides backwards on the ground. Swing: it arcs forward."""
    leg = src["thigh"] + src["calf"]
    stride, lift, duty = gait["stride"] * leg, gait["lift"] * leg, gait["duty"]
    if phase < duty:                                   # stance
        s = phase / duty
        return (0.5 - s) * stride, -src["stand"]
    s = (phase - duty) / (1 - duty)                    # swing
    return (-0.5 + s) * stride, -src["stand"] + lift * math.sin(math.pi * s)


def gait_reference(gait_name, source="dog", frames=60):
    """One cycle of the source animal's motion: per frame, foot offsets of all
    four legs (hip frame) and the body speed."""
    g, src = GAITS[gait_name], SOURCES[source]
    out = []
    for f in range(frames):
        t = f / frames
        out.append({leg: foot_offset((t + g["offsets"][leg]) % 1.0, g, src) for leg in LEGS})
    speed = g["stride"] * (src["thigh"] + src["calf"]) * g["freq"] / g["duty"] * 0.5
    return out, speed


# ---------------------------------------------------------------------------
# Stage 1: retargeting onto the Go2
# ---------------------------------------------------------------------------
def retarget(gait_name, source="dog", frames=60, naive=False):
    """Scale the source foot targets by leg length and solve Go2 IK per frame.
    Returns the Go2 joint trajectory and how many targets had to be clipped
    into the Go2's reachable workspace. naive=True copies the source foot
    positions in metres without scaling, to show why scaling is needed."""
    src = SOURCES[source]
    ref, _ = gait_reference(gait_name, source, frames)
    scale = 1.0 if naive else (GO2["thigh"] + GO2["calf"]) / (src["thigh"] + src["calf"])
    joints, feet, clipped = [], [], 0
    for fr in ref:
        q, fp = {}, {}
        for leg, (x, z) in fr.items():
            if naive:
                tx, tz = x, z
            else:
                tx, tz = x * scale, z * GO2["stand"] / src["stand"]     # match Go2 stand height
            hip, knee, c = ik_2link(tx, tz, GO2["thigh"], GO2["calf"])
            clipped += c
            q[leg] = (hip, knee)
            fp[leg] = fk_2link(hip, knee, GO2["thigh"], GO2["calf"])[1]
        joints.append(q)
        feet.append(fp)
    return dict(joints=joints, feet=feet, ref=ref, scale=scale,
                clipped_fraction=clipped / (len(ref) * 4))


# ---------------------------------------------------------------------------
# Stage 2: the imitation reward (paper Appendix C.1, Eqs. 1-8)
# ---------------------------------------------------------------------------
WEIGHTS = dict(q=0.5, qdot=0.05, ee=0.2, pose=0.15, vel=0.1)


def imitation_reward(sim, ref):
    """Kine2Go's reward. `sim` and `ref` are dicts with:
    q (joint angles), qdot (joint velocities), feet (foot positions relative
    to the root), pos (root position), rot (root pitch, rad),
    vel (root linear velocity), angvel (root angular velocity)."""
    r = {}
    r["q"] = math.exp(-5 * np.sum((sim["q"] - ref["q"]) ** 2))                          # Eq. 3
    r["qdot"] = math.exp(-0.1 * np.sum((sim["qdot"] - ref["qdot"]) ** 2))               # Eq. 4
    r["ee"] = math.exp(-40 * np.sum((sim["feet"] - ref["feet"]) ** 2))                  # Eq. 5
    theta = abs(math.atan2(math.sin(sim["rot"] - ref["rot"]), math.cos(sim["rot"] - ref["rot"])))
    r["pose"] = math.exp(-20 * np.sum((sim["pos"] - ref["pos"]) ** 2)) + math.exp(-10 * theta ** 2)   # Eqs. 6-7
    r["vel"] = (math.exp(-2 * np.sum((sim["vel"] - ref["vel"]) ** 2))
                + math.exp(-0.2 * np.sum((sim["angvel"] - ref["angvel"]) ** 2)))                    # Eq. 8
    total = sum(WEIGHTS[k] * r[k] for k in r)                                           # Eq. 1
    return total, r


# The k in exp(-k * error^2) for every tracking term (paper Eqs. 3-8)
KERNELS = dict(q=5.0, qdot=0.1, ee=40.0, pos=20.0, rot=10.0, vel=2.0, angvel=0.2)


def workspace(l1, l2):
    """Inner and outer radius of the foot's reachable ring around the hip
    (same limits that ik_2link uses)."""
    return abs(l1 - l2) + 0.05, 0.98 * (l1 + l2)


MAX_REWARD = WEIGHTS["q"] + WEIGHTS["qdot"] + WEIGHTS["ee"] + 2 * WEIGHTS["pose"] + 2 * WEIGHTS["vel"]

# Episode termination (paper Table 3)
TERMINATE_DISTANCE_M = 1.0
TERMINATE_ROTATION_RAD = math.pi / 2


def state_from_joints(joints_t, joints_prev, root_x, pitch, dt):
    q = np.array([a for leg in LEGS for a in joints_t[leg]])
    qp = np.array([a for leg in LEGS for a in joints_prev[leg]])
    feet = np.array([fk_2link(*joints_t[leg], GO2["thigh"], GO2["calf"])[1] for leg in LEGS])
    return dict(q=q, qdot=(q - qp) / dt, feet=feet, pos=np.array([root_x, GO2["stand"]]),
                rot=pitch, vel=np.array([0.0, 0.0]), angvel=np.array([0.0]))


def tracker_rewards(gait="trot", source="dog", noise_deg=(0, 2, 5, 10, 20), seed=0, cycles=3):
    """Average imitation reward of a noisy tracker (stand-in for a trained
    PPO policy) at different joint tracking errors."""
    rng = np.random.default_rng(seed)
    rt = retarget(gait, source)
    J = rt["joints"]
    dt = 1.0 / (GAITS[gait]["freq"] * len(J))
    out = {}
    for nd in noise_deg:
        tot = []
        prev_s = None
        for c in range(cycles):
            for i in range(len(J)):
                ref = state_from_joints(J[i], J[i - 1], 0.0, 0.0, dt)
                noisy = {leg: (J[i][leg][0] + math.radians(rng.normal(0, nd)),
                               J[i][leg][1] + math.radians(rng.normal(0, nd))) for leg in LEGS}
                sim = state_from_joints(noisy, prev_s or J[i - 1], 0.0, math.radians(rng.normal(0, nd / 4)), dt)
                prev_s = noisy
                tot.append(imitation_reward(sim, ref)[0])
        out[nd] = float(np.mean(tot)) / MAX_REWARD
    return out


# ---------------------------------------------------------------------------
# Stage 3: rollouts with random headings (paper Fig. 4b)
# ---------------------------------------------------------------------------
def rollout_paths(kind="circle", n=20, seed=0, steps=240):
    """Top-down root paths for n rollouts of one policy: the same motion,
    each started with a random heading and with a little policy noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, steps)
    if kind == "circle":
        base = np.stack([1.2 * np.sin(2 * np.pi * t), 1.2 * (1 - np.cos(2 * np.pi * t))], 1)
    elif kind == "strafe":
        base = np.stack([0.4 * np.sin(6 * np.pi * t), 3.0 * t], 1)
    else:   # straight trot
        base = np.stack([3.5 * t, 0.0 * t], 1)
    paths = []
    for _ in range(n):
        h = rng.uniform(0, 2 * np.pi)
        R = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
        drift = np.cumsum(rng.normal(0, 0.006, (steps, 2)), 0)
        paths.append(base @ R.T + drift)
    return paths


DATASET = dict(policies=40, rollouts_per_policy=20, trajectories=800,
               sources={"AI4Animation dog mocap": 15, "AI4Animation interactive wolf": 7,
                        "VHDC horses": 12, "Solo8 robot": 6})


def evaluate():
    out = {"retargeting": {}, "tracker_reward_vs_noise_deg": tracker_rewards()}
    for src in SOURCES:
        for g in GAITS:
            rt = retarget(g, src)
            # IK check: forward kinematics of the solved angles hits the (clipped) targets
            out["retargeting"][f"{src}/{g}"] = dict(
                scale=round(rt["scale"], 3), clipped_fraction=round(rt["clipped_fraction"], 3),
                naive_clipped_fraction=round(retarget(g, src, naive=True)["clipped_fraction"], 3))
    out["dataset"] = DATASET
    return out


if __name__ == "__main__":
    import json
    # sanity check: IK then FK returns the target
    for x, z in [(0.1, -0.27), (-0.12, -0.25), (0.0, -0.35)]:
        h, k, c = ik_2link(x, z, GO2["thigh"], GO2["calf"])
        fx, fz = fk_2link(h, k, GO2["thigh"], GO2["calf"])[1]
        assert c or (abs(fx - x) < 1e-9 and abs(fz - z) < 1e-9), (x, z, fx, fz)
    print(json.dumps(evaluate(), indent=2))
