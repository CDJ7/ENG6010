"""
voxposer_demo.py - Article 2: VoxPoser (Huang et al., CoRL 2023).

VoxPoser does not ask the LLM for motor commands. The LLM writes Python code
that (1) calls perception ("detect('vase')"), (2) fills voxel arrays with
NumPy, and (3) returns "value maps":
    * affordance map - high value = attract the entity ("go to the handle")
    * avoidance map  - high value = repel the entity ("watch out for the vase")
A motion planner then searches the combined cost map, the robot executes
only the first waypoint, and everything is replanned (5 Hz in the paper):
model-predictive control. No component is trained.

This file reimplements that pipeline on a 2-D top-down slice of a table
(100 x 100 voxels, 1 voxel = 1 cm) instead of the paper's 100^3 grid:
    detect()                    stand-in for the VLM perception stack
                                (OWL-ViT + Segment Anything + XMem in the paper)
    affordance_map()            paper: Euclidean distance transform
    avoidance_map()             paper: Gaussian filter around the obstacle
    cost_map()                  paper: negative weighted sum, weights 2 and 1
    greedy_plan()               paper: greedy search over the cost map
    run_episode()               closed-loop MPC: execute first step, replan

The "LLM code" for each instruction (`INSTRUCTIONS[...]["code"]`) is a hand-
written example in the style of the paper's Fig. 2. No LLM is called.
"""

import numpy as np

N = 100                    # voxels per side (1 voxel = 1 cm)
YY, XX = np.mgrid[0:N, 0:N]


# ---------------------------------------------------------------------------
# Perception stand-in
# ---------------------------------------------------------------------------
def make_scene(rng=None, handle=(80, 62), vase=(52, 55), mug=(30, 25), bowl=(80, 20), start=(12, 50)):
    """A table with a drawer handle, a vase, a mug and a bowl (cm, top-down)."""
    return {"gripper": np.array(start, float),
            "objects": {"top drawer handle": dict(pos=np.array(handle, float), radius=3),
                        "vase": dict(pos=np.array(vase, float), radius=7),
                        "mug": dict(pos=np.array(mug, float), radius=5),
                        "bowl": dict(pos=np.array(bowl, float), radius=8)}}


def detect(scene, name):
    """What the VLM returns in the paper: position and occupancy of an object."""
    o = scene["objects"][name]
    occ = (XX - o["pos"][0]) ** 2 + (YY - o["pos"][1]) ** 2 <= o["radius"] ** 2
    return dict(name=name, pos=o["pos"].copy(), radius=o["radius"], occupancy=occ)


# ---------------------------------------------------------------------------
# Value maps (what the LLM-written code produces)
# ---------------------------------------------------------------------------
def affordance_map(target):
    """1 at the target, falling off with distance (Euclidean distance transform)."""
    d = np.hypot(XX - target["pos"][0], YY - target["pos"][1])
    return 1.0 - d / d.max()


def avoidance_map(obstacle, margin_cm=0.0, sigma=6.0):
    """High near the obstacle, smoothed with a Gaussian so the planner keeps clear."""
    d = np.hypot(XX - obstacle["pos"][0], YY - obstacle["pos"][1]) - obstacle["radius"] - margin_cm
    return np.exp(-np.maximum(d, 0.0) ** 2 / (2 * sigma ** 2))


def cost_map(affordances, avoidances, w_aff=2.0, w_avoid=1.0):
    """Paper: 'the negative of the weighted sum of normalized affordance and
    avoidance maps with weights 2 and 1' (avoidance enters with a minus sign
    because high avoidance values repel)."""
    def norm(m):
        return (m - m.min()) / (m.max() - m.min() + 1e-9)
    aff = sum(affordances) if affordances else np.zeros((N, N))
    avo = sum(avoidances) if avoidances else np.zeros((N, N))
    return -(w_aff * norm(aff) - w_avoid * (norm(avo) if avoidances else avo))


# ---------------------------------------------------------------------------
# The "LLM programs" for three instructions (hand-written examples)
# ---------------------------------------------------------------------------
INSTRUCTIONS = [
    dict(text="Move to the top drawer handle.",
         code=["def affordance_map():",
               "    handle = detect('top drawer handle')",
               "    return distance_transform(handle.pos)",
               "",
               "def avoidance_map():",
               "    return None     # nothing to avoid"],
         target="top drawer handle", avoid=[]),
    dict(text="Open the top drawer, and watch out for that vase!",
         code=["def affordance_map():",
               "    handle = detect('top drawer handle')",
               "    return distance_transform(handle.pos)",
               "",
               "def avoidance_map():",
               "    vase = detect('vase')",
               "    map[vase.occupancy] = 1",
               "    return gaussian_smooth(map)"],
         target="top drawer handle", avoid=[("vase", 0.0)]),
    dict(text="Go to the bowl, staying at least 15 cm from the mug and the vase.",
         code=["def affordance_map():",
               "    bowl = detect('bowl')",
               "    return distance_transform(bowl.pos)",
               "",
               "def avoidance_map():",
               "    for name in ['mug', 'vase']:",
               "        obj = detect(name)",
               "        set_voxel_by_radius(map, obj.pos, 15, 1)",
               "    return gaussian_smooth(map)"],
         target="bowl", avoid=[("mug", 10.0), ("vase", 10.0)]),
]


def compose(scene, instr):
    """Run the instruction's 'LLM program' against the current scene."""
    tgt = detect(scene, instr["target"])
    affs = [affordance_map(tgt)]
    avos = [avoidance_map(detect(scene, n), margin) for n, margin in instr["avoid"]]
    occ = np.zeros((N, N), bool)                 # cells inside avoided objects (+2 cm)
    for n, _ in instr["avoid"]:
        o = scene["objects"][n]
        occ |= (XX - o["pos"][0]) ** 2 + (YY - o["pos"][1]) ** 2 <= (o["radius"] + 2) ** 2
    return dict(affordance=affs[0], avoidance=sum(avos) if avos else np.zeros((N, N)),
                cost=cost_map(affs, avos), target=tgt, occupied=occ)


# ---------------------------------------------------------------------------
# Planner + MPC
# ---------------------------------------------------------------------------
STEPS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def greedy_plan(cost, start, goal, max_len=250, goal_tol=2.5):
    """Greedy search: from the start, repeatedly step to the cheapest
    neighbouring voxel. Returns (path, reached_goal)."""
    x, y = int(round(start[0])), int(round(start[1]))
    path, seen = [(x, y)], {(x, y)}
    for _ in range(max_len):
        if np.hypot(x - goal[0], y - goal[1]) <= goal_tol:
            return path, True
        best = None
        for dx, dy in STEPS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and (nx, ny) not in seen:
                c = cost[ny, nx] + 0.002 * np.hypot(dx, dy)
                if best is None or c < best[0]:
                    best = (c, nx, ny)
        if best is None:
            return path, False
        x, y = best[1], best[2]
        path.append((x, y))
        seen.add((x, y))
    return path, False


def astar_plan(cost, start, goal, goal_tol=2.5, blocked=None):
    """A* / Dijkstra over the same cost map, for comparison with greedy search.
    Step cost = distance x (1 + 6 * normalised map cost), so it trades path
    length against staying in low-cost voxels. Unlike greedy search it cannot
    get stuck in a local minimum (the classic potential-field trap)."""
    import heapq
    c = (cost - cost.min()) / (cost.max() - cost.min() + 1e-9)
    sx, sy = int(round(start[0])), int(round(start[1]))
    gx, gy = goal
    best = {(sx, sy): 0.0}
    prev = {}
    pq = [(0.0, 0.0, sx, sy)]
    while pq:
        f, g, x, y = heapq.heappop(pq)
        if np.hypot(x - gx, y - gy) <= goal_tol:
            path = [(x, y)]
            while (x, y) in prev:
                x, y = prev[(x, y)]
                path.append((x, y))
            return path[::-1], True
        if g > best.get((x, y), 1e18):
            continue
        for dx, dy in STEPS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and not (blocked is not None and blocked[ny, nx]):
                ng = g + np.hypot(dx, dy) * (1 + 6 * c[ny, nx])
                if ng < best.get((nx, ny), 1e18):
                    best[(nx, ny)] = ng
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (ng + np.hypot(nx - gx, ny - gy), ng, nx, ny))
    return [(sx, sy)], False


TRAP_INSTRUCTION = dict(
    text="Reach the handle, avoiding the mug, the vase and the bowl.",
    code=["def avoidance_map():",
          "    for name in ['mug', 'vase', 'bowl']:",
          "        map[detect(name).occupancy] = 1",
          "    return gaussian_smooth(map)"],
    target="top drawer handle", avoid=[("vase", 0.0), ("mug", 0.0), ("bowl", 0.0)])


def trap_scene():
    """Three objects form a cup that opens towards the gripper, right on the
    straight line to the handle: a textbook local minimum ("potential-field
    trap") for greedy descent. Greedy search walks into the cup first."""
    s = make_scene(handle=(88, 50), mug=(55, 38), vase=(62, 50), bowl=(55, 62), start=(12, 50))
    for n in ("mug", "vase", "bowl"):
        s["objects"][n]["radius"] = 8
    return s


def collides(scene, p, names):
    return any(np.hypot(p[0] - scene["objects"][n]["pos"][0], p[1] - scene["objects"][n]["pos"][1])
               <= scene["objects"][n]["radius"] for n in names)


def run_episode(scene, instr, replan=True, disturb_at=None, disturb=None, max_steps=250):
    """Execute an instruction. With replan=True (MPC) the value maps and plan
    are recomputed every step from the latest scene; with replan=False the
    first plan is followed blindly. `disturb` moves an object at step
    `disturb_at` (the paper moves targets/obstacles during execution)."""
    pos = scene["gripper"].copy()
    traj, collided = [pos.copy()], False
    plan = None
    names = [n for n in scene["objects"] if n != instr["target"]]
    for t in range(max_steps):
        if disturb_at is not None and t == disturb_at:
            disturb(scene)
        if replan or plan is None:
            maps = compose(scene, instr)
            plan, _ = greedy_plan(maps["cost"], pos, maps["target"]["pos"])
            k = 1
        if k >= len(plan):
            break
        pos = np.array(plan[k], float)
        k += 1
        traj.append(pos.copy())
        collided |= collides(scene, pos, names)
        goal = scene["objects"][instr["target"]]["pos"]
        if np.hypot(*(pos - goal)) <= 2.5:
            return dict(traj=traj, reached=True, collided=collided, steps=t + 1)
    return dict(traj=traj, reached=False, collided=collided, steps=max_steps)


def random_scene(rng):
    """Random scene with the vase roughly between the gripper and the handle."""
    start = np.array([rng.uniform(8, 20), rng.uniform(20, 80)])
    handle = np.array([rng.uniform(78, 92), rng.uniform(20, 80)])
    f = rng.uniform(0.35, 0.65)
    vase = start + f * (handle - start) + rng.normal(0, 3, 2)
    return make_scene(handle=tuple(handle), vase=tuple(vase), start=tuple(start),
                      mug=(rng.uniform(20, 80), rng.uniform(85, 95)), bowl=(rng.uniform(20, 80), 6))


def evaluate(n=300, seed=0):
    """Toy versions of the paper's questions, on random scenes:
    1. does the avoidance map stop the gripper hitting the vase?
    2. does closed-loop replanning (MPC) survive a disturbance
       (the handle moved half-way through), compared with open-loop?"""
    rng = np.random.default_rng(seed)
    res = {k: {"reached": 0, "collided": 0} for k in ("aff_only", "aff_avoid", "open_loop_dist", "mpc_dist")}
    outcomes = {k: [] for k in res}      # per scene: 0 clean arrival, 1 hit an obstacle, 2 did not arrive
    for _ in range(n):
        s = random_scene(rng)
        shift = rng.uniform(-25, 25)

        def disturb(sc, d=shift):
            p = sc["objects"]["top drawer handle"]["pos"]
            p[1] = np.clip(p[1] + d, 8, 92)
        for key, instr, kw in (("aff_only", INSTRUCTIONS[0], {}),
                               ("aff_avoid", INSTRUCTIONS[1], {}),
                               ("open_loop_dist", INSTRUCTIONS[1], dict(replan=False, disturb_at=30, disturb=disturb)),
                               ("mpc_dist", INSTRUCTIONS[1], dict(replan=True, disturb_at=30, disturb=disturb))):
            sc = {"gripper": s["gripper"].copy(),
                  "objects": {k: dict(v, pos=v["pos"].copy()) for k, v in s["objects"].items()}}
            out = run_episode(sc, instr, **kw)
            res[key]["reached"] += out["reached"] and not out["collided"]
            res[key]["collided"] += out["collided"]
            outcomes[key].append(1 if out["collided"] else (0 if out["reached"] else 2))
    summary = {k: {m: v / n for m, v in d.items()} for k, d in res.items()}
    for k in summary:
        summary[k]["outcomes"] = outcomes[k]
    return summary | {"n_scenes": n}


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate(), indent=2))
