"""
slides_kine2go_end.py - Article 4 (Kine2Go) slides, the common threads, and the takeaway.
"""

import math
import os
import sys

import numpy as np

from engine import *  # noqa: F401,F403
from engine import Slide

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from demos import kine2go_demo as K  # noqa: E402
from slides_intro_gorilla import ARTICLES  # noqa: E402

C4 = TEAL
LEG_COLORS = {"FL": TEAL, "FR": SKY, "RL": AMBER, "RR": PINK}


# ============================================================ 12. Kine2Go idea
class KineIdeaSlide(Slide):
    kicker, title = "Article 4 · Kine2Go", "800 Go2 trajectories, made from animal motion capture"
    kicker_color = C4
    hints = "→ next: retarget a horse onto the Go2 yourself"

    def draw(self, s):
        stages = [("horse", "1. Kinematic retargeting", "Map dog, horse, Solo8 and wolf motion onto the Go2's 12 joints "
                                                       "with inverse kinematics: feet become targets the robot can reach."),
                  ("brain", "2. RL motion imitation", "One PPO policy per motion. Reward = how closely the robot tracks the "
                                                      "reference. 8,192 parallel envs in Genesis, ~2 B steps, ~3 h per motion."),
                  ("film", "3. Rollouts + filtering", "Run every policy 20 times with random headings; record states AND "
                                                      "motor actions; drop falls and bad tracking by watching the videos.")]
        for i, (ic, h1, body) in enumerate(stages):
            a = self.appear(0.2 * i)
            y = 150 + i * 132 + 12 * (1 - a)
            rrect(s, CARD, (60, y, 640, 116), 16)
            icon(s, ic, (110, y + 58), 62, C4)
            text(s, h1, (158, y + 16), 20, TEXT, True)
            para(s, body, 158, y + 48, 520, 15, MUTED)
            if i < 2:
                arrow(s, FAINT, (380, y + 117), (380, y + 131), 2, 6)
        rrect(s, CARD2, (60, 552, 640, 68), 14)
        para(s, "Why it matters: motion capture has no motor commands (\"stop-motion frames\"), so it can't train "
                "a controller directly. The RL policies fill in the actions.", 80, 562, 600, 15, TEXT)
        # sources
        x0 = 730
        rrect(s, CARD, (x0, 150, 490, 470), 16)
        text(s, "40 policies × 20 rollouts = 800 trajectories", (x0 + 22, 166), 17, TEXT, True)
        g = ease(self.t / 1.0)
        cols = [TEAL, SKY, AMBER, LAV]
        for i, ((name, n), col) in enumerate(zip(K.DATASET["sources"].items(), cols)):
            y = 210 + i * 62
            text(s, name, (x0 + 22, y), 15, TEXT)
            rrect(s, col, (x0 + 22, y + 24, max(2, 18 * n * g), 18), 6)
            text(s, f"{n} motions", (x0 + 30 + 18 * n * g, y + 24), 14, MUTED)
        para(s, "Motions include walking, trotting, pacing, cantering, turning, strafing, crawling and "
                "figure-eights. Missing: most jumps, sitting, rough terrain, foot-pressure sensors.",
             x0 + 22, 470, 446, 15, MUTED)


# ============================================================ 13. retargeting demo
class KineRetargetSlide(Slide):
    kicker, title = "Article 4 · live demo", "Retargeting: scale the animal's feet, then solve the Go2's joints"
    kicker_color = C4
    hints = "1-4: walk / trot / pace / canter · S: source animal · N: naive copy (no scaling) · ↑/↓: speed"

    def enter(self):
        super().enter()
        self.gait, self.src, self.naive, self.speed = "trot", "horse", False, 0.5
        self.phase = 0.0
        self.compute()

    def compute(self):
        self.frames = 120
        self.rt = K.retarget(self.gait, self.src, self.frames, naive=self.naive)

    def key(self, k):
        gaits = list(K.GAITS)
        if pygame.K_1 <= k <= pygame.K_4:
            self.gait = gaits[k - pygame.K_1]
        elif k == pygame.K_s:
            srcs = list(K.SOURCES)
            self.src = srcs[(srcs.index(self.src) + 1) % len(srcs)]
        elif k == pygame.K_n:
            self.naive = not self.naive
        elif k == pygame.K_UP:
            self.speed = min(1.5, self.speed + 0.1)
            return True
        elif k == pygame.K_DOWN:
            self.speed = max(0.1, self.speed - 0.1)
            return True
        else:
            return False
        self.compute()
        return True

    def update(self, dt):
        super().update(dt)
        self.phase = (self.phase + dt * K.GAITS[self.gait]["freq"] * self.speed) % 1.0

    def body(self, s, origin, scale, morph, feet, color, targets=None, joints=None, stand=None):
        """Side view: body bar, then hip -> knee -> foot for the four legs."""
        ox, oy = origin
        half = morph["body"] / 2
        hips = {"FL": half, "FR": half, "RL": -half, "RR": -half}
        hy = oy - (stand or morph["stand"]) * scale
        draw.line(s, GRID, (ox - 180, oy), (ox + 180, oy), 2)
        thick_line(s, color, (ox - half * scale, hy), (ox + half * scale, hy), 12)
        for leg in ("FR", "RR", "FL", "RL"):              # far legs first, faded
            hx = ox + hips[leg] * scale
            far = leg in ("FR", "RR")
            col = tuple(int(c * (0.5 if far else 1.0)) for c in LEG_COLORS[leg])
            if joints:
                (kx, kz), (fx, fz) = K.fk_2link(*joints[leg], morph["thigh"], morph["calf"])
            else:
                fx, fz = feet[leg]
                hip, knee, _ = K.ik_2link(fx, fz, morph["thigh"], morph["calf"])
                (kx, kz), (fx, fz) = K.fk_2link(hip, knee, morph["thigh"], morph["calf"])
            knee = (hx + kx * scale, hy - kz * scale)
            foot = (hx + fx * scale, hy - fz * scale)
            thick_line(s, col, (hx, hy), knee, 7)
            thick_line(s, col, knee, foot, 6)
            circle(s, col, foot, 5)
            if targets and not far:
                tx, tz = targets[leg]
                if math.hypot(tx - fx, tz - fz) > 0.01:
                    # target out of reach: it is far below, so point straight down to it
                    arrow(s, CORAL, (foot[0], foot[1] + 4), (foot[0], oy + 36), 2, 8)

    def draw(self, s):
        src = K.SOURCES[self.src]
        ref, _ = K.gait_reference(self.gait, self.src, self.frames)
        f = int(self.phase * self.frames) % self.frames
        # left panel: the source animal at its own scale
        rrect(s, CARD, (60, 150, 420, 330), 16)
        text(s, f"SOURCE: {src['name']}", (82, 164), 14, C4, True)
        sc = 210 / (src["stand"] + 0.25 * (src["thigh"] + src["calf"]))
        with sublayer(s, (60, 150, 420, 330)) as L:
            self.body(L, (270, 430), sc, src, ref[f], (210, 214, 224))
        text(s, f"leg {src['thigh'] + src['calf']:.2f} m · stands {src['stand']:.2f} m", (82, 186), 13, MUTED)
        # right panel: the Go2
        rrect(s, CARD, (500, 150, 420, 330), 16)
        text(s, "TARGET: Unitree Go2 (retargeted)" + ("  · NAIVE COPY" if self.naive else ""), (522, 164), 14,
             CORAL if self.naive else C4, True)
        go = K.GO2
        sc2 = 185 / (go["stand"] + 0.25 * (go["thigh"] + go["calf"]))
        targets = None
        if self.naive:
            targets = ref[f]
        else:
            scale = self.rt["scale"]
            targets = {leg: (x * scale, z * go["stand"] / src["stand"]) for leg, (x, z) in ref[f].items()}
        stand = None
        if self.naive and self.rt["clipped_fraction"] > 0:
            stand = 0.98 * (go["thigh"] + go["calf"])      # legs stretched as far as they go
            sc2 = 185 / (stand + 0.25 * (go["thigh"] + go["calf"]))
        with sublayer(s, (500, 150, 420, 330)) as L:
            self.body(L, (725, 420), sc2, go, None, C4, targets=targets, joints=self.rt["joints"][f], stand=stand)
        text(s, f"leg {go['thigh'] + go['calf']:.2f} m · stands {go['stand']:.2f} m (approx.)", (522, 186), 13, MUTED)
        if stand:
            text(s, "legs fully stretched; targets are ~1 m lower (arrows)", (710, 460), 13, CORAL, anchor="midtop")
        # footfall pattern
        rrect(s, CARD, (60, 496, 860, 124), 16)
        text(s, f"Footfall pattern · {self.gait} (stance = foot on the ground)", (82, 506), 14, TEXT, True)
        g = K.GAITS[self.gait]
        for i, leg in enumerate(K.LEGS):
            y = 532 + i * 20
            text(s, leg, (82, y), 13, LEG_COLORS[leg], True)
            for c in range(80):
                ph = (c / 80 + g["offsets"][leg]) % 1.0
                if ph < g["duty"]:
                    draw.rect(s, LEG_COLORS[leg], (120 + c * 9.5, y + 2, 9.6, 12))
        cx = 120 + self.phase * 760
        draw.line(s, TEXT, (cx, 528), (cx, 612), 2)
        # right: numbers
        x0 = 940
        rrect(s, CARD, (x0, 150, 280, 470), 16)
        text(s, "This frame", (x0 + 18, 164), 15, TEXT, True)
        q = self.rt["joints"][f]["FL"]
        text(s, f"FL hip   {math.degrees(q[0]):6.1f}°", (x0 + 18, 196), 15, TEXT, mono=True)
        text(s, f"FL knee  {math.degrees(q[1]):6.1f}°", (x0 + 18, 220), 15, TEXT, mono=True)
        text(s, f"foot scale × {self.rt['scale']:.3f}", (x0 + 18, 256), 15, MUTED)
        clipped = self.rt["clipped_fraction"]
        col = CORAL if clipped > 0 else GREEN
        text(s, "unreachable targets", (x0 + 18, 296), 14, MUTED)
        text(s, f"{clipped:.0%}", (x0 + 18, 318), 40, col, True)
        para(s, "Naive copy (N) reuses the animal's foot positions in metres. A horse's feet are ~1.3 m below "
                "its hips, far beyond a 0.43 m Go2 leg, so no joint angles reach them (red arrows). The Solo8 is "
                "already about Go2-sized, so it fits either way. Scaling first keeps every target reachable: "
                "that is Kine2Go's retargeting stage.", x0 + 18, 380, 244, 13, MUTED)


# ============================================================ 14. imitation reward demo
class KineRewardSlide(Slide):
    kicker, title = "Article 4 · live demo", "The imitation reward: how well does it track?"
    kicker_color = C4
    hints = "↑/↓: tracking error · K: shove the robot (big shoves end the episode) · 1-4: gait"
    TERMS = [("q", "joint angles", 0.5, 1), ("qdot", "joint velocities", 0.05, 1), ("ee", "feet positions", 0.2, 1),
             ("pose", "root pose", 0.15, 2), ("vel", "root velocity", 0.1, 2)]

    def enter(self):
        super().enter()
        self.gait, self.noise = "trot", 3.0
        self.rt = K.retarget(self.gait, "dog", 120)
        self.phase = 0.0
        rng = np.random.default_rng(1)
        # smooth tracking error per joint: a slow sine with its own frequency and phase
        self.err_f = {leg: rng.uniform(0.6, 1.6, 2) for leg in K.LEGS}
        self.err_p = {leg: rng.uniform(0, 2 * np.pi, 2) for leg in K.LEGS}
        self.prev_ref = None
        self.pitch, self.pitch_v = 0.0, 0.0
        self.prev = None
        self.kept, self.dropped, self.ep_t, self.fallen = 0, 0, 0.0, 0.0
        self.rng = np.random.default_rng(0)
        self.r, self.terms = 1.0, {}

    def key(self, k):
        if k == pygame.K_UP:
            self.noise = min(25.0, self.noise + 1.0)
        elif k == pygame.K_DOWN:
            self.noise = max(0.0, self.noise - 1.0)
        elif k == pygame.K_k:
            self.pitch_v += self.rng.choice([-1, 1]) * self.rng.uniform(4, 12)
        elif pygame.K_1 <= k <= pygame.K_4:
            self.gait = list(K.GAITS)[k - pygame.K_1]
            self.rt = K.retarget(self.gait, "dog", 120)
            self.prev = self.prev_ref = None
        else:
            return False
        return True

    def update(self, dt):
        super().update(dt)
        if self.fallen:
            self.fallen -= dt
            if self.fallen <= 0:
                self.pitch, self.pitch_v, self.ep_t, self.fallen = 0.0, 0.0, 0.0, 0.0
            return
        J = self.rt["joints"]
        n = len(J)
        self.phase = (self.phase + dt * K.GAITS[self.gait]["freq"] * 0.5) % 1.0
        f = int(self.phase * n) % n
        sd = math.radians(self.noise)
        e = {leg: sd * np.sin(2 * np.pi * self.err_f[leg] * self.t + self.err_p[leg]) for leg in K.LEGS}
        self.sim_joints = {leg: (J[f][leg][0] + e[leg][0], J[f][leg][1] + e[leg][1]) for leg in K.LEGS}
        # root pitch: a damped spring, disturbed by shoves
        self.pitch_v += (-30 * self.pitch - 3 * self.pitch_v) * dt
        self.pitch += self.pitch_v * dt
        # velocities from the real elapsed time, for both the reference and the robot
        ref = K.state_from_joints(J[f], self.prev_ref or J[f], 0.0, 0.0, max(dt, 1e-3))
        sim = K.state_from_joints(self.sim_joints, self.prev or self.sim_joints, 0.0, self.pitch, max(dt, 1e-3))
        self.prev, self.prev_ref = self.sim_joints, J[f]
        total, self.terms = K.imitation_reward(sim, ref)
        self.r = total / K.MAX_REWARD
        self.ep_t += dt
        if abs(self.pitch) > K.TERMINATE_ROTATION_RAD:      # paper Table 3: rotation threshold 90 deg
            self.dropped += 1
            self.fallen = 1.6
        elif self.ep_t > 6.0:
            self.kept += 1
            self.ep_t = 0.0

    def draw(self, s):
        J = self.rt["joints"]
        f = int(self.phase * len(J)) % len(J)
        go = K.GO2
        rrect(s, CARD, (60, 150, 520, 470), 16)
        text(s, "grey ghost = reference motion · teal = policy", (82, 164), 14, MUTED)
        ox, oy, sc = 320, 500, 560
        for joints, col, alpha in ((J[f], (150, 156, 170), 120), (getattr(self, "sim_joints", J[f]), C4, None)):
            cs, sn = math.cos(self.pitch if col == C4 else 0), math.sin(self.pitch if col == C4 else 0)
            hy = oy - go["stand"] * sc

            def rot(px, py, cs=cs, sn=sn, hy=hy):
                dx, dy = px - ox, py - hy
                return (ox + dx * cs - dy * sn, hy + dx * sn + dy * cs)
            half = go["body"] / 2
            with sublayer(s, (60, 150, 520, 470), alpha) as L:
                thick_line(L, col, rot(ox - half * sc, hy), rot(ox + half * sc, hy), 12)
                for leg, hipx in (("FL", half), ("RL", -half)):
                    (kx, kz), (fx, fz) = K.fk_2link(*joints[leg], go["thigh"], go["calf"])
                    hp = rot(ox + hipx * sc, hy)
                    kp = rot(ox + (hipx + kx) * sc, hy - kz * sc)
                    fp = rot(ox + (hipx + fx) * sc, hy - fz * sc)
                    thick_line(L, col, hp, kp, 8)
                    thick_line(L, col, kp, fp, 7)
        draw.line(s, GRID, (90, oy), (550, oy), 2)
        if self.fallen:
            rrect(s, CORAL, (82, 192, 476, 56), 12)
            text(s, "Fell over (tilt > 90°): episode ends, trajectory filtered out", (320, 220), 16, BG, True,
                 anchor="center")
        text(s, f"tracking error ≈ {self.noise:.0f}° per joint", (82, 590), 15, TEXT)
        # reward bars
        x0 = 600
        rrect(s, CARD, (x0, 150, 620, 330), 16)
        text(s, "Kine2Go reward: R = Σ wᵢ · exp(−kᵢ · errorᵢ²)  (paper Eqs. 1-8)", (x0 + 20, 164), 15, TEXT, True)
        for i, (k, lab, w, mx) in enumerate(self.TERMS):
            y = 200 + i * 48
            v = self.terms.get(k, mx) / mx
            text(s, f"{lab}", (x0 + 20, y + 4), 15, TEXT)
            text(s, f"w = {w}", (x0 + 190, y + 4), 14, MUTED)
            draw.rect(s, GRID, (x0 + 270, y + 6, 300, 16), border_radius=6)
            draw.rect(s, C4, (x0 + 270, y + 6, max(2, 300 * v), 16), border_radius=6)
            text(s, f"{v:.2f}", (x0 + 580, y + 4), 14, TEXT, anchor="topright")
        text(s, f"total: {self.r:.0%} of the maximum", (x0 + 20, 448), 18, C4, True)
        rrect(s, CARD2, (x0, 494, 620, 126), 16)
        text(s, f"Episodes kept: {self.kept}    filtered out (fell): {self.dropped}", (x0 + 20, 508), 16, TEXT, True)
        para(s, "Joint angles carry half the weight, and the feet term is the sharpest (k = 40): a few "
                "centimetres of foot error already costs a lot. Episodes that fall are thrown away, as "
                "Kine2Go does when filtering rollouts.", x0 + 20, 538, 580, 14, MUTED)


# ============================================================ 15. dataset slide
class KineDatasetSlide(Slide):
    kicker, title = "Article 4 · the dataset", "One policy, 20 random headings, diverse data"
    kicker_color = C4
    hints = "M: motion (circle / strafe / straight)   ·   paths are drawn like the paper's Figure 4b"
    KINDS = ["circle", "strafe", "straight"]

    def enter(self):
        super().enter()
        self.ki = 0
        self.t0 = 0.0

    def key(self, k):
        if k == pygame.K_m:
            self.ki = (self.ki + 1) % 3
            self.t0 = self.t
            return True
        return False

    def draw(self, s):
        paths = K.rollout_paths(self.KINDS[self.ki], 20, seed=self.ki)
        rrect(s, CARD, (60, 150, 520, 470), 16)
        text(s, f"Root position of 20 rollouts · motion: {self.KINDS[self.ki]}", (82, 164), 14, TEXT, True)
        cx, cy, sc = 320, 395, 52
        draw.line(s, GRID, (cx - 220, cy), (cx + 220, cy), 1)
        draw.line(s, GRID, (cx, cy - 210), (cx, cy + 210), 1)
        prog = min(1.0, (self.t - self.t0) / 3.0)
        pal = [TEAL, SKY, AMBER, LAV, PINK]
        for i, p in enumerate(paths):
            n = max(2, int(len(p) * prog))
            pts = [(cx + x * sc, cy - y * sc) for x, y in p[:n]]
            draw.lines(s, pal[i % 5], False, pts, 2)
        text(s, "1 m grid lines · origin = start", (82, 590), 12, FAINT)
        x0 = 600
        rrect(s, CARD, (x0, 150, 620, 230), 16)
        text(s, "What each trajectory stores (per frame)", (x0 + 20, 164), 16, TEXT, True)
        items = ["joint positions and velocities (12 DoF)", "base orientation (quaternion) + angular velocity",
                 "the policy's motor actions (offsets from default pose)", "world positions and velocities of the joints"]
        for i, it in enumerate(items):
            circle(s, C4, (x0 + 30, 206 + i * 38), 4)
            text(s, it, (x0 + 44, 196 + i * 38), 15, TEXT)
        rrect(s, CARD2, (x0, 394, 620, 226), 16)
        text(s, "Intended use", (x0 + 20, 408), 16, C4, True)
        para(s, "Behaviour cloning, offline control, and especially behavioural foundation models (e.g. Meta "
                "Motivo, BFM-Zero) that learn many gaits in one policy. Because those models should capture a "
                "motion's style rather than copy it frame by frame, the authors suggest judging them with Earth "
                "Mover's Distance instead of plain mean-squared error.", x0 + 20, 438, 580, 15, TEXT)


# ============================================================ 16. common threads
class ThreadsSlide(Slide):
    kicker, title = "Putting them together", "Five threads that run through all four papers"
    hints = "Click a card (or press 1-5) to flip it"

    def enter(self):
        super().enter()
        S = self.app.context["summary"]
        l2r_ok = sum(r["reward"] >= 0.9 for r in S["l2r"]["tasks"])
        cap_ok = sum(r["cap"] >= 0.9 for r in S["l2r"]["tasks"])
        self.cards = [
            ("ban", CORAL, "Hand-made skills are the bottleneck",
             f"VoxPoser 88% vs 24% for LLM + primitives. Language to Rewards 90% vs 50%. My pendulum: rewards solved "
             f"{l2r_ok} tasks, primitives {cap_ok}."),
            ("eye", SKY, "Ground the LLM in reality",
             "Gorilla: retrieved docs + AST check. VoxPoser: camera perception. L2R: a motion template. "
             "Kine2Go: the Go2's real joint limits."),
            ("code", LAV, "Code is the interface",
             "API calls (Gorilla), NumPy value maps (VoxPoser), reward code (L2R). LLMs write code well, so "
             "code becomes the bridge."),
            ("sync", AMBER, "MPC does the low-level work",
             "VoxPoser replans at 5 Hz; L2R uses MuJoCo MPC. The same plan-act-replan loop as my Readings 6 CEM "
             "planner."),
            ("brain", TEAL, "RL is a tool, not the link",
             "Only Kine2Go trains with RL (PPO). Gorilla fine-tunes; VoxPoser and L2R optimise with MPC "
             "and train nothing."),
        ]
        self.flip = [0.0] * 5
        self.target = [0] * 5

    def rects(self):
        return [(60 + i * 236, 160, 220, 440) for i in range(5)]

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_5:
            i = k - pygame.K_1
            self.target[i] = 1 - self.target[i]
            return True
        return False

    def click(self, pos, button):
        for i, (x, y, w, h) in enumerate(self.rects()):
            if x <= pos[0] <= x + w and y <= pos[1] <= y + h:
                self.target[i] = 1 - self.target[i]
                return True
        return False

    def update(self, dt):
        super().update(dt)
        for i in range(5):
            self.flip[i] += (self.target[i] - self.flip[i]) * min(1, dt * 8)

    def draw(self, s):
        for i, (x, y, w, h) in enumerate(self.rects()):
            f = self.flip[i]
            back = f > 0.5
            sx = abs(math.cos(f * math.pi))
            card = pygame.Surface((PX(w), PX(h)), pygame.SRCALPHA)
            ic, col, head, body = self.cards[i]
            with local_origin():
                rrect(card, CARD2 if back else CARD, (0, 0, w, h), 18)
                icon(card, ic, (110, 80), 76, col)
                if not back:
                    para(card, head, 20, 150, w - 40, 22, TEXT, True)
                    text(card, f"press {i + 1}", (20, h - 36), 13, FAINT)
                else:
                    yb = para(card, head, 20, 136, w - 40, 16, col, True)
                    para(card, body, 20, yb + 10, w - 40, 15, TEXT)
            cw = max(1, int(PX(w) * sx))
            card = pygame.transform.smoothscale(card, (cw, PX(h)))
            x0, y0 = P((x + (w - w * sx) / 2, y))
            s.blit(card, (int(round(x0)), int(round(y0))))


# ============================================================ 17. takeaway
class TakeawaySlide(Slide):
    deep = True
    hints = "Thanks!   ·   Esc to quit"

    def draw(self, s):
        text(s, "TAKEAWAY", (80, 120), 18, AMBER, True)
        text(s, "Let the LLM say what to do.", (76, 156), 44, TEXT, True)
        text(s, "Let something grounded do it.", (76, 212), 44, TEAL, True)
        lessons = ["LLMs are good at goals, code and common sense, but they hallucinate and can't output torques.",
                   "Each paper picks a bridge (tool call, value map, reward, data) that a reliable system can execute.",
                   "Optimisers (MPC) and learned policies (RL on data like Kine2Go) turn that bridge into motion.",
                   "Every bridge must be checked against reality: docs, cameras, physics and joint limits."]
        for i, l in enumerate(lessons):
            a = self.appear(0.3 + 0.25 * i)
            y = 300 + i * 56
            circle(s, ARTICLES[i]["color"], (92, y + 12), 7)
            para(s, l, 112, y, 640, 18, MUTED, alpha=int(255 * a))
        rrect(s, CODE_BG, (80, 540, 660, 92), 14)
        for i, l in enumerate(["cd readings/Reading_7", "python presentation/interactive_presentation.py"]):
            text(s, l, (104, 558 + i * 32), 18, (220, 227, 239), mono=True)
        # the stack, with the pulse
        layers = ["Language"] + [a["layer"] for a in ARTICLES]
        cols = [TEXT] + [a["color"] for a in ARTICLES]
        f = (self.t * 0.45) % 1.0
        for i, (lab, col) in enumerate(zip(layers, cols)):
            y = 140 + i * 96
            rrect(s, CARD, (850, y, 330, 76), 16)
            text(s, lab, (1015, y + 38), 20, col, True, anchor="center")
            if i < 4:
                arrow(s, FAINT, (1015, y + 78), (1015, y + 94), 2, 7)
        circle(s, AMBER, (1172, 140 + f * 460), 6)
