"""
slides_kine2go.py - Article 4 (Kine2Go): an animated assembly line, an IK
playground with the leg's workspace, the reward kernels with sliders, and the
dataset coverage explorer.
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

C4 = TEAL
LEG_COLORS = {"FL": TEAL, "FR": SKY, "RL": AMBER, "RR": PINK}
SRC_COLORS = {"dog": AMBER, "wolf": LAV, "horse": PINK, "solo8": SKY}


def mini_go2(s, origin, scale, joints, color, pitch=0.0, offset=None):
    """Side view of the Go2 from a joints dict {leg: (hip, knee)}."""
    ox, oy = origin
    go = K.GO2
    half = go["body"] / 2
    hy = oy - go["stand"] * scale
    ca, sa = math.cos(pitch), math.sin(pitch)
    hips = {}
    for leg, hx in (("FL", half), ("FR", half), ("RL", -half), ("RR", -half)):
        hips[leg] = (ox + hx * ca * scale, hy - hx * sa * scale)
    thick_line(s, color, hips["RL"], hips["FL"], 12)
    circle(s, color, (hips["FL"][0] + 16 * ca, hips["FL"][1] - 16 * sa - 6), 9)
    for leg in ("FR", "RR", "FL", "RL"):
        hip, knee = joints[leg]
        if offset:
            hip, knee = hip + offset, knee + offset * 0.8
        (kx, kz), (fx, fz) = K.fk_2link(hip + pitch, knee, go["thigh"], go["calf"])
        h = hips[leg]
        far = leg in ("FR", "RR")
        col = tuple(int(c * (0.5 if far else 1.0)) for c in LEG_COLORS[leg])
        kp = (h[0] + kx * scale, h[1] - kz * scale)
        fp = (h[0] + fx * scale, h[1] - fz * scale)
        thick_line(s, col, h, kp, 6)
        thick_line(s, col, kp, fp, 5)
        circle(s, col, fp, 4)


# ============================================================ 13. idea: assembly line
class KineIdeaSlide(Slide):
    kicker, title = "Article 4 · Kine2Go", "A factory that turns animal motion into robot training data"
    kicker_color = C4
    hints = "Click a station to explain it   ·   + / −: belt speed   ·   → next: the IK playground"
    STATIONS = [("horse", "Mocap clips", "40 source motions: 15 dog, 7 wolf, 12 horse clips and 6 Solo8 "
                                         "robot motions (walk, trot, pace, canter, turns, sideways)."),
                ("cogs", "Retarget", "Scale each clip to the Go2's legs and solve inverse kinematics for its "
                                     "12 joints, inside the joint limits."),
                ("brain", "RL gym", "Train one PPO policy per motion in simulation to imitate it with the "
                                    "reward on slide 15. Falls end the episode."),
                ("route", "Rollouts", "Run each trained policy 20 times from random headings, logging "
                                      "states and motor actions at every step."),
                ("database", "Dataset", "40 policies × 20 rollouts = 800 trajectories of states + actions, "
                                        "ready to train world models or new policies.")]
    XS = [150, 395, 640, 885, 1130]

    def enter(self):
        super().enter()
        self.speed, self.sel = 1.0, None
        self.clock = 0.0
        rng = np.random.default_rng(1)
        kinds = ["dog"] * 15 + ["wolf"] * 7 + ["horse"] * 12 + ["solo8"] * 6
        rng.shuffle(kinds)
        self.tokens = kinds

    def key(self, k):
        if k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(4.0, self.speed * 1.5)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(0.25, self.speed / 1.5)
        else:
            return False
        return True

    def click(self, pos, button):
        for i, x in enumerate(self.XS):
            if inside(pos, (x - 90, 220, 180, 150)):
                self.sel = None if self.sel == i else i
                return True
        return False

    def update(self, dt):
        super().update(dt)
        self.clock += dt * self.speed
        cycle = 40 * 1.6 + (self.XS[4] - 60) / 90 + 4.0      # all 40 clips through, then a short pause
        if self.clock > cycle:
            self.clock = 0.0                                   # start the factory again

    def draw(self, s):
        belt_y = 410
        draw.rect(s, GRID, (60, belt_y, 1160, 12), border_radius=6)
        for k in range(30):
            x = 60 + ((k * 40 + self.clock * 50) % 1160)
            draw.line(s, FAINT, (x, belt_y), (x - 6, belt_y + 12), 2)
        for i, (ic, lab, _) in enumerate(self.STATIONS):
            x = self.XS[i]
            on = self.sel == i
            rrect(s, CARD2 if on else CARD, (x - 90, 220, 180, 150), 16)
            if on:
                draw.rect(s, C4, (x - 90, 220, 180, 150), 2, border_radius=16)
            icon(s, ic, (x, 268), 56, C4)
            text(s, lab, (x, 312), 17, TEXT, True, anchor="midtop")
            draw.rect(s, CARD, (x - 30, 370, 60, 40))           # pillar to the belt
        # tokens: one per source clip, released one after another
        period = 1.6
        n_out = 0
        released = int(self.clock / period) + 1
        for j in range(min(released, 40)):
            age = self.clock - j * period
            x = 60 + age * 90
            kind = self.tokens[j]
            col = SRC_COLORS[kind]
            if x < self.XS[2]:                                   # a clip (before training)
                if x < self.XS[1]:
                    circle(s, col, (x, belt_y - 12), 10)
                else:                                            # retargeted: becomes a Go2-shaped square
                    rrect(s, col, (x - 10, belt_y - 22, 20, 20), 4)
            elif x < self.XS[3]:                                 # a policy (brain dot)
                circle(s, col, (x, belt_y - 12), 12)
                circle(s, BG, (x, belt_y - 12), 5)
            elif x < self.XS[4]:                                 # 20 rollouts fan out
                f = (x - self.XS[3]) / (self.XS[4] - self.XS[3])
                for r in range(20):
                    a = r / 20 * 2 * math.pi
                    circle(s, col, (x + math.cos(a) * 22 * f, belt_y - 30 + math.sin(a) * 22 * f), 3)
            else:
                n_out += 1
        # the dataset bin fills up
        total = n_out * 20
        rrect(s, CARD, (1060, 450, 160, 140), 14)
        fill = total / 800
        draw.rect(s, mix(CARD, C4, 0.6), (1066, 584 - 128 * fill, 148, 128 * fill), border_radius=10)
        text(s, f"{total}", (1140, 500), 30, TEXT, True, anchor="center")
        text(s, "of 800 trajectories", (1140, 532), 12, MUTED, anchor="center")
        # legend
        for k, (kind, n) in enumerate([("dog", 15), ("wolf", 7), ("horse", 12), ("solo8", 6)]):
            x = 60 + k * 170
            circle(s, SRC_COLORS[kind], (x + 8, 160), 7)
            text(s, f"{kind} ×{n}", (x + 22, 151), 15, TEXT)
        text(s, f"belt speed ×{self.speed:.2g}", (1220, 151), 14, FAINT, anchor="topright")
        text(s, "circle = clip · square = retargeted · ring = trained policy · spray = 20 rollouts",
             (60, 182), 13, FAINT)
        if self.sel is not None:
            bubble(s, 60, 460, 960, [self.STATIONS[self.sel][2]], C4, 16)
        else:
            rrect(s, CARD, (60, 460, 980, 130), 14)
            para(s, "Why bother? Real robot data is slow, expensive and risky to collect. Kine2Go is "
                    "the first open dataset of this kind for the Go2: varied, physically valid motion "
                    "with the motor actions that produced it.", 84, 476, 930, 16, TEXT)
        para(s, "Paper: Pałucki et al., 2026. No LLM here: Kine2Go builds the data layer that low-level "
                "controllers learn from.", 60, 610, 960, 13, MUTED)


# ============================================================ 14. IK playground
class KineIKSlide(Slide):
    kicker, title = "Article 4 · Kine2Go · IK playground", "Where can a Go2 foot go? Drag it and find out"
    kicker_color = C4
    hints = "Drag inside the leg panel to move the foot   ·   S: source animal · G: gait · N: naive copy (no scaling)"
    HIP = (340, 330)
    SC = 470            # px per metre

    def enter(self):
        super().enter()
        self.target = (0.05, -0.30)
        self.src, self.gait, self.naive = "dog", "trot", False
        self.drag, self.phase = False, 0.0
        self.compute()

    def compute(self):
        self.rt = K.retarget(self.gait, self.src, 60, naive=self.naive)
        self.ref, _ = K.gait_reference(self.gait, self.src, 60)

    def key(self, k):
        if k == pygame.K_s:
            keys = list(K.SOURCES)
            self.src = keys[(keys.index(self.src) + 1) % len(keys)]
        elif k == pygame.K_g:
            keys = list(K.GAITS)
            self.gait = keys[(keys.index(self.gait) + 1) % len(keys)]
        elif k == pygame.K_n:
            self.naive = not self.naive
        else:
            return False
        self.compute()
        return True

    def to_m(self, pos):
        return (pos[0] - self.HIP[0]) / self.SC, -(pos[1] - self.HIP[1]) / self.SC

    def to_px(self, x, z):
        return self.HIP[0] + x * self.SC, self.HIP[1] - z * self.SC

    def click(self, pos, button):
        if inside(pos, (60, 140, 560, 516)):
            self.drag = True
            self.target = self.to_m(pos)
            return True
        return False

    def release(self, pos):
        self.drag = False

    def motion(self, pos, buttons):
        if self.drag:
            p = (min(max(pos[0], 64), 616), min(max(pos[1], 144), 652))
            self.target = self.to_m(p)

    def update(self, dt):
        super().update(dt)
        self.phase = (self.phase + dt * K.GAITS[self.gait]["freq"]) % 1.0

    def draw(self, s):
        go = K.GO2
        l1, l2 = go["thigh"], go["calf"]
        rmin, rmax = K.workspace(l1, l2)
        rrect(s, CARD, (60, 140, 560, 516), 18)
        with sublayer(s, (60, 140, 560, 516)) as L:
            # workspace annulus
            circle(L, (22, 52, 60), self.HIP, rmax * self.SC)
            circle(L, CARD, self.HIP, rmin * self.SC)
            draw.circle(L, mix(CARD, C4, 0.6), self.HIP, rmax * self.SC, 1)
            draw.circle(L, mix(CARD, C4, 0.6), self.HIP, rmin * self.SC, 1)
            # ground at stand height
            gy = self.to_px(0, -go["stand"])[1]
            draw.line(L, GRID, (60, gy), (620, gy), 2)
            # this gait's foot path for the FL leg
            src = K.SOURCES[self.src]
            sc = 1.0 if self.naive else self.rt["scale"]
            zs = 1.0 if self.naive else go["stand"] / src["stand"]
            path = [self.to_px(fr["FL"][0] * sc, fr["FL"][1] * zs) for fr in self.ref]
            col = CORAL if self.naive and self.rt["clipped_fraction"] > 0 else LEG_COLORS["FL"]
            draw.lines(L, col, True, path, 2)
            # the foot the user drags
            tx, tz = self.target
            hip, knee, clipped = K.ik_2link(tx, tz, l1, l2)
            (kx, kz), (fx, fz) = K.fk_2link(hip, knee, l1, l2)
            kp, fp = self.to_px(kx, kz), self.to_px(fx, fz)
            tp = self.to_px(tx, tz)
            # law of cosines triangle
            draw.line(L, FAINT, self.HIP, fp, 1)
            thick_line(L, (230, 234, 242), self.HIP, kp, 12)
            thick_line(L, (230, 234, 242), kp, fp, 10)
            circle(L, C4, self.HIP, 9)
            circle(L, C4, kp, 7)
            circle(L, GREEN if not clipped else AMBER, fp, 9)
            if clipped:
                draw.line(L, CORAL, fp, tp, 2)
                draw.circle(L, CORAL, tp, 9, 2)
            d = math.hypot(fx, fz)
            text(L, f"d = {d:.3f} m", (max(self.HIP[0], fp[0]) + 16, (self.HIP[1] + fp[1]) / 2), 12, MUTED)
            if self.naive and self.src == "horse":
                text(L, "horse foot targets are ~1.3 m below the hip ↓ (off the panel)", (340, 620), 13, CORAL,
                     anchor="midtop")
        text(s, "reachable ring (workspace)", (80, 156), 13, C4, True)
        text(s, "foot path of this gait (FL leg)", (80, 176), 13, CORAL if self.naive else LEG_COLORS["FL"])
        text(s, "hip", (self.HIP[0] + 14, self.HIP[1] - 22), 12, C4)
        text(s, "thigh l₁ = calf l₂ = 0.213 m", (600, 156), 13, MUTED, anchor="topright")
        # ---------------- right: the maths + a moving Go2
        x0 = 640
        rrect(s, CARD, (x0, 140, 580, 200), 18)
        text(s, "INVERSE KINEMATICS (2 links)", (x0 + 20, 154), 12, C4, True)
        rows = [("cos(knee) = (d² − l₁² − l₂²) / (2 l₁ l₂)", f"knee bend = {math.degrees(knee):5.1f}°"),
                ("hip = atan2(x, −z) − atan2(l₂ sin k, l₁ + l₂ cos k)", f"hip angle = {math.degrees(hip):+6.1f}°"),
                ("reach:  " f"{rmin:.3f} m ≤ d ≤ {rmax:.3f} m",
                 "out of reach → clipped" if clipped else "inside the ring: reachable")]
        for j, (eq, val) in enumerate(rows):
            y = 182 + j * 46
            text(s, eq, (x0 + 20, y), 14, TEXT, mono=True)
            text(s, val, (x0 + 20, y + 20), 13, (AMBER if clipped and j == 2 else C4), True)
        # mini Go2 walking the retargeted gait
        rrect(s, CARD, (x0, 356, 580, 300), 18)
        f = int(self.phase * 60) % 60
        status = "NAIVE COPY (no scaling)" if self.naive else f"scaled ×{self.rt['scale']:.3f} + IK"
        text(s, f"{K.SOURCES[self.src]['name']} · {self.gait} → Go2 · {status}", (x0 + 20, 370), 13,
             CORAL if self.naive else C4, True)
        with sublayer(s, (x0, 390, 580, 210)) as L:
            draw.line(L, GRID, (x0 + 20, 580), (x0 + 560, 580), 2)
            mini_go2(L, (x0 + 290, 580), 500, self.rt["joints"][f], (230, 234, 242))
        frac = self.rt["clipped_fraction"]
        col = CORAL if frac > 0 else GREEN
        text(s, f"foot targets out of reach: {frac:.0%}", (x0 + 20, 612), 15, col, True)
        text(s, "S source · G gait · N naive", (x0 + 560, 614), 12, FAINT, anchor="topright")


# ============================================================ 15. reward kernels
class KineRewardSlide(Slide):
    kicker, title = "Article 4 · Kine2Go · the imitation reward", "Each error is squashed by exp(−k·e²)"
    kicker_color = C4
    hints = "Drag the sliders (or Tab + ↑/↓)   ·   past 1 m or 90° the episode ends, as in the paper (Table 3)"
    TERMS = [  # key, label, unit, slider max, to-SI, kernel, weight, how many copies are summed
        ("q", "joint angle error", "°", 40, math.radians, K.KERNELS["q"], K.WEIGHTS["q"], 12),
        ("qdot", "joint speed error", "rad/s", 8, float, K.KERNELS["qdot"], K.WEIGHTS["qdot"], 12),
        ("ee", "foot position error", "cm", 20, lambda v: v / 100, K.KERNELS["ee"], K.WEIGHTS["ee"], 4),
        ("pos", "body position error", "cm", 120, lambda v: v / 100, K.KERNELS["pos"], K.WEIGHTS["pose"], 1),
        ("rot", "body tilt error", "°", 100, math.radians, K.KERNELS["rot"], K.WEIGHTS["pose"], 1),
        ("vel", "body speed error", "m/s", 2, float, K.KERNELS["vel"], K.WEIGHTS["vel"], 1),
    ]

    def enter(self):
        super().enter()
        self.vals = [5.0, 1.0, 2.0, 10.0, 8.0, 0.2]
        self.sel, self.drag = 0, None

    def srect(self, i):
        return 84, 180 + i * 72, 260

    def key(self, k):
        if k == pygame.K_TAB:
            self.sel = (self.sel + 1) % 6
        elif k in (pygame.K_UP, pygame.K_DOWN):
            mx = self.TERMS[self.sel][3]
            d = mx / 40 * (1 if k == pygame.K_UP else -1)
            self.vals[self.sel] = float(np.clip(self.vals[self.sel] + d, 0, mx))
        else:
            return False
        return True

    def click(self, pos, button):
        for i in range(6):
            x, y, w = self.srect(i)
            if inside(pos, (x - 10, y + 10, w + 20, 34)):
                self.sel, self.drag = i, i
                self.motion(pos, (1, 0, 0))
                return True
        return False

    def release(self, pos):
        self.drag = None

    def motion(self, pos, buttons):
        if self.drag is not None:
            x, y, w = self.srect(self.drag)
            self.vals[self.drag] = float(np.clip((pos[0] - x) / w, 0, 1) * self.TERMS[self.drag][3])

    def term_r(self, i, v=None):
        key, _, _, _, si, kk, _, n = self.TERMS[i]
        e = si(self.vals[i] if v is None else v)
        return math.exp(-kk * n * e * e)

    def draw(self, s):
        rrect(s, CARD, (60, 140, 310, 516), 18)
        text(s, "TRACKING ERRORS", (84, 154), 12, C4, True)
        for i, (key, lab, unit, mx, *_r) in enumerate(self.TERMS):
            x, y, w = self.srect(i)
            slider(s, x, y, w, self.vals[i], 0, mx, lab, C4, self.sel == i, "{:.1f} " + unit)
        # ---- small multiples: one kernel curve per term
        for i, (key, lab, unit, mx, si, kk, wt, n) in enumerate(self.TERMS):
            cx0, cy0 = 390 + (i % 3) * 196, 140 + (i // 3) * 200
            rrect(s, CARD, (cx0, cy0, 184, 188), 14)
            text(s, f"r_{key} · w = {wt:g}", (cx0 + 12, cy0 + 10), 13, C4 if i == self.sel else TEXT, True)
            text(s, f"k = {kk:g}", (cx0 + 172, cy0 + 12), 11, FAINT, anchor="topright")
            px0, py0, pw, ph = cx0 + 14, cy0 + 40, 156, 110
            draw.line(s, GRID, (px0, py0 + ph), (px0 + pw, py0 + ph), 1)
            draw.line(s, GRID, (px0, py0), (px0, py0 + ph), 1)
            pts = [(px0 + pw * j / 60, py0 + ph * (1 - self.term_r(i, mx * j / 60))) for j in range(61)]
            draw.lines(s, C4, False, pts, 2)
            v = self.vals[i]
            r = self.term_r(i)
            mxp = (px0 + pw * v / mx, py0 + ph * (1 - r))
            draw.line(s, FAINT, (mxp[0], py0 + ph), mxp, 1)
            circle(s, AMBER, mxp, 6)
            text(s, f"{r:.2f}", (cx0 + 172, cy0 + 158), 15, AMBER, True, anchor="topright")
            text(s, f"0 … {mx} {unit}", (cx0 + 14, cy0 + 162), 11, FAINT)
        # ---- total + robot
        r = [self.term_r(i) for i in range(6)]
        total = (K.WEIGHTS["q"] * r[0] + K.WEIGHTS["qdot"] * r[1] + K.WEIGHTS["ee"] * r[2]
                 + K.WEIGHTS["pose"] * (r[3] + r[4]) + K.WEIGHTS["vel"] * (r[5] + 1.0))
        x0 = 390
        rrect(s, CARD, (x0, 540, 576, 116), 14)
        text(s, "TOTAL REWARD (Eq. 1, angular-speed term held at 1)", (x0 + 16, 552), 12, C4, True)
        parts = [(K.WEIGHTS["q"] * r[0], TEAL), (K.WEIGHTS["qdot"] * r[1], SKY), (K.WEIGHTS["ee"] * r[2], LAV),
                 (K.WEIGHTS["pose"] * (r[3] + r[4]), AMBER), (K.WEIGHTS["vel"] * (r[5] + 1), PINK)]
        xx = x0 + 16
        for v, col in parts:
            w = 470 * v / K.MAX_REWARD
            draw.rect(s, col, (xx, 580, max(0, w - 2), 26), border_radius=4)
            xx += w
        draw.rect(s, GRID, (x0 + 16, 580, 470, 26), 1, border_radius=4)
        text(s, f"{total / K.MAX_REWARD:.0%}", (x0 + 560, 580), 22, TEXT, True, anchor="topright")
        text(s, "joints · joint speed · feet · pose · velocity", (x0 + 16, 616), 12, FAINT)
        terminated = self.vals[3] >= 100 or self.vals[4] >= 90
        # robot reacting to the errors
        rrect(s, CARD, (984, 140, 236, 516), 18)
        text(s, "THE ROBOT", (1004, 154), 12, C4, True)
        f = int(self.t * 40) % 60
        rt = getattr(self, "_rt", None) or K.retarget("trot", "dog", 60)
        self._rt = rt
        wob = math.radians(self.vals[0]) * math.sin(self.t * 9)
        pitch = math.radians(self.vals[4]) * (1 if not terminated else 1)
        with sublayer(s, (984, 170, 236, 300)) as L:
            gx = 1102 + min(self.vals[3], 120) * 0.6
            draw.line(L, GRID, (994, 420), (1210, 420), 2)
            draw.line(L, mix(CARD, C4, 0.5), (1102, 300), (1102, 440), 1)
            mini_go2(L, (gx, 420), 260, rt["joints"][f], CORAL if terminated else (230, 234, 242),
                     pitch=pitch, offset=wob)
        text(s, "thin line = reference body position", (1102, 470), 11, FAINT, anchor="midtop")
        if terminated:
            rrect(s, CORAL, (1000, 500, 204, 70), 12)
            para(s, "EPISODE ENDS: fell or drifted too far", 1012, 508, 184, 14, BG, True)
        else:
            para(s, "Errors shrink each term smoothly; the policy is pulled back towards the reference.",
                 1004, 500, 200, 13, MUTED)


# ============================================================ 16. dataset explorer
class KineDatasetSlide(Slide):
    kicker, title = "Article 4 · Kine2Go · the dataset", "Twenty headings per policy: coverage you can see"
    kicker_color = C4
    hints = "M: motion (circle / strafe / trot)   ·   click the field to add a rollout aimed there   ·   C: clear"
    MOTIONS = ["circle", "strafe", "trot"]
    FX, FY, FW = 60, 140, 516

    def enter(self):
        super().enter()
        self.mi = 0
        self.load()

    def load(self):
        self.paths = K.rollout_paths(self.MOTIONS[self.mi], 20, seed=self.mi)
        self.t0 = self.t
        self.extra = []

    def key(self, k):
        if k == pygame.K_m:
            self.mi = (self.mi + 1) % 3
            self.load()
        elif k == pygame.K_c:
            self.paths, self.extra = [], []
        else:
            return False
        return True

    def click(self, pos, button):
        if inside(pos, (self.FX, self.FY, self.FW, self.FW)):
            x, y = (pos[0] - self.FX - self.FW / 2) / 60, -(pos[1] - self.FY - self.FW / 2) / 60
            base = K.rollout_paths(self.MOTIONS[self.mi], 1, seed=len(self.paths) + 99)[0]
            end = base[len(base) // 2] if self.MOTIONS[self.mi] == "circle" else base[-1]
            h = math.atan2(y, x) - math.atan2(end[1], end[0])
            Rm = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
            self.paths.append(base @ Rm.T)
            self.extra.append(len(self.paths) - 1)
            return True
        return False

    def heading(self, p):
        d = p[min(20, len(p) - 1)] - p[0]
        return math.atan2(d[1], d[0])

    def draw(self, s):
        cx, cy = self.FX + self.FW / 2, self.FY + self.FW / 2
        rrect(s, CARD, (self.FX, self.FY, self.FW, self.FW), 18)
        for r in (1, 2, 3, 4):
            draw.circle(s, GRID, (cx, cy), r * 60, 1)
        draw.line(s, GRID, (self.FX + 10, cy), (self.FX + self.FW - 10, cy))
        draw.line(s, GRID, (cx, self.FY + 10), (cx, self.FY + self.FW - 10))
        text(s, "1 m rings · top-down", (self.FX + 16, self.FY + 12), 12, FAINT)
        el = self.t - self.t0
        for i, p in enumerate(self.paths):
            g = min(1.0, max(0.0, (el - i * 0.12) / 1.5)) if i not in self.extra else 1.0
            n = max(2, int(len(p) * g))
            pts = [(cx + q[0] * 60, cy - q[1] * 60) for q in p[:n]]
            col = AMBER if i in self.extra else mix(CARD, C4, 0.4 + 0.6 * (i % 5) / 4)
            draw.lines(s, col, False, pts, 2)
            circle(s, col, pts[-1], 4)
        circle(s, TEXT, (cx, cy), 6)
        # ---- heading coverage rose
        x0 = 600
        rrect(s, CARD, (x0, 140, 300, 300), 18)
        text(s, "START HEADINGS", (x0 + 20, 154), 12, C4, True)
        bins = np.zeros(12)
        for p in self.paths:
            bins[int(((self.heading(p) + 2 * math.pi) % (2 * math.pi)) / (2 * math.pi) * 12) % 12] += 1
        rx, ry = x0 + 150, 300
        for b in range(12):
            a0 = b / 12 * 2 * math.pi
            a1 = (b + 1) / 12 * 2 * math.pi
            r = 20 + 100 * bins[b] / max(4, bins.max())
            pts = [(rx, ry)] + [(rx + r * math.cos(a0 + (a1 - a0) * t / 6), ry - r * math.sin(a0 + (a1 - a0) * t / 6))
                                for t in range(7)]
            poly(s, mix(CARD, C4, min(1, 0.3 + bins[b] / 4)) if bins[b] else GRID, pts)
        empty = int((bins == 0).sum())
        text(s, f"{empty} of 12 directions empty", (x0 + 150, 412), 13, CORAL if empty else GREEN, True,
             anchor="midtop")
        # ---- frame inspector
        x1 = 920
        rrect(s, CARD, (x1, 140, 300, 300), 18)
        text(s, "ONE LOGGED FRAME (illustrative)", (x1 + 20, 154), 12, C4, True)
        rt = getattr(self, "_rt", None) or K.retarget("trot", "dog", 60)
        self._rt = rt
        f = int(self.t * 30) % 60
        p = self.paths[-1] if self.paths else np.zeros((240, 2))
        k = int(self.t * 30) % len(p)
        q = rt["joints"][f]
        rows = [("t", f"{k * 0.02:.2f} s"), ("root x, y", f"{p[k][0]:+.2f}, {p[k][1]:+.2f} m"),
                ("heading", f"{math.degrees(self.heading(p)):+.0f}°")]
        rows += [(f"{leg} hip/knee", f"{math.degrees(q[leg][0]):+5.1f}°  {math.degrees(q[leg][1]):5.1f}°")
                 for leg in K.LEGS]
        rows += [("actions", "12 joint targets")]
        for j, (a, b) in enumerate(rows):
            y = 182 + j * 30
            text(s, a, (x1 + 20, y), 13, MUTED)
            text(s, b, (x1 + 280, y), 13, TEXT, mono=True, anchor="topright")
        # ---- numbers
        D = self.app.context["summary"]["kine2go"]["dataset"]
        rrect(s, CARD2, (600, 456, 620, 200), 18)
        stats = [(f"{D['policies']}", "policies"), (f"{D['rollouts_per_policy']}", "rollouts each"),
                 (f"{D['trajectories']}", "trajectories")]
        for j, (v, lab) in enumerate(stats):
            x = 640 + j * 200
            text(s, v, (x, 476), 40, C4, True)
            text(s, lab, (x, 528), 14, MUTED)
        para(s, "Random headings stop a learned model from thinking the robot only ever walks one way. "
                "Click the empty directions on the field to fill the rose; it is the same idea "
                "(paper Fig. 4b).", 620, 566, 580, 14, TEXT)
