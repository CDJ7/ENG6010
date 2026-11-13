"""
slides_l2r.py - Article 3 (Language to Rewards): the Motion Descriptor
template being filled in, a phase-plane lab where the reward landscape and
the optimiser's 256 sampled futures are visible, and results as rings.
"""

import math
import os
import re
import sys

import numpy as np

from engine import *  # noqa: F401,F403
from engine import Slide, SimMixin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from demos import l2r_demo as R  # noqa: E402

C3 = VIOLET
CODE_TXT = (220, 227, 239)
TERM_COLORS = {"angle": AMBER, "speed": SKY, "effort": TEAL}
SHORT = ["swing up", "lean 20°", "spin fast", "hang still", "horizontal"]
LAND_STOPS = [(12, 14, 34), (46, 30, 90), (110, 70, 180), (190, 150, 250), (250, 240, 255)]


def term_line(spec, name):
    t = spec.terms[name]
    if name == "angle":
        return f"set_angle_reward({t.target:g}, weight={t.weight:g})"
    if name == "speed":
        return f"set_speed_reward({t.target:g}, weight={t.weight:g})"
    return f"set_effort_reward(weight={t.weight:g})"


def slot_values(spec):
    """What the Motion Descriptor puts into each template slot for this reward."""
    a, sp, ef = spec.terms["angle"], spec.terms["speed"], spec.terms["effort"]
    spin = sp.target != 0
    return [None,
            f"{a.target:g}" if a.weight > 0 else "—",
            "spin" if spin else "hold still",
            f"{sp.target:g}" if spin else "—",
            "kept" if ef.weight > 0 else "dropped",
            None]


# ============================================================ 10. idea: fill the template
class L2RIdeaSlide(Slide, SimMixin):
    kicker, title = "Article 3 · Language to Rewards", "Fill in a template, write a reward, let the optimiser move"
    kicker_color = C3
    hints = "1-5 or click: pick an instruction and watch the three stages run   ·   → next: the reward lab"

    def enter(self):
        super().enter()
        self.ii = 0
        self.load()

    def load(self):
        self.t0 = self.t
        ins = R.INSTRUCTIONS[self.ii]
        self.spec = R.run_reward_code(ins["reward_code"])
        self.ctrl = R.PredictiveSampling(self.spec, samples=128, seed=self.ii)
        self.th, self.w, self.u = math.pi - 0.05, 0.0, 0.0
        self.sim_reset()

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_5:
            self.ii = k - pygame.K_1
            self.load()
            return True
        return False

    def click(self, pos, button):
        for i, r in enumerate(getattr(self, "chips", [])):
            if inside(pos, r):
                self.ii = i
                self.load()
                return True
        return False

    def update(self, dt):
        super().update(dt)
        if self.t - self.t0 > 4.6:
            for _ in range(self.sim_ticks(dt, max_steps=1)):
                self.u = self.ctrl.act(self.th, self.w)
                th, w = R.step(self.th, self.w, self.u)
                self.th, self.w = float(th), float(w)

    def draw(self, s):
        el = self.t - self.t0
        ins = R.INSTRUCTIONS[self.ii]
        self.chips = []
        x = 60
        for i, lab in enumerate(SHORT):
            r = chip(s, x, 136, f"{i + 1} {lab}", C3, i == self.ii, 13)
            self.chips.append(r)
            x += r[2] + 8
        text(s, f"\"{ins['text']}\"", (1220, 142), 16, TEXT, True, anchor="topright")
        # ---- stage 1: Motion Descriptor fills the template
        rrect(s, CARD, (60, 186, 470, 470), 18)
        icon(s, "language", (88, 214), 34, C3)
        text(s, "1 · MOTION DESCRIPTOR (LLM)", (114, 204), 14, C3, True)
        text(s, "fills the slots of a fixed template", (114, 224), 12, MUTED)
        vals = slot_values(self.spec)
        y = 262
        for j, line in enumerate(R.TEMPLATE):
            m = re.search(r"\[NUM:[^\]]*\]|\{CHOICE:[^}]*\}", line)
            t_fill = 0.5 + 0.5 * j
            filled = el > t_fill
            if m and vals[j] is not None:
                before, slot, after = line[:m.start()], m.group(0), line[m.end():]
                show = vals[j] if filled else slot
                pieces = [(before, MUTED), (show, C3 if filled else AMBER), (after, MUTED)]
            elif line.startswith("[optional]"):
                kept = vals[4] == "kept"
                pieces = [(line.replace("[optional] ", ""), (MUTED if kept else FAINT) if filled else MUTED)]
            else:
                pieces = [(line, FAINT)]
            # wrap the line manually (pieces keep their colours)
            xx = 84
            for txt, col in pieces:
                chunks = [txt] if col in (C3, AMBER) else re.split(r"(\s+)", txt)   # a slot stays in one piece
                for word in chunks:
                    if not word:
                        continue
                    ww = text_width(word, 14)
                    if xx + ww > 506 and word.strip():
                        xx, y = 84, y + 22
                    if word.strip():
                        if col in (C3, AMBER):
                            rrect(s, mix(CARD, col, 0.25), (xx - 3, y - 2, ww + 6, 21), 5)
                        text(s, word, (xx, y), 14, col, col == C3)
                    xx += ww
            if line.startswith("[optional]") and filled:
                text(s, vals[4], (506, y), 13, GREEN if vals[4] == "kept" else FAINT, True, anchor="topright")
            y += 32
        text(s, "THE PAPER'S QUADRUPED TEMPLATE (abridged)", (84, 462), 11, FAINT, True)
        rrect(s, CODE_BG, (78, 480, 434, 82), 10)
        for j, ln in enumerate(["The torso should pitch upward at [NUM: 0.0] degrees.",
                                "The height of the torso center should be at [NUM: 0.3] m.",
                                "The robot should {CHOICE: face a direction, turn}."]):
            text(s, ln, (90, 490 + j * 23), 13, (200, 205, 225))
        para(s, "The template keeps the LLM's answer inside what the reward API can express. "
                "Paper: without it, success drops sharply on hard tasks.", 84, 578, 420, 13, MUTED)
        # ---- stage 2: Reward Coder writes code
        arrow(s, C3 if el > 3.0 else GRID, (536, 420), (560, 420), 3, 9)
        rrect(s, CARD, (566, 186, 360, 470), 18)
        icon(s, "code", (594, 214), 34, C3)
        text(s, "2 · REWARD CODER (LLM)", (620, 204), 14, C3, True)
        text(s, "turns the description into API calls", (620, 224), 12, MUTED)
        rrect(s, CODE_BG, (584, 252, 324, 200), 12)
        chars = int(max(0, el - 3.0) * 60)
        used = 0
        for j, line in enumerate(ins["reward_code"]):
            part = line[:max(0, chars - used)]
            used += len(line)
            col = CODE_TXT
            for name, c in TERM_COLORS.items():
                if f"set_{name}" in line:
                    col = c
            text(s, part, (600, 266 + j * 30), 13, col, mono=True)
        text(s, "R(s, a) = − Σ  wᵢ · rᵢ²", (746, 476), 22, TEXT, True, anchor="midtop")
        para(s, "r = how far a quantity is from its target ψ; w = how much the term matters (paper Eq. 1).",
             590, 516, 316, 13, MUTED)
        for k, (name, c) in enumerate(TERM_COLORS.items()):
            t = self.spec.terms[name]
            circle(s, c if t.weight > 0 else GRID, (600 + k * 104, 592), 6)
            text(s, f"{name} {t.weight:g}", (612 + k * 104, 584), 13, c if t.weight > 0 else FAINT)
        # ---- stage 3: the optimiser
        arrow(s, C3 if el > 4.6 else GRID, (932, 420), (956, 420), 3, 9)
        rrect(s, CARD, (962, 186, 258, 470), 18)
        icon(s, "cogs", (990, 214), 34, C3)
        text(s, "3 · MOTION", (1016, 198), 14, C3, True)
        text(s, "CONTROLLER", (1016, 216), 14, C3, True)
        para(s, "MuJoCo MPC in the paper; predictive sampling here", 984, 244, 220, 12, MUTED)
        cx, cy = 1091, 440
        if el > 4.6:
            a = self.spec.terms["angle"]
            if a.weight > 0:
                ta = math.radians(a.target)
                draw_pendulum(s, (cx, cy), 100, (math.cos(ta), math.sin(ta)), AMBER, 8, pivot=False, alpha=90)
            draw_pendulum(s, (cx, cy), 100, (math.cos(self.th), math.sin(self.th)), (230, 234, 242), 12,
                          torque=self.u)
            text(s, f"{math.degrees(R.wrap(self.th)):+.0f}°", (cx, 600), 18, TEXT, True, anchor="midtop")
        else:
            text(s, "waiting for", (cx, 420), 14, FAINT, anchor="center")
            text(s, "a reward…", (cx, 442), 14, FAINT, anchor="center")
        if "note" in ins:
            text(s, "can't be held: motor too weak", (cx, 626), 12, CORAL, anchor="midtop")


# ============================================================ 11. the reward lab
CORR_SHORT = ["further right", "steadier", "less energy", "spin faster", "lean the other way"]


class L2RLabSlide(Slide, SimMixin):
    kicker, title = "Article 3 · reward lab", "See the reward landscape and the optimiser's 256 guesses"
    kicker_color = C3
    hints = ("Q: next instruction · 1-5: say a correction · click the plane: drop the pole in that state · "
             "P: primitives · K: kick · R: reset")
    PX0, PY0, PW, PH = 60, 140, 640, 420

    def enter(self):
        super().enter()
        self.ii, self.mode = 1, "reward"
        self.load()

    def load(self):
        ins = R.INSTRUCTIONS[self.ii]
        self.spec = R.run_reward_code(ins["reward_code"])
        self.ctrl = R.PredictiveSampling(self.spec, seed=self.ii, record=True)
        self.th, self.w, self.u = math.pi - 0.05, 0.0, 0.0
        self.trail = []
        self.diff = None
        self.said = []
        self.relandscape()
        self.sim_reset()

    def relandscape(self):
        Rl, _, _ = R.reward_landscape(self.spec, 120, 72)
        a = 1 - np.sqrt((Rl.max() - Rl) / (Rl.max() - Rl.min() + 1e-9))
        self.img = colormap(a[::-1], LAND_STOPS)

    def key(self, k):
        if k == pygame.K_q:
            self.ii = (self.ii + 1) % 5
            self.load()
        elif pygame.K_1 <= k <= pygame.K_5:
            corr = R.CORRECTIONS[k - pygame.K_1]
            old = term_line(self.spec, corr[1])
            new = R.apply_correction(self.spec, corr)
            self.diff = (old, new, self.t, corr[0])
            self.said = (self.said + [corr[0]])[-3:]
            self.relandscape()
        elif k == pygame.K_p:
            self.mode = "cap" if self.mode == "reward" else "reward"
        elif k == pygame.K_k:
            self.w = float(np.clip(self.w + np.random.choice([-6, 6]), -8, 8))
        elif k == pygame.K_r:
            self.load()
        else:
            return False
        return True

    def click(self, pos, button):
        if inside(pos, (self.PX0, self.PY0, self.PW, self.PH)):
            self.th = (pos[0] - self.PX0) / self.PW * 2 * math.pi - math.pi
            self.w = (1 - (pos[1] - self.PY0) / self.PH) * 2 * R.MAX_SPEED - R.MAX_SPEED
            self.trail = []
            return True
        return False

    def update(self, dt):
        super().update(dt)
        ins = R.INSTRUCTIONS[self.ii]
        for _ in range(self.sim_ticks(dt, max_steps=1)):
            if self.mode == "reward":
                self.u = self.ctrl.act(self.th, self.w)
            else:
                self.u = R.PRIMITIVES[ins["cap_fn"]](self.th, self.w)
                self.ctrl.rollouts = None
            th, w = R.step(self.th, self.w, self.u)
            self.th, self.w = float(th), float(w)
            self.trail = (self.trail + [(self.th, self.w)])[-80:]

    def pp(self, th, w):
        x = self.PX0 + (R.wrap(th) + math.pi) / (2 * math.pi) * self.PW
        y = self.PY0 + (1 - (w + R.MAX_SPEED) / (2 * R.MAX_SPEED)) * self.PH
        return x, y

    def polyline(self, s, ths, ws, col, width):
        seg = []
        last = None
        for th, w in zip(ths, ws):
            p = self.pp(th, w)
            if last is not None and abs(p[0] - last[0]) > self.PW / 2:
                if len(seg) > 1:
                    draw.lines(s, col, False, seg, width)
                seg = []
            seg.append(p)
            last = p
        if len(seg) > 1:
            draw.lines(s, col, False, seg, width)

    def draw(self, s):
        ins = R.INSTRUCTIONS[self.ii]
        # ---------------- phase plane
        blit_array(s, self.img, (self.PX0, self.PY0, self.PW, self.PH))
        draw.rect(s, GRID, (self.PX0, self.PY0, self.PW, self.PH), 2)
        for deg in (-180, -90, 0, 90, 180):
            x = self.PX0 + (deg + 180) / 360 * self.PW
            draw.line(s, (60, 70, 100), (x, self.PY0), (x, self.PY0 + self.PH), 1)
            text(s, f"{deg}°", (x, self.PY0 + self.PH + 4), 12, MUTED, anchor="midtop")
        y0 = self.pp(0, 0)[1]
        draw.line(s, (60, 70, 100), (self.PX0, y0), (self.PX0 + self.PW, y0), 1)
        text(s, "pole angle (0 = up)", (self.PX0 + self.PW, self.PY0 + self.PH + 22), 12, FAINT, anchor="topright")
        text(s, "spin speed ↑", (self.PX0 + 8, self.PY0 + 6), 12, TEXT)
        ro = self.ctrl.rollouts if self.mode == "reward" else None
        if ro is not None:
            n = ro["theta"].shape[0]
            for i in range(0, n, 6):
                self.polyline(s, ro["theta"][i], ro["thdot"][i], (120, 110, 160), 1)
            b = ro["best"]
            self.polyline(s, ro["theta"][b], ro["thdot"][b], GREEN, 3)
        if len(self.trail) > 1:
            self.polyline(s, [p[0] for p in self.trail], [p[1] for p in self.trail], AMBER, 2)
        circle(s, TEXT, self.pp(self.th, self.w), 8)
        circle(s, AMBER, self.pp(self.th, self.w), 4)
        # legend under the plane
        ly = 600
        for k, (lab, col) in enumerate([("bright = high reward", (250, 240, 255)),
                                         ("grey = 43 of the 256 sampled futures", (120, 110, 160)),
                                         ("green = the best one (its 1st action is used)", GREEN),
                                         ("amber = what actually happened", AMBER)]):
            x = 60 + (k % 2) * 330
            yy = ly + (k // 2) * 22
            rrect(s, col, (x, yy + 4, 16, 8), 3)
            text(s, lab, (x + 24, yy), 13, MUTED)
        # ---------------- right column
        x0 = 726
        rrect(s, CARD, (x0, 140, 494, 196), 16)
        text(s, f"INSTRUCTION {self.ii + 1}/5 (Q)", (x0 + 20, 154), 12, C3, True)
        para(s, f"\"{ins['text']}\"", x0 + 20, 172, 280, 15, TEXT, True)
        cx, cy = x0 + 400, 250
        a = self.spec.terms["angle"]
        if a.weight > 0 and self.mode == "reward":
            ta = math.radians(a.target)
            draw_pendulum(s, (cx, cy), 70, (math.cos(ta), math.sin(ta)), AMBER, 7, pivot=False, alpha=90)
        draw_pendulum(s, (cx, cy), 70, (math.cos(self.th), math.sin(self.th)), (230, 234, 242), 10, torque=self.u)
        toggle(s, x0 + 20, 290, self.mode == "reward", "reward + MPC" if self.mode == "reward"
               else "fixed primitive: " + ins["cap_fn"] + "()", C3)
        # code with the live diff
        rrect(s, CODE_BG, (x0, 348, 494, 132), 14)
        text(s, "REWARD CODE (live)", (x0 + 18, 358), 12, C3, True)
        for j, name in enumerate(("angle", "speed", "effort")):
            line = term_line(self.spec, name)
            y = 382 + j * 30
            changed = self.diff and self.diff[1] == line and self.t - self.diff[2] < 5
            if changed:
                rrect(s, (24, 60, 44), (x0 + 12, y - 3, 470, 24), 5)
                text(s, "+", (x0 + 18, y), 14, GREEN, True, mono=True)
            w = self.spec.terms[name].weight
            text(s, line, (x0 + 34, y), 14, TERM_COLORS[name] if w > 0 else FAINT, mono=True)
        if self.diff and self.t - self.diff[2] < 5:
            text(s, "− " + self.diff[0], (x0 + 18, 462), 12, CORAL, mono=True)
        # corrections
        rrect(s, CARD, (x0, 492, 494, 164), 16)
        text(s, "SAY A CORRECTION (paper Sec. 4.5)", (x0 + 20, 504), 12, C3, True)
        for i, lab in enumerate(CORR_SHORT):
            xx = x0 + 20 + (i % 3) * 156
            yy = 528 + (i // 3) * 36
            rrect(s, CARD2, (xx, yy, 148, 28), 14)
            text(s, f"{i + 1}  {lab}", (xx + 12, yy + 6), 13, TEXT)
        if self.said:
            text(s, "you: \"" + self.said[-1] + "\"", (x0 + 20, 610), 14, C3, True)
            text(s, "→ the LLM edits one line of the reward; the optimiser adapts at once", (x0 + 20, 632), 12,
                 MUTED)
        else:
            para(s, "Each correction edits one weight or target, like the paper's chat with the robot.",
                 x0 + 20, 608, 450, 13, MUTED)


# ============================================================ 12. results: rings + matrix + chat
class L2RResultsSlide(Slide):
    kicker, title = "Article 3 · results", "Rewards reach tasks a fixed skill library cannot"
    kicker_color = C3
    hints = "N: next turn of the paper's moonwalk conversation   ·   rings = paper, ticks = my pendulum"
    CHAT = [("Make the robot stand upright on two back feet like a human.", "set_torso_targets(0.65, 90°, …)"),
            ("You don't need to keep the front paws at a certain height; leave it to the controller.",
             "front-paw terms removed"),
            ("Now make the robot do a moonwalk while standing up like this.", "velocity = (−0.2, 0)"),
            ("Moonwalk means walking backward while the feet swing as if moving forward. Also go 0.5 m/s.",
             "velocity = (−0.5, 0) · feet swing +0.2")]

    def enter(self):
        super().enter()
        self.n, self.t_n = 1, 0.0

    def key(self, k):
        if k == pygame.K_n:
            self.n = self.n % 4 + 1
            self.t_n = self.t
            return True
        return False

    def draw(self, s):
        S = self.app.context["summary"]["l2r"]
        # ---- paper rings
        rrect(s, CARD, (60, 140, 400, 300), 18)
        text(s, "PAPER · 17 tasks, quadruped + dexterous hand", (82, 156), 12, C3, True)
        for k, (lab, v, col) in enumerate([("reward + MPC", 0.90, C3), ("Code-as-Policies", 0.50, GRAY)]):
            cx = 160 + k * 200
            g = ease((self.t - 0.2 - 0.3 * k) / 1.2)
            ring(s, (cx, 290), 70, v * g, col, 16)
            text(s, f"{v * g:.0%}", (cx, 290), 30, TEXT, True, anchor="center")
            text(s, lab, (cx, 376), 14, col, True, anchor="midtop")
        text(s, "of tasks solved", (260, 404), 12, FAINT, anchor="midtop")
        # ---- my matrix
        rrect(s, CARD, (60, 456, 400, 200), 18)
        text(s, "MY PENDULUM · 10 starts each", (82, 470), 12, C3, True)
        text(s, "reward", (300, 492), 12, C3, True, anchor="midtop")
        text(s, "primitives", (390, 492), 12, GRAY, True, anchor="midtop")
        for i, row in enumerate(S["tasks"]):
            y = 512 + i * 26
            text(s, SHORT[i], (82, y), 14, TEXT)
            for j, m in enumerate(("reward", "cap")):
                v = row[m]
                cx = 300 + j * 90
                col = GREEN if v >= 0.9 else (AMBER if v > 0 else CORAL)
                mark(s, v >= 0.9, (cx - 24, y + 8), 12, col)
                text(s, f"{v:.0%}", (cx - 12, y), 14, col, True)
        # ---- horizon sparkline + horizontal-pose note
        hs = S["horizon_sweep"]
        x0 = 480
        rrect(s, CARD, (x0, 140, 330, 300), 18)
        text(s, "PLANNING HORIZON (swing-up success)", (x0 + 20, 156), 12, C3, True)
        ks = sorted(hs, key=lambda k: int(k))
        pts = []
        for i, k in enumerate(ks):
            x = x0 + 50 + i * 60
            y = 400 - hs[k] * 190
            pts.append((x, y))
            text(s, f"{k}", (x, 410), 12, MUTED, anchor="midtop")
        draw.lines(s, C3, False, pts, 3)
        for (x, y), k in zip(pts, ks):
            circle(s, C3, (x, y), 6)
            text(s, f"{hs[k]:.0%}", (x, y - 12), 12, TEXT, True, anchor="midbottom")
        text(s, "steps ahead (×0.05 s)", (x0 + 165, 424), 11, FAINT, anchor="midtop")
        rrect(s, CARD, (x0, 456, 330, 200), 18)
        lean = S["tasks"][4].get("achieved_lean_deg", 0)
        text(s, "THE IMPOSSIBLE REQUEST", (x0 + 20, 470), 12, CORAL, True)
        para(s, f"\"Hold the pole horizontal\" needs 5 N·m; the motor has 2. The optimiser settled at "
                f"{lean:.1f}° from up: the closest pose it can hold (theory 156.4°). The reward said what, "
                "physics said how much.", x0 + 20, 494, 290, 14, TEXT)
        # ---- moonwalk chat
        x1 = 830
        rrect(s, CARD, (x1, 140, 390, 516), 18)
        text(s, "PAPER'S MOONWALK CHAT (Fig. 4)", (x1 + 20, 156), 12, C3, True)
        y = 184
        for i in range(self.n):
            user, change = self.CHAT[i]
            a = ease((self.t - self.t_n) / 0.4) if i == self.n - 1 else 1
            lines = wrap(user, 13, 290)
            h = len(lines) * 18 + 14
            bx = x1 + 70
            rrect(s, (45, 70, 110), (bx, y, 300, h), 12, alpha=int(255 * a))
            for j, ln in enumerate(lines):
                text(s, ln, (bx + 10, y + 7 + j * 18), 13, TEXT, alpha=int(255 * a))
            y += h + 6
            circle(s, C3, (x1 + 36, y + 13), 13)
            text(s, "R", (x1 + 36, y + 13), 13, BG, True, anchor="center")
            cw = text_width(change, 12, False, True) + 20
            rrect(s, CODE_BG, (x1 + 56, y, min(cw, 320), 26), 10)
            text(s, change, (x1 + 66, y + 6), 12, C3, mono=True, alpha=int(255 * a))
            y += 40
        if self.n == 4:
            para(s, "Each reply edits the reward, not the motion; MPC re-optimises in real time.",
                 x1 + 20, 600, 350, 13, MUTED)
