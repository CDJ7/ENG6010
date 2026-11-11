"""
slides_l2r.py - Article 3 (Language to Rewards) slides.
"""

import math
import os
import sys

import numpy as np

from engine import *  # noqa: F401,F403
from engine import SIM_DT, SimMixin, Slide

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from demos import l2r_demo as R  # noqa: E402

C3 = VIOLET


# ============================================================ 9. L2R idea
class L2RIdeaSlide(Slide):
    kicker, title = "Article 3 · Language to Rewards", "The LLM writes the reward; an optimiser finds the motion"
    kicker_color = C3
    hints = "The moonwalk conversation plays out by itself (N: restart / skip)   ·   → next: steer a robot with rewards"
    CHAT = [("Make the robot stand upright on two back feet like a human.",
             "set_torso_targets(0.65, deg2rad(90), ...)  # stand up"),
            ("Good, you actually don't need to keep the front paws at certain height, just leave it to the controller.",
             "# front-paw height terms removed"),
            ("Good, now make the robot do a moonwalk while stand up like this.",
             "velocity=(-0.2, 0.0) · back feet stepping"),
            ("Moon walk means the robot should walk backward while the feet swings as if they are moving forward. "
             "Correct your answer and also make it walk at a speed of 0.5m/s.",
             "velocity=(-0.5, 0.0) · feet swing forward (+0.2)")]

    def enter(self):
        super().enter()
        self.n = 1
        self.t_n = 0.0

    def key(self, k):
        if k == pygame.K_n:
            self.n = self.n % 4 + 1
            self.t_n = self.t
            return True
        return False

    def update(self, dt):
        super().update(dt)
        if self.n < 4 and self.t - self.t_n > 2.5:       # the conversation plays out by itself
            self.n += 1
            self.t_n = self.t

    def draw(self, s):
        # pipeline
        stages = [("comments", "User", "\"Make robot dog stand up on two feet.\""),
                  ("language", "Motion Descriptor (LLM)", "fills a motion template: torso pitch 90°, height 0.65 m, ..."),
                  ("code", "Reward Coder (LLM)", "turns it into reward code: set_torso_rewards(...), set_feet_..."),
                  ("cogs", "Motion Controller", "MuJoCo MPC optimises actions for that reward in real time")]
        f = (self.t * 0.4) % 1.0
        for i, (ic, h1, body) in enumerate(stages):
            a = self.appear(0.15 * i)
            y = 150 + i * 102 + 12 * (1 - a)
            rrect(s, CARD, (60, y, 560, 88), 14)
            icon(s, ic, (104, y + 44), 52, C3)
            text(s, h1, (144, y + 12), 19, TEXT, True)
            para(s, body, 144, y + 42, 456, 15, MUTED)
            if i < 3:
                arrow(s, FAINT, (340, y + 89), (340, y + 101), 2, 6)
        circle(s, C3, (612, 150 + f * 394), 5)
        rrect(s, CODE_BG, (60, 562, 560, 58), 12)
        text(s, "R(s, a) = − Σᵢ wᵢ · nᵢ( rᵢ(s, a, ψᵢ) )", (84, 572), 18, C3, True)
        text(s, "weights wᵢ and targets ψᵢ are what the LLM chooses", (84, 598), 13, MUTED)
        # the moonwalk conversation (paper Fig. 5a / Appendix A.9)
        x0 = 650
        rrect(s, CARD, (x0, 150, 570, 470), 16)
        text(s, "Interactive correction: teaching a moonwalk (paper Fig. 5a)", (x0 + 22, 164), 15, C3, True)
        y = 196
        for i in range(self.n):
            user, code = self.CHAT[i]
            a = ease((self.t - self.t_n) / 0.5) if i == self.n - 1 else 1.0
            ylines = wrap(user, 14, 500)
            h = 14 + 19 * len(ylines)
            rrect(s, CARD2, (x0 + 22, y, 526, h), 10, alpha=int(255 * a))
            para(s, user, x0 + 34, y + 7, 500, 14, TEXT, alpha=int(255 * a))
            y += h + 6
            rrect(s, CODE_BG, (x0 + 50, y, 498, 30), 8, alpha=int(255 * a))
            text(s, code[:64], (x0 + 60, y + 7), 12, (200, 210, 225), mono=True, alpha=int(255 * a))
            y += 40
        para(s, "Because MPC re-optimises in real time, each correction changes the robot's motion immediately.",
             x0 + 22, 572, 526, 14, MUTED)


# ============================================================ 10. L2R demo
TERM_COLORS = {"angle": AMBER, "speed": SKY, "effort": TEAL}


class L2RDemoSlide(Slide, SimMixin):
    kicker, title = "Article 3 · live demo", "New reward, new motion: same optimiser"
    kicker_color = C3
    hints = ("1-5: instruction · Tab: pick a term · W/S: weight · A/D: target · P: primitives baseline · "
             "K: kick · R: reset")

    def enter(self):
        super().enter()
        self.ii, self.mode, self.term = 0, "reward", 0
        self.load()

    def load(self):
        instr = R.INSTRUCTIONS[self.ii]
        self.spec = R.run_reward_code(instr["reward_code"])
        self.ctrl = R.PredictiveSampling(self.spec, seed=self.ii)
        self.th, self.w, self.u = math.pi, 0.0, 0.0
        self.hist_th, self.hist_w = [], []
        self.edited = False
        self.sim_reset()

    def key(self, k):
        names = ["angle", "speed", "effort"]
        t = self.spec.terms[names[self.term]]
        if pygame.K_1 <= k <= pygame.K_5:
            self.ii = k - pygame.K_1
            self.load()
        elif k == pygame.K_TAB:
            self.term = (self.term + 1) % 3
        elif k == pygame.K_w:
            t.weight = round(min(2.0, t.weight + (0.05 if t.name != "effort" else 0.01)), 3)
            self.edited = True
        elif k == pygame.K_s:
            t.weight = round(max(0.0, t.weight - (0.05 if t.name != "effort" else 0.01)), 3)
            self.edited = True
        elif k in (pygame.K_a, pygame.K_d) and t.name != "effort":
            step = 10 if t.name == "angle" else 40
            t.target += step if k == pygame.K_d else -step
            self.edited = True
        elif k == pygame.K_p:
            self.mode = "cap" if self.mode == "reward" else "reward"
            self.th, self.w = math.pi, 0.0
        elif k == pygame.K_k:
            self.w = float(np.clip(self.w + np.random.choice([-5, 5]), -8, 8))
        elif k == pygame.K_r:
            self.load()
        else:
            return False
        return True

    def update(self, dt):
        super().update(dt)
        instr = R.INSTRUCTIONS[self.ii]
        for _ in range(self.sim_ticks(dt, max_steps=1)):
            if self.mode == "reward":
                self.u = self.ctrl.act(self.th, self.w)
            else:
                self.u = R.PRIMITIVES[instr["cap_fn"]](self.th, self.w)
            th, w = R.step(self.th, self.w, self.u)
            self.th, self.w = float(th), float(w)
            self.hist_th.append(self.th)
            self.hist_w.append(self.w)
            self.hist_th, self.hist_w = self.hist_th[-60:], self.hist_w[-60:]

    def draw(self, s):
        instr = R.INSTRUCTIONS[self.ii]
        # left: the robot
        cx, cy, Rr = 260, 400, 160
        rrect(s, CARD, (60, 150, 400, 470), 16)
        text(s, f"INSTRUCTION {self.ii + 1}/5", (82, 164), 13, C3, True)
        para(s, f"\"{instr['text']}\"", 82, 186, 360, 17, TEXT, True)
        if self.mode == "reward":
            # the plan the optimiser currently believes in
            a, b = self.th, self.w
            pts = []
            for u in self.ctrl.plan:
                a, b = R.step(a, b, u)
                pts.append((cx + Rr * math.sin(float(a)), cy - Rr * math.cos(float(a))))
            if len(pts) > 1:
                draw.lines(s, (*LAV,), False, pts, 2)
        tgt = self.spec.terms["angle"]
        if tgt.weight > 0 and self.mode == "reward":
            ta = math.radians(tgt.target)
            draw_pendulum(s, (cx, cy), Rr, (math.cos(ta), math.sin(ta)), AMBER, 10, pivot=False, alpha=90)
        draw_pendulum(s, (cx, cy), Rr, (math.cos(self.th), math.sin(self.th)), (230, 234, 242), 16, torque=self.u)
        ok = len(self.hist_th) >= 60 and bool(instr["check"](np.array(self.hist_th), np.array(self.hist_w)))
        lab, col = ("Goal met (last 3 s)", GREEN) if ok else ("Not there yet", FAINT)
        bw = text_width(lab, 14, True) + 24
        rrect(s, col, (82, 586, bw, 26), 13)
        text(s, lab, (82 + bw / 2, 599), 14, BG, True, anchor="center")
        text(s, "amber ghost = target · lilac = planned motion", (440, 594), 12, FAINT, anchor="topright")

        # middle: LLM outputs or the primitive program
        x0 = 480
        if self.mode == "reward":
            rrect(s, CODE_BG, (x0, 150, 370, 210), 14)
            text(s, "Motion Descriptor output (example)", (x0 + 18, 162), 13, C3, True)
            y = 186
            for line in instr["description"]:
                y = para(s, line, x0 + 18, y, 336, 13, (200, 210, 225), gap=2)
            rrect(s, CODE_BG, (x0, 372, 370, 248), 14)
            text(s, "Reward Coder output (example)" + ("  · edited by you" if self.edited else ""),
                 (x0 + 18, 384), 13, C3, True)
            for j, line in enumerate(instr["reward_code"]):
                text(s, line, (x0 + 18, 410 + j * 24), 13, (220, 227, 239), mono=True)
            if instr.get("note"):
                para(s, instr["note"], x0 + 18, 540, 336, 13, CORAL)
        else:
            rrect(s, CODE_BG, (x0, 150, 370, 470), 14)
            text(s, "Code-as-Policies baseline", (x0 + 18, 162), 15, CORAL, True)
            para(s, "The LLM may only call skills from a fixed library:", x0 + 18, 190, 336, 14, MUTED)
            for j, name in enumerate(R.PRIMITIVES):
                text(s, f"• {name}()", (x0 + 18, 226 + j * 24), 14, TEXT, mono=True)
            text(s, "Program written for this instruction:", (x0 + 18, 300), 14, MUTED)
            for j, line in enumerate(instr["cap"]):
                text(s, line, (x0 + 18, 328 + j * 24), 13, (220, 227, 239), mono=True)
            para(s, "When no skill matches the request, the best it can do is run the closest one. "
                    "Rewards don't have that limit: the optimiser makes a new motion for any goal it can express.",
                 x0 + 18, 450, 336, 14, TEXT)

        # right: the reward terms
        x1 = 870
        rrect(s, CARD, (x1, 150, 350, 470), 14)
        text(s, "Reward terms  R = −Σ w·r²", (x1 + 18, 162), 15, TEXT, True)
        if self.mode == "cap":
            para(s, "(not used by the primitives baseline)", x1 + 18, 192, 316, 14, FAINT)
            return
        res = self.spec.residuals(self.th, self.w, self.u)
        for j, name in enumerate(["angle", "speed", "effort"]):
            t = self.spec.terms[name]
            y = 196 + j * 116
            on = j == self.term
            rrect(s, CARD2 if on else BG, (x1 + 14, y, 322, 104), 12)
            if on:
                draw.rect(s, TERM_COLORS[name], (x1 + 14, y, 322, 104), 2, border_radius=12)
            text(s, name, (x1 + 30, y + 10), 17, TERM_COLORS[name], True)
            text(s, f"weight w = {t.weight:g}", (x1 + 140, y + 12), 14, TEXT)
            tgt_txt = f"target ψ = {t.target:g} {t.unit}" if name != "effort" else "target ψ = 0"
            text(s, tgt_txt, (x1 + 30, y + 40), 14, MUTED)
            contrib = t.weight * float(res[name]) ** 2
            bw = min(190, 95 * contrib)
            rrect(s, TERM_COLORS[name], (x1 + 30, y + 72, max(2, bw), 14), 5)
            text(s, f"cost now {contrib:.2f}", (x1 + 320, y + 70), 12, MUTED, anchor="topright")
        para(s, "Tab selects a term. W/S change its weight, A/D its target: like the paper's "
                "language corrections, the motion changes immediately.", x1 + 18, 548, 316, 13, MUTED)


# ============================================================ 11. L2R results
class L2RResultsSlide(Slide):
    kicker, title = "Article 3 · results", "Rewards cover more tasks than a fixed skill library"
    kicker_color = C3
    hints = "Left: paper · middle: my pendulum version (results/summary.json) · right: planning-horizon sweep"

    def draw(self, s):
        g = ease(self.t / 1.0)
        S = self.app.context["summary"]["l2r"]
        rrect(s, CARD, (60, 150, 340, 470), 18)
        text(s, "Paper: 17 tasks", (82, 166), 19, TEXT, True)
        text(s, "quadruped (9) + dexterous hand (8), GPT-4", (82, 196), 13, FAINT)
        for i, (lab, v, col) in enumerate((("Reward interface (theirs)", 0.9, C3), ("Code as Policies", 0.5, GRAY))):
            y = 240 + i * 90
            text(s, lab, (82, y), 15, TEXT)
            rrect(s, col, (82, y + 26, max(2, 280 * v * g), 30), 8)
            text(s, f"{v:.0%} of tasks", (92, y + 32), 15, BG if v * g > 0.3 else TEXT, True)
        para(s, "Removing the Motion Descriptor also hurt: the Reward Coder alone forgot steps, e.g. moving the "
                "hand to the drawer handle before pulling.", 82, 440, 300, 14, MUTED)
        # my results
        x0 = 420
        rrect(s, CARD, (x0, 150, 470, 470), 18)
        text(s, "My pendulum version (10 random starts each)", (x0 + 22, 166), 17, TEXT, True)
        circle(s, C3, (x0 + 30, 206), 6)
        text(s, "reward + MPC", (x0 + 42, 197), 13, TEXT)
        circle(s, GRAY, (x0 + 160, 206), 6)
        text(s, "fixed primitives", (x0 + 172, 197), 13, TEXT)
        for i, row in enumerate(S["tasks"]):
            y = 228 + i * 70
            text(s, row["instruction"][:46], (x0 + 22, y), 14, TEXT)
            rrect(s, C3, (x0 + 22, y + 22, max(2, 330 * row["reward"] * g), 12), 4)
            rrect(s, GRAY, (x0 + 22, y + 38, max(2, 330 * row["cap"] * g), 12), 4)
            text(s, f"{row['reward']:.0%} / {row['cap']:.0%}", (x0 + 448, y + 26), 13, MUTED, anchor="topright")
        lean = S["tasks"][4].get("achieved_lean_deg", 0)
        para(s, f"Task 5 is physically impossible (needs 5 N·m, motor gives 2). MPC settled at {lean:.1f}° from "
                f"upright, the closest holdable pose (theory: 156.4°).", x0 + 22, 576, 430, 13, CORAL)
        # horizon sweep
        x1 = 910
        rrect(s, CARD, (x1, 150, 310, 470), 18)
        text(s, "Swing-up success vs horizon", (x1 + 20, 166), 16, TEXT, True)
        text(s, "predictive sampling, 256 samples", (x1 + 20, 192), 13, FAINT)
        hs = S["horizon_sweep"]
        for i, (h, v) in enumerate(hs.items()):
            x = x1 + 30 + i * 54
            bh = 260 * v * g
            rrect(s, C3 if int(h) <= 30 else CORAL, (x, 500 - bh, 36, max(2, bh)), 6)
            text(s, h, (x + 18, 508), 13, MUTED, anchor="midtop")
            text(s, f"{v:.0%}", (x + 18, 494 - bh), 12, TEXT, anchor="midbottom")
        text(s, "steps (×0.05 s)", (x1 + 155, 530), 12, FAINT, anchor="midtop")
        para(s, "Random sampling gets worse with long horizons: too many possible plans to search.",
             x1 + 20, 556, 270, 13, MUTED)
