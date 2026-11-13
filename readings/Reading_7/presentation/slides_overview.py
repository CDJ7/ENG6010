"""
slides_overview.py - title slide and the "subway map" overview of the four bridges.
"""

import math
import os
import sys

import numpy as np

from engine import *  # noqa: F401,F403
from engine import Slide

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from demos import voxposer_demo as V  # noqa: E402

# One colour per article, used on every slide
C1, C2, C3, C4 = SKY, AMBER, VIOLET, TEAL
ARTICLES = [
    dict(n=1, name="Gorilla", short="Gorilla", who="Patil et al., 2023 · UC Berkeley", color=C1, icon="search",
         layer="API call"),
    dict(n=2, name="VoxPoser", short="VoxPoser", who="Huang et al., CoRL 2023 · Stanford", color=C2, icon="map",
         layer="3D value map"),
    dict(n=3, name="Language to Rewards", short="L2R", who="Yu et al., 2023 · Google DeepMind", color=C3,
         icon="sliders", layer="reward function"),
    dict(n=4, name="Kine2Go", short="Kine2Go", who="Pałucki et al., 2026 · Univ. of Warsaw", color=C4, icon="dog",
         layer="motion data"),
]


# ============================================================ 1. title: words fall, motors move
class TitleSlide(Slide):
    deep = True
    hints = "Click a band to see which paper it is   ·   → next slide"
    PHRASE = "open the top drawer and watch out for the vase"
    BANDS = [(260, "API call", C1, "Gorilla: the LLM picks a real tool and calls it correctly"),
             (340, "3D value map", C2, "VoxPoser: the LLM's code paints where to go and what to avoid"),
             (420, "reward function", C3, "Language to Rewards: the LLM writes the goal, an optimiser moves"),
             (500, "motion data", C4, "Kine2Go: states + motor actions that train low-level control")]

    def enter(self):
        super().enter()
        self.rng = np.random.default_rng(3)
        self.letters, self.spawn, self.pulses, self.sel = [], 0.0, [], None

    def click(self, pos, button):
        for i, (y, *_r) in enumerate(self.BANDS):
            if 60 <= pos[0] <= 1220 and y <= pos[1] <= y + 64:
                self.sel = None if self.sel == i else i
                return True
        return False

    def update(self, dt):
        super().update(dt)
        self.spawn += dt
        while self.spawn > 0.12:
            self.spawn -= 0.12
            ch = self.PHRASE[int(self.t * 8) % len(self.PHRASE)]
            if ch != " ":
                self.letters.append([self.rng.uniform(260, 1180), 232.0, ch, self.rng.uniform(70, 120)])
        for l in self.letters:
            l[1] += l[3] * dt
        for l in [l for l in self.letters if l[1] > 584]:
            self.pulses.append([l[0], self.t])
        self.letters = [l for l in self.letters if l[1] <= 584]
        self.pulses = [p for p in self.pulses if self.t - p[1] < 2.5]

    def draw(self, s):
        a = self.appear(0)
        text(s, "ENG 6010 · ARTIFICIAL INTELLIGENCE IN ROBOTICS · READING 7", (640, 70), 16, AMBER, True,
             anchor="center", alpha=int(255 * a))
        text(s, "From Words to Motors", (640, 128), 60, TEXT, True, anchor="center", alpha=int(255 * a))
        text(s, "four papers, four ways to turn a sentence into robot motion", (640, 186), 22, MUTED,
             anchor="center", alpha=int(255 * self.appear(0.3)))
        for i, (y, lab, col, _) in enumerate(self.BANDS):
            rrect(s, col, (60, y, 1160, 64), 14, alpha=60 if self.sel == i else 26)
            text(s, lab.upper(), (82, y + 14), 15, col, True)
            text(s, ARTICLES[i]["name"], (82, y + 36), 14, MUTED)
        for x, y, ch, _ in self.letters:
            band = max([i for i, (by, *_r) in enumerate(self.BANDS) if y >= by] or [-1])
            col = TEXT if band < 0 else self.BANDS[band][2]
            text(s, ch, (x, y), 20, col, True, mono=True, anchor="center")
        # the "motor" waveform: every arriving letter becomes a decaying wiggle
        xs = np.linspace(60, 1220, 240)
        ys = np.zeros_like(xs)
        for px, t0 in self.pulses:
            age = self.t - t0
            ys += 14 * np.exp(-age * 2.2) * np.exp(-((xs - px) / 26) ** 2) * np.sin(age * 18 + (xs - px) / 9)
        pts = np.stack([xs, 628 + ys], 1)
        draw.lines(s, AMBER, False, pts, 2)
        text(s, "MOTOR COMMANDS", (1220, 600), 12, FAINT, True, anchor="topright")
        text(s, "WORDS", (60, 222), 12, FAINT, True, anchor="bottomleft")
        if self.sel is not None:
            y, lab, col, msg = self.BANDS[self.sel]
            bubble(s, 640, y - 64, 520, [msg], col, 15, tail=(700, y + 10))


# ============================================================ 2. subway map
class SubwaySlide(Slide):
    kicker, title = "The key link", "One line, five stations: what travels from words to motors"
    hints = "Click a station (or 1-5) to stop the train there   ·   0: let it run"
    STATIONS = [("Language", TEXT, None), ("API call", C1, 0), ("Value map", C2, 1),
                ("Reward", C3, 2), ("Motor data", C4, 3)]
    XS = [150, 395, 640, 885, 1130]

    def enter(self):
        super().enter()
        self.pinned = None
        scene = V.make_scene()
        self.cost = V.compose(scene, V.INSTRUCTIONS[1])["cost"]

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_5:
            self.pinned = k - pygame.K_1
        elif k == pygame.K_0:
            self.pinned = None
        else:
            return False
        return True

    def click(self, pos, button):
        for i, x in enumerate(self.XS):
            if abs(pos[0] - x) < 60 and 220 <= pos[1] <= 330:
                self.pinned = i
                return True
        return False

    def train_x(self):
        if self.pinned is not None:
            return self.XS[self.pinned], self.pinned
        cyc = (self.t / 2.4) % 5            # 2.4 s per leg: move 1 s, dwell 1.4 s
        i, f = int(cyc), cyc - int(cyc)
        j = (i + 1) % 5
        if j == 0:
            return self.XS[i], i
        g = ease(max(0, (f - 0.58) / 0.42))
        return lerp(self.XS[i], self.XS[j], g), (j if g > 0.5 else i)

    def draw(self, s):
        tx, cur = self.train_x()
        # the line
        for i in range(4):
            draw.line(s, self.STATIONS[i + 1][1], (self.XS[i], 290), (self.XS[i + 1], 290), 8)
        for i, (lab, col, _) in enumerate(self.STATIONS):
            on = i == cur
            circle(s, col, (self.XS[i], 290), 18 if on else 14)
            circle(s, BG, (self.XS[i], 290), 8 if on else 6)
            text(s, lab, (self.XS[i], 252), 19 if on else 16, col if on else MUTED, on, anchor="midbottom")
            if i:
                text(s, ARTICLES[i - 1]["short"], (self.XS[i], 316), 13, FAINT, anchor="midtop")
        rrect(s, TEXT, (tx - 34, 274, 68, 32), 10)
        text(s, "goal", (tx, 290), 13, BG, True, anchor="center")
        text(s, "more abstract", (60, 150), 13, FAINT)
        arrow(s, FAINT, (60, 176), (1220, 176), 1, 8)
        text(s, "closer to the motors", (1220, 150), 13, FAINT, anchor="topright")
        # what travels at this stop
        lab, col, art = self.STATIONS[cur]
        rrect(s, CARD, (60, 360, 1160, 262), 18)
        text(s, f"WHAT TRAVELS AT \"{lab.upper()}\"", (84, 376), 14, col, True)
        x0 = 84
        if cur == 0:
            para(s, "\"Open the top drawer, and watch out for that vase!\"", x0, 412, 700, 30, TEXT, True)
            para(s, "Rich in meaning, but a robot can't execute it: which drawer? how hard to pull? where is the vase?",
                 x0, 500, 700, 17, MUTED)
        elif cur == 1:
            rrect(s, CODE_BG, (x0, 410, 700, 60), 10)
            text(s, "torch.hub.load('pytorch/vision', 'fcn_resnet50')", (x0 + 16, 428), 18, (220, 227, 239), mono=True)
            para(s, "A single tool call. The tool (a segmentation model) does the actual work. "
                    "Grounding: it must exist in the API database, or it is a hallucination.", x0, 492, 700, 17, MUTED)
        elif cur == 2:
            c = self.cost
            a = (c - c.min()) / (c.max() - c.min() + 1e-9)
            blit_array(s, colormap(1 - a, [(16, 24, 44), (30, 80, 110), (45, 212, 191), (245, 165, 36)]),
                       (x0, 410, 190, 190))
            para(s, "A cost map over the scene: bright = go here (the handle), dark = stay away (the vase). "
                    "A planner searches it and MPC replans 5 times a second.", x0 + 216, 420, 480, 17, MUTED)
        elif cur == 3:
            rrect(s, CODE_BG, (x0, 410, 700, 96), 10)
            for j, line in enumerate(["set_torso_targets(0.65, np.deg2rad(90), ...)",
                                      "set_feet_pos_parameters('back_left', 0.0, None, None)"]):
                text(s, line, (x0 + 16, 424 + j * 34), 16, (220, 227, 239), mono=True)
            para(s, "Reward terms say what success looks like; MuJoCo MPC finds the motor actions "
                    "(paper example: stand up on two back feet).", x0, 524, 700, 17, MUTED)
        else:
            t = np.linspace(0, 2, 160)
            for j, colj in enumerate((C4, SKY, AMBER)):
                y = 0.8 * np.sin(2 * np.pi * 1.4 * t + j * 2.1) + 0.25 * np.sin(2 * np.pi * 2.8 * t + j)
                pts = np.stack([x0 + t * 340, 470 + j * 34 - y * 14], 1)
                draw.lines(s, colj, False, pts, 2)
            para(s, "Per frame: joint angles, velocities, base orientation and the motor actions that produced "
                    "them. Learned policies train on this.", x0 + 370, 420, 330, 17, MUTED)
        # right: who turns it into motion
        x1 = 820
        info = [("Who writes it", ["a person", "the LLM (fine-tuned)", "the LLM (writes code)", "the LLM (writes code)",
                                   "RL policies imitating mocap"][cur]),
                ("Turned into motion by", ["(nothing yet)", "the tool / library", "planner + MPC at 5 Hz",
                                           "MuJoCo MPC optimiser", "a trained controller"][cur]),
                ("Kept honest by", ["(nothing yet)", "docs retrieval + AST check", "camera perception (VLMs)",
                                    "physics in the optimiser", "Go2 joint limits + filtering"][cur])]
        for j, (h1, v) in enumerate(info):
            y = 404 + j * 70
            text(s, h1.upper(), (x1, y), 12, FAINT, True)
            text(s, v, (x1, y + 20), 18, col if j < 2 else TEXT, True)
