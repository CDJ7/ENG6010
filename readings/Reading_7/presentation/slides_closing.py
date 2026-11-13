"""
slides_closing.py - connections board, a "which bridge?" quiz and the takeaway.
"""

import math

from engine import *  # noqa: F401,F403
from engine import Slide
from slides_overview import ARTICLES

THREADS = [
    ("Who writes the bridge?", ["LLM writes an API call", "LLM writes value-map code", "LLM writes reward code",
                                "No LLM: retargeting + RL"],
     ["Gorilla is a fine-tuned LLaMA-7B that answers with one call, e.g. torch.hub.load(...).",
      "GPT-4 writes Python that calls detect() and builds NumPy voxel maps.",
      "GPT-4 first fills a Motion Descriptor, then writes set_..._reward() calls.",
      "Motion capture is retargeted with IK, and PPO policies learn to imitate it."]),
    ("What keeps it honest?", ["docs + AST check", "camera perception", "physics in MPC", "joint limits, falls"],
     ["A retriever supplies the current doc; AST matching flags calls that don't exist.",
      "OWL-ViT, SAM and XMem locate the real objects every cycle, so maps follow the scene.",
      "MuJoCo MPC only finds motions the simulated robot can really do (my 156° 'horizontal').",
      "Retargeted poses respect the Go2's joints; episodes that fall or drift are cut."]),
    ("Closed loop?", ["no: one answer", "yes: replan at 5 Hz", "yes: MPC + chat", "yes: policy feedback"],
     ["Gorilla answers once; checking happens after the fact.",
      "The planner re-runs on fresh maps 5 times a second, so moved objects are handled.",
      "MPC re-optimises every step, and the user can correct the reward in conversation.",
      "Each trained policy reacts to the simulated robot's state at every step."]),
    ("What is learned?", ["fine-tuning", "nothing (zero-shot)", "nothing (zero-shot)", "PPO policies"],
     ["Supervised fine-tuning on 16,450 self-instruct pairs.",
      "No training: an off-the-shelf LLM + VLMs + a planner. (Optional: learn dynamics with its prior.)",
      "No training: GPT-4 + MuJoCo MPC.",
      "Reinforcement learning, 40 policies, then the dataset itself is the product."]),
    ("Where it fails", ["bad retriever, numbers", "perception, local minima", "fixed reward terms", "flat ground only"],
     ["BM25 can fetch the wrong doc and lower accuracy; it ignores numeric constraints.",
      "Most real-robot errors came from perception; greedy descent can also get trapped (my cup demo).",
      "Hand-written templates per robot, a fixed set of reward terms, and rewards cannot change over time.",
      "Flat terrain only, almost no jumps, no sitting and no foot-pressure data (paper's own list)."]),
]


# ============================================================ 17. connections board
class ConnectionsSlide(Slide):
    kicker, title = "Connections", "Five questions, four papers: compare them cell by cell"
    hints = "Click any cell (or hover) to read the evidence   ·   → next: quiz"

    def enter(self):
        super().enter()
        self.sel = (0, 0)

    def cell(self, r, c):
        return 290 + c * 232, 186 + r * 66, 222, 58

    def click(self, pos, button):
        for r in range(5):
            for c in range(4):
                if inside(pos, self.cell(r, c)):
                    self.sel = (r, c)
                    return True
        return False

    def motion(self, pos, buttons):
        for r in range(5):
            for c in range(4):
                if inside(pos, self.cell(r, c)):
                    self.sel = (r, c)

    def draw(self, s):
        for c, art in enumerate(ARTICLES):
            x = 290 + c * 232
            icon(s, art["icon"], (x + 20, 158), 28, art["color"])
            text(s, art["short"], (x + 42, 148), 17, art["color"], True)
        sr, sc = self.sel
        for r, (q, cells, _) in enumerate(THREADS):
            y = 186 + r * 66
            text(s, q, (60, y + 18), 16, TEXT if r == sr else MUTED, r == sr)
            for c in range(4):
                x, yy, w, h = self.cell(r, c)
                on = (r, c) == self.sel
                col = ARTICLES[c]["color"]
                rrect(s, mix(CARD, col, 0.35) if on else (CARD if r != sr and c != sc else CARD2), (x, yy, w, h), 12)
                if on:
                    draw.rect(s, col, (x, yy, w, h), 2, border_radius=12)
                text(s, cells[c], (x + w / 2, yy + h / 2), 14, TEXT if on else MUTED, on, anchor="center")
        q, cells, ev = THREADS[sr]
        col = ARTICLES[sc]["color"]
        rrect(s, CARD, (60, 524, 1160, 132), 16)
        draw.rect(s, col, (60, 524, 8, 132), border_radius=4)
        text(s, f"{ARTICLES[sc]['name'].upper()} · {q.upper()}", (88, 540), 13, col, True)
        text(s, cells[sc], (88, 562), 22, TEXT, True)
        para(s, ev[sc], 88, 600, 1100, 16, MUTED)


# ============================================================ 18. quiz
QUIZ = [("You need a robot to call a weather service and a map service correctly, even after their "
         "APIs change next month.", 0,
         "An API call is the bridge, and retrieval keeps it current when the docs change (Gorilla)."),
        ("A kitchen robot must reach the sink while keeping 10 cm from a hot pan that people keep moving.", 1,
         "Value maps compose 'go here' and 'avoid that', and MPC replans as the pan moves (VoxPoser)."),
        ("A quadruped should 'moonwalk', a skill nobody programmed, and you want to fix it by chatting.", 2,
         "Describe the goal as a reward, let MPC find the motion, then correct the reward in words (L2R)."),
        ("You want to train a world model for a Go2 but have no hardware time for data collection.", 3,
         "Retarget animal motion, train imitation policies in simulation, log 800 trajectories (Kine2Go)."),
        ("A manipulator must 'open the top drawer' in a cluttered scene it has never seen, no training "
         "allowed.", 1,
         "Zero-shot: an LLM writes the maps, perception grounds them, a planner follows them (VoxPoser).")]


class QuizSlide(Slide):
    kicker, title = "Check yourself", "Which bridge would you build?"
    hints = "1-4 or click: answer   ·   N: next scenario   ·   your score is kept until you leave the slide"

    def enter(self):
        super().enter()
        self.qi, self.answer, self.score, self.done = 0, None, 0, 0
        self.t_a = 0.0

    def choose(self, i):
        if self.answer is None:
            self.answer, self.t_a = i, self.t
            self.done += 1
            self.score += i == QUIZ[self.qi][1]

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_4:
            self.choose(k - pygame.K_1)
        elif k == pygame.K_n:
            self.qi, self.answer = (self.qi + 1) % len(QUIZ), None
        else:
            return False
        return True

    def click(self, pos, button):
        for i in range(4):
            if inside(pos, (60 + i * 292, 330, 280, 150)):
                self.choose(i)
                return True
        return False

    def draw(self, s):
        q, right, why = QUIZ[self.qi]
        rrect(s, CARD, (60, 140, 1160, 160), 18)
        text(s, f"SCENARIO {self.qi + 1}/{len(QUIZ)}", (84, 156), 13, AMBER, True)
        para(s, q, 84, 186, 1100, 26, TEXT, True)
        text(s, f"score {self.score}/{self.done}", (1196, 156), 15, GREEN if self.score == self.done else MUTED,
             True, anchor="topright")
        for i, art in enumerate(ARTICLES):
            x = 60 + i * 292
            col = art["color"]
            if self.answer is None:
                bg = CARD
            elif i == right:
                bg = mix(CARD, GREEN, 0.35)
            elif i == self.answer:
                bg = mix(CARD, CORAL, 0.35)
            else:
                bg = CARD
            wob = 6 * math.sin((self.t - self.t_a) * 30) * max(0, 1 - (self.t - self.t_a) * 3) \
                if self.answer == i and i != right else 0
            rrect(s, bg, (x + wob, 330, 280, 150), 18)
            icon(s, art["icon"], (x + wob + 44, 380), 52, col)
            text(s, f"{i + 1}", (x + wob + 260, 344), 14, FAINT, anchor="topright")
            text(s, art["layer"], (x + wob + 80, 360), 20, TEXT, True)
            text(s, art["name"], (x + wob + 80, 390), 14, col)
            if self.answer is not None and i == right:
                mark(s, True, (x + 32, 450), 18, GREEN)
                text(s, "best fit", (x + 50, 440), 16, GREEN, True)
            elif self.answer == i:
                mark(s, False, (x + 32, 450), 16, CORAL)
                text(s, "not the best fit", (x + 50, 440), 16, CORAL, True)
        if self.answer is None:
            rrect(s, CARD, (60, 504, 1160, 110), 18)
            text(s, "ASK YOURSELF", (84, 520), 13, FAINT, True)
            para(s, "Does the robot need a tool to call, a place to go (and avoid), a goal to optimise, "
                    "or data to learn from?", 84, 546, 1100, 20, MUTED)
        else:
            rrect(s, CARD2, (60, 504, 1160, 110), 18)
            text(s, "WHY", (84, 520), 13, ARTICLES[right]["color"], True)
            para(s, why, 84, 546, 1100, 20, TEXT)
            text(s, "N: next scenario", (1196, 590), 13, FAINT, anchor="topright")


# ============================================================ 19. takeaway
class TakeawaySlide(Slide):
    deep = True
    hints = "Thanks!   ·   Esc to quit"
    LESSONS = ["Gorilla: give the LLM the docs, and check its output against reality (AST).",
               "VoxPoser: an LLM can write the map; a planner should walk it, and replan.",
               "Language to Rewards: describe what success looks like; let an optimiser find how.",
               "Kine2Go: the bottom layer still needs data, and simulation can make it at scale."]

    def draw(self, s):
        text(s, "TAKEAWAY", (640, 110), 16, AMBER, True, anchor="center")
        text(s, "Let the LLM say what to do.", (640, 160), 44, TEXT, True, anchor="center")
        text(s, "Let something grounded do it.", (640, 214), 44, TEAL, True, anchor="center")
        # a spectrum from flexible to grounded, with the four papers placed on it
        y = 312
        x0, x1 = 160, 1120
        for k in range(60):
            draw.rect(s, mix(VIOLET, TEAL, k / 59), (x0 + k * (x1 - x0) / 60, y, (x1 - x0) / 60 + 1, 8))
        text(s, "flexible (language)", (x0, y + 18), 13, VIOLET)
        text(s, "grounded (motors)", (x1, y + 18), 13, TEAL, anchor="topright")
        for i, art in enumerate(ARTICLES):
            px = x0 + (0.18 + 0.22 * i) * (x1 - x0)
            g = ease((self.t - 0.2 - 0.2 * i) / 0.6)
            circle(s, art["color"], (px, y + 4), 6 + 6 * g)
            text(s, art["layer"], (px, y - 14), 14, art["color"], True, anchor="midbottom")
        for i, l in enumerate(self.LESSONS):
            a = self.appear(0.8 + 0.25 * i)
            yy = 380 + i * 52
            rrect(s, CARD, (160, yy, 960, 42), 12, alpha=int(255 * a))
            circle(s, ARTICLES[i]["color"], (184, yy + 21), 7)
            text(s, l, (204, yy + 11), 17, TEXT, alpha=int(255 * a))
        text(s, "python presentation/interactive_presentation.py", (640, 610), 15, FAINT, mono=True,
             anchor="midtop")
