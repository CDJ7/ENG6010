"""
slides_intro_gorilla.py - title, overview and Article 1 (Gorilla) slides.
"""

import math
import os
import sys

import numpy as np

from engine import *  # noqa: F401,F403  (drawing helpers, palette, Slide)
from engine import Slide, W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from demos import gorilla_demo as G  # noqa: E402

# One colour per article, used on every slide
C1, C2, C3, C4 = SKY, AMBER, VIOLET, TEAL
ARTICLES = [
    dict(n=1, name="Gorilla", who="Patil et al., 2023 · UC Berkeley", color=C1, icon="search",
         layer="API call", llm_out="the right tool call", fills="the tool / library",
         demo="BM25 retrieval + AST hallucination check"),
    dict(n=2, name="VoxPoser", who="Huang et al., CoRL 2023 · Stanford", color=C2, icon="map",
         layer="3D value map", llm_out="code that marks where to go / avoid", fills="motion planner + MPC",
         demo="value maps, greedy planner, live replanning"),
    dict(n=3, name="Language to Rewards", who="Yu et al., 2023 · Google DeepMind", color=C3, icon="sliders",
         layer="reward function", llm_out="reward terms and targets", fills="MuJoCo MPC optimiser",
         demo="reward API + predictive-sampling MPC"),
    dict(n=4, name="Kine2Go", who="Pałucki et al., 2026 · Univ. of Warsaw", color=C4, icon="dog",
         layer="motion data", llm_out="(no LLM) 800 state + action trajectories", fills="RL policies (PPO)",
         demo="retargeting IK + imitation reward"),
]


def badge(s, label, x, y, color, size=14):
    w = text_width(label, size, True) + 20
    rrect(s, color, (x, y, w, size + 12), (size + 12) // 2)
    text(s, label, (x + w / 2, y + (size + 12) / 2), size, BG, True, anchor="center")
    return w


def verdict_color(v):
    return {"correct": GREEN, "error": AMBER, "hallucination": CORAL}.get(v, MUTED)


# ============================================================ 1. title
class TitleSlide(Slide):
    deep = True
    hints = "Click a layer of the stack to highlight it   ·   → next slide"

    def enter(self):
        super().enter()
        self.sel = None

    def click(self, pos, button):
        for i in range(5):
            x, y, w, h = 820, 150 + i * 92, 380, 72
            if x <= pos[0] <= x + w and y <= pos[1] <= y + h:
                self.sel = None if self.sel == i else i
                return True
        return False

    def draw(self, s):
        a = self.appear(0)
        text(s, "ENG 6010 · ARTIFICIAL INTELLIGENCE IN ROBOTICS · READING 7", (80, 160 - 20 * (1 - a)), 18, AMBER,
             True, alpha=int(255 * a))
        a = self.appear(0.15)
        text(s, "From Words to Motors", (76, 200 - 20 * (1 - a)), 58, TEXT, True, alpha=int(255 * a))
        a = self.appear(0.3)
        para(s, "How four papers connect language models to robot control, "
                "one layer at a time", 80, 290, 640, 28, MUTED, alpha=int(255 * a))
        for i, art in enumerate(ARTICLES):
            a = self.appear(0.45 + 0.1 * i)
            y = 420 + i * 44
            icon(s, art["icon"], (98, y + 14), 30, art["color"])
            text(s, f"{art['n']}. {art['name']}", (126, y + 2), 21, TEXT, True, alpha=int(255 * a))
            text(s, art["who"], (430, y + 5), 16, MUTED, alpha=int(255 * a))
        # the stack: language at the top, motors at the bottom
        layers = [("Language", "\"open the drawer\"", TEXT)] + \
                 [(art["layer"], art["name"], art["color"]) for art in ARTICLES]
        layers[-1] = ("Motor data", "Kine2Go", C4)
        pulse = (self.t * 0.45) % 1.0
        for i, (lab, src, col) in enumerate(layers):
            a = self.appear(0.2 + 0.12 * i)
            x, y, w, h = 820, 150 + i * 92 + 20 * (1 - a), 380, 72
            on = self.sel == i
            rrect(s, CARD2 if on else CARD, (x, y, w, h), 16)
            if on:
                draw.rect(s, col, (x, y, w, h), 2, border_radius=16)
            text(s, lab, (x + 24, y + 14), 22, col, True)
            text(s, src, (x + 24, y + 42), 15, MUTED)
            if i < len(layers) - 1:
                arrow(s, FAINT, (x + w / 2, y + h + 2), (x + w / 2, y + h + 18), 2, 8)
        py = 150 + pulse * (4 * 92 + 72)
        circle(s, AMBER, (1206, py), 6)
        text(s, "high-level intent", (1200, 146), 13, FAINT, anchor="bottomright")
        text(s, "motor commands", (1200, 600), 13, FAINT, anchor="topright")


# ============================================================ 2. the stack
class StackSlide(Slide):
    kicker, title = "The key link", "Every paper picks a different bridge from words to motors"
    hints = "Click an article (or press 1-4) to see what sits between language and the robot"

    def enter(self):
        super().enter()
        self.sel = 0

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_4:
            self.sel = k - pygame.K_1
            return True
        return False

    def click(self, pos, button):
        for i in range(4):
            x, y = 60, 160 + i * 118
            if x <= pos[0] <= x + 460 and y <= pos[1] <= y + 100:
                self.sel = i
                return True
        return False

    def draw(self, s):
        for i, art in enumerate(ARTICLES):
            x, y = 60, 160 + i * 118
            on = i == self.sel
            rrect(s, CARD2 if on else CARD, (x, y, 460, 100), 16)
            if on:
                draw.rect(s, art["color"], (x, y, 460, 100), 2, border_radius=16)
            icon(s, art["icon"], (x + 48, y + 50), 56, art["color"] if on else FAINT)
            text(s, f"{art['n']}. {art['name']}", (x + 92, y + 22), 22, TEXT, True)
            text(s, f"bridge: {art['layer']}", (x + 92, y + 56), 17, art["color"] if on else MUTED, True)
        art = ARTICLES[self.sel]
        x0 = 560
        rrect(s, CARD, (x0, 160, 660, 454), 18)
        text(s, art["name"], (x0 + 30, 182), 30, art["color"], True)
        text(s, art["who"], (x0 + 30, 224), 16, MUTED)
        # flow diagram for the selected article
        boxes = [("User / goal", "\"open the drawer\"" if art["n"] != 4 else "motion capture"),
                 ("LLM writes", art["llm_out"]),
                 ("Filled in by", art["fills"]),
                 ("Robot", "joint torques / motions")]
        if art["n"] == 4:
            boxes[1] = ("Pipeline makes", art["llm_out"])
        f = (self.t * 0.5) % 1.0
        for j, (h1, h2) in enumerate(boxes):
            y = 272 + j * 78
            rrect(s, CARD2, (x0 + 30, y, 600, 60), 12)
            text(s, h1, (x0 + 50, y + 8), 15, art["color"] if j in (1, 2) else MUTED, True)
            text(s, h2, (x0 + 50, y + 30), 18, TEXT)
            if j < 3:
                arrow(s, FAINT, (x0 + 330, y + 61), (x0 + 330, y + 76), 2, 7)
        circle(s, art["color"], (x0 + 640, 272 + f * 294), 5)
        text(s, "My demo: " + art["demo"], (x0 + 30, 590), 16, MUTED)


# ============================================================ 3. Gorilla idea
class GorillaIdeaSlide(Slide):
    kicker, title = "Article 1 · Gorilla", "LLMs hallucinate API calls; Gorilla is trained to get them right"
    kicker_color = C1
    hints = "The three answers are the paper's Figure 1 example   ·   → next: try the retriever and checker"

    FIG1 = [("GPT-4", "torch.hub.load('snakers4/silero-models',", "    'asr', source='local')", "Hallucinated model", CORAL),
            ("Claude", "torchaudio.pipelines.WAV2VEC2_ASR_", "    PIPELINE('audio.wav')", "Wrong library", CORAL),
            ("Gorilla", "torch.hub.load('snakers4/silero-models',", "    'silero_stt')", "Good to go", GREEN)]

    def draw(self, s):
        rrect(s, CARD, (60, 150, 600, 470), 18)
        text(s, "Prompt (paper Fig. 1)", (84, 168), 15, C1, True)
        para(s, "\"Help me find an API to convert the spoken language in a recorded audio "
                "to text using Torch Hub.\"", 84, 192, 552, 18, TEXT)
        for i, (who, l1, l2, verdict, col) in enumerate(self.FIG1):
            a = self.appear(0.4 + 0.5 * i)
            y = 270 + i * 112
            rrect(s, CODE_BG, (84, y, 552, 96), 12, alpha=int(255 * a))
            text(s, who, (100, y + 10), 17, TEXT, True, alpha=int(255 * a))
            text(s, l1, (100, y + 38), 15, (220, 227, 239), mono=True, alpha=int(255 * a))
            text(s, l2, (100, y + 60), 15, (220, 227, 239), mono=True, alpha=int(255 * a))
            if a > 0.9:
                badge(s, verdict, 520 - text_width(verdict, 13, True), y + 8, col, 13)
        # how Gorilla is built
        x0 = 690
        steps = [("database", "APIBench: 1,645 API docs", "94 TorchHub · 626 TensorFlow Hub · 925 HuggingFace"),
                 ("comments", "Self-instruct: 16,450 prompts", "GPT-4 writes 10 user requests per API"),
                 ("brain", "Fine-tune LLaMA-7B", "learn to answer with the right API call"),
                 ("search", "Optional retriever", "paste the newest API doc into the prompt")]
        for i, (ic, t1, t2) in enumerate(steps):
            a = self.appear(0.2 + 0.25 * i)
            y = 150 + i * 98 + 12 * (1 - a)
            rrect(s, CARD, (x0, y, 530, 84), 14)
            icon(s, ic, (x0 + 42, y + 42), 50, C1)
            text(s, t1, (x0 + 82, y + 14), 19, TEXT, True)
            text(s, t2, (x0 + 82, y + 46), 15, MUTED)
        rrect(s, CARD2, (x0, 550, 530, 70), 14)
        para(s, "Hallucination = calling an API that does not exist. Error = a real API, "
                "but the wrong one. Both are measured by matching code trees (AST).", x0 + 20, 560, 490, 15, TEXT)


# ============================================================ 4. Gorilla demo
class GorillaDemoSlide(Slide):
    kicker, title = "Article 1 · Gorilla · live demo", "Retrieve the documentation, then check the call"
    kicker_color = C1
    hints = "Q: next prompt · T: type your own · C: next answer · U: update the docs (paper Fig. 6)"

    def enter(self):
        super().enter()
        self.qi, self.ci, self.updated = 0, 0, False
        self.custom = None
        self.typing = False
        self.buf = ""
        self.t_anim = 0.0

    # text entry ---------------------------------------------------------
    def text_input(self, s):
        if len(self.buf) < 110:
            self.buf += s

    def key(self, k):
        if self.typing:
            if k == pygame.K_RETURN:
                self.typing = False
                self.custom = self.buf.strip() or None
                self.t_anim = self.t
            elif k == pygame.K_ESCAPE:
                self.typing = False
            elif k == pygame.K_BACKSPACE:
                self.buf = self.buf[:-1]
            return True
        if k == pygame.K_q:
            self.qi, self.ci, self.custom = (self.qi + 1) % len(G.QUERIES), 0, None
            self.updated = False
        elif k == pygame.K_c:
            self.ci = (self.ci + 1) % 3
        elif k == pygame.K_t:
            self.typing, self.buf, self.updated = True, "", False
        elif k == pygame.K_u:
            self.updated = not self.updated
            if self.updated:
                self.qi, self.custom = 1, None
        else:
            return False
        self.t_anim = self.t
        return True

    def draw(self, s):
        db = G.database(self.updated)
        bm = G.BM25(db)
        q = G.QUERIES[self.qi]
        prompt = self.custom if self.custom else q.prompt
        scores = bm.scores(prompt)
        order = sorted(range(len(db)), key=lambda i: -scores[i])
        g = ease((self.t - self.t_anim) / 0.6)
        # ---- left: prompt + BM25 ranking ---------------------------------
        rrect(s, CARD, (60, 150, 600, 120), 16)
        text(s, "USER PROMPT" + ("   (typing: Enter to run, Esc to cancel)" if self.typing else ""),
             (82, 164), 13, C1, True)
        shown = (self.buf + ("|" if int(self.t * 2) % 2 == 0 else "")) if self.typing else prompt
        para(s, shown, 82, 188, 556, 17, TEXT)
        rrect(s, CARD, (60, 284, 600, 336), 16)
        text(s, "BM25 retriever: score of every API doc (top 6)", (82, 298), 15, TEXT, True)
        top = max(scores[order[0]], 1e-6)
        for r, i in enumerate(order[:6]):
            y = 330 + r * 40
            d = db[i]
            col = C1 if r == 0 else GRID
            w = 300 * scores[i] / top * g
            draw.rect(s, col, (250, y + 4, max(2, w), 22), border_radius=5)
            text(s, d["id"], (82, y + 6), 15, TEXT if r == 0 else MUTED, r == 0)
            text(s, f"{scores[i]:.2f}", (258 + w, y + 6), 14, MUTED)
        doc = db[order[0]]
        rrect(s, CODE_BG, (82, 572, 556, 38), 8)
        text(s, "prompt + \"Use this API documentation for reference: <" + doc["id"] + " doc>\"",
             (92, 582), 13, (200, 210, 225), mono=True)

        # ---- right: candidate answers + AST check ------------------------
        x0 = 690
        rrect(s, CARD, (x0, 150, 530, 470), 16)
        if self.updated:
            text(s, "DOCS UPDATED AFTER TRAINING (paper Fig. 6)", (x0 + 22, 164), 13, AMBER, True)
            para(s, "fcn_resnet50  →  fcn_resnet101 in the API database", x0 + 22, 186, 490, 16, TEXT)
            new_db = db
            aware, _, _ = G.retriever_aware_model(G.QUERIES[1].prompt, new_db)
            frozen = G.frozen_model("fcn_resnet50")
            rows = [("Retriever-aware model (reads the doc)", aware), ("Model without retrieval (memorised)", frozen)]
            for j, (lab, code) in enumerate(rows):
                v = G.ast_verdict(code, "fcn_resnet50", new_db)[0]
                y = 240 + j * 150
                rrect(s, CARD2, (x0 + 22, y, 486, 130), 12)
                text(s, lab, (x0 + 40, y + 12), 16, TEXT, True)
                para(s, code, x0 + 40, y + 42, 450, 14, (220, 227, 239))
                badge(s, v.upper(), x0 + 40, y + 92, verdict_color(v), 13)
            para(s, "The memorised call no longer exists in the new docs, so the AST check marks it a "
                    "hallucination. The retriever-aware model copies the new API from the doc.",
                 x0 + 22, 548, 490, 14, MUTED)
            return
        if self.custom:
            text(s, "YOUR PROMPT", (x0 + 22, 164), 13, C1, True)
            para(s, "A retriever-aware model would answer with the call from the top document:",
                 x0 + 22, 190, 486, 16, TEXT)
            rrect(s, CODE_BG, (x0 + 22, 250, 486, 70), 10)
            para(s, doc["api_call"], x0 + 36, 262, 460, 14, (220, 227, 239))
            para(s, f"Domain: {doc['domain']}. {doc['description']}", x0 + 22, 340, 486, 15, MUTED)
            para(s, "Check: is this what you asked for? Retrieval only works when the right document "
                    "ranks first (the paper found a poor retriever can lower accuracy).",
                 x0 + 22, 430, 486, 15, MUTED)
            return
        text(s, "CANDIDATE ANSWERS · AST SUB-TREE CHECK", (x0 + 22, 164), 13, C1, True)
        for j, (lab, code) in enumerate(q.candidates):
            v = G.ast_verdict(code, q.reference, db)[0]
            y = 190 + j * 54
            on = j == self.ci
            rrect(s, CARD2 if on else CODE_BG, (x0 + 22, y, 486, 46), 10)
            text(s, lab, (x0 + 36, y + 13), 15, TEXT if on else MUTED, on)
            badge(s, v, x0 + 494 - text_width(v, 13, True) - 20, y + 10, verdict_color(v), 13)
        lab, code = q.candidates[self.ci]
        v, matched, nodes = G.ast_verdict(code, q.reference, db)
        rrect(s, CODE_BG, (x0 + 22, 360, 486, 46), 8)
        para(s, code, x0 + 34, 366, 466, 13, (220, 227, 239))
        # draw the AST path: call -> key arguments
        text(s, "parsed tree (green = matches a real API, red = no match)", (x0 + 22, 418), 13, MUTED)
        y = 442
        for k, (label, ok) in enumerate(nodes[:5]):
            a = ease((self.t - self.t_anim - 0.15 * k) / 0.4)
            col = GREEN if ok else (CORAL if ok is False else FAINT)
            xx = x0 + 40 + k * 18
            circle(s, col, (xx, y + 10), 6)
            text(s, label[:52], (xx + 16, y), 15, TEXT if ok is not None else FAINT, alpha=int(255 * a))
            y += 30
        note = {"correct": "matches the reference API for this task",
                "error": f"a real API ({matched['id'] if matched else ''}), but the wrong one",
                "hallucination": "matches no API in the database: an imagined tool"}[v]
        badge(s, f"{v.upper()}: {note}", x0 + 22, 590, verdict_color(v), 13)


# ============================================================ 5. Gorilla results
PAPER_T1 = {   # paper Table 1: overall accuracy / hallucination (%), per hub and retriever
    "zero-shot": dict(acc=[[0, 48.38, 38.70, 18.81, 59.13], [0.00, 16.81, 19.80, 6.19, 71.68], [0, 41.75, 18.20, 9.19, 83.79]],
                      hal=[[100, 18.81, 36.55, 65.59, 6.98], [97.57, 35.73, 37.16, 77.65, 10.95], [100, 47.88, 78.65, 88.46, 5.40]]),
    "BM25": dict(acc=[[8.60, 38.17, 35.48, 39.78, 40.32], [3.00, 17.26, 16.48, 14.60, 17.03], [8.90, 54.16, 34.01, 35.18, 41.89]],
                 hal=[[76.88, 6.98, 11.29, 5.37, 4.30], [77.99, 8.30, 15.93, 15.82, 6.42], [77.37, 3.64, 37.08, 21.16, 2.77]]),
    "GPT-Index": dict(acc=[[14.51, 60.21, 59.13, 60.21, 61.82], [10.18, 29.08, 44.58, 41.37, 47.46], [15.62, 65.59, 43.94, 55.62, 64.96]],
                      hal=[[75.8, 1.61, 1.07, 3.76, 0], [75.66, 7.85, 11.18, 18.81, 8.19], [77.66, 3.79, 31.53, 16.20, 2.33]]),
    "Oracle": dict(acc=[[16.12, 66.31, 66.12, 63.44, 67.20], [17.70, 89.71, 85.07, 77.21, 91.26], [12.55, 95.03, 55.91, 74.74, 94.16]],
                   hal=[[79.03, 1.60, 0.53, 3.76, 0], [77.10, 6.64, 10.62, 19.58, 7.08], [87.00, 0.29, 37.95, 21.60, 1.89]]),
}
MODELS = ["LLaMA", "GPT-3.5", "GPT-4", "Claude", "Gorilla"]
MODEL_COLORS = [GRAY, LAV, PINK, AMBER, C1]
HUBS = ["TorchHub", "HuggingFace", "TensorFlow Hub"]


class GorillaResultsSlide(Slide):
    kicker, title = "Article 1 · Gorilla · paper results", "Fine-tuning beats prompting; a bad retriever can hurt"
    kicker_color = C1
    hints = "R: switch retriever (zero-shot / BM25 / GPT-Index / Oracle)   ·   H: accuracy ↔ hallucination"
    SETTINGS = ["zero-shot", "BM25", "GPT-Index", "Oracle"]

    def enter(self):
        super().enter()
        self.si, self.hal = 0, False
        self.t_anim = 0.0

    def key(self, k):
        if k == pygame.K_r:
            self.si = (self.si + 1) % 4
        elif k == pygame.K_h:
            self.hal = not self.hal
        else:
            return False
        self.t_anim = self.t
        return True

    def draw(self, s):
        st = self.SETTINGS[self.si]
        data = PAPER_T1[st]["hal" if self.hal else "acc"]
        g = ease((self.t - self.t_anim) / 0.7)
        rrect(s, CARD, (60, 150, 780, 470), 18)
        text(s, f"{'Hallucination' if self.hal else 'Overall accuracy'} (%) · retriever: {st}",
             (84, 166), 19, TEXT, True)
        text(s, "Source: paper Table 1 (higher is better for accuracy, lower for hallucination)",
             (84, 194), 13, FAINT)
        base, top_h = 560, 300
        for v in (0, 25, 50, 75, 100):
            y = base - top_h * v / 100
            draw.line(s, GRID, (110, y), (820, y))
            text(s, str(v), (102, y), 12, MUTED, anchor="midright")
        for h, hub in enumerate(HUBS):
            gx = 130 + h * 236
            for m in range(5):
                v = data[h][m]
                bh = top_h * v / 100 * g
                x = gx + m * 40
                draw.rect(s, MODEL_COLORS[m], (x, base - bh, 32, max(1, bh)),
                          border_top_left_radius=5, border_top_right_radius=5)
                if m == 4:
                    text(s, f"{v:.0f}", (x + 16, base - bh - 4), 13, TEXT, True, anchor="midbottom")
            text(s, hub, (gx + 96, base + 10), 15, MUTED, anchor="midtop")
        for m, name in enumerate(MODELS):
            circle(s, MODEL_COLORS[m], (130 + m * 132, 604), 6)
            text(s, name, (142 + m * 132, 595), 14, TEXT)
        # take-aways
        x0 = 870
        notes = [
            ("Zero-shot: Gorilla wins", "59 / 72 / 84% vs GPT-4's 39 / 20 / 18% on the three hubs."),
            ("Bad retriever hurts", "With BM25, Gorilla falls to 40 / 17 / 42%: the wrong doc misleads it."),
            ("Good retriever helps", "With the Oracle (perfect) doc, Gorilla reaches 67 / 91 / 94%."),
            ("GPT-3.5 < GPT-4 hallucination", "The authors suggest RLHF tuning makes a model more truthful."),
        ]
        for i, (h1, body) in enumerate(notes):
            y = 150 + i * 118
            rrect(s, CARD2 if (i == 0 and st == "zero-shot") or (i == 1 and st == "BM25") or
                  (i == 2 and st == "Oracle") else CARD, (x0, y, 350, 106), 14)
            text(s, h1, (x0 + 18, y + 12), 17, C1, True)
            para(s, body, x0 + 18, y + 42, 316, 15, TEXT)
