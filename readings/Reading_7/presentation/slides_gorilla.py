"""
slides_gorilla.py - Article 1 (Gorilla): chat replay + build pipeline, BM25
heatmap, AST lab with an API builder, and the paper's results as a scatter.
"""

import math
import os
import sys

import numpy as np

from engine import *  # noqa: F401,F403
from engine import Slide

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from demos import gorilla_demo as G  # noqa: E402

C1 = SKY
CODE_TXT = (220, 227, 239)
VERDICT_COL = {"correct": GREEN, "error": AMBER, "hallucination": CORAL}


def stamp(s, label, center, color, t, size=22, angle=-8):
    """A rubber stamp that thumps in (scale 1.6 -> 1) when t goes 0 -> 1."""
    if t <= 0:
        return
    k = 1 + 0.6 * (1 - ease(min(1, t)))
    w = text_width(label, size, True) + 30
    h = size + 20
    surf = pygame.Surface((PX(w), PX(h)), pygame.SRCALPHA)
    with local_origin():
        draw.rect(surf, color, (2, 2, w - 4, h - 4), 3, border_radius=8)
        text(surf, label, (w / 2, h / 2), size, color, True, anchor="center")
    surf = pygame.transform.rotozoom(surf, angle, k)
    surf.set_alpha(int(255 * min(1, t * 2)))
    x, y = P(center)
    s.blit(surf, surf.get_rect(center=(int(x), int(y))))


# ============================================================ 3. idea: chat + conveyor
class GorillaIdeaSlide(Slide):
    kicker, title = "Article 1 · Gorilla", "Same question, three chatbots: only one gives a real API call"
    kicker_color = C1
    hints = "1 / 2 / 3 or click a tab: GPT-4 · Claude · Gorilla   ·   click a machine on the right to explain it"
    PROMPT = "Help me find an API to convert the spoken language in a recorded audio to text using Torch Hub."
    BOTS = [("GPT-4", PINK, ["<domain>: Speech-to-Text", "<api_provider>: TorchHub",
                              "<code>: asr_model = torch.hub.load(", "    'snakers4/silero-models', 'asr',",
                              "    source='local')"], "HALLUCINATED", CORAL,
             "The repo is real, but there is no model called 'asr' in it."),
            ("Claude", AMBER, ["<domain>: Audio-Translation", "<api_provider>: Pytorch",
                               "<code>: import torchaudio", "translation = Torchaudio.pipelines.",
                               "    WAV2VEC2_ASR_PIPELINE('audio.wav')"], "WRONG LIBRARY", CORAL,
             "Not TorchHub, as the user asked, and it would not run as written."),
            ("Gorilla", C1, ["<domain>: Speech-to-Text", "<api_provider>: TorchHub",
                             "<code>: asr_model = torch.hub.load(", "    'snakers4/silero-models',",
                             "    'silero_stt', language='en')"], "GOOD TO GO", GREEN,
             "A real model from the requested hub, with the right arguments.")]
    MACHINES = [("database", "API docs", 1645, "APIBench: every model card from TorchHub (94), TensorFlow Hub (626) "
                                               "and HuggingFace (925), stored as JSON docs."),
                ("comments", "Self-instruct", 16450, "GPT-4 is shown each doc and writes 10 user requests that "
                                                     "this API answers: 16,450 instruction-API pairs."),
                ("brain", "Fine-tune", 7, "LLaMA-7B is fine-tuned on the pairs, so it learns to answer "
                                          "with the right API call (7 = billions of parameters)."),
                ("search", "Retriever", 1, "Optional: at test time a retriever pastes the newest doc into "
                                           "the prompt, so the answer can follow API changes.")]

    def enter(self):
        super().enter()
        self.bot, self.t_bot, self.sel = 0, 0.0, None

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_3:
            self.bot, self.t_bot = k - pygame.K_1, self.t
            return True
        return False

    def click(self, pos, button):
        for i in range(3):
            if inside(pos, (84 + i * 160, 150, 150, 40)):
                self.bot, self.t_bot = i, self.t
                return True
        for i in range(4):
            if inside(pos, (*self.machine_xy(i), 230, 120)):
                self.sel = None if self.sel == i else i
                return True
        return False

    @staticmethod
    def machine_xy(i):
        """Snake layout: 1 -> 2 on the top belt, then 3 <- 4 on the bottom belt."""
        return [(720, 176), (980, 176), (980, 340), (720, 340)][i]

    def update(self, dt):
        super().update(dt)
        if self.t - self.t_bot > 6.0:                      # auto-advance through the three bots
            self.bot, self.t_bot = (self.bot + 1) % 3, self.t

    def draw(self, s):
        # ---------------- left: a chat app window
        rrect(s, CARD, (60, 140, 600, 500), 20)
        for i, (name, col, *_r) in enumerate(self.BOTS):
            on = i == self.bot
            rrect(s, col if on else CARD2, (84 + i * 160, 152, 150, 36), 18)
            text(s, name, (159 + i * 160, 170), 16, BG if on else MUTED, True, anchor="center")
        draw.line(s, GRID, (60, 202), (660, 202), 1)
        # user's message (right-aligned bubble)
        lines = wrap(self.PROMPT, 15, 380)
        h = len(lines) * 21 + 20
        rrect(s, (45, 70, 110), (250, 220, 390, h), 14)
        for j, ln in enumerate(lines):
            text(s, ln, (266, 230 + j * 21), 15, TEXT)
        # the bot's reply types in
        name, col, code, verdict, vcol, why = self.BOTS[self.bot]
        el = self.t - self.t_bot
        circle(s, col, (100, 236 + h + 22), 18)
        text(s, name[0], (100, 236 + h + 22), 16, BG, True, anchor="center")
        y0 = 236 + h
        rrect(s, CODE_BG, (128, y0, 512, 150), 14)
        shown = int(el * 70)
        for j, ln in enumerate(code):
            part = ln[:max(0, shown - sum(len(c) for c in code[:j]))]
            text(s, part, (144, y0 + 14 + j * 25), 15, CODE_TXT, mono=True)
        if shown < sum(len(c) for c in code):
            if int(self.t * 3) % 2:
                draw.rect(s, col, (144, y0 + 14 + min(4, shown // 30) * 25, 8, 18))
        done = (sum(len(c) for c in code)) / 70
        stamp(s, verdict, (470, y0 + 168), vcol, (el - done) / 0.35)
        if el > done + 0.3:
            para(s, why, 84, y0 + 200, 330, 15, MUTED)
        # ---------------- right: the build pipeline as machines on a belt
        text(s, "HOW GORILLA IS BUILT", (720, 146), 14, C1, True)
        belts = [308, 472]
        for row in range(2):
            draw.rect(s, GRID, (700, belts[row], 520, 10), border_radius=5)
            for k in range(13):                             # moving belt marks
                x = 704 + ((k * 40 + self.t * 60 * (1 if row == 0 else -1)) % 512)
                draw.line(s, FAINT, (x, belts[row] + 1), (x - 5, belts[row] + 9), 2)
        draw.line(s, GRID, (1214, 318), (1214, 472), 4)
        for i, (ic, lab, n, _) in enumerate(self.MACHINES):
            x, y = self.machine_xy(i)
            on = self.sel == i
            rrect(s, CARD2 if on else CARD, (x, y, 230, 120), 16)
            if on:
                draw.rect(s, C1, (x, y, 230, 120), 2, border_radius=16)
            icon(s, ic, (x + 40, y + 42), 48, C1)
            text(s, lab, (x + 76, y + 20), 18, TEXT, True)
            # counters tick up when the slide opens
            g = ease((self.t - 0.3 * i) / 1.6)
            val = int(n * g)
            lab2 = {0: f"{val:,} docs", 1: f"{val:,} pairs", 2: f"LLaMA-{val}B", 3: "+ newest doc"}[i]
            text(s, lab2, (x + 76, y + 48), 17, C1, True, mono=True)
            text(s, ["step 1", "step 2", "step 3", "step 4 (optional)"][i], (x + 16, y + 90), 13, FAINT)
        # packets ride the belts: left to right on top, then right to left below
        for k in range(6):
            f = (self.t * 0.12 + k / 6) % 1.0
            if f < 0.5:
                x, y = 700 + f * 2 * 510, belts[0] - 12
            else:
                x, y = 1210 - (f - 0.5) * 2 * 510, belts[1] - 12
            rrect(s, C1, (x - 6, y, 12, 12), 3)
        if self.sel is not None:
            bubble(s, 700, 506, 520, [self.MACHINES[self.sel][3]], C1, 14)
        else:
            rrect(s, CARD2, (700, 506, 520, 78), 14)
            para(s, "Training prompt with a retriever: \"<user request> Use this API documentation for "
                    "reference: <retrieved doc>\" (the model learns to read docs).", 718, 518, 486, 14, TEXT)


# ============================================================ 4. BM25 heatmap
class GorillaRetrieverSlide(Slide):
    kicker, title = "Article 1 · Gorilla · the retriever", "BM25: which words point at which API doc?"
    kicker_color = C1
    hints = "Q: next example prompt   ·   T: type your own prompt (Enter to finish)   ·   click a row to retrieve that doc"

    def enter(self):
        super().enter()
        self.db = G.database()
        self.bm = G.BM25(self.db)
        self.qi, self.typed, self.typing, self.pick = 0, "", False, None
        self.t_q = 0.0

    def prompt(self):
        return self.typed if (self.typing or self.typed) else G.QUERIES[self.qi].prompt

    def key(self, k):
        if self.typing:
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
                self.typing = False
                pygame.key.stop_text_input()
            elif k == pygame.K_BACKSPACE:
                self.typed = self.typed[:-1]
                self.t_q = self.t
            return True
        if k == pygame.K_q:
            self.qi, self.typed, self.pick, self.t_q = (self.qi + 1) % len(G.QUERIES), "", None, self.t
        elif k == pygame.K_t:
            self.typing, self.typed, self.pick = True, "", None
            pygame.key.start_text_input()
        else:
            return False
        return True

    def text_input(self, s):
        self.typed += s
        self.t_q = self.t

    def click(self, pos, button):
        for i in range(len(self.db)):
            if inside(pos, (60, 236 + i * 26, 760, 24)):
                self.pick = i
                return True
        return False

    def draw(self, s):
        q = self.prompt()
        # prompt box
        rrect(s, CARD2 if self.typing else CARD, (60, 140, 1160, 50), 12)
        if self.typing:
            draw.rect(s, C1, (60, 140, 1160, 50), 2, border_radius=12)
        shown = q if text_width(q, 17) < 1120 else "…" + q[-110:]
        tw = text(s, shown or "type a request, e.g. 'turn my text into a spoken voice'", (80, 154), 17,
                  TEXT if q else FAINT)
        if self.typing and int(self.t * 2) % 2:
            draw.line(s, C1, (82 + tw.width, 154), (82 + tw.width, 176), 2)
        contrib = self.bm.contributions(q)
        words = list(contrib)[:7]
        scores = self.bm.scores(q)
        best = int(np.argmax(scores)) if max(scores, default=0) > 0 else None
        sel = self.pick if self.pick is not None else best
        mx = max([max(v) for v in contrib.values()] + [1e-9])
        # header: query words as columns
        x0, cw = 250, 66
        text(s, "API doc", (72, 208), 13, FAINT, True)
        for j, w in enumerate(words):
            ww = w if len(w) <= 8 else w[:7] + "…"
            text(s, ww, (x0 + j * cw + cw / 2, 212), 13, C1, True, anchor="center")
        if not words:
            text(s, "(no searchable words yet)", (x0, 206), 13, FAINT)
        sx = x0 + max(1, len(words)) * cw + 14
        text(s, "BM25 score", (sx, 208), 13, FAINT, True)
        smax = max(max(scores), 1e-9)
        g = ease((self.t - self.t_q) / 0.5)
        for i, d in enumerate(self.db):
            y = 236 + i * 26
            if i == sel:
                rrect(s, CARD2, (60, y - 2, 760, 25), 6)
            text(s, d["id"], (72, y + 2), 14, TEXT if i == best else MUTED, i == best, mono=True)
            for j, w in enumerate(words):
                v = contrib[w][i] / mx
                col = mix(CARD, C1, v) if v > 0 else BG_DEEP
                rrect(s, col, (x0 + j * cw + 3, y, cw - 6, 21), 4)
                if v > 0.15:
                    text(s, f"{contrib[w][i]:.1f}", (x0 + j * cw + cw / 2, y + 10), 12, BG if v > 0.6 else TEXT,
                         True, anchor="center")
            bw = 150 * scores[i] / smax * g
            draw.rect(s, GREEN if i == best else GRAY, (sx, y + 4, max(1, bw), 14), border_radius=4)
        # right: retrieved doc + augmented prompt
        x1 = 850
        rrect(s, CARD, (x1, 206, 370, 426), 16)
        if sel is None:
            para(s, "No document shares a word with this prompt, so BM25 has nothing to retrieve.",
                 x1 + 20, 226, 330, 16, MUTED)
            return
        d = self.db[sel]
        text(s, "RETRIEVED DOC" if sel == best else "DOC YOU PICKED", (x1 + 20, 222), 13, C1, True)
        text(s, d["id"], (x1 + 20, 244), 20, TEXT, True, mono=True)
        text(s, f"{d['domain']} · {d['api_provider']}", (x1 + 20, 274), 13, MUTED)
        para(s, d["description"], x1 + 20, 298, 330, 14, TEXT)
        ok = G.QUERIES[self.qi].reference == d["id"] if not self.typed else None
        if ok is not None:
            mark(s, ok, (x1 + 28, 381), 14, GREEN if ok else CORAL)
            text(s, "the correct API for this prompt" if ok else "not the API this prompt needs",
                 (x1 + 42, 372), 14, GREEN if ok else CORAL, True)
        text(s, "PROMPT SENT TO GORILLA", (x1 + 20, 404), 12, FAINT, True)
        rrect(s, CODE_BG, (x1 + 14, 424, 342, 196), 10)
        full = G.retrieval_prompt(q[:90] + ("…" if len(q) > 90 else ""), d)
        yy = 432
        for ln in wrap(full, 13, 320):
            if yy > 600:
                break
            hl = "Use" in ln or "torch" in ln or "pipeline" in ln or "hub" in ln
            text(s, ln, (x1 + 26, yy), 13, C1 if hl else CODE_TXT)
            yy += 19
        if G.QUERIES[self.qi].reference == "mobilenet_v2" and not self.typed:
            para(s, "BM25 matches words, not numbers: '< 10M parameters' is invisible to it.",
                 60, 632, 760, 14, AMBER)


# ============================================================ 5. AST lab
class GorillaASTSlide(Slide):
    kicker, title = "Article 1 · Gorilla · AST lab", "Build an API call; the checker compares code trees"
    kicker_color = C1
    hints = ("1 / 2 / 3 or click: change function · repo · model   ·   Q: next task   ·   "
             "U: update the docs (fcn_resnet50 → 101)")
    PARTS = ["function", "repo", "model"]

    def enter(self):
        super().enter()
        self.qi = 1
        self.choice = [0, 0, 2]          # torch.hub.load('pytorch/vision', 'fcn_resnet50')
        self.updated = False
        self.t_change = 0.0
        self.tally = {"correct": 0, "error": 0, "hallucination": 0}
        self.seen = set()

    def code(self):
        f, r, m = (G.BUILDER[p][c] for p, c in zip(self.PARTS, self.choice))
        return G.build_call(f, r, m)

    def bump(self, i):
        self.choice[i] = (self.choice[i] + 1) % len(G.BUILDER[self.PARTS[i]])
        self.t_change = self.t

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_3:
            self.bump(k - pygame.K_1)
        elif k == pygame.K_q:
            self.qi = (self.qi + 1) % len(G.QUERIES)
            self.t_change = self.t
        elif k == pygame.K_u:
            self.updated = not self.updated
            self.t_change = self.t
        else:
            return False
        return True

    def click(self, pos, button):
        for i in range(3):
            if inside(pos, (84 + i * 250, 290, 236, 60)):
                self.bump(i)
                return True
        return False

    def tree(self, s, code, nodes, cx, top, title, col_title, width=500):
        """Root call node with one child per argument (green = matches, red = not)."""
        text(s, title, (cx, top), 13, col_title, True, anchor="midtop")
        parsed = G.parse_call(code)
        root = parsed[0] if parsed else "(does not parse)"
        rw = text_width(root, 14, True, True) + 24
        ry = top + 26
        status = {lbl.split(":")[0]: ok for lbl, ok in nodes[1:]} if nodes else {}
        root_ok = nodes[0][1] if nodes else None
        rc = GREEN if root_ok else CORAL if root_ok is False else GRAY
        kids = []
        if parsed:
            kids = [(k, v) for k, v in parsed[2].items()]
        n = max(1, len(kids))
        for j, (k, v) in enumerate(kids):
            x = cx - width / 2 + (j + 0.5) * width / n
            y = ry + 70
            ok = status.get(k)
            col = GREEN if ok else CORAL if ok is False else GRAY
            lab1, lab2 = k, repr(v)
            if len(lab2) > 18:
                lab2 = lab2[:16] + "…'"
            bw = min(width / n - 6, max(text_width(lab1, 11, True), text_width(lab2, 11, False, True)) + 14)
            y += (j % 2) * 40 if n > 2 else 0            # stagger crowded rows
            draw.line(s, col, (cx, ry + 30), (x, y), 2)
            rrect(s, CARD2, (x - bw / 2, y, bw, 42), 10)
            draw.rect(s, col, (x - bw / 2, y, bw, 42), 2, border_radius=10)
            text(s, lab1, (x, y + 5), 11, col, True, anchor="midtop")
            text(s, lab2, (x, y + 22), 11, TEXT, mono=True, anchor="midtop")
        rrect(s, CARD2, (cx - rw / 2, ry, rw, 32), 10)
        draw.rect(s, rc, (cx - rw / 2, ry, rw, 32), 2, border_radius=10)
        text(s, root, (cx, ry + 16), 14, TEXT, True, mono=True, anchor="center")

    def draw(self, s):
        q = G.QUERIES[self.qi]
        db = G.database(self.updated)
        code = self.code()
        verdict, ref, nodes = G.ast_verdict(code, q.reference, db)
        key = (code, self.qi, self.updated)
        if key not in self.seen:
            self.seen.add(key)
            self.tally[verdict] += 1
        # task
        rrect(s, CARD, (60, 140, 1160, 62), 14)
        text(s, f"TASK {self.qi + 1}/{len(G.QUERIES)}", (80, 150), 12, C1, True)
        para(s, q.prompt, 80, 170, 1000, 15, TEXT)
        if self.updated:
            chip(s, 1040, 152, "docs updated", AMBER, True, 13)
        # the builder: three clickable slots
        text(s, "YOUR CALL (click a slot to change it)", (84, 218), 12, FAINT, True)
        rrect(s, CODE_BG, (60, 236, 1160, 44), 10)
        text(s, code, (80, 248), 18, CODE_TXT, mono=True)
        for i, p in enumerate(self.PARTS):
            x = 84 + i * 250
            val = G.BUILDER[p][self.choice[i]]
            rrect(s, CARD, (x, 290, 236, 60), 12)
            text(s, f"{i + 1} · {p}", (x + 14, 298), 12, C1, True)
            v = val if len(val) <= 26 else val[:24] + "…"
            text(s, v, (x + 14, 320), 14, TEXT, True, mono=True)
        # verdict
        vc = VERDICT_COL[verdict]
        stamp(s, verdict.upper(), (1000, 318), vc, (self.t - self.t_change) / 0.3, 24, -6)
        expl = {"correct": "Sub-tree match with the reference API for this task.",
                "error": f"A real API ({ref['id'] if ref else ''}), but not the one this task needs.",
                "hallucination": "Matches nothing in the database: this API does not exist."}[verdict]
        text(s, expl, (1000, 352), 13, MUTED, anchor="midtop")
        # two trees: candidate vs reference
        rrect(s, CARD, (60, 380, 760, 254), 16)
        target = next(d for d in db if d["id"] == q.reference)
        _, ref_nodes = G.matches(target["api_call"], target["api_call"])
        self.tree(s, code, nodes if verdict != "error" else G.matches(code, target["api_call"])[1],
                  250, 394, "YOUR CALL'S TREE", TEXT, 360)
        self.tree(s, target["api_call"], ref_nodes, 630, 394, "REFERENCE API (from the database)", C1, 360)
        draw.line(s, GRID, (440, 400), (440, 620), 1)
        text(s, "optional arguments (pretrained=True, language=...) are not checked, as in the paper",
             (440, 612), 12, FAINT, anchor="midtop")
        # tally + how
        x1 = 850
        rrect(s, CARD, (x1, 380, 370, 254), 16)
        text(s, "CALLS YOU HAVE TRIED", (x1 + 20, 396), 13, C1, True)
        tot = max(1, sum(self.tally.values()))
        for j, v in enumerate(("correct", "error", "hallucination")):
            y = 426 + j * 40
            text(s, v, (x1 + 20, y), 15, VERDICT_COL[v], True)
            draw.rect(s, GRID, (x1 + 150, y + 4, 160, 14), border_radius=6)
            draw.rect(s, VERDICT_COL[v], (x1 + 150, y + 4, 160 * self.tally[v] / tot, 14), border_radius=6)
            text(s, str(self.tally[v]), (x1 + 350, y), 15, TEXT, True, anchor="topright")
        para(s, "Paper (Sec. 3.3): parse the call into an AST, then check whether it is a sub-tree of an API "
                "in the database. No match at all = hallucination." +
             ("  After the update the old fcn_resnet50 call no longer exists." if self.updated else ""),
             x1 + 20, 548, 330, 13, MUTED)


# ============================================================ 6. results scatter
PAPER_T1 = {   # paper Table 1: overall accuracy / hallucination (%), [hub][model]
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
SETTINGS = ["zero-shot", "BM25", "GPT-Index", "Oracle"]


class GorillaResultsSlide(Slide):
    kicker, title = "Article 1 · Gorilla · paper results", "Accuracy vs hallucination: where each model lands"
    kicker_color = C1
    hints = "R / click a chip: retriever   ·   H: hub (TorchHub / HuggingFace / TF Hub)   ·   trails show zero-shot → now"

    def enter(self):
        super().enter()
        self.si, self.hub = 0, 0
        self.pos = {m: np.array(self.target(m), float) for m in range(5)}

    def target(self, m, si=None):
        st = SETTINGS[self.si if si is None else si]
        return PAPER_T1[st]["hal"][self.hub][m], PAPER_T1[st]["acc"][self.hub][m]

    def key(self, k):
        if k == pygame.K_r:
            self.si = (self.si + 1) % 4
        elif k == pygame.K_h:
            self.hub = (self.hub + 1) % 3
        else:
            return False
        return True

    def click(self, pos, button):
        for i in range(4):
            if inside(pos, self.chips[i]) if hasattr(self, "chips") else False:
                self.si = i
                return True
        for i in range(3):
            if inside(pos, (900 + i * 106, 140, 100, 30)):
                self.hub = i
                return True
        return False

    def update(self, dt):
        super().update(dt)
        for m in range(5):
            self.pos[m] += (np.array(self.target(m)) - self.pos[m]) * min(1, dt * 5)

    def xy(self, hal, acc):
        return 124 + hal / 100 * 640, 596 - acc / 100 * 360

    def draw(self, s):
        # retriever chips
        self.chips = []
        x = 60
        for i, st in enumerate(SETTINGS):
            r = chip(s, x, 140, st, C1, i == self.si, 14)
            self.chips.append(r)
            x += r[2] + 10
        for i, h in enumerate(HUBS):
            on = i == self.hub
            rrect(s, CARD2 if on else CARD, (900 + i * 106, 140, 100, 30), 8)
            text(s, h if len(h) < 12 else "TF Hub", (950 + i * 106, 155), 13, TEXT if on else MUTED, on,
                 anchor="center")
        # plot
        x0, y0 = self.xy(0, 0)
        x1, y1 = self.xy(100, 100)
        rrect(s, CARD, (60, 186, 790, 470), 16)
        poly(s, (24, 52, 50), [(x0, y1), (x0 + 200, y1), (x0, y1 + 140)])
        text(s, "the good corner", (x0 + 8, y1 + 6), 12, GREEN)
        for v in (0, 25, 50, 75, 100):
            xa, _ = self.xy(v, 0)
            _, ya = self.xy(0, v)
            draw.line(s, GRID, (xa, y0), (xa, y1))
            draw.line(s, GRID, (x0, ya), (x1, ya))
            text(s, f"{v}", (xa, y0 + 6), 12, MUTED, anchor="midtop")
            text(s, f"{v}", (x0 - 8, ya), 12, MUTED, anchor="midright")
        text(s, "hallucination (%) →  lower is better", ((x0 + x1) / 2, y0 + 24), 13, FAINT, anchor="midtop")
        text(s, "accuracy (%) ↑", (x0 - 40, y1 - 28), 13, FAINT)
        for m in range(5):
            zx, zy = self.xy(*self.target(m, 0))
            cx, cy = self.xy(*self.pos[m])
            if self.si:
                draw.line(s, mix(CARD, MODEL_COLORS[m], 0.5), (zx, zy), (cx, cy), 2)
                circle(s, mix(CARD, MODEL_COLORS[m], 0.5), (zx, zy), 4)
            r = 13 if m == 4 else 9
            circle(s, MODEL_COLORS[m], (cx, cy), r)
            if cx > 700:
                text(s, MODELS[m], (cx - r - 6, cy - 8), 14, MODEL_COLORS[m], True, anchor="topright")
            else:
                text(s, MODELS[m], (cx + r + 6, cy - 8), 14, MODEL_COLORS[m], True)
        text(s, "Source: paper Table 1", (836, 640), 12, FAINT, anchor="bottomright")
        # notes
        x2 = 880
        st = SETTINGS[self.si]
        acc = PAPER_T1[st]["acc"][self.hub]
        hal = PAPER_T1[st]["hal"][self.hub]
        rrect(s, CARD, (x2, 186, 340, 230), 16)
        text(s, f"{HUBS[self.hub]} · {st}", (x2 + 20, 202), 15, C1, True)
        for m in range(5):
            y = 234 + m * 34
            circle(s, MODEL_COLORS[m], (x2 + 28, y + 9), 6)
            text(s, MODELS[m], (x2 + 42, y), 15, TEXT, m == 4)
            text(s, f"{acc[m]:.0f}% acc", (x2 + 230, y), 14, TEXT, anchor="topright")
            text(s, f"{hal[m]:.0f}% hal", (x2 + 320, y), 14, CORAL if hal[m] > 30 else MUTED, anchor="topright")
        note = {"zero-shot": "No docs at test time. Fine-tuning alone puts Gorilla ahead of GPT-4 on every hub.",
                "BM25": "A weak retriever often fetches the wrong doc and drags Gorilla's accuracy down (e.g. "
                        "TorchHub 59 → 40%). A bad retriever can hurt.",
                "GPT-Index": "A better retriever helps the prompted models catch up, and hallucination drops "
                             "towards 0 for Gorilla.",
                "Oracle": "With the perfect doc, Gorilla reaches 67 / 91 / 94%: reading docs is the skill "
                          "that keeps answers current."}[st]
        rrect(s, CARD2, (x2, 430, 340, 226), 16)
        text(s, "WHAT TO NOTICE", (x2 + 20, 446), 13, C1, True)
        para(s, note, x2 + 20, 472, 300, 16, TEXT)
