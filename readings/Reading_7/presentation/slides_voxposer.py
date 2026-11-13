"""
slides_voxposer.py - Article 2 (VoxPoser): a rotatable 3D value surface built
line by line, a planner lab (greedy vs A*, weights, the trap), and results as
a dot matrix + a waffle chart of 300 scenes.
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

C2 = AMBER
CODE_TXT = (220, 227, 239)
OBJ_COLORS = {"top drawer handle": TEAL, "vase": CORAL, "mug": LAV, "bowl": SKY}
SURF_STOPS = [(30, 27, 75), (88, 28, 135), (190, 50, 90), (245, 140, 40), (250, 230, 120)]


def norm01(m):
    return (m - m.min()) / (m.max() - m.min() + 1e-9)


# ============================================================ 7. idea: code builds a 3D surface
class VoxIdeaSlide(Slide):
    kicker, title = "Article 2 · VoxPoser", "Each line of LLM code reshapes a 3D value map"
    kicker_color = C2
    hints = "Drag the surface (or A / D) to rotate it   ·   R: replay the program   ·   → next: the planner lab"
    CODE = ["# \"Open the top drawer. Watch out for that vase!\"",
            "def affordance_map():",
            "    handle = detect('top drawer handle')",
            "    return distance_transform(handle.pos)",
            "",
            "def avoidance_map():",
            "    vase = detect('vase')",
            "    map[vase.occupancy] = 1",
            "    return gaussian_smooth(map)",
            "",
            "cost = -(2 * affordance - 1 * avoidance)"]
    # which code line is "running" at each moment (seconds after entry)
    TIMELINE = [(0.4, 1), (0.9, 2), (1.6, 3), (3.0, 5), (3.5, 6), (4.0, 7), (4.6, 8), (6.0, 10), (7.5, None)]
    GRID_N = 26

    def enter(self):
        super().enter()
        self.ang = 0.6
        self.drag = None
        sc = V.make_scene()
        m = V.compose(sc, V.INSTRUCTIONS[1])
        idx = np.linspace(0, V.N - 1, self.GRID_N).astype(int)
        self.aff = norm01(m["affordance"])[np.ix_(idx, idx)]
        self.avo = norm01(m["avoidance"])[np.ix_(idx, idx)]
        self.idx = idx
        self.scene = sc

    def key(self, k):
        if k == pygame.K_a:
            self.ang -= 0.25
        elif k == pygame.K_d:
            self.ang += 0.25
        elif k == pygame.K_r:
            self.t = 0.0
        else:
            return False
        return True

    def click(self, pos, button):
        if inside(pos, (560, 140, 660, 520)):
            self.drag = (pos[0], self.ang)
            return True
        return False

    def release(self, pos):
        self.drag = None

    def motion(self, pos, buttons):
        if self.drag:
            self.ang = self.drag[1] + (pos[0] - self.drag[0]) * 0.01

    def update(self, dt):
        super().update(dt)
        if not self.drag:
            self.ang += dt * 0.15

    def line_now(self):
        cur = None
        for t0, ln in self.TIMELINE:
            if self.t >= t0:
                cur = ln
        return cur

    def project(self, x, y, h):
        """x, y in [-1, 1] (table), h in [0, 1] (value) -> screen point."""
        ca, sa = math.cos(self.ang), math.sin(self.ang)
        rx, ry = x * ca - y * sa, x * sa + y * ca
        return 890 + rx * 230, 460 + ry * 105 - h * 190, ry

    def draw(self, s):
        cur = self.line_now()
        ga = ease((self.t - 1.6) / 1.2)          # affordance appears
        gb = ease((self.t - 4.6) / 1.2)          # avoidance carves the dip
        # ---- code panel
        rrect(s, CODE_BG, (60, 140, 480, 330), 16)
        for j, ln in enumerate(self.CODE):
            y = 156 + j * 27
            on = cur == j
            if on:
                rrect(s, (40, 46, 70), (66, y - 3, 468, 25), 6)
                circle(s, C2, (76, y + 10), 4)
            seen = cur is None or j <= (cur if cur is not None else 99)
            col = C2 if on else (CODE_TXT if seen else FAINT)
            if ln.startswith("#"):
                col = FAINT if not on else C2
            text(s, ln, (88, y), 14, col, on, mono=True)
        # ---- pipeline strip under the code
        stages = [("eye", "Perception", "OWL-ViT + SAM + XMem find objects", 0.4),
                  ("code", "LLM code", "writes the map functions above", 0.9),
                  ("layers", "Value maps", "100 × 100 × 100 voxels", 1.6),
                  ("route", "Planner", "search for a low-cost path", 6.0),
                  ("sync", "MPC at 5 Hz", "replan as the scene changes", 7.5)]
        for i, (ic, h1, h2, t0) in enumerate(stages):
            y = 490 + i * 34
            on = self.t >= t0 and (i == 4 or self.t < stages[i + 1][3])
            done = self.t >= t0
            icon(s, ic, (80, y + 14), 26, C2 if done else GRID)
            text(s, h1, (102, y + 4), 15, TEXT if on else (MUTED if done else FAINT), on)
            text(s, h2, (230, y + 6), 13, MUTED if done else FAINT)
        # ---- the surface
        rrect(s, CARD, (560, 140, 660, 520), 18)
        n = self.GRID_N
        hgt = np.clip(2 * self.aff * ga - 1.0 * self.avo * gb, -1, 2)
        h01 = (hgt + 1) / 3
        quads = []
        xs = np.linspace(-1, 1, n)
        for i in range(n - 1):
            for j in range(n - 1):
                c = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
                pts = [self.project(xs[b], xs[a], h01[a, b] * 0.9) for a, b in c]
                depth = sum(p[2] for p in pts) / 4
                hv = float(np.mean([h01[a, b] for a, b in c]))
                quads.append((depth, pts, hv))
        quads.sort(key=lambda q: q[0])
        cols = colormap(np.array([q[2] for q in quads]), SURF_STOPS)
        for (depth, pts, hv), col in zip(quads, cols):
            shade = 0.75 + 0.25 * (depth + 1.4) / 2.8
            c = tuple(int(min(255, v * shade)) for v in col)
            poly(s, c, [(p[0], p[1]) for p in pts])
        # object pins
        for name, lab in (("top drawer handle", "handle"), ("vase", "vase")):
            o = self.scene["objects"][name]["pos"]
            x, y = o[0] / (V.N - 1) * 2 - 1, o[1] / (V.N - 1) * 2 - 1
            ai, bi = int(round(o[1] / (V.N - 1) * (n - 1))), int(round(o[0] / (V.N - 1) * (n - 1)))
            base = self.project(x, y, h01[ai, bi] * 0.9)
            top = (base[0], base[1] - 60)
            draw.line(s, OBJ_COLORS[name], (base[0], base[1]), top, 2)
            circle(s, OBJ_COLORS[name], top, 6)
            text(s, lab, (top[0] + 10, top[1] - 8), 14, OBJ_COLORS[name], True)
        # legend
        text(s, "height = value (high = go here)", (584, 156), 13, MUTED)
        if gb > 0.5:
            text(s, "the vase digs a pit the planner keeps out of", (584, 176), 13, CORAL)
        elif ga > 0.5:
            text(s, "a smooth hill rising towards the handle", (584, 176), 13, TEAL)
        if self.t > 6.0:
            rrect(s, CARD2, (590, 590, 600, 54), 12)
            para(s, "Paper: cost = −(2 × affordance − 1 × avoidance) on a 100³ voxel grid. "
                    "The planner then descends this map; no training is needed.", 606, 598, 570, 14, TEXT)


# ============================================================ 8. the planner lab
class VoxLabSlide(Slide):
    kicker, title = "Article 2 · VoxPoser · planner lab", "Tune the map, pick a planner, spring the trap"
    kicker_color = C2
    hints = ("1-3: instructions · 4: the trap   ·   G: greedy ↔ A*   ·   drag sliders or objects   ·   "
             "M: map view   ·   P: replay")
    VIEWS = ["cost", "affordance", "avoidance"]
    MX, MY, MS = 700, 140, 516           # map rect

    def enter(self):
        super().enter()
        self.ii, self.astar, self.view = 1, False, 0
        self.w = [2.0, 1.0]
        self.drag, self.slide_drag = None, None
        self.load()

    def load(self):
        self.scene = V.trap_scene() if self.ii == 3 else V.make_scene()
        self.dirty, self.t_comp = True, -1.0
        self.compute()

    def instr(self):
        return V.TRAP_INSTRUCTION if self.ii == 3 else V.INSTRUCTIONS[self.ii]

    def compute(self):
        ins = self.instr()
        m = V.compose(self.scene, ins)
        tgt = m["target"]
        affs = [V.affordance_map(tgt)]
        avos = [V.avoidance_map(V.detect(self.scene, n), mg) for n, mg in ins["avoid"]]
        self.cost = V.cost_map(affs, avos, self.w[0], self.w[1])
        self.maps = {"cost": self.cost, "affordance": affs[0],
                     "avoidance": sum(avos) if avos else np.zeros((V.N, V.N))}
        start, goal = self.scene["gripper"], tgt["pos"]
        self.greedy = V.greedy_plan(self.cost, start, goal)
        self.astar_p = V.astar_plan(self.cost, start, goal, blocked=m["occupied"])
        self.dirty, self.t_comp, self.t_anim = False, self.t, self.t
        self.img = None

    def path(self, astar=None):
        a = self.astar if astar is None else astar
        return (self.astar_p if a else self.greedy)

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_4:
            self.ii = k - pygame.K_1
            self.load()
        elif k == pygame.K_g:
            self.astar = not self.astar
            self.t_anim = self.t
        elif k == pygame.K_m:
            self.view = (self.view + 1) % 3
            self.img = None
        elif k == pygame.K_p:
            self.t_anim = self.t
        else:
            return False
        return True

    def to_map(self, p):
        k = self.MS / V.N
        return self.MX + (p[0] + 0.5) * k, self.MY + (p[1] + 0.5) * k

    def from_map(self, pos):
        k = self.MS / V.N
        return np.clip((pos[0] - self.MX) / k - 0.5, 2, V.N - 3), np.clip((pos[1] - self.MY) / k - 0.5, 2, V.N - 3)

    def sliders(self):
        return [(84, 372, 300), (84, 432, 300)]

    def click(self, pos, button):
        for i, (x, y, w) in enumerate(self.sliders()):
            if inside(pos, (x - 10, y + 14, w + 20, 28)):
                self.slide_drag = i
                self.motion(pos, (1, 0, 0))
                return True
        for i in range(4):
            if inside(pos, (84 + (i % 2) * 150, 180 + (i // 2) * 42, 140, 34)):
                self.key(pygame.K_1 + i)
                return True
        if inside(pos, (84, 268, 300, 30)):
            self.key(pygame.K_g)
            return True
        if inside(pos, (self.MX, self.MY, self.MS, self.MS)):
            items = [("gripper", self.scene["gripper"])] + [(n, o["pos"]) for n, o in self.scene["objects"].items()]
            for name, p in items:
                if math.dist(self.to_map(p), pos) < 22:
                    self.drag = name
                    return True
        return False

    def release(self, pos):
        if self.drag or self.slide_drag is not None:
            self.dirty = True
        self.drag, self.slide_drag = None, None

    def motion(self, pos, buttons):
        if self.slide_drag is not None:
            x, y, w = self.sliders()[self.slide_drag]
            f = np.clip((pos[0] - x) / w, 0, 1)
            self.w[self.slide_drag] = round(float(f * 4.0), 1)
            self.dirty = True
        elif self.drag:
            p = np.array(self.from_map(pos))
            if self.drag == "gripper":
                self.scene["gripper"] = p
            else:
                self.scene["objects"][self.drag]["pos"] = p
            self.dirty = True

    def update(self, dt):
        super().update(dt)
        if self.dirty and self.t - self.t_comp > 0.18:
            self.compute()

    def stats(self, path):
        names = [n for n in self.scene["objects"] if n != self.instr()["target"]]
        pts = np.array(path, float)
        length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0
        clear = min(float(np.min(np.hypot(pts[:, 0] - o["pos"][0], pts[:, 1] - o["pos"][1]) - o["radius"]))
                    for o in (self.scene["objects"][n] for n in names))
        hits = [n for n in names if any(V.collides(self.scene, p, [n]) for p in path)]
        return length, clear, hits

    def draw(self, s):
        # ---------------- left: controls
        ins = self.instr()
        rrect(s, CARD, (60, 140, 610, 516), 18)
        text(s, "INSTRUCTION", (84, 156), 12, C2, True)
        labels = ["1 · drawer", "2 · drawer + vase", "3 · bowl, keep 15 cm", "4 · the trap"]
        for i, lab in enumerate(labels):
            x, y = 84 + (i % 2) * 150, 180 + (i // 2) * 42
            on = i == self.ii
            rrect(s, C2 if on else CARD2, (x, y, 140, 34), 10)
            text(s, lab, (x + 70, y + 17), 12 if len(lab) > 16 else 13, BG if on else TEXT, True, anchor="center")
        para(s, f"\"{ins['text']}\"", 400, 182, 250, 14, TEXT)
        toggle(s, 84, 270, self.astar, "A* search" if self.astar else "greedy descent (paper)", C2)
        text(s, "G", (370, 274), 13, FAINT, anchor="topright")
        text(s, "COST WEIGHTS", (84, 340), 12, C2, True)
        for i, (x, y, w) in enumerate(self.sliders()):
            slider(s, x, y, w, self.w[i], 0, 4, ["affordance weight (paper: 2)", "avoidance weight (paper: 1)"][i],
                   [TEAL, CORAL][i], self.slide_drag == i, "{:.1f}")
        # readouts
        path, reached = self.path()
        length, clear, hits = self.stats(path)
        o_path, o_reached = self.path(not self.astar)
        o_len = self.stats(o_path)[0]
        cards = [("path", f"{length:.0f} cm", MUTED),
                 ("closest pass", f"{clear:+.0f} cm", CORAL if clear < 0 else (AMBER if clear < 5 else GREEN)),
                 ("result", "arrived" if reached and not hits else ("hit " + hits[0].split()[-1] if hits else "stuck"),
                  GREEN if reached and not hits else CORAL)]
        for j, (h1, v, col) in enumerate(cards):
            x = 400 + 0 * j
            y = 330 + j * 50
            text(s, h1.upper(), (x, y), 11, FAINT, True)
            text(s, v, (x, y + 16), 20, col, True)
        rrect(s, CARD2, (84, 500, 562, 136), 14)
        if self.ii == 3:
            msg = (f"Greedy descent walks into the cup formed by the mug, vase and bowl: it only looks one step "
                   f"ahead, so it must back out ({len(self.greedy[0])} steps). A* searches the whole map "
                   f"({len(self.astar_p[0])} steps). VoxPoser's greedy planner relies on MPC replanning and "
                   f"sensible maps to avoid traps like this.")
        elif self.w[1] == 0 and ins["avoid"]:
            msg = "Avoidance weight 0: the map no longer knows about the vase, so the path goes straight through it."
        elif not ins["avoid"]:
            msg = ("No avoidance map in this program: the path is a straight line, even if a vase is in the way. "
                   "Drag the vase onto the line and watch it get hit.")
        else:
            msg = (f"The other planner's path (dashed) is {o_len:.0f} cm. Raising the avoidance weight buys "
                   "clearance at the cost of a longer path. Drag any object: maps and plans recompute live.")
        para(s, msg, 102, 512, 528, 14, TEXT)
        # ---------------- right: the map
        if self.img is None:
            m = self.maps[self.VIEWS[self.view]]
            a = norm01(m)
            if self.view == 0:
                a = 1 - a
                stops = [(16, 24, 44), (30, 80, 110), (45, 212, 191), (245, 165, 36)]
            elif self.view == 1:
                stops = [(16, 24, 44), (20, 90, 90), (45, 212, 191)]
            else:
                stops = [(16, 24, 44), (120, 40, 60), (248, 113, 113)]
            self.img = colormap(a, stops)
        blit_array(s, self.img, (self.MX, self.MY, self.MS, self.MS))
        draw.rect(s, GRID, (self.MX, self.MY, self.MS, self.MS), 2)
        text(s, f"{self.VIEWS[self.view]} map (M)", (self.MX + 10, self.MY + 8), 13, TEXT, True)
        # other planner's path (ghost)
        pts = [self.to_map(p) for p in o_path]
        for k in range(0, len(pts) - 1, 2):
            draw.line(s, (200, 205, 215), pts[k], pts[k + 1], 1)
        # this planner's path, drawn up to the animated gripper
        pts = [self.to_map(p) for p in path]
        n_show = min(len(pts), int((self.t - self.t_anim) * 60) + 2)
        if n_show > 1:
            draw.lines(s, TEXT, False, pts[:n_show], 3)
        for name, o in self.scene["objects"].items():
            c = self.to_map(o["pos"])
            r = o["radius"] * self.MS / V.N
            circle(s, OBJ_COLORS[name], c, r)
            if name in [n for n, _ in ins["avoid"]]:
                draw.circle(s, OBJ_COLORS[name], c, r + 6, 1)
        for name, o in self.scene["objects"].items():           # labels last, so circles never hide them
            c = self.to_map(o["pos"])
            r = o["radius"] * self.MS / V.N
            lab = "handle" if name.startswith("top") else name
            w = text_width(lab, 12, True) + 10
            if lab == "handle":                                   # above the target
                x, y = c[0] - w / 2, c[1] - r - 26
            else:                                                 # to the right of an obstacle
                x, y = c[0] + r + 4, c[1] - 10
                if x + w > self.MX + self.MS:
                    x = c[0] - r - 4 - w
            rrect(s, BG_DEEP, (x, y, w, 20), 6, alpha=190)
            text(s, lab, (x + 5, y + 10), 12, OBJ_COLORS[name], True, anchor="midleft")
        g = pts[min(n_show, len(pts)) - 1]
        circle(s, TEXT, g, 9)
        circle(s, BG, g, 4)
        gp = self.to_map(self.scene["gripper"])
        draw.circle(s, TEXT, gp, 11, 2)
        text(s, "start", (gp[0], gp[1] - 26), 12, TEXT, anchor="midtop")


# ============================================================ 9. results: dot matrix + waffle
class VoxResultsSlide(Slide):
    kicker, title = "Article 2 · VoxPoser · results", "Every dot is a trial: the paper's robot and my 300 scenes"
    kicker_color = C2
    hints = "1-4 or click: switch the condition in the waffle   ·   each square = one random scene"
    PAPER = [("Move & Avoid", 0, 0, 9, 8), ("Set Up Table", 7, 0, 9, 7), ("Close Drawer", 0, 0, 10, 7),
             ("Open Bottle", 5, 0, 7, 5), ("Sweep Trash", 0, 0, 9, 8)]
    CONDS = [("aff_only", "affordance only"), ("aff_avoid", "+ avoidance"),
             ("open_loop_dist", "moved, plan once"), ("mpc_dist", "moved, MPC replans")]
    OUT_COL = [GREEN, CORAL, GRAY]

    def enter(self):
        super().enter()
        self.ci, self.t_c = 1, 0.0

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_4:
            self.ci, self.t_c = k - pygame.K_1, self.t
            return True
        return False

    def click(self, pos, button):
        for i, r in enumerate(getattr(self, "chips", [])):
            if inside(pos, r):
                self.ci, self.t_c = i, self.t
                return True
        return False

    def draw(self, s):
        # ---- paper dot matrix
        rrect(s, CARD, (60, 140, 560, 516), 18)
        text(s, "REAL ROBOT · successes out of 10 (paper Table 1)", (82, 156), 13, C2, True)
        heads = [("LLM + primitives", GRAY), ("VoxPoser", C2)]
        for c, (h, col) in enumerate(heads):
            text(s, h, (250 + c * 180, 186), 14, col, True)
            text(s, "static", (250 + c * 180, 206), 11, FAINT)
            text(s, "disturbed", (340 + c * 180, 206), 11, FAINT)
        for i, row in enumerate(self.PAPER):
            y = 234 + i * 70
            text(s, row[0], (82, y + 12), 15, TEXT)
            for c, val in enumerate(row[1:]):
                x0 = 250 + (c // 2) * 180 + (c % 2) * 90
                col = GRAY if c < 2 else C2
                for d in range(10):
                    dx, dy = x0 + (d % 5) * 15, y + (d // 5) * 15 + 4
                    on = d < val and self.t > 0.2 + i * 0.12 + d * 0.03
                    circle(s, col if on else GRID, (dx + 5, dy + 5), 5)
        para(s, "Totals: primitives 24% static / 0% disturbed · VoxPoser 88% / 70%. "
                "The maps generalise to new instructions with zero training.", 82, 590, 516, 14, MUTED)
        # ---- my waffle
        S = self.app.context["summary"]["voxposer"]
        x0 = 650
        rrect(s, CARD, (x0, 140, 570, 516), 18)
        text(s, f"MY 2-D DEMO · {S['n_scenes']} random scenes", (x0 + 22, 156), 13, C2, True)
        self.chips = []
        cx = x0 + 22
        for i, (_, lab) in enumerate(self.CONDS):          # two rows of two chips
            if i == 2:
                cx = x0 + 22
            r = chip(s, cx, 180 + (i // 2) * 36, f"{i + 1} {lab}", C2, i == self.ci, 12)
            self.chips.append(r)
            cx += r[2] + 6
        key, lab = self.CONDS[self.ci]
        outs = S[key].get("outcomes", [])
        cols, cell = 25, 18
        gx, gy = x0 + 30, 266
        for i, o in enumerate(outs):
            r, c = divmod(i, cols)
            g = ease((self.t - self.t_c - i * 0.003) / 0.25)
            col = mix(GRID, self.OUT_COL[o], g)
            rrect(s, col, (gx + c * (cell + 2), gy + r * (cell + 2), cell, cell), 3)
        counts = [outs.count(k) for k in range(3)]
        y = gy + 12 * (cell + 2) + 14
        for k, lab2 in enumerate(["arrived cleanly", "hit an object", "never arrived"]):
            rrect(s, self.OUT_COL[k], (x0 + 30 + k * 180, y + 2, 14, 14), 3)
            text(s, f"{lab2}: {counts[k]}", (x0 + 52 + k * 180, y), 14, TEXT)
        notes = {"aff_only": "Without the avoidance map the path runs straight through the vase in over half "
                             "the scenes.",
                 "aff_avoid": "Adding the LLM's avoidance map (weight 1) keeps the gripper clear in most scenes.",
                 "open_loop_dist": "The handle moves mid-way. Following the first plan blindly almost never "
                                   "arrives.",
                 "mpc_dist": "Replanning every step (MPC, like the paper's 5 Hz loop) recovers from the move."}
        para(s, notes[key], x0 + 30, y + 30, 510, 14, MUTED)
