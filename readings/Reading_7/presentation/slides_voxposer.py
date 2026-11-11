"""
slides_voxposer.py - Article 2 (VoxPoser) slides.
"""

import os
import sys

import numpy as np

from engine import *  # noqa: F401,F403
from engine import Slide

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from demos import voxposer_demo as V  # noqa: E402

C2 = AMBER
OBJ_COLORS = {"top drawer handle": TEAL, "vase": CORAL, "mug": LAV, "bowl": SKY}


def colormap(a, stops):
    """Map an array in [0, 1] to RGB by linear interpolation between colour stops."""
    a = np.clip(a, 0, 1)
    xs = np.linspace(0, 1, len(stops))
    st = np.array(stops, float)
    return np.stack([np.interp(a, xs, st[:, c]) for c in range(3)], -1).astype(np.uint8)


def blit_array(s, rgb, rect):
    """Draw an (h, w, 3) uint8 image into a design-space rectangle."""
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
    r = RECT(rect)
    s.blit(pygame.transform.smoothscale(surf, (max(1, r.w), max(1, r.h))), r.topleft)


# ============================================================ 6. VoxPoser idea
class VoxIdeaSlide(Slide):
    kicker, title = "Article 2 · VoxPoser", "The LLM writes code that paints a 3D map of where to go"
    kicker_color = C2
    hints = "Watch the program being written, then follow the pipeline   ·   → next: drive it yourself"
    CODE = ["# Open the top drawer. Watch out for that vase!",
            "def affordance_map():",
            "    map = np.zeros((100, 100, 100))",
            "    handles = detect('handle')",
            "    top = max(handles, key=lambda h: h.pos[2])",
            "    map[top.pos] = 1",
            "    return smooth(map)",
            "",
            "def constraint_map():",
            "    map = np.zeros((100, 100, 100))",
            "    vase = detect('vase')[0]",
            "    map[vase.occupancy_grid] = -1",
            "    return smooth(map)"]

    def draw(self, s):
        rrect(s, CODE_BG, (60, 150, 600, 470), 16)
        text(s, "What GPT-4 writes (simplified from the paper's Fig. 2)", (82, 166), 14, C2, True)
        chars = int(self.t * 110)
        y = 200
        for line in self.CODE:
            shown = line[:max(0, chars)]
            chars -= len(line) + 1
            col = (124, 139, 168) if line.strip().startswith("#") else (220, 227, 239)
            text(s, shown, (84, y), 15, col, mono=True)
            y += 29
        x0 = 690
        stages = [("eye", "Perception (VLM)", "OWL-ViT finds objects, Segment Anything masks them, XMem tracks them"),
                  ("map", "Value maps", "affordance = attract (handle), avoidance = repel (vase); 100³ voxels"),
                  ("route", "Motion planner", "greedy search over the combined cost map → end-effector path"),
                  ("sync", "MPC at 5 Hz", "execute the first waypoint, re-observe, replan; nothing is trained")]
        for i, (ic, h1, body) in enumerate(stages):
            a = self.appear(0.6 + 0.4 * i)
            y = 150 + i * 118 + 14 * (1 - a)
            rrect(s, CARD, (x0, y, 530, 104), 14)
            icon(s, ic, (x0 + 44, y + 52), 54, C2 if a > 0.5 else FAINT)
            text(s, h1, (x0 + 86, y + 16), 19, TEXT, True)
            para(s, body, x0 + 86, y + 46, 424, 15, MUTED)
            if i < 3:
                arrow(s, FAINT, (x0 + 265, y + 105), (x0 + 265, y + 117), 2, 6)


# ============================================================ 7. VoxPoser demo
class VoxDemoSlide(Slide):
    kicker, title = "Article 2 · VoxPoser · live demo", "Value maps + replanning: drag the objects while it moves"
    kicker_color = C2
    hints = "1-3: instruction · M: cost / affordance / avoidance view · drag any object · R: restart"
    VIEWS = ["cost", "affordance", "avoidance"]
    PANEL = (60, 150, 470, 470)

    def enter(self):
        super().enter()
        self.ii, self.vi = 1, 0
        self.reset()

    def reset(self):
        self.scene = V.make_scene()
        self.trail = [self.scene["gripper"].copy()]
        self.plan, self.drag, self.acc = [], None, 0.0
        self.done_t, self.collided, self.reached = 0.0, False, False

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_3:
            self.ii = k - pygame.K_1
            self.reset()
        elif k == pygame.K_m:
            self.vi = (self.vi + 1) % 3
        elif k == pygame.K_r:
            self.reset()
        else:
            return False
        return True

    # grid <-> design coordinates
    def to_px(self, p):
        x, y, w, h = self.PANEL
        return (x + p[0] / V.N * w, y + p[1] / V.N * h)

    def to_grid(self, pos):
        x, y, w, h = self.PANEL
        return np.array([(pos[0] - x) / w * V.N, (pos[1] - y) / h * V.N])

    def click(self, pos, button):
        g = self.to_grid(pos)
        cands = list(self.scene["objects"].items()) + [("gripper", dict(pos=self.scene["gripper"], radius=4))]
        for name, o in cands:
            if np.hypot(*(g - o["pos"])) <= o["radius"] + 4:
                self.drag = name
                return True
        return False

    def motion(self, pos, buttons):
        if self.drag and buttons[0]:
            g = np.clip(self.to_grid(pos), 3, V.N - 4)
            if self.drag == "gripper":
                self.scene["gripper"][:] = g
            else:
                self.scene["objects"][self.drag]["pos"][:] = g
            self.done_t, self.reached = 0.0, False

    def release(self, pos):
        self.drag = None

    def update(self, dt):
        super().update(dt)
        instr = V.INSTRUCTIONS[self.ii]
        self.maps = V.compose(self.scene, instr)
        if self.reached:
            self.done_t += dt
            if self.done_t > 1.8:
                self.scene["gripper"][:] = (12, 50)
                self.trail, self.reached, self.collided = [self.scene["gripper"].copy()], False, False
            return
        self.acc += dt
        # MPC: replan from the current state every step, execute the first move
        while self.acc > 0.045:
            self.acc -= 0.045
            self.plan, _ = V.greedy_plan(self.maps["cost"], self.scene["gripper"], self.maps["target"]["pos"])
            if len(self.plan) > 1:
                self.scene["gripper"][:] = self.plan[1]
                self.trail.append(self.scene["gripper"].copy())
                self.trail = self.trail[-400:]
            others = [n for n in self.scene["objects"] if n != instr["target"]]
            self.collided |= V.collides(self.scene, self.scene["gripper"], [n for n, _ in instr["avoid"]] or others)
            if np.hypot(*(self.scene["gripper"] - self.maps["target"]["pos"])) <= 2.5:
                self.reached = True
                break

    def draw(self, s):
        instr = V.INSTRUCTIONS[self.ii]
        view = self.VIEWS[self.vi]
        m = self.maps[view]
        if view == "cost":
            a = (m - m.min()) / (m.max() - m.min() + 1e-9)
            rgb = colormap(1 - a, [(16, 24, 44), (30, 80, 110), (45, 212, 191), (245, 165, 36)])
        elif view == "affordance":
            rgb = colormap(m, [(16, 24, 44), (40, 90, 120), (45, 212, 191)])
        else:
            mx = m.max() if m.max() > 0 else 1
            rgb = colormap(m / mx, [(16, 24, 44), (120, 40, 60), (248, 113, 113)])
        x, y, w, h = self.PANEL
        rrect(s, CARD, (x - 8, y - 8, w + 16, h + 16), 14)
        blit_array(s, rgb, self.PANEL)
        # planned path (what MPC would do next) and executed trail
        if len(self.plan) > 1:
            draw.lines(s, (230, 234, 242), False, [self.to_px(p) for p in self.plan], 2)
        if len(self.trail) > 1:
            draw.lines(s, AMBER, False, [self.to_px(p) for p in self.trail], 3)
        for name, o in self.scene["objects"].items():
            cx, cy = self.to_px(o["pos"])
            r = o["radius"] / V.N * w
            circle(s, OBJ_COLORS[name], (cx, cy), max(5, r))
            text(s, name, (cx, cy + max(5, r) + 4), 13, TEXT, True, anchor="midtop")
        gx, gy = self.to_px(self.scene["gripper"])
        circle(s, (240, 244, 250), (gx, gy), 7)
        circle(s, BG, (gx, gy), 3)
        text(s, "gripper", (gx, gy - 12), 12, TEXT, anchor="midbottom")
        text(s, f"view: {view} map  (top-down slice, 100 × 100 voxels)", (x, y + h + 12), 13, MUTED)
        # right panel
        x0 = 570
        rrect(s, CARD, (x0, 150, 650, 92), 14)
        text(s, f"INSTRUCTION {self.ii + 1}/3", (x0 + 22, 162), 13, C2, True)
        para(s, f"\"{instr['text']}\"", x0 + 22, 186, 606, 19, TEXT, True)
        rrect(s, CODE_BG, (x0, 256, 650, 246), 14)
        text(s, "LLM program (example of what the LLM writes)", (x0 + 22, 268), 13, C2, True)
        for j, line in enumerate(instr["code"]):
            text(s, line, (x0 + 24, 294 + j * 23), 14, (220, 227, 239), mono=True)
        status, col = (("Reached the target", GREEN) if self.reached else ("Moving: replanning every step", C2))
        if self.collided:
            status, col = "Touched an obstacle", CORAL
        rrect(s, CARD2, (x0, 516, 650, 104), 14)
        badge_w = text_width(status, 15, True) + 24
        rrect(s, col, (x0 + 22, 530, badge_w, 28), 14)
        text(s, status, (x0 + 22 + badge_w / 2, 544), 15, BG, True, anchor="center")
        para(s, "Cost = −(2 × affordance − 1 × avoidance), the paper's weights. Drag the vase onto the path or "
                "move the handle: the map and plan update instantly (closed-loop MPC).",
             x0 + 22, 566, 606, 14, MUTED)


# ============================================================ 8. VoxPoser results
class VoxResultsSlide(Slide):
    kicker, title = "Article 2 · VoxPoser · results", "Maps beat primitives; replanning beats disturbances"
    kicker_color = C2
    hints = "Left: paper's real-robot table · right: my 2-D demo on 300 random scenes (results/summary.json)"
    PAPER = [("Move & Avoid", 0, 0, 9, 8), ("Set Up Table", 7, 0, 9, 7), ("Close Drawer", 0, 0, 10, 7),
             ("Open Bottle", 5, 0, 7, 5), ("Sweep Trash", 0, 0, 9, 8)]

    def draw(self, s):
        g = ease(self.t / 1.0)
        rrect(s, CARD, (60, 150, 560, 470), 18)
        text(s, "Real robot, successes out of 10 (paper Table 1)", (82, 166), 17, TEXT, True)
        cols = [("LLM + primitives", GRAY), ("VoxPoser", C2)]
        for c, (lab, col) in enumerate(cols):
            circle(s, col, (98 + c * 150, 206), 6)
            text(s, lab, (110 + c * 150, 197), 14, TEXT)
        text(s, "solid = static · faded = disturbed", (600, 198), 12, FAINT, anchor="topright")
        for i, (task, ps, pd, vs, vd) in enumerate(self.PAPER):
            y = 232 + i * 68
            text(s, task, (82, y + 18), 15, TEXT)
            for j, (val, col, alpha) in enumerate(((ps, GRAY, None), (pd, GRAY, 110), (vs, C2, None), (vd, C2, 110))):
                bw = 32 * val * g
                yy = y + j * 14
                rrect(s, col, (230, yy + 4, max(2, bw), 11), 4, alpha=alpha)
            text(s, f"{vs}/10 · {vd}/10", (560, y + 18), 13, C2, True, anchor="topright")
        text(s, "Total: primitives 24% / 0%   ·   VoxPoser 88% / 70%", (82, 584), 16, C2, True)
        # my demo
        S = self.app.context["summary"]["voxposer"]
        x0 = 650
        rrect(s, CARD, (x0, 150, 570, 300), 18)
        text(s, f"My 2-D demo ({S['n_scenes']} random scenes, vase in the way)", (x0 + 22, 166), 17, TEXT, True)
        rows = [("Affordance map only", S["aff_only"]["reached"], S["aff_only"]["collided"]),
                ("+ avoidance map", S["aff_avoid"]["reached"], S["aff_avoid"]["collided"]),
                ("Handle moved mid-way, plan once", S["open_loop_dist"]["reached"], None),
                ("Handle moved mid-way, MPC replans", S["mpc_dist"]["reached"], None)]
        for i, (lab, ok, col_rate) in enumerate(rows):
            y = 206 + i * 56
            text(s, lab, (x0 + 22, y), 15, TEXT)
            bw = 300 * ok * g
            rrect(s, GREEN if i in (1, 3) else GRAY, (x0 + 22, y + 22, max(2, bw), 16), 5)
            extra = f"  ·  hit the vase {col_rate:.0%}" if col_rate is not None else ""
            text(s, f"{ok:.0%} clean arrivals{extra}", (x0 + 30 + bw, y + 20), 13, MUTED)
        rrect(s, CARD2, (x0, 466, 570, 154), 18)
        text(s, "Learning dynamics with a VoxPoser prior (paper Table 3)", (x0 + 22, 480), 16, C2, True)
        para(s, "Door / window / fridge: zero-shot 7% / 3% / 18% success. Using the zero-shot trajectory to guide "
                "exploration reached 88% / 80% / 92% in under 3 minutes of interaction; exploring without the "
                "prior went past the 12-hour limit.", x0 + 22, 510, 530, 15, TEXT)
