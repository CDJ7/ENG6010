"""
interactive_presentation.py - the Readings 6 talk as a live, animated Python app.

Unlike the .pptx, the models on these slides are *running*: the pendulums are
simulated with the exact Pendulum-v1 physics and controlled in real time by
the world model, CEM planner and distilled policy that main.py trained
(results/models/*.pt). All reported numbers come from results/summary.json.

    cd readings/Readings_6
    python main.py                                      # once, to create results/
    python presentation/interactive_presentation.py     # start the talk
    python presentation/interactive_presentation.py --fullscreen

Global keys
    Right / Space / PageDown   next slide         Left / PageUp   previous slide
    F                          toggle fullscreen  Esc             quit
Each slide lists its own interactive keys in the bar at the bottom.

How the slides connect to the readings
    * "You are the controller"  - Pendulum-v1 and LeCun's fixed intrinsic cost
    * "Two modes"               - LeCun's Mode-2 (plan) vs Mode-1 (react), timing budget
    * "Compounding error"       - Diffusion World Model Fig. 1.1, why MPC replans
    * "Imagination"             - CEM plans 15 steps ahead (DreamerV3 imag_length: 15)
    * "DAgger"                  - LeCun Fig. 5: compile Mode-2 into Mode-1
    * "The race"                - DWM: the deployed policy is fast
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # readings/Readings_6
RESULTS = os.path.join(ROOT, "results")
ICONS = os.path.join(HERE, "icons")
sys.path.insert(0, ROOT)

# utils.py (imported by planner.py) forces SDL's "dummy" video driver so the
# training code can render off-screen. Here we want a real window, so we
# remember whether the user set a driver and undo the default afterwards.
_user_driver = os.environ.get("SDL_VIDEODRIVER")
import torch  # noqa: E402

from config import get_config  # noqa: E402
from planner import A_HIGH, A_LOW, CEMPlanner, RandomShootingPlanner  # noqa: E402
from policy import Policy, PolicyController  # noqa: E402
from collect_data import RandomController  # noqa: E402
from cost import pendulum_cost  # noqa: E402
from world_model import WorldModel  # noqa: E402

if _user_driver is None:
    os.environ.pop("SDL_VIDEODRIVER", None)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
import pygame.gfxdraw  # noqa: E402

# ------------------------------------------------------------------ palette
# Same colours as the .pptx: amber = Mode-2 planner, teal = Mode-1 policy.
BG = (13, 20, 36)
BG_DEEP = (10, 16, 32)
CARD = (23, 33, 58)
CARD2 = (31, 44, 74)
CODE_BG = (10, 16, 32)
TEXT = (242, 245, 250)
MUTED = (167, 179, 201)
FAINT = (86, 99, 125)
GRID = (42, 55, 86)
AMBER = (245, 165, 36)
TEAL = (45, 212, 191)
VIOLET = (167, 139, 250)
CORAL = (248, 113, 113)
GRAY = (138, 148, 168)
SKY = (125, 211, 252)
LAV = (196, 181, 253)
CTRL_COLORS = {"Random": GRAY, "Random-shooting MPC": VIOLET, "CEM MPC": AMBER, "Distilled policy": TEAL}

W, H = 1280, 720
SIM_DT = 0.05            # Pendulum-v1 time step (20 Hz control)


# ====================================================================== utils
def ease(t):
    """Ease-out cubic on [0, 1]."""
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def fmt_num(v, f=".0f"):
    return format(v, f).replace("-", "−")


class Fonts:
    """Font cache. Calibri/Segoe on Windows, Carlito/DejaVu/Arial elsewhere."""

    SANS = "calibri,carlito,segoeui,helveticaneue,arial,dejavusans"
    MONO = "consolas,menlo,dejavusansmono,couriernew,freemono"

    def __init__(self):
        self.cache = {}

    def get(self, size, bold=False, mono=False):
        key = (size, bold, mono)
        if key not in self.cache:
            self.cache[key] = pygame.font.SysFont(self.MONO if mono else self.SANS, size, bold=bold)
        return self.cache[key]


FONTS = None


def text(surf, s, pos, size=20, color=TEXT, bold=False, mono=False, anchor="topleft", alpha=255):
    img = FONTS.get(size, bold, mono).render(s, True, color)
    if alpha < 255:
        img.set_alpha(alpha)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def wrap(s, size, width, bold=False):
    f = FONTS.get(size, bold)
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.size(t)[0] <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def para(surf, s, x, y, width, size=18, color=MUTED, bold=False, gap=4, alpha=255):
    for line in wrap(s, size, width, bold):
        r = text(surf, line, (x, y), size, color, bold, alpha=alpha)
        y += r.height + gap
    return y


def rrect(surf, color, rect, radius=14, alpha=None):
    if alpha is None:
        pygame.draw.rect(surf, color, rect, border_radius=radius)
    else:
        tmp = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(tmp, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
        surf.blit(tmp, (rect[0], rect[1]))


def circle(surf, color, center, r):
    x, y = int(center[0]), int(center[1])
    pygame.gfxdraw.aacircle(surf, x, y, int(r), color)
    pygame.gfxdraw.filled_circle(surf, x, y, int(r), color)


def thick_line(surf, color, p1, p2, width):
    """Anti-aliased thick line with round caps (used for the pendulum rod)."""
    (x1, y1), (x2, y2) = p1, p2
    d = math.hypot(x2 - x1, y2 - y1) or 1.0
    nx, ny = -(y2 - y1) / d * width / 2, (x2 - x1) / d * width / 2
    pts = [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)]
    pygame.gfxdraw.aapolygon(surf, pts, color)
    pygame.gfxdraw.filled_polygon(surf, pts, color)
    circle(surf, color, p1, width / 2)
    circle(surf, color, p2, width / 2)


def arrow(surf, color, p1, p2, width=2, head=10):
    pygame.draw.line(surf, color, p1, p2, width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    a1 = (p2[0] - head * math.cos(ang - 0.45), p2[1] - head * math.sin(ang - 0.45))
    a2 = (p2[0] - head * math.cos(ang + 0.45), p2[1] - head * math.sin(ang + 0.45))
    pygame.gfxdraw.aapolygon(surf, [p2, a1, a2], color)
    pygame.gfxdraw.filled_polygon(surf, [p2, a1, a2], color)


_icon_cache = {}


def icon(surf, name, center, d, color):
    circle(surf, color, center, d / 2)
    key = (name, int(d * 0.52))
    if key not in _icon_cache:
        img = pygame.image.load(os.path.join(ICONS, name + ".png")).convert_alpha()
        _icon_cache[key] = pygame.transform.smoothscale(img, (key[1], key[1]))
    img = _icon_cache[key]
    surf.blit(img, img.get_rect(center=(int(center[0]), int(center[1]))))


# ============================================================ the pendulum
class Pendulum:
    """Exact Pendulum-v1 dynamics (gymnasium/envs/classic_control/pendulum.py),
    reimplemented so slides can set the state, kick it and run it live."""

    g, m, l, max_speed, max_torque = 10.0, 1.0, 1.0, 8.0, 2.0

    def __init__(self, th=math.pi, thdot=0.0):
        self.reset(th, thdot)

    def reset(self, th=math.pi, thdot=0.0):
        self.th, self.thdot, self.u = th, thdot, 0.0
        self.ret, self.steps, self.last_cost = 0.0, 0, 0.0

    def obs(self):
        return np.array([math.cos(self.th), math.sin(self.th), self.thdot], dtype=np.float32)

    def cost_terms(self, u=None):
        u = self.u if u is None else u
        a = ((self.th + math.pi) % (2 * math.pi)) - math.pi
        return a * a, 0.1 * self.thdot ** 2, 0.001 * u * u

    def step(self, u):
        u = float(np.clip(u, -self.max_torque, self.max_torque))
        self.u = u
        c = sum(self.cost_terms(u))
        newthdot = self.thdot + (3 * self.g / (2 * self.l) * math.sin(self.th) + 3.0 / (self.m * self.l ** 2) * u) * SIM_DT
        self.thdot = float(np.clip(newthdot, -self.max_speed, self.max_speed))
        self.th += self.thdot * SIM_DT
        self.ret -= c
        self.steps += 1
        self.last_cost = c
        return c

    def kick(self, dv):
        self.thdot = float(np.clip(self.thdot + dv, -self.max_speed, self.max_speed))

    def upright(self):
        a = ((self.th + math.pi) % (2 * math.pi)) - math.pi
        return abs(a) < 0.25 and abs(self.thdot) < 1.0


def tip_pts(cs, center, R):
    """Pendulum-tip screen points for an array of (cos, sin) rows, as plain floats."""
    cs = np.asarray(cs, dtype=float).reshape(-1, 2)
    return np.stack([center[0] + R * cs[:, 1], center[1] - R * cs[:, 0]], 1).tolist()


def draw_pendulum(surf, center, R, cos_sin, color, width=None, pivot=True, torque=0.0, alpha=None):
    """Draw the pole. cos_sin = (cos th, sin th); th = 0 is upright."""
    c, s = cos_sin
    cx, cy = center
    tip = (cx + R * float(s), cy - R * float(c))
    width = width or max(6, int(R * 0.11))
    if alpha is not None:
        # draw opaque on a transparent layer, then fade the whole layer
        tmp = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        thick_line(tmp, color, (cx, cy), tip, width)
        tmp.set_alpha(alpha)
        surf.blit(tmp, (0, 0))
    else:
        thick_line(surf, color, (cx, cy), tip, width)
    if pivot:
        circle(surf, (230, 234, 242), (cx, cy), max(4, width * 0.35))
    if abs(torque) > 0.05:
        # torque arc around the pivot, like Gymnasium's renderer
        r = R * 0.28
        span = math.radians(150) * abs(torque) / 2.0
        start = math.pi / 2
        a0, a1 = (start, start + span) if torque > 0 else (start - span, start)
        rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
        pygame.draw.arc(surf, MUTED, rect, a0, a1, 3)
    return tip


# ============================================================ the models
class Models:
    """Loads the seed-0 networks saved by main.py and builds every controller."""

    def __init__(self):
        torch.set_num_threads(1)                        # same as the reported timings
        self.summary = json.load(open(os.path.join(RESULTS, "summary.json")))
        self.cfg = get_config()
        c = self.summary["config"]
        for k in ("wm_hidden", "wm_layers", "use_symlog", "pi_hidden", "plan_horizon",
                  "cem_population", "cem_elites", "cem_iterations", "rs_samples"):
            setattr(self.cfg, k, c[k])
        self.wm = WorldModel(3, 1, self.cfg.wm_hidden, self.cfg.wm_layers, self.cfg.use_symlog)
        self.wm.load_state_dict(torch.load(os.path.join(RESULTS, "models", "world_model_seed0.pt")))
        self.wm.eval()
        self.pi = Policy(3, 1, self.cfg.pi_hidden)
        self.pi.load_state_dict(torch.load(os.path.join(RESULTS, "models", "policy_seed0.pt")))
        self.pi.eval()

    def controller(self, name, seed=0, horizon=None):
        cfg = self.cfg if horizon is None else replace(self.cfg, plan_horizon=horizon)
        if name == "Random":
            return RandomController(seed)
        if name == "Random-shooting MPC":
            return RandomShootingPlanner(self.wm, cfg, seed=seed)
        if name == "CEM MPC":
            return CEMPlanner(self.wm, cfg, seed=seed)
        return PolicyController(self.pi)


class Timed:
    """Wraps a controller and measures how long each act() call takes."""

    def __init__(self, ctrl):
        self.ctrl, self.times = ctrl, []

    def reset(self):
        self.ctrl.reset()

    def act(self, obs):
        t0 = time.perf_counter()
        a = self.ctrl.act(obs)
        self.times.append(time.perf_counter() - t0)
        self.times = self.times[-200:]
        return float(a[0])

    @property
    def ms(self):
        return 1000 * float(np.mean(self.times)) if self.times else 0.0


class VisualCEM(CEMPlanner):
    """CEMPlanner that also records what it imagined, so we can draw it.

    For every CEM iteration we keep the imagined states of all sampled action
    sequences, which ones were elites and their costs. The optimisation is the
    same as planner.CEMPlanner.act.
    """

    def __init__(self, model, cfg, seed=0):
        super().__init__(model, cfg, seed)
        self.history = []

    @torch.no_grad()
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mean = self.mean.clone()
        std = torch.full((self.H,), self.init_std)
        hist = []
        for _ in range(self.iters):
            noise = torch.randn(self.pop, self.H, generator=self.gen)
            samples = torch.clamp(mean + std * noise, A_LOW, A_HIGH)
            s = obs_t.expand(self.pop, -1)
            states = [s]
            total = torch.zeros(self.pop)
            for t in range(self.H):
                a = samples[:, t:t + 1]
                total += pendulum_cost(s, a)
                s = self.model.predict(s, a)            # imagine one step
                states.append(s)
            total += pendulum_cost(s, torch.zeros(self.pop, 1))
            elite_idx = torch.topk(total, self.n_elite, largest=False).indices
            elites = samples[elite_idx]
            mean = self.momentum * mean + (1 - self.momentum) * elites.mean(0)
            std = self.momentum * std + (1 - self.momentum) * elites.std(0)
            st = torch.stack(states, 1)[:, :, :2].numpy()       # (pop, H+1, [cos, sin])
            hist.append({"states": st, "elites": elite_idx.numpy(), "costs": total.numpy(),
                         "best": float(total[elite_idx[0]])})
        # plan of the final mean, imagined once more for drawing
        s = obs_t.unsqueeze(0)
        plan = [s]
        for t in range(self.H):
            s = self.model.predict(s, mean[t:t + 1].unsqueeze(0))
            plan.append(s)
        self.plan_states = torch.cat(plan)[:, :2].numpy()
        self.history = hist
        action = mean[0].item()
        self.mean = torch.cat([mean[1:], torch.zeros(1)])
        return np.array([action], dtype=np.float32)


# ============================================================ slide base
class Slide:
    kicker, title, hints = "", "", ""
    kicker_color = AMBER

    def __init__(self, app):
        self.app = app
        self.t = 0.0

    def enter(self):
        self.t = 0.0

    def update(self, dt):
        self.t += dt

    def draw(self, s):
        pass

    def key(self, k):
        return False

    def click(self, pos, button):
        return False

    # helpers -------------------------------------------------------------
    def appear(self, delay, dur=0.6):
        """0 -> 1 entrance progress for an element shown `delay` s after entry."""
        return ease((self.t - delay) / dur)


class SimMixin:
    """Fixed-rate (20 Hz) stepping of one or more pendulums."""

    def sim_reset(self):
        self.acc = 0.0

    def sim_ticks(self, dt, rate=1.0, max_steps=2):
        self.acc += dt * rate
        n = 0
        while self.acc >= SIM_DT and n < max_steps:
            self.acc -= SIM_DT
            n += 1
        if n == max_steps:
            self.acc = min(self.acc, SIM_DT)
        return n


# ============================================================ 1. title
class TitleSlide(Slide, SimMixin):
    hints = "Click or K: kick the pendulum   ·   R: restart from hanging down"

    def enter(self):
        super().enter()
        self.p = Pendulum()
        self.ctrl = Timed(self.app.models.controller("Distilled policy"))
        self.balanced = 0.0
        self.trail = []
        self.sim_reset()

    def update(self, dt):
        super().update(dt)
        for _ in range(self.sim_ticks(dt)):
            self.p.step(self.ctrl.act(self.p.obs()))
            self.trail.append(self.p.obs()[:2].copy())
            self.trail = self.trail[-40:]
        self.balanced = self.balanced + dt if self.p.upright() else 0.0
        if self.balanced > 5.0:
            self.p.reset(math.pi + np.random.uniform(-0.3, 0.3), 0.0)
            self.balanced = 0.0

    def key(self, k):
        if k == pygame.K_k:
            self.p.kick(np.random.choice([-5.0, 5.0]))
            return True
        if k == pygame.K_r:
            self.p.reset()
            return True
        return False

    def click(self, pos, button):
        self.p.kick(5.0 if pos[0] < 1010 else -5.0)
        return True

    def draw(self, s):
        S = self.app.models.summary
        a = self.appear(0.0)
        text(s, "ENG 6010 · ARTIFICIAL INTELLIGENCE IN ROBOTICS · READINGS 6", (80, 150 - 20 * (1 - a)), 18,
             AMBER, True, alpha=int(255 * a))
        a = self.appear(0.15)
        text(s, "From Mode-2 to Mode-1", (76, 196 - 20 * (1 - a)), 58, TEXT, True, alpha=int(255 * a))
        a = self.appear(0.3)
        para(s, "Planning with a learned world model, then distilling it into a fast policy",
             80, 290, 600, 30, MUTED, alpha=int(255 * a))
        a = self.appear(0.5)
        hd = S["headline"]
        chips = [(f"{hd['speedup_policy_vs_cem']:.0f}× faster", TEAL), (f"{100 * hd['policy_normalized_score']:.0f}% of the skill", TEAL),
                 ("15-step imagination", AMBER)]
        x = 80
        for lab, col in chips:
            w = FONTS.get(20, True).size(lab)[0] + 40
            rrect(s, CARD, (x, 420 + 20 * (1 - a), w, 46), 23)
            text(s, lab, (x + w / 2, 443 + 20 * (1 - a)), 20, col, True, anchor="center", alpha=int(255 * a))
            x += w + 14
        text(s, "This pendulum is being balanced live by the distilled policy (Mode-1).", (80, 520), 18, MUTED)
        text(s, f"Decision time right now: {self.ctrl.ms:.3f} ms", (80, 548), 18, TEAL, True)
        # live pendulum
        cx, cy, R = 1010, 360, 180
        circle(s, CARD, (cx, cy), R + 45)
        for i, cs in enumerate(self.trail):
            al = int(120 * (i + 1) / len(self.trail))
            tmp = (cx + R * cs[1], cy - R * cs[0])
            pygame.gfxdraw.filled_circle(s, int(tmp[0]), int(tmp[1]), 3, (*TEAL, al))
        draw_pendulum(s, (cx, cy), R, self.p.obs()[:2], TEAL, 18, torque=self.p.u)


# ============================================================ 2. readings
class ReadingsSlide(Slide):
    kicker, title = "Background", "Four readings, one idea: imagine in training, act fast"
    hints = "Move the mouse over a card to see where the idea lives in my code"
    CARDS = [
        ("brain", AMBER, "LeCun (2022)", "A Path Towards Autonomous Machine Intelligence",
         ["Mode-2: plan with a world model + fixed intrinsic cost, act, replan",
          "Fig. 5: compile Mode-2 into a fast Mode-1 policy"], "planner.py · cost.py · policy.py"),
        ("layers", VIOLET, "Ding et al. (2024)", "Diffusion World Model",
         ["One-step models compound error over long rollouts (Fig. 1.1)",
          "Imagination only in training, so the deployed policy is fast"], "multi_step_error() · evaluate.py"),
        ("moon", TEAL, "Hafner et al. (2023)", "DreamerV3",
         ["Learn a world model from experience; behave in short imagined rollouts",
          "symlog makes regression targets robust"], "world_model.py · --symlog"),
        ("github", GRAY, "danijar/dreamerv3", "Official code",
         ["imag_length: 15 became my planning horizon",
          "symlog/symexp in nets.py; configs.yaml became config.py"], "config.py · utils.py"),
    ]

    def draw(self, s):
        mx, my = pygame.mouse.get_pos()
        for i, (ic, col, who, what, pts, code) in enumerate(self.CARDS):
            a = self.appear(0.1 + 0.12 * i)
            x, y, w, h = 60 + i * 296, 160 + 40 * (1 - a), 276, 470
            hover = x <= mx <= x + w and y <= my <= y + h
            if hover:
                y -= 8
            rrect(s, CARD2 if hover else CARD, (x, y, w, h), 18)
            if hover:
                pygame.draw.rect(s, col, (x, y, w, h), 2, border_radius=18)
            icon(s, ic, (x + 50, y + 55), 60, col)
            text(s, who, (x + 24, y + 100), 24, TEXT, True)
            yy = para(s, what, x + 24, y + 134, w - 44, 17, MUTED)
            yy += 10
            for p in pts:
                circle(s, col, (x + 30, yy + 11), 3)
                yy = para(s, p, x + 42, yy, w - 64, 18, TEXT) + 8
            rrect(s, BG_DEEP, (x + 16, y + h - 74, w - 32, 56), 10)
            text(s, "In my code", (x + 28, y + h - 68), 14, col if hover else FAINT, True)
            text(s, code, (x + 28, y + h - 46), 15, TEXT if hover else MUTED)


# ============================================================ 3. try it
class TryItSlide(Slide, SimMixin):
    kicker, title = "The robot · Pendulum-v1", "You are the controller: can you swing it up?"
    hints = "Hold A / D: torque −2 / +2   ·   T: let the trained policy drive   ·   R: reset"

    def enter(self):
        super().enter()
        self.p = Pendulum()
        self.policy = Timed(self.app.models.controller("Distilled policy"))
        self.auto = False
        self.best = None
        self.last = None
        self.sim_reset()

    def key(self, k):
        if k == pygame.K_t:
            self.auto = not self.auto
            self.p.reset()
            return True
        if k == pygame.K_r:
            self.p.reset()
            return True
        return False

    def update(self, dt):
        super().update(dt)
        keys = pygame.key.get_pressed()
        for _ in range(self.sim_ticks(dt)):
            if self.auto:
                u = self.policy.act(self.p.obs())
            else:
                u = (2.0 if keys[pygame.K_d] else 0.0) - (2.0 if keys[pygame.K_a] else 0.0)
            self.p.step(u)
            if self.p.steps >= 200:
                self.last = (self.p.ret, "policy" if self.auto else "you")
                if not self.auto:
                    self.best = self.p.ret if self.best is None else max(self.best, self.p.ret)
                self.p.reset()

    def draw(self, s):
        cx, cy, R = 330, 380, 170
        rrect(s, CARD, (60, 150, 540, 500), 18)
        draw_pendulum(s, (cx, cy), R, self.p.obs()[:2], TEAL if self.auto else AMBER, 18, torque=self.p.u)
        who = "Distilled policy is driving" if self.auto else "You are driving (A / D)"
        text(s, who, (cx, 182), 20, TEAL if self.auto else AMBER, True, anchor="center")
        # episode timer
        pygame.draw.rect(s, GRID, (100, 615, 460, 8), border_radius=4)
        pygame.draw.rect(s, MUTED, (100, 615, int(460 * self.p.steps / 200), 8), border_radius=4)
        text(s, f"step {self.p.steps}/200", (330, 590), 14, MUTED, anchor="center")

        x0 = 650
        o = self.p.obs()
        rows = [("State s", f"[cos θ, sin θ, θ̇] = [{o[0]:+.2f}, {o[1]:+.2f}, {o[2]:+.2f}]"),
                ("Action u", f"torque {self.p.u:+.1f} N·m  (limit ±2: too weak to lift it straight up)")]
        y = 160
        for t, d in rows:
            text(s, t, (x0, y), 22, TEXT, True)
            text(s, d, (x0, y + 30), 18, MUTED)
            y += 76
        # live cost bar - LeCun's intrinsic cost
        rrect(s, CARD, (x0 - 10, y, 580, 200), 16)
        text(s, "Intrinsic cost (LeCun): fixed, not learned", (x0 + 10, y + 14), 20, TEXT, True)
        text(s, "θ² + 0.1·θ̇² + 0.001·u²", (x0 + 10, y + 46), 22, AMBER, True)
        terms = self.p.cost_terms()
        tot = sum(terms)
        bx, bw = x0 + 10, 540
        scale = bw / 16.3
        cols = [AMBER, VIOLET, TEAL]
        xx = bx
        pygame.draw.rect(s, GRID, (bx, y + 92, bw, 26), border_radius=6)
        for v, c in zip(terms, cols):
            w = v * scale
            pygame.draw.rect(s, c, (xx, y + 92, max(0, w), 26))
            xx += w
        text(s, f"cost now {tot:5.2f}   (max 16.3)", (bx, y + 128), 18, MUTED)
        for i, (lab, c) in enumerate([("angle θ²", AMBER), ("speed 0.1·θ̇²", VIOLET), ("effort", TEAL)]):
            circle(s, c, (bx + 8 + i * 170, y + 172), 6)
            text(s, lab, (bx + 20 + i * 170, y + 162), 16, MUTED)
        y += 220
        text(s, f"Return this episode: {fmt_num(self.p.ret)}", (x0, y), 24, TEXT, True)
        y += 36
        if self.last:
            text(s, f"Last episode ({self.last[1]}): {fmt_num(self.last[0])}", (x0, y), 18, MUTED)
            y += 26
        if self.best is not None:
            text(s, f"Your best: {fmt_num(self.best)}", (x0, y), 18, AMBER, True)
            y += 26
        pol = self.app.models.summary["controllers"]["Distilled policy"]["return_mean"]
        text(s, f"Test means over random starts: random {fmt_num(self.app.models.summary['controllers']['Random']['return_mean'])},"
                f" distilled policy {fmt_num(pol)}. Hanging down is the hardest start.", (x0, y), 16, FAINT)


# ============================================================ 4. two modes
class ModesSlide(Slide):
    kicker, title = "LeCun's architecture → my code", "Mode-2 thinks, Mode-1 reacts"
    hints = "1 / 2: focus on Mode-2 / Mode-1   ·   0: show both"

    def enter(self):
        super().enter()
        self.focus = 0
        self.pulses2 = []
        self.pulses1 = []
        self.spawn2 = self.spawn1 = 0.0
        S = self.app.models.summary
        self.cem_ms = S["controllers"]["CEM MPC"]["decision_time_ms_mean"]
        self.pol_ms = S["controllers"]["Distilled policy"]["decision_time_ms_mean"]
        c = S["config"]
        self.imag = c["cem_iterations"] * c["cem_population"] * (c["plan_horizon"] + 1)

    def key(self, k):
        for kk, f in ((pygame.K_0, 0), (pygame.K_1, 2), (pygame.K_2, 1)):
            if k == kk:
                self.focus = f
                return True
        return False

    def update(self, dt):
        super().update(dt)
        # Mode-2: a burst of "imagination" pulses between planner and model/cost
        self.spawn2 += dt
        while self.spawn2 > 0.025:
            self.spawn2 -= 0.025
            self.pulses2.append([np.random.choice(["wm", "cost"]), 0.0])
        for p in self.pulses2:
            p[1] += dt * 1.6
        self.pulses2 = [p for p in self.pulses2 if p[1] < 1]
        self.spawn1 += dt
        while self.spawn1 > 0.12:
            self.spawn1 -= 0.12
            self.pulses1.append(0.0)
        self.pulses1 = [p + dt * 0.8 for p in self.pulses1 if p < 1]

    def box(self, s, x, y, w, h, title, sub, col, alpha):
        rrect(s, CARD2, (x, y, w, h), 14)
        text(s, title, (x + w / 2, y + 22), 20, col, True, anchor="center", alpha=alpha)
        text(s, sub, (x + w / 2, y + 50), 15, MUTED, mono=True, anchor="center", alpha=alpha)

    def draw(self, s):
        dim2 = 255 if self.focus in (0, 2) else 70
        dim1 = 255 if self.focus in (0, 1) else 70
        # lanes
        rrect(s, CARD, (60, 150, 1160, 250), 18, alpha=dim2)
        rrect(s, CARD, (60, 440, 1160, 170), 18, alpha=dim1)
        text(s, "MODE-2 · DELIBERATE", (84, 166), 16, AMBER, True, alpha=dim2)
        text(s, f"{self.cem_ms:.1f} ms per decision · {self.imag:,} imagined steps each", (1196, 166), 16, MUTED, anchor="topright", alpha=dim2)
        text(s, "MODE-1 · REACTIVE", (84, 456), 16, TEAL, True, alpha=dim1)
        text(s, f"{self.pol_ms:.3f} ms per decision · one forward pass", (1196, 456), 16, MUTED, anchor="topright", alpha=dim1)
        # Mode-2 boxes
        self.box(s, 90, 240, 200, 76, "Observation s", "[cos, sin, dθ/dt]", TEXT, dim2)
        self.box(s, 360, 240, 230, 76, "CEM planner", "planner.py", AMBER, dim2)
        self.box(s, 680, 190, 230, 70, "World model", "world_model.py", TEXT, dim2)
        self.box(s, 680, 305, 230, 70, "Intrinsic cost", "cost.py", TEXT, dim2)
        self.box(s, 990, 240, 200, 76, "Execute a₀", "then replan", AMBER, dim2)
        col2 = AMBER if dim2 == 255 else FAINT
        arrow(s, col2, (290, 278), (360, 278))
        arrow(s, col2, (590, 278), (990, 278), 3)
        # imagination pulses
        for kind, f in self.pulses2:
            y1 = 225 if kind == "wm" else 340
            # out and back: planner -> box -> planner
            g = f * 2 if f < 0.5 else 2 - f * 2
            x = lerp(590, 680, g)
            y = lerp(278, y1, g)
            if dim2 == 255:
                circle(s, AMBER if kind == "wm" else LAV, (x, y), 3)
        # distil arrow
        arrow(s, TEAL, (475, 316), (475, 500), 3)
        text(s, "distil: BC + DAgger on planner labels", (492, 404), 17, TEAL, True)
        # Mode-1 boxes
        self.box(s, 90, 500, 200, 76, "Observation s", "[cos, sin, dθ/dt]", TEXT, dim1)
        self.box(s, 360, 500, 230, 76, "Policy π(s)", "policy.py", TEAL, dim1)
        self.box(s, 990, 500, 200, 76, "Execute a", "every step", TEAL, dim1)
        col1 = TEAL if dim1 == 255 else FAINT
        arrow(s, col1, (290, 538), (360, 538))
        arrow(s, col1, (590, 538), (990, 538), 3)
        if dim1 == 255:
            for f in self.pulses1:
                circle(s, TEAL, (lerp(590, 990, f), 538), 5)
        # time-budget bar: 50 ms control period at 20 Hz
        y = 636
        text(s, "50 ms control period (20 Hz):", (60, y), 16, MUTED)
        bx, bw = 300, 600
        pygame.draw.rect(s, GRID, (bx, y + 2, bw, 16), border_radius=8)
        sweep = ease(self.t / 2.0)
        pygame.draw.rect(s, AMBER, (bx, y + 2, int(bw * self.cem_ms / 50 * sweep), 16), border_radius=8)
        pygame.draw.rect(s, TEAL, (bx, y + 2, max(3, int(bw * self.pol_ms / 50 * sweep)), 16), border_radius=8)
        text(s, f"CEM uses {100 * self.cem_ms / 50:.0f}%  ·  policy {100 * self.pol_ms / 50:.1f}%", (bx + bw + 14, y), 16, TEXT, True)


# ============================================================ 5. pipeline
class PipelineSlide(Slide):
    kicker, title = "How the code works", "One command, six stages: python main.py"
    hints = "Click a stage to pin it (click again to release)"
    STAGES = [
        ("database", "Collect data", "collect_data.py", "50 random episodes → 10,000 (s, a, s′, r) transitions, saved as .npz"),
        ("network", "Train world model", "world_model.py", "3×128 MLP predicts Δs = s′ − s from normalised (s, a)"),
        ("chart", "Compounding error", "multi_step_error()", "Open-loop rollouts for 1-30 steps on held-out trajectories"),
        ("route", "Mode-2: CEM MPC", "planner.py", "300 plans × 4 iterations, 15 steps each; run first action, replan"),
        ("bolt", "Mode-1: distil", "policy.py", "Behaviour cloning, then 4 DAgger iterations (600 → 3,000 labels)"),
        ("balance", "Evaluate + plot", "evaluate.py · plots.py", "4 controllers × 15 unseen episodes: return and ms per decision"),
    ]

    def enter(self):
        super().enter()
        self.pinned = None

    def rects(self):
        return [(60 + (i % 3) * 395, 160 + (i // 3) * 230, 370, 205) for i in range(6)]

    def click(self, pos, button):
        for i, (x, y, w, h) in enumerate(self.rects()):
            if x <= pos[0] <= x + w and y <= pos[1] <= y + h:
                self.pinned = None if self.pinned == i else i
                return True
        return False

    def draw(self, s):
        active = self.pinned if self.pinned is not None else int(self.t / 1.6) % 6
        rs = self.rects()
        # travelling dot along the stage order
        f = (self.t / 1.6) % 1
        for i, ((ic, t, f_, d), (x, y, w, h)) in enumerate(zip(self.STAGES, rs)):
            a = self.appear(0.08 * i)
            on = i == active
            col = AMBER if i < 4 else TEAL
            rrect(s, CARD2 if on else CARD, (x, y + 30 * (1 - a), w, h), 18)
            if on:
                pygame.draw.rect(s, col, (x, y, w, h), 2, border_radius=18)
            icon(s, ic, (x + 46, y + 46), 52, col if on or self.pinned is None else FAINT)
            text(s, f"{i + 1}. {t}", (x + 86, y + 30), 23, TEXT, True)
            text(s, f_, (x + 24, y + 86), 16, col, mono=True)
            para(s, d, x + 24, y + 116, w - 48, 17, TEXT if on else MUTED)
        if self.pinned is None:
            i = active
            x, y, w, h = rs[i]
            nx, ny, _, _ = rs[(i + 1) % 6]
            p0 = (x + w - 10, y + h / 2)
            p1 = (nx + 10, ny + h / 2)
            circle(s, AMBER if i < 4 else TEAL, (lerp(p0[0], p1[0], f), lerp(p0[1], p1[1], f)), 6)
        c = self.app.models.summary
        text(s, f"3 seeds · fixed random seeds · {c['runtime_seconds'] / 60:.1f} min on one CPU thread · "
                "every setting in config.py", (60, 630), 16, MUTED)


# ============================================================ 6. compounding error
class CompoundingSlide(Slide):
    kicker, title = "Diffusion World Model, Fig. 1.1", "Small one-step errors compound over the horizon"
    kicker_color = VIOLET
    hints = "N: new trajectory   ·   S: re-sync from real state every H steps (MPC)   ·   Up / Down: change H"

    def enter(self):
        super().enter()
        self.resync = False
        self.H = 15
        self.new_traj()

    def new_traj(self):
        rng = np.random.default_rng(int(time.time() * 1000) % 2 ** 32)
        self.real = Pendulum(rng.uniform(-math.pi, math.pi), rng.uniform(-1, 1))
        self.ghost = self.real.obs().copy()
        self.actions = rng.uniform(-2, 2, size=80)
        self.k = 0
        self.errs = []
        self.trail_r, self.trail_g = [], []
        self.acc = 0.0
        self.pause = 0.0

    def key(self, k):
        if k == pygame.K_n:
            self.new_traj()
        elif k == pygame.K_s:
            self.resync = not self.resync
            self.new_traj()
        elif k == pygame.K_UP:
            self.H = min(30, self.H + 1)
        elif k == pygame.K_DOWN:
            self.H = max(1, self.H - 1)
        else:
            return False
        return True

    @torch.no_grad()
    def update(self, dt):
        super().update(dt)
        if self.k >= len(self.actions):
            self.pause += dt
            if self.pause > 2.0:
                self.new_traj()
            return
        self.acc += dt
        while self.acc > 0.08 and self.k < len(self.actions):     # slow motion: 12.5 steps/s
            self.acc -= 0.08
            a = self.actions[self.k]
            g = torch.as_tensor(self.ghost).unsqueeze(0)
            self.ghost = self.app.models.wm.predict(g, torch.tensor([[a]], dtype=torch.float32))[0].numpy()
            self.real.step(a)
            self.k += 1
            if self.resync and self.k % self.H == 0:
                self.ghost = self.real.obs().copy()          # MPC: replan from a real observation
            r = self.real.obs()
            d = math.atan2(self.ghost[1], self.ghost[0]) - math.atan2(r[1], r[0])
            d = abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))
            self.errs.append(d)
            self.trail_r.append(r[:2].copy())
            self.trail_g.append(self.ghost[:2].copy())

    def draw(self, s):
        cx, cy, R = 300, 400, 185
        rrect(s, CARD, (60, 150, 480, 490), 18)
        for tr, col in ((self.trail_r, TEXT), (self.trail_g, AMBER)):
            pts = tip_pts(tr[-30:], (cx, cy), R) if tr else []
            if len(pts) > 1:
                pygame.draw.aalines(s, col, False, pts)
        draw_pendulum(s, (cx, cy), R, self.real.obs()[:2], (225, 230, 240), 16)
        draw_pendulum(s, (cx, cy), R, self.ghost[:2], AMBER, 16, pivot=True, alpha=170)
        circle(s, TEXT, (90, 178), 6)
        text(s, "real physics", (102, 168), 16, TEXT)
        circle(s, AMBER, (220, 178), 6)
        text(s, "world model, open loop", (232, 168), 16, AMBER)
        text(s, f"step {self.k}   ·   same random torques for both", (cx, 610), 15, MUTED, anchor="center")

        # live error chart (angle error in degrees)
        x0, y0, w, h = 590, 170, 630, 330
        rrect(s, CARD, (x0 - 20, y0 - 20, w + 40, h + 80), 18)
        text(s, "Angle error of the imagined pendulum (degrees)", (x0, y0 - 6), 18, TEXT, True)
        peak = max(self.errs) * 1.1 if self.errs else 0.0
        top = next(v for v in (10, 20, 45, 90, 180, 1e9) if v >= peak)
        top = min(top, 180)
        gy0 = y0 + 30
        gh = h - 40
        for v in np.linspace(0, top, 5):
            yy = gy0 + gh - gh * v / top
            pygame.draw.line(s, GRID, (x0 + 40, yy), (x0 + w, yy))
            text(s, f"{v:g}", (x0 + 32, yy), 13, MUTED, anchor="midright")
        n = len(self.actions)
        if self.resync:
            for k in range(self.H, n, self.H):
                xx = x0 + 40 + (w - 40) * k / n
                pygame.draw.line(s, (60, 80, 70), (xx, gy0), (xx, gy0 + gh), 1)
        pts = [(x0 + 40 + (w - 40) * (i + 1) / n, gy0 + gh - gh * min(e, top) / top) for i, e in enumerate(self.errs)]
        if len(pts) > 1:
            pygame.draw.lines(s, AMBER, False, pts, 3)
        text(s, "step", (x0 + w, gy0 + gh + 8), 13, MUTED, anchor="topright")
        S = self.app.models.summary["world_model"]["seed0_prediction_error"]["model 0 (random data)"]
        mode = (f"MPC mode: re-sync to the real state every H = {self.H} steps" if self.resync
                else "Open loop: the model only ever sees its own predictions")
        text(s, mode, (x0 - 20, 545), 18, TEAL if self.resync else AMBER, True)
        para(s, f"Measured on held-out data: {S['angle_err_deg'][0]:.2f}° after 1 step, "
                f"{S['angle_err_deg'][14]:.2f}° after 15, {S['angle_err_deg'][29]:.1f}° after 30. "
                "That is why MPC (and DreamerV3) imagine only ~15 steps and then start again from reality.",
             x0 - 20, 575, 650, 16, MUTED)


# ============================================================ 7. imagination
class ImaginationSlide(Slide, SimMixin):
    kicker, title = "Mode-2 = Model-Predictive Control", "Watching the CEM planner imagine"
    hints = "S: slow-motion CEM   ·   H: horizon 5 / 15 / 30   ·   K / click: kick   ·   R: reset"
    HORIZONS = [15, 30, 5]

    def enter(self):
        super().enter()
        self.h_i = 0
        self.slow = False
        self.make()

    def make(self):
        m = self.app.models
        cfg = replace(m.cfg, plan_horizon=self.HORIZONS[self.h_i])
        self.cem = VisualCEM(m.wm, cfg, seed=0)
        self.p = Pendulum()
        self.ms = []
        self.phase = 0.0            # slow-motion phase through iterations
        self.pending = None
        self.sim_reset()

    def key(self, k):
        if k == pygame.K_s:
            self.slow = not self.slow
        elif k == pygame.K_h:
            self.h_i = (self.h_i + 1) % len(self.HORIZONS)
            self.make()
        elif k == pygame.K_k:
            self.p.kick(np.random.choice([-5.0, 5.0]))
        elif k == pygame.K_r:
            self.make()
        else:
            return False
        return True

    def click(self, pos, button):
        self.p.kick(5.0 if pos[0] < 240 else -5.0)
        return True

    def decide(self):
        t0 = time.perf_counter()
        a = float(self.cem.act(self.p.obs())[0])
        self.ms.append(1000 * (time.perf_counter() - t0))
        self.ms = self.ms[-50:]
        return a

    def update(self, dt):
        super().update(dt)
        if self.p.steps >= 300:
            self.make()
        if self.slow:
            # one decision every (iterations × 0.7 s): show each CEM iteration
            if self.pending is None:
                self.pending = self.decide()
                self.phase = 0.0
            self.phase += dt
            if self.phase > 0.7 * self.cem.iters + 0.4:
                self.p.step(self.pending)
                self.pending = None
        else:
            self.pending = None
            for _ in range(self.sim_ticks(dt, max_steps=1)):
                self.p.step(self.decide())

    def time_plot(self, s, fan, elites, plan, H_, scale_states):
        """Imagined angle change over the horizon for every sampled plan.

        The circle view hides how different the plans are (they overlap near
        the tip); unrolling them in time shows the spread and how CEM narrows it.
        """
        x0, x1, y0, y1 = 425, 680, 200, 585
        def dtheta(st):
            th = np.unwrap(np.arctan2(st[..., 1], st[..., 0]), axis=-1)
            return np.degrees(th - th[..., :1])
        f = dtheta(fan)
        ref = dtheta(scale_states)            # fixed axis for the whole decision
        lo, hi = min(ref.min(), -5), max(ref.max(), 5)
        pad = 0.08 * (hi - lo)
        lo, hi = float(lo - pad), float(hi + pad)
        def to_xy(row):
            xs = x0 + (x1 - x0) * np.arange(len(row)) / (len(row) - 1)
            ys = y1 - (y1 - y0) * (np.clip(row, lo, hi) - lo) / (hi - lo)
            return np.stack([xs, ys], 1).tolist()
        yz = y1 - (y1 - y0) * (0 - lo) / (hi - lo)
        pygame.draw.line(s, GRID, (x0, yz), (x1, yz))
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        for row in f:
            pygame.draw.lines(ov, (*LAV, 60), False, to_xy(row), 1)
        for row in dtheta(elites):
            pygame.draw.lines(ov, (*AMBER, 170), False, to_xy(row), 2)
        s.blit(ov, (0, 0))
        if plan is not None:
            pygame.draw.lines(s, AMBER, False, to_xy(dtheta(plan[None])[0]), 4)
        text(s, "imagined angle change Δθ (°)", (x0, y0 - 28), 14, MUTED)
        text(s, f"{hi:+.0f}°", (x0 - 6, y0), 12, FAINT, anchor="topright")
        text(s, f"{lo:+.0f}°", (x0 - 6, y1), 12, FAINT, anchor="bottomright")
        text(s, "now", (x0, y1 + 6), 12, FAINT)
        text(s, f"+{H_} steps", (x1, y1 + 6), 12, FAINT, anchor="topright")

    def draw(self, s):
        cx, cy, R = 240, 400, 150
        rrect(s, CARD, (60, 150, 640, 500), 18)
        pygame.draw.line(s, GRID, (395, 175), (395, 600))
        hist = self.cem.history
        if hist:
            slow_now = self.slow and self.pending is not None
            it = min(int(self.phase / 0.7), len(hist) - 1) if slow_now else len(hist) - 1
            # faint fan: the samples of the current iteration in slow motion,
            # otherwise the first (widest) iteration so the search is visible
            fan = hist[it] if slow_now else hist[0]
            el_h = hist[it]
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            el = set(el_h["elites"].tolist())
            for i in range(fan["states"].shape[0]):
                if slow_now and i in el:
                    continue
                pygame.draw.lines(ov, (*LAV, 70), False, tip_pts(fan["states"][i], (cx, cy), R), 1)
            for i in el:
                pygame.draw.lines(ov, (*AMBER, 170), False, tip_pts(el_h["states"][i], (cx, cy), R), 2)
            s.blit(ov, (0, 0))
            show_plan = not (slow_now and it < len(hist) - 1)
            if show_plan:
                plan = self.cem.plan_states
                pts = tip_pts(plan, (cx, cy), R)
                pygame.draw.lines(s, AMBER, False, pts, 4)
                draw_pendulum(s, (cx, cy), R, plan[-1], AMBER, 12, pivot=False, alpha=90)
            self.time_plot(s, fan["states"], el_h["states"][el_h["elites"]],
                           self.cem.plan_states if show_plan else None, self.HORIZONS[self.h_i], hist[0]["states"])
            if slow_now:
                text(s, f"Slow motion: CEM iteration {it + 1} of {len(hist)}", (84, 168), 18, AMBER, True)
        draw_pendulum(s, (cx, cy), R, self.p.obs()[:2], (230, 234, 242), 16, torque=self.p.u)
        text(s, "lilac: sampled plans   ·   amber: 30 elites   ·   bold: chosen plan", (380, 625), 15, MUTED, anchor="center")

        # right panel
        x0 = 740
        m = self.app.models
        Hh = self.HORIZONS[self.h_i]
        imag = self.cem.iters * self.cem.pop * (Hh + 1)
        rrect(s, CARD, (x0, 150, 480, 220), 18)
        text(s, f"Horizon H = {Hh} steps ({Hh * SIM_DT:.2f} s ahead)", (x0 + 24, 168), 22,
             AMBER if Hh == 15 else CORAL, True)
        # Measured once from hanging down (200 steps, seed 0):
        # H=5 -> -1130, never upright; H=15 -> -386; H=30 -> -360 at about 2x the time.
        note = {15: "15 = DreamerV3's imag_length. Test from hanging down: −386",
                5: "Too short: from hanging down it never got up (−1130)",
                30: "Test from hanging down: −360 vs −386, at ~2× the compute"}[Hh]
        text(s, note, (x0 + 24, 202), 16, MUTED)
        text(s, f"{imag:,}", (x0 + 24, 236), 40, AMBER, True)
        text(s, "imagined model steps per decision", (x0 + 24, 284), 16, MUTED)
        ms = float(np.mean(self.ms)) if self.ms else 0.0
        text(s, f"{ms:.1f} ms", (x0 + 280, 236), 40, TEXT, True)
        text(s, "per decision, measured live", (x0 + 280, 284), 16, MUTED)
        text(s, f"Return so far: {fmt_num(self.p.ret)}  (step {self.p.steps})", (x0 + 24, 322), 18, TEXT)

        # CEM iterations panel
        rrect(s, CARD, (x0, 390, 480, 260), 18)
        text(s, "One decision = 4 CEM iterations", (x0 + 24, 406), 20, TEXT, True)
        if hist:
            costs = [h_["best"] for h_ in hist]
            mx_ = max(costs) or 1.0
            cur = int(self.phase / 0.7) if self.slow and self.pending is not None else len(hist) - 1
            for i, c in enumerate(costs):
                bx = x0 + 34 + i * 108
                on = i <= cur
                hgt = max(3, 80 * c / mx_)
                pygame.draw.rect(s, AMBER if on else GRID, (bx, 590 - hgt, 70, hgt), border_radius=6)
                text(s, f"it {i + 1}", (bx + 35, 596), 14, MUTED, anchor="midtop")
                text(s, f"{c:.2f}", (bx + 35, 584 - hgt), 14, TEXT if on else FAINT, anchor="midbottom")
            text(s, "imagined cost of the best plan after each iteration (lower is better)", (x0 + 24, 620), 13, FAINT)
        steps = ["Sample 300 plans", "Imagine in the world model", "Keep 30 cheapest, refit", "Execute first action"]
        text(s, "  →  ".join(steps[:2]), (x0 + 24, 440), 14, MUTED)
        text(s, "  →  ".join(steps[2:]) + ", replan", (x0 + 24, 460), 14, MUTED)


# ============================================================ 8. DAgger
class DaggerSlide(Slide):
    kicker, title = "Mode-1 · LeCun Fig. 5", "DAgger turns the planner into a reflex"
    kicker_color = TEAL
    hints = "D: reveal the next iteration   ·   A: reveal all   ·   R: start over"

    def enter(self):
        super().enter()
        self.shown = 1

    def key(self, k):
        n = len(self.app.models.summary["dagger"]["0"]["iteration"])
        if k == pygame.K_d:
            self.shown = min(n, self.shown + 1)
        elif k == pygame.K_a:
            self.shown = n
        elif k == pygame.K_r:
            self.shown = 1
        else:
            return False
        self.t_reveal = self.t
        return True

    def draw(self, s):
        S = self.app.models.summary
        # loop diagram
        cx, cy, r = 300, 380, 150
        rrect(s, CARD, (60, 150, 480, 490), 18)
        nodes = [("play", "Policy drives", "its own mistakes happen"),
                 ("route", "Planner labels", "every state it visited"),
                 ("database", "Aggregate + retrain", "dataset grows each round")]
        ang0 = -math.pi / 2
        pos = [(cx + r * math.cos(ang0 + i * 2 * math.pi / 3), cy + r * math.sin(ang0 + i * 2 * math.pi / 3)) for i in range(3)]
        pygame.draw.circle(s, GRID, (cx, cy), r, 2)
        f = (self.t * 0.35) % 1
        a = ang0 + f * 2 * math.pi
        circle(s, TEAL, (cx + r * math.cos(a), cy + r * math.sin(a)), 7)
        for (ic, t1, t2), (x, y) in zip(nodes, pos):
            icon(s, ic, (x, y), 58, TEAL)
            text(s, t1, (x, y + 36), 17, TEXT, True, anchor="midtop")
            text(s, t2, (x, y + 58), 14, MUTED, anchor="midtop")
        text(s, "DAgger", (cx, cy), 26, TEAL, True, anchor="center")
        bc = [S["dagger"][k]["return_mean"][0] for k in S["dagger"]]
        para(s, f"Iteration 0 is plain behaviour cloning. It scored about {fmt_num(round(min(bc), -1))} on 2 of 3 seeds, "
                "worse than random: the clone never saw how to recover from its own mistakes.",
             84, 548, 440, 15, MUTED)

        # progressive chart
        x0, y0, w, h = 600, 205, 600, 360
        rrect(s, CARD, (580, 150, 640, 490), 18)
        lo, hi = -1800, 0
        for v in range(lo, hi + 1, 300):
            yy = y0 + h - h * (v - lo) / (hi - lo)
            pygame.draw.line(s, GRID, (x0 + 50, yy), (x0 + w - 10, yy))
            text(s, fmt_num(v), (x0 + 44, yy), 13, MUTED, anchor="midright")
        its = S["dagger"]["0"]["iteration"]
        xs = [x0 + 80 + (w - 130) * i / (len(its) - 1) for i in its]
        for i, xx in enumerate(xs):
            text(s, "BC" if i == 0 else f"DAgger {i}", (xx, y0 + h + 10), 14, MUTED if i < self.shown else FAINT, anchor="midtop")
        cem = S["controllers"]["CEM MPC"]["return_mean"]
        rnd = S["controllers"]["Random"]["return_mean"]
        for v, col, lab in ((cem, AMBER, "CEM teacher"), (rnd, GRAY, "random")):
            yy = y0 + h - h * (v - lo) / (hi - lo)
            for xx in range(x0 + 50, x0 + w - 10, 14):
                pygame.draw.line(s, col, (xx, yy), (xx + 7, yy), 2)
            text(s, lab, (x0 + 58, yy - 4), 13, col, anchor="bottomleft")
        prog = ease((self.t - getattr(self, "t_reveal", 0)) / 0.5)
        for (k, d), col in zip(S["dagger"].items(), [TEAL, SKY, LAV]):
            ys = [y0 + h - h * (v - lo) / (hi - lo) for v in d["return_mean"]]
            n = self.shown
            pts = list(zip(xs[:n], ys[:n]))
            if n >= 2:
                last = (lerp(xs[n - 2], xs[n - 1], prog), lerp(ys[n - 2], ys[n - 1], prog))
                pts = pts[:-1] + [last]
            if len(pts) > 1:
                pygame.draw.lines(s, col, False, pts, 3)
            for p in pts:
                circle(s, col, p, 6)
            text(s, f"seed {k}", (x0 + 60 + int(k) * 90, 166), 15, col, True)
        if self.shown == len(its):
            fin = S["controllers"]["Distilled policy"]["return_mean"]
            text(s, f"Final distilled policy on 15 test episodes: {fmt_num(fin)} vs CEM {fmt_num(cem)}",
                 (900, 612), 16, TEAL, True, anchor="center")


# ============================================================ 9. race
class RaceSlide(Slide):
    kicker, title = "Live evaluation", "Same start, four controllers, running now"
    kicker_color = TEAL
    hints = "K / click: kick all four   ·   R: restart from hanging down"
    NAMES = ["Random", "Random-shooting MPC", "CEM MPC", "Distilled policy"]

    def enter(self):
        super().enter()
        m = self.app.models
        self.ctrls = [Timed(m.controller(n, seed=7)) for n in self.NAMES]
        self.restart()

    def restart(self):
        self.ps = [Pendulum() for _ in self.NAMES]
        for c in self.ctrls:
            c.reset()
        self.done_t = 0.0

    def key(self, k):
        if k == pygame.K_k:
            for p in self.ps:
                p.kick(5.0)
        elif k == pygame.K_r:
            self.restart()
        else:
            return False
        return True

    def click(self, pos, button):
        for p in self.ps:
            p.kick(5.0)
        return True

    def update(self, dt):
        super().update(dt)
        if self.ps[0].steps >= 200:
            self.done_t += dt
            if self.done_t > 3.0:
                self.restart()
            return
        # one control step per frame for every controller (planners are slow!)
        for p, c in zip(self.ps, self.ctrls):
            p.step(c.act(p.obs()))

    def draw(self, s):
        for i, (n, p, c) in enumerate(zip(self.NAMES, self.ps, self.ctrls)):
            x = 60 + i * 296
            col = CTRL_COLORS[n]
            rrect(s, CARD, (x, 150, 276, 480), 18)
            text(s, n, (x + 138, 176), 19, col, True, anchor="center")
            cx, cy, R = x + 138, 330, 100
            draw_pendulum(s, (cx, cy), R, p.obs()[:2], col, 14, torque=p.u)
            text(s, fmt_num(p.ret), (x + 138, 470), 40, TEXT, True, anchor="center")
            text(s, "return so far", (x + 138, 500), 14, MUTED, anchor="center")
            # decision time on a log scale from 1 µs to 100 ms
            ms = max(c.ms, 1e-3)
            f = (math.log10(ms) + 3) / 5
            pygame.draw.rect(s, GRID, (x + 24, 556, 228, 10), border_radius=5)
            pygame.draw.rect(s, col, (x + 24, 556, int(228 * max(0.02, min(1, f))), 10), border_radius=5)
            text(s, f"{c.ms:.3f} ms / decision" if c.ms < 1 else f"{c.ms:.1f} ms / decision", (x + 138, 584), 16,
                 TEXT, True, anchor="center")
            text(s, "log scale, 1 µs → 100 ms", (x + 138, 606), 12, FAINT, anchor="center")
        text(s, f"step {self.ps[0].steps}/200   ·   all four start hanging down at rest", (640, 650), 15, MUTED,
             anchor="center")


# ============================================================ 10. results
class ResultsSlide(Slide):
    kicker, title = "Reported results · results/summary.json", "Close to the planner's return at a fraction of the cost"
    kicker_color = TEAL
    hints = "L: switch decision time between linear and log scale"

    def enter(self):
        super().enter()
        self.log = False

    def key(self, k):
        if k == pygame.K_l:
            self.log = not self.log
            self.t = 0.6
            return True
        return False

    def draw(self, s):
        S = self.app.models.summary
        C = S["controllers"]
        names = list(C)
        short = {"Random": "Random", "Random-shooting MPC": "Rand. shoot", "CEM MPC": "CEM MPC", "Distilled policy": "Distilled"}
        g = ease(self.t / 1.2)
        # returns
        x0, y0, w, h = 60, 150, 560, 420
        rrect(s, CARD, (x0, y0, w, h), 18)
        text(s, f"Return, mean ± std of {C['CEM MPC']['n_episodes']} episodes", (x0 + 24, y0 + 16), 19, TEXT, True)
        base = y0 + 70
        scale = 300 / 1450
        for i, n in enumerate(names):
            bx = x0 + 50 + i * 125
            v, sd = C[n]["return_mean"], C[n]["return_std"]
            hgt = -v * scale * g
            pygame.draw.rect(s, CTRL_COLORS[n], (bx, base, 70, hgt), border_bottom_left_radius=8, border_bottom_right_radius=8)
            near, far = base + max(0, -v - sd) * scale * g, base + (-v + sd) * scale * g
            pygame.draw.line(s, TEXT, (bx + 35, near), (bx + 35, far), 2)
            pygame.draw.line(s, TEXT, (bx + 27, far), (bx + 43, far), 2)
            text(s, fmt_num(v * g), (bx + 35, far + 6), 17, TEXT, True, anchor="midtop")
            text(s, short[n], (bx + 35, base - 8), 14, MUTED, anchor="midbottom")
        pygame.draw.line(s, FAINT, (x0 + 30, base), (x0 + w - 30, base))
        # decision time
        x0 = 660
        rrect(s, CARD, (x0, y0, w, h), 18)
        text(s, f"Milliseconds per decision ({'log' if self.log else 'linear'} scale)", (x0 + 24, y0 + 16), 19, TEXT, True)
        base = y0 + h - 50
        for i, n in enumerate(names):
            bx = x0 + 50 + i * 125
            v = C[n]["decision_time_ms_mean"]
            if self.log:
                f = (math.log10(v) + 3) / (math.log10(60) + 3)
            else:
                f = v / 50
            hgt = max(2, 300 * f * g)
            pygame.draw.rect(s, CTRL_COLORS[n], (bx, base - hgt, 70, hgt), border_top_left_radius=8, border_top_right_radius=8)
            text(s, f"{v * g:.3f}" if v < 1 else f"{v * g:.1f}", (bx + 35, base - hgt - 6), 17, TEXT, True, anchor="midbottom")
            text(s, short[n], (bx + 35, base + 10), 14, MUTED, anchor="midtop")
        pygame.draw.line(s, FAINT, (x0 + 30, base), (x0 + w - 30, base))
        hd = S["headline"]
        text(s, f"{hd['speedup_policy_vs_cem'] * g:.0f}× faster than CEM", (60, 600), 30, TEAL, True)
        text(s, f"{100 * hd['policy_normalized_score'] * g:.0f}% of CEM's improvement over random kept", (500, 606), 22, TEXT, True)


# ============================================================ 11. readings check
class ClaimsSlide(Slide):
    kicker, title = "Back to the readings", "What the readings claimed, and what my run showed"
    hints = "Click a card or press 1-4 to flip it   ·   A: flip all"

    def enter(self):
        super().enter()
        S = self.app.models.summary
        hd = S["headline"]
        pe = S["world_model"]["seed0_prediction_error"]["model 0 (random data)"]
        r30 = pe["rmse"][29] / pe["rmse"][0]
        self.cards = [
            ("brain", AMBER, "LeCun", "Mode-2 planning can be compiled into a fast Mode-1 policy",
             f"{hd['speedup_policy_vs_cem']:.0f}× faster, {100 * hd['policy_normalized_score']:.0f}% of the planner's gain kept", True),
            ("layers", VIOLET, "Ding et al. (DWM)", "One-step models compound error; keep imagination in training",
             f"Error ×{r30:.0f} after 30 open-loop steps; the deployed policy never touches the model", True),
            ("moon", TEAL, "DreamerV3", "Short imagined rollouts (15 steps) are enough to learn behaviour",
             f"Angle error {pe['angle_err_deg'][14]:.2f}° at 15 steps: enough to swing up from any start", True),
            ("github", GRAY, "DreamerV3 code", "symlog makes regression targets robust across domains",
             "Slightly worse here: Pendulum's targets are already small (an honest negative result)", False),
        ]
        self.flip = [0.0] * 4        # 0 = claim side, 1 = observed side
        self.target = [0] * 4

    def rects(self):
        return [(60 + i * 296, 160, 276, 440) for i in range(4)]

    def key(self, k):
        if pygame.K_1 <= k <= pygame.K_4:
            i = k - pygame.K_1
            self.target[i] = 1 - self.target[i]
            return True
        if k == pygame.K_a:
            v = 0 if all(self.target) else 1
            self.target = [v] * 4
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
        for i in range(4):
            self.flip[i] += (self.target[i] - self.flip[i]) * min(1, dt * 8)

    def draw(self, s):
        for i, ((ic, col, who, claim, obs, ok), (x, y, w, h)) in enumerate(zip(self.cards, self.rects())):
            f = self.flip[i]
            back = f > 0.5
            sx = abs(math.cos(f * math.pi))           # card "turns" around its vertical axis
            card = pygame.Surface((w, h), pygame.SRCALPHA)
            rrect(card, CARD2 if back else CARD, (0, 0, w, h), 18)
            icon(card, ic, (50, 55), 60, col)
            text(card, who, (24, 100), 24, TEXT, True)
            if not back:
                text(card, "CLAIM", (24, 150), 15, MUTED, True)
                para(card, claim, 24, 178, w - 48, 22, TEXT)
                text(card, f"press {i + 1} to see the result", (24, h - 40), 14, FAINT)
            else:
                text(card, "WHAT I OBSERVED", (24, 150), 15, col, True)
                para(card, obs, 24, 178, w - 48, 22, TEXT)
                icon(card, "check" if ok else "warn", (w - 46, h - 46), 48, TEAL if ok else CORAL)
            cw = max(1, int(w * sx))
            card = pygame.transform.smoothscale(card, (cw, h))
            s.blit(card, (x + (w - cw) // 2, y))


# ============================================================ 12. takeaway
class TakeawaySlide(Slide, SimMixin):
    hints = "Click or K: kick the pendulum: the 0.1 ms policy recovers"

    def enter(self):
        super().enter()
        self.p = Pendulum(0.0, 0.0)
        self.ctrl = Timed(self.app.models.controller("Distilled policy"))
        self.sim_reset()

    def update(self, dt):
        super().update(dt)
        for _ in range(self.sim_ticks(dt)):
            self.p.step(self.ctrl.act(self.p.obs()))

    def key(self, k):
        if k == pygame.K_k:
            self.p.kick(np.random.choice([-6.0, 6.0]))
            return True
        return False

    def click(self, pos, button):
        self.p.kick(6.0 if pos[0] < 1000 else -6.0)
        return True

    def draw(self, s):
        a = self.appear(0.0)
        text(s, "TAKEAWAY", (80, 150), 18, AMBER, True)
        text(s, "Think slowly while training.", (76, 190), 46, AMBER, True, alpha=int(255 * a))
        a2 = self.appear(0.4)
        text(s, "React quickly when deployed.", (76, 250), 46, TEAL, True, alpha=int(255 * a2))
        para(s, "A learned world model plus a planner gives good behaviour (Mode-2). Distilling that planner gives the "
                "same behaviour cheaply enough for a real-time robot (Mode-1).", 80, 340, 680, 22, MUTED)
        rrect(s, CODE_BG, (80, 470, 660, 150), 14)
        lines = ["cd readings/Readings_6", "python main.py", "python presentation/interactive_presentation.py"]
        for i, l in enumerate(lines):
            text(s, l, (104, 492 + i * 36), 20, (220, 227, 239), mono=True)
        cx, cy, R = 1040, 370, 150
        circle(s, CARD, (cx, cy), R + 40)
        draw_pendulum(s, (cx, cy), R, self.p.obs()[:2], TEAL, 16, torque=self.p.u)
        text(s, "github.com/CDJ7/ENG6010 · readings/Readings_6", (cx, 620), 16, MUTED, anchor="center")


# ================================================================ the app
class App:
    SLIDES = [TitleSlide, ReadingsSlide, TryItSlide, ModesSlide, PipelineSlide, CompoundingSlide,
              ImaginationSlide, DaggerSlide, RaceSlide, ResultsSlide, ClaimsSlide, TakeawaySlide]

    def __init__(self, fullscreen=False, start=0):
        global FONTS
        pygame.display.init()             # no audio needed
        pygame.font.init()
        pygame.display.set_caption("ENG 6010 · Readings 6 · From Mode-2 to Mode-1")
        flags = pygame.SCALED | pygame.RESIZABLE
        if fullscreen:
            flags |= pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode((W, H), flags, vsync=1)
        except pygame.error:                      # no vsync on this machine
            self.screen = pygame.display.set_mode((W, H), flags)
        FONTS = Fonts()
        self.models = Models()
        self.slides = [S(self) for S in self.SLIDES]
        self.i = max(0, min(start, len(self.slides) - 1))
        self.slides[self.i].enter()
        self.fade = 1.0
        self.clock = pygame.time.Clock()

    def go(self, d):
        j = max(0, min(len(self.slides) - 1, self.i + d))
        if j != self.i:
            self.i = j
            self.slides[j].enter()
            self.fade = 0.0

    def frame(self, dt):
        sl = self.slides[self.i]
        sl.update(dt)
        s = self.screen
        s.fill(BG_DEEP if self.i in (0, len(self.slides) - 1) else BG)
        if sl.title:
            text(s, sl.kicker.upper(), (60, 46), 15, sl.kicker_color, True)
            text(s, sl.title, (58, 70), 40, TEXT, True)
        sl.draw(s)
        # footer: hints, progress dots, page number
        if sl.hints:
            text(s, sl.hints, (60, 688), 14, FAINT)
        for k in range(len(self.slides)):
            col = TEXT if k == self.i else GRID
            circle(s, col, (W - 60 - (len(self.slides) - 1 - k) * 14, 695), 4)
        if self.fade < 1.0:
            self.fade = min(1.0, self.fade + dt * 4)
            veil = pygame.Surface((W, H))
            veil.fill(BG)
            veil.set_alpha(int(255 * (1 - ease(self.fade))))
            s.blit(veil, (0, 0))

    def run(self):
        while True:
            dt = min(self.clock.tick(60) / 1000.0, 0.1)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        return
                    if e.key == pygame.K_f:
                        pygame.display.toggle_fullscreen()
                    elif self.slides[self.i].key(e.key):
                        pass
                    elif e.key in (pygame.K_RIGHT, pygame.K_SPACE, pygame.K_PAGEDOWN, pygame.K_RETURN):
                        self.go(1)
                    elif e.key in (pygame.K_LEFT, pygame.K_PAGEUP, pygame.K_BACKSPACE):
                        self.go(-1)
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.slides[self.i].click(e.pos, e.button)
            self.frame(dt)
            pygame.display.flip()

    def screenshots(self, out_dir, seconds=4.0, keys=None):
        """Headless QA: simulate every slide for a few seconds and save a PNG."""
        os.makedirs(out_dir, exist_ok=True)
        keys = keys or {}
        for i in range(len(self.slides)):
            self.i = i
            self.slides[i].enter()
            self.fade = 1.0
            for k in keys.get(i, []):
                self.slides[i].key(k)
            for _ in range(int(seconds * 20)):
                self.frame(0.05)
            pygame.image.save(self.screen, os.path.join(out_dir, f"slide_{i + 1:02d}.png"))
            print("saved slide", i + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--start", type=int, default=1, help="slide number to start on")
    ap.add_argument("--screenshots", metavar="DIR", help="save a PNG of every slide and exit (no window needed)")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(RESULTS, "models", "policy_seed0.pt")):
        sys.exit("results/models/*.pt not found - run `python main.py` first.")
    if args.screenshots:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    app = App(args.fullscreen, args.start - 1)
    if args.screenshots:
        app.screenshots(args.screenshots)
    else:
        app.run()
    pygame.quit()


if __name__ == "__main__":
    main()
