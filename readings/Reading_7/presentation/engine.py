"""
engine.py - shared rendering engine for the Reading 7 interactive slideshow.

Taken from the Readings 6 presentation: every slide is laid out on a
1280x720 design canvas, and all coordinates, line widths and font sizes are
multiplied up at draw time, so the slides are drawn at the screen's native
resolution (sharp on any monitor, DPI-aware on Windows).
"""

import argparse
import math
import os
import sys

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
import pygame.gfxdraw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ICONS = os.path.join(HERE, "icons")

# ------------------------------------------------------------------ palette
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
PINK = (244, 114, 182)
GREEN = (74, 222, 128)

W, H = 1280, 720        # design canvas; drawn at native resolution

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


class View:
    """Maps the 1280x720 design canvas onto the real window.

    Every slide is laid out in 1280x720 "design units". Instead of drawing a
    small 1280x720 image and stretching it (which looks blurry on a large or
    high-DPI screen), all coordinates, line widths and font sizes are
    multiplied by `k` at draw time, so everything is rendered at the screen's
    native resolution. (ox, oy) letterbox the 16:9 canvas on other aspect ratios.
    """

    k, ox, oy = 1.0, 0.0, 0.0

    @classmethod
    def fit(cls, w, h):
        cls.k = min(w / W, h / H)
        cls.ox, cls.oy = (w - W * cls.k) / 2, (h - H * cls.k) / 2


def P(p):
    """Design-space point -> screen pixels."""
    return (View.ox + p[0] * View.k, View.oy + p[1] * View.k)


def PX(v):
    """Design-space length -> pixels (at least 1)."""
    return max(1, int(round(v * View.k)))


def RECT(r):
    x, y = P((r[0], r[1]))
    return pygame.Rect(round(x), round(y), round(r[2] * View.k), round(r[3] * View.k))


def to_design(pos):
    """Screen pixels (mouse) -> design-space point."""
    return ((pos[0] - View.ox) / View.k, (pos[1] - View.oy) / View.k)


class local_origin:
    """Draw into an off-screen card surface whose top-left is design (0, 0)."""

    def __enter__(self):
        self.saved = (View.ox, View.oy)
        View.ox = View.oy = 0.0

    def __exit__(self, *exc):
        View.ox, View.oy = self.saved
        return False


class sublayer:
    """Transparent layer covering only `rect` (design units), for translucent drawing.

        with sublayer(screen, (x, y, w, h), alpha=120) as ov:
            draw.lines(ov, ...)      # same design coordinates as usual

    Only that region is allocated and blitted, which keeps full-resolution
    rendering fast. `alpha` fades the whole layer when it is blitted.
    """

    def __init__(self, surf, rect, alpha=None):
        self.surf, self.rect, self.alpha = surf, rect, alpha

    def __enter__(self):
        self.r = RECT(self.rect).clip(self.surf.get_rect())
        self.layer = pygame.Surface((max(1, self.r.w), max(1, self.r.h)), pygame.SRCALPHA)
        self.saved = (View.ox, View.oy)
        View.ox, View.oy = View.ox - self.r.x, View.oy - self.r.y
        return self.layer

    def __exit__(self, *exc):
        View.ox, View.oy = self.saved
        if self.alpha is not None:
            self.layer.set_alpha(self.alpha)
        self.surf.blit(self.layer, self.r.topleft)
        return False


class draw:
    """pygame.draw with design-space coordinates."""

    @staticmethod
    def rect(surf, color, rect, width=0, border_radius=0, **radii):
        radii = {k: int(round(v * View.k)) for k, v in radii.items()}
        pygame.draw.rect(surf, color, RECT(rect), PX(width) if width else 0,
                         border_radius=int(round(border_radius * View.k)), **radii)

    @staticmethod
    def line(surf, color, p1, p2, width=1):
        pygame.draw.line(surf, color, P(p1), P(p2), PX(width))

    @staticmethod
    def lines(surf, color, closed, pts, width=1):
        # vectorised: the CEM slide draws hundreds of polylines per frame
        xy = np.asarray(pts, dtype=float) * View.k + (View.ox, View.oy)
        pygame.draw.lines(surf, color, closed, xy.tolist(), PX(width))

    @staticmethod
    def aalines(surf, color, closed, pts):
        xy = np.asarray(pts, dtype=float) * View.k + (View.ox, View.oy)
        pygame.draw.aalines(surf, color, closed, xy.tolist())

    @staticmethod
    def circle(surf, color, center, r, width=0):
        pygame.draw.circle(surf, color, P(center), r * View.k, PX(width) if width else 0)

    @staticmethod
    def arc(surf, color, rect, a0, a1, width=1):
        pygame.draw.arc(surf, color, RECT(rect), a0, a1, PX(width))


class gfx:
    """pygame.gfxdraw (anti-aliased) with design-space coordinates."""

    @staticmethod
    def filled_circle(surf, x, y, r, color):
        cx, cy = P((x, y))
        pygame.gfxdraw.filled_circle(surf, int(cx), int(cy), PX(r), color)


class Fonts:
    """Font cache keyed by *pixel* size. Calibri/Segoe on Windows, Carlito/DejaVu/Arial elsewhere."""

    SANS = "calibri,carlito,segoeui,helveticaneue,arial,dejavusans"
    MONO = "consolas,menlo,dejavusansmono,couriernew,freemono"

    def __init__(self):
        self.cache = {}

    def get(self, size, bold=False, mono=False):
        """`size` is in design units; the font is created at the scaled pixel size."""
        px = max(6, int(round(size * View.k)))
        key = (px, bold, mono)
        if key not in self.cache:
            self.cache[key] = pygame.font.SysFont(self.MONO if mono else self.SANS, px, bold=bold)
        return self.cache[key]


FONTS = None


class TextBox:
    """Size of rendered text in design units (what callers lay out with)."""

    def __init__(self, w, h):
        self.width, self.height = w, h


def text_width(s, size, bold=False, mono=False):
    return FONTS.get(size, bold, mono).size(s)[0] / View.k


def text(surf, s, pos, size=20, color=TEXT, bold=False, mono=False, anchor="topleft", alpha=255):
    img = FONTS.get(size, bold, mono).render(s, True, color)
    if alpha < 255:
        img.set_alpha(alpha)
    x, y = P(pos)
    r = img.get_rect(**{anchor: (round(x), round(y))})
    surf.blit(img, r)
    return TextBox(r.width / View.k, r.height / View.k)


def wrap(s, size, width, bold=False):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if text_width(t, size, bold) <= width:
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
        draw.rect(surf, color, rect, border_radius=radius)
    else:
        r = RECT(rect)
        tmp = pygame.Surface((max(1, r.w), max(1, r.h)), pygame.SRCALPHA)
        pygame.draw.rect(tmp, (*color, alpha), (0, 0, r.w, r.h), border_radius=int(round(radius * View.k)))
        surf.blit(tmp, r.topleft)


def circle(surf, color, center, r):
    """Anti-aliased filled circle."""
    x, y = P(center)
    x, y, rr = int(round(x)), int(round(y)), max(1, int(round(r * View.k)))
    pygame.gfxdraw.aacircle(surf, x, y, rr, color)
    pygame.gfxdraw.filled_circle(surf, x, y, rr, color)


def thick_line(surf, color, p1, p2, width):
    """Anti-aliased thick line with round caps (used for the pendulum rod)."""
    (x1, y1), (x2, y2) = p1, p2
    d = math.hypot(x2 - x1, y2 - y1) or 1.0
    nx, ny = -(y2 - y1) / d * width / 2, (x2 - x1) / d * width / 2
    pts = [P(q) for q in ((x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny))]
    pygame.gfxdraw.aapolygon(surf, pts, color)
    pygame.gfxdraw.filled_polygon(surf, pts, color)
    circle(surf, color, p1, width / 2)
    circle(surf, color, p2, width / 2)


def arrow(surf, color, p1, p2, width=2, head=10):
    draw.line(surf, color, p1, p2, width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    a1 = (p2[0] - head * math.cos(ang - 0.45), p2[1] - head * math.sin(ang - 0.45))
    a2 = (p2[0] - head * math.cos(ang + 0.45), p2[1] - head * math.sin(ang + 0.45))
    tri = [P(p2), P(a1), P(a2)]
    pygame.gfxdraw.aapolygon(surf, tri, color)
    pygame.gfxdraw.filled_polygon(surf, tri, color)


_icon_cache = {}


def icon(surf, name, center, d, color):
    circle(surf, color, center, d / 2)
    px = PX(d * 0.52)
    key = (name, px)
    if key not in _icon_cache:
        img = pygame.image.load(os.path.join(ICONS, name + ".png")).convert_alpha()
        _icon_cache[key] = pygame.transform.smoothscale(img, (px, px))
    img = _icon_cache[key]
    x, y = P(center)
    surf.blit(img, img.get_rect(center=(int(round(x)), int(round(y)))))



# ============================================================ pendulum drawing
def tip_pts(cs, center, R):
    """Pendulum-tip points (design units) for an array of (cos, sin) rows."""
    cs = np.asarray(cs, dtype=float).reshape(-1, 2)
    return np.stack([center[0] + R * cs[:, 1], center[1] - R * cs[:, 0]], 1)


def draw_pendulum(surf, center, R, cos_sin, color, width=None, pivot=True, torque=0.0, alpha=None):
    """Draw the pole. cos_sin = (cos th, sin th); th = 0 is upright."""
    c, s = cos_sin
    cx, cy = center
    tip = (cx + R * float(s), cy - R * float(c))
    width = width or max(6, int(R * 0.11))
    if alpha is not None:
        # draw opaque on a transparent layer, then fade the whole layer
        m = R + width
        with sublayer(surf, (cx - m, cy - m, 2 * m, 2 * m), alpha) as tmp:
            thick_line(tmp, color, (cx, cy), tip, width)
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
        draw.arc(surf, MUTED, (cx - r, cy - r, 2 * r, 2 * r), a0, a1, 3)
    return tip


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

    def release(self, pos):
        pass

    def motion(self, pos, buttons):
        pass

    typing = False          # True while the slide is capturing typed text

    def text_input(self, s):
        pass

    # helpers -------------------------------------------------------------
    def appear(self, delay, dur=0.6):
        """0 -> 1 entrance progress for an element shown `delay` s after entry."""
        return ease((self.t - delay) / dur)


SIM_DT = 0.05            # Pendulum-v1 time step (20 Hz control)


class SimMixin:
    """Fixed-rate (20 Hz) stepping of a simulation."""

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


def make_dpi_aware():
    """Stop Windows from bitmap-stretching the window when display scaling is
    125 % / 150 % (that stretching is what makes text look blurry)."""
    if sys.platform != "win32":
        return
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)      # per-monitor aware
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass




class App:
    """Slide deck runner: navigation, fades, footer, text input, screenshots."""

    def __init__(self, slide_classes, caption, fullscreen=False, start=0, size=None, context=None):
        global FONTS
        make_dpi_aware()
        pygame.display.init()             # no audio needed
        pygame.font.init()
        pygame.display.set_caption(caption)
        self.fixed_size = size            # only used for --screenshots
        self.context = context or {}
        self.open_window(fullscreen)
        FONTS = Fonts()
        self.slides = [S(self) for S in slide_classes]
        self.i = max(0, min(start, len(self.slides) - 1))
        self.slides[self.i].enter()
        self.fade = 1.0
        self.clock = pygame.time.Clock()

    def open_window(self, fullscreen):
        """Windowed: the largest 16:9 window that fits ~90% of the desktop.
        Fullscreen: the monitor's native resolution."""
        self.fullscreen = fullscreen
        if self.fixed_size:
            size, flags = self.fixed_size, 0
        elif fullscreen:
            size, flags = (0, 0), pygame.FULLSCREEN
        else:
            try:
                dw, dh = pygame.display.get_desktop_sizes()[0]
            except Exception:
                dw, dh = W, H
            k = min(dw * 0.9 / W, dh * 0.85 / H)
            size, flags = (int(W * k), int(H * k)), pygame.RESIZABLE
        self.screen = pygame.display.set_mode(size, flags)
        View.fit(*self.screen.get_size())

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
        View.fit(*s.get_size())
        s.fill(BG_DEEP if getattr(sl, "deep", False) else BG)
        if sl.title:
            text(s, sl.kicker.upper(), (60, 46), 15, sl.kicker_color, True)
            text(s, sl.title, (58, 70), 40, TEXT, True)
        sl.draw(s)
        if sl.hints:
            text(s, sl.hints, (60, 688), 14, FAINT)
        n = len(self.slides)
        for k in range(n):
            circle(s, TEXT if k == self.i else GRID, (W - 60 - (n - 1 - k) * 14, 695), 4)
        if self.fade < 1.0:
            self.fade = min(1.0, self.fade + dt * 4)
            veil = pygame.Surface(s.get_size())
            veil.fill(BG)
            veil.set_alpha(int(255 * (1 - ease(self.fade))))
            s.blit(veil, (0, 0))

    def run(self):
        while True:
            dt = min(self.clock.tick(60) / 1000.0, 0.1)
            for e in pygame.event.get():
                sl = self.slides[self.i]
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.screen = pygame.display.get_surface()
                if e.type == pygame.TEXTINPUT and sl.typing:
                    sl.text_input(e.text)
                    continue
                if e.type == pygame.KEYDOWN:
                    if sl.typing:                 # the slide owns the keyboard while typing
                        sl.key(e.key)
                        continue
                    if e.key == pygame.K_ESCAPE:
                        return
                    if e.key == pygame.K_f:
                        self.open_window(not self.fullscreen)
                    elif sl.key(e.key):
                        pass
                    elif e.key in (pygame.K_RIGHT, pygame.K_SPACE, pygame.K_PAGEDOWN, pygame.K_RETURN):
                        self.go(1)
                    elif e.key in (pygame.K_LEFT, pygame.K_PAGEUP, pygame.K_BACKSPACE):
                        self.go(-1)
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    sl.click(to_design(e.pos), e.button)
                if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    sl.release(to_design(e.pos))
                if e.type == pygame.MOUSEMOTION:
                    sl.motion(to_design(e.pos), e.buttons)
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


def run_cli(slide_classes, caption, doc, context=None):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--start", type=int, default=1, help="slide number to start on")
    ap.add_argument("--screenshots", metavar="DIR", help="save a PNG of every slide and exit (no window needed)")
    ap.add_argument("--size", default="1280x720", help="image size for --screenshots, e.g. 1920x1080")
    args = ap.parse_args()
    if args.screenshots:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.screenshots else None
    app = App(slide_classes, caption, args.fullscreen, args.start - 1, size, context)
    if args.screenshots:
        app.screenshots(args.screenshots)
    else:
        app.run()
    pygame.quit()


# ============================================================ widgets (Reading 7 v2)
def mouse():
    """Mouse position in design units."""
    return to_design(pygame.mouse.get_pos())


def inside(pos, rect):
    x, y, w, h = rect
    return x <= pos[0] <= x + w and y <= pos[1] <= y + h


def poly(surf, color, pts):
    """Anti-aliased filled polygon in design units."""
    q = [P(p) for p in pts]
    pygame.gfxdraw.aapolygon(surf, q, color)
    pygame.gfxdraw.filled_polygon(surf, q, color)


def slider(surf, x, y, w, value, vmin, vmax, label, color, active=False, fmt="{:.2f}"):
    """A horizontal slider: label above, track, knob, value on the right."""
    f = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    f = max(0.0, min(1.0, f))
    text(surf, label, (x, y), 14, TEXT if active else MUTED, active)
    text(surf, fmt.format(value), (x + w, y), 14, color if active else MUTED, True, anchor="topright")
    draw.rect(surf, GRID, (x, y + 24, w, 6), border_radius=3)
    draw.rect(surf, color, (x, y + 24, w * f, 6), border_radius=3)
    circle(surf, color, (x + w * f, y + 27), 9 if active else 7)
    if active:
        circle(surf, BG, (x + w * f, y + 27), 3)


def toggle(surf, x, y, on, label, color):
    """An on/off switch with a label to its right."""
    rrect(surf, color if on else GRID, (x, y, 44, 24), 12)
    circle(surf, TEXT, (x + (32 if on else 12), y + 12), 9)
    text(surf, label, (x + 54, y + 3), 15, TEXT if on else MUTED, on)


def ring(surf, center, r, frac, color, width=16, track=GRID):
    """Donut / progress ring, drawn as polygons so it is smooth at any size."""
    cx, cy = center
    def band(a0, a1, col):
        n = max(2, int(abs(a1 - a0) / 0.05))
        outer = [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
        ri = r - width
        inner = [(cx + ri * math.cos(a0 + (a1 - a0) * i / n), cy + ri * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
        poly(surf, col, outer + inner[::-1])
    start = -math.pi / 2
    band(start, start + 2 * math.pi * 0.9999, track)
    if frac > 0.002:
        band(start, start + 2 * math.pi * min(frac, 0.9999), color)


def bubble(surf, x, y, w, lines, color, size=14, tail=None):
    """Speech bubble with wrapped text; tail = (x, y) point it points at."""
    h = 16
    rows = []
    for ln in lines:
        rows += wrap(ln, size, w - 28) or [""]
    h += len(rows) * (size + 6) + 4
    if tail:
        tx, ty = tail
        bx = min(max(tx, x + 20), x + w - 20)
        poly(surf, CARD2, [(bx - 10, y + (h if ty > y + h else 0)), (bx + 10, y + (h if ty > y + h else 0)), (tx, ty)])
    rrect(surf, CARD2, (x, y, w, h), 12)
    draw.rect(surf, color, (x, y, w, h), 2, border_radius=12)
    yy = y + 10
    for ln in rows:
        text(surf, ln, (x + 14, yy), size, TEXT)
        yy += size + 6
    return h


def chip(surf, x, y, label, color, on=False, size=14):
    """Pill-shaped button; returns its rect (design units) for click tests."""
    w = text_width(label, size, True) + 26
    h = size + 14
    rrect(surf, color if on else CARD2, (x, y, w, h), h // 2)
    if not on:
        draw.rect(surf, color, (x, y, w, h), 1, border_radius=h // 2)
    text(surf, label, (x + w / 2, y + h / 2), size, BG if on else TEXT, True, anchor="center")
    return (x, y, w, h)


def colormap(a, stops):
    """Map an array in [0, 1] to RGB by interpolating between colour stops."""
    a = np.clip(a, 0, 1)
    xs = np.linspace(0, 1, len(stops))
    st = np.array(stops, float)
    return np.stack([np.interp(a, xs, st[:, c]) for c in range(3)], -1).astype(np.uint8)


def blit_array(surf, rgb, rect):
    """Draw an (h, w, 3) uint8 image into a design-space rectangle."""
    img = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
    r = RECT(rect)
    surf.blit(pygame.transform.smoothscale(img, (max(1, r.w), max(1, r.h))), r.topleft)


def mark(surf, ok, center, size, color):
    """A drawn tick (ok=True) or cross: the UI fonts have no check-mark glyph."""
    x, y = center
    h = size / 2
    if ok:
        pts = [(x - h, y), (x - h * 0.25, y + h * 0.7), (x + h, y - h * 0.7)]
        thick_line(surf, color, pts[0], pts[1], max(2, size / 6))
        thick_line(surf, color, pts[1], pts[2], max(2, size / 6))
    else:
        thick_line(surf, color, (x - h * 0.7, y - h * 0.7), (x + h * 0.7, y + h * 0.7), max(2, size / 6))
        thick_line(surf, color, (x - h * 0.7, y + h * 0.7), (x + h * 0.7, y - h * 0.7), max(2, size / 6))
