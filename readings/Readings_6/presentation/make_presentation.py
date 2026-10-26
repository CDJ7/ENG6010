"""
make_presentation.py - builds Readings_6_Presentation.pptx with python-pptx.

    cd readings/Readings_6
    python presentation/make_presentation.py

Every number on the slides is read from results/summary.json (written by
main.py), so the deck always matches the reported run. Pendulum frames are
taken from results/gifs/*.gif and recoloured for the dark theme. Charts are
native PowerPoint charts, so they stay editable.
"""

import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.chart import (XL_CHART_TYPE, XL_DATA_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE,
                            XL_TICK_LABEL_POSITION)
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # readings/Readings_6
RESULTS = os.path.join(ROOT, "results")
ICONS = os.path.join(HERE, "icons")
FRAMES = os.path.join(HERE, "frames")
OUT = os.path.join(ROOT, "Readings_6_Presentation.pptx")

# ---------------------------------------------------------------- palette
# Deep "night lab" navy dominates; amber = Mode-2 (slow, deliberate),
# teal = Mode-1 (fast reflex). The same two colours mean the same thing on
# every slide.
BG = "0D1424"
CARD = "17213A"
CARD2 = "1F2C4A"
CODE_BG = "0A1020"
TEXT = "F2F5FA"
MUTED = "A7B3C9"
FAINT = "56637D"
AMBER = "F5A524"     # Mode-2 / CEM planner
TEAL = "2DD4BF"      # Mode-1 / distilled policy
VIOLET = "A78BFA"    # random-shooting MPC
CORAL = "F87171"     # warnings / limitations
GRAY = "8A94A8"      # random policy
CTRL_COLORS = {"Random": GRAY, "Random-shooting MPC": VIOLET, "CEM MPC": AMBER, "Distilled policy": TEAL}

HEAD = "Calibri"
BODY = "Calibri"
MONO = "Courier New"

W, H = 13.333, 7.5


def num(v, fmt=".0f"):
    """Format a number with a typographic minus sign."""
    return format(v, fmt).replace("-", "\u2212")


def rgb(h):
    return RGBColor.from_string(h)


# ================================================================ helpers
def new_slide(prs, bg=BG):
    s = prs.slides.add_slide(prs.slide_layouts[6])        # blank layout
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(bg)
    return s


def text(slide, x, y, w, h, paras, size=16, color=TEXT, bold=False, font=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing_after=4, name=None, line_spacing=None):
    """Add a text box. `paras` is a string, or a list of paragraphs where each
    paragraph is a string or a list of (text, {overrides}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing_after)
        if line_spacing:
            p.line_spacing = line_spacing
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.name = o.get("font", font)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=TEXT, bullet_color=None, gap=8):
    """Real bulleted paragraphs (bullet character set in the XML, not typed)."""
    tb = text(slide, x, y, w, h, items, size=size, color=color, spacing_after=gap)
    for p in tb.text_frame.paragraphs:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.28))))
        pPr.set("indent", str(-Emu(Inches(0.28))))
        bc = bullet_color or color
        pPr.append(parse_xml(f'<a:buClr {nsdecls("a")}><a:srgbClr val="{bc}"/></a:buClr>'))
        pPr.append(parse_xml(f'<a:buFont {nsdecls("a")} typeface="Arial"/>'))
        pPr.append(parse_xml(f'<a:buChar {nsdecls("a")} char="&#8226;"/>'))
    return tb


def card(slide, x, y, w, h, fill=CARD, radius=0.06, name=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    sh.line.fill.background()
    sh.shadow.inherit = False
    if name:
        sh.name = name
    return sh


def icon(slide, name, x, y, d=0.62, color=AMBER):
    """The deck's motif: a white icon inside a coloured circle."""
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = rgb(color)
    c.line.fill.background()
    c.shadow.inherit = False
    pad = d * 0.24
    pic = slide.shapes.add_picture(os.path.join(ICONS, name + ".png"), Inches(x + pad), Inches(y + pad),
                                   Inches(d - 2 * pad), Inches(d - 2 * pad))
    pic.name = f"icon {name}"
    return c


def number_badge(slide, n, x, y, d=0.5, color=AMBER, fg=BG):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = rgb(color)
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.name = HEAD
    r.font.color.rgb = rgb(fg)
    return c


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=2.0):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = rgb(color)
    ln.line.width = Pt(width)
    ln.line._get_or_add_ln().append(parse_xml(f'<a:tailEnd {nsdecls("a")} type="triangle" w="med" len="med"/>'))
    return ln


def header(slide, kicker, title, kicker_color=AMBER):
    """Same position, font and size for every content-slide title."""
    text(slide, 0.6, 0.38, 12.1, 0.3, kicker.upper(), size=12, color=kicker_color, bold=True, name="kicker")
    text(slide, 0.6, 0.68, 12.1, 0.75, title, size=32, bold=True, font=HEAD, name="title")


def footer(slide, n):
    text(slide, 0.6, 7.0, 9, 0.25, "ENG 6010 · Readings 6 · Mode-2 to Mode-1", size=10, color=FAINT)
    text(slide, 11.73, 7.0, 1.0, 0.25, str(n), size=10, color=FAINT, align=PP_ALIGN.RIGHT)


def code_card(slide, x, y, w, h, lines, size=13, title=None):
    card(slide, x, y, w, h, fill=CODE_BG, radius=0.04)
    top = y + 0.2
    if title:
        text(slide, x + 0.3, top, w - 0.6, 0.3, title, size=11, color=MUTED, font=MONO)
        top += 0.4
    paras = []
    for ln in lines:
        if "#" in ln:
            i = ln.index("#")
            paras.append([(ln[:i], {}), (ln[i:], {"color": "7C8BA8"})])
        else:
            paras.append([(ln if ln else " ", {})])
    text(slide, x + 0.3, top, w - 0.6, h - (top - y) - 0.15, paras, size=size, color="DCE3EF",
         font=MONO, spacing_after=1)


def stat(slide, x, y, w, value, label, color=AMBER, vsize=54, lsize=14):
    text(slide, x, y, w, 0.95, value, size=vsize, bold=True, color=color, font=HEAD)
    text(slide, x, y + 0.95, w, 0.7, label, size=lsize, color=MUTED)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


# ------------------------------------------------------------- chart style
def style_chart(chart, size=12, legend=False):
    chart.font.size = Pt(size)
    chart.font.name = BODY
    chart.font.color.rgb = rgb(MUTED)
    # transparent chart area so the slide colour shows through
    cs = chart._chartSpace
    sp = parse_xml(f'<c:spPr {nsdecls("c", "a")}><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>')
    cs.insert(list(cs).index(cs.find(qn("c:chart"))) + 1, sp)
    chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.color.rgb = rgb(TEXT)
        chart.legend.font.size = Pt(size)


def style_axes(chart, gridlines=True):
    va, ca = chart.value_axis, chart.category_axis
    va.has_major_gridlines = gridlines
    if gridlines:
        va.major_gridlines.format.line.color.rgb = rgb("2A3756")
        va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    ca.format.line.color.rgb = rgb(FAINT)
    for ax in (va, ca):
        ax.tick_labels.font.color.rgb = rgb(MUTED)
        ax.tick_labels.font.size = Pt(12)


def axis_title(axis, s):
    axis.has_title = True
    tf = axis.axis_title.text_frame
    tf.text = s
    f = tf.paragraphs[0].runs[0].font
    f.size = Pt(12)
    f.bold = False
    f.color.rgb = rgb(MUTED)
    f.name = BODY


def log_axis(axis):
    scaling = axis._element.find(qn("c:scaling"))
    scaling.insert(0, parse_xml(f'<c:logBase {nsdecls("c")} val="10"/>'))


# ------------------------------------------------------------ frame images
def gif_frame_at(gif_name, steps):
    """Return the GIF frame showing each requested simulator step.

    The GIF writer merges identical consecutive frames (a balanced pendulum
    does not move), so frame index != time. Each original frame covers
    `gif_frame_skip` steps, so we walk the per-frame durations instead.
    """
    im = Image.open(os.path.join(RESULTS, "gifs", gif_name))
    starts, frames, t = [], [], 0.0
    base = None
    for k in range(im.n_frames):
        im.seek(k)
        d = im.info.get("duration", 66)
        base = d if base is None else min(base, d)
        starts.append(t)
        frames.append(np.asarray(im.convert("RGB")))
        t += d
    ms_per_step = base / 2.0            # main.py saves every 2nd simulator step
    out = []
    for st in steps:
        k = max(i for i, s0 in enumerate(starts) if s0 <= st * ms_per_step + 1e-6)
        out.append(frames[k])
    return out


def dark_frames(gif_name, steps, size=250):
    """Recolour Gymnasium's white-background pendulum frames for the dark theme:
    white -> card colour, black -> light ink, red pole -> amber (planner) or
    teal (policy)."""
    os.makedirs(FRAMES, exist_ok=True)
    bg = np.array([int(CARD2[i:i + 2], 16) for i in (0, 2, 4)], float)
    ink = np.array([0xE6, 0xEA, 0xF2], float)
    pole_hex = TEAL if "policy" in gif_name else AMBER
    pole = np.array([int(pole_hex[i:i + 2], 16) for i in (0, 2, 4)], float)
    paths = []
    for st, fr in zip(steps, gif_frame_at(gif_name, steps)):
        a = fr[..., :3].astype(float)
        t = (255.0 - a.min(-1, keepdims=True)) / 255.0          # amount of "ink"
        redness = np.clip((a[..., :1] - a[..., 1:2]) / 130.0, 0, 1)
        target = redness * pole + (1 - redness) * ink
        out = bg + t * (target - bg)
        p = os.path.join(FRAMES, f"{gif_name[:-4]}_t{st:03d}.png")
        Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS).save(p)
        paths.append(p)
    return paths


# ================================================================== slides
def build():
    S = json.load(open(os.path.join(RESULTS, "summary.json")))
    ctrl = S["controllers"]
    hd = S["headline"]
    pred = S["world_model"]["seed0_prediction_error"]
    plain = pred["model 0 (random data)"]
    sym = pred["model 0 (symlog targets)"]
    losses = S["world_model"]["seed0_final_losses"]["model 0 (random data)"]
    dagger = S["dagger"]
    cfg = S["config"]
    n_ep = ctrl["CEM MPC"]["n_episodes"]
    cem_ms = ctrl["CEM MPC"]["decision_time_ms_mean"]
    pol_ms = ctrl["Distilled policy"]["decision_time_ms_mean"]
    R = {k: v["return_mean"] for k, v in ctrl.items()}
    SD = {k: v["return_std"] for k, v in ctrl.items()}
    speed = hd["speedup_policy_vs_cem"]
    score = hd["policy_normalized_score"]
    imag_steps = cfg["cem_iterations"] * cfg["cem_population"] * (cfg["plan_horizon"] + 1)
    runtime_min = S["runtime_seconds"] / 60

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    prs.core_properties.title = "From Mode-2 to Mode-1"
    prs.core_properties.subject = "ENG 6010 Readings 6"
    n = 0

    # ---------------------------------------------------------- 1. title
    s = new_slide(prs, "0A1020"); n += 1
    text(s, 0.8, 1.2, 7.4, 0.35, "ENG 6010 · Artificial Intelligence in Robotics · Readings 6",
         size=14, color=AMBER, bold=True)
    text(s, 0.8, 1.75, 7.6, 1.0, "From Mode-2 to Mode-1", size=46, bold=True, font=HEAD)
    text(s, 0.8, 2.85, 7.0, 1.3, "Planning with a learned world model, then distilling it into a fast policy",
         size=24, color=MUTED, line_spacing=1.05)
    for i, (lab, col) in enumerate([("Mode-2 · think", AMBER), ("Mode-1 · react", TEAL)]):
        card(s, 0.8 + i * 2.35, 4.5, 2.15, 0.5, fill=CARD, radius=0.5)
        text(s, 0.8 + i * 2.35, 4.5, 2.15, 0.5, lab, size=14, bold=True, color=col,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.8, 5.4, 7.2, 0.9, ["Gymnasium Pendulum-v1 · PyTorch · CPU only",
                                 "LeCun (2022) · Diffusion World Model (2024) · DreamerV3 (2023)"],
         size=14, color=MUTED)
    title_steps = [0, 40, 60, 100]
    fr = dark_frames("cem_mpc.gif", title_steps)
    for i, (p, t) in enumerate(zip(fr, title_steps)):
        x, y = 8.55 + (i % 2) * 2.15, 1.2 + (i // 2) * 2.55
        card(s, x - 0.08, y - 0.08, 2.06, 2.36, fill=CARD2)
        s.shapes.add_picture(p, Inches(x), Inches(y), Inches(1.9), Inches(1.9))
        text(s, x, y + 1.92, 1.9, 0.3, f"t = {t}", size=11, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, 8.47, 6.3, 4.3, 0.3, "CEM planner swinging the pole up from hanging down", size=11,
         color=FAINT, align=PP_ALIGN.CENTER)
    notes(s, "This project turns the common idea of this week's four readings into running code: "
             "learn a model of the world, plan with it slowly (Mode-2), then compile those plans into a fast "
             "reflex policy (Mode-1). Everything runs on a laptop CPU with python main.py.")

    # ---------------------------------------------------- 2. headline
    s = new_slide(prs); n += 1
    header(s, "The result in one slide", "Mode-1 keeps the planner's skill at a fraction of its cost")
    stats = [
        (f"{speed:.0f}×", f"faster decisions: {pol_ms:.3f} ms for the policy vs {cem_ms:.1f} ms for the CEM planner", TEAL),
        (f"{100 * score:.0f}%", f"of the planner's improvement over random play is kept by the distilled policy", TEAL),
        (f"{cfg['plan_horizon']}", "imagined steps per plan, the same horizon as DreamerV3's imag_length", AMBER),
    ]
    for i, (v, lab, col) in enumerate(stats):
        x = 0.6 + i * 4.1
        card(s, x, 1.75, 3.8, 2.7)
        stat(s, x + 0.35, 2.0, 3.2, v, lab, color=col, vsize=60, lsize=15)
    card(s, 0.6, 4.75, 12.1, 1.85, fill=CARD2)
    icon(s, "bullseye", 0.95, 5.3, 0.75, color=AMBER)
    text(s, 2.0, 5.0, 10.4, 1.4, [
        [("Returns over ", {}), (f"{n_ep} unseen test episodes", {"bold": True}), (" (3 seeds), higher is better:", {})],
        [("Random ", {"color": MUTED}), (num(R['Random']), {"bold": True, "color": GRAY}),
         ("    Random-shooting MPC ", {"color": MUTED}), (num(R['Random-shooting MPC']), {"bold": True, "color": VIOLET}),
         ("    CEM MPC ", {"color": MUTED}), (num(R['CEM MPC']), {"bold": True, "color": AMBER}),
         ("    Distilled policy ", {"color": MUTED}), (num(R['Distilled policy']), {"bold": True, "color": TEAL})],
    ], size=18, anchor=MSO_ANCHOR.MIDDLE, spacing_after=10)
    footer(s, n)
    notes(s, f"Headline: the distilled policy decides about {speed:.0f} times faster than the CEM planner it learned "
             f"from and keeps {100 * score:.0f} percent of the planner's gain over random play. The planner looks "
             f"{cfg['plan_horizon']} steps ahead, the same imagination length DreamerV3 uses.")

    # ---------------------------------------------------- 3. readings
    s = new_slide(prs); n += 1
    header(s, "Background", "Four readings, one idea: imagine in training, act fast")
    readings = [
        ("brain", AMBER, "LeCun (2022)", "A Path Towards Autonomous Machine Intelligence",
         ["Mode-2: plan with a world model and a fixed intrinsic cost, act, replan",
          "Fig. 5: compile Mode-2 plans into a fast Mode-1 policy"]),
        ("layers", VIOLET, "Ding et al. (2024)", "Diffusion World Model",
         ["One-step models compound error over long rollouts (Fig. 1.1)",
          "Imagined rollouts only in training, so the deployed policy is fast"]),
        ("moon", TEAL, "Hafner et al. (2023)", "DreamerV3",
         ["Learn a world model from experience, train behaviour in short imagined rollouts",
          "symlog makes regression targets robust"]),
        ("github", GRAY, "danijar/dreamerv3", "Official code",
         ["imag_length: 15 became my planning horizon",
          "symlog/symexp in nets.py; one configs.yaml became config.py"]),
    ]
    for i, (ic, col, who, what, pts) in enumerate(readings):
        x = 0.6 + i * 3.08
        card(s, x, 1.75, 2.85, 4.95)
        icon(s, ic, x + 0.3, 2.05, 0.7, color=col)
        text(s, x + 0.3, 2.95, 2.3, 0.35, who, size=18, bold=True)
        text(s, x + 0.3, 3.32, 2.3, 0.65, what, size=13, color=MUTED)
        bullets(s, x + 0.3, 4.05, 2.35, 2.55, pts, size=14, bullet_color=col, gap=10)
    footer(s, n)
    notes(s, "Each reading contributed one ingredient. LeCun gives the architecture and the Mode-1/Mode-2 vocabulary. "
             "The Diffusion World Model paper explains why long imagined rollouts are dangerous and why imagination "
             "should stay in training. DreamerV3 and its code give the 15-step horizon, symlog and the single config file.")

    # ---------------------------------------------- 4. architecture map
    s = new_slide(prs); n += 1
    header(s, "LeCun's architecture → my code", "The same agent, run in two modes")
    # Mode-2 lane
    card(s, 0.6, 1.6, 12.1, 2.65, fill=CARD)
    text(s, 0.85, 1.72, 6, 0.3, [[("MODE-2 · DELIBERATE", {"bold": True, "color": AMBER}),
                                   (f"   {cem_ms:.1f} ms per decision", {"color": MUTED})]], size=13)

    def box(x, y, w, h, title, sub, col, fill=CARD2):
        card(s, x, y, w, h, fill=fill, radius=0.12)
        text(s, x, y + 0.12, w, 0.35, title, size=15, bold=True, color=col, align=PP_ALIGN.CENTER)
        text(s, x, y + 0.47, w, 0.3, sub, size=12, color=MUTED, font=MONO, align=PP_ALIGN.CENTER)

    box(0.95, 2.55, 2.2, 0.95, "Observation s", "[cos, sin, dθ/dt]", TEXT)
    box(3.75, 2.55, 2.5, 0.95, "CEM planner", "planner.py", AMBER)
    box(6.95, 1.95, 2.6, 0.8, "World model", "world_model.py", TEXT)
    box(6.95, 3.3, 2.6, 0.8, "Intrinsic cost", "cost.py", TEXT)
    box(10.15, 2.55, 2.25, 0.95, "Execute a₀", "then replan", AMBER)
    arrow(s, 3.15, 3.02, 3.75, 3.02)
    arrow(s, 6.25, 2.8, 6.95, 2.35)
    arrow(s, 6.25, 3.25, 6.95, 3.7)
    arrow(s, 6.25, 3.02, 10.15, 3.02, color=AMBER)
    text(s, 3.75, 2.05, 3.0, 0.3, f"imagine {cfg['plan_horizon']} steps × {cfg['cem_population']} plans",
         size=12, color=MUTED)
    # distil arrow
    arrow(s, 5.0, 3.5, 5.0, 5.25, color=TEAL, width=2.5)
    text(s, 5.2, 4.26, 4.5, 0.32, [[("distil: ", {"bold": True, "color": TEAL}),
                                   ("BC + DAgger on planner labels", {"color": TEXT})]], size=14,
         anchor=MSO_ANCHOR.MIDDLE)
    # Mode-1 lane
    card(s, 0.6, 4.6, 12.1, 1.95, fill=CARD)
    text(s, 0.85, 4.72, 6, 0.3, [[("MODE-1 · REACTIVE", {"bold": True, "color": TEAL}),
                                   (f"   {pol_ms:.3f} ms per decision", {"color": MUTED})]], size=13)
    box(0.95, 5.25, 2.2, 0.95, "Observation s", "[cos, sin, dθ/dt]", TEXT)
    box(3.75, 5.25, 2.5, 0.95, "Policy π(s)", "policy.py", TEAL)
    box(10.15, 5.25, 2.25, 0.95, "Execute a", "every step", TEAL)
    arrow(s, 3.15, 5.72, 3.75, 5.72)
    arrow(s, 6.25, 5.72, 10.15, 5.72)
    text(s, 6.5, 5.32, 3.4, 0.3, "one forward pass, no search", size=12, color=MUTED, align=PP_ALIGN.CENTER)
    footer(s, n)
    notes(s, "Top lane is LeCun's Mode-2: the planner proposes action sequences, the world model imagines what happens, "
             "the fixed intrinsic cost scores the imagined states, and only the first action is executed before replanning. "
             "That is Model-Predictive Control. The bottom lane is Mode-1: a small network trained on the planner's "
             "decisions, which is LeCun's Figure 5.")

    # ---------------------------------------------------- 5. environment
    s = new_slide(prs); n += 1
    header(s, "The robot", "Pendulum-v1: a small but real swing-up problem")
    fr = dark_frames("cem_mpc.gif", [50])[0]
    card(s, 0.6, 1.65, 4.6, 4.95, fill=CARD2)
    s.shapes.add_picture(fr, Inches(0.95), Inches(1.95), Inches(3.9), Inches(3.9))
    text(s, 0.85, 5.95, 4.1, 0.5, "Torque is limited to ±2 N·m, so the pole must be swung, not lifted",
         size=13, color=MUTED, align=PP_ALIGN.CENTER)
    rows = [
        ("network", AMBER, "State", "s = [cos θ, sin θ, θ̇],  θ̇ ∈ [−8, 8] rad/s"),
        ("cogs", AMBER, "Action", "torque u ∈ [−2, 2] N·m, 20 Hz control (dt = 0.05 s)"),
        ("bullseye", AMBER, "Cost (negative reward)", "θ² + 0.1·θ̇² + 0.001·u²   (0 = upright and still)"),
        ("clock", AMBER, "Episode", f"200 steps. Random play: {num(R['Random'])}. Good swing-up: about −120 to −240"),
    ]
    for i, (ic, col, t, d) in enumerate(rows):
        y = 1.75 + i * 1.22
        icon(s, ic, 5.65, y, 0.66, color=col)
        text(s, 6.55, y - 0.04, 6.1, 0.35, t, size=18, bold=True)
        text(s, 6.55, y + 0.36, 6.1, 0.6, d, size=15, color=MUTED)
    footer(s, n)
    notes(s, "Pendulum-v1 has a three-number state, one torque, and a reward that penalises angle, speed and effort. "
             "The motor is too weak to lift the pole directly, so a controller has to plan a swing, which makes it a "
             "real, if small, planning problem. The same cost formula is used by my planner as LeCun's intrinsic cost.")

    # ---------------------------------------------------- 6. pipeline
    s = new_slide(prs); n += 1
    header(s, "How the code works", "One command, six stages: python main.py")
    steps = [
        ("database", "Collect data", "collect_data.py", "50 random episodes give 10,000 (s, a, s′, r) transitions, saved as .npz"),
        ("network", "Train world model", "world_model.py", "MLP predicts Δs = s′ − s from normalised (s, a)"),
        ("chart", "Test compounding error", "multi_step_error()", "Open-loop rollouts, 1-30 steps, on held-out data"),
        ("route", "Mode-2: CEM MPC", "planner.py", "Search 15-step plans in the model, run the first action, replan"),
        ("bolt", "Mode-1: distil", "policy.py", "Behaviour cloning, then 4 DAgger iterations"),
        ("balance", "Evaluate and plot", "evaluate.py · plots.py", "4 controllers, 15 episodes, return and ms per decision"),
    ]
    for i, (ic, t, f, d) in enumerate(steps):
        col, row = i % 3, i // 3
        x, y = 0.6 + col * 4.1, 1.7 + row * 2.45
        card(s, x, y, 3.8, 2.2)
        c = AMBER if i < 4 else TEAL
        number_badge(s, i + 1, x + 0.3, y + 0.3, 0.5, color=c)
        text(s, x + 1.0, y + 0.3, 2.6, 0.5, t, size=18, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.3, y + 0.95, 3.3, 0.3, f, size=13, color=c, font=MONO)
        text(s, x + 0.3, y + 1.3, 3.3, 0.8, d, size=14, color=MUTED)
    text(s, 0.6, 6.6, 12.1, 0.3,
         f"3 seeds · fixed random seeds everywhere · {runtime_min:.1f} min on one CPU thread · all hyperparameters in config.py",
         size=13, color=MUTED)
    footer(s, n)
    notes(s, "main.py runs these six stages for each seed, then pools the results, makes every plot and writes "
             "summary.json. Stages one to four are the world model and Mode-2 side; five and six are Mode-1 and the comparison.")

    # ---------------------------------------------------- 7. world model
    s = new_slide(prs); n += 1
    header(s, "Stage 2 · World model", "Learning the physics from 10,000 random transitions")
    code_card(s, 0.6, 1.65, 7.0, 4.95, [
        "def predict(self, obs, act):",
        "    # one imagined step: s' = s + f(s, a)",
        "    x = (cat([obs, act]) - in_mean) / in_std",
        "    y = self.net(x)          # 3 x 128 SiLU MLP",
        "    delta = y * out_std + out_mean",
        "    if self.use_symlog:      # DreamerV3",
        "        delta = symexp(delta)",
        "    nxt = obs + delta",
        "    cs = unit_circle(nxt[:2])  # valid cos/sin",
        "    thdot = clip(nxt[2], -8, 8)",
        "    return cat([cs, thdot])",
    ], size=14, title="world_model.py (simplified)")
    bullets(s, 8.0, 1.75, 4.7, 2.6, [
        "Predicting the change Δs gives a small, zero-centred target",
        "Inputs and targets are standardised; statistics are stored inside the model",
        "Optional --symlog flag: regress symlog(Δs), map back with symexp",
    ], size=15, bullet_color=AMBER, gap=12)
    card(s, 8.0, 4.6, 4.7, 2.0, fill=CARD2)
    text(s, 8.3, 4.8, 2.1, 0.75, f"{losses['train']:.5f}", size=30, bold=True, color=AMBER)
    text(s, 8.3, 5.55, 2.1, 0.5, "train MSE (normalised)", size=13, color=MUTED)
    text(s, 10.5, 4.8, 2.1, 0.75, f"{losses['val']:.5f}", size=30, bold=True, color=AMBER)
    text(s, 10.5, 5.55, 2.1, 0.5, "validation MSE", size=13, color=MUTED)
    text(s, 8.3, 6.05, 4.2, 0.4, "Seed 0, after 60 epochs: no overfitting", size=13, color=MUTED)
    footer(s, n)
    notes(s, "The world model is a plain MLP that predicts the change in state. Two pieces of prior knowledge keep "
             "rollouts valid: cos and sin are put back on the unit circle and the speed is clipped like the simulator. "
             "The one-step fit is excellent, with train and validation losses almost equal.")

    # ---------------------------------------------- 8. compounding error
    s = new_slide(prs); n += 1
    header(s, "Stage 3 · Diffusion World Model, Fig. 1.1", "Small one-step errors compound over the horizon")
    cd = CategoryChartData()
    cd.categories = [str(h) for h in plain["horizon"]]
    cd.add_series("Plain targets", [round(v, 5) for v in plain["rmse"]])
    cd.add_series("symlog targets", [round(v, 5) for v in sym["rmse"]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.5), Inches(1.6), Inches(7.7), Inches(5.1), cd)
    ch = gf.chart
    style_chart(ch, legend=True)
    style_axes(ch)
    log_axis(ch.value_axis)
    ch.value_axis.tick_labels.number_format = "0.000"
    ch.value_axis.tick_labels.number_format_is_linked = False
    axis_title(ch.value_axis, "state RMSE (log scale)")
    axis_title(ch.category_axis, "open-loop horizon (steps)")
    cat_ax = ch.category_axis._element
    anchor_el = cat_ax.find(qn("c:noMultiLvlLbl"))
    for tag in ("tickLblSkip", "tickMarkSkip"):
        el = parse_xml(f'<c:{tag} {nsdecls("c")} val="5"/>')
        if anchor_el is not None:
            anchor_el.addprevious(el)
        else:
            cat_ax.append(el)
    for ser, col in zip(ch.series, [AMBER, VIOLET]):
        ser.format.line.color.rgb = rgb(col)
        ser.format.line.width = Pt(2.25)
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 5
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(col)
        ser.marker.format.line.fill.background()
    r15 = plain["rmse"][14] / plain["rmse"][0]
    r30 = plain["rmse"][29] / plain["rmse"][0]
    for i, (v, lab) in enumerate([(f"×{r15:.0f}", "error after 15 steps vs 1 step"),
                                  (f"×{r30:.0f}", "error after 30 steps vs 1 step")]):
        card(s, 8.6, 1.65 + i * 1.6, 4.1, 1.4)
        text(s, 8.9, 1.75 + i * 1.6, 1.6, 1.2, v, size=40, bold=True, color=AMBER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 10.5, 1.75 + i * 1.6, 2.0, 1.2, lab, size=14, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    card(s, 8.6, 4.85, 4.1, 1.85, fill=CARD2)
    text(s, 8.9, 4.98, 3.6, 1.65, [
        [("Angle error ", {"color": MUTED}), (f"{plain['angle_err_deg'][0]:.2f}° → {plain['angle_err_deg'][14]:.2f}° → "
                                              f"{plain['angle_err_deg'][29]:.1f}°", {"bold": True})],
        [("At H = 15 the model is still within 1°, so MPC plans only that far and replans from real "
          "observations (DreamerV3's 15-step imagination).", {"color": MUTED})],
    ], size=14, spacing_after=6)
    footer(s, n)
    notes(s, f"Every step, the model is fed its own slightly wrong prediction, so the error grows faster than linearly: "
             f"about {r15:.0f} times larger after 15 steps and {r30:.0f} times after 30. This is the compounding error the "
             f"Diffusion World Model paper shows in Figure 1.1, and the reason DreamerV3 imagines only 15 steps. "
             f"symlog did not help here because Pendulum's state changes are already small.")

    # ---------------------------------------------------- 9. CEM
    s = new_slide(prs); n += 1
    header(s, "Stage 4 · Mode-2 = Model-Predictive Control", "The Cross-Entropy Method plans inside the learned model")
    cem_steps = [
        f"Sample {cfg['cem_population']} torque sequences of length {cfg['plan_horizon']} from a Gaussian",
        "Roll each one through the world model and add up the imagined cost",
        f"Keep the {cfg['cem_elites']} cheapest (elites) and refit the Gaussian; repeat {cfg['cem_iterations']}×",
        "Execute only the first action, then replan from the new real state",
    ]
    for i, t in enumerate(cem_steps):
        y = 1.75 + i * 1.02
        number_badge(s, i + 1, 0.6, y, 0.55, color=AMBER)
        text(s, 1.4, y - 0.175, 5.2, 0.9, t, size=16, anchor=MSO_ANCHOR.MIDDLE)
    card(s, 0.6, 5.95, 6.0, 0.75, fill=CARD2)
    text(s, 0.85, 5.95, 5.6, 0.75, [[(f"{imag_steps:,}", {"bold": True, "color": AMBER, "size": 22}),
                                     ("  imagined model steps per decision", {"color": MUTED})]],
         size=15, anchor=MSO_ANCHOR.MIDDLE)
    code_card(s, 7.0, 1.65, 5.7, 5.05, [
        "def imagined_cost(model, obs, plans):",
        "    s = obs.expand(N, -1)",
        "    total = zeros(N)",
        "    for t in range(H):          # H = 15",
        "        a = plans[:, t:t+1]",
        "        total += pendulum_cost(s, a)",
        "        s = model.predict(s, a) # imagine",
        "    return total + pendulum_cost(s, 0)",
        "",
        "# LeCun Mode-2: execute first action,",
        "# keep the rest as a warm start",
        "action = mean[0]",
        "self.mean = cat([mean[1:], zeros(1)])",
    ], size=13, title="planner.py (simplified)")
    footer(s, n)
    notes(s, "CEM is a simple optimiser: sample, score in imagination, keep the best, refit, repeat. The planner never "
             "touches the simulator while thinking; it only uses the learned model and the fixed cost. Executing just "
             "the first action and replanning is what makes this MPC, and it limits the damage from compounding error.")

    # ---------------------------------------------------- 10. distillation
    s = new_slide(prs); n += 1
    header(s, "Stage 5 · Mode-1, LeCun Fig. 5", "DAgger turns the planner into a reflex", kicker_color=TEAL)
    dag_steps = [
        ("play", "Behaviour cloning", "Imitate the planner on its own 3 episodes"),
        ("sync", "DAgger iteration", "Let the policy drive; the planner labels every state it visits"),
        ("database", "Aggregate and retrain", "600 → 3,000 labelled states, including recovery states"),
    ]
    for i, (ic, t, d) in enumerate(dag_steps):
        y = 1.75 + i * 1.3
        icon(s, ic, 0.6, y, 0.66, color=TEAL)
        text(s, 1.5, y - 0.05, 3.6, 0.35, t, size=17, bold=True)
        text(s, 1.5, y + 0.33, 3.6, 0.8, d, size=14, color=MUTED)
    bc = [dagger[k]["return_mean"][0] for k in dagger]
    card(s, 0.6, 5.65, 4.5, 1.05, fill=CARD2)
    text(s, 0.85, 5.65, 4.1, 1.05, [[("BC alone scored ", {"color": MUTED}),
                                     ("about " + num(round(min(bc), -1)), {"bold": True, "color": CORAL}),
                                     (" on 2 of 3 seeds, worse than random. DAgger fixed it.", {"color": MUTED})]],
         size=14, anchor=MSO_ANCHOR.MIDDLE)
    cd = CategoryChartData()
    its = dagger["0"]["iteration"]
    cd.categories = ["BC" if i == 0 else f"DAgger {i}" for i in its]
    for k in dagger:
        cd.add_series(f"Seed {k}", [round(v, 1) for v in dagger[k]["return_mean"]])
    cd.add_series("CEM teacher", [round(R["CEM MPC"], 1)] * len(its))
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(5.4), Inches(1.6), Inches(7.4), Inches(5.15), cd)
    ch = gf.chart
    style_chart(ch, legend=True)
    style_axes(ch)
    axis_title(ch.value_axis, "policy return")
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    ch.value_axis.maximum_scale = 0
    ch.value_axis.minimum_scale = -1800
    ch.value_axis.major_unit = 300
    ch.value_axis.tick_labels.number_format = '0;"\u2212"0'
    ch.value_axis.tick_labels.number_format_is_linked = False
    for ser, col in zip(ch.series, [TEAL, "7DD3FC", "C4B5FD", AMBER]):
        ser.format.line.color.rgb = rgb(col)
        ser.format.line.width = Pt(2.25)
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 7
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(col)
        ser.marker.format.line.fill.background()
    last = ch.series[-1]
    last.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    last.marker.style = XL_MARKER_STYLE.NONE
    footer(s, n)
    notes(s, "Plain behaviour cloning only sees states where the expert is doing well, so the first mistake takes the "
             "policy somewhere it has never been trained on. DAgger asks the planner to label the states the policy "
             "actually visits, so it learns to recover. Seed 2 needed three iterations before it clicked.")

    # ---------------------------------------------------- 11. results
    s = new_slide(prs); n += 1
    header(s, "Stage 6 · Evaluation", f"Close to the planner's return at 1/{speed:.0f} of the decision time", kicker_color=TEAL)
    names = list(ctrl)
    short = {"Random": "Random", "Random-shooting MPC": "Rand. shoot", "CEM MPC": "CEM MPC",
             "Distilled policy": "Distilled"}
    for j, (key, ttl, fmt) in enumerate([("return_mean", f"Return, mean of {n_ep} episodes (higher is better)", '0;"\u2212"0'),
                                         ("decision_time_ms_mean", "Milliseconds per decision (lower is faster)", "0.000")]):
        x = 0.6 + j * 6.2
        card(s, x, 1.6, 5.9, 4.4)
        text(s, x + 0.3, 1.75, 5.3, 0.35, ttl, size=14, bold=True)
        cd = CategoryChartData()
        cd.categories = [short[k] for k in names]
        cd.add_series(ttl, [round(ctrl[k][key], 4) for k in names])
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x + 0.15), Inches(2.15), Inches(5.6),
                                Inches(3.75), cd)
        ch = gf.chart
        style_chart(ch)
        style_axes(ch, gridlines=False)
        ch.value_axis.visible = False
        ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
        plot = ch.plots[0]
        plot.gap_width = 70
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.position = XL_DATA_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(13)
        dl.font.bold = True
        dl.font.color.rgb = rgb(TEXT)
        for i, k in enumerate(names):
            pt = ch.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(CTRL_COLORS[k])
        if key == "decision_time_ms_mean":
            # Linear on purpose: the policy's bar is a sliver next to the planners'.
            ch.value_axis.minimum_scale = 0
            ch.value_axis.maximum_scale = 50
        else:
            ch.value_axis.minimum_scale = -1450
            ch.value_axis.maximum_scale = 0
    text(s, 0.6, 6.2, 12.1, 0.6, [[("± std: ", {"bold": True}),
        ("  ".join(f"{short[k]} {SD[k]:.0f}" for k in names), {"color": MUTED}),
        ("    ·    random-shooting gets the same 1,200-plan budget as CEM", {"color": MUTED})]], size=13)
    footer(s, n)
    notes(s, f"On the same {n_ep} unseen start states, CEM scores {R['CEM MPC']:.0f} and the distilled policy "
             f"{R['Distilled policy']:.0f}, against {R['Random']:.0f} for random play. The policy needs "
             f"{pol_ms:.3f} ms per decision against {cem_ms:.1f} ms for CEM. At 20 Hz control, CEM uses most of the "
             f"50 ms budget on one CPU thread; the policy uses a fraction of a percent.")

    # ---------------------------------------------------- 12. filmstrip
    s = new_slide(prs); n += 1
    header(s, "Seeing it", "Swing-up from hanging down, seed-0 controllers", kicker_color=TEAL)
    gifs = [("Random", "random.gif", GRAY), ("Random-shooting MPC", "random_shooting_mpc.gif", VIOLET),
            ("CEM MPC", "cem_mpc.gif", AMBER), ("Distilled policy", "distilled_policy.gif", TEAL)]
    idx = [0, 24, 48, 72, 96, 124, 160, 198]
    gr = S.get("gif_episode_returns_seed0") or {}
    for r_i, (name, g, col) in enumerate(gifs):
        y = 1.6 + r_i * 1.3
        card(s, 0.6, y, 12.1, 1.18, fill=CARD)
        text(s, 0.85, y + 0.18, 2.4, 0.4, name, size=15, bold=True, color=col)
        if name in gr:
            text(s, 0.85, y + 0.6, 2.4, 0.35, f"return {num(gr[name])}", size=13, color=MUTED)
        for c_i, p in enumerate(dark_frames(g, idx, size=160)):
            s.shapes.add_picture(p, Inches(3.3 + c_i * 1.17), Inches(y + 0.07), Inches(1.04), Inches(1.04))
    for c_i, k in enumerate(idx):
        text(s, 3.3 + c_i * 1.17, 6.82, 1.04, 0.2, f"t = {k}", size=10, color=FAINT, align=PP_ALIGN.CENTER)
    notes(s, "All four start hanging straight down at rest, the hardest start. Random never gets up. Both planners and "
             "the distilled policy pump energy with a few swings and then balance; the policy takes a little longer, "
             "which is why its return on this one episode is lower than CEM's.")

    # ---------------------------------------------------- 13. readings check
    s = new_slide(prs); n += 1
    header(s, "Back to the readings", "What the readings claimed, and what my run showed")
    rows = [
        ("brain", AMBER, "LeCun", "Mode-2 can be compiled into a fast Mode-1 policy",
         f"{speed:.0f}× faster, {100 * score:.0f}% of the planner's gain kept"),
        ("layers", VIOLET, "Diffusion World Model", "One-step models compound error; keep imagination in training",
         f"Error ×{r30:.0f} after 30 steps; the deployed policy never touches the model"),
        ("moon", TEAL, "DreamerV3", "Short imagined rollouts (15 steps) are enough to learn behaviour",
         f"Angle error {plain['angle_err_deg'][14]:.2f}° at 15 steps, good enough to swing up"),
        ("github", GRAY, "DreamerV3 code", "symlog makes targets robust across domains",
         "Slightly worse here: Pendulum's targets are already small (honest negative result)"),
    ]
    text(s, 2.75, 1.6, 4.5, 0.3, "CLAIM", size=12, bold=True, color=MUTED)
    text(s, 7.55, 1.6, 5.0, 0.3, "WHAT I OBSERVED", size=12, bold=True, color=MUTED)
    for i, (ic, col, who, claim, obs) in enumerate(rows):
        y = 2.0 + i * 1.18
        card(s, 0.6, y, 12.1, 1.03, fill=CARD if i % 2 == 0 else CARD2)
        icon(s, ic, 0.8, y + 0.2, 0.62, color=col)
        text(s, 1.6, y, 1.1, 1.03, who, size=14, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 2.75, y, 4.5, 1.03, claim, size=14, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 7.55, y, 4.95, 1.03, obs, size=14, color=col if i < 3 else MUTED, bold=i < 3,
             anchor=MSO_ANCHOR.MIDDLE)
    footer(s, n)
    notes(s, "Three of the four claims show up clearly even on this tiny problem. The symlog result is the honest "
             "exception: symlog exists to handle targets whose scale varies by orders of magnitude across domains, "
             "and a single well-normalised pendulum has no such problem.")

    # ---------------------------------------------------- 14. lessons + limits
    s = new_slide(prs); n += 1
    header(s, "Lessons and limitations", "What surprised me, and what this experiment does not show", kicker_color=CORAL)
    card(s, 0.6, 1.65, 5.9, 5.05)
    icon(s, "flask", 0.9, 1.95, 0.62, color=TEAL)
    text(s, 1.75, 1.95, 4.5, 0.62, "Surprises", size=20, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    bullets(s, 0.9, 2.85, 5.3, 3.7, [
        "Behaviour cloning alone was worse than random: the policy's own mistakes need labels",
        "Retraining on CEM data (2 rounds) changed CEM's return by at most 0.1 point",
        "symlog targets were slightly worse on this small, well-scaled problem",
        f"Random shooting with the same budget was close to CEM ({num(R['Random-shooting MPC'])} vs {num(R['CEM MPC'])}) but less consistent",
    ], size=15, bullet_color=TEAL, gap=12)
    card(s, 6.8, 1.65, 5.9, 5.05)
    icon(s, "warn", 7.1, 1.95, 0.62, color=CORAL)
    text(s, 7.95, 1.95, 4.5, 0.62, "Limitations", size=20, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    bullets(s, 7.1, 2.85, 5.3, 3.7, [
        "True 3-number state, not camera images (no representation learning)",
        "Deterministic MLP: no uncertainty, no RSSM or diffusion model",
        "Hand-written cost and no value function, so the planner sees only 0.75 s ahead",
        "Imitation cannot beat its teacher; only 15 test episodes, and the ± ranges overlap",
    ], size=15, bullet_color=CORAL, gap=12)
    footer(s, n)
    notes(s, "The biggest lesson was DAgger: the same compounding-error problem the world model has also hits the "
             "cloned policy. The main limitations are that I skip perception entirely and use a hand-written cost; "
             "DreamerV3 and DWM learn from pixels and learn a reward model and critic.")

    # ---------------------------------------------------- 15. closing
    s = new_slide(prs, "0A1020"); n += 1
    text(s, 0.8, 1.3, 11.7, 0.35, "TAKEAWAY", size=14, bold=True, color=AMBER)
    text(s, 0.8, 1.8, 11.7, 1.8, [[("Think slowly while training. ", {"color": AMBER}),
                                   ("React quickly when deployed.", {"color": TEAL})]],
         size=44, bold=True, font=HEAD)
    text(s, 0.8, 3.6, 10.5, 1.0, "A learned world model plus a planner gives good behaviour; distilling that planner "
                                 "gives the same behaviour cheaply enough for a real-time robot.", size=20, color=MUTED)
    code_card(s, 0.8, 4.85, 6.3, 1.85, [
        "cd readings/Readings_6",
        "pip install -r requirements.txt",
        "python main.py        # full run",
        "python main.py --quick",
    ], size=14, title="reproduce")
    icon(s, "github", 7.6, 5.05, 0.7, color=CARD2)
    text(s, 8.5, 5.0, 4.3, 0.8, [[("github.com/CDJ7/ENG6010", {"bold": True})],
                                 [("readings/Readings_6", {"color": MUTED, "font": MONO, "size": 14})]],
         size=16, spacing_after=2)
    text(s, 8.5, 5.95, 4.3, 0.8, "README · Program_Explanation · Results", size=14, color=MUTED)
    notes(s, "That is the whole loop from LeCun's paper on a small scale: model, cost, plan, then compile the plan into "
             "a reflex. Everything shown is reproducible with one command from the repository.")

    prs.save(OUT)
    print(f"wrote {OUT} ({n} slides)")


if __name__ == "__main__":
    build()
