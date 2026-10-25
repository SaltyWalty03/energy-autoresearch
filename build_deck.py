"""
Build a presentation-ready slide deck for the XLE energy-autoresearch capstone.
Outputs: presentation_assets/deck.pptx
"""
import sys, io, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path

# python-pptx
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from prepare import load_and_split_data
from model import build_model

OUT = Path("presentation_assets")
OUT.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY    = "#0D1B2A"    # slide background
PANEL   = "#1B2B3E"    # chart area background
WHITE   = "#F0F4F8"    # primary text
MUTED   = "#8BA0B4"    # secondary text / grid
CYAN    = "#38BDF8"    # keep / improvement
GREEN   = "#4ADE80"    # final / best
RED     = "#F87171"    # discard
AMBER   = "#FCD34D"    # baseline
PURPLE  = "#C084FC"    # XGB
GRID    = "#22374D"    # chart gridlines

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def mpl_hex(h):
    """Return matplotlib-compatible colour from hex string."""
    return h

# ── Matplotlib global style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":     NAVY,
    "axes.facecolor":       PANEL,
    "axes.edgecolor":       GRID,
    "axes.labelcolor":      MUTED,
    "axes.titlecolor":      WHITE,
    "text.color":           WHITE,
    "xtick.color":          MUTED,
    "ytick.color":          MUTED,
    "grid.color":           GRID,
    "grid.linewidth":       0.7,
    "legend.facecolor":     PANEL,
    "legend.edgecolor":     GRID,
    "legend.labelcolor":    WHITE,
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     False,
    "axes.spines.bottom":   False,
})

SLIDE_W, SLIDE_H = 13.33, 7.5   # inches (widescreen 16:9)

# ─────────────────────────────────────────────────────────────────────────────
# Fit the model once for reuse
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data and fitting model...")
train, val = load_and_split_data()
model = build_model()
model.fit(train[["ret_lag"]], train["target"])
preds = model.predict(val[["ret_lag"]])

strat_rets = np.sign(preds) * val["target"].values
bh_rets    = val["target"].values
equity_s   = np.exp(np.cumsum(strat_rets))
equity_bh  = np.exp(np.cumsum(bh_rets))
dates      = val.index
roll_max   = np.maximum.accumulate(equity_s)
dd         = equity_s / roll_max - 1
sharpe_val = (np.mean(strat_rets) / (np.std(strat_rets) + 1e-9)) * np.sqrt(252)
total_ret  = equity_s[-1] - 1
max_dd     = dd.min()
win_rate   = (strat_rets > 0).mean()

print(f"  Val Sharpe: {sharpe_val:.2f} | Return: {total_ret:+.1%} | Max DD: {max_dd:.1%}")

def save_fig(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    return path

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Sharpe Comparison (all named experiments)
# ─────────────────────────────────────────────────────────────────────────────
experiments = [
    # label, sharpe, kind, session
    ("LinearRegression — true baseline",           0.8584, "baseline", "Session 1"),
    ("RetFeatures + Ridge",                        0.0653, "discard",  "Session 1"),
    ("GradientBoostingRegressor",                  0.1524, "discard",  "Session 1"),
    ("Rolling features + Ridge",                   0.4703, "discard",  "Session 1"),
    ("Rolling features + LogisticRegression",      0.9581, "keep",     "Session 1"),
    ("LogisticReg + extra lags / RSI",             0.8824, "discard",  "Session 1"),
    ("RF  depth=3, msl=30",                        1.2623, "keep",     "Session 1"),
    ("RF  depth=2, msl=25, n=200",                 1.3092, "keep",     "Session 1"),
    ("MLP (128,) — unstable",                      1.1365, "discard",  "Session 1"),
    ("RF  n=300, depth=2, msl=20",                 1.3717, "keep",     "Session 1"),
    ("RF  n=200, depth=2, msl=22  ← S1 best",     1.3954, "keep",     "Session 1"),
    ("Direct Sharpe optimizer — overfit",          0.0267, "discard",  "Session 1"),
    ("OLS  24 macro / weather features",          -0.7390, "discard",  "Macro"),
    ("OLS  10 selected features",                  0.0390, "discard",  "Macro"),
    ("Lasso / Ridge",                              0.6130, "discard",  "Macro"),
    ("RF  on macro features — severe overfit",    -0.7050, "discard",  "Macro"),
    ("Best single signal (trend_energy_stocks)",   1.6870, "keep",     "Macro"),
    ("Normalised signal composite",                0.6630, "discard",  "Macro"),
    ("Walk-forward: signal + Lasso ensemble",      1.0420, "keep",     "Macro"),
    ("Sweep · run_001  baseline",                  2.4041, "baseline", "Sweep"),
    ("Sweep · run_002  depth=3",                   2.1359, "discard",  "Sweep"),
    ("Sweep · run_003  depth=4",                   2.4652, "keep",     "Sweep"),
    ("Sweep · run_004  depth=1 stumps",            1.2735, "discard",  "Sweep"),
    ("Sweep · run_005  msl=10",                    2.3837, "discard",  "Sweep"),
    ("Sweep · run_006  msl=50",                    1.4882, "discard",  "Sweep"),
    ("Sweep · run_007  msl=100",                   1.5910, "discard",  "Sweep"),
    ("Sweep · run_008  n=100",                     2.4742, "keep",     "Sweep"),
    ("Sweep · run_009  n=600  ← best sweep",       2.4793, "keep",     "Sweep"),
    ("Sweep · run_010  window=10",                 2.3446, "discard",  "Sweep"),
    ("Sweep · run_011  window=40",                 1.9068, "discard",  "Sweep"),
    ("Sweep · run_012  WTI thresh=3 %",            2.2201, "discard",  "Sweep"),
    ("Sweep · run_013  WTI thresh=1 %",            1.4536, "discard",  "Sweep"),
    ("Sweep · run_014  train window=2 yr",         1.2470, "discard",  "Sweep"),
    ("Sweep · run_015  depth=3 + msl=15 + w=10",  2.0270, "discard",  "Sweep"),
    ("S2 · WTI filter ≥ 3 %",                     2.0219, "keep",     "Session 2"),
    ("S2 · WTI filter ≥ 2 %  ← best committed",  2.2119, "final",    "Session 2"),
    ("S2 · WTI filter ≥ 1.5 %",                   2.0065, "discard",  "Session 2"),
    ("S2 · WTI as RF input feature",              1.3403, "discard",  "Session 2"),
    ("S2 · Contrarian on shock days",             1.9532, "discard",  "Session 2"),
    ("Post · RF smoke test",                       2.5903, "keep",     "Post"),
    ("Post · XGBoost baseline  (WF −0.40)",        0.5408, "xgb",      "Post"),
    ("Post · XGBoost depth=2   (WF −0.68)",        0.2908, "xgb",      "Post"),
    ("Post · RF + walk-forward  ← final",          2.5949, "final",    "Post"),
]

cmap = {"baseline": AMBER, "discard": RED, "keep": CYAN,
        "final": GREEN, "xgb": PURPLE}

labels  = [e[0] for e in experiments]
sharpes = [e[1] for e in experiments]
kinds   = [e[2] for e in experiments]
sessions = [e[3] for e in experiments]
colors  = [cmap[k] for k in kinds]

# Section divider y positions
session_breaks = []
prev = None
for i, sg in enumerate(sessions):
    if sg != prev and i != 0:
        session_breaks.append(i - 0.5)
    prev = sg

fig, ax = plt.subplots(figsize=(SLIDE_W - 1, 13), facecolor=NAVY)
ax.set_facecolor(PANEL)

for yb in session_breaks:
    ax.axhline(yb, color=GRID, linewidth=1.2, linestyle="-", zorder=1)

bars = ax.barh(range(len(labels)), sharpes, color=colors,
               edgecolor=NAVY, linewidth=0.4, height=0.72, zorder=2)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7.4, color=WHITE)
ax.axvline(0,      color=MUTED, linewidth=0.8, linestyle="--", zorder=3)
ax.axvline(0.8584, color=AMBER, linewidth=1.0, linestyle=":", alpha=0.55, zorder=3)
ax.axvline(2.5949, color=GREEN, linewidth=1.0, linestyle=":", alpha=0.55, zorder=3)

for bar, v in zip(bars, sharpes):
    pad = 0.04
    ax.text(v + pad if v >= 0 else v - pad,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}", va="center",
            ha="left" if v >= 0 else "right",
            fontsize=6.5, color=WHITE, zorder=4)

legend_els = [
    mpatches.Patch(facecolor=GREEN,  label="Final / best committed"),
    mpatches.Patch(facecolor=CYAN,   label="Kept (improvement)"),
    mpatches.Patch(facecolor=AMBER,  label="Baseline reference"),
    mpatches.Patch(facecolor=RED,    label="Discarded"),
    mpatches.Patch(facecolor=PURPLE, label="XGBoost (rejected)"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=8,
          framealpha=0.9, edgecolor=GRID)
ax.set_xlabel("Annualised Sharpe Ratio", fontsize=9, color=MUTED)
ax.set_title("All ~60 Experiments — Sharpe Ratio by Configuration",
             fontsize=13, fontweight="bold", color=WHITE, pad=14)
ax.tick_params(axis="y", labelsize=7.4, colors=WHITE, length=0)
ax.tick_params(axis="x", colors=MUTED, labelsize=8)
ax.grid(axis="x", zorder=0)
ax.set_axisbelow(True)

# Section labels on right margin
section_ys = {}
prev = None
for i, sg in enumerate(sessions):
    if sg != prev:
        section_ys[sg] = i
    prev = sg
for sg, y_start in section_ys.items():
    y_end = y_start
    for i2, sg2 in enumerate(sessions):
        if sg2 == sg:
            y_end = i2
    mid = (y_start + y_end) / 2
    ax.text(ax.get_xlim()[1] + 0.06, mid, sg,
            va="center", ha="left", fontsize=7, color=MUTED,
            fontstyle="italic", clip_on=False)

plt.tight_layout(pad=1.2)
chart1 = save_fig(fig, "slide_sharpe_comparison.png")
print("saved slide_sharpe_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Research Progression (milestone line chart)
# ─────────────────────────────────────────────────────────────────────────────
milestones = [
    ("LinReg\nbaseline",             0.8584, AMBER),
    ("LogisticReg\n+ rolling feat",  0.9581, CYAN),
    ("RF\ndepth=2",                  1.2623, CYAN),
    ("RF n=200\nmsl=22 (S1★)",      1.3954, CYAN),
    ("Sweep\nbaseline",              2.4041, AMBER),
    ("Sweep\nn=600",                 2.4793, CYAN),
    ("+ WTI\nfilter 3%",            2.0219, CYAN),
    ("+ WTI\nfilter 2% ★",         2.2119, GREEN),
    ("Post-cleanup\nRF",             2.5903, CYAN),
    ("RF +\nwalk-forward",           2.5949, GREEN),
]
m_x      = list(range(len(milestones)))
m_vals   = [m[1] for m in milestones]
m_colors = [m[2] for m in milestones]
m_labels = [m[0] for m in milestones]

fig, ax = plt.subplots(figsize=(SLIDE_W - 1.5, 4.6), facecolor=NAVY)
ax.set_facecolor(PANEL)

ax.fill_between(m_x, m_vals, 0.8584, where=[v > 0.8584 for v in m_vals],
                alpha=0.12, color=GREEN, interpolate=True)
ax.plot(m_x, m_vals, color=CYAN, linewidth=2.2, zorder=3)

for i, (v, c, lbl) in enumerate(zip(m_vals, m_colors, m_labels)):
    ax.scatter(i, v, color=c, s=90, zorder=5, edgecolors=NAVY, linewidths=1.2)
    offset = 12 if i % 2 == 0 else -18
    ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                xytext=(0, offset), ha="center", fontsize=9,
                color=WHITE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8) if abs(offset) > 12 else None)

ax.axhline(0.8584, color=AMBER, linewidth=1.0, linestyle=":", alpha=0.6,
           label="True baseline  0.86")
ax.axhline(2.5949, color=GREEN, linewidth=1.0, linestyle=":", alpha=0.6,
           label="Final model  2.59")
ax.set_xticks(m_x)
ax.set_xticklabels(m_labels, fontsize=8.5, color=WHITE)
ax.set_ylabel("Annualised Sharpe", fontsize=9, color=MUTED)
ax.set_title("Research Progression — Key Milestones",
             fontsize=13, fontweight="bold", color=WHITE, pad=12)
ax.legend(fontsize=9, framealpha=0.85, loc="upper left")
ax.grid(axis="y", zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0, labelsize=8)
ax.tick_params(axis="y", colors=MUTED, labelsize=8)

plt.tight_layout(pad=1.2)
chart2 = save_fig(fig, "slide_progression.png")
print("saved slide_progression.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Equity Curve
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(SLIDE_W - 1.5, 4.6), facecolor=NAVY)
ax.set_facecolor(PANEL)

ax.plot(dates, equity_s,  color=GREEN, linewidth=2.2,
        label="Final RF (DirectionModel)", zorder=3)
ax.plot(dates, equity_bh, color=MUTED, linewidth=1.2, linestyle="--",
        alpha=0.6, label="Buy & Hold XLE", zorder=2)
ax.axhline(1.0, color=GRID, linewidth=0.8, linestyle=":")
ax.fill_between(dates, equity_s, roll_max, where=(dd < 0),
                alpha=0.25, color=RED, label="Drawdown", zorder=1)

# Annotate final value
ax.annotate(f"${equity_s[-1]:.2f}",
            xy=(dates[-1], equity_s[-1]),
            xytext=(-60, 10), textcoords="offset points",
            fontsize=10, fontweight="bold", color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

# Stat box
stats_txt = (f"Sharpe  {sharpe_val:.2f}\n"
             f"Return  {total_ret:+.1%}\n"
             f"Max DD  {max_dd:.1%}\n"
             f"Win %    {win_rate:.1%}")
ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
        va="top", ha="left", fontsize=9, color=WHITE,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=NAVY,
                  edgecolor=GRID, alpha=0.9))

ax.set_ylabel("Portfolio Value ($1 start)", fontsize=9, color=MUTED)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
ax.set_title("Equity Curve — Final RF Model vs. Buy & Hold (Validation Set)",
             fontsize=13, fontweight="bold", color=WHITE, pad=12)
ax.legend(fontsize=9, loc="upper left", framealpha=0.85,
          bbox_to_anchor=(0.25, 1.0))
ax.grid(axis="y", zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8)

plt.tight_layout(pad=1.2)
chart3 = save_fig(fig, "slide_equity_curve.png")
print("saved slide_equity_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
feature_names = [
    "Raw return  (ret_lag)",
    "Vol-adjusted return  (ret / 5d σ)",
    "5-day mean return",
    "20-day mean return",
    "MACD ratio  (5d mean / 20d mean)",
    "Vol ratio  (5d σ / 20d σ)",
]
importances = model.clf_.feature_importances_
order   = np.argsort(importances)
s_names = [feature_names[i] for i in order]
s_imp   = importances[order]
bar_clr = [GREEN if v == s_imp.max() else CYAN for v in s_imp]

fig, ax = plt.subplots(figsize=(SLIDE_W - 3.5, 4.0), facecolor=NAVY)
ax.set_facecolor(PANEL)

bars = ax.barh(s_names, s_imp, color=bar_clr, edgecolor=NAVY,
               linewidth=0.4, height=0.62)
for bar, v in zip(bars, s_imp):
    ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=9.5, color=WHITE, fontweight="bold")

ax.set_xlabel("Mean Decrease in Impurity", fontsize=9, color=MUTED)
ax.set_title("Feature Importances — Final Random Forest",
             fontsize=13, fontweight="bold", color=WHITE, pad=12)
ax.tick_params(axis="y", colors=WHITE, labelsize=9.5, length=0)
ax.tick_params(axis="x", colors=MUTED, labelsize=8)
ax.grid(axis="x", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout(pad=1.4)
chart4 = save_fig(fig, "slide_feature_importance.png")
print("saved slide_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — Rolling Sharpe
# ─────────────────────────────────────────────────────────────────────────────
WINDOW = 63
roll_rets = pd.Series(strat_rets, index=dates)
roll_sh = (roll_rets.rolling(WINDOW).mean()
           / (roll_rets.rolling(WINDOW).std() + 1e-9)) * np.sqrt(252)

fig, ax = plt.subplots(figsize=(SLIDE_W - 1.5, 4.0), facecolor=NAVY)
ax.set_facecolor(PANEL)

ax.plot(dates, roll_sh, color=CYAN, linewidth=1.8,
        label=f"{WINDOW}-day Rolling Sharpe", zorder=3)
ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--", zorder=2)
ax.axhline(sharpe_val, color=GREEN, linewidth=1.0, linestyle=":",
           alpha=0.75, zorder=2,
           label=f"Full-period Sharpe  ({sharpe_val:.2f})")
ax.fill_between(dates, roll_sh, 0, where=(roll_sh >= 0),
                alpha=0.18, color=GREEN, zorder=1)
ax.fill_between(dates, roll_sh, 0, where=(roll_sh <  0),
                alpha=0.18, color=RED,   zorder=1)

ax.set_ylabel("Annualised Sharpe", fontsize=9, color=MUTED)
ax.set_title(f"Rolling {WINDOW}-Day Sharpe — Validation Set",
             fontsize=13, fontweight="bold", color=WHITE, pad=12)
ax.legend(fontsize=9, framealpha=0.85)
ax.grid(axis="y", zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8)

plt.tight_layout(pad=1.2)
chart5 = save_fig(fig, "slide_rolling_sharpe.png")
print("saved slide_rolling_sharpe.png")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PPTX DECK
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding deck...")

prs = Presentation()
prs.slide_width  = Inches(SLIDE_W)
prs.slide_height = Inches(SLIDE_H)

BLANK = 6   # blank layout index

def rgb(h):
    r, g, b = hex2rgb(h)
    return RGBColor(r, g, b)

def add_slide(prs):
    layout = prs.slide_layouts[BLANK]
    slide = prs.slides.add_slide(layout)
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = rgb(NAVY)
    return slide

def add_text(slide, text, left, top, width, height,
             size=24, bold=False, color=WHITE, align=PP_ALIGN.LEFT, italic=False):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(color)
    return txBox

def add_image(slide, path, left, top, width, height=None):
    if height:
        slide.shapes.add_picture(str(path),
            Inches(left), Inches(top), Inches(width), Inches(height))
    else:
        slide.shapes.add_picture(str(path),
            Inches(left), Inches(top), Inches(width))

def add_accent_bar(slide, color=GREEN, height=0.06):
    from pptx.util import Pt
    bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), Inches(0),
        Inches(SLIDE_W), Inches(height))
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(color)
    bar.line.fill.background()

def header(slide, title, subtitle="", accent=GREEN):
    add_accent_bar(slide, accent)
    add_text(slide, title, 0.35, 0.08, 12, 0.55,
             size=26, bold=True, color=WHITE)
    if subtitle:
        add_text(slide, subtitle, 0.35, 0.58, 12, 0.35,
                 size=13, color=MUTED, italic=True)

# ── Slide 1: Title ────────────────────────────────────────────────────────────
slide = add_slide(prs)
add_accent_bar(slide, GREEN, height=0.08)
add_accent_bar(slide, CYAN,  height=0.03)

add_text(slide, "XLE Energy ETF", 1.1, 1.4, 11, 0.7,
         size=16, color=MUTED, italic=True)
add_text(slide, "AutoResearch Capstone", 1.1, 1.9, 11, 1.0,
         size=38, bold=True, color=WHITE)
add_text(slide, "ML-Driven Sharpe Ratio Optimization", 1.1, 2.85, 11, 0.6,
         size=20, color=CYAN)

# Stat pills (manual text boxes)
stats = [
    (f"True Baseline  0.86", 1.1,  4.3, AMBER),
    (f"Best Committed  2.21", 4.4,  4.3, GREEN),
    (f"Final Sharpe  2.59",  7.7,  4.3, CYAN),
    (f"~60 Experiments",    11.0, 4.3, PURPLE),
]
for txt, lx, ly, col in stats:
    box = slide.shapes.add_shape(1,
        Inches(lx), Inches(ly), Inches(2.0), Inches(0.55))
    box.fill.solid()
    box.fill.fore_color.rgb = rgb(PANEL)
    box.line.color.rgb = rgb(col)
    box.line.width = Pt(1.5)
    tf = box.text_frame
    tf.margin_left = Pt(6)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = txt
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = rgb(col)

add_text(slide, "Sessions: Apr 15 – May 18 2026  ·  Model: RF DirectionModel  ·  Data: XLE 2020–present",
         1.1, 5.1, 11, 0.4, size=10, color=MUTED, italic=True)

add_text(slide, "Only model.py was modified. prepare.py and run.py were frozen.",
         1.1, 5.55, 11, 0.4, size=10, color=MUTED, italic=True)

# ── Slide 2: All experiments bar chart ───────────────────────────────────────
slide = add_slide(prs)
header(slide, "All ~60 Experiments — Sharpe Ratio",
       "Grouped by session · Color = keep / discard / final / XGBoost", CYAN)
add_image(slide, chart1, 0.2, 0.85, SLIDE_W - 0.4, SLIDE_H - 1.0)

# ── Slide 3: Research progression ────────────────────────────────────────────
slide = add_slide(prs)
header(slide, "Research Progression — Key Milestones",
       "True baseline 0.86 → best committed 2.21 → post-cleanup 2.59", GREEN)
add_image(slide, chart2, 0.4, 0.92, SLIDE_W - 0.8, SLIDE_H - 1.05)

# ── Slide 4: Equity curve ─────────────────────────────────────────────────────
slide = add_slide(prs)
header(slide, "Equity Curve — Final RF Model vs. Buy & Hold",
       "Validation set · Red shading = drawdown periods", GREEN)
add_image(slide, chart3, 0.4, 0.92, SLIDE_W - 0.8, SLIDE_H - 1.05)

# ── Slide 5: Feature importance ───────────────────────────────────────────────
slide = add_slide(prs)
header(slide, "Feature Importances — Final Random Forest",
       "6 rolling momentum / volatility features derived from ret_lag only", CYAN)
add_image(slide, chart4, 1.5, 0.95, SLIDE_W - 3.0, SLIDE_H - 1.1)

# ── Slide 6: Rolling Sharpe ───────────────────────────────────────────────────
slide = add_slide(prs)
header(slide, f"Rolling 63-Day Sharpe — Validation Period",
       "Green fill = positive Sharpe periods · Red fill = negative", CYAN)
add_image(slide, chart5, 0.4, 0.92, SLIDE_W - 0.8, SLIDE_H - 1.05)

# ── Slide 7: What Worked / Failed ─────────────────────────────────────────────
slide = add_slide(prs)
header(slide, "What Worked vs. What Failed", accent=AMBER)

left_items = [
    ("✓  3-year rolling window",         "Dropping it to 2yr cost −1.3 Sharpe"),
    ("✓  WTI shock filter at 2%",        "+0.37 Sharpe; 1.5% and 3% both hurt"),
    ("✓  Direction classification",      "sign(pred)×ret metric → only direction matters"),
    ("✓  Shallow RF (depth=2)",          "Depth=4 overfit; depth=1 underfit"),
    ("✓  Exponential sample decay",      "Upweights recent data within rolling window"),
]
right_items = [
    ("✗  Direct Sharpe optimizer",       "Training obj didn't correlate with val (0.03)"),
    ("✗  Macro / weather features",      "OOS −0.74 vs. IS 1.22; too sparse"),
    ("✗  XGBoost",                       "WF Sharpe −0.40; sequential boosting fits noise"),
    ("✗  WTI as RF input",               "Works as filter only, not as feature (1.34)"),
    ("✗  More features (lags/RSI/Bol.)", "Every extra feature reduced val Sharpe"),
]

for i, (title, sub) in enumerate(left_items):
    y = 1.05 + i * 0.96
    add_text(slide, title, 0.35, y,       6.0, 0.38, size=11, bold=True,  color=GREEN)
    add_text(slide, sub,   0.55, y + 0.36, 5.8, 0.38, size=9.5, color=MUTED)

for i, (title, sub) in enumerate(right_items):
    y = 1.05 + i * 0.96
    add_text(slide, title, 6.9,  y,       6.0, 0.38, size=11, bold=True,  color=RED)
    add_text(slide, sub,   7.1,  y + 0.36, 5.8, 0.38, size=9.5, color=MUTED)

# vertical divider
div = slide.shapes.add_shape(1,
    Inches(6.6), Inches(0.9), Inches(0.02), Inches(SLIDE_H - 1.0))
div.fill.solid()
div.fill.fore_color.rgb = rgb(GRID)
div.line.fill.background()

# ── Slide 8: Final Model Card ────────────────────────────────────────────────
slide = add_slide(prs)
header(slide, "Final Model — DirectionModel (RF)", accent=GREEN)

spec_lines = [
    ("Classifier",       "RandomForestClassifier  n=600, depth=2, msl=22, max_features='sqrt'"),
    ("Training window",  "Rolling 756 days (3 yr) — drops COVID 2020–21 regime"),
    ("Sample weights",   "Exponential decay  exp(linspace(−0.3, 0, N))[cutoff:]"),
    ("Features",         "6 rolling momentum / volatility features derived from ret_lag"),
    ("Shock filter",     "Go flat when |WTI log-ret yesterday| > 2 %  (EIA cache → yfinance)"),
    ("Val Sharpe",       "2.59  (single holdout)  ·  0.32 mean across 4 walk-forward folds"),
    ("Val return",       f"{total_ret:+.1%}  ·  Max DD {max_dd:.1%}  ·  Win rate {win_rate:.1%}"),
]
for i, (k, v) in enumerate(spec_lines):
    y = 1.05 + i * 0.72
    add_text(slide, k, 0.5, y,      2.8, 0.45, size=11, bold=True, color=CYAN)
    add_text(slide, v, 3.3, y,      9.5, 0.45, size=11, color=WHITE)
    sep = slide.shapes.add_shape(1,
        Inches(0.5), Inches(y + 0.58), Inches(12.3), Inches(0.01))
    sep.fill.solid()
    sep.fill.fore_color.rgb = rgb(GRID)
    sep.line.fill.background()

add_text(slide,
    "⚠  Walk-forward Sharpe (0.32) is the conservative estimate. "
    "The single-holdout (2.59) reflects favorable 2025–2026 market conditions.",
    0.5, 6.8, 12.3, 0.45, size=9.5, color=AMBER, italic=True)

# ─────────────────────────────────────────────────────────────────────────────
deck_path = OUT / "deck.pptx"
prs.save(str(deck_path))
print(f"\nDeck saved: {deck_path.resolve()}")
print(f"Slides: {len(prs.slides)}  |  Size: {deck_path.stat().st_size / 1024:.0f} KB")
