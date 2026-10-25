"""
fix_deck.py
Structural fixes for deck.pptx:
  1. Insert Problem Framing as Slide 2 (push existing 2-8 down to 3-9)
  2. Append Reflection as Slide 10
  3. Add speaker notes to all 10 slides
  4. Save as presentation_assets/deck_final.pptx
  5. Programmatic verification report
"""
import io, shutil, sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Users\wpbro\OneDrive\Documents\Stat 303-1\Workspace\energy-autoresearch")
ASSETS  = BASE / "presentation_assets"
SRC     = ASSETS / "deck.pptx"
OUT     = ASSETS / "deck_final.pptx"
TEMP    = Path(r"C:\Users\wpbro\AppData\Local\Temp\deck_fix_work.pptx")

# ── Colour palette (matches build_deck.py exactly) ───────────────────────────
NAVY   = (0x0D, 0x1B, 0x2A)
PANEL  = (0x1B, 0x2B, 0x3E)
WHITE  = (0xF0, 0xF4, 0xF8)
MUTED  = (0x8B, 0xA0, 0xB4)
CYAN   = (0x38, 0xBD, 0xF8)
GREEN  = (0x4A, 0xDE, 0x80)
RED    = (0xF8, 0x71, 0x71)
AMBER  = (0xFC, 0xD3, 0x4D)
GRID   = (0x22, 0x37, 0x4D)

def rgb(t): return RGBColor(*t)

SLIDE_W, SLIDE_H = 13.33, 7.5

# ── Load (via temp copy to avoid OneDrive lock) ───────────────────────────────
shutil.copy2(SRC, TEMP)
with open(TEMP, "rb") as f:
    prs = Presentation(io.BytesIO(f.read()))

print(f"Loaded: {len(prs.slides)} slides")

# ── Helper: find blank layout ─────────────────────────────────────────────────
def _blank_layout(prs):
    for layout in prs.slide_layouts:
        if layout.name.lower() in ("blank", ""):
            return layout
    # fall back: layout with fewest placeholders
    return min(prs.slide_layouts, key=lambda l: len(l.placeholders))

BLANK_LAYOUT = _blank_layout(prs)
print(f"Blank layout: '{BLANK_LAYOUT.name}'")

# ── Helper: set navy background on a slide ───────────────────────────────────
def set_navy_bg(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = rgb(NAVY)

# ── Helper: add accent bar at top ────────────────────────────────────────────
def add_accent_bar(slide, color=GREEN, h=0.06):
    bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE.RECTANGLE
        Inches(0), Inches(0),
        Inches(SLIDE_W), Inches(h))
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(color)
    bar.line.fill.background()

# ── Helper: add text box ─────────────────────────────────────────────────────
def add_text(slide, text, left, top, width, height,
             size=11, bold=False, italic=False,
             color=WHITE, align=PP_ALIGN.LEFT):
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(color)
    return txb

# ── Helper: thin horizontal rule ─────────────────────────────────────────────
def add_rule(slide, top, color=GRID):
    line = slide.shapes.add_shape(
        1,
        Inches(0.4), Inches(top),
        Inches(12.5), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = rgb(color)
    line.line.fill.background()

# ── Helper: labeled content block ─────────────────────────────────────────────
def add_block(slide, label, body, left, top, width,
              label_color=CYAN, label_size=12, body_size=10.5):
    add_text(slide, label, left, top, width, 0.4,
             size=label_size, bold=True, color=label_color)
    add_text(slide, body, left, top + 0.4, width, 1.5,
             size=body_size, color=WHITE)

# ── Helper: header bar ────────────────────────────────────────────────────────
def add_header(slide, title, subtitle="", accent=GREEN):
    add_accent_bar(slide, accent)
    add_text(slide, title, 0.35, 0.08, 12.5, 0.55,
             size=26, bold=True, color=WHITE)
    if subtitle:
        add_text(slide, subtitle, 0.35, 0.58, 12.5, 0.35,
                 size=13, italic=True, color=MUTED)

# ── Helper: speaker notes ─────────────────────────────────────────────────────
def set_notes(slide, text):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    # Clear all existing content and set fresh
    for para in list(tf.paragraphs)[1:]:
        para._p.getparent().remove(para._p)
    p0 = tf.paragraphs[0]
    # Remove all existing runs from first paragraph
    for run in list(p0.runs):
        run._r.getparent().remove(run._r)
    run = p0.add_run()
    run.text = text

# ── Helper: move slide ────────────────────────────────────────────────────────
def move_slide(prs, old_idx, new_idx):
    xml_slides = prs.slides._sldIdLst
    slides = list(xml_slides)
    el = slides[old_idx]
    xml_slides.remove(el)
    xml_slides.insert(new_idx, el)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build Problem Framing slide (appended at end, then moved to slot 1)
# ─────────────────────────────────────────────────────────────────────────────
slide_framing = prs.slides.add_slide(BLANK_LAYOUT)
set_navy_bg(slide_framing)
add_header(slide_framing, "The Problem", accent=AMBER)

# 3 full-width stacked blocks, each ~2.0" tall, with dividers
blocks_framing = [
    (
        "Why XLE?",
        "Energy sector returns are driven by oil price shocks, geopolitical events, "
        "and macro regime shifts. These create non-stationary return dynamics that "
        "make prediction genuinely hard.",
        CYAN,
    ),
    (
        "The Research Contract",
        "Can a machine learning model predict the daily direction of XLE well enough "
        "to achieve an annualized Sharpe ratio above 1.0 on a held-out validation set, "
        "using only model.py modifications with prepare.py and run.py frozen?",
        GREEN,
    ),
    (
        "Why It's Hard",
        "Energy returns exhibit fat tails, regime changes, and low signal-to-noise "
        "ratios. Naive feature engineering consistently overfits. The baseline linear "
        "model achieves Sharpe 0.86, meaning there is real signal but capturing it "
        "requires disciplined methodology.",
        AMBER,
    ),
]

y_positions = [1.0, 3.05, 5.1]
for (label, body, lcol), y in zip(blocks_framing, y_positions):
    add_block(slide_framing, label, body,
              left=0.4, top=y, width=12.5, label_color=lcol, label_size=13, body_size=11)

add_rule(slide_framing, top=2.95, color=GRID)
add_rule(slide_framing, top=5.00, color=GRID)

# Move from last position (index 8) to index 1 (slide 2)
move_slide(prs, len(prs.slides) - 1, 1)
print(f"Problem Framing inserted as slide 2. Now {len(prs.slides)} slides.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Build Reflection slide (append as slide 10)
# ─────────────────────────────────────────────────────────────────────────────
slide_reflection = prs.slides.add_slide(BLANK_LAYOUT)
set_navy_bg(slide_reflection)
add_header(slide_reflection, "Reflection", accent=CYAN)

blocks_left = [
    (
        "What the agent did well",
        "Rapid hyperparameter sweeps, consistent experiment logging, and enforcing "
        "the frozen-file contract across all runs.",
        CYAN,
    ),
    (
        "Walk-forward caveat",
        "The single-holdout Sharpe of 2.59 reflects favorable 2025-2026 market "
        "conditions. The walk-forward mean of 0.32 is the conservative generalization "
        "estimate. Both are reported honestly.",
        AMBER,
    ),
]

blocks_right = [
    (
        "What required human judgment",
        "Identifying that the direct Sharpe optimizer had a 0.03 train-val correlation "
        "and was fundamentally broken. Deciding to abandon macro/weather features after "
        "recognizing the sparsity problem. Choosing the 2% WTI threshold by reasoning "
        "about oil market microstructure, not grid search.",
        RED,
    ),
    (
        "If I redesigned the loop",
        "Add a regime-detection gate before model selection. Force the agent to run "
        "walk-forward evaluation on every committed experiment, not just the final one.",
        GREEN,
    ),
]

# Left column
for i, (label, body, lcol) in enumerate(blocks_left):
    y = 1.0 + i * 2.9
    add_block(slide_reflection, label, body,
              left=0.4, top=y, width=6.1, label_color=lcol, label_size=12, body_size=10.5)

# Right column
for i, (label, body, lcol) in enumerate(blocks_right):
    y = 1.0 + i * 2.9
    add_block(slide_reflection, label, body,
              left=7.0, top=y, width=6.0, label_color=lcol, label_size=12, body_size=10.5)

# Vertical divider between columns
vdiv = slide_reflection.shapes.add_shape(
    1,
    Inches(6.65), Inches(0.95),
    Inches(0.02), Inches(6.3))
vdiv.fill.solid()
vdiv.fill.fore_color.rgb = rgb(GRID)
vdiv.line.fill.background()

# Horizontal mid-rule in each column
add_rule(slide_reflection, top=3.78)

print(f"Reflection appended as slide 10. Now {len(prs.slides)} slides.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Speaker notes — all 10 slides in final order
# ─────────────────────────────────────────────────────────────────────────────
notes = [
    # Slide 1 — Title
    ("This project asks whether a machine learning model can predict the daily direction "
     "of XLE well enough to produce risk-adjusted returns above a naive buy-and-hold baseline. "
     "Starting from a true baseline Sharpe of 0.86, the AutoResearch loop ran roughly 60 "
     "experiments over five weeks and converged on a Random Forest direction classifier with "
     "a single-holdout validation Sharpe of 2.59. I'll walk through how we got there and "
     "what the agent actually contributed."),
    # Slide 2 — Problem Framing
    ("Energy return prediction is hard for structural reasons. Oil prices respond to supply "
     "shocks, geopolitical events, and macro regime changes in ways that are difficult to model "
     "with fixed features. The research contract was narrow by design: only model.py could be "
     "modified, which forced the agent to find signal within a disciplined scope rather than "
     "engineering its way out of the problem."),
    # Slide 3 — All 60 Experiments
    ("This chart is the full experiment trace. Every bar is one run, colored by outcome. "
     "The macro feature experiments produced some of the worst results including negative "
     "Sharpe. The sweep phase is where most of the signal came from. XGBoost, despite being "
     "a reasonable prior, was consistently worse than the RF baseline across all configurations "
     "tried."),
    # Slide 4 — Research Progression
    ("This is the same story told as a timeline of committed improvements. The big jump "
     "happened during the hyperparameter sweep, where moving to n=600 trees and tuning "
     "minimum samples per leaf pushed past Sharpe 2.4. The WTI shock filter then added "
     "0.37 on top of that. The final post-cleanup value of 2.59 reflects a rerun after "
     "removing experimental cruft from the pipeline."),
    # Slide 5 — Equity Curve
    ("The model compounds steadily and largely avoids the drawdown periods where buy-and-hold "
     "gave back gains. The win rate of 43.9% is worth flagging: the model wins less than half "
     "the time but sizes its correct bets better, which is characteristic of a momentum-style "
     "classifier."),
    # Slide 6 — Feature Importances
    ("The final model uses only 6 features, all derived from the lagged return series. Every "
     "attempt to add external features hurt out-of-sample performance. The model is telling us "
     "the signal lives in price momentum and short-term volatility, not in external "
     "fundamentals."),
    # Slide 7 — Rolling Sharpe
    ("The rolling 63-day Sharpe is the credibility check. It stays consistently positive for "
     "most of the validation period, with one notable exception around November-December 2025 "
     "corresponding to a regime shift in energy markets."),
    # Slide 8 — What Worked vs Failed
    ("The most important lesson is about parsimony. Every intervention that added complexity "
     "hurt. The direct Sharpe optimizer had a training-to-validation correlation of 0.03, "
     "meaning the objective it was optimizing had essentially no relationship to what we "
     "actually care about."),
    # Slide 9 — Final Model Card
    ("The final model is fully specified here. The single-holdout Sharpe of 2.59 reflects a "
     "favorable market environment. The walk-forward mean of 0.32 is the more conservative "
     "and honest generalization estimate. The project achieved its contract objective, but "
     "this gap is a real limitation."),
    # Slide 10 — Reflection
    ("The agent accelerated the sweep phase dramatically, but the two most important decisions "
     "in the project required human judgment: recognizing the direct Sharpe optimizer was "
     "broken, and abandoning macro features after understanding the sparsity problem. The loop "
     "worked because the contract was narrow. A looser scope would have produced a messier "
     "result."),
]

assert len(prs.slides) == 10, f"Expected 10 slides, got {len(prs.slides)}"
assert len(notes) == 10, "Notes list must have exactly 10 entries"

for i, (slide, note_text) in enumerate(zip(prs.slides, notes)):
    set_notes(slide, note_text)
    print(f"  Notes set: Slide {i+1} ({len(note_text)} chars)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Save
# ─────────────────────────────────────────────────────────────────────────────
buf = io.BytesIO()
prs.save(buf)
buf.seek(0)
OUT.write_bytes(buf.read())
print(f"\nSaved: {OUT}  ({OUT.stat().st_size // 1024} KB)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Programmatic verification
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- VERIFICATION ---")
# Reload from saved file
with open(OUT, "rb") as f:
    prs2 = Presentation(io.BytesIO(f.read()))

issues = []

# 5a. Slide count
n = len(prs2.slides)
status = "OK" if n == 10 else f"FAIL (expected 10, got {n})"
print(f"Slide count: {n}  [{status}]")
if n != 10:
    issues.append(f"Wrong slide count: {n}")

# 5b. Per-slide checks
slide_titles = [
    "Title",
    "Problem Framing",
    "All Experiments",
    "Research Progression",
    "Equity Curve",
    "Feature Importances",
    "Rolling Sharpe",
    "What Worked / Failed",
    "Final Model Card",
    "Reflection",
]
for i, slide in enumerate(prs2.slides):
    # Background check
    bg_fill = slide.background.fill
    try:
        bg_type = bg_fill.type
    except Exception:
        bg_type = None

    # Shape count
    n_shapes = len(slide.shapes)

    # Notes check
    notes_text = slide.notes_slide.notes_text_frame.text.strip()
    has_notes = len(notes_text) > 20

    # Image count (slides 3-7 should have images from the chart PNGs)
    n_images = sum(1 for s in slide.shapes
                   if s.shape_type == 13)  # MSO_SHAPE_TYPE.PICTURE

    status_parts = []
    if not has_notes:
        status_parts.append("MISSING NOTES")
        issues.append(f"Slide {i+1} missing notes")
    if i in (1, 9) and n_shapes < 4:
        status_parts.append(f"LOW SHAPE COUNT ({n_shapes})")
        issues.append(f"Slide {i+1} has only {n_shapes} shapes (new slide may be broken)")

    status_str = ", ".join(status_parts) if status_parts else "OK"
    print(f"  Slide {i+1:2d} [{slide_titles[i]:22s}]: "
          f"{n_shapes:2d} shapes | "
          f"{n_images} imgs | "
          f"notes={'yes' if has_notes else 'NO '} | "
          f"{status_str}")

# 5c. New slides: check they have content text boxes
print()
slide2 = prs2.slides[1]
slide2_texts = [s.text_frame.text for s in slide2.shapes
                if s.has_text_frame and s.text_frame.text.strip()]
print(f"Slide 2 text blocks ({len(slide2_texts)}): "
      + " | ".join(t[:30] for t in slide2_texts))

slide10 = prs2.slides[9]
slide10_texts = [s.text_frame.text for s in slide10.shapes
                 if s.has_text_frame and s.text_frame.text.strip()]
print(f"Slide 10 text blocks ({len(slide10_texts)}): "
      + " | ".join(t[:30] for t in slide10_texts))

# 5d. Verify original slides 2-8 (now 3-9) still have their chart images
print()
for orig_i, new_i in [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]:
    slide = prs2.slides[new_i - 1]
    n_img = sum(1 for s in slide.shapes if s.shape_type == 13)
    status = "OK" if n_img >= 1 else "WARN: no image found"
    print(f"  Chart slide (was {orig_i}, now {new_i}): {n_img} image(s)  [{status}]")

# Summary
print()
if issues:
    print(f"ISSUES FOUND ({len(issues)}):")
    for iss in issues:
        print(f"  - {iss}")
else:
    print("All checks passed. deck_final.pptx is structurally sound.")
