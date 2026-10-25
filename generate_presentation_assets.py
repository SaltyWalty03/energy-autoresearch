"""Generate presentation_assets/ PNG charts — full experiment history."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from prepare import load_and_split_data
from model import build_model

OUT = Path("presentation_assets")
OUT.mkdir(exist_ok=True)

BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
AMBER  = "#D97706"
PURPLE = "#7C3AED"
GRAY   = "#6B7280"
BG     = "#F9FAFB"

# ─────────────────────────────────────────────────────────────────────────────
# 1. SHARPE COMPARISON — full named experiment history
# ─────────────────────────────────────────────────────────────────────────────
# (label, sharpe, kind, session_group)
# kind: "baseline" | "discard" | "keep" | "final" | "xgb"
experiments = [
    # ── True baseline ────────────────────────────────────────────────────────
    ("S1 · LinearRegression (true baseline)",     0.8584, "baseline", "Session 1"),
    # ── Session 1 model search ────────────────────────────────────────────────
    ("S1 · RetFeatures + Ridge",                  0.0653, "discard",  "Session 1"),
    ("S1 · GradBoost regressor",                  0.1524, "discard",  "Session 1"),
    ("S1 · Rolling feat + Ridge",                 0.4703, "discard",  "Session 1"),
    ("S1 · Rolling feat + LogisticReg",           0.9581, "keep",     "Session 1"),
    ("S1 · LogisticReg + extra lags/RSI",         0.8824, "discard",  "Session 1"),
    ("S1 · RF depth=3, msl=30",                   1.2623, "keep",     "Session 1"),
    ("S1 · RF depth=2, msl=25, n=200",            1.3092, "keep",     "Session 1"),
    ("S1 · MLP (128,)",                           1.1365, "discard",  "Session 1"),
    ("S1 · RF n=300, depth=2, msl=20",            1.3717, "keep",     "Session 1"),
    ("S1 · RF n=200, depth=2, msl=22 ★",         1.3954, "keep",     "Session 1"),
    ("S1 · Direct Sharpe optimizer (overfit)",    0.0267, "discard",  "Session 1"),
    # ── Detailed macro experiments (2026-04-20) ───────────────────────────────
    ("S1d · OLS 24 macro/weather feat",          -0.739,  "discard",  "Macro Exps"),
    ("S1d · OLS 10 selected feat",                0.039,  "discard",  "Macro Exps"),
    ("S1d · Lasso/Ridge",                         0.613,  "discard",  "Macro Exps"),
    ("S1d · RF (macro, overfit)",                -0.705,  "discard",  "Macro Exps"),
    ("S1d · Signal audit (best single)",          1.687,  "keep",     "Macro Exps"),
    ("S1d · Norm. signal composite",              0.663,  "discard",  "Macro Exps"),
    ("S1d · WF best signal + Lasso ensemble",     1.042,  "keep",     "Macro Exps"),
    # ── Hyperparameter sweep ──────────────────────────────────────────────────
    ("Sweep · run_001 baseline",                  2.4041, "baseline", "Sweep"),
    ("Sweep · run_002 depth=3",                   2.1359, "discard",  "Sweep"),
    ("Sweep · run_003 depth=4",                   2.4652, "keep",     "Sweep"),
    ("Sweep · run_004 depth=1 stumps",            1.2735, "discard",  "Sweep"),
    ("Sweep · run_005 msl=10",                    2.3837, "discard",  "Sweep"),
    ("Sweep · run_006 msl=50",                    1.4882, "discard",  "Sweep"),
    ("Sweep · run_007 msl=100",                   1.5910, "discard",  "Sweep"),
    ("Sweep · run_008 n=100",                     2.4742, "keep",     "Sweep"),
    ("Sweep · run_009 n=600 ★",                  2.4793, "keep",     "Sweep"),
    ("Sweep · run_010 window=10",                 2.3446, "discard",  "Sweep"),
    ("Sweep · run_011 window=40",                 1.9068, "discard",  "Sweep"),
    ("Sweep · run_012 WTI thresh=3%",            2.2201, "discard",  "Sweep"),
    ("Sweep · run_013 WTI thresh=1%",            1.4536, "discard",  "Sweep"),
    ("Sweep · run_014 train 2yr",                 1.2470, "discard",  "Sweep"),
    ("Sweep · run_015 combo",                     2.0270, "discard",  "Sweep"),
    # ── Session 2 — WTI filter tuning ─────────────────────────────────────────
    ("S2 · WTI filter 3%",                        2.0219, "keep",     "Session 2"),
    ("S2 · WTI filter 2% ★ (best committed)",    2.2119, "final",    "Session 2"),
    ("S2 · WTI filter 1.5%",                      2.0065, "discard",  "Session 2"),
    ("S2 · WTI as RF input feat",                 1.3403, "discard",  "Session 2"),
    ("S2 · Contrarian on shock days",             1.9532, "discard",  "Session 2"),
    # ── Post-cleanup / XGBoost comparisons ───────────────────────────────────
    ("Post · RF smoke test",                      2.5903, "keep",     "Post-cleanup"),
    ("Post · XGBoost baseline (WF: -0.40)",       0.5408, "xgb",      "Post-cleanup"),
    ("Post · XGBoost depth=2 (WF: -0.68)",        0.2908, "xgb",      "Post-cleanup"),
    ("Post · RF baseline + WF",                   2.5949, "final",    "Post-cleanup"),
]

labels  = [e[0] for e in experiments]
sharpes = [e[1] for e in experiments]
kinds   = [e[2] for e in experiments]

color_map = {
    "baseline": AMBER,
    "discard":  RED,
    "keep":     BLUE,
    "final":    GREEN,
    "xgb":      PURPLE,
}
colors = [color_map[k] for k in kinds]

fig, ax = plt.subplots(figsize=(14, 18), facecolor=BG)
ax.set_facecolor(BG)

# horizontal separator lines between session groups
session_groups = [e[3] for e in experiments]
prev = None
for i, sg in enumerate(session_groups):
    if sg != prev and i != 0:
        ax.axhline(i - 0.5, color="#D1D5DB", linewidth=0.8, linestyle="-")
    prev = sg

bars = ax.barh(range(len(labels)), sharpes, color=colors,
               edgecolor="white", linewidth=0.5, height=0.72)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7.2)
ax.axvline(0,      color=GRAY,  linewidth=0.7, linestyle="--")
ax.axvline(0.8584, color=AMBER, linewidth=1.0, linestyle=":", alpha=0.6,
           label="True baseline (0.86)")
ax.axvline(2.2119, color=GREEN, linewidth=1.0, linestyle=":", alpha=0.6,
           label="Best committed (2.21)")

for bar, v in zip(bars, sharpes):
    ax.text(v + 0.03 if v >= 0 else v - 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=6.5, color="#111827")

from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor=GREEN,  label="Final / best committed"),
    Patch(facecolor=BLUE,   label="Kept (improved vs. prior best)"),
    Patch(facecolor=AMBER,  label="Baseline reference"),
    Patch(facecolor=RED,    label="Discarded"),
    Patch(facecolor=PURPLE, label="XGBoost (rejected)"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=8, framealpha=0.9)
ax.set_xlabel("Annualised Sharpe Ratio (val set)", fontsize=10)
ax.set_title("Sharpe Ratio — All ~60 Experiments Across Sessions",
             fontsize=13, fontweight="bold", pad=12)
ax.tick_params(axis="y", labelsize=7.2)
plt.tight_layout()
plt.savefig(OUT / "sharpe_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved sharpe_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPERIMENT PROGRESSION — Sharpe over time (milestone experiments only)
# ─────────────────────────────────────────────────────────────────────────────
milestones = [
    ("LinReg baseline",            0.8584),
    ("LogisticReg + rolling feat", 0.9581),
    ("RF depth=3",                 1.2623),
    ("RF depth=2, msl=22 (S1★)",  1.3954),
    ("Sweep run_001",              2.4041),
    ("Sweep run_009 n=600",        2.4793),
    ("+ WTI filter 3%",            2.0219),
    ("+ WTI filter 2% (final)",    2.2119),
    ("Post-cleanup RF",            2.5903),
    ("RF + walk-forward",          2.5949),
]
m_labels = [x[0] for x in milestones]
m_values = [x[1] for x in milestones]
m_colors = [GREEN if v == max(m_values) else BLUE for v in m_values]
m_colors[0] = AMBER  # true baseline

fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(range(len(milestones)), m_values, color=BLUE, linewidth=1.5,
        marker="o", markersize=7, zorder=3)
for i, (lbl, v, c) in enumerate(zip(m_labels, m_values, m_colors)):
    ax.scatter(i, v, color=c, s=70, zorder=4)
    ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7.5, color="#111827")
ax.axhline(0.8584, color=AMBER, linewidth=0.9, linestyle=":", alpha=0.6,
           label="True baseline (0.86)")
ax.set_xticks(range(len(milestones)))
ax.set_xticklabels(m_labels, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Annualised Sharpe", fontsize=10)
ax.set_title("Research Progression — Key Milestone Sharpe Values",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=9, framealpha=0.85)
plt.tight_layout()
plt.savefig(OUT / "experiment_progression.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved experiment_progression.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EQUITY CURVE (final model on val set)
# ─────────────────────────────────────────────────────────────────────────────
train, val = load_and_split_data()
model = build_model()
model.fit(train[["ret_lag"]], train["target"])
preds = model.predict(val[["ret_lag"]])

strat_rets = np.sign(preds) * val["target"].values
bh_rets    = val["target"].values
equity_s   = np.exp(np.cumsum(strat_rets))
equity_bh  = np.exp(np.cumsum(bh_rets))
dates      = val.index

roll_max = np.maximum.accumulate(equity_s)
dd = equity_s / roll_max - 1

sharpe_val   = (np.mean(strat_rets) / (np.std(strat_rets) + 1e-9)) * np.sqrt(252)
total_return = equity_s[-1] - 1
max_dd       = dd.min()
win_rate     = (strat_rets > 0).mean()

fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(dates, equity_s,  color=GREEN, linewidth=1.8, label="Final RF (DirectionModel)")
ax.plot(dates, equity_bh, color=GRAY,  linewidth=1.2, linestyle="--", alpha=0.7,
        label="Buy & Hold XLE")
ax.axhline(1.0, color=GRAY, linewidth=0.5, linestyle=":")
ax.fill_between(dates, equity_s, roll_max, where=(dd < 0),
                alpha=0.13, color=RED, label="Drawdown")
ax.set_title(
    f"Equity Curve — Final RF Model (Validation Set)\n"
    f"Return: {total_return:+.1%}  |  Sharpe: {sharpe_val:.2f}  "
    f"|  Max DD: {max_dd:.1%}  |  Win rate: {win_rate:.1%}",
    fontsize=11, fontweight="bold", pad=10
)
ax.set_ylabel("Portfolio Value ($1 start)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(OUT / "equity_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved equity_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
feature_names = [
    "Raw return (ret_lag)",
    "Vol-adjusted return",
    "5-day mean return",
    "20-day mean return",
    "MACD ratio (5d/20d)",
    "Vol ratio (5d/20d std)",
]
importances = model.clf_.feature_importances_
order = np.argsort(importances)
s_names = [feature_names[i] for i in order]
s_imp   = importances[order]
bar_colors = [GREEN if v == s_imp.max() else BLUE for v in s_imp]

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)
bars = ax.barh(s_names, s_imp, color=bar_colors, edgecolor="white", height=0.6)
for bar, v in zip(bars, s_imp):
    ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=8.5)
ax.set_xlabel("Mean Decrease in Impurity", fontsize=10)
ax.set_title("Feature Importances — Final Random Forest", fontsize=12,
             fontweight="bold", pad=10)
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ROLLING SHARPE (63-day window)
# ─────────────────────────────────────────────────────────────────────────────
WINDOW = 63
roll_rets = pd.Series(strat_rets, index=dates)
roll_sh = (roll_rets.rolling(WINDOW).mean()
           / (roll_rets.rolling(WINDOW).std() + 1e-9)) * np.sqrt(252)

fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(dates, roll_sh, color=BLUE, linewidth=1.5,
        label=f"{WINDOW}-day Rolling Sharpe")
ax.axhline(0, color=GRAY, linewidth=0.7, linestyle="--")
ax.axhline(sharpe_val, color=GREEN, linewidth=1.0, linestyle=":", alpha=0.8,
           label=f"Full-period Sharpe ({sharpe_val:.2f})")
ax.fill_between(dates, roll_sh, 0, where=(roll_sh >= 0), alpha=0.14, color=GREEN)
ax.fill_between(dates, roll_sh, 0, where=(roll_sh < 0),  alpha=0.14, color=RED)
ax.set_ylabel("Annualised Sharpe", fontsize=10)
ax.set_title(f"Rolling {WINDOW}-Day Sharpe — Final RF Model (Validation Set)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=9, framealpha=0.85)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(OUT / "rolling_sharpe.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved rolling_sharpe.png")

print(f"\nAll 5 charts saved to: {OUT.resolve()}")
print(f"Final model val Sharpe: {sharpe_val:.4f}")
print(f"Total return: {total_return:+.2%} | Max DD: {max_dd:.2%} | Win rate: {win_rate:.2%}")
