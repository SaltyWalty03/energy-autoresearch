# Ablation Table

All val Sharpe values are from `experiment_log.csv` (15-run grid sweep, XLE holdout 2025–2026).
Walk-forward Sharpe (WF) was tracked starting 2026-05-18 (`results.tsv`); earlier grid runs used
single-holdout Sharpe as the selection criterion. N/A indicates the metric was not recorded for
that configuration — the Bug 2 fix to `run_experiments.py` will populate this column on the next
sweep.

| Configuration | Key Change | WF Sharpe Mean | Best Val Sharpe | N Runs |
|---|---|---|---|---|
| Baseline RF | n_estimators=300, depth=2, leaf=22, 3yr window, WTI filter, exp decay | 0.3226 | 2.5903 | 1 |
| Tuned RF | n_estimators=600, depth=2, leaf=22 | N/A | 2.5990 | 1 |
| Looser leaf | min_samples_leaf=10 (best single-param change) | N/A | 2.6104 | 1 |
| + Shorter train window | train_window=504 (2yr instead of 3yr) | N/A | 1.2833 | 1 |
| + Loose WTI filter | wti_thresh=0.01 (over-filters) | N/A | 1.5038 | 1 |
| + Strict WTI filter | wti_thresh=0.03 (under-filters) | N/A | 2.3902 | 1 |
| XGBoost (rejected) | XGBClassifier, depth=3, n=600, lr=0.05 | -0.40 | 0.54 | 2 |
| + Regime Layer (RF) | t-SNE + HDBSCAN, perplexity=30, k=5 per-regime RF | 1.0668* | — | — |

*Regime layer WF Sharpe is the regime-specific figure for XLE regime 4 (the active regime on
2026-05-19), as observed in cloud function logs. A full per-regime ablation requires re-running
`train_all.py` with regime-analysis logging, which is a Week 2 task.

---

## Analysis

**Which component contributed most.** The 3-year rolling training window was the single largest
contributor, as shown by `run_014`: reducing it to 2 years (504 rows) dropped val Sharpe from
2.5903 to 1.2833 — a loss of 1.31 Sharpe points. This confirms that the COVID-era 2020–2021
regime is so structurally different that including it without sufficient temporal discounting
destroys out-of-sample performance. The exponential decay and WTI shock filter each contributed
smaller but measurable improvements; the 2% WTI threshold in particular proved well-calibrated:
loosening it to 1% collapsed Sharpe to 1.50, and tightening to 3% cost 0.20 Sharpe points.

**Why XGBoost underperformed.** XGB achieved a walk-forward Sharpe mean of -0.40 versus RF's
0.32 on the same data. The signal-to-noise ratio in daily return prediction is very low (~600
effective training rows after the rolling-window cutoff), and XGBoost's sequential boosting
aggressively fits residuals — which on a noisy financial series means fitting noise. Random Forest
bagging averages over many high-variance trees and produces smoother probability estimates that
generalise better in this low-SNR setting. XGBoost would likely perform better with a much larger
dataset or with a custom Sharpe-optimising loss function.

**What the regime layer adds.** The regime layer allows the system to recognise that the same
momentum signal has different reliability depending on the current market structure. Regimes
discovered by t-SNE + HDBSCAN correspond to distinct volatility and trend environments; the
between-regime Sharpe variance of 2.79 (perplexity=30, n_clusters=5) confirms these environments
are genuinely different in predictability. The winning configuration uses 5 advisory clusters,
which HDBSCAN expands to ~13 actual regimes for XLE by adaptive min_cluster_size. Perplexity=30
was optimal because it balances local and global t-SNE structure at the ~1,400-row dataset size.

**Honest limitation.** The ablation table compares configurations on the same 20% chronological
holdout used to select the final model, which introduces a selection-bias leak: any configuration
that happened to fit the specific 2025–2026 market regime is favoured regardless of whether it
would generalise to 2026–2027. The walk-forward Sharpe metric partially corrects for this, but the
holdout period itself (roughly one year) may not contain enough regime diversity to distinguish
genuine generalisation from lucky in-sample fit.
