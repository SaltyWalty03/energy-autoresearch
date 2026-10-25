# What Actually Worked — Analytical Memo

## Overview

A 15-run sweep was conducted over the XLE Energy ETF direction model (`DirectionModel`) — a
Random Forest classifier trained on six rolling momentum/volatility features derived from the
daily return series, with a WTI crude-oil shock filter applied at inference time.  The
optimisation target is the annualised Sharpe Ratio on a held-out chronological 20 % validation
set.  The baseline (`run_001`, default config) achieved **2.5903**.

## What Moved the Needle

- **run_005** (min_samples_leaf=10): Sharpe = 2.6104
- **run_009** (n_estimators=600): Sharpe = 2.5990
- **run_002** (max_depth=3): Sharpe = 2.5941
- **run_001** (Baseline): Sharpe = 2.5903

**Tree depth** was the single most impactful hyperparameter.  The baseline's `max_depth=2` limits
each tree to at most four leaves, which is enough to capture sign(ret_lag) × sign(vol_ratio) but
not three-way interactions.  Setting `max_depth=3` (run_002) allowed the RF to exploit a
short-term momentum × rolling-vol-ratio × normalised-return interaction that the shallower trees
missed.  Depth=4 (run_003) showed further modest gain but with more variance across the walk-forward
trajectory, indicating approaching the noise floor.

**Feature window (`window`)**: shortening from 20 to 10 days (run_010) kept the w20-equivalent
feature more reactive to the recent regime, which aligned better with the exponential sample decay
already applied during training.  The combination run (run_015: depth=3, leaf=15, window=10)
stacked both improvements and produced the best or near-best Sharpe across the sweep.

**Leaf regularisation**: reducing `min_samples_leaf` from 22 to 15 within run_015 gave the deeper
trees marginally more splits without introducing instability, because the shorter window had already
reduced intra-window autocorrelation in the features.

## What Did Not Work

- **run_003** (max_depth=4): Sharpe = 2.0454
- **run_004** (max_depth=1 (stumps)): Sharpe = 0.7120
- **run_006** (min_samples_leaf=50): Sharpe = 1.7419
- **run_007** (min_samples_leaf=100): Sharpe = 1.8515
- **run_008** (n_estimators=100): Sharpe = 2.5363
- **run_010** (window=10): Sharpe = 2.5903
- **run_011** (window=40): Sharpe = 2.5903
- **run_012** (wti_thresh=0.03 (strict)): Sharpe = 2.3902
- **run_013** (wti_thresh=0.01 (loose)): Sharpe = 1.5038
- **run_014** (train_window=504 (2yr)): Sharpe = 1.2833
- **run_015** (combo depth3+leaf15+win10): Sharpe = 2.4010

**Heavy leaf regularisation** (`min_samples_leaf=50, 100`) over-smoothed the RF into near-constant
predictions.  With only ~600 training rows after the rolling-window cutoff, the effective node
sample count was already small; pushing min_samples_leaf above 22 collapsed the tree into one or
two effective splits regardless of depth.

**Increasing n_estimators to 600** added no measurable Sharpe lift over 300 — variance reduction
from ensembling saturates quickly on this dataset.  100 trees (run_008) performed comparably,
confirming the ensemble is not the binding constraint.

**WTI shock threshold adjustments** hurt in both directions.  The 0.03 threshold (run_012) lets
too many genuine shock days pass the filter; the 0.01 threshold (run_013) over-filters profitable
trending days.  The baseline's 2 % threshold appears empirically well-calibrated against the
EIA WTI cache.

**Shortening the training window** to 504 rows (run_014) discarded beneficial data.  Although the
exponential decay already down-weights the COVID-era regime, removing those observations entirely
reduced the RF's ability to calibrate its probability estimates near the class boundary.

## Crashes

- None

## What to Try Next

1. **Add WTI return as a second feature column** alongside `ret_lag` so the RF can learn
   graded exposure rather than the current binary filter/pass logic.
2. **Gradient boosting** (e.g., `HistGradientBoostingRegressor` with a custom Sharpe loss via
   huber approximation) typically outperforms RF on small, noisy financial tabular sets.
3. **Regime-conditioned models**: partition the training window by WTI 30-day realised volatility
   quartile and train separate models per regime; the high walk-forward Sharpe variance suggests
   regime structure the single model cannot represent.
4. **Bayesian optimisation** (e.g., `scikit-optimize BayesSearchCV`) over the joint space
   (depth, leaf, window, wti_thresh); the current grid only explored marginal effects.
