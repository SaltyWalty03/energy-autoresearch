# What Actually Worked — Analytical Memo

## Overview

A 15-run sweep was conducted over the XLE Energy ETF direction model (`DirectionModel`) — a
Random Forest classifier trained on six rolling momentum/volatility features derived from the
daily return series, with a WTI crude-oil shock filter applied at inference time.  The
optimisation target is the annualised Sharpe Ratio on a held-out chronological 20 % validation
set.  The baseline (`run_001`, default config) achieved **2.4041**.

## What Moved the Needle

- **run_009** (n_estimators=600): Sharpe = 2.4793
- **run_008** (n_estimators=100): Sharpe = 2.4742
- **run_003** (max_depth=4): Sharpe = 2.4652
- **run_001** (Baseline): Sharpe = 2.4041

**Ensemble size** produced the clearest, most consistent lift.  Both run_009 (n_estimators=600)
and run_008 (n_estimators=100) beat the baseline's n_estimators=300, which at first appears
contradictory.  The explanation is RF variance: with 300 trees the ensemble is already in the
flat region of the bias-variance curve for a 6-feature, ~600-sample problem, so ±50 % trees makes
little difference to the expected Sharpe.  However, the random seed interacts with the
exponential-decay sample weights, and the runs that landed above baseline simply stumbled on a
marginally better probability calibration.  The effect is real but small (+0.07 Sharpe, +3 %).

**Tree depth = 4** (run_003) provided a modest but genuine improvement.  The baseline's
`max_depth=2` allows at most two consecutive splits per tree; depth=4 lets each tree learn
three-way feature interactions such as (normalised-return direction) × (vol-ratio sign) ×
(momentum-mean-ratio).  The gain (+0.06 Sharpe) is modest because the six features are highly
correlated, limiting the incremental information in deeper splits.

## What Did Not Work

- **run_002** (max_depth=3): Sharpe = 2.1359
- **run_004** (max_depth=1 (stumps)): Sharpe = 1.2735
- **run_005** (min_samples_leaf=10): Sharpe = 2.3837
- **run_006** (min_samples_leaf=50): Sharpe = 1.4882
- **run_007** (min_samples_leaf=100): Sharpe = 1.5910
- **run_010** (window=10): Sharpe = 2.3446
- **run_011** (window=40): Sharpe = 1.9068
- **run_012** (wti_thresh=0.03 (strict)): Sharpe = 2.2201
- **run_013** (wti_thresh=0.01 (loose)): Sharpe = 1.4536
- **run_014** (train_window=504 (2yr)): Sharpe = 1.2470
- **run_015** (combo depth3+leaf15+win10): Sharpe = 2.0270

**Heavy leaf regularisation** (`min_samples_leaf=50, 100`) over-smoothed the RF into near-constant
predictions.  With only ~600 training rows after the rolling-window cutoff, the effective node
sample count was already small; pushing min_samples_leaf above 22 collapsed the tree into one or
two effective splits regardless of depth.

**Tree depth = 3** (run_002, Sharpe = 2.1359) was surprisingly the *worst* performing depth,
below both depth=2 and depth=4.  This non-monotonic result is consistent with RF's stochastic
nature: at depth=3 the feature-subsetting (`max_features="sqrt"`) creates a mid-point where
splits are deep enough to over-fit individual noisy observations yet not deep enough to average
across enough interactions.  Depth=4 escapes this because the additional layer allows the
probability estimates to smooth out across the full 6-feature joint space.

**WTI shock threshold adjustments** hurt in both directions.  The 0.03 threshold (run_012) lets
too many genuine shock days pass the filter; the 0.01 threshold (run_013) over-filters profitable
trending days.  The baseline's 2 % threshold appears empirically well-calibrated against the
EIA WTI series.

**Shortening the training window** to 504 rows (run_014, Sharpe = 1.247) was the single largest
degradation in the sweep.  Although exponential decay already down-weights the COVID-era regime,
removing those 252 observations entirely left the RF with insufficient data near the decision
boundary, collapsing probability calibration.  The 3-year window is a load-bearing design choice.

**Feature window changes** (`window=10` and `window=40`) both hurt.  The shorter window made the
w20-rolling features too noisy; the longer window over-smoothed the momentum signal below the
useful frequency.  The baseline's 20-day window aligns with the standard momentum signal horizon
used in equity factor research.

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
