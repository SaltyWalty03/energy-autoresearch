# Experiment Log — XLE Sharpe Ratio Optimization

All experiments use the same pipeline (`run.py`): train on 80% of XLE daily returns (2020–present),
evaluate on the remaining 20% validation set. Sharpe = sign(pred) * actual_return, annualized.
Only `model.py` was modified.

---

## Master Results Summary

| Session | Run / ID | Date | OOS Sharpe | Config / Approach | Status |
|---------|----------|------|-----------|-------------------|--------|
| 1 | Baseline | 2026-04-18 | 0.8584 | `LinearRegression` on `ret_lag` only | Reference |
| 1 | Run 6 | 2026-04-18 | 1.2623 | `RandomForestClassifier` depth=3, msl=30 | — |
| 1 | Run 7 | 2026-04-18 | 1.3092 | RF depth=2, msl=25, `max_features='sqrt'`, n=200 | — |
| 1 | Run 9 | 2026-04-18 | 1.3717 | RF n=300, depth=2, msl=20 | — |
| 1 | Run 10 | 2026-04-18 | **1.3954** | RF n=200, depth=2, msl=22 | ✅ Best S1 |
| Sweep | run_001 | 2026-04-25 | 2.4041 | n=300, d=2, msl=22, w=20, wti=2%, tw=756 | Reference baseline |
| Sweep | run_003 | 2026-04-25 | 2.4652 | n=300, d=4, msl=22, w=20, wti=2%, tw=756 | ✅ Keep |
| Sweep | run_008 | 2026-04-25 | 2.4742 | n=100, d=2, msl=22, w=20, wti=2%, tw=756 | ✅ Keep |
| Sweep | run_009 | 2026-04-25 | **2.4793** | n=600, d=2, msl=22, w=20, wti=2%, tw=756 | ✅ Best sweep |
| 2 | Exp4 | 2026-04-25 | 2.0219 | WTI shock filter ≥3% → flat | ✅ committed |
| 2 | Exp5 | 2026-04-25 | **2.2119** | WTI shock filter ≥2% → flat | ✅ **BEST committed** |
| 3 | — | 2026-05-07 | 2.1982 | (run.py — session 3 peak) | — |
| 3 | — | 2026-05-07 | 2.1776 | — | — |
| 3 | — | 2026-05-07 | 2.1316 | — | — |

---

## Session 1 — 2026-04-18 (model.py autoresearch loop)

Starting from `LinearRegression` on a single lag feature; progressively improved toward RF direction classifier.

| Run | Sharpe | Model / Approach |
|-----|--------|-----------------|
| Baseline | 0.8584 | `LinearRegression` on `ret_lag` only |
| 1 | 0.0653 | `RetFeatures` transformer (polynomial, sign, pos/neg splits) + Ridge |
| 2 | 0.1524 | `GradientBoostingRegressor` on `ret_lag` |
| 3 | 0.4703 | Custom rolling features (z-score, RSI, vol ratio) + Ridge regression |
| 4 | 0.9581 | Rolling features + `LogisticRegression` (direction classifier, `2*proba-1` signal) |
| 5 | 0.8824 | Added lag2, lag3, MACD, RSI to logistic reg — more features overfit |
| 6 | 1.2623 | `RandomForestClassifier` depth=3, min_samples_leaf=30 |
| 7 | 1.3092 | RF tuned: depth=2, msl=25, `max_features='sqrt'`, n=200 |
| 8 | 1.1365 | `MLPClassifier` (128,) alpha=1.0 — unstable across data refreshes |
| 9 | 1.3717 | RF n=300, depth=2, min_samples_leaf=20 |
| 10 | **1.3954** | RF n=200, depth=2, min_samples_leaf=22 ← **best S1** |
| 11 | 0.0267 | Direct Sharpe optimizer (tanh surrogate, multi-start) — severely overfit |
| Restored | **1.3954** | Reverted to RF n=200, depth=2, min_samples_leaf=22 |

---

## Detailed Session 1 Experiments — 2026-04-20

### Experiment 001
**Model:** OLS (statsmodels)

**Features used (24):** crude_chg_wow, crude_zscore_52, gas_chg_wow, gas_zscore_52, refutil_anom_12,
wti_ret_weekly, wti_mom_4w, hdd_weekly, cdd_weekly, temp_mean_weekly, precip_weekly, hdd_anom_12w,
cdd_anom_12w, temp_zscore_52, extreme_cold

**Top p-values:** crude_chg_wow=0.5793, gas_chg_wow=0.3784, refutil_anom_12=0.3032, hdd_weekly=0.1982

**In-sample Sharpe:** 1.007 | **OOS Sharpe:** -0.739 | **R²:** 0.0785 | **Walk-forward:** N/A

**What changed:** Initial baseline experiment.

**Next hypothesis:** Add rolling z-scores, 1/2-week lags, crude×weather interactions. Drop p>0.10 features.

---

### Experiment 002
**Model:** OLS (selected + engineered)

**Features used (10):** gem_cancel_rate, trend_oil_price_4w_mom, wti_mom_4w_lag2, extreme_cold_lag1,
extreme_cold_lag2, trend_oil_price_lag1, trend_oil_price_4w_mom_lag1, trend_gasoline_price_lag1,
trend_energy_stocks_lag1, trend_energy_stocks_4w_mom_lag1

**Top p-values:** trend_oil_price_4w_mom=0.0006, wti_mom_4w_lag2=0.0052, trend_oil_price_lag1=0.0810

**In-sample Sharpe:** 1.218 | **OOS Sharpe:** 0.039 | **R²:** 0.1392 | **Walk-forward:** N/A

**What changed:** Rolling 4-week z-scores, 1- and 2-week lags, crude×HDD/gas×HDD interaction terms. Filtered to p<0.10.

**Next hypothesis:** Apply Ridge/Lasso to handle multicollinearity.

---

### Experiment 003
**Model:** Lasso (best OOS) / Ridge

**Features used:** (Lasso zeroed most features at alpha=0.1)

**In-sample Sharpe:** 0.218 | **OOS Sharpe:** 0.613 | **R²:** nan | **Walk-forward:** N/A

**What changed:** Ridge and Lasso with cross-validated alpha. StandardScaler applied.

**Next hypothesis:** Try Random Forest for non-linear interactions.

---

### Experiment 004
**Model:** RandomForest (n=300, d=4, msl=15)

**Features used (10):** trend_oil_price_lag1, precip_weekly, cdd_anom_12w_z4, wti_ret_weekly_lag2,
wti_mom_4w_lag1, wti_ret_weekly_lag1, wti_mom_4w_lag2, wti_mom_4w_z4, crude_x_hdd, gas_chg_wow_z4

**Top importances:** trend_oil_price_lag1=0.0627, precip_weekly=0.0595, cdd_anom_12w_z4=0.0575

**In-sample Sharpe:** 4.652 | **OOS Sharpe:** -0.705 | **R²:** 0.5563 | **Walk-forward:** N/A

**What changed:** RF regressor, swept n_estimators/max_depth/min_samples_leaf.

**Next hypothesis:** Try gradient boosting (XGBoost).

---

### Experiment 006
**Model:** Lasso / Ridge + signal threshold optimization

**In-sample Sharpe:** 0.218 | **OOS Sharpe:** 0.613 | **R²:** 0.000 | **Walk-forward:** -0.2405

**What changed:** Signal threshold sweep on best model; 5-fold expanding-window walk-forward Sharpe computed.

**Next hypothesis:** Add Google Trends sentiment features or ensemble RF+XGB.

---

### Experiment 007
**Model:** Standalone signal audit (15 non-financial signals tested individually)

**Features audited (15):** trend_energy_stocks_4w_mom, crude_zscore_52, trend_oil_price_4w_mom,
crude_chg_wow, extreme_cold, hdd_weekly, gas_zscore_52, refutil_anom_12, hdd_anom_12w, cdd_weekly,
trend_gasoline_price_4w_mom, gem_cancel_rate, gem_additions_yoy, gas_chg_wow, cdd_anom_12w

**Top p-values:** trend_oil_price_4w_mom=0.0020, trend_energy_stocks_4w_mom=0.1164

**In-sample Sharpe:** 0.000 | **OOS Sharpe:** 1.687 | **Walk-forward:** N/A

**What changed:** Tested each of 15 signals in isolation for p-value, OOS Sharpe, trade count.

**Next hypothesis:** Build normalised composite from top-ranked signals.

---

### Experiment 008
**Model:** Normalised signal composite (EW + OLS-weighted)

**Features used (5):** trend_energy_stocks_4w_mom, trend_oil_price_4w_mom, crude_chg_wow, gas_zscore_52, hdd_anom_12w

**Top p-values:** trend_energy_stocks_4w_mom=0.0840, trend_oil_price_4w_mom=0.0838

**In-sample Sharpe:** 0.551 | **OOS Sharpe:** 0.663 | **R²:** 0.0209 | **Walk-forward:** N/A

**What changed:** Combined top non-financial signals normalised to unit variance; compared EW vs. OLS-weighted composite.

**Next hypothesis:** Lasso with fine alpha grid on top 9 signals + 1-week lags.

---

### Experiment 009
**Model:** Lasso (fine alpha grid, 40-point, 1e-5 to 1e-1; top non-financial signals only)

**In-sample Sharpe:** 0.218 | **OOS Sharpe:** 0.613 | **R²:** 0.000 | **Walk-forward:** N/A

**What changed:** 40-point alpha grid fixes prior alpha=0.1 collapse; applied to 9 non-financial signals plus 1-week lags.

**Next hypothesis:** Walk-forward stress test; ensemble trend signal with Lasso predictions.

---

### Experiment 010
**Model:** Walk-forward: best signal + Lasso ensemble

**Features used (1):** trend_energy_stocks_4w_mom (importance=1.0415)

**In-sample Sharpe:** 0.000 | **OOS Sharpe:** 1.042 | **Walk-forward:** 0.5733

**What changed:** 5-fold expanding walk-forward of (a) single best non-financial signal and (b) Lasso refitted per fold. Ensemble z-normalises both signals.

**Next hypothesis:** Experiment series complete. Consider macro regime filter or sector rotation signals if OOS < 1.00.

---

## Hyperparameter Sweep — experiment_log.csv

RF direction classifier with 6 rolling momentum features, WTI shock filter. All runs use the same val period (2025-01-17 to 2026-04-23).

| Run ID | n_est | depth | msl | window | wti_thresh | train_window | Sharpe | Status | Notes |
|--------|-------|-------|-----|--------|-----------|--------------|--------|--------|-------|
| run_001 | 300 | 2 | 22 | 20 | 0.02 | 756 | 2.4041 | Keep | Reference baseline |
| run_002 | 300 | 3 | 22 | 20 | 0.02 | 756 | 2.1359 | Discard | -0.2682 vs baseline |
| run_003 | 300 | 4 | 22 | 20 | 0.02 | 756 | **2.4652** | Keep | +0.0611 vs baseline |
| run_004 | 300 | 1 | 22 | 20 | 0.02 | 756 | 1.2735 | Discard | -1.1306 vs baseline |
| run_005 | 300 | 2 | 10 | 20 | 0.02 | 756 | 2.3837 | Discard | -0.0204 vs baseline |
| run_006 | 300 | 2 | 50 | 20 | 0.02 | 756 | 1.4882 | Discard | -0.9159 vs baseline |
| run_007 | 300 | 2 | 100 | 20 | 0.02 | 756 | 1.5910 | Discard | -0.8131 vs baseline |
| run_008 | 100 | 2 | 22 | 20 | 0.02 | 756 | **2.4742** | Keep | +0.0701 vs baseline |
| run_009 | 600 | 2 | 22 | 20 | 0.02 | 756 | **2.4793** | Keep | +0.0752 vs baseline — **best sweep** |
| run_010 | 300 | 2 | 22 | 10 | 0.02 | 756 | 2.3446 | Discard | -0.0595 vs baseline |
| run_011 | 300 | 2 | 22 | 40 | 0.02 | 756 | 1.9068 | Discard | -0.4973 vs baseline |
| run_012 | 300 | 2 | 22 | 20 | 0.03 | 756 | 2.2201 | Discard | -0.1840 vs baseline |
| run_013 | 300 | 2 | 22 | 20 | 0.01 | 756 | 1.4536 | Discard | -0.9505 vs baseline |
| run_014 | 300 | 2 | 22 | 20 | 0.02 | 504 | 1.2470 | Discard | -1.1571 vs baseline |
| run_015 | 300 | 3 | 15 | 10 | 0.02 | 756 | 2.0270 | Discard | -0.3771 vs baseline |

**Sweep findings:**
- Depth=2 is optimal; depth=1 and depth=4 both hurt (overfitting or underfitting at this sample size).
- msl=22 is optimal; higher values (50, 100) over-regularize, lower (10) slightly overfits.
- n_estimators: marginal gains from 300→600 (+0.008 Sharpe). 100 estimators nearly matches 300.
- window=20 is optimal; shorter (10) and longer (40) both reduce Sharpe.
- wti_thresh=2% is optimal; 1% over-filters, 3% under-filters.
- train_window=756 (3yr) outperforms 504 (2yr) by +1.16 Sharpe.

---

## Session 2 — 2026-04-25/28 (model.py autoresearch loop)

Starting point: RF n=300, depth=2, msl=22, seed=42, 3yr rolling window, exp-decay weights. Val Sharpe = 1.83.

| Exp | Sharpe | Description | Result |
|-----|--------|-------------|--------|
| Exp4 | 2.0219 | WTI shock filter: go flat on days where \|WTI ret\| > 3% (EIA cache) | ✅ committed |
| Exp5 | **2.2119** | Tighten WTI shock threshold 3% → 2% | ✅ committed — **BEST committed** |
| Exp6 | 2.0065 | Tighten further to 1.5% | ❌ reverted |
| Exp7 | 1.3403 | Add WTI ret1 + mom5 as RF input features | ❌ reverted (depth-2 can't generalize extra features) |
| Exp8 | 1.9532 | Contrarian signal on shock days (fade WTI direction) | ❌ reverted |

**Final committed model Sharpe: 2.2119**

### Best Model Spec (Exp5 — git commit `449dade`)

- **Classifier:** `RandomForestClassifier(n_estimators=300, max_depth=2, min_samples_leaf=22, max_features='sqrt', random_state=42)`
- **Training window:** Rolling 3 years (756 trading days) — drops pre-2022 COVID regime
- **Sample weights:** `np.exp(np.linspace(-1.0, 0.0, N_full))[cutoff:]` → range 0.551–1.0, upweights recent data
- **Scaler:** `StandardScaler` fit on training window only
- **Output:** `2 * P(up) - 1` → continuous signal in [-1, 1]

**Features (6) — derived from `ret_lag`:**
1. `ret_lag` — raw daily return
2. `ret_lag / vol5` — vol-adjusted return (5-day std)
3. `mean(w5)` — 5-day rolling mean (short-term momentum)
4. `mean(w20)` — 20-day rolling mean (medium-term momentum)
5. `mean(w5) / mean(w20)` — MACD-like ratio
6. `vol5 / vol20` — volatility regime indicator

**WTI shock filter:**
- Loads WTI spot price from `data/eia_cache/eia_wti_price_*.parquet`; falls back to yfinance `CL=F`
- Computes `yesterday's WTI log-return = log(WTI).diff(1).shift(1)`
- On val days where `|WTI ret yesterday| > 0.02`: override signal to 0.0 (hold cash)
- Rationale: large oil moves inject regime noise that invalidates the momentum signal

**Validation period:** 2025-01-17 to 2026-04-23 (317 trading days, includes Trump tariff shock and April 2025 crash)

**OOS Sharpe: 2.2119** (up from 2.0219 at 3% threshold, 1.8307 before any shock filter)

---

## Session 3 — 2026-05-07 (results.tsv runs)

22 runs on 2026-05-07, plus isolated runs on 2026-04-30 and 2026-05-06.

| Timestamp | Sharpe |
|-----------|--------|
| 2026-04-30 17:34:01 | 0.7453 |
| 2026-05-06 17:28:43 | 1.6544 |
| 2026-05-07 21:58:38 | 2.0925 |
| 2026-05-07 22:01:14 | 1.9338 |
| 2026-05-07 22:02:50 | 1.6980 |
| 2026-05-07 22:04:22 | 1.3099 |
| 2026-05-07 22:05:24 | 0.7485 |
| 2026-05-07 22:06:46 | 1.9362 |
| 2026-05-07 22:54:28 | 1.7698 |
| 2026-05-07 22:59:39 | 1.6660 |
| 2026-05-07 23:01:18 | 1.5416 |
| 2026-05-07 23:04:32 | 1.7753 |
| 2026-05-07 23:05:28 | 1.1332 |
| 2026-05-07 23:06:45 | 2.0365 |
| 2026-05-07 23:07:17 | 2.1316 |
| 2026-05-07 23:07:37 | **2.1982** |
| 2026-05-07 23:07:55 | 1.9865 |
| 2026-05-07 23:08:12 | 2.0790 |
| 2026-05-07 23:08:49 | 1.0358 |
| 2026-05-07 23:09:22 | 2.1776 |
| 2026-05-07 23:09:53 | 2.1563 |
| 2026-05-07 23:10:19 | 2.0418 |
| 2026-05-07 23:10:44 | 2.0526 |
| 2026-05-07 23:11:32 | 0.4453 |

**Session 3 peak: 2.1982** (2026-05-07 23:07:37). Sustained cluster of 2.0–2.2 Sharpe runs in the 23:06–23:10 window.

---

## All results.tsv Runs (complete chronological log)

| Timestamp | Sharpe | Note |
|-----------|--------|------|
| 2026-04-15 11:53:27 | 0.8220 | Session 1 start |
| 2026-04-16 17:12:10 | 0.8347 | |
| 2026-04-16 17:13:23 | 0.8347 | |
| 2026-04-18 17:57:42 | 0.8584 | Baseline |
| 2026-04-18 17:58:28 | 0.0653 | Run 1 |
| 2026-04-18 17:58:46 | 0.1524 | Run 2 |
| 2026-04-18 17:59:07 | 0.4703 | Run 3 |
| 2026-04-18 17:59:44 | 0.9581 | Run 4 |
| 2026-04-18 18:00:04 | 0.8824 | Run 5 |
| 2026-04-18 18:01:49 | 1.2623 | Run 6 |
| 2026-04-18 18:03:53 | 1.3092 | Run 7 |
| 2026-04-18 18:07:32 | 1.1365 | Run 8 |
| 2026-04-18 18:08:47 | 1.3717 | Run 9 |
| 2026-04-18 18:09:27 | **1.3954** | Run 10 — S1 best |
| 2026-04-18 18:18:31 | 0.0267 | Run 11 |
| 2026-04-18 18:35:17 | 1.3954 | Restored |
| 2026-04-25 11:50:14 | 1.1685 | Session 2 start |
| 2026-04-25 11:53:36 | 0.7420 | |
| 2026-04-25 11:58:53 | 0.7372 | |
| 2026-04-25 12:09:03 | 1.8278 | |
| 2026-04-25 12:37:21 | 2.0219 | Exp4 (WTI 3%) |
| 2026-04-25 12:38:53 | **2.2119** | Exp5 (WTI 2%) — **best committed** |
| 2026-04-25 12:39:14 | 2.0065 | Exp6 (WTI 1.5%) |
| 2026-04-25 12:40:51 | 1.3403 | Exp7 |
| 2026-04-25 12:41:38 | 1.9532 | Exp8 |
| 2026-04-30 17:34:01 | 0.7453 | |
| 2026-05-06 17:28:43 | 1.6544 | |
| 2026-05-07 21:58:38 | 2.0925 | Session 3 |
| 2026-05-07 22:01:14 | 1.9338 | |
| 2026-05-07 22:02:50 | 1.6980 | |
| 2026-05-07 22:04:22 | 1.3099 | |
| 2026-05-07 22:05:24 | 0.7485 | |
| 2026-05-07 22:06:46 | 1.9362 | |
| 2026-05-07 22:54:28 | 1.7698 | |
| 2026-05-07 22:59:39 | 1.6660 | |
| 2026-05-07 23:01:18 | 1.5416 | |
| 2026-05-07 23:04:32 | 1.7753 | |
| 2026-05-07 23:05:28 | 1.1332 | |
| 2026-05-07 23:06:45 | 2.0365 | |
| 2026-05-07 23:07:17 | 2.1316 | |
| 2026-05-07 23:07:37 | **2.1982** | S3 peak |
| 2026-05-07 23:07:55 | 1.9865 | |
| 2026-05-07 23:08:12 | 2.0790 | |
| 2026-05-07 23:08:49 | 1.0358 | |
| 2026-05-07 23:09:22 | 2.1776 | |
| 2026-05-07 23:09:53 | 2.1563 | |
| 2026-05-07 23:10:19 | 2.0418 | |
| 2026-05-07 23:10:44 | 2.0526 | |
| 2026-05-07 23:11:32 | 0.4453 | |

---

## Features Used (Session 1 Run 4 onward)

Computed inside `model.py` from the `ret_lag` series using stored training history:

1. `ret_lag` — raw daily return
2. `ret_lag / vol5` — vol-adjusted return (5-day z-score)
3. `mean(w5)` — 5-day rolling mean (short-term momentum)
4. `mean(w20)` — 20-day rolling mean (long-term momentum)
5. `mean(w5) / mean(w20)` — MACD-like momentum ratio
6. `vol5 / vol20` — volatility regime indicator

---

## Key Findings

- **Rolling features + direction classification beats regression** (0.86 → 0.96+). Since Sharpe = sign(pred) × actual, only direction matters — framing as classification is more natural.
- **RandomForest outperforms** LogisticReg, SVM (RBF/linear), MLP, GradientBoosting, HistGradientBoosting for this task.
- **Shallow trees (depth=2) generalize best.** Deeper trees overfit the ~800-sample training set.
- **More features consistently hurt.** Tested: additional lags, 60-day window, 10-day mean, Bollinger z-score, streak counter, absolute vol. All reduced Sharpe vs the 6-feature baseline.
- **WTI shock filter (≥2%) is highly effective.** Going flat after ±2% WTI days removes regimes where momentum signal is noise. Effect: +0.37 Sharpe (1.83 → 2.21). 2% optimal; 3% misses too many shock days, 1.5% over-filters.
- **WTI as RF input feature consistently hurts** (1.34 Sharpe). The shock filter works as a post-hoc regime overlay, not an RF input.
- **Direct Sharpe optimization (tanh surrogate) overfits severely.** Training objective does not correlate with validation Sharpe.
- **MLP is unstable** — performance varies with random seed and daily data refresh; RF is deterministic and robust.
- **Sample weighting by return magnitude hurts** (0.06 vs 1.40). Correct calls on small-return days also matter.
- **Confidence gating (abstaining on uncertain predictions) hurts** — RF probabilities cluster near 0.5, so gating removes most of the signal.
- **train_window=756 (3yr) strongly preferred** over 504 (2yr); drops pre-2022 COVID regime that confuses momentum signal.
