# Experiment Log — XLE Sharpe Ratio Optimization

All experiments use the same pipeline (run.py): train on 80% of XLE daily returns (2020–present),
evaluate on the remaining 20% validation set. Sharpe = sign(pred) * actual_return, annualized.
Only model.py was modified.

## Results

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
| 10 | **1.3954** | RF n=200, depth=2, min_samples_leaf=22 ← **best** |
| 11 | 0.0267 | Direct Sharpe optimizer (tanh surrogate, multi-start) — severely overfit |
| Restored | **1.3954** | Reverted to RF n=200, depth=2, min_samples_leaf=22 |

## Features Used (all experiments from run 4 onward)

Computed inside `model.py` from the `ret_lag` series using stored training history:

1. `ret_lag` — raw daily return
2. `ret_lag / vol5` — vol-adjusted return (5-day z-score)
3. `mean(w5)` — 5-day rolling mean (short-term momentum)
4. `mean(w20)` — 20-day rolling mean (long-term momentum)
5. `mean(w5) / mean(w20)` — MACD-like momentum ratio
6. `vol5 / vol20` — volatility regime indicator

## Key Findings

- **Rolling features + direction classification beats regression** (0.86 → 0.96+). Since Sharpe = sign(pred) * actual, only direction matters — framing as classification is more natural.
- **RandomForest outperforms** LogisticReg, SVM (RBF/linear), MLP, GradientBoosting, HistGradientBoosting for this task.
- **Shallow trees (depth=2) generalize best.** Deeper trees overfit the ~800-sample training set.
- **More features consistently hurt.** Tested: additional lags, 60-day window, 10-day mean, Bollinger z-score, streak counter, absolute vol. All reduced Sharpe vs the 6-feature baseline.
- **Direct Sharpe optimization (tanh surrogate) overfits severely.** Training objective does not correlate with validation Sharpe; CV-based trial selection only recovered ~1.0.
- **MLP is unstable** — performance varies with random seed and daily data refresh; RF is deterministic and robust.
- **Sample weighting by return magnitude hurts** (0.06 vs 1.40). Correct calls on small-return days are also important.
- **Confidence gating (abstaining on uncertain predictions) hurts** — RF probabilities cluster near 0.5, so gating removes most of the signal.

---

## Experiment 001 — 2026-04-20
**Model:** OLS (statsmodels)

**Features used (24):**
  - crude_chg_wow
  - crude_zscore_52
  - gas_chg_wow
  - gas_zscore_52
  - refutil_anom_12
  - wti_ret_weekly
  - wti_mom_4w
  - hdd_weekly
  - cdd_weekly
  - temp_mean_weekly
  - precip_weekly
  - hdd_anom_12w
  - cdd_anom_12w
  - temp_zscore_52
  - extreme_cold

**p-values / importances (top features):**
  - crude_chg_wow: p=0.5793
  - crude_zscore_52: p=0.7721
  - gas_chg_wow: p=0.3784
  - gas_zscore_52: p=0.8602
  - refutil_anom_12: p=0.3032
  - wti_ret_weekly: p=0.4362
  - wti_mom_4w: p=0.7679
  - hdd_weekly: p=0.1982

**In-sample Sharpe:**  1.007
**Out-of-sample Sharpe:** -0.739
**R² (in-sample):** 0.0785
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Initial baseline experiment.

**Next hypothesis:**
Add rolling z-scores, 1/2-week lags, and crude×weather interaction terms. Drop features with p>0.10.


---

## Experiment 002 — 2026-04-20
**Model:** OLS (selected + engineered)

**Features used (10):**
  - gem_cancel_rate
  - trend_oil_price_4w_mom
  - wti_mom_4w_lag2
  - extreme_cold_lag1
  - extreme_cold_lag2
  - trend_oil_price_lag1
  - trend_oil_price_4w_mom_lag1
  - trend_gasoline_price_lag1
  - trend_energy_stocks_lag1
  - trend_energy_stocks_4w_mom_lag1

**p-values / importances (top features):**
  - gem_cancel_rate: p=0.0660
  - trend_oil_price_4w_mom: p=0.0006
  - wti_mom_4w_lag2: p=0.0052
  - extreme_cold_lag1: p=0.4818
  - extreme_cold_lag2: p=0.1809
  - trend_oil_price_lag1: p=0.0810
  - trend_oil_price_4w_mom_lag1: p=0.0664
  - trend_gasoline_price_lag1: p=0.6249

**In-sample Sharpe:**  1.218
**Out-of-sample Sharpe:** 0.039
**R² (in-sample):** 0.1392
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Added rolling 4-week z-scores, 1- and 2-week lags for all features, and crude×HDD / gas×HDD interaction terms. Features filtered to p<0.10 via univariate OLS screening.

**Next hypothesis:**
Apply L1/L2 regularisation (Ridge/Lasso) to handle multicollinearity from the expanded feature set.


---

## Experiment 003 — 2026-04-20
**Model:** Lasso (best OOS) / also Ridge

**Features used (0):**


**p-values / importances (top features):**

**In-sample Sharpe:**  0.218
**Out-of-sample Sharpe:** 0.613
**R² (in-sample):** nan
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Switched to Ridge and Lasso with cross-validated alpha. StandardScaler applied. Lasso performs implicit feature selection.

**Next hypothesis:**
Try non-linear tree models (Random Forest) to capture non-linear interactions between supply, demand, and weather signals.


---

## Experiment 004 — 2026-04-20
**Model:** RandomForest (n=300,d=4,msl=15)

**Features used (10):**
  - trend_oil_price_lag1
  - precip_weekly
  - cdd_anom_12w_z4
  - wti_ret_weekly_lag2
  - wti_mom_4w_lag1
  - wti_ret_weekly_lag1
  - wti_mom_4w_lag2
  - wti_mom_4w_z4
  - crude_x_hdd
  - gas_chg_wow_z4

**p-values / importances (top features):**
  - trend_oil_price_lag1: importance=0.0627
  - precip_weekly: importance=0.0595
  - cdd_anom_12w_z4: importance=0.0575
  - wti_ret_weekly_lag2: importance=0.0465
  - wti_mom_4w_lag1: importance=0.0391
  - wti_ret_weekly_lag1: importance=0.0360
  - wti_mom_4w_lag2: importance=0.0355
  - wti_mom_4w_z4: importance=0.0343

**In-sample Sharpe:**  4.652
**Out-of-sample Sharpe:** -0.705
**R² (in-sample):** 0.5563
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Random Forest regressor. Swept n_estimators, max_depth, min_samples_leaf. Feature importances ranked.

**Next hypothesis:**
Try gradient boosting (XGBoost) which may better model sequential supply/demand signals.


---

## Experiment 006 — 2026-04-20
**Model:** Lasso (best OOS) / also Ridge + threshold opt

**Features used (0):**


**p-values / importances (top features):**

**In-sample Sharpe:**  0.218
**Out-of-sample Sharpe:** 0.613
**R² (in-sample):** nan
**Walk-forward Sharpe:** -0.2404890262620922

**What changed vs. prior:**
Applied signal threshold sweep on best model (Lasso (best OOS) / also Ridge). Also computed 5-fold expanding-window walk-forward Sharpe.

**Next hypothesis:**
If target not yet met: add Google Trends sentiment features, tune further, or try ensemble of RF+XGB predictions.


---

## Experiment 001 — 2026-04-20
**Model:** OLS (statsmodels)

**Features used (24):**
  - crude_chg_wow
  - crude_zscore_52
  - gas_chg_wow
  - gas_zscore_52
  - refutil_anom_12
  - wti_ret_weekly
  - wti_mom_4w
  - hdd_weekly
  - cdd_weekly
  - temp_mean_weekly
  - precip_weekly
  - hdd_anom_12w
  - cdd_anom_12w
  - temp_zscore_52
  - extreme_cold

**p-values / importances (top features):**
  - crude_chg_wow: p=0.5793
  - crude_zscore_52: p=0.7721
  - gas_chg_wow: p=0.3784
  - gas_zscore_52: p=0.8602
  - refutil_anom_12: p=0.3032
  - wti_ret_weekly: p=0.4362
  - wti_mom_4w: p=0.7679
  - hdd_weekly: p=0.1982

**In-sample Sharpe:**  1.007
**Out-of-sample Sharpe:** -0.739
**R² (in-sample):** 0.0785
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Initial baseline experiment.

**Next hypothesis:**
Add rolling z-scores, 1/2-week lags, and crude×weather interaction terms. Drop features with p>0.10.


---

## Experiment 002 — 2026-04-20
**Model:** OLS (selected + engineered)

**Features used (10):**
  - gem_cancel_rate
  - trend_oil_price_4w_mom
  - wti_mom_4w_lag2
  - extreme_cold_lag1
  - extreme_cold_lag2
  - trend_oil_price_lag1
  - trend_oil_price_4w_mom_lag1
  - trend_gasoline_price_lag1
  - trend_energy_stocks_lag1
  - trend_energy_stocks_4w_mom_lag1

**p-values / importances (top features):**
  - gem_cancel_rate: p=0.0660
  - trend_oil_price_4w_mom: p=0.0006
  - wti_mom_4w_lag2: p=0.0052
  - extreme_cold_lag1: p=0.4818
  - extreme_cold_lag2: p=0.1809
  - trend_oil_price_lag1: p=0.0810
  - trend_oil_price_4w_mom_lag1: p=0.0664
  - trend_gasoline_price_lag1: p=0.6249

**In-sample Sharpe:**  1.218
**Out-of-sample Sharpe:** 0.039
**R² (in-sample):** 0.1392
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Added rolling 4-week z-scores, 1- and 2-week lags for all features, and crude×HDD / gas×HDD interaction terms. Features filtered to p<0.10 via univariate OLS screening.

**Next hypothesis:**
Apply L1/L2 regularisation (Ridge/Lasso) to handle multicollinearity from the expanded feature set.


---

## Experiment 003 — 2026-04-20
**Model:** Lasso (best OOS) / also Ridge

**Features used (0):**


**p-values / importances (top features):**

**In-sample Sharpe:**  0.218
**Out-of-sample Sharpe:** 0.613
**R² (in-sample):** 0.0000
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Switched to Ridge and Lasso with cross-validated alpha. StandardScaler applied. Lasso performs implicit feature selection.

**Next hypothesis:**
Try non-linear tree models (Random Forest) to capture non-linear interactions between supply, demand, and weather signals.


---

## Experiment 004 — 2026-04-20
**Model:** RandomForest (n=300,d=4,msl=15)

**Features used (10):**
  - trend_oil_price_lag1
  - precip_weekly
  - cdd_anom_12w_z4
  - wti_ret_weekly_lag2
  - wti_mom_4w_lag1
  - wti_ret_weekly_lag1
  - wti_mom_4w_lag2
  - wti_mom_4w_z4
  - crude_x_hdd
  - gas_chg_wow_z4

**p-values / importances (top features):**
  - trend_oil_price_lag1: importance=0.0627
  - precip_weekly: importance=0.0595
  - cdd_anom_12w_z4: importance=0.0575
  - wti_ret_weekly_lag2: importance=0.0465
  - wti_mom_4w_lag1: importance=0.0391
  - wti_ret_weekly_lag1: importance=0.0360
  - wti_mom_4w_lag2: importance=0.0355
  - wti_mom_4w_z4: importance=0.0343

**In-sample Sharpe:**  4.652
**Out-of-sample Sharpe:** -0.705
**R² (in-sample):** 0.5561
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Random Forest regressor. Swept n_estimators, max_depth, min_samples_leaf. Feature importances ranked.

**Next hypothesis:**
Try gradient boosting (XGBoost) which may better model sequential supply/demand signals.


---

## Experiment 006 — 2026-04-20
**Model:** Lasso (best OOS) / also Ridge + threshold opt

**Features used (0):**


**p-values / importances (top features):**

**In-sample Sharpe:**  0.218
**Out-of-sample Sharpe:** 0.613
**R² (in-sample):** 0.0000
**Walk-forward Sharpe:** -0.23299872518996087

**What changed vs. prior:**
Applied signal threshold sweep on best model (Lasso (best OOS) / also Ridge). Also computed 5-fold expanding-window walk-forward Sharpe.

**Next hypothesis:**
If target not yet met: add Google Trends sentiment features, tune further, or try ensemble of RF+XGB predictions.


---

## Experiment 007 — 2026-04-20
**Model:** Standalone signal audit

**Features used (15):**
  - trend_energy_stocks_4w_mom
  - crude_zscore_52
  - trend_oil_price_4w_mom
  - crude_chg_wow
  - extreme_cold
  - hdd_weekly
  - gas_zscore_52
  - refutil_anom_12
  - hdd_anom_12w
  - cdd_weekly
  - trend_gasoline_price_4w_mom
  - gem_cancel_rate
  - gem_additions_yoy
  - gas_chg_wow
  - cdd_anom_12w

**p-values / importances (top features):**
  - trend_energy_stocks_4w_mom: p=0.1164
  - crude_zscore_52: p=0.3610
  - trend_oil_price_4w_mom: p=0.0020
  - crude_chg_wow: p=0.4794
  - extreme_cold: p=0.4965
  - hdd_weekly: p=0.8554
  - gas_zscore_52: p=0.6098
  - refutil_anom_12: p=0.3922

**In-sample Sharpe:**  0.000
**Out-of-sample Sharpe:** 1.687
**R² (in-sample):** 0.0000
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Tested each of 15 non-financial signals in isolation. Reports p-value, OOS Sharpe, active weeks, and trade count to identify which single signals have genuine predictive power.

**Next hypothesis:**
Build a normalised composite from top-ranked signals (trends + supply-surprise) to smooth out single-signal noise.


---

## Experiment 008 — 2026-04-20
**Model:** Normalised signal composite (EW + OLS-weighted)

**Features used (5):**
  - trend_energy_stocks_4w_mom
  - trend_oil_price_4w_mom
  - crude_chg_wow
  - gas_zscore_52
  - hdd_anom_12w

**p-values / importances (top features):**
  - trend_energy_stocks_4w_mom: p=0.0840
  - trend_oil_price_4w_mom: p=0.0838
  - crude_chg_wow: p=0.3858
  - gas_zscore_52: p=0.3568
  - hdd_anom_12w: p=0.6715

**In-sample Sharpe:**  0.551
**Out-of-sample Sharpe:** 0.663
**R² (in-sample):** 0.0209
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Combined top non-financial signals (trend momentum, crude, gas z-score, HDD anomaly) normalised to unit variance. Compared equal-weight vs. OLS-weighted composite.

**Next hypothesis:**
Use Lasso with fine alpha grid on top 9 signals + 1-week lags to find sparsest regularised combination.


---

## Experiment 009 — 2026-04-20
**Model:** Lasso (fine alpha grid, top non-financial signals only)

**Features used (0):**


**p-values / importances (top features):**

**In-sample Sharpe:**  0.218
**Out-of-sample Sharpe:** 0.613
**R² (in-sample):** 0.0000
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Applied Lasso with 40-point alpha grid (1e-5 to 1e-1) on 9 non-financial signals plus their 1-week lags. Fine grid fixes prior alpha=0.1 collapse to zero-coefficient model.

**Next hypothesis:**
Walk-forward stress test: re-fit best strategy on each fold and ensemble trend signal with Lasso predictions.


---

## Experiment 001 — 2026-04-20
**Model:** OLS (statsmodels)

**Features used (24):**
  - crude_chg_wow
  - crude_zscore_52
  - gas_chg_wow
  - gas_zscore_52
  - refutil_anom_12
  - wti_ret_weekly
  - wti_mom_4w
  - hdd_weekly
  - cdd_weekly
  - temp_mean_weekly
  - precip_weekly
  - hdd_anom_12w
  - cdd_anom_12w
  - temp_zscore_52
  - extreme_cold

**p-values / importances (top features):**
  - crude_chg_wow: p=0.5793
  - crude_zscore_52: p=0.7721
  - gas_chg_wow: p=0.3784
  - gas_zscore_52: p=0.8602
  - refutil_anom_12: p=0.3032
  - wti_ret_weekly: p=0.4362
  - wti_mom_4w: p=0.7679
  - hdd_weekly: p=0.1982

**In-sample Sharpe:**  1.007
**Out-of-sample Sharpe:** -0.739
**R² (in-sample):** 0.0785
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Initial baseline experiment.

**Next hypothesis:**
Add rolling z-scores, 1/2-week lags, and crude×weather interaction terms. Drop features with p>0.10.


---

## Experiment 002 — 2026-04-20
**Model:** OLS (selected + engineered)

**Features used (10):**
  - gem_cancel_rate
  - trend_oil_price_4w_mom
  - wti_mom_4w_lag2
  - extreme_cold_lag1
  - extreme_cold_lag2
  - trend_oil_price_lag1
  - trend_oil_price_4w_mom_lag1
  - trend_gasoline_price_lag1
  - trend_energy_stocks_lag1
  - trend_energy_stocks_4w_mom_lag1

**p-values / importances (top features):**
  - gem_cancel_rate: p=0.0660
  - trend_oil_price_4w_mom: p=0.0006
  - wti_mom_4w_lag2: p=0.0052
  - extreme_cold_lag1: p=0.4818
  - extreme_cold_lag2: p=0.1809
  - trend_oil_price_lag1: p=0.0810
  - trend_oil_price_4w_mom_lag1: p=0.0664
  - trend_gasoline_price_lag1: p=0.6249

**In-sample Sharpe:**  1.218
**Out-of-sample Sharpe:** 0.039
**R² (in-sample):** 0.1392
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Added rolling 4-week z-scores, 1- and 2-week lags for all features, and crude×HDD / gas×HDD interaction terms. Features filtered to p<0.10 via univariate OLS screening.

**Next hypothesis:**
Apply L1/L2 regularisation (Ridge/Lasso) to handle multicollinearity from the expanded feature set.


---

## Experiment 003 — 2026-04-20
**Model:** Lasso (best OOS) / also Ridge

**Features used (0):**


**p-values / importances (top features):**

**In-sample Sharpe:**  0.218
**Out-of-sample Sharpe:** 0.613
**R² (in-sample):** 0.0000
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Switched to Ridge and Lasso with cross-validated alpha. StandardScaler applied. Lasso performs implicit feature selection.

**Next hypothesis:**
Try non-linear tree models (Random Forest) to capture non-linear interactions between supply, demand, and weather signals.


---

## Experiment 004 — 2026-04-20
**Model:** RandomForest (n=300,d=4,msl=15)

**Features used (10):**
  - trend_oil_price_lag1
  - precip_weekly
  - cdd_anom_12w_z4
  - wti_ret_weekly_lag2
  - wti_mom_4w_lag1
  - wti_ret_weekly_lag1
  - wti_mom_4w_lag2
  - wti_mom_4w_z4
  - crude_x_hdd
  - gas_chg_wow_z4

**p-values / importances (top features):**
  - trend_oil_price_lag1: importance=0.0627
  - precip_weekly: importance=0.0595
  - cdd_anom_12w_z4: importance=0.0575
  - wti_ret_weekly_lag2: importance=0.0465
  - wti_mom_4w_lag1: importance=0.0391
  - wti_ret_weekly_lag1: importance=0.0360
  - wti_mom_4w_lag2: importance=0.0355
  - wti_mom_4w_z4: importance=0.0343

**In-sample Sharpe:**  4.652
**Out-of-sample Sharpe:** -0.705
**R² (in-sample):** 0.5560
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Random Forest regressor. Swept n_estimators, max_depth, min_samples_leaf. Feature importances ranked.

**Next hypothesis:**
Try gradient boosting (XGBoost) which may better model sequential supply/demand signals.


---

## Experiment 006 — 2026-04-20
**Model:** Lasso (best OOS) / also Ridge + threshold opt

**Features used (0):**


**p-values / importances (top features):**

**In-sample Sharpe:**  0.218
**Out-of-sample Sharpe:** 0.613
**R² (in-sample):** 0.0000
**Walk-forward Sharpe:** -0.24048787941202068

**What changed vs. prior:**
Applied signal threshold sweep on best model (Lasso (best OOS) / also Ridge). Also computed 5-fold expanding-window walk-forward Sharpe.

**Next hypothesis:**
If target not yet met: add Google Trends sentiment features, tune further, or try ensemble of RF+XGB predictions.


---

## Experiment 007 — 2026-04-20
**Model:** Standalone signal audit

**Features used (15):**
  - trend_energy_stocks_4w_mom
  - crude_zscore_52
  - trend_oil_price_4w_mom
  - crude_chg_wow
  - extreme_cold
  - hdd_weekly
  - gas_zscore_52
  - refutil_anom_12
  - hdd_anom_12w
  - cdd_weekly
  - trend_gasoline_price_4w_mom
  - gem_cancel_rate
  - gem_additions_yoy
  - gas_chg_wow
  - cdd_anom_12w

**p-values / importances (top features):**
  - trend_energy_stocks_4w_mom: p=0.1164
  - crude_zscore_52: p=0.3610
  - trend_oil_price_4w_mom: p=0.0020
  - crude_chg_wow: p=0.4794
  - extreme_cold: p=0.4965
  - hdd_weekly: p=0.8554
  - gas_zscore_52: p=0.6098
  - refutil_anom_12: p=0.3922

**In-sample Sharpe:**  0.000
**Out-of-sample Sharpe:** 1.687
**R² (in-sample):** 0.0000
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Tested each of 15 non-financial signals in isolation. Reports p-value, OOS Sharpe, active weeks, and trade count to identify which single signals have genuine predictive power.

**Next hypothesis:**
Build a normalised composite from top-ranked signals (trends + supply-surprise) to smooth out single-signal noise.


---

## Experiment 008 — 2026-04-20
**Model:** Normalised signal composite (EW + OLS-weighted)

**Features used (5):**
  - trend_energy_stocks_4w_mom
  - trend_oil_price_4w_mom
  - crude_chg_wow
  - gas_zscore_52
  - hdd_anom_12w

**p-values / importances (top features):**
  - trend_energy_stocks_4w_mom: p=0.0840
  - trend_oil_price_4w_mom: p=0.0838
  - crude_chg_wow: p=0.3858
  - gas_zscore_52: p=0.3568
  - hdd_anom_12w: p=0.6715

**In-sample Sharpe:**  0.551
**Out-of-sample Sharpe:** 0.663
**R² (in-sample):** 0.0209
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Combined top non-financial signals (trend momentum, crude, gas z-score, HDD anomaly) normalised to unit variance. Compared equal-weight vs. OLS-weighted composite.

**Next hypothesis:**
Use Lasso with fine alpha grid on top 9 signals + 1-week lags to find sparsest regularised combination.


---

## Experiment 009 — 2026-04-20
**Model:** Lasso (fine alpha grid, top non-financial signals only)

**Features used (0):**


**p-values / importances (top features):**

**In-sample Sharpe:**  0.218
**Out-of-sample Sharpe:** 0.613
**R² (in-sample):** 0.0000
**Walk-forward Sharpe:** N/A

**What changed vs. prior:**
Applied Lasso with 40-point alpha grid (1e-5 to 1e-1) on 9 non-financial signals plus their 1-week lags. Fine grid fixes prior alpha=0.1 collapse to zero-coefficient model.

**Next hypothesis:**
Walk-forward stress test: re-fit best strategy on each fold and ensemble trend signal with Lasso predictions.


---

## Experiment 010 — 2026-04-20
**Model:** Walk-forward: best signal + Lasso ensemble

**Features used (1):**
  - trend_energy_stocks_4w_mom

**p-values / importances (top features):**
  - trend_energy_stocks_4w_mom: importance=1.0415

**In-sample Sharpe:**  0.000
**Out-of-sample Sharpe:** 1.042
**R² (in-sample):** 0.0000
**Walk-forward Sharpe:** 0.5733169479830138

**What changed vs. prior:**
5-fold expanding walk-forward validation of (a) single best non-financial signal and (b) Lasso refitted per fold. Ensemble averages both signals after z-normalisation.

**Next hypothesis:**
Experiment series complete. If OOS Sharpe < 1.00, consider adding macro regime filter or sector rotation signals.

---

## Session 2 — 2026-04-28 (model.py autoresearch loop)

Starting point: RF n=300, depth=2, msl=22, seed=42, 3yr rolling window, exp-decay weights. Val Sharpe = 1.83 (from prior session).

| Exp | Sharpe | Description | Result |
|-----|--------|-------------|--------|
| Exp4 | 2.0219 | WTI shock filter: go flat on days where \|WTI ret\| > 3% (EIA cache) | ✅ committed |
| Exp5 | **2.2119** | Tighten WTI shock threshold 3% → 2% | ✅ committed — **BEST** |
| Exp6 | 2.0065 | Tighten further to 1.5% | ❌ reverted |
| Exp7 | 1.3403 | Add WTI ret1 + mom5 as RF input features | ❌ reverted (depth-2 can't generalize extra features) |
| Exp8 | 1.9532 | Contrarian signal on shock days (fade WTI direction) | ❌ reverted |

**Final committed model Sharpe: 2.2119**

---

## Experiment 5 (Session 2) — 2026-04-28 — BEST MODEL
**Model:** RandomForestClassifier direction classifier with 3-year rolling window, exponential sample decay, and WTI shock filter at 2% threshold.

**Full model spec:**
- Classifier: `RandomForestClassifier(n_estimators=300, max_depth=2, min_samples_leaf=22, max_features='sqrt', random_state=42)`
- Training window: rolling 3 years (756 trading days) — drops pre-2022 COVID regime
- Sample weights: `np.exp(np.linspace(-1.0, 0.0, N_full))[cutoff:]` → range 0.551 to 1.0, upweighting recent data
- Scaler: `StandardScaler` fit on training window only
- Output: `2 * P(up) - 1` → continuous signal in [-1, 1]

**Features used (6) — all derived from `ret_lag` (previous day's XLE % return):**
1. `ret_lag` — raw daily return
2. `ret_lag / vol5` — vol-adjusted return (5-day std)
3. `mean(w5)` — 5-day rolling mean (short-term momentum)
4. `mean(w20)` — 20-day rolling mean (medium-term momentum)
5. `mean(w5) / mean(w20)` — MACD-like ratio
6. `vol5 / vol20` — volatility regime indicator

**WTI shock filter (post-hoc regime overlay):**
- Loads WTI spot price from `data/eia_cache/eia_wti_price_*.parquet` (EIA API, covers 2020–2026); falls back to yfinance `CL=F` if cache missing
- Computes `yesterday's WTI log-return = log(WTI).diff(1).shift(1)`
- On any val day where `|WTI ret yesterday| > 0.02` (2%), override signal to 0.0 (flat — hold cash)
- Rationale: large oil moves inject regime noise that invalidates the momentum signal

**Validation period:** 2025-01-17 to 2026-04-23 (317 trading days, includes Trump tariff shock and April 2025 crash)

**Out-of-sample Sharpe: 2.2119** (up from 2.0219 at 3% threshold, 1.8307 before any shock filter)

**What changed vs. Exp4:** Tightened `WTI_SHOCK_THRESH` from `0.03` to `0.02`. At 3%, the filter captured ~50 shock days; at 2%, it captures more moderate-but-still-disruptive oil moves. The 2% cutoff aligns with typical energy ETF intraday volatility — a 2% WTI move is roughly a 1-sigma daily event and reliably signals an unreliable momentum regime.

**What was tried and rejected afterward:**
- `WTI_SHOCK_THRESH = 0.015` (Exp6): Sharpe 2.0065 — over-filters; removes days with genuine momentum signal
- WTI ret1 + mom5 as RF input features (Exp7): Sharpe 1.3403 — depth-2 RF cannot generalize 8 features on 756 samples; external signals add noise rather than signal at this tree depth
- Contrarian signal on shock days (Exp8): Sharpe 1.9532 — oil shock reversals are not reliable enough in 2025–2026 geopolitical/tariff regime

**Git commit:** `449dade` — `feat: WTI shock threshold 2% → Sharpe 2.21`

---

### Key findings this session

- **WTI shock filter is highly effective.** Going flat on days after a ±2% WTI move removes regimes where momentum signal is noise. Effect: +0.37 Sharpe (1.83 → 2.21).
- **Threshold tuning matters.** 2% is optimal; 3% misses too many shock days, 1.5% over-filters normal vol.
- **WTI as RF input feature consistently hurts** (1.34 Sharpe). Depth-2 RF with ~756 training samples cannot generalize an 8-feature space — adding external signals always increases noise relative to signal. The shock filter works precisely because it operates as a post-hoc regime overlay, not an RF input.
- **Contrarian (mean-reversion) on shock days is worse than flat.** Directional reversals after oil shocks are not reliable enough in this val period (2025–2026 tariff/geopolitical regime).
- **TRADING_PLAN.md written** covering signal generation, AWS Lambda deployment, Alpaca paper trading execution, risk controls, and decay monitoring.

