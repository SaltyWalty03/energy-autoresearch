# Agent Strategy

## Pipeline Flowchart

```
Weekly Retrain (Friday close)
  └─ train_all.py
       ├─ fetch XLE / XLB / XOP returns + commodity prices
       ├─ build 6 rolling features per asset
       ├─ fit t-SNE + HDBSCAN → regime map saved
       ├─ train RF per regime per asset → models/{ticker}_regime{k}.pkl
       └─ walk-forward Sharpe logged to results.tsv

Daily Job (Cloud Scheduler, market open)
  └─ run_daily.py / cloud_function/main.py
       ├─ classify today's feature vector → regime k
       ├─ load regime-specific model
       ├─ generate signal + compute half-Kelly fraction
       ├─ acquire GCS lock
       ├─ reconcile live Alpaca position
       ├─ place/hold/close order per asset
       └─ log to trades_log.csv

Intraday Bot (every 5 minutes during market hours)
  └─ run_intraday.py / cloud_function/main.py (Layer 3)
       ├─ read daily bias from compute_daily_bias()
       ├─ compute RSI(14) + MACD on 5-min Alpaca bars
       ├─ check pending/partial orders
       ├─ reconcile position via _current_position()
       └─ execute only within daily signal bounds
```

---

## Stage 1: Weekly Retrain (`train_all.py`)

**Inputs:** Yahoo Finance daily OHLCV for XLE, XLB, XOP; EIA WTI crude price cache; yfinance copper prices.
**Outputs:** `models/{ticker}_model.pkl`, `models/{ticker}_regime.pkl`, `models/{ticker}_regime{k}_model.pkl`, `models/{ticker}_meta.json`, `models/{ticker}_regime{k}_meta.json`.
**Key design decisions:** The full history (from 2020-01-01) is used for regime discovery so the t-SNE embedding sees complete market structure; the 3-year rolling window is enforced only inside `DirectionModel.fit()`, not at the data-loading level. Walk-forward Sharpe is computed per regime and stored in the meta JSON so the cloud function can read it without re-running training.

---

## Stage 2: Feature Engineering (`model.py → build_feature_matrix`)

**Inputs:** 1-D array of daily percentage returns for one asset.
**Outputs:** (n, 6) float64 array — raw return, volatility-normalised return, 5-day mean, 20-day mean, 5-to-20-day momentum ratio, short-to-long volatility ratio.
**Key design decisions:** All six features are derived from the return series alone; no external data is needed at inference time. This makes the feature vector computable from a 90-day Yahoo Finance download in under a second, which is critical for the cloud function's 300-second timeout budget.

---

## Stage 3: Regime Classification (`src/regime.py → classify_regime`)

**Inputs:** Today's (1, 6) feature vector; a `RegimeModel` loaded from GCS.
**Outputs:** Integer regime label k, or -1 on failure.
**Key design decisions:** t-SNE is never called at inference — the `RegimeModel` contains a KNN classifier (k=5) trained on the original six-feature space during `fit_regimes()`. This means regime classification is O(n_training_points) not O(t-SNE), and takes roughly 1ms. A -1 return triggers a graceful fallback to the full (non-regime) model rather than raising an exception, so a corrupted or missing regime file cannot halt trading.

---

## Stage 4: Daily Signal + Kelly Sizing

**Inputs:** Regime-specific `DirectionModel`, today's `ret_lag` series as a DataFrame, regime-specific `wf_sharpe_mean`.
**Outputs:** Directional signal in [-1, 1], Kelly fraction, dollar allocation.
**Key design decisions:** `kelly_f = max(0, (wf_sharpe_mean / √252) × 0.5)` converts the annualised Sharpe to a daily edge estimate and applies a 50% haircut (half-Kelly) to account for estimation error in the Sharpe itself. If `wf_sharpe_mean ≤ 0`, kelly_f is forced to zero and no order is placed — the model explicitly abstains when the historical edge is negative. Signals with absolute value below 0.05 are treated as flat (no trade).

---

## Stage 5: GCS Distributed Lock + Order Placement

**Inputs:** Alpaca account equity, per-asset signal and kelly_f, existing Alpaca positions.
**Outputs:** Market orders submitted to Alpaca paper-trading API; `trades_log.csv` row appended.
**Key design decisions:** A GCS blob (`trade_lock.json`) with a 90-second TTL prevents simultaneous order submission from the cloud function and any locally running scripts. Position reconciliation is direction-aware: if the current position already matches the signal direction, the asset is held without placing a new order; only direction flips or flat signals trigger closes. Total gross allocation across all three assets is capped at 95% of equity via a uniform scale factor.

---

## Stage 6: Intraday Confirmation (Layer 3)

**Inputs:** 5-minute Alpaca bar data (60 bars lookback), daily directional bias per asset.
**Outputs:** Entry signal — `long`, `short`, or `flat` — per asset.
**Key design decisions:** The intraday layer does not override the daily bias; it only refines entry timing within the bias direction. Long entries require RSI(14) < 35 AND a MACD histogram sign flip from negative to positive; short entries require RSI > 65 AND a histogram flip from positive to negative. This means the cloud function may fire 96 times per day but will place an order only once per asset per day at most, on the first bar where the intraday confirmation aligns with the daily bias.
