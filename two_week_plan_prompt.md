# Claude Code Prompt: Fix Bugs + Generate Assignment Deliverables

Read these files before changing or writing anything:
`run_daily.py`, `run_experiments.py`, `scripts/plot_regimes.py`,
`prepare.py`, `program.md`, `what_actually_worked.md`, `results.tsv`,
`TRADING_PLAN.md`, `model.py`

Think through the full project state before touching any file.

---

## Step 1: Fix Three Bugs

**Bug 1 — run_daily.py unconditional position close**
Only close a position for an asset if today's signal direction differs
from the currently held direction. If signal is flat, close it. If signal
matches the current position, do nothing. Do not close positions for
assets whose signal has not changed.

**Bug 2 — run_experiments.py grid sweep uses wrong metric**
Replace the single-holdout Sharpe with `walk_forward_sharpe()` from
`prepare.py` as the primary sort key in the grid sweep. Update results.tsv
columns so `wf_sharpe_mean` is the metric the autoresearch loop optimizes
against. Keep the existing trajectory chart logic unchanged.

**Bug 3 — regime plots never confirmed**
Run `scripts/plot_regimes.py` for XLE. If it errors, fix it and re-run.
Confirm both files exist and are non-empty:
- `results/regime_embedding.png`
- `results/regime_returns.png`

---

## Step 2: Add Trade Logging

Add a `trades_log.csv` to the project root. Every time `run_daily.py`
places or skips an order, append one row with these columns:

```
timestamp, asset, regime, n_regimes, signal, kelly_fraction, action,
fill_price, shares, account_equity
```

- `action` is one of: `buy`, `sell`, `close`, `hold`, `flat`
- `fill_price` should be the ask price used for sizing (already fetched)
- If an order is skipped due to the GCS lock or flat signal, still log
  the row with the appropriate action
- Write a `src/trade_logger.py` module with a single function
  `log_trade(**kwargs)` that appends to `trades_log.csv`, creating it
  with headers if it does not exist
- Import and call `log_trade()` from `run_daily.py` after every order
  decision

---

## Step 3: Generate Assignment Deliverables

Save all four files to `deliverables/`. Ground every claim in actual
values from `results.tsv` and the project files — do not invent numbers.

---

### `deliverables/project_statement.md`

400-500 words. Cover:
- What the system does: self-improving quantitative research pipeline
  predicting daily direction for XLE, XLB, and XOP using momentum
  features and commodity shock filters
- Core model: RandomForestClassifier with 6 rolling features, 3-year
  training window, exponential sample decay, WTI/copper shock filter
- Regime layer: t-SNE embedding + HDBSCAN clustering, per-regime RF
  models, KNN inference, optimal config perplexity=30 n_clusters=5
- Execution: Alpaca paper trading, half-Kelly sizing from live account
  equity, GCS distributed lock, Cloud Scheduler deployment
- Autoresearch loop: grid search over hyperparameters, walk-forward
  Sharpe as selection criterion, weekly retraining cycle
- What is explicitly not in scope: real capital, options, COMTRADE
  features, intraday regime switching

---

### `deliverables/agent_strategy.md`

Document the full pipeline as an agent loop. Open with this flowchart:

```
Weekly Retrain (Friday close)
  └─ train_all.py
       ├─ fetch XLE / XLB / XOP returns + commodity prices
       ├─ build 6 rolling features per asset
       ├─ fit t-SNE + HDBSCAN → regime map saved
       ├─ train RF per regime per asset → models/{ticker}_regime{k}.pkl
       └─ walk-forward Sharpe logged to results.tsv

Daily Job (Cloud Scheduler, market open)
  └─ run_daily.py
       ├─ classify today's feature vector → regime k
       ├─ load regime-specific model
       ├─ generate signal + compute half-Kelly fraction
       ├─ acquire GCS lock
       ├─ reconcile live Alpaca position
       ├─ place/hold/close order per asset
       └─ log to trades_log.csv

Intraday Bot (every N minutes during market hours)
  └─ run_intraday.py
       ├─ read daily bias from compute_daily_bias()
       ├─ compute RSI + MACD on intraday bars
       ├─ check pending/partial orders
       ├─ reconcile position via _current_position()
       └─ execute only within daily signal bounds
```

Then document each stage in 2-3 sentences describing inputs, outputs,
and key design decisions.

---

### `deliverables/ablation_table.md`

Build this table using actual values from results.tsv. For configurations
with multiple runs, report the mean wf_sharpe_mean across those runs.

| Configuration | Key Change | WF Sharpe Mean | Best Val Sharpe | N Runs |
|---|---|---|---|---|
| Baseline RF | n_estimators=300, depth=2 | | | |
| Tuned RF | min_samples_leaf=22, n=600 | | | |
| + WTI Shock Filter | crude return > 2% → flat | | | |
| + Exponential Decay | sample weights exp(-0.3→0) | | | |
| + 3yr Rolling Window | drop pre-2021 regime | | | |
| XGBoost (rejected) | XGBClassifier | -0.40 | 0.54 | 2 |
| + Regime Layer (RF) | t-SNE + HDBSCAN, k=5 | | | |

Below the table write 4-5 sentences covering:
- Which single component contributed most to Sharpe improvement
- Why XGBoost underperformed (low signal-to-noise, RF bagging more robust)
- What the regime layer adds and why perplexity=30 / n_clusters=5 won
- One honest limitation of the current ablation methodology

---

### `deliverables/two_week_plan.md`

**Week 1 — Stabilize and Collect**

| Day | Task |
|---|---|
| 1 | Run this prompt — fix 3 bugs, add trade logging, generate deliverables |
| 2 | Retrain all models via train_all.py, confirm regime plots render cleanly |
| 3-5 | Let paper trading run untouched — collect 3 days of trades_log.csv data |

**Week 2 — Measure and Present**

| Day | Task |
|---|---|
| 6-7 | Collect 2 more trading days, compute live Sharpe from trades_log.csv |
| 8 | Write scripts/live_performance.py — loads trades_log.csv, computes annualised Sharpe, max drawdown, win rate per asset, saves results/live_performance.md |
| 9 | Final presentation prep — compile regime visuals, ablation table, live vs backtested Sharpe comparison |
| 10 | Buffer / polish |

**Not in scope (deliberate)**
- Real capital deployment
- Options or derivatives
- Intraday regime switching
- COMTRADE feature integration
- Sentiment or alternative data beyond EIA

Close with 2 sentences explaining why the scope boundary was drawn here
and what the natural next research direction would be after the semester.
