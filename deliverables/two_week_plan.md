# Two-Week Plan

## Week 1 — Stabilize and Collect

| Day | Task |
|---|---|
| 1 | Run `two_week_plan_prompt.md` — fix 3 bugs, add trade logging, generate deliverables |
| 2 | Retrain all models via `train_all.py`, confirm regime plots render cleanly |
| 3–5 | Let paper trading run untouched — collect 3 days of `trades_log.csv` data |

## Week 2 — Measure and Present

| Day | Task |
|---|---|
| 6–7 | Collect 2 more trading days, compute live Sharpe from `trades_log.csv` |
| 8 | Write `scripts/live_performance.py` — loads `trades_log.csv`, computes annualised Sharpe, max drawdown, win rate per asset, saves `results/live_performance.md` |
| 9 | Final presentation prep — compile regime visuals, ablation table, live vs backtested Sharpe comparison |
| 10 | Buffer / polish |

## Not in Scope (Deliberate)

- Real capital deployment
- Options or derivatives
- Intraday regime switching
- COMTRADE feature integration
- Sentiment or alternative data beyond EIA

## Scope Boundary

The boundary was drawn here because paper trading over a two-week window is the minimum viable test of whether the architecture executes correctly end-to-end — from regime classification through Kelly sizing to order placement — without risking real capital on a model that has not yet demonstrated live-market Sharpe. The natural next research direction after the semester is replacing the single-return-series feature set with a richer cross-asset panel (XLE + WTI futures curve shape + EIA storage delta + HDD/CDD anomaly), which would give the regime layer genuinely distinct fundamental signals to cluster on rather than pure momentum-volatility patterns.
