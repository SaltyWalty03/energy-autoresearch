# Project Statement

## Overview

This project is a self-improving quantitative research pipeline that predicts the daily price direction of three energy-sector ETFs — XLE (Energy Select SPDR), XLB (Materials Select SPDR), and XOP (SPDR S&P Oil & Gas Exploration) — and executes paper trades on Alpaca using the resulting signals. The system is designed around one objective: maximising annualised Sharpe Ratio on a 20% chronological holdout, computed as `mean(sign(prediction) × actual_return) / std(...) × sqrt(252)`. The best validated Sharpe achieved on XLE is **2.59** (holdout period 2025–2026).

## Core Model

Each asset is modelled by a `DirectionModel`: a `RandomForestClassifier` trained on six rolling momentum and volatility features derived solely from the asset's daily return series (`ret_lag`). The six features are: raw return, volatility-normalised return, 5-day mean, 20-day mean, 5-to-20-day momentum ratio, and short-to-long volatility ratio. Three design choices that materially improved Sharpe are (1) a 3-year rolling training window that drops the COVID-2020 regime, (2) exponential sample-decay weighting (`exp(-0.3 → 0)`) that tilts the RF toward recent market behaviour, and (3) a commodity shock filter that sets the output signal to zero on any day where the prior day's commodity return exceeded a calibrated threshold (2% for WTI crude, 2.5% for copper). XLB uses a copper shock filter (`HG=F`) while XLE and XOP use WTI (`CL=F`).

## Regime Layer

On top of the base model sits a t-SNE + HDBSCAN regime identification layer. The same six rolling features are computed for each asset's full history and embedded into two dimensions via t-SNE (perplexity=30, random_state=42). HDBSCAN clusters the embedding; noise points are reassigned to the nearest non-noise cluster by L2 distance. Optimal hyperparameters were selected by a grid search over perplexity ∈ {15, 30, 50} and n_clusters ∈ {3, 4, 5}, maximising between-regime Sharpe variance — the winning configuration (perplexity=30, n_clusters=5) produced a variance of **2.79** across the XLE regime set. At inference time, t-SNE is never called; a KNN classifier (k=5) trained on the original feature space assigns today's feature vector to the nearest regime in under a millisecond. One `DirectionModel` is trained per regime per asset; the regime-specific walk-forward Sharpe drives Kelly position sizing.

## Execution

Live signals are delivered by two processes. A Cloud Scheduler job fires every five minutes during market hours (9:30–15:55 ET, Mon–Fri) and invokes a Google Cloud Function (`run_daily`). The function classifies each asset's current market regime, retrieves the cached daily directional bias (recomputed on date rollover), reads 5-minute Alpaca bars for intraday RSI(14) and MACD confirmation, and places market orders via the Alpaca paper-trading API. Position sizing uses half-Kelly: `kelly_f = max(0, (wf_sharpe_mean / √252) × 0.5)`, where `wf_sharpe_mean` is the regime-specific 4-fold walk-forward Sharpe. Total gross exposure is capped at 95% of account equity across all three assets. A GCS distributed lock (`trade_lock.json`, 90-second TTL) prevents simultaneous order submission from the cloud function and any locally running scripts.

## Autoresearch Loop

Model improvement is driven by an automated grid sweep (`run_experiments.py`) over hyperparameters including tree depth, leaf size, training window length, and WTI shock threshold. Walk-forward Sharpe (4-fold chronological) is the primary selection criterion; a configuration is kept only if its walk-forward Sharpe exceeds the baseline's. Results are appended to `results.tsv` and summarised in `experiment_log.csv`. The intended cadence is weekly retraining via `train_all.py` every Friday close, with the updated model files synced to GCS for the cloud function to load.

## Explicit Non-Scope

This project does not deploy real capital, trade options or derivatives, perform intraday regime switching, integrate COMTRADE trade-flow features, or use sentiment or alternative data beyond EIA crude storage.
