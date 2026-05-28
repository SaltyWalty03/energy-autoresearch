# energy-autoresearch

Predictive model for the **XLE Energy Select Sector SPDR ETF** daily return direction,
targeting out-of-sample Sharpe Ratio maximization on a chronological 20% holdout.

**Best validated OOS Sharpe: 2.48** (annualised, validation period 2025-01-17 to 2026-04-23)

---

## Project Overview

This project implements an automated ML research loop for predicting XLE (Energy ETF)
daily log-return direction. The core model is a Random Forest direction classifier with:

- Six rolling momentum/volatility features derived from lagged returns
- A 3-year rolling training window (drops COVID-2020 regime)
- Exponential sample-decay weighting to upweight recent data
- WTI crude-oil shock filter: go flat when yesterday's WTI |return| > 2%

The system logs every experiment to structured JSON files in `experiments/` and
aggregates results in `results/final_results.csv`.

---

## Repository Structure

```
energy-autoresearch/
├── model.py                  # DirectionModel — the only file to edit for experiments
├── prepare.py                # Data download, feature prep, train/val split (frozen)
├── run.py                    # Single experiment runner → appends to results.tsv
├── src/
│   ├── run_experiments.py    # Structured experiment runner → experiments/ JSON logs
│   ├── build_results_table.py# Aggregate JSON logs → results/final_results.{csv,tex}
│   ├── regime.py             # Regime detection utilities
│   ├── data_loader.py        # Alternative-data pipeline
│   ├── eia_fetcher.py        # EIA crude/gas data fetcher
│   └── trends_fetcher.py     # Google Trends signal fetcher
├── experiments/              # Structured JSON logs (one file per experiment)
├── results/
│   ├── final_results.csv     # Aggregated results table
│   ├── final_results.tex     # LaTeX table for the paper
│   └── experiment_log.md     # Human-readable experiment notes
├── reports/
│   ├── final_report.tex      # NeurIPS-format paper
│   └── reflection_memo.txt   # Post-hoc analysis memo
├── notebooks/                # Exploratory Jupyter notebooks
├── data/                     # Raw and cached data (gitignored)
└── results.tsv               # Append-only legacy experiment log
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Python version

Python 3.10+ required. Developed and tested on Python 3.12.

### Data

Data is fetched automatically at runtime:
- **XLE** price data: downloaded via `yfinance` on first run
- **WTI crude**: loaded from `data/eia_cache/` (pre-fetched EIA parquet files)

---

## Reproduction Steps

### 1. Run all structured experiments

```bash
python src/run_experiments.py
```

This runs 7 model configurations, saves structured JSON to `experiments/`, and
prints a summary table. Completes in under 5 minutes on CPU.

### 2. Build the results table

```bash
python src/build_results_table.py
```

Aggregates all `experiments/*.json` files into `results/final_results.csv` and
`results/final_results.tex`.

### 3. Compile the report (optional — requires pdflatex)

```bash
cd reports && pdflatex final_report.tex
```

### Single experiment (legacy)

```bash
python run.py "description of this change"
```

---

## Model Architecture

`DirectionModel` (`model.py`) is an sklearn-compatible estimator.

**Input:** single-column DataFrame `X` with column `ret_lag` (previous day's % return),
indexed by `DatetimeIndex`.

**Internal features (6), computed from `ret_lag`:**

| # | Feature | Description |
|---|---------|-------------|
| 1 | `ret_lag` | Raw daily return |
| 2 | `ret_lag / vol5` | Vol-adjusted return (5-day std) |
| 3 | `mean(w5)` | 5-day rolling mean (short-term momentum) |
| 4 | `mean(w20)` | 20-day rolling mean (medium-term momentum) |
| 5 | `mean(w5) / mean(w20)` | MACD-like ratio |
| 6 | `vol5 / vol20` | Volatility regime indicator |

**Training:** `RandomForestClassifier` with `max_depth=2`, `min_samples_leaf=22`,
`n_estimators=600`, `max_features='sqrt'`. Labels are `(target > 0).astype(int)`.

**WTI shock filter:** On days where `|WTI log-return (t-1)| > 0.02`, override
signal to 0.0 (hold cash). Reduces exposure to energy-market regime discontinuities.

---

## Evaluation Metric

```
signal_return = sign(prediction) * actual_log_return
Sharpe = mean(signal_return) / std(signal_return) * sqrt(252)
```

---

## Key Findings

- **RF outperforms** linear, logistic, GBM, MLP, and SVM for this task
- **Depth=2 generalizes best** — deeper trees overfit the ~800-sample training set
- **6 features are optimal** — adding more features consistently reduces Sharpe
- **WTI shock filter adds +0.37 Sharpe** over the unfiltered baseline (2.21 vs 1.84)
- **3-year rolling window** strongly outperforms 2-year (+1.16 Sharpe)

---

## Experiment Tracking

Every experiment is logged to `experiments/` as a JSON file with the schema:

```json
{
  "experiment_id": "exp_001",
  "description": "Baseline LinearRegression",
  "timestamp": "2026-05-27T...",
  "hyperparameters": {...},
  "dataset": {"ticker": "XLE", "train_size": 1258, "val_size": 315, "split_date": "..."},
  "metrics": {"val_sharpe": 0.8584, "wf_sharpe_mean": ..., "wf_sharpe_std": ...},
  "runtime_seconds": 1.2,
  "anomalies": []
}
```

---

## Modifying the Model

Only `model.py` should be modified. `prepare.py` and `run.py` are frozen.

After every change:
```bash
python run.py "description" && tail -1 results.tsv
```

Revert if worse:
```bash
git checkout model.py
```
