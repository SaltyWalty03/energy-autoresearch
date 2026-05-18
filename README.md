# energy-autoresearch

Predictive model for the **XLE Energy ETF** daily return direction, targeting Sharpe Ratio maximization on a 20% chronological holdout.

**Best validated Sharpe: 2.48** (annualised, val period ~2025–2026)

---

## Project Structure

```
energy-autoresearch/
├── model.py               # DirectionModel — edit this to experiment
├── prepare.py             # Data download, feature prep, train/val split
├── run.py                 # Single experiment runner → appends to results.tsv
├── run_experiments.py     # Hyperparameter grid sweep → produces reports
├── run_daily.py           # Local one-shot paper-trading signal via Alpaca
├── run_intraday.py        # Intraday loop: model bias + RSI/MACD via Alpaca
├── train_model.py         # Re-trains model.pkl from latest XLE data
├── results.tsv            # Append-only experiment log (source of truth)
├── results/               # Generated reports and charts
├── src/
│   ├── data_loader.py     # Alternative-data pipeline (future feature work)
│   ├── eia_fetcher.py
│   ├── openmeteo_fetcher.py
│   ├── gem_loader.py
│   └── trends_fetcher.py
├── cloud_function/        # Google Cloud Function for automated daily trading
│   ├── main.py
│   ├── model.py           # Kept in sync with root model.py (see note inside)
│   └── requirements.txt
├── scripts/
│   └── generate_deliverables.py   # One-shot report generator (already run)
└── data/                  # Raw data (caches excluded from git)
```

---

## How It Works

**`prepare.py`** downloads daily XLE data from Yahoo Finance (from 2020-01-01), computes `ret_lag` (previous day's % return) and `target` (next-day log return), and splits 80/20 chronologically.

**`model.py`** defines `DirectionModel` — a Random Forest classifier that:
- Internally builds six rolling momentum/volatility features from `ret_lag`
- Trains with a 3-year rolling window and exponential sample-decay
- Applies a WTI crude shock filter at inference (goes flat when yesterday's WTI |return| > 2%)
- Outputs a continuous signal in `[-1, 1]` representing directional confidence

**Evaluation metric:**
```
Sharpe = mean(sign(prediction) × actual_return) / std(...) × sqrt(252)
```

---

## Quickstart

```bash
pip install yfinance scikit-learn numpy pandas alpaca-py python-dotenv pytz
```

Run a single experiment and log it:
```bash
python run.py "description of this change"
```

---

## Entry Points

### `run.py` — Single experiment
```bash
python run.py "add wti momentum feature"
```
Fits the model defined in `model.py`, evaluates on val, appends to `results.tsv`.

### `run_experiments.py` — Hyperparameter grid sweep
```bash
python run_experiments.py
```
Runs every config in `EXPERIMENT_GRID`, writes `experiment_log.csv`,
`metric_trajectory.png`, `keep_discard_crash_summary.md`,
`best_vs_baseline.md`, and `what_actually_worked.md` to `results/`.

### `run_daily.py` — Local paper-trading signal
```bash
python run_daily.py
```
Loads `model.pkl`, generates today's signal, and places a market order via
the Alpaca paper API. Use as a fallback when the Cloud Function is unavailable.

### `run_intraday.py` — Intraday trading loop
```bash
python run_intraday.py
```
Runs continuously during market hours (Mon–Fri 9:30 AM–3:55 PM ET). Uses a
two-layer signal: `DirectionModel` daily bias gates intraday RSI(14) + MACD
entries on 5-minute Alpaca bars.

---

## Modifying the Model

Only `model.py` needs to change for experiments. `prepare.py` signatures are
stable — do not modify them.

`build_model()` must return a scikit-learn compatible estimator. Current best config:
```python
def build_model():
    return DirectionModel(n_estimators=600, max_depth=2, min_samples_leaf=22,
                          train_window=756, wti_thresh=0.02)
```

Key `DirectionModel` parameters:

| Parameter | Default | Effect |
|---|---|---|
| `n_estimators` | 600 | Number of trees |
| `max_depth` | 2 | Tree depth — main Sharpe lever |
| `min_samples_leaf` | 22 | Leaf regularisation |
| `window` | 20 | Rolling feature window (days) |
| `train_window` | 756 | ~3 years of training data |
| `wti_thresh` | 0.02 | WTI shock filter threshold |

After every change, run the smoke test:
```bash
python run.py "description" && tail -1 results.tsv
```

Revert if worse:
```bash
git checkout model.py
```

---

## Experiment Log

`results.tsv` is the append-only record of every run:
```
timestamp    description    sharpe
```

---

## Cloud Deployment

The Cloud Function in `cloud_function/` runs the intraday two-layer logic
automatically every 5 minutes during market hours via Cloud Scheduler.

```bash
# Redeploy after changes to cloud_function/main.py or cloud_function/model.py
gcloud functions deploy run_daily \
    --runtime python311 --region us-central1 \
    --trigger-http --allow-unauthenticated \
    --source cloud_function/ \
    --memory 512MB --timeout 120s

# Upload a retrained model to GCS
gsutil cp model.pkl gs://energy-autoresearch-model/model.pkl

# View recent logs
gcloud functions logs read run_daily --region us-central1 --limit 20
```

---

## Next Steps

From `what_actually_worked.md`:

1. **WTI as a second feature** — add `wti_ret_lag` alongside `ret_lag` so the RF learns graded oil-momentum exposure rather than a binary shock filter.
2. **Gradient boosting** — `HistGradientBoostingRegressor` with a Sharpe-approximating loss typically outperforms RF on small, noisy financial tabular datasets.
3. **Regime-conditioned models** — partition the training window by WTI 30-day realised volatility quartile; the walk-forward Sharpe variance suggests unmodeled regime structure.
4. **Bayesian hyperparameter search** — `scikit-optimize BayesSearchCV` over the joint space (depth, leaf, window, wti_thresh); the grid only explored marginal effects.
5. **Alternative-data features** — `src/data_loader.py` assembles EIA, weather (HDD/CDD), GEM pipeline, and Google Trends signals into a weekly feature matrix, ready to plug into a weekly version of the model.
