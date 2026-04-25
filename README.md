# Energy-Autoresearch

A structured research sandbox for developing and evaluating predictive models on the **XLE Energy ETF**, using a combination of price momentum features and weather/commodity interaction signals.

The goal is to **minimize prediction error** and **maximize the Sharpe Ratio** for Energy ETF returns.

---

## Project Structure

```
energy-autoresearch/
├── data/                  # Raw and processed data files
├── results/               # Saved experiment outputs
├── src/                   # Additional source modules
├── data_loader.py         # (auxiliary) data loading utilities
├── model.py               # The file you modify
├── prepare.py             # Data download, feature prep, and evaluation (do not edit)
├── run.py                 # Experiment runner (do not edit)
├── results.tsv            # Append-only experiment log (timestamp, description, Sharpe)
└── program.md             # Task rules and data dictionary
```

---

## How It Works

**`prepare.py`** downloads daily XLE price data from Yahoo Finance (from 2020-01-01), computes a lagged return feature (`ret_lag`) and a next-day log return target (`target`), then splits the dataset 80/20 into train and validation sets.

**`model.py`** defines `DirectionModel` — a scikit-learn compatible regressor that:
- Computes rolling window features (5-day and 20-day momentum, volatility ratios)
- Trains a `RandomForestClassifier` to predict the direction of the next day's return
- Outputs a continuous signal in `[-1, 1]` (from class probabilities)

**`run.py`** wires everything together: it loads data, fits the model, evaluates it on the validation set, and appends the result to `results.tsv`.

**Evaluation metric:** Annualized Sharpe Ratio, computed as:
```
Sharpe = mean(sign(prediction) * actual_return) / std(...) * sqrt(252)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install yfinance scikit-learn numpy pandas
```

### 2. Run the baseline experiment

```bash
python run.py "baseline description"
```

Results are appended to `results.tsv` with a timestamp, description, and Sharpe Ratio.

---

## Rules for Experimentation

1. **Only modify `model.py`** — `prepare.py` and `run.py` are locked.
2. Focus on building **interaction features** between weather (`Temp_Anomaly`) and natural gas (`NatGas_Storage`).
3. Log every experiment with a descriptive message: `python run.py "your description"`.

---

## Data Dictionary

| Field | Description |
|---|---|
| `XLE_Close` | Closing price of the XLE Energy ETF |
| `Temp_Anomaly` | Departure from average temperature |
| `NatGas_Storage` | Current natural gas storage levels |
| `ret_lag` | Previous day's percentage return (baseline feature) |
| `target` | Next day's log return (prediction target) |

---

## Extending `model.py`

The `build_model()` function returns a scikit-learn compatible estimator. To experiment, implement a new class or modify `DirectionModel`. Ideas to try:

- Add `Temp_Anomaly × NatGas_Storage` interaction terms
- Incorporate lagged weather or storage features
- Swap the classifier (e.g., GradientBoosting, Ridge regression on signals)
- Tune `window`, `n_estimators`, `max_depth`, `min_samples_leaf`

Example skeleton:

```python
def build_model():
    return DirectionModel(n_estimators=300, max_depth=3, min_samples_leaf=15)
```

---

## Results Log

Experiments are tracked in `results.tsv`:

```
timestamp    description    sharpe
```

Each run appends a new row, making it easy to compare experiments over time.
