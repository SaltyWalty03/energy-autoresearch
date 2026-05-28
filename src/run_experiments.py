"""
src/run_experiments.py — Structured experiment runner for XLE direction model.

Runs 7 pre-defined model configurations, logs results as JSON to experiments/,
and prints a summary table. Never overwrites prior experiment records.

Usage:
    python src/run_experiments.py
"""
import sys
import time
import json
import datetime
from pathlib import Path

# Allow importing prepare.py and model.py from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin

from prepare import load_and_split_data, evaluate, walk_forward_sharpe
from model import DirectionModel

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def _next_exp_id():
    existing = sorted(EXPERIMENTS_DIR.glob("exp_*.json"))
    if not existing:
        return "exp_001"
    last = existing[-1].stem  # e.g. "exp_007"
    num = int(last.split("_")[1]) + 1
    return f"exp_{num:03d}"


class LinearBaselineModel(BaseEstimator, RegressorMixin):
    """Naive baseline: LinearRegression on raw ret_lag only."""

    def fit(self, X, y):
        self.lr_ = LinearRegression()
        self.lr_.fit(np.array(X).reshape(-1, 1), y)
        return self

    def predict(self, X):
        return self.lr_.predict(np.array(X).reshape(-1, 1))


EXPERIMENT_CONFIGS = [
    {
        "description": "Baseline: LinearRegression on ret_lag only",
        "model_cls": "linear_baseline",
        "hyperparameters": {
            "model_type": "LinearRegression",
            "features": ["ret_lag"],
        },
        "ablation_flags": {"shock_filter": False, "rolling_features": False, "sample_weights": False},
    },
    {
        "description": "RF direction classifier — Session 1 best (n=200, depth=2, msl=22, no shock filter)",
        "model_cls": "direction",
        "hyperparameters": {
            "n_estimators": 200, "max_depth": 2, "min_samples_leaf": 22,
            "train_window": 756, "wti_thresh": 99.0, "window": 20,
        },
        "ablation_flags": {"shock_filter": False, "rolling_features": True, "sample_weights": True},
    },
    {
        "description": "RF sweep baseline (n=300, depth=2, msl=22, wti_thresh=2%)",
        "model_cls": "direction",
        "hyperparameters": {
            "n_estimators": 300, "max_depth": 2, "min_samples_leaf": 22,
            "train_window": 756, "wti_thresh": 0.02, "window": 20,
        },
        "ablation_flags": {"shock_filter": True, "rolling_features": True, "sample_weights": True},
    },
    {
        "description": "RF depth ablation (n=300, depth=4, msl=22, wti_thresh=2%) — run_003",
        "model_cls": "direction",
        "hyperparameters": {
            "n_estimators": 300, "max_depth": 4, "min_samples_leaf": 22,
            "train_window": 756, "wti_thresh": 0.02, "window": 20,
        },
        "ablation_flags": {"shock_filter": True, "rolling_features": True, "sample_weights": True},
    },
    {
        "description": "RF best sweep (n=600, depth=2, msl=22, wti_thresh=2%) — run_009",
        "model_cls": "direction",
        "hyperparameters": {
            "n_estimators": 600, "max_depth": 2, "min_samples_leaf": 22,
            "train_window": 756, "wti_thresh": 0.02, "window": 20,
        },
        "ablation_flags": {"shock_filter": True, "rolling_features": True, "sample_weights": True},
    },
    {
        "description": "RF shock-filter ablation — no WTI filter (n=300, depth=2, msl=22)",
        "model_cls": "direction",
        "hyperparameters": {
            "n_estimators": 300, "max_depth": 2, "min_samples_leaf": 22,
            "train_window": 756, "wti_thresh": 99.0, "window": 20,
        },
        "ablation_flags": {"shock_filter": False, "rolling_features": True, "sample_weights": True},
    },
    {
        "description": "RF training-window ablation (n=300, depth=2, msl=22, train_window=504, wti_thresh=2%)",
        "model_cls": "direction",
        "hyperparameters": {
            "n_estimators": 300, "max_depth": 2, "min_samples_leaf": 22,
            "train_window": 504, "wti_thresh": 0.02, "window": 20,
        },
        "ablation_flags": {"shock_filter": True, "rolling_features": True, "sample_weights": True},
    },
]


def run_one(config, train, val, full_data):
    exp_id = _next_exp_id()
    anomalies = []
    t0 = time.time()

    try:
        if config["model_cls"] == "linear_baseline":
            model = LinearBaselineModel()
            X_tr = train[["ret_lag"]].values.ravel()
            y_tr = train["target"].values
            X_te = val[["ret_lag"]].values.ravel()
            y_te = val["target"].values
            model.fit(X_tr.reshape(-1, 1), y_tr)
            preds = model.predict(X_te.reshape(-1, 1))
            wf_mean, wf_std = 0.0, 0.0
        else:
            hp = config["hyperparameters"]
            model = DirectionModel(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                min_samples_leaf=hp["min_samples_leaf"],
                train_window=hp["train_window"],
                wti_thresh=hp["wti_thresh"],
                window=hp["window"],
            )
            X_tr = train[["ret_lag"]]
            y_tr = train["target"]
            X_te = val[["ret_lag"]]
            y_te = val["target"]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            wf_model = DirectionModel(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                min_samples_leaf=hp["min_samples_leaf"],
                train_window=hp["train_window"],
                wti_thresh=hp["wti_thresh"],
                window=hp["window"],
            )
            wf_mean, wf_std = walk_forward_sharpe(full_data, wf_model, n_folds=4)

        metrics = evaluate(y_te, preds)
        val_sharpe = metrics["sharpe"]
    except Exception as e:
        anomalies.append(f"ERROR: {e}")
        val_sharpe, wf_mean, wf_std = 0.0, 0.0, 0.0

    runtime = round(time.time() - t0, 2)

    record = {
        "experiment_id": exp_id,
        "description": config["description"],
        "timestamp": datetime.datetime.now().isoformat(),
        "hyperparameters": config["hyperparameters"],
        "ablation_flags": config["ablation_flags"],
        "dataset": {
            "ticker": "XLE",
            "train_size": len(train),
            "val_size": len(val),
            "split_date": str(val.index[0].date()),
            "val_end_date": str(val.index[-1].date()),
        },
        "metrics": {
            "val_sharpe": val_sharpe,
            "wf_sharpe_mean": wf_mean,
            "wf_sharpe_std": wf_std,
        },
        "runtime_seconds": runtime,
        "anomalies": anomalies,
    }

    out_path = EXPERIMENTS_DIR / f"{exp_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return record


def main():
    print("Loading XLE data...")
    train, val = load_and_split_data()
    full_data = pd.concat([train, val])
    print(f"Train: {len(train)} rows | Val: {len(val)} rows | Split: {val.index[0].date()}\n")

    header = f"{'ID':<9} {'Val Sharpe':>11} {'WF Mean':>9} {'WF Std':>8}  Description"
    print(header)
    print("-" * len(header))

    for cfg in EXPERIMENT_CONFIGS:
        rec = run_one(cfg, train, val, full_data)
        m = rec["metrics"]
        flag = " [ERROR]" if rec["anomalies"] else ""
        print(f"{rec['experiment_id']:<9} {m['val_sharpe']:>11.4f} {m['wf_sharpe_mean']:>9.4f} "
              f"{m['wf_sharpe_std']:>8.4f}  {rec['description'][:60]}{flag}")

    print(f"\nExperiment records saved to: {EXPERIMENTS_DIR}/")


if __name__ == "__main__":
    main()
