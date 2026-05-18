"""
train_all.py — Train and save one DirectionModel per asset in ASSETS.

For each asset, downloads price data, trains a DirectionModel with the
asset's commodity shock filter, runs 4-fold purged walk-forward validation,
and saves:
    models/{ticker}_model.pkl  — fitted model (trained on full history)
    models/{ticker}_meta.json  — wf_sharpe_mean, wf_sharpe_std

Prints a summary table of walk-forward Sharpe per asset at the end.

Usage:
    python train_all.py
"""
import json
import pickle
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)

from src.universe import ASSETS
from prepare import load_and_split_data, walk_forward_sharpe
from model import DirectionModel


def train_asset(ticker: str, cfg: dict) -> dict:
    print(f"\n-- {ticker} (commodity={cfg['commodity']}) --------------------")

    train, val = load_and_split_data(ticker)
    full = pd.concat([train, val])

    model = DirectionModel(
        n_estimators=600,
        max_depth=2,
        min_samples_leaf=22,
        train_window=756,
        wti_thresh=cfg["shock_thresh"],
        model_type="rf",
        commodity_ticker=cfg["commodity"],
    )

    wf_mean, wf_std = walk_forward_sharpe(full, model, n_folds=4)
    print(f"  Walk-forward Sharpe: mean={wf_mean:.4f}  std={wf_std:.4f}")

    # Refit on full history for deployment
    model.fit(full[["ret_lag"]], full["target"])

    model_path = MODELS_DIR / f"{ticker}_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"  Saved → {model_path.relative_to(HERE)}")

    meta = {
        "ticker": ticker,
        "wf_sharpe_mean": wf_mean,
        "wf_sharpe_std": wf_std,
        "commodity": cfg["commodity"],
        "shock_thresh": cfg["shock_thresh"],
    }
    meta_path = MODELS_DIR / f"{ticker}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved → {meta_path.relative_to(HERE)}")

    return meta


if __name__ == "__main__":
    results = []
    for ticker, cfg in ASSETS.items():
        results.append(train_asset(ticker, cfg))

    print("\n" + "=" * 56)
    print(f"{'Asset':<8} {'WF Sharpe Mean':>16} {'WF Sharpe Std':>14} {'Edge?':>8}")
    print("-" * 56)

    for r in results:
        edge = "YES" if r["wf_sharpe_mean"] > 0 else "no"
        print(f"{r['ticker']:<8} {r['wf_sharpe_mean']:>16.4f} {r['wf_sharpe_std']:>14.4f} {edge:>8}")
    print("=" * 56)
