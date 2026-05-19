"""
train_all.py — Train per-asset and per-regime DirectionModels.

For each asset in ASSETS:
  1. Downloads price data and computes walk-forward Sharpe.
  2. Builds the 6-feature rolling matrix and discovers regimes via t-SNE + HDBSCAN.
  3. Trains one DirectionModel per regime, evaluating walk-forward Sharpe within
     each regime's rows.
  4. Saves:
       models/{ticker}_model.pkl        — full model (trained on all data)
       models/{ticker}_meta.json        — wf_sharpe_mean, wf_sharpe_std
       models/{ticker}_regime.pkl       — RegimeModel (embedding + KNN)
       models/{ticker}_regime{k}_model.pkl  — regime-specific DirectionModel
       models/{ticker}_regime{k}_meta.json  — regime-specific wf_sharpe

Usage:
    python train_all.py
"""
import copy
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)

from src.universe import ASSETS
from src.regime import fit_regimes
from prepare import load_and_split_data, walk_forward_sharpe
from model import DirectionModel, build_feature_matrix


def _fold_sharpe(y_true, y_pred):
    rets = np.sign(y_pred) * np.array(y_true)
    return float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() > 0 else 0.0


def _regime_wf_sharpe(regime_df: pd.DataFrame, model_template, n_folds: int = 4):
    """Walk-forward Sharpe computed only on rows belonging to a specific regime."""
    if len(regime_df) < 20:
        return 0.0, 0.0
    return walk_forward_sharpe(regime_df, model_template, n_folds=min(n_folds, len(regime_df) // 10))


def train_asset(ticker: str, cfg: dict) -> dict:
    print(f"\n-- {ticker} (commodity={cfg['commodity']}) --------------------")

    train, val = load_and_split_data(ticker)
    full = pd.concat([train, val])

    base_model_cfg = dict(
        n_estimators=600,
        max_depth=2,
        min_samples_leaf=22,
        train_window=756,
        wti_thresh=cfg["shock_thresh"],
        model_type="rf",
        commodity_ticker=cfg["commodity"],
    )

    # ── Full-data model ────────────────────────────────────────────────────────
    model = DirectionModel(**base_model_cfg)
    wf_mean, wf_std = walk_forward_sharpe(full, model, n_folds=4)
    print(f"  Full WF Sharpe: mean={wf_mean:.4f}  std={wf_std:.4f}")

    model.fit(full[["ret_lag"]], full["target"])
    _save_pkl(MODELS_DIR / f"{ticker}_model.pkl", model)
    _save_json(MODELS_DIR / f"{ticker}_meta.json", {
        "ticker": ticker, "wf_sharpe_mean": wf_mean, "wf_sharpe_std": wf_std,
        "commodity": cfg["commodity"], "shock_thresh": cfg["shock_thresh"],
    })

    # ── Regime discovery ───────────────────────────────────────────────────────
    print("  Fitting regimes via t-SNE + HDBSCAN...")
    returns_arr = full["ret_lag"].values.astype(np.float64)
    feat_matrix = build_feature_matrix(returns_arr)

    # Drop rows with NaN/Inf that can arise from early rolling windows
    valid = np.isfinite(feat_matrix).all(axis=1)
    feat_matrix_clean = feat_matrix[valid]
    full_clean = full.iloc[np.where(valid)[0]]   # keep DatetimeIndex

    regime_model = fit_regimes(feat_matrix_clean, perplexity=30, n_clusters=4)
    _save_pkl(MODELS_DIR / f"{ticker}_regime.pkl", regime_model)
    print(f"  Regimes found: {regime_model.n_regimes}")

    # ── Per-regime models ──────────────────────────────────────────────────────
    regime_results = []
    for k in range(regime_model.n_regimes):
        mask = regime_model.labels == k
        regime_df = full_clean[mask].copy()
        n_rows = mask.sum()
        print(f"  Regime {k}: {n_rows} rows", end="")

        if n_rows < 30:
            print(" — too few rows, using full model as fallback.")
            # Copy full model as placeholder
            _save_pkl(MODELS_DIR / f"{ticker}_regime{k}_model.pkl", copy.deepcopy(model))
            _save_json(MODELS_DIR / f"{ticker}_regime{k}_meta.json", {
                "ticker": ticker, "regime": k, "n_rows": int(n_rows),
                "wf_sharpe_mean": wf_mean, "wf_sharpe_std": wf_std, "fallback": True,
            })
            regime_results.append({"regime": k, "n_rows": int(n_rows),
                                    "wf_sharpe_mean": wf_mean, "wf_sharpe_std": wf_std})
            continue

        r_model = DirectionModel(**base_model_cfg)
        r_wf_mean, r_wf_std = _regime_wf_sharpe(regime_df, r_model)
        print(f"  WF Sharpe: mean={r_wf_mean:.4f}  std={r_wf_std:.4f}")

        r_model.fit(regime_df[["ret_lag"]], regime_df["target"])
        _save_pkl(MODELS_DIR / f"{ticker}_regime{k}_model.pkl", r_model)
        _save_json(MODELS_DIR / f"{ticker}_regime{k}_meta.json", {
            "ticker": ticker, "regime": k, "n_rows": int(n_rows),
            "wf_sharpe_mean": r_wf_mean, "wf_sharpe_std": r_wf_std, "fallback": False,
        })
        regime_results.append({"regime": k, "n_rows": int(n_rows),
                                "wf_sharpe_mean": r_wf_mean, "wf_sharpe_std": r_wf_std})

    return {"ticker": ticker, "wf_sharpe_mean": wf_mean, "wf_sharpe_std": wf_std,
            "n_regimes": regime_model.n_regimes, "regime_results": regime_results}


def _save_pkl(path: Path, obj):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"  Saved -> {path.relative_to(HERE)}")


def _save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    all_results = []
    for ticker, cfg in ASSETS.items():
        all_results.append(train_asset(ticker, cfg))

    print("\n" + "=" * 70)
    print(f"{'Asset':<6} {'WF Mean':>9} {'WF Std':>8} {'Regimes':>8}  Per-regime WF Sharpes")
    print("-" * 70)
    for r in all_results:
        regime_str = "  ".join(
            f"R{rr['regime']}={rr['wf_sharpe_mean']:.3f}"
            for rr in r["regime_results"]
        )
        print(f"{r['ticker']:<6} {r['wf_sharpe_mean']:>9.4f} {r['wf_sharpe_std']:>8.4f}"
              f" {r['n_regimes']:>8}  {regime_str}")
    print("=" * 70)
