"""
scripts/plot_regimes.py — Visualise the t-SNE regime embedding for XLE.

Produces two publication-quality figures saved to results/:
  regime_embedding.png  — t-SNE coloured by regime label
  regime_returns.png    — t-SNE coloured by forward 5-day return

Usage:
    python scripts/plot_regimes.py
"""
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from prepare import load_and_split_data

MODELS_DIR  = HERE / "models"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TICKER = "XLE"


def _load_regime_model(ticker: str):
    path = MODELS_DIR / f"{ticker}_regime.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run train_all.py first.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _forward_returns(full: pd.DataFrame, window: int = 5) -> np.ndarray:
    """Annualised forward window-day return aligned to the same rows as the feature matrix."""
    fwd = full["target"].rolling(window).sum().shift(-(window - 1))
    return fwd.values * (252 / window)  # annualise


def plot_embedding_by_label(embedding, labels, n_regimes, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab10", n_regimes)
    for k in range(n_regimes):
        mask = labels == k
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[cmap(k)], label=f"Regime {k}  (n={mask.sum()})",
            s=18, alpha=0.75, linewidths=0,
        )
    ax.set_title(f"{TICKER} — t-SNE Embedding by Regime Label", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path.relative_to(HERE)}")


def plot_embedding_by_returns(embedding, fwd_rets, out_path: Path):
    # Clip to 5th–95th percentile for colour saturation
    vmin, vmax = np.nanpercentile(fwd_rets, 5), np.nanpercentile(fwd_rets, 95)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=fwd_rets, cmap="RdYlGn", vmin=vmin, vmax=vmax,
        s=18, alpha=0.8, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Annualised forward 5-day return", fontsize=10)
    ax.set_title(f"{TICKER} — t-SNE Embedding by Forward 5-Day Return", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path.relative_to(HERE)}")


if __name__ == "__main__":
    regime_model = _load_regime_model(TICKER)

    train, val = load_and_split_data(TICKER)
    full = pd.concat([train, val]).reset_index(drop=True)

    from model import build_feature_matrix
    returns_arr = full["ret_lag"].values.astype(np.float64)
    feat_matrix = build_feature_matrix(returns_arr)
    valid = np.isfinite(feat_matrix).all(axis=1)
    full_clean = full.iloc[np.where(valid)[0]].reset_index(drop=True)

    embedding = regime_model.tsne_embedding
    labels    = regime_model.labels

    fwd_rets  = _forward_returns(full_clean)
    # Align fwd_rets to embedding length (some rows at the tail will be NaN)
    fwd_rets  = fwd_rets[:len(embedding)]

    plot_embedding_by_label(
        embedding, labels, regime_model.n_regimes,
        RESULTS_DIR / "regime_embedding.png",
    )
    plot_embedding_by_returns(
        embedding, fwd_rets,
        RESULTS_DIR / "regime_returns.png",
    )
    print("Done.")
