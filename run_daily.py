"""
run_daily.py — Multi-asset local paper-trading signal via Alpaca.

Loads per-asset regime models from models/, classifies today's market regime
for each asset, then uses the regime-specific DirectionModel and its
walk-forward Sharpe to size a half-Kelly position.

Two-layer inference:
  Layer 1 — Regime: classify_regime() → k (or -1 → fall back to full model)
  Layer 2 — Signal: regime-k DirectionModel predicts directional bias

Kelly formula (half-Kelly, annualised-to-daily):
    kelly_f = max(0, (wf_sharpe_mean / sqrt(252)) * 0.5)
    dollar_alloc = kelly_f * equity
Total gross exposure capped at 95% of equity across all assets.

Usage:
    python run_daily.py
"""
import json
import math
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv, find_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

try:
    from google.cloud import storage as gcs_storage
    _GCS_AVAILABLE = True
except ImportError:
    _GCS_AVAILABLE = False

load_dotenv(find_dotenv())

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
GCS_BUCKET = "energy-autoresearch-model"
LOCK_OBJECT = "trade_lock.json"
LOCK_TTL_SECONDS = 90
MAX_EXPOSURE = 0.95
MAX_SHARES_PER_ASSET = 100
MODEL_THRESH = 0.05

from src.universe import ASSETS
from src.regime import classify_regime
from model import build_feature_matrix


# ── GCS lock ──────────────────────────────────────────────────────────────────

def _gcs_client():
    if not _GCS_AVAILABLE:
        return None
    try:
        return gcs_storage.Client()
    except Exception:
        return None


def _acquire_lock(gcs, owner: str) -> bool:
    if gcs is None:
        print("GCS unavailable — proceeding without distributed lock.")
        return True
    blob = gcs.bucket(GCS_BUCKET).blob(LOCK_OBJECT)
    try:
        existing = json.loads(blob.download_as_text())
        acquired_at = datetime.fromisoformat(existing["acquired_at"]).replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - acquired_at).total_seconds()
        if age < LOCK_TTL_SECONDS:
            print(f"Lock held by '{existing.get('owner')}' (age={age:.0f}s) — aborting.")
            return False
        print(f"Overwriting stale lock (age={age:.0f}s).")
    except Exception:
        pass
    blob.upload_from_string(
        json.dumps({"owner": owner, "acquired_at": datetime.now(timezone.utc).isoformat()}),
        content_type="application/json",
    )
    print(f"Lock acquired by '{owner}'.")
    return True


def _release_lock(gcs):
    if gcs is None:
        return
    try:
        gcs.bucket(GCS_BUCKET).blob(LOCK_OBJECT).delete()
        print("Lock released.")
    except Exception:
        pass


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_pkl(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_meta(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"wf_sharpe_mean": 0.0}


def _load_regime_stack(ticker: str):
    """
    Returns (full_model, full_meta, regime_model, regime_meta_list).
    regime_meta_list[k] is the meta dict for regime k.
    Falls back to full model if regime files are missing.
    """
    full_model = _load_pkl(MODELS_DIR / f"{ticker}_model.pkl")
    full_meta  = _load_meta(MODELS_DIR / f"{ticker}_meta.json")

    regime_path = MODELS_DIR / f"{ticker}_regime.pkl"
    if not regime_path.exists():
        return full_model, full_meta, None, []

    regime_model = _load_pkl(regime_path)
    regime_metas = []
    for k in range(regime_model.n_regimes):
        regime_metas.append(_load_meta(MODELS_DIR / f"{ticker}_regime{k}_meta.json"))

    return full_model, full_meta, regime_model, regime_metas


# ── Feature vector for today ──────────────────────────────────────────────────

def _today_feature_vector(ticker: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (feature_vector_for_latest_row, ret_lag_series_as_frame)."""
    raw = yf.download(ticker, period="90d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    ret_lag = raw["Close"].dropna().pct_change().dropna().shift(1).dropna()
    feat_matrix = build_feature_matrix(ret_lag.values.astype(np.float64))
    return feat_matrix[-1], ret_lag.to_frame()


# ── Signal generation ──────────────────────────────────────────────────────────

def _get_signal_and_meta(
    ticker: str,
    full_model,
    full_meta: dict,
    regime_model,
    regime_metas: list,
    feat_vec: np.ndarray,
    ret_lag_frame: pd.DataFrame,
) -> tuple[float, dict, int]:
    """
    Returns (signal, meta_used, regime_k).
    regime_k == -1 means the full model was used as fallback.
    """
    if regime_model is None:
        signal = float(full_model.predict(ret_lag_frame)[-1])
        return signal, full_meta, -1

    k = classify_regime(feat_vec, regime_model)
    if k == -1 or k >= len(regime_metas):
        print(f"  Regime classify returned {k} — using full model fallback.")
        signal = float(full_model.predict(ret_lag_frame)[-1])
        return signal, full_meta, -1

    r_model_path = MODELS_DIR / f"{ticker}_regime{k}_model.pkl"
    if not r_model_path.exists():
        print(f"  No regime-{k} model found — using full model fallback.")
        signal = float(full_model.predict(ret_lag_frame)[-1])
        return signal, full_meta, k

    r_model = _load_pkl(r_model_path)
    signal  = float(r_model.predict(ret_lag_frame)[-1])
    return signal, regime_metas[k], k


# ── Latest ask price ──────────────────────────────────────────────────────────

def _get_price(data_client: StockHistoricalDataClient, ticker: str) -> float:
    quote = data_client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=ticker)
    )
    return float(quote[ticker].ask_price)


# ── Kelly sizing ───────────────────────────────────────────────────────────────

def _kelly_fraction(wf_sharpe_mean: float) -> float:
    if wf_sharpe_mean <= 0:
        return 0.0
    return (wf_sharpe_mean / math.sqrt(252)) * 0.5


# ── Order placement ────────────────────────────────────────────────────────────

def _close_position(client: TradingClient, ticker: str):
    try:
        client.close_position(ticker)
        print(f"  Closed existing {ticker} position.")
    except Exception:
        pass


def _place_order(client: TradingClient, ticker: str, shares: int, signal: float, price: float):
    side = OrderSide.BUY if signal > 0 else OrderSide.SELL
    order = MarketOrderRequest(
        symbol=ticker, qty=shares, side=side, time_in_force=TimeInForce.DAY,
    )
    client.submit_order(order)
    print(f"  [{side.value.upper()}] {shares} shares {ticker} @ ~${price:.2f}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    trade_client = TradingClient(api_key, secret_key, paper=True)
    data_client  = StockHistoricalDataClient(api_key, secret_key)
    gcs          = _gcs_client()

    if not _acquire_lock(gcs, owner="run_daily"):
        raise SystemExit("Could not acquire trade lock — another process is trading.")

    try:
        equity = float(trade_client.get_account().equity)
        print(f"Account equity: ${equity:,.2f}")

        plan = []  # (ticker, signal, shares_raw, price, kelly_f)

        for ticker in ASSETS:
            print(f"\n[{ticker}]")
            full_model, full_meta, regime_model, regime_metas = _load_regime_stack(ticker)
            feat_vec, ret_lag_frame = _today_feature_vector(ticker)

            signal, meta_used, regime_k = _get_signal_and_meta(
                ticker, full_model, full_meta, regime_model, regime_metas,
                feat_vec, ret_lag_frame,
            )
            regime_label = f"regime {regime_k}" if regime_k >= 0 else "full model"
            n_regimes    = regime_model.n_regimes if regime_model else 0
            print(f"  {regime_label} (of {n_regimes})  signal={signal:.4f}  "
                  f"WF-Sharpe={meta_used['wf_sharpe_mean']:.4f}")

            if abs(signal) < MODEL_THRESH:
                print("  Signal in flat zone — skipping.")
                continue

            kelly_f = _kelly_fraction(meta_used["wf_sharpe_mean"])
            if kelly_f == 0.0:
                print("  WF-Sharpe <= 0 — no edge, skipping.")
                continue

            price = _get_price(data_client, ticker)
            dollar_alloc = kelly_f * equity
            shares_raw   = int(dollar_alloc / price)
            plan.append((ticker, signal, shares_raw, price, kelly_f))
            print(f"  kelly_f={kelly_f:.4f}  alloc=${dollar_alloc:,.0f}  raw_shares={shares_raw}")

        # ── Cap total gross exposure ───────────────────────────────────────────
        scale = 1.0
        if plan:
            total_alloc = sum(s * p for _, _, s, p, _ in plan)
            max_alloc   = MAX_EXPOSURE * equity
            scale = min(1.0, max_alloc / total_alloc) if total_alloc > 0 else 1.0
            if scale < 1.0:
                print(f"\nScaling by {scale:.3f} (alloc ${total_alloc:,.0f} > cap ${max_alloc:,.0f})")

        # ── Close then open ────────────────────────────────────────────────────
        print("\n-- Closing existing positions --")
        for ticker in ASSETS:
            _close_position(trade_client, ticker)

        print("\n-- Placing orders --")
        for ticker, signal, shares_raw, price, kelly_f in plan:
            shares = min(int(shares_raw * scale), MAX_SHARES_PER_ASSET)
            if shares < 1:
                print(f"  {ticker}: scaled to <1 share — skipping.")
                continue
            _place_order(trade_client, ticker, shares, signal, price)

        if not plan:
            print("No actionable signals — staying flat.")

    finally:
        _release_lock(gcs)
