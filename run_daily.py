"""
run_daily.py — Multi-asset paper-trading signal; Cloud Functions HTTP entry point.

Deploy with:
    gcloud functions deploy run_daily \
        --runtime python311 --trigger-http \
        --entry-point run_daily --region us-central1 \
        --set-env-vars ALPACA_API_KEY=...,ALPACA_SECRET_KEY=...

Two-layer inference:
  Layer 1 — Regime: classify_regime() → k (or -1 → fall back to full model)
  Layer 2 — Signal: regime-k DirectionModel predicts directional bias

Position sizing — equal-weight allocation:
    per_position_notional = MAX_PORTFOLIO_EXPOSURE_PCT * equity / num_active_signals
    safe_notional         = min(per_position_notional, alpaca_buying_power)
    shares                = int(safe_notional / price)
Kelly is retained as an edge gate only: wf_sharpe ≤ 0 → skip asset, not size it.

Position reconciliation:
  - Signal matches current direction → hold (no order placed)
  - Signal flat → close any open position
  - Signal differs from current → close then open new direction

Environment variables:
    ALPACA_API_KEY      — required
    ALPACA_SECRET_KEY   — required
    DRY_RUN             — set to "true" to run signal generation without placing orders
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

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
GCS_BUCKET = "energy-autoresearch-model"
LOCK_OBJECT = "trade_lock.json"
LOCK_TTL_SECONDS = 90
MAX_PORTFOLIO_EXPOSURE_PCT = 0.80   # fraction of equity deployed across all concurrent positions
MODEL_THRESH = 0.02

from src.universe import ASSETS
from src.regime import classify_regime
from src.trade_logger import log_trade
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

    if regime_metas[k].get("wf_sharpe_mean", 0.0) <= 0:
        print(f"  Regime-{k} wf_sharpe={regime_metas[k]['wf_sharpe_mean']:.3f} ≤ 0 — using full model fallback.")
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


# ── Kelly edge guard ──────────────────────────────────────────────────────────

def _kelly_fraction(wf_sharpe_mean: float) -> float:
    """Skip gate only — returns 0 when the model has no validated edge (wf_sharpe ≤ 0).
    No longer used for dollar sizing; see _compute_position_size."""
    if wf_sharpe_mean <= 0:
        return 0.0
    return (wf_sharpe_mean / math.sqrt(252)) * 0.5


# ── Position sizing (swap body here to change weighting scheme) ────────────────

def _compute_position_size(
    equity: float,
    price: float,
    num_positions: int,
    buying_power: float,
    *,
    max_exposure: float = MAX_PORTFOLIO_EXPOSURE_PCT,
) -> int:
    """
    Equal-weight allocation across num_positions concurrent positions.

    Formula
    -------
        per_position_notional = max_exposure * equity / num_positions
        safe_notional         = min(per_position_notional, buying_power)
        shares                = int(safe_notional / price)

    The buying_power clamp enforces the 80 % cap in real time against Alpaca's
    live account state, not just at signal-generation time.

    To swap to inverse-volatility weighting later, replace this function body
    while keeping the signature unchanged — order-execution code is untouched.
    """
    if num_positions <= 0 or price <= 0:
        return 0
    raw_notional  = max_exposure * equity / num_positions
    safe_notional = min(raw_notional, max(0.0, buying_power))
    return int(safe_notional / price)


# ── Position management ────────────────────────────────────────────────────────

def _current_position(client: TradingClient, ticker: str) -> tuple[str, int]:
    try:
        pos = client.get_open_position(ticker)
        qty = int(pos.qty)
        if qty > 0:
            return "long", qty
        elif qty < 0:
            return "short", abs(qty)
    except Exception:
        pass
    return "flat", 0


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


# ── Cloud Functions HTTP entry point ──────────────────────────────────────────

def run_daily(request):
    """
    HTTP entry point for Google Cloud Functions.
    `request` is a Flask Request object (ignored — all config comes from env vars).
    Returns a (body, status_code) tuple per Cloud Functions HTTP convention.
    """
    dry_run    = os.environ.get("DRY_RUN") == "true"
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    if not dry_run:
        trade_client = TradingClient(api_key, secret_key, paper=True)
        data_client  = StockHistoricalDataClient(api_key, secret_key)
        gcs          = _gcs_client()

        if not _acquire_lock(gcs, owner="run_daily"):
            for ticker in ASSETS:
                log_trade(asset=ticker, regime=-1, n_regimes=0, signal=0.0,
                          kelly_fraction=0.0, action="flat", fill_price=0.0,
                          shares=0, account_equity=0.0)
            return ("Lock held, skipped run", 200)
    else:
        trade_client = data_client = gcs = None

    try:
        if not dry_run:
            equity = float(trade_client.get_account().equity)
            print(f"Account equity: ${equity:,.2f}")
        else:
            equity = 0.0

        # plan entries: (ticker, signal, price, kelly_f, regime_k, n_regimes)
        plan = []

        for ticker in ASSETS:
            full_model, full_meta, regime_model, regime_metas = _load_regime_stack(ticker)
            feat_vec, ret_lag_frame = _today_feature_vector(ticker)
            n_regimes = regime_model.n_regimes if regime_model else 0

            signal, meta_used, regime_k = _get_signal_and_meta(
                ticker, full_model, full_meta, regime_model, regime_metas,
                feat_vec, ret_lag_frame,
            )
            wf_sharpe = meta_used["wf_sharpe_mean"]
            direction = "long" if signal > MODEL_THRESH else (
                "short" if signal < -MODEL_THRESH else "flat"
            )
            kelly_f = _kelly_fraction(wf_sharpe)

            print(f"[{ticker}] regime={regime_k} (of {n_regimes})  bias={signal:.4f}  "
                  f"direction={direction}  wf_sharpe={wf_sharpe:.3f}  kelly_f={kelly_f:.4f}")

            if dry_run:
                continue

            if abs(signal) < MODEL_THRESH:
                log_trade(asset=ticker, regime=regime_k, n_regimes=n_regimes,
                          signal=round(signal, 4), kelly_fraction=0.0,
                          action="flat", fill_price=0.0, shares=0,
                          account_equity=round(equity, 2))
                continue

            if kelly_f == 0.0:
                log_trade(asset=ticker, regime=regime_k, n_regimes=n_regimes,
                          signal=round(signal, 4), kelly_fraction=0.0,
                          action="flat", fill_price=0.0, shares=0,
                          account_equity=round(equity, 2))
                continue

            price = _get_price(data_client, ticker)
            plan.append((ticker, signal, price, kelly_f, regime_k, n_regimes))

        if dry_run:
            return ("Dry run complete — no orders placed", 200)

        # ── Real-time buying-power check ───────────────────────────────────────
        num_positions = len(plan)
        buying_power  = float(trade_client.get_account().buying_power)
        cap_notional  = MAX_PORTFOLIO_EXPOSURE_PCT * equity
        print(f"\nActive signals: {num_positions}  |  Equity: ${equity:,.2f}  |  "
              f"Buying power: ${buying_power:,.2f}  |  "
              f"Target deployment: ${cap_notional:,.2f} ({MAX_PORTFOLIO_EXPOSURE_PCT:.0%})")

        plan_tickers = {t for t, *_ in plan}

        # ── Close flat-signal assets that still have open positions ───────────
        print("\n-- Reconciling flat assets --")
        for ticker in ASSETS:
            if ticker not in plan_tickers:
                current_dir, current_qty = _current_position(trade_client, ticker)
                if current_dir != "flat":
                    print(f"  {ticker}: signal flat — closing {current_dir} position.")
                    _close_position(trade_client, ticker)
                    log_trade(asset=ticker, regime=-1, n_regimes=0,
                              signal=0.0, kelly_fraction=0.0, action="close",
                              fill_price=0.0, shares=current_qty,
                              account_equity=round(equity, 2))

        # ── Place, hold, or flip per-plan assets ──────────────────────────────
        print("\n-- Placing orders --")
        for ticker, signal, price, kelly_f, regime_k, n_regimes in plan:
            current_dir, current_qty = _current_position(trade_client, ticker)
            signal_dir = "long" if signal > 0 else "short"
            shares = _compute_position_size(equity, price, num_positions, buying_power)

            if current_dir == signal_dir:
                print(f"  {ticker}: already {current_dir} — holding.")
                log_trade(asset=ticker, regime=regime_k, n_regimes=n_regimes,
                          signal=round(signal, 4), kelly_fraction=round(kelly_f, 4),
                          action="hold", signal_price=round(price, 2), fill_price=round(price, 2), shares=current_qty,
                          account_equity=round(equity, 2))
                continue

            if current_dir != "flat":
                _close_position(trade_client, ticker)

            if shares < 1:
                print(f"  {ticker}: scaled to <1 share — skipping.")
                log_trade(asset=ticker, regime=regime_k, n_regimes=n_regimes,
                          signal=round(signal, 4), kelly_fraction=round(kelly_f, 4),
                          action="flat", signal_price=round(price, 2), fill_price=round(price, 2), shares=0,
                          account_equity=round(equity, 2))
                continue

            _place_order(trade_client, ticker, shares, signal, price)
            action = "buy" if signal > 0 else "sell"
            log_trade(asset=ticker, regime=regime_k, n_regimes=n_regimes,
                      signal=round(signal, 4), kelly_fraction=round(kelly_f, 4),
                      action=action, signal_price=round(price, 2), fill_price=round(price, 2), shares=shares,
                      account_equity=round(equity, 2))

        if not plan:
            print("No actionable signals — staying flat.")

        return ("Run completed", 200)

    except Exception as exc:
        print(f"run_daily error: {exc}")
        return (f"Error: {exc}", 500)

    finally:
        if not dry_run:
            _release_lock(gcs)


# ── Local testing shim ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_daily(None)
