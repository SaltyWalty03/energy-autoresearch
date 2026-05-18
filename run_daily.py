"""
run_daily.py — Multi-asset local paper-trading signal via Alpaca.

Loads per-asset models from models/, generates directional signals for each
asset in ASSETS (XLE, XLB, XOP), sizes positions with half-Kelly calibrated
from each asset's walk-forward Sharpe, caps total gross exposure at 95% of
equity, and places orders under the GCS distributed lock.

Kelly formula (half-Kelly, annualised-to-daily):
    kelly_f = max(0, (wf_sharpe_mean / sqrt(252)) * 0.5)
    dollar_alloc = kelly_f * equity
If walk-forward Sharpe <= 0 for an asset, it is skipped entirely.

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
MAX_EXPOSURE = 0.95      # max gross exposure as fraction of equity
MAX_SHARES_PER_ASSET = 100
MODEL_THRESH = 0.05      # flat-zone for signal magnitude

from src.universe import ASSETS


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

def _load_asset_model(ticker: str):
    model_path = MODELS_DIR / f"{ticker}_model.pkl"
    meta_path  = MODELS_DIR / f"{ticker}_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model for {ticker}. Run: python train_all.py"
        )
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"wf_sharpe_mean": 0.0}
    return model, meta


# ── Signal generation ──────────────────────────────────────────────────────────

def _generate_signal(ticker: str, model) -> float:
    raw = yf.download(ticker, period="90d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    ret_lag = raw["Close"].dropna().pct_change().dropna().shift(1).dropna()
    signal = model.predict(ret_lag.to_frame())[-1]
    return float(signal)


# ── Latest ask price ──────────────────────────────────────────────────────────

def _get_price(data_client: StockHistoricalDataClient, ticker: str) -> float:
    quote = data_client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=ticker)
    )
    return float(quote[ticker].ask_price)


# ── Kelly sizing ───────────────────────────────────────────────────────────────

def _kelly_fraction(wf_sharpe_mean: float) -> float:
    """Half-Kelly with annualised-to-daily Sharpe conversion."""
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
        symbol=ticker,
        qty=shares,
        side=side,
        time_in_force=TimeInForce.DAY,
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

        # ── Build per-asset allocation plan ───────────────────────────────────
        plan = []  # list of (ticker, signal, shares_raw, price, kelly_f)

        for ticker in ASSETS:
            print(f"\n[{ticker}]")
            model, meta = _load_asset_model(ticker)
            signal = _generate_signal(ticker, model)
            print(f"  Signal: {signal:.4f}  WF-Sharpe: {meta['wf_sharpe_mean']:.4f}")

            if abs(signal) < MODEL_THRESH:
                print("  Signal in flat zone — skipping.")
                continue

            kelly_f = _kelly_fraction(meta["wf_sharpe_mean"])
            if kelly_f == 0.0:
                print("  WF-Sharpe <= 0 — no edge, skipping.")
                continue

            price = _get_price(data_client, ticker)
            dollar_alloc = kelly_f * equity
            shares_raw = int(dollar_alloc / price)
            plan.append((ticker, signal, shares_raw, price, kelly_f))
            print(f"  kelly_f={kelly_f:.4f}  alloc=${dollar_alloc:,.0f}  raw_shares={shares_raw}")

        # ── Cap total gross exposure at MAX_EXPOSURE ───────────────────────────
        if plan:
            total_alloc = sum(s * p for _, _, s, p, _ in plan)
            max_alloc   = MAX_EXPOSURE * equity
            scale       = min(1.0, max_alloc / total_alloc) if total_alloc > 0 else 1.0
            if scale < 1.0:
                print(f"\nScaling positions by {scale:.3f} (total alloc ${total_alloc:,.0f} > cap ${max_alloc:,.0f})")

        # ── Close all ASSET positions first, then open new ones ───────────────
        print("\n── Closing existing positions ──────────────────────────")
        for ticker in ASSETS:
            _close_position(trade_client, ticker)

        # ── Place orders ───────────────────────────────────────────────────────
        print("\n── Placing orders ──────────────────────────────────────")
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
