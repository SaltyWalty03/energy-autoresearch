"""
run_intraday.py — Intraday XLE trading bot using Alpaca paper account.

Two-layer signal architecture:
  1. DAILY BIAS  — DirectionModel (model.pkl) predicts today's XLE direction
                   from lagged daily returns + WTI shock filter.
                   Recomputed once per trading day at startup / date rollover.
                     bias > +MODEL_THRESH  → only allow long entries today
                     bias < -MODEL_THRESH  → only allow short entries today
                     |bias| ≤ MODEL_THRESH → model says flat; no trades today

  2. INTRADAY TIMING — RSI(14) + MACD histogram on 5-min bars.
                   Entry fires only when both the model bias AND the intraday
                   indicator agree on direction.
                     long entry : bias=long  AND RSI < RSI_OVERSOLD  AND MACD hist flips +
                     short entry: bias=short AND RSI > RSI_OVERBOUGHT AND MACD hist flips -

Usage:
    python run_intraday.py
"""

import os
import json
import pickle
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from dotenv import load_dotenv, find_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

try:
    from google.cloud import storage as gcs_storage
    _GCS_AVAILABLE = True
except ImportError:
    _GCS_AVAILABLE = False

load_dotenv(find_dotenv())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET       = "energy-autoresearch-model"
LOCK_OBJECT      = "trade_lock.json"
LOCK_TTL_SECONDS = 90
TICKER          = "XLE"
BAR_INTERVAL    = TimeFrame(5, TimeFrameUnit.Minute)
LOOKBACK_BARS   = 60          # 5-min bars to fetch (covers RSI + MACD warmup)
RSI_OVERSOLD    = 35          # intraday entry threshold (long)
RSI_OVERBOUGHT  = 65          # intraday entry threshold (short)
MODEL_THRESH    = 0.05        # |bias| below this → model says flat, no trades
MAX_SHARES      = 50
POLL_SECONDS    = 300         # re-evaluate every 5 minutes
ET              = pytz.timezone("America/New_York")
MARKET_OPEN     = (9, 30)
MARKET_CLOSE    = (15, 55)    # stop 5 min before close

# Order statuses that block new order placement
BLOCKING_STATUSES = {"partially_filled", "pending_new", "accepted", "new"}

MODEL_PATH = Path(__file__).parent / "model.pkl"


# ── Position dataclass ────────────────────────────────────────────────────────

@dataclass
class Position:
    direction:    str    # "long", "short", or "flat"
    qty:          int    # absolute share count (0 if flat)
    market_value: float  # dollar value of position (0.0 if flat)

    def __str__(self) -> str:
        if self.direction == "flat":
            return "flat"
        return f"{self.direction}({self.qty} shares, ${self.market_value:,.2f})"


# ── Indicator math (pure pandas) ─────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    macd_line   = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    return macd_line - signal_line


# ── Model (daily bias) ────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    log.info("Loaded model from %s  (%s)", MODEL_PATH, type(model).__name__)
    return model


def compute_daily_bias(model) -> float:
    """
    Run DirectionModel on today's daily data.
    Returns a signal in [-1, 1]: positive = bullish bias, negative = bearish.
    The model's WTI shock filter will return 0.0 on high-volatility crude days.
    """
    raw = yf.download(TICKER, period="90d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    closes  = raw["Close"].dropna()
    returns = closes.pct_change().dropna()
    ret_lag = returns.shift(1).dropna()

    signal = model.predict(ret_lag.to_frame())
    bias   = float(signal[-1])
    log.info("Daily model bias = %.4f", bias)
    return bias


def bias_direction(bias: float) -> str:
    """Translate raw model output to 'long', 'short', or 'flat'."""
    if bias > MODEL_THRESH:
        return "long"
    elif bias < -MODEL_THRESH:
        return "short"
    return "flat"


# ── Alpaca clients ────────────────────────────────────────────────────────────

def _clients():
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    trade      = TradingClient(api_key, secret_key, paper=True)
    data       = StockHistoricalDataClient(api_key, secret_key)
    return trade, data


# ── GCS distributed lock ──────────────────────────────────────────────────────

def _gcs_client():
    if not _GCS_AVAILABLE:
        return None
    try:
        return gcs_storage.Client()
    except Exception as exc:
        log.warning("GCS client unavailable — lock disabled: %s", exc)
        return None


def _acquire_lock(gcs, owner: str) -> bool:
    """Returns True if lock acquired. Falls back to True if GCS unavailable."""
    if gcs is None:
        log.warning("GCS unavailable — proceeding without distributed lock.")
        return True
    blob = gcs.bucket(GCS_BUCKET).blob(LOCK_OBJECT)
    try:
        existing = json.loads(blob.download_as_text())
        acquired_at = datetime.fromisoformat(existing["acquired_at"]).replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - acquired_at).total_seconds()
        if age < LOCK_TTL_SECONDS:
            log.warning("Lock held by '%s' (age=%.0fs) — skipping cycle.", existing.get("owner"), age)
            return False
        log.info("Overwriting stale lock (age=%.0fs).", age)
    except Exception:
        pass
    blob.upload_from_string(
        json.dumps({"owner": owner, "acquired_at": datetime.now(timezone.utc).isoformat()}),
        content_type="application/json",
    )
    log.info("Lock acquired by '%s'.", owner)
    return True


def _release_lock(gcs):
    if gcs is None:
        return
    try:
        gcs.bucket(GCS_BUCKET).blob(LOCK_OBJECT).delete()
        log.info("Lock released.")
    except Exception:
        pass


# ── Kelly sizing ──────────────────────────────────────────────────────────────

def _kelly_shares(bias: float, equity: float, price: float) -> tuple[float, int]:
    """
    Half-Kelly sizing from live account equity.

    kelly_f      = abs(bias) * 0.5   (bias is model signal in [-1, 1])
    dollar_alloc = kelly_f * equity
    shares       = min(int(dollar_alloc / price), MAX_SHARES)

    The non-XLE portion of equity stays in cash — no bond leg is traded.
    """
    kelly_f      = abs(bias) * 0.5
    dollar_alloc = kelly_f * equity
    shares       = min(int(dollar_alloc / price), MAX_SHARES)
    log.info("Kelly: bias=%.4f  kelly_f=%.3f  equity=$%.2f  alloc=$%.2f  shares=%d",
             bias, kelly_f, equity, dollar_alloc, shares)
    return kelly_f, shares


# ── Market hours ──────────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t <= MARKET_CLOSE


# ── Intraday data + indicators ────────────────────────────────────────────────

def _fetch_bars(data_client: StockHistoricalDataClient) -> pd.DataFrame:
    start = datetime.now(ET) - timedelta(hours=10)
    req   = StockBarsRequest(
        symbol_or_symbols=TICKER,
        timeframe=BAR_INTERVAL,
        start=start,
        feed="iex",
    )
    bars = data_client.get_stock_bars(req)
    df   = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(TICKER, level=0)
    return df.sort_index().tail(LOOKBACK_BARS).copy()


def _intraday_signal(df: pd.DataFrame, allowed_bias: str) -> tuple[str, float, float]:
    """
    Compute intraday entry direction filtered by the model's daily bias.

    allowed_bias: 'long', 'short', or 'flat' (from DirectionModel)
    Returns (direction, rsi, macd_hist) where direction respects the bias.
    """
    close = df["close"]
    rsi   = _rsi(close, period=14)
    hist  = _macd_hist(close)

    latest_rsi  = float(rsi.iloc[-1])
    latest_hist = float(hist.iloc[-1])
    prev_hist   = float(hist.iloc[-2])

    hist_flipped_up   = prev_hist < 0 and latest_hist >= 0
    hist_flipped_down = prev_hist > 0 and latest_hist <= 0

    want_long  = (latest_rsi < RSI_OVERSOLD  and hist_flipped_up   and allowed_bias == "long")
    want_short = (latest_rsi > RSI_OVERBOUGHT and hist_flipped_down and allowed_bias == "short")

    if want_long:
        direction = "long"
    elif want_short:
        direction = "short"
    else:
        direction = "flat"

    return direction, latest_rsi, latest_hist


# ── Position management ───────────────────────────────────────────────────────

def _current_position(trade_client: TradingClient) -> Position:
    """Read live position directly from Alpaca — never inferred from order history."""
    try:
        pos = trade_client.get_open_position(TICKER)
        qty = int(pos.qty)
        mv  = float(pos.market_value)
        if qty > 0:
            return Position("long",  qty,  mv)
        elif qty < 0:
            return Position("short", abs(qty), mv)
    except Exception:
        pass
    return Position("flat", 0, 0.0)


def _check_pending_orders(trade_client: TradingClient) -> str | None:
    """
    Returns a warning string if any open order for TICKER is in a blocking
    status (partially_filled, pending_new, accepted, new), else None.
    Blocking prevents stacking new orders on top of unresolved fills.
    """
    try:
        orders = trade_client.get_orders(
            filter=GetOrdersRequest(status="open", symbols=[TICKER])
        )
        for o in orders:
            status = str(o.status.value) if hasattr(o.status, "value") else str(o.status)
            if status in BLOCKING_STATUSES:
                filled = int(o.filled_qty) if o.filled_qty else 0
                total  = int(o.qty) if o.qty else 0
                return (
                    f"Partial/pending order {o.id} status={status} "
                    f"({filled}/{total} filled) — waiting for resolution."
                )
    except Exception as exc:
        log.warning("_check_pending_orders failed: %s", exc)
    return None


def _shares_to_trade(target_direction: str, current: Position, target_shares: int) -> int:
    """
    Returns the net share delta to reach `target_shares` from the actual live position.
    Positive = need to buy, negative = need to sell.
    Uses live reconciled qty so partial fills are accounted for.
    """
    target_signed  = (target_shares  if target_direction == "long"  else
                      -target_shares if target_direction == "short" else 0)
    current_signed = (current.qty  if current.direction == "long"  else
                      -current.qty if current.direction == "short" else 0)
    return target_signed - current_signed


def _close_position(trade_client: TradingClient):
    try:
        trade_client.close_position(TICKER)
        log.info("Closed existing %s position.", TICKER)
    except Exception as exc:
        log.debug("close_position: %s", exc)


def _place_order(trade_client: TradingClient, delta: int, price: float):
    """Submit a market order for `delta` shares (positive=buy, negative=sell)."""
    if delta == 0:
        log.info("Delta is 0 — no order needed.")
        return

    side   = OrderSide.BUY if delta > 0 else OrderSide.SELL
    shares = abs(delta)
    order  = MarketOrderRequest(
        symbol=TICKER,
        qty=shares,
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    trade_client.submit_order(order)
    log.info("Submitted %s %d shares of %s @ ~$%.2f", side.value.upper(), shares, TICKER, price)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    log.info("Intraday bot starting. Polls every %ds during market hours.", POLL_SECONDS)

    model                     = load_model()
    trade_client, data_client = _clients()
    gcs                       = _gcs_client()

    # ── Startup reconciliation ────────────────────────────────────────────────
    startup_pos     = _current_position(trade_client)
    startup_pending = _check_pending_orders(trade_client)
    log.info("[STARTUP] Live position: %s", startup_pos)
    log.info("[STARTUP] Open orders: %s", startup_pending or "none")
    if startup_pending:
        log.warning("[STARTUP] Partial fill detected — will wait one full cycle before trading.")
    log.info("[STARTUP] Ready to trade.")

    _bias_date:    str   = ""
    _bias:         float = 0.0
    _allowed_bias: str   = "flat"
    _skip_first_cycle    = bool(startup_pending)

    while True:
        if not _is_market_open():
            now = datetime.now(ET)
            log.info("Market closed (%s). Sleeping 60s.", now.strftime("%a %H:%M ET"))
            time.sleep(60)
            continue

        today = datetime.now(ET).strftime("%Y-%m-%d")

        # Refresh daily model bias once per trading day
        if today != _bias_date:
            try:
                _bias         = compute_daily_bias(model)
                _allowed_bias = bias_direction(_bias)
                _bias_date    = today
                log.info("Daily bias refreshed for %s: raw=%.4f  direction=%s",
                         today, _bias, _allowed_bias)
                if _allowed_bias == "flat":
                    log.info("Model says flat today — no trades will be placed.")
            except Exception as exc:
                log.exception("Failed to compute daily bias: %s", exc)

        # ── Pending order guard ───────────────────────────────────────────────
        try:
            pending = _check_pending_orders(trade_client)
        except Exception as exc:
            log.warning("Could not check pending orders: %s", exc)
            pending = None

        if pending or _skip_first_cycle:
            log.warning("Pending/partial orders — skipping cycle. (%s)", pending)
            _skip_first_cycle = False
            time.sleep(POLL_SECONDS)
            continue

        # ── Distributed lock — blocks simultaneous cloud function orders ──────
        if not _acquire_lock(gcs, owner="run_intraday"):
            time.sleep(POLL_SECONDS)
            continue

        try:
            df = _fetch_bars(data_client)

            if len(df) < 30:
                log.warning("Only %d bars available — need more data. Waiting.", len(df))
                _release_lock(gcs)
                time.sleep(POLL_SECONDS)
                continue

            latest_price         = float(df["close"].iloc[-1])
            direction, rsi, hist = _intraday_signal(df, _allowed_bias)
            current              = _current_position(trade_client)

            # ── Half-Kelly sizing from live equity ────────────────────────────
            account = trade_client.get_account()
            equity  = float(account.equity)
            _, target_shares = _kelly_shares(_bias, equity, latest_price)

            log.info(
                "Price=$%.2f  RSI=%.1f  MACDh=%.4f  ModelBias=%s(%.3f)  Signal=%s  Position=%s",
                latest_price, rsi, hist, _allowed_bias, _bias, direction, current,
            )

            if direction == current.direction:
                log.info("No change needed — holding %s.", current)
            else:
                if current.direction != "flat":
                    _close_position(trade_client)
                    time.sleep(2)
                    current = _current_position(trade_client)

                if direction != "flat":
                    delta = _shares_to_trade(direction, current, target_shares)
                    if delta == 0:
                        log.info("Already at Kelly target qty=%d — no order needed.", target_shares)
                    else:
                        _place_order(trade_client, delta, latest_price)
                else:
                    log.info("Intraday conditions not met or model bias blocks trade — staying out.")

        except Exception as exc:
            log.exception("Loop error: %s", exc)
        finally:
            _release_lock(gcs)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run()
