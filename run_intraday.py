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
import pickle
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from dotenv import load_dotenv, find_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv(find_dotenv())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
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

MODEL_PATH = Path(__file__).parent / "model.pkl"


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
    ret_lag = returns.shift(1).dropna()   # feature: yesterday's return predicts today

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

    # Intraday conditions — gated by daily model bias
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

def _current_position(trade_client: TradingClient) -> str:
    try:
        pos = trade_client.get_open_position(TICKER)
        qty = int(pos.qty)
        if qty > 0:
            return "long"
        elif qty < 0:
            return "short"
    except Exception:
        pass
    return "flat"


def _close_position(trade_client: TradingClient):
    try:
        trade_client.close_position(TICKER)
        log.info("Closed existing %s position.", TICKER)
    except Exception as exc:
        log.debug("close_position: %s", exc)


def _place_order(trade_client: TradingClient, direction: str, price: float):
    account = trade_client.get_account()
    equity  = float(account.buying_power)
    shares  = min(int(equity * 0.10 / price), MAX_SHARES)

    if shares < 1:
        log.info("Allocation < 1 share — skipping.")
        return

    side  = OrderSide.BUY if direction == "long" else OrderSide.SELL
    order = MarketOrderRequest(
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

    model                  = load_model()
    trade_client, data_client = _clients()

    _bias_date: str  = ""   # tracks which calendar day the bias was computed for
    _bias: float     = 0.0
    _allowed_bias: str = "flat"

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
                _bias          = compute_daily_bias(model)
                _allowed_bias  = bias_direction(_bias)
                _bias_date     = today
                log.info(
                    "Daily bias refreshed for %s: raw=%.4f  direction=%s",
                    today, _bias, _allowed_bias,
                )
                if _allowed_bias == "flat":
                    log.info("Model says flat today — no trades will be placed.")
            except Exception as exc:
                log.exception("Failed to compute daily bias: %s", exc)

        try:
            df = _fetch_bars(data_client)

            if len(df) < 30:
                log.warning("Only %d bars available — need more data. Waiting.", len(df))
                time.sleep(POLL_SECONDS)
                continue

            latest_price              = float(df["close"].iloc[-1])
            direction, rsi, hist      = _intraday_signal(df, _allowed_bias)
            current_pos               = _current_position(trade_client)

            log.info(
                "Price=$%.2f  RSI=%.1f  MACDh=%.4f  ModelBias=%s(%.3f)  Signal=%s  Position=%s",
                latest_price, rsi, hist, _allowed_bias, _bias, direction, current_pos,
            )

            if direction == current_pos:
                log.info("No change needed — holding %s.", current_pos)
            else:
                if current_pos != "flat":
                    _close_position(trade_client)
                    time.sleep(2)

                if direction != "flat":
                    _place_order(trade_client, direction, latest_price)
                else:
                    log.info("Intraday conditions not met or model bias blocks trade — staying out.")

        except Exception as exc:
            log.exception("Loop error: %s", exc)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run()
