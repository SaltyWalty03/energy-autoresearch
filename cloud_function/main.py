"""
run_daily — Google Cloud Function entry point (intraday edition).

Triggered every 5 minutes by Cloud Scheduler (Mon–Fri, ~9:25 AM – 4:05 PM ET).
Returns immediately outside market hours so idle invocations are free/cheap.

Two-layer signal architecture:
  1. DAILY BIAS  — DirectionModel loaded from GCS model.pkl.
                   Computed once per trading day; cached in GCS bias_cache.json.
                     bias > +MODEL_THRESH  → allowed direction = long
                     bias < -MODEL_THRESH  → allowed direction = short
                     |bias| ≤ MODEL_THRESH → flat; no trades today

  2. INTRADAY TIMING — RSI(14) + MACD histogram on 5-min Alpaca bars.
                   Entry fires only when intraday indicator agrees with daily bias.
                     long entry : bias=long  AND RSI < 35 AND MACD hist flips +
                     short entry: bias=short AND RSI > 65 AND MACD hist flips -
                     otherwise  → hold / stay flat

Deploy:
    gcloud functions deploy run_daily \
        --runtime python311 --region us-central1 \
        --trigger-http --allow-unauthenticated \
        --memory 512MB --timeout 120s \
        --set-env-vars GCP_PROJECT=energy-autoresearch

Scheduler (update to every 5 min during market hours):
    gcloud scheduler jobs update http run-daily-signal \
        --schedule "*/5 13-20 * * 1-5" \
        --location us-central1
"""
import io
import json
import logging
import pickle
from datetime import datetime, timedelta

import functions_framework
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from google.cloud import secretmanager, storage

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GCS_BUCKET      = "energy-autoresearch-model"
MODEL_OBJECT    = "model.pkl"
BIAS_CACHE_OBJ  = "bias_cache.json"
GCP_PROJECT     = "energy-autoresearch"
TICKER          = "XLE"
MAX_SHARES      = 100
MODEL_THRESH    = 0.05    # |bias| below this → flat today
RSI_OVERSOLD    = 35
RSI_OVERBOUGHT  = 65
LOOKBACK_BARS   = 60      # 5-min bars (covers RSI + MACD warmup)
ET              = pytz.timezone("America/New_York")
MARKET_OPEN     = (9, 30)
MARKET_CLOSE    = (15, 55)


# ── Secrets ───────────────────────────────────────────────────────────────────

def _get_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name   = f"projects/{GCP_PROJECT}/secrets/{secret_id}/versions/latest"
    resp   = client.access_secret_version(request={"name": name})
    return resp.payload.data.decode("utf-8").strip()


# ── Market hours ──────────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t <= MARKET_CLOSE


def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


# ── GCS model loading ─────────────────────────────────────────────────────────

def _load_model(gcs: storage.Client):
    blob  = gcs.bucket(GCS_BUCKET).blob(MODEL_OBJECT)
    model = pickle.load(io.BytesIO(blob.download_as_bytes()))
    log.info("Model loaded from gs://%s/%s  (%s)", GCS_BUCKET, MODEL_OBJECT, type(model).__name__)
    return model


# ── Daily bias cache (GCS JSON) ───────────────────────────────────────────────

def _load_bias_cache(gcs: storage.Client) -> dict:
    try:
        blob = gcs.bucket(GCS_BUCKET).blob(BIAS_CACHE_OBJ)
        return json.loads(blob.download_as_text())
    except Exception:
        return {}


def _save_bias_cache(gcs: storage.Client, data: dict):
    blob = gcs.bucket(GCS_BUCKET).blob(BIAS_CACHE_OBJ)
    blob.upload_from_string(json.dumps(data), content_type="application/json")
    log.info("Bias cache saved to gs://%s/%s", GCS_BUCKET, BIAS_CACHE_OBJ)


def _get_daily_bias(gcs: storage.Client, model) -> tuple[float, str]:
    """
    Return (bias_float, direction_str) for today, using GCS cache.
    Recomputes and saves if the cache is from a prior trading day.
    """
    today = _today_et()
    cache = _load_bias_cache(gcs)

    if cache.get("date") == today:
        bias      = float(cache["bias"])
        direction = cache["direction"]
        log.info("Bias cache hit for %s: bias=%.4f  direction=%s", today, bias, direction)
        return bias, direction

    # Cache miss — compute fresh bias
    log.info("Cache miss for %s — computing daily bias...", today)
    raw = yf.download(TICKER, period="90d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    closes  = raw["Close"].dropna()
    returns = closes.pct_change().dropna()
    ret_lag = returns.shift(1).dropna()

    signals   = model.predict(ret_lag.to_frame())
    bias      = float(signals[-1])
    direction = "long" if bias > MODEL_THRESH else ("short" if bias < -MODEL_THRESH else "flat")

    _save_bias_cache(gcs, {"date": today, "bias": bias, "direction": direction})
    log.info("Daily bias computed: %.4f  direction=%s", bias, direction)
    return bias, direction


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


# ── Intraday bars ─────────────────────────────────────────────────────────────

def _fetch_intraday_bars(api_key: str, secret_key: str) -> pd.DataFrame:
    data_client = StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=TICKER,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=datetime.now(ET) - timedelta(hours=10),
        feed="iex",
    )
    bars = data_client.get_stock_bars(req)
    df   = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(TICKER, level=0)
    return df.sort_index().tail(LOOKBACK_BARS).copy()


# ── Intraday signal (gated by daily bias) ─────────────────────────────────────

def _intraday_signal(df: pd.DataFrame, allowed_bias: str) -> tuple[str, float, float]:
    close = df["close"]
    rsi   = _rsi(close)
    hist  = _macd_hist(close)

    latest_rsi  = float(rsi.iloc[-1])
    latest_hist = float(hist.iloc[-1])
    prev_hist   = float(hist.iloc[-2])

    hist_flipped_up   = prev_hist < 0 and latest_hist >= 0
    hist_flipped_down = prev_hist > 0 and latest_hist <= 0

    if latest_rsi < RSI_OVERSOLD  and hist_flipped_up   and allowed_bias == "long":
        direction = "long"
    elif latest_rsi > RSI_OVERBOUGHT and hist_flipped_down and allowed_bias == "short":
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
        trade_client.cancel_orders()
    except Exception:
        pass
    try:
        trade_client.close_position(TICKER)
        log.info("Closed existing %s position.", TICKER)
    except Exception as exc:
        log.info("close_position: %s", exc)


def _place_order(trade_client: TradingClient, direction: str, price: float) -> dict:
    account = trade_client.get_account()
    equity  = float(account.buying_power)
    shares  = min(int(equity * 0.10 / price), MAX_SHARES)

    if shares < 1:
        log.info("Allocation < 1 share at $%.2f — skipping.", price)
        return {"action": "skipped", "reason": "allocation_too_small"}

    side  = OrderSide.BUY if direction == "long" else OrderSide.SELL
    order = trade_client.submit_order(MarketOrderRequest(
        symbol=TICKER,
        qty=shares,
        side=side,
        time_in_force=TimeInForce.DAY,
    ))
    log.info("Order: %s %d %s @ ~$%.2f", side.value.upper(), shares, TICKER, price)
    return {"action": side.value, "qty": shares, "order_id": str(order.id)}


# ── Entry point ───────────────────────────────────────────────────────────────

@functions_framework.http
def run_daily(request):
    """HTTP Cloud Function — called every 5 min by Cloud Scheduler."""
    try:
        if not _is_market_open():
            now = datetime.now(ET)
            msg = f"Outside market hours ({now.strftime('%a %H:%M ET')}) — no action."
            log.info(msg)
            return (json.dumps({"status": "closed", "message": msg}), 200,
                    {"Content-Type": "application/json"})

        api_key    = _get_secret("ALPACA_API_KEY")
        secret_key = _get_secret("ALPACA_SECRET_KEY")
        gcs        = storage.Client()

        # Layer 1: daily model bias (cached in GCS)
        model          = _load_model(gcs)
        bias, allowed  = _get_daily_bias(gcs, model)

        if allowed == "flat":
            log.info("Model bias flat (%.4f) — no trades today.", bias)
            return (json.dumps({"status": "flat", "bias": bias}), 200,
                    {"Content-Type": "application/json"})

        # Layer 2: intraday RSI + MACD timing
        df = _fetch_intraday_bars(api_key, secret_key)
        if len(df) < 30:
            log.warning("Only %d intraday bars — insufficient data.", len(df))
            return (json.dumps({"status": "no_data"}), 200,
                    {"Content-Type": "application/json"})

        latest_price          = float(df["close"].iloc[-1])
        direction, rsi, hist  = _intraday_signal(df, allowed)
        trade_client          = TradingClient(api_key, secret_key, paper=True)
        current_pos           = _current_position(trade_client)

        log.info(
            "Price=$%.2f  RSI=%.1f  MACDh=%.4f  Bias=%s(%.3f)  Signal=%s  Position=%s",
            latest_price, rsi, hist, allowed, bias, direction, current_pos,
        )

        result = {
            "price":       latest_price,
            "rsi":         round(rsi, 2),
            "macd_hist":   round(hist, 5),
            "model_bias":  round(bias, 4),
            "allowed":     allowed,
            "signal":      direction,
            "position":    current_pos,
        }

        if direction == current_pos:
            log.info("No change — holding %s.", current_pos)
            result["action"] = "hold"
        else:
            if current_pos != "flat":
                _close_position(trade_client)

            if direction != "flat":
                order_result = _place_order(trade_client, direction, latest_price)
                result.update(order_result)
            else:
                log.info("Intraday conditions not met — staying out.")
                result["action"] = "flat"

        return (json.dumps(result), 200, {"Content-Type": "application/json"})

    except Exception as exc:
        log.exception("run_daily failed: %s", exc)
        return (json.dumps({"error": str(exc)}), 500, {"Content-Type": "application/json"})
