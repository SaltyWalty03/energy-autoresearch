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

Position sizing — half-Kelly:
  kelly_f       = abs(model_bias) * 0.5
  dollar_alloc  = kelly_f * account.equity   (live equity, not buying_power)
  shares        = min(int(dollar_alloc / price), MAX_SHARES)
  The non-XLE portion of equity is held as cash — no bond leg is traded.

Distributed lock:
  GCS object trade_lock.json prevents simultaneous orders from this function
  and run_intraday.py. Lock TTL = LOCK_TTL_SECONDS; stale locks are overwritten.

Deploy:
    gcloud functions deploy run_daily \
        --runtime python311 --region us-central1 \
        --trigger-http --allow-unauthenticated \
        --source cloud_function/ \
        --memory 512MB --timeout 120s \
        --set-env-vars GCP_PROJECT=energy-autoresearch

Scheduler:
    gcloud scheduler jobs update http run-daily-signal \
        --schedule "*/5 13-20 * * 1-5" --location us-central1
"""
import io
import json
import logging
import pickle
from datetime import datetime, timedelta, timezone

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
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from google.cloud import secretmanager, storage

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GCS_BUCKET       = "energy-autoresearch-model"
MODEL_OBJECT     = "model.pkl"
BIAS_CACHE_OBJ   = "bias_cache.json"
LOCK_OBJECT      = "trade_lock.json"
LOCK_TTL_SECONDS = 90
GCP_PROJECT      = "energy-autoresearch"
TICKER           = "XLE"
MAX_SHARES       = 100
MODEL_THRESH     = 0.05
RSI_OVERSOLD     = 35
RSI_OVERBOUGHT   = 65
LOOKBACK_BARS    = 60
ET               = pytz.timezone("America/New_York")
MARKET_OPEN      = (9, 30)
MARKET_CLOSE     = (15, 55)
BLOCKING_STATUSES = {"partially_filled", "pending_new", "accepted", "new"}


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
    return MARKET_OPEN <= (now.hour, now.minute) <= MARKET_CLOSE


def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


# ── GCS distributed lock ──────────────────────────────────────────────────────

def _acquire_lock(gcs: storage.Client, owner: str) -> bool:
    """
    Try to acquire the trade lock. Returns True if acquired.
    Reads the existing lock; if held and not stale, returns False.
    Stale locks (older than LOCK_TTL_SECONDS) are overwritten.
    """
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
        pass  # no lock exists or unreadable — safe to acquire

    blob.upload_from_string(
        json.dumps({"owner": owner, "acquired_at": datetime.now(timezone.utc).isoformat()}),
        content_type="application/json",
    )
    log.info("Lock acquired by '%s'.", owner)
    return True


def _release_lock(gcs: storage.Client):
    try:
        gcs.bucket(GCS_BUCKET).blob(LOCK_OBJECT).delete()
        log.info("Lock released.")
    except Exception:
        pass


# ── GCS model loading ─────────────────────────────────────────────────────────

def _load_model(gcs: storage.Client):
    blob  = gcs.bucket(GCS_BUCKET).blob(MODEL_OBJECT)
    model = pickle.load(io.BytesIO(blob.download_as_bytes()))
    log.info("Model loaded from gs://%s/%s  (%s)", GCS_BUCKET, MODEL_OBJECT, type(model).__name__)
    return model


# ── Daily bias cache (GCS JSON) ───────────────────────────────────────────────

def _load_bias_cache(gcs: storage.Client) -> dict:
    try:
        return json.loads(gcs.bucket(GCS_BUCKET).blob(BIAS_CACHE_OBJ).download_as_text())
    except Exception:
        return {}


def _save_bias_cache(gcs: storage.Client, data: dict):
    gcs.bucket(GCS_BUCKET).blob(BIAS_CACHE_OBJ).upload_from_string(
        json.dumps(data), content_type="application/json"
    )
    log.info("Bias cache saved to gs://%s/%s", GCS_BUCKET, BIAS_CACHE_OBJ)


def _get_daily_bias(gcs: storage.Client, model) -> tuple[float, str]:
    today = _today_et()
    cache = _load_bias_cache(gcs)
    if cache.get("date") == today:
        bias, direction = float(cache["bias"]), cache["direction"]
        log.info("Bias cache hit for %s: bias=%.4f  direction=%s", today, bias, direction)
        return bias, direction

    log.info("Cache miss for %s — computing daily bias...", today)
    raw = yf.download(TICKER, period="90d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    ret_lag   = raw["Close"].dropna().pct_change().dropna().shift(1).dropna()
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
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    macd_line   = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd_line - macd_line.ewm(span=sig, adjust=False).mean()


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


# ── Intraday signal ───────────────────────────────────────────────────────────

def _intraday_signal(df: pd.DataFrame, allowed_bias: str) -> tuple[str, float, float]:
    close = df["close"]
    rsi   = _rsi(close)
    hist  = _macd_hist(close)
    latest_rsi, latest_hist, prev_hist = float(rsi.iloc[-1]), float(hist.iloc[-1]), float(hist.iloc[-2])

    if latest_rsi < RSI_OVERSOLD  and prev_hist < 0 <= latest_hist and allowed_bias == "long":
        return "long",  latest_rsi, latest_hist
    if latest_rsi > RSI_OVERBOUGHT and prev_hist > 0 >= latest_hist and allowed_bias == "short":
        return "short", latest_rsi, latest_hist
    return "flat", latest_rsi, latest_hist


# ── Position management ───────────────────────────────────────────────────────

def _current_position(trade_client: TradingClient) -> tuple[str, int]:
    """Returns (direction, abs_qty) read directly from live Alpaca account."""
    try:
        pos = trade_client.get_open_position(TICKER)
        qty = int(pos.qty)
        if qty > 0:
            return "long", qty
        elif qty < 0:
            return "short", abs(qty)
    except Exception:
        pass
    return "flat", 0


def _check_pending_orders(trade_client: TradingClient) -> str | None:
    try:
        orders = trade_client.get_orders(filter=GetOrdersRequest(status="open", symbols=[TICKER]))
        for o in orders:
            status = str(o.status.value) if hasattr(o.status, "value") else str(o.status)
            if status in BLOCKING_STATUSES:
                filled = int(o.filled_qty) if o.filled_qty else 0
                return f"Order {o.id} status={status} ({filled}/{int(o.qty)} filled) — blocked."
    except Exception as exc:
        log.warning("_check_pending_orders failed: %s", exc)
    return None


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


def _kelly_shares(bias: float, equity: float, price: float) -> tuple[float, int]:
    """
    Half-Kelly position sizing from live account equity.

    kelly_f      = abs(bias) * 0.5   (bias is model signal in [-1,1])
    dollar_alloc = kelly_f * equity
    shares       = min(int(dollar_alloc / price), MAX_SHARES)

    The non-XLE portion of equity is held as cash.
    No bond leg is traded — the Kelly fraction applies to XLE only.
    """
    kelly_f      = abs(bias) * 0.5
    dollar_alloc = kelly_f * equity
    shares       = min(int(dollar_alloc / price), MAX_SHARES)
    log.info("Kelly sizing: bias=%.4f  kelly_f=%.3f  equity=$%.2f  alloc=$%.2f  shares=%d",
             bias, kelly_f, equity, dollar_alloc, shares)
    return kelly_f, shares


def _shares_to_trade(target_direction: str, current_qty_signed: int, target_qty: int) -> int:
    """Net delta to reach target from actual live position."""
    target_signed = target_qty if target_direction == "long" else -target_qty
    return target_signed - current_qty_signed


def _place_order(trade_client: TradingClient, direction: str, bias: float, price: float) -> dict:
    account = trade_client.get_account()
    equity  = float(account.equity)          # live equity, not buying_power
    _, target_shares = _kelly_shares(bias, equity, price)

    if target_shares < 1:
        log.info("Kelly allocation < 1 share at $%.2f — staying flat.", price)
        return {"action": "skipped", "reason": "kelly_too_small"}

    current_dir, current_qty = _current_position(trade_client)
    current_signed = current_qty if current_dir == "long" else -current_qty
    delta = _shares_to_trade(direction, current_signed, target_shares)

    if delta == 0:
        log.info("Already at target qty=%d — no order needed.", target_shares)
        return {"action": "hold", "qty": current_qty}

    side  = OrderSide.BUY if delta > 0 else OrderSide.SELL
    order = trade_client.submit_order(MarketOrderRequest(
        symbol=TICKER, qty=abs(delta), side=side, time_in_force=TimeInForce.DAY,
    ))
    log.info("Order: %s %d %s @ ~$%.2f  (kelly_f=%.3f)", side.value.upper(), abs(delta), TICKER, price, abs(bias)*0.5)
    return {"action": side.value, "qty": abs(delta), "order_id": str(order.id)}


# ── Entry point ───────────────────────────────────────────────────────────────

@functions_framework.http
def run_daily(request):
    """HTTP Cloud Function — called every 5 min by Cloud Scheduler."""
    gcs = None
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

        # ── Distributed lock — only one script trades at a time ───────────────
        if not _acquire_lock(gcs, owner="cloud_function"):
            return (json.dumps({"status": "locked", "message": "Trade lock held — skipping."}),
                    200, {"Content-Type": "application/json"})

        # ── Pending order guard ───────────────────────────────────────────────
        trade_client = TradingClient(api_key, secret_key, paper=True)
        pending = _check_pending_orders(trade_client)
        if pending:
            log.warning("Pending orders detected — skipping cycle. %s", pending)
            _release_lock(gcs)
            return (json.dumps({"status": "pending", "message": pending}), 200,
                    {"Content-Type": "application/json"})

        # ── Layer 1: daily model bias ─────────────────────────────────────────
        model         = _load_model(gcs)
        bias, allowed = _get_daily_bias(gcs, model)

        if allowed == "flat":
            log.info("Model bias flat (%.4f) — no trades today.", bias)
            _release_lock(gcs)
            return (json.dumps({"status": "flat", "bias": bias}), 200,
                    {"Content-Type": "application/json"})

        # ── Layer 2: intraday RSI + MACD timing ───────────────────────────────
        df = _fetch_intraday_bars(api_key, secret_key)
        if len(df) < 30:
            log.warning("Only %d intraday bars — insufficient data.", len(df))
            _release_lock(gcs)
            return (json.dumps({"status": "no_data"}), 200,
                    {"Content-Type": "application/json"})

        latest_price         = float(df["close"].iloc[-1])
        direction, rsi, hist = _intraday_signal(df, allowed)
        current_dir, current_qty = _current_position(trade_client)

        log.info(
            "Price=$%.2f  RSI=%.1f  MACDh=%.4f  Bias=%s(%.3f)  Signal=%s  Position=%s(%d)",
            latest_price, rsi, hist, allowed, bias, direction, current_dir, current_qty,
        )

        result = {
            "price": latest_price, "rsi": round(rsi, 2), "macd_hist": round(hist, 5),
            "model_bias": round(bias, 4), "allowed": allowed, "signal": direction,
            "position": current_dir, "position_qty": current_qty,
        }

        if direction == current_dir:
            log.info("No change — holding %s(%d).", current_dir, current_qty)
            result["action"] = "hold"
        else:
            if current_dir != "flat":
                _close_position(trade_client)

            if direction != "flat":
                order_result = _place_order(trade_client, direction, bias, latest_price)
                result.update(order_result)
            else:
                log.info("Intraday conditions not met — staying out.")
                result["action"] = "flat"

        _release_lock(gcs)
        return (json.dumps(result), 200, {"Content-Type": "application/json"})

    except Exception as exc:
        log.exception("run_daily failed: %s", exc)
        if gcs:
            _release_lock(gcs)
        return (json.dumps({"error": str(exc)}), 500, {"Content-Type": "application/json"})
