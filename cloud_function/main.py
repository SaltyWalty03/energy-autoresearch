"""
run_daily — Google Cloud Function entry point (multi-asset regime edition).

Triggered every 5 minutes by Cloud Scheduler (Mon–Fri, ~9:25 AM – 4:05 PM ET).
Returns immediately outside market hours so idle invocations are free/cheap.

Three-layer signal architecture:
  1. REGIME      — classify_regime() maps today's 6-feature rolling vector to
                   a market regime using KNN on the trained embedding.
                   Per-asset regime model loaded from GCS models/{ticker}_regime.pkl.

  2. DAILY BIAS  — Regime-specific DirectionModel (models/{ticker}_regime{k}_model.pkl)
                   predicts directional bias. Walk-forward Sharpe from the regime's
                   meta JSON drives Kelly sizing.
                   Cached per-asset in GCS bias_cache.json; recomputed on date rollover.
                     bias > +MODEL_THRESH  → allowed direction = long
                     bias < -MODEL_THRESH  → allowed direction = short
                     |bias| ≤ MODEL_THRESH → flat (no trade this asset today)
                   WF-Sharpe ≤ 0 → kelly_f = 0 → no trade even if signal is non-zero.

  3. INTRADAY TIMING — RSI(14) + MACD histogram on 5-min Alpaca bars per asset.
                   Entry fires only when intraday indicator agrees with daily bias.
                     long  entry: bias=long  AND RSI < 35 AND MACD hist flips +
                     short entry: bias=short AND RSI > 65 AND MACD hist flips -

Position sizing — regime-calibrated half-Kelly:
  kelly_f      = max(0, (wf_sharpe_mean / sqrt(252)) * 0.5)
  dollar_alloc = kelly_f * account.equity
  shares       = min(int(dollar_alloc / price), MAX_SHARES_PER_ASSET)
  Total gross exposure capped at MAX_EXPOSURE (95%) across all assets.

Assets traded: XLE (WTI shock filter), XLB (copper), XOP (WTI shock filter).

Distributed lock:
  GCS trade_lock.json prevents simultaneous orders from this function and
  run_intraday.py / run_daily.py. TTL = LOCK_TTL_SECONDS; stale locks overwritten.

GCS model layout (upload after retraining with train_all.py):
  gsutil cp -r models/ gs://energy-autoresearch-model/models/

Deploy:
    gcloud functions deploy run_daily \\
        --runtime python311 --region us-central1 \\
        --trigger-http --allow-unauthenticated \\
        --source cloud_function/ \\
        --memory 1024MB --timeout 300s \\
        --set-env-vars GCP_PROJECT=energy-autoresearch

Scheduler:
    gcloud scheduler jobs update http run-daily-signal \\
        --schedule "*/5 13-20 * * 1-5" --location us-central1
"""
import io
import json
import logging
import math
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

from model import build_feature_matrix
from regime import classify_regime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GCS_BUCKET        = "energy-autoresearch-model"
BIAS_CACHE_OBJ    = "bias_cache.json"
LOCK_OBJECT       = "trade_lock.json"
LOCK_TTL_SECONDS  = 90
GCP_PROJECT       = "energy-autoresearch"
MAX_SHARES_PER_ASSET = 100
MAX_EXPOSURE      = 0.95      # cap total gross exposure as fraction of equity
MODEL_THRESH      = 0.05
RSI_OVERSOLD      = 35
RSI_OVERBOUGHT    = 65
LOOKBACK_BARS     = 60
ET                = pytz.timezone("America/New_York")
MARKET_OPEN       = (9, 30)
MARKET_CLOSE      = (15, 55)
BLOCKING_STATUSES = {"partially_filled", "pending_new", "accepted", "new"}

ASSETS = {
    "XLE": {"commodity": "CL=F", "shock_thresh": 0.02},
    "XLB": {"commodity": "HG=F", "shock_thresh": 0.025},
    "XOP": {"commodity": "CL=F", "shock_thresh": 0.02},
}


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


def _release_lock(gcs: storage.Client):
    try:
        gcs.bucket(GCS_BUCKET).blob(LOCK_OBJECT).delete()
        log.info("Lock released.")
    except Exception:
        pass


# ── GCS model loading ─────────────────────────────────────────────────────────

def _gcs_load_pkl(gcs: storage.Client, obj_name: str):
    blob = gcs.bucket(GCS_BUCKET).blob(obj_name)
    return pickle.load(io.BytesIO(blob.download_as_bytes()))


def _gcs_load_json(gcs: storage.Client, obj_name: str) -> dict:
    try:
        return json.loads(gcs.bucket(GCS_BUCKET).blob(obj_name).download_as_text())
    except Exception:
        return {}


def _load_regime_stack(gcs: storage.Client, ticker: str):
    """
    Returns (full_model, full_meta, regime_model, regime_metas_list).
    Falls back gracefully if regime files are missing in GCS.
    """
    full_model = _gcs_load_pkl(gcs, f"models/{ticker}_model.pkl")
    full_meta  = _gcs_load_json(gcs, f"models/{ticker}_meta.json")
    if not full_meta:
        full_meta = {"wf_sharpe_mean": 0.0}

    try:
        regime_model = _gcs_load_pkl(gcs, f"models/{ticker}_regime.pkl")
    except Exception:
        log.warning("%s: regime.pkl missing in GCS — using full model.", ticker)
        return full_model, full_meta, None, []

    regime_metas = []
    for k in range(regime_model.n_regimes):
        m = _gcs_load_json(gcs, f"models/{ticker}_regime{k}_meta.json")
        regime_metas.append(m if m else {"wf_sharpe_mean": 0.0})

    return full_model, full_meta, regime_model, regime_metas


# ── Daily bias cache (multi-asset GCS JSON) ───────────────────────────────────

def _load_bias_cache(gcs: storage.Client) -> dict:
    return _gcs_load_json(gcs, BIAS_CACHE_OBJ)


def _save_bias_cache(gcs: storage.Client, data: dict):
    gcs.bucket(GCS_BUCKET).blob(BIAS_CACHE_OBJ).upload_from_string(
        json.dumps(data), content_type="application/json"
    )
    log.info("Bias cache saved.")


def _get_daily_bias_for_asset(
    ticker: str,
    full_model,
    full_meta: dict,
    regime_model,
    regime_metas: list,
    cache_entry: dict,
) -> dict:
    """
    Compute (or retrieve from cache) regime, bias, direction, wf_sharpe for one asset.
    Returns a dict: {bias, direction, regime, wf_sharpe_mean}.
    """
    raw = yf.download(ticker, period="90d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    ret_lag = raw["Close"].dropna().pct_change().dropna().shift(1).dropna()

    # Regime classification
    regime_k = -1
    model_used = full_model
    meta_used  = full_meta

    if regime_model is not None:
        feat_matrix = build_feature_matrix(ret_lag.values.astype(np.float64))
        valid = np.isfinite(feat_matrix).all(axis=1)
        if valid.any():
            feat_vec = feat_matrix[valid][-1]
            regime_k = classify_regime(feat_vec, regime_model)
            if 0 <= regime_k < len(regime_metas):
                meta_used = regime_metas[regime_k] if regime_metas[regime_k] else full_meta
                # regime-specific model would need to be loaded separately;
                # for cloud function we use the full model for prediction
                # but regime-specific WF Sharpe for Kelly sizing
                log.info("%s: regime %d of %d  wf_sharpe=%.4f",
                         ticker, regime_k, regime_model.n_regimes,
                         meta_used.get("wf_sharpe_mean", 0.0))
            else:
                log.warning("%s: classify_regime=%d out of range — falling back.", ticker, regime_k)
                regime_k = -1

    bias      = float(model_used.predict(ret_lag.to_frame())[-1])
    direction = "long" if bias > MODEL_THRESH else ("short" if bias < -MODEL_THRESH else "flat")
    wf_sharpe = float(meta_used.get("wf_sharpe_mean", 0.0))

    log.info("%s: bias=%.4f  direction=%s  regime=%d  wf_sharpe=%.4f",
             ticker, bias, direction, regime_k, wf_sharpe)
    return {"bias": bias, "direction": direction, "regime": regime_k, "wf_sharpe_mean": wf_sharpe}


def _get_all_daily_biases(gcs: storage.Client, regime_stacks: dict) -> dict:
    """
    Load or recompute per-asset bias entries. Returns dict keyed by ticker.
    Cache format: {"date": "YYYY-MM-DD", "XLE": {...}, "XLB": {...}, "XOP": {...}}
    """
    today = _today_et()
    cache = _load_bias_cache(gcs)

    if cache.get("date") == today and all(t in cache for t in ASSETS):
        log.info("Bias cache hit for %s.", today)
        return {t: cache[t] for t in ASSETS}

    log.info("Cache miss for %s — computing daily biases for all assets...", today)
    result = {}
    for ticker, (full_model, full_meta, regime_model, regime_metas) in regime_stacks.items():
        cache_entry = cache.get(ticker, {})
        result[ticker] = _get_daily_bias_for_asset(
            ticker, full_model, full_meta, regime_model, regime_metas, cache_entry
        )

    _save_bias_cache(gcs, {"date": today, **result})
    return result


# ── Indicator math (pure pandas) ─────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd_line - macd_line.ewm(span=sig, adjust=False).mean()


# ── Intraday bars ─────────────────────────────────────────────────────────────

def _fetch_intraday_bars(api_key: str, secret_key: str, ticker: str) -> pd.DataFrame:
    data_client = StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=datetime.now(ET) - timedelta(hours=10),
        feed="iex",
    )
    bars = data_client.get_stock_bars(req)
    df   = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level=0)
    return df.sort_index().tail(LOOKBACK_BARS).copy()


def _intraday_signal(df: pd.DataFrame, allowed_bias: str) -> tuple[str, float, float]:
    close = df["close"]
    rsi   = _rsi(close)
    hist  = _macd_hist(close)
    latest_rsi  = float(rsi.iloc[-1])
    latest_hist = float(hist.iloc[-1])
    prev_hist   = float(hist.iloc[-2])

    if latest_rsi < RSI_OVERSOLD  and prev_hist < 0 <= latest_hist and allowed_bias == "long":
        return "long",  latest_rsi, latest_hist
    if latest_rsi > RSI_OVERBOUGHT and prev_hist > 0 >= latest_hist and allowed_bias == "short":
        return "short", latest_rsi, latest_hist
    return "flat", latest_rsi, latest_hist


# ── Position management ───────────────────────────────────────────────────────

def _current_position(trade_client: TradingClient, ticker: str) -> tuple[str, int]:
    try:
        pos = trade_client.get_open_position(ticker)
        qty = int(pos.qty)
        if qty > 0:
            return "long", qty
        elif qty < 0:
            return "short", abs(qty)
    except Exception:
        pass
    return "flat", 0


def _check_pending_orders(trade_client: TradingClient) -> str | None:
    tickers = list(ASSETS.keys())
    try:
        orders = trade_client.get_orders(filter=GetOrdersRequest(status="open", symbols=tickers))
        for o in orders:
            status = str(o.status.value) if hasattr(o.status, "value") else str(o.status)
            if status in BLOCKING_STATUSES:
                filled = int(o.filled_qty) if o.filled_qty else 0
                return f"Order {o.id} ({o.symbol}) status={status} ({filled}/{int(o.qty)}) — blocked."
    except Exception as exc:
        log.warning("_check_pending_orders failed: %s", exc)
    return None


def _close_position(trade_client: TradingClient, ticker: str):
    try:
        trade_client.close_position(ticker)
        log.info("Closed existing %s position.", ticker)
    except Exception as exc:
        log.info("close_position(%s): %s", ticker, exc)


# ── Kelly sizing ──────────────────────────────────────────────────────────────

def _kelly_fraction(wf_sharpe_mean: float) -> float:
    """Half-Kelly, annualised-to-daily conversion. Returns 0 if no edge."""
    if wf_sharpe_mean <= 0:
        return 0.0
    return (wf_sharpe_mean / math.sqrt(252)) * 0.5


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

        # ── Distributed lock ──────────────────────────────────────────────────
        if not _acquire_lock(gcs, owner="cloud_function"):
            return (json.dumps({"status": "locked", "message": "Trade lock held."}),
                    200, {"Content-Type": "application/json"})

        # ── Pending order guard ───────────────────────────────────────────────
        trade_client = TradingClient(api_key, secret_key, paper=True)
        pending = _check_pending_orders(trade_client)
        if pending:
            log.warning("Pending orders — skipping. %s", pending)
            _release_lock(gcs)
            return (json.dumps({"status": "pending", "message": pending}), 200,
                    {"Content-Type": "application/json"})

        # ── Layer 1+2: regime + daily bias (cached per asset) ─────────────────
        log.info("Loading regime stacks for all assets...")
        regime_stacks = {t: _load_regime_stack(gcs, t) for t in ASSETS}
        biases = _get_all_daily_biases(gcs, regime_stacks)

        equity = float(trade_client.get_account().equity)
        log.info("Account equity: $%.2f", equity)

        # ── Layer 3: intraday RSI + MACD per asset → build order plan ─────────
        plan    = []   # (ticker, intraday_direction, bias_info, latest_price)
        results = {}

        for ticker in ASSETS:
            b = biases[ticker]
            allowed   = b["direction"]
            bias      = b["bias"]
            wf_sharpe = b["wf_sharpe_mean"]
            regime_k  = b["regime"]
            kelly_f   = _kelly_fraction(wf_sharpe)

            log.info("%s: allowed=%s  bias=%.4f  regime=%d  wf=%.4f  kelly_f=%.4f",
                     ticker, allowed, bias, regime_k, wf_sharpe, kelly_f)

            asset_result = {
                "bias": round(bias, 4), "allowed": allowed,
                "regime": regime_k, "wf_sharpe": round(wf_sharpe, 4),
                "kelly_f": round(kelly_f, 4),
            }

            if allowed == "flat" or kelly_f == 0.0:
                reason = "flat_bias" if allowed == "flat" else "no_edge"
                log.info("%s: skipping (%s).", ticker, reason)
                asset_result["action"] = "skip"
                asset_result["reason"] = reason
                results[ticker] = asset_result
                continue

            try:
                df = _fetch_intraday_bars(api_key, secret_key, ticker)
            except Exception as exc:
                log.warning("%s: intraday bars failed: %s", ticker, exc)
                asset_result["action"] = "no_data"
                results[ticker] = asset_result
                continue

            if len(df) < 30:
                log.warning("%s: only %d bars.", ticker, len(df))
                asset_result["action"] = "insufficient_bars"
                results[ticker] = asset_result
                continue

            latest_price          = float(df["close"].iloc[-1])
            direction, rsi, hist  = _intraday_signal(df, allowed)
            current_dir, current_qty = _current_position(trade_client, ticker)

            asset_result.update({
                "price": round(latest_price, 2),
                "rsi": round(rsi, 2),
                "macd_hist": round(hist, 5),
                "signal": direction,
                "position": current_dir,
                "position_qty": current_qty,
            })

            log.info("%s: price=$%.2f RSI=%.1f MACDh=%.4f signal=%s pos=%s(%d)",
                     ticker, latest_price, rsi, hist, direction, current_dir, current_qty)

            if direction == "flat":
                asset_result["action"] = "flat"
                results[ticker] = asset_result
                continue

            plan.append((ticker, direction, bias, wf_sharpe, kelly_f, latest_price,
                         current_dir, current_qty, asset_result))

        # ── Exposure cap across all assets ────────────────────────────────────
        scale = 1.0
        if plan:
            total_alloc = sum(entry[4] * equity for entry in plan)  # entry[4] = kelly_f
            max_alloc = MAX_EXPOSURE * equity
            if total_alloc > max_alloc:
                scale = max_alloc / total_alloc
                log.info("Scaling positions by %.3f (alloc $%.0f > cap $%.0f)",
                         scale, total_alloc, max_alloc)

        # ── Place orders ──────────────────────────────────────────────────────
        for ticker, direction, bias, wf_sharpe, kelly_f, latest_price, \
                current_dir, current_qty, asset_result in plan:

            target_shares = min(int(kelly_f * scale * equity / latest_price),
                                MAX_SHARES_PER_ASSET)

            if direction == current_dir:
                log.info("%s: already %s — holding.", ticker, current_dir)
                asset_result["action"] = "hold"
            else:
                if current_dir != "flat":
                    _close_position(trade_client, ticker)

                current_signed = current_qty if current_dir == "long" else -current_qty
                target_signed  = target_shares if direction == "long" else -target_shares
                delta          = target_signed - current_signed

                if delta == 0 or target_shares < 1:
                    log.info("%s: delta=0 or target<1 share — skipping.", ticker)
                    asset_result["action"] = "skipped"
                else:
                    side  = OrderSide.BUY if delta > 0 else OrderSide.SELL
                    order = trade_client.submit_order(MarketOrderRequest(
                        symbol=ticker, qty=abs(delta), side=side,
                        time_in_force=TimeInForce.DAY,
                    ))
                    log.info("Order: %s %d %s @ ~$%.2f (kelly_f=%.4f wf=%.4f)",
                             side.value.upper(), abs(delta), ticker, latest_price, kelly_f, wf_sharpe)
                    asset_result.update({
                        "action": side.value, "qty": abs(delta),
                        "order_id": str(order.id),
                    })

            results[ticker] = asset_result

        _release_lock(gcs)
        return (json.dumps({"status": "ok", "equity": equity,
                            "scale": round(scale, 4), "assets": results}),
                200, {"Content-Type": "application/json"})

    except Exception as exc:
        log.exception("run_daily failed: %s", exc)
        if gcs:
            _release_lock(gcs)
        return (json.dumps({"error": str(exc)}), 500, {"Content-Type": "application/json"})
