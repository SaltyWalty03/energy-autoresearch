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
                     long  entry: bias=long  AND RSI < 45 AND MACD hist > 0
                     short entry: bias=short AND RSI > 55 AND MACD hist < 0

Position sizing — equal-weight allocation:
  per_position_notional = MAX_PORTFOLIO_EXPOSURE_PCT * equity / num_active_signals
  safe_notional         = min(per_position_notional, buying_power)   [live Alpaca check]
  shares                = int(safe_notional / price)
  WF-Sharpe ≤ 0 → kelly_f = 0 → asset skipped (edge gate, not sizing).

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
import math
import os
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

from model import (build_feature_matrix, N_BASE_FEATURES, N_EXTENDED_FEATURES,
                   FEATURE_NAMES_BASE, FEATURE_NAMES_EXTENDED)
from regime import classify_regime
from src.universe import ASSETS  # canonical asset config — never duplicate inline

import sys as _sys


class _StructuredLogger:
    """Emit Cloud Logging-compatible JSON to stdout (required for Gen 2 functions)."""

    def _emit(self, severity: str, msg: str) -> None:
        entry = json.dumps({"severity": severity, "message": msg})
        print(entry, flush=True)

    def info(self, msg: str, *args) -> None:
        self._emit("INFO", msg % args if args else msg)

    def warning(self, msg: str, *args) -> None:
        self._emit("WARNING", msg % args if args else msg)

    def error(self, msg: str, *args) -> None:
        self._emit("ERROR", msg % args if args else msg)

    def exception(self, msg: str, *args) -> None:
        import traceback
        tb = traceback.format_exc()
        self._emit("ERROR", (msg % args if args else msg) + "\n" + tb)


log = _StructuredLogger()

# ── Constants ─────────────────────────────────────────────────────────────────
GCS_BUCKET           = "energy-autoresearch-model"
BIAS_CACHE_OBJ       = "bias_cache.json"
LOCK_OBJECT          = "trade_lock.json"
MODEL_POINTER_OBJ    = "models/pointer.json"
MODEL_PREFIX_DEFAULT = "models/"  # fallback when pointer is absent
LOCK_TTL_SECONDS     = 90
GCP_PROJECT          = "energy-autoresearch"
MODEL_THRESH         = 0.02
RSI_OVERSOLD         = 45
RSI_OVERBOUGHT       = 55
LOOKBACK_BARS        = 60
ET                   = pytz.timezone("America/New_York")
MARKET_OPEN          = (9, 30)
MARKET_CLOSE         = (15, 55)
BLOCKING_STATUSES    = {"partially_filled", "pending_new", "accepted", "new"}

# ── Runtime config (override via env vars for local testing) ──────────────────
CFG = {
    # Maximum gross notional as a fraction of account equity.
    # Override: GROSS_CAP_PCT=0.75 python cloud_function/main.py --dry-run
    "gross_cap_pct": float(os.environ.get("GROSS_CAP_PCT", "0.80")),
}
MAX_PORTFOLIO_EXPOSURE_PCT = CFG["gross_cap_pct"]  # backward-compat alias


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
    except Exception as exc:
        # NotFound is the normal "no lock held" case; anything else is worth seeing.
        if type(exc).__name__ == "NotFound":
            log.info("No existing lock object.")
        else:
            log.warning("Could not read existing lock (%s: %s) — acquiring anyway.",
                        type(exc).__name__, exc)
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
    except Exception as exc:
        log.warning("Lock release failed (%s: %s) — it will expire after %ds.",
                    type(exc).__name__, exc, LOCK_TTL_SECONDS)


# ── GCS model loading ─────────────────────────────────────────────────────────

def _gcs_load_pkl(gcs: storage.Client, obj_name: str):
    blob = gcs.bucket(GCS_BUCKET).blob(obj_name)
    return pickle.load(io.BytesIO(blob.download_as_bytes()))


def _gcs_load_json(gcs: storage.Client, obj_name: str) -> dict:
    try:
        return json.loads(gcs.bucket(GCS_BUCKET).blob(obj_name).download_as_text())
    except Exception as exc:
        # A missing meta file silently becomes wf_sharpe=0 → no_edge, so always say so.
        log.warning("GCS JSON %s unavailable (%s: %s) — using empty dict.",
                    obj_name, type(exc).__name__, exc)
        return {}


def _get_model_prefix(gcs: storage.Client) -> str:
    """Read models/pointer.json to find the active versioned prefix."""
    try:
        data = json.loads(
            gcs.bucket(GCS_BUCKET).blob(MODEL_POINTER_OBJ).download_as_text()
        )
        prefix = data.get("prefix", MODEL_PREFIX_DEFAULT)
        log.info("Model prefix from pointer: %s (uploaded_at=%s)",
                 prefix, data.get("uploaded_at", "?"))
        return prefix
    except Exception as exc:
        log.warning("pointer.json unavailable (%s: %s) — using default prefix %s",
                    type(exc).__name__, exc, MODEL_PREFIX_DEFAULT)
        return MODEL_PREFIX_DEFAULT


def _load_regime_stack(gcs: storage.Client, ticker: str, prefix: str = MODEL_PREFIX_DEFAULT):
    """
    Returns (full_model, full_meta, regime_model, regime_metas_list).
    Falls back gracefully if regime files are missing in GCS.
    """
    full_model = _gcs_load_pkl(gcs, f"{prefix}{ticker}_model.pkl")
    full_meta  = _gcs_load_json(gcs, f"{prefix}{ticker}_meta.json")
    if not full_meta:
        full_meta = {"wf_sharpe_mean": 0.0}

    log.info("%s: full model from prefix=%s  n_features=%d", ticker, prefix, full_model.n_features_)

    try:
        regime_model = _gcs_load_pkl(gcs, f"{prefix}{ticker}_regime.pkl")
    except Exception as exc:
        log.warning("%s: regime.pkl unavailable in GCS (%s: %s) — using full model.",
                    ticker, type(exc).__name__, exc)
        return full_model, full_meta, None, []

    regime_metas = []
    for k in range(regime_model.n_regimes):
        m = _gcs_load_json(gcs, f"{prefix}{ticker}_regime{k}_meta.json")
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
    gcs: storage.Client,
    ticker: str,
    full_model,
    full_meta: dict,
    regime_model,
    regime_metas: list,
    cache_entry: dict,
    prefix: str = MODEL_PREFIX_DEFAULT,
) -> dict:
    """
    Compute (or retrieve from cache) regime, bias, direction, wf_sharpe for one asset.
    Returns a dict: {bias, direction, regime, wf_sharpe_mean}.
    """
    raw = yf.download(ticker, period="90d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty or "Close" not in raw:
        raise ValueError(f"{ticker}: yfinance returned no daily data")
    log.info("%s: daily data rows=%d latest=%s close_NaNs=%d",
             ticker, len(raw), raw.index[-1].date(), int(raw["Close"].isna().sum()))
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
                try:
                    r_model = _gcs_load_pkl(gcs, f"{prefix}{ticker}_regime{regime_k}_model.pkl")
                    model_used = r_model
                except Exception as exc:
                    log.warning("%s: regime-%d model unavailable in GCS (%s: %s) — using full model.",
                                ticker, regime_k, type(exc).__name__, exc)
                log.info("%s: regime %d of %d  wf_sharpe=%.4f",
                         ticker, regime_k, regime_model.n_regimes,
                         meta_used.get("wf_sharpe_mean", 0.0))
            else:
                log.warning("%s: classify_regime=%d out of range — falling back.", ticker, regime_k)
                regime_k = -1

    # Schema assertion: compare model's stored feature names against meta JSON.
    # Both must be populated by train_all.py >= 2026-10-24; older pickles that
    # lack feature_names_ are rejected so the mismatch is caught on deploy not at runtime.
    model_feat_names = getattr(model_used, "feature_names_", None)
    meta_feat_names  = meta_used.get("feature_names")
    if model_feat_names is None:
        raise ValueError(
            f"{ticker}: model pickle lacks feature_names_ — retrain with train_all.py "
            f">= 2026-10-24 and push to GCS."
        )
    if meta_feat_names is not None and list(model_feat_names) != list(meta_feat_names):
        raise ValueError(
            f"{ticker}: feature name mismatch between pickle "
            f"({model_feat_names}) and meta JSON ({meta_feat_names}). "
            f"Push matching artifacts to GCS."
        )
    n_feat = model_used.n_features_
    if n_feat not in (N_BASE_FEATURES, N_EXTENDED_FEATURES):
        raise ValueError(
            f"{ticker}: model has {n_feat} features — expected one of "
            f"{FEATURE_NAMES_BASE} or {FEATURE_NAMES_EXTENDED}."
        )

    bias      = float(model_used.predict(ret_lag.to_frame())[-1])
    direction = "long" if bias > MODEL_THRESH else ("short" if bias < -MODEL_THRESH else "flat")
    wf_sharpe = float(meta_used.get("wf_sharpe_mean", 0.0))

    log.info("%s: bias=%.4f  thresh=±%.2f  direction=%s  regime=%d  wf_sharpe=%.4f  model_features=%d",
             ticker, bias, MODEL_THRESH, direction, regime_k, wf_sharpe, model_used.n_features_)
    return {"bias": bias, "direction": direction, "regime": regime_k, "wf_sharpe_mean": wf_sharpe}


def _get_all_daily_biases(gcs: storage.Client, regime_stacks: dict,
                          prefix: str = MODEL_PREFIX_DEFAULT,
                          dry_run: bool = False) -> dict:
    """
    Load or recompute per-asset bias entries. Returns dict keyed by ticker.
    Cache format: {"date": "YYYY-MM-DD", "model_prefix": "models/...", "XLE": {...}, ...}
    Invalidated when date rolls over OR when model_prefix changes (e.g. after GCS rollback).
    dry_run=True: cache is read-only — computed biases are NOT written back to GCS.
    """
    today = _today_et()
    cache = _load_bias_cache(gcs)

    if (cache.get("date") == today
            and cache.get("model_prefix") == prefix
            and all(t in cache for t in ASSETS)):
        log.info("Bias cache hit for %s (prefix=%s).", today, prefix)
        return {t: cache[t] for t in ASSETS}

    log.info("Cache miss for %s — computing daily biases for all assets...", today)
    result = {}
    failed = []
    for ticker, (full_model, full_meta, regime_model, regime_metas) in regime_stacks.items():
        cache_entry = cache.get(ticker, {})
        try:
            result[ticker] = _get_daily_bias_for_asset(
                gcs, ticker, full_model, full_meta, regime_model, regime_metas,
                cache_entry, prefix=prefix,
            )
        except Exception as exc:
            # One asset's failure must not block the others; record why it is flat.
            log.exception("%s: daily bias failed — asset skipped this cycle: %s", ticker, exc)
            failed.append(ticker)
            result[ticker] = {"bias": 0.0, "direction": "flat", "regime": -1,
                              "wf_sharpe_mean": 0.0, "error": f"{type(exc).__name__}: {exc}"}

    if dry_run:
        log.info("DRY_RUN: bias cache not saved (read-only mode).")
    elif failed:
        log.warning("Bias cache NOT saved — failed assets %s will be retried next cycle.", failed)
    else:
        _save_bias_cache(gcs, {"date": today, "model_prefix": prefix, **result})
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

    if latest_rsi < RSI_OVERSOLD  and latest_hist > 0 and allowed_bias == "long":
        return "long",  latest_rsi, latest_hist
    if latest_rsi > RSI_OVERBOUGHT and latest_hist < 0 and allowed_bias == "short":
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
    except Exception as exc:
        # Alpaca answers 404 "position does not exist" when flat; anything else
        # means the position is unknown and is being treated as flat.
        if getattr(exc, "status_code", None) == 404 or "does not exist" in str(exc):
            log.info("%s: no open position.", ticker)
        else:
            log.error("%s: get_open_position failed (%s: %s) — treating as flat.",
                      ticker, type(exc).__name__, exc)
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


def _close_position(trade_client: TradingClient, ticker: str) -> bool:
    """Returns True on success. Caller must skip the opening order on False."""
    try:
        trade_client.close_position(ticker)
        log.info("Closed existing %s position.", ticker)
        return True
    except Exception as exc:
        log.error("close_position(%s) FAILED — skipping open order: %s: %s",
                  ticker, type(exc).__name__, exc)
        return False


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

    The buying_power clamp enforces the 80 % cap against Alpaca's live account
    state before every order, not just at signal-generation time.

    To swap to inverse-volatility weighting later, replace this function body
    while keeping the signature unchanged — order-execution code is untouched.
    """
    if num_positions <= 0 or price <= 0:
        return 0
    raw_notional  = max_exposure * equity / num_positions
    safe_notional = min(raw_notional, max(0.0, buying_power))
    return int(safe_notional / price)


# ── Decision summary ──────────────────────────────────────────────────────────

_ORDER_ACTIONS = {"buy", "sell"}


def _log_decision_summary(results: dict) -> None:
    """One line per cycle stating, for every asset, the order placed or the stage that filtered it."""
    parts = []
    for ticker, r in results.items():
        action = r.get("action", "?")
        if action in _ORDER_ACTIONS:
            parts.append(f"{ticker}={action.upper()} {r.get('qty')}")
        else:
            parts.append(f"{ticker}={action}({r.get('reason', 'no reason recorded')})")
    traded = any(r.get("action") in _ORDER_ACTIONS for r in results.values())
    log.info("%s: %s", "DECISION" if traded else "NO TRADE", "; ".join(parts))


# ── Entry point ───────────────────────────────────────────────────────────────

@functions_framework.http
def run_daily(request):
    """HTTP Cloud Function — called every 5 min by Cloud Scheduler.

    DRY_RUN mode (env var DRY_RUN=true):
      Runs the full pipeline — loads models from GCS, fetches intraday bars,
      computes the order plan — but does NOT acquire the lock and does NOT
      place any orders. Prints a human-readable plan to stdout.
      Useful for local pre-deploy verification.

    Local credentials:
      In production, ALPACA_API_KEY and ALPACA_SECRET_KEY come from Secret
      Manager. Locally, set them as env vars and they take precedence.
    """
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
    gcs = None
    try:
        if not dry_run and not _is_market_open():
            now = datetime.now(ET)
            msg = f"Outside market hours ({now.strftime('%a %H:%M ET')}) — no action."
            log.info(msg)
            return (json.dumps({"status": "closed", "message": msg}), 200,
                    {"Content-Type": "application/json"})

        # Prefer env-var credentials for local dry-runs; fall back to Secret Manager
        api_key    = (os.environ.get("ALPACA_API_KEY")
                      or _get_secret("ALPACA_API_KEY"))
        secret_key = (os.environ.get("ALPACA_SECRET_KEY")
                      or _get_secret("ALPACA_SECRET_KEY"))
        gcs        = storage.Client()

        # ── Distributed lock (skipped in dry-run) ─────────────────────────────
        if not dry_run:
            if not _acquire_lock(gcs, owner="cloud_function"):
                log.info("NO TRADE: stage=lock (trade lock held by another process).")
                return (json.dumps({"status": "locked", "message": "Trade lock held."}),
                        200, {"Content-Type": "application/json"})

        # ── Pending order guard ───────────────────────────────────────────────
        trade_client = TradingClient(api_key, secret_key, paper=True)
        if not dry_run:
            pending = _check_pending_orders(trade_client)
            if pending:
                log.warning("NO TRADE: stage=pending_orders. %s", pending)
                _release_lock(gcs)
                return (json.dumps({"status": "pending", "message": pending}), 200,
                        {"Content-Type": "application/json"})

        # ── Layer 1+2: regime + daily bias (cached per asset) ─────────────────
        log.info("Loading regime stacks for all assets...")
        model_prefix  = _get_model_prefix(gcs)
        regime_stacks = {t: _load_regime_stack(gcs, t, model_prefix) for t in ASSETS}
        biases = _get_all_daily_biases(gcs, regime_stacks, prefix=model_prefix,
                                       dry_run=dry_run)

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

            if b.get("error") or allowed == "flat" or kelly_f == 0.0:
                if b.get("error"):
                    reason = f"bias_error: {b['error']}"
                elif allowed == "flat":
                    reason = f"flat_bias |{bias:.4f}| <= {MODEL_THRESH}"
                else:
                    reason = f"no_edge wf_sharpe={wf_sharpe:.4f} <= 0"
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
                asset_result["reason"] = f"intraday_bars_error: {exc}"
                results[ticker] = asset_result
                continue

            if len(df) < 30:
                log.warning("%s: only %d bars.", ticker, len(df))
                asset_result["action"] = "insufficient_bars"
                asset_result["reason"] = f"only {len(df)} intraday bars (< 30)"
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
                asset_result["reason"] = (
                    f"intraday not confirmed for {allowed}: RSI={rsi:.1f} "
                    + (f"(need < {RSI_OVERSOLD}) MACDh={hist:.4f} (need > 0)" if allowed == "long"
                       else f"(need > {RSI_OVERBOUGHT}) MACDh={hist:.4f} (need < 0)")
                )
                results[ticker] = asset_result
                continue

            plan.append((ticker, direction, bias, wf_sharpe, kelly_f, latest_price,
                         current_dir, current_qty, asset_result))

        # ── Portfolio gross exposure cap ──────────────────────────────────────
        # Held:  direction == current_dir → position stays unchanged.
        # Opens: direction != current_dir (new entry or flip) → close first, then open.
        #
        # held_gross = sum(abs(market_value)) for HOLDS only (flips will be closed).
        # remaining_cap = gross_cap - held_gross.
        # per_new_position = remaining_cap / #opens.
        #
        # If held_gross > gross_cap (e.g. 2x book), emit reduce orders proportionally
        # before opening new ones.
        all_positions   = trade_client.get_all_positions()
        pos_by_symbol   = {p.symbol: p for p in all_positions}

        holds = [(t, d, b, w, k, p, cd, cq, ar) for t, d, b, w, k, p, cd, cq, ar in plan
                 if d == cd]
        opens = [(t, d, b, w, k, p, cd, cq, ar) for t, d, b, w, k, p, cd, cq, ar in plan
                 if d != cd]

        # abs(market_value) handles short positions (Alpaca returns negative for shorts)
        held_gross = sum(
            abs(float(pos_by_symbol[t].market_value))
            for t, *_ in holds
            if t in pos_by_symbol
        )
        gross_cap        = CFG["gross_cap_pct"] * equity
        remaining_cap    = max(0.0, gross_cap - held_gross)
        num_opens        = len(opens)
        per_new_position = remaining_cap / num_opens if num_opens else 0.0

        log.info("Gross cap: $%.2f  |  Held: $%.2f  |  Remaining: $%.2f  |  "
                 "Opens: %d  |  Per-new: $%.2f  |  Holds: %d",
                 gross_cap, held_gross, remaining_cap, num_opens, per_new_position, len(holds))

        # ── Reduce step: held_gross exceeds cap → scale held positions down ────
        if held_gross > gross_cap and holds:
            scale = gross_cap / held_gross
            log.warning("held_gross $%.2f > gross_cap $%.2f — scaling held positions to %.1f%%",
                        held_gross, gross_cap, scale * 100)
            for ticker, direction, _, _, _, latest_price, _, current_qty, asset_result in holds:
                target_notional = abs(float(pos_by_symbol[ticker].market_value)) * scale
                target_shares   = int(target_notional / latest_price) if latest_price > 0 else 0
                delta_shares    = current_qty - target_shares  # always > 0 (we're reducing)
                if delta_shares > 0:
                    reduce_side = OrderSide.SELL if direction == "long" else OrderSide.BUY
                    log.info("REDUCE %s: %s %d sh (from %d → %d target)%s",
                             ticker, reduce_side.value.upper(), delta_shares,
                             current_qty, target_shares, "  [DRY RUN]" if dry_run else "")
                    asset_result.update({"action": "reduce" + ("_dry" if dry_run else ""),
                                         "qty": delta_shares,
                                         "reason": f"held_gross {held_gross:.0f} > cap {gross_cap:.0f}"})
                    if not dry_run:
                        try:
                            order = trade_client.submit_order(MarketOrderRequest(
                                symbol=ticker, qty=delta_shares, side=reduce_side,
                                time_in_force=TimeInForce.DAY,
                            ))
                            asset_result["order_id"] = str(order.id)
                        except Exception as exc:
                            log.error("REDUCE %s failed: %s", ticker, exc)
                            asset_result["action"] = "reduce_failed"
                            asset_result["reason"] = str(exc)
                    results[ticker] = asset_result

        # Remove holds from plan — only process opens below
        plan = opens

        # ── Place opens/flips ─────────────────────────────────────────────────
        for ticker, direction, bias, wf_sharpe, kelly_f, latest_price, \
                current_dir, current_qty, asset_result in plan:  # plan is now opens only

            target_shares = int(per_new_position / latest_price) if latest_price > 0 else 0
            log.info("%s: allocated=$%.0f (%d sh)  kelly_f=%.4f",
                     ticker, per_new_position, target_shares, kelly_f)
            asset_result.update({"target_shares": target_shares,
                                 "notional_allocated": round(per_new_position, 2)})

            if current_dir != "flat":
                if dry_run:
                    log.info("%s: would close %s %d sh  [DRY RUN]", ticker, current_dir, current_qty)
                    current_qty = 0
                elif not _close_position(trade_client, ticker):
                    asset_result["action"] = "close_failed"
                    asset_result["reason"] = f"close of {current_dir} {current_qty} sh failed — open skipped"
                    results[ticker] = asset_result
                    continue
                else:
                    current_qty = 0

            current_signed = current_qty if current_dir == "long" else -current_qty
            target_signed  = target_shares if direction == "long" else -target_shares
            delta          = target_signed - current_signed

            if delta == 0 or target_shares < 1:
                log.info("%s: delta=0 or target<1 share — skipping.", ticker)
                asset_result["action"] = "skipped"
                asset_result["reason"] = (
                    f"size rounds to {target_shares} sh "
                    f"(allocated ${per_new_position:.0f} @ ${latest_price:.2f})"
                    if target_shares < 1 else "delta=0"
                )
            else:
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                log.info("Order%s: %s %d %s @ ~$%.2f  notional=$%.0f  wf=%.4f",
                         " [DRY RUN]" if dry_run else "",
                         side.value.upper(), abs(delta), ticker, latest_price,
                         abs(delta) * latest_price, wf_sharpe)
                asset_result.update({
                    "action": side.value + ("_dry" if dry_run else ""),
                    "qty": abs(delta),
                })
                if not dry_run:
                    order = trade_client.submit_order(MarketOrderRequest(
                        symbol=ticker, qty=abs(delta), side=side,
                        time_in_force=TimeInForce.DAY,
                    ))
                    asset_result["order_id"] = str(order.id)

            results[ticker] = asset_result

        _log_decision_summary(results)
        if not dry_run:
            _release_lock(gcs)

        payload = {"status": "dry_run" if dry_run else "ok",
                   "equity": equity, "assets": results}
        if dry_run:
            log.info("=== DRY RUN COMPLETE — no orders placed ===")
        return (json.dumps(payload), 200, {"Content-Type": "application/json"})

    except Exception as exc:
        log.exception("run_daily failed: %s", exc)
        if gcs and not (os.environ.get("DRY_RUN", "false").lower() == "true"):
            _release_lock(gcs)
        return (json.dumps({"error": str(exc)}), 500, {"Content-Type": "application/json"})


# ── Local dry-run entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Dry-run the cloud function locally against the paper account."
    )
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Simulate orders without placing them (default: True)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Actually place orders (dangerous — confirm before using)")
    parser.add_argument("--skip-hours-check", action="store_true",
                        help="Run outside market hours for testing")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    if args.skip_hours_check:
        # Monkey-patch market hours check so it always returns True
        _is_market_open_orig = _is_market_open
        def _is_market_open():  # noqa: F811
            return True

    class _MockRequest:
        method = "POST"
        args = {}
        get_json = lambda s, **kw: None

    body, status, headers = run_daily(_MockRequest())
    import json as _json
    data = _json.loads(body)

    print(f"\nHTTP {status}")
    print(_json.dumps(data, indent=2))
