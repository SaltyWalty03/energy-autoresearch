# Phase 0 Audit Report
*Generated 2026-10-24*

---

## 1. Module Map and Data Flow

### Files and Their Roles

| Path | Purpose |
|------|---------|
| `model.py` (root) | Training definition + `build_feature_matrix`. Source of truth for `DirectionModel`. |
| `cloud_function/model.py` | Byte-identical copy deployed to Cloud Run. **Drifts independently.** |
| `deploy/model.py` | Third copy in `deploy/`. Same content, independent from both. |
| `src/regime.py` | `fit_regimes()` + `classify_regime()` for KNN-based regime inference. |
| `cloud_function/regime.py` | Byte-identical copy used by the deployed function. |
| `cloud_function/src/regime.py` | Fourth copy, also byte-identical. |
| `deploy/src/` | Fifth copy of src/. |
| `src/universe.py` | Defines `ASSETS` dict `{XLE, XLB, XOP}` with commodity tickers and shock thresholds. |
| `cloud_function/main.py` | **The deployed Cloud Function** (intraday 3-layer: regime → daily bias → RSI+MACD). |
| `run_daily.py` (root) | Local runner — daily-only (no intraday layer). Regime + bias only. |
| `cloud_function/run_daily.py` | An intermediate daily-only version, also in the cloud_function tree. |
| `models/` | Local pkl/json artifacts. Must be manually `gsutil cp`-ed to GCS after retrain. |
| `deploy/models/` | Another copy of model artifacts. |
| `src/trade_logger.py` | Appends rows to `trades_log.csv`. Local only (not in the deployed function). |

### Data Flow (Deployed Path)

```
Cloud Scheduler (every 5 min, Mon-Fri 13-20 UTC)
  → cloud_function/main.py::run_daily()
      ├─ _is_market_open() → exits early outside 9:30–15:55 ET
      ├─ Secrets Manager → ALPACA_API_KEY, ALPACA_SECRET_KEY
      ├─ GCS lock (trade_lock.json, TTL 90s)
      ├─ _load_regime_stack() ← GCS: models/{ticker}_model.pkl,
      │                                    _regime.pkl, _regime{k}_meta.json
      ├─ _get_all_daily_biases() ← GCS: bias_cache.json (invalidated daily by ET date)
      │     └─ yfinance(ticker, period=90d) → pct_change().shift(1)
      │          → build_feature_matrix() [6-feature rolling momentum/vol]
      │          → classify_regime() [KNN on 6-feature space]
      │          → DirectionModel.predict() [9-feature: base6 + WTI_ret + temp×ret + ng×ret]
      │          → signal ∈ (-1, 1), threshold ±0.02 → direction
      ├─ Alpaca: get_account().equity, buying_power
      ├─ _fetch_intraday_bars() ← Alpaca 5-min bars, last 60 bars
      │     → RSI(14) + MACD histogram
      │     → intraday signal must CONFIRM daily bias
      └─ _compute_position_size() → submit_order()
```

### Shared Coupling Points

Every location where a name, schema, or value must match across files:

| Coupling | Files That Must Agree |
|----------|----------------------|
| Feature count (6 vs 9) | `model.py` (all 3 copies), pickles in GCS |
| `ASSETS` dict | `src/universe.py`, `cloud_function/main.py` (inline copy at line 118) |
| `MAX_PORTFOLIO_EXPOSURE_PCT = 0.80` | `cloud_function/main.py`, `run_daily.py`, `cloud_function/run_daily.py` |
| `MODEL_THRESH = 0.02` | Same three files |
| Model filenames `{ticker}_model.pkl`, `{ticker}_regime.pkl`, etc. | `_load_regime_stack()` in all three entry points + GCS layout |
| `RegimeModel.n_regimes` | Pickle in GCS must match regime count expected by the loop |
| `build_feature_matrix()` output | Used by regime layer AND `_today_feature_vector()` — must produce 6 features |

**Critical observation:** `ASSETS` is defined in `src/universe.py` for local paths, but `cloud_function/main.py` has its own inline copy (line 118–122). They are currently identical; a future edit to one will not propagate to the other.

---

## 2. Why Gross Exposure is ~2x Equity

**The bug is not in the formula — it's in when positions are held vs. resized.**

The sizing formula:
```python
per_position_notional = 0.80 * equity / num_positions
safe_notional = min(raw_notional, buying_power)
shares = int(safe_notional / price)
```

`num_positions = len(plan)` counts only assets generating **new** signals on the current invocation. Assets in "hold" mode (current direction matches signal) are not in `plan` and are not counted. Their existing notional is never rebalanced.

### Scenario That Produced the Current Book

| Day | Event | Result |
|-----|-------|--------|
| Day A | XOP fires alone | `num_positions=1`, notional = `0.80 × $100k / 1 = $80k`. XOP long opened at ~$80k. |
| Day B | XLE fires (short), XOP is "hold" | `num_positions=1`, notional = `0.80 × $98k / 1 = $78k`. XLE short opened at ~$80k. Gross ~$160k = 1.6× equity. |
| Day C | XLB fires (short), XOP/XLE are "hold" | `num_positions=1`, notional = ~$40k. XLB short opened. Gross ~$200k = 2× equity. |

This matches the observed book exactly: XOP ~$80k, XLE ~$80k, XLB ~$40k.

### Why the Buying-Power Clamp Does Not Help

The intent of `min(raw_notional, buying_power)` is to limit sizing to available capital. However, in Alpaca's paper trading environment, `buying_power` is typically 2–4× equity for a margin account (~$200–400k against $98k equity). The clamp almost never fires. For short positions specifically, Alpaca paper trading does not correctly reduce `buying_power` by the short notional the way a real broker would.

### Does the 80% Cap Apply to Net or Gross?

It is applied per new position as a fraction of equity, **not** as a portfolio gross-exposure cap. There is no check of total gross notional anywhere in the code. Existing positions accumulate unbounded.

---

## 3. "Stops Trading After an Update" — Root-Cause Diagnosis

### Finding 1 — Feature Schema Mismatch (Highest Probability Root Cause)

`model.py` documents two schemas:
- `N_BASE_FEATURES = 6` (pre-2026-09-14 pickles)
- `N_EXTENDED_FEATURES = 9` (current)

When a model is retrained locally and NOT pushed to GCS, the cloud function keeps loading the old pickle. When `model.py` is updated in the cloud but the pickle is not, `predict()` raises `ValueError` at lines 317–320:

```python
raise ValueError(
    f"DirectionModel fitted on {n_expected} features; predict() can only build "
    f"{N_BASE_FEATURES} (legacy) or {N_EXTENDED_FEATURES} (extended). Retrain the model."
)
```

This propagates to the outer `except` in `run_daily()` (line 666), which returns HTTP 500. Cloud Scheduler logs a failure but sends no alert. All trading stops silently.

### Finding 2 — `_close_position` Swallows Errors Without Logging (Local `run_daily.py`)

```python
def _close_position(client, ticker):
    try:
        client.close_position(ticker)
        print(f"  Closed existing {ticker} position.")
    except Exception:
        pass  # ← silent, no log
```

If a close fails (PDT restriction, already-closed position, etc.), the code proceeds to `_place_order`. That order may then be rejected by Alpaca (e.g., for trying to sell a position that was not actually closed), and the rejection is not logged — it propagates as an unhandled exception.

### Finding 3 — `_place_order` Has No Exception Handling (Local `run_daily.py`, Line 269)

```python
def _place_order(client, ticker, shares, signal, price):
    ...
    client.submit_order(order)  # ← raises on rejection, no try/except
    print(f"  [{side.value.upper()}] ...")
```

In the cloud function (`main.py`), `submit_order` is inside a try/except for the whole asset loop, so a rejection on one asset is logged. In the local `run_daily.py`, the exception propagates to the `finally`, releases the lock, and exits — the rejection reason is never printed.

### Finding 4 — GCS Model Sync Is Manual

There is no automated step to push `models/` to GCS after retraining. The docs say:
> `gsutil cp -r models/ gs://energy-autoresearch-model/models/`

If someone runs `train_all.py` locally but forgets this command, the cloud function continues using stale pickles indefinitely. The only guard is the `n_features_` check, which only fires if the feature *count* changed — not if model quality deteriorated or the regime structure changed.

### Finding 5 — Regime Count Mismatch After Retrain

Current artifact counts: XLE has 13 regimes (0–12), XOP has 10 (0–9), XLB has 2 (0–1). After retraining, HDBSCAN may produce a different number of regimes. `_load_regime_stack` loops `for k in range(regime_model.n_regimes)` using the new pickle's count, but if GCS still has old `_regime{k}_model.pkl` files from the prior training run, missing files fall back to the full model silently. No error is surfaced to signal the mismatch.

### Finding 6 — Code Paths Where Order Failure Is Not Logged

| Location | Status |
|----------|--------|
| `cloud_function/main.py` `submit_order` | **Logged.** Exception caught by outer `except Exception` at line 666, which calls `log.exception`. |
| Local `run_daily.py` `_close_position` | **Not logged.** `except Exception: pass` swallows the failure completely. |
| Local `run_daily.py` `_place_order` | **Not logged.** No exception handling; failure exits silently with no `log_trade()` call. |
| Local `run_daily.py` outer exception path | No `log_trade()` is called on crash, so the trades log shows no evidence of the failure. |

---

## 4. Correlation and Hedging Analysis

### Asset Correlations (Qualitative)

- **XLE vs. XOP:** Historically 0.85–0.95. Both are U.S. oil & gas ETFs. XOP is more E&P-concentrated with higher WTI beta; XLE has a larger integrated-major component (Exxon, Chevron). They track each other very closely day-to-day.
- **XLE vs. XLB:** Moderate, ~0.50–0.65. XLB (materials) is correlated to energy through chemicals and metals, but diverges significantly.
- **XOP vs. XLB:** ~0.45–0.60. Lower correlation; XLB is driven by industrial metals and chemicals rather than oil.

### Current Book Implication

| Position | Direction | Notional |
|----------|-----------|----------|
| XOP | Long | $79.9k |
| XLE | Short | $80.1k |
| XLB | Short | $40.3k |

The long XOP / short XLE pair is near-perfectly offsetting: approximately 90–95% of the day-to-day P/L on each leg cancels the other. What remains is XOP idiosyncratic alpha (E&P outperformance vs. integrated majors) and basis risk on the spread.

**Net energy exposure:** roughly long `$79.9k − $80.1k × 0.90 ≈ +$8k` (XOP outperformance vs. XLE), plus a directional short on materials of $40.3k.

The portfolio is effectively a **long XOP/XLE spread** plus a **short materials bet**. The XOP/XLE pair hedges away most of the energy-sector P/L. If the intent was a directional energy bet, the structure is inadvertently self-cancelling while paying double transaction costs and short borrow fees on both legs.

---

## 5. Summary of Critical Issues

| # | Severity | Issue |
|---|----------|-------|
| 1 | **Critical** | Three independent copies of `model.py` and five copies of `regime.py` across root, `cloud_function/`, and `deploy/`. Any edit drifts silently. |
| 2 | **Critical** | Gross exposure 2× equity because `num_positions` counts only new signals; existing "hold" positions never rebalance. No portfolio-level gross-exposure cap exists. |
| 3 | **Critical** | Feature schema mismatch (6 vs. 9 features) between pickle and code after updates is the primary "stops trading" root cause — returns HTTP 500 silently. |
| 4 | **High** | Local `_close_position` swallows exceptions with no logging. Local `_place_order` has no exception handling. Order rejections leave no trace. |
| 5 | **High** | GCS model sync is manual. A local retrain that is not pushed leaves the cloud function trading a stale model indefinitely. |
| 6 | **High** | The 80% parameter is a per-new-position fraction of equity, not a portfolio gross-exposure cap. Positions compound across days without a ceiling. |
| 7 | **Medium** | `ASSETS` is defined in both `src/universe.py` and inline in `cloud_function/main.py` (line 118). Edits to one do not propagate. |
| 8 | **Medium** | Long XOP + short XLE largely cancel — the system pays double fees and borrow costs for near-zero net energy exposure on that pair. |
| 9 | **Info** | `env.yaml` and `.env` contain plaintext API keys. Both files should be confirmed absent from git history. |

---

## 6. Open Questions for Phase 1+

1. Should the gross-exposure bug (Issue 2) be fixed urgently before any model work, or addressed as part of Phase 2's risk-config refactor?
2. Should live Alpaca account equity be pulled via the API to confirm the $98k figure and verify whether the paper account has margin enabled (which drives the buying-power clamp behavior)?
3. Are the three copies of `model.py` intentionally maintained separately, or should they be replaced with a single shared module?
