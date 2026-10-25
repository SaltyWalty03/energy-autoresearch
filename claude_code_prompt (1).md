# Claude Code Prompt: Diagnose & Fix Energy Autoresearch Trading System

## Context

This is a multi-asset paper trading system (`energy-autoresearch/`) that trades XLE, XLB, and
XOP on Alpaca paper trading. It runs via two paths:
- **`run_daily.py`** — runs locally (or as a cron job), pure Python
- **`cloud_function/main.py`** — runs on Google Cloud Functions, triggered every 5 min by Cloud
  Scheduler during market hours

The most recent commit (May 18) rewrote both files to a three-layer regime-aware architecture.
Since then, **no trades are executing on Alpaca**. Your job is to find and fix all bugs, then
loosen parameters so trades actually fire on a paper account.

---

## Known Architecture

**Three-layer signal (cloud function):**
1. **Regime**: `classify_regime(feat_vec, regime_model)` → regime index k
2. **Daily bias**: regime-k `DirectionModel` → `bias` float; cached in GCS `bias_cache.json`
3. **Intraday timing**: RSI(14) + MACD histogram on 5-min Alpaca bars — entry fires only when
   intraday indicators confirm the daily bias direction

**Kelly sizing**: `kelly_f = max(0, (wf_sharpe_mean / sqrt(252)) * 0.5)`

**Assets**: XLE, XLB, XOP (defined in `src/universe.py` and hardcoded in `cloud_function/main.py`)

---

## Bug Investigation Tasks

### 1. Diagnose why no trades are executing

Work through each layer of the signal pipeline and identify what is blocking trades. The likely
culprits are:

**a) Stale or wrong GCS `bias_cache.json`**
- The cloud function caches `bias_cache.json` keyed by date. If it was written with a wrong date
  or bad bias values (all `"flat"`), every 5-min invocation returns early without placing orders.
- Check: `gsutil cat gs://energy-autoresearch-model/bias_cache.json`
- Fix: delete the stale cache so it recomputes: `gsutil rm gs://energy-autoresearch-model/bias_cache.json`

**b) Kelly fraction always 0 (no edge detected)**
- `_kelly_fraction` returns 0 when `wf_sharpe_mean <= 0`, which skips the asset entirely.
- In `cloud_function/main.py` → `_get_daily_bias_for_asset`, the code uses `meta_used` for
  WF Sharpe but `model_used = full_model` (not the regime-specific model). The regime-specific
  `wf_sharpe_mean` is used for Kelly, but the **full model's prediction** is used for bias
  direction. This is intentional per the comment, but verify `meta_used` is correctly set to
  `regime_metas[regime_k]` and not defaulting to `full_meta` (which has lower Sharpe).
- Check that `regime_metas[k]` is being populated from GCS: the cloud function calls
  `_gcs_load_json(gcs, f"models/{ticker}_regime{k}_meta.json")`. Confirm these objects exist
  in GCS with: `gsutil ls gs://energy-autoresearch-model/models/ | grep meta`

**c) Intraday RSI/MACD conditions too tight**
- Long entry requires: RSI < 35 AND MACD histogram flips from negative to positive (prev < 0,
  current ≥ 0). Short entry requires: RSI > 65 AND MACD histogram flips from positive to
  negative. These simultaneous conditions rarely align on 5-min bars. This is a primary reason
  for no trades even when bias is non-flat.

**d) `run_daily.py` — MODEL_THRESH too tight**
- `MODEL_THRESH = 0.05`. If `abs(signal) < 0.05` the asset is skipped. Check what signal values
  are actually being generated for XLE, XOP, XLB today. Print them during a dry run.

**e) GCS lock stuck**
- If a previous run crashed before releasing the lock, `trade_lock.json` may still exist.
- Check: `gsutil cat gs://energy-autoresearch-model/trade_lock.json`
- Fix if stale: `gsutil rm gs://energy-autoresearch-model/trade_lock.json`

**f) Cloud function not deploying latest code**
- The `logging` import in the old version was replaced with `_StructuredLogger`, but the
  `logging.basicConfig` call was left in temporarily during the diff. Verify the deployed
  function matches `cloud_function/main.py` exactly.
- Also verify that `cloud_function/regime.py` and `cloud_function/model.py` are present and
  match `src/regime.py` and `model.py` respectively — the function imports from them directly.

**g) XOP and XLB have no regime-specific model prediction in the cloud function**
- In `_get_daily_bias_for_asset`, `model_used = full_model` always — it never loads the
  regime-specific model for prediction, only for WF Sharpe. For XLE this was intentional,
  but it means XOP and XLB also use the full model bias even when a better regime model
  exists. This may or may not cause wrong signals. See Task 3 below for the fix.

---

### 2. Add dry-run diagnostic output

Add a `--dry-run` flag to `run_daily.py` that prints the full signal pipeline per asset without
placing orders. Output should include:

```
[XLE] regime=5 (of 13)  bias=0.0823  direction=long  wf_sharpe=3.726  kelly_f=0.1172
[XOP] regime=1 (of 10)  bias=0.1142  direction=long  wf_sharpe=2.729  kelly_f=0.0860
[XLB] regime=0 (of 2)   bias=0.0291  direction=flat  wf_sharpe=0.284  kelly_f=0.0089
```

Run it and paste the output here before making any changes to trading parameters.

---

### 3. Add XOP and XLB regime-specific model prediction

In `cloud_function/main.py` → `_get_daily_bias_for_asset`, the function always uses
`full_model` for prediction (`model_used = full_model`). XLE was designed this way as a
deliberate fallback, but XOP and XLB have regime-specific models in GCS and should use them.

**Fix**: after identifying `regime_k`, load the regime-specific model for prediction if
it exists, exactly as `run_daily.py` does:

```python
# After: regime_k = classify_regime(feat_vec, regime_model)
if 0 <= regime_k < len(regime_metas):
    try:
        r_model = _gcs_load_pkl(gcs, f"models/{ticker}_regime{regime_k}_model.pkl")
        model_used = r_model
    except Exception:
        log.warning("%s: regime-%d model missing in GCS — using full model.", ticker, regime_k)
    meta_used = regime_metas[regime_k] if regime_metas[regime_k] else full_meta
```

Apply this same pattern in `run_daily.py` if it also skips regime-specific prediction for XOP/XLB
(check whether `_get_signal_and_meta` already handles it — it does, but verify the cloud function
path mirrors it).

---

### 4. Loosen parameters for paper trading

After diagnosing the no-trade root cause, apply the following parameter relaxations so trades
fire regularly. These are intentionally loose since this is a paper account for live-data testing:

**In `cloud_function/main.py`:**
```python
MODEL_THRESH  = 0.02   # was 0.05  — smaller flat zone
RSI_OVERSOLD  = 45     # was 35    — RSI below 45 qualifies for long entry
RSI_OVERBOUGHT = 55    # was 65    — RSI above 55 qualifies for short entry
```

With `RSI_OVERSOLD = 45` and `RSI_OVERBOUGHT = 55`, the RSI gate is nearly always passable
during normal trading. Combined with `MODEL_THRESH = 0.02`, any non-trivial bias will generate
a signal.

**Remove the strict MACD histogram flip requirement** in `_intraday_signal` — change it to
simply require the histogram to be on the correct side (positive for long, negative for short),
not require a zero-crossing:

```python
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
```

Apply the same looser threshold in `run_daily.py`:
```python
MODEL_THRESH = 0.02   # was 0.05
```

---

### 5. Verify GCS model objects are current

After retraining with `train_all.py`, models must be re-uploaded. Verify that the GCS bucket
contains the latest models matching the local `models/` directory:

```bash
gsutil ls gs://energy-autoresearch-model/models/ | wc -l
ls models/ | wc -l
```

They should match. If GCS is behind, upload:
```bash
gsutil -m cp -r models/ gs://energy-autoresearch-model/
```

Also check that `cloud_function/src/regime.py` (used locally) and `cloud_function/regime.py`
(deployed to GCF) are in sync:
```bash
diff cloud_function/src/regime.py cloud_function/regime.py
```

If they differ, copy: `cp cloud_function/src/regime.py cloud_function/regime.py`

---

### 6. Redeploy the cloud function

After applying all fixes:

```bash
gcloud functions deploy run_daily \
    --runtime python311 \
    --region us-central1 \
    --trigger-http \
    --allow-unauthenticated \
    --source cloud_function/ \
    --memory 1024MB \
    --timeout 300s \
    --set-env-vars GCP_PROJECT=energy-autoresearch
```

Then manually trigger it and check logs:
```bash
gcloud functions call run_daily --region us-central1
gcloud functions logs read run_daily --region us-central1 --limit 50
```

---

### 7. Verify Alpaca paper account connectivity

Run `test_alpaca.py` to confirm credentials and API access are working:
```bash
python test_alpaca.py
```

Also confirm the account is not flagged for pattern day trading (PDT) restrictions — paper
accounts can sometimes block short selling.

---

## Success Criteria

1. `python run_daily.py --dry-run` prints non-zero signals for at least one asset.
2. `python run_daily.py` places at least one order on Alpaca paper account.
3. Cloud function logs show `action: buy` or `action: sell` for at least one asset during
   market hours.
4. `bias_cache.json` in GCS reflects today's date and non-flat directions for ≥1 asset.
5. XOP and XLB use regime-specific models for prediction (not just for WF Sharpe Kelly sizing).
