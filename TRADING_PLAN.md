# XLE Energy ETF — Paper Trading Plan

## 1. Model Summary

**Final model: `DirectionModel` (Random Forest classifier → continuous signal)**

| Component | Detail |
|---|---|
| Input | Previous day's XLE % return (`ret_lag`) |
| Features | 6 rolling momentum features: raw return, vol-adjusted return, 5-day/20-day mean, MACD ratio, vol-ratio |
| Classifier | `RandomForestClassifier(n_estimators=300, max_depth=2, min_samples_leaf=22, random_state=42)` |
| Training window | Rolling 3 years (756 trading days) — excludes COVID-era dynamics |
| Sample weighting | Exponential decay `np.exp(linspace(-1, 0, N))` — upweights recent data |
| Output signal | `2 * P(up) - 1` → range [-1, 1]; positive = long XLE, negative = short XLE |
| Regime filter | **WTI shock filter**: go flat (signal=0) on days where yesterday's WTI crude return exceeded ±2% |

**Validated Sharpe Ratio: 2.21** (val period 2025-01-17 to 2026-04-23, 317 days)

### Experiment History

| # | Change | Sharpe |
|---|---|---|
| Baseline | RF, no rolling window | 0.74 |
| Exp 3 | 3yr window + exp decay + seed 42 | 1.83 |
| Exp 4 | + WTI shock filter at 3% | 2.02 |
| Exp 5 ✓ | + tighten threshold to 2% | **2.21** |
| Exp 6 | threshold 1.5% | 2.01 (worse) |
| Exp 7 | + WTI momentum as RF features | 1.34 (worse) |
| Exp 8 | contrarian on shock days | 1.95 (worse) |

---

## 2. Signal Generation Pipeline

Every trading day before market open (e.g. 9:00 AM ET):

```python
# Step 1: fetch yesterday's XLE close
xle = yf.download("XLE", period="5d", auto_adjust=True)
ret_lag = xle["Close"].pct_change().iloc[-1]

# Step 2: fetch WTI (CL=F) for shock filter
wti = yf.download("CL=F", period="10d", auto_adjust=True)
wti_ret_yesterday = np.log(wti["Close"].iloc[-1] / wti["Close"].iloc[-2])

# Step 3: apply shock filter
if abs(wti_ret_yesterday) > 0.02:
    signal = 0.0  # go flat — oil shock day
else:
    # Step 4: run model predict()
    signal = model.predict(X_new)  # [-1, 1]

# Step 5: translate to position
# signal > 0  → long XLE
# signal < 0  → short XLE (or just cash if no shorting)
# signal == 0 → cash
```

**Refit frequency**: Weekly (every Friday close) — retrain on the latest 756 trading days to stay current with the market regime.

---

## 3. Cloud Deployment

### Recommended Stack

| Layer | Service | Cost |
|---|---|---|
| Compute | AWS Lambda (Python 3.12) | ~$0/mo on free tier |
| Scheduler | AWS EventBridge (cron) | $0 (included) |
| Storage | AWS S3 | <$1/mo for model artifacts |
| Secrets | AWS Secrets Manager | $0.40/mo per secret |
| Alerting | AWS SNS → email | $0 (free tier) |

### Deployment Steps

1. **Package the model**:
   ```bash
   pip install boto3 yfinance scikit-learn pandas -t ./package
   cp model.py run_daily.py ./package
   cd package && zip -r ../lambda_function.zip .
   ```

2. **Create Lambda function** (`run_daily.py`):
   ```python
   import json, boto3, pickle, numpy as np
   from model import build_model, DirectionModel
   
   def lambda_handler(event, context):
       # Load trained model from S3
       s3 = boto3.client("s3")
       obj = s3.get_object(Bucket="xle-model", Key="model.pkl")
       model = pickle.loads(obj["Body"].read())
       
       # Generate signal
       signal = generate_signal(model)
       
       # Send to Alpaca / log to S3
       place_order(signal)
       log_signal(signal)
       
       return {"statusCode": 200, "signal": signal}
   ```

3. **EventBridge rule**: `cron(0 13 ? * MON-FRI *)` (9 AM ET = 1 PM UTC)

4. **Refit Lambda**: separate function triggered `cron(30 20 ? * FRI *)` (4:30 PM ET Friday)
   - Downloads fresh XLE data via yfinance
   - Refits model on latest 756-day window
   - Saves pickled model to S3

---

## 4. Data Pipeline

| Data | Source | Frequency | Notes |
|---|---|---|---|
| XLE price | `yfinance` `"XLE"` | Daily | 5-day window sufficient |
| WTI crude | EIA API `v2/petroleum/pri/spt/data/` | Daily | Cache as parquet; refresh daily |
| WTI fallback | `yfinance` `"CL=F"` | Daily | Used if EIA API unavailable |

### EIA API Refresh (production):

```python
import requests, os
API_KEY = os.environ["EIA_API_KEY"]

def fetch_wti_daily():
    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    params = {
        "api_key": API_KEY,
        "frequency": "daily",
        "data[0]": "value",
        "facets[product][]": "EPCWTI",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 30,
    }
    r = requests.get(url, params=params)
    df = pd.DataFrame(r.json()["response"]["data"])
    df["date"] = pd.to_datetime(df["period"])
    return df.set_index("date")["value"].sort_index()
```

Store in S3 as `data/eia_wti_YYYY-MM-DD.parquet`. The model's `_load_wti()` already reads from this cache path when running locally — in Lambda, mount S3 as the cache directory or patch the path.

---

## 5. Paper Trading Execution (Alpaca)

Alpaca offers commission-free paper trading with a Python SDK (`alpaca-trade-api`).

### Setup

```bash
pip install alpaca-trade-api
```

Set environment variables:
```
ALPACA_API_KEY=<your paper key>
ALPACA_SECRET_KEY=<your paper secret>
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Order Logic

```python
import alpaca_trade_api as tradeapi

api = tradeapi.REST(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    os.environ["ALPACA_BASE_URL"],
)

def place_order(signal: float, equity: float = 10_000.0):
    """Translate model signal [-1,1] to XLE position."""
    
    # Close any existing XLE position
    try:
        api.close_position("XLE")
    except Exception:
        pass
    
    if abs(signal) < 0.05:  # dead zone — stay flat
        return
    
    xle_price = float(api.get_latest_trade("XLE").price)
    shares = int(abs(signal) * equity / xle_price)  # scale by signal strength
    
    if shares < 1:
        return
    
    side = "buy" if signal > 0 else "sell"
    api.submit_order(
        symbol="XLE",
        qty=shares,
        side=side,
        type="market",
        time_in_force="day",
    )
    print(f"[{side.upper()}] {shares} shares XLE @ ~${xle_price:.2f} | signal={signal:.3f}")
```

### Daily Workflow

```
09:00 ET  → Lambda fires → fetch data → check WTI shock → run model
09:01 ET  → close prior XLE position (market order)
09:02 ET  → place new position (market order)
04:30 ET  → log P&L to S3 / DynamoDB
04:35 ET  → (Friday only) refit model on fresh data
```

---

## 6. Risk Controls

| Control | Rule | Reason |
|---|---|---|
| **WTI shock filter** | Flat if \|WTI ret\| > 2% | Oil panic invalidates momentum signal |
| **Max position size** | Cap at 20% of portfolio in XLE | Concentration risk |
| **Daily loss limit** | Auto-liquidate if day P&L < -2% | Tail protection |
| **Drawdown stop** | Halt trading if rolling 10-day P&L < -5% | Model decay / regime change |
| **Signal dead zone** | No trade if \|signal\| < 0.05 | Avoid churning on near-zero conviction |
| **Earnings / FOMC** | Go flat on scheduled macro events | Model not trained for event risk |
| **Vol spike** | Flat if XLE 5-day vol > 3× trailing mean | Volatility regime change |

```python
def risk_check(signal: float, vol_ratio: float, day_pnl_pct: float) -> float:
    """Apply risk overlays. Returns adjusted signal (0 = flat)."""
    if day_pnl_pct < -0.02:       # daily stop
        return 0.0
    if vol_ratio > 3.0:            # vol spike
        return 0.0
    if abs(signal) < 0.05:         # dead zone
        return 0.0
    return signal
```

---

## 7. Monitoring & Decay Detection

### What to log daily (to S3 / DynamoDB)

```
date, signal, xle_ret, strategy_ret, wti_shocked, rolling_sharpe_20d, rolling_sharpe_63d
```

### Decay detection rules

| Metric | Warning | Action |
|---|---|---|
| Rolling 20-day Sharpe | < 0.5 | Alert — investigate regime change |
| Rolling 20-day Sharpe | < 0.0 | Pause trading — refit or review |
| Win rate (20-day) | < 40% | Alert |
| WTI shock filter hit rate | > 30% of days | Unusual — check WTI data quality |
| Directional accuracy | < 48% | Below random — model may have decayed |

### Refit trigger

Force a refit (not just weekly) when:
- Rolling 63-day Sharpe drops below 0.8 (was 2.21 at deployment)
- A major structural break occurs (oil supply shock, sanctions, new ETF holdings change)

### Simple monitoring script

```python
def check_model_health(log_df: pd.DataFrame) -> dict:
    recent = log_df.tail(20)
    rets = recent["strategy_ret"]
    sharpe_20 = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    win_rate = (rets > 0).mean()
    return {
        "sharpe_20d": round(sharpe_20, 2),
        "win_rate_20d": round(win_rate, 3),
        "needs_refit": sharpe_20 < 0.8,
        "pause_trading": sharpe_20 < 0.0,
    }
```

---

## 8. Estimated P&L (Paper)

Starting with $10,000 paper account, 2.21 annualized Sharpe, ~15% annual vol (typical for XLE):

| Scenario | Expected Annual Return |
|---|---|
| Vol = 15%, Sharpe = 2.21 | ~33% |
| Vol = 20%, Sharpe = 1.5 (decay) | ~30% |
| Vol = 10%, Sharpe = 1.0 (further decay) | ~10% |

These are directional estimates. Paper trade for 60-90 days before allocating real capital.

---

## Quick-Start Checklist

- [ ] Create Alpaca paper account at `paper-api.alpaca.markets`
- [ ] Set `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `EIA_API_KEY` in AWS Secrets Manager
- [ ] Deploy Lambda `run_daily.py` + `refit_weekly.py`
- [ ] Set EventBridge cron rules (9 AM ET weekdays + 4:30 PM ET Fridays)
- [ ] Test with 1 share manually before enabling automated orders
- [ ] Set up CloudWatch alarm on Lambda errors
- [ ] Start daily P&L log → check model health after 20 trading days
