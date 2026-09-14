"""
fetch_trade_history.py — Pull Alpaca FILL activities and join to local signal log.

Writes data/alpaca_trade_history.csv with columns:
  timestamp, symbol, side, qty, expected_price, fill_price, commission

expected_price comes from trades_log.csv (signal_price column, or fill_price for older rows
that used the column before it was renamed).  Rows where no signal log entry exists within
WINDOW_HOURS before the fill get expected_price=NaN.

Run:
    python fetch_trade_history.py
"""
import os
from datetime import timedelta
from pathlib import Path

# Load .env manually to avoid python-dotenv dependency
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import pandas as pd
from alpaca.trading.client import TradingClient

HERE       = Path(__file__).parent
TRADES_LOG = HERE / "trades_log.csv"
OUT_PATH   = HERE / "data" / "alpaca_trade_history.csv"
WINDOW_HOURS = 36  # signal must fire within this many hours before the fill

api_key    = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET_KEY"]

# ── 1. Pull FILL activities from Alpaca paper account ─────────────────────────
# alpaca-py 0.43 doesn't expose get_account_activities on TradingClient;
# drive the underlying REST client directly.  The base URL is already /v2/,
# so path must start without /v2/.
client = TradingClient(api_key, secret_key, paper=True)

print("Fetching account activities (activity_type=FILL)...")
raw: list[dict] = []
page = client.get("/account/activities", {"activity_type": "FILL", "page_size": "100"})
raw.extend(page if isinstance(page, list) else [])
while len(raw) % 100 == 0 and raw:
    last_id = raw[-1].get("id", "")
    if not last_id:
        break
    nxt = client.get("/account/activities", {
        "activity_type": "FILL", "page_size": "100", "page_token": last_id,
    })
    if not nxt:
        break
    raw.extend(nxt if isinstance(nxt, list) else [])

# The endpoint occasionally returns mixed types (e.g. FEE records sneak in).
# Keep only genuine trade fills (activity_type == "FILL").
fills_raw = [r for r in raw if r.get("activity_type") == "FILL"]
print(f"  Records returned: {len(raw)}  |  actual FILLs: {len(fills_raw)}")

fills_df = pd.DataFrame([
    {
        "transaction_time": r.get("transaction_time") or r.get("date"),
        "symbol":           r.get("symbol", ""),
        "side":             str(r.get("side", "")).lower(),
        "qty":              float(r.get("qty", 0)),
        "fill_price":       float(r.get("price", 0)),
        "commission":       0.0,   # paper trading; no commission field on TradeActivity
    }
    for r in fills_raw
])

if fills_df.empty:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fills_df.assign(expected_price=None)[
        ["timestamp","symbol","side","qty","expected_price","fill_price","commission"]
    ].to_csv(OUT_PATH, index=False)
    print("No fills found — wrote empty file.")
    raise SystemExit(0)

fills_df["transaction_time"] = pd.to_datetime(
    fills_df["transaction_time"], format="mixed", utc=True
)

# ── 2. Load signal log ─────────────────────────────────────────────────────────
# trades_log.csv records the ask-price queried at signal time as `signal_price`
# (or `fill_price` in older rows before the column was added).  Only buy/sell/hold
# rows have a non-zero price; flat/close rows get 0.0.
if not TRADES_LOG.exists():
    print(
        "\nWARNING: trades_log.csv not found — no expected_price can be recovered.\n"
        "  run_daily.py already logs signal_price for future buy/sell orders;\n"
        "  see src/trade_logger.py for the signal_price column added in this commit."
    )
    signal_df = pd.DataFrame()
else:
    sig = pd.read_csv(TRADES_LOG)
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True)

    # Support both the new `signal_price` column and the old `fill_price` column
    price_col = "signal_price" if "signal_price" in sig.columns else "fill_price"
    sig = sig[
        sig["action"].isin(["buy", "sell", "hold"]) & (sig[price_col] > 0)
    ].copy()
    sig = sig.rename(columns={"asset": "symbol", price_col: "expected_price"})
    signal_df = sig[["timestamp", "symbol", "expected_price"]].copy()
    print(f"  Signal log entries with a price: {len(signal_df)}")
    if not signal_df.empty:
        print(f"  Signal log date range: "
              f"{signal_df['timestamp'].min()} to {signal_df['timestamp'].max()}")

# ── 3. Join on symbol + nearest preceding signal ───────────────────────────────
rows = []
matched   = 0
unmatched = 0

for _, fill in fills_df.iterrows():
    sym   = fill["symbol"]
    ftime = fill["transaction_time"]

    expected = None
    if not signal_df.empty:
        candidates = signal_df[
            (signal_df["symbol"] == sym) &
            (signal_df["timestamp"] <= ftime) &
            (signal_df["timestamp"] >= ftime - timedelta(hours=WINDOW_HOURS))
        ]
        if not candidates.empty:
            idx = (ftime - candidates["timestamp"]).abs().idxmin()
            expected = float(candidates.loc[idx, "expected_price"])
            matched += 1
        else:
            unmatched += 1
    else:
        unmatched += 1

    rows.append({
        "timestamp":      ftime.isoformat(),
        "symbol":         sym,
        "side":           fill["side"],
        "qty":            fill["qty"],
        "expected_price": expected,
        "fill_price":     fill["fill_price"],
        "commission":     fill["commission"],
    })

# ── 4. Write output ────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out_df = pd.DataFrame(rows, columns=[
    "timestamp","symbol","side","qty","expected_price","fill_price","commission"
])
out_df.to_csv(OUT_PATH, index=False)

print(f"\n{'-'*55}")
print(f"Output:                {OUT_PATH}")
print(f"Total fills found:     {len(fills_df)}")
print(f"Joined (expected_price recovered): {matched}")
print(f"Missing expected_price:            {unmatched}")
if unmatched:
    print(
        "\nNOTE: All fills are missing expected_price.  Root cause:\n"
        "  • Fills from 2026-05-08 predate the trade logger (added 2026-06-03).\n"
        "  • Fills from 2026-07-17 onward: the Cloud Function writes trades_log.csv\n"
        "    to a local/ephemeral path; those entries are not persisted locally.\n"
        "  • The signal log has only 'hold' entries with prices — holds do not\n"
        "    generate fills, so there is no timestamp overlap.\n"
        "  run_daily.py now also passes signal_price= to log_trade() so future\n"
        "  local runs will log the expected price for buy/sell actions explicitly."
    )
