"""
src/trade_logger.py — Append-only trade log for run_daily.py.

log_trade(**kwargs) appends one row to trades_log.csv at the project root.
Creates the file with headers if it does not exist.

Columns: timestamp, asset, regime, n_regimes, signal, kelly_fraction,
         action, signal_price, fill_price, shares, account_equity

signal_price — ask price queried at signal time (expected execution price).
               Set for buy/sell/hold actions; 0.0 for flat/close.
fill_price   — kept for backward compatibility (same value as signal_price
               for rows written by this version; older rows only have fill_price).
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path(__file__).parent.parent / "trades_log.csv"
_COLUMNS  = [
    "timestamp", "asset", "regime", "n_regimes", "signal",
    "kelly_fraction", "action", "signal_price", "fill_price", "shares", "account_equity",
]


def log_trade(**kwargs) -> None:
    """Append one row to trades_log.csv. Missing columns default to empty string."""
    write_header = not _LOG_PATH.exists() or _LOG_PATH.stat().st_size == 0
    row = {col: kwargs.get(col, "") for col in _COLUMNS}
    if not row["timestamp"]:
        row["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(_LOG_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
