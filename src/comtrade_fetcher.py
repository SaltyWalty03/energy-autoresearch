"""
UN Comtrade+ API — US Energy Trade Flow Fetcher
------------------------------------------------
Fetches monthly US imports/exports of crude oil and petroleum products.
Trade flow data provides a macro supply-side signal for energy ETF prediction.

Key commodities (HS codes):
  2709 — Petroleum oils, crude
  2710 — Petroleum oils, not crude (refined products)
  2711 — Petroleum gases (LNG, LPG, natural gas)

Authentication:
    Register at https://comtradedeveloper.un.org/ (free tier available)
    Set env var: COMTRADE_KEY=your_subscription_key
    Or pass api_key= argument.

Free tier limits: ~500 requests/day, monthly data only.

Typical usage:
    from src.comtrade_fetcher import fetch_trade, build_trade_features
    df = fetch_trade("2020", "2025")
    features = build_trade_features(df)
"""

import os
import time
import hashlib
from pathlib import Path

import requests
import pandas as pd
import numpy as np

BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/M"
CACHE_DIR = Path(__file__).parent.parent / "data" / "comtrade_cache"

HS_CODES = {
    "crude_oil":         "2709",
    "refined_products":  "2710",
    "petroleum_gas":     "2711",
}

REPORTER_USA = "842"   # UN country code for USA
FLOW_IMPORT = "M"
FLOW_EXPORT = "X"


def _cache_path(label: str, start_year: str, end_year: str) -> Path:
    key = f"{label}_{start_year}_{end_year}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"comtrade_{label}_{h}.parquet"


def fetch_trade(
    start_year: str,
    end_year: str,
    hs_codes: dict = None,
    api_key: str = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch monthly US trade data for energy commodities.

    Args:
        start_year:  "YYYY"
        end_year:    "YYYY"
        hs_codes:    Dict of {label: hs_code}. Defaults to crude, refined, gas.
        api_key:     Comtrade+ subscription key (falls back to COMTRADE_KEY env var)
        use_cache:   Cache to parquet

    Returns:
        Wide DataFrame indexed by date with import/export netflow columns.
        All trade values in USD.
    """
    api_key = api_key or os.environ.get("COMTRADE_KEY", "")
    if not api_key:
        raise ValueError(
            "UN Comtrade API key required. Register at https://comtradedeveloper.un.org/ "
            "then set env var COMTRADE_KEY=your_key or pass api_key= argument."
        )

    hs_codes = hs_codes or HS_CODES
    cache_key = "_".join(sorted(hs_codes.keys()))
    cache_file = _cache_path(cache_key, start_year, end_year)
    if use_cache and cache_file.exists():
        print(f"Loading Comtrade cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    # Build period list: "202001,202002,...,202512"
    periods = []
    for yr in range(int(start_year), int(end_year) + 1):
        for mo in range(1, 13):
            periods.append(f"{yr}{mo:02d}")
    period_str = ",".join(periods)

    headers = {"Ocp-Apim-Subscription-Key": api_key}
    records = []

    for label, cmd in hs_codes.items():
        for flow_code, flow_name in [(FLOW_IMPORT, "import"), (FLOW_EXPORT, "export")]:
            params = {
                "reporterCode": REPORTER_USA,
                "period": period_str,
                "flowCode": flow_code,
                "cmdCode": cmd,
                "includeDesc": "false",
            }
            url = f"{BASE_URL}/{cmd}"
            print(f"Fetching Comtrade: {label} {flow_name} ({start_year}–{end_year})")
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            for row in data:
                records.append({
                    "date": pd.to_datetime(f"{row['period'][:4]}-{row['period'][4:6]}-01"),
                    "commodity": label,
                    "flow": flow_name,
                    "trade_value_usd": float(row.get("primaryValue", 0) or 0),
                    "qty_kg": float(row.get("netWgt", 0) or 0),
                })
            time.sleep(0.5)

    if not records:
        raise RuntimeError("No Comtrade data returned. Check API key and parameters.")

    df = pd.DataFrame(records)

    # Pivot to wide: one column per commodity/flow
    df["col"] = df["commodity"] + "_" + df["flow"] + "_usd"
    wide = df.pivot_table(index="date", columns="col", values="trade_value_usd", aggfunc="sum")
    wide.columns.name = None
    wide = wide.sort_index().reset_index()

    if use_cache:
        wide.to_parquet(cache_file, index=False)

    return wide


def build_trade_features(
    df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Engineer model-ready features from monthly trade data.

    Output columns (where data is available):
      - crude_net_flow:      crude imports - exports (supply pressure)
      - crude_import_yoy:    crude import value YoY % change
      - gas_net_flow:        LNG/gas net imports
      - total_energy_import: sum of all energy commodity imports
    """
    df = df.set_index("date") if "date" in df.columns else df

    feats = pd.DataFrame(index=df.index)

    imp_cols = [c for c in df.columns if c.endswith("_import_usd")]
    exp_cols = [c for c in df.columns if c.endswith("_export_usd")]

    for col in imp_cols:
        commodity = col.replace("_import_usd", "")
        exp_col = f"{commodity}_export_usd"
        if exp_col in df.columns:
            feats[f"{commodity}_net_flow"] = df[col] - df[exp_col]
        feats[f"{commodity}_import_yoy"] = df[col].pct_change(12)

    if imp_cols:
        feats["total_energy_import"] = df[imp_cols].sum(axis=1)

    if trading_dates is not None:
        feats = feats.reindex(trading_dates, method="ffill")

    return feats


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COMTRADE_KEY", "")
    df = fetch_trade("2022", "2024", api_key=key)
    print(df.tail(6).to_string())
    feats = build_trade_features(df)
    print("\nFeatures:")
    print(feats.tail(6).to_string())
