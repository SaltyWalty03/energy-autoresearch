"""
EIA Open Data API v2 — Energy Market Data Fetcher
--------------------------------------------------
Fetches the most XLE-relevant series from the US Energy Information Administration:
  - WTI crude oil spot price (daily)
  - Henry Hub natural gas spot price (daily)
  - Weekly crude oil commercial stocks (US)
  - Weekly natural gas in storage (Lower 48)
  - US refinery utilisation rate (weekly)

Authentication:
    Free API key at https://www.eia.gov/opendata/
    Set env var:  EIA_API_KEY=your_key
    Or pass api_key= to each fetch function.

Typical usage:
    from src.eia_fetcher import fetch_all, build_eia_features
    df = fetch_all("2020-01-01", "2025-01-01")
    features = build_eia_features(df)
"""

import os
import time
import hashlib
from pathlib import Path

import requests
import pandas as pd
import numpy as np

BASE_URL = "https://api.eia.gov/v2"
CACHE_DIR = Path(__file__).parent.parent / "data" / "eia_cache"

# (route, facets, frequency)
SERIES = {
    "wti_price": (
        "petroleum/pri/spt/data/",
        {"product": "EPCWTI"},
        "daily",
    ),
    "henry_hub": (
        "natural-gas/pri/fut/data/",
        {"series": "RNGC1"},
        "daily",
    ),
    "crude_stocks": (
        "petroleum/stoc/wstk/data/",
        {"duoarea": "NUS", "product": "EPC0"},
        "weekly",
    ),
    "gas_storage": (
        "natural-gas/stor/wkly/data/",
        {"series": "NW2_EPG0_SWO_R48_BCF"},
        "weekly",
    ),
    "refinery_util": (
        "petroleum/pnp/wiup/data/",
        {"series": "WPULEUS3"},
        "weekly",
    ),
}


def _cache_path(series_name: str, start: str, end: str) -> Path:
    key = f"{series_name}_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"eia_{series_name}_{h}.parquet"


def _fetch_series(
    route: str,
    filters: dict,
    start: str,
    end: str,
    api_key: str,
    frequency: str = "weekly",
) -> pd.DataFrame:
    """Low-level fetch for one EIA v2 series. Returns tidy DataFrame."""
    params = {
        "api_key": api_key,
        "data[]": "value",
        "frequency": frequency,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
        **{f"facets[{k}][]": v for k, v in filters.items()},
    }
    url = f"{BASE_URL}/{route}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    records = payload.get("response", {}).get("data", [])
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)


def fetch_series(
    name: str,
    start_date: str,
    end_date: str,
    api_key: str = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch a single named EIA series.

    Args:
        name:        One of: wti_price, henry_hub, crude_stocks, gas_storage, refinery_util
        start_date:  "YYYY-MM-DD"
        end_date:    "YYYY-MM-DD"
        api_key:     EIA API key (falls back to EIA_API_KEY env var)
        use_cache:   Cache to parquet

    Returns:
        DataFrame with columns [date, value]
    """
    api_key = api_key or os.environ.get("EIA_API_KEY", "")
    if not api_key:
        raise ValueError(
            "EIA API key required. Register at https://www.eia.gov/opendata/ "
            "then set env var EIA_API_KEY=your_key or pass api_key= argument."
        )
    if name not in SERIES:
        raise ValueError(f"Unknown series '{name}'. Choose from: {list(SERIES.keys())}")

    cache_file = _cache_path(name, start_date, end_date)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    route, filters, frequency = SERIES[name]
    print(f"Fetching EIA: {name} ({start_date} to {end_date})")
    df = _fetch_series(route, filters, start_date, end_date, api_key, frequency)
    if use_cache and not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


def fetch_all(
    start_date: str,
    end_date: str,
    api_key: str = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch all five EIA series and merge into a single wide DataFrame.

    Returns DataFrame with columns: [wti_price, henry_hub, crude_stocks,
                                      gas_storage, refinery_util]
    indexed by date. Weekly series are forward-filled to daily.
    """
    api_key = api_key or os.environ.get("EIA_API_KEY", "")
    if not api_key:
        raise ValueError(
            "EIA API key required. Register at https://www.eia.gov/opendata/ "
            "then set env var EIA_API_KEY=your_key or pass api_key= argument."
        )
    frames = {}
    for name in SERIES:
        try:
            df = fetch_series(name, start_date, end_date, api_key, use_cache)
            if not df.empty:
                s = df.set_index("date")["value"].rename(name)
                s = s[~s.index.duplicated(keep="last")]
                frames[name] = s
            time.sleep(0.3)
        except Exception as e:
            print(f"Warning: could not fetch {name}: {e}")

    if not frames:
        raise RuntimeError("No EIA data fetched. Check API key and series config.")

    combined = pd.concat(frames.values(), axis=1)
    daily_idx = pd.date_range(start_date, end_date, freq="D")
    combined = combined.reindex(daily_idx).ffill()
    return combined


def build_eia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer model-ready features from the raw EIA wide DataFrame.

    Returns columns:
      - wti_ret:          daily log return of WTI
      - wti_ma5_ret:      5-day rolling mean of WTI log return
      - wti_vol20:        20-day rolling std of WTI log return
      - hhub_ret:         daily log return of Henry Hub gas price
      - crude_chg_yoy:    crude stocks % change vs 52 weeks prior
      - gas_stor_chg_yoy: gas storage % change vs 52 weeks prior
      - refutil_anom20:   refinery utilisation deviation from 20-day mean
    """
    feats = pd.DataFrame(index=df.index)

    if "wti_price" in df.columns:
        wti = df["wti_price"].ffill()
        feats["wti_ret"] = np.log(wti / wti.shift(1))
        feats["wti_ma5_ret"] = feats["wti_ret"].rolling(5).mean()
        feats["wti_vol20"] = feats["wti_ret"].rolling(20).std()

    if "henry_hub" in df.columns:
        hh = df["henry_hub"].ffill()
        feats["hhub_ret"] = np.log(hh / hh.shift(1))

    if "crude_stocks" in df.columns:
        cs = df["crude_stocks"].ffill()
        feats["crude_chg_yoy"] = cs.pct_change(252)

    if "gas_storage" in df.columns:
        gs = df["gas_storage"].ffill()
        feats["gas_stor_chg_yoy"] = gs.pct_change(52)

    if "refinery_util" in df.columns:
        ru = df["refinery_util"].ffill()
        feats["refutil_anom20"] = ru - ru.rolling(20).mean()

    return feats.dropna(how="all")


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EIA_API_KEY", "")
    df = fetch_all("2024-01-01", "2024-06-30", api_key=key)
    print(df.tail(10).to_string())
    feats = build_eia_features(df)
    print("\nFeatures:")
    print(feats.tail(10).to_string())
