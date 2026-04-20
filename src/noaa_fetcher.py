"""
NOAA NCEI Climate Data Fetcher
-------------------------------
Fetches daily climate summaries from the NCEI Access Data Service v1 API.
Useful features for energy demand prediction: TAVG, TMAX, TMIN, HDD, CDD.

Authentication:
    Register for a free API token at https://www.ncdc.noaa.gov/cdo-web/token
    Then either:
      - Set env var:  NOAA_TOKEN=your_token
      - Or pass token= directly to fetch_climate()

Rate limits: 5 req/s, 10,000 req/day (free tier)

Typical usage:
    from src.noaa_fetcher import fetch_climate, build_temp_features
    df = fetch_climate("2020-01-01", "2025-01-01")
    features = build_temp_features(df)
"""

import os
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np

BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
CACHE_DIR = Path(__file__).parent.parent / "data" / "noaa_cache"

# Key US energy-relevant stations (Houston, Chicago, New York, Los Angeles)
DEFAULT_STATIONS = [
    "GHCND:USW00012918",  # Houston Intercontinental
    "GHCND:USW00094846",  # Chicago O'Hare
    "GHCND:USW00094728",  # New York JFK
    "GHCND:USW00023174",  # Los Angeles Intl
]

DEFAULT_DATATYPES = ["TAVG", "TMAX", "TMIN"]


def _cache_path(start: str, end: str, stations: list, datatypes: list) -> Path:
    key = f"{start}_{end}_{'_'.join(sorted(stations))}_{'_'.join(sorted(datatypes))}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"noaa_{h}.parquet"


def fetch_climate(
    start_date: str,
    end_date: str,
    stations: list = None,
    datatypes: list = None,
    token: str = None,
    use_cache: bool = True,
    chunk_days: int = 365,
) -> pd.DataFrame:
    """
    Fetch daily climate data from NCEI and return a tidy DataFrame.

    Args:
        start_date:  ISO date string, e.g. "2020-01-01"
        end_date:    ISO date string, e.g. "2025-01-01"
        stations:    List of GHCND station IDs. Defaults to 4 major US cities.
        datatypes:   List of data type codes. Defaults to TAVG, TMAX, TMIN.
        token:       NOAA API token. Falls back to NOAA_TOKEN env var.
        use_cache:   Cache results to disk as parquet (avoids re-fetching).
        chunk_days:  API query window size in days (max ~1 year recommended).

    Returns:
        DataFrame with columns [date, station, TAVG, TMAX, TMIN, ...]
        Temperature values are in tenths of Celsius (divide by 10 for °C).
    """
    stations = stations or DEFAULT_STATIONS
    datatypes = datatypes or DEFAULT_DATATYPES
    token = token or os.environ.get("NOAA_TOKEN", "")

    if not token:
        raise ValueError(
            "NOAA API token required. Register at https://www.ncdc.noaa.gov/cdo-web/token "
            "then set env var NOAA_TOKEN=your_token or pass token= argument."
        )

    cache_file = _cache_path(start_date, end_date, stations, datatypes)
    if use_cache and cache_file.exists():
        print(f"Loading NOAA data from cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    headers = {"token": token}
    params_base = {
        "dataset": "daily-summaries",
        "stations": ",".join(stations),
        "dataTypes": ",".join(datatypes),
        "format": "json",
        "units": "metric",
        "includeAttributes": "false",
    }

    # Chunk into yearly windows to stay within API limits
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    chunks = []
    cursor = start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_dt)
        params = {
            **params_base,
            "startDate": cursor.strftime("%Y-%m-%d"),
            "endDate": chunk_end.strftime("%Y-%m-%d"),
        }
        print(f"Fetching NOAA: {params['startDate']} → {params['endDate']}")
        resp = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data:
            chunks.extend(data if isinstance(data, list) else data.get("results", []))
        time.sleep(0.25)  # stay under 5 req/s rate limit
        cursor = chunk_end + timedelta(days=1)

    if not chunks:
        raise ValueError("No data returned. Check station IDs, date range, and token.")

    df = pd.DataFrame(chunks)
    df["date"] = pd.to_datetime(df["date"])

    # Pivot so each datatype is a column; one row per (date, station)
    df = df.pivot_table(index=["date", "station"], columns="datatype", values="value", aggfunc="first")
    df.columns.name = None
    df = df.reset_index()
    df = df.sort_values("date").reset_index(drop=True)

    if use_cache:
        df.to_parquet(cache_file, index=False)
        print(f"Cached to {cache_file}")

    return df


def build_temp_features(df: pd.DataFrame, trading_dates: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Aggregate multi-station daily climate data into model-ready features.

    Computes for each date:
      - temp_avg_C:        national average of TAVG across stations (°C)
      - temp_anom_5d:      deviation of temp_avg from its 5-year daily mean
      - temp_anom_20d:     20-day rolling deviation from seasonal baseline
      - hdd:               heating degree days (max(0, 18 - temp_avg_C))
      - cdd:               cooling degree days (max(0, temp_avg_C - 18))
      - hdd_anom_20d:      HDD deviation from 20-day rolling mean

    Args:
        df:             DataFrame from fetch_climate()
        trading_dates:  Optional DatetimeIndex to reindex output to trading calendar.

    Returns:
        DataFrame indexed by date with feature columns.
    """
    numeric_cols = [c for c in ["TAVG", "TMAX", "TMIN"] if c in df.columns]
    if not numeric_cols:
        raise ValueError("No temperature columns found in data.")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Average across stations per day
    daily = df.groupby("date")[numeric_cols].mean()
    daily.columns = [c.lower() for c in daily.columns]

    # Use TAVG if available, else average TMAX and TMIN
    if "tavg" in daily.columns:
        daily["temp_avg_C"] = daily["tavg"]
    else:
        daily["temp_avg_C"] = (daily["tmax"] + daily["tmin"]) / 2

    # 20-day rolling mean as seasonal baseline proxy
    daily["temp_baseline_20d"] = daily["temp_avg_C"].rolling(20, min_periods=5).mean()
    daily["temp_anom_20d"] = daily["temp_avg_C"] - daily["temp_baseline_20d"]

    # 5-day anomaly
    daily["temp_baseline_5d"] = daily["temp_avg_C"].rolling(5, min_periods=2).mean()
    daily["temp_anom_5d"] = daily["temp_avg_C"] - daily["temp_baseline_5d"].shift(1)

    # Degree days (base 18°C)
    daily["hdd"] = np.maximum(0, 18 - daily["temp_avg_C"])
    daily["cdd"] = np.maximum(0, daily["temp_avg_C"] - 18)
    daily["hdd_anom_20d"] = daily["hdd"] - daily["hdd"].rolling(20, min_periods=5).mean()

    features = daily[["temp_avg_C", "temp_anom_20d", "temp_anom_5d", "hdd", "cdd", "hdd_anom_20d"]]

    if trading_dates is not None:
        features = features.reindex(trading_dates, method="ffill")

    return features


if __name__ == "__main__":
    import sys
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NOAA_TOKEN", "")
    df = fetch_climate("2024-01-01", "2024-03-31", token=token)
    print(df.head(10).to_string())
    feats = build_temp_features(df)
    print("\nFeatures:")
    print(feats.head(10).to_string())
