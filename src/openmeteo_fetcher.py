"""
Open-Meteo Historical Weather API — Energy-Demand Weather Fetcher
-----------------------------------------------------------------
No API key required. Fetches daily weather for key US energy-demand cities
and engineers heating/cooling degree day features relevant to XLE prediction.

Cities covered (chosen for energy consumption significance):
  Houston TX   — refining/petrochemical hub, Gulf Coast
  Chicago IL   — Midwest heating demand, NG hub
  New York NY  — Northeast demand corridor
  Atlanta GA   — Southeast demand / mild winters amplify swings

API endpoint: https://archive-api.open-meteo.com/v1/archive  (historical)
              https://api.open-meteo.com/v1/forecast          (recent/forecast)

Typical usage:
    from src.openmeteo_fetcher import fetch_weather, build_weather_features
    df = fetch_weather("2020-01-01", "2025-01-01")
    features = build_weather_features(df)
"""

import time
import hashlib
from pathlib import Path

import requests
import pandas as pd
import numpy as np

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_DIR = Path(__file__).parent.parent / "data" / "openmeteo_cache"

CITIES = {
    "houston":  {"latitude": 29.76, "longitude": -95.37, "timezone": "America/Chicago"},
    "chicago":  {"latitude": 41.88, "longitude": -87.63, "timezone": "America/Chicago"},
    "new_york": {"latitude": 40.71, "longitude": -74.01, "timezone": "America/New_York"},
    "atlanta":  {"latitude": 33.75, "longitude": -84.39, "timezone": "America/New_York"},
}

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
    "shortwave_radiation_sum",  # solar — proxy for renewable generation
]

HDD_BASE = 18.0  # °C (≈ 65°F)
CDD_BASE = 18.0


def _cache_path(city: str, start: str, end: str) -> Path:
    key = f"{city}_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"openmeteo_{city}_{h}.parquet"


def _fetch_city(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily weather for one city. Handles historical/forecast boundary."""
    from datetime import date
    today = date.today().isoformat()
    coords = CITIES[city]

    # Open-Meteo splits history vs forecast at ~today-5d
    if end_date <= today:
        url = HISTORICAL_URL
    else:
        url = FORECAST_URL

    params = {
        **coords,
        "start_date": start_date,
        "end_date": min(end_date, today),
        "daily": ",".join(DAILY_VARS),
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df["city"] = city
    df = df.drop(columns=["time"])
    return df


def fetch_weather(
    start_date: str,
    end_date: str,
    cities: list = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily weather for all cities and return a stacked DataFrame.

    Args:
        start_date:  "YYYY-MM-DD"
        end_date:    "YYYY-MM-DD"
        cities:      List of city names (keys of CITIES dict). Defaults to all 4.
        use_cache:   Cache each city's data to parquet.

    Returns:
        Long-format DataFrame with columns [date, city, temperature_2m_mean, ...]
    """
    cities = cities or list(CITIES.keys())
    frames = []
    for city in cities:
        cache_file = _cache_path(city, start_date, end_date)
        if use_cache and cache_file.exists():
            print(f"Loading Open-Meteo cache: {city}")
            frames.append(pd.read_parquet(cache_file))
            continue
        print(f"Fetching Open-Meteo: {city} ({start_date} to {end_date})")
        df = _fetch_city(city, start_date, end_date)
        if use_cache and not df.empty:
            df.to_parquet(cache_file, index=False)
        frames.append(df)
        time.sleep(0.2)

    return pd.concat(frames, ignore_index=True)


def build_weather_features(
    df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Aggregate multi-city weather into daily model-ready features.

    Output columns:
      - temp_mean_C:       population-weighted mean temperature across cities
      - hdd:               heating degree days (max(0, base - temp))
      - cdd:               cooling degree days (max(0, temp - base))
      - hdd_anom_20d:      HDD vs 20-day rolling mean (demand surprise)
      - temp_anom_20d:     temperature vs 20-day rolling mean
      - precip_sum:        total precipitation (weather disruption proxy)
      - wind_max:          max wind speed (renewable generation proxy)
      - solar_sum:         total shortwave radiation (solar generation proxy)
    """
    numeric = [c for c in DAILY_VARS if c in df.columns]
    daily = df.groupby("date")[numeric].mean().sort_index()

    temp = daily.get("temperature_2m_mean", daily.get("temperature_2m_max"))

    feats = pd.DataFrame(index=daily.index)
    feats["temp_mean_C"] = temp
    feats["hdd"] = np.maximum(0, HDD_BASE - temp)
    feats["cdd"] = np.maximum(0, temp - CDD_BASE)

    roll20 = temp.rolling(20, min_periods=5)
    feats["temp_anom_20d"] = temp - roll20.mean()
    feats["hdd_anom_20d"] = feats["hdd"] - feats["hdd"].rolling(20, min_periods=5).mean()

    if "precipitation_sum" in daily.columns:
        feats["precip_sum"] = daily["precipitation_sum"]
    if "windspeed_10m_max" in daily.columns:
        feats["wind_max"] = daily["windspeed_10m_max"]
    if "shortwave_radiation_sum" in daily.columns:
        feats["solar_sum"] = daily["shortwave_radiation_sum"]

    if trading_dates is not None:
        feats = feats.reindex(trading_dates, method="ffill")

    return feats


if __name__ == "__main__":
    df = fetch_weather("2024-01-01", "2024-03-31")
    print(f"Fetched {len(df)} rows across {df['city'].nunique()} cities")
    print(df.head(8).to_string())
    feats = build_weather_features(df)
    print("\nFeatures:")
    print(feats.head(10).to_string())
