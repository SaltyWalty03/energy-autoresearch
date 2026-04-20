"""
Google Trends — Energy Sentiment Fetcher
-----------------------------------------
Fetches weekly search interest for energy-related keywords via pytrends.
Search volume acts as a retail-investor sentiment and demand-awareness proxy.

No API key required. Uses pytrends (unofficial Google Trends wrapper).
Recommended: 60s+ delays between requests to avoid IP throttling.

Keywords tracked:
  "oil price"        — consumer awareness of crude prices
  "gas prices"       — retail fuel price concern (demand elasticity signal)
  "natural gas"      — heating/utility demand awareness
  "energy stocks"    — retail investor sentiment toward the sector
  "XLE"              — direct ETF attention

Typical usage:
    from src.trends_fetcher import fetch_trends, build_trends_features
    df = fetch_trends("2020-01-01", "2025-01-01")
    features = build_trends_features(df)
"""

import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    from pytrends.request import TrendReq
except ImportError:
    raise ImportError("pytrends not installed. Run: pip install pytrends")

CACHE_DIR = Path(__file__).parent.parent / "data" / "trends_cache"

# Keywords grouped by theme (5 max per request — Google Trends limit)
KEYWORD_GROUPS = {
    "energy_prices": ["oil price", "gas prices", "natural gas"],
    "energy_invest":  ["energy stocks", "XLE"],
}

# Google Trends returns 0-100 index; normalise across keywords within each group


def _cache_path(group: str, start: str, end: str) -> Path:
    key = f"{group}_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"trends_{group}_{h}.parquet"


def _sanitize_colname(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def fetch_group(
    keywords: list,
    group_name: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    request_delay: float = 60.0,
) -> pd.DataFrame:
    """Fetch one keyword group from Google Trends (weekly data)."""
    cache_file = _cache_path(group_name, start_date, end_date)
    if use_cache and cache_file.exists():
        print(f"Loading Trends cache: {group_name}")
        return pd.read_parquet(cache_file)

    print(f"Fetching Google Trends: {keywords} ({start_date} to {end_date})")
    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 30))

    timeframe = f"{start_date} {end_date}"
    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo="US")

    # Small sleep before each request to avoid rate limiting
    time.sleep(request_delay)
    df = pytrends.interest_over_time()

    if df.empty:
        print(f"Warning: no data returned for {keywords}")
        return pd.DataFrame()

    df = df.drop(columns=["isPartial"], errors="ignore")
    df.columns = [_sanitize_colname(c) for c in df.columns]
    df.index.name = "date"
    df = df.reset_index()

    if use_cache:
        df.to_parquet(cache_file, index=False)
    return df


def fetch_trends(
    start_date: str,
    end_date: str,
    keyword_groups: dict = None,
    use_cache: bool = True,
    request_delay: float = 60.0,
) -> pd.DataFrame:
    """
    Fetch all keyword groups and merge into a single weekly DataFrame.

    Args:
        start_date:      "YYYY-MM-DD"
        end_date:        "YYYY-MM-DD"
        keyword_groups:  Dict of {group_name: [kw1, kw2, ...]}. Defaults to KEYWORD_GROUPS.
        use_cache:       Cache to parquet
        request_delay:   Seconds to wait between requests (default 60s — avoids throttle)

    Returns:
        DataFrame with one column per keyword, weekly indexed by date.
        Values are Google's 0-100 relative interest index.
    """
    keyword_groups = keyword_groups or KEYWORD_GROUPS
    frames = []
    for group_name, keywords in keyword_groups.items():
        df = fetch_group(keywords, group_name, start_date, end_date, use_cache, request_delay)
        if not df.empty:
            frames.append(df.set_index("date"))

    if not frames:
        raise RuntimeError("No Trends data fetched. Check internet connection.")

    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index()
    return combined.reset_index()


def build_trends_features(
    df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Engineer model-ready features from weekly Google Trends data.

    Raw weekly data is forward-filled to daily and the following features
    are computed:
      - <keyword>_idx:     raw 0-100 interest index (forward-filled)
      - <keyword>_chg4w:  4-week change in interest (momentum/spike detection)
      - energy_price_composite: average of oil/gas price search terms
      - sentiment_spike:  1 if any keyword is >1.5 std above its 26w mean

    Args:
        df:             DataFrame from fetch_trends()
        trading_dates:  Optional DatetimeIndex to reindex to trading calendar.

    Returns:
        DataFrame of daily features.
    """
    df = df.set_index("date") if "date" in df.columns else df
    df.index = pd.to_datetime(df.index)

    keyword_cols = [c for c in df.columns if not c.startswith("_")]

    # Forward-fill weekly data to daily
    daily_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    daily = df.reindex(daily_idx, method="ffill")

    feats = pd.DataFrame(index=daily.index)

    for col in keyword_cols:
        feats[f"{col}_idx"] = daily[col]
        feats[f"{col}_chg4w"] = daily[col] - daily[col].shift(28)

    # Composite energy price search (average of oil/gas keywords)
    price_cols = [f"{c}_idx" for c in keyword_cols if any(k in c for k in ["oil", "gas"])]
    if price_cols:
        feats["energy_price_composite"] = feats[price_cols].mean(axis=1)

    # Sentiment spike: any keyword > 1.5 std above its 26-week rolling mean
    spike_flags = []
    for col in keyword_cols:
        roll = daily[col].rolling(182, min_periods=30)
        z = (daily[col] - roll.mean()) / (roll.std() + 1e-8)
        spike_flags.append(z > 1.5)
    feats["sentiment_spike"] = pd.concat(spike_flags, axis=1).any(axis=1).astype(int)

    if trading_dates is not None:
        feats = feats.reindex(trading_dates, method="ffill")

    return feats


if __name__ == "__main__":
    # Single small test — uses real network request
    df = fetch_trends("2024-01-01", "2024-06-30", request_delay=5.0)
    print(df.head(8).to_string())
    feats = build_trends_features(df)
    print("\nFeatures:")
    print(feats.head(10).to_string())
