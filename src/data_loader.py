"""
src/data_loader.py — Alternative-data pipeline for future XLE feature research.

Builds a weekly feature matrix combining EIA petroleum/gas signals, Open-Meteo
weather (HDD/CDD), GEM pipeline capacity, and Google Trends. All signals are
lagged 1 week to prevent lookahead bias.

This module is NOT used by the current core pipeline (run.py / model.py), which
relies only on ret_lag and the WTI shock filter. It is the foundation for the
next research phase: adding alternative-data features to the DirectionModel.

See what_actually_worked.md § "What to Try Next" for the planned extension.

Usage:
    from src.data_loader import build_feature_matrix
    df = build_feature_matrix(start="2018-01-01")

Keys loaded from .env via python-dotenv (never hardcoded).
"""

import os
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent / ".env")

# ── helpers ──────────────────────────────────────────────────────────────────

def _to_weekly_friday(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily-indexed df to week-ending-Friday."""
    return df.resample("W-FRI").last()


def _zscore(s: pd.Series, window: int = 52) -> pd.Series:
    roll = s.rolling(window, min_periods=window // 2)
    return (s - roll.mean()) / (roll.std() + 1e-8)


# ── 1. ETF returns ────────────────────────────────────────────────────────────

def load_etf_weekly(tickers=("XLE", "VDE"), start="2015-01-01", end=None) -> pd.DataFrame:
    """Weekly log returns for target ETFs via yfinance."""
    print("[ETF] Downloading weekly returns...")
    end = end or datetime.today().strftime("%Y-%m-%d")
    frames = {}
    for t in tickers:
        try:
            raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"].resample("W-FRI").last()
            frames[t] = np.log(close / close.shift(1))
        except Exception as e:
            print(f"  [ETF] WARNING: could not download {t}: {e}")
    df = pd.DataFrame(frames).dropna(how="all")
    print(f"  [ETF] {len(df)} weekly rows, {df.index.min().date()} to {df.index.max().date()}")
    return df


# ── 2. EIA weekly signals ─────────────────────────────────────────────────────

def load_eia_weekly(start="2015-01-01", end=None) -> pd.DataFrame:
    """EIA crude stocks, gas storage, refinery utilisation — weekly."""
    from src.eia_fetcher import fetch_series
    end = end or datetime.today().strftime("%Y-%m-%d")
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        print("  [EIA] WARNING: EIA_API_KEY not set, skipping EIA data.")
        return pd.DataFrame()

    results = {}
    series_map = {
        "crude_stocks":  ("petroleum/stoc/wstk/data/",  {"duoarea": "NUS", "product": "EPC0"}, "weekly"),
        "gas_storage":   ("natural-gas/stor/wkly/data/", {"series": "NW2_EPG0_SWO_R48_BCF"},   "weekly"),
        "refinery_util": ("petroleum/pnp/wiup/data/",    {"series": "WPULEUS3"},                "weekly"),
        "wti_price":     ("petroleum/pri/spt/data/",     {"product": "EPCWTI"},                 "daily"),
    }

    for name, (route, facets, freq) in series_map.items():
        try:
            df = fetch_series(name, start, end, api_key=api_key)
            if df.empty:
                raise ValueError("Empty response")
            s = df.set_index("date")["value"]
            s = s[~s.index.duplicated(keep="last")]
            s = s.resample("W-FRI").last()
            results[name] = s
            print(f"  [EIA] {name}: {len(s)} weekly rows")
        except Exception as e:
            print(f"  [EIA] WARNING: {name} failed: {e}")

    if not results:
        return pd.DataFrame()

    eia = pd.DataFrame(results)

    # Feature engineering
    eia["crude_chg_wow"] = eia["crude_stocks"].diff(1)
    eia["crude_zscore_52"] = _zscore(eia["crude_stocks"], 52)
    eia["gas_chg_wow"] = eia["gas_storage"].diff(1)
    eia["gas_zscore_52"] = _zscore(eia["gas_storage"], 52)
    eia["refutil_anom_12"] = eia["refinery_util"] - eia["refinery_util"].rolling(12, min_periods=6).mean()
    if "wti_price" in eia.columns:
        eia["wti_ret_weekly"] = np.log(eia["wti_price"] / eia["wti_price"].shift(1))
        eia["wti_mom_4w"] = eia["wti_price"].pct_change(4)

    keep = ["crude_chg_wow", "crude_zscore_52", "gas_chg_wow",
            "gas_zscore_52", "refutil_anom_12", "wti_ret_weekly", "wti_mom_4w"]
    return eia[[c for c in keep if c in eia.columns]]


# ── 3. Weather signals (Open-Meteo, no key needed) ───────────────────────────

def load_weather_weekly(start="2015-01-01", end=None) -> pd.DataFrame:
    """Weekly HDD/CDD and temperature anomaly from Open-Meteo."""
    from src.openmeteo_fetcher import fetch_weather, CITIES
    end = end or datetime.today().strftime("%Y-%m-%d")

    print("[Weather] Fetching Open-Meteo data...")
    try:
        df = fetch_weather(start, end, use_cache=True)
    except Exception as e:
        print(f"  [Weather] WARNING: fetch failed: {e}")
        return pd.DataFrame()

    numeric = [c for c in ["temperature_2m_mean", "temperature_2m_max",
                            "temperature_2m_min", "precipitation_sum",
                            "windspeed_10m_max"] if c in df.columns]
    daily = df.groupby("date")[numeric].mean()
    daily.index = pd.to_datetime(daily.index)

    temp = daily.get("temperature_2m_mean", (daily.get("temperature_2m_max", 0)
                                              + daily.get("temperature_2m_min", 0)) / 2)

    daily["hdd"] = np.maximum(0, 18.0 - temp)
    daily["cdd"] = np.maximum(0, temp - 18.0)

    # Resample to weekly sums/means
    weekly = pd.DataFrame({
        "hdd_weekly": daily["hdd"].resample("W-FRI").sum(),
        "cdd_weekly": daily["cdd"].resample("W-FRI").sum(),
        "temp_mean_weekly": temp.resample("W-FRI").mean(),
        "precip_weekly": daily["precipitation_sum"].resample("W-FRI").sum()
                         if "precipitation_sum" in daily.columns else np.nan,
    })

    # Anomalies vs 52-week rolling mean
    weekly["hdd_anom_12w"] = weekly["hdd_weekly"] - weekly["hdd_weekly"].rolling(12, min_periods=6).mean()
    weekly["cdd_anom_12w"] = weekly["cdd_weekly"] - weekly["cdd_weekly"].rolling(12, min_periods=6).mean()
    weekly["temp_zscore_52"] = _zscore(weekly["temp_mean_weekly"], 52)

    # Extreme cold flag: HDD > rolling mean + 1.5 std
    hdd_roll = weekly["hdd_weekly"].rolling(52, min_periods=26)
    weekly["extreme_cold"] = (weekly["hdd_weekly"] > hdd_roll.mean() + 1.5 * hdd_roll.std()).astype(int)

    print(f"  [Weather] {len(weekly)} weekly rows")
    return weekly


# ── 4. GEM pipeline signals ───────────────────────────────────────────────────

def load_gem_weekly(start="2015-01-01", end=None, trading_index=None) -> pd.DataFrame:
    """
    Quarterly pipeline capacity signals from GEM dataset, forward-filled weekly.
    """
    from src.gem_loader import load_gem_pipelines
    end = end or datetime.today().strftime("%Y-%m-%d")
    print("[GEM] Building pipeline capacity signals...")

    try:
        df = load_gem_pipelines()
    except Exception as e:
        print(f"  [GEM] WARNING: load failed: {e}")
        return pd.DataFrame()

    # Filter to North America operating + recently cancelled
    na = df[df["is_north_america"]].copy()
    cap_col = "capacityboed"
    na[cap_col] = pd.to_numeric(na[cap_col], errors="coerce")

    # Quarterly capacity additions: sum CapacityBOEd by StartYear1/quarter
    op = na[na["status"] == "operating"].dropna(subset=["start_year", cap_col]).copy()
    op["start_year"] = op["start_year"].astype(int)

    # Build annual time series 2010–present, then interpolate quarterly
    years = range(2010, datetime.today().year + 1)
    records = []
    for yr in years:
        active = op[op["start_year"] <= yr]
        new = op[op["start_year"] == yr]
        cancelled = na[(na.get("cancelledyear", pd.Series(dtype=float)).fillna(0) == yr)]
        under_con = na[na["status"] == "construction"]
        records.append({
            "year": yr,
            "gem_capacity_total": active[cap_col].sum(),
            "gem_additions": new[cap_col].sum(),
            "gem_cancellations": len(cancelled),
            "gem_construction_count": len(under_con),
        })

    annual = pd.DataFrame(records).set_index("year")
    annual["gem_additions_yoy"] = annual["gem_additions"].pct_change(1)
    annual["gem_cancel_rate"] = annual["gem_cancellations"] / (annual["gem_construction_count"] + 1)

    # Assign to start of each year, forward-fill to weekly
    annual.index = pd.to_datetime([f"{y}-01-01" for y in annual.index])

    idx = trading_index if trading_index is not None else pd.date_range(start, end, freq="W-FRI")
    weekly = annual.reindex(idx, method="ffill")
    print(f"  [GEM] {len(weekly)} weekly rows (forward-filled from annual)")
    return weekly[["gem_additions_yoy", "gem_cancel_rate", "gem_construction_count"]]


# ── 5. Google Trends signals ──────────────────────────────────────────────────

def load_trends_weekly(start="2015-01-01", end=None) -> pd.DataFrame:
    """Weekly Google Trends search volume for energy keywords."""
    from src.trends_fetcher import fetch_group
    end = end or datetime.today().strftime("%Y-%m-%d")
    print("[Trends] Fetching Google Trends data...")

    keywords = {
        "energy_prices": ["oil price", "gasoline price", "energy stocks"],
    }

    frames = {}
    for group, kws in keywords.items():
        try:
            df = fetch_group(kws, group, start, end, use_cache=True, request_delay=5.0)
            if df.empty:
                raise ValueError("Empty")
            df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
            # Resample to W-FRI to match other series
            df = df.resample("W-FRI").mean()
            for col in df.columns:
                col_safe = col.replace(" ", "_")
                frames[f"trend_{col_safe}"] = df[col]
                frames[f"trend_{col_safe}_4w_mom"] = df[col] - df[col].shift(4)
            print(f"  [Trends] {group}: {len(df)} weekly rows")
        except Exception as e:
            print(f"  [Trends] WARNING: {group} failed: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


# ── 6. Master feature matrix ──────────────────────────────────────────────────

def build_feature_matrix(
    start: str = "2015-01-01",
    end: str = None,
    target: str = "XLE",
    lag_periods: int = 1,
) -> pd.DataFrame:
    """
    Build aligned weekly feature matrix with all alternative data signals.

    All features are lagged by `lag_periods` weeks before merging with targets
    to prevent lookahead bias. Returns a DataFrame with:
      - columns prefixed with feature source
      - 'target_ret' column = next-week return of target ETF (not lagged)
      - 'xle_ret', 'vde_ret' = concurrent returns for reference
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"Building feature matrix: {start} to {end}, target={target}")
    print(f"{'='*60}")

    # Load targets
    etf = load_etf_weekly(start=start, end=end)
    idx = etf.index  # use ETF trading dates as master index

    # Load all signals
    sources = {}
    sources["eia"] = load_eia_weekly(start=start, end=end)
    sources["weather"] = load_weather_weekly(start=start, end=end)
    sources["gem"] = load_gem_weekly(start=start, end=end, trading_index=idx)
    sources["trends"] = load_trends_weekly(start=start, end=end)

    # Merge signals onto ETF index
    combined = pd.DataFrame(index=idx)
    for src_name, src_df in sources.items():
        if src_df.empty:
            print(f"  [merge] {src_name}: SKIPPED (empty)")
            continue
        src_df.index = pd.to_datetime(src_df.index)
        aligned = src_df.reindex(idx, method="ffill")
        # LAG all features by lag_periods to prevent lookahead
        aligned = aligned.shift(lag_periods)
        for col in aligned.columns:
            combined[col] = aligned[col]
        print(f"  [merge] {src_name}: {aligned.shape[1]} features merged")

    # Add target returns (NOT lagged — these are what we predict)
    if target in etf.columns:
        combined["target_ret"] = etf[target]
    if "XLE" in etf.columns:
        combined["xle_ret"] = etf["XLE"]
    if "VDE" in etf.columns:
        combined["vde_ret"] = etf["VDE"]

    combined = combined.dropna(subset=["target_ret"])

    # Fill any remaining NaN with 0 (forward-fill already applied)
    feature_cols = [c for c in combined.columns if c not in ("target_ret", "xle_ret", "vde_ret")]
    combined[feature_cols] = combined[feature_cols].fillna(0)

    n_features = len(feature_cols)
    print(f"\nFeature matrix: {len(combined)} rows x {n_features} features")
    print(f"Date range: {combined.index.min().date()} to {combined.index.max().date()}")
    print(f"Features: {feature_cols}")
    return combined


if __name__ == "__main__":
    df = build_feature_matrix(start="2018-01-01")
    print(df.tail(5).to_string())
