"""
GEM Global Oil Infrastructure Tracker — Pipeline Data Loader
-------------------------------------------------------------
Parses the GEM Excel file (data/GEM-GOIT-Oil-NGL-Pipelines-2025-03.xlsx)
and extracts time-series-compatible features for use in the prediction pipeline.

Useful signals for XLE prediction:
  - US pipeline capacity additions per year (new projects coming online)
  - Proportion of capacity under construction vs operating (supply-side pressure)
  - Pipeline capacity by fuel source (Oil vs NGL)

Typical usage:
    from src.gem_loader import load_gem_pipelines, build_pipeline_features
    df = load_gem_pipelines()
    features = build_pipeline_features(df, trading_dates=price_index)
"""

from pathlib import Path
import pandas as pd
import numpy as np

GEM_FILE = Path(__file__).parent.parent / "data" / "GEM-GOIT-Oil-NGL-Pipelines-2025-03.xlsx"

# Status values treated as operational/active
OPERATING_STATUSES = {"operating"}
CONSTRUCTION_STATUSES = {"construction", "under construction"}
PROPOSED_STATUSES = {"proposed", "pre-construction", "announced"}
CANCELLED_STATUSES = {"cancelled", "shelved", "mothballed", "retired"}


def load_gem_pipelines(path: Path = GEM_FILE) -> pd.DataFrame:
    """
    Load and clean the GEM pipeline Excel file.

    Returns a DataFrame where each row is one pipeline record with
    standardised column names and numeric capacity/length fields.
    """
    df = pd.read_excel(path, sheet_name="Pipelines")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

    # Clean capacity: strip commas and convert to float
    if "capacity" in df.columns:
        df["capacity"] = (
            df["capacity"].astype(str).str.replace(",", "").str.strip()
        )
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    df["capacityboed"] = pd.to_numeric(df.get("capacityboed", pd.Series(dtype=float)), errors="coerce")

    # Normalise status field
    df["status"] = df["status"].astype(str).str.strip().str.lower()

    # Parse year fields
    for col in ["startyear1", "startyear2", "startyear3", "cancelledyear", "constructionyear"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Best-estimate operational start year
    df["start_year"] = df.get("startyear1", pd.Series(dtype=float))

    # Broad geographic filter: rows touching the US or Canada (most relevant to XLE)
    us_canada_mask = df["countries"].astype(str).str.contains(
        r"United States|Canada", case=False, na=False
    )
    df["is_north_america"] = us_canada_mask

    return df


def build_pipeline_features(
    df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex = None,
    region_filter: str = "north_america",
) -> pd.DataFrame:
    """
    Convert static pipeline records into annual/daily time-series features.

    For each year between the earliest start year and 2025 this computes:
      - capacity_operating_bpd:     total bpd of operating pipelines
      - capacity_construction_bpd:  total bpd under construction
      - capacity_additions_bpd:     new bpd that came online that year
      - n_operating:                count of operating pipelines
      - construction_ratio:         construction / (operating + 1) capacity ratio

    The annual values are then forward-filled to a daily index if trading_dates
    is provided, simulating the information available at each trading date.

    Args:
        df:             DataFrame from load_gem_pipelines().
        trading_dates:  Optional DatetimeIndex for daily reindexing.
        region_filter:  "north_america" (default) or "global".

    Returns:
        DataFrame with pipeline feature columns indexed by year (or by date).
    """
    if region_filter == "north_america":
        df = df[df["is_north_america"]].copy()

    cap_col = "capacityboed"

    # --- Operating capacity per year ---
    op = df[df["status"].isin(OPERATING_STATUSES)].copy()
    op = op.dropna(subset=["start_year", cap_col])
    op["start_year"] = op["start_year"].astype(int)

    years = range(int(op["start_year"].min()), 2026)
    annual = []
    for yr in years:
        active = op[op["start_year"] <= yr]
        new_this_yr = op[op["start_year"] == yr]
        cons = df[df["status"].isin(CONSTRUCTION_STATUSES)]
        annual.append({
            "year": yr,
            "capacity_operating_bpd": active[cap_col].sum(),
            "capacity_additions_bpd": new_this_yr[cap_col].sum(),
            "n_operating": len(active),
            "capacity_construction_bpd": cons[cap_col].sum(),
        })

    feats = pd.DataFrame(annual).set_index("year")
    feats["construction_ratio"] = (
        feats["capacity_construction_bpd"] / (feats["capacity_operating_bpd"] + 1)
    )
    feats["capacity_growth_yoy"] = feats["capacity_operating_bpd"].pct_change()

    if trading_dates is None:
        return feats

    # Forward-fill annual values to daily trading dates
    daily_index = pd.DatetimeIndex([pd.Timestamp(yr, 1, 1) for yr in feats.index])
    feats_daily = feats.copy()
    feats_daily.index = daily_index
    feats_daily = feats_daily.reindex(trading_dates, method="ffill")

    return feats_daily


if __name__ == "__main__":
    df = load_gem_pipelines()
    print(f"Loaded {len(df)} pipeline records")
    print(f"Statuses: {df['status'].value_counts().to_dict()}")
    print(f"North America records: {df['is_north_america'].sum()}")
    feats = build_pipeline_features(df)
    print("\nPipeline features (recent years):")
    print(feats.tail(10).to_string())
