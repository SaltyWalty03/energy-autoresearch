import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

_CACHE_DIR = Path(__file__).parent / "data" / "model_cache"


def _load_commodity(ticker: str = "CL=F") -> pd.Series:
    """
    Load daily commodity closing prices.
    For CL=F (WTI crude) tries the EIA parquet cache first.
    Falls back to yfinance for all tickers.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("=", "_").replace("/", "_")
    cache_file = _CACHE_DIR / f"{safe}_daily.parquet"

    # WTI-specific: prefer EIA parquet (offline, covers 2020-2026)
    if ticker == "CL=F":
        eia_dir = Path(__file__).parent / "data" / "eia_cache"
        eia_files = sorted(eia_dir.glob("eia_wti_price_*.parquet"),
                           key=lambda f: f.stat().st_size, reverse=True)
        if eia_files:
            df = pd.read_parquet(eia_files[0])
            s = df.set_index("date")["value"]
            s.index = pd.to_datetime(s.index)
            return s.sort_index()

    # yfinance fallback (all tickers), cached for 23 h
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 82800:
        return pd.read_parquet(cache_file)["close"]
    try:
        raw = yf.download(ticker, start="2019-01-01", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw[["Close"]].rename(columns={"Close": "close"}).to_parquet(cache_file)
        return raw["Close"]
    except Exception:
        return pd.Series(dtype=float)


class DirectionModel(BaseEstimator, RegressorMixin):
    """
    RF direction classifier — improvements over baseline:

    1. 3-year rolling training window  (drops COVID 2020-21 regime)
    2. Exponential sample-decay (0.55→1.0) within that window
    3. WTI shock filter: output 0 (flat) on days where yesterday's
       WTI crude return exceeded ±2% — energy-market shocks make
       the momentum signal unreliable on those days.
    """

    def __init__(self, window=20, n_estimators=300, max_depth=2,
                 min_samples_leaf=22, train_window=756, wti_thresh=0.02,
                 model_type="rf", learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, commodity_ticker="CL=F"):
        self.window           = window
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.train_window     = train_window
        self.wti_thresh       = wti_thresh
        self.model_type       = model_type
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.commodity_ticker = commodity_ticker

    def _rolling_features(self, r, start_idx):
        feats = []
        for i in range(start_idx, len(r)):
            w5  = r[max(0, i - 4): i + 1]
            w20 = r[max(0, i - 19): i + 1]
            v5  = np.std(w5)  + 1e-8
            v20 = np.std(w20) + 1e-8
            feats.append([
                r[i],
                r[i] / v5,
                np.mean(w5),
                np.mean(w20),
                np.mean(w5) / (np.mean(w20) + 1e-8),
                v5 / v20,
            ])
        return np.array(feats)

    def _wti_lag_ret(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """WTI lagged log-return aligned to idx (continuous cross-asset signal)."""
        prices = _load_commodity(self.commodity_ticker)
        if len(prices) == 0:
            return np.zeros(len(idx))
        ret = np.log(prices.clip(lower=1e-8)).diff(1).shift(1)
        return ret.reindex(idx, method="ffill").fillna(0.0).values

    def _temp_anomaly(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """
        Yesterday's Houston temperature anomaly (temp - 30d rolling mean), shifted 1d.
        Uses the largest cached OpenMeteo Houston parquet; falls back to zeros.
        """
        cache_dir = Path(__file__).parent / "data" / "openmeteo_cache"
        files = sorted(cache_dir.glob("openmeteo_houston_*.parquet"),
                       key=lambda f: f.stat().st_size, reverse=True)
        if not files:
            return np.zeros(len(idx))
        df = pd.read_parquet(files[0])
        df["date"] = pd.to_datetime(df["date"])
        temp = df.sort_values("date").set_index("date")["temperature_2m_mean"]
        rolling_mean = temp.rolling(30, min_periods=15).mean()
        anomaly = (temp - rolling_mean).shift(1)  # yesterday's anomaly
        return anomaly.reindex(idx, method="ffill").fillna(0.0).values

    def _natgas_zscore(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """
        NatGas storage 20-day rolling z-score (stationarised, lagged 1 period).
        Uses the largest EIA gas-storage parquet (weekly, ffilled to daily).
        Falls back to zeros if unavailable.
        """
        eia_dir = Path(__file__).parent / "data" / "eia_cache"
        files = sorted(eia_dir.glob("eia_gas_storage_*.parquet"),
                       key=lambda f: f.stat().st_size, reverse=True)
        if not files:
            return np.zeros(len(idx))
        df = pd.read_parquet(files[0])
        storage = df.set_index("date")["value"]
        storage.index = pd.to_datetime(storage.index)
        storage = storage.sort_index()
        # Forward-fill weekly EIA releases to daily, then z-score
        storage_daily = storage.reindex(idx, method="ffill")
        roll_mean = storage_daily.rolling(20, min_periods=10).mean()
        roll_std  = storage_daily.rolling(20, min_periods=10).std() + 1e-8
        zscore = ((storage_daily - roll_mean) / roll_std).shift(1)
        return zscore.fillna(0.0).values

    def _commodity_shock_mask(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """
        Returns a boolean array (len(idx)) that is True on days where
        yesterday's commodity |return| > wti_thresh. Falls back to all-False
        if data is unavailable so the model degrades gracefully.
        """
        prices = _load_commodity(self.commodity_ticker)
        if len(prices) == 0:
            return np.zeros(len(idx), dtype=bool)
        ret = np.log(prices + 1e-8).diff(1).shift(1)   # yesterday's return
        shock = ret.abs() > self.wti_thresh
        return shock.reindex(idx, method="ffill").fillna(False).values

    def fit(self, X, y):
        """
        Fit the direction model.

        X must be a single-column DataFrame of daily percentage returns (ret_lag),
        indexed by date. The model ignores any additional columns and internally
        computes six rolling features (raw return, vol-adjusted return, 5-day/20-day
        mean, MACD ratio, vol-ratio). Passing multi-column X will silently use only
        the first column after ravel().
        """
        r_full  = np.array(X).ravel()
        y_full  = np.array(y)
        n_full  = len(r_full)

        # ── 3-year rolling window ──────────────────────────────────────────────
        cutoff  = max(0, n_full - self.train_window)
        r       = r_full[cutoff:]
        y_w     = y_full[cutoff:]

        # Preserve full tail for rolling-feature continuity in predict()
        self._tail = r_full[-self.window:]

        # ── Exponential sample-decay (steeper tilt toward recent data) ──────────
        weights = np.exp(np.linspace(-0.3, 0.0, n_full))[cutoff:]

        # ── Train ─────────────────────────────────────────────────────────────
        F        = self._rolling_features(r, 0)
        wti_full = self._wti_lag_ret(X.index)
        wti_win  = wti_full[cutoff:]
        # ret_lag × Temp_Anomaly (program.md priority #1)
        temp_full = self._temp_anomaly(X.index)
        temp_win  = temp_full[cutoff:]
        interact_temp = r * temp_win
        # ret_lag × NatGas_Storage_zscore (program.md priority #2)
        ng_full = self._natgas_zscore(X.index)
        ng_win  = ng_full[cutoff:]
        interact_ng = r * ng_win
        F        = np.column_stack([F, wti_win, interact_temp, interact_ng])

        labels = (y_w > 0).astype(int)

        self.scaler_ = StandardScaler()
        F_s = self.scaler_.fit_transform(F)

        if self.model_type == "xgb":
            if not _XGB_AVAILABLE:
                raise ImportError("xgboost is not installed. Run: pip install xgboost")
            self.clf_ = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        else:
            self.clf_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
                random_state=42,
            )
        self.clf_.fit(F_s, labels, sample_weight=weights)
        return self

    def predict(self, X):
        r     = np.concatenate([self._tail, np.array(X).ravel()])
        F     = self._rolling_features(r, len(self._tail))
        r_val = r[len(self._tail):]
        wti   = self._wti_lag_ret(X.index)
        temp  = self._temp_anomaly(X.index)
        ng    = self._natgas_zscore(X.index)
        F     = np.column_stack([F, wti, r_val * temp, r_val * ng])
        F_s = self.scaler_.transform(F)
        proba = self.clf_.predict_proba(F_s)[:, 1]
        signal = 2 * proba - 1

        # ── Commodity shock filter ────────────────────────────────────────────
        shock = self._commodity_shock_mask(X.index)
        signal[shock] = 0.0

        return signal


def build_model():
    return DirectionModel(n_estimators=600, max_depth=2, min_samples_leaf=22,
                          train_window=756, wti_thresh=0.02, model_type="rf")


def build_feature_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Compute the same 6 rolling momentum/volatility features used inside
    DirectionModel, given a 1-D array of daily percentage returns.
    Returns an (n, 6) float64 array — one row per input observation.
    Used externally by the regime layer and plotting scripts.
    """
    feats = []
    for i in range(len(returns)):
        w5  = returns[max(0, i - 4): i + 1]
        w20 = returns[max(0, i - 19): i + 1]
        v5  = np.std(w5)  + 1e-8
        v20 = np.std(w20) + 1e-8
        feats.append([
            returns[i],
            returns[i] / v5,
            float(np.mean(w5)),
            float(np.mean(w20)),
            float(np.mean(w5)) / (float(np.mean(w20)) + 1e-8),
            v5 / v20,
        ])
    return np.array(feats, dtype=np.float64)
