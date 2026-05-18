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
_WTI_CACHE  = _CACHE_DIR / "wti_daily.parquet"


def _load_wti():
    """
    Load WTI crude spot from EIA-sourced cache (covers 2020-2026).
    Falls back to yfinance CL=F if cache is missing.
    """
    # Prefer the fresh EIA parquet saved in data/eia_cache/
    eia_dir = Path(__file__).parent / "data" / "eia_cache"
    eia_files = sorted(eia_dir.glob("eia_wti_price_*.parquet"),
                       key=lambda f: f.stat().st_size, reverse=True)
    if eia_files:
        df = pd.read_parquet(eia_files[0])
        s = df.set_index("date")["value"]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    # Fallback: yfinance CL=F (requires internet)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _WTI_CACHE.exists() and (time.time() - _WTI_CACHE.stat().st_mtime) < 82800:
        return pd.read_parquet(_WTI_CACHE)["close"]
    try:
        raw = yf.download("CL=F", start="2019-01-01", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw[["Close"]].rename(columns={"Close": "close"}).to_parquet(_WTI_CACHE)
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
                 model_type="xgb", learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8):
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

    def _wti_shock_mask(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """
        Returns a boolean array (len(idx)) that is True on days where
        yesterday's WTI |return| > WTI_SHOCK_THRESH.  Falls back to all-False
        if WTI data is unavailable, so the model degrades gracefully.
        """
        wti = _load_wti()
        if len(wti) == 0:
            return np.zeros(len(idx), dtype=bool)
        wti_ret = np.log(wti + 1e-8).diff(1).shift(1)   # yesterday's return
        shock = wti_ret.abs() > self.wti_thresh
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
        F      = self._rolling_features(r, 0)
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
        r   = np.concatenate([self._tail, np.array(X).ravel()])
        F   = self._rolling_features(r, len(self._tail))
        F_s = self.scaler_.transform(F)
        proba = self.clf_.predict_proba(F_s)[:, 1]
        signal = 2 * proba - 1

        # ── WTI shock filter ──────────────────────────────────────────────────
        shock = self._wti_shock_mask(X.index)
        signal[shock] = 0.0

        return signal


def build_model():
    return DirectionModel(n_estimators=600, max_depth=2, min_samples_leaf=22,
                          train_window=756, wti_thresh=0.02, model_type="rf")
