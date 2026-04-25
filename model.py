import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class DirectionModel(BaseEstimator, RegressorMixin):
    """
    RF direction classifier with two improvements over the prior best:

    1. 3-year rolling training window  — excludes COVID-era dynamics (2020-2021)
       that hurt generalisation to the current 2025-2026 regime.
    2. Exponential sample-decay weights  — gently upweights recent data
       (weight range ~0.55→1.0) so the model is more adapted to the
       market regime just before the validation period.

    Core features (6) are unchanged: ret, ret_adj5, mom5, mom20, macd, vol_ratio.
    """

    TRAIN_WINDOW = 756   # ~3 years of trading days
    DECAY_LINSPACE_RANGE = (-1.0, 0.0)  # applied over len(full_train), sliced to window

    def __init__(self, window=20, n_estimators=300, max_depth=2, min_samples_leaf=22):
        self.window = window
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

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

    def fit(self, X, y):
        r_full  = np.array(X).ravel()
        y_full  = np.array(y)
        n_full  = len(r_full)

        # ── 3-year rolling window ──────────────────────────────────────────────
        cutoff  = max(0, n_full - self.TRAIN_WINDOW)
        r       = r_full[cutoff:]
        y_w     = y_full[cutoff:]

        # Store tail from the FULL series (correct history for predict)
        self._tail = r_full[-self.window:]

        # ── Exponential sample-decay weights ──────────────────────────────────
        # Build weights over the full training series, then slice to [cutoff:].
        # This produces a gentle 0.55→1.0 taper rather than a steep 0.37→1.0.
        lo, hi = self.DECAY_LINSPACE_RANGE
        weights = np.exp(np.linspace(lo, hi, n_full))[cutoff:]

        # ── Feature matrix & classifier ───────────────────────────────────────
        F      = self._rolling_features(r, 0)
        labels = (y_w > 0).astype(int)

        self.scaler_ = StandardScaler()
        F_s = self.scaler_.fit_transform(F)

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
        return 2 * proba - 1


def build_model():
    return DirectionModel(n_estimators=300, max_depth=2, min_samples_leaf=22)
