import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class DirectionModel(BaseEstimator, RegressorMixin):
    def __init__(self, window=20, n_estimators=300, max_depth=2, min_samples_leaf=20):
        self.window = window
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def _rolling_features(self, r, start_idx):
        feats = []
        for i in range(start_idx, len(r)):
            w5 = r[max(0, i - 4):i + 1]
            w20 = r[max(0, i - 19):i + 1]
            vol5 = np.std(w5) + 1e-8
            vol20 = np.std(w20) + 1e-8
            feats.append([
                r[i],
                r[i] / vol5,
                np.mean(w5),
                np.mean(w20),
                np.mean(w5) / (np.mean(w20) + 1e-8),
                vol5 / vol20,
            ])
        return np.array(feats)

    def fit(self, X, y):
        r = np.array(X).ravel()
        self._tail = r[-self.window:]
        F = self._rolling_features(r, 0)
        labels = (np.array(y) > 0).astype(int)
        self.scaler_ = StandardScaler()
        F_s = self.scaler_.fit_transform(F)
        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features='sqrt',
            random_state=42,
        )
        self.clf_.fit(F_s, labels)
        return self

    def predict(self, X):
        r = np.concatenate([self._tail, np.array(X).ravel()])
        F = self._rolling_features(r, len(self._tail))
        F_s = self.scaler_.transform(F)
        proba = self.clf_.predict_proba(F_s)[:, 1]
        return 2 * proba - 1


def build_model():
    return DirectionModel(n_estimators=200, max_depth=2, min_samples_leaf=22)
