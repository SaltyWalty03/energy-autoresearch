import logging
import pickle

import numpy as np
import pandas as pd
import pytest

import model as M
from conftest import ROOT

LEGACY_PICKLE = ROOT / "models" / "XLE_regime9_model.pkl"   # fitted 2026-05-18, 6 features


@pytest.fixture
def offline(monkeypatch):
    """No network: synthetic WTI prices, no weather/NatGas data."""
    idx = pd.bdate_range("2019-01-01", "2026-09-30")
    rng = np.random.default_rng(0)
    wti = pd.Series(70 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    monkeypatch.setattr(M, "_load_commodity", lambda ticker="CL=F": wti)
    monkeypatch.setattr(M.DirectionModel, "_temp_anomaly", lambda self, i: rng.normal(0, 3, len(i)))
    monkeypatch.setattr(M.DirectionModel, "_natgas_zscore", lambda self, i: rng.normal(0, 1, len(i)))


def _returns(n, start="2023-01-02", seed=1):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"ret_lag": np.random.default_rng(seed).normal(0, 0.015, n)}, index=idx)


@pytest.mark.skipif(not LEGACY_PICKLE.exists(), reason="legacy pickle not present")
def test_legacy_6_feature_pickle_predicts(offline):
    """Regression: 9-feature predict() crashed on every pre-update pickle (Cloud Run 500s since 2026-09-15)."""
    with open(LEGACY_PICKLE, "rb") as fh:
        mdl = pickle.load(fh)
    out = mdl.predict(_returns(60, start="2026-06-01"))
    assert mdl.scaler_.n_features_in_ == 6
    assert out.shape == (60,)
    assert np.isfinite(out).all() and (np.abs(out) <= 1).all()


def test_extended_9_feature_model_predicts(offline):
    X = _returns(300)
    y = X["ret_lag"].shift(-1).fillna(0).values
    mdl = M.DirectionModel(n_estimators=20, train_window=250).fit(X, y)
    assert mdl.n_features_ == M.N_EXTENDED_FEATURES
    out = mdl.predict(_returns(30, start="2024-03-01", seed=2))
    assert out.shape == (30,) and np.isfinite(out).all()


def test_unknown_feature_count_raises_clear_error(offline):
    X = _returns(300)
    mdl = M.DirectionModel(n_estimators=20, train_window=250).fit(X, X["ret_lag"].values)
    mdl.scaler_.n_features_in_ = 7
    with pytest.raises(ValueError, match="fitted on 7 features"):
        mdl.predict(_returns(30, start="2024-03-01"))


def test_commodity_cache_write_failure_keeps_data(monkeypatch, tmp_path, caplog):
    """Regression: without pyarrow, to_parquet raised and the downloaded prices were discarded."""
    idx = pd.bdate_range("2026-01-01", periods=50)
    raw = pd.DataFrame({"Close": np.linspace(60, 70, 50)}, index=idx)
    monkeypatch.setattr(M, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(M.yf, "download", lambda *a, **k: raw.copy())

    def boom(*a, **k):
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow'")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)

    with caplog.at_level(logging.WARNING, logger=M.log.name):
        s = M._load_commodity("HG=F")
    assert len(s) == 50
    assert "cache write failed" in caplog.text


def test_commodity_download_failure_is_logged(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(M, "_CACHE_DIR", tmp_path)

    def boom(*a, **k):
        raise ConnectionError("no network")
    monkeypatch.setattr(M.yf, "download", boom)
    with caplog.at_level(logging.WARNING, logger=M.log.name):
        s = M._load_commodity("HG=F")
    assert s.empty
    assert "download failed" in caplog.text
