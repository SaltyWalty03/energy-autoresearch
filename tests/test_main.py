import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import main


class APIError(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


class FakeTrading:
    def __init__(self, positions=None, equity=100_000.0, buying_power=200_000.0):
        self.positions = positions or {}
        self.account = SimpleNamespace(equity=equity, buying_power=buying_power)
        self.orders, self.closed = [], []

    def get_account(self):
        return self.account

    def get_open_position(self, ticker):
        if ticker not in self.positions:
            raise APIError('{"code":40410000,"message":"position does not exist"}', 404)
        return SimpleNamespace(qty=str(self.positions[ticker]))

    def close_position(self, ticker):
        self.closed.append(ticker)

    def submit_order(self, req):
        self.orders.append(req)
        return SimpleNamespace(id=f"order-{len(self.orders)}")


def _bias(direction, bias, wf=1.0, **extra):
    return {"bias": bias, "direction": direction, "regime": 0, "wf_sharpe_mean": wf, **extra}


@pytest.fixture
def run(monkeypatch):
    """Wire run_daily() to fakes; returns a callable taking biases + intraday signals."""
    def _run(biases, signals, trading, price=100.0):
        monkeypatch.setattr(main, "_is_market_open", lambda: True)
        monkeypatch.setattr(main, "_get_secret", lambda s: "x")
        monkeypatch.setattr(main.storage, "Client", lambda: object())
        monkeypatch.setattr(main, "_acquire_lock", lambda g, owner: True)
        monkeypatch.setattr(main, "_release_lock", lambda g: None)
        monkeypatch.setattr(main, "_check_pending_orders", lambda c: None)
        monkeypatch.setattr(main, "TradingClient", lambda *a, **k: trading)
        monkeypatch.setattr(main, "_load_regime_stack", lambda g, t: None)
        monkeypatch.setattr(main, "_get_all_daily_biases", lambda g, s: biases)
        bars = pd.DataFrame({"close": np.full(60, price)})
        monkeypatch.setattr(main, "_fetch_intraday_bars", lambda k, s, t: bars)
        monkeypatch.setattr(main, "_intraday_signal", lambda df, allowed: signals[allowed])
        body, status, _ = main.run_daily(None)
        return json.loads(body), status
    return _run


SIG = {"long": ("long", 40.0, 0.1), "short": ("flat", 50.0, 0.1), "flat": ("flat", 50.0, 0.0)}


def test_successful_cycle_returns_200(run):
    """Regression: response referenced undefined `scale` → NameError → HTTP 500 after orders were sent."""
    trading = FakeTrading()
    body, status = run({"XLE": _bias("long", 0.2), "XLB": _bias("flat", 0.0),
                        "XOP": _bias("long", 0.3, wf=-0.5)}, SIG, trading)
    assert status == 200, body
    assert body["assets"]["XLE"]["action"] == "buy"
    assert body["assets"]["XLE"]["qty"] == 800            # 0.80 * 100k / 1 signal / $100
    assert len(trading.orders) == 1


def test_reversal_orders_target_size_not_target_plus_old(run):
    """Regression: after close_position(), delta still netted the closed short → oversized buy."""
    trading = FakeTrading(positions={"XLE": -100})
    body, status = run({"XLE": _bias("long", 0.2), "XLB": _bias("flat", 0.0),
                        "XOP": _bias("flat", 0.0)}, SIG, trading)
    assert status == 200
    assert trading.closed == ["XLE"]
    assert trading.orders[0].qty == 800                    # was 900
    assert trading.orders[0].side == main.OrderSide.BUY


def test_every_skipped_asset_records_a_reason(run, capsys):
    trading = FakeTrading()
    body, _ = run({"XLE": _bias("flat", 0.01), "XLB": _bias("long", 0.2, wf=-0.1),
                   "XOP": _bias("short", -0.2)}, SIG, trading)
    a = body["assets"]
    assert a["XLE"]["reason"].startswith("flat_bias")
    assert a["XLB"]["reason"].startswith("no_edge")
    assert a["XOP"]["reason"].startswith("intraday not confirmed for short")
    out = capsys.readouterr().out
    assert "NO TRADE: XLE=skip(flat_bias" in out
    assert trading.orders == []


def test_zero_share_size_records_reason(run):
    trading = FakeTrading(equity=100.0, buying_power=100.0)
    body, _ = run({"XLE": _bias("long", 0.2), "XLB": _bias("flat", 0.0),
                   "XOP": _bias("flat", 0.0)}, SIG, trading, price=500.0)
    assert body["assets"]["XLE"]["action"] == "skipped"
    assert "size rounds to 0 sh" in body["assets"]["XLE"]["reason"]


def test_one_asset_bias_failure_does_not_block_others(monkeypatch, capsys):
    """Regression: a single predict() exception aborted the whole run for all assets."""
    saved = []
    monkeypatch.setattr(main, "_load_bias_cache", lambda g: {})
    monkeypatch.setattr(main, "_save_bias_cache", lambda g, d: saved.append(d))

    def fake_bias(gcs, ticker, *a):
        if ticker == "XLB":
            raise ValueError("X has 9 features, but StandardScaler is expecting 6 features as input.")
        return _bias("long", 0.2)
    monkeypatch.setattr(main, "_get_daily_bias_for_asset", fake_bias)

    out = main._get_all_daily_biases(None, {t: (None, {}, None, []) for t in main.ASSETS})
    assert out["XLE"]["direction"] == "long" and out["XOP"]["direction"] == "long"
    assert out["XLB"]["direction"] == "flat" and "9 features" in out["XLB"]["error"]
    assert saved == []                                     # don't cache a failed day
    assert "XLB: daily bias failed" in capsys.readouterr().out


def test_current_position_logs_unexpected_errors(capsys):
    class Broken:
        def get_open_position(self, t):
            raise APIError("forbidden", 403)
    assert main._current_position(Broken(), "XLE") == ("flat", 0)
    assert '"severity": "ERROR"' in capsys.readouterr().out


def test_current_position_404_is_flat_without_error(capsys):
    assert main._current_position(FakeTrading(), "XLE") == ("flat", 0)
    assert "ERROR" not in capsys.readouterr().out


def test_missing_gcs_json_is_logged(capsys):
    class Bucket:
        def blob(self, name):
            raise RuntimeError("404 not found")
    gcs = SimpleNamespace(bucket=lambda b: Bucket())
    assert main._gcs_load_json(gcs, "models/XLE_regime3_meta.json") == {}
    assert "models/XLE_regime3_meta.json unavailable" in capsys.readouterr().out
