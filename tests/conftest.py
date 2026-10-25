"""
Test setup for cloud_function/.

Puts cloud_function/ first on sys.path (so `import main` / `import model` resolve
to the deployed code, not the repo-root research copies) and stubs the
Cloud-only SDKs that are not installed locally.
"""
import enum
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CF_DIR = ROOT / "cloud_function"
sys.path.insert(0, str(CF_DIR))


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _cls(name):
    return type(name, (), {"__init__": lambda self, *a, **k: None})


try:
    import functions_framework  # noqa: F401
except ImportError:
    _stub("functions_framework", http=lambda f: f)

try:
    import alpaca  # noqa: F401
except ImportError:
    class OrderSide(enum.Enum):
        BUY = "buy"
        SELL = "sell"

    class TimeInForce(enum.Enum):
        DAY = "day"

    class MarketOrderRequest:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _stub("alpaca")
    _stub("alpaca.data")
    _stub("alpaca.data.historical", StockHistoricalDataClient=_cls("StockHistoricalDataClient"))
    _stub("alpaca.data.requests", StockBarsRequest=_cls("StockBarsRequest"))
    _stub("alpaca.data.timeframe", TimeFrame=_cls("TimeFrame"),
          TimeFrameUnit=types.SimpleNamespace(Minute="Minute"))
    _stub("alpaca.trading")
    _stub("alpaca.trading.client", TradingClient=_cls("TradingClient"))
    _stub("alpaca.trading.enums", OrderSide=OrderSide, TimeInForce=TimeInForce)
    _stub("alpaca.trading.requests", GetOrdersRequest=_cls("GetOrdersRequest"),
          MarketOrderRequest=MarketOrderRequest)

try:
    from google.cloud import secretmanager, storage  # noqa: F401
except ImportError:
    sm = _stub("google.cloud.secretmanager",
               SecretManagerServiceClient=_cls("SecretManagerServiceClient"))
    st = _stub("google.cloud.storage", Client=_cls("Client"))
    gc = sys.modules.get("google.cloud") or _stub("google.cloud")
    gc.secretmanager, gc.storage = sm, st
    if "google" in sys.modules:
        sys.modules["google"].cloud = gc
