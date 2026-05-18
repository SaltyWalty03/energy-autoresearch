"""
run_daily.py — Local one-shot paper-trading signal via Alpaca.

Loads model.pkl, generates today's XLE directional signal, sizes a position
using the signal magnitude × equity, and submits a market order to the Alpaca
paper account. Intended for manual runs or as a fallback when the Cloud
Function is unavailable.

Usage:
    python run_daily.py
"""
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import os
from dotenv import load_dotenv, find_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


load_dotenv(find_dotenv())

MAX_SHARES = 100

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

def generate_signal(model) -> float:
    # Fetch XLE returns
    xle = yf.download("XLE", period="60d", auto_adjust=True, progress=False)
    if isinstance(xle.columns, pd.MultiIndex):
        xle.columns = xle.columns.get_level_values(0)
    
    closes = xle["Close"].dropna()
    returns = closes.pct_change().dropna()
    ret_lag = returns.shift(1).dropna()

    # WTI shock check
    wti = yf.download("CL=F", period="10d", auto_adjust=True, progress=False)
    if isinstance(wti.columns, pd.MultiIndex):
        wti.columns = wti.columns.get_level_values(0)
    wti_ret = float(np.log(wti["Close"].iloc[-1] / wti["Close"].iloc[-2]))

    if abs(wti_ret) > 0.02:
        print(f"WTI shock detected ({wti_ret:.3f}) — going flat.")
        return 0.0

    signal = model.predict(ret_lag.to_frame())[-1]
    print(f"Signal: {signal:.4f}")
    return float(signal)

def risk_check(signal: float) -> float:
    if abs(signal) < 0.05:
        print("Signal in dead zone — no trade.")
        return 0.0
    return signal

def place_order(signal: float):
    client = TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"],
        paper=True
    )

    # Close existing XLE position
    try:
        client.close_position("XLE")
        print("Closed existing XLE position.")
    except Exception:
        pass

    if signal == 0.0:
        print("Flat — no order placed.")
        return

    data_client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"]
    )
    quote = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols="XLE"))
    price = float(quote["XLE"].ask_price)

    equity = float(client.get_account().equity)
    kelly_f = abs(signal) * 0.5
    shares = min(int(kelly_f * equity / price), MAX_SHARES)

    if shares < 1:
        print("Signal too weak for even 1 share — skipping.")
        return

    side = OrderSide.BUY if signal > 0 else OrderSide.SELL
    order = MarketOrderRequest(
        symbol="XLE",
        qty=shares,
        side=side,
        time_in_force=TimeInForce.DAY
    )
    client.submit_order(order)
    print(f"[{side}] {shares} shares XLE @ ~${price:.2f} | signal={signal:.3f}  equity=${equity:,.2f}  kelly_f={kelly_f:.3f}")

if __name__ == "__main__":
    model = load_model()
    signal = generate_signal(model)
    signal = risk_check(signal)
    place_order(signal)