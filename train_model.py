import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from model import build_model

def train_and_save():
    print("Downloading XLE data...")
    xle = yf.download("XLE", start="2019-01-01", auto_adjust=True, progress=False)
    
    if isinstance(xle.columns, pd.MultiIndex):
        xle.columns = xle.columns.get_level_values(0)
    
    # Build return series
    closes = xle["Close"].dropna()
    returns = closes.pct_change().dropna()
    
    # Features: lagged return (previous day predicts next day)
    ret_lag = returns.shift(1).dropna()
    ret_fwd = returns.loc[ret_lag.index]  # aligned forward return (label)
    
    print(f"Training on {len(ret_lag)} days of data...")
    
    model = build_model()
    model.fit(ret_lag.to_frame(), ret_fwd.values)
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("model.pkl saved successfully.")

if __name__ == "__main__":
    train_and_save()