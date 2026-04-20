import yfinance as yf
import pandas as pd
import numpy as np

def load_and_split_data(ticker="XLE"):
    data = yf.download(ticker, start="2020-01-01", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Target: Next day log return
    data['target'] = np.log(data['Close'] / data['Close'].shift(1)).shift(-1)
    data['ret_lag'] = data['Close'].pct_change()
    data = data.dropna()

    split = int(len(data) * 0.8)
    return data.iloc[:split], data.iloc[split:]

def evaluate(y_true, y_pred):
    rets = np.sign(y_pred) * y_true
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() != 0 else 0
    return {"sharpe": round(sharpe, 4)}