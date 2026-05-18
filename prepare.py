import yfinance as yf
import pandas as pd
import numpy as np
import copy

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

def _fold_sharpe(y_true, y_pred):
    rets = np.sign(y_pred) * np.array(y_true)
    if rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * np.sqrt(252))

def walk_forward_sharpe(returns: pd.DataFrame, model, n_folds: int = 4) -> tuple[float, float]:
    """
    Purged walk-forward Sharpe across n_folds chronological folds.

    Each fold trains on all prior data, tests on the next fold.
    The model is deep-copied per fold so no state bleeds between folds.
    Returns (mean_sharpe, std_sharpe) across the n_folds-1 test folds.

    `returns` must be a DataFrame with columns ['ret_lag', 'target'],
    indexed by date, in chronological order — the same format produced
    by load_and_split_data() concatenated back together.
    """
    n = len(returns)
    fold_size = n // n_folds
    sharpes = []

    for fold in range(1, n_folds):
        train_end = fold * fold_size
        test_end  = min((fold + 1) * fold_size, n)

        train_block = returns.iloc[:train_end]
        test_block  = returns.iloc[train_end:test_end]

        if len(test_block) < 10:
            continue

        X_tr = train_block[["ret_lag"]]
        y_tr = train_block["target"]
        X_te = test_block[["ret_lag"]]
        y_te = test_block["target"]

        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        sharpes.append(_fold_sharpe(y_te, preds))

    if not sharpes:
        return 0.0, 0.0
    arr = np.array(sharpes)
    return round(float(arr.mean()), 4), round(float(arr.std()), 4)