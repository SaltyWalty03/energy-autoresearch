"""
run.py — Single experiment runner for the XLE direction model.

Loads data, fits the model defined in model.py, evaluates on the validation
set, and appends the result to results.tsv.

Usage:
    python run.py "description of this experiment"
"""
import sys
import datetime
import pandas as pd
from prepare import load_and_split_data, evaluate, walk_forward_sharpe
from model import build_model

description = sys.argv[1] if len(sys.argv) > 1 else "unnamed"

train, val = load_and_split_data()
X_train, y_train = train[['ret_lag']], train['target']
X_val, y_val = val[['ret_lag']], val['target']

model = build_model()
model.fit(X_train, y_train)
preds = model.predict(X_val)
m = evaluate(y_val, preds)

# Walk-forward Sharpe on the full dataset (train + val combined)
full_data = pd.concat([train, val])
wf_mean, wf_std = walk_forward_sharpe(full_data, build_model(), n_folds=4)

with open("results.tsv", "a") as f:
    f.write(f"{datetime.datetime.now()}\t{description}\t{m['sharpe']}\t{wf_mean}\t{wf_std}\n")

print(f"Val Sharpe:          {m['sharpe']}")
print(f"Walk-Forward Sharpe: mean={wf_mean}  std={wf_std}")
