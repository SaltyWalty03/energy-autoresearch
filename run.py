"""
run.py — Single experiment runner for the XLE direction model.

Loads data, fits the model defined in model.py, evaluates on the validation
set, and appends the result to results.tsv.

Usage:
    python run.py "description of this experiment"
"""
import sys
import datetime
from prepare import load_and_split_data, evaluate
from model import build_model

description = sys.argv[1] if len(sys.argv) > 1 else "unnamed"

train, val = load_and_split_data()
X_train, y_train = train[['ret_lag']], train['target']
X_val, y_val = val[['ret_lag']], val['target']

model = build_model()
model.fit(X_train, y_train)
preds = model.predict(X_val)
m = evaluate(y_val, preds)

with open("results.tsv", "a") as f:
    f.write(f"{datetime.datetime.now()}\t{description}\t{m['sharpe']}\n")

print(f"Sharpe Ratio: {m['sharpe']}")
