import sys, datetime
from prepare import load_and_split_data, evaluate
from model import build_model

train, val = load_and_split_data()
X_train, y_train = train[['ret_lag']], train['target']
X_val, y_val = val[['ret_lag']], val['target']

model = build_model()
model.fit(X_train, y_train)
preds = model.predict(X_val)
m = evaluate(y_val, preds)

with open("results.tsv", "a") as f:
    f.write(f"{datetime.datetime.now()}\tBaseline\t{m['sharpe']}\n")

print(f"Sharpe Ratio: {m['sharpe']}")