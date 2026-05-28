"""
src/build_results_table.py — Aggregate experiment JSON logs into results tables.

Reads all experiments/*.json files, produces:
  - results/final_results.csv
  - results/final_results.tex  (LaTeX booktabs table)

Usage:
    python src/build_results_table.py
"""
import sys
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_all_experiments():
    records = []
    for path in sorted(EXPERIMENTS_DIR.glob("exp_*.json")):
        with open(path) as f:
            rec = json.load(f)
        hp = rec.get("hyperparameters", {})
        abl = rec.get("ablation_flags", {})
        ds = rec.get("dataset", {})
        m = rec.get("metrics", {})
        records.append({
            "experiment_id": rec["experiment_id"],
            "model": _model_name(rec),
            "dataset": ds.get("ticker", "XLE"),
            "train_size": ds.get("train_size", ""),
            "val_size": ds.get("val_size", ""),
            "split_date": ds.get("split_date", ""),
            "n_estimators": hp.get("n_estimators", "N/A"),
            "max_depth": hp.get("max_depth", "N/A"),
            "min_samples_leaf": hp.get("min_samples_leaf", "N/A"),
            "train_window": hp.get("train_window", "N/A"),
            "wti_thresh": hp.get("wti_thresh", "N/A"),
            "shock_filter": abl.get("shock_filter", False),
            "rolling_features": abl.get("rolling_features", False),
            "sample_weights": abl.get("sample_weights", False),
            "val_sharpe": m.get("val_sharpe", 0.0),
            "wf_sharpe_mean": m.get("wf_sharpe_mean", 0.0),
            "wf_sharpe_std": m.get("wf_sharpe_std", 0.0),
            "runtime_seconds": rec.get("runtime_seconds", 0.0),
            "description": rec.get("description", ""),
        })
    return pd.DataFrame(records)


def _model_name(rec):
    hp = rec.get("hyperparameters", {})
    mt = hp.get("model_type", "rf")
    if mt == "LinearRegression":
        return "LinearRegression"
    d = hp.get("max_depth", "?")
    n = hp.get("n_estimators", "?")
    msl = hp.get("min_samples_leaf", "?")
    thresh = hp.get("wti_thresh", 99.0)
    shock = "" if float(thresh) >= 1.0 else f"+ShockFilter({thresh})"
    return f"RF(n={n},d={d},msl={msl}){shock}"


def write_csv(df):
    out = RESULTS_DIR / "final_results.csv"
    cols = [
        "experiment_id", "model", "dataset", "train_size", "val_size", "split_date",
        "n_estimators", "max_depth", "min_samples_leaf", "train_window", "wti_thresh",
        "shock_filter", "rolling_features", "sample_weights",
        "val_sharpe", "wf_sharpe_mean", "wf_sharpe_std", "runtime_seconds", "description",
    ]
    df[cols].to_csv(out, index=False)
    print(f"CSV  -> {out}")


def write_latex(df):
    out = RESULTS_DIR / "final_results.tex"

    display_df = df[[
        "experiment_id", "model", "val_sharpe", "wf_sharpe_mean",
        "shock_filter", "rolling_features", "train_window",
    ]].copy()

    display_df.columns = [
        "ID", "Model / Configuration", "Val Sharpe", "WF Sharpe",
        "Shock Filter", "Roll. Feat.", "Train Window",
    ]
    display_df = display_df.sort_values("Val Sharpe", ascending=False).reset_index(drop=True)
    display_df["Val Sharpe"] = display_df["Val Sharpe"].map("{:.4f}".format)
    display_df["WF Sharpe"] = display_df["WF Sharpe"].map("{:.4f}".format)
    display_df["Shock Filter"] = display_df["Shock Filter"].map({True: "\\checkmark", False: "—"})
    display_df["Roll. Feat."] = display_df["Roll. Feat."].map({True: "\\checkmark", False: "—"})

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Model comparison on XLE Energy ETF direction prediction.")
    lines.append(r"Val Sharpe is annualised on the 20\% chronological holdout.")
    lines.append(r"WF Sharpe is the mean walk-forward Sharpe over 3 folds.")
    lines.append(r"Train window is the number of trading days used for fitting.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrccr}")
    lines.append(r"\toprule")
    lines.append(r"ID & Model & Val Sharpe & WF Sharpe & Shock Filter & Roll.\ Feat. & Train Win. \\")
    lines.append(r"\midrule")

    for _, row in display_df.iterrows():
        model_safe = row["Model / Configuration"].replace("_", r"\_").replace("%", r"\%")
        line = (
            f"{row['ID']} & "
            f"\\texttt{{{model_safe}}} & "
            f"{row['Val Sharpe']} & "
            f"{row['WF Sharpe']} & "
            f"{row['Shock Filter']} & "
            f"{row['Roll. Feat.']} & "
            f"{row['Train Window']} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out.write_text(tex, encoding="utf-8")
    print(f"LaTeX -> {out}")
    return tex


def main():
    df = load_all_experiments()
    if df.empty:
        print("No experiment JSON files found in experiments/. Run src/run_experiments.py first.")
        sys.exit(1)

    print(f"\nLoaded {len(df)} experiments.\n")
    print(df[["experiment_id", "model", "val_sharpe", "wf_sharpe_mean"]].to_string(index=False))
    print()

    write_csv(df)
    write_latex(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
