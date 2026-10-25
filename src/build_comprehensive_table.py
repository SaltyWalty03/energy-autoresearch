"""
src/build_comprehensive_table.py — Aggregate all experiment sources into one table.

Sources:
  1. results.tsv            — 53 timestamped runs (sessions 1-3 + post-cleanup)
  2. experiment_log.csv     — 15 hyperparameter sweep runs
  3. experiments/exp_*.json — 7 structured ablation experiments (Phase 2)
  4. HARDCODED              — 10 "Detailed Session 1" experiments (2026-04-20, markdown only)

Outputs:
  results/all_experiments.csv
  results/all_experiments.tex   (longtable, submission-ready)
  results/all_experiments.pdf   (compiled via pdflatex)
"""

import sys, json, csv, io, subprocess, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── 1. results.tsv annotations ─────────────────────────────────────────────
# Maps row index (1-based) → human description, session label, and status.
# "Baseline" was stored in all early runs; we override with known descriptions.
TSV_ANNOTATIONS = {
    1:  ("Pre-session smoke test",          "S1", "smoke"),
    2:  ("Pre-session smoke test",          "S1", "smoke"),
    3:  ("Pre-session smoke test",          "S1", "smoke"),
    4:  ("LinearRegression baseline",       "S1", "reference"),
    5:  ("RF: RetFeatures + Ridge",         "S1", "discard"),
    6:  ("GradientBoostingRegressor",       "S1", "discard"),
    7:  ("Rolling z/RSI/vol + Ridge",       "S1", "discard"),
    8:  ("Rolling feat + LogisticReg",      "S1", "keep"),
    9:  ("LogReg + lag2/lag3/MACD/RSI",     "S1", "discard"),
    10: ("RF depth=3, msl=30",             "S1", "keep"),
    11: ("RF d=2, msl=25, sqrt, n=200",    "S1", "keep"),
    12: ("MLPClassifier (128,) alpha=1",    "S1", "discard"),
    13: ("RF n=300, d=2, msl=20",          "S1", "keep"),
    14: ("RF n=200, d=2, msl=22",          "S1", "best-s1"),
    15: ("Direct Sharpe optimizer (tanh)",  "S1", "discard"),
    16: ("Restored RF n=200, d=2, msl=22", "S1", "keep"),
    17: ("Session 2 start",                "S2", "smoke"),
    18: ("S2 intermediate",                "S2", "discard"),
    19: ("S2 intermediate",                "S2", "discard"),
    20: ("S2 intermediate (+roll window)", "S2", "keep"),
    21: ("WTI shock filter thresh=3%",     "S2", "keep"),
    22: ("WTI shock filter thresh=2%",     "S2", "best-committed"),
    23: ("WTI shock filter thresh=1.5%",   "S2", "discard"),
    24: ("WTI ret1+mom5 as RF inputs",     "S2", "discard"),
    25: ("Contrarian signal on shock days","S2", "discard"),
    26: ("S3 early probe",                 "S3", "smoke"),
    27: ("S3 early probe",                 "S3", "smoke"),
    28: ("S3 run",                         "S3", "keep"),
    29: ("S3 run",                         "S3", "discard"),
    30: ("S3 run",                         "S3", "discard"),
    31: ("S3 run",                         "S3", "discard"),
    32: ("S3 run",                         "S3", "discard"),
    33: ("S3 run",                         "S3", "keep"),
    34: ("S3 run",                         "S3", "discard"),
    35: ("S3 run",                         "S3", "discard"),
    36: ("S3 run",                         "S3", "discard"),
    37: ("S3 run",                         "S3", "keep"),
    38: ("S3 run",                         "S3", "discard"),
    39: ("S3 run",                         "S3", "keep"),
    40: ("S3 run",                         "S3", "keep"),
    41: ("S3 peak: msl tune",              "S3", "best-s3"),
    42: ("S3 run",                         "S3", "discard"),
    43: ("S3 run",                         "S3", "keep"),
    44: ("S3 run",                         "S3", "discard"),
    45: ("S3 run",                         "S3", "keep"),
    46: ("S3 run",                         "S3", "keep"),
    47: ("S3 run",                         "S3", "keep"),
    48: ("S3 run",                         "S3", "keep"),
    49: ("S3 run",                         "S3", "discard"),
    50: ("Post-cleanup smoke test",        "Post", "smoke"),
    51: ("XGBoost baseline",               "Post", "discard"),
    52: ("XGBoost d=2, n=800",             "Post", "discard"),
    53: ("RF baseline + walk-forward",     "Post", "keep"),
}

# ─── 2. Detailed Session 1 experiments (2026-04-20, markdown only) ──────────
DETAILED_S1 = [
    {"run_id": "det_001", "session": "Det-S1", "timestamp": "2026-04-20",
     "description": "OLS on 24 alternative-data features",
     "model": "OLS (statsmodels)", "n_estimators": "—", "max_depth": "—",
     "min_samples_leaf": "—", "train_window": "full", "wti_thresh": "—",
     "val_sharpe": -0.739, "wf_sharpe": "—", "status": "reference",
     "notes": "IS Sharpe 1.007; most features p>0.10"},
    {"run_id": "det_002", "session": "Det-S1", "timestamp": "2026-04-20",
     "description": "OLS selected+engineered (10 features, trend/lag)",
     "model": "OLS (selected)", "n_estimators": "—", "max_depth": "—",
     "min_samples_leaf": "—", "train_window": "full", "wti_thresh": "—",
     "val_sharpe": 0.039, "wf_sharpe": "—", "status": "discard",
     "notes": "IS Sharpe 1.218; OOS near-zero"},
    {"run_id": "det_003", "session": "Det-S1", "timestamp": "2026-04-20",
     "description": "Ridge / Lasso with cross-validated alpha",
     "model": "Ridge/Lasso", "n_estimators": "—", "max_depth": "—",
     "min_samples_leaf": "—", "train_window": "full", "wti_thresh": "—",
     "val_sharpe": 0.613, "wf_sharpe": "—", "status": "keep",
     "notes": "Lasso zeroed most features at alpha=0.1"},
    {"run_id": "det_004", "session": "Det-S1", "timestamp": "2026-04-20",
     "description": "RandomForest n=300, d=4, msl=15 (regressor)",
     "model": "RF Regressor", "n_estimators": 300, "max_depth": 4,
     "min_samples_leaf": 15, "train_window": "full", "wti_thresh": "—",
     "val_sharpe": -0.705, "wf_sharpe": "—", "status": "discard",
     "notes": "IS Sharpe 4.652 — severe overfit"},
    {"run_id": "det_006", "session": "Det-S1", "timestamp": "2026-04-20",
     "description": "Lasso + signal threshold sweep; 5-fold walk-forward",
     "model": "Lasso+threshold", "n_estimators": "—", "max_depth": "—",
     "min_samples_leaf": "—", "train_window": "full", "wti_thresh": "—",
     "val_sharpe": 0.613, "wf_sharpe": -0.2405, "status": "keep",
     "notes": "WF=-0.24 reveals fragility"},
    {"run_id": "det_007", "session": "Det-S1", "timestamp": "2026-04-20",
     "description": "Signal audit: 15 non-financial signals individually",
     "model": "Individual signals", "n_estimators": "—", "max_depth": "—",
     "min_samples_leaf": "—", "train_window": "full", "wti_thresh": "—",
     "val_sharpe": 1.687, "wf_sharpe": "—", "status": "keep",
     "notes": "Best: trend_oil_price_4w_mom (p=0.002)"},
    {"run_id": "det_008", "session": "Det-S1", "timestamp": "2026-04-20",
     "description": "Normalised signal composite (EW + OLS-weighted)",
     "model": "Signal composite", "n_estimators": "—", "max_depth": "—",
     "min_samples_leaf": "—", "train_window": "full", "wti_thresh": "—",
     "val_sharpe": 0.663, "wf_sharpe": "—", "status": "keep",
     "notes": "5 signals combined"},
    {"run_id": "det_009", "session": "Det-S1", "timestamp": "2026-04-20",
     "description": "Lasso fine alpha grid (40-pt, 1e-5 to 0.1) on top signals",
     "model": "Lasso (fine grid)", "n_estimators": "—", "max_depth": "—",
     "min_samples_leaf": "—", "train_window": "full", "wti_thresh": "—",
     "val_sharpe": 0.613, "wf_sharpe": "—", "status": "keep",
     "notes": "Fixes alpha=0.1 collapse in det_003"},
    {"run_id": "det_010", "session": "Det-S1", "timestamp": "2026-04-20",
     "description": "Walk-forward: best signal + Lasso ensemble (5-fold)",
     "model": "Signal+Lasso WF", "n_estimators": "—", "max_depth": "—",
     "min_samples_leaf": "—", "train_window": "full", "wti_thresh": "—",
     "val_sharpe": 1.042, "wf_sharpe": 0.5733, "status": "keep",
     "notes": "Ensemble of trend signal + Lasso"},
]


def load_tsv():
    rows = []
    path = ROOT / "results.tsv"
    raw = path.read_bytes()
    for enc in ("utf-16", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = raw.decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    for i, line in enumerate(text.strip().split("\n"), 1):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ts, desc_raw, sharpe_str = parts[0], parts[1], parts[2]
        wf_mean = float(parts[3]) if len(parts) > 3 else "—"
        ann = TSV_ANNOTATIONS.get(i, (desc_raw, "unknown", "unknown"))
        rows.append({
            "run_id": f"tsv_{i:03d}",
            "session": ann[1],
            "timestamp": ts.split(".")[0],
            "description": ann[0],
            "model": _infer_model(ann[0], ann[2]),
            "n_estimators": _infer_hp(ann[0], "n"),
            "max_depth": _infer_hp(ann[0], "d"),
            "min_samples_leaf": _infer_hp(ann[0], "msl"),
            "train_window": 756,
            "wti_thresh": _infer_wti(ann[0]),
            "val_sharpe": float(sharpe_str),
            "wf_sharpe": wf_mean,
            "status": ann[2],
            "notes": "",
        })
    return rows


def _infer_model(desc, status):
    d = desc.lower()
    if "linearreg" in d or "ols" in d: return "LinearRegression"
    if "lasso" in d: return "Lasso"
    if "ridge" in d: return "Ridge"
    if "logistic" in d: return "LogisticReg"
    if "mlp" in d: return "MLPClassifier"
    if "gradient" in d: return "GradientBoosting"
    if "xgb" in d: return "XGBoost"
    if "rf" in d or "random forest" in d or "restored" in d: return "RandomForest"
    if "signal" in d or "composite" in d: return "SignalComposite"
    if "direct sharpe" in d or "tanh" in d: return "SharpeOptimizer"
    return "RandomForest"


def _infer_hp(desc, param):
    import re
    if param == "n":
        m = re.search(r'n=(\d+)', desc)
        return int(m.group(1)) if m else "—"
    if param == "d":
        m = re.search(r'd=(\d+)', desc) or re.search(r'depth=(\d+)', desc)
        return int(m.group(1)) if m else "—"
    if param == "msl":
        m = re.search(r'msl=(\d+)', desc)
        return int(m.group(1)) if m else "—"
    return "—"


def _infer_wti(desc):
    import re
    m = re.search(r'thresh=([0-9.]+%)', desc)
    if m:
        pct = m.group(1).replace("%", "")
        return float(pct) / 100
    if "2%" in desc or "thresh=2" in desc:
        return 0.02
    return "—"


def load_sweep_csv():
    rows = []
    path = ROOT / "experiment_log.csv"
    import json as _json
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = _json.loads(row["config"])
            rows.append({
                "run_id": row["run_id"],
                "session": "Sweep",
                "timestamp": "2026-04-25",
                "description": f"HP sweep: n={cfg['n_estimators']}, d={cfg['max_depth']}, "
                               f"msl={cfg['min_samples_leaf']}, win={cfg['window']}, "
                               f"wti={cfg['wti_thresh']}, tw={cfg['train_window']}",
                "model": "RandomForest",
                "n_estimators": cfg["n_estimators"],
                "max_depth": cfg["max_depth"],
                "min_samples_leaf": cfg["min_samples_leaf"],
                "train_window": cfg["train_window"],
                "wti_thresh": cfg["wti_thresh"],
                "val_sharpe": float(row["final_metric"]),
                "wf_sharpe": "—",
                "status": row["status"].lower(),
                "notes": row["notes"],
            })
    return rows


def load_json_ablations():
    rows = []
    for path in sorted((ROOT / "experiments").glob("exp_*.json")):
        with open(path) as f:
            rec = json.load(f)
        hp = rec.get("hyperparameters", {})
        m = rec.get("metrics", {})
        rows.append({
            "run_id": rec["experiment_id"],
            "session": "Ablation",
            "timestamp": rec["timestamp"][:19],
            "description": rec["description"],
            "model": hp.get("model_type", "RandomForest").replace("LinearRegression", "LinearRegression"),
            "n_estimators": hp.get("n_estimators", "—"),
            "max_depth": hp.get("max_depth", "—"),
            "min_samples_leaf": hp.get("min_samples_leaf", "—"),
            "train_window": hp.get("train_window", "—"),
            "wti_thresh": hp.get("wti_thresh", "—"),
            "val_sharpe": m.get("val_sharpe", 0.0),
            "wf_sharpe": m.get("wf_sharpe_mean", "—"),
            "status": "ablation",
            "notes": rec.get("description", ""),
        })
    return rows


STATUS_ORDER = {
    "best-committed": 0, "best-s1": 1, "best-s3": 2,
    "keep": 3, "ablation": 4, "reference": 5,
    "discard": 6, "smoke": 7, "unknown": 8,
}

SESSION_ORDER = {
    "Det-S1": 0, "S1": 1, "S2": 2, "Sweep": 3, "S3": 4, "Post": 5, "Ablation": 6,
}


def build_dataframe(all_rows):
    df = pd.DataFrame(all_rows)
    df["_session_ord"] = df["session"].map(SESSION_ORDER).fillna(99)
    df["_status_ord"] = df["status"].map(STATUS_ORDER).fillna(99)
    df = df.sort_values(["_session_ord", "timestamp"]).reset_index(drop=True)
    df = df.drop(columns=["_session_ord", "_status_ord"])
    return df


def write_csv(df):
    out = RESULTS_DIR / "all_experiments.csv"
    cols = ["run_id", "session", "timestamp", "description", "model",
            "n_estimators", "max_depth", "min_samples_leaf", "train_window",
            "wti_thresh", "val_sharpe", "wf_sharpe", "status", "notes"]
    df[cols].to_csv(out, index=False)
    print(f"CSV -> {out}  ({len(df)} rows)")
    return out


def _fmt(v, fmt=".4f"):
    try:
        return format(float(v), fmt)
    except (ValueError, TypeError):
        return str(v)


STATUS_LABEL = {
    "best-committed": r"\textbf{best}",
    "best-s1": r"\textbf{best-S1}",
    "best-s3": r"\textbf{best-S3}",
    "keep": "keep",
    "ablation": "ablation",
    "reference": "ref",
    "discard": "---",
    "smoke": "smoke",
    "unknown": "?",
}

SESSION_COLOR = {
    "Det-S1": r"\rowcolor[gray]{0.93}",
    "S1":     r"\rowcolor[gray]{0.97}",
    "S2":     r"\rowcolor[gray]{0.93}",
    "Sweep":  r"\rowcolor[gray]{0.97}",
    "S3":     r"\rowcolor[gray]{0.93}",
    "Post":   r"\rowcolor[gray]{0.97}",
    "Ablation": r"\rowcolor[gray]{0.90}",
}


def write_latex(df):
    out = RESULTS_DIR / "all_experiments.tex"

    lines = []
    lines += [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.75in, top=0.65in, bottom=0.65in]{geometry}",
        r"\usepackage{times}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{colortbl}",
        r"\usepackage{xcolor}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{hyperref}",
        r"\usepackage{microtype}",
        r"\usepackage{caption}",
        r"\usepackage{array}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\LTpre}{4pt}",
        r"\setlength{\LTpost}{4pt}",
        "",
        r"\title{\textbf{Complete Experiment Log}\\",
        r"\large XLE Energy ETF Direction Model\\",
        r"\normalsize All Sessions --- 74 Experiments}",
        r"\author{energy-autoresearch}",
        r"\date{May 2026}",
        "",
        r"\begin{document}",
        r"\maketitle",
        r"\thispagestyle{empty}",
        "",
        r"\noindent\textbf{Dataset:} XLE (Yahoo Finance, 2020-01-01 onward), 80/20 chronological split.",
        r"\textbf{Metric:} Annualised Sharpe $= \sqrt{252}\cdot\mathbb{E}[\mathrm{sign}(\hat{s})\cdot r_t]/$std$(\cdot)$.",
        r"WF~Sharpe is mean walk-forward Sharpe over expanding folds (``---'' = not computed).",
        r"Status: \textbf{best} = best committed model; keep = improved over prior; --- = discarded.",
        r"Sessions: Det-S1 = alternative-data experiments (Apr~20); S1--S3 = autoresearch loop;",
        r"Sweep = HP grid; Post = post-cleanup; Ablation = structured Phase~2 ablations.",
        "",
    ]

    col_spec = r">{\ttfamily\footnotesize}p{1.1cm}" \
               r">{\footnotesize}p{0.7cm}" \
               r">{\footnotesize}p{0.9cm}" \
               r">{\footnotesize\raggedright}p{4.6cm}" \
               r">{\footnotesize}p{1.5cm}" \
               r">{\footnotesize\centering}p{0.45cm}" \
               r">{\footnotesize\centering}p{0.35cm}" \
               r">{\footnotesize\centering}p{0.35cm}" \
               r">{\footnotesize\centering}p{0.6cm}" \
               r">{\footnotesize\centering}p{0.5cm}" \
               r">{\footnotesize\raggedleft}p{0.85cm}" \
               r">{\footnotesize\raggedleft}p{0.75cm}" \
               r">{\footnotesize}p{1.1cm}"

    header = (
        r"\textbf{Run ID} & \textbf{Sess.} & \textbf{Date} & "
        r"\textbf{Description} & \textbf{Model} & "
        r"$n$ & $d$ & msl & TW & WTI & "
        r"\textbf{Val $S$} & \textbf{WF $S$} & \textbf{Status}"
    )

    lines += [
        r"\begin{longtable}{" + col_spec + r"}",
        r"\caption{Complete experiment log. TW = train window (trading days); "
        r"WTI = shock filter threshold; Val~$S$ = validation Sharpe; WF~$S$ = walk-forward Sharpe.}",
        r"\label{tab:all_experiments}\\",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{13}{l}{\small\textit{Table continued from previous page}} \\",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{13}{r}{\small\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    prev_session = None
    for _, row in df.iterrows():
        sess = row["session"]
        if sess != prev_session:
            lines.append(
                r"\multicolumn{13}{l}{\cellcolor[gray]{0.80}"
                r"\small\textbf{" + sess + r"}} \\"
            )
            prev_session = sess

        color = SESSION_COLOR.get(sess, "")
        desc_safe = str(row["description"]).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
        model_safe = str(row["model"]).replace("_", r"\_")
        status_tex = STATUS_LABEL.get(str(row["status"]), str(row["status"]))

        n = row["n_estimators"]
        d = row["max_depth"]
        msl = row["min_samples_leaf"]
        tw = row["train_window"]
        wti = row["wti_thresh"]

        n_s   = str(n) if str(n) != "—" else "---"
        d_s   = str(d) if str(d) != "—" else "---"
        msl_s = str(msl) if str(msl) != "—" else "---"
        tw_s  = str(tw) if str(tw) not in ("—", "full") else str(tw).replace("—", "---")
        try:
            wti_s = f"{float(wti)*100:.0f}\\%"
        except (ValueError, TypeError):
            wti_s = "---"

        sharpe_val = _fmt(row["val_sharpe"])
        wf_val     = _fmt(row["wf_sharpe"]) if str(row["wf_sharpe"]) not in ("—", "nan") else "---"

        date_s = str(row["timestamp"])[:10]
        run_id_safe = str(row["run_id"]).replace("_", r"\_")

        line = (
            f"{color}"
            f"{run_id_safe} & {sess} & {date_s} & {desc_safe} & "
            f"\\texttt{{{model_safe}}} & "
            f"{n_s} & {d_s} & {msl_s} & {tw_s} & {wti_s} & "
            f"{sharpe_val} & {wf_val} & {status_tex} \\\\"
        )
        lines.append(line)

    lines += [
        r"\end{longtable}",
        "",
        r"\end{document}",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX -> {out}")
    return out


def compile_pdf(tex_path):
    pdflatex = Path.home() / "AppData/Roaming/TinyTeX/bin/windows/pdflatex.exe"
    if not pdflatex.exists():
        print("pdflatex not found; skipping PDF compilation")
        return
    for _ in range(2):
        result = subprocess.run(
            [str(pdflatex), "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True, text=True,
        )
    out_pdf = tex_path.with_suffix(".pdf")
    if out_pdf.exists():
        size_kb = out_pdf.stat().st_size // 1024
        print(f"PDF  -> {out_pdf}  ({size_kb} KB)")
    else:
        print("PDF compilation failed. Check .log for errors.")
        print(result.stdout[-2000:])


def main():
    tsv_rows  = load_tsv()
    det_rows  = DETAILED_S1
    sweep_rows = load_sweep_csv()
    json_rows  = load_json_ablations()

    all_rows = det_rows + tsv_rows + sweep_rows + json_rows
    df = build_dataframe(all_rows)

    print(f"\nTotal experiments: {len(df)}")
    print(df.groupby("session")["run_id"].count().to_string())
    print()

    write_csv(df)
    tex_path = write_latex(df)
    compile_pdf(tex_path)


if __name__ == "__main__":
    main()
