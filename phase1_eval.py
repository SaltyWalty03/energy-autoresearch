"""
phase1_eval.py — Phase 1 walk-forward evaluation (corrected harness).

Models
------
  M0_RF9   incumbent DirectionModel  (RF, 9 features)
  M1_XGB9  XGBoost on the same 9 features
  M2_RF11  DirectionModelV2 (RF, 11 features)

Corrections vs first run
-------------------------
1. Deflated Sharpe: daily (non-annualized) SR; raw-kurtosis term
   (excess_kurtosis + 2)/4; SR0 from cross-trial Sharpe variance (two-pass).
2. Transaction costs: |Δsign| × TC_BPS per one-way leg — a long→short flip
   charges 2 × TC (close long + open short).
3. Both ±0.00 (raw sign) and ±0.02 (deployed threshold) signal variants are
   evaluated and reported side by side.
4. Intraday RSI/MACD gate approximated from daily bars (same RSI_OVERSOLD=45
   / RSI_OVERBOUGHT=55 thresholds as cloud_function/main.py).  Approximation
   quality is noted; daily RSI has different dynamics from 5-min RSI.
5. Regime model is refit inside each training fold (t-SNE + HDBSCAN).
   Per-regime Sharpe is reported with sample size n and SE.
6. Feature count: N_V2_FEATURES = 11 (resolves earlier 11/12 inconsistency).
7. Rolling 252-day beta of XOP on XLE; net book exposure in XLE-equivalents.

Run with:
    python phase1_eval.py
"""

import warnings
warnings.filterwarnings("ignore")

import copy, json, logging, math, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

import prepare
from model import DirectionModel, DirectionModelV2, N_V2_FEATURES, build_feature_matrix
from src.regime import fit_regimes, classify_regime

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("phase1_eval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HERE        = Path(__file__).parent
MODELS_DIR  = HERE / "models"

ASSETS   = ["XLE", "XLB", "XOP"]
N_FOLDS  = 4
TC_BPS   = 5                  # one-way, basis points
SHORT_BORROW_ANNUAL = 0.015   # 1.5 % p.a. short-borrow cost
MODEL_THRESH  = 0.02          # mirrors cloud_function/main.py MODEL_THRESH
RSI_OVERSOLD  = 45
RSI_OVERBOUGHT = 55
N_TRIALS = 9                  # 3 models × 3 assets (threshold variants are
                              # harness settings, not independent strategies)

ASSET_CFG = {
    "XLE": {"commodity": "CL=F", "shock_thresh": 0.02},
    "XLB": {"commodity": "HG=F", "shock_thresh": 0.025},
    "XOP": {"commodity": "CL=F", "shock_thresh": 0.02},
}
CROSS_ASSET_FOR = {"XLE": "XOP", "XOP": "XLE", "XLB": "XLE"}

_GAMMA_EM = 0.5772156649
_E        = math.e


# ---------------------------------------------------------------------------
# 1.  PSR / Deflated Sharpe (two-pass)
# ---------------------------------------------------------------------------

def _psr(sr_daily: float, T: int, skew: float, excess_kurtosis: float,
         sr0: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio — PSR(sr0).

    Uses daily (non-annualized) SR and the exact Bailey & López de Prado 2014
    formula:
        adj = 1 - skew*SR + (excess_kurtosis + 2)/4 * SR^2
        z   = (SR - sr0) * sqrt(T-1) / sqrt(adj)
        PSR = Phi(z)

    The (excess_kurtosis + 2)/4 term equals (gamma4_raw - 1)/4 with gamma4_raw
    = excess_kurtosis + 3, which is what the paper denotes as (gamma4 - 1)/4.
    """
    adj = 1.0 - skew * sr_daily + (excess_kurtosis + 2.0) / 4.0 * sr_daily ** 2
    adj = max(adj, 1e-8)
    z   = (sr_daily - sr0) * math.sqrt(max(T - 1, 1)) / math.sqrt(adj)
    return float(norm.cdf(z))


def _sr_variance(sr_daily: float, T: int, skew: float, excess_kurtosis: float) -> float:
    """Asymptotic variance of the SR estimator (BLP 2014 eq. 4)."""
    adj = 1.0 - skew * sr_daily + (excess_kurtosis + 2.0) / 4.0 * sr_daily ** 2
    adj = max(adj, 1e-8)
    return adj / max(T - 1, 1)


def add_deflated_sharpes(results: dict) -> None:
    """
    Two-pass DSR computation.

    Pass 1 — collect (sr_daily, T, skew, excess_kurtosis) from every trial.
    Pass 2 — compute V_bar = mean of per-trial SR variances, derive SR0 as
             E[max SR] under N(0, V_bar), then compute DSR for each trial.

    Modifies each result dict in-place by adding 'defl_sharpe_{variant}'.
    """
    for variant in ("thresh0", "thresh002"):
        key_sr  = f"sr_daily_{variant}"
        key_T   = "T"
        key_sk  = f"skew_{variant}"
        key_ek  = f"excess_kurtosis_{variant}"
        key_out = f"defl_sharpe_{variant}"

        trial_stats = []
        for res in results.values():
            sr   = res.get(key_sr, float("nan"))
            T    = res.get(key_T, 0)
            sk   = res.get(key_sk, 0.0)
            ek   = res.get(key_ek, 0.0)
            if not (math.isnan(sr) or T < 2):
                trial_stats.append((sr, T, sk, ek))

        if not trial_stats:
            for res in results.values():
                res[key_out] = float("nan")
            continue

        V_bar = float(np.mean([_sr_variance(s, t, sk, ek)
                               for s, t, sk, ek in trial_stats]))
        K     = len(trial_stats)
        if K >= 2:
            sr0 = math.sqrt(V_bar) * (
                (1.0 - _GAMMA_EM) * norm.ppf(1.0 - 1.0 / K)
                + _GAMMA_EM * norm.ppf(1.0 - 1.0 / (K * _E))
            )
        else:
            sr0 = 0.0

        for res in results.values():
            sr  = res.get(key_sr, float("nan"))
            T   = res.get(key_T, 0)
            sk  = res.get(key_sk, 0.0)
            ek  = res.get(key_ek, 0.0)
            if math.isnan(sr) or T < 2:
                res[key_out] = float("nan")
            else:
                res[key_out] = round(_psr(sr, T, sk, ek, sr0=sr0), 4)


# ---------------------------------------------------------------------------
# 2.  Signal & cost helpers
# ---------------------------------------------------------------------------

def _strategy_signs(preds: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Map continuous predictions to {-1, 0, +1}."""
    signs = np.zeros(len(preds), dtype=float)
    signs[preds >  threshold] =  1.0
    signs[preds < -threshold] = -1.0
    return signs


def _compute_costs(signs: np.ndarray) -> np.ndarray:
    """
    Daily cost array.

    One-way TC: |Δsign| × TC_BPS × 1e-4 per leg.
    A flat→long or long→flat move incurs 1 TC.
    A long→short flip (|Δsign| = 2) incurs 2 TC — one to close, one to open.
    Short borrow: SHORT_BORROW_ANNUAL / 252 on short days.
    """
    prev      = np.concatenate([[0.0], signs[:-1]])
    tc        = np.abs(signs - prev) * TC_BPS * 1e-4
    tc[signs < 0] += SHORT_BORROW_ANNUAL / 252.0
    return tc


def _metrics_from_net_rets(net_rets: np.ndarray, preds: np.ndarray,
                            targets: np.ndarray) -> dict:
    """Compute all scalar metrics from net daily returns."""
    T    = len(net_rets)
    mu   = float(net_rets.mean())
    sig  = float(net_rets.std())

    sr_daily = mu / sig if sig > 0 else 0.0
    sharpe   = sr_daily * math.sqrt(252)

    down     = net_rets[net_rets < 0]
    down_std = float(down.std()) if len(down) > 1 else float("nan")
    sortino  = (mu / down_std * math.sqrt(252)) if down_std > 0 else float("nan")

    cum    = np.cumsum(net_rets)
    peak   = np.maximum.accumulate(cum)
    max_dd = float((cum - peak).min())

    signs    = np.sign(preds)
    prev_s   = np.concatenate([[0.0], signs[:-1]])
    turnover = float((signs != prev_s).mean())

    active   = np.abs(signs) > 0
    hit_rate = float((np.sign(preds[active]) * np.sign(targets[active]) > 0).mean()) \
               if active.sum() > 0 else float("nan")

    skew_val = float(pd.Series(net_rets).skew())
    ek_val   = float(pd.Series(net_rets).kurtosis())   # excess kurtosis

    return dict(
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4) if not math.isnan(sortino) else float("nan"),
        max_dd=round(max_dd, 6),
        turnover=round(turnover, 4),
        hit_rate=round(hit_rate, 4) if not math.isnan(hit_rate) else float("nan"),
        sr_daily=sr_daily,
        T=T,
        skew=skew_val,
        excess_kurtosis=ek_val,
    )


# ---------------------------------------------------------------------------
# 3.  Intraday RSI/MACD approximation (daily bars)
# ---------------------------------------------------------------------------

def _daily_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))


def _daily_macd_hist(close: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd - macd.ewm(span=sig, adjust=False).mean()


def _intraday_mask(full_df: pd.DataFrame, test_idx: pd.Index) -> np.ndarray:
    """
    Approximate the intraday RSI/MACD gate using daily close prices.

    NOTE: The deployed gate uses 5-minute bars with RSI_OVERSOLD=45 and
    RSI_OVERBOUGHT=55. Daily RSI typically stays in a narrower range than
    intraday RSI, so these thresholds produce different hit rates. This is
    an approximation — treat results as indicative, not exact.

    Returns a boolean array (True = intraday gate passes) aligned to test_idx.
    """
    if "Close" not in full_df.columns:
        return np.ones(len(test_idx), dtype=bool)

    close = full_df["Close"].reindex(full_df.index)
    rsi   = _daily_rsi(close)
    hist  = _daily_macd_hist(close)

    rsi_t  = rsi.reindex(test_idx, method="nearest")
    hist_t = hist.reindex(test_idx, method="nearest")

    # Same direction-specific thresholds as cloud_function/main.py
    # (will be applied after sign of signal is known in caller)
    return rsi_t.values, hist_t.values


# ---------------------------------------------------------------------------
# 4.  Regime breakdown with per-fold refit
# ---------------------------------------------------------------------------

def _regime_breakdown_refitted(fold_data: list, ticker: str) -> dict:
    """
    Refit a regime model on each fold's training set, classify test rows,
    then aggregate per-regime Sharpe ± SE across all folds.

    fold_data: list of dicts with keys
        train_df, test_df, net_rets_thresh002, signs_thresh002
    Returns: {regime_label: {sharpe, n, se}}
    """
    per_regime: dict[int, list] = {}

    for fold in fold_data:
        train_df     = fold["train_df"]
        test_df      = fold["test_df"]
        net_rets     = fold["net_rets_thresh002"]
        signs        = fold["signs_thresh002"]

        r_train = train_df["ret_lag"].values.astype(np.float64)
        feat_tr = build_feature_matrix(r_train)
        valid_tr = np.isfinite(feat_tr).all(axis=1)
        feat_tr_clean = feat_tr[valid_tr]

        if len(feat_tr_clean) < 30:
            continue

        try:
            regime_model = fit_regimes(feat_tr_clean, perplexity=30, n_clusters=4)
        except Exception as exc:
            log.warning("Regime refit failed for %s fold: %s", ticker, exc)
            continue

        r_test  = test_df["ret_lag"].values.astype(np.float64)
        feat_te = build_feature_matrix(r_test)
        valid_te = np.isfinite(feat_te).all(axis=1)

        try:
            labels = regime_model.knn.predict(feat_te[valid_te])
        except Exception as exc:
            log.warning("Regime classify failed for %s fold: %s", ticker, exc)
            continue

        # Align valid indices to net_rets (which has the same length as test_df)
        te_indices = np.where(valid_te)[0]
        for i, lbl in zip(te_indices, labels):
            if i < len(net_rets):
                per_regime.setdefault(int(lbl), []).append(net_rets[i])

    result = {}
    for lbl, daily_rets in per_regime.items():
        arr  = np.array(daily_rets)
        n    = len(arr)
        mu   = arr.mean()
        sig  = arr.std()
        sh   = float(mu / sig * math.sqrt(252)) if sig > 0 else 0.0
        se   = float(sig / math.sqrt(n) * math.sqrt(252)) if n > 1 else float("nan")
        result[lbl] = {"sharpe": round(sh, 4), "n": n,
                       "se": round(se, 4) if not math.isnan(se) else float("nan")}
    return result


# ---------------------------------------------------------------------------
# 5.  Main walk-forward evaluator
# ---------------------------------------------------------------------------

def extended_wf_eval(full_df: pd.DataFrame, model, n_folds: int = 4,
                     ticker: str = "XLE") -> dict:
    """
    Run walk-forward folds, then compute metrics for two signal variants:
      thresh0   : sign(pred) — no threshold
      thresh002 : sign after ±MODEL_THRESH=0.02 filter (deployed behaviour)

    Also computes a third variant:
      intraday  : thresh002 signals further filtered by daily RSI/MACD gate

    Returns a dict with all metrics and raw fold data for regime breakdown.
    """
    n         = len(full_df)
    fold_size = n // n_folds

    fold_data  = []
    all_preds  = []
    all_tgts   = []

    for fold in range(1, n_folds):
        train_end = fold * fold_size
        test_end  = min((fold + 1) * fold_size, n)

        train_block = full_df.iloc[:train_end]
        test_block  = full_df.iloc[train_end:test_end]

        if len(test_block) < 10:
            continue

        X_tr = train_block[["ret_lag"]]
        y_tr = train_block["target"]
        X_te = test_block[["ret_lag"]]
        y_te = test_block["target"].values

        try:
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
        except Exception as exc:
            log.warning("Fold %d failed for %s/%s: %s",
                        fold, ticker, type(model).__name__, exc)
            continue

        all_preds.append(preds)
        all_tgts.append(y_te)

        # Build per-fold net_rets for regime refit (thresh002)
        s002    = _strategy_signs(preds, MODEL_THRESH)
        c002    = _compute_costs(s002)
        nr002   = np.sign(preds) * y_te - c002   # gross ret via thresh002 sign
        # Recompute gross with correct sign mapping
        nr002   = s002 * y_te - c002

        fold_data.append(dict(
            train_df=train_block,
            test_df=test_block,
            net_rets_thresh002=nr002,
            signs_thresh002=s002,
        ))

    if not all_preds:
        return _empty_result()

    preds_arr = np.concatenate(all_preds)
    tgts_arr  = np.concatenate(all_tgts)
    T         = len(preds_arr)

    # ── variant: thresh0 ───────────────────────────────────────────────────
    s0       = _strategy_signs(preds_arr, 0.0)
    c0       = _compute_costs(s0)
    nr0      = s0 * tgts_arr - c0
    m0       = _metrics_from_net_rets(nr0, preds_arr, tgts_arr)

    # Per-fold net Sharpe (thresh0)
    fold_sharpes_t0 = []
    for fp, ft in zip(all_preds, all_tgts):
        fs = _strategy_signs(fp, 0.0)
        fc = _compute_costs(fs)
        fn = fs * ft - fc
        fstd = fn.std()
        fold_sharpes_t0.append(
            round(float(fn.mean() / fstd * math.sqrt(252)) if fstd > 0 else 0.0, 4)
        )

    # ── variant: thresh002 ─────────────────────────────────────────────────
    s002     = _strategy_signs(preds_arr, MODEL_THRESH)
    c002     = _compute_costs(s002)
    nr002    = s002 * tgts_arr - c002
    m002     = _metrics_from_net_rets(nr002, preds_arr, tgts_arr)

    fold_sharpes_t002 = []
    for fp, ft in zip(all_preds, all_tgts):
        fs = _strategy_signs(fp, MODEL_THRESH)
        fc = _compute_costs(fs)
        fn = fs * ft - fc
        fstd = fn.std()
        fold_sharpes_t002.append(
            round(float(fn.mean() / fstd * math.sqrt(252)) if fstd > 0 else 0.0, 4)
        )

    # ── variant: intraday approx (daily RSI/MACD gate on thresh002) ────────
    # Rebuild test-block close prices for RSI/MACD
    if "Close" in full_df.columns:
        intraday_sharpe = _intraday_approx_sharpe(
            full_df, all_preds, all_tgts, n_folds, fold_size, n
        )
    else:
        intraday_sharpe = float("nan")

    # ── regime breakdown with refit ────────────────────────────────────────
    regime_bkdown = _regime_breakdown_refitted(fold_data, ticker)

    # ── cross-check: prepare.walk_forward_sharpe ──────────────────────────
    try:
        wf_mean, wf_std = prepare.walk_forward_sharpe(
            full_df[["ret_lag", "target"]], copy.deepcopy(model), n_folds=n_folds
        )
    except Exception as exc:
        log.warning("prepare.wf_sharpe failed for %s/%s: %s",
                    ticker, type(model).__name__, exc)
        wf_mean, wf_std = float("nan"), float("nan")

    return dict(
        # thresh0
        sharpe_t0      = m0["sharpe"],
        sortino_t0     = m0["sortino"],
        max_dd_t0      = m0["max_dd"],
        turnover_t0    = m0["turnover"],
        hit_rate_t0    = m0["hit_rate"],
        fold_sharpes_t0= fold_sharpes_t0,
        sr_daily_thresh0      = m0["sr_daily"],
        skew_thresh0          = m0["skew"],
        excess_kurtosis_thresh0 = m0["excess_kurtosis"],
        # thresh002 (deployed)
        sharpe_t002    = m002["sharpe"],
        sortino_t002   = m002["sortino"],
        max_dd_t002    = m002["max_dd"],
        turnover_t002  = m002["turnover"],
        hit_rate_t002  = m002["hit_rate"],
        fold_sharpes_t002 = fold_sharpes_t002,
        sr_daily_thresh002      = m002["sr_daily"],
        skew_thresh002          = m002["skew"],
        excess_kurtosis_thresh002 = m002["excess_kurtosis"],
        # intraday approx
        sharpe_intraday = intraday_sharpe,
        # shared
        T              = T,
        wf_mean        = wf_mean,
        wf_std         = wf_std,
        regime_breakdown = regime_bkdown,
        fold_data      = fold_data,   # kept for regime refit
    )


def _intraday_approx_sharpe(full_df, all_preds, all_tgts,
                             n_folds, fold_size, n) -> float:
    """Apply daily RSI/MACD gate to thresh002 signals. Returns net Sharpe."""
    close = full_df["Close"]
    rsi   = _daily_rsi(close)
    hist  = _daily_macd_hist(close)

    filtered_preds = []
    fold_idx = 0
    for fold in range(1, n_folds):
        train_end = fold * fold_size
        test_end  = min((fold + 1) * fold_size, n)
        test_block = full_df.iloc[train_end:test_end]
        if len(test_block) < 10:
            continue
        if fold_idx >= len(all_preds):
            break

        fp = all_preds[fold_idx]
        fold_idx += 1
        fs  = _strategy_signs(fp, MODEL_THRESH)

        rsi_t  = rsi.reindex(test_block.index, method="nearest").values
        hist_t = hist.reindex(test_block.index, method="nearest").values

        long_ok  = (rsi_t < RSI_OVERSOLD)  & (hist_t > 0)
        short_ok = (rsi_t > RSI_OVERBOUGHT) & (hist_t < 0)
        gate     = ((fs == 1) & long_ok) | ((fs == -1) & short_ok) | (fs == 0)
        filtered_preds.append(np.where(gate, fs, 0.0))

    if not filtered_preds:
        return float("nan")

    fp_arr = np.concatenate(filtered_preds)
    ft_arr = np.concatenate(all_tgts)
    fc     = _compute_costs(fp_arr)
    fn     = fp_arr * ft_arr - fc
    fstd   = fn.std()
    return round(float(fn.mean() / fstd * math.sqrt(252)) if fstd > 0 else 0.0, 4)


def _empty_result():
    nan = float("nan")
    return dict(
        sharpe_t0=nan, sortino_t0=nan, max_dd_t0=nan,
        turnover_t0=nan, hit_rate_t0=nan, fold_sharpes_t0=[],
        sr_daily_thresh0=nan, skew_thresh0=0.0, excess_kurtosis_thresh0=0.0,
        sharpe_t002=nan, sortino_t002=nan, max_dd_t002=nan,
        turnover_t002=nan, hit_rate_t002=nan, fold_sharpes_t002=[],
        sr_daily_thresh002=nan, skew_thresh002=0.0, excess_kurtosis_thresh002=0.0,
        sharpe_intraday=nan, T=0, wf_mean=nan, wf_std=nan,
        regime_breakdown={}, fold_data=[],
    )


# ---------------------------------------------------------------------------
# 6.  Promotion helper
# ---------------------------------------------------------------------------

def _beats_incumbent(cand_folds: list, m0_folds: list, min_wins: int = 2) -> bool:
    wins = sum(c > b for c, b in zip(cand_folds, m0_folds))
    return wins >= min_wins


# ---------------------------------------------------------------------------
# 7.  Rolling beta and net exposure
# ---------------------------------------------------------------------------

def compute_rolling_beta_and_exposure(window: int = 252) -> dict:
    """
    Compute rolling 252-day OLS beta of XOP returns on XLE returns.
    Restate current live book in XLE-equivalent notional.

    Current book (from Phase 0 audit):
        Long  XOP  $79,900
        Short XLE  $80,100
        Short XLB  $40,300
    """
    print("  Fetching XOP/XLE for rolling beta...", end=" ", flush=True)
    try:
        raw = yf.download(["XOP", "XLE"], start="2021-01-01",
                          auto_adjust=True, progress=False)
        close = raw["Close"] if not isinstance(raw.columns, pd.MultiIndex) \
                else raw.xs("Close", axis=1, level=0)
        rets  = close.pct_change().dropna()
        roll_cov = rets["XOP"].rolling(window).cov(rets["XLE"])
        roll_var = rets["XLE"].rolling(window).var()
        beta     = (roll_cov / roll_var).dropna()
        latest   = float(beta.iloc[-1])
        beta_mean = float(beta.mean())
        beta_std  = float(beta.std())
        print(f"done. Latest beta={latest:.3f}, mean={beta_mean:.3f} ± {beta_std:.3f}")
    except Exception as exc:
        log.warning("Rolling beta fetch failed: %s", exc)
        return {"error": str(exc)}

    # Current book
    xop_notl  =  79_900.0
    xle_notl  = -80_100.0
    xlb_notl  = -40_300.0

    # XLE-equivalent
    xop_xle_eq  = xop_notl * latest
    net_energy  = xop_xle_eq + xle_notl   # combined XOP+XLE in XLE-equivalent $
    gross_notl  = abs(xop_notl) + abs(xle_notl) + abs(xlb_notl)

    return dict(
        latest_beta          = round(latest, 4),
        rolling_mean_beta    = round(beta_mean, 4),
        rolling_std_beta     = round(beta_std, 4),
        xop_notional         = xop_notl,
        xle_notional         = xle_notl,
        xlb_notional         = xlb_notl,
        xop_xle_equivalent   = round(xop_xle_eq, 0),
        net_energy_xle_equiv = round(net_energy, 0),
        gross_notional       = round(gross_notl, 0),
    )


# ---------------------------------------------------------------------------
# 8.  Main
# ---------------------------------------------------------------------------

def main():
    # ── Rolling beta ──────────────────────────────────────────────────────
    print("\n=== Rolling Beta / Net Exposure ===")
    beta_info = compute_rolling_beta_and_exposure()
    for k, v in beta_info.items():
        print(f"  {k}: {v}")

    results: dict[tuple[str, str], dict] = {}

    for ticker in ASSETS:
        cfg     = ASSET_CFG[ticker]
        ca_tick = CROSS_ASSET_FOR[ticker]

        print(f"\n{'='*60}")
        print(f"  Asset: {ticker}")
        print(f"{'='*60}")

        # Load daily data (also need Close for intraday approx)
        print(f"  Loading data for {ticker}...")
        import yfinance as yf2
        raw_all = yf2.download(ticker, start="2020-01-01", auto_adjust=True,
                               progress=False)
        if isinstance(raw_all.columns, pd.MultiIndex):
            raw_all.columns = raw_all.columns.get_level_values(0)
        raw_all["target"] = np.log(raw_all["Close"] / raw_all["Close"].shift(1)).shift(-1)
        raw_all["ret_lag"] = raw_all["Close"].pct_change()
        full_df = raw_all.dropna()

        models = {
            "M0_RF9": DirectionModel(
                n_estimators=600, max_depth=2, min_samples_leaf=22,
                train_window=756, wti_thresh=cfg["shock_thresh"],
                model_type="rf", commodity_ticker=cfg["commodity"],
            ),
            "M1_XGB9": DirectionModel(
                n_estimators=600, max_depth=2, min_samples_leaf=22,
                train_window=756, wti_thresh=cfg["shock_thresh"],
                model_type="xgb", learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, commodity_ticker=cfg["commodity"],
            ),
            "M2_RF11": DirectionModelV2(
                n_estimators=600, max_depth=2, min_samples_leaf=22,
                train_window=756, wti_thresh=cfg["shock_thresh"],
                model_type="rf", commodity_ticker=cfg["commodity"],
                cross_asset_ticker=ca_tick,
            ),
        }

        for model_name, model in models.items():
            print(f"  Evaluating {model_name}...", end=" ", flush=True)
            res = extended_wf_eval(full_df, model, n_folds=N_FOLDS, ticker=ticker)
            results[(ticker, model_name)] = res
            print(
                f"Sharpe(t0)={res['sharpe_t0']:.4f}  "
                f"Sharpe(t.02)={res['sharpe_t002']:.4f}  "
                f"Sharpe(intra)={res['sharpe_intraday']}"
            )

    # ── Two-pass DSR ──────────────────────────────────────────────────────
    add_deflated_sharpes(results)

    # ── Print summary table ────────────────────────────────────────────────
    _print_table(results)

    # ── Write report ──────────────────────────────────────────────────────
    _write_results_md(results, beta_info)
    print("\nResults written to phase1_results.md")


def _print_table(results):
    hdr = (f"\n{'Asset':<6} {'Model':<10} "
           f"{'Sh(t0)':>8} {'Sh(t.02)':>9} {'Sh(intra)':>10} "
           f"{'Srt':>7} {'MaxDD':>9} {'Turn':>6} {'Hit':>6} "
           f"{'DSR(t.02)':>10}  FoldSharpes(t.02)")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for ticker in ASSETS:
        for model_name in ["M0_RF9", "M1_XGB9", "M2_RF11"]:
            r = results.get((ticker, model_name), {})
            def f(v, w=8):
                return f"{v:{w}.4f}" if isinstance(v, float) and not math.isnan(v) else f"{'n/a':>{w}}"
            print(
                f"{ticker:<6} {model_name:<10} "
                f"{f(r.get('sharpe_t0', float('nan')))} "
                f"{f(r.get('sharpe_t002', float('nan')), 9)} "
                f"{f(r.get('sharpe_intraday', float('nan')), 10)} "
                f"{f(r.get('sortino_t002', float('nan')), 7)} "
                f"{f(r.get('max_dd_t002', float('nan')), 9)} "
                f"{f(r.get('turnover_t002', float('nan')), 6)} "
                f"{f(r.get('hit_rate_t002', float('nan')), 6)} "
                f"{f(r.get('defl_sharpe_thresh002', float('nan')), 10)}  "
                f"{r.get('fold_sharpes_t002', [])}"
            )


def _fmt(v, dp=4):
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{dp}f}"
    return str(v)


def _write_results_md(results, beta_info):
    lines = []

    lines += [
        "# Phase 1 Evaluation Results (Corrected Harness)",
        "",
        f"**Assets:** {', '.join(ASSETS)}  ",
        f"**Models:** M0_RF9 (incumbent RF-9), M1_XGB9 (XGB-9), M2_RF11 (RF-11)  ",
        f"**Folds:** {N_FOLDS} (walk-forward, expanding window)  ",
        f"**TC:** {TC_BPS} bps one-way; flip (long↔short) = 2×TC  ",
        f"**Short borrow:** {SHORT_BORROW_ANNUAL*100:.1f}% p.a.  ",
        f"**N_TRIALS (DSR):** {N_TRIALS}  ",
        "",
        "## Harness Corrections",
        "",
        "| # | Correction |",
        "|---|-----------|",
        "| 1 | Deflated Sharpe: daily SR, `(excess_kurtosis+2)/4` raw-kurtosis term, SR0 from cross-trial variance. |",
        "| 2 | TC: \\|Δsign\\| × TC per leg — long↔short flip costs 2×TC. |",
        "| 3 | Both ±0.00 (raw) and ±0.02 (deployed) signal thresholds reported. |",
        "| 4 | Intraday RSI/MACD gate approximated on daily bars (RSI<45/RSI>55 + MACD sign). **Approximation**: daily RSI has narrower range than 5-min RSI. |",
        "| 5 | Regime model refit inside each training fold (t-SNE + HDBSCAN). |",
        "| 6 | N_V2_FEATURES = 11 (was incorrectly 12 in first run). |",
        "",
    ]

    # ── Rolling beta ─────────────────────────────────────────────────────
    lines += [
        "## Rolling Beta and Net Exposure",
        "",
        "Rolling 252-day OLS beta of XOP daily returns on XLE daily returns.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in beta_info.items():
        lines.append(f"| {k} | {v} |")

    if "net_energy_xle_equiv" in beta_info and "latest_beta" in beta_info:
        b = beta_info["latest_beta"]
        xle_eq = beta_info["net_energy_xle_equiv"]
        lines += [
            "",
            f"**Interpretation:** At beta = {b:.3f}, the long XOP ${abs(beta_info['xop_notional']):,.0f} "
            f"position is equivalent to ${abs(beta_info['xop_xle_equivalent']):,.0f} long XLE. "
            f"Net after the short XLE ${abs(beta_info['xle_notional']):,.0f}: "
            f"**${xle_eq:+,.0f} net long XLE-equivalent energy exposure** "
            f"(plus short XLB ${abs(beta_info['xlb_notional']):,.0f} which is materials, not energy).",
            "",
        ]

    # ── Summary tables ────────────────────────────────────────────────────
    for variant, v_label, fs_key, dsr_key in [
        ("t0",   "±0.00 threshold (raw sign)",        "fold_sharpes_t0",   "defl_sharpe_thresh0"),
        ("t002", "±0.02 threshold (deployed logic)",  "fold_sharpes_t002", "defl_sharpe_thresh002"),
    ]:
        lines += [
            f"## Summary Table — {v_label}",
            "",
            "| Asset | Model | Sharpe | Sortino | MaxDD | Turnover | HitRate | DSR |",
            "|-------|-------|-------:|--------:|------:|---------:|--------:|----:|",
        ]
        for ticker in ASSETS:
            for mname in ["M0_RF9", "M1_XGB9", "M2_RF11"]:
                r = results.get((ticker, mname), {})
                lines.append(
                    f"| {ticker} | {mname} "
                    f"| {_fmt(r.get(f'sharpe_{variant}', float('nan')))} "
                    f"| {_fmt(r.get(f'sortino_{variant}', float('nan')))} "
                    f"| {_fmt(r.get(f'max_dd_{variant}', float('nan')))} "
                    f"| {_fmt(r.get(f'turnover_{variant}', float('nan')))} "
                    f"| {_fmt(r.get(f'hit_rate_{variant}', float('nan')))} "
                    f"| {_fmt(r.get(dsr_key, float('nan')))} |"
                )
        lines.append("")

    # ── Intraday approx table ─────────────────────────────────────────────
    lines += [
        "## Intraday Filter Comparison (daily RSI/MACD approximation)",
        "",
        "> **Note:** The deployed function uses 5-minute bar RSI/MACD. Daily RSI",
        "> stays in a narrower range, making RSI<45 and RSI>55 trigger more",
        "> frequently than in the live system. Treat this as indicative only.",
        "",
        "| Asset | Model | Sharpe (no gate) | Sharpe (±0.02) | Sharpe (daily intraday gate) |",
        "|-------|-------|----------------:|---------------:|-----------------------------:|",
    ]
    for ticker in ASSETS:
        for mname in ["M0_RF9", "M1_XGB9", "M2_RF11"]:
            r = results.get((ticker, mname), {})
            lines.append(
                f"| {ticker} | {mname} "
                f"| {_fmt(r.get('sharpe_t0', float('nan')))} "
                f"| {_fmt(r.get('sharpe_t002', float('nan')))} "
                f"| {_fmt(r.get('sharpe_intraday', float('nan')))} |"
            )
    lines.append("")

    # ── Per-fold Sharpe ───────────────────────────────────────────────────
    lines += [
        "## Per-Fold Net Sharpe (±0.02 threshold, after costs)",
        "",
        f"| Asset | Model | Fold 1 | Fold 2 | Fold 3 |",
        "|-------|-------|-------:|-------:|-------:|",
    ]
    for ticker in ASSETS:
        for mname in ["M0_RF9", "M1_XGB9", "M2_RF11"]:
            r  = results.get((ticker, mname), {})
            fs = r.get("fold_sharpes_t002", [])
            cells = " | ".join(_fmt(v) for v in fs)
            lines.append(f"| {ticker} | {mname} | {cells} |")
    lines.append("")

    # ── Cross-check ───────────────────────────────────────────────────────
    lines += [
        "## Cross-Check: prepare.walk_forward_sharpe (gross, no costs, no threshold)",
        "",
        "| Asset | Model | WF Mean | WF Std |",
        "|-------|-------|--------:|-------:|",
    ]
    for ticker in ASSETS:
        for mname in ["M0_RF9", "M1_XGB9", "M2_RF11"]:
            r = results.get((ticker, mname), {})
            lines.append(
                f"| {ticker} | {mname} "
                f"| {_fmt(r.get('wf_mean', float('nan')))} "
                f"| {_fmt(r.get('wf_std', float('nan')))} |"
            )
    lines.append("")

    # ── Regime breakdown ──────────────────────────────────────────────────
    lines += [
        "## Regime Breakdown (regime model refit inside each training fold)",
        "",
        "Sharpe ± SE (annualized). n = sample size in test folds for that regime.",
        "",
    ]
    for ticker in ASSETS:
        lines.append(f"### {ticker}")
        lines.append("")
        all_labels: set[int] = set()
        for mname in ["M0_RF9", "M1_XGB9", "M2_RF11"]:
            rb = results.get((ticker, mname), {}).get("regime_breakdown", {})
            all_labels.update(rb.keys())
        if not all_labels:
            lines.append("_No regime data available._")
            lines.append("")
            continue
        lbl_list = sorted(all_labels)
        hdr_cols = " | ".join(f"Regime {l} (n, Sh±SE)" for l in lbl_list)
        lines.append(f"| Model | {hdr_cols} |")
        lines.append("|-------|" + "|".join(["------------------"] * len(lbl_list)) + "|")
        for mname in ["M0_RF9", "M1_XGB9", "M2_RF11"]:
            rb = results.get((ticker, mname), {}).get("regime_breakdown", {})
            cells = []
            for l in lbl_list:
                entry = rb.get(l, {})
                if entry:
                    cells.append(
                        f"n={entry['n']} {entry['sharpe']:.3f}±{_fmt(entry['se'])}"
                    )
                else:
                    cells.append("—")
            lines.append(f"| {mname} | " + " | ".join(cells) + " |")
        lines.append("")

    # ── Promotion decision ────────────────────────────────────────────────
    lines += [
        "## Promotion Decision",
        "",
        "Promotion criteria: beats M0_RF9 on net Sharpe in **≥ 2 of 3 test folds** "
        "(±0.02 threshold) **AND** DSR > 0.50.",
        "",
    ]
    for ticker in ASSETS:
        lines.append(f"### {ticker}")
        lines.append("")
        m0_folds = results.get((ticker, "M0_RF9"), {}).get("fold_sharpes_t002", [])
        m0_sh    = results.get((ticker, "M0_RF9"), {}).get("sharpe_t002", float("nan"))
        for cand in ["M1_XGB9", "M2_RF11"]:
            r          = results.get((ticker, cand), {})
            cf         = r.get("fold_sharpes_t002", [])
            csh        = r.get("sharpe_t002", float("nan"))
            dsr        = r.get("defl_sharpe_thresh002", float("nan"))
            wins       = sum(c > b for c, b in zip(cf, m0_folds))
            beats      = wins >= 2
            dsr_ok     = (not math.isnan(dsr)) and dsr > 0.5
            decision   = "**PROMOTE**" if beats and dsr_ok else "hold"
            lines += [
                f"- **{cand}** vs M0_RF9:",
                f"  - Net Sharpe (±0.02): {_fmt(csh)} vs {_fmt(m0_sh)}",
                f"  - Fold wins: {wins}/{len(m0_folds)} (need ≥ 2)",
                f"  - DSR: {_fmt(dsr)} (need > 0.50)",
                f"  - Decision: {decision}",
                "",
            ]

    out = HERE / "phase1_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
