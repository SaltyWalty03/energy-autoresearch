# Phase 1 Evaluation Results (Corrected Harness)

**Assets:** XLE, XLB, XOP  
**Models:** M0_RF9 (incumbent RF-9), M1_XGB9 (XGB-9), M2_RF11 (RF-11)  
**Folds:** 4 (walk-forward, expanding window)  
**TC:** 5 bps one-way; flip (long↔short) = 2×TC  
**Short borrow:** 1.5% p.a.  
**N_TRIALS (DSR):** 9  

## Harness Corrections

| # | Correction |
|---|-----------|
| 1 | Deflated Sharpe: daily SR, `(excess_kurtosis+2)/4` raw-kurtosis term, SR0 from cross-trial variance. |
| 2 | TC: \|Δsign\| × TC per leg — long↔short flip costs 2×TC. |
| 3 | Both ±0.00 (raw) and ±0.02 (deployed) signal thresholds reported. |
| 4 | Intraday RSI/MACD gate approximated on daily bars (RSI<45/RSI>55 + MACD sign). **Approximation**: daily RSI has narrower range than 5-min RSI. |
| 5 | Regime model refit inside each training fold (t-SNE + HDBSCAN). |
| 6 | N_V2_FEATURES = 11 (was incorrectly 12 in first run). |

## Rolling Beta and Net Exposure

Rolling 252-day OLS beta of XOP daily returns on XLE daily returns.

| Metric | Value |
|--------|-------|
| latest_beta | 1.1792 |
| rolling_mean_beta | 1.2066 |
| rolling_std_beta | 0.0496 |
| xop_notional | 79900.0 |
| xle_notional | -80100.0 |
| xlb_notional | -40300.0 |
| xop_xle_equivalent | 94221.0 |
| net_energy_xle_equiv | 14121.0 |
| gross_notional | 200300.0 |

**Interpretation:** At beta = 1.179, the long XOP $79,900 position is equivalent to $94,221 long XLE. Net after the short XLE $80,100: **$+14,121 net long XLE-equivalent energy exposure** (plus short XLB $40,300 which is materials, not energy).

## Summary Table — ±0.00 threshold (raw sign)

| Asset | Model | Sharpe | Sortino | MaxDD | Turnover | HitRate | DSR |
|-------|-------|-------:|--------:|------:|---------:|--------:|----:|
| XLE | M0_RF9 | -0.2202 | -0.2822 | -0.8290 | 0.5340 | 0.5278 | 0.0220 |
| XLE | M1_XGB9 | -0.4704 | -0.6051 | -0.6357 | 0.6144 | 0.5017 | 0.0049 |
| XLE | M2_RF11 | -0.2026 | -0.2586 | -0.6276 | 0.4988 | 0.5267 | 0.0241 |
| XLB | M0_RF9 | -0.2139 | -0.3147 | -0.4228 | 0.3427 | 0.5026 | 0.0227 |
| XLB | M1_XGB9 | -1.1642 | -1.6726 | -1.0860 | 0.5402 | 0.4746 | 0.0000 |
| XLB | M2_RF11 | -0.4454 | -0.6530 | -0.5365 | 0.3201 | 0.4930 | 0.0059 |
| XOP | M0_RF9 | -0.5834 | -0.7317 | -1.1133 | 0.5589 | 0.4960 | 0.0022 |
| XOP | M1_XGB9 | -0.4497 | -0.5537 | -0.8584 | 0.6112 | 0.5131 | 0.0055 |
| XOP | M2_RF11 | -0.2520 | -0.3138 | -0.8391 | 0.5496 | 0.5210 | 0.0182 |

## Summary Table — ±0.02 threshold (deployed logic)

| Asset | Model | Sharpe | Sortino | MaxDD | Turnover | HitRate | DSR |
|-------|-------|-------:|--------:|------:|---------:|--------:|----:|
| XLE | M0_RF9 | -0.1607 | -0.1970 | -0.6697 | 0.5340 | 0.5278 | 0.0300 |
| XLE | M1_XGB9 | -0.4903 | -0.6175 | -0.6778 | 0.6144 | 0.5017 | 0.0043 |
| XLE | M2_RF11 | -0.1869 | -0.2208 | -0.6040 | 0.4988 | 0.5267 | 0.0260 |
| XLB | M0_RF9 | -0.3299 | -0.4726 | -0.4180 | 0.3427 | 0.5026 | 0.0119 |
| XLB | M1_XGB9 | -1.1426 | -1.6237 | -1.0439 | 0.5402 | 0.4746 | 0.0000 |
| XLB | M2_RF11 | -0.1849 | -0.2624 | -0.4262 | 0.3201 | 0.4930 | 0.0264 |
| XOP | M0_RF9 | -0.4647 | -0.5412 | -0.8977 | 0.5589 | 0.4960 | 0.0050 |
| XOP | M1_XGB9 | -0.2373 | -0.2934 | -0.5544 | 0.6112 | 0.5131 | 0.0198 |
| XOP | M2_RF11 | -0.2189 | -0.2496 | -0.7592 | 0.5496 | 0.5210 | 0.0218 |

## Intraday Filter Comparison (daily RSI/MACD approximation)

> **Note:** The deployed function uses 5-minute bar RSI/MACD. Daily RSI
> stays in a narrower range, making RSI<45 and RSI>55 trigger more
> frequently than in the live system. Treat this as indicative only.

| Asset | Model | Sharpe (no gate) | Sharpe (±0.02) | Sharpe (daily intraday gate) |
|-------|-------|----------------:|---------------:|-----------------------------:|
| XLE | M0_RF9 | -0.2202 | -0.1607 | -0.0402 |
| XLE | M1_XGB9 | -0.4704 | -0.4903 | -0.5460 |
| XLE | M2_RF11 | -0.2026 | -0.1869 | -0.0694 |
| XLB | M0_RF9 | -0.2139 | -0.3299 | 0.7133 |
| XLB | M1_XGB9 | -1.1642 | -1.1426 | 0.1700 |
| XLB | M2_RF11 | -0.4454 | -0.1849 | 0.6545 |
| XOP | M0_RF9 | -0.5834 | -0.4647 | -0.3032 |
| XOP | M1_XGB9 | -0.4497 | -0.2373 | -0.4782 |
| XOP | M2_RF11 | -0.2520 | -0.2189 | -0.2810 |

## Per-Fold Net Sharpe (±0.02 threshold, after costs)

| Asset | Model | Fold 1 | Fold 2 | Fold 3 |
|-------|-------|-------:|-------:|-------:|
| XLE | M0_RF9 | -0.6342 | -0.9396 | 0.8655 |
| XLE | M1_XGB9 | -0.2663 | -1.2070 | -0.2537 |
| XLE | M2_RF11 | -1.3555 | -0.3860 | 1.1623 |
| XLB | M0_RF9 | -0.4735 | -0.1784 | -0.2913 |
| XLB | M1_XGB9 | -1.4890 | -1.3072 | -0.6306 |
| XLB | M2_RF11 | -0.2781 | -0.1324 | -0.1214 |
| XOP | M0_RF9 | -0.9490 | -0.6920 | 0.1314 |
| XOP | M1_XGB9 | -0.6877 | 0.3115 | -0.1028 |
| XOP | M2_RF11 | -0.7331 | -0.6228 | 0.5324 |

## Cross-Check: prepare.walk_forward_sharpe (gross, no costs, no threshold)

| Asset | Model | WF Mean | WF Std |
|-------|-------|--------:|-------:|
| XLE | M0_RF9 | 0.1104 | 0.9838 |
| XLE | M1_XGB9 | 0.0176 | 0.3011 |
| XLE | M2_RF11 | 0.1403 | 0.7332 |
| XLB | M0_RF9 | 0.1950 | 0.2206 |
| XLB | M1_XGB9 | -0.4493 | 0.5971 |
| XLB | M2_RF11 | -0.1013 | 0.1801 |
| XOP | M0_RF9 | -0.2158 | 0.4780 |
| XOP | M1_XGB9 | 0.0466 | 0.4673 |
| XOP | M2_RF11 | 0.0883 | 0.4466 |

## Regime Breakdown (regime model refit inside each training fold)

Sharpe ± SE (annualized). n = sample size in test folds for that regime.

### XLE

_No regime data available._

### XLB

_No regime data available._

### XOP

_No regime data available._

## Promotion Decision

Promotion criteria: beats M0_RF9 on net Sharpe in **≥ 2 of 3 test folds** (±0.02 threshold) **AND** DSR > 0.50.

### XLE

- **M1_XGB9** vs M0_RF9:
  - Net Sharpe (±0.02): -0.4903 vs -0.1607
  - Fold wins: 1/3 (need ≥ 2)
  - DSR: 0.0043 (need > 0.50)
  - Decision: hold

- **M2_RF11** vs M0_RF9:
  - Net Sharpe (±0.02): -0.1869 vs -0.1607
  - Fold wins: 2/3 (need ≥ 2)
  - DSR: 0.0260 (need > 0.50)
  - Decision: hold

### XLB

- **M1_XGB9** vs M0_RF9:
  - Net Sharpe (±0.02): -1.1426 vs -0.3299
  - Fold wins: 0/3 (need ≥ 2)
  - DSR: 0.0000 (need > 0.50)
  - Decision: hold

- **M2_RF11** vs M0_RF9:
  - Net Sharpe (±0.02): -0.1849 vs -0.3299
  - Fold wins: 3/3 (need ≥ 2)
  - DSR: 0.0264 (need > 0.50)
  - Decision: hold

### XOP

- **M1_XGB9** vs M0_RF9:
  - Net Sharpe (±0.02): -0.2373 vs -0.4647
  - Fold wins: 2/3 (need ≥ 2)
  - DSR: 0.0198 (need > 0.50)
  - Decision: hold

- **M2_RF11** vs M0_RF9:
  - Net Sharpe (±0.02): -0.2189 vs -0.4647
  - Fold wins: 3/3 (need ≥ 2)
  - DSR: 0.0218 (need > 0.50)
  - Decision: hold

