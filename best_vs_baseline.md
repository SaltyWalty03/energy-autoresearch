# Best Run vs. Baseline

## Run Identifiers
- **Baseline**: run_001 — Baseline
- **Best**:     run_009 — n_estimators=600

## Metric Comparison
| metric | baseline | best run | delta | % change |
|--------|----------|----------|-------|----------|
| Sharpe (annualised) | 2.4041 | 2.4793 | +0.0752 | +3.13% |

## Hyperparameter Diff (baseline → best)
| parameter | baseline | best run |
|-----------|----------|----------|
| `max_depth` | 2 | 2 |
| `min_samples_leaf` | 22 | 22 |
| `n_estimators` | 300 | 600 | **<-- changed**
| `train_window` | 756 | 756 |
| `window` | 20 | 20 |
| `wti_thresh` | 0.02 | 0.02 |

## Summary
The best configuration (run_009: n_estimators=600) improved Sharpe by
**+0.0752** (+3.13%) relative to the baseline.
This is a meaningful gain in risk-adjusted return.
