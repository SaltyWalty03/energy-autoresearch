# Best Run vs. Baseline

## Run Identifiers
- **Baseline**: run_001 — Baseline
- **Best**:     run_005 — min_samples_leaf=10

## Metric Comparison
| metric | baseline | best run | delta | % change |
|--------|----------|----------|-------|----------|
| Sharpe (annualised) | 2.5903 | 2.6104 | +0.0201 | +0.78% |

## Hyperparameter Diff (baseline → best)
| parameter | baseline | best run |
|-----------|----------|----------|
| `max_depth` | 2 | 2 |
| `min_samples_leaf` | 22 | 10 | **<-- changed**
| `n_estimators` | 300 | 300 |
| `train_window` | 756 | 756 |
| `window` | 20 | 20 |
| `wti_thresh` | 0.02 | 0.02 |

## Summary
The best configuration (run_005: min_samples_leaf=10) improved Sharpe by
**+0.0201** (+0.78%) relative to the baseline.
The gain is modest, suggesting the baseline is already well-tuned for this dataset.
