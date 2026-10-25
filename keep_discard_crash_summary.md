# Keep / Discard / Crash Summary

Baseline Sharpe (run_001): **2.5903**

| run_id | description | status | sharpe | notes |
|--------|-------------|--------|--------|-------|
| run_001 | Baseline | **Keep** | 2.5903 | reference baseline |
| run_002 | max_depth=3 | **Keep** | 2.5941 | +0.0038 Sharpe over baseline |
| run_003 | max_depth=4 | **Discard** | 2.0454 | -0.5449 Sharpe vs baseline (2.5903) |
| run_004 | max_depth=1 (stumps) | **Discard** | 0.7120 | -1.8783 Sharpe vs baseline (2.5903) |
| run_005 | min_samples_leaf=10 | **Keep** | 2.6104 | +0.0201 Sharpe over baseline |
| run_006 | min_samples_leaf=50 | **Discard** | 1.7419 | -0.8484 Sharpe vs baseline (2.5903) |
| run_007 | min_samples_leaf=100 | **Discard** | 1.8515 | -0.7388 Sharpe vs baseline (2.5903) |
| run_008 | n_estimators=100 | **Discard** | 2.5363 | -0.0540 Sharpe vs baseline (2.5903) |
| run_009 | n_estimators=600 | **Keep** | 2.5990 | +0.0087 Sharpe over baseline |
| run_010 | window=10 | **Discard** | 2.5903 | 0.0000 Sharpe vs baseline (2.5903) |
| run_011 | window=40 | **Discard** | 2.5903 | 0.0000 Sharpe vs baseline (2.5903) |
| run_012 | wti_thresh=0.03 (strict) | **Discard** | 2.3902 | -0.2001 Sharpe vs baseline (2.5903) |
| run_013 | wti_thresh=0.01 (loose) | **Discard** | 1.5038 | -1.0865 Sharpe vs baseline (2.5903) |
| run_014 | train_window=504 (2yr) | **Discard** | 1.2833 | -1.3070 Sharpe vs baseline (2.5903) |
| run_015 | combo depth3+leaf15+win10 | **Discard** | 2.4010 | -0.1893 Sharpe vs baseline (2.5903) |