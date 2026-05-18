# Keep / Discard / Crash Summary

Baseline Sharpe (run_001): **2.4041**

| run_id | description | status | sharpe | notes |
|--------|-------------|--------|--------|-------|
| run_001 | Baseline | **Keep** | 2.4041 | reference baseline |
| run_002 | max_depth=3 | **Discard** | 2.1359 | -0.2682 Sharpe vs baseline (2.4041) |
| run_003 | max_depth=4 | **Keep** | 2.4652 | +0.0611 Sharpe over baseline |
| run_004 | max_depth=1 (stumps) | **Discard** | 1.2735 | -1.1306 Sharpe vs baseline (2.4041) |
| run_005 | min_samples_leaf=10 | **Discard** | 2.3837 | -0.0204 Sharpe vs baseline (2.4041) |
| run_006 | min_samples_leaf=50 | **Discard** | 1.4882 | -0.9159 Sharpe vs baseline (2.4041) |
| run_007 | min_samples_leaf=100 | **Discard** | 1.5910 | -0.8131 Sharpe vs baseline (2.4041) |
| run_008 | n_estimators=100 | **Keep** | 2.4742 | +0.0701 Sharpe over baseline |
| run_009 | n_estimators=600 | **Keep** | 2.4793 | +0.0752 Sharpe over baseline |
| run_010 | window=10 | **Discard** | 2.3446 | -0.0595 Sharpe vs baseline (2.4041) |
| run_011 | window=40 | **Discard** | 1.9068 | -0.4973 Sharpe vs baseline (2.4041) |
| run_012 | wti_thresh=0.03 (strict) | **Discard** | 2.2201 | -0.1840 Sharpe vs baseline (2.4041) |
| run_013 | wti_thresh=0.01 (loose) | **Discard** | 1.4536 | -0.9505 Sharpe vs baseline (2.4041) |
| run_014 | train_window=504 (2yr) | **Discard** | 1.2470 | -1.1571 Sharpe vs baseline (2.4041) |
| run_015 | combo depth3+leaf15+win10 | **Discard** | 2.0270 | -0.3771 Sharpe vs baseline (2.4041) |