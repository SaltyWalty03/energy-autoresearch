# Prompt 1: XGBoost Swap + Walk-Forward Validation
# Run this first. Do not start Prompt 2 until this confirms Sharpe >= 2.48.

Swap the RandomForestClassifier in `model.py`'s `DirectionModel` for XGBoost.
Specifically:

1. Add `xgboost` to `requirements.txt` and `cloud_function/requirements.txt`

2. Replace `RandomForestClassifier` with `xgb.XGBClassifier` in `DirectionModel`.
   Starting hyperparameters: `n_estimators=600, max_depth=3, learning_rate=0.05,
   subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
   eval_metric='logloss', random_state=42`. Keep the StandardScaler, sample
   weights, WTI shock filter, and 3-year rolling window exactly as they are.

3. Add a `model_type` parameter to `DirectionModel.__init__()` that accepts
   `"xgb"` or `"rf"` so both are available for the autoresearch loop to compare.
   Default to `"xgb"`.

4. Add a `walk_forward_sharpe()` function to `prepare.py` that:
   - Takes the full return series and a model instance
   - Splits into 4 folds of equal length
   - Trains on folds 1-N, tests on fold N+1, slides forward
   - Returns mean and std of Sharpe across folds
   - This must be used in `run_experiments.py` INSTEAD of the current
     single holdout Sharpe — update the results columns in `results.tsv`
     to include `wf_sharpe_mean` and `wf_sharpe_std`

5. Run `python run.py "xgb-baseline"` and confirm walk-forward Sharpe mean
   >= 2.0. If it is lower, try `max_depth=2` and `n_estimators=800` before
   concluding RF is better.

Do not touch `run_intraday.py`, `run_daily.py`, or any cloud infrastructure.


---


# Prompt 2: Multi-Asset Expansion (XLB + XOP)
# Only run after Prompt 1 confirms XGBoost Sharpe >= 2.48.

Extend the pipeline to trade XLB (materials) and XOP (E&P) alongside XLE.

1. In `model.py`, make `DirectionModel` fully ticker-agnostic — remove any
   hardcoded "XLE" or "CL=F" references. Add a `commodity_ticker` parameter
   to `__init__()` (default `"CL=F"`) so XLB can use `"HG=F"` (copper) and
   XOP uses `"CL=F"` as well.

2. Create `src/universe.py` defining:
   ```python
   ASSETS = {
       "XLE": {"commodity": "CL=F", "shock_thresh": 0.02},
       "XLB": {"commodity": "HG=F", "shock_thresh": 0.025},
       "XOP": {"commodity": "CL=F", "shock_thresh": 0.02},
   }
   ```

3. Update `prepare.py`'s `load_and_split_data()` to accept a `ticker`
   parameter (default `"XLE"`).

4. Create `train_all.py` that iterates over `ASSETS`, trains one
   `DirectionModel` per ticker, runs walk-forward validation, and saves
   each model as `models/{ticker}_model.pkl`. Print a summary table of
   walk-forward Sharpe per asset.

5. Update `run_daily.py` to:
   - Load all three models from `models/`
   - Generate a signal per asset
   - Compute a Kelly fraction per asset using the asset's own walk-forward
     Sharpe as the edge estimate: `kelly_f = (sharpe / 252**0.5) * 0.5`
     (half-Kelly, annualised-to-daily conversion)
   - Allocate portfolio across all three, capped so total exposure <= 95%
     of account equity
   - Place orders for all three assets under the existing GCS lock

6. Update `.gitignore` to include `models/*.pkl`.

Do not touch `run_intraday.py` yet — intraday will follow the multi-asset
pattern in a later prompt.


---


# Prompt 3: t-SNE Regime Layer
# Only run after Prompt 2. This is the most complex change — read fully before starting.

Add a regime identification layer using t-SNE + HDBSCAN that conditions
the DirectionModel on the current market regime.

1. Add to `requirements.txt`: `hdbscan`, `scikit-learn` (already present),
   `joblib` (already present).

2. Create `src/regime.py` with:

   `fit_regimes(feature_matrix: np.ndarray, perplexity=30, n_clusters=4)`:
   - Fits t-SNE (n_components=2, perplexity=perplexity, random_state=42)
     on the feature matrix
   - Fits HDBSCAN(min_cluster_size=20) on the 2D embedding
   - Returns a `RegimeModel` dataclass containing:
     - `tsne_embedding`: the 2D coordinates
     - `labels`: cluster label per row (-1 = noise, reassign to nearest cluster)
     - `cluster_centers`: mean feature vector per cluster
     - `n_regimes`: number of distinct regimes found
   - Saves a `regime_map.pkl` to `models/`

   `classify_regime(feature_vector: np.ndarray, regime_model: RegimeModel) -> int`:
   - Uses KNN (k=5) on the t-SNE embedding to assign a new feature vector
     to the nearest regime
   - This is the inference-time function — t-SNE itself is never called
     at inference time

3. Update `train_all.py` to:
   - After building features for each ticker, call `fit_regimes()` and
     save the regime map per ticker as `models/{ticker}_regime.pkl`
   - Train one `DirectionModel` per regime per ticker, saved as
     `models/{ticker}_regime{k}_model.pkl`
   - Log walk-forward Sharpe per regime so you can see which regimes
     the model predicts well vs. poorly

4. Update `run_daily.py` inference to:
   - Load `{ticker}_regime.pkl` and classify today's feature vector
   - Load the regime-specific model `{ticker}_regime{k}_model.pkl`
   - Log which regime each asset is in: `"XLE: regime 2 (of 4)"`
   - Use regime-specific walk-forward Sharpe for Kelly sizing

5. Create `scripts/plot_regimes.py` that:
   - Loads `models/XLE_regime.pkl`
   - Plots the t-SNE embedding colored by:
     (a) regime label
     (b) forward 5-day return (to validate regimes have different return profiles)
   - Saves both plots to `results/regime_embedding.png` and
     `results/regime_returns.png`
   - These are your portfolio piece visuals — make them clean

6. Add a `--regime-analysis` flag to `run_experiments.py` that runs the
   autoresearch loop over `perplexity` in [15, 30, 50] and `n_clusters`
   in [3, 4, 5], selecting the config that maximises between-regime
   Sharpe variance (i.e. regimes that are most distinct in predictability).

Constraint: if `classify_regime()` returns -1 (noise point at inference),
fall back to the non-regime model for that asset rather than erroring.
