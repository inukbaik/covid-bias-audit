# covid_bias.evaluation

## Overview

Model-evaluation helpers shared by the analysis modules in
[`../analysis/`](../analysis/README.md). The package has three parts:

- A quasi-binomial GLM fit wrapper that computes the Pearson dispersion itself.
- Weighted regression metrics, binary-classification metrics, per-group fairness
  tables, and confidence intervals.
- Cross-validation helpers that return held-out weighted scores per fold and
  per-fold permutation importance on raw (unencoded) columns.

`__init__.py` holds only a docstring and exports nothing. Import from the submodules
directly, for example `from covid_bias.evaluation.metrics import weighted_r2`.

## Interface

### `glm.py`

- `fit_quasi_binomial(model, **fit_kwargs)`: fits the model once to get
  `phi = pearson_chi2 / df_resid`, refits with `scale=phi` and `fit_kwargs`, sets
  `res.dispersion = phi`, and returns the refitted results. The model is expected to
  be a statsmodels GLM, but this file does not import statsmodels.

### `metrics.py`

- `weighted_r2(y_true, y_pred, sample_weight=None) -> float`: `1 - SSE/SST` around
  the weighted mean.
- `weighted_mae(y_true, y_pred, sample_weight=None) -> float`
- `binary_metrics(y_true, y_prob, threshold=0.5) -> dict[str, float]`: keys
  `accuracy`, `balanced_accuracy`, `f1`, `roc_auc`, `brier`.
- `group_fairness_table(y_true, y_pred, groups, y_prob=None) -> pd.DataFrame`: one
  row per group, sorted, with `group`, `n`, `base_rate`, `selection_rate`, `tpr`,
  `fpr`, `fnr`, `precision`. Adds `mean_pred_prob` and `observed_rate` when `y_prob`
  is given.
- `fairness_summary(table) -> dict[str, float]`: keys
  `demographic_parity_difference`, `tpr_range`, `fpr_range`,
  `equalized_odds_difference`.
- `bootstrap_ci(values, weights=None, stat=None, n_boot=2000, seed=42, alpha=0.05)
  -> (estimate, low, high)`: percentile bootstrap. The default statistic is the
  weighted mean.
- `wilson_interval(successes, n, z=1.96) -> (low, high)`

### `validation.py`

- `cv_scores(model, X, y, weights, splits, scale=100.0) -> pd.DataFrame`: columns
  `fold`, `r2`, `mae`. The MAE is multiplied by `scale`.
- `summarise_scores(scores, model_name) -> dict`: keys `model`, `r2_mean`, `r2_std`,
  `mae_mean`, `mae_std`, `n_folds`.
- `permutation_importance_cv(model, X, y, weights, splits, seed=42, n_repeats=5,
  metric=weighted_r2) -> pd.DataFrame`: columns `feature`, `importance_mean`,
  `importance_std`, sorted by `importance_mean` in descending order.

## Invariants

- The fold scheme is set by the caller. `cv_scores` and `permutation_importance_cv`
  use whatever `splits` iterable they are given, so leakage control (for example,
  grouping by state) is the caller's job.
- `_fit` clones the estimator before every fit, so the caller's model object is never
  fitted in place. For a `Pipeline`, sample weights go to the last step as
  `<step>__sample_weight`.
- Randomness is seeded and local. `bootstrap_ci` and `permutation_importance_cv` each
  create `np.random.default_rng(seed)` with a default seed of 42.
- Undefined values come back as NaN rather than raising an error:
  - `weighted_r2` when SST is 0
  - `binary_metrics["roc_auc"]` when `y_true` has fewer than 2 classes
  - per-group rates whose denominator is 0 or less (through `_safe_div`)
  - `_weighted_mean` when the weights sum to 0 or less
  - `wilson_interval` when `n <= 0`
  - `summarise_scores` standard deviations when there is only one fold
    (otherwise `ddof=1`)
- `fairness_summary` ignores a NaN range when computing
  `equalized_odds_difference`, and returns NaN only if both ranges are NaN.
- `binary_metrics` treats `y_prob >= threshold` as a positive prediction. F1 uses
  `zero_division=0`.

## Side Effects

- `fit_quasi_binomial` fits the model twice and adds a `dispersion` attribute to the
  results object it returns.
- None of these files read or write disk, use the network, log, or print, and none
  changes the global NumPy random state.

## Verification

- Run: `make test` (runs `python -m pytest`; see `tests/test_metrics.py`)
