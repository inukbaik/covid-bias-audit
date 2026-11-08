# covid_bias.analysis

## Overview

One module per analysis. Each module exposes `run(cfg: Config, out: Path) -> dict`.
That function loads data, fits models, writes CSV tables to `out/tables/` and PNG
figures to `out/figures/`, and returns headline numbers. The CLI
([`../cli.py`](../cli.py)) merges those numbers into `out/metrics.json`.

| Module | Analysis | Data loader |
|---|---|---|
| `education.py` | Race, education, sex, and age vs the COVID share of deaths (quasi-binomial GLM, plus cross-validated ridge and random forest) | `load_education` |
| `county.py` | County disparity ratios by race with bootstrap confidence intervals, state-grouped CV prediction, and a state-clustered GLM | `load_county` |
| `risk.py` | Fairness audit of a hypothetical high-COVID-burden classifier (trained on 2020, tested on 2021) | `load_monthly` |
| `original.py` | Re-runs of the original paper's pipelines, to measure how misleading they were | raw CSVs read directly |

Dependencies: loaders come from [`../data/`](../data/README.md), metrics and CV from
[`../evaluation/`](../evaluation/README.md), and figure helpers from
[`../plotting.py`](../plotting.py).

## Interface

### `__init__.py` (shared output helpers)

- `table_path(out, name) -> Path` and `figure_path(out, name) -> Path`: return
  `out/tables/<name>` or `out/figures/<name>`, creating the directory first.
- `write_table(df, out, name) -> DataFrame`: writes a CSV (`index=False`,
  `float_format="%.6g"`) and returns `df`.

### `education.py`

- `run(cfg, out) -> dict`
- `FEATURES = ["race", "education", "sex", "age_group"]`
- `model_frame(df)`: casts `FEATURES` to `str`.
- `pooled_shares(df)`: the pooled COVID share for each factor level, with a Wilson
  95% CI.
- `fit_glm(df)`: a quasi-binomial GLM with reference levels `REF_RACE`,
  `REF_EDUCATION`, `REF_SEX`, and the first age group.
- `odds_ratio_table(res)`: columns `factor`, `level`, `odds_ratio`, `ci_low`,
  `ci_high`, `p_value`.
- `splits_for(df, cfg)`: the `RepeatedKFold` train/test index pairs.

### `county.py`

- `run(cfg, out) -> dict`
- `POP_COLS`, `LINEAR_POP_COLS` (the same list without `pop_share_white`), and
  `RATIO_TYPES` (`covid`, `allcause`, `covid_vs_allcause`).
- `disparity_ratios(df)`: long format with `fips`, `race`, `ratio_type`, `ratio`,
  `covid_deaths`.
- `summarise_ratios(long, cfg)`: for each race and ratio type, gives `n_counties`,
  the median, quartiles, and a COVID-death-weighted mean with a bootstrap CI.
- `model_data(df)`: drops rows with no `log_income` and casts `urban_rural` to
  `str`.
- `fit_county_glm(d)`: a quasi-binomial GLM with state-clustered standard errors.
  Population shares are scaled by 10, so each odds ratio is per 10 percentage
  points.
- `glm_table(res)`: `term`, `odds_ratio`, `ci_low`, `ci_high`, `p_value`, without
  the intercept.

### `risk.py`

- `run(cfg, out) -> dict`
- `THRESHOLD = 0.5`
- `SPECS`: five `(name, kind, features)` model specifications. They are a majority
  baseline, logistic regression on age, age + sex, and age + sex + race, and
  HistGradientBoosting on age + sex + race.
- `build_cells(monthly, min_deaths)`, `compute_tau(train)` (computed as
  `sum(covid_deaths) / sum(all_cause)`), `add_label(df, tau)` (adds
  `high_burden = covid_share > tau`), and `split_temporal(cells) -> (train, test, tau)`.

### `original.py`

- `run(cfg, out) -> dict`
- `education_original(cfg)`, `county_original(cfg)`, `risk_original(cfg)`: each
  returns a dict of reproduced metrics and diagnostics (for example `seed_r2`,
  `importance`, `majority_baseline`, `double_count_share_of_sum`).
- `EDU_FEATURES`: the raw column names the original code used.
- `run` also calls the private `education._models` and the public
  `education.splits_for`, to compare against the corrected pipeline.

## Invariants

- **All randomness is seeded from `cfg.seed`.** This covers the random forests,
  HistGradientBoosting, `RepeatedKFold`, the bootstrap, and permutation importance.
  The exception is `original.py`, which hard-codes `random_state=42` to match the
  original code. It also uses split seeds `range(cfg.n_original_seeds)` plus seed 42,
  and `default_rng(0)` for figure jitter.
- **Cross-validation:**
  - county: `GroupKFold(n_splits=5)` grouped by state, so no state appears in both
    train and test, and permutation importance uses `n_repeats=5`
  - education: `RepeatedKFold(n_splits=5, n_repeats=cfg.cv_repeats)`, and
    permutation importance uses `n_repeats=3`
- **The risk model has no time leakage.** It trains on 2020 and tests on 2021, `tau`
  comes from the training data only, cells need `all_cause >= cfg.min_cell_deaths`,
  and the decision threshold is `THRESHOLD`.
- **Reference levels:** the county GLM uses `Large central metro`. The education GLM
  uses `config.REF_*`, and the headline odds ratios it reports are for `HS or less`
  and `Some college`.
- **County ratios** are kept only where the population share is above 0 and the
  share is not NaN. `covid_vs_allcause` also needs the all-cause share to be above 0.
- **Model settings:** ridge uses `alpha=1.0`. Random forests use
  `min_samples_leaf=3` and `cfg.rf_estimators` trees.
- **`original.py` keeps the original errors on purpose:**
  - label encoding and an 80/20 split for education
  - zero-filled NaNs in the county distribution indicator
  - MULTIPLE + UNDERLYING double counting, a mean-score label threshold, and a 70/30
    split for risk

  Do not "fix" this module. Its job is to reproduce those errors.
- The package contains no asserts and raises no errors of its own. Validation happens
  in the loaders.

## Side Effects

- Creates `out/tables/` and `out/figures/` (`mkdir(parents=True, exist_ok=True)`).
- Writes these files:

  | Module | Tables (`out/tables/`) | Figures (`out/figures/`) |
  |---|---|---|
  | `education` | `education_pooled_share.csv`, `education_glm_odds_ratios.csv`, `education_cv_scores.csv`, `education_permutation_importance.csv` | `education_odds_ratios.png`, `education_permutation_importance.png` |
  | `county` | `county_disparity_ratios.csv`, `county_cv_scores.csv`, `county_permutation_importance.csv`, `county_glm_odds_ratios.csv` | `county_disparity_ratios.png`, `county_permutation_importance.png` |
  | `risk` | `risk_model_scores.csv`, `risk_group_fairness.csv`, `risk_fairness_summary.csv` | `risk_group_fairness.png` |
  | `original` | `original_reproduction.csv` | `original_r2_instability.png` |

- Reads `data/raw/` through the loaders. `original.py` reads all four raw CSVs
  directly with pandas, bypassing the loaders' checks.
- Figures change matplotlib's global `rcParams` through `plotting.new_figure`.
- INFO-level logging: counties dropped, cell counts, and `tau` and the base rate.

## Verification

- Run: `make test` (runs `python -m pytest`; see `tests/test_risk.py` and
  `tests/test_smoke.py`)
- Full rerun: `make reproduce`
