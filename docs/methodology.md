# Methodology audit

This document records each problem found in the original analyses (`git show 00b0933`) and how the
rebuilt pipeline fixes it. Every "original" number below is reproduced by `analysis/original.py` and
written to `reports/tables/original_reproduction.csv`.

## 1. Education × race model (originally "Feature Importance SEP + Race")

| # | Problem in the original | Why it matters | Fix |
|---|---|---|---|
| 1.1 | **Single random train/test split.** R² = 0.918 at `random_state=42` | 252 rows is too few for one split. Across 50 seeds the same model scores **−0.53 to 0.99** (median 0.81) | Repeated 5-fold CV (5 × 10 = 50 folds), reported as mean ± sd |
| 1.2 | **Target is a raw death count** (`COVID-19 Deaths`) | Counts mostly measure the *size* of each demographic cell, so "age is most important" partly means "older cells have more deaths" | Target is the COVID share of deaths (`covid / total`), weighted by total deaths. A binomial GLM models the counts with the correct exposure |
| 1.3 | **`LabelEncoder` on nominal features** (race, sex) | Imposes an arbitrary order (Hispanic < NH AIAN < …) that trees can split on | One-hot encoding in a `Pipeline` |
| 1.4 | **Impurity (MDI) feature importance**, training data, no uncertainty | MDI is biased and computed in-sample, and it is not a measure of association with the outcome | Permutation importance on held-out folds (mean ± sd). Effect sizes come from GLM odds ratios with 95% CIs |
| 1.5 | **No baseline.** MAE, MSE and R² are reported without context and are absent from the code | Can't tell whether the model learned anything | Weighted-mean `DummyRegressor` baseline in every comparison |
| 1.6 | **Data quality** | A truncated label (`…Pacific Islande`) and a corrupted apostrophe (`Bachelor�s`) went unnoticed. `procdata.csv` was written to the repo root but read from `datasets/` | Loader normalises labels, drops `Unknown` and 2019 (no COVID), and validates the schema |

**Corrected result.** The quasi-binomial GLM (dispersion ≈ 54) gives race odds ratios vs NH White of
Hispanic 3.17, NH AIAN 2.73, NH NHPI 2.11, NH Black 1.72 and NH Asian 1.72, adjusted for education, age and sex.
Education's effect is small (HS or less: 1.17; some college: 1.00). In CV, ridge (R² 0.75 ± 0.11) beats the
random forest (0.67 ± 0.19), which suggests the structure is largely additive.

> **Implementation note.** In statsmodels 0.15, `GLM(..., Binomial()).fit(scale="X2")` with two-column
> (successes, failures) endog estimates the dispersion as 0.026 instead of 54.4, which shrinks standard
> errors by a factor of about 45. `evaluation/glm.py::fit_quasi_binomial` computes the Pearson dispersion explicitly.

## 2. County model (originally "Feature Importance By County")

| # | Problem in the original | Why it matters | Fix |
|---|---|---|---|
| 2.1 | **Target is `Total deaths`**, but the paper calls it a "death rate" | Total deaths is a proxy for county population. "Large central metro" ranked first (0.44) because big counties have many deaths | Target is the county COVID share of deaths (a rate), weighted by total deaths |
| 2.2 | **Features derived from the outcome.** Race shares *of COVID deaths* were used to predict deaths | Circular: the features are computed from the target population, and they are compositional (they sum to about 1) | Use race shares of the **population** (exogenous). Keep NH White out of linear models as the reference |
| 2.3 | **`fillna(0)` on suppressed cells** (5,114 cells) | NCHS blanks counts of 1–9. Zero-filling marks small groups as having no deaths | Keep them as NaN. Disparity ratios use only unsuppressed cells, and the selection effect is documented |
| 2.4 | **Name-based inner join** to BEA income | Silently dropped 70 of 1,229 counties (BEA footnote asterisks, independent cities, Virginia combination areas) | Normalised name key with explicit handling of these cases: 1,229/1,229 matched, logged and tested |
| 2.5 | **Random split across neighbouring counties** | Spatial leakage: counties in the same state share policy, timing and reporting | `GroupKFold` by state |
| 2.6 | **Causal reading of importance.** "Asian or Hispanic people died for racial and ethnic reasons" | Importance is not an effect size, and the race shares are confounded with metro status | Disparity ratios with bootstrap CIs, plus a GLM with state-clustered SEs |

**Corrected result.** With a rate target and state-grouped CV, no model beats the baseline: R² is −0.12
for ridge and −0.16 for both the random forest and the baseline. Between-state differences dominate, and county
traits do not generalise across states. That is a real finding. The original 0.51 measured county size.
In the GLM, higher income (OR 0.83 per log unit) and a larger Hispanic (1.06 per 10 pp) or Black
(1.03 per 10 pp) population share are associated with a higher COVID share of deaths.

Disparity ratios (share of deaths ÷ share of population, death-weighted mean across counties):

| Group | COVID deaths | All-cause deaths | COVID vs all-cause |
|---|---|---|---|
| NH AIAN | 2.08 | 0.86 | 1.51 |
| NH NHPI | 2.01 | 1.02 | 1.69 |
| NH White | 1.13 | 1.29 | 0.89 |
| NH Black | 1.01 | 0.96 | 1.03 |
| Hispanic | 0.69 | 0.42 | 1.57 |
| NH Asian | 0.56 | 0.44 | 1.19 |

The crude COVID ratio for Hispanic residents (0.69) looks favourable only because the Hispanic population
is young and young people die less of every cause. Relative to all-cause mortality, Hispanic and NH AIAN
deaths are about 1.5× over-represented among COVID deaths. This matches the age-adjusted GLM above.

## 3. Risk classifier (originally "Racial Prescription analysis")

| # | Problem in the original | Why it matters | Fix |
|---|---|---|---|
| 3.1 | **Double-counted COVID deaths:** `Multiple Cause + Underlying Cause` | Multiple-cause deaths already include underlying-cause deaths (true in 100% of rows). The share was inflated about 2×, and 7 rows exceeded 100% | Use the multiple-cause count only |
| 3.2 | **Zero-death rows** produced NaN scores, which were silently labelled "not risky" | Label noise | Drop zero-death cells, and require ≥ 20 deaths per cell for a stable ratio |
| 3.3 | **Threshold = unweighted mean of the ratio over all data** | Leaks test data into label construction, and gives a 1-death cell the same weight as a 10,000-death cell | Threshold = death-weighted pooled share of the **training** set only (tested) |
| 3.4 | **Random split across months** | The same demographic cell appears in train and test in adjacent months. Adding month and year as features lifts a random forest from 0.75 to 0.89 accuracy, so pandemic waves dominate | Temporal split: train on 2020, test on 2021 (Jan–Sep) |
| 3.5 | **Accuracy with no baseline** (0.75) | The majority class already gives 0.61 | Report a majority baseline, balanced accuracy, F1, ROC AUC and Brier score, plus feature-set ablations |
| 3.6 | **Sex-only SVM** used to conclude sex "did not significantly affect" risk | The SVM predicted one class for every row, so its accuracy *is* the baseline | Removed. Sex is evaluated as an ablation (age → age+sex) |
| 3.7 | **"Bias" assessed with a heatmap** of mean labels | Descriptive base rates, not model behaviour | Per-group TPR, FPR, FNR, precision, selection rate and calibration. Demographic parity and equalized odds differences for race-aware vs race-blind models |
| 3.8 | **Paper vs code mismatch.** The paper says C was tuned by CV, sex was a feature and the kernel was chosen for non-linearity | None of these is in the code | Every number in the docs traces to a generated table in `reports/tables/` |

**Corrected result.** Race adds +0.03 ROC AUC (0.725 → 0.753). Fairness at threshold 0.5:

| Model | Demographic parity diff. | Equalized odds diff. | Notable |
|---|---|---|---|
| Race-aware (age + sex + race) | 0.65 | 0.81 | Flags 0% of NH White cells (24% are truly high-burden). Flags 55% of Hispanic and 65% of NH AIAN cells |
| Race-blind (age + sex) | 0.08 | 0.11 | Misses 64% of high-burden Hispanic cells and 56% of NH AIAN cells, vs 53% of NH White cells |

Adding race turns the model into a race lookup. Removing race equalises selection rates but
under-serves the groups with the highest burden, because base rates differ (49% Hispanic vs 24% NH White
in 2021). When base rates differ, no threshold satisfies both criteria at once, so the choice has to be
made explicitly and not left to the model.

## Unused data removed

`owid-covid-data.csv` (99 MB), `country_covid_totals.csv` and `us_racial.csv` were never used by any model.
`By Countries.ipynb` also summed non-additive columns (rates and indices) over time. These files remain
in git history.
