# Errata

**Paper:** I. Baik, R. Reagan-Redko and E. Hewitt, *AI's Racism: How Racially Biased Data Affect Machine
Learning*, EST 441 Interdisciplinary Senior Project, Stony Brook University, May 2024
([PDF](ai-racism-final-paper.pdf)).

**Errata date:** November 2026

**Corrected results:** commit `f820754` of this repository. The original code is at commit `00b0933`.

<!-- TODO before publishing: co-author review and sign-off (R. Reagan-Redko, E. Hewitt). -->

Page numbers refer to the PDF. Section numbers are as printed in the paper. (The paper has two sections
numbered III and none numbered IV; see D6.)

## Summary

A rebuild of the paper's analyses ([methodology audit](../methodology.md)) found errors that invalidate
most of the paper's quantitative results. Please do not cite the following:

- **R² = 0.918 for Model 1 (p. 10).** This came from one lucky train/test split of 252 rows. With the
  same model and 50 different splits, R² ranges from −0.53 to 0.99.
- **R² = 0.51 for Model 2 and "large central metro" as the most important factor (pp. 10–12).** The
  target was the county's *all-cause* death total, so the model was learning county size. Rebuilt with
  a COVID death rate as the target and validated across states, no model beats a simple baseline.
- **"Asian or Hispanic people died for racial and ethnic reasons" (p. 10).** This is a causal claim
  drawn from feature importance, and the features were computed from the outcome itself.
- **75% classifier accuracy (pp. 14–15).** Predicting the majority class for everyone already scores
  61% on the same split, and the risk label double-counted COVID deaths.
- **"Gender did not significantly affect" risk (p. 14).** The sex-only SVM predicted the same class
  for every row, so its accuracy was the baseline.
- **Several methods described in the paper do not match the code**, including how C was tuned, which
  features were used, and what Fig. 6 shows (R3–R5, R10).

Some conclusions do survive correction, a few of them in stronger form (see [What still holds](#what-still-holds)).
The central finding is clearer after correction than before: **race is strongly associated with COVID-19
mortality after adjusting for education, age and sex.**

## Status key

| Status | Meaning |
|---|---|
| **Error** | A factual or computational mistake |
| **Not supported** | The number or conclusion came from a flawed method and should not be relied on |
| **Reversed** | The corrected analysis contradicts the claim |
| **Paper ≠ code** | The paper describes something the code did not do |
| **Holds** | The claim survives correction (the evidence may still have been flawed) |
| **Editorial** | Citation, caption or typo |

Evidence paths are relative to the repository root. "M x.y" refers to item x.y in
[docs/methodology.md](../methodology.md).

## Model 1: Feature importance, education × race (pp. 5–10)

| # | Page | Paper says | Status | Correction | Evidence |
|---|---|---|---|---|---|
| E1 | 8, 9 | The target is "the deaths", and the results describe "the mortality rate of the coronavirus" | **Error** | The target was the raw COVID-19 death *count* per demographic cell, not a rate. Counts mostly reflect how large each cell is. The corrected target is the COVID share of deaths, weighted by total deaths | M 1.2 |
| E2 | 9 | "The age group had the greatest influence" | **Not supported** | This came from in-sample impurity (MDI) importance on a count target, with race and sex label-encoded as if they were ordered numbers. On the share target with held-out permutation importance, race ranks first (1.62 ± 0.84) and age second (0.98 ± 0.87). Sex and education are about 0 | `reports/tables/education_permutation_importance.csv`; M 1.3, 1.4 |
| E3 | 9 | Education had "an unexpectedly lower influence than race" | **Holds** | Confirmed, and stronger. In a quasi-binomial GLM, HS or less vs bachelor's+ gives OR 1.17 (95% CI 1.09–1.26) and some college gives 1.00. Race ORs vs NH White range from 1.72 to 3.17 | `reports/tables/education_glm_odds_ratios.csv` |
| E4 | 10 | R² = 0.9178, "strong predictive power" | **Not supported** | One 80/20 split of 252 rows (51 test rows). Over 50 split seeds, R² ranges from −0.53 to 0.99 (median 0.81). Corrected repeated 5-fold CV on the share target gives: random forest 0.67 ± 0.19, ridge 0.75 ± 0.11, baseline −0.04 | `reports/tables/original_reproduction.csv`, `reports/tables/education_cv_scores.csv`; M 1.1 |
| E5 | 9–10 | MAE = 221.31, MSE = 212,519.90 | **Not supported** | Error on a count scale, with no baseline to compare against. On the corrected target, the random forest MAE is 1.89 percentage points vs 4.21 for the baseline | `reports/tables/education_cv_scores.csv`; M 1.5 |
| E6 | 5, 6 | Feature importance shows how much race "has a significant impact", and "gives a clear indication of how each feature contributes to the accuracy of the model" | **Error** | MDI measures impurity reduction on training data and is biased. It does not measure contribution to accuracy (permutation importance does) or statistical significance. Effect sizes should come from a model with confidence intervals, such as the GLM in E3 | M 1.4 |
| E7 | 7 | Hypothesis: "Race would not have had much of an impact on the coronavirus" | **Reversed** | After adjusting for education, age and sex, the odds that a death was due to COVID-19 (vs NH White) are 3.17× for Hispanic (2.94–3.41), 2.73× for NH AIAN, 2.11× for NH NHPI, and 1.72× for NH Black and NH Asian decedents | `reports/metrics.json` → `education.or_race` |

## Model 2: Feature importance by county (pp. 8, 10–12)

| # | Page | Paper says | Status | Correction | Evidence |
|---|---|---|---|---|---|
| C1 | 8, 11 | "The number of deaths was used as a target feature" to identify features of coronavirus death by county. Fig. 4 is titled "COVID-19 Death Rates" | **Error** | The target was `Total deaths`: the county's all-cause death total, which is neither COVID-specific nor a rate. It is essentially a proxy for county population. The corrected target is the county COVID share of deaths | `data/raw/county_deaths.csv`; M 2.1 |
| C2 | 8 | Race is used as a contributing feature | **Error** | The race columns were each group's share *of COVID-19 deaths* (the code filters to `Distribution of COVID-19 deaths (%)`), so features were derived from the outcome population. The paper does not disclose this. The corrected analysis uses each group's share of the *population* | M 2.2 |
| C3 | 8 | Empty cells "referred to as 0, so they were filled with zero" | **Error** | NCHS leaves cells blank when the count is 1–9 (suppression). They are not zeros. 5,114 suppressed cells were set to 0 | `reports/tables/original_reproduction.csv`; M 2.3 |
| C4 | n/a | (Not reported) | **Error (omission)** | Merging income data by county name silently dropped 70 of 1,229 counties. The corrected join matches 1,229/1,229 | M 2.4 |
| C5 | 10 | "Large central metro" is most important: "the more downtown the county, the more COVID virus spreads" | **Not supported** | Large-metro counties rank first because they have the most deaths of any cause. In the corrected GLM, no urban–rural category differs significantly from large central metro (ORs 0.98–1.10, all p > 0.08) | `reports/tables/county_glm_odds_ratios.csv` |
| C6 | 10 | "Asian or Hispanic people died for racial and ethnic reasons" | **Reversed / withdrawn** | Feature importance cannot show cause, and the features were circular (C2). Measured against population share, NH Asian (0.56) and Hispanic (0.69) residents are *under*-represented among COVID deaths. Measured against all-cause deaths, they are over-represented (1.19 and 1.57), largely because the Hispanic population is younger. A higher Hispanic population share is associated with a higher county COVID share of deaths (OR 1.06 per 10 pp, p < 0.001). The Asian population share is not (OR 1.06, CI 0.94–1.21). None of these analyses identifies a mechanism | `reports/tables/county_disparity_ratios.csv`, `reports/tables/county_glm_odds_ratios.csv`; M 2.6 |
| C7 | 12, 15 | R² = 0.51. The remaining error suggests underfitting or missing predictors, to be fixed with more model complexity | **Not supported** | The 0.51 came from a random split on a size proxy. With a rate target and cross-validation grouped by state, no model beats the baseline (ridge −0.12, random forest −0.16, baseline −0.16). Differences between states dominate, and adding complexity would not help | `reports/tables/county_cv_scores.csv`; M 2.5 |
| C8 | 11–12 | MAE ≈ 4,546, MSE ≈ 3.3 × 10⁸, and residuals that grow with predicted deaths | **Not supported** | These follow from the count target: errors scale with county size. They say nothing about COVID mortality | M 2.1 |

## Patient classification model (pp. 13–15)

| # | Page | Paper says | Status | Correction | Evidence |
|---|---|---|---|---|---|
| R1 | 13 | Deaths "from COVID-19 alone and … from COVID-19 and other complications were all treated as COVID-19-induced deaths" | **Error** | The code *added* multiple-cause and underlying-cause COVID counts. In every row the underlying-cause deaths are already included in the multiple-cause count, so COVID deaths were roughly doubled. 7 rows had a COVID share above 100% | `reports/tables/original_reproduction.csv`; M 3.1 |
| R2 | 13 | (Label definition not described) | **Error (omission)** | A cell was labelled "risky" if its COVID share exceeded the unweighted mean share across *all* data (0.189), test rows included. Zero-death rows became NaN and were silently labelled not risky | M 3.2, 3.3 |
| R3 | 13 | Features are age group, gender and race | **Paper ≠ code** | The main SVM used age group and race only. Sex was used alone in a separate SVM | M 3.8 |
| R4 | 13–14 | "Regularization parameter C optimized through cross-validation" | **Paper ≠ code** | No tuning was done. C was left at the default (1.0) | M 3.8 |
| R5 | 13 | SVM chosen to capture effects that "may not be linear" | **Paper ≠ code** | This contradicts the linear kernel described on the same page, which is what the code used | M 3.8 |
| R6 | 13 | A random 70/30 split ensures "a robust assessment" | **Not supported** | Rows are month × demographic cells, so the same cell in adjacent months falls on both sides of a random split. The corrected analysis trains on 2020 and tests on 2021 | M 3.4 |
| R7 | 14, 15 | Accuracy = 0.75 | **Not supported** | Predicting the majority class gives 0.61 on the same split (the paper notes on p. 14 that a benchmark is needed). With the corrected label and the temporal split, the best model reaches ROC AUC 0.76 and accuracy 0.68, vs 0.63 for the majority baseline | `reports/tables/risk_model_scores.csv`; M 3.5 |
| R8 | 14 | "Gender did not significantly affect the classification of high-risk groups" | **Evidence invalid; conclusion roughly holds** | The sex-only SVM predicted a single class for every row, so its 0.61 accuracy *is* the baseline. In the corrected models, adding sex to age does not improve performance (ROC AUC 0.730 → 0.725) | `reports/tables/risk_model_scores.csv`; M 3.6 |
| R9 | 14 | Older patients are classified as high risk | **Holds** | Age is the strongest single predictor (age-only ROC AUC 0.73) | `reports/tables/risk_model_scores.csv` |
| R10 | 14 | Fig. 6: "Classification model without and with the race indicator" | **Error** | Fig. 6 shows heatmaps of the *mean training label* (base rates) by age × race and age × sex. It does not show model predictions | M 3.7 |
| R11 | 14 | Non-Hispanic Whites are classified as low risk at all ages and may be filtered out of selective treatment | **Holds (for a race-aware model)** | The paper's evidence (Fig. 6) did not show this, but the corrected analysis does. A race-aware model flags 0% of NH White cells, although 24% are truly high-burden. Dropping race does not fix it: the race-blind model misses 64% of high-burden Hispanic cells vs 53% of NH White cells. Equalized odds difference is 0.81 (race-aware) vs 0.11 (race-blind) | `reports/tables/risk_group_fairness.csv`, `reports/tables/risk_fairness_summary.csv` |
| R12 | 15 | 75% accuracy shows "a significant but incomplete ability to predict risk", and "a better patient classification model can be developed with this 75% indicator" | **Withdrawn** | See R7 | |

## Conclusion (pp. 14–15)

| # | Page | Paper says | Status | Correction |
|---|---|---|---|---|
| K1 | 15 | "Race and socioeconomic factors also play an important role" | **Partly holds** | Race: yes, strongly (E7). Socioeconomic factors: weakly. Education has a small effect (E3). At county level, higher income per capita is associated with a lower COVID share of deaths (OR 0.83 per log unit, CI 0.73–0.94) |
| K2 | 15 | Model 2's errors call for more features and a more complex model | **Not supported** | See C7 |

## Citations, figures and editorial

| # | Page | Paper says | Status | Correction |
|---|---|---|---|---|
| D1 | 7, 8 | Cites [19], [20] for NCHS data and [21] for BEA (p. 7). Cites [20] for the county death dataset and [21] for BEA income (p. 8) | **Editorial** | In the reference list, [20] is BEA and [21] is the NCHS county dataset. The county death data is [21] and the income data is [20] |
| D2 | 17 | Reference [20]: BEA, "GDP by County, Metro, and Other Areas" | **Editorial** | The data used was per-capita personal income, not GDP. Correct source: BEA, *Personal Income by County, Metro, and Other Areas*, https://www.bea.gov/data/income-saving/personal-income-county-metro-and-other-areas |
| D3 | 8 | "home per capita" | **Editorial** | "income per capita" |
| D4 | 11 | Fig. 4 caption: "Result of Feature Importance Model 1" | **Editorial** | "Model 2" |
| D5 | 8 | Python 3.8, Pandas 1.2, Scikit-learn 0.24 | **Error** | The county script calls `get_feature_names_out`, which appeared in scikit-learn 1.0, so it could not have run on 0.24. The actual environment was not recorded. The rebuild requires Python ≥ 3.10, pandas ≥ 2.0 and scikit-learn ≥ 1.3 (`pyproject.toml`) |
| D6 | 5 | Section numbering: "III. Literature Review" followed by "III. Theoretical Framework", then V | **Editorial** | The Theoretical Framework should be section IV |

## What still holds

- Race is strongly associated with COVID-19 mortality after adjusting for education, age and sex. This is
  clearer after correction than in the paper (E7).
- Education explains less than race (E3).
- Age is the dominant predictor in the classification model, and sex adds little (R8, R9).
- A race-aware screening model would under-select non-Hispanic White patients (R11). The corrected
  analysis also shows the reverse problem for a race-blind model, so neither choice is neutral.
- The paper's broader argument stands: bias can enter machine learning at data collection, modelling and
  deployment, and simply removing race does not remove bias. The errors above are themselves examples
  of modelling-stage bias.

## Reproducing the corrections

```bash
make install
make reproduce   # writes every table cited above to reports/
```

The original scripts are at commit `00b0933` (`git show 00b0933:"Feature Importance_SEP + Race.py"`). The
Model 1 script writes `procdata.csv` to the repository root but reads it from `datasets/`, so it does not run
from a clean checkout. `src/covid_bias/analysis/original.py` re-implements all three original pipelines
faithfully, and is the source of every "original" number above.
