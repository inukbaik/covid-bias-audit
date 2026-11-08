# covid_bias.data

## Overview

Loaders for the four raw inputs in `data/raw/`: three CDC/NCHS death files and one
BEA income file. Each loader reads a CSV, renames and normalises its columns, checks
the schema and category values, and returns a tidy pandas DataFrame. Anything
unexpected (a missing column, an unknown category, a duplicate cell, or an impossible
count) raises `ValueError` instead of being dropped silently.

Column maps, category lists, reference levels, and `Config` are defined in
[`../config.py`](../config.py), not here. The analysis modules in
[`../analysis/`](../analysis/README.md) consume these loaders.

## Interface

`__all__ = [county_key, load_county, load_education, load_income, load_monthly]`

### `county.py`

- `load_county(path=None, income_path=None, cfg=None, with_income=True) -> DataFrame`:
  one row per county FIPS. Columns are `fips` (int), `state`, `county_name`,
  `urban_rural` (ordered Categorical), `total_deaths`, `covid_deaths`,
  `covid_share_of_deaths`, and then `pop_share_{k}`, `covid_share_{k}`, and
  `allcause_share_{k}` for each key in `COUNTY_RACE_COLUMNS`. When
  `with_income=True`, it adds `income` and `log_income` through `merge_income`. The
  default path is `cfg.raw_dir / "county_deaths.csv"`.
- `load_income(path=None, cfg=None) -> DataFrame`: columns `state`, `name`, `key`,
  `income`, `combined` (bool). `income` is the mean of the 2020–2022 per-capita
  values. Virginia "+" combined areas are split, and each component county gets the
  combined value. The default path is `cfg.raw_dir / "county_income.csv"`.
- `merge_income(counties, income) -> DataFrame`: left-joins income onto a copy of the
  county table on `county_key(county_name, state)`. It adds `income` (NaN when there
  is no match) and `log_income`. It is not listed in `__all__`.
- `county_key(name, state) -> tuple[str, str]`: the normalised
  `(STATE, name)` join key. It lowercases, strips accents, `*`, and text in
  parentheses, drops the first matching entry in `SUFFIXES`, rewrites
  `saint`/`st.` as `st`, and keeps only the letters a–z. Independent cities get a
  `#city` marker. The result then passes through `ALIASES`.
- Constants: `INDICATORS` maps the three CDC indicator strings to the `allcause_share`,
  `covid_share`, and `pop_share` prefixes. `SUFFIXES` lists the county-type words
  that are removed. `ALIASES` maps renamed counties, for example Shannon to Oglala
  Lakota in South Dakota.

### `education.py`

- `load_education(path=None, cfg=None) -> DataFrame`: columns `year`, `education`
  (ordered over `EDUCATION_LEVELS`), `race` (mapped through `EDU_RACE_MAP`), `sex`,
  `age_group` (ordered by its leading number), `total_deaths`, `covid_deaths`,
  `covid_share`. The default path is `cfg.raw_dir / "education_deaths.csv"`.
- `RAW_COLUMNS`: maps the seven raw CDC headers to the names above.

### `monthly.py`

- `load_monthly(path=None, cfg=None) -> DataFrame`: columns `year`, `month`, `t`
  (months since January 2020), `sex`, `race` (mapped through `MONTHLY_RACE_MAP`),
  `age_group` (ordered over `MONTHLY_AGE_GROUPS`), `all_cause`, `covid_deaths`,
  `covid_share`. The default path is `cfg.raw_dir / "monthly_deaths.csv"`.
- `MULTIPLE`, `UNDERLYING`: the CDC column names for COVID-19 (U071) as a multiple
  cause and as the underlying cause. `REQUIRED` lists the raw columns that must be
  present.

## Invariants

- **The schema is checked when the file is read.** Every loader raises `ValueError`
  if a required column is missing:
  - county: the CDC columns plus every `COUNTY_RACE_COLUMNS` value
  - income: `Location`, `2020`, `2021`, and `2022`
  - education: all of `RAW_COLUMNS`
  - monthly: all of `REQUIRED`
- **Categories form closed sets.** `ValueError` is raised for an unknown value:
  - county: `Indicator` outside `INDICATORS`, or `Urban Rural Description` outside
    `URBAN_RURAL_LEVELS`
  - education: an education level outside `EDUCATION_LEVELS`, or a race outside
    `EDU_RACE_MAP`
  - monthly: a sex other than Male or Female, a race outside `MONTHLY_RACE_MAP`, or an
    age group outside `MONTHLY_AGE_GROUPS`
- **Each cell is unique.** `ValueError` is raised on a duplicate:
  - county: a FIPS repeated within one indicator, or in the final table
  - education: a repeated `(education, race, sex, age_group)`
  - monthly: a repeated `(year, month, sex, race, age_group)`
- **Counts must be possible.** Education requires
  `0 <= covid_deaths <= total_deaths`, and monthly requires
  `covid_deaths <= all_cause`. Either violation raises `ValueError`. The county
  `pop_share_*` columns must have no NaN.
- **Some rows are filtered rather than rejected:**
  - education keeps 2020 only, and drops `Unknown` education or race and rows with
    `total_deaths <= 0`
  - monthly keeps 2020–2021 and drops rows with `all_cause <= 0`
  - income keeps rows whose `Location` ends in `, XX` (a two-letter state code)
- **Monthly COVID deaths use only the multiple-cause column.** `UNDERLYING` is never
  added in. Adding it was the double-counting error in the original paper.
- **Label fixes:** education labels are normalised, for example `Bachelor*` becomes
  `Bachelor's or more`, and the truncated NHPI label (`TRUNCATED_NHPI`) is replaced
  with `FULL_NHPI`.
- **Income rules:** income values are coerced to numbers, and the 2020–2022 mean uses
  `skipna=False`, so a missing year makes `income` NaN. When a key appears more than
  once, the code sorts by income in ascending order and keeps the first row, which is
  the *lowest* non-missing income. The comment at `county.py:112` says "first
  non-missing", which does not match this.
- **No zero guard:** `covid_share_of_deaths` in `load_county` divides by
  `total_deaths` without checking for zero.
- **No suppression handling:** the loaders contain no code for suppressed NCHS
  counts (1–9).

## Side Effects

- Disk reads only: `pd.read_csv` on `cfg.raw_dir` (`data/raw/`) or the path passed
  in. `load_county(with_income=True)` also reads the income file. The package writes
  no files and has no cache or network access.
- INFO-level logging through module loggers:
  - `merge_income`: the income match count and rate, and the unmatched counties
  - `load_education`: rows dropped for zero total deaths
  - `load_monthly`: rows dropped for zero all-cause deaths

## Verification

- Run: `make test` (runs `python -m pytest`; see `tests/test_loaders.py`)
