# covid_bias

## Overview

The installable package behind the `covid-bias` command (distribution
`covid-bias-audit` 0.1.0, Python 3.10 or later). The top-level files hold the CLI, a
frozen run configuration with the shared label and level constants, and the shared
matplotlib style. The analysis work lives in three subpackages:
`data` → `evaluation` → `analysis`.

## Interface

### Subpackages

- [`analysis/`](./analysis/README.md): one module per analysis (`education`,
  `county`, `risk`, `original`). Each exposes `run(cfg, out) -> dict` and writes
  tables and figures.
- [`data/`](./data/README.md): validated loaders for the four raw CSVs
  (`load_county`, `load_income`, `load_education`, `load_monthly`).
- [`evaluation/`](./evaluation/README.md): the quasi-binomial GLM wrapper, weighted
  and fairness metrics, and cross-validation and permutation-importance helpers.

### `cli.py`

- `main(argv=None) -> int`: the entry point for the `covid-bias` console script and
  for `python -m covid_bias` (through `__main__.py`). It returns 0.
- `build_parser()`: one required subcommand, `run`, which takes these flags:
  - `--analysis {education,county,risk,original,all}` (default `all`, which runs them
    in that order)
  - `--output PATH` (default `Config().reports_dir`)
  - `--fast`, which applies `Config.fast_mode()`
  - `-v/--verbose`, which sets the log level to INFO
- `run_analyses(names, cfg, out) -> dict`: calls `ANALYSES[name](cfg, out)` for each
  name and merges the results into `out/metrics.json`.
- `ANALYSES`: maps each analysis name to its `analysis.<name>.run`.

### `config.py`

- `@dataclass(frozen=True) Config`: fields `root`, `seed=42`, `min_cell_deaths=20`,
  `n_bootstrap=2000`, `cv_repeats=10`, `n_original_seeds=50`, `rf_estimators=300`,
  `fast=False`.
  - Properties: `raw_dir` (`root/data/raw`) and `reports_dir` (`root/reports`).
  - `fast_mode() -> Config` returns a copy with `n_bootstrap=100`, `cv_repeats=1`,
    `n_original_seeds=5`, `rf_estimators=40`, and `fast=True`.
- `find_repo_root(start=None) -> Path`: the first ancestor folder that contains both
  `pyproject.toml` and `data/`. If none does, it returns `Path.cwd()`.
- Label and level constants:
  - race maps: `EDU_RACE_MAP`, `MONTHLY_RACE_MAP`, `COUNTY_RACE_COLUMNS`,
    `COUNTY_RACE_KEYS`, `COUNTY_RACE_LABELS`
  - level lists: `EDUCATION_LEVELS`, `URBAN_RURAL_LEVELS`, `MONTHLY_AGE_GROUPS`
  - reference levels: `REF_RACE="NH White"`, `REF_EDUCATION="Bachelor's or more"`,
    `REF_SEX="Female"`
  - NHPI label fix: `TRUNCATED_NHPI` and `FULL_NHPI`
- Module constants `SEED`, `MIN_CELL_DEATHS`, `N_BOOTSTRAP`, and `CV_REPEATS` repeat
  the `Config` defaults.

### `plotting.py`

- `PALETTE` (8 Okabe-Ito colours), `COLORS`, and `DPI = 200`.
- `apply_style()`: updates `plt.rcParams`.
- `new_figure(width=7.0, height=4.5, **kw)`: applies the style and returns
  `plt.subplots(...)`.
- `plain_log_axis(ax, axis="x", ticks=...)`: plain-number tick labels on an axis that
  is already log-scaled. It does not set the log scale itself.
- `save(fig, path) -> Path`: creates the parent directory, saves at `DPI`, and closes
  the figure.

### `__init__.py`

- `__version__ = "0.1.0"`

## Invariants

- `Config` is immutable. Variants are made with `dataclasses.replace`, as
  `fast_mode` does. Every seed in the analyses comes from `Config.seed`, which
  defaults to 42.
- **Constants live in one place.** Category labels, level orders, and reference
  levels are defined only in `config.py`. Both `data/` and `analysis/` import them
  from here.
- **Import direction:** `cli` → `analysis` → (`data`, `evaluation`, `plotting`,
  `config`). The top-level files import nothing from `data` or `evaluation`.
- **CLI argument rules:** `run` is required, and `--analysis` accepts only the four
  names or `all`.
- **`metrics.json` handling:** if the existing file is not valid JSON,
  `run_analyses` starts from an empty dict instead of raising. Values are serialised
  with `default=float`.
- **Repo root fallback:** `find_repo_root` never raises. Outside the repository it
  falls back to the current folder, so `raw_dir` may not exist.

## Side Effects

- `run_analyses` reads `out/metrics.json` if it exists and merges the new results
  over it, so keys from earlier runs are kept. It then writes the file back with
  `indent=2` and creates `out/` if needed.
- `main` configures root logging (WARNING, or INFO with `-v`) and prints
  `Wrote results for <names> to <out>`.
- `apply_style` and `new_figure` change matplotlib's global `rcParams`. No backend
  is set anywhere in the package. `make reproduce` sets `MPLBACKEND=Agg` instead.
- `save` writes figure files and closes the figure.
- The analysis subpackage does all the table and figure writing (see
  [`analysis/README.md`](./analysis/README.md)).

## Verification

- Run: `make test` (runs `python -m pytest`)
- End-to-end: `covid-bias run --analysis all --fast --output <tmpdir>`
