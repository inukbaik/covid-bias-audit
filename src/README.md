# src

## Overview

The source root for the `src/` layout. It holds a single installable package,
`covid_bias`, which `pyproject.toml` declares and `make install`
(`pip install -e ".[dev]"`) installs.

## Subdirectories & Boundaries

- covid_bias: [`./covid_bias/README.md`](./covid_bias/README.md) - The `covid-bias`
  command, the frozen `Config`, the shared plotting style, and three subpackages:
  validated raw-data loaders (`data`), metrics, CV, and GLM helpers (`evaluation`),
  and the four analyses (`analysis`). The analyses write `reports/tables`,
  `reports/figures`, and `reports/metrics.json`.
