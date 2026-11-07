"""Analyses; each module exposes ``run(cfg, out) -> dict`` of headline results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def table_path(out: Path, name: str) -> Path:
    """Path of a CSV table under ``out/tables`` (directory created)."""
    path = Path(out) / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(out: Path, name: str) -> Path:
    """Path of a figure under ``out/figures`` (directory created)."""
    path = Path(out) / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_table(df: pd.DataFrame, out: Path, name: str) -> pd.DataFrame:
    """Write ``df`` to ``out/tables/name`` and return it."""
    df.to_csv(table_path(out, name), index=False, float_format="%.6g")
    return df
