"""Loader for CDC deaths by educational attainment."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from covid_bias.config import (
    EDU_RACE_MAP,
    EDUCATION_LEVELS,
    FULL_NHPI,
    TRUNCATED_NHPI,
    Config,
)

log = logging.getLogger(__name__)

RAW_COLUMNS = {
    "Year": "year",
    "Education Level": "education",
    "Race or Hispanic Origin": "race",
    "Sex": "sex",
    "Age Group": "age_group",
    "Total Deaths": "total_deaths",
    "COVID-19 Deaths": "covid_deaths",
}


def _normalize_education(value: str) -> str:
    if re.match(r"^Bachelor", value):
        return "Bachelor's or more"
    if value.startswith("High school"):
        return "HS or less"
    if value.startswith("Associate"):
        return "Some college"
    return value


def _age_sort_key(label: str) -> int:
    return int(re.match(r"\d+", label).group())


def load_education(path: Path | None = None, cfg: Config | None = None) -> pd.DataFrame:
    """Load 2020 deaths by education, race, sex and age as a tidy frame."""
    cfg = cfg or Config()
    path = path or cfg.raw_dir / "education_deaths.csv"
    raw = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"education file missing columns: {sorted(missing)}")
    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS).copy()
    for col in ("education", "race", "sex", "age_group"):
        df[col] = df[col].astype(object).str.strip()
    df["education"] = df["education"].map(_normalize_education)
    df["race"] = df["race"].replace({TRUNCATED_NHPI: FULL_NHPI})

    df = df[(df["education"] != "Unknown") & (df["race"] != "Unknown")]
    df = df[df["year"] == 2020]
    n_zero = int((df["total_deaths"] == 0).sum())
    df = df[df["total_deaths"] > 0]
    log.info("education: dropped %d rows with zero total deaths", n_zero)

    unknown_edu = set(df["education"]) - set(EDUCATION_LEVELS)
    if unknown_edu:
        raise ValueError(f"unexpected education levels: {sorted(unknown_edu)}")
    unknown_race = set(df["race"]) - set(EDU_RACE_MAP)
    if unknown_race:
        raise ValueError(f"unexpected race labels: {sorted(unknown_race)}")
    if not ((df["covid_deaths"] >= 0) & (df["covid_deaths"] <= df["total_deaths"])).all():
        raise ValueError("covid_deaths must satisfy 0 <= covid_deaths <= total_deaths")

    df["race"] = df["race"].map(EDU_RACE_MAP)
    df["education"] = pd.Categorical(df["education"], categories=EDUCATION_LEVELS, ordered=True)
    ages = sorted(df["age_group"].unique(), key=_age_sort_key)
    df["age_group"] = pd.Categorical(df["age_group"], categories=ages, ordered=True)
    df["sex"] = df["sex"].astype(object)
    df["race"] = df["race"].astype(object)
    df["covid_share"] = df["covid_deaths"] / df["total_deaths"]
    df = df.reset_index(drop=True)
    key = ["education", "race", "sex", "age_group"]
    if df.duplicated(key, keep=False).any():
        raise ValueError("duplicate education cells found")
    return df
