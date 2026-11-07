"""Loader for CDC monthly provisional death counts by sex, age and race."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from covid_bias.config import MONTHLY_AGE_GROUPS, MONTHLY_RACE_MAP, Config

log = logging.getLogger(__name__)

MULTIPLE = "COVID-19 (U071, Multiple Cause of Death)"
UNDERLYING = "COVID-19 (U071, Underlying Cause of Death)"
REQUIRED = [
    "Date Of Death Year",
    "Date Of Death Month",
    "Sex",
    "Race/Ethnicity",
    "AgeGroup",
    "AllCause",
    MULTIPLE,
]


def load_monthly(path: Path | None = None, cfg: Config | None = None) -> pd.DataFrame:
    """Load 2020-2021 monthly cells.

    ``covid_deaths`` uses the *multiple cause* column only: it already contains deaths
    where COVID-19 was the underlying cause, so adding both would double count.
    """
    cfg = cfg or Config()
    path = path or cfg.raw_dir / "monthly_deaths.csv"
    raw = pd.read_csv(path)
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"monthly file missing columns: {sorted(missing)}")
    df = pd.DataFrame(
        {
            "year": raw["Date Of Death Year"],
            "month": raw["Date Of Death Month"],
            "sex": raw["Sex"].astype(object).replace({"M": "Male", "F": "Female"}),
            "race": raw["Race/Ethnicity"].astype(object),
            "age_group": raw["AgeGroup"].astype(object),
            "all_cause": raw["AllCause"],
            "covid_deaths": raw[MULTIPLE],
        }
    )
    df = df[df["year"].isin([2020, 2021])]
    if not set(df["sex"]) <= {"Male", "Female"}:
        raise ValueError(f"unexpected sex values: {sorted(set(df['sex']))}")
    if not set(df["race"]) <= set(MONTHLY_RACE_MAP):
        raise ValueError(f"unexpected race values: {sorted(set(df['race']))}")
    if not set(df["age_group"]) <= set(MONTHLY_AGE_GROUPS):
        raise ValueError(f"unexpected age groups: {sorted(set(df['age_group']))}")
    df["race"] = df["race"].map(MONTHLY_RACE_MAP)
    n_zero = int((df["all_cause"] == 0).sum())
    df = df[df["all_cause"] > 0]
    log.info("monthly: dropped %d rows with zero all-cause deaths", n_zero)
    if (df["covid_deaths"] > df["all_cause"]).any():
        raise ValueError("covid_deaths exceeds all_cause deaths")
    df["t"] = (df["year"] - 2020) * 12 + df["month"] - 1
    df["age_group"] = pd.Categorical(
        df["age_group"], categories=list(MONTHLY_AGE_GROUPS), ordered=True
    )
    df["covid_share"] = df["covid_deaths"] / df["all_cause"]
    key = ["year", "month", "sex", "race", "age_group"]
    if df.duplicated(key).any():
        raise ValueError("duplicate monthly cells found")
    cols = [
        "year",
        "month",
        "t",
        "sex",
        "race",
        "age_group",
        "all_cause",
        "covid_deaths",
        "covid_share",
    ]
    return df[cols].reset_index(drop=True)
