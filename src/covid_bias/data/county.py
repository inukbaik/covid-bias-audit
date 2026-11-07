"""Loaders for CDC county deaths and BEA per-capita income, and their merge."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from covid_bias.config import COUNTY_RACE_COLUMNS, URBAN_RURAL_LEVELS, Config

log = logging.getLogger(__name__)

INDICATORS = {
    "Distribution of all-cause deaths (%)": "allcause_share",
    "Distribution of COVID-19 deaths (%)": "covid_share",
    "Distribution of population (%)": "pop_share",
}
SUFFIXES = (
    "city and borough",
    "census area",
    "municipality",
    "borough",
    "parish",
    "county",
)
# Hand-checked renames of BEA/CDC name differences that normalisation cannot resolve.
# Keys and values are *normalised* (state, name) keys.
ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("SD", "shannon"): ("SD", "oglalalakota"),
}


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def county_key(name: str, state: str) -> tuple[str, str]:
    """Normalised ``(state, name)`` join key for CDC and BEA county names.

    Lowercases, strips accents, footnote asterisks, parentheticals and punctuation, and removes
    geographic suffixes (county, parish, borough, ...). Independent cities (CDC ``Baltimore
    city``, BEA ``Baltimore (Independent City)``) get a ``#city`` marker so they do not
    collide with the same-named county.
    """
    text = _strip_accents(str(name)).lower().replace("*", "").strip()
    is_city = bool(re.search(r"\(independent city\)", text)) or (
        bool(re.search(r"\bcity$", text)) and not text.endswith("city and borough")
    )
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.strip()
    if is_city:
        text = re.sub(r"\bcity$", "", text).strip()
    else:
        for suffix in SUFFIXES:
            if text.endswith(" " + suffix):
                text = text[: -len(suffix)].strip()
                break
    text = re.sub(r"^(saint|st\.?)\s", "st", text)
    text = re.sub(r"[^a-z]", "", text)
    if is_city:
        text += "#city"
    st = str(state).strip().upper()
    return ALIASES.get((st, text), (st, text))


def load_income(path: Path | None = None, cfg: Config | None = None) -> pd.DataFrame:
    """Load BEA per-capita personal income as ``state, name, key, income`` rows.

    Virginia combination areas such as ``Albemarle + Charlottesville, VA`` are split on ``+``
    and the combined income is assigned to every component. This is an approximation: BEA
    publishes no separate estimates for the components.
    """
    cfg = cfg or Config()
    path = path or cfg.raw_dir / "county_income.csv"
    raw = pd.read_csv(path, dtype=str)
    if "Location" not in raw.columns:
        raise ValueError("income file missing 'Location' column")
    year_cols = [c for c in ("2020", "2021", "2022") if c in raw.columns]
    if len(year_cols) != 3:
        raise ValueError("income file must contain 2020, 2021 and 2022 columns")
    raw = raw[raw["Location"].str.contains(r",\s*[A-Z]{2}\*?\s*$", regex=True, na=False)].copy()
    vals = raw[year_cols].apply(pd.to_numeric, errors="coerce")
    raw["income"] = vals.mean(axis=1, skipna=False)
    rows: list[dict] = []
    for loc, income in zip(raw["Location"], raw["income"], strict=True):
        body, state = re.match(r"^(.*),\s*([A-Z]{2})\*?\s*$", loc).groups()
        if "+" in body:
            # BEA lists the county first, then one or two independent cities.
            parts = [p.strip() for p in body.split("+")]
            first = parts[0].split(",")[0].strip()
            comps = [(first, False)] + [(p, True) for p in parts[1:]]
            if "," in parts[0]:
                comps.append((parts[0].split(",", 1)[1].strip(), True))
        else:
            comps = [(body.strip(), False)]
        for comp, city in comps:
            label = f"{comp} (Independent City)" if city else comp
            rows.append(
                {
                    "state": state,
                    "name": comp,
                    "key": county_key(label, state),
                    "income": income,
                    "combined": "+" in body,
                }
            )
    out = pd.DataFrame(rows)
    # Keep the first non-missing estimate per key (e.g. Alaska pre-2019 vs new areas).
    out = out.sort_values("income", na_position="last").drop_duplicates("key")
    return out.reset_index(drop=True)


def load_county(
    path: Path | None = None,
    income_path: Path | None = None,
    cfg: Config | None = None,
    with_income: bool = True,
) -> pd.DataFrame:
    """Load county deaths (one row per county) merged with per-capita income."""
    cfg = cfg or Config()
    path = path or cfg.raw_dir / "county_deaths.csv"
    raw = pd.read_csv(path)
    needed = {"State", "County Name", "FIPS Code", "Indicator", "Total deaths", "COVID-19 Deaths"}
    needed |= set(COUNTY_RACE_COLUMNS.values()) | {"Urban Rural Description"}
    if needed - set(raw.columns):
        raise ValueError(f"county file missing columns: {sorted(needed - set(raw.columns))}")
    unknown = set(raw["Indicator"]) - set(INDICATORS)
    if unknown:
        raise ValueError(f"unexpected indicators: {sorted(unknown)}")
    if set(raw["Urban Rural Description"]) - set(URBAN_RURAL_LEVELS):
        raise ValueError("unexpected urban/rural categories")

    base = (
        raw.drop_duplicates("FIPS Code")
        .rename(
            columns={
                "FIPS Code": "fips",
                "State": "state",
                "County Name": "county_name",
                "Urban Rural Description": "urban_rural",
                "Total deaths": "total_deaths",
                "COVID-19 Deaths": "covid_deaths",
            }
        )[["fips", "state", "county_name", "urban_rural", "total_deaths", "covid_deaths"]]
        .set_index("fips")
    )
    pieces = []
    for indicator, prefix in INDICATORS.items():
        sub = raw[raw["Indicator"] == indicator].set_index("FIPS Code")
        if sub.index.duplicated().any():
            raise ValueError(f"duplicate counties for indicator {indicator!r}")
        sub = sub[list(COUNTY_RACE_COLUMNS.values())]
        sub.columns = [f"{prefix}_{k}" for k in COUNTY_RACE_COLUMNS]
        pieces.append(sub)
    wide = base.join(pieces)
    wide = wide.reset_index()
    wide["fips"] = wide["fips"].astype(int)
    wide["urban_rural"] = pd.Categorical(
        wide["urban_rural"], categories=list(URBAN_RURAL_LEVELS), ordered=True
    )
    wide["covid_share_of_deaths"] = wide["covid_deaths"] / wide["total_deaths"]
    pop_cols = [f"pop_share_{k}" for k in COUNTY_RACE_COLUMNS]
    if wide[pop_cols].isna().any().any():
        raise ValueError("population shares must not contain missing values")
    if wide["fips"].duplicated().any():
        raise ValueError("county table must have one row per fips")
    order = ["fips", "state", "county_name", "urban_rural", "total_deaths", "covid_deaths"]
    order += ["covid_share_of_deaths"]
    for k in COUNTY_RACE_COLUMNS:
        order += [f"pop_share_{k}", f"covid_share_{k}", f"allcause_share_{k}"]
    wide = wide[order]

    if not with_income:
        return wide
    return merge_income(wide, load_income(income_path, cfg))


def merge_income(counties: pd.DataFrame, income: pd.DataFrame) -> pd.DataFrame:
    """Left-join income onto counties by normalised name; log the match rate."""
    keys = [
        county_key(n, s) for n, s in zip(counties["county_name"], counties["state"], strict=True)
    ]
    lookup = income.set_index("key")["income"]
    mapped = pd.Series([lookup.get(k, np.nan) for k in keys], index=counties.index)
    out = counties.copy()
    out["income"] = mapped
    out["log_income"] = np.log(out["income"])
    n_ok = int(out["income"].notna().sum())
    log.info("income match: %d/%d counties (%.1f%%)", n_ok, len(out), 100 * n_ok / len(out))
    unmatched = out.loc[out["income"].isna(), ["county_name", "state"]]
    if len(unmatched):
        log.info(
            "unmatched counties: %s",
            "; ".join(f"{n}, {s}" for n, s in unmatched.itertuples(index=False)),
        )
    return out
