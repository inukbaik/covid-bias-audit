import numpy as np
import pandas as pd
import pytest

from covid_bias.config import COUNTY_RACE_KEYS, EDUCATION_LEVELS, URBAN_RURAL_LEVELS
from covid_bias.data import county_key, load_education, load_income
from covid_bias.data.monthly import MULTIPLE

MIN_INCOME_MATCH = 0.99  # achieved: 100% (1229/1229), minus ~1pp


def test_education_schema(education):
    cols = ["year", "education", "race", "sex", "age_group", "total_deaths", "covid_deaths"]
    assert set(cols + ["covid_share"]) <= set(education.columns)
    assert list(education["education"].cat.categories) == list(EDUCATION_LEVELS)
    assert education["education"].notna().all()


def test_education_cleaning(education):
    assert (education["year"] == 2020).all()
    assert not education["race"].str.contains("Unknown").any()
    assert "Unknown" not in set(education["education"].astype(str))
    assert not any(r.endswith("Islande") for r in education["race"])
    assert "NH NHPI" in set(education["race"])
    assert "Bachelor's or more" in set(education["education"].astype(str))
    assert (education["total_deaths"] > 0).all()
    assert (education["covid_deaths"] <= education["total_deaths"]).all()
    assert education["covid_share"].between(0, 1).all()
    assert not education.duplicated(["education", "race", "sex", "age_group"]).any()


def test_education_schema_violation(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Year": [2020]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_education(bad)


def test_monthly_uses_multiple_cause_only(monthly, cfg):
    raw = pd.read_csv(cfg.raw_dir / "monthly_deaths.csv")
    raw = raw[(raw["Date Of Death Year"] >= 2020) & (raw["AllCause"] > 0)]
    assert monthly["covid_deaths"].sum() == raw[MULTIPLE].sum()
    assert len(monthly) == len(raw)


def test_monthly_schema(monthly):
    assert set(monthly["year"]) == {2020, 2021}
    assert set(monthly["sex"]) == {"Female", "Male"}
    assert monthly["t"].min() == 0 and monthly["t"].max() == 20
    assert (monthly["all_cause"] > 0).all()
    assert (monthly["covid_deaths"] <= monthly["all_cause"]).all()
    assert monthly["age_group"].cat.ordered
    assert monthly["race"].nunique() == 6
    assert monthly["age_group"].nunique() == 10
    assert not monthly.duplicated(["year", "month", "sex", "race", "age_group"]).any()


def test_county_one_row_per_fips(county):
    assert county["fips"].is_unique
    assert len(county) == 1229
    assert list(county["urban_rural"].cat.categories) == list(URBAN_RURAL_LEVELS)
    for k in COUNTY_RACE_KEYS:
        for prefix in ("pop_share", "covid_share", "allcause_share"):
            assert f"{prefix}_{k}" in county.columns


def test_suppressed_shares_stay_nan(county):
    assert county["pop_share_black"].notna().all()
    assert county["covid_share_nhpi"].isna().mean() > 0.5
    assert county["covid_share_aian"].isna().any()
    assert (
        county["covid_share_of_deaths"] == county["covid_deaths"] / county["total_deaths"]
    ).all()


def test_income_match_rate(county):
    rate = county["income"].notna().mean()
    assert rate >= MIN_INCOME_MATCH
    assert np.allclose(county["log_income"].dropna(), np.log(county["income"].dropna()))


def test_county_key_handles_known_cases():
    assert county_key("Baltimore city", "MD") == county_key("Baltimore (Independent City)", "MD")
    assert county_key("Baltimore County", "MD") != county_key("Baltimore city", "MD")
    assert county_key("Doña Ana County", "NM") == county_key("Dona Ana", "NM")
    assert county_key("St. Louis County", "MO") == county_key("St Louis", "MO")
    assert county_key("Anchorage Municipality", "AK") == county_key("Anchorage", "AK")
    assert county_key("Orleans Parish", "LA") == county_key("Orleans", "LA")


def test_virginia_combination_areas_split(cfg):
    inc = load_income(cfg=cfg)
    a = inc.set_index("key")["income"]
    assert a[county_key("Albemarle County", "VA")] == a[county_key("Charlottesville city", "VA")]
