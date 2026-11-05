"""Analysis 1: race and COVID-19 share of deaths, adjusted for education, age and sex."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from covid_bias.analysis import figure_path, write_table
from covid_bias.config import REF_EDUCATION, REF_RACE, REF_SEX, Config
from covid_bias.data.education import load_education
from covid_bias.evaluation.glm import fit_quasi_binomial
from covid_bias.evaluation.metrics import wilson_interval
from covid_bias.evaluation.validation import (
    cv_scores,
    permutation_importance_cv,
    summarise_scores,
)
from covid_bias.plotting import PALETTE, new_figure, plain_log_axis, save

log = logging.getLogger(__name__)

FEATURES = ["race", "education", "sex", "age_group"]


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw feature frame with plain string columns (permutable before encoding)."""
    return df[FEATURES].astype(str).reset_index(drop=True)


def pooled_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled COVID share (sum covid / sum total) per factor level with Wilson 95% CI."""
    rows = []
    for factor in FEATURES:
        for level, d in df.groupby(factor, observed=True):
            covid, total = d["covid_deaths"].sum(), d["total_deaths"].sum()
            lo, hi = wilson_interval(covid, total)
            rows.append(
                {
                    "factor": factor,
                    "level": str(level),
                    "covid_deaths": covid,
                    "total_deaths": total,
                    "pooled_share": covid / total,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def fit_glm(df: pd.DataFrame):
    """Quasi-binomial GLM (Pearson-chi2 scale) of COVID vs other deaths on the four factors."""
    d = df.copy()
    d["other_deaths"] = d["total_deaths"] - d["covid_deaths"]
    ref_age = str(d["age_group"].cat.categories[0])
    d["age_group"] = d["age_group"].astype(str)
    d["education"] = d["education"].astype(str)
    formula = (
        "covid_deaths + other_deaths ~ "
        f"C(race, Treatment('{REF_RACE}')) + C(education, Treatment(\"{REF_EDUCATION}\")) + "
        f"C(sex, Treatment('{REF_SEX}')) + C(age_group, Treatment('{ref_age}'))"
    )
    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    return fit_quasi_binomial(model)


def odds_ratio_table(res) -> pd.DataFrame:
    """Odds ratios, 95% CI and p-values from a fitted GLM (intercept excluded)."""
    ci = res.conf_int()
    out = pd.DataFrame(
        {
            "term": res.params.index,
            "odds_ratio": np.exp(res.params.values),
            "ci_low": np.exp(ci[0].values),
            "ci_high": np.exp(ci[1].values),
            "p_value": res.pvalues.values,
        }
    )
    out = out[out["term"] != "Intercept"].copy()
    parsed = out["term"].str.extract(r"^C\((\w+), .*\)\[T\.(.*)\]$")
    out["factor"] = parsed[0]
    out["level"] = parsed[1]
    return out[["factor", "level", "odds_ratio", "ci_low", "ci_high", "p_value"]].reset_index(
        drop=True
    )


def _plot_forest(ors: pd.DataFrame, path: Path) -> None:
    sub = ors[ors["factor"].isin(["race", "education"])].reset_index(drop=True)
    fig, ax = new_figure(7, 0.45 * len(sub) + 1.5)
    y = np.arange(len(sub))[::-1]
    colors = [PALETTE["blue"] if f == "race" else PALETTE["vermillion"] for f in sub["factor"]]
    for yi, row, c in zip(y, sub.itertuples(), colors, strict=True):
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=c, lw=2)
        ax.plot(row.odds_ratio, yi, "o", color=c)
    ax.axvline(1, color=PALETTE["grey"], ls="--", lw=1)
    ax.set_xscale("log")
    plain_log_axis(ax)
    ax.set_yticks(y, [f"{r.level} ({r.factor})" for r in sub.itertuples()])
    ax.set_xlabel("Odds ratio of COVID-19 among all deaths (log scale, 95% CI)")
    ax.set_title(
        f"Adjusted odds ratios (refs: {REF_RACE}, {REF_EDUCATION}, {REF_SEX}, youngest age)"
    )
    ax.grid(axis="y", visible=False)
    save(fig, path)


def _models(cfg: Config) -> dict[str, Pipeline | DummyRegressor]:
    def pre() -> ColumnTransformer:
        return ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)], remainder="drop"
        )

    return {
        "Weighted-mean baseline": DummyRegressor(strategy="mean"),
        "Ridge (one-hot)": Pipeline([("pre", pre()), ("model", Ridge(alpha=1.0))]),
        "Random forest": Pipeline(
            [
                ("pre", pre()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=cfg.rf_estimators,
                        min_samples_leaf=3,
                        random_state=cfg.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def splits_for(df: pd.DataFrame, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated 5-fold splits (deterministic)."""
    rkf = RepeatedKFold(n_splits=5, n_repeats=cfg.cv_repeats, random_state=cfg.seed)
    return list(rkf.split(df))


def run(cfg: Config, out: Path) -> dict:
    """Run the education analysis and write tables and figures under ``out``."""
    df = load_education(cfg=cfg)
    log.info("education: %d cells, %d total deaths", len(df), df["total_deaths"].sum())

    write_table(pooled_shares(df), out, "education_pooled_share.csv")

    res = fit_glm(df)
    ors = write_table(odds_ratio_table(res), out, "education_glm_odds_ratios.csv")
    _plot_forest(ors, figure_path(out, "education_odds_ratios.png"))
    dispersion = float(res.dispersion)

    X = model_frame(df)
    y = df["covid_share"].to_numpy()
    w = df["total_deaths"].to_numpy(dtype=float)
    splits = splits_for(df, cfg)
    rows = []
    for name, model in _models(cfg).items():
        rows.append(summarise_scores(cv_scores(model, X, y, w, splits), name))
    cv = write_table(pd.DataFrame(rows), out, "education_cv_scores.csv")

    imp = permutation_importance_cv(
        _models(cfg)["Random forest"], X, y, w, splits, seed=cfg.seed, n_repeats=3
    )
    imp = write_table(imp, out, "education_permutation_importance.csv")
    fig, ax = new_figure(6, 3.5)
    d = imp.sort_values("importance_mean")
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color=PALETTE["blue"])
    ax.set_xlabel("Drop in held-out weighted R$^2$ when permuted (mean ± sd over folds)")
    ax.set_title("Permutation importance, random forest on COVID share")
    ax.grid(axis="y", visible=False)
    save(fig, figure_path(out, "education_permutation_importance.png"))

    def odds(factor: str, level: str) -> dict[str, float]:
        r = ors[(ors["factor"] == factor) & (ors["level"] == level)].iloc[0]
        return {
            "or": round(float(r.odds_ratio), 3),
            "ci_low": round(float(r.ci_low), 3),
            "ci_high": round(float(r.ci_high), 3),
        }

    rf = cv.set_index("model").loc["Random forest"]
    return {
        "n_cells": int(len(df)),
        "total_deaths": int(df["total_deaths"].sum()),
        "covid_deaths": int(df["covid_deaths"].sum()),
        "glm_dispersion": round(dispersion, 2),
        "or_race": {lvl: odds("race", lvl) for lvl in sorted(set(df["race"]) - {REF_RACE})},
        "or_education": {lvl: odds("education", lvl) for lvl in ("HS or less", "Some college")},
        "cv_r2_rf": round(float(rf.r2_mean), 3),
        "cv_r2_rf_std": round(float(rf.r2_std), 3),
        "cv_mae_pp_rf": round(float(rf.mae_mean), 2),
        "permutation_importance": {
            r.feature: round(float(r.importance_mean), 3) for r in imp.itertuples()
        },
    }
