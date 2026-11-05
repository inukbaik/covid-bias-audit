"""Analysis 2: county-level disparities, prediction and inference for COVID-19 share of deaths."""

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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from covid_bias.analysis import figure_path, write_table
from covid_bias.config import COUNTY_RACE_KEYS, COUNTY_RACE_LABELS, Config
from covid_bias.data.county import load_county
from covid_bias.evaluation.glm import fit_quasi_binomial
from covid_bias.evaluation.metrics import bootstrap_ci
from covid_bias.evaluation.validation import (
    cv_scores,
    permutation_importance_cv,
    summarise_scores,
)
from covid_bias.plotting import PALETTE, new_figure, save

log = logging.getLogger(__name__)

POP_COLS = [f"pop_share_{k}" for k in COUNTY_RACE_KEYS]
LINEAR_POP_COLS = [c for c in POP_COLS if c != "pop_share_white"]
RATIO_TYPES = {
    "covid": "COVID-19 deaths / population",
    "allcause": "All-cause deaths / population",
    "covid_vs_allcause": "COVID-19 / all-cause",
}


def disparity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Per-county, per-race disparity ratios in long format.

    NOTE (selection bias): NCHS suppresses death shares for groups with 1-9 deaths, so a ratio
    only exists for counties where the group had at least 10 deaths. Counties with small
    minority populations therefore drop out, and the surviving ones are not representative.
    """
    rows = []
    for k in COUNTY_RACE_KEYS:
        pop, cov, allc = df[f"pop_share_{k}"], df[f"covid_share_{k}"], df[f"allcause_share_{k}"]
        ok_pop = pop > 0
        frames = {
            "covid": (cov / pop).where(ok_pop & cov.notna()),
            "allcause": (allc / pop).where(ok_pop & allc.notna()),
            "covid_vs_allcause": (cov / allc).where((allc > 0) & cov.notna()),
        }
        for kind, ratio in frames.items():
            part = pd.DataFrame(
                {
                    "fips": df["fips"],
                    "race": COUNTY_RACE_LABELS[k],
                    "ratio_type": kind,
                    "ratio": ratio,
                    "covid_deaths": df["covid_deaths"],
                }
            ).dropna(subset=["ratio"])
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarise_ratios(long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Median, IQR and covid-death-weighted mean (bootstrap 95% CI over counties) per race."""
    rows = []
    for (race, kind), d in long.groupby(["race", "ratio_type"], sort=False):
        est, lo, hi = bootstrap_ci(
            d["ratio"].to_numpy(),
            d["covid_deaths"].to_numpy(dtype=float),
            n_boot=cfg.n_bootstrap,
            seed=cfg.seed,
        )
        rows.append(
            {
                "race": race,
                "ratio_type": kind,
                "n_counties": len(d),
                "median": d["ratio"].median(),
                "q25": d["ratio"].quantile(0.25),
                "q75": d["ratio"].quantile(0.75),
                "weighted_mean": est,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def _plot_ratios(table: pd.DataFrame, path: Path) -> None:
    races = [COUNTY_RACE_LABELS[k] for k in COUNTY_RACE_KEYS]
    fig, ax = new_figure(7, 4.5)
    offsets = {"covid": 0.18, "allcause": -0.18}
    colors = {"covid": PALETTE["vermillion"], "allcause": PALETTE["blue"]}
    for kind in ("covid", "allcause"):
        d = table[table["ratio_type"] == kind].set_index("race").reindex(races)
        y = np.arange(len(races))[::-1] + offsets[kind]
        ax.errorbar(
            d["weighted_mean"],
            y,
            xerr=[d["weighted_mean"] - d["ci_low"], d["ci_high"] - d["weighted_mean"]],
            fmt="o",
            color=colors[kind],
            label=RATIO_TYPES[kind],
            capsize=2,
        )
    ax.axvline(1, color=PALETTE["grey"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(races))[::-1], races)
    ax.set_xlabel("Share of deaths / share of population (log scale; death-weighted mean, 95% CI)")
    ax.set_title("County disparity ratios by race (only counties with unsuppressed data)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, path)


def model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with income available, as a plain frame for modelling."""
    d = df.dropna(subset=["log_income"]).copy()
    d["urban_rural"] = d["urban_rural"].astype(str)
    return d.reset_index(drop=True)


def _pipelines(cfg: Config) -> dict[str, tuple[Pipeline | DummyRegressor, list[str]]]:
    cats = ["urban_rural"]
    lin_nums = ["log_income", *LINEAR_POP_COLS]
    rf_nums = ["log_income", *POP_COLS]
    ridge = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                        ("num", StandardScaler(), lin_nums),
                    ]
                ),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    rf = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                        ("num", "passthrough", rf_nums),
                    ]
                ),
            ),
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
    )
    return {
        "Weighted-mean baseline": (DummyRegressor(strategy="mean"), cats + lin_nums),
        "Ridge (standardized)": (ridge, cats + lin_nums),
        "Random forest": (rf, cats + rf_nums),
    }


def fit_county_glm(d: pd.DataFrame):
    """Quasi-binomial GLM with state-clustered standard errors."""
    g = d.copy()
    g["other_deaths"] = g["total_deaths"] - g["covid_deaths"]
    for c in LINEAR_POP_COLS:
        g[c] = g[c] * 10  # odds ratio per 10 percentage points
    terms = " + ".join(LINEAR_POP_COLS)
    formula = (
        "covid_deaths + other_deaths ~ C(urban_rural, Treatment('Large central metro')) + "
        f"log_income + {terms}"
    )
    model = smf.glm(formula, data=g, family=sm.families.Binomial())
    groups = pd.factorize(g["state"])[0]
    return fit_quasi_binomial(model, cov_type="cluster", cov_kwds={"groups": groups})


def glm_table(res) -> pd.DataFrame:
    """Odds ratios with 95% CI from a county GLM."""
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
    out["term"] = (
        out["term"]
        .str.replace(r"C\(urban_rural, .*\)\[T\.(.*)\]", r"urban_rural: \1", regex=True)
        .str.replace("log_income", "log_income (per log unit)")
        .str.replace(r"pop_share_(\w+)", r"pop_share_\1 (per 10 pp)", regex=True)
    )
    return out.reset_index(drop=True)


def run(cfg: Config, out: Path) -> dict:
    """Run the county analysis and write tables and figures under ``out``."""
    df = load_county(cfg=cfg)

    # 1. Disparity ratios
    long = disparity_ratios(df)
    summary = write_table(summarise_ratios(long, cfg), out, "county_disparity_ratios.csv")
    _plot_ratios(summary, figure_path(out, "county_disparity_ratios.png"))

    # 2. Prediction with state-grouped CV
    d = model_data(df)
    n_dropped = len(df) - len(d)
    log.info("county: dropped %d counties with missing income", n_dropped)
    y = d["covid_share_of_deaths"].to_numpy()
    w = d["total_deaths"].to_numpy(dtype=float)
    splits = list(GroupKFold(n_splits=5).split(d, groups=d["state"]))
    rows, models = [], _pipelines(cfg)
    for name, (model, cols) in models.items():
        rows.append(summarise_scores(cv_scores(model, d[cols], y, w, splits), name))
    cv = write_table(pd.DataFrame(rows), out, "county_cv_scores.csv")

    rf, cols = models["Random forest"]
    imp = permutation_importance_cv(rf, d[cols], y, w, splits, seed=cfg.seed, n_repeats=5)
    imp = write_table(imp, out, "county_permutation_importance.csv")
    fig, ax = new_figure(6.5, 4)
    s = imp.sort_values("importance_mean")
    ax.barh(s["feature"], s["importance_mean"], xerr=s["importance_std"], color=PALETTE["blue"])
    ax.set_xlabel("Drop in held-out weighted R$^2$ when permuted (mean ± sd over state folds)")
    ax.set_title("Permutation importance, random forest on county COVID share")
    ax.grid(axis="y", visible=False)
    save(fig, figure_path(out, "county_permutation_importance.png"))

    # 3. Inference
    res = fit_county_glm(d)
    ors = write_table(glm_table(res), out, "county_glm_odds_ratios.csv")

    def ratio_row(race: str, kind: str) -> dict[str, float]:
        r = summary[(summary["race"] == race) & (summary["ratio_type"] == kind)].iloc[0]
        return {
            "n": int(r.n_counties),
            "weighted_mean": round(float(r.weighted_mean), 3),
            "ci_low": round(float(r.ci_low), 3),
            "ci_high": round(float(r.ci_high), 3),
        }

    rfm = cv.set_index("model").loc["Random forest"]
    ridge = cv.set_index("model").loc["Ridge (standardized)"]
    return {
        "n_counties": int(len(df)),
        "n_counties_modelled": int(len(d)),
        "income_match_rate": round(float(df["income"].notna().mean()), 4),
        "disparity_covid": {
            r: ratio_row(r, "covid") for r in (COUNTY_RACE_LABELS[k] for k in COUNTY_RACE_KEYS)
        },
        "disparity_allcause": {
            r: ratio_row(r, "allcause") for r in (COUNTY_RACE_LABELS[k] for k in COUNTY_RACE_KEYS)
        },
        "cv_r2_rf": round(float(rfm.r2_mean), 3),
        "cv_r2_ridge": round(float(ridge.r2_mean), 3),
        "cv_mae_pp_rf": round(float(rfm.mae_mean), 2),
        "permutation_importance": {
            r.feature: round(float(r.importance_mean), 3) for r in imp.itertuples()
        },
        "glm_dispersion": round(float(res.dispersion), 2),
        "glm_odds_ratios": {r.term: round(float(r.odds_ratio), 3) for r in ors.itertuples()},
    }
