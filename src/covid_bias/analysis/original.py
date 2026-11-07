"""Analysis 4: faithful reproductions of the original (flawed) pipelines.

The goal is to quantify how misleading they were, not to endorse them. Known problems:

* education: target is a raw death *count* (driven by cell size), label-encoded categoricals,
  a single random split whose R^2 depends heavily on the seed;
* county: target is the total death count, suppressed (NaN) shares silently set to 0;
* risk: COVID deaths double counted (multiple-cause already contains underlying-cause deaths),
  a mean-score threshold, a random (leaky) split, and a sex-only model that is constant.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from covid_bias.analysis import education as education_analysis
from covid_bias.analysis import figure_path, write_table
from covid_bias.config import COUNTY_RACE_COLUMNS, Config
from covid_bias.data.monthly import MULTIPLE, UNDERLYING
from covid_bias.plotting import PALETTE, new_figure, save

log = logging.getLogger(__name__)

EDU_FEATURES = ["Education Level", "Race or Hispanic Origin", "Sex", "Age Group"]


def education_original(cfg: Config) -> dict:
    """Count-target random forest with label encoding and a single 80/20 split per seed."""
    df = pd.read_csv(cfg.raw_dir / "education_deaths.csv", encoding_errors="replace")
    df = df[df["Year"] != 2019]
    df = df[(df["Education Level"] != "Unknown") & (df["Race or Hispanic Origin"] != "Unknown")]
    df = df.drop(columns=["Data as of"]).copy()
    for col in EDU_FEATURES:
        df[col] = LabelEncoder().fit_transform(df[col])
    X, y = df[EDU_FEATURES], df["COVID-19 Deaths"]

    def split_r2(seed: int) -> float:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
        return float(r2_score(yte, rf.predict(Xte)))

    seeds = list(range(cfg.n_original_seeds))
    r2s = np.array([split_r2(s) for s in seeds])
    seed42 = split_r2(42)

    # Same model and target, properly cross-validated (unweighted R^2 on held-out folds).
    rkf = RepeatedKFold(n_splits=5, n_repeats=cfg.cv_repeats, random_state=cfg.seed)
    cv_r2 = []
    for tr, te in rkf.split(X):
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X.iloc[tr], y.iloc[tr])
        cv_r2.append(float(r2_score(y.iloc[te], rf.predict(X.iloc[te]))))
    return {
        "seed_r2": r2s,
        "cv_r2": np.array(cv_r2),
        "seed42_r2": seed42,
        "n_rows": len(df),
    }


def county_original(cfg: Config) -> dict:
    """Original county pipeline: naive string merge, zero-filled NaNs, total-death target."""
    deaths = pd.read_csv(cfg.raw_dir / "county_deaths.csv")
    deaths["Location"] = deaths["County Name"] + ", " + deaths["State"]
    deaths["Location"] = deaths["Location"].str.replace(r"\sCounty,", ",", regex=True)
    n_before = deaths["FIPS Code"].nunique()

    inc = pd.read_csv(cfg.raw_dir / "county_income.csv", dtype=str)
    for col in ("2020", "2021", "2022"):
        inc[col] = pd.to_numeric(inc[col], errors="coerce")
    inc["Income_per_Capita"] = inc[["2020", "2021", "2022"]].mean(axis=1)
    merged = pd.merge(deaths, inc[["Location", "Income_per_Capita"]], on="Location")
    n_after = merged["FIPS Code"].nunique()

    covid = merged[merged["Indicator"] == "Distribution of COVID-19 deaths (%)"]
    race_cols = list(COUNTY_RACE_COLUMNS.values())
    zero_filled = int(covid[race_cols].isna().sum().sum())
    covid = covid.fillna(0)
    X = covid[[*race_cols, "Urban Rural Description", "Income_per_Capita"]]
    y = covid["Total deaths"]
    cats = ["Urban Rural Description"]
    nums = X.columns.difference(cats).tolist()
    pre = ColumnTransformer([("num", "passthrough", nums), ("cat", OneHotEncoder(), cats)])
    pipe = Pipeline(
        [("pre", pre), ("rf", RandomForestRegressor(n_estimators=100, random_state=42))]
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    names = nums + list(pre.named_transformers_["cat"].get_feature_names_out(cats))
    imp = pd.Series(pipe["rf"].feature_importances_, index=names).sort_values(ascending=False)
    return {
        "r2": float(r2_score(yte, pipe.predict(Xte))),
        "importance": imp,
        "zero_filled_cells": zero_filled,
        "n_counties_after_merge": int(n_after),
        "n_dropped": int(n_before - n_after),
    }


def risk_original(cfg: Config) -> dict:
    """Double-counted COVID score, mean threshold, random 70/30 split, linear SVM."""
    df = pd.read_csv(cfg.raw_dir / "monthly_deaths.csv")
    df = df[df["Date Of Death Year"] != 2019].copy()
    df["Sex"] = df["Sex"].replace({"M": "Male", "F": "Female"})
    df["covid"] = df[MULTIPLE] + df[UNDERLYING]
    with np.errstate(divide="ignore", invalid="ignore"):
        score = df["covid"] / df["AllCause"]
    threshold = score.mean()
    # NaN scores (0/0) compare False, i.e. label 0, exactly as in the original code.
    label = (score > threshold).astype(int)

    X = pd.get_dummies(df[["AgeGroup", "Race/Ethnicity"]]).astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, label, test_size=0.3, random_state=42)
    svm = Pipeline([("scale", StandardScaler()), ("svc", SVC(kernel="linear"))])
    svm.fit(Xtr, ytr)
    acc = accuracy_score(yte, svm.predict(Xte))
    majority = max(yte.mean(), 1 - yte.mean())

    Xs = LabelEncoder().fit_transform(df["Sex"]).reshape(-1, 1)
    Xtr, Xte, ytr, yte = train_test_split(Xs, label, test_size=0.3, random_state=42)
    sex_pred = SVC(kernel="linear").fit(Xtr, ytr).predict(Xte)
    return {
        "accuracy": float(acc),
        "majority_baseline": float(majority),
        "n_rows": len(df),
        "n_score_gt_1": int((score > 1).sum()),
        "threshold": float(threshold),
        "sex_accuracy": float(accuracy_score(yte, sex_pred)),
        "sex_unique_predictions": int(len(np.unique(sex_pred))),
        "double_count_share_of_sum": float(df[UNDERLYING].sum() / df["covid"].sum()),
    }


def _plot_instability(edu: dict, proper_r2: float, path: Path) -> None:
    fig, ax = new_figure(7, 3.6)
    rng = np.random.default_rng(0)
    groups = [
        ("Original: one random 80/20 split\n(one point per seed)", edu["seed_r2"], "vermillion"),
        ("Same model, 5-fold CV\n(one point per fold)", edu["cv_r2"], "blue"),
    ]
    lo, hi = -1.0, 1.05
    for i, (label, vals, c) in enumerate(groups):
        n_clipped = int((vals < lo).sum())
        if n_clipped:
            ax.annotate(
                f"{n_clipped} point(s) below {lo:g}\nnot shown",
                (lo, i),
                xytext=(6, 14),
                textcoords="offset points",
                fontsize=8,
            )
        vals = vals[vals >= lo]
        ax.scatter(
            vals,
            i + rng.uniform(-0.12, 0.12, len(vals)),
            s=18,
            alpha=0.7,
            color=PALETTE[c],
            label=label,
        )
        ax.plot([np.median(vals)] * 2, [i - 0.3, i + 0.3], color="black", lw=2)
    ax.axvline(edu["seed42_r2"], color=PALETTE["vermillion"], ls=":", lw=1.5)
    ax.annotate(
        f"seed 42: {edu['seed42_r2']:.3f}\n(reported in paper)",
        (edu["seed42_r2"] + 0.02, 1.55),
        fontsize=8,
        ha="left",
    )
    ax.axvline(proper_r2, color=PALETTE["green"], ls="--", lw=1.5)
    ax.annotate(
        f"Analysis 1 (share target,\nweighted CV R$^2$): {proper_r2:.3f}",
        (proper_r2 - 0.02, -0.65),
        fontsize=8,
        ha="right",
    )
    ax.set_yticks([0, 1], [g[0] for g in groups])
    ax.set_ylim(-0.9, 1.9)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Held-out R$^2$ (random forest on COVID-19 death counts)")
    ax.set_title("Single-split R$^2$ depends on the seed")
    ax.grid(axis="y", visible=False)
    save(fig, path)


def run(cfg: Config, out: Path) -> dict:
    """Reproduce the three original pipelines; write the long-format summary table."""
    edu = education_original(cfg)
    cnty = county_original(cfg)
    risk = risk_original(cfg)

    # Properly cross-validated analysis-1 model for the comparison line in the figure.
    d = education_analysis.load_education(cfg=cfg)
    X = education_analysis.model_frame(d)
    from covid_bias.evaluation.validation import cv_scores

    model = education_analysis._models(cfg)["Random forest"]
    cv = cv_scores(
        model,
        X,
        d["covid_share"].to_numpy(),
        d["total_deaths"].to_numpy(dtype=float),
        education_analysis.splits_for(d, cfg),
    )
    proper_r2 = float(cv["r2"].mean())
    _plot_instability(edu, proper_r2, figure_path(out, "original_r2_instability.png"))

    r2s = edu["seed_r2"]
    rows = [
        ("education", "r2_seed42", edu["seed42_r2"]),
        ("education", "r2_min_over_seeds", r2s.min()),
        ("education", "r2_median_over_seeds", np.median(r2s)),
        ("education", "r2_max_over_seeds", r2s.max()),
        ("education", "n_seeds", len(r2s)),
        ("education", "r2_same_model_5fold_cv_mean", edu["cv_r2"].mean()),
        ("education", "r2_analysis1_weighted_cv_share_mean", proper_r2),
        ("county", "r2_random_split_seed42", cnty["r2"]),
        ("county", "n_suppressed_cells_filled_with_zero", cnty["zero_filled_cells"]),
        ("county", "n_counties_after_merge", cnty["n_counties_after_merge"]),
        ("county", "n_dropped", cnty["n_dropped"]),
        *[("county", f"importance_{k}", v) for k, v in cnty["importance"].items()],
        ("risk", "svm_accuracy", risk["accuracy"]),
        ("risk", "majority_baseline_accuracy", risk["majority_baseline"]),
        ("risk", "n_rows_score_gt_1", risk["n_score_gt_1"]),
        ("risk", "mean_score_threshold", risk["threshold"]),
        ("risk", "sex_only_svm_accuracy", risk["sex_accuracy"]),
        ("risk", "sex_only_unique_predicted_classes", risk["sex_unique_predictions"]),
        ("risk", "underlying_share_of_double_counted_sum", risk["double_count_share_of_sum"]),
    ]
    table = pd.DataFrame(rows, columns=["analysis", "metric", "value"])
    write_table(table, out, "original_reproduction.csv")
    return {
        "education_r2_seed42": round(edu["seed42_r2"], 3),
        "education_r2_min": round(float(r2s.min()), 3),
        "education_r2_median": round(float(np.median(r2s)), 3),
        "education_r2_max": round(float(r2s.max()), 3),
        "education_r2_same_model_cv": round(float(edu["cv_r2"].mean()), 3),
        "county_r2": round(cnty["r2"], 3),
        "county_n_after_merge": cnty["n_counties_after_merge"],
        "county_n_dropped": cnty["n_dropped"],
        "county_top_feature": str(cnty["importance"].index[0]),
        "risk_svm_accuracy": round(risk["accuracy"], 3),
        "risk_majority_baseline": round(risk["majority_baseline"], 3),
        "risk_rows_score_gt_1": risk["n_score_gt_1"],
        "risk_sex_only_accuracy": round(risk["sex_accuracy"], 3),
        "risk_sex_only_unique_predictions": risk["sex_unique_predictions"],
    }
