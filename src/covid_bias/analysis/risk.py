"""Analysis 3: audit of a hypothetical screening model that flags high-COVID-burden cells."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from covid_bias.analysis import figure_path, write_table
from covid_bias.config import Config
from covid_bias.data.monthly import load_monthly
from covid_bias.evaluation.metrics import binary_metrics, fairness_summary, group_fairness_table
from covid_bias.plotting import PALETTE, new_figure, save

log = logging.getLogger(__name__)

THRESHOLD = 0.5


def build_cells(monthly: pd.DataFrame, min_deaths: int) -> pd.DataFrame:
    """Keep demographic-month cells with at least ``min_deaths`` all-cause deaths."""
    cells = monthly[monthly["all_cause"] >= min_deaths].copy()
    for col in ("sex", "race", "age_group"):
        cells[col] = cells[col].astype(str)
    return cells.reset_index(drop=True)


def compute_tau(train: pd.DataFrame) -> float:
    """Death-weighted pooled COVID share of the *training* cells."""
    return float(train["covid_deaths"].sum() / train["all_cause"].sum())


def add_label(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Add the ``high_burden`` label: cell COVID share above ``tau``."""
    out = df.copy()
    out["high_burden"] = (out["covid_share"] > tau).astype(int)
    return out


def split_temporal(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Train on 2020, test on 2021; tau and labels derive from the training data only."""
    train = cells[cells["year"] == 2020]
    test = cells[cells["year"] == 2021]
    tau = compute_tau(train)
    return add_label(train, tau), add_label(test, tau), tau


def _make_model(kind: str, features: list[str], cfg: Config) -> Pipeline | DummyClassifier:
    if kind == "majority":
        return DummyClassifier(strategy="prior")
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features)]
    )
    if kind == "logistic":
        clf = LogisticRegression(max_iter=10_000)
    else:
        clf = HistGradientBoostingClassifier(random_state=cfg.seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def _fit_predict(kind, features, train, test, cfg) -> np.ndarray:
    model = _make_model(kind, features, cfg)
    if kind == "majority":
        model.fit(train[["age_group"]], train["high_burden"])
        return model.predict_proba(test[["age_group"]])[:, 1]
    model.fit(train[features], train["high_burden"])
    return model.predict_proba(test[features])[:, 1]


SPECS = [
    ("Majority baseline", "majority", ["age_group"]),
    ("Logistic: age", "logistic", ["age_group"]),
    ("Logistic: age+sex", "logistic", ["age_group", "sex"]),
    ("Logistic: age+sex+race", "logistic", ["age_group", "sex", "race"]),
    ("Gradient boosting: age+sex+race", "hgb", ["age_group", "sex", "race"]),
]


def _plot_fairness(table: pd.DataFrame, path: Path) -> None:
    races = sorted(table["group"].unique())
    metrics = [
        ("tpr", "True positive rate"),
        ("fnr", "False negative rate"),
        ("selection_rate", "Selection rate"),
    ]
    fig, axes = new_figure(11, 3.8, ncols=3, sharey=True)
    colors = {
        "Race-aware (age+sex+race)": PALETTE["vermillion"],
        "Race-blind (age+sex)": PALETTE["blue"],
    }
    width = 0.38
    x = np.arange(len(races))
    for ax, (col, title) in zip(axes, metrics, strict=True):
        for i, (model, c) in enumerate(colors.items()):
            d = table[table["model"] == model].set_index("group").reindex(races)
            ax.bar(x + (i - 0.5) * width, d[col], width, label=model, color=c)
        ax.set_xticks(x, races, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Group fairness on the 2021 test set (threshold 0.5)", y=1.04)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, path)


def run(cfg: Config, out: Path) -> dict:
    """Run the risk-model audit and write tables and figures under ``out``."""
    monthly = load_monthly(cfg=cfg)
    cells = build_cells(monthly, cfg.min_cell_deaths)
    train, test, tau = split_temporal(cells)
    log.info(
        "risk: %d train / %d test cells, tau=%.4f, test base rate %.3f",
        len(train),
        len(test),
        tau,
        test["high_burden"].mean(),
    )

    rows, probs = [], {}
    for name, kind, feats in SPECS:
        p = _fit_predict(kind, feats, train, test, cfg)
        probs[name] = p
        m = binary_metrics(test["high_burden"], p, THRESHOLD)
        rows.append(
            {
                "model": name,
                "features": "+".join(feats) if kind != "majority" else "none",
                **m,
                "test_base_rate": float(test["high_burden"].mean()),
                "n_train": len(train),
                "n_test": len(test),
                "tau": tau,
            }
        )
    scores = write_table(pd.DataFrame(rows), out, "risk_model_scores.csv")

    audited = {
        "Race-aware (age+sex+race)": "Logistic: age+sex+race",
        "Race-blind (age+sex)": "Logistic: age+sex",
    }
    tables, summaries = [], []
    for label, name in audited.items():
        p = probs[name]
        t = group_fairness_table(
            test["high_burden"], (p >= THRESHOLD).astype(int), test["race"], y_prob=p
        )
        t.insert(0, "model", label)
        tables.append(t)
        summaries.append({"model": label, **fairness_summary(t)})
    fair = write_table(pd.concat(tables, ignore_index=True), out, "risk_group_fairness.csv")
    summ = write_table(pd.DataFrame(summaries), out, "risk_fairness_summary.csv")
    _plot_fairness(fair, figure_path(out, "risk_group_fairness.png"))

    s = scores.set_index("model")
    return {
        "n_train_cells": int(len(train)),
        "n_test_cells": int(len(test)),
        "tau": round(tau, 4),
        "train_base_rate": round(float(train["high_burden"].mean()), 3),
        "test_base_rate": round(float(test["high_burden"].mean()), 3),
        "scores": {
            m: {
                k: round(float(s.loc[m, k]), 3)
                for k in ("accuracy", "balanced_accuracy", "roc_auc")
            }
            for m in s.index
        },
        "fairness": {
            r.model: {
                "demographic_parity_difference": round(float(r.demographic_parity_difference), 3),
                "equalized_odds_difference": round(float(r.equalized_odds_difference), 3),
            }
            for r in summ.itertuples()
        },
    }
