"""Cross-validation helpers returning fold-level scores and held-out permutation importance."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from covid_bias.evaluation.metrics import weighted_mae, weighted_r2


def _fit(model: BaseEstimator, X, y, w) -> BaseEstimator:
    """Fit a (possibly Pipeline) estimator passing sample weights to its final step."""
    model = clone(model)
    if hasattr(model, "steps"):
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)
    return model


def cv_scores(
    model: BaseEstimator,
    X: pd.DataFrame,
    y,
    weights,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    scale: float = 100.0,
) -> pd.DataFrame:
    """Held-out weighted R^2 and weighted MAE (in ``scale`` units, e.g. pp) for each fold."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    rows = []
    for k, (tr, te) in enumerate(splits):
        m = _fit(model, X.iloc[tr], y[tr], w[tr])
        pred = m.predict(X.iloc[te])
        rows.append(
            {
                "fold": k,
                "r2": weighted_r2(y[te], pred, w[te]),
                "mae": weighted_mae(y[te], pred, w[te]) * scale,
            }
        )
    return pd.DataFrame(rows)


def summarise_scores(scores: pd.DataFrame, model_name: str) -> dict[str, float | str]:
    """Mean and std over folds of R^2 and MAE."""
    return {
        "model": model_name,
        "r2_mean": scores["r2"].mean(),
        "r2_std": scores["r2"].std(ddof=1) if len(scores) > 1 else float("nan"),
        "mae_mean": scores["mae"].mean(),
        "mae_std": scores["mae"].std(ddof=1) if len(scores) > 1 else float("nan"),
        "n_folds": len(scores),
    }


def permutation_importance_cv(
    model: BaseEstimator,
    X: pd.DataFrame,
    y,
    weights,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    seed: int = 42,
    n_repeats: int = 5,
    metric: Callable = weighted_r2,
) -> pd.DataFrame:
    """Permutation importance of raw columns on each held-out fold.

    Each column is shuffled in the *raw* frame (before encoding), so a categorical feature is
    permuted as a whole. Importance is the drop in weighted R^2 relative to the unpermuted
    held-out score, averaged over repeats within a fold; the table aggregates across folds.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    records = []
    for k, (tr, te) in enumerate(splits):
        m = _fit(model, X.iloc[tr], y[tr], w[tr])
        X_te = X.iloc[te].reset_index(drop=True)
        base = metric(y[te], m.predict(X_te), w[te])
        for col in X.columns:
            drops = []
            for _ in range(n_repeats):
                Xp = X_te.copy()
                Xp[col] = Xp[col].to_numpy()[rng.permutation(len(Xp))]
                drops.append(base - metric(y[te], m.predict(Xp), w[te]))
            records.append({"fold": k, "feature": col, "r2_drop": float(np.mean(drops))})
    per_fold = pd.DataFrame(records)
    agg = (
        per_fold.groupby("feature")["r2_drop"]
        .agg(importance_mean="mean", importance_std="std")
        .reset_index()
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return agg
