"""Weighted regression metrics, binary classification metrics and group fairness."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def weighted_r2(y_true, y_pred, sample_weight=None) -> float:
    """Weighted coefficient of determination (1 - SSE / SST around the weighted mean)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.ones_like(y_true) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    mean = np.average(y_true, weights=w)
    sst = np.sum(w * (y_true - mean) ** 2)
    sse = np.sum(w * (y_true - y_pred) ** 2)
    return float("nan") if sst == 0 else float(1 - sse / sst)


def weighted_mae(y_true, y_pred, sample_weight=None) -> float:
    """Weighted mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.average(np.abs(y_true - y_pred), weights=sample_weight))


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, balanced accuracy, F1, ROC AUC and Brier score for probabilistic output."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_hat = (y_prob >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def group_fairness_table(y_true, y_pred, groups, y_prob=None) -> pd.DataFrame:
    """Per-group n, base rate, selection rate, TPR, FPR, FNR, precision (and calibration).

    ``y_pred`` is the hard 0/1 decision. Rates with an empty denominator are NaN.
    """
    df = pd.DataFrame(
        {
            "group": np.asarray(groups, dtype=object),
            "y": np.asarray(y_true).astype(int),
            "yhat": np.asarray(y_pred).astype(int),
        }
    )
    if y_prob is not None:
        df["prob"] = np.asarray(y_prob, dtype=float)
    rows = []
    for g, d in df.groupby("group", sort=True):
        tp = int(((d.y == 1) & (d.yhat == 1)).sum())
        fp = int(((d.y == 0) & (d.yhat == 1)).sum())
        fn = int(((d.y == 1) & (d.yhat == 0)).sum())
        tn = int(((d.y == 0) & (d.yhat == 0)).sum())
        row = {
            "group": g,
            "n": len(d),
            "base_rate": d.y.mean(),
            "selection_rate": d.yhat.mean(),
            "tpr": _safe_div(tp, tp + fn),
            "fpr": _safe_div(fp, fp + tn),
            "fnr": _safe_div(fn, tp + fn),
            "precision": _safe_div(tp, tp + fp),
        }
        if y_prob is not None:
            row["mean_pred_prob"] = d.prob.mean()
            row["observed_rate"] = d.y.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def fairness_summary(table: pd.DataFrame) -> dict[str, float]:
    """Demographic parity difference and equalized odds difference from a group table.

    Equalized odds difference is the larger of the TPR range and the FPR range (NaN ignored).
    """
    sel = table["selection_rate"]
    tpr_range = table["tpr"].max() - table["tpr"].min()
    fpr_range = table["fpr"].max() - table["fpr"].min()
    ranges = [r for r in (tpr_range, fpr_range) if not np.isnan(r)]
    return {
        "demographic_parity_difference": float(sel.max() - sel.min()),
        "tpr_range": float(tpr_range),
        "fpr_range": float(fpr_range),
        "equalized_odds_difference": float(max(ranges)) if ranges else float("nan"),
    }


def _weighted_mean(a: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(a, weights=w)) if w.sum() > 0 else float("nan")


def bootstrap_ci(
    values,
    weights=None,
    stat: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI of a (weighted) statistic; returns (estimate, low, high).

    Default statistic is the weighted mean. Rows are resampled with replacement.
    """
    x = np.asarray(values, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    stat = stat or _weighted_mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    draws = np.array([stat(x[i], w[i]) for i in idx])
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(stat(x, w)), float(lo), float(hi)


def wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)
