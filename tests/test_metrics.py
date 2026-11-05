import warnings

import numpy as np
import pandas as pd

from covid_bias.evaluation.metrics import (
    binary_metrics,
    bootstrap_ci,
    fairness_summary,
    group_fairness_table,
    weighted_mae,
    weighted_r2,
    wilson_interval,
)


def test_weighted_r2_toy():
    y, p, w = [1.0, 2.0, 4.0], [1.0, 3.0, 3.0], [1.0, 1.0, 2.0]
    # weighted mean 11/4; SST = 49/16 + 9/16 + 2*(25/16) = 108/16; SSE = 0 + 1 + 2 = 3
    assert np.isclose(weighted_r2(y, p, w), 1 - 3 / (108 / 16))
    assert weighted_r2(y, y, w) == 1.0
    assert np.isclose(weighted_r2(y, [np.average(y, weights=w)] * 3, w), 0.0)


def test_weighted_mae_toy():
    assert np.isclose(weighted_mae([1, 2, 4], [1, 3, 3], [1, 1, 2]), (0 + 1 + 2) / 4)
    assert np.isclose(weighted_mae([1, 2], [2, 4]), 1.5)


def test_binary_metrics():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["accuracy"] == 0.5
    assert m["balanced_accuracy"] == 0.5
    assert np.isclose(m["roc_auc"], 0.75)
    assert np.isclose(m["brier"], (0.01 + 0.36 + 0.36 + 0.01) / 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(binary_metrics([1, 1], [0.2, 0.9])["roc_auc"])


def _toy():
    y = [1, 1, 0, 0, 1, 0, 0, 0]
    yhat = [1, 0, 1, 0, 1, 0, 0, 0]
    g = ["a", "a", "a", "a", "b", "b", "b", "b"]
    return y, yhat, g


def test_group_fairness_table():
    t = group_fairness_table(*_toy()).set_index("group")
    assert t.loc["a", "n"] == 4
    assert t.loc["a", "base_rate"] == 0.5
    assert t.loc["a", "selection_rate"] == 0.5
    assert t.loc["a", "tpr"] == 0.5 and t.loc["a", "fnr"] == 0.5
    assert t.loc["a", "fpr"] == 0.5 and t.loc["a", "precision"] == 0.5
    assert t.loc["b", "tpr"] == 1.0 and t.loc["b", "fpr"] == 0.0
    assert t.loc["b", "selection_rate"] == 0.25 and t.loc["b", "precision"] == 1.0


def test_group_without_positives_is_nan():
    t = group_fairness_table([0, 0, 1, 1], [0, 0, 1, 0], ["x", "x", "y", "y"]).set_index("group")
    assert np.isnan(t.loc["x", "tpr"]) and np.isnan(t.loc["x", "precision"])
    assert t.loc["x", "fpr"] == 0.0
    s = fairness_summary(t.reset_index())
    # Only one group has a defined TPR, so its range is zero; NaNs are ignored.
    assert s["tpr_range"] == 0.0
    assert s["equalized_odds_difference"] == 0.0


def test_fairness_summary():
    s = fairness_summary(group_fairness_table(*_toy()))
    assert np.isclose(s["demographic_parity_difference"], 0.25)
    assert np.isclose(s["tpr_range"], 0.5)
    assert np.isclose(s["fpr_range"], 0.5)
    assert np.isclose(s["equalized_odds_difference"], 0.5)


def test_bootstrap_ci_deterministic():
    x = np.random.default_rng(0).normal(size=50)
    a = bootstrap_ci(x, n_boot=200, seed=1)
    assert a == bootstrap_ci(x, n_boot=200, seed=1)
    assert a != bootstrap_ci(x, n_boot=200, seed=2)
    assert a[1] < a[0] < a[2]
    est, _, _ = bootstrap_ci([1.0, 3.0], weights=[1, 3], n_boot=10)
    assert np.isclose(est, 2.5)


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert all(np.isnan(v) for v in wilson_interval(0, 0))
    assert isinstance(pd.Series([lo]).iloc[0], float)
