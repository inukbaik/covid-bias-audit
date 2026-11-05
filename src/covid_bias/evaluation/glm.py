"""Quasi-binomial GLM helper."""

from __future__ import annotations


def fit_quasi_binomial(model, **fit_kwargs):
    """Fit a binomial GLM with the dispersion estimated from the Pearson chi-square.

    ``scale="X2"`` in statsmodels mis-scales the dispersion for two-column (successes,
    failures) endog (it returned 0.026 instead of the true 54 on the education data), so the
    Pearson dispersion is computed explicitly on the count scale and passed as a fixed scale.
    """
    first = model.fit()
    phi = float(first.pearson_chi2 / first.df_resid)
    res = model.fit(scale=phi, **fit_kwargs)
    res.dispersion = phi
    return res
