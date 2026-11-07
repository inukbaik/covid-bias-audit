import numpy as np

from covid_bias.analysis.risk import build_cells, compute_tau, split_temporal


def test_tau_uses_training_data_only(monthly, cfg):
    cells = build_cells(monthly, cfg.min_cell_deaths)
    train, test, tau = split_temporal(cells)
    assert (train["year"] == 2020).all() and (test["year"] == 2021).all()
    assert np.isclose(tau, train["covid_deaths"].sum() / train["all_cause"].sum())

    altered = cells.copy()
    mask = altered["year"] == 2021
    altered.loc[mask, "covid_deaths"] = altered.loc[mask, "all_cause"]
    altered["covid_share"] = altered["covid_deaths"] / altered["all_cause"]
    train2, test2, tau2 = split_temporal(altered)
    assert tau2 == tau
    assert train2["high_burden"].equals(train["high_burden"])
    assert test2["high_burden"].mean() > test["high_burden"].mean()


def test_min_cell_deaths_filter(monthly):
    cells = build_cells(monthly, 20)
    assert (cells["all_cause"] >= 20).all()
    assert len(cells) < len(monthly)


def test_compute_tau_weighted(monthly):
    sub = monthly.iloc[:10]
    assert np.isclose(compute_tau(sub), sub["covid_deaths"].sum() / sub["all_cause"].sum())
