import json

import pytest

from covid_bias.cli import main

EXPECTED = {
    "education": (
        [
            "education_pooled_share.csv",
            "education_glm_odds_ratios.csv",
            "education_cv_scores.csv",
            "education_permutation_importance.csv",
        ],
        ["education_odds_ratios.png", "education_permutation_importance.png"],
    ),
    "county": (
        [
            "county_disparity_ratios.csv",
            "county_cv_scores.csv",
            "county_permutation_importance.csv",
            "county_glm_odds_ratios.csv",
        ],
        ["county_disparity_ratios.png", "county_permutation_importance.png"],
    ),
    "risk": (
        ["risk_model_scores.csv", "risk_group_fairness.csv", "risk_fairness_summary.csv"],
        ["risk_group_fairness.png"],
    ),
    "original": (["original_reproduction.csv"], ["original_r2_instability.png"]),
}


@pytest.mark.parametrize("analysis", list(EXPECTED))
def test_analysis_fast(analysis, tmp_path):
    assert main(["run", "--analysis", analysis, "--output", str(tmp_path), "--fast"]) == 0
    tables, figures = EXPECTED[analysis]
    for t in tables:
        assert (tmp_path / "tables" / t).stat().st_size > 0
    for f in figures:
        assert (tmp_path / "figures" / f).stat().st_size > 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert analysis in metrics
