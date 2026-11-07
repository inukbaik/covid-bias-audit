"""Project configuration: paths, seeds, thresholds and label constants."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# Short readable race labels used throughout the package.
EDU_RACE_MAP: dict[str, str] = {
    "Hispanic": "Hispanic",
    "Non-Hispanic White": "NH White",
    "Non-Hispanic Black": "NH Black",
    "Non-Hispanic Asian": "NH Asian",
    "Non-Hispanic American Indian or Alaska Native": "NH AIAN",
    "Non-Hispanic Native Hawaiian or Other Pacific Islander": "NH NHPI",
    "Non-Hispanic More than one race": "NH Multiracial",
}
# The raw file truncates the NHPI label at 50 characters.
TRUNCATED_NHPI = "Non-Hispanic Native Hawaiian or Other Pacific Islande"
FULL_NHPI = "Non-Hispanic Native Hawaiian or Other Pacific Islander"

MONTHLY_RACE_MAP: dict[str, str] = {
    "Hispanic": "Hispanic",
    "Non-Hispanic White": "NH White",
    "Non-Hispanic Black": "NH Black",
    "Non-Hispanic Asian": "NH Asian",
    "Non-Hispanic American Indian or Alaska Native": "NH AIAN",
    "Other": "Other",
}

# Short keys for the county-level race columns -> raw CDC column names.
COUNTY_RACE_COLUMNS: dict[str, str] = {
    "white": "Non-Hispanic White",
    "black": "Non-Hispanic Black",
    "aian": "Non-Hispanic American Indian or Alaska Native",
    "asian": "Non-Hispanic Asian",
    "nhpi": "Non-Hispanic Native Hawaiian or Other Pacific Islander",
    "hispanic": "Hispanic",
    "other": "Other",
}
COUNTY_RACE_KEYS: tuple[str, ...] = tuple(COUNTY_RACE_COLUMNS)
COUNTY_RACE_LABELS: dict[str, str] = {
    "white": "NH White",
    "black": "NH Black",
    "aian": "NH AIAN",
    "asian": "NH Asian",
    "nhpi": "NH NHPI",
    "hispanic": "Hispanic",
    "other": "Other",
}

EDUCATION_LEVELS: tuple[str, ...] = ("HS or less", "Some college", "Bachelor's or more")
URBAN_RURAL_LEVELS: tuple[str, ...] = (
    "Large central metro",
    "Large fringe metro",
    "Medium metro",
    "Small metro",
    "Micropolitan",
    "Noncore",
)
MONTHLY_AGE_GROUPS: tuple[str, ...] = (
    "0-4 years",
    "5-14 years",
    "15-24 years",
    "25-34 years",
    "35-44 years",
    "45-54 years",
    "55-64 years",
    "65-74 years",
    "75-84 years",
    "85 years and over",
)

REF_RACE = "NH White"
REF_EDUCATION = "Bachelor's or more"
REF_SEX = "Female"


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking up to a directory holding ``pyproject.toml``."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "data").exists():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class Config:
    """Immutable run configuration."""

    root: Path = field(default_factory=find_repo_root)
    seed: int = 42
    min_cell_deaths: int = 20
    n_bootstrap: int = 2000
    cv_repeats: int = 10
    n_original_seeds: int = 50
    rf_estimators: int = 300
    fast: bool = False

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def fast_mode(self) -> Config:
        """Return a copy with reduced repeats for quick runs and tests."""
        return replace(
            self,
            n_bootstrap=100,
            cv_repeats=1,
            n_original_seeds=5,
            rf_estimators=40,
            fast=True,
        )


SEED = 42
MIN_CELL_DEATHS = 20
N_BOOTSTRAP = 2000
CV_REPEATS = 10
