import pytest

from covid_bias.config import Config
from covid_bias.data import load_county, load_education, load_monthly


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config().fast_mode()


@pytest.fixture(scope="session")
def education(cfg):
    return load_education(cfg=cfg)


@pytest.fixture(scope="session")
def monthly(cfg):
    return load_monthly(cfg=cfg)


@pytest.fixture(scope="session")
def county(cfg):
    return load_county(cfg=cfg)
