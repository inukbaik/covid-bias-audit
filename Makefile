.PHONY: install lint format test reproduce

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	python -m pytest

reproduce:
	MPLBACKEND=Agg python -m covid_bias run --analysis all --output reports
