.PHONY: install test lint format typecheck check results clean

install:          ## editable install with dev extras
	pip install -e ".[dev]"

test:             ## run the test-suite with coverage
	MPLBACKEND=Agg pytest --cov

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

check: lint typecheck test   ## everything CI runs

results:          ## regenerate the figures and numbers quoted in the README (needs network once)
	MPLBACKEND=Agg python scripts/make_readme_figures.py
	MPLBACKEND=Agg sarb backtest --n-boot 5000 --out docs/figures
	MPLBACKEND=Agg sarb sweep --out docs/figures
	MPLBACKEND=Agg sarb portfolio --out docs/figures
	sarb scan

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
