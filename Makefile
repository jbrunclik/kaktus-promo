.PHONY: setup lint format test clean

setup:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -e ".[dev]"

lint:
	venv/bin/ruff check .
	venv/bin/mypy .

format:
	venv/bin/ruff format .
	venv/bin/ruff check --fix .

test:
	venv/bin/pytest -v

clean:
	rm -rf venv __pycache__ .mypy_cache .ruff_cache .pytest_cache *.egg-info
