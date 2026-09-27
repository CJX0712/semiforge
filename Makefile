.PHONY: install test lint format demo bench hpo clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q -W ignore::UserWarning

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .

demo:
	$(PY) examples/run_demo.py

bench:
	$(PY) -m semiforge.cli run --datasets moons blobs circle spiral highdim digits --out benchmark.json

hpo:
	$(PY) -m semiforge.cli run --datasets moons circle spiral --hpo --hpo-trials 20 --hpo-timeout 120 --out benchmark_hpo.json

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
