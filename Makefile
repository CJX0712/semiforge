# SemiForge Makefile (作者: 晨星)
.PHONY: install test lint demo ci clean

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff

test:
	pytest -q

lint:
	ruff check .

demo:
	python cli.py demo

ci: lint test demo

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache benchmark.json
