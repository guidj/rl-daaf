pip-sync:
	uv sync

format:
	uv run ruff format src tests
	uv run ruff check --fix

test-coverage:
	uv run pytest --cov-report=html --cov=src tests
