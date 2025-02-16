pip-sync:
	uv sync

format:
	uv run ruff check --select I --fix
	uv run ruff format src tests

test-coverage:
	uv run pytest --cov-report=html --cov=src tests
