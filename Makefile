pip-compile: requirements.in test-requirements.in nb-requirements.in dev-requirements.in ray-env-requirements.in rendering-requirements.txt
	uv pip compile --no-emit-index-url --no-emit-find-links requirements.in -o requirements.txt
	uv pip compile --no-emit-index-url --no-emit-find-links test-requirements.in -o test-requirements.txt
	uv pip compile --no-emit-index-url --no-emit-find-links nb-requirements.in -o nb-requirements.txt
	uv pip compile --no-emit-index-url --no-emit-find-links dev-requirements.in -o dev-requirements.txt
	uv pip compile --no-emit-index-url --no-emit-find-links ray-env-requirements.in -o ray-env-requirements.txt --unsafe-package ray
	uv pip compile --no-emit-index-url --no-emit-find-links rendering-requirements.in -o rendering-requirements.txt

pip-install: pip-compile
	uv pip install -r dev-requirements.txt -e .

format:
	ruff format src tests --line-length 88
	ruff check --fix

test-coverage:
	pytest --cov-report=html --cov=src tests
