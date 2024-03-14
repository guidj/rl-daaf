pip-compile: requirements.in test-requirements.in nb-requirements.in dev-requirements.in ray-env-requirements.in rendering-requirements.in 
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links requirements.in
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links test-requirements.in
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links nb-requirements.in
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links dev-requirements.in
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links ray-env-requirements.in --unsafe-package ray
	pip-compile --no-emit-index-url --no-emit-options --no-emit-find-links rendering-requirements.in

pip-install: pip-compile
	pip install -r dev-requirements.txt -e .

format:
	ruff format src tests
	ruff check --fix

test-coverage:
	pytest --cov-report=html --cov=src tests
