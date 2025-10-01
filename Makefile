PYTHON ?= python

.PHONY: fmt lint check test ci

fmt:
	$(PYTHON) -m isort . --profile=black
	$(PYTHON) -m black .

lint:
	$(PYTHON) -m flake8 .

check:
	$(PYTHON) -m isort . --profile=black --check-only
	$(PYTHON) -m black . --check
	$(PYTHON) -m flake8 .

test:
	$(PYTHON) -m pytest -q

ci: check test
