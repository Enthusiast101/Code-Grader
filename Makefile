PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install sandbox test run clean

install:
	$(PIP) install -e .

sandbox:
	docker build -f Dockerfile.sandbox -t agent-sandbox:latest .

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

run:
	$(PYTHON) -m grading_agent.main

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info reports .grader-workspace
