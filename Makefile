PYTHON ?= python3.12
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
VENV_PYTEST := $(VENV_DIR)/bin/pytest

.PHONY: help venv setup test clean

help:
	@echo "Targets:"
	@echo "  make setup  - Create virtualenv and install dependencies for tests"
	@echo "  make test   - Run pytest with project config (includes coverage)"
	@echo "  make clean  - Remove test and coverage artifacts"

venv:
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)

setup: venv
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@$(VENV_PIP) install -e . pytest pytest-cov
	@$(VENV_PYTHON) -c "import nltk; [nltk.download(r, quiet=True) for r in ('punkt', 'punkt_tab', 'stopwords')]"

test: setup
	@$(VENV_PYTEST) -q

clean:
	@rm -rf .pytest_cache htmlcov .coverage .coverage.*
