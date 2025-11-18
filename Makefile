SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# Default: create the dev environment
dev: uv.lock | .venv
.PHONY: dev

fmt:
	uv run --frozen ruff format 
	uv run --frozen ruff check --fix .
.PHONY: fmt

typing:
	uv run --frozen zmypy
.PHONY: typing

lint: fmt typing
.PHONY: lint

test:
	uv run --dev --frozen pytest
.PHONY: test

docs:
	uv run --dev pdoc 'logic_asts' --math --docformat=google -o docs/_build/
.PHONY: docs

uv.lock .venv &: pyproject.toml
	uv sync --frozen --dev

# Automatic make target for scripts with locking
%.py.lock: %.py
	uv lock --script $<
