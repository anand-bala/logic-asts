# Set shell options for safety

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Default: create the dev environment
default: dev

# Set up development environment
dev: sync-venv

# Format and lint code
[no-cd]
fmt:
    uv run --frozen ruff format
    uv run --frozen ruff check --output-format concise --fix --exit-non-zero-on-fix .

# Run type checkers
[no-cd]
[private]
ty-check:
    uv run --frozen ty check --output-format concise

[no-cd]
[private]
pyrefly-check:
    uv run --frozen pyrefly check --output-format min-text

[no-cd]
[private]
mypy-check:
    uv run --frozen mypy --strict

[parallel]
type-check: ty-check pyrefly-check mypy-check

# Run both formatting and type checking
[no-cd]
lint: fmt type-check

# Run tests
[no-cd]
test:
    uv run --dev --frozen pytest --lf

# Sync virtual environment
sync-venv:
    uv sync --all-packages --frozen --inexact --dev

# Lock a Python script's dependencies
lock-script script:
    uv lock --script {{ script }}

# Release workflow

bump-version *args: lint test
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Will run: uv version --bump {{ args }}"
    read -p "Are you sure? [y/n] " -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      uv version --bump {{ args }}
    fi

tag-package:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Will run: git tag $(printf "v%s" $(uv version --short))"
    read -p "Are you sure? [y/n] " -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      git tag "$(printf "v%s" $(uv version --short))"
    fi
