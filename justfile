# Set shell options for safety

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

[private]
_frozen := if env("FORCE_UV_NO_FREEZE", "") != "1" { " --frozen " } else { "" }
DEFAULT_UV_ARGS := "--dev" + " " + _frozen

# Default: create the dev environment
default: dev

# Set up development environment
dev: sync-venv

# Format and lint code
[no-cd]
fmt:
    uv run {{ DEFAULT_UV_ARGS }} ruff format
    uv run {{ DEFAULT_UV_ARGS }} ruff check --output-format concise --fix --exit-non-zero-on-fix .

# Run type checkers
[no-cd]
[private]
ty-check:
    uv run {{ DEFAULT_UV_ARGS }} ty check --output-format concise

[no-cd]
[private]
pyrefly-check:
    uv run {{ DEFAULT_UV_ARGS }} pyrefly check --output-format min-text

[no-cd]
[private]
mypy-check:
    uv run {{ DEFAULT_UV_ARGS }} mypy --strict

[no-cd]
[private]
pyright-check:
    uv run {{ DEFAULT_UV_ARGS }} basedpyright

[parallel]
type-check: mypy-check pyright-check

# ty-check pyrefly-check

# Run both formatting and type checking
[no-cd]
lint: fmt type-check

# Run tests
[no-cd]
test:
    uv run  {{ DEFAULT_UV_ARGS }} pytest --lf

docs:
    uv run  {{ DEFAULT_UV_ARGS }} sphinx-build -b html docs/ docs/_build/html

# Sync virtual environment
sync-venv:
    uv sync {{ DEFAULT_UV_ARGS }}

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
