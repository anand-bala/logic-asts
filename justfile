# Set shell options for safety

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Default: create the dev environment
default: dev

# Set up development environment
dev: sync-venv

# Format and lint code
[no-cd]
fmt:
    ruff format
    ruff check --output-format concise --fix --exit-non-zero-on-fix .

# Run type checkers
[no-cd]
[private]
ty-check:
    ty check --output-format concise

[no-cd]
[private]
pyrefly-check:
    pyrefly check --output-format min-text

[no-cd]
[private]
mypy-check:
    mypy --strict

[no-cd]
[private]
pyright-check:
    basedpyright

# [parallel]
type-check: mypy-check

# pyright-check
# ty-check pyrefly-check

# Run both formatting and type checking
[no-cd]
lint: fmt type-check

# Run tests
[no-cd]
test:
    pytest --lf

docs:
    sphinx-build -b html docs/ docs/_build/html

# Sync virtual environment
sync-venv:
    pixi install --frozen

# Release workflow

bump-version *args: lint test
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Will run: uv version --bump {{ args }}"
    read -p "Are you sure? [y/n] " -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      uv version --bump {{ args }}
      pixi lock
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
