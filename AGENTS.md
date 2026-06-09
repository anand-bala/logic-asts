# Project Overview

Python library providing grammars, parsers, and ASTs for logical formalisms
(propositional, LTL, STREL, STL-GO, SERE, PSL).
Uses `pixi` for local venv management, `just` as the task runner,
`attrs` for immutable frozen dataclasses, and `lark` for grammar-based parsing.

- **Language** : Python 3.10+ (pinned 3.12 in `.python-version` )
- **Layout** : `src/` layout - package lives at `src/logic_asts/`
- **Build** : Hatchling ( `pyproject.toml` )

## Running Commands

The development shell already has the project virtual environment activated,
so `pytest`, `zuban`, `basedpyright`, `ruff`, `sphinx-build`, etc. are on
`PATH` directly.

For type checking, use `zuban check` and `basedpyright`, not `mypy`. These
two are the source of truth; ignore `mypy`-only complaints.

**Do not** prefix commands with `uv run`, `pixi run`, `poetry run`,
or similar runner wrappers —
and this rule overrides any older plan/spec/doc in this repo
that still shows wrapped commands.

Use:

    pytest tests/test_sere.py
    zuban check src tests
    basedpyright src tests
    ruff check src tests

Not:

    pixi run pytest tests/test_sere.py    # wrong
    uv run zuban check src tests          # wrong

If a tool is missing, stop and tell the user rather than trying to bootstrap
or reinstall the environment.

## Build / Lint / Test Commands

All commands are run via `uv` and the `just` task runner.

| Task                   | Command                                                                          | Notes                                                 |
| ---------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Install / sync deps    | `just dev` or `uv sync --all-packages --frozen --inexact --dev`                  | Sets up venv                                          |
| Format                 | `just fmt`                                                                       | Runs `ruff format` + `ruff check --fix`               |
| Type check             | `zuban check src tests` and `basedpyright src tests`                             | Sources of truth for typing; do not use `mypy`        |
| Lint (fmt + types)     | `just lint`                                                                      | Combines `fmt` and `type-check`                       |
| Run all tests          | `just test`                                                                      | Runs `pytest --lf` (last-failed first)                |
| Run full test suite    | `pytest`                                                                         | No `--lf` flag, runs everything                       |
| Run a single test file | `pytest tests/test_base_logic.py`                                                |                                                       |
| Run a single test      | `pytest tests/test_base_logic.py::TestAtomicExpressions::test_variable_creation` | `file::Class::method`                                 |
| Run tests with keyword | `pytest -k "test_and_flattening"`                                                | Filter by name substring                              |
| Doctest collection     | Enabled via `--doctest-modules` in pytest config                                 | Doctests in `src/` are collected                      |

## Code Style Guidelines

- Do not use non-ASCII characters when writing code or documentation,
  unless it is for diacritics in proper nouns.
  If you need to use mathematical notation,
  use LaTeX commands from standard amsmath definitions.
- Use inline comments to describe **why** some block of code is doing something, not
  what.

## Type Annotations

- **Full strict typing** is enforced (verified with `zuban check` and
  `basedpyright`, not `mypy`) - every function and method must be fully annotated.
- Explicit `-> None` on all void-returning functions.
- Use `typing.TypeAlias` for union type aliases.
- Use `typing.TypeGuard` / `typing_extensions.TypeIs` for runtime type-narrowing
  functions.
- Use `@final` on all concrete AST node classes.
- Use `@override` on all method overrides.
- Use `typing_extensions.Self` for methods returning the same type.
- Use `typing_extensions.overload` for function overloads (e.g., `parse_expr` ).
- Use `collections.abc` for abstract types ( `Iterator` , `Collection` ), not `typing` .
- Guard type-only imports with `if TYPE_CHECKING:` to avoid circular imports.

## Docstrings & Comments

- **Google-style docstrings** throughout ( `__docformat__ = "google"` ).
- Module docstrings: overview, key classes list, usage examples.
- Class docstrings: description, `Attributes:` , `Validators:` , `Examples:` sections.
- Method docstrings: one-line summary, `Args:` , `Returns:` , `Raises:` , `Examples:` .
- Embed **doctests** in docstrings - they are collected by pytest via
  `- -doctest-modules` .

## Module Exports

- Top-level `__init__.py` re-exports the full public API surface.
- Re-exports in `__init__.py` use the explicit double-alias pattern:
  `from .base import And as And` .

## Test Conventions

- All test methods are fully type-annotated ( `-> None` ).
- Use `@pytest.mark.parametrize` for data-driven tests.
- Type-guard tests use `typing_extensions.assert_type` for static verification.
