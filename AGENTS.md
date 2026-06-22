# Guide for Coding Agents

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
