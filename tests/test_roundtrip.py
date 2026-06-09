"""Property-based round-trip tests for the lark grammars.

For each supported syntax we use ``hypothesis.extra.lark.from_lark`` to
generate strings directly from the grammar, parse them inside the strategy
to get an ``Expr``, then assert ``parse(str(expr)) == expr``.

We do the initial parse inside the strategy (rather than ``assume``-ing in
the test body) because ``from_lark`` ignores regex lookahead/boundary
terminals -- e.g. it happily emits ``1xor1``, which the real lexer rejects
because ``1`` followed by ``x`` violates the TRUE-terminal's ``(?!\\w)``
guard. Filtering those out in the strategy keeps hypothesis's shrinker and
example database focused on inputs that actually produced an ``Expr``.

We can't always assert ``parse(s)`` matches a re-stringified form character
for character because the AST canonicalizes during construction (constant
folding via ``Literal.__and__``/``__or__``, ``And``/``Or`` arg flattening,
double-negation elimination, etc.). Stability after the first parse is the
right invariant.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.lark import from_lark
from lark import Lark
from lark.exceptions import LarkError

from logic_asts import parse_expr
from logic_asts.base import Variable
from logic_asts.grammars import SupportedGrammars
from logic_asts.psl import (
    StrongClosure,
    SuffixImpliesExist,
    SuffixImpliesUniv,
    WeakClosure,
)
from logic_asts.sere import Alt, Concat, Fusion, Inter, Repeat
from logic_asts.spec import Expr

# Cache the Lark parser per syntax so each example doesn't pay parser-build cost.
_GRAMMARS: dict[SupportedGrammars, Lark] = {
    syntax: Lark.open_from_package("logic_asts", f"{syntax.value}.lark", ["grammars"]) for syntax in SupportedGrammars
}


def _expr_strategy(syntax: SupportedGrammars) -> st.SearchStrategy[Expr]:
    """Generate grammar strings and parse them, dropping ones the real lexer rejects."""

    def _try_parse(source: str) -> Expr | None:
        try:
            return parse_expr(source, syntax=syntax)  # type: ignore[call-overload, no-any-return]
        except LarkError:
            return None

    parsed: st.SearchStrategy[Expr | None] = from_lark(_GRAMMARS[syntax]).map(_try_parse)
    return parsed.filter(lambda e: e is not None)  # type: ignore[return-value]


_COMMON_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much, HealthCheck.data_too_large],
)


def _assert_roundtrip(expr: Expr, syntax: SupportedGrammars) -> None:
    rendered = str(expr)
    reparsed = parse_expr(rendered, syntax=syntax)  # type: ignore[call-overload]
    assert expr == reparsed, (
        f"round-trip not stable for syntax={syntax.value!r}:\n"
        f"  expr:     {expr!r}\n"
        f"  rendered: {rendered!r}\n"
        f"  reparsed: {reparsed!r}"
    )


@_COMMON_SETTINGS
@given(expr=_expr_strategy(SupportedGrammars.BASE))
def test_base_grammar_roundtrip(expr: Expr) -> None:
    _assert_roundtrip(expr, SupportedGrammars.BASE)


@_COMMON_SETTINGS
@given(expr=_expr_strategy(SupportedGrammars.LTL))
def test_ltl_grammar_roundtrip(expr: Expr) -> None:
    _assert_roundtrip(expr, SupportedGrammars.LTL)


@pytest.mark.parametrize(
    "name",
    [
        "0",
        "1",
        "true",
        "False",
        "TRUE",
        "hello world",
        'has"quote',
        "back\\slash",
        "with spaces and !@#",
    ],
)
def test_known_unsafe_variable_names_roundtrip(name: str) -> None:
    """Names that would otherwise collide with literals or special chars must round-trip."""
    expr = Variable(name)
    parsed = parse_expr(str(expr), syntax="base")
    assert parsed == expr


def _bool_atom_strategy() -> st.SearchStrategy[Expr]:
    return st.builds(Variable, st.sampled_from(["a", "b", "c", "d"]))


def sere_strategy(max_leaves: int = 8) -> st.SearchStrategy[Expr]:
    return st.recursive(
        _bool_atom_strategy(),
        lambda children: st.one_of(
            st.builds(
                Concat,
                st.lists(children, min_size=2, max_size=4).map(tuple),
            ),
            st.builds(
                Fusion,
                st.lists(children, min_size=2, max_size=3).map(tuple),
            ),
            st.builds(
                Alt,
                st.lists(children, min_size=2, max_size=3).map(tuple),
            ),
            st.builds(
                Inter,
                st.lists(children, min_size=2, max_size=3).map(tuple),
            ),
            st.tuples(
                children,
                st.integers(min_value=0, max_value=4),
                st.one_of(st.none(), st.integers(min_value=0, max_value=6)),
            )
            .filter(lambda t: t[2] is None or t[1] <= t[2])  # zuban: ignore[index]
            .map(lambda t: Repeat(t[0], t[1], t[2])),  # zuban: ignore[index]
        ),
        max_leaves=max_leaves,
    )


@_COMMON_SETTINGS
@given(expr=sere_strategy())
def test_sere_round_trips(expr: Expr) -> None:
    assert parse_expr(str(expr), syntax="sere") == expr


@pytest.mark.parametrize(
    "src",
    [
        "a & b",
        "a & b & c",
        "a & b && c",
        "a | b & c",
        "(a ; b[*]) & c",
    ],
)
def test_sere_nlm_inter_roundtrip(src: str) -> None:
    expr = parse_expr(src, syntax="sere")
    assert parse_expr(str(expr), syntax="sere") == expr


@pytest.mark.parametrize(
    "src",
    [
        "~a",
        "~~a",
        "~{a;b}",
        "~a[*]",
        "~a | b",
        "first_match(a)",
        "first_match(a;b)",
        "first_match(~a)",
        "~first_match(a)",
        "first_match(a)[*]",
        "(a & b) | ~c",
        "{first_match(a;b) && ~c}",
    ],
)
def test_sere_complement_and_first_match_roundtrip(src: str) -> None:
    expr = parse_expr(src, syntax="sere")
    assert parse_expr(str(expr), syntax="sere") == expr


@pytest.mark.parametrize(
    "src",
    [
        "a[:*]",
        "a[:+]",
        "a[:*3]",
        "a[:*2..5]",
        "a[:*2..]",
        "a[->]",
        "a[->2]",
        "a[->2..5]",
        "a[->3..]",
        "a[=]",
        "a[=3]",
        "a[=2..5]",
        "a[=3..]",
        "(a;b)[->2]",
        "~a[=2]",
        "first_match(a)[:+]",
        "{a[:*3] && b[->2]}",
    ],
)
def test_sere_extended_repetition_roundtrip(src: str) -> None:
    expr = parse_expr(src, syntax="sere")
    assert parse_expr(str(expr), syntax="sere") == expr


def psl_strategy(max_leaves: int = 8) -> st.SearchStrategy[Expr]:
    sere = sere_strategy(max_leaves=max_leaves)
    formulas = _bool_atom_strategy()

    return st.one_of(
        st.builds(SuffixImpliesUniv, sere, formulas),
        st.builds(SuffixImpliesExist, sere, formulas),
        st.builds(WeakClosure, sere),
        st.builds(StrongClosure, sere),
        formulas,
    )


@_COMMON_SETTINGS
@given(expr=psl_strategy())
def test_psl_round_trips(expr: Expr) -> None:
    assert parse_expr(str(expr), syntax="psl") == expr
