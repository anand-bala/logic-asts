import pytest

import logic_asts
from logic_asts.base import And, Equiv, Expr, Variable


def test_base_logic() -> None:
    expr = "(x1 <-> x2) | x3"
    expected = Equiv(Variable("x1"), Variable("x2")) | Variable("x3")
    parsed = logic_asts.parse_expr(expr, syntax="base")
    assert parsed == expected, (parsed, expected)


@pytest.mark.parametrize(
    "n",
    [3, 5, 10, 20, 30, 40, 80, 100],
)
def test_parse_large_expr(n: int) -> None:
    expr = " & ".join((f"(x{i} <-> y{i})" for i in range(n)))
    expected: Expr = And([Equiv(Variable(f"x{i}"), Variable(f"y{i}")) for i in range(n)])
    parsed = logic_asts.parse_expr(expr, syntax="base")
    assert parsed == expected
