import math
import pytest

import logic_asts
import logic_asts.ltl as ltl
from logic_asts.base import Expr, Not, Variable

CASES = [
    (
        "X(Gp2 U Fp2)",
        ltl.Next(
            ltl.Until(
                ltl.Always(Variable("p2")),
                ltl.Eventually(Variable("p2")),
            ),
        ),
        math.inf,
    ),
    ("!Fp2", Not(ltl.Eventually(Variable("p2"))), math.inf),
    (
        "F(a & F(b & F[,20]c))",
        ltl.Eventually(
            Variable("a") & ltl.Eventually(Variable("b") & ltl.Eventually(Variable("c"), ltl.TimeInterval(None, 20)))
        ),
        math.inf,
    ),
    (
        "X(a & F[,10](b & F[,20]c))",
        ltl.Next(
            Variable("a")
            & ltl.Eventually(
                interval=ltl.TimeInterval(None, 10),
                arg=Variable("b") & ltl.Eventually(Variable("c"), ltl.TimeInterval(None, 20)),
            )
        ),
        1 + 10 + 20,
    ),
    (
        "X(a U[0,5](b & F[5,20]c))",
        ltl.Next(
            ltl.Until(
                interval=ltl.TimeInterval(0, 5),
                lhs=Variable("a"),
                rhs=Variable("b") & ltl.Eventually(Variable("c"), ltl.TimeInterval(5, 20)),
            )
        ),
        1 + 5 + 20,
    ),
]


@pytest.mark.parametrize("expr,expected_ast,expected_horizon", CASES)
def test_ltl_parsing(expr: str, expected_ast: Expr, expected_horizon: int | float) -> None:
    parsed = logic_asts.parse_expr(expr, syntax="ltl")
    assert parsed == expected_ast, (parsed, expected_ast)
    assert parsed.horizon() == expected_ast.horizon() == expected_horizon
