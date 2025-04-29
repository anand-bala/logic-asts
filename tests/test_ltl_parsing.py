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
    ),
    ("!Fp2", Not(ltl.Eventually(Variable("p2")))),
]


@pytest.mark.parametrize("expr,expected_ast", CASES)
def test_ltl_parsing(expr: str, expected_ast: Expr) -> None:
    parsed = logic_asts.parse_expr(expr, syntax="ltl")
    assert parsed == expected_ast, (parsed, expected_ast)
