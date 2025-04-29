import pytest
import rich

import logic_asts
import logic_asts.ltl as ltl
import logic_asts.strel as strel
from logic_asts.base import Expr, Variable

CASES = [
    (
        "(G ! obstacle) & ((somewhere^hops [0,2] groundstation) U goal)",
        (
            ltl.Always(~Variable("obstacle"))
            & (ltl.Until(strel.Somewhere(Variable("groundstation"), strel.DistanceInterval(0, 2), "hops"), Variable("goal")))
        ),
    ),
    (
        "G( (somewhere[1,2] drone) | (F[0, 100] somewhere[1,2] (drone | groundstation)) )",
        ltl.Always(
            strel.Somewhere(Variable("drone"), strel.DistanceInterval(1, 2))
            | ltl.Eventually(
                strel.Somewhere(Variable("drone") | Variable("groundstation"), strel.DistanceInterval(1, 2)),
                ltl.TimeInterval(0, 100),
            )
        ),
    ),
]


@pytest.mark.parametrize("expr,expected_ast", CASES)
def test_strel_parsing(expr: str, expected_ast: Expr) -> None:
    parsed = logic_asts.parse_expr(expr, syntax="strel")
    try:
        assert parsed == expected_ast
    except AssertionError as e:
        rich.print("parsed=", parsed)
        rich.print("expected=", expected_ast)
        raise e
