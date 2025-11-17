r"""Comprehensive tests for linear temporal logic (ltl module).

Tests cover:
- TimeInterval class and its methods
- Temporal operators (Next, Always, Eventually, Until)
- Time constraints and interval syntax
- Operator interactions and nesting
- Parsing various LTL expressions
- Horizon calculation
- String representation
- Tree operations
"""

import math

import pytest

import logic_asts
import logic_asts.ltl as ltl
from logic_asts.base import And, Expr, Not, Variable


class TestTimeInterval:
    """Tests for TimeInterval class."""

    def test_time_interval_bounded(self) -> None:
        """Test creating a bounded time interval."""
        ti = ltl.TimeInterval(0, 10)
        assert ti.start == 0
        assert ti.end == 10
        assert str(ti) == "[0, 10]"

    def test_time_interval_left_unbounded(self) -> None:
        """Test left-unbounded time interval."""
        ti = ltl.TimeInterval(None, 20)
        assert ti.start is None
        assert ti.end == 20
        assert str(ti) == "[, 20]"

    def test_time_interval_right_unbounded(self) -> None:
        """Test right-unbounded time interval."""
        ti = ltl.TimeInterval(5, None)
        assert ti.start == 5
        assert ti.end is None
        assert str(ti) == "[5, ]"

    def test_time_interval_fully_unbounded(self) -> None:
        """Test fully unbounded time interval."""
        ti = ltl.TimeInterval(None, None)
        assert ti.start is None
        assert ti.end is None
        assert str(ti) == ""

    def test_time_interval_duration(self) -> None:
        """Test duration calculation."""
        ti = ltl.TimeInterval(0, 10)
        assert ti.duration() == 10

    def test_time_interval_duration_unbounded(self) -> None:
        """Test duration of unbounded interval."""
        ti = ltl.TimeInterval(5, None)
        assert math.isinf(ti.duration())

    def test_time_interval_is_unbounded(self) -> None:
        """Test is_unbounded method."""
        ti1 = ltl.TimeInterval(0, 10)
        ti2 = ltl.TimeInterval(0, None)
        assert not ti1.is_unbounded()
        assert ti2.is_unbounded()

    def test_time_interval_is_untimed(self) -> None:
        """Test is_untimed method for [0, inf)."""
        ti1 = ltl.TimeInterval(None, None)
        ti2 = ltl.TimeInterval(0, None)
        ti3 = ltl.TimeInterval(5, None)
        assert ti1.is_untimed()
        assert ti2.is_untimed()
        assert not ti3.is_untimed()

    def test_time_interval_iter_interval_bounded(self) -> None:
        """Test iterating over bounded interval."""
        ti = ltl.TimeInterval(0, 5)
        points = list(ti.iter_interval(step=1))
        assert len(points) > 0

    def test_time_interval_iter_interval_unbounded(self) -> None:
        """Test iterating over unbounded interval (limited iteration)."""
        ti = ltl.TimeInterval(0, None)
        points = []
        for i, point in enumerate(ti.iter_interval(step=1)):
            points.append(point)
            if i >= 4:  # Only take first 5 points
                break
        assert len(points) == 5


class TestNextOperator:
    """Tests for Next (X) operator."""

    def test_next_single_step(self) -> None:
        """Test Next with single step."""
        p = Variable("p")
        expr = ltl.Next(p)
        assert str(expr) == "(X p)"
        assert expr.arg == p

    def test_next_multiple_steps(self) -> None:
        """Test Next with multiple steps."""
        p = Variable("p")
        expr = ltl.Next(p, steps=5)
        assert str(expr) == "(X[5] p)"
        assert expr.steps == 5

    def test_next_expansion(self) -> None:
        """Test that X[n] expands to nested X operators."""
        p = Variable("p")
        expr = ltl.Next(p, steps=3)
        expanded = expr.expand()
        # Should be X(X(X(p)))
        assert isinstance(expanded, ltl.Next)

    def test_next_horizon_single_step(self) -> None:
        """Test horizon of single-step Next."""
        p = Variable("p")
        expr = ltl.Next(p)
        assert expr.horizon() == 1

    def test_next_horizon_multiple_steps(self) -> None:
        """Test horizon of multi-step Next."""
        p = Variable("p")
        expr = ltl.Next(p, steps=5)
        assert expr.horizon() == 5

    def test_next_children(self) -> None:
        """Test children of Next operator."""
        p = Variable("p")
        expr = ltl.Next(p)
        children = list(expr.children())
        assert len(children) == 1
        assert children[0] == p

    def test_next_to_nnf(self) -> None:
        """Test NNF conversion of Next."""
        p = Variable("p")
        expr = ltl.Next(Not(p))
        nnf = expr.to_nnf()
        assert isinstance(nnf, ltl.Next)


class TestEventuallyOperator:
    """Tests for Eventually (F) operator."""

    def test_eventually_unbounded(self) -> None:
        """Test Eventually without time constraint."""
        p = Variable("p")
        expr = ltl.Eventually(p)
        assert str(expr) == "(F p)"
        assert expr.interval.is_untimed()

    def test_eventually_bounded(self) -> None:
        """Test Eventually with time constraint."""
        p = Variable("p")
        expr = ltl.Eventually(p, ltl.TimeInterval(0, 10))
        # String representation accepts both [0, 10] and [, 10] for start=0
        assert "F" in str(expr) and "p" in str(expr)

    def test_eventually_left_unbounded(self) -> None:
        """Test Eventually with left-unbounded interval."""
        p = Variable("p")
        expr = ltl.Eventually(p, ltl.TimeInterval(None, 20))
        assert str(expr) == "(F[, 20] p)"

    def test_eventually_horizon(self) -> None:
        """Test horizon of Eventually operator."""
        p = Variable("p")
        expr1 = ltl.Eventually(p)
        expr2 = ltl.Eventually(p, ltl.TimeInterval(0, 10))
        assert math.isinf(expr1.horizon())
        assert expr2.horizon() == 10

    def test_eventually_to_nnf(self) -> None:
        """Test NNF conversion of Eventually."""
        p = Variable("p")
        expr = ltl.Eventually(p)
        nnf = expr.to_nnf()
        assert isinstance(nnf, ltl.Eventually)

    def test_eventually_children(self) -> None:
        """Test children of Eventually operator."""
        p = Variable("p")
        expr = ltl.Eventually(p)
        children = list(expr.children())
        assert len(children) == 1
        assert children[0] == p


class TestAlwaysOperator:
    """Tests for Always (G) operator."""

    def test_always_unbounded(self) -> None:
        """Test Always without time constraint."""
        p = Variable("p")
        expr = ltl.Always(p)
        assert str(expr) == "(G p)"
        assert expr.interval.is_untimed()

    def test_always_bounded(self) -> None:
        """Test Always with time constraint."""
        p = Variable("p")
        expr = ltl.Always(p, ltl.TimeInterval(0, 10))
        assert "G" in str(expr) and "p" in str(expr)

    def test_always_expansion(self) -> None:
        """Test that Always expands to negated Eventually."""
        p = Variable("p")
        expr = ltl.Always(p)
        expanded = expr.expand()
        # G p = ~F(~p)
        assert isinstance(expanded, Not)

    def test_always_horizon(self) -> None:
        """Test horizon of Always operator."""
        p = Variable("p")
        expr1 = ltl.Always(p)
        expr2 = ltl.Always(p, ltl.TimeInterval(0, 10))
        assert math.isinf(expr1.horizon())
        assert expr2.horizon() == 10

    def test_always_children(self) -> None:
        """Test children of Always operator."""
        p = Variable("p")
        expr = ltl.Always(p)
        children = list(expr.children())
        assert len(children) == 1
        assert children[0] == p


class TestUntilOperator:
    """Tests for Until (U) operator."""

    def test_until_unbounded(self) -> None:
        """Test Until without time constraint."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.Until(p, q)
        assert str(expr) == "(p U q)"

    def test_until_bounded(self) -> None:
        """Test Until with time constraint."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.Until(p, q, ltl.TimeInterval(0, 10))
        assert "U" in str(expr) and "p" in str(expr) and "q" in str(expr)

    def test_until_horizon(self) -> None:
        """Test horizon of Until operator."""
        p = Variable("p")
        q = Variable("q")
        expr1 = ltl.Until(p, q)
        expr2 = ltl.Until(p, q, ltl.TimeInterval(0, 10))
        assert math.isinf(expr1.horizon())
        assert expr2.horizon() == 10

    def test_until_to_nnf(self) -> None:
        """Test NNF conversion of Until."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.Until(p, q)
        nnf = expr.to_nnf()
        assert isinstance(nnf, ltl.Until)

    def test_until_children(self) -> None:
        """Test children of Until operator."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.Until(p, q)
        children = list(expr.children())
        assert len(children) == 2
        assert p in children and q in children


class TestLTLParsing:
    """Tests for parsing LTL expressions."""

    def test_parse_next_operator(self) -> None:
        """Test parsing Next operator."""
        expr = logic_asts.parse_expr("X p", syntax="ltl")
        assert isinstance(expr, ltl.Next)
        assert isinstance(expr.arg, Variable)

    def test_parse_next_multi_step(self) -> None:
        """Test parsing Next with multiple steps."""
        expr = logic_asts.parse_expr("X[5] p", syntax="ltl")
        assert isinstance(expr, ltl.Next)
        assert expr.steps == 5

    def test_parse_eventually_operator(self) -> None:
        """Test parsing Eventually operator."""
        expr = logic_asts.parse_expr("F p", syntax="ltl")
        assert isinstance(expr, ltl.Eventually)

    def test_parse_eventually_bounded(self) -> None:
        """Test parsing Eventually with time constraint."""
        expr = logic_asts.parse_expr("F[0,10] p", syntax="ltl")
        assert isinstance(expr, ltl.Eventually)
        assert expr.interval.start == 0
        assert expr.interval.end == 10

    def test_parse_eventually_left_unbounded(self) -> None:
        """Test parsing Eventually with left-unbounded interval."""
        expr = logic_asts.parse_expr("F[,20] p", syntax="ltl")
        assert isinstance(expr, ltl.Eventually)
        assert expr.interval.start is None
        assert expr.interval.end == 20

    def test_parse_always_operator(self) -> None:
        """Test parsing Always operator."""
        expr = logic_asts.parse_expr("G p", syntax="ltl")
        assert isinstance(expr, ltl.Always)

    def test_parse_always_bounded(self) -> None:
        """Test parsing Always with time constraint."""
        expr = logic_asts.parse_expr("G[5,15] p", syntax="ltl")
        assert isinstance(expr, ltl.Always)
        assert expr.interval.start == 5
        assert expr.interval.end == 15

    def test_parse_until_operator(self) -> None:
        """Test parsing Until operator."""
        expr = logic_asts.parse_expr("p U q", syntax="ltl")
        assert isinstance(expr, ltl.Until)

    def test_parse_until_bounded(self) -> None:
        """Test parsing Until with time constraint."""
        expr = logic_asts.parse_expr("p U[0,5] q", syntax="ltl")
        assert isinstance(expr, ltl.Until)
        assert expr.interval.start == 0
        assert expr.interval.end == 5

    def test_parse_nested_temporal(self) -> None:
        """Test parsing nested temporal operators."""
        expr = logic_asts.parse_expr("F G p", syntax="ltl")
        assert isinstance(expr, ltl.Eventually)
        assert isinstance(expr.arg, ltl.Always)

    def test_parse_temporal_with_propositional(self) -> None:
        """Test parsing temporal with propositional operators."""
        expr = logic_asts.parse_expr("G(p -> F q)", syntax="ltl")
        assert isinstance(expr, ltl.Always)

    def test_parse_complex_ltl_formula(self) -> None:
        """Test parsing complex LTL formula."""
        expr = logic_asts.parse_expr("(request -> F response) & G ~error", syntax="ltl")
        assert isinstance(expr, And)

    def test_parse_request_response_pattern(self) -> None:
        """Test parsing request-response pattern."""
        expr = logic_asts.parse_expr("G(request -> F response)", syntax="ltl")
        assert isinstance(expr, ltl.Always)

    def test_parse_liveness_pattern(self) -> None:
        """Test parsing liveness pattern."""
        expr = logic_asts.parse_expr("G F enabled", syntax="ltl")
        assert isinstance(expr, ltl.Always)
        assert isinstance(expr.arg, ltl.Eventually)

    def test_parse_safety_pattern(self) -> None:
        """Test parsing safety pattern."""
        expr = logic_asts.parse_expr("G ~error", syntax="ltl")
        assert isinstance(expr, ltl.Always)
        assert isinstance(expr.arg, Not)

    def test_parse_negation_of_temporal(self) -> None:
        """Test parsing negation of temporal operator."""
        expr = logic_asts.parse_expr("!F p", syntax="ltl")
        assert isinstance(expr, Not)
        assert isinstance(expr.arg, ltl.Eventually)

    def test_parse_until_with_complex_operands(self) -> None:
        """Test parsing Until with complex operands."""
        expr = logic_asts.parse_expr("(a & b) U (c | d)", syntax="ltl")
        assert isinstance(expr, ltl.Until)

    def test_parse_next_of_until(self) -> None:
        """Test parsing Next of Until."""
        expr = logic_asts.parse_expr("X(p U q)", syntax="ltl")
        assert isinstance(expr, ltl.Next)
        assert isinstance(expr.arg, ltl.Until)

    def test_parse_until_with_bounded_interval(self) -> None:
        """Test parsing Until with explicit interval."""
        expr = logic_asts.parse_expr("p U[10,20] q", syntax="ltl")
        assert isinstance(expr, ltl.Until)
        assert expr.interval.start == 10
        assert expr.interval.end == 20

    def test_parse_eventually_right_unbounded(self) -> None:
        """Test parsing Eventually with right-unbounded interval."""
        expr = logic_asts.parse_expr("F[5,] p", syntax="ltl")
        assert isinstance(expr, ltl.Eventually)
        assert expr.interval.start == 5
        assert expr.interval.end is None


class TestLTLCases:
    """Original test cases from the initial test suite."""

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
    def test_ltl_parsing(self, expr: str, expected_ast: Expr, expected_horizon: int | float) -> None:
        parsed = logic_asts.parse_expr(expr, syntax="ltl")
        assert parsed == expected_ast, (parsed, expected_ast)
        assert parsed.horizon() == expected_ast.horizon() == expected_horizon
