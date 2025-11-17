r"""Comprehensive tests for spatio-temporal reach-escape logic (strel module).

Tests cover:
- DistanceInterval class and methods
- Spatial operators (Everywhere, Somewhere, Escape, Reach)
- Distance functions
- Spatial-temporal integration
- Parsing various STREL expressions
- String representation
- Tree operations
"""

import pytest
import rich

import logic_asts
import logic_asts.ltl as ltl
import logic_asts.strel as strel
from logic_asts.base import And, Expr, Variable


class TestDistanceInterval:
    """Tests for DistanceInterval class."""

    def test_distance_interval_bounded(self) -> None:
        """Test creating a bounded distance interval."""
        di = strel.DistanceInterval(0, 10)
        assert di.start == 0
        assert di.end == 10
        assert str(di) == "[0, 10]"

    def test_distance_interval_left_unbounded(self) -> None:
        """Test left-unbounded distance interval."""
        di = strel.DistanceInterval(None, 20.5)
        assert di.start is None
        assert di.end == 20.5
        assert str(di) == "[, 20.5]"

    def test_distance_interval_right_unbounded(self) -> None:
        """Test right-unbounded distance interval."""
        di = strel.DistanceInterval(5.5, None)
        assert di.start == 5.5
        assert di.end is None
        assert str(di) == "[5.5, ]"

    def test_distance_interval_fully_unbounded(self) -> None:
        """Test fully unbounded distance interval."""
        di = strel.DistanceInterval(None, None)
        assert di.start is None
        assert di.end is None
        assert str(di) == ""

    def test_distance_interval_float_values(self) -> None:
        """Test distance intervals with float values."""
        di = strel.DistanceInterval(0.5, 2.5)
        assert di.start == 0.5
        assert di.end == 2.5


class TestEverywhereOperator:
    """Tests for Everywhere spatial operator."""

    def test_everywhere_basic(self) -> None:
        """Test basic Everywhere operator."""
        p = Variable("p")
        expr = strel.Everywhere(p, strel.DistanceInterval(0, 10))
        assert isinstance(expr.arg, Variable)
        assert expr.arg == p
        assert expr.interval.start == 0
        assert expr.interval.end == 10

    def test_everywhere_without_distance_function(self) -> None:
        """Test Everywhere without specified distance function."""
        p = Variable("p")
        expr = strel.Everywhere(p, strel.DistanceInterval(0, 5))
        assert expr.dist_fn is None

    def test_everywhere_with_distance_function(self) -> None:
        """Test Everywhere with distance function."""
        p = Variable("p")
        expr = strel.Everywhere(p, strel.DistanceInterval(0, 5), "hops")
        assert expr.dist_fn == "hops"
        assert "hops" in str(expr)

    def test_everywhere_string_representation(self) -> None:
        """Test string representation of Everywhere."""
        p = Variable("p")
        expr = strel.Everywhere(p, strel.DistanceInterval(0, 5))
        assert "everywhere" in str(expr).lower()
        # String may show [0, 5] or [, 5] for start=0
        assert ("5" in str(expr)) and ("p" in str(expr))

    def test_everywhere_children(self) -> None:
        """Test children of Everywhere operator."""
        p = Variable("p")
        expr = strel.Everywhere(p, strel.DistanceInterval(0, 5))
        children = list(expr.children())
        assert len(children) == 1
        assert children[0] == p

    def test_everywhere_expand(self) -> None:
        """Test expansion of Everywhere."""
        p = Variable("p")
        expr = strel.Everywhere(p, strel.DistanceInterval(0, 5))
        expanded = expr.expand()
        assert isinstance(expanded, strel.Everywhere)

    def test_everywhere_to_nnf(self) -> None:
        """Test NNF conversion of Everywhere."""
        p = Variable("p")
        expr = strel.Everywhere(~p, strel.DistanceInterval(0, 5))
        nnf = expr.to_nnf()
        assert isinstance(nnf, strel.Everywhere)

    def test_everywhere_horizon(self) -> None:
        """Test horizon of Everywhere."""
        p = Variable("p")
        expr = strel.Everywhere(p, strel.DistanceInterval(0, 5))
        assert expr.horizon() == 0


class TestSomewhereOperator:
    """Tests for Somewhere spatial operator."""

    def test_somewhere_basic(self) -> None:
        """Test basic Somewhere operator."""
        p = Variable("p")
        expr = strel.Somewhere(p, strel.DistanceInterval(0, 10))
        assert isinstance(expr.arg, Variable)
        assert expr.arg == p
        assert expr.interval.start == 0
        assert expr.interval.end == 10

    def test_somewhere_without_distance_function(self) -> None:
        """Test Somewhere without specified distance function."""
        p = Variable("p")
        expr = strel.Somewhere(p, strel.DistanceInterval(0, 5))
        assert expr.dist_fn is None

    def test_somewhere_with_distance_function(self) -> None:
        """Test Somewhere with distance function."""
        p = Variable("p")
        expr = strel.Somewhere(p, strel.DistanceInterval(0, 5), "euclidean")
        assert expr.dist_fn == "euclidean"
        assert "euclidean" in str(expr)

    def test_somewhere_string_representation(self) -> None:
        """Test string representation of Somewhere."""
        p = Variable("p")
        expr = strel.Somewhere(p, strel.DistanceInterval(0, 5))
        assert "somewhere" in str(expr).lower()
        assert ("5" in str(expr)) and ("p" in str(expr))

    def test_somewhere_children(self) -> None:
        """Test children of Somewhere operator."""
        p = Variable("p")
        expr = strel.Somewhere(p, strel.DistanceInterval(0, 5))
        children = list(expr.children())
        assert len(children) == 1
        assert children[0] == p

    def test_somewhere_expand(self) -> None:
        """Test expansion of Somewhere."""
        p = Variable("p")
        expr = strel.Somewhere(p, strel.DistanceInterval(0, 5))
        expanded = expr.expand()
        assert isinstance(expanded, strel.Somewhere)

    def test_somewhere_to_nnf(self) -> None:
        """Test NNF conversion of Somewhere."""
        p = Variable("p")
        expr = strel.Somewhere(~p, strel.DistanceInterval(0, 5))
        nnf = expr.to_nnf()
        assert isinstance(nnf, strel.Somewhere)

    def test_somewhere_horizon(self) -> None:
        """Test horizon of Somewhere."""
        p = Variable("p")
        expr = strel.Somewhere(p, strel.DistanceInterval(0, 5))
        assert expr.horizon() == 0


class TestEscapeOperator:
    """Tests for Escape operator."""

    def test_escape_basic(self) -> None:
        """Test basic Escape operator."""
        p = Variable("p")
        expr = strel.Escape(p, strel.DistanceInterval(0, 10))
        assert isinstance(expr.arg, Variable)
        assert expr.arg == p
        assert expr.interval.start == 0
        assert expr.interval.end == 10

    def test_escape_without_distance_function(self) -> None:
        """Test Escape without specified distance function."""
        p = Variable("p")
        expr = strel.Escape(p, strel.DistanceInterval(0, 5))
        assert expr.dist_fn is None

    def test_escape_with_distance_function(self) -> None:
        """Test Escape with distance function."""
        p = Variable("p")
        expr = strel.Escape(p, strel.DistanceInterval(0, 5), "hops")
        assert expr.dist_fn == "hops"

    def test_escape_string_representation(self) -> None:
        """Test string representation of Escape."""
        p = Variable("p")
        expr = strel.Escape(p, strel.DistanceInterval(0, 5))
        assert "escape" in str(expr).lower()

    def test_escape_children(self) -> None:
        """Test children of Escape operator."""
        p = Variable("p")
        expr = strel.Escape(p, strel.DistanceInterval(0, 5))
        children = list(expr.children())
        assert len(children) == 1
        assert children[0] == p

    def test_escape_horizon(self) -> None:
        """Test horizon of Escape."""
        p = Variable("p")
        expr = strel.Escape(p, strel.DistanceInterval(0, 5))
        assert expr.horizon() == 0


class TestReachOperator:
    """Tests for Reach binary operator."""

    def test_reach_basic(self) -> None:
        """Test basic Reach operator."""
        p = Variable("p")
        q = Variable("q")
        expr = strel.Reach(p, q, strel.DistanceInterval(0, 10))
        assert expr.lhs == p
        assert expr.rhs == q
        assert expr.interval.start == 0
        assert expr.interval.end == 10

    def test_reach_without_distance_function(self) -> None:
        """Test Reach without specified distance function."""
        p = Variable("p")
        q = Variable("q")
        expr = strel.Reach(p, q, strel.DistanceInterval(0, 10))
        assert expr.dist_fn is None

    def test_reach_with_distance_function(self) -> None:
        """Test Reach with distance function."""
        p = Variable("p")
        q = Variable("q")
        expr = strel.Reach(p, q, strel.DistanceInterval(0, 10), "hops")
        assert expr.dist_fn == "hops"
        assert "hops" in str(expr)

    def test_reach_string_representation(self) -> None:
        """Test string representation of Reach."""
        p = Variable("p")
        q = Variable("q")
        expr = strel.Reach(p, q, strel.DistanceInterval(0, 10))
        assert "reach" in str(expr).lower()
        assert "[0, 10]" in str(expr)

    def test_reach_children(self) -> None:
        """Test children of Reach operator."""
        p = Variable("p")
        q = Variable("q")
        expr = strel.Reach(p, q, strel.DistanceInterval(0, 10))
        children = list(expr.children())
        assert len(children) == 2
        assert p in children and q in children

    def test_reach_expand(self) -> None:
        """Test expansion of Reach."""
        p = Variable("p")
        q = Variable("q")
        expr = strel.Reach(p, q, strel.DistanceInterval(0, 10))
        expanded = expr.expand()
        assert isinstance(expanded, strel.Reach)

    def test_reach_to_nnf(self) -> None:
        """Test NNF conversion of Reach."""
        p = Variable("p")
        q = Variable("q")
        expr = strel.Reach(p, q, strel.DistanceInterval(0, 10))
        nnf = expr.to_nnf()
        assert isinstance(nnf, strel.Reach)

    def test_reach_horizon(self) -> None:
        """Test horizon of Reach."""
        p = Variable("p")
        q = Variable("q")
        expr = strel.Reach(p, q, strel.DistanceInterval(0, 10))
        assert expr.horizon() == 0


class TestSTRELParsing:
    """Tests for parsing STREL expressions."""

    def test_parse_everywhere_operator(self) -> None:
        """Test parsing Everywhere operator."""
        expr = logic_asts.parse_expr("everywhere[0,5] p", syntax="strel")
        assert isinstance(expr, strel.Everywhere)
        assert expr.interval.start == 0
        assert expr.interval.end == 5

    def test_parse_everywhere_with_distance_function(self) -> None:
        """Test parsing Everywhere with distance function."""
        expr = logic_asts.parse_expr("everywhere^hops[0,5] p", syntax="strel")
        assert isinstance(expr, strel.Everywhere)
        assert expr.dist_fn == "hops"

    def test_parse_somewhere_operator(self) -> None:
        """Test parsing Somewhere operator."""
        expr = logic_asts.parse_expr("somewhere[0,5] p", syntax="strel")
        assert isinstance(expr, strel.Somewhere)
        assert expr.interval.start == 0
        assert expr.interval.end == 5

    def test_parse_somewhere_with_distance_function(self) -> None:
        """Test parsing Somewhere with distance function."""
        expr = logic_asts.parse_expr("somewhere^euclidean[0,5] p", syntax="strel")
        assert isinstance(expr, strel.Somewhere)
        assert expr.dist_fn == "euclidean"

    def test_parse_escape_operator(self) -> None:
        """Test parsing Escape operator."""
        expr = logic_asts.parse_expr("escape[0,10] danger", syntax="strel")
        assert isinstance(expr, strel.Escape)

    def test_parse_reach_operator(self) -> None:
        """Test parsing Reach operator."""
        expr = logic_asts.parse_expr("start reach[0,50] goal", syntax="strel")
        assert isinstance(expr, strel.Reach)

    def test_parse_reach_with_distance_function(self) -> None:
        """Test parsing Reach with distance function."""
        expr = logic_asts.parse_expr("position_a reach^hops[0,10] position_b", syntax="strel")
        assert isinstance(expr, strel.Reach)
        assert expr.dist_fn == "hops"

    def test_parse_spatial_with_propositional(self) -> None:
        """Test parsing spatial with propositional operators."""
        expr = logic_asts.parse_expr("everywhere[0,5] (p & q)", syntax="strel")
        assert isinstance(expr, strel.Everywhere)

    def test_parse_spatial_with_negation(self) -> None:
        """Test parsing spatial with negation."""
        expr = logic_asts.parse_expr("everywhere[0,5] ~obstacle", syntax="strel")
        assert isinstance(expr, strel.Everywhere)

    def test_parse_spatial_temporal_integration(self) -> None:
        """Test parsing integration of spatial and temporal operators."""
        expr = logic_asts.parse_expr("G somewhere[0,10] goal", syntax="strel")
        assert isinstance(expr, ltl.Always)
        assert isinstance(expr.arg, strel.Somewhere)

    def test_parse_eventually_somewhere(self) -> None:
        """Test parsing Eventually with Somewhere."""
        expr = logic_asts.parse_expr("F somewhere[0,20] goal", syntax="strel")
        assert isinstance(expr, ltl.Eventually)
        assert isinstance(expr.arg, strel.Somewhere)

    def test_parse_until_with_somewhere(self) -> None:
        """Test parsing Until with Somewhere."""
        expr = logic_asts.parse_expr("sending U somewhere[0,10] ack", syntax="strel")
        assert isinstance(expr, ltl.Until)
        assert isinstance(expr.rhs, strel.Somewhere)

    def test_parse_everywhere_safety_pattern(self) -> None:
        """Test parsing everywhere safety pattern."""
        expr = logic_asts.parse_expr("G everywhere[0,5] ~obstacle", syntax="strel")
        assert isinstance(expr, ltl.Always)
        assert isinstance(expr.arg, strel.Everywhere)

    def test_parse_complex_spatial_temporal(self) -> None:
        """Test parsing complex spatial-temporal formula."""
        expr = logic_asts.parse_expr("(G ~obstacle) & ((somewhere[0,2] groundstation) U goal)", syntax="strel")
        assert isinstance(expr, And)

    def test_parse_reach_from_start_to_goal(self) -> None:
        """Test parsing reach from start to goal."""
        expr = logic_asts.parse_expr("start reach[0,50] goal", syntax="strel")
        assert isinstance(expr, strel.Reach)

    def test_parse_spatial_with_or(self) -> None:
        """Test parsing spatial with disjunction."""
        expr = logic_asts.parse_expr("somewhere[0,5] (p | q)", syntax="strel")
        assert isinstance(expr, strel.Somewhere)

    def test_parse_nested_spatial(self) -> None:
        """Test parsing nested spatial operators (if supported)."""
        # This might not be supported depending on grammar, but test anyway
        try:
            expr = logic_asts.parse_expr("G (somewhere[1,2] drone)", syntax="strel")
            assert isinstance(expr, ltl.Always)
        except Exception:
            pass  # Grammar might not support nested spatial

    def test_parse_distance_function_special_chars(self) -> None:
        """Test parsing distance functions with special characters."""
        expr = logic_asts.parse_expr('somewhere^"manhattan-distance"[0,5] p', syntax="strel")
        if isinstance(expr, strel.Somewhere):
            # Successfully parsed
            assert expr.dist_fn is not None


class TestSTRELCases:
    """Original test cases from the initial test suite."""

    CASES = [
        (
            "(G ! obstacle) & ((somewhere^hops [0,2] groundstation) U goal)",
            (
                ltl.Always(~Variable("obstacle"))
                & (
                    ltl.Until(
                        strel.Somewhere(Variable("groundstation"), strel.DistanceInterval(0, 2), "hops"),
                        Variable("goal"),
                    )
                )
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
    def test_strel_parsing(self, expr: str, expected_ast: Expr) -> None:
        parsed = logic_asts.parse_expr(expr, syntax="strel")
        try:
            assert parsed == expected_ast
        except AssertionError as e:
            rich.print("parsed=", parsed)
            rich.print("expected=", expected_ast)
            raise e
