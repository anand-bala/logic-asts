r"""Integration tests for cross-logic interactions.

Tests cover interactions between different logical systems:
- Propositional logic combined with temporal operators
- Propositional logic combined with spatial operators
- Temporal operators combined with spatial operators
- Complex nested formulas spanning multiple logics
- Parsing, NNF conversion, horizon calculation across logics
"""

import math

import logic_asts
from logic_asts.base import And, Implies, Not, Or, Variable
from logic_asts.ltl import Always, Eventually, Next, TimeInterval, Until
from logic_asts.strel import DistanceInterval, Escape, Everywhere, Reach, Somewhere


class TestLTLWithPropositional:
    """Tests for temporal operators with propositional logic."""

    def test_next_with_conjunction(self) -> None:
        """Test Next operator containing conjunction."""
        p = Variable("p")
        q = Variable("q")
        expr = Next(p & q)
        assert isinstance(expr.arg, And)
        assert expr.horizon() == 1

    def test_eventually_with_disjunction(self) -> None:
        """Test Eventually operator containing disjunction."""
        p = Variable("p")
        q = Variable("q")
        expr = Eventually(p | q)
        assert isinstance(expr.arg, Or)
        assert math.isinf(expr.horizon())

    def test_always_with_implication(self) -> None:
        """Test Always operator containing implication."""
        p = Variable("p")
        q = Variable("q")
        expr = Always(Implies(p, q))
        assert isinstance(expr.arg, Implies)
        assert math.isinf(expr.horizon())

    def test_until_with_complex_propositional(self) -> None:
        """Test Until with complex propositional formulas."""
        p = Variable("p")
        q = Variable("q")
        r = Variable("r")
        lhs = (p & q) | r
        rhs = ~p
        expr = Until(lhs, rhs)
        assert isinstance(expr.lhs, Or)
        assert isinstance(expr.rhs, Not)
        assert math.isinf(expr.horizon())

    def test_nested_temporal_with_propositional(self) -> None:
        """Test nested temporal operators with propositional."""
        p = Variable("p")
        expr = Eventually(Always(Not(p)))
        assert isinstance(expr.arg, Always)
        assert math.isinf(expr.horizon())

    def test_temporal_with_propositional_parsing(self) -> None:
        """Test parsing temporal operators with propositional operators."""
        expr = logic_asts.parse_expr("G(p & q -> F r)", syntax="ltl")
        assert isinstance(expr, Always)
        assert isinstance(expr.arg, Implies)

    def test_next_multi_step_with_propositional(self) -> None:
        """Test multi-step Next with propositional operators."""
        p = Variable("p")
        q = Variable("q")
        expr = Next(p | q, steps=3)
        assert expr.steps == 3
        assert expr.horizon() == 3

    def test_eventually_with_time_constraint_and_propositional(self) -> None:
        """Test Eventually with time constraint and propositional logic."""
        p = Variable("p")
        q = Variable("q")
        expr = Eventually(p & q, TimeInterval(0, 10))
        assert expr.interval.end == 10
        assert expr.horizon() == 10


class TestSTRELWithPropositional:
    """Tests for spatial operators with propositional logic."""

    def test_everywhere_with_conjunction(self) -> None:
        """Test Everywhere operator containing conjunction."""
        p = Variable("p")
        q = Variable("q")
        expr = Everywhere(p & q, DistanceInterval(0, 10))
        assert isinstance(expr.arg, And)
        assert expr.horizon() == 0

    def test_somewhere_with_disjunction(self) -> None:
        """Test Somewhere operator containing disjunction."""
        p = Variable("p")
        q = Variable("q")
        expr = Somewhere(p | q, DistanceInterval(0, 5))
        assert isinstance(expr.arg, Or)
        assert expr.horizon() == 0

    def test_escape_with_negation(self) -> None:
        """Test Escape operator containing negation."""
        p = Variable("p")
        expr = Escape(~p, DistanceInterval(0, 20))
        assert isinstance(expr.arg, Not)
        assert expr.horizon() == 0

    def test_reach_with_implication(self) -> None:
        """Test Reach operator with implication."""
        p = Variable("p")
        q = Variable("q")
        expr = Reach(Implies(p, q), q, DistanceInterval(0, 10))
        assert isinstance(expr.lhs, Implies)
        assert expr.horizon() == 0

    def test_spatial_with_propositional_parsing(self) -> None:
        """Test parsing spatial operators with propositional operators."""
        expr = logic_asts.parse_expr("everywhere[0,5] (p & q)", syntax="strel")
        assert isinstance(expr, Everywhere)
        assert isinstance(expr.arg, And)


class TestLTLWithSTREL:
    """Tests for temporal operators with spatial operators."""

    def test_next_containing_spatial(self) -> None:
        """Test Next operator containing spatial operator."""
        p = Variable("p")
        expr = Next(Somewhere(p, DistanceInterval(0, 5)))
        assert isinstance(expr.arg, Somewhere)
        assert expr.horizon() == 1

    def test_eventually_containing_spatial(self) -> None:
        """Test Eventually operator containing spatial operator."""
        p = Variable("p")
        expr = Eventually(Everywhere(p, DistanceInterval(0, 10)))
        assert isinstance(expr.arg, Everywhere)
        assert math.isinf(expr.horizon())

    def test_always_containing_spatial(self) -> None:
        """Test Always operator containing spatial operator."""
        p = Variable("p")
        expr = Always(Reach(p, p, DistanceInterval(0, 15)))
        assert isinstance(expr.arg, Reach)
        assert math.isinf(expr.horizon())

    def test_until_with_spatial_operands(self) -> None:
        """Test Until operator with spatial operands."""
        p = Variable("p")
        q = Variable("q")
        expr = Until(Somewhere(p, DistanceInterval(0, 5)), Everywhere(q, DistanceInterval(0, 10)))
        assert isinstance(expr.lhs, Somewhere)
        assert isinstance(expr.rhs, Everywhere)

    def test_spatial_containing_temporal(self) -> None:
        """Test spatial operator containing temporal operator."""
        p = Variable("p")
        expr = Somewhere(Eventually(p), DistanceInterval(0, 5))
        assert isinstance(expr.arg, Eventually)
        assert math.isinf(expr.horizon())

    def test_temporal_spatial_temporal_nesting(self) -> None:
        """Test deeply nested temporal and spatial operators."""
        p = Variable("p")
        expr = Eventually(Somewhere(p, DistanceInterval(0, 5)), TimeInterval(0, 10))
        assert isinstance(expr.arg, Somewhere)
        assert isinstance(expr.arg.arg, Variable)
        assert expr.horizon() == 10

    def test_cross_logic_parsing(self) -> None:
        """Test parsing expressions mixing temporal and spatial."""
        expr = logic_asts.parse_expr("G somewhere[0,10] F p", syntax="strel")
        assert isinstance(expr, Always)
        assert isinstance(expr.arg, Somewhere)

    def test_cross_logic_nnf_conversion(self) -> None:
        """Test NNF conversion with temporal and spatial."""
        p = Variable("p")
        expr = ~Always(Somewhere(p, DistanceInterval(0, 5)))
        nnf = expr.to_nnf()
        # ~G spatial = F ~spatial
        assert isinstance(nnf, Eventually)

    def test_next_of_spatial_expansion(self) -> None:
        """Test expansion of Next containing spatial operator."""
        p = Variable("p")
        expr = Next(Somewhere(p, DistanceInterval(0, 5)), steps=2)
        expanded = expr.expand()
        # X[2] somewhere[0,5] p -> X(X(somewhere[0,5] p))
        assert isinstance(expanded, Next)

    def test_horizon_calculation_cross_logic(self) -> None:
        """Test horizon calculation with mixed temporal and spatial."""
        p = Variable("p")
        # Always contains Somewhere, temporal contribution comes from Always
        expr = Always(Somewhere(p, DistanceInterval(0, 5)), TimeInterval(0, 20))
        assert expr.horizon() == 20


class TestComplexIntegration:
    """Tests for complex multi-logic formulas."""

    def test_request_response_with_spatial_constraint(self) -> None:
        """Test request-response pattern with spatial constraint."""
        request = Variable("request")
        response = Variable("response")
        server = Variable("server")

        # G (request -> F somewhere[0,10] (response & server))
        expr = Always(Implies(request, Eventually(Somewhere(response & server, DistanceInterval(0, 10)))))
        assert isinstance(expr.arg, Implies)
        assert math.isinf(expr.horizon())

    def test_safety_with_spatial_everywhere(self) -> None:
        """Test safety property with universal spatial constraint."""
        safe = Variable("safe")
        obstacle = Variable("obstacle")

        # G everywhere[0,50] (safe & ~obstacle)
        expr = Always(Everywhere(safe & ~obstacle, DistanceInterval(0, 50)))
        assert math.isinf(expr.horizon())

    def test_liveness_with_reach_constraint(self) -> None:
        """Test liveness property with reachability constraint."""
        enabled = Variable("enabled")
        goal = Variable("goal")

        # G F reach[0,100](enabled, goal)
        expr = Always(Eventually(Reach(enabled, goal, DistanceInterval(0, 100))))
        assert math.isinf(expr.horizon())

    def test_temporal_spatial_propositional_nnf(self) -> None:
        """Test NNF conversion for deeply mixed formula."""
        p = Variable("p")
        _q = Variable("q")

        # ~(G somewhere[0,10] p)
        expr = ~Always(Somewhere(p, DistanceInterval(0, 10)))
        nnf = expr.to_nnf()
        # ~G F somewhere = F E everywhere (roughly)
        assert isinstance(nnf, Eventually)

    def test_multi_operator_combination(self) -> None:
        """Test multiple operators in different combinations."""
        a = Variable("a")
        b = Variable("b")
        c = Variable("c")

        # F[0,10] (G everywhere[0,5] (a & b) | X F somewhere[0,20] c)
        expr = Eventually(
            Always(Everywhere(a & b, DistanceInterval(0, 5))) | Next(Eventually(Somewhere(c, DistanceInterval(0, 20)))),
            TimeInterval(0, 10),
        )
        assert math.isinf(expr.horizon())

    def test_escape_from_temporal_property(self) -> None:
        """Test escape operator with temporal constraints."""
        danger = Variable("danger")
        safe = Variable("safe")

        # X[5] (escape[0,50] danger & F[0,10] safe)
        expr = Next(Escape(danger, DistanceInterval(0, 50)) & Eventually(safe, TimeInterval(0, 10)), steps=5)
        # Next contributes 5, Eventually contributes 10, total max
        assert expr.horizon() == 15

    def test_parsing_complex_cross_logic_formula(self) -> None:
        """Test parsing of complex cross-logic formula."""
        expr = logic_asts.parse_expr("G(a -> F somewhere[0,10] (b & c))", syntax="strel")
        assert isinstance(expr, Always)
        assert isinstance(expr.arg, Implies)
        assert isinstance(expr.arg.rhs, Eventually)

    def test_children_iteration_cross_logic(self) -> None:
        """Test children iteration with cross-logic formula."""
        p = Variable("p")
        q = Variable("q")

        expr = Eventually(Somewhere(p & q, DistanceInterval(0, 10)))

        children = list(expr.children())
        assert len(children) == 1
        assert isinstance(children[0], Somewhere)

        spatial_children = list(children[0].children())
        assert len(spatial_children) == 1
        assert isinstance(spatial_children[0], And)

    def test_tree_operations_cross_logic(self) -> None:
        """Test full tree iteration with cross-logic formula."""
        p = Variable("p")
        q = Variable("q")

        expr = Always(Somewhere(Eventually(p & q), DistanceInterval(0, 5)))

        # Collect all subtrees
        subtrees = list(expr.iter_subtree())
        assert len(subtrees) > 1
        assert any(isinstance(st, Always) for st in subtrees)
        assert any(isinstance(st, Somewhere) for st in subtrees)
        assert any(isinstance(st, Eventually) for st in subtrees)


class TestParsingSyntaxConsistency:
    """Tests for parsing consistency across logics."""

    def test_base_operators_in_ltl_context(self) -> None:
        """Test that base operators work correctly in LTL context."""
        expr_ltl = logic_asts.parse_expr("p & q | r", syntax="ltl")
        expr_base = logic_asts.parse_expr("p & q | r", syntax="base")

        # Should have same structure
        assert expr_ltl == expr_base

    def test_base_operators_in_strel_context(self) -> None:
        """Test that base operators work correctly in STREL context."""
        expr_strel = logic_asts.parse_expr("p & q | r", syntax="strel")
        expr_base = logic_asts.parse_expr("p & q | r", syntax="base")

        # Should have same structure
        assert expr_strel == expr_base

    def test_temporal_in_strel_context(self) -> None:
        """Test that temporal operators work in STREL context."""
        expr = logic_asts.parse_expr("F p", syntax="strel")
        assert isinstance(expr, Eventually)

    def test_spatial_operators_require_strel(self) -> None:
        """Test that spatial operators work only in STREL context."""
        # Parsing spatial in STREL should work
        expr = logic_asts.parse_expr("somewhere[0,10] p", syntax="strel")
        assert isinstance(expr, Somewhere)

    def test_cross_logic_mixed_operators(self) -> None:
        """Test parsing with mixed operator types."""
        # This should parse both temporal and spatial in STREL
        expr = logic_asts.parse_expr("G(p -> F somewhere[0,5] q)", syntax="strel")
        assert isinstance(expr, Always)
        assert isinstance(expr.arg, Implies)
        assert isinstance(expr.arg.rhs, Eventually)


class TestHorizonCalculationConsistency:
    """Tests for horizon calculation across logics."""

    def test_horizon_unbounded_temporal(self) -> None:
        """Test horizon of unbounded temporal operators."""
        p = Variable("p")

        expr1 = Eventually(p)
        expr2 = Always(p)

        assert math.isinf(expr1.horizon())
        assert math.isinf(expr2.horizon())

    def test_horizon_bounded_temporal(self) -> None:
        """Test horizon of bounded temporal operators."""
        p = Variable("p")

        expr1 = Eventually(p, TimeInterval(0, 10))
        expr2 = Always(p, TimeInterval(0, 10))

        assert expr1.horizon() == 10
        assert expr2.horizon() == 10

    def test_horizon_spatial_no_contribution(self) -> None:
        """Test that spatial operators don't contribute to horizon."""
        p = Variable("p")

        expr1 = Somewhere(p, DistanceInterval(0, 100))
        expr2 = Everywhere(p, DistanceInterval(0, 100))

        # Spatial operators don't add to horizon
        assert expr1.horizon() == 0
        assert expr2.horizon() == 0

    def test_horizon_mixed_temporal_spatial(self) -> None:
        """Test horizon calculation with mixed temporal and spatial."""
        p = Variable("p")

        # Temporal contributes to horizon, spatial doesn't
        expr = Eventually(Somewhere(p, DistanceInterval(0, 100)), TimeInterval(0, 20))
        assert expr.horizon() == 20

    def test_horizon_nested_eventually(self) -> None:
        """Test horizon of nested Eventually operators."""
        p = Variable("p")

        expr = Eventually(Eventually(p, TimeInterval(0, 5)), TimeInterval(0, 10))
        # Outer contributes 10, inner contributes 5
        # horizon = 10 + 5 = 15
        assert expr.horizon() == 15
