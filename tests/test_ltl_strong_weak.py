r"""Tests for strong/weak LTL operators (StrongNext, WeakUntil, StrongRelease).

Tests cover:
- AST construction and string representation
- Negative Normal Form (NNF) rules from Spot section 2.1
- Expand correctness for strong/weak operators
- Parser acceptance table for new operator forms
- Round-trip parsing stability
"""

import typing

import pytest

import logic_asts
import logic_asts.ltl as ltl
from logic_asts.base import Not, Variable
from logic_asts.spec import Expr


class TestASTConstruction:
    """Test AST construction and string representation of new operators."""

    def test_strong_next_single_step(self) -> None:
        """Test StrongNext with single step."""
        p = Variable("p")
        expr = ltl.StrongNext(p)
        assert str(expr) == "(X[!] p)"
        assert expr.arg == p

    def test_strong_next_multiple_steps(self) -> None:
        """Test StrongNext with multiple steps."""
        p = Variable("p")
        expr = ltl.StrongNext(p, steps=3)
        assert str(expr) == "(X[3!] p)"
        assert expr.steps == 3

    def test_weak_until(self) -> None:
        """Test WeakUntil construction and string representation."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.WeakUntil(p, q)
        assert str(expr) == "(p W q)"
        assert expr.lhs == p
        assert expr.rhs == q

    def test_strong_release(self) -> None:
        """Test StrongRelease construction and string representation."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.StrongRelease(p, q)
        assert str(expr) == "(p M q)"
        assert expr.lhs == p
        assert expr.rhs == q


class TestNNFRules:
    """Test Negative Normal Form rules from Spot section 2.1."""

    def test_nnf_negation_of_weak_next(self) -> None:
        """Test ~X p = X[!] ~p (strong-X dual of weak-X)."""
        p = Variable("p")
        expr = ~ltl.Next(p)
        nnf = expr.to_nnf()
        expected = ltl.StrongNext(Not(p))
        assert nnf == expected, f"Expected {expected}, got {nnf}"

    def test_nnf_negation_of_strong_next(self) -> None:
        """Test ~X[!] p = X ~p (weak-X dual of strong-X)."""
        p = Variable("p")
        expr = ~ltl.StrongNext(p)
        nnf = expr.to_nnf()
        expected = ltl.Next(Not(p))
        assert nnf == expected, f"Expected {expected}, got {nnf}"

    def test_nnf_negation_of_eventually(self) -> None:
        """Test ~F p = G ~p."""
        p = Variable("p")
        expr = ~ltl.Eventually(p)
        nnf = expr.to_nnf()
        expected = ltl.Always(Not(p))
        assert nnf == expected, f"Expected {expected}, got {nnf}"

    def test_nnf_negation_of_always(self) -> None:
        """Test ~G p = F ~p."""
        p = Variable("p")
        expr = ~ltl.Always(p)
        nnf = expr.to_nnf()
        expected = ltl.Eventually(Not(p))
        assert nnf == expected, f"Expected {expected}, got {nnf}"

    def test_nnf_negation_of_until(self) -> None:
        """Test ~(p U q) = (~p) R (~q)."""
        p = Variable("p")
        q = Variable("q")
        expr = ~ltl.Until(p, q)
        nnf = expr.to_nnf()
        expected = ltl.Release(Not(p), Not(q))
        assert nnf == expected, f"Expected {expected}, got {nnf}"

    def test_nnf_negation_of_weak_until(self) -> None:
        """Test ~(p W q) = (~p) M (~q)."""
        p = Variable("p")
        q = Variable("q")
        expr = ~ltl.WeakUntil(p, q)
        nnf = expr.to_nnf()
        expected = ltl.StrongRelease(Not(p), Not(q))
        assert nnf == expected, f"Expected {expected}, got {nnf}"

    def test_nnf_negation_of_release(self) -> None:
        """Test ~(p R q) = (~p) U (~q)."""
        p = Variable("p")
        q = Variable("q")
        expr = ~ltl.Release(p, q)
        nnf = expr.to_nnf()
        expected = ltl.Until(Not(p), Not(q))
        assert nnf == expected, f"Expected {expected}, got {nnf}"

    def test_nnf_negation_of_strong_release(self) -> None:
        """Test ~(p M q) = (~p) W (~q)."""
        p = Variable("p")
        q = Variable("q")
        expr = ~ltl.StrongRelease(p, q)
        nnf = expr.to_nnf()
        expected = ltl.WeakUntil(Not(p), Not(q))
        assert nnf == expected, f"Expected {expected}, got {nnf}"

    def test_nnf_idempotence_mixed_formula(self) -> None:
        """Test that to_nnf(to_nnf(e)) == to_nnf(e) for mixed formula."""
        p = Variable("p")
        q = Variable("q")
        # Create a mixed formula with multiple operators
        expr = ~(ltl.Until(ltl.Next(p), ltl.WeakUntil(q, ~p)))
        nnf1 = expr.to_nnf()
        nnf2 = nnf1.to_nnf()
        assert nnf1 == nnf2, "NNF should be idempotent"


class TestExpandCorrectness:
    """Test expand() correctness for new operators."""

    def test_strong_next_expand_multiple_steps(self) -> None:
        """Test that StrongNext with steps > 1 expands to nested StrongNext."""
        p = Variable("p")
        expr = ltl.StrongNext(p, steps=3)
        expanded = expr.expand()
        # Should be StrongNext(StrongNext(StrongNext(p)))
        assert isinstance(expanded, ltl.StrongNext)
        # Walk down and verify all are StrongNext
        current: object = expanded
        depth = 0
        while isinstance(current, ltl.StrongNext):
            depth += 1
            current = current.arg
        assert depth == 3
        assert current == p

    def test_eventually_strong_flag_produces_strong_next(self) -> None:
        """Test that Eventually with strong=True and bounded interval expands to StrongNext."""
        p = Variable("p")
        expr = ltl.Eventually(p, ltl.TimeInterval(0, 3), strong=True)
        expanded = expr.expand()
        # Walk the tree and collect all Next/StrongNext operators
        all_next_types: set[str] = set()

        def collect_next_types(node: Expr) -> None:
            if isinstance(node, ltl.StrongNext):
                all_next_types.add("StrongNext")
            elif isinstance(node, ltl.Next):
                all_next_types.add("Next")
            for child in node.children():
                collect_next_types(child)

        collect_next_types(expanded)
        # Should contain StrongNext and not contain Next
        assert "StrongNext" in all_next_types, "Expected StrongNext in expanded tree"
        assert "Next" not in all_next_types, "Should not have weak Next when strong=True"

    def test_eventually_weak_flag_produces_weak_next(self) -> None:
        """Test that Eventually with strong=False and bounded interval expands to Next."""
        p = Variable("p")
        expr = ltl.Eventually(p, ltl.TimeInterval(0, 3), strong=False)
        expanded = expr.expand()
        # Walk the tree and collect all Next/StrongNext operators
        all_next_types: set[str] = set()

        def collect_next_types(node: Expr) -> None:
            if isinstance(node, ltl.StrongNext):
                all_next_types.add("StrongNext")
            elif isinstance(node, ltl.Next):
                all_next_types.add("Next")
            for child in node.children():
                collect_next_types(child)

        collect_next_types(expanded)
        # Should contain Next and not contain StrongNext
        assert "Next" in all_next_types, "Expected Next in expanded tree"
        assert "StrongNext" not in all_next_types, "Should not have StrongNext when strong=False"


class TestParserAcceptance:
    """Test parser acceptance of all operator forms."""

    @pytest.mark.parametrize(
        ["input_str", "expected_type", "extra_checks"],
        [
            # Weak Next forms
            ("X p", ltl.Next, lambda e: e.steps is None or e.steps == 1),
            ("X[3] p", ltl.Next, lambda e: e.steps == 3),
            # Strong Next forms
            ("X[!] p", ltl.StrongNext, lambda e: e.steps is None or e.steps == 1),
            ("X[3!] p", ltl.StrongNext, lambda e: e.steps == 3),
            # Binary operators
            ("p U q", ltl.Until, lambda e: True),
            ("p W q", ltl.WeakUntil, lambda e: True),
            ("p R q", ltl.Release, lambda e: True),
            ("p M q", ltl.StrongRelease, lambda e: True),
            # Bounded Eventually and Always
            ("F[0,3] p", ltl.Eventually, lambda e: e.strong is False),
            ("F[0,3!] p", ltl.Eventually, lambda e: e.strong is True),
            ("G[0,3] p", ltl.Always, lambda e: e.strong is False),
            ("G[0,3!] p", ltl.Always, lambda e: e.strong is True),
        ],
    )
    def test_parser_forms(
        self, input_str: str, expected_type: type, extra_checks: typing.Callable[[typing.Any], bool]
    ) -> None:
        """Test parser produces correct AST types and attributes."""
        expr = logic_asts.parse_expr(input_str, syntax="ltl")
        assert isinstance(expr, expected_type), f"Expected {expected_type}, got {type(expr)}"
        assert extra_checks(expr), f"Extra checks failed for {input_str}"


class TestRoundTrip:
    """Test round-trip parsing stability."""

    @pytest.mark.parametrize(
        "input_str",
        [
            "X[!] p",
            "X[3!] p",
            "p W q",
            "p M q",
        ],
    )
    def test_round_trip_stability(self, input_str: str) -> None:
        """Test that parse_expr(str(parse_expr(s))) is stable."""
        parsed1 = logic_asts.parse_expr(input_str, syntax="ltl")
        str_repr = str(parsed1)
        parsed2 = logic_asts.parse_expr(str_repr, syntax="ltl")
        assert parsed1 == parsed2, f"Round-trip failed: {input_str} -> {str_repr} -> {parsed2}"

    def test_round_trip_complex_formula(self) -> None:
        """Test round-trip on a complex formula with mixed operators."""
        input_str = "(p W q) & (X[3!] r) & (s M t)"
        parsed1 = logic_asts.parse_expr(input_str, syntax="ltl")
        str_repr = str(parsed1)
        parsed2 = logic_asts.parse_expr(str_repr, syntax="ltl")
        assert parsed1 == parsed2

    def test_round_trip_bounded_eventually_weak(self) -> None:
        """Test round-trip on bounded weak Eventually."""
        input_str = "F[0,3] p"
        parsed1 = logic_asts.parse_expr(input_str, syntax="ltl")
        str_repr = str(parsed1)
        parsed2 = logic_asts.parse_expr(str_repr, syntax="ltl")
        assert parsed1 == parsed2

    def test_round_trip_bounded_always_weak(self) -> None:
        """Test round-trip on bounded weak Always."""
        input_str = "G[0,3] p"
        parsed1 = logic_asts.parse_expr(input_str, syntax="ltl")
        str_repr = str(parsed1)
        parsed2 = logic_asts.parse_expr(str_repr, syntax="ltl")
        assert parsed1 == parsed2


class TestWeakUntilProperties:
    """Test WeakUntil-specific properties."""

    def test_weak_until_semantics(self) -> None:
        """Test WeakUntil semantics: (f U g) | G f."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.WeakUntil(p, q)
        # Create equivalent formula: (p U q) | G p
        equivalent = ltl.Until(p, q) | ltl.Always(p)
        # String representations should differ but semantics match
        assert str(expr) == "(p W q)"
        assert str(equivalent) == "((p U q) | (G p))"

    def test_weak_until_children(self) -> None:
        """Test WeakUntil children iteration."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.WeakUntil(p, q)
        children = list(expr.children())
        assert len(children) == 2
        assert p in children and q in children


class TestStrongReleaseProperties:
    """Test StrongRelease-specific properties."""

    def test_strong_release_semantics(self) -> None:
        """Test StrongRelease as dual of WeakUntil."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.StrongRelease(p, q)
        # Strong Release is ~(~p W ~q)
        assert str(expr) == "(p M q)"

    def test_strong_release_children(self) -> None:
        """Test StrongRelease children iteration."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.StrongRelease(p, q)
        children = list(expr.children())
        assert len(children) == 2
        assert p in children and q in children

    def test_strong_release_unbounded_expand(self) -> None:
        """Test that unbounded StrongRelease expands with StrongRelease."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.StrongRelease(p, q)
        expanded = expr.expand()
        # Unbounded should return StrongRelease with expanded operands
        assert isinstance(expanded, ltl.StrongRelease)

    def test_strong_release_bounded_expand(self) -> None:
        """Test that bounded StrongRelease uses dual form."""
        p = Variable("p")
        q = Variable("q")
        expr = ltl.StrongRelease(p, q, ltl.TimeInterval(0, 5))
        expanded = expr.expand()
        # Bounded should return Not(WeakUntil(...))
        assert isinstance(expanded, Not)


class TestNextOperatorConsistency:
    """Test consistency between Next and StrongNext."""

    def test_next_string_representation(self) -> None:
        """Test that Next prints as weak form (no [!])."""
        p = Variable("p")
        expr = ltl.Next(p)
        assert str(expr) == "(X p)"

    def test_next_vs_strong_next_string(self) -> None:
        """Test difference in string representation."""
        p = Variable("p")
        weak = ltl.Next(p)
        strong = ltl.StrongNext(p)
        assert str(weak) == "(X p)"
        assert str(strong) == "(X[!] p)"
        assert str(weak) != str(strong)

    def test_next_expand_produces_next_not_strong(self) -> None:
        """Test that Next.expand() produces Next, not StrongNext."""
        p = Variable("p")
        expr = ltl.Next(p, steps=2)
        expanded = expr.expand()
        # Should be Next(Next(p))
        assert isinstance(expanded, ltl.Next)
        assert isinstance(expanded.arg, ltl.Next)
        assert isinstance(expanded.arg.arg, Variable)


class TestAlwaysWithStrongFlag:
    """Test Always with strong flag."""

    def test_always_strong_flag_true(self) -> None:
        """Test Always with strong=True."""
        p = Variable("p")
        expr = ltl.Always(p, ltl.TimeInterval(0, 3), strong=True)
        assert expr.strong is True

    def test_always_strong_flag_false(self) -> None:
        """Test Always with strong=False (default)."""
        p = Variable("p")
        expr = ltl.Always(p, ltl.TimeInterval(0, 3), strong=False)
        assert expr.strong is False

    def test_always_strong_expand_produces_strong_next(self) -> None:
        """Test that Always with strong=True expands with StrongNext."""
        p = Variable("p")
        expr = ltl.Always(p, ltl.TimeInterval(0, 3), strong=True)
        expanded = expr.expand()
        all_next_types: set[str] = set()

        def collect_next_types(node: Expr) -> None:
            if isinstance(node, ltl.StrongNext):
                all_next_types.add("StrongNext")
            elif isinstance(node, ltl.Next):
                all_next_types.add("Next")
            for child in node.children():
                collect_next_types(child)

        collect_next_types(expanded)
        assert "StrongNext" in all_next_types
        assert "Next" not in all_next_types


class TestEventuallyWithStrongFlag:
    """Test Eventually with strong flag."""

    def test_eventually_strong_flag_true(self) -> None:
        """Test Eventually with strong=True."""
        p = Variable("p")
        expr = ltl.Eventually(p, ltl.TimeInterval(0, 3), strong=True)
        assert expr.strong is True

    def test_eventually_strong_flag_false(self) -> None:
        """Test Eventually with strong=False (default)."""
        p = Variable("p")
        expr = ltl.Eventually(p, ltl.TimeInterval(0, 3), strong=False)
        assert expr.strong is False


class TestNNFWithBoundedIntervals:
    """Test NNF on formulas with bounded intervals."""

    def test_nnf_of_negated_bounded_eventually_strong_expands(self) -> None:
        """Test ~F[0,3!] p expands and converts to disjunction."""
        p = Variable("p")
        expr = ~ltl.Eventually(p, ltl.TimeInterval(0, 3), strong=True)
        nnf = expr.to_nnf()
        # After expanding, ~F[0,3!] p should produce a disjunction
        # because F[0,3!] expands to (p | X[!] p | X[!] X[!] p | ...)
        from logic_asts.base import Or

        assert isinstance(nnf, Or), f"Expected Or, got {type(nnf)}"

    def test_nnf_of_negated_bounded_always_weak_expands(self) -> None:
        """Test ~G[0,3] p expands and converts to disjunction."""
        p = Variable("p")
        expr = ~ltl.Always(p, ltl.TimeInterval(0, 3), strong=False)
        nnf = expr.to_nnf()
        # After expanding, ~G[0,3] p should produce a disjunction
        # because G[0,3] expands to (p & X p & X X p & ...)
        from logic_asts.base import Or

        assert isinstance(nnf, Or), f"Expected Or, got {type(nnf)}"
