r"""Comprehensive tests for propositional logic (base module).

Tests cover:
- Atomic expressions (Literal, Variable)
- Binary operators (And, Or, Implies, Equiv, Xor)
- Unary operator (Not)
- Operator overloading and precedence
- Formula expansion
- Negation Normal Form (NNF) conversion
- Tree traversal and operations
- Parsing and string representation
- Simple evaluation with truth assignments
"""

import operator
from functools import reduce

import pytest

import logic_asts
from logic_asts.base import (
    And,
    Equiv,
    Expr,
    Implies,
    Literal,
    Not,
    Or,
    Variable,
    Xor,
    simple_eval,
)


class TestAtomicExpressions:
    """Tests for Literal and Variable atomic expressions."""

    @pytest.mark.parametrize(
        ["expr", "expected"],
        [
            ("0", Literal(False)),
            ("1", Literal(True)),
            ("False", Literal(False)),
            ("True", Literal(True)),
            ("FALSE", Literal(False)),
            ("TRUE", Literal(True)),
        ],
    )
    def test_atoms(self, expr: str, expected: Expr) -> None:
        """Test parsing of literal constants."""
        parsed = logic_asts.parse_expr(expr, syntax="base")
        assert parsed == expected

    def test_literal_true(self) -> None:
        """Test Literal(True) creation and properties."""
        lit = Literal(True)
        assert lit.value is True
        assert str(lit) == "t"
        assert lit.horizon() == 0

    def test_literal_false(self) -> None:
        """Test Literal(False) creation and properties."""
        lit = Literal(False)
        assert lit.value is False
        assert str(lit) == "f"
        assert lit.horizon() == 0

    def test_variable_string(self) -> None:
        """Test Variable with string name."""
        var = Variable("p")
        assert var.name == "p"
        assert str(var) == "p"
        assert var.horizon() == 0

    def test_variable_tuple(self) -> None:
        """Test Variable with tuple name."""
        var = Variable(("agent", 0))
        assert var.name == ("agent", 0)
        assert str(var) == "('agent', 0)"

    def test_variable_int(self) -> None:
        """Test Variable with integer name."""
        var = Variable(42)
        assert var.name == 42
        assert str(var) == "42"


class TestBinaryOperators:
    """Tests for binary operators: And, Or, Implies, Equiv, Xor."""

    def test_and_two_operands(self) -> None:
        """Test And with two operands."""
        p = Variable("p")
        q = Variable("q")
        and_expr = And((p, q))
        assert len(list(and_expr.children())) == 2
        assert str(and_expr) == "(p & q)"

    def test_and_multiple_operands(self) -> None:
        """Test And with more than two operands."""
        p, q, r = Variable("p"), Variable("q"), Variable("r")
        and_expr = And((p, q, r))
        assert len(list(and_expr.children())) == 3

    def test_and_operator_overload(self) -> None:
        """Test And creation using & operator."""
        p = Variable("p")
        q = Variable("q")
        expr1 = And((p, q))
        expr2 = p & q
        assert expr1 == expr2

    def test_and_flattening(self) -> None:
        """Test that And flattens nested And expressions."""
        p, q, r = Variable("p"), Variable("q"), Variable("r")
        expr = (p & q) & r
        assert isinstance(expr, And)
        assert len(list(expr.children())) == 3

    def test_or_two_operands(self) -> None:
        """Test Or with two operands."""
        p = Variable("p")
        q = Variable("q")
        or_expr = Or((p, q))
        assert len(list(or_expr.children())) == 2
        assert str(or_expr) == "(p | q)"

    def test_or_multiple_operands(self) -> None:
        """Test Or with more than two operands."""
        p, q, r = Variable("p"), Variable("q"), Variable("r")
        or_expr = Or((p, q, r))
        assert len(list(or_expr.children())) == 3

    def test_or_operator_overload(self) -> None:
        """Test Or creation using | operator."""
        p = Variable("p")
        q = Variable("q")
        expr1 = Or((p, q))
        expr2 = p | q
        assert expr1 == expr2

    def test_or_flattening(self) -> None:
        """Test that Or flattens nested Or expressions."""
        p, q, r = Variable("p"), Variable("q"), Variable("r")
        expr = (p | q) | r
        assert isinstance(expr, Or)
        assert len(list(expr.children())) == 3

    def test_implies(self) -> None:
        """Test Implies operator."""
        p = Variable("p")
        q = Variable("q")
        impl = Implies(p, q)
        assert str(impl) == "p -> q"
        assert impl.lhs == p
        assert impl.rhs == q

    def test_equiv(self) -> None:
        """Test Equiv (biconditional) operator."""
        p = Variable("p")
        q = Variable("q")
        equiv = Equiv(p, q)
        assert str(equiv) == "p <-> q"
        assert equiv.lhs == p
        assert equiv.rhs == q

    def test_xor(self) -> None:
        """Test Xor (exclusive or) operator."""
        p = Variable("p")
        q = Variable("q")
        xor = Xor(p, q)
        assert str(xor) == "p ^ q"
        assert xor.lhs == p
        assert xor.rhs == q


class TestUnaryOperator:
    """Tests for Not (negation) operator."""

    def test_not_variable(self) -> None:
        """Test Not of a variable."""
        p = Variable("p")
        not_p = Not(p)
        assert str(not_p) == "!p"
        assert not_p.arg == p

    def test_not_operator_overload(self) -> None:
        """Test Not creation using ~ operator."""
        p = Variable("p")
        expr1 = Not(p)
        expr2 = ~p
        assert expr1 == expr2

    def test_double_negation_elimination(self) -> None:
        """Test that ~~p = p."""
        p = Variable("p")
        double_neg = ~(~p)
        assert double_neg == p

    def test_not_literal_true(self) -> None:
        """Test Not of Literal(True)."""
        not_true = ~Literal(True)
        assert not_true == Literal(False)

    def test_not_literal_false(self) -> None:
        """Test Not of Literal(False)."""
        not_false = ~Literal(False)
        assert not_false == Literal(True)

    def test_not_and(self) -> None:
        """Test Not of And (De Morgan's law)."""
        p = Variable("p")
        q = Variable("q")
        # ~(p & q) should contain negations over p and q
        expr = ~(p & q)
        assert isinstance(expr, Not)


class TestOperatorOverloading:
    """Tests for operator overloading precedence and interactions."""

    def test_precedence_and_over_or(self) -> None:
        """Test that & has higher precedence than |."""
        p = Variable("p")
        q = Variable("q")
        r = Variable("r")
        # p | q & r should parse as p | (q & r)
        expr = p | (q & r)
        assert isinstance(expr, Or)
        assert len(list(expr.children())) == 2

    def test_negation_binds_tighter(self) -> None:
        """Test that ~ binds tighter than & and |."""
        p = Variable("p")
        q = Variable("q")
        # ~p & q should be (~p) & q, not ~(p & q)
        expr = (~p) & q
        assert isinstance(expr, And)

    def test_mixed_operators(self) -> None:
        """Test expressions with mixed operators."""
        p = Variable("p")
        q = Variable("q")
        r = Variable("r")
        expr = (p & q) | ~r
        assert isinstance(expr, Or)


class TestExpansion:
    """Tests for formula expansion of derived operators."""

    def test_implies_expansion(self) -> None:
        r"""Test that p -> q expands to ~p | q."""
        p = Variable("p")
        q = Variable("q")
        impl = Implies(p, q)
        expanded = impl.expand()
        # Should be equivalent to ~p | q
        expected = (~p) | q
        assert expanded == expected

    def test_equiv_expansion(self) -> None:
        r"""Test that p <-> q expands to (p | ~q) & (~p | q)."""
        p = Variable("p")
        q = Variable("q")
        equiv = Equiv(p, q)
        expanded = equiv.expand()
        # Should be (p | ~q) & (~p | q)
        expected = (p | ~q) & ((~p) | q)
        assert expanded == expected

    def test_xor_expansion(self) -> None:
        r"""Test that p ^ q expands to (p & ~q) | (~p & q)."""
        p = Variable("p")
        q = Variable("q")
        xor = Xor(p, q)
        expanded = xor.expand()
        # Should be (p & ~q) | (~p & q)
        expected = (p & (~q)) | ((~p) & q)
        assert expanded == expected

    def test_expansion_preserves_semantics(self) -> None:
        """Test that expansion preserves truth values."""
        p = Variable("p")
        q = Variable("q")
        impl = Implies(p, q)
        expanded = impl.expand()
        assert isinstance(expanded, Or)

        # Test with all truth assignments
        for p_val in [set(), {"p"}]:
            for q_val in [set(), {"q"}]:
                assignment = p_val | q_val
                original = simple_eval(impl, assignment)
                exp_result = simple_eval(expanded, assignment)
                assert original == exp_result


class TestNNFConversion:
    """Tests for conversion to Negation Normal Form (NNF)."""

    def test_nnf_literal(self) -> None:
        """Test NNF of literals."""
        assert Literal(True).to_nnf() == Literal(True)
        assert Literal(False).to_nnf() == Literal(False)

    def test_nnf_variable(self) -> None:
        """Test NNF of variables."""
        p = Variable("p")
        assert p.to_nnf() == p

    def test_nnf_double_negation(self) -> None:
        """Test that ~~p = p in NNF."""
        p = Variable("p")
        expr = Not(Not(p))
        nnf = expr.to_nnf()
        assert nnf == p

    def test_nnf_de_morgan_and(self) -> None:
        r"""Test De Morgan's law: ~(p & q) = ~p | ~q."""
        p = Variable("p")
        q = Variable("q")
        expr = ~(p & q)
        nnf = expr.to_nnf()
        # Should be Or of negated parts
        assert isinstance(nnf, Or)

    def test_nnf_de_morgan_or(self) -> None:
        r"""Test De Morgan's law: ~(p | q) = ~p & ~q."""
        p = Variable("p")
        q = Variable("q")
        expr = ~(p | q)
        nnf = expr.to_nnf()
        # Should be And of negated parts
        assert isinstance(nnf, And)

    def test_nnf_implies_expansion(self) -> None:
        r"""Test NNF of implication: p -> q becomes ~p | q."""
        p = Variable("p")
        q = Variable("q")
        impl = Implies(p, q)
        nnf = impl.to_nnf()
        # Should be ~p | q
        expected = (~p) | q
        assert nnf == expected

    def test_nnf_complex_formula(self) -> None:
        """Test NNF of complex nested formula."""
        p = Variable("p")
        q = Variable("q")
        r = Variable("r")
        # Complex: ~((p & q) | r)
        expr = ~((p & q) | r)
        nnf = expr.to_nnf()
        # No negations should be above non-atomic nodes
        for node in nnf.iter_subtree():
            if isinstance(node, Not):
                assert isinstance(node.arg, (Variable, Literal))


class TestTreeOperations:
    """Tests for tree traversal and introspection."""

    def test_iter_subtree_single_node(self) -> None:
        """Test iter_subtree on a single variable."""
        p = Variable("p")
        nodes = list(p.iter_subtree())
        assert len(nodes) == 1
        assert nodes[0] == p

    def test_iter_subtree_post_order(self) -> None:
        """Test that iter_subtree follows post-order traversal."""
        p = Variable("p")
        q = Variable("q")
        expr = p & q
        nodes = list(expr.iter_subtree())
        # Post-order: children before parents
        # Verify that expr is last (parent comes after children)
        assert nodes[-1] == expr
        # Verify p and q come before expr
        assert p in nodes and q in nodes
        assert nodes.index(p) < nodes.index(expr)
        assert nodes.index(q) < nodes.index(expr)

    def test_iter_subtree_no_duplicates(self) -> None:
        """Test that iter_subtree doesn't return the same object twice."""
        p = Variable("p")
        q = Variable("q")
        # Create expression where p and q appear multiple times
        expr = (p & q) | (p & q)
        nodes = list(expr.iter_subtree())
        # Check that no node appears twice in the visited set
        node_ids = [id(n) for n in nodes]
        assert len(node_ids) == len(set(node_ids)), "iter_subtree returned duplicate objects"

    def test_children_and_operator(self) -> None:
        """Test children() for And operator."""
        p = Variable("p")
        q = Variable("q")
        r = Variable("r")
        expr = And((p, q, r))
        children = list(expr.children())
        assert len(children) == 3
        assert p in children and q in children and r in children

    def test_children_or_operator(self) -> None:
        """Test children() for Or operator."""
        p = Variable("p")
        q = Variable("q")
        expr = Or((p, q))
        children = list(expr.children())
        assert len(children) == 2

    def test_children_not_operator(self) -> None:
        """Test children() for Not operator."""
        p = Variable("p")
        expr = Not(p)
        children = list(expr.children())
        assert len(children) == 1
        assert children[0] == p

    def test_children_literal(self) -> None:
        """Test that literals have no children."""
        lit = Literal(True)
        children = list(lit.children())
        assert len(children) == 0

    def test_horizon_atomic(self) -> None:
        """Test horizon of atomic expressions."""
        p = Variable("p")
        lit = Literal(True)
        assert p.horizon() == 0
        assert lit.horizon() == 0

    def test_horizon_and(self) -> None:
        """Test horizon of And expression."""
        p = Variable("p")
        q = Variable("q")
        expr = p & q
        assert expr.horizon() == 0

    def test_horizon_or(self) -> None:
        """Test horizon of Or expression."""
        p = Variable("p")
        q = Variable("q")
        expr = p | q
        assert expr.horizon() == 0


class TestParsing:
    """Tests for parsing propositional logic expressions."""

    @pytest.mark.parametrize(
        "expr_str",
        [
            "p",
            "~p",
            "p & q",
            "p | q",
            "p -> q",
            "p <-> q",
            "p ^ q",
            "(p & q) | r",
            "~(p & q)",
            "((p | q) & r) | s",
        ],
    )
    def test_parse_valid_operators(self, expr_str: str) -> None:
        """Test parsing of various operators."""
        parsed = logic_asts.parse_expr(expr_str, syntax="base")
        assert parsed is not None

    def test_parse_parentheses(self) -> None:
        """Test parsing with parentheses."""
        expr1 = logic_asts.parse_expr("(p & q) | r", syntax="base")
        expr2 = logic_asts.parse_expr("p & (q | r)", syntax="base")
        # These should be different
        assert str(expr1) != str(expr2)

    def test_base_logic_complex(self) -> None:
        """Test parsing of a complex propositional formula."""
        expr = "(x1 <-> x2) | x3"
        expected = Equiv(Variable("x1"), Variable("x2")) | Variable("x3")
        parsed = logic_asts.parse_expr(expr, syntax="base")
        assert parsed == expected
        assert parsed.horizon() == 0

    @pytest.mark.parametrize(
        "n",
        [3, 5, 10, 20, 30, 40, 80, 100],
    )
    def test_parse_large_expr(self, n: int) -> None:
        """Test parsing of large expressions."""
        expr = " & ".join((f"(x{i} <-> y{i})" for i in range(n)))
        expected: Expr = reduce(operator.__and__, (Equiv(Variable(f"x{i}"), Variable(f"y{i}")) for i in range(n)))
        parsed = logic_asts.parse_expr(expr, syntax="base")
        assert parsed == expected
        assert parsed.horizon() == expected.horizon() == 0


class TestSimpleEvaluation:
    """Tests for simple_eval function."""

    def test_eval_true_literal(self) -> None:
        """Test evaluation of Literal(True)."""
        lit = Literal(True)
        assert simple_eval(lit, set()) is True

    def test_eval_false_literal(self) -> None:
        """Test evaluation of Literal(False)."""
        lit = Literal(False)
        assert simple_eval(lit, set()) is False

    def test_eval_variable_true(self) -> None:
        """Test evaluation of variable that is true."""
        p = Variable("p")
        assert simple_eval(p, {"p"}) is True

    def test_eval_variable_false(self) -> None:
        """Test evaluation of variable that is false."""
        p = Variable("p")
        assert simple_eval(p, set()) is False

    def test_eval_and_both_true(self) -> None:
        """Test And evaluation when both operands are true."""
        p = Variable("p")
        q = Variable("q")
        expr = p & q
        assert isinstance(expr, And)
        assert simple_eval(expr, {"p", "q"}) is True

    def test_eval_and_one_false(self) -> None:
        """Test And evaluation when one operand is false."""
        p = Variable("p")
        q = Variable("q")
        expr = p & q
        assert isinstance(expr, And)
        assert simple_eval(expr, {"p"}) is False

    def test_eval_or_both_true(self) -> None:
        """Test Or evaluation when both operands are true."""
        p = Variable("p")
        q = Variable("q")
        expr = p | q
        assert isinstance(expr, Or)
        assert simple_eval(expr, {"p", "q"}) is True

    def test_eval_or_one_true(self) -> None:
        """Test Or evaluation when one operand is true."""
        p = Variable("p")
        q = Variable("q")
        expr = p | q
        assert isinstance(expr, Or)
        assert simple_eval(expr, {"p"}) is True

    def test_eval_or_both_false(self) -> None:
        """Test Or evaluation when both operands are false."""
        p = Variable("p")
        q = Variable("q")
        expr = p | q
        assert isinstance(expr, Or)
        assert simple_eval(expr, set()) is False

    def test_eval_not(self) -> None:
        """Test Not evaluation."""
        p = Variable("p")
        assert simple_eval(~p, {"p"}) is False  # type: ignore[arg-type]
        assert simple_eval(~p, set()) is True  # type: ignore[arg-type]

    def test_eval_implies_true(self) -> None:
        """Test Implies evaluation (true implication)."""
        p = Variable("p")
        q = Variable("q")
        impl = Implies(p, q)
        # True -> True = True
        assert simple_eval(impl, {"p", "q"}) is True
        # False -> True = True
        assert simple_eval(impl, {"q"}) is True
        # False -> False = True
        assert simple_eval(impl, set()) is True

    def test_eval_implies_false(self) -> None:
        """Test Implies evaluation (false implication)."""
        p = Variable("p")
        q = Variable("q")
        impl = Implies(p, q)
        # True -> False = False
        assert simple_eval(impl, {"p"}) is False

    def test_eval_equiv_true(self) -> None:
        """Test Equiv evaluation (true biconditional)."""
        p = Variable("p")
        q = Variable("q")
        equiv = Equiv(p, q)
        # True <-> True = True
        assert simple_eval(equiv, {"p", "q"}) is True
        # False <-> False = True
        assert simple_eval(equiv, set()) is True

    def test_eval_equiv_false(self) -> None:
        """Test Equiv evaluation (false biconditional)."""
        p = Variable("p")
        q = Variable("q")
        equiv = Equiv(p, q)
        # True <-> False = False
        assert simple_eval(equiv, {"p"}) is False
        # False <-> True = False
        assert simple_eval(equiv, {"q"}) is False

    def test_eval_xor_true(self) -> None:
        """Test Xor evaluation (true exclusive or)."""
        p = Variable("p")
        q = Variable("q")
        xor = Xor(p, q)
        # True ^ False = True
        assert simple_eval(xor, {"p"}) is True
        # False ^ True = True
        assert simple_eval(xor, {"q"}) is True

    def test_eval_xor_false(self) -> None:
        """Test Xor evaluation (false exclusive or)."""
        p = Variable("p")
        q = Variable("q")
        xor = Xor(p, q)
        # True ^ True = False
        assert simple_eval(xor, {"p", "q"}) is False
        # False ^ False = False
        assert simple_eval(xor, set()) is False

    def test_eval_complex_formula(self) -> None:
        """Test evaluation of complex formula."""
        p = Variable("p")
        q = Variable("q")
        r = Variable("r")
        # (p & q) | ~r
        expr = (p & q) | ~r
        assert isinstance(expr, Or)
        # p=T, q=T, r=T -> (T & T) | ~T = T | F = T
        assert simple_eval(expr, {"p", "q", "r"}) is True
        # p=T, q=F, r=F -> (T & F) | ~F = F | T = T
        assert simple_eval(expr, {"p"}) is True
        # p=F, q=F, r=F -> (F & F) | ~F = F | T = T
        assert simple_eval(expr, set()) is True
        # p=F, q=F, r=T -> (F & F) | ~T = F | F = F
        assert simple_eval(expr, {"r"}) is False

    def test_eval_tautology(self) -> None:
        """Test evaluation of tautology (always true)."""
        p = Variable("p")
        # p | ~p is always true
        tautology = p | ~p
        assert isinstance(tautology, Or)
        assert simple_eval(tautology, set()) is True
        assert simple_eval(tautology, {"p"}) is True

    def test_eval_contradiction(self) -> None:
        """Test evaluation of contradiction (always false)."""
        p = Variable("p")
        # p & ~p is always false
        contradiction = p & ~p
        assert isinstance(contradiction, And)
        assert simple_eval(contradiction, set()) is False
        assert simple_eval(contradiction, {"p"}) is False

    def test_eval_with_tuple_variables(self) -> None:
        """Test evaluation with tuple variable names."""
        p = Variable(("agent", 0))
        q = Variable(("agent", 1))
        expr = p & q
        assert isinstance(expr, And)
        assignment = {("agent", 0), ("agent", 1)}
        assert simple_eval(expr, assignment) is True
