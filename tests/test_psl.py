"""Tests for the psl module."""

import pytest

from logic_asts.base import Literal, Not, Variable
from logic_asts.ltl import Always, Eventually, Release, Until
from logic_asts.psl import (
    PSLFormula,
    StrongClosure,
    SuffixImpliesExist,
    SuffixImpliesUniv,
    WeakClosure,
)
from logic_asts.sere import Concat, Repeat
from logic_asts.utils import to_nnf


class TestPslNodes:
    def test_suffix_implies_univ_str(self) -> None:
        a, f = Variable("a"), Variable("f")
        node = SuffixImpliesUniv(a, f)
        assert str(node) == "{a}[]-> f"

    def test_suffix_implies_exist_str(self) -> None:
        a, f = Variable("a"), Variable("f")
        node = SuffixImpliesExist(a, f)
        assert str(node) == "{a}<>-> f"

    def test_weak_closure_str(self) -> None:
        a = Variable("a")
        assert str(WeakClosure(a)) == "{a}"

    def test_strong_closure_str(self) -> None:
        a = Variable("a")
        assert str(StrongClosure(a)) == "{a}!"

    def test_not_weak_closure_str(self) -> None:
        a = Variable("a")
        assert str(Not(WeakClosure(a))) == "!{a}"

    def test_children_for_binding(self) -> None:
        a, f = Variable("a"), Variable("f")
        node = SuffixImpliesUniv(a, f)
        assert list(node.children()) == [a, f]

    def test_children_for_closure(self) -> None:
        a = Variable("a")
        assert list(WeakClosure(a).children()) == [a]

    def test_horizon_sums_for_binding(self) -> None:
        a, f = Variable("a"), Variable("f")
        node = SuffixImpliesUniv(a, f)
        assert node.horizon() == a.horizon() + f.horizon()

    def test_horizon_closure_is_sere_horizon(self) -> None:
        a = Variable("a")
        r = Repeat(a, 0, 3)
        assert WeakClosure(r).horizon() == r.horizon()


class TestPslParser:
    def test_parse_suffix_implies_univ(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{a}[]-> b", syntax="psl")
        assert expr == SuffixImpliesUniv(Variable("a"), Variable("b"))

    def test_parse_suffix_implies_exist(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{a}<>-> b", syntax="psl")
        assert expr == SuffixImpliesExist(Variable("a"), Variable("b"))

    def test_parse_weak_closure(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{a ; b}", syntax="psl")
        assert expr == WeakClosure(Concat((Variable("a"), Variable("b"))))

    def test_parse_strong_closure(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{a}!", syntax="psl")
        assert expr == StrongClosure(Variable("a"))

    def test_parse_neg_weak_closure(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("!{a}", syntax="psl")
        assert expr == Not(WeakClosure(Variable("a")))

    def test_sugar_universal_implies_then(self) -> None:
        """{r}[]=> f rewrites to SuffixImpliesUniv(Concat((r, Literal(True))), f)."""
        from logic_asts import parse_expr
        from logic_asts.base import Literal

        expr = parse_expr("{a}[]=> b", syntax="psl")
        assert expr == SuffixImpliesUniv(Concat((Variable("a"), Literal(True))), Variable("b"))

    def test_sugar_existential_implies_then(self) -> None:
        from logic_asts import parse_expr
        from logic_asts.base import Literal

        expr = parse_expr("{a}<>=> b", syntax="psl")
        assert expr == SuffixImpliesExist(Concat((Variable("a"), Literal(True))), Variable("b"))

    def test_ltl_inside_psl_round_trip(self) -> None:
        from logic_asts import parse_expr

        src = "G {a ; b}[]-> F c"
        expr = parse_expr(src, syntax="psl")
        # Top-level should be an LTL operator (Always) wrapping the binding.
        assert isinstance(expr, Always)
        assert parse_expr(str(expr), syntax="psl") == expr


def test_psl_expr_iter_walks_mixed_tree() -> None:
    from logic_asts.psl import psl_expr_iter
    from logic_asts.sere import SEREExpr

    a, b, c = Variable("a"), Variable("b"), Variable("c")
    sere_part: SEREExpr[str] = Concat((a, b))
    formula = Eventually(c)
    node: SuffixImpliesUniv[str] = SuffixImpliesUniv(sere_part, formula)
    nodes = list(psl_expr_iter(node))
    assert nodes[-1] == node
    assert sere_part in nodes
    assert formula in nodes


def test_is_psl_expr_accepts_mixed_tree() -> None:
    from logic_asts import is_psl_expr

    expr: SuffixImpliesUniv[str] = SuffixImpliesUniv(Concat((Variable("a"), Variable("b"))), Eventually(Variable("c")))
    assert is_psl_expr(expr)


def test_is_psl_expr_rejects_strel_node() -> None:
    from logic_asts import is_psl_expr
    from logic_asts.strel import DistanceInterval, Somewhere

    expr = Somewhere(Variable("a"), DistanceInterval(0, 5), None)
    assert not is_psl_expr(expr)


def test_psl_strong_closure_over_nlm_inter() -> None:
    """PSL reuses the SERE grammar; ``{a & b}!`` should parse."""
    from logic_asts import parse_expr

    expr = parse_expr("{a & b}!", syntax="psl")
    assert parse_expr(str(expr), syntax="psl") == expr


class TestPslComplementAndFirstMatch:
    """PSL reuses the SERE grammar wholesale, so ~ and first_match should
    parse inside ``{...}``-wrapped PSL forms."""

    def test_psl_strong_closure_over_complement(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{~a}!", syntax="psl")
        assert parse_expr(str(expr), syntax="psl") == expr

    def test_psl_strong_closure_over_first_match(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{first_match(a;b)}!", syntax="psl")
        assert parse_expr(str(expr), syntax="psl") == expr

    def test_psl_suffix_implies_with_complement(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{~a;b}[]-> c", syntax="psl")
        assert parse_expr(str(expr), syntax="psl") == expr


class TestPslExtendedRepetition:
    """PSL reuses the SERE grammar; the new repeat suffixes must parse inside {...} forms."""

    def test_psl_strong_closure_over_fusion_repeat(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{a[:*]}!", syntax="psl")
        assert parse_expr(str(expr), syntax="psl") == expr

    def test_psl_strong_closure_over_goto_repeat(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{a[->2]}!", syntax="psl")
        assert parse_expr(str(expr), syntax="psl") == expr

    def test_psl_suffix_implies_with_equal_repeat(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("{a[=1..3]}[]-> c", syntax="psl")
        assert parse_expr(str(expr), syntax="psl") == expr


class TestPslSuffixImpliesNnf:
    def test_univ_no_negate_recurses_into_formula(self) -> None:
        a, b, r = Variable("a"), Variable("b"), Variable("r")
        phi: PSLFormula[str] = Until(a, b)
        expr = SuffixImpliesUniv(r, phi)
        assert to_nnf(expr) == SuffixImpliesUniv(r, to_nnf(phi))

    def test_exist_no_negate_recurses_into_formula(self) -> None:
        a, b, r = Variable("a"), Variable("b"), Variable("r")
        phi: PSLFormula[str] = Until(a, b)
        expr = SuffixImpliesExist(r, phi)
        assert to_nnf(expr) == SuffixImpliesExist(r, to_nnf(phi))

    def test_negate_univ_becomes_exist(self) -> None:
        a, b, r = Variable("a"), Variable("b"), Variable("r")
        phi: PSLFormula[str] = Until(a, b)
        expr = SuffixImpliesUniv(r, phi)
        expected = SuffixImpliesExist(r, to_nnf(phi, negate=True))
        assert to_nnf(expr, negate=True) == expected

    def test_negate_exist_becomes_univ(self) -> None:
        a, b, r = Variable("a"), Variable("b"), Variable("r")
        phi: PSLFormula[str] = Until(a, b)
        expr = SuffixImpliesExist(r, phi)
        expected = SuffixImpliesUniv(r, to_nnf(phi, negate=True))
        assert to_nnf(expr, negate=True) == expected

    def test_double_negation_univ_round_trip(self) -> None:
        a, b, r = Variable("a"), Variable("b"), Variable("r")
        phi = Until(a, b)
        expr = SuffixImpliesUniv(r, phi)
        assert to_nnf(~~expr) == to_nnf(expr)

    def test_double_negation_exist_round_trip(self) -> None:
        a, b, r = Variable("a"), Variable("b"), Variable("r")
        phi = Until(a, b)
        expr = SuffixImpliesExist(r, phi)
        assert to_nnf(~~expr) == to_nnf(expr)

    def test_sere_argument_not_recursed(self) -> None:
        a, b, phi = Variable("a"), Variable("b"), Variable("phi")
        r = Concat((a, b))
        expr = SuffixImpliesUniv(r, phi)
        assert to_nnf(expr) == SuffixImpliesUniv(r, phi)

    def test_mixed_negation_pushes_into_until(self) -> None:
        a, b, r = Variable("a"), Variable("b"), Variable("r")
        na: PSLFormula[str] = Not(a)
        nb: PSLFormula[str] = Not(b)
        expr = SuffixImpliesUniv(r, Until(a, b))
        result = to_nnf(expr, negate=True)
        assert result == SuffixImpliesExist(r, Release(na, nb))


class TestPslClosureNnf:
    def test_weak_closure_no_negate_unchanged(self) -> None:
        r = Variable("r")
        expr = WeakClosure(r)
        assert to_nnf(expr) == expr

    def test_weak_closure_negate_wraps_in_not(self) -> None:
        r = Variable("r")
        expr = WeakClosure(r)
        assert to_nnf(expr, negate=True) == Not(WeakClosure(r))

    def test_not_weak_closure_double_negation(self) -> None:
        r = Variable("r")
        expr = Not(WeakClosure(r))
        assert to_nnf(~expr) == to_nnf(WeakClosure(r))

    def test_strong_closure_nnf_desugars(self) -> None:
        r = Variable("r")
        # to_nnf calls .expand() first, so StrongClosure desugars.
        assert to_nnf(StrongClosure(r)) == SuffixImpliesExist(r, Literal(True))

    def test_strong_closure_negate_becomes_universal_zero(self) -> None:
        r = Variable("r")
        # !{r}!  ==  !({r}<>-> 1)  ==  {r}[]-> 0
        assert to_nnf(StrongClosure(r), negate=True) == SuffixImpliesUniv(r, Literal(False))

    def test_strong_closure_expand_desugars_to_suffix_implies_exist(self) -> None:
        r = Variable("r")
        expanded = StrongClosure(r).expand()
        assert expanded == SuffixImpliesExist(r, Literal(True))

    def test_strong_closure_expand_then_nnf_negation(self) -> None:
        r = Variable("r")
        expr = StrongClosure(r).expand()
        # !({r}<>-> 1)  ==  {r}[]-> 0
        assert to_nnf(expr, negate=True) == SuffixImpliesUniv(r, Literal(False))

    def test_sere_argument_to_closure_not_recursed(self) -> None:
        a, b = Variable("a"), Variable("b")
        r = Concat((a, b))
        expr: WeakClosure[str] = WeakClosure(r)
        assert to_nnf(expr, negate=True) == Not(WeakClosure(r))


class TestPslValidators:
    def test_weak_closure_rejects_ltl_formula(self) -> None:

        ltl_formula = Always(Variable("p"))
        with pytest.raises(TypeError):
            WeakClosure(ltl_formula)  # type: ignore[arg-type]

    def test_strong_closure_rejects_ltl_formula(self) -> None:

        ltl_formula = Eventually(Variable("p"))
        with pytest.raises(TypeError):
            StrongClosure(ltl_formula)  # type: ignore[arg-type]

    def test_suffix_implies_univ_rejects_ltl_in_sere_slot(self) -> None:

        ltl_formula = Always(Variable("p"))
        with pytest.raises(TypeError):
            SuffixImpliesUniv(ltl_formula, Variable("q"))  # type: ignore[arg-type]

    def test_suffix_implies_univ_rejects_bare_sere_in_formula_slot(self) -> None:

        sere = Concat((Variable("a"), Variable("b")))
        with pytest.raises(TypeError):
            SuffixImpliesUniv(Variable("r"), sere)  # type: ignore[arg-type]

    def test_suffix_implies_exist_rejects_ltl_in_sere_slot(self) -> None:

        ltl_formula = Until(Variable("p"), Variable("q"))
        with pytest.raises(TypeError):
            SuffixImpliesExist(ltl_formula, Variable("r"))  # type: ignore[arg-type]

    def test_well_typed_constructions_succeed(self) -> None:
        # Should not raise.
        WeakClosure(Variable("a"))
        WeakClosure(Concat((Variable("a"), Variable("b"))))
        StrongClosure(Variable("a"))
        SuffixImpliesUniv(Variable("a"), Variable("b"))
        SuffixImpliesUniv(
            Concat((Variable("a"), Variable("b"))),
            Always(Variable("p")),
        )
        # PSL nesting: formula slot may itself be a closure / suffix implication.
        SuffixImpliesUniv(
            Variable("r"),
            WeakClosure(Variable("a")),
        )
