"""Tests for the psl module."""

from logic_asts.base import Variable
from logic_asts.ltl import Always, Eventually
from logic_asts.psl import (
    NegStrongClosure,
    StrongClosure,
    SuffixImpliesExist,
    SuffixImpliesUniv,
    WeakClosure,
)
from logic_asts.sere import Concat, Repeat


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

    def test_neg_strong_closure_str(self) -> None:
        a = Variable("a")
        assert str(NegStrongClosure(a)) == "!{a}"

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

    def test_parse_neg_strong_closure(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("!{a}", syntax="psl")
        assert expr == NegStrongClosure(Variable("a"))

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
    from logic_asts.psl import PSLExpr, psl_expr_iter

    a, b, c = Variable("a"), Variable("b"), Variable("c")
    sere_part = Concat((a, b))
    formula = Eventually(c)
    node = SuffixImpliesUniv(sere_part, formula)
    nodes: list[PSLExpr[str]] = list(psl_expr_iter(node))
    assert nodes[-1] == node
    assert sere_part in nodes
    assert formula in nodes


def test_is_psl_expr_accepts_mixed_tree() -> None:
    from logic_asts import is_psl_expr

    expr = SuffixImpliesUniv(Concat((Variable("a"), Variable("b"))), Eventually(Variable("c")))
    assert is_psl_expr(expr)


def test_is_psl_expr_rejects_strel_node() -> None:
    from logic_asts import is_psl_expr
    from logic_asts.strel import DistanceInterval, Somewhere

    expr = Somewhere(Variable("a"), DistanceInterval(0, 5), None)
    assert not is_psl_expr(expr)
