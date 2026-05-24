"""Tests for the sere module."""

import math

import pytest

from logic_asts.base import Variable
from logic_asts.sere import Alt, Concat, Fusion, Inter, Repeat


class TestRepeat:
    def test_unbounded_star_str(self) -> None:
        a = Variable("a")
        assert str(Repeat(a, 0, None)) == "a[*]"
        assert str(Repeat(a, None, None)) == "a[*]"

    def test_unbounded_plus_str(self) -> None:
        a = Variable("a")
        assert str(Repeat(a, 1, None)) == "a[+]"

    def test_point_repeat_str(self) -> None:
        a = Variable("a")
        assert str(Repeat(a, 3, 3)) == "a[*3]"

    def test_range_repeat_str(self) -> None:
        a = Variable("a")
        assert str(Repeat(a, 2, 5)) == "a[*2..5]"

    def test_open_upper_repeat_str(self) -> None:
        a = Variable("a")
        assert str(Repeat(a, 2, None)) == "a[*2..]"

    def test_low_must_be_non_negative(self) -> None:
        a = Variable("a")
        with pytest.raises(ValueError):
            Repeat(a, -1, None)

    def test_low_must_be_le_high(self) -> None:
        a = Variable("a")
        with pytest.raises(ValueError):
            Repeat(a, 3, 2)

    def test_children_yields_arg(self) -> None:
        a = Variable("a")
        r = Repeat(a, 0, None)
        assert list(r.children()) == [a]

    def test_horizon_bounded(self) -> None:
        a = Variable("a")
        assert Repeat(a, 0, 5).horizon() == 5 * a.horizon()

    def test_horizon_unbounded(self) -> None:
        a = Variable("a")
        assert math.isinf(Repeat(a, 0, None).horizon())

    def test_expand_singleton_collapses(self) -> None:
        a = Variable("a")
        assert Repeat(a, 1, 1).expand() == a

    def test_expand_preserves_general_form(self) -> None:
        a = Variable("a")
        r = Repeat(a, 0, None)
        assert r.expand() == r


class TestConcat:
    def test_str_uses_semicolons(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        assert str(Concat((a, b, c))) == "(a ; b ; c)"

    def test_min_length_two(self) -> None:
        a = Variable("a")
        with pytest.raises(ValueError):
            Concat((a,))

    def test_children_yields_args(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert list(Concat((a, b)).children()) == [a, b]

    def test_horizon_sums(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert Concat((a, b)).horizon() == a.horizon() + b.horizon()

    def test_expand_flattens_nested(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        nested = Concat((Concat((a, b)), c))
        assert nested.expand() == Concat((a, b, c))


class TestFusion:
    def test_str_uses_colons(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert str(Fusion((a, b))) == "(a : b)"

    def test_expand_flattens(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        assert Fusion((Fusion((a, b)), c)).expand() == Fusion((a, b, c))


class TestAlt:
    def test_str_uses_pipes(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert str(Alt((a, b))) == "(a | b)"

    def test_horizon_is_max(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert Alt((a, b)).horizon() == max(a.horizon(), b.horizon())

    def test_expand_flattens(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        assert Alt((Alt((a, b)), c)).expand() == Alt((a, b, c))


class TestInter:
    def test_str_uses_double_amp(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert str(Inter((a, b))) == "(a && b)"

    def test_horizon_is_max(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert Inter((a, b)).horizon() == max(a.horizon(), b.horizon())

    def test_expand_flattens(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        assert Inter((Inter((a, b)), c)).expand() == Inter((a, b, c))


def test_sere_expr_iter_yields_postorder() -> None:
    from logic_asts.sere import SEREExpr, sere_expr_iter

    a, b = Variable("a"), Variable("b")
    expr = Concat((Repeat(a, 0, None), b))
    nodes: list[SEREExpr[str]] = list(sere_expr_iter(expr))
    # Leaves come before parents in post-order.
    assert nodes[-1] == expr
    assert a in nodes and b in nodes


def test_sere_expr_iter_rejects_non_sere_node() -> None:
    from logic_asts.ltl import Eventually
    from logic_asts.sere import sere_expr_iter

    bad = Concat((Variable("a"), Eventually(Variable("b"))))
    with pytest.raises(TypeError):
        list(sere_expr_iter(bad))


class TestSereParser:
    def test_parse_atom(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("a", syntax="sere")
        assert expr == Variable("a")

    def test_parse_star(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("a[*]", syntax="sere")
        assert expr == Repeat(Variable("a"), 0, None)

    def test_parse_plus(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("a[+]", syntax="sere")
        assert expr == Repeat(Variable("a"), 1, None)

    def test_parse_range(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("a[*2..5]", syntax="sere")
        assert expr == Repeat(Variable("a"), 2, 5)

    def test_parse_concat(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("a ; b ; c", syntax="sere")
        assert expr == Concat((Variable("a"), Variable("b"), Variable("c")))

    def test_parse_fusion(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("a : b", syntax="sere")
        assert expr == Fusion((Variable("a"), Variable("b")))

    def test_parse_alt_with_concat(self) -> None:
        """Alt has lower precedence than concat."""
        from logic_asts import parse_expr

        expr = parse_expr("a ; b | c", syntax="sere")
        assert expr == Alt((Concat((Variable("a"), Variable("b"))), Variable("c")))

    def test_parse_inter_higher_than_concat(self) -> None:
        from logic_asts import parse_expr

        # ; binds tighter than &&, so this is a && (b ; c)
        expr = parse_expr("a && b ; c", syntax="sere")
        assert expr == Inter((Variable("a"), Concat((Variable("b"), Variable("c")))))

    def test_round_trip_complex(self) -> None:
        from logic_asts import parse_expr

        src = "(a ; b[*0..3]) | (c && d[+])"
        expr = parse_expr(src, syntax="sere")
        # str(expr) must reparse to the same AST.
        assert parse_expr(str(expr), syntax="sere") == expr


class TestIsSereExpr:
    def test_accepts_sere_tree(self) -> None:
        from logic_asts import is_sere_expr

        expr = Concat((Variable("a"), Repeat(Variable("b"), 0, None)))
        assert is_sere_expr(expr)

    def test_rejects_ltl_node(self) -> None:
        from logic_asts import is_sere_expr
        from logic_asts.ltl import Eventually

        expr = Eventually(Variable("a"))
        assert not is_sere_expr(expr)
