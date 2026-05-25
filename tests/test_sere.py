"""Tests for the sere module."""

import math

import pytest

from logic_asts.base import Variable
from logic_asts.sere import (
    Alt,
    Complement,
    Concat,
    EqualRepeat,
    FirstMatch,
    Fusion,
    FusionRepeat,
    GotoRepeat,
    Inter,
    NLMInter,
    Repeat,
)


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


class TestNLMInter:
    def test_str_binary(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert str(NLMInter((a, b))) == "(a & b)"

    def test_str_ternary(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        assert str(NLMInter((a, b, c))) == "(a & b & c)"

    def test_horizon(self) -> None:
        a = Variable("a")
        b = Repeat(Variable("b"), low=0, high=3)
        assert NLMInter((a, b)).horizon() == max(a.horizon(), b.horizon())

    def test_expand_flattens(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        assert NLMInter((NLMInter((a, b)), c)).expand() == NLMInter((a, b, c))

    def test_children_order(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        assert tuple(NLMInter((a, b, c)).children()) == (a, b, c)

    def test_min_len_two(self) -> None:
        with pytest.raises(ValueError):
            NLMInter((Variable("a"),))


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


class TestComplement:
    def test_str_atom(self) -> None:
        a = Variable("a")
        assert str(Complement(a)) == "~a"

    def test_str_double(self) -> None:
        a = Variable("a")
        assert str(Complement(Complement(a))) == "~~a"

    def test_str_braced_compound(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert str(Complement(Concat((a, b)))) == "~{a ; b}"

    def test_str_with_repeat_around(self) -> None:
        a = Variable("a")
        # ~ binds tighter than [*]: Repeat(Complement(a), ...) -> "~a[*]"
        assert str(Repeat(Complement(a), 0, None)) == "~a[*]"

    def test_str_complement_around_repeat(self) -> None:
        a = Variable("a")
        # Complement(Repeat(...)) must brace-wrap so the suffix doesn't
        # rebind to the outer ~ on re-parse: ~{a[*]} not ~a[*].
        assert str(Complement(Repeat(a, 0, None))) == "~{a[*]}"

    def test_horizon_is_inf(self) -> None:
        a = Variable("a")
        assert math.isinf(Complement(a).horizon())

    def test_children_yields_arg(self) -> None:
        a = Variable("a")
        c = Complement(a)
        assert list(c.children()) == [a]

    def test_expand_recurses(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        # Inner NLMInter flattens via NLMInter.expand(), wrapped by Complement.
        inner = NLMInter((NLMInter((a, b)), c))
        assert Complement(inner).expand() == Complement(NLMInter((a, b, c)))


class TestFirstMatch:
    def test_str_atom(self) -> None:
        a = Variable("a")
        assert str(FirstMatch(a)) == "first_match(a)"

    def test_str_compound(self) -> None:
        a, b = Variable("a"), Variable("b")
        assert str(FirstMatch(Concat((a, b)))) == "first_match(a ; b)"

    def test_str_nested_with_complement(self) -> None:
        a = Variable("a")
        assert str(FirstMatch(Complement(a))) == "first_match(~a)"

    def test_horizon_propagates(self) -> None:
        a = Variable("a")
        assert FirstMatch(Repeat(a, 0, 3)).horizon() == Repeat(a, 0, 3).horizon()

    def test_children_yields_arg(self) -> None:
        a = Variable("a")
        node = FirstMatch(a)
        assert list(node.children()) == [a]

    def test_expand_recurses(self) -> None:
        a, b, c = Variable("a"), Variable("b"), Variable("c")
        inner = NLMInter((NLMInter((a, b)), c))
        assert FirstMatch(inner).expand() == FirstMatch(NLMInter((a, b, c)))


class TestFusionRepeat:
    def test_str_star(self) -> None:
        a = Variable("a")
        assert str(FusionRepeat(a, 0, None)) == "a[:*]"

    def test_str_plus(self) -> None:
        a = Variable("a")
        assert str(FusionRepeat(a, 1, None)) == "a[:+]"

    def test_str_point(self) -> None:
        a = Variable("a")
        assert str(FusionRepeat(a, 3, 3)) == "a[:*3]"

    def test_str_range(self) -> None:
        a = Variable("a")
        assert str(FusionRepeat(a, 2, 5)) == "a[:*2..5]"

    def test_str_open_upper(self) -> None:
        a = Variable("a")
        assert str(FusionRepeat(a, 2, None)) == "a[:*2..]"

    def test_low_must_be_non_negative(self) -> None:
        a = Variable("a")
        with pytest.raises(ValueError):
            FusionRepeat(a, -1, None)

    def test_low_must_be_le_high(self) -> None:
        a = Variable("a")
        with pytest.raises(ValueError):
            FusionRepeat(a, 3, 2)

    def test_children_yields_arg(self) -> None:
        a = Variable("a")
        assert list(FusionRepeat(a, 0, None).children()) == [a]

    def test_horizon_unbounded(self) -> None:
        a = Variable("a")
        assert math.isinf(FusionRepeat(a, 0, None).horizon())

    def test_horizon_bounded(self) -> None:
        a = Variable("a")
        assert FusionRepeat(a, 0, 5).horizon() == 5 * a.horizon()

    def test_expand_zero_zero_is_true(self) -> None:
        from logic_asts.base import Literal

        a = Variable("a")
        assert FusionRepeat(a, 0, 0).expand() == Literal(True)

    def test_expand_one_one_collapses(self) -> None:
        a = Variable("a")
        assert FusionRepeat(a, 1, 1).expand() == a

    def test_expand_point_k_uses_fusion(self) -> None:
        a = Variable("a")
        assert FusionRepeat(a, 3, 3).expand() == Fusion((a, a, a))

    def test_expand_bounded_range_to_alt(self) -> None:
        from logic_asts.base import Literal

        a = Variable("a")
        expanded = FusionRepeat(a, 0, 2).expand()
        assert expanded == Alt((Literal(True), a, Fusion((a, a))))

    def test_expand_unbounded_stays_primitive(self) -> None:
        a = Variable("a")
        node = FusionRepeat(a, 0, None)
        assert node.expand() == node


class TestGotoRepeat:
    def test_str_default(self) -> None:
        a = Variable("a")
        # GotoRepeat(a, 1, 1) is the shorthand [->]
        assert str(GotoRepeat(a, 1, 1)) == "a[->]"

    def test_str_point(self) -> None:
        a = Variable("a")
        assert str(GotoRepeat(a, 2, 2)) == "a[->2]"

    def test_str_range(self) -> None:
        a = Variable("a")
        assert str(GotoRepeat(a, 2, 5)) == "a[->2..5]"

    def test_str_open_upper(self) -> None:
        a = Variable("a")
        assert str(GotoRepeat(a, 3, None)) == "a[->3..]"

    def test_low_must_be_non_negative(self) -> None:
        a = Variable("a")
        with pytest.raises(ValueError):
            GotoRepeat(a, -1, None)

    def test_low_must_be_le_high(self) -> None:
        a = Variable("a")
        with pytest.raises(ValueError):
            GotoRepeat(a, 3, 2)

    def test_children_yields_arg(self) -> None:
        a = Variable("a")
        assert list(GotoRepeat(a, 1, 1).children()) == [a]

    def test_horizon_is_inf(self) -> None:
        a = Variable("a")
        assert math.isinf(GotoRepeat(a, 1, 1).horizon())

    def test_expand_boolean_operand(self) -> None:
        # GotoRepeat(a, 1, 1).expand() builds:
        #   Repeat(Concat((Repeat(Complement(a), 0, None), a)), 1, 1).expand()
        # which collapses via Repeat's [*1..1] -> Concat((Repeat(Complement(a), 0, None), a))
        a = Variable("a")
        expanded = GotoRepeat(a, 1, 1).expand()
        expected = Concat((Repeat(Complement(a), 0, None), a))
        assert expanded == expected

    def test_expand_sere_operand(self) -> None:
        a, b = Variable("a"), Variable("b")
        inner = Concat((a, b))
        expanded = GotoRepeat(inner, 2, 3).expand()
        expected = Repeat(
            Concat((Repeat(Complement(inner), 0, None), inner)),
            2,
            3,
        )
        assert expanded == expected


class TestEqualRepeat:
    def test_str_default(self) -> None:
        a = Variable("a")
        # EqualRepeat(a, 0, None) is the shorthand [=]
        assert str(EqualRepeat(a, 0, None)) == "a[=]"

    def test_str_point(self) -> None:
        a = Variable("a")
        assert str(EqualRepeat(a, 2, 2)) == "a[=2]"

    def test_str_range(self) -> None:
        a = Variable("a")
        assert str(EqualRepeat(a, 2, 5)) == "a[=2..5]"

    def test_str_open_upper(self) -> None:
        a = Variable("a")
        assert str(EqualRepeat(a, 3, None)) == "a[=3..]"

    def test_low_must_be_non_negative(self) -> None:
        a = Variable("a")
        with pytest.raises(ValueError):
            EqualRepeat(a, -1, None)

    def test_low_must_be_le_high(self) -> None:
        a = Variable("a")
        with pytest.raises(ValueError):
            EqualRepeat(a, 3, 2)

    def test_children_yields_arg(self) -> None:
        a = Variable("a")
        assert list(EqualRepeat(a, 0, None).children()) == [a]

    def test_horizon_is_inf(self) -> None:
        a = Variable("a")
        assert math.isinf(EqualRepeat(a, 0, None).horizon())

    def test_expand_appends_complement_tail(self) -> None:
        # EqualRepeat(a, 2, 3).expand() should produce:
        #   Concat((GotoRepeat(a, 2, 3).expand(), Repeat(Complement(a), 0, None)))
        # which (per the GotoRepeat operand-preserving expand) is:
        #   Concat((Repeat(Concat((Repeat(Complement(a), 0, None), a)), 2, 3),
        #           Repeat(Complement(a), 0, None)))
        a = Variable("a")
        expanded = EqualRepeat(a, 2, 3).expand()
        goto_body = Repeat(Concat((Repeat(Complement(a), 0, None), a)), 2, 3)
        tail = Repeat(Complement(a), 0, None)
        expected = Concat((goto_body, tail))
        assert expanded == expected


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

    def test_parse_nlm_inter_binary(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("a & b", syntax="sere")
        assert expr == NLMInter((Variable("a"), Variable("b")))

    def test_parse_nlm_inter_lower_than_inter(self) -> None:
        from logic_asts import parse_expr

        # && binds tighter than &, so this is a & (b && c)
        expr = parse_expr("a & b && c", syntax="sere")
        assert expr == NLMInter((Variable("a"), Inter((Variable("b"), Variable("c")))))

    def test_parse_nlm_inter_higher_than_alt(self) -> None:
        from logic_asts import parse_expr

        # & binds tighter than |, so this is a | (b & c)
        expr = parse_expr("a | b & c", syntax="sere")
        assert expr == Alt((Variable("a"), NLMInter((Variable("b"), Variable("c")))))

    def test_parse_nlm_inter_chains(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("a & b & c", syntax="sere")
        assert expr == NLMInter((Variable("a"), Variable("b"), Variable("c")))

    def test_parse_nlm_inter_roundtrip(self) -> None:
        from logic_asts import parse_expr

        src = "((a & b) | (c && d))"
        expr = parse_expr(src, syntax="sere")
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


class TestParseComplement:
    def test_parse_complement_atom(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("~a", syntax="sere")
        assert expr == Complement(Variable("a"))

    def test_parse_double_complement(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("~~a", syntax="sere")
        assert expr == Complement(Complement(Variable("a")))

    def test_parse_complement_brace_compound(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("~{a;b}", syntax="sere")
        assert expr == Complement(Concat((Variable("a"), Variable("b"))))

    def test_complement_binds_tighter_than_concat(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("~a;b", syntax="sere")
        assert expr == Concat((Complement(Variable("a")), Variable("b")))

    def test_complement_binds_tighter_than_repeat(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("~a[*]", syntax="sere")
        assert expr == Repeat(Complement(Variable("a")), 0, None)

    def test_complement_binds_tighter_than_alt(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("~a | b", syntax="sere")
        assert expr == Alt((Complement(Variable("a")), Variable("b")))


def test_tilde_rejected_as_boolean_negation() -> None:
    """After grammar change, ``~`` is SERE-only; Boolean ``~`` is rejected."""
    import pytest as _pytest
    from lark.exceptions import LarkError

    from logic_asts import parse_expr

    with _pytest.raises(LarkError):
        parse_expr("~a", syntax="base")


class TestParseFirstMatch:
    def test_parse_first_match_atom(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("first_match(a)", syntax="sere")
        assert expr == FirstMatch(Variable("a"))

    def test_parse_first_match_compound(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("first_match(a;b)", syntax="sere")
        assert expr == FirstMatch(Concat((Variable("a"), Variable("b"))))

    def test_first_match_inside_complement(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("~first_match(a)", syntax="sere")
        assert expr == Complement(FirstMatch(Variable("a")))

    def test_complement_inside_first_match(self) -> None:
        from logic_asts import parse_expr

        expr = parse_expr("first_match(~a)", syntax="sere")
        assert expr == FirstMatch(Complement(Variable("a")))

    def test_first_match_with_repeat(self) -> None:
        from logic_asts import parse_expr

        # first_match(...) is an atom, so it accepts a repeat suffix.
        expr = parse_expr("first_match(a)[*]", syntax="sere")
        assert expr == Repeat(FirstMatch(Variable("a")), 0, None)


class TestParseFusionRepeat:
    def test_parse_fusion_star(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[:*]", syntax="sere") == FusionRepeat(Variable("a"), 0, None)

    def test_parse_fusion_plus(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[:+]", syntax="sere") == FusionRepeat(Variable("a"), 1, None)

    def test_parse_fusion_range(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[:*2..5]", syntax="sere") == FusionRepeat(Variable("a"), 2, 5)

    def test_parse_fusion_point(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[:*3]", syntax="sere") == FusionRepeat(Variable("a"), 3, 3)

    def test_parse_fusion_open_upper(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[:*2..]", syntax="sere") == FusionRepeat(Variable("a"), 2, None)


class TestParseGotoRepeat:
    def test_parse_goto_default(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[->]", syntax="sere") == GotoRepeat(Variable("a"), 1, 1)

    def test_parse_goto_point(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[->2]", syntax="sere") == GotoRepeat(Variable("a"), 2, 2)

    def test_parse_goto_range(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[->2..5]", syntax="sere") == GotoRepeat(Variable("a"), 2, 5)

    def test_parse_goto_sere_operand(self) -> None:
        from logic_asts import parse_expr

        # Non-Boolean operand: extension beyond Spot.
        expr = parse_expr("(a;b)[->2]", syntax="sere")
        assert expr == GotoRepeat(Concat((Variable("a"), Variable("b"))), 2, 2)


class TestParseEqualRepeat:
    def test_parse_equal_default(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[=]", syntax="sere") == EqualRepeat(Variable("a"), 0, None)

    def test_parse_equal_point(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[=3]", syntax="sere") == EqualRepeat(Variable("a"), 3, 3)

    def test_parse_equal_range(self) -> None:
        from logic_asts import parse_expr

        assert parse_expr("a[=2..5]", syntax="sere") == EqualRepeat(Variable("a"), 2, 5)


def test_complement_precedence_with_equal_repeat() -> None:
    """`~` still binds tighter than the new repeat suffixes."""
    from logic_asts import parse_expr

    assert parse_expr("~a[=2]", syntax="sere") == EqualRepeat(Complement(Variable("a")), 2, 2)


def test_concat_with_fusion_repeat() -> None:
    """`[:*]` binds tighter than `;`, mirroring `[*]`."""
    from logic_asts import parse_expr

    expr = parse_expr("a[:*3];b", syntax="sere")
    assert expr == Concat((FusionRepeat(Variable("a"), 3, 3), Variable("b")))
