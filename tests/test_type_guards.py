"""Type guard tests using typing.assert_type for static type checking.

These tests verify that the TypeIs guards correctly narrow types
for static type checkers like mypy. Run with: mypy tests/test_type_guards.py
"""

from typing_extensions import assert_type

import logic_asts
from logic_asts import base, ltl, psl, sere, stl_go, strel
from logic_asts.base import Variable


def test_is_propositional_logic_guard() -> None:
    """Test is_propositional_logic narrows to BaseExpr."""
    obj: object = logic_asts.parse_expr("p & q", syntax="base")

    if logic_asts.is_propositional_logic(obj, str):
        # Type checker should narrow obj to base.BaseExpr[Any]
        _ = assert_type(obj, base.BaseExpr[str])
        # Should be able to access BaseExpr methods
        _ = obj.to_nnf()
        _ = obj.expand()

    # Test with Variable
    var: object = Variable("p")
    if logic_asts.is_propositional_logic(var, str):
        _ = assert_type(var, base.BaseExpr[str])


def test_is_ltl_expr_guard() -> None:
    """Test is_ltl_expr narrows to LTLExpr."""
    obj: object = logic_asts.parse_expr("G(p -> F q)", syntax="ltl")

    if logic_asts.is_ltl_expr(obj, str):
        # Type checker should narrow obj to ltl.LTLExpr[Any]
        _ = assert_type(obj, ltl.LTLExpr[str])
        # Should be able to access LTLExpr methods
        _ = obj.horizon()
        _ = obj.to_nnf()

    # Test with propositional logic (subset of LTL)
    prop: object = logic_asts.parse_expr("p & q", syntax="base")
    if logic_asts.is_ltl_expr(prop, str):
        _ = assert_type(prop, ltl.LTLExpr[str])


def test_is_strel_expr_guard() -> None:
    """Test is_strel_expr narrows to STRELExpr."""
    obj: object = logic_asts.parse_expr("somewhere[0,10] p", syntax="strel")

    if logic_asts.is_strel_expr(obj, str):
        # Type checker should narrow obj to strel.STRELExpr[Any]
        _ = assert_type(obj, strel.STRELExpr[str])
        # Should be able to access STRELExpr methods
        _ = obj.horizon()
        _ = obj.to_nnf()

    # Test with LTL (subset of STREL)
    ltl_expr: object = logic_asts.parse_expr("G p", syntax="ltl")
    if logic_asts.is_strel_expr(ltl_expr, str):
        _ = assert_type(ltl_expr, strel.STRELExpr[str])


def test_is_stl_go_expr_guard() -> None:
    """Test is_stl_go_expr narrows to STLGOExpr."""
    obj: object = logic_asts.parse_expr("G p", syntax="stl_go")

    if logic_asts.is_stl_go_expr(obj, str):
        # Type checker should narrow obj to stl_go.STLGOExpr[Any]
        _ = assert_type(obj, stl_go.STLGOExpr[str])
        # Should be able to access STLGOExpr methods
        _ = obj.horizon()
        _ = obj.to_nnf()

    # Test with propositional logic (subset of STL-GO)
    prop: object = logic_asts.parse_expr("p & q", syntax="base")
    if logic_asts.is_stl_go_expr(prop, str):
        _ = assert_type(prop, stl_go.STLGOExpr[str])


def test_is_sere_expr_guard() -> None:
    """Test is_sere_expr narrows to SEREExpr."""
    obj: object = logic_asts.parse_expr("a ; b[+] ; c", syntax="sere")

    if logic_asts.is_sere_expr(obj, str):
        # Type checker should narrow obj to sere.SEREExpr[Any]
        _ = assert_type(obj, sere.SEREExpr[str])

    # Pure Boolean tree is a subset of SERE
    prop: object = logic_asts.parse_expr("p & q", syntax="base")
    assert logic_asts.is_sere_expr(prop, str)
    if logic_asts.is_sere_expr(prop, str):
        _ = assert_type(prop, sere.SEREExpr[str])

    # Cross-dialect rejection: LTL temporal operators are not SERE
    ltl_expr: object = logic_asts.parse_expr("G p", syntax="ltl")
    assert not logic_asts.is_sere_expr(ltl_expr, str)


def test_is_psl_expr_guard() -> None:
    """Test is_psl_expr narrows to PSLExpr."""
    obj: object = logic_asts.parse_expr("{a;b}[]-> F c", syntax="psl")

    if logic_asts.is_psl_expr(obj, str):
        # Type checker should narrow obj to psl.PSLExpr[Any]
        _ = assert_type(obj, psl.PSLExpr[str])

    # LTL is a subset of PSL
    ltl_expr: object = logic_asts.parse_expr("G(p -> F q)", syntax="ltl")
    assert logic_asts.is_psl_expr(ltl_expr, str)

    # Pure Boolean tree is a subset of PSL
    prop: object = logic_asts.parse_expr("p & q", syntax="base")
    assert logic_asts.is_psl_expr(prop, str)
    if logic_asts.is_psl_expr(prop, str):
        _ = assert_type(prop, psl.PSLExpr[str])

    # Cross-dialect rejection: STREL spatial operators are not PSL
    strel_expr: object = logic_asts.parse_expr("somewhere[0,10] p", syntax="strel")
    assert not logic_asts.is_psl_expr(strel_expr, str)


def test_bool_accepted_by_sere_and_psl() -> None:
    """A pure Boolean tree should be accepted by both is_sere_expr and is_psl_expr."""
    prop: object = logic_asts.parse_expr("(p & q) | !r", syntax="base")
    assert logic_asts.is_sere_expr(prop, str)
    assert logic_asts.is_psl_expr(prop, str)


def test_ltl_boolean_connective_over_temporal() -> None:
    """Regression: Boolean connectives wrapping temporal ops must pass is_ltl_expr.

    Previously, is_ltl_expr called is_propositional_logic on each node from
    iter_subtree(), which recursed into sub-subtrees and found temporal
    operators inside Boolean nodes, causing false negatives.
    """
    # (F a) & (F b)
    and_of_eventually: object = logic_asts.parse_expr("(F(a) & F(b))", syntax="ltl")
    assert logic_asts.is_ltl_expr(and_of_eventually, str)
    assert logic_asts.is_ltl_expr(and_of_eventually, None)

    # G a | G b
    or_of_globally: object = logic_asts.parse_expr("(G(a) | G(b))", syntax="ltl")
    assert logic_asts.is_ltl_expr(or_of_globally, str)

    # !(F a)
    not_eventually: object = logic_asts.parse_expr("!(F(a))", syntax="ltl")
    assert logic_asts.is_ltl_expr(not_eventually, str)

    # Nested: G(a & F b)
    nested: object = logic_asts.parse_expr("G(a & F(b))", syntax="ltl")
    assert logic_asts.is_ltl_expr(nested, str)

    # Same expressions must still fail is_propositional_logic
    assert not logic_asts.is_propositional_logic(and_of_eventually, str)
    assert not logic_asts.is_propositional_logic(or_of_globally, str)


def test_strel_boolean_connective_over_spatial() -> None:
    """Regression: Boolean connectives wrapping spatial ops must pass is_strel_expr."""
    expr: object = logic_asts.parse_expr("somewhere[0,10] p & everywhere[0,5] q", syntax="strel")
    assert logic_asts.is_strel_expr(expr, str)
    assert not logic_asts.is_ltl_expr(expr, str)


def test_negative_guards() -> None:
    """Test that guards correctly reject non-matching types."""
    not_an_expr: object = "just a string"

    # These should all be False
    assert not logic_asts.is_propositional_logic(not_an_expr)
    assert not logic_asts.is_ltl_expr(not_an_expr)
    assert not logic_asts.is_strel_expr(not_an_expr)
    assert not logic_asts.is_stl_go_expr(not_an_expr)

    # Parse STREL-specific expression - should not be propositional logic
    strel_only: object = logic_asts.parse_expr("somewhere[0,10] p", syntax="strel")
    assert not logic_asts.is_propositional_logic(strel_only, str)


def test_var_type_checking() -> None:
    """Test that var_type parameter correctly validates variable types."""
    # Create expressions with string variables
    str_expr: object = Variable("p") & Variable("q")

    # Should pass with str var_type
    assert logic_asts.is_propositional_logic(str_expr, str)
    assert logic_asts.is_ltl_expr(str_expr, str)

    # Should pass with None (no type checking)
    assert logic_asts.is_propositional_logic(str_expr, None)
    assert logic_asts.is_ltl_expr(str_expr, None)

    # Should fail with wrong var_type
    assert not logic_asts.is_propositional_logic(str_expr, int)
    assert not logic_asts.is_ltl_expr(str_expr, int)

    # Create expression with int variables
    int_expr: object = Variable(1) & Variable(2)

    # Should pass with int var_type
    assert logic_asts.is_propositional_logic(int_expr, int)

    # Should fail with str var_type
    assert not logic_asts.is_propositional_logic(int_expr, str)


def test_var_type_with_tuple_variables() -> None:
    """Test var_type checking with tuple variable names."""
    # Create expression with tuple variables
    tuple_expr: object = Variable(("agent", 0)) & Variable(("agent", 1))

    # Should pass with tuple var_type
    assert logic_asts.is_propositional_logic(tuple_expr, tuple)
    assert logic_asts.is_ltl_expr(tuple_expr, tuple)
    assert logic_asts.is_strel_expr(tuple_expr, tuple)
    assert logic_asts.is_stl_go_expr(tuple_expr, tuple)

    # Should fail with wrong var_type
    assert not logic_asts.is_propositional_logic(tuple_expr, str)
    assert not logic_asts.is_ltl_expr(tuple_expr, str)

    # Should pass with None
    assert logic_asts.is_propositional_logic(tuple_expr, None)


def test_subscripted_var_types() -> None:
    """Test var_type checking with subscripted generic types."""
    # Create expression with tuple variables
    tuple_expr: object = Variable(("agent", 0)) & Variable(("sensor", 1))

    # Should pass with subscripted tuple type (checks origin type only)
    if logic_asts.is_propositional_logic(tuple_expr, tuple[str, int]):
        # Type should be narrowed to BaseExpr[tuple[str, int]]
        _ = assert_type(tuple_expr, base.BaseExpr[tuple[str, int]])
        _ = tuple_expr.to_nnf()

    # Test with LTL
    if logic_asts.is_ltl_expr(tuple_expr, tuple[str, int]):
        _ = assert_type(tuple_expr, ltl.LTLExpr[tuple[str, int]])
        _ = tuple_expr.horizon()

    # Test with STREL
    if logic_asts.is_strel_expr(tuple_expr, tuple[str, int]):
        _ = assert_type(tuple_expr, strel.STRELExpr[tuple[str, int]])

    # Test with STL-GO
    if logic_asts.is_stl_go_expr(tuple_expr, tuple[str, int]):
        _ = assert_type(tuple_expr, stl_go.STLGOExpr[tuple[str, int]])


def test_strel_destructuring_is_precise() -> None:
    """An And destructured from an STREL/STL-GO expr yields children of that dialect type.

    The refactor's payoff: a destructured child can be passed to a function expecting the
    dialect union with NO cast. Before the node classes were parameterized, ``args`` was
    ``tuple[Expr, ...]`` and these calls would be type errors. The value is typed as the
    union via a function PARAMETER so both mypy and basedpyright treat it as the declared
    union (not the narrower assigned-value type).
    """
    from logic_asts import stl_go, strel
    from logic_asts.base import And, Variable

    def want_strel(x: strel.STRELExpr[str]) -> None: ...
    def want_stlgo(x: stl_go.STLGOExpr[str]) -> None: ...

    def check_strel(e: strel.STRELExpr[str]) -> None:
        match e:
            case And(args):
                for child in args:
                    want_strel(child)  # no cast: child is STRELExpr[str]
            case _:
                pass

    def check_stlgo(e: stl_go.STLGOExpr[str]) -> None:
        match e:
            case And(args):
                for child in args:
                    want_stlgo(child)  # no cast: child is STLGOExpr[str]
            case _:
                pass

    check_strel(And((Variable("p"), Variable("q"))))
    check_stlgo(And((Variable("p"), Variable("q"))))


def test_psl_destructuring_is_precise() -> None:
    """A destructured PSL And yields children of type PSLFormula (cast-free)."""
    from logic_asts import psl
    from logic_asts.base import And, Variable

    def want_psl(x: psl.PSLFormula[str]) -> None: ...

    def check(e: psl.PSLFormula[str]) -> None:
        match e:
            case And(args):
                for child in args:
                    want_psl(child)  # no cast: child is PSLFormula[str]
            case _:
                pass

    check(And((Variable("p"), Variable("q"))))


def test_subscripted_types_runtime_behavior() -> None:
    """Test runtime behavior with subscripted types."""
    # With tuple[str, int], should check origin tuple type
    tuple_expr: object = Variable(("a", "b")) & Variable(("c", "d"))

    # Should pass - origin type is tuple
    assert logic_asts.is_propositional_logic(tuple_expr, tuple[str, int])
    assert logic_asts.is_ltl_expr(tuple_expr, tuple[str, int])

    # String variables should fail with tuple[str, int]
    str_expr: object = Variable("p") & Variable("q")
    assert not logic_asts.is_propositional_logic(str_expr, tuple[str, int])

    # List would work if we had list variables (just showing the concept)
    # list_expr: object = Variable([1, 2])
    # assert not logic_asts.is_propositional_logic(list_expr, tuple[str, int])


def test_ltl_destructuring_is_precise() -> None:
    """A destructured LTL Until yields lhs/rhs of type LTLExpr (cast-free)."""
    from logic_asts import ltl
    from logic_asts.base import Variable
    from logic_asts.ltl import Until

    def want_ltl(x: ltl.LTLExpr[str]) -> None: ...

    def check(e: ltl.LTLExpr[str]) -> None:
        match e:
            case Until(lhs, rhs):
                want_ltl(lhs)  # no cast: lhs is LTLExpr[str]
                want_ltl(rhs)  # no cast: rhs is LTLExpr[str]
            case _:
                pass

    check(Until(Variable("p"), Variable("q")))


def test_sere_destructuring_is_precise() -> None:
    """A destructured SERE Concat yields children of type SEREExpr (cast-free)."""
    from logic_asts import sere
    from logic_asts.base import Variable
    from logic_asts.sere import Concat

    def want_sere(x: sere.SEREExpr[str]) -> None: ...

    def check(e: sere.SEREExpr[str]) -> None:
        match e:
            case Concat(args):
                for child in args:
                    want_sere(child)  # no cast: child is SEREExpr[str]
            case _:
                pass

    check(Concat((Variable("p"), Variable("q"))))


def test_bool_destructuring_is_precise() -> None:
    """A destructured propositional And yields children of type BaseExpr (cast-free)."""
    from logic_asts import base
    from logic_asts.base import And, Variable

    def want_bool(x: base.BaseExpr[str]) -> None: ...

    def check(e: base.BaseExpr[str]) -> None:
        match e:
            case And(args):
                for child in args:
                    want_bool(child)  # no cast: child is BaseExpr[str]
            case _:
                pass

    check(And((Variable("p"), Variable("q"))))
