"""Type guard tests using typing.assert_type for static type checking.

These tests verify that the TypeIs guards correctly narrow types
for static type checkers like mypy. Run with: mypy tests/test_type_guards.py
"""

from typing_extensions import assert_type

import logic_asts
from logic_asts import base, ltl, stl_go, strel
from logic_asts.base import Variable


def test_is_propositional_logic_guard() -> None:
    """Test is_propositional_logic narrows to BaseExpr."""
    obj: object = logic_asts.parse_expr("p & q", syntax="base")

    if logic_asts.is_propositional_logic(obj, str):
        # Type checker should narrow obj to base.BaseExpr[Any]
        assert_type(obj, base.BaseExpr[str])
        # Should be able to access BaseExpr methods
        _ = obj.to_nnf()
        _ = obj.expand()

    # Test with Variable
    var: object = Variable("p")
    if logic_asts.is_propositional_logic(var, str):
        assert_type(var, base.BaseExpr[str])


def test_is_ltl_expr_guard() -> None:
    """Test is_ltl_expr narrows to LTLExpr."""
    obj: object = logic_asts.parse_expr("G(p -> F q)", syntax="ltl")

    if logic_asts.is_ltl_expr(obj, str):
        # Type checker should narrow obj to ltl.LTLExpr[Any]
        assert_type(obj, ltl.LTLExpr[str])
        # Should be able to access LTLExpr methods
        _ = obj.horizon()
        _ = obj.to_nnf()

    # Test with propositional logic (subset of LTL)
    prop: object = logic_asts.parse_expr("p & q", syntax="base")
    if logic_asts.is_ltl_expr(prop, str):
        assert_type(prop, ltl.LTLExpr[str])


def test_is_strel_expr_guard() -> None:
    """Test is_strel_expr narrows to STRELExpr."""
    obj: object = logic_asts.parse_expr("somewhere[0,10] p", syntax="strel")

    if logic_asts.is_strel_expr(obj, str):
        # Type checker should narrow obj to strel.STRELExpr[Any]
        assert_type(obj, strel.STRELExpr[str])
        # Should be able to access STRELExpr methods
        _ = obj.horizon()
        _ = obj.to_nnf()

    # Test with LTL (subset of STREL)
    ltl_expr: object = logic_asts.parse_expr("G p", syntax="ltl")
    if logic_asts.is_strel_expr(ltl_expr, str):
        assert_type(ltl_expr, strel.STRELExpr[str])


def test_is_stl_go_expr_guard() -> None:
    """Test is_stl_go_expr narrows to STLGOExpr."""
    obj: object = logic_asts.parse_expr("G p", syntax="stl_go")

    if logic_asts.is_stl_go_expr(obj, str):
        # Type checker should narrow obj to stl_go.STLGOExpr[Any]
        assert_type(obj, stl_go.STLGOExpr[str])
        # Should be able to access STLGOExpr methods
        _ = obj.horizon()
        _ = obj.to_nnf()

    # Test with propositional logic (subset of STL-GO)
    prop: object = logic_asts.parse_expr("p & q", syntax="base")
    if logic_asts.is_stl_go_expr(prop, str):
        assert_type(prop, stl_go.STLGOExpr[str])


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
        assert_type(tuple_expr, base.BaseExpr[tuple[str, int]])
        _ = tuple_expr.to_nnf()

    # Test with LTL
    if logic_asts.is_ltl_expr(tuple_expr, tuple[str, int]):
        assert_type(tuple_expr, ltl.LTLExpr[tuple[str, int]])
        _ = tuple_expr.horizon()

    # Test with STREL
    if logic_asts.is_strel_expr(tuple_expr, tuple[str, int]):
        assert_type(tuple_expr, strel.STRELExpr[tuple[str, int]])

    # Test with STL-GO
    if logic_asts.is_stl_go_expr(tuple_expr, tuple[str, int]):
        assert_type(tuple_expr, stl_go.STLGOExpr[tuple[str, int]])


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
