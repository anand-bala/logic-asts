"""Type guard tests using typing.assert_type for static type checking.

These tests verify that the TypeIs guards correctly narrow types
for static type checkers like mypy. Run with: mypy tests/test_type_guards.py
"""

from typing import assert_type

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
    assert not logic_asts.is_propositional_logic(not_an_expr, str)
    assert not logic_asts.is_ltl_expr(not_an_expr, str)
    assert not logic_asts.is_strel_expr(not_an_expr, str)
    assert not logic_asts.is_stl_go_expr(not_an_expr, str)

    # Parse STREL-specific expression - should not be propositional logic
    strel_only: object = logic_asts.parse_expr("somewhere[0,10] p", syntax="strel")
    assert not logic_asts.is_propositional_logic(strel_only, str)
