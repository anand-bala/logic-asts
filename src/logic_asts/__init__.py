"""
.. include:: ../../README.md

# API Reference
"""

# mypy: allow_untyped_calls
import typing
from collections.abc import Hashable

from lark import Lark, Transformer
from typing_extensions import overload

import logic_asts.base as base
import logic_asts.ltl as ltl
import logic_asts.stl_go as stl_go
import logic_asts.strel as strel
from logic_asts.base import And as And
from logic_asts.base import BoolExpr as BoolExpr
from logic_asts.base import Equiv as Equiv
from logic_asts.base import Implies as Implies
from logic_asts.base import Literal as Literal
from logic_asts.base import Not as Not
from logic_asts.base import Or as Or
from logic_asts.base import Variable as Variable
from logic_asts.base import Xor as Xor
from logic_asts.base import bool_expr_iter as bool_expr_iter
from logic_asts.grammars import SupportedGrammars
from logic_asts.ltl import LTLExpr as LTLExpr
from logic_asts.ltl import ltl_expr_iter as ltl_expr_iter
from logic_asts.spec import Expr as Expr
from logic_asts.spec import ExprVisitor as ExprVisitor
from logic_asts.stl_go import STLGOExpr as STLGOExpr
from logic_asts.stl_go import stlgo_expr_iter as stlgo_expr_iter
from logic_asts.strel import STRELExpr as STRELExpr
from logic_asts.strel import strel_expr_iter as strel_expr_iter

SupportedGrammarsStr: typing.TypeAlias = typing.Literal["base", "ltl", "strel", "stl_go"]

_VarT = typing.TypeVar("_VarT", bound=Hashable)


def is_propositional_logic(obj: object, var_type: type[_VarT] | None = None) -> typing.TypeGuard[base.BaseExpr[_VarT]]:
    """Checks if the given object is an `Expr` and then checks if all the subexpressions are instances of `BaseExpr`

    > [!WARNING]
    > Using `None` as the `var_type` will automatically make the variable type check pass.
    """
    if isinstance(obj, Expr):
        # Extract origin if it's a subscripted generic
        check_type = typing.get_origin(var_type) or var_type if var_type else None
        return all(
            isinstance(expr, Implies | Equiv | Xor | And | Or | Not | Literal)
            or (isinstance(expr, Variable) and (check_type is None or isinstance(expr.name, check_type)))
            for expr in obj.iter_subtree()
        )
    return False


def is_ltl_expr(obj: object, var_type: type[_VarT] | None = None) -> typing.TypeGuard[ltl.LTLExpr[_VarT]]:
    """Checks if the given object is an `Expr` and then checks if all the subexpressions are instances of `LTLExpr`

    > [!WARNING]
    > Using `None` as the `var_type` will automatically make the variable type check pass.
    """
    if isinstance(obj, Expr):
        return all(
            is_propositional_logic(expr, var_type)
            or isinstance(expr, ltl.Next | ltl.Always | ltl.Eventually | ltl.Until | ltl.Release)
            for expr in obj.iter_subtree()
        )

    return False


def is_strel_expr(obj: object, var_type: type[_VarT] | None = None) -> typing.TypeGuard[strel.STRELExpr[_VarT]]:
    """Checks if the given object is an `Expr` and then checks if all the subexpressions are instances of `STRELExpr`

    > [!WARNING]
    > Using `None` as the `var_type` will automatically make the variable type check pass.
    """
    if isinstance(obj, Expr):
        return all(
            is_propositional_logic(expr, var_type)
            or is_ltl_expr(expr, var_type)
            or isinstance(expr, strel.Everywhere | strel.Somewhere | strel.Reach | strel.Escape)
            for expr in obj.iter_subtree()
        )
    return False


def is_stl_go_expr(obj: object, var_type: type[_VarT] | None = None) -> typing.TypeGuard[stl_go.STLGOExpr[_VarT]]:
    """Checks if the given object is an `Expr` and then checks if all the subexpressions are instances of `STLGOExpr`

    > [!WARNING]
    > Using `None` as the `var_type` will automatically make the variable type check pass.
    """
    if isinstance(obj, Expr):
        return all(
            is_propositional_logic(expr, var_type)
            or is_ltl_expr(expr, var_type)
            or isinstance(expr, stl_go.GraphIncoming | stl_go.GraphOutgoing)
            for expr in obj.iter_subtree()
        )
    return False


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["base", SupportedGrammars.BASE] = ...,
) -> base.BaseExpr[str]: ...


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["ltl", SupportedGrammars.LTL] = ...,
) -> ltl.LTLExpr[str]: ...


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["strel", SupportedGrammars.STREL] = ...,
) -> strel.STRELExpr[str]: ...


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["stl_go", SupportedGrammars.STL_GO] = ...,
) -> stl_go.STLGOExpr[str]: ...


def parse_expr(
    expr: str,
    *,
    syntax: SupportedGrammars | SupportedGrammarsStr = SupportedGrammars.BASE,
) -> Expr:
    syntax = SupportedGrammars(syntax)

    grammar = Lark.open_from_package(
        __name__,
        f"{str(syntax.value)}.lark",
        ["grammars"],
    )
    transformer = syntax.get_transformer()
    assert isinstance(transformer, Transformer), f"{transformer=}"

    parse_tree = grammar.parse(expr)
    return transformer.transform(tree=parse_tree)


__all__ = [
    "And",
    "BoolExpr",
    "Equiv",
    "Expr",
    "ExprVisitor",
    "Implies",
    "LTLExpr",
    "Literal",
    "Not",
    "Or",
    "STLGOExpr",
    "STRELExpr",
    "SupportedGrammars",
    "SupportedGrammarsStr",
    "Variable",
    "Xor",
    "base",
    "bool_expr_iter",
    "ltl",
    "ltl_expr_iter",
    "parse_expr",
    "stl_go",
    "stlgo_expr_iter",
    "strel",
    "strel_expr_iter",
]
