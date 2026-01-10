"""
.. include:: ../../README.md

# API Reference
"""

# mypy: allow_untyped_calls
import typing

from lark import Lark, Transformer
from typing_extensions import TypeIs, overload

import logic_asts.base as base
import logic_asts.ltl as ltl
import logic_asts.stl_go as stl_go
import logic_asts.strel as strel
from logic_asts.base import And as And
from logic_asts.base import Equiv as Equiv
from logic_asts.base import Expr
from logic_asts.base import Implies as Implies
from logic_asts.base import Literal as Literal
from logic_asts.base import Not as Not
from logic_asts.base import Or as Or
from logic_asts.base import Variable as Variable
from logic_asts.base import Xor as Xor
from logic_asts.grammars import SupportedGrammars

SupportedGrammarsStr: typing.TypeAlias = typing.Literal["base", "ltl", "strel", "stl_go"]


def is_propositional_logic(obj: object) -> TypeIs[base.BaseExpr[base.Var]]:
    return isinstance(obj, Implies | Equiv | Xor | And | Or | Not | Variable[base.Var] | Literal)


def is_ltl_expr(obj: object) -> TypeIs[ltl.LTLExpr[base.Var]]:
    return is_propositional_logic(obj) or isinstance(obj, ltl.Next | ltl.Always | ltl.Eventually | ltl.Until)


def is_strel_expr(obj: object) -> TypeIs[strel.STRELExpr[base.Var]]:
    return (
        is_propositional_logic(obj)
        or is_ltl_expr(obj)
        or isinstance(obj, strel.Everywhere | strel.Somewhere | strel.Reach | strel.Escape)
    )


def is_stl_go_expr(obj: object) -> TypeIs[stl_go.STLGOExpr[base.Var]]:
    return is_propositional_logic(obj) or is_ltl_expr(obj) or isinstance(obj, stl_go.GraphIncoming | stl_go.GraphOutgoing)


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["base", SupportedGrammars.BASE],
) -> base.BaseExpr[str]: ...


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["ltl", SupportedGrammars.LTL],
) -> ltl.LTLExpr[str]: ...


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["strel", SupportedGrammars.STREL],
) -> strel.STRELExpr[str]: ...


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["stl_go", SupportedGrammars.STL_GO],
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
    "parse_expr",
    "SupportedGrammars",
    "SupportedGrammarsStr",
    "Expr",
    "base",
    "ltl",
    "strel",
    "stl_go",
]
