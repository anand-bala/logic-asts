"""
.. include:: ../../README.md

# API Reference
"""

# mypy: allow_untyped_calls
import typing

from lark import Lark, Transformer
from typing_extensions import overload

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
