"""Top-level package for logic-asts.

Provides grammars, parsers, and ASTs for logical formalisms including
propositional logic, LTL, STREL, and STL-GO.
"""

# mypy: allow_untyped_calls
import typing
from collections.abc import Hashable

from lark import Lark, Transformer
from typing_extensions import overload

import logic_asts.base as base
import logic_asts.ltl as ltl
import logic_asts.psl as psl
import logic_asts.sere as sere
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
from logic_asts.psl import PSLExpr as PSLExpr
from logic_asts.psl import psl_expr_iter as psl_expr_iter
from logic_asts.sere import SEREExpr as SEREExpr
from logic_asts.sere import sere_expr_iter as sere_expr_iter
from logic_asts.spec import Expr as Expr
from logic_asts.spec import ExprVisitor as ExprVisitor
from logic_asts.stl_go import STLGOExpr as STLGOExpr
from logic_asts.stl_go import stlgo_expr_iter as stlgo_expr_iter
from logic_asts.strel import STRELExpr as STRELExpr
from logic_asts.strel import strel_expr_iter as strel_expr_iter

SupportedGrammarsStr: typing.TypeAlias = typing.Literal["base", "ltl", "strel", "stl_go", "sere", "psl"]

_VarT = typing.TypeVar("_VarT", bound=Hashable)


def is_propositional_logic(obj: object, var_type: type[_VarT] | None = None) -> typing.TypeGuard[base.BaseExpr[_VarT]]:
    """Checks if the given object is an ``Expr`` and then checks if all the subexpressions are instances of ``BaseExpr``.

    Warning:
        Using ``None`` as the ``var_type`` will automatically make the variable type check pass.
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
    """Checks if the given object is an ``Expr`` and then checks if all the subexpressions are instances of ``LTLExpr``.

    Warning:
        Using ``None`` as the ``var_type`` will automatically make the variable type check pass.
    """
    if isinstance(obj, Expr):
        check_type = typing.get_origin(var_type) or var_type if var_type else None
        return all(
            isinstance(expr, Implies | Equiv | Xor | And | Or | Not | Literal)
            or (isinstance(expr, Variable) and (check_type is None or isinstance(expr.name, check_type)))
            or isinstance(
                expr,
                ltl.Next
                | ltl.StrongNext
                | ltl.Always
                | ltl.Eventually
                | ltl.Until
                | ltl.WeakUntil
                | ltl.Release
                | ltl.StrongRelease,
            )
            for expr in obj.iter_subtree()
        )

    return False


def is_strel_expr(obj: object, var_type: type[_VarT] | None = None) -> typing.TypeGuard[strel.STRELExpr[_VarT]]:
    """Checks if the given object is an ``Expr`` and then checks if all the subexpressions are instances of ``STRELExpr``.

    Warning:
        Using ``None`` as the ``var_type`` will automatically make the variable type check pass.
    """
    if isinstance(obj, Expr):
        check_type = typing.get_origin(var_type) or var_type if var_type else None
        return all(
            isinstance(expr, Implies | Equiv | Xor | And | Or | Not | Literal)
            or (isinstance(expr, Variable) and (check_type is None or isinstance(expr.name, check_type)))
            or isinstance(
                expr,
                ltl.Next
                | ltl.StrongNext
                | ltl.Always
                | ltl.Eventually
                | ltl.Until
                | ltl.WeakUntil
                | ltl.Release
                | ltl.StrongRelease,
            )
            or isinstance(expr, strel.Everywhere | strel.Somewhere | strel.Reach | strel.Escape)
            for expr in obj.iter_subtree()
        )
    return False


def is_stl_go_expr(obj: object, var_type: type[_VarT] | None = None) -> typing.TypeGuard[stl_go.STLGOExpr[_VarT]]:
    """Checks if the given object is an ``Expr`` and then checks if all the subexpressions are instances of ``STLGOExpr``.

    Warning:
        Using ``None`` as the ``var_type`` will automatically make the variable type check pass.
    """
    if isinstance(obj, Expr):
        check_type = typing.get_origin(var_type) or var_type if var_type else None
        return all(
            isinstance(expr, Implies | Equiv | Xor | And | Or | Not | Literal)
            or (isinstance(expr, Variable) and (check_type is None or isinstance(expr.name, check_type)))
            or isinstance(
                expr,
                ltl.Next
                | ltl.StrongNext
                | ltl.Always
                | ltl.Eventually
                | ltl.Until
                | ltl.WeakUntil
                | ltl.Release
                | ltl.StrongRelease,
            )
            or isinstance(expr, stl_go.GraphIncoming | stl_go.GraphOutgoing)
            for expr in obj.iter_subtree()
        )
    return False


def is_sere_expr(obj: object, var_type: type[_VarT] | None = None) -> typing.TypeGuard[sere.SEREExpr[_VarT]]:
    """Check that ``obj`` is a SERE-only expression tree."""
    if isinstance(obj, Expr):
        check_type = typing.get_origin(var_type) or var_type if var_type else None
        return all(
            isinstance(expr, Implies | Equiv | Xor | And | Or | Not | Literal)
            or (isinstance(expr, Variable) and (check_type is None or isinstance(expr.name, check_type)))
            or isinstance(
                expr,
                sere.Concat
                | sere.Fusion
                | sere.Alt
                | sere.Inter
                | sere.NLMInter
                | sere.Complement
                | sere.FirstMatch
                | sere.FusionRepeat
                | sere.GotoRepeat
                | sere.EqualRepeat
                | sere.Repeat,
            )
            for expr in obj.iter_subtree()
        )
    return False


def is_psl_expr(obj: object, var_type: type[_VarT] | None = None) -> typing.TypeGuard[psl.PSLExpr[_VarT]]:
    """Check that ``obj`` is a PSL-only expression tree (LTL + SERE + PSL bindings)."""
    if isinstance(obj, Expr):
        check_type = typing.get_origin(var_type) or var_type if var_type else None
        return all(
            isinstance(expr, Implies | Equiv | Xor | And | Or | Not | Literal)
            or (isinstance(expr, Variable) and (check_type is None or isinstance(expr.name, check_type)))
            or isinstance(
                expr,
                ltl.Next
                | ltl.StrongNext
                | ltl.Always
                | ltl.Eventually
                | ltl.Until
                | ltl.WeakUntil
                | ltl.Release
                | ltl.StrongRelease,
            )
            or isinstance(
                expr,
                sere.Concat
                | sere.Fusion
                | sere.Alt
                | sere.Inter
                | sere.NLMInter
                | sere.Complement
                | sere.FirstMatch
                | sere.FusionRepeat
                | sere.GotoRepeat
                | sere.EqualRepeat
                | sere.Repeat,
            )
            or isinstance(
                expr,
                psl.SuffixImpliesUniv | psl.SuffixImpliesExist | psl.WeakClosure | psl.StrongClosure,
            )
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
def parse_expr(  # pyright: ignore[reportOverlappingOverload]
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


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["sere", SupportedGrammars.SERE] = ...,
) -> sere.SEREExpr[str]: ...


@overload
def parse_expr(
    expr: str,
    *,
    syntax: typing.Literal["psl", SupportedGrammars.PSL] = ...,
) -> psl.PSLExpr[str]: ...


def parse_expr(
    expr: str,
    *,
    syntax: SupportedGrammars | SupportedGrammarsStr = SupportedGrammars.BASE,
) -> Expr:
    """Parse a logical expression string into an AST.

    For LTL expressions, uses Spot syntax with support for weak/strong
    operators (X, X[!], W, M) and bounded temporal operators.
    """
    syntax = SupportedGrammars(syntax)

    # These grammars are LALR-compatible (efficient deterministic parsing);
    # sere/psl still need the Earley parser until their grammar conflicts are
    # resolved.
    _LALR_GRAMMARS = {"base", "ltl", "strel", "stl_go"}
    parser = "lalr" if str(syntax.value) in _LALR_GRAMMARS else "earley"

    grammar = Lark.open_from_package(
        __name__,
        f"{str(syntax.value)}.lark",
        ["grammars"],
        parser=parser,
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
    "PSLExpr",
    "SEREExpr",
    "STLGOExpr",
    "STRELExpr",
    "SupportedGrammars",
    "SupportedGrammarsStr",
    "Variable",
    "Xor",
    "base",
    "bool_expr_iter",
    "is_psl_expr",
    "is_sere_expr",
    "ltl",
    "ltl_expr_iter",
    "parse_expr",
    "psl",
    "psl_expr_iter",
    "sere",
    "sere_expr_iter",
    "stl_go",
    "stlgo_expr_iter",
    "strel",
    "strel_expr_iter",
]
