r"""Abstract syntax trees for Sequential Extended Regular Expressions (SERE).

This module implements the core subset of Spot's SERE syntax (see Spot's
``tl.pdf``, section 2.5). SERE is a regex-like layer over Boolean state
formulas with concatenation, fusion, alternation, length-matching
intersection, and repetition. Boolean leaves reuse ``base.BaseExpr``.

Operators implemented:
    - Concat ``;``
    - Fusion ``:``
    - Alt    ``|`` (alternation, distinct from Boolean Or at the SERE level)
    - Inter  ``&&`` (length-matching intersection)
    - Repeat ``[*]``, ``[+]``, ``[*i]``, ``[*i..j]``, ``[*i..]``
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TypeAlias, TypeVar, final

import attrs
from attrs import frozen
from typing_extensions import override

from logic_asts.base import And as And
from logic_asts.base import BoolExpr as BoolExpr
from logic_asts.base import Equiv as Equiv
from logic_asts.base import Implies as Implies
from logic_asts.base import Literal as Literal
from logic_asts.base import Not as Not
from logic_asts.base import Or as Or
from logic_asts.base import Variable as Variable
from logic_asts.base import Xor as Xor
from logic_asts.spec import Expr, ExprVisitor


def _normalize_low(value: int | None) -> int:
    return 0 if value is None else value


@final
@frozen
class Repeat(Expr):
    r"""Repetition: ``r[*low..high]``.

    ``low=None`` is treated as 0; ``high=None`` is unbounded.
    """

    arg: Expr
    low: int | None = attrs.field(default=None)
    high: int | None = attrs.field(default=None)

    def __attrs_post_init__(self) -> None:
        lo = _normalize_low(self.low)
        if lo < 0:
            raise ValueError(f"Repeat.low must be non-negative, got {self.low}")
        if self.high is not None and lo > self.high:
            raise ValueError(f"Repeat.low ({self.low}) must be <= Repeat.high ({self.high})")

    @override
    def __str__(self) -> str:
        lo = _normalize_low(self.low)
        hi = self.high
        if lo == 0 and hi is None:
            suffix = "[*]"
        elif lo == 1 and hi is None:
            suffix = "[+]"
        elif hi is None:
            suffix = f"[*{lo}..]"
        elif lo == hi:
            suffix = f"[*{lo}]"
        else:
            suffix = f"[*{lo}..{hi}]"
        arg_str = f"({self.arg})" if isinstance(self.arg, Repeat) else str(self.arg)
        return f"{arg_str}{suffix}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def expand(self) -> Expr:
        arg = self.arg.expand()
        if _normalize_low(self.low) == 1 and self.high == 1:
            return arg
        return Repeat(arg, self.low, self.high)

    @override
    def horizon(self) -> int | float:
        arg_hrz = self.arg.horizon()
        if self.high is None:
            return math.inf
        return self.high * arg_hrz


def _flatten_same_kind(cls: type[Expr], args: tuple[Expr, ...]) -> tuple[Expr, ...]:
    out: list[Expr] = []
    for a in args:
        if isinstance(a, cls):
            out.extend(a.args)  # type: ignore[attr-defined]
        else:
            out.append(a)
    return tuple(out)


@final
@frozen
class Concat(Expr):
    r"""SERE concatenation: ``r1 ; r2 ; ... ; rn``."""

    args: tuple[Expr, ...] = attrs.field(validator=attrs.validators.min_len(2))

    @override
    def __str__(self) -> str:
        return "(" + " ; ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[Expr]:
        yield from self.args

    @override
    def expand(self) -> Expr:
        expanded = tuple(a.expand() for a in self.args)
        return Concat(_flatten_same_kind(Concat, expanded))

    @override
    def horizon(self) -> int | float:
        return sum((a.horizon() for a in self.args), start=0)


@final
@frozen
class Fusion(Expr):
    r"""SERE fusion: ``r1 : r2 : ... : rn``."""

    args: tuple[Expr, ...] = attrs.field(validator=attrs.validators.min_len(2))

    @override
    def __str__(self) -> str:
        return "(" + " : ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[Expr]:
        yield from self.args

    @override
    def expand(self) -> Expr:
        expanded = tuple(a.expand() for a in self.args)
        return Fusion(_flatten_same_kind(Fusion, expanded))

    @override
    def horizon(self) -> int | float:
        return sum((a.horizon() for a in self.args), start=0)


@final
@frozen
class Alt(Expr):
    r"""SERE alternation: ``r1 | r2 | ... | rn``."""

    args: tuple[Expr, ...] = attrs.field(validator=attrs.validators.min_len(2))

    @override
    def __str__(self) -> str:
        return "(" + " | ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[Expr]:
        yield from self.args

    @override
    def expand(self) -> Expr:
        expanded = tuple(a.expand() for a in self.args)
        return Alt(_flatten_same_kind(Alt, expanded))

    @override
    def horizon(self) -> int | float:
        return max(a.horizon() for a in self.args)


@final
@frozen
class Inter(Expr):
    r"""SERE length-matching intersection: ``r1 && r2 && ... && rn``."""

    args: tuple[Expr, ...] = attrs.field(validator=attrs.validators.min_len(2))

    @override
    def __str__(self) -> str:
        return "(" + " && ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[Expr]:
        yield from self.args

    @override
    def expand(self) -> Expr:
        expanded = tuple(a.expand() for a in self.args)
        return Inter(_flatten_same_kind(Inter, expanded))

    @override
    def horizon(self) -> int | float:
        return max(a.horizon() for a in self.args)


Var = TypeVar("Var")
SEREExpr: TypeAlias = BoolExpr[Var] | Concat | Fusion | Alt | Inter | Repeat
"""Union of all SERE expression node types."""


def sere_expr_iter(expr: SEREExpr[Var]) -> Iterator[SEREExpr[Var]]:
    """Post-order iterator over a SERE expression, validating dialect membership."""
    return iter(
        ExprVisitor[SEREExpr[Var]](
            (
                Concat,
                Fusion,
                Alt,
                Inter,
                Repeat,
                Implies,
                Equiv,
                Xor,
                And,
                Or,
                Not,
                Variable[Var],
                Literal,
            ),
            expr,
        )
    )


__all__ = [
    "SEREExpr",
    "Concat",
    "Fusion",
    "Alt",
    "Inter",
    "Repeat",
    "sere_expr_iter",
]
