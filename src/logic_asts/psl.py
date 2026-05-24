r"""Abstract syntax trees for PSL (Property Specification Logic).

PSL adds the SERE-LTL binding operators on top of LTL. See Spot's
``tl.pdf``, section 2.6.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeAlias, TypeVar, final

from attrs import frozen
from typing_extensions import override

from logic_asts.base import (
    And,
    Equiv,
    Implies,
    Literal,
    Not,
    Or,
    Variable,
    Xor,
)
from logic_asts.ltl import (
    Always,
    Eventually,
    LTLExpr,
    Next,
    Release,
    StrongNext,
    StrongRelease,
    Until,
    WeakUntil,
)
from logic_asts.sere import (
    Alt,
    Concat,
    Fusion,
    Inter,
    Repeat,
    SEREExpr,
)
from logic_asts.spec import Expr, ExprVisitor


@final
@frozen
class SuffixImpliesUniv(Expr):
    r"""``{r}[]-> f`` (universal suffix implication)."""

    sere: Expr
    formula: Expr

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}[]-> {self.formula}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere
        yield self.formula

    @override
    def expand(self) -> Expr:
        return SuffixImpliesUniv(self.sere.expand(), self.formula.expand())

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon() + self.formula.horizon()


@final
@frozen
class SuffixImpliesExist(Expr):
    r"""``{r}<>-> f`` (existential suffix implication)."""

    sere: Expr
    formula: Expr

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}<>-> {self.formula}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere
        yield self.formula

    @override
    def expand(self) -> Expr:
        return SuffixImpliesExist(self.sere.expand(), self.formula.expand())

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon() + self.formula.horizon()


@final
@frozen
class WeakClosure(Expr):
    r"""``{r}`` (weak closure)."""

    sere: Expr

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere

    @override
    def expand(self) -> Expr:
        return WeakClosure(self.sere.expand())

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon()


@final
@frozen
class StrongClosure(Expr):
    r"""``{r}!`` (strong closure)."""

    sere: Expr

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}!"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere

    @override
    def expand(self) -> Expr:
        return StrongClosure(self.sere.expand())

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon()


@final
@frozen
class NegStrongClosure(Expr):
    r"""``!{r}`` (negated strong closure, primitive per Spot)."""

    sere: Expr

    @override
    def __str__(self) -> str:
        return f"!{{{self.sere}}}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere

    @override
    def expand(self) -> Expr:
        return NegStrongClosure(self.sere.expand())

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon()


Var = TypeVar("Var")
PSLExpr: TypeAlias = (
    LTLExpr[Var] | SEREExpr[Var] | SuffixImpliesUniv | SuffixImpliesExist | WeakClosure | StrongClosure | NegStrongClosure
)


def psl_expr_iter(expr: PSLExpr[Var]) -> Iterator[PSLExpr[Var]]:
    """Post-order iterator over a PSL expression, validating dialect membership."""
    return iter(
        ExprVisitor[PSLExpr[Var]](
            (
                # PSL bindings
                SuffixImpliesUniv,
                SuffixImpliesExist,
                WeakClosure,
                StrongClosure,
                NegStrongClosure,
                # LTL
                Next,
                StrongNext,
                Always,
                Eventually,
                Until,
                WeakUntil,
                Release,
                StrongRelease,
                # SERE
                Concat,
                Fusion,
                Alt,
                Inter,
                Repeat,
                # Boolean
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
    "PSLExpr",
    "SuffixImpliesUniv",
    "SuffixImpliesExist",
    "WeakClosure",
    "StrongClosure",
    "NegStrongClosure",
    "psl_expr_iter",
]
