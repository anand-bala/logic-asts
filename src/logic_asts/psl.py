r"""Abstract syntax trees for PSL (Property Specification Logic).

PSL adds the SERE-LTL binding operators on top of LTL. See Spot's
``tl.pdf``, section 2.6.

Scope: only a *subset* of Spot's PSL is supported. Bindings provided here
are ``{r}[]-> f``, ``{r}<>-> f``, ``{r}``, ``{r}!``, ``!{r}``, plus the
sugar ``{r}[]=> f`` and ``{r}<>=> f``. The PSL surface inherits the SERE
subset from :mod:`logic_asts.sere` (no delays, no goto/equal/non-consecutive
repetitions, no ``first_match``, no non-length-matching intersection).
Quantifiers, Spot's automatic simplification rules, and trace evaluation
are out of scope for this module.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeAlias, TypeVar, cast, final

from attrs import field, frozen
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
    SEREExpr,
)
from logic_asts.spec import Expr, ExprVisitor

Var = TypeVar("Var")


def _validates_sere(instance: object, attribute: object, value: object) -> None:
    # Lazy import to break the circular dependency with logic_asts/__init__.py.
    from logic_asts import is_sere_expr

    if not is_sere_expr(value):
        raise TypeError(f"{getattr(attribute, 'name', '<field>')} must be a SERE expression, got {type(value).__name__}")


_SERE_ONLY_TOP_NODES: tuple[type, ...] = (
    Concat,
    Fusion,
    Alt,
    Inter,
    NLMInter,
    Complement,
    FirstMatch,
    FusionRepeat,
    GotoRepeat,
    EqualRepeat,
    Repeat,
)
"""Node types that are SERE constructors but never valid as the top of a
PSL formula. Boolean atoms (``Variable``, ``Literal``, ``And``, ...) are
valid in both a SERE slot and a PSL-formula slot, so they are
deliberately omitted here."""


def _validates_psl_formula(instance: object, attribute: object, value: object) -> None:
    # Lazy import to break the circular dependency with logic_asts/__init__.py.
    from logic_asts import is_psl_expr

    if not is_psl_expr(value) or isinstance(value, _SERE_ONLY_TOP_NODES):
        raise TypeError(
            f"{getattr(attribute, 'name', '<field>')} must be a PSL formula (no bare SEREs), got {type(value).__name__}"
        )


@final
@frozen
class SuffixImpliesUniv(Expr, Generic[Var]):
    r"""``{r}[]-> f`` (universal suffix implication)."""

    sere: SEREExpr[Var] = field(validator=_validates_sere)  # type: ignore[assignment]
    formula: PSLFormula[Var] = field(validator=_validates_psl_formula)  # type: ignore[assignment]

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}[]-> {self.formula}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere
        yield self.formula

    @override
    def expand(self) -> SuffixImpliesUniv[Var]:
        return SuffixImpliesUniv(
            cast(SEREExpr[Var], self.sere.expand()),
            cast(PSLFormula[Var], self.formula.expand()),
        )

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon() + self.formula.horizon()


@final
@frozen
class SuffixImpliesExist(Expr, Generic[Var]):
    r"""``{r}<>-> f`` (existential suffix implication)."""

    sere: SEREExpr[Var] = field(validator=_validates_sere)  # type: ignore[assignment]
    formula: PSLFormula[Var] = field(validator=_validates_psl_formula)  # type: ignore[assignment]

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}<>-> {self.formula}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere
        yield self.formula

    @override
    def expand(self) -> SuffixImpliesExist[Var]:
        return SuffixImpliesExist(
            cast(SEREExpr[Var], self.sere.expand()),
            cast(PSLFormula[Var], self.formula.expand()),
        )

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon() + self.formula.horizon()


@final
@frozen
class WeakClosure(Expr, Generic[Var]):
    r"""``{r}`` (weak closure)."""

    sere: SEREExpr[Var] = field(validator=_validates_sere)  # type: ignore[assignment]

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere

    @override
    def expand(self) -> WeakClosure[Var]:
        return WeakClosure(cast(SEREExpr[Var], self.sere.expand()))

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon()


@final
@frozen
class StrongClosure(Expr, Generic[Var]):
    r"""``{r}!`` (strong closure)."""

    sere: SEREExpr[Var] = field(validator=_validates_sere)  # type: ignore[assignment]

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}!"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere

    @override
    def expand(self) -> SuffixImpliesExist[Var]:
        return SuffixImpliesExist(cast(SEREExpr[Var], self.sere.expand()), Literal(True))

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon()


PSLFormula: TypeAlias = LTLExpr[Var] | SuffixImpliesUniv[Var] | SuffixImpliesExist[Var] | WeakClosure[Var] | StrongClosure[Var]
"""Type of a PSL formula (no bare SEREs)."""

PSLExpr: TypeAlias = PSLFormula[Var] | SEREExpr[Var]
"""Permissive umbrella: any node the PSL parser can produce."""


def psl_expr_iter(expr: PSLExpr[Var]) -> Iterator[PSLExpr[Var]]:
    """Post-order iterator over a PSL expression, validating dialect membership."""
    return iter(
        ExprVisitor[PSLExpr[Var]](
            (  # type: ignore[arg-type]
                # PSL bindings
                SuffixImpliesUniv,
                SuffixImpliesExist,
                WeakClosure,
                StrongClosure,
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
    "PSLFormula",
    "SuffixImpliesUniv",
    "SuffixImpliesExist",
    "WeakClosure",
    "StrongClosure",
    "psl_expr_iter",
]
