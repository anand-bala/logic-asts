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

from collections.abc import Hashable, Iterator
from typing import Generic, TypeVar, cast, final

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


def _validates_psl(instance: object, attribute: object, value: object) -> None:
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

    sere: SEREExpr[Var] = field(validator=_validates_sere)
    formula: PSLFormula[Var] = field(validator=_validates_psl)

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
            self.sere.expand(),
            self.formula.expand(),
        )

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> PSLFormula[Var]:
        if expand:
            return self.expand().to_nnf(negate=negate, expand=False)
        new_formula = self.formula.to_nnf(negate=negate, expand=False)
        if negate:
            # !({r}[]-> f) = {r}<>-> !f
            return SuffixImpliesExist(self.sere, new_formula)
        return SuffixImpliesUniv(self.sere, new_formula)

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon() + self.formula.horizon()


@final
@frozen
class SuffixImpliesExist(Expr, Generic[Var]):
    r"""``{r}<>-> f`` (existential suffix implication)."""

    sere: SEREExpr[Var] = field(validator=_validates_sere)
    formula: PSLFormula[Var] = field(validator=_validates_psl)

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
            self.sere.expand(),
            self.formula.expand(),
        )

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> PSLFormula[Var]:
        if expand:
            return self.expand().to_nnf(negate=negate, expand=False)
        new_formula = self.formula.to_nnf(negate=negate, expand=False)
        if negate:
            # !({r}<>-> f) = {r}[]-> !f
            return SuffixImpliesUniv(self.sere, new_formula)
        return SuffixImpliesExist(self.sere, new_formula)

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon() + self.formula.horizon()


@final
@frozen
class WeakClosure(Expr, Generic[Var]):
    r"""``{r}`` (weak closure)."""

    sere: SEREExpr[Var] = field(validator=_validates_sere)

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere

    @override
    def expand(self) -> WeakClosure[Var]:
        return WeakClosure(self.sere.expand())

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> PSLFormula[Var]:
        if expand:
            return self.expand().to_nnf(negate=negate, expand=False)
        # No NNF dual for a weak closure; block negation from passing through.
        return Not(self) if negate else self

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon()


@final
@frozen
class StrongClosure(Expr, Generic[Var]):
    r"""``{r}!`` (strong closure)."""

    sere: SEREExpr[Var] = field(validator=_validates_sere)

    @override
    def __str__(self) -> str:
        return f"{{{self.sere}}}!"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.sere

    @override
    def expand(self) -> SuffixImpliesExist[Var]:
        return SuffixImpliesExist(self.sere.expand(), Literal(True))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> PSLFormula[Var]:
        if expand:
            return self.expand().to_nnf(negate=negate, expand=False)
        # No NNF dual for a strong closure; block negation from passing through.
        return Not(self) if negate else self

    @override
    def horizon(self) -> int | float:
        return self.sere.horizon()


type PSLFormula[Var: Hashable] = (
    Variable[Var]
    | Literal
    | And[PSLFormula[Var]]
    | Or[PSLFormula[Var]]
    | Not[PSLFormula[Var]]
    | Implies[PSLFormula[Var]]
    | Equiv[PSLFormula[Var]]
    | Xor[PSLFormula[Var]]
    | Next[PSLFormula[Var]]
    | StrongNext[PSLFormula[Var]]
    | Always[PSLFormula[Var]]
    | Eventually[PSLFormula[Var]]
    | Until[PSLFormula[Var]]
    | WeakUntil[PSLFormula[Var]]
    | Release[PSLFormula[Var]]
    | StrongRelease[PSLFormula[Var]]
    | SuffixImpliesUniv[Var]
    | SuffixImpliesExist[Var]
    | WeakClosure[Var]
    | StrongClosure[Var]
)
"""Type of a PSL formula (no bare SEREs)."""

type PSLExpr[Var: Hashable] = PSLFormula[Var] | SEREExpr[Var]
"""Permissive umbrella: any node the PSL parser can produce."""


def psl_expr_iter(expr: PSLExpr[Var]) -> Iterator[PSLExpr[Var]]:
    """Post-order iterator over a PSL expression, validating dialect membership."""
    return iter(
        ExprVisitor(
            cast(
                list[type[PSLExpr[Var]]],
                [
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
                ],
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
