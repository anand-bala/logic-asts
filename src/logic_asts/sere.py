r"""Abstract syntax trees for Sequential Extended Regular Expressions (SERE).

This module implements the core subset of Spot's SERE syntax (see Spot's
``tl.pdf``, section 2.5). SERE is a regex-like layer over Boolean state
formulas with concatenation, fusion, alternation, length-matching and
non-length-matching intersection, and repetition. Boolean leaves reuse
``base.BaseExpr``.

Operators implemented:
    - Concat ``;``
    - Fusion ``:``
    - Alt    ``|`` (alternation, distinct from Boolean Or at the SERE level)
    - Inter    ``&&`` (length-matching intersection)
    - NLMInter ``&``  (non-length-matching intersection)
    - Complement ``~`` (SERE complement; extension beyond Spot)
    - FirstMatch ``first_match(r)`` (SVA-derived; matches Spot's surface syntax)
    - FusionRepeat ``[:*]``, ``[:+]``, ``[:*i]``, ``[:*i..j]``, ``[:*i..]`` (fusion-iteration)
    - GotoRepeat ``[->]``, ``[->i]``, ``[->i..j]``, ``[->i..]`` (goto-repetition; extension beyond Spot for non-Boolean operands)
    - EqualRepeat ``[=]``, ``[=i]``, ``[=i..j]``, ``[=i..]`` (equal-count repetition; extension beyond Spot for non-Boolean operands)
    - Repeat ``[*]``, ``[+]``, ``[*i]``, ``[*i..j]``, ``[*i..]``

Out of scope (not supported): delay operators ``##i`` / ``##[i..j]``.

Extensions beyond Spot: ``~`` at the SERE level denotes SERE complement
(``L(~r) = Sigma* \ L(r)``), not Boolean negation. ``!`` is the sole
Boolean negation glyph across every grammar in this package. Spot's
SERE grammar has no complement operator, so any string emitted by this
module containing ``~`` will be rejected by Spot. ``first_match``
matches Spot's surface syntax, but combinations with ``~`` remain
Spot-incompatible. Goto and equal repetition (``[->]``, ``[=]``) accept
arbitrary SERE operands here, not just Boolean formulas as in Spot.
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


@final
@frozen
class NLMInter(Expr):
    r"""SERE non-length-matching intersection: ``r1 & r2 & ... & rn``.

    A word ``w`` matches iff one operand matches ``w`` exactly and every
    other operand matches some prefix of ``w``. Equivalently::

        L(r1 & r2) = (L(r1) & L(r2) . Sigma*) | (L(r1) . Sigma* & L(r2))

    Contrast with :class:`Inter` (``&&``), which requires all operands
    to match the same word of the same length.
    """

    args: tuple[Expr, ...] = attrs.field(validator=attrs.validators.min_len(2))

    @override
    def __str__(self) -> str:
        return "(" + " & ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[Expr]:
        yield from self.args

    @override
    def expand(self) -> Expr:
        expanded = tuple(a.expand() for a in self.args)
        return NLMInter(_flatten_same_kind(NLMInter, expanded))

    @override
    def horizon(self) -> int | float:
        return max(a.horizon() for a in self.args)


@final
@frozen
class Complement(Expr):
    r"""SERE complement: ``~r``.

    Language: ``Sigma* \ L(r)``. Distinct from Boolean negation on a
    leaf (``!a``), which yields a single-letter language. ``~`` binds
    tighter than every other SERE operator (``[*]``, ``;``, ``:``,
    ``&``, ``&&``, ``|``).

    Extension beyond Spot: Spot's SERE grammar does not include a
    complement operator. See the module docstring.
    """

    arg: Expr

    @override
    def __str__(self) -> str:
        inner = self.arg
        if isinstance(inner, (Concat, Fusion, Alt, Inter, NLMInter, Repeat)):
            # Brace-group operands that would otherwise create round-trip
            # ambiguity. Multi-operator nodes already wrap themselves in
            # "(...)" so we strip those and re-wrap in braces. ``Repeat``
            # carries a postfix suffix (``[*]`` etc.) that would otherwise
            # bind to the outer ``~`` and re-parse as ``Repeat(Complement)``.
            body = str(inner)
            if body.startswith("(") and body.endswith(")"):
                body = body[1:-1]
            return f"~{{{body}}}"
        return f"~{inner}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def expand(self) -> Expr:
        return Complement(self.arg.expand())

    @override
    def horizon(self) -> int | float:
        return math.inf


@final
@frozen
class FirstMatch(Expr):
    r"""SERE first-match restriction: ``first_match(r)``.

    Language: words ``w`` in ``L(r)`` whose strictly shorter prefixes
    are all outside ``L(r)``. Matches Spot's ``first_match`` operator.
    """

    arg: Expr

    @override
    def __str__(self) -> str:
        body = str(self.arg)
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1]
        return f"first_match({body})"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def expand(self) -> Expr:
        return FirstMatch(self.arg.expand())

    @override
    def horizon(self) -> int | float:
        return self.arg.horizon()


@final
@frozen
class FusionRepeat(Expr):
    r"""Fusion-iteration: ``r[:*low..high]``.

    Like ``Repeat`` (`[*]`) but the separator is fusion (`:`) instead of
    concatenation (`;`). Bounded form is syntactic sugar over :class:`Fusion`;
    unbounded ``r[:*i..]`` (`high is None`) is a primitive operator
    (Dax et al.). See :meth:`expand`.

    ``low=None`` is treated as 0; ``high=None`` is unbounded.
    """

    arg: Expr
    low: int | None = attrs.field(default=None)
    high: int | None = attrs.field(default=None)

    def __attrs_post_init__(self) -> None:
        lo = _normalize_low(self.low)
        if lo < 0:
            raise ValueError(f"FusionRepeat.low must be non-negative, got {self.low}")
        if self.high is not None and lo > self.high:
            raise ValueError(f"FusionRepeat.low ({self.low}) must be <= FusionRepeat.high ({self.high})")

    @override
    def __str__(self) -> str:
        lo = _normalize_low(self.low)
        hi = self.high
        if lo == 0 and hi is None:
            suffix = "[:*]"
        elif lo == 1 and hi is None:
            suffix = "[:+]"
        elif hi is None:
            suffix = f"[:*{lo}..]"
        elif lo == hi:
            suffix = f"[:*{lo}]"
        else:
            suffix = f"[:*{lo}..{hi}]"
        arg_str = f"({self.arg})" if isinstance(self.arg, (Repeat, FusionRepeat)) else str(self.arg)
        return f"{arg_str}{suffix}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def expand(self) -> Expr:
        e = self.arg.expand()
        lo = _normalize_low(self.low)
        hi = self.high
        if hi is None:
            return FusionRepeat(e, self.low, None)
        if lo == 0 and hi == 0:
            return Literal(True)
        if lo == 1 and hi == 1:
            return e
        if lo == hi:
            return Fusion(tuple(e for _ in range(lo)))
        parts: list[Expr] = []
        for k in range(lo, hi + 1):
            if k == 0:
                parts.append(Literal(True))
            elif k == 1:
                parts.append(e)
            else:
                parts.append(Fusion(tuple(e for _ in range(k))))
        return Alt(tuple(parts))

    @override
    def horizon(self) -> int | float:
        if self.high is None:
            return math.inf
        return self.high * self.arg.horizon()


@final
@frozen
class GotoRepeat(Expr):
    r"""Goto-repetition: ``r[->low..high]``.

    Generalized from Spot's Boolean-operand definition to arbitrary SERE
    operand via :class:`Complement`. Extension beyond Spot.

    Semantics (always desugarable)::

        r[->i..j]  ==  (~r[*] ; r)[*i..j]

    ``low=None`` is treated as 0; ``high=None`` is unbounded.
    """

    arg: Expr
    low: int | None = attrs.field(default=None)
    high: int | None = attrs.field(default=None)

    def __attrs_post_init__(self) -> None:
        lo = _normalize_low(self.low)
        if lo < 0:
            raise ValueError(f"GotoRepeat.low must be non-negative, got {self.low}")
        if self.high is not None and lo > self.high:
            raise ValueError(f"GotoRepeat.low ({self.low}) must be <= GotoRepeat.high ({self.high})")

    @override
    def __str__(self) -> str:
        lo = _normalize_low(self.low)
        hi = self.high
        if lo == 1 and hi == 1:
            suffix = "[->]"
        elif hi is None:
            suffix = f"[->{lo}..]"
        elif lo == hi:
            suffix = f"[->{lo}]"
        else:
            suffix = f"[->{lo}..{hi}]"
        arg_str = f"({self.arg})" if isinstance(self.arg, (Repeat, FusionRepeat, GotoRepeat)) else str(self.arg)
        return f"{arg_str}{suffix}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def expand(self) -> Expr:
        # Preserve operand identity in the expansion: do NOT call .expand()
        # on the outer Repeat, because Concat.expand would flatten a SERE
        # operand ``r`` (which may itself be a Concat) into the surrounding
        # body. Downstream consumers like morphata pattern-match on the
        # expanded shape, so the body's tuple must stay (~r[*], r).
        e = self.arg.expand()
        body = Concat((Repeat(Complement(e), 0, None), e))
        lo = _normalize_low(self.low)
        if lo == 1 and self.high == 1:
            return body
        return Repeat(body, self.low, self.high)

    @override
    def horizon(self) -> int | float:
        return math.inf


@final
@frozen
class EqualRepeat(Expr):
    r"""Equal-count repetition: ``r[=low..high]``.

    Generalized from Spot's Boolean-operand definition to arbitrary SERE
    operand via :class:`Complement`. Extension beyond Spot.

    Semantics (always desugarable)::

        r[=i..j]  ==  (~r[*] ; r)[*i..j] ; ~r[*]

    ``low=None`` is treated as 0; ``high=None`` is unbounded.
    """

    arg: Expr
    low: int | None = attrs.field(default=None)
    high: int | None = attrs.field(default=None)

    def __attrs_post_init__(self) -> None:
        lo = _normalize_low(self.low)
        if lo < 0:
            raise ValueError(f"EqualRepeat.low must be non-negative, got {self.low}")
        if self.high is not None and lo > self.high:
            raise ValueError(f"EqualRepeat.low ({self.low}) must be <= EqualRepeat.high ({self.high})")

    @override
    def __str__(self) -> str:
        lo = _normalize_low(self.low)
        hi = self.high
        if lo == 0 and hi is None:
            suffix = "[=]"
        elif hi is None:
            suffix = f"[={lo}..]"
        elif lo == hi:
            suffix = f"[={lo}]"
        else:
            suffix = f"[={lo}..{hi}]"
        arg_str = f"({self.arg})" if isinstance(self.arg, (Repeat, FusionRepeat, GotoRepeat, EqualRepeat)) else str(self.arg)
        return f"{arg_str}{suffix}"

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def expand(self) -> Expr:
        # Preserve operand identity by reusing GotoRepeat.expand (which
        # already avoids the flattening trap) and tacking on the
        # complement-closure tail without re-flattening through
        # Concat.expand.
        goto_part = GotoRepeat(self.arg, self.low, self.high).expand()
        tail = Repeat(Complement(self.arg.expand()), 0, None)
        return Concat((goto_part, tail))

    @override
    def horizon(self) -> int | float:
        return math.inf


Var = TypeVar("Var")
SEREExpr: TypeAlias = (
    BoolExpr[Var]
    | Concat
    | Fusion
    | Alt
    | Inter
    | NLMInter
    | Complement
    | FirstMatch
    | FusionRepeat
    | GotoRepeat
    | EqualRepeat
    | Repeat
)
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
                NLMInter,
                Complement,
                FirstMatch,
                FusionRepeat,
                GotoRepeat,
                EqualRepeat,
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
    "NLMInter",
    "Complement",
    "FirstMatch",
    "FusionRepeat",
    "GotoRepeat",
    "EqualRepeat",
    "Repeat",
    "sere_expr_iter",
]
