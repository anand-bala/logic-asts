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
import typing
from abc import ABC
from collections.abc import Hashable, Iterator
from typing import Generic, Self, TypeVar, cast, final

import attrs
from attrs import frozen
from typing_extensions import TypeGuard, override

from logic_asts.base import And as And
from logic_asts.base import Equiv as Equiv
from logic_asts.base import Implies as Implies
from logic_asts.base import Literal as Literal
from logic_asts.base import Not as Not
from logic_asts.base import Or as Or
from logic_asts.base import Variable as Variable
from logic_asts.base import Xor as Xor
from logic_asts.base import is_bool_node as is_bool_node
from logic_asts.spec import ChildExpr, Expr, ExprVisitor
from logic_asts.utils import flatten_nary_args, nary_fold


def is_sere_node[_T: Hashable](node: object, check_type: type[_T] | None = None) -> TypeGuard[SEREExpr[_T]]:
    """Shallow membership test: is ``node`` a SERE node (bool or SERE op)?

    >>> is_sere_node(Concat((Variable("a"), Variable("b"))))
    True
    """
    return is_bool_node(node, check_type) or isinstance(
        node,
        (Empty, Concat, Fusion, Alt, Inter, NLMInter, Complement, FirstMatch, FusionRepeat, GotoRepeat, EqualRepeat, Repeat),
    )


def _validate_sere_child(_instance: object, attribute: attrs.Attribute, value: object) -> None:  # type: ignore[type-arg]
    if not is_sere_node(value):
        raise TypeError(f"{attribute.name} must be a SERE expression, got {type(value).__name__}")


def _normalize_low(value: int | None) -> int:
    return 0 if value is None else value


class SEREOp(Expr, ABC):
    r"""Mixin supplying SERE-level operator dunders for SERE-specific nodes.

    For SERE operators the Python dunders denote the SERE connectives,
    not the Boolean ones inherited from :class:`Expr`:

    - ``~r``      -> :class:`Complement`
    - ``r1 | r2`` -> :class:`Alt`   (alternation)
    - ``r1 & r2`` -> :class:`Inter` (length-matching intersection)

    Boolean leaves (:class:`Variable`, :class:`Literal`, :class:`And`,
    :class:`Or`, :class:`Not`) keep their Boolean dunders, so a Boolean
    state formula ``a | b`` is still a single-letter ``Or`` (which
    coincides with ``Alt`` on Boolean operands).

    Note:
        Operator dispatch is left-biased: ``r | a`` (SERE node on the
        left) yields ``Alt``, but ``a | r`` (a Boolean leaf on the left)
        yields a Boolean ``Or`` because the leaf's ``__or__`` never
        defers. Lead with the SERE node, or use the constructor, when
        combining a Boolean leaf with a compound SERE operand.
    """

    @override
    def __invert__(self) -> Self:
        return cast(Self, Complement(self))

    @override
    def __or__(self, other: Expr) -> Self:
        return cast(Self, nary_fold(Alt, (self, other)))

    @override
    def __and__(self, other: Expr) -> Self:
        return cast(Self, nary_fold(Inter, (self, other)))


@final
@frozen
class Empty(SEREOp):
    r"""Zero-length matching input: :math:`\varepsilon`

    This is equivalent to ``Repeat(Literal(True), 0, 0)``.
    """

    @override
    def __str__(self) -> str:
        return str(Repeat(Literal(True), 0, 0))

    @override
    def expand(self) -> Self:
        return self

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> Repeat | Empty:
        _ = expand
        if negate:
            return Repeat(Literal(True), low=1, high=None)
        else:
            return Empty()

    @override
    def children(self) -> Iterator[Expr]:
        yield from iter(())

    @override
    def horizon(self) -> int | float:
        return 0

    @override
    def __invert__(self) -> SEREExpr[typing.Any]:  # type: ignore[override]
        comp: SEREExpr[typing.Any] = Repeat(Literal(True), 1, None)
        return comp


@final
@frozen
class Repeat(SEREOp, Generic[ChildExpr]):
    r"""Repetition: ``r[*low..high]``.

    ``low=None`` is treated as 0; ``high=None`` is unbounded.
    """

    arg: ChildExpr = attrs.field(validator=_validate_sere_child)
    low: int | None = attrs.field(default=None)
    high: int | None = attrs.field(default=None)

    def __attrs_post_init__(self) -> None:
        lo = _normalize_low(self.low)
        if lo < 0:
            raise ValueError(f"Repeat.low must be non-negative, got {self.low}")
        if self.high is not None and lo > self.high:
            raise ValueError(f"Repeat.low ({self.low}) must be <= Repeat.high ({self.high})")
        if self.high == 0:
            # Effectively "accepts the empty string only" and we should canonicalize `arg`
            object.__setattr__(self, "arg", Literal(True))

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
    def children(self) -> Iterator[ChildExpr]:
        yield self.arg

    @override
    def expand(self) -> ChildExpr:
        arg = self.arg.expand()
        low = _normalize_low(self.low)
        high = self.high
        if isinstance(arg, Empty):
            # epsilon repeated any number of times is still epsilon
            return cast(ChildExpr, Empty())
        if low == high == 0:
            return cast(ChildExpr, Empty())
        if low == high == 1:
            return cast(ChildExpr, arg)
        return cast(ChildExpr, Repeat(arg, low, high))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            if isinstance(self.arg, Literal) and self.arg.value is True and self.low == 1 and self.high is None:
                # this is the complement of the epsilon node, do return epsilon.
                return cast(ChildExpr, Empty())
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        arg_hrz = self.arg.horizon()
        if self.high is None:
            return math.inf
        return self.high * arg_hrz


@final
@frozen
class Concat(SEREOp, Generic[ChildExpr]):
    r"""SERE concatenation: ``r1 ; r2 ; ... ; rn``."""

    args: tuple[ChildExpr, ...] = attrs.field(
        validator=attrs.validators.deep_iterable(
            member_validator=_validate_sere_child,
            iterable_validator=attrs.validators.min_len(2),
        )
    )

    @override
    def __str__(self) -> str:
        return "(" + " ; ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield from self.args

    @override
    def expand(self) -> ChildExpr:
        args = (a.expand() for a in self.args)
        # Flatten nested Concat
        args = flatten_nary_args(Concat, args)
        # Empty is the identity of concatenation; drop it. nary_fold
        # collapses to the single survivor, or to Empty() when every
        # operand is absorbed, avoiding the Concat min_len(2) validator.
        args = (a for a in args if not isinstance(a, Empty))
        return cast(ChildExpr, nary_fold(Concat, args, identity=Empty()))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        return sum((a.horizon() for a in self.args), start=0)


@final
@frozen
class Fusion(SEREOp, Generic[ChildExpr]):
    r"""SERE fusion: ``r1 : r2 : ... : rn``."""

    args: tuple[ChildExpr, ...] = attrs.field(
        validator=attrs.validators.deep_iterable(
            member_validator=_validate_sere_child,
            iterable_validator=attrs.validators.min_len(2),
        )
    )

    @override
    def __str__(self) -> str:
        return "(" + " : ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield from self.args

    @override
    def expand(self) -> ChildExpr:
        args = (a.expand() for a in self.args)
        args = flatten_nary_args(Fusion, args)
        # TODO: Maybe handle the Epsilon case?
        return cast(ChildExpr, Fusion(tuple(args)))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        return sum((a.horizon() for a in self.args), start=0)


@final
@frozen
class Alt(SEREOp, Generic[ChildExpr]):
    r"""SERE alternation: ``r1 | r2 | ... | rn``."""

    args: tuple[ChildExpr, ...] = attrs.field(
        validator=attrs.validators.deep_iterable(
            member_validator=_validate_sere_child,
            iterable_validator=attrs.validators.min_len(2),
        )
    )

    @override
    def __str__(self) -> str:
        return "(" + " | ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield from self.args

    @override
    def expand(self) -> ChildExpr:
        args = (a.expand() for a in self.args)
        args = flatten_nary_args(Alt, args)
        return cast(ChildExpr, Alt(tuple(args)))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        return max(a.horizon() for a in self.args)


@final
@frozen
class Inter(SEREOp, Generic[ChildExpr]):
    r"""SERE length-matching intersection: ``r1 && r2 && ... && rn``."""

    args: tuple[ChildExpr, ...] = attrs.field(
        validator=attrs.validators.deep_iterable(
            member_validator=_validate_sere_child,
            iterable_validator=attrs.validators.min_len(2),
        )
    )

    @override
    def __str__(self) -> str:
        return "(" + " && ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield from self.args

    @override
    def expand(self) -> ChildExpr:
        args = (a.expand() for a in self.args)
        args = flatten_nary_args(Inter, args)
        return cast(ChildExpr, Inter(tuple(args)))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        return max(a.horizon() for a in self.args)


@final
@frozen
class NLMInter(SEREOp, Generic[ChildExpr]):
    r"""SERE non-length-matching intersection: ``r1 & r2 & ... & rn``.

    A word ``w`` matches iff one operand matches ``w`` exactly and every
    other operand matches some prefix of ``w``. Equivalently::

        L(r1 & r2) = (L(r1) & L(r2) . Sigma*) | (L(r1) . Sigma* & L(r2))

    Contrast with :class:`Inter` (``&&``), which requires all operands
    to match the same word of the same length.
    """

    args: tuple[ChildExpr, ...] = attrs.field(
        validator=attrs.validators.deep_iterable(
            member_validator=_validate_sere_child,
            iterable_validator=attrs.validators.min_len(2),
        )
    )

    @override
    def __str__(self) -> str:
        return "(" + " & ".join(str(a) for a in self.args) + ")"

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield from self.args

    @override
    def expand(self) -> ChildExpr:
        args = (a.expand() for a in self.args)
        args = flatten_nary_args(NLMInter, args)
        return cast(ChildExpr, NLMInter(tuple(args)))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        return max(a.horizon() for a in self.args)


@final
@frozen
class Complement(SEREOp, Generic[ChildExpr]):
    r"""SERE complement: ``~r``.

    Language: ``Sigma* \ L(r)``. Distinct from Boolean negation on a
    leaf (``!a``), which yields a single-letter language. ``~`` binds
    tighter than every other SERE operator (``[*]``, ``;``, ``:``,
    ``&``, ``&&``, ``|``).

    Extension beyond Spot: Spot's SERE grammar does not include a
    complement operator. See the module docstring.
    """

    arg: ChildExpr = attrs.field(validator=_validate_sere_child)

    @override
    def __invert__(self) -> SEREExpr[typing.Any]:  # type: ignore[override]
        # ~~r = r: double-complement elimination (mirrors Not.__invert__).
        return cast("SEREExpr[typing.Any]", self.arg)

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
    def children(self) -> Iterator[ChildExpr]:
        yield self.arg

    @override
    def expand(self) -> ChildExpr:
        if isinstance(self.arg, Complement):
            # Flatten it
            return cast(ChildExpr, self.arg.arg.expand())
        if isinstance(self.arg, Literal):
            # Push the negation inwards
            return cast(ChildExpr, Literal(not self.arg.value))
        return cast(ChildExpr, Complement(self.arg.expand()))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, self.arg.to_nnf(negate=False, expand=False))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        return math.inf


@final
@frozen
class FirstMatch(SEREOp, Generic[ChildExpr]):
    r"""SERE first-match restriction: ``first_match(r)``.

    Language: words ``w`` in ``L(r)`` whose strictly shorter prefixes
    are all outside ``L(r)``. Matches Spot's ``first_match`` operator.
    """

    arg: ChildExpr = attrs.field(validator=_validate_sere_child)

    @override
    def __str__(self) -> str:
        body = str(self.arg)
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1]
        return f"first_match({body})"

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield self.arg

    @override
    def expand(self) -> ChildExpr:
        return cast(ChildExpr, FirstMatch(self.arg.expand()))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        return self.arg.horizon()


@final
@frozen
class FusionRepeat(SEREOp, Generic[ChildExpr]):
    r"""Fusion-iteration: ``r[:*low..high]``.

    Like ``Repeat`` (`[*]`) but the separator is fusion (`:`) instead of
    concatenation (`;`). Bounded form is syntactic sugar over :class:`Fusion`;
    unbounded ``r[:*i..]`` (`high is None`) is a primitive operator
    (Dax et al.). See :meth:`expand`.

    ``low=None`` is treated as 0; ``high=None`` is unbounded.
    """

    arg: ChildExpr = attrs.field(validator=_validate_sere_child)
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
    def children(self) -> Iterator[ChildExpr]:
        yield self.arg

    @override
    def expand(self) -> ChildExpr:
        e = self.arg.expand()
        lo = _normalize_low(self.low)
        hi = self.high
        if hi is None:
            return cast(ChildExpr, FusionRepeat(e, self.low, None))
        if lo == 0 and hi == 0:
            return cast(ChildExpr, Literal(True))
        if lo == 1 and hi == 1:
            return cast(ChildExpr, e)
        if lo == hi:
            return cast(ChildExpr, Fusion(tuple(e for _ in range(lo))))
        parts: list[Expr] = []
        for k in range(lo, hi + 1):
            if k == 0:
                parts.append(Literal(True))
            elif k == 1:
                parts.append(e)
            else:
                parts.append(Fusion(tuple(e for _ in range(k))))
        return cast(ChildExpr, Alt(tuple(parts)))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        if self.high is None:
            return math.inf
        return self.high * self.arg.horizon()


@final
@frozen
class GotoRepeat(SEREOp, Generic[ChildExpr]):
    r"""Goto-repetition: ``r[->low..high]``.

    Generalized from Spot's Boolean-operand definition to arbitrary SERE
    operand via :class:`Complement`. Extension beyond Spot.

    Semantics (always desugarable)::

        r[->i..j]  ==  (~r[*] ; r)[*i..j]

    ``low=None`` is treated as 0; ``high=None`` is unbounded.
    """

    arg: ChildExpr = attrs.field(validator=_validate_sere_child)
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
    def children(self) -> Iterator[ChildExpr]:
        yield self.arg

    @override
    def expand(self) -> ChildExpr:
        # Preserve operand identity in the expansion: do NOT call .expand()
        # on the outer Repeat, because Concat.expand would flatten a SERE
        # operand ``r`` (which may itself be a Concat) into the surrounding
        # body. Downstream consumers like morphata pattern-match on the
        # expanded shape, so the body's tuple must stay (~r[*], r).
        e = self.arg.expand()
        body = Concat((Repeat(Complement(e), 0, None), e))
        lo = _normalize_low(self.low)
        if lo == 1 and self.high == 1:
            return cast(ChildExpr, body)
        return cast(ChildExpr, Repeat(body, self.low, self.high))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        return math.inf


@final
@frozen
class EqualRepeat(SEREOp, Generic[ChildExpr]):
    r"""Equal-count repetition: ``r[=low..high]``.

    Generalized from Spot's Boolean-operand definition to arbitrary SERE
    operand via :class:`Complement`. Extension beyond Spot.

    Semantics (always desugarable)::

        r[=i..j]  ==  (~r[*] ; r)[*i..j] ; ~r[*]

    ``low=None`` is treated as 0; ``high=None`` is unbounded.
    """

    arg: ChildExpr = attrs.field(validator=_validate_sere_child)
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
    def children(self) -> Iterator[ChildExpr]:
        yield self.arg

    @override
    def expand(self) -> ChildExpr:
        # Preserve operand identity by reusing GotoRepeat.expand (which
        # already avoids the flattening trap) and tacking on the
        # complement-closure tail without re-flattening through
        # Concat.expand.
        goto_part = GotoRepeat(self.arg, self.low, self.high).expand()
        tail = Repeat(Complement(self.arg.expand()), 0, None)
        return cast(ChildExpr, Concat((goto_part, tail)))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            return cast(ChildExpr, Complement(self))
        return cast(ChildExpr, self)

    @override
    def horizon(self) -> int | float:
        return math.inf


Var = TypeVar("Var")
type SEREExpr[Var: Hashable] = (
    Variable[Var]
    | Literal
    | And[SEREExpr[Var]]
    | Or[SEREExpr[Var]]
    | Not[SEREExpr[Var]]
    | Implies[SEREExpr[Var]]
    | Equiv[SEREExpr[Var]]
    | Xor[SEREExpr[Var]]
    | Empty
    | Concat[SEREExpr[Var]]
    | Fusion[SEREExpr[Var]]
    | Alt[SEREExpr[Var]]
    | Inter[SEREExpr[Var]]
    | NLMInter[SEREExpr[Var]]
    | Complement[SEREExpr[Var]]
    | FirstMatch[SEREExpr[Var]]
    | FusionRepeat[SEREExpr[Var]]
    | GotoRepeat[SEREExpr[Var]]
    | EqualRepeat[SEREExpr[Var]]
    | Repeat[SEREExpr[Var]]
)
"""Union of all SERE expression node types."""


def sere_expr_iter(expr: SEREExpr[Var]) -> Iterator[SEREExpr[Var]]:
    """Post-order iterator over a SERE expression, validating dialect membership."""
    return iter(
        ExprVisitor(
            cast(
                list[type[SEREExpr[Var]]],
                [
                    Empty,
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
                ],
            ),
            expr,
        )
    )


__all__ = [
    "SEREExpr",
    "Empty",
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
    "is_sere_node",
]
