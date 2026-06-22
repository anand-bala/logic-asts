# mypy: disable-error-code="no-untyped-call"

from __future__ import annotations

import enum
import typing
from pathlib import Path

from lark import Token, Transformer, v_args
from lark.visitors import merge_transformers

from logic_asts.base import And, BoolExpr, Equiv, Implies, Literal, Not, Or, Variable, Xor
from logic_asts.ltl import (
    Always,
    Eventually,
    LTLExpr,
    Next,
    Release,
    StrongNext,
    StrongRelease,
    TimeInterval,
    Until,
    WeakUntil,
)
from logic_asts.psl import PSLFormula, StrongClosure, SuffixImpliesExist, SuffixImpliesUniv, WeakClosure
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
from logic_asts.spec import Expr
from logic_asts.stl_go import EdgeCountInterval, GraphIncoming, GraphOutgoing, Quantifier, STLGOExpr, WeightInterval
from logic_asts.strel import DistanceInterval, Escape, Everywhere, Reach, Somewhere, STRELExpr
from logic_asts.utils import nary_fold

GRAMMARS_DIR = Path(__file__).parent


def _unquote_identifier(text: str) -> str:
    """Strip the outer quotes and undo grammar-level escapes for ``ESCAPED_STRING``.

    The base/ltl grammars wrap escaped identifiers in ``"..."``; strel/stl_go
    also accept ``'...'``. Inside the quotes only ``\\\\`` and the matching
    quote character are escaped.
    """
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        inner = text[1:-1]
        out: list[str] = []
        i = 0
        while i < len(inner):
            ch = inner[i]
            if ch == "\\" and i + 1 < len(inner):
                out.append(inner[i + 1])
                i += 2
            else:
                out.append(ch)
                i += 1
        return "".join(out)
    return text


class MaybeStrongStep(typing.NamedTuple):
    steps: int | None
    strong: bool


class MaybeStrongInterval(typing.NamedTuple):
    interval: TimeInterval
    strong: bool


@typing.final
@v_args(inline=True)
class BaseTransform(Transformer[Token, BoolExpr[str]]):
    def mul(self, lhs: BoolExpr[str], rhs: BoolExpr[str]) -> BoolExpr[str]:
        # ``&`` is a pure constructor now; flatten at construction via nary_fold
        # so ``a & b & c`` stays a flat n-ary And.
        return typing.cast(BoolExpr[str], nary_fold(And, (lhs, rhs)))

    def add(self, lhs: BoolExpr[str], rhs: BoolExpr[str]) -> BoolExpr[str]:
        return typing.cast(BoolExpr[str], nary_fold(Or, (lhs, rhs)))

    def neg(self, arg: BoolExpr[str]) -> BoolExpr[str]:
        return Not(arg)

    def xor(self, lhs: BoolExpr[str], rhs: BoolExpr[str]) -> BoolExpr[str]:
        return Xor(lhs, rhs)

    def equiv(self, lhs: BoolExpr[str], rhs: BoolExpr[str]) -> BoolExpr[str]:
        return Equiv(lhs, rhs)

    def implies(self, lhs: BoolExpr[str], rhs: BoolExpr[str]) -> BoolExpr[str]:
        return Implies(lhs, rhs)

    def var(self, value: Token | str) -> BoolExpr[str]:
        return Variable(str(value))

    def literal(self, value: Token | str) -> BoolExpr[str]:
        value = str(value)
        match value:
            case "0" | "FALSE":
                return Literal(False)
            case "1" | "TRUE":
                return Literal(True)
            case _:
                raise RuntimeError(f"unknown literal string: {value}")

    def CNAME(self, value: Token | str) -> str:  # noqa: N802
        return str(value)

    def ESCAPED_STRING(self, value: Token | str) -> str:  # noqa: N802
        parsed = str(value)
        # trim the quotes at the end
        return parsed[1:-1]

    def TRUE(self, _value: Token | str) -> Literal:  # noqa: N802
        return Literal(True)

    def FALSE(self, _value: Token | str) -> Literal:  # noqa: N802
        return Literal(False)

    def start(self, expr: BoolExpr[str]) -> BoolExpr[str]:
        return expr

    def IDENTIFIER(self, value: Token | str) -> Variable[str]:  # noqa: N802
        return Variable(_unquote_identifier(str(value)))


@typing.final
@v_args(inline=True)
class LtlTransform(Transformer[Token, LTLExpr[str]]):
    def start(self, expr: LTLExpr[str]) -> LTLExpr[str]:
        return expr

    def mul(self, lhs: LTLExpr[str], rhs: LTLExpr[str]) -> LTLExpr[str]:
        return typing.cast(LTLExpr[str], nary_fold(And, (lhs, rhs)))

    def until(self, lhs: LTLExpr[str], interval: TimeInterval | None, rhs: LTLExpr[str]) -> LTLExpr[str]:
        interval = interval or TimeInterval()
        return Until(lhs, rhs, interval)

    def weak_until(self, lhs: LTLExpr[str], interval: TimeInterval | None, rhs: LTLExpr[str]) -> LTLExpr[str]:
        interval = interval or TimeInterval()
        return WeakUntil(lhs, rhs, interval)

    def release(self, lhs: LTLExpr[str], interval: TimeInterval | None, rhs: LTLExpr[str]) -> LTLExpr[str]:
        interval = interval or TimeInterval()
        return Release(lhs, rhs, interval)

    def strong_release(self, lhs: LTLExpr[str], interval: TimeInterval | None, rhs: LTLExpr[str]) -> LTLExpr[str]:
        interval = interval or TimeInterval()
        return StrongRelease(lhs, rhs, interval)

    def always(self, interval: TimeInterval | MaybeStrongInterval | None, arg: LTLExpr[str]) -> LTLExpr[str]:
        # interval can be:
        # - None (no interval)
        # - TimeInterval (regular time interval)
        # - tuple (TimeInterval, strong) from time_interval_strong
        strong = False
        if isinstance(interval, tuple):
            interval, strong = interval
        elif interval is None:
            interval = TimeInterval()
        return Always(arg, interval, strong=strong)

    def eventually(self, interval: TimeInterval | MaybeStrongInterval | None, arg: LTLExpr[str]) -> LTLExpr[str]:
        # interval can be:
        # - None (no interval)
        # - TimeInterval (regular time interval)
        # - tuple (TimeInterval, strong) from time_interval_strong
        strong = False
        if isinstance(interval, tuple):
            interval, strong = interval
        elif interval is None:
            interval = TimeInterval()
        return Eventually(arg, interval, strong=strong)

    def next(self, modifier: MaybeStrongStep | None, arg: LTLExpr[str]) -> LTLExpr[str]:
        if modifier is None:
            # weak_next: X p
            return Next(arg, None)
        else:
            if modifier.strong:
                # strong_next: X[!] p
                steps = modifier.steps
                return StrongNext(arg, steps)
            else:
                # weak_step: X[n] p
                steps = modifier.steps
                return Next(arg, steps)

    def strong_next(self) -> MaybeStrongStep:
        # X[!] -> strong next, no steps
        return MaybeStrongStep(None, True)

    def weak_step(self, steps: int) -> MaybeStrongStep:
        # X[n] -> weak next with steps
        return MaybeStrongStep(steps, False)

    def strong_step(self, steps: int) -> MaybeStrongStep:
        # X[n!] -> strong next with steps
        return MaybeStrongStep(steps, True)

    def weak_next(self) -> None:
        # X -> weak next with no steps
        return None

    def time_interval(self, start: int | None, end: int | None) -> TimeInterval:
        return TimeInterval(start, end)

    def time_interval_strong(self, start: int | None, end: int | None) -> MaybeStrongInterval:
        return MaybeStrongInterval(TimeInterval(start, end), True)

    def INT(self, value: Token | int) -> int:  # noqa: N802
        return int(value)


@typing.final
@v_args(inline=True)
class SereTransform(Transformer[Token, SEREExpr[str]]):
    def start(self, expr: SEREExpr[str]) -> SEREExpr[str]:
        return expr

    def bool_atom(self, expr: SEREExpr[str]) -> SEREExpr[str]:
        return expr

    @v_args(inline=False)
    def alt(self, args: list[SEREExpr[str]]) -> SEREExpr[str]:
        return Alt(tuple(args))

    @v_args(inline=False)
    def inter(self, args: list[SEREExpr[str]]) -> SEREExpr[str]:
        return Inter(tuple(args))

    @v_args(inline=False)
    def nlm_inter(self, args: list[SEREExpr[str]]) -> SEREExpr[str]:
        return NLMInter(tuple(args))

    def complement(self, arg: SEREExpr[str]) -> SEREExpr[str]:
        return Complement(arg)

    def first_match(self, arg: SEREExpr[str]) -> SEREExpr[str]:
        return FirstMatch(arg)

    @v_args(inline=False)
    def concat(self, args: list[SEREExpr[str]]) -> SEREExpr[str]:
        return Concat(tuple(args))

    @v_args(inline=False)
    def fusion(self, args: list[SEREExpr[str]]) -> SEREExpr[str]:
        return Fusion(tuple(args))

    def repeat(self, arg: SEREExpr[str], suffix: tuple[int | None, int | None]) -> SEREExpr[str]:
        low, high = suffix
        return Repeat(arg, low, high)

    def fusion_repeat(self, arg: SEREExpr[str], suffix: tuple[int | None, int | None]) -> SEREExpr[str]:
        low, high = suffix
        return FusionRepeat(arg, low, high)

    def goto_repeat(self, arg: SEREExpr[str], suffix: tuple[int | None, int | None]) -> SEREExpr[str]:
        low, high = suffix
        return GotoRepeat(arg, low, high)

    def equal_repeat(self, arg: SEREExpr[str], suffix: tuple[int | None, int | None]) -> SEREExpr[str]:
        low, high = suffix
        return EqualRepeat(arg, low, high)

    def fusion_star_unbounded(self) -> tuple[int | None, int | None]:
        return (0, None)

    def fusion_plus_unbounded(self) -> tuple[int | None, int | None]:
        return (1, None)

    def goto_unbounded_default(self) -> tuple[int | None, int | None]:
        # Spot shorthand: [->] == [->1..1]
        return (1, 1)

    def equal_unbounded_default(self) -> tuple[int | None, int | None]:
        # Spot shorthand: [=] == [=0..]
        return (0, None)

    def star_unbounded(self) -> tuple[int | None, int | None]:
        return (0, None)

    def plus_unbounded(self) -> tuple[int | None, int | None]:
        return (1, None)

    def bare_star(self) -> tuple[int | None, int | None]:
        return (0, None)

    def bare_plus(self) -> tuple[int | None, int | None]:
        return (1, None)

    def star_range(self, rng: tuple[int | None, int | None]) -> tuple[int | None, int | None]:
        return rng

    def range_point(self, n: int) -> tuple[int | None, int | None]:
        return (n, n)

    def range_closed(self, lo: int, hi: int) -> tuple[int | None, int | None]:
        return (lo, hi)

    def range_open_high(self, lo: int) -> tuple[int | None, int | None]:
        return (lo, None)

    def range_open_low(self, hi: int) -> tuple[int | None, int | None]:
        return (0, hi)

    def INT(self, value: Token | int) -> int:  # noqa: N802
        return int(value)


@typing.final
@v_args(inline=True)
class StrelTransform(Transformer[Token, STRELExpr[str]]):
    def start(self, expr: STRELExpr[str]) -> STRELExpr[str]:
        return expr

    def mul(self, lhs: STRELExpr[str], rhs: STRELExpr[str]) -> STRELExpr[str]:
        return typing.cast(STRELExpr[str], nary_fold(And, (lhs, rhs)))

    def reach(
        self, lhs: STRELExpr[str], dist_fn: str | None, interval: DistanceInterval, rhs: STRELExpr[str]
    ) -> STRELExpr[str]:
        return Reach(lhs, rhs, interval, dist_fn)

    def escape(self, dist_fn: str | None, interval: DistanceInterval, arg: STRELExpr[str]) -> STRELExpr[str]:
        return Escape(arg, interval, dist_fn)

    def somewhere(self, dist_fn: str | None, interval: DistanceInterval, arg: STRELExpr[str]) -> STRELExpr[str]:
        return Somewhere(arg, interval, dist_fn)

    def everywhere(self, dist_fn: str | None, interval: DistanceInterval, arg: STRELExpr[str]) -> STRELExpr[str]:
        return Everywhere(arg, interval, dist_fn)

    def dist_interval(self, start: float | None, end: float | None) -> DistanceInterval:
        return DistanceInterval(start, end)

    def dist_fn(self, value: str | Token) -> str:
        return str(value)

    def NUMBER(self, value: Token | float) -> float:  # noqa: N802
        return float(value)


@typing.final
class StlGoTransform(Transformer[Token, STLGOExpr[str]]):
    """Transformer for STL-GO grammar, extending LTL transformations."""

    @v_args(inline=True)
    def start(self, expr: STLGOExpr[str]) -> STLGOExpr[str]:
        return expr

    @v_args(inline=True)
    def mul(self, lhs: STLGOExpr[str], rhs: STLGOExpr[str]) -> STLGOExpr[str]:
        return typing.cast(STLGOExpr[str], nary_fold(And, (lhs, rhs)))

    @v_args(inline=True)
    def graph_incoming(
        self,
        weight_interval: WeightInterval,
        quantifier: Quantifier,
        graphs: frozenset[str],
        edge_count: EdgeCountInterval,
        arg: STLGOExpr[str],
    ) -> STLGOExpr[str]:
        return GraphIncoming(
            arg=arg,
            graphs=graphs,
            edge_count=edge_count,
            weights=weight_interval,
            quantifier=quantifier,
        )

    @v_args(inline=True)
    def graph_outgoing(
        self,
        weight_interval: WeightInterval,
        quantifier: Quantifier,
        graphs: frozenset[str],
        edge_count: EdgeCountInterval,
        arg: STLGOExpr[str],
    ) -> STLGOExpr[str]:
        return GraphOutgoing(
            arg=arg,
            graphs=graphs,
            edge_count=edge_count,
            weights=weight_interval,
            quantifier=quantifier,
        )

    @v_args(inline=True)
    def weight_interval(self, start: float | None, end: float | None) -> WeightInterval:
        """Parse weight interval.

        Note:
            - [None, None]: unbounded interval
            - [n1, n2]: bounded interval
            - None values are converted to actual infinities by WeightInterval
        """
        return WeightInterval(start, end)

    @v_args(inline=True)
    def weight_bound(self, value: float) -> float | None:
        return float(value)

    @v_args(inline=True)
    def edge_count_interval(self, start: int | None, end: int | None) -> EdgeCountInterval:
        """Parse edge count interval.

        Note:
            - [None, None]: unbounded interval
            - [n1, n2]: bounded interval
        """
        return EdgeCountInterval(start, end)

    def graph_list(self, graph_types: str | list[str]) -> frozenset[str]:
        # graph_types can be a list or individual items depending on grammar
        if isinstance(graph_types, list):
            return frozenset(graph_types)
        return frozenset([graph_types]) if graph_types else frozenset()

    @v_args(inline=True)
    def graph_type(self, identifier: str) -> str:
        """Pass through graph type identifier."""
        return identifier

    @v_args(inline=False)
    def exists_q(self, _: Token) -> Quantifier:
        """Quantifier: exists"""
        return Quantifier.EXISTS

    @v_args(inline=False)
    def forall_q(self, _: Token) -> Quantifier:
        """Quantifier: forall"""
        return Quantifier.FORALL

    def IDENTIFIER(self, value: Token | str) -> str:  # noqa: N802
        """Convert identifier token to string."""
        return str(value)

    def NUMBER(self, value: Token | float) -> float:  # noqa: N802
        """Convert NUMBER token to float."""
        return float(value)

    def INF(self, value: Token | float) -> float:  # noqa: N802
        return float(value)

    def NEG_INF(self, value: Token | float) -> float:  # noqa: N802
        return float(value)

    def INT(self, value: Token | int) -> int:  # noqa: N802
        """Convert INT token to int."""
        return int(value)


@typing.final
@v_args(inline=True)
class PslTransform(Transformer[Token, PSLFormula[str]]):
    def start(self, expr: PSLFormula[str]) -> PSLFormula[str]:
        return expr

    def suffix_implies_univ(self, sere: SEREExpr[str], formula: PSLFormula[str]) -> PSLFormula[str]:
        return SuffixImpliesUniv(sere, formula)

    def suffix_implies_exist(self, sere: SEREExpr[str], formula: PSLFormula[str]) -> PSLFormula[str]:
        return SuffixImpliesExist(sere, formula)

    def suffix_implies_univ_then(self, sere: SEREExpr[str], formula: PSLFormula[str]) -> PSLFormula[str]:
        # {r}[]=> f  ==  {r ; 1}[]-> f
        return SuffixImpliesUniv(Concat[SEREExpr[str]]((sere, Literal(True))), formula)

    def suffix_implies_exist_then(self, sere: SEREExpr[str], formula: PSLFormula[str]) -> PSLFormula[str]:
        # {r}<>=> f  ==  {r ; 1}<>-> f
        return SuffixImpliesExist(Concat[SEREExpr[str]]((sere, Literal(True))), formula)

    def weak_closure(self, sere: SEREExpr[str]) -> PSLFormula[str]:
        return WeakClosure(sere)

    def strong_closure(self, sere: SEREExpr[str]) -> PSLFormula[str]:
        return StrongClosure(sere)


@enum.unique
class SupportedGrammars(enum.Enum):
    BASE = "base"
    """Base Boolean propositional logic, without quantifiers or modal operators.

    .. seealso:: :mod:`logic_asts.base`
    """

    LTL = "ltl"
    """Linear Temporal Logic.

    .. seealso:: :mod:`logic_asts.ltl`
    """

    STREL = "strel"
    """Spatio-Temporal Reach Escape Logic.

    .. seealso:: :mod:`logic_asts.strel`
    """

    STL_GO = "stl_go"
    """Spatio-Temporal Logic with Graph Operators.

    .. seealso:: :mod:`logic_asts.stl_go`
    """

    SERE = "sere"
    """Sequential Extended Regular Expressions.

    .. seealso:: :mod:`logic_asts.sere`
    """

    PSL = "psl"
    """Property Specification Logic.

    .. seealso:: :mod:`logic_asts.psl`
    """

    def get_transformer(self) -> Transformer[Token, Expr]:
        """Return the Lark transformer for this grammar."""
        syntax = str(self.value)

        match syntax:
            case "base":
                return typing.cast(Transformer[Token, Expr], BaseTransform())
            case "ltl":
                return merge_transformers(
                    LtlTransform(),
                    base=BaseTransform(),
                )
            case "strel":
                return merge_transformers(
                    StrelTransform(),
                    ltl=merge_transformers(
                        LtlTransform(),
                        base=BaseTransform(),
                    ),
                )
            case "stl_go":
                return merge_transformers(
                    StlGoTransform(),
                    ltl=merge_transformers(
                        LtlTransform(),
                        base=BaseTransform(),
                    ),
                )
            case "sere":
                return merge_transformers(
                    SereTransform(),
                    base=BaseTransform(),
                )
            case "psl":
                return merge_transformers(
                    PslTransform(),
                    ltl=merge_transformers(
                        LtlTransform(),
                        base=BaseTransform(),
                    ),
                    sere=merge_transformers(
                        SereTransform(),
                        base=BaseTransform(),
                    ),
                )
            case _:
                raise ValueError(f"Unsupported grammar reference: {syntax}")
