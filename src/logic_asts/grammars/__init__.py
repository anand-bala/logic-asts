# mypy: disable-error-code="no-untyped-call"

from __future__ import annotations

import enum
import typing
from pathlib import Path

from lark import Token, Transformer, v_args
from lark.visitors import merge_transformers

from logic_asts.base import Equiv, Expr, Implies, Literal, Variable, Xor
from logic_asts.ltl import Always, Eventually, Next, TimeInterval, Until
from logic_asts.stl_go import EdgeCountInterval, GraphIncoming, GraphOutgoing, Quantifier, WeightInterval
from logic_asts.strel import DistanceInterval, Escape, Everywhere, Reach, Somewhere

GRAMMARS_DIR = Path(__file__).parent


@typing.final
@v_args(inline=True)
class BaseTransform(Transformer[Token, Expr]):
    def mul(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs & rhs

    def add(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs | rhs

    def neg(self, arg: Expr) -> Expr:
        return ~arg

    def xor(self, lhs: Expr, rhs: Expr) -> Expr:
        return Xor(lhs, rhs)

    def equiv(self, lhs: Expr, rhs: Expr) -> Expr:
        return Equiv(lhs, rhs)

    def implies(self, lhs: Expr, rhs: Expr) -> Expr:
        return Implies(lhs, rhs)

    def var(self, value: Token | str) -> Expr:
        return Variable(str(value))

    def literal(self, value: Token | str) -> Expr:
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

    def IDENTIFIER(self, value: Token | str) -> Variable[str]:  # noqa: N802
        return Variable(str(value))


@typing.final
@v_args(inline=True)
class LtlTransform(Transformer[Token, Expr]):
    def mul(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs & rhs

    def until(self, lhs: Expr, interval: TimeInterval | None, rhs: Expr) -> Expr:
        interval = interval or TimeInterval()
        return Until(lhs, rhs, interval)

    def always(self, interval: TimeInterval | None, arg: Expr) -> Expr:
        interval = interval or TimeInterval()
        return Always(arg, interval)

    def eventually(self, interval: TimeInterval | None, arg: Expr) -> Expr:
        interval = interval or TimeInterval()
        return Eventually(arg, interval)

    def next(self, steps: int | None, arg: Expr) -> Expr:
        return Next(arg, steps)

    def time_interval(self, start: int | None, end: int | None) -> TimeInterval:
        return TimeInterval(start, end)

    def INT(self, value: Token | int) -> int:  # noqa: N802
        return int(value)


@typing.final
@v_args(inline=True)
class StrelTransform(Transformer[Token, Expr]):
    def mul(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs & rhs

    def reach(self, lhs: Expr, dist_fn: str | None, interval: DistanceInterval, rhs: Expr) -> Expr:
        return Reach(lhs, rhs, interval, dist_fn)

    def escape(self, dist_fn: str | None, interval: DistanceInterval, arg: Expr) -> Expr:
        return Escape(arg, interval, dist_fn)

    def somewhere(self, dist_fn: str | None, interval: DistanceInterval, arg: Expr) -> Expr:
        return Somewhere(arg, interval, dist_fn)

    def everywhere(self, dist_fn: str | None, interval: DistanceInterval, arg: Expr) -> Expr:
        return Everywhere(arg, interval, dist_fn)

    def dist_interval(self, start: float | None, end: float | None) -> DistanceInterval:
        return DistanceInterval(start, end)

    def dist_fn(self, value: str | Token) -> str:
        return str(value)

    def NUMBER(self, value: Token | float) -> float:  # noqa: N802
        return float(value)


@typing.final
class StlGoTransform(Transformer[Token, Expr]):
    """Transformer for STL-GO grammar, extending LTL transformations."""

    @v_args(inline=True)
    def mul(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs & rhs

    @v_args(inline=True)
    def graph_incoming(
        self,
        weight_interval: WeightInterval,
        quantifier: Quantifier,
        graphs: frozenset[str],
        edge_count: EdgeCountInterval,
        arg: Expr,
    ) -> Expr:
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
        arg: Expr,
    ) -> Expr:
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

        Handles:
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

        Handles:
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


@enum.unique
class SupportedGrammars(enum.Enum):
    BASE = "base"
    """Base Boolean propositional logic, without quantifiers or modal operators

    See:
        `logic_asts.base`
    """

    LTL = "ltl"
    """Linear Temporal Logic

    See:
        `logic_asts.ltl`
    """

    STREL = "strel"
    """Spatio-Temporal Reach Escape Logic

    See:
        `logic_asts.strel`
    """

    STL_GO = "stl_go"
    """Spatio-Temporal Logic with Graph Operators

    See:
        `logic_asts.stl_go`
    """

    def get_transformer(self) -> Transformer[Token, Expr]:
        """
        @private
        """
        syntax = str(self.value)

        transformer: Transformer[Token, Expr]
        match syntax:
            case "base":
                transformer = BaseTransform()
            case "ltl":
                transformer = merge_transformers(
                    LtlTransform(),
                    base=BaseTransform(),
                )
            case "strel":
                transformer = merge_transformers(
                    StrelTransform(),
                    ltl=merge_transformers(
                        LtlTransform(),
                        base=BaseTransform(),
                    ),
                )
            case "stl_go":
                transformer = merge_transformers(
                    StlGoTransform(),
                    ltl=merge_transformers(
                        LtlTransform(),
                        base=BaseTransform(),
                    ),
                )
            case _:
                raise ValueError(f"Unsupported grammar reference: {syntax}")
        return transformer
