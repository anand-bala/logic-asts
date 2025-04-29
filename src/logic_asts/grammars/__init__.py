from __future__ import annotations

import typing
from pathlib import Path

from lark import Token, Transformer, v_args

from logic_asts.base import Equiv, Expr, Implies, Literal, Variable, Xor
from logic_asts.ltl import Always, Eventually, Next, TimeInterval, Until
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

    def literal(self, value: typing.Literal["0", "1", "TRUE", "FALSE"]) -> Expr:
        match value:
            case "0" | "FALSE":
                return Literal(False)
            case "1" | "TRUE":
                return Literal(True)

    def CNAME(self, value: Token | str) -> str:  # noqa: N802
        return str(value)

    def ESCAPED_STRING(self, value: Token | str) -> str:  # noqa: N802
        parsed = str(value)
        # trim the quotes at the end
        return parsed[1:-1]

    def identifier(self, value: Token | str) -> str:
        return str(value)


@typing.final
@v_args(inline=True)
class LtlTransform(Transformer[Token, Expr]):
    def mul(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs & rhs

    def until(self, lhs: Expr, interval: TimeInterval | None, rhs: Expr) -> Expr:
        return Until(lhs, rhs, interval)

    def always(self, interval: TimeInterval | None, arg: Expr) -> Expr:
        return Always(arg, interval)

    def eventually(self, interval: TimeInterval | None, arg: Expr) -> Expr:
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
