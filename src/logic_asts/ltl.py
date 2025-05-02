from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import attrs
from attrs import frozen
from typing_extensions import final, override

from logic_asts.base import And, Equiv, Expr, Implies, Literal, Not, Or, Variable, Xor
from logic_asts.utils import check_positive, check_start


@final
@frozen
class TimeInterval:
    start: None | int = attrs.field(default=None, validator=[check_positive, check_start])
    end: int | None = attrs.field(default=None, validator=[check_positive])

    @override
    def __str__(self) -> str:
        match (self.start, self.end):
            case None, None:
                return ""
            case _:
                return f"[{self.start or ''}, {self.end or ''}]"

    def is_unbounded(self) -> bool:
        return self.end is None or math.isinf(self.end)

    def is_untimed(self) -> bool:
        """If the interval is [0, inf]"""
        return (self.start is None or self.start == 0.0) and (self.end is None or math.isinf(self.end))

    def iter_interval(self, *, step: float | int = 1) -> Iterator[float | int]:
        """Return an iterator over the discrete (determined by `step`) range of the time interval

        !!! note

            If the time interval is unbounded, this will return a generator that goes on forever
        """

        def _bounded_iter_with_float(start: float | int, stop: float | int, step: float | int) -> Iterator[float | int]:
            pos = start
            while pos < stop:
                yield start
                pos += step
            return

        start = self.start or 0.0
        end = self.end or math.inf

        if math.isinf(end):
            # Unbounded iteration
            yield from itertools.count(start, step=step)
        else:
            # Bounded iter
            yield from _bounded_iter_with_float(start, end, step)


@final
@frozen
class Next(Expr):
    arg: Expr
    steps: int | None = attrs.field(default=None)

    @override
    def __str__(self) -> str:
        match self.steps:
            case None | 1:
                step_str = ""
            case t:
                step_str = f"[{t}]"
        return f"(X{step_str} {self.arg})"

    @override
    def expand(self) -> Expr:
        arg = self.arg.expand()
        match self.steps:
            case None:
                return Next(arg)
            case t:
                expr = arg
                for _ in range(t):
                    expr = Next(expr)
                return expr

    @override
    def to_nnf(self) -> Expr:
        return Next(self.arg.to_nnf(), steps=self.steps)


@final
@frozen
class Always(Expr):
    arg: Expr
    interval: TimeInterval | None = None

    @override
    def __str__(self) -> str:
        return f"(G{self.interval or ''} {self.arg})"

    @override
    def expand(self) -> Expr:
        return ~Eventually(~self.arg, self.interval)

    @override
    def to_nnf(self) -> Expr:
        return self.expand().to_nnf()


@final
@frozen
class Eventually(Expr):
    arg: Expr
    interval: TimeInterval | None = None

    @override
    def __str__(self) -> str:
        return f"(F{self.interval or ''} {self.arg})"

    @override
    def expand(self) -> Expr:
        match self.interval:
            case None | TimeInterval(None, None) | TimeInterval(0, None):
                # Unbounded F
                return Eventually(self.arg.expand())
            case TimeInterval(0, int(t2)) | TimeInterval(None, int(t2)):
                # F[0, t2]
                arg = self.arg.expand()
                expr = arg
                for _ in range(t2):
                    expr = expr & Next(arg)
                return expr
            case TimeInterval(int(t1), None):
                # F[t1,inf]
                assert t1 > 0
                return Next(Eventually(self.arg), t1).expand()
            case TimeInterval(int(t1), int(t2)):
                # F[t1, t2]
                assert t1 > 0
                # F[t1, t2] = X[t1] F[0,t2-t1] arg
                # Nested nexts until t1
                return Next(Eventually(self.arg, TimeInterval(0, t2 - t1)), t1).expand()
            case TimeInterval():
                raise RuntimeError(f"Unexpected time interval {self.interval}")

    @override
    def to_nnf(self) -> Expr:
        return Eventually(self.arg.to_nnf(), self.interval)


@final
@frozen
class Until(Expr):
    lhs: Expr
    rhs: Expr
    interval: TimeInterval | None = None

    @override
    def __str__(self) -> str:
        return f"({self.lhs} U{self.interval or ''} {self.rhs})"

    @override
    def to_nnf(self) -> Expr:
        return Until(self.lhs.to_nnf(), self.rhs.to_nnf(), interval=self.interval)

    @override
    def expand(self) -> Expr:
        new_lhs = self.lhs.expand()
        new_rhs = self.rhs.expand()
        match self.interval:
            case None | TimeInterval(None | 0, None):
                # Just make an unbounded one here
                return Until(new_lhs, new_rhs)
            case TimeInterval(t1, None):  # Unbounded end
                return Always(
                    arg=Until(interval=None, lhs=new_lhs, rhs=new_rhs),
                    interval=TimeInterval(0, t1),
                ).expand()
            case TimeInterval(t1, _):
                z1 = Eventually(interval=self.interval, arg=new_lhs).expand()
                until_interval = TimeInterval(t1, None)
                z2 = Until(interval=until_interval, lhs=new_lhs, rhs=new_rhs).expand()
                return z1 & z2


__all__ = [
    "Expr",
    "Implies",
    "Equiv",
    "Xor",
    "And",
    "Or",
    "Not",
    "Variable",
    "Literal",
    "TimeInterval",
    "Next",
    "Always",
    "Eventually",
    "Until",
]
