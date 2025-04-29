from __future__ import annotations

import shlex

import attrs
from attrs import frozen
from typing_extensions import final, override

from logic_asts.base import Expr
from logic_asts.utils import check_positive, check_start


@final
@frozen
class DistanceInterval:
    start: float | None = attrs.field(default=None, validator=[check_positive, check_start])
    end: float | None = attrs.field(default=None, validator=[check_positive])

    @override
    def __str__(self) -> str:
        match (self.start, self.end):
            case None, None:
                return ""
            case _:
                return f"[{self.start or ''}, {self.end or ''}]"


@final
@frozen
class Everywhere(Expr):
    arg: Expr
    interval: DistanceInterval
    dist_fn: str | None = None

    @override
    def __str__(self) -> str:
        dist_fn = self.dist_fn and f"^{shlex.quote(self.dist_fn)}" or ""
        return f"(everywhere{dist_fn}{self.interval} {self.arg})"

    @override
    def expand(self) -> Expr:
        return Everywhere(self.arg.expand(), self.interval, self.dist_fn)

    @override
    def to_nnf(self) -> Expr:
        return Everywhere(self.arg.to_nnf(), self.interval, self.dist_fn)


@final
@frozen
class Somewhere(Expr):
    arg: Expr
    interval: DistanceInterval
    dist_fn: str | None = None

    @override
    def __str__(self) -> str:
        dist_fn = self.dist_fn and f"^{shlex.quote(self.dist_fn)}" or ""
        return f"(somewhere{dist_fn}{self.interval} {self.arg})"

    @override
    def expand(self) -> Expr:
        return Somewhere(self.arg.expand(), self.interval, self.dist_fn)

    @override
    def to_nnf(self) -> Expr:
        return Somewhere(self.arg.to_nnf(), self.interval, self.dist_fn)


@final
@frozen
class Escape(Expr):
    arg: Expr
    interval: DistanceInterval
    dist_fn: str | None = None

    @override
    def __str__(self) -> str:
        dist_fn = self.dist_fn and f"^{shlex.quote(self.dist_fn)}" or ""
        return f"(escape{dist_fn}{self.interval} {self.arg})"

    @override
    def expand(self) -> Expr:
        return Escape(self.arg.expand(), self.interval, self.dist_fn)

    @override
    def to_nnf(self) -> Expr:
        return Escape(self.arg.to_nnf(), self.interval, self.dist_fn)


@final
@frozen
class Reach(Expr):
    lhs: Expr
    rhs: Expr
    interval: DistanceInterval
    dist_fn: str | None = None

    @override
    def __str__(self) -> str:
        dist_fn = self.dist_fn and f"^{shlex.quote(self.dist_fn)}" or ""
        return f"({self.lhs} reach{dist_fn}{self.interval} {self.rhs})"

    @override
    def expand(self) -> Expr:
        return Reach(
            lhs=self.lhs.expand(),
            rhs=self.rhs.expand(),
            interval=self.interval,
            dist_fn=self.dist_fn,
        )

    @override
    def to_nnf(self) -> Expr:
        return Reach(
            lhs=self.lhs.to_nnf(),
            rhs=self.rhs.to_nnf(),
            interval=self.interval,
            dist_fn=self.dist_fn,
        )
