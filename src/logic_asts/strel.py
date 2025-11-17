r"""Abstract syntax trees for spatio-temporal reach-escape logic.

This module extends linear temporal logic with spatial operators for specifying
properties about multi-agent systems and spatially distributed environments.
Spatio-Temporal Reach-Escape Logic (STREL) combines temporal quantifiers
(from LTL) with spatial distance constraints.

Core Spatial Operators:
    - Everywhere: $\text{everywhere}^d \phi$ - property holds within distance d
    - Somewhere: $\text{somewhere}^d \phi$ - property holds somewhere within distance d
    - Reach: $\phi \leadsto^d \psi$ - from phi, can reach psi within distance d
    - Escape: $\text{escape}^d \phi$ - can escape region where phi holds

Space and Distance:
    - Distance intervals [start, end] constrain spatial neighborhoods.
    - Distance functions (hops, euclidean, etc.) determine neighborhood metrics.

Key Classes:
    - DistanceInterval: Represents distance bounds [start, end]
    - Everywhere: Universal spatial operator
    - Somewhere: Existential spatial operator
    - Reach: Binary spatial reachability operator
    - Escape: Spatial escape from a region

Integration with Temporal Logic:
    - Spatial operators can be nested with LTL operators (X, F, G, U) to
    - create spatio-temporal formulas combining time and space constraints.

Examples:
    - Safety: G everywhere[0,5] ~obstacle
    - Reachability: F somewhere[0,10] goal
    - Coordination: G (agent1 reach[0,20] agent2)
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator
from typing import TypeAlias, TypeVar, final

import attrs
from attrs import frozen
from typing_extensions import override

from logic_asts.base import Expr
from logic_asts.ltl import LTLExpr
from logic_asts.utils import check_positive, check_start


@final
@frozen
class DistanceInterval:
    r"""Distance constraint for spatial operators: interval $[d_1, d_2]$.

    Represents a distance interval for constraining spatial neighborhoods.
    The interval is closed on both ends. None represents unboundedness at
    that end. Distances are continuous (float) values.

    Attributes:
        start: Lower bound on distance (inclusive), or None for no lower bound.
            Defaults to 0 when None.
        end: Upper bound on distance (inclusive), or None for unbounded.

    Examples:
        - Bounded: `DistanceInterval(0, 10)`     represents $[0,10]$
        - Left unbounded: `DistanceInterval(None, 20)`  represents $[0,20]$
        - Right unbounded: `DistanceInterval(5.5, None)`  represents $[5.5,\infty)$
        - Fully unbounded: `DistanceInterval(None, None)`  represents $[0,\infty)$

    Validators:
        - start and end must be non-negative
        - start must be <= end if both are non-None
        - No point intervals (start == end not allowed)
    """

    start: float | None = attrs.field(default=None, validator=[check_positive, check_start])
    end: float | None = attrs.field(default=None, validator=[check_positive])

    @override
    def __str__(self) -> str:
        match (self.start, self.end):
            case None, None:
                return ""
            case _:
                start_str = str(self.start) if self.start is not None else ""
                end_str = str(self.end) if self.end is not None else ""
                return f"[{start_str}, {end_str}]"


@final
@frozen
class Everywhere(Expr):
    r"""Universal spatial operator: $\square^d \phi$.

    Asserts that the formula holds everywhere within the distance interval.
    The formula $\square^d \phi$ holds if $\phi$ holds at all points within
    distance $d$ from the reference point.

    Attributes:
        arg: The sub-formula that must hold everywhere.
        interval: Distance constraint [start, end] for the neighborhood.
        dist_fn: Optional distance metric function name (e.g., 'hops',
            'euclidean'). None uses default metric.

    Examples:
        - No obstacles nearby: everywhere[0,5] ~obstacle
        - With distance metric: everywhere^hops[0,3] safe
        - Nested: G everywhere[0,10] property
    """

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

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        return self.arg.horizon()


@final
@frozen
class Somewhere(Expr):
    r"""Existential spatial operator: $\diamond^d \phi$.

    Asserts that the formula holds somewhere within the distance interval.
    The formula $\diamond^d \phi$ holds if $\phi$ holds at some point within
    distance $d$ from the reference point.

    Attributes:
        arg: The sub-formula that must hold somewhere.
        interval: Distance constraint [start, end] for the neighborhood.
        dist_fn: Optional distance metric function name (e.g., 'hops',
            'euclidean'). None uses default metric.

    Examples:
        - Goal exists nearby: somewhere[0,20] goal
        - With metric: somewhere^euclidean[0,5.5] target
        - Temporal-spatial: F somewhere[0,10] ally
    """

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

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        return self.arg.horizon()


@final
@frozen
class Escape(Expr):
    r"""Escape operator: escape from a region.

    Asserts that the system can escape from the region where the formula holds,
    within a given distance. This represents the ability to move away from
    states satisfying the formula.

    Attributes:
        arg: The sub-formula defining the region to escape from.
        interval: Distance constraint [start, end] for the escape neighborhood.
        dist_fn: Optional distance metric function name. None uses default.

    Examples:
        - Escape from danger zone: escape[0,10] danger
        - With metric: escape^hops[0,5] blocked_region
    """

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

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        return self.arg.horizon()


@final
@frozen
class Reach(Expr):
    r"""Reachability operator: $\phi \leadsto^d \psi$.

    Binary spatial operator asserting reachability. The formula $\phi \leadsto^d \psi$
    holds if, starting from a state where $\phi$ is true, one can reach a state
    within distance $d$ where $\psi$ is true.

    Attributes:
        lhs: The starting condition formula ($\phi$).
        rhs: The target condition formula ($\psi$).
        interval: Distance constraint [start, end] for the reachability.
        dist_fn: Optional distance metric function name. None uses default.

    Examples:
        - Reach goal from start: start reach[0,50] goal
        - Bounded reachability: position_a reach^hops[0,10] position_b
        - Spatio-temporal: (F location1) reach[0,30] (F location2)
    """

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

    @override
    def children(self) -> Iterator[Expr]:
        yield self.lhs
        yield self.rhs

    @override
    def horizon(self) -> int | float:
        return max(self.lhs.horizon(), self.rhs.horizon())


Var = TypeVar("Var")
STRELExpr: TypeAlias = LTLExpr[Var] | Everywhere | Somewhere | Reach | Escape
"""STREL Expression Types"""

__all__ = [
    "STRELExpr",
    "DistanceInterval",
    "Everywhere",
    "Somewhere",
    "Escape",
    "Reach",
]
