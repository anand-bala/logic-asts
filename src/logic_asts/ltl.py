r"""Abstract syntax trees for linear temporal logic (LTL).

This module extends propositional logic with temporal operators for specifying
LTL properties:
    - X (Next): $X\phi$ asserts that $\phi$ holds at the next time step
    - F (Eventually): $F\phi$ asserts that $\phi$ holds at some future time
    - G (Always): $G\phi$ asserts that $\phi$ holds at all future times
    - U (Until): $\phi U \psi$ asserts $\phi$ holds until $\psi$ becomes true

Time Constraints:
    Operators can be constrained with time intervals [start, end]:
    - $F_{[0,10]}\phi$: phi holds within the next 10 time steps
    - $G_{[5,\infty)}\phi$: phi always holds from time 5 onward

Key Classes:
    - TimeInterval: Represents time bounds [start, end]
    - Next: Single and multi-step next operator
    - Eventually: Existential temporal operator
    - Always: Universal temporal operator
    - Until: Binary temporal operator

Examples:
    Request-response property: `request -> F response`
    >>> from logic_asts.base import Variable, Implies
    >>> request = Variable("request")
    >>> response = Variable("response")
    >>> print(Implies(request, Eventually(response)))
    request -> (F response)

    Safety property: `G ~error`
    >>> error = Variable("error")
    >>> print(Always(~error))
    (G !error)

    Liveness property: `G F (process_ready)`
    >>> process_ready = Variable("process_ready")
    >>> print(Always(Eventually(process_ready)))
    (G (F process_ready))
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from typing import TypeAlias, TypeVar, final

import attrs
from attrs import frozen
from typing_extensions import override

from logic_asts.base import BaseExpr, Expr
from logic_asts.utils import check_positive, check_start


@final
@frozen
class TimeInterval:
    r"""Time constraint for temporal operators: interval $[a,b]$.

    Represents a time interval for constraining when temporal properties must
    hold. The interval is closed on both ends. None represents unboundedness
    at that end.

    Attributes:
        start: Lower bound (inclusive), or None for no lower bound. Defaults to 0
            when None and used with duration().
        end: Upper bound (inclusive), or None for unbounded.

    Examples:
        - Bounded: `TimeInterval(0, 10)`    represents $[0,10]$
        - Left unbounded: `TimeInterval(None, 20)`  represents $[0,20]$
        - Right unbounded: `TimeInterval(5, None)`  represents $[5,\infty)$
        - Fully unbounded: `TimeInterval(None, None)`  represents $[0,\infty)$

    Validators:
        - start and end must be non-negative
        - start must be <= end
        - No point intervals (start == end not allowed if both are non-None)
    """

    start: int | None = attrs.field(default=None, validator=[check_positive, check_start])
    end: int | None = attrs.field(default=None, validator=[check_positive])

    @override
    def __str__(self) -> str:
        match (self.start, self.end):
            case None, None:
                return ""
            case _:
                start_str = str(self.start) if self.start is not None else ""
                end_str = str(self.end) if self.end is not None else ""
                return f"[{start_str}, {end_str}]"

    def duration(self) -> int | float:
        r"""Calculate the duration of the interval.

        Computes the length of the interval, treating None as:
        - start = None as 0
        - end = None as infinity

        Returns:
            The length $b - a$ where $a$ is start and $b$ is end.
        """
        start = self.start or 0
        end = self.end or math.inf

        return end - start

    def is_unbounded(self) -> bool:
        r"""Check if the interval has no upper bound.

        Returns:
            True if end is None or infinity.
        """
        return self.end is None or math.isinf(self.end)

    def is_untimed(self) -> bool:
        r"""Check if the interval represents the unbounded future $[0, \infty)$.

        Returns:
            True if this is effectively [0, infinity), False otherwise.
        """
        return (self.start is None or self.start == 0.0) and (self.end is None or math.isinf(self.end))

    def iter_interval(self, *, step: float | int = 1) -> Iterator[float | int]:
        r"""Generate time points in the interval at fixed step sizes.

        Yields discrete time points from start to end with given step size.
        For unbounded intervals, yields infinitely many points.

        Arguments:
            step: Time increment between consecutive points. Defaults to 1.

        Yields:
            Time points in [start, end] at intervals of step.

        Warning:
            For unbounded intervals (end is None), this generates infinitely
            many values and will never terminate.
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
    r"""Next operator: $X\phi$ or $X^n\phi$.

    Asserts that the formula holds in the next time step(s). A formula $X\phi$
    holds at time $t$ if $\phi$ holds at time $t+1$.

    For $X^n\phi$, the formula must hold at time $t+n$, which is equivalent to
    nesting n Next operators.

    Attributes:
        arg: The sub-formula to evaluate in the next state(s).
        steps: Number of steps to look ahead. None or 1 means single step;
            any positive integer specifies multiple steps. Defaults to None
            (equivalent to 1).

    Examples:
        - Single step: `X p`  (p holds next)
        - Multiple steps: `X[5] p`  (p holds in 5 time steps)
        - Nested: `X(X(X p))`  (p holds 3 steps ahead)

    Horizon:
        - The horizon is `1 + horizon(arg)` for single step.
        - For $X^n$: `n + horizon(arg)`.
    """

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

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        arg_hrz = self.arg.horizon()
        assert isinstance(arg_hrz, int) or math.isinf(arg_hrz), (
            "`Next` cannot be used for continuous-time specifications, horizon cannot be computed"
        )
        steps = self.steps if self.steps is not None else 1
        return steps + arg_hrz


@final
@frozen
class Always(Expr):
    r"""Always (globally) operator: $G\phi$ or $G_{[a,b]}\phi$.

    Asserts that the formula holds at all future time steps. The formula $G\phi$
    holds at time $t$ if $\phi$ holds at all times $\geq t$.

    With time constraint $G_{[a,b]}\phi$, the formula must hold for all times
    in the interval `[a,b]`.

    Attributes:
        arg: The sub-formula that must always hold.
        interval: Time constraint for when the formula must hold. Defaults to
            unbounded $[0,\infty)$.

    Examples:
        - Unbounded: G ~error  (error never occurs)
        - Bounded: G[0,10] ready  (ready holds for the next 10 steps)
        - With propositional: G (request -> F response)

    Semantics:
        `G phi` is equivalent to `~F(~phi)` (negation of eventually not phi).
    """

    arg: Expr
    interval: TimeInterval = attrs.field(factory=lambda: TimeInterval(None, None))

    @override
    def __str__(self) -> str:
        return f"(G{self.interval or ''} {self.arg})"

    @override
    def expand(self) -> Expr:
        return ~Eventually(~self.arg, self.interval)

    @override
    def to_nnf(self) -> Expr:
        return self.expand().to_nnf()

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        return (self.interval.end or math.inf) + self.arg.horizon()


@final
@frozen
class Eventually(Expr):
    r"""Eventually (future) operator: $F\phi$ or $F_{[a,b]}\phi$.

    Asserts that the formula will hold at some future time. The formula $F\phi$
    holds at time $t$ if $\phi$ holds at some time $\geq t$.

    With time constraint $F_{[a,b]}\phi$, the formula must hold at some time
    within the interval [a,b].

    Attributes:
        arg: The sub-formula that must eventually hold.
        interval: Time constraint for when the formula must hold. Defaults to
            unbounded $[0,\infty)$.

    Examples:
        Unbounded: F start  (system eventually starts)
        Bounded: F[0,100] goal  (goal reached within 100 steps)
        Nested: F G stable  (system eventually becomes stable forever)

    Semantics:
        F phi is equivalent to true U phi (true until phi becomes true).
    """

    arg: Expr
    interval: TimeInterval = attrs.field(factory=lambda: TimeInterval(None, None))

    @override
    def __str__(self) -> str:
        return f"(F{self.interval or ''} {self.arg})"

    @override
    def expand(self) -> Expr:
        match self.interval:
            case TimeInterval(None, None) | TimeInterval(0, None):
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
                # F[t1, inf]
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

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        return (self.interval.end or math.inf) + self.arg.horizon()


@final
@frozen
class Until(Expr):
    r"""Until operator: $\phi U \psi$ or $\phi U_{[a,b]} \psi$.

    Binary temporal operator asserting that lhs holds continuously until rhs
    becomes true. The formula $\phi U \psi$ holds at time $t$ if there exists
    a time $\geq t$ where $\psi$ holds, and $\phi$ holds at all times from $t$
    until that moment.

    With time constraint $\phi U_{[a,b]} \psi$, psi must become true within
    the interval [a,b] while phi holds continuously until then.

    Attributes:
        lhs: The left operand formula ($\phi$, must hold until rhs).
        rhs: The right operand formula ($\psi$, becomes true).
        interval: Time constraint for when rhs must hold. Defaults to
            unbounded $[0,\infty)$.

    Examples:
        Unbounded: request U grant  (request holds until grant)
        Bounded: sending U[0,10] ack  (sending holds until ack within 10 steps)
        Nested: (a | b) U c  (a or b holds until c)

    Semantics:
        phi U psi asserts: at some future point, psi will be true, and phi
        will hold up to that point.
    """

    lhs: Expr
    rhs: Expr
    interval: TimeInterval = attrs.field(factory=lambda: TimeInterval(None, None))

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
            case TimeInterval(None | 0, None):
                # Just make an unbounded one here
                return Until(new_lhs, new_rhs)
            case TimeInterval(t1, None):  # Unbounded end
                return Always(
                    arg=Until(lhs=new_lhs, rhs=new_rhs),
                    interval=TimeInterval(0, t1),
                ).expand()
            case TimeInterval(t1, _):
                z1 = Eventually(interval=self.interval, arg=new_lhs).expand()
                until_interval = TimeInterval(t1, None)
                z2 = Until(interval=until_interval, lhs=new_lhs, rhs=new_rhs).expand()
                return z1 & z2

    @override
    def children(self) -> Iterator[Expr]:
        yield self.lhs
        yield self.rhs

    @override
    def horizon(self) -> int | float:
        end = self.interval.end or math.inf
        return max(self.lhs.horizon() + end - 1, self.rhs.horizon() + end)


Var = TypeVar("Var")
LTLExpr: TypeAlias = BaseExpr[Var] | Next | Always | Eventually | Until
"""LTL expression types"""

__all__ = [
    "LTLExpr",
    "TimeInterval",
    "Next",
    "Always",
    "Eventually",
    "Until",
]

__docformat__ = "google"
