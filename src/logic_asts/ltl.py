r"""Abstract syntax trees for linear temporal logic (LTL).

This module extends propositional logic with temporal operators for specifying
LTL properties. Supports both weak and strong variants via Spot syntax:

Unary operators:
    - X (Next): weak next, vacuous on finite traces
    - X[!] (StrongNext): strong next, false on finite traces
    - F (Eventually): existential temporal operator
    - G (Always): universal temporal operator

Binary operators:
    - U (Until): strong until, requires right to eventually hold
    - W (WeakUntil): weak until, allows left to hold forever
    - R (Release): weak release, allows right to hold forever
    - M (StrongRelease): strong release, requires left to eventually hold

Key Classes:
    - TimeInterval: Represents time bounds [start, end]
    - Next, StrongNext: Next-step operators (weak and strong)
    - Always, Eventually: Unary temporal operators
    - Until, WeakUntil: Until operators (strong and weak)
    - Release, StrongRelease: Release operators (weak and strong)

Examples:
    Request-response property: `request -> F response`
    >>> from logic_asts.base import Variable, Implies
    >>> request = Variable("request")
    >>> response = Variable("response")
    >>> print(Implies(request, Eventually(response)))
    (request -> (F response))

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
from collections.abc import Hashable, Iterator
from typing import Generic, cast, final

import attrs
from attrs import frozen
from typing_extensions import TypeGuard, TypeVar, override

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
from logic_asts.utils import check_positive, check_start, convert_next_step

_T = TypeVar("_T", bound=Hashable, default=Hashable)


def is_ltl_node[_T: Hashable](node: object, check_type: type[_T] | None = None) -> TypeGuard[LTLExpr[_T]]:
    """Shallow membership test: is ``node`` an LTL node (bool or temporal)?

    >>> is_ltl_node(Always(Variable("p")))
    True
    """
    return is_bool_node(node, check_type) or isinstance(
        node,
        (Next, StrongNext, Always, Eventually, Until, WeakUntil, Release, StrongRelease),
    )


@final
@frozen
class TimeInterval:
    r"""Time constraint for temporal operators: interval :math:`[a,b]`.

    Represents a time interval for constraining when temporal properties must
    hold. The interval is closed on both ends. None represents unboundedness
    at that end.

    Attributes:
        start: Lower bound (inclusive), or None for no lower bound. Defaults to 0
            when None and used with duration().
        end: Upper bound (inclusive), or None for unbounded.

    Examples:
        - Bounded: `TimeInterval(0, 10)`    represents :math:`[0,10]`
        - Left unbounded: `TimeInterval(None, 20)`  represents :math:`[0,20]`
        - Right unbounded: `TimeInterval(5, None)`  represents :math:`[5,\infty)`
        - Fully unbounded: `TimeInterval(None, None)`  represents :math:`[0,\infty)`

    Note:
        - start and end must be non-negative
        - start must be <= end
        - No point intervals (start == end not allowed if both are non-None)
    """

    start: int | None = attrs.field(default=None, validator=[check_positive, check_start])
    end: int | None = attrs.field(default=None, validator=[check_positive])

    @override
    def __str__(self) -> str:
        return self.format()

    def format(self, *, strong: bool = False) -> str:
        r"""Render the interval, optionally with the strong ``!`` marker.

        ``strong=True`` produces ``[a, b!]`` and is required by the grammar
        when an enclosing Always/Eventually is strong; without it the strong
        flag would be lost on round-trip.
        """
        if self.start is None and self.end is None and not strong:
            return ""
        start_str = str(self.start) if self.start is not None else ""
        end_str = str(self.end) if self.end is not None else ""
        suffix = "!" if strong else ""
        return f"[{start_str}, {end_str}{suffix}]"

    def duration(self) -> int | float:
        r"""Calculate the duration of the interval.

        Computes the length of the interval, treating None as:
        - start = None as 0
        - end = None as infinity

        Returns:
            The length :math:`b - a` where :math:`a` is start and :math:`b` is end.
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
        r"""Check if the interval represents the unbounded future :math:`[0, \infty)`.

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
class Next(Expr, Generic[ChildExpr]):
    r"""Next operator: :math:`X\phi` or :math:`X^n\phi`.

    Asserts that the formula holds in the next time step(s). A formula :math:`X\phi`
    holds at time :math:`t` if :math:`\phi` holds at time :math:`t+1`.

    For :math:`X^n\phi`, the formula must hold at time :math:`t+n`, which is equivalent to
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

    Note:
        - The horizon is ``1 + horizon(arg)`` for single step.
        - For :math:`X^n`: ``n + horizon(arg)``.
    """

    arg: ChildExpr
    steps: int | None = attrs.field(default=None, converter=convert_next_step)

    @override
    def __str__(self) -> str:
        match self.steps:
            case None | 1:
                step_str = ""
            case t:
                step_str = f"[{t}]"
        return f"(X{step_str} {self.arg})"

    @override
    def expand(self) -> ChildExpr:
        arg = self.arg.expand()
        if self.steps is None:
            return cast(ChildExpr, Next(arg))
        else:
            assert isinstance(self.steps, int)
            expr: Expr = arg
            for _ in range(self.steps):
                expr = Next(expr)
            return cast(ChildExpr, expr)

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            # !X f = X[!] !f (strong-X dual of weak-X)
            return cast(ChildExpr, StrongNext(self.arg.to_nnf(negate=True, expand=False)))
        # X f = X f
        return cast(ChildExpr, Next(self.arg.to_nnf(expand=False)))

    @override
    def children(self) -> Iterator[ChildExpr]:
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
class StrongNext(Expr, Generic[ChildExpr]):
    r"""Strong Next operator: :math:`X[!]\phi` or :math:`X[n!]\phi`.

    Asserts that the formula holds in the next time step(s) on finite traces.
    Unlike the weak Next, :math:`X[!]\phi` is false at the last position of a
    finite trace.

    For :math:`X[n!]\phi`, the formula must hold at time :math:`t+n`, which is
    equivalent to nesting n StrongNext operators.

    Attributes:
        arg: The sub-formula to evaluate in the next state(s).
        steps: Number of steps to look ahead. None or 1 means single step;
            any positive integer specifies multiple steps. Defaults to None
            (equivalent to 1).

    Examples:
        - Single step: `X[!] p`  (p holds next on finite traces)
        - Multiple steps: `X[3!] p`  (p holds in 5 time steps on finite traces)

    Note:
        On infinite traces, StrongNext is equivalent to Next.
        The horizon is ``1 + horizon(arg)`` for single step.
        For :math:`X[n!]`: ``n + horizon(arg)``.
    """

    arg: ChildExpr
    steps: int | None = attrs.field(default=None, converter=convert_next_step)

    @override
    def __str__(self) -> str:
        match self.steps:
            case None | 1:
                step_str = "[!]"
            case t:
                step_str = f"[{t}!]"
        return f"(X{step_str} {self.arg})"

    @override
    def expand(self) -> ChildExpr:
        arg = self.arg.expand()
        if self.steps is None:
            return cast(ChildExpr, StrongNext(arg))
        else:
            assert isinstance(self.steps, int)
            expr: Expr = arg
            for _ in range(self.steps):
                expr = StrongNext(expr)
            return cast(ChildExpr, expr)

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            # !X[!] f = X !f (weak-X dual of strong-X)
            return cast(ChildExpr, Next(self.arg.to_nnf(negate=True, expand=False)))
        # X[!] f = X[!] f
        return cast(ChildExpr, StrongNext(self.arg.to_nnf(expand=False)))

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        arg_hrz = self.arg.horizon()
        assert isinstance(arg_hrz, int) or math.isinf(arg_hrz), (
            "`StrongNext` cannot be used for continuous-time specifications, horizon cannot be computed"
        )
        steps = self.steps if self.steps is not None else 1
        return steps + arg_hrz


@final
@frozen
class Always(Expr, Generic[ChildExpr]):
    r"""Always (globally) operator: :math:`G\phi` or :math:`G_{[a,b]}\phi`.

    Asserts that the formula holds at all future time steps. The formula :math:`G\phi`
    holds at time :math:`t` if :math:`\phi` holds at all times :math:`\geq t`.

    With time constraint :math:`G_{[a,b]}\phi`, the formula must hold for all times
    in the interval ``[a,b]``.

    Attributes:
        arg: The sub-formula that must always hold.
        interval: Time constraint for when the formula must hold. Defaults to
            unbounded :math:`[0,\infty)`.
        strong: If True, expand bounded G using StrongNext instead of Next.
            Defaults to False (weak/standard next).

    Examples:
        - Unbounded: G ~error  (error never occurs)
        - Bounded: G[0,10] ready  (ready holds for the next 10 steps)
        - With propositional: G (request -> F response)

    Note:
        Semantics: ``G phi`` is equivalent to ``~F(~phi)`` (negation of eventually not phi).
    """

    arg: ChildExpr
    interval: TimeInterval = attrs.field(factory=lambda: TimeInterval(None, None))
    strong: bool = attrs.field(default=False)

    @override
    def __str__(self) -> str:
        return f"(G{self.interval.format(strong=self.strong)} {self.arg})"

    @override
    def expand(self) -> ChildExpr:
        # Choose Next or StrongNext based on self.strong
        next_op = StrongNext if self.strong else Next

        match self.interval:
            case TimeInterval(None, None) | TimeInterval(0, None):
                # Unbounded G
                return cast(ChildExpr, Always(self.arg.expand(), strong=self.strong))
            case TimeInterval(0, int(t2)) | TimeInterval(None, int(t2)):
                # G[0, t2]
                arg = self.arg.expand()  # zuban: ignore[unreachable]
                expr: Expr = arg
                for _ in range(t2):
                    expr = expr & next_op(arg)
                return cast(ChildExpr, expr)
            case TimeInterval(int(t1), None):
                # G[t1, inf]
                assert t1 > 0  # zuban: ignore[unreachable]
                return cast(ChildExpr, next_op(Always(self.arg, strong=self.strong), t1).expand())
            case TimeInterval(int(t1), int(t2)):
                # G[t1, t2]
                assert t1 > 0  # zuban: ignore[unreachable]
                # G[t1, t2] = X[t1] G[0,t2-t1] arg
                # Nested nexts until t1
                return cast(ChildExpr, next_op(Always(self.arg, TimeInterval(0, t2 - t1), strong=self.strong), t1).expand())
            case _:
                raise RuntimeError(f"Unexpected time interval {self.interval}")

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            # ! G x = F !x
            return cast(ChildExpr, Eventually(self.arg.to_nnf(negate=True, expand=False), self.interval))
        # G x = G x
        return cast(ChildExpr, Always(self.arg.to_nnf(expand=False), self.interval))

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        return (self.interval.end or math.inf) + self.arg.horizon()


@final
@frozen
class Eventually(Expr, Generic[ChildExpr]):
    r"""Eventually (future) operator: :math:`F\phi` or :math:`F_{[a,b]}\phi`.

    Asserts that the formula will hold at some future time. The formula :math:`F\phi`
    holds at time :math:`t` if :math:`\phi` holds at some time :math:`\geq t`.

    With time constraint :math:`F_{[a,b]}\phi`, the formula must hold at some time
    within the interval [a,b].

    Attributes:
        arg: The sub-formula that must eventually hold.
        interval: Time constraint for when the formula must hold. Defaults to
            unbounded :math:`[0,\infty)`.
        strong: If True, expand bounded F using StrongNext instead of Next.
            Defaults to False (weak/standard next).

    Examples:
        Unbounded: F start  (system eventually starts)
        Bounded: F[0,100] goal  (goal reached within 100 steps)
        Nested: F G stable  (system eventually becomes stable forever)

    Note:
        Semantics: F phi is equivalent to true U phi (true until phi becomes true).
    """

    arg: ChildExpr
    interval: TimeInterval = attrs.field(factory=lambda: TimeInterval(None, None))
    strong: bool = attrs.field(default=False)

    @override
    def __str__(self) -> str:
        return f"(F{self.interval.format(strong=self.strong)} {self.arg})"

    @override
    def expand(self) -> ChildExpr:
        # Choose Next or StrongNext based on self.strong
        next_op = StrongNext if self.strong else Next

        match self.interval:
            case TimeInterval(None, None) | TimeInterval(0, None):
                # Unbounded F
                return cast(ChildExpr, Eventually(self.arg.expand(), strong=self.strong))
            case TimeInterval(0, int(t2)) | TimeInterval(None, int(t2)):
                # F[0, t2]
                arg = self.arg.expand()  # zuban: ignore[unreachable]
                expr: Expr = arg
                for _ in range(t2):
                    expr = expr & next_op(arg)
                return cast(ChildExpr, expr)
            case TimeInterval(int(t1), None):
                # F[t1, inf]
                assert t1 > 0  # zuban: ignore[unreachable]
                return cast(ChildExpr, next_op(Eventually(self.arg, strong=self.strong), t1).expand())
            case TimeInterval(int(t1), int(t2)):
                # F[t1, t2]
                assert t1 > 0  # zuban: ignore[unreachable]
                # F[t1, t2] = X[t1] F[0,t2-t1] arg
                # Nested nexts until t1
                return cast(
                    ChildExpr, next_op(Eventually(self.arg, TimeInterval(0, t2 - t1), strong=self.strong), t1).expand()
                )
            case _:
                raise RuntimeError(f"Unexpected time interval {self.interval}")

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            # ! F x = G !x
            return cast(ChildExpr, Always(self.arg.to_nnf(negate=True, expand=False), self.interval))
        # F x = F x
        return cast(ChildExpr, Eventually(self.arg.to_nnf(expand=False), self.interval))

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        return (self.interval.end or math.inf) + self.arg.horizon()


@final
@frozen
class Until(Expr, Generic[ChildExpr]):
    r"""Until operator: :math:`\phi U \psi` or :math:`\phi U_{[a,b]} \psi`.

    Binary temporal operator asserting that lhs holds continuously until rhs
    becomes true. The formula :math:`\phi U \psi` holds at time :math:`t` if there exists
    a time :math:`\geq t` where :math:`\psi` holds, and :math:`\phi` holds at all times from :math:`t`
    until that moment.

    With time constraint :math:`\phi U_{[a,b]} \psi`, psi must become true within
    the interval [a,b] while phi holds continuously until then.

    Attributes:
        lhs: The left operand formula (:math:`\phi`, must hold until rhs).
        rhs: The right operand formula (:math:`\psi`, becomes true).
        interval: Time constraint for when rhs must hold. Defaults to
            unbounded :math:`[0,\infty)`.

    Examples:
        Unbounded: request U grant  (request holds until grant)
        Bounded: sending U[0,10] ack  (sending holds until ack within 10 steps)
        Nested: (a | b) U c  (a or b holds until c)

    Note:
        Semantics: phi U psi asserts: at some future point, psi will be true, and phi
        will hold up to that point.
    """

    lhs: ChildExpr
    rhs: ChildExpr
    interval: TimeInterval = attrs.field(factory=lambda: TimeInterval(None, None))

    @override
    def __str__(self) -> str:
        return f"({self.lhs} U{self.interval or ''} {self.rhs})"

    @override
    def expand(self) -> ChildExpr:
        new_lhs = self.lhs.expand()
        new_rhs = self.rhs.expand()
        match self.interval:
            case TimeInterval(None | 0, None):
                # Just make an unbounded one here
                return cast(ChildExpr, Until(new_lhs, new_rhs))
            case TimeInterval(t1, None):  # Unbounded end
                return cast(
                    ChildExpr,
                    Always(  # zuban: ignore[unreachable]
                        arg=Until(lhs=new_lhs, rhs=new_rhs),
                        interval=TimeInterval(0, t1),
                    ).expand(),
                )
            case TimeInterval(t1, _):
                z1 = Eventually(interval=self.interval, arg=new_lhs).expand()  # zuban: ignore[unreachable]
                until_interval = TimeInterval(t1, None)
                z2 = Until(interval=until_interval, lhs=new_lhs, rhs=new_rhs).expand()
                return cast(ChildExpr, z1 & z2)
            case _:
                raise RuntimeError(f"Unexpected time interval {self.interval}")

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            # ! (f U g) = (!f) R (!g)
            return cast(
                ChildExpr,
                Release(
                    self.lhs.to_nnf(negate=True, expand=False),
                    self.rhs.to_nnf(negate=True, expand=False),
                    self.interval,
                ),
            )
        return cast(
            ChildExpr,
            Until(
                self.lhs.to_nnf(expand=False),
                self.rhs.to_nnf(expand=False),
                self.interval,
            ),
        )

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield self.lhs
        yield self.rhs

    @override
    def horizon(self) -> int | float:
        end = self.interval.end or math.inf
        return max(self.lhs.horizon() + end - 1, self.rhs.horizon() + end)


@final
@frozen
class WeakUntil(Expr, Generic[ChildExpr]):
    r"""Weak Until operator: :math:`\phi W \psi` or :math:`\phi W_{[a,b]} \psi`.

    Binary temporal operator asserting that lhs holds continuously until rhs
    becomes true, or lhs holds forever. The formula :math:`\phi W \psi` holds
    at time :math:`t` if either :math:`\phi` holds at all times from :math:`t`
    onward, or there exists a time :math:`\geq t` where :math:`\psi` holds and
    :math:`\phi` holds at all times from :math:`t` until that moment.

    With time constraint :math:`\phi W_{[a,b]} \psi`, psi must become true within
    the interval [a,b] while phi holds continuously until then, or phi holds
    throughout [a,b] and beyond.

    Attributes:
        lhs: The left operand formula (:math:`\phi`, must hold until rhs or forever).
        rhs: The right operand formula (:math:`\psi`, becomes true).
        interval: Time constraint for when rhs must hold. Defaults to
            unbounded :math:`[0,\infty)`.

    Examples:
        Unbounded: request W grant  (request holds until grant or forever)
        Bounded: sending W[0,10] ack  (sending holds until ack within 10 steps or forever)

    Note:
        Semantics: phi W psi is equivalent to (phi U psi) | G phi.
    """

    lhs: ChildExpr
    rhs: ChildExpr
    interval: TimeInterval = attrs.field(factory=lambda: TimeInterval(None, None))

    @override
    def __str__(self) -> str:
        return f"({self.lhs} W{self.interval or ''} {self.rhs})"

    @override
    def expand(self) -> ChildExpr:
        new_lhs = self.lhs.expand()
        new_rhs = self.rhs.expand()
        match self.interval:
            case TimeInterval(None | 0, None):
                # Just make an unbounded one here
                return cast(ChildExpr, WeakUntil(new_lhs, new_rhs))
            case TimeInterval(t1, None):  # Unbounded end
                return cast(
                    ChildExpr,
                    Always(  # zuban: ignore[unreachable]
                        arg=WeakUntil(lhs=new_lhs, rhs=new_rhs),
                        interval=TimeInterval(0, t1),
                    ).expand(),
                )
            case TimeInterval(t1, _):
                z1 = Eventually(interval=self.interval, arg=new_lhs).expand()  # zuban: ignore[unreachable]
                until_interval = TimeInterval(t1, None)
                z2 = WeakUntil(interval=until_interval, lhs=new_lhs, rhs=new_rhs).expand()
                return cast(ChildExpr, z1 & z2)
            case _:
                raise RuntimeError(f"Unexpected time interval {self.interval}")

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            # ! (f W g) = (!f) M (!g)
            return cast(
                ChildExpr,
                StrongRelease(
                    self.lhs.to_nnf(negate=True, expand=False),
                    self.rhs.to_nnf(negate=True, expand=False),
                    self.interval,
                ),
            )
        return cast(
            ChildExpr,
            WeakUntil(
                self.lhs.to_nnf(expand=False),
                self.rhs.to_nnf(expand=False),
                self.interval,
            ),
        )

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield self.lhs
        yield self.rhs

    @override
    def horizon(self) -> int | float:
        end = self.interval.end or math.inf
        return max(self.lhs.horizon() + end - 1, self.rhs.horizon() + end)


@final
@frozen
class Release(Expr, Generic[ChildExpr]):
    r"""Release operator: :math:`\phi R \psi` or :math:`\phi R_{[a,b]} \psi`.

    Binary temporal operator asserting that rhs holds continuously unless
    and until lhs becomes true. The formula :math:`\phi R \psi` holds at time :math:`t` if
    either :math:`\psi` holds forever from :math:`t` onward, or :math:`\phi` becomes true at some
    time :math:`\geq t` and :math:`\psi` holds continuously from :math:`t` until that moment.

    Release is the dual of Until: :math:`\phi R \psi \equiv \neg(\neg\phi U \neg\psi)`.

    With time constraint :math:`\phi R_{[a,b]} \psi`, if phi becomes true, it must do
    so within the interval [a,b], while psi holds continuously until then.

    Attributes:
        lhs: The left operand formula (:math:`\phi`, releases rhs when true).
        rhs: The right operand formula (:math:`\psi`, must hold until released).
        interval: Time constraint for when lhs may release rhs. Defaults to
            unbounded :math:`[0,\infty)`.

    Examples:
        >>> from logic_asts.base import Variable
        >>> safe = Variable("safe")
        >>> error = Variable("error")
        >>> print(Release(safe, ~error))
        (safe R !error)

        Release with time constraint:
        >>> standby = Variable("standby")
        >>> ready = Variable("ready")
        >>> print(Release(standby, ready, TimeInterval(0, 5)))
        (standby R[0, 5] ready)

    Note:
        Semantics: phi R psi asserts: psi holds continuously unless and until phi becomes
        true. Unlike Until, psi may hold forever if phi never becomes true.
    """

    lhs: ChildExpr
    rhs: ChildExpr
    interval: TimeInterval = attrs.field(factory=lambda: TimeInterval(None, None))

    @override
    def __str__(self) -> str:
        return f"({self.lhs} R{self.interval or ''} {self.rhs})"

    @override
    def expand(self) -> ChildExpr:
        # Expands as the dual of Until
        return cast(ChildExpr, Not(Until(~self.lhs, ~self.rhs, self.interval)))

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            # ! (f R g) = (!f) U (!g)
            return cast(
                ChildExpr,
                Until(
                    self.lhs.to_nnf(negate=True, expand=False),
                    self.rhs.to_nnf(negate=True, expand=False),
                    self.interval,
                ),
            )
        return cast(
            ChildExpr,
            Release(
                self.lhs.to_nnf(expand=False),
                self.rhs.to_nnf(expand=False),
                self.interval,
            ),
        )

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield self.lhs
        yield self.rhs

    @override
    def horizon(self) -> int | float:
        # Release has same horizon as Until
        end = self.interval.end or math.inf
        return max(self.lhs.horizon() + end - 1, self.rhs.horizon() + end)


@final
@frozen
class StrongRelease(Expr, Generic[ChildExpr]):
    r"""Strong Release operator: :math:`\phi M \psi` or :math:`\phi M_{[a,b]} \psi`.

    Binary temporal operator asserting that rhs holds continuously until lhs
    becomes true, and lhs must eventually become true. The formula :math:`\phi M \psi`
    holds at time :math:`t` if there exists a time :math:`\geq t` where :math:`\phi`
    becomes true, and :math:`\psi` holds at all times from :math:`t` up to and
    including that moment.

    Strong Release is the dual of Weak Until: :math:`\phi M \psi \equiv \neg(\neg\phi W \neg\psi)`.

    With time constraint :math:`\phi M_{[a,b]} \psi`, phi must become true within
    the interval [a,b], while psi holds continuously until then.

    Attributes:
        lhs: The left operand formula (:math:`\phi`, releases rhs when true).
        rhs: The right operand formula (:math:`\psi`, must hold until released).
        interval: Time constraint for when lhs may release rhs. Defaults to
            unbounded :math:`[0,\infty)`.

    Examples:
        >>> from logic_asts.base import Variable
        >>> safe = Variable("safe")
        >>> error = Variable("error")
        >>> print(StrongRelease(safe, ~error))
        (safe M !error)

    Note:
        Semantics: phi M psi asserts: psi holds continuously until phi becomes
        true, and phi must eventually become true. Unlike Release, psi cannot
        hold forever without phi becoming true.
    """

    lhs: ChildExpr
    rhs: ChildExpr
    interval: TimeInterval = attrs.field(factory=lambda: TimeInterval(None, None))

    @override
    def __str__(self) -> str:
        return f"({self.lhs} M{self.interval or ''} {self.rhs})"

    @override
    def expand(self) -> ChildExpr:
        new_lhs = self.lhs.expand()
        new_rhs = self.rhs.expand()
        match self.interval:
            case TimeInterval(None | 0, None):
                # Unbounded: return StrongRelease with expanded operands
                return cast(ChildExpr, StrongRelease(new_lhs, new_rhs))
            case _:
                # Bounded: use dual form Not(WeakUntil(~lhs, ~rhs, interval))
                return cast(ChildExpr, Not(WeakUntil(~new_lhs, ~new_rhs, self.interval)))  # zuban: ignore[unreachable]

    @override
    def to_nnf(self, *, negate: bool = False, expand: bool = True) -> ChildExpr:
        if expand:
            return cast(ChildExpr, self.expand().to_nnf(negate=negate, expand=False))
        if negate:
            # ! (f M g) = (!f) W (!g)
            return cast(
                ChildExpr,
                WeakUntil(
                    self.lhs.to_nnf(negate=True, expand=False),
                    self.rhs.to_nnf(negate=True, expand=False),
                    self.interval,
                ),
            )
        return cast(
            ChildExpr,
            StrongRelease(
                self.lhs.to_nnf(expand=False),
                self.rhs.to_nnf(expand=False),
                self.interval,
            ),
        )

    @override
    def children(self) -> Iterator[ChildExpr]:
        yield self.lhs
        yield self.rhs

    @override
    def horizon(self) -> int | float:
        end = self.interval.end or math.inf
        return max(self.lhs.horizon() + end - 1, self.rhs.horizon() + end)


Var = TypeVar("Var")
type LTLExpr[Var: Hashable] = (
    Variable[Var]
    | Literal
    | And[LTLExpr[Var]]
    | Or[LTLExpr[Var]]
    | Not[LTLExpr[Var]]
    | Implies[LTLExpr[Var]]
    | Equiv[LTLExpr[Var]]
    | Xor[LTLExpr[Var]]
    | Next[LTLExpr[Var]]
    | StrongNext[LTLExpr[Var]]
    | Always[LTLExpr[Var]]
    | Eventually[LTLExpr[Var]]
    | Until[LTLExpr[Var]]
    | WeakUntil[LTLExpr[Var]]
    | Release[LTLExpr[Var]]
    | StrongRelease[LTLExpr[Var]]
)
"""LTL expression types.

Use :func:`logic_asts.ltl_expr_iter` to iterate over the subtree of an
``LTLExpr[AP]`` with full static type information (returns
``Iterator[LTLExpr[AP]]``).
"""


def ltl_expr_iter(expr: LTLExpr[Var]) -> Iterator[LTLExpr[Var]]:
    """Returns an post-order iterator over the LTL expression

    Iterates over all sub-expressions in post-order, visiting each
    expression exactly once. In post-order, children are yielded before
    their parents, making this suitable for bottom-up processing.

    Moreover, it ensures that each subexpression is a ``LTLExpr``.

    Yields:
        Each node in the expression tree in post-order sequence.

    Raises:
        TypeError: If the expression contains a subexpression that is not a ``LTLExpr``

    """
    return iter(
        ExprVisitor(
            cast(
                list[type[LTLExpr[Var]]],
                [
                    Next,
                    StrongNext,
                    Always,
                    Eventually,
                    Until,
                    WeakUntil,
                    Release,
                    StrongRelease,
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
    "LTLExpr",
    "TimeInterval",
    "Next",
    "StrongNext",
    "Always",
    "Eventually",
    "Until",
    "WeakUntil",
    "Release",
    "StrongRelease",
    "ltl_expr_iter",
    "is_ltl_node",
]
