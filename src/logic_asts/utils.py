# flake8: noqa: ANN401
# pyright: reportExplicitAny=false
from __future__ import annotations

from numbers import Real
from typing import Any

import attrs

import logic_asts as logic


def check_positive(_instance: Any, attribute: attrs.Attribute[None], value: Real | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"attribute {attribute.name} cannot have negative value ({value})")


def check_start(instance: Any, attribute: attrs.Attribute[None], value: Real | None) -> None:
    end: Real | None = getattr(instance, "end", None)
    if value is None or end is None:
        return
    if value == end:
        raise ValueError(f"{attribute.name} cannot be point values [a,a]")
    if value > end:
        raise ValueError(f"{attribute.name} [a,b] cannot have a > b")


def check_weight_start(instance: Any, attribute: attrs.Attribute[None], value: float | None) -> None:
    """Validator for weight interval start - ensures start <= end."""
    end: float | None = getattr(instance, "end", None)
    if value is None or end is None:
        return
    if value > end:
        raise ValueError(f"{attribute.name} [a,b] cannot have a > b")


def to_nnf(expr: logic.Expr, *, negate: bool = False) -> logic.Expr:
    """Use the NNF/negation identities for all supported logics"""

    match expr:
        case logic.Literal() | logic.Variable():
            return expr if not negate else ~expr
        case logic.Not(arg):
            return to_nnf(arg, negate=not negate)
        case logic.And(args):
            args = tuple(to_nnf(arg, negate=negate) for arg in args)
            if negate:
                return logic.Or(args)
            else:
                return logic.And(args)
        case logic.Or(args):
            args = tuple(to_nnf(arg, negate=negate) for arg in args)
            if negate:
                return logic.And(args)
            else:
                return logic.Or(args)
        case logic.Implies(lhs, rhs):
            return to_nnf(~lhs | rhs, negate=negate)
        case logic.Equiv(x, y):
            return to_nnf((x | ~y) & (~x | y), negate=negate)
        case logic.Xor(x, y):
            return to_nnf((x & ~y) | (~x & y), negate=negate)
        case logic.ltl.Next(arg):
            return logic.ltl.Next(to_nnf(arg, negate=negate))
        case logic.ltl.Always(arg, interval):
            if negate:
                # ~ G x = F ~x
                return logic.ltl.Eventually(to_nnf(arg, negate=negate), interval)
            # G x = G x
            return logic.ltl.Always(to_nnf(arg, negate=negate), interval)
        case logic.ltl.Eventually(arg, interval):
            if negate:
                # ~ F x = G ~x
                return logic.ltl.Always(to_nnf(arg, negate=negate), interval)
            # F x = F x
            return logic.ltl.Eventually(to_nnf(arg, negate=negate), interval)
        case logic.ltl.Until(lhs, rhs, interval):
            if negate:
                # ~ (p U q) = ~p R ~q
                return logic.ltl.Release(to_nnf(lhs, negate=negate), to_nnf(rhs, negate=negate), interval)
            return logic.ltl.Until(to_nnf(lhs, negate=negate), to_nnf(rhs, negate=negate), interval)

        case logic.ltl.Release(lhs, rhs, interval):
            if negate:
                # ~ (p R q) = ~p U ~q
                return logic.ltl.Release(to_nnf(lhs, negate=negate), to_nnf(rhs, negate=negate), interval)
            return logic.ltl.Release(to_nnf(lhs, negate=negate), to_nnf(rhs, negate=negate), interval)

        case logic.strel.Everywhere(arg, interval, dist_fn):
            if negate:
                # somewhere is dual to everywhere
                return logic.strel.Somewhere(to_nnf(arg, negate=negate), interval, dist_fn)
            return logic.strel.Everywhere(to_nnf(arg, negate=negate), interval, dist_fn)
        case logic.strel.Somewhere(arg, interval, dist_fn):
            if negate:
                # somewhere is dual to somewhere
                return logic.strel.Somewhere(to_nnf(arg, negate=negate), interval, dist_fn)
            return logic.strel.Somewhere(to_nnf(arg, negate=negate), interval, dist_fn)
        case logic.strel.Escape():
            # TODO: there isn't a real dual to Escape
            # prevent negation from passing through
            expr = attrs.evolve(expr, arg=to_nnf(expr.arg))
            if negate:
                return logic.Not(expr)
            else:
                return expr
        case logic.strel.Reach():
            # TODO: there isn't a real dual to Reach
            # prevent negation from passing through
            expr = attrs.evolve(expr, lhs=to_nnf(expr.lhs), rhs=to_nnf(expr.rhs))
            if negate:
                return logic.Not(expr)
            else:
                return expr
        case logic.stl_go.GraphIncoming() | logic.stl_go.GraphOutgoing():
            # TODO: unsure what the dual to these is
            expr = attrs.evolve(expr, arg=to_nnf(expr.arg))
            if negate:
                return logic.Not(expr)
            else:
                return expr
        case _:
            # When unsure, just return
            if negate:
                return logic.Not(expr)
            else:
                return expr
