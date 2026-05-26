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


def _convert_next_step(value: int | None) -> int | None:
    """Convert the `steps` parameter for Next and StrongNext into `None` if equal to `1`."""
    if value is None or value == 1:
        return None
    else:
        return value


def to_nnf(expr: logic.Expr, *, negate: bool = False, _expanded: bool = False) -> logic.Expr:
    """Use the Spot NNF/negation identities for all supported logics.

    Applies Spot's negative normal form rules, which properly handle strong/weak
    temporal operators. For LTL, this includes:

    - !X f = X[!] !f (strong-X dual of weak-X)
    - !X[!] f = X !f
    - !(f U g) = (!f) R (!g)
    - !(f W g) = (!f) M (!g)
    - !(f R g) = (!f) U (!g)
    - !(f M g) = (!f) W (!g)
    - !G f = F !f
    - !F f = G !f
    """
    if not _expanded:
        expr = expr.expand()
        _expanded = True

    match expr:
        case logic.Literal() | logic.Variable():
            return expr if not negate else ~expr
        case logic.Not(arg):
            return to_nnf(arg, negate=not negate, _expanded=_expanded)
        case logic.And(args):
            args = tuple(to_nnf(arg, negate=negate, _expanded=_expanded) for arg in args)
            if negate:
                return logic.Or(args)
            else:
                return logic.And(args)
        case logic.Or(args):
            args = tuple(to_nnf(arg, negate=negate, _expanded=_expanded) for arg in args)
            if negate:
                return logic.And(args)
            else:
                return logic.Or(args)
        case logic.Implies(lhs, rhs):
            return to_nnf(~lhs | rhs, negate=negate, _expanded=_expanded)
        case logic.Equiv(x, y):
            return to_nnf((x | ~y) & (~x | y), negate=negate, _expanded=_expanded)
        case logic.Xor(x, y):
            return to_nnf((x & ~y) | (~x & y), negate=negate, _expanded=_expanded)
        case logic.ltl.Next(arg):
            if negate:
                # !X f = X[!] !f (strong-X dual of weak-X)
                return logic.ltl.StrongNext(to_nnf(arg, negate=True, _expanded=_expanded))
            # X f = X f
            return logic.ltl.Next(to_nnf(arg, _expanded=_expanded))
        case logic.ltl.StrongNext(arg):
            if negate:
                # !X[!] f = X !f (weak-X dual of strong-X)
                return logic.ltl.Next(to_nnf(arg, negate=True, _expanded=_expanded))
            # X[!] f = X[!] f
            return logic.ltl.StrongNext(to_nnf(arg, _expanded=_expanded))
        case logic.ltl.Always(arg, interval):
            if negate:
                # ! G x = F !x
                return logic.ltl.Eventually(to_nnf(arg, negate=True, _expanded=_expanded), interval)
            # G x = G x
            return logic.ltl.Always(to_nnf(arg, _expanded=_expanded), interval)
        case logic.ltl.Eventually(arg, interval):
            if negate:
                # ! F x = G !x
                return logic.ltl.Always(to_nnf(arg, negate=True, _expanded=_expanded), interval)
            # F x = F x
            return logic.ltl.Eventually(to_nnf(arg, _expanded=_expanded), interval)
        case logic.ltl.Until(lhs, rhs, interval):
            if negate:
                # ! (f U g) = (!f) R (!g)
                return logic.ltl.Release(
                    to_nnf(lhs, negate=True, _expanded=_expanded),
                    to_nnf(rhs, negate=True, _expanded=_expanded),
                    interval,
                )
            return logic.ltl.Until(
                to_nnf(lhs, _expanded=_expanded),
                to_nnf(rhs, _expanded=_expanded),
                interval,
            )
        case logic.ltl.WeakUntil(lhs, rhs, interval):
            if negate:
                # ! (f W g) = (!f) M (!g)
                return logic.ltl.StrongRelease(
                    to_nnf(lhs, negate=True, _expanded=_expanded),
                    to_nnf(rhs, negate=True, _expanded=_expanded),
                    interval,
                )
            return logic.ltl.WeakUntil(
                to_nnf(lhs, _expanded=_expanded),
                to_nnf(rhs, _expanded=_expanded),
                interval,
            )
        case logic.ltl.Release(lhs, rhs, interval):
            if negate:
                # ! (f R g) = (!f) U (!g)  [FIX: was wrongly returning Release]
                return logic.ltl.Until(
                    to_nnf(lhs, negate=True, _expanded=_expanded),
                    to_nnf(rhs, negate=True, _expanded=_expanded),
                    interval,
                )
            return logic.ltl.Release(
                to_nnf(lhs, _expanded=_expanded),
                to_nnf(rhs, _expanded=_expanded),
                interval,
            )
        case logic.ltl.StrongRelease(lhs, rhs, interval):
            if negate:
                # ! (f M g) = (!f) W (!g)
                return logic.ltl.WeakUntil(
                    to_nnf(lhs, negate=True, _expanded=_expanded),
                    to_nnf(rhs, negate=True, _expanded=_expanded),
                    interval,
                )
            return logic.ltl.StrongRelease(
                to_nnf(lhs, _expanded=_expanded),
                to_nnf(rhs, _expanded=_expanded),
                interval,
            )
        case logic.strel.Everywhere(arg, interval, dist_fn):
            if negate:
                # !(E A) = E S (negate dual: Everywhere to Somewhere)
                return logic.strel.Somewhere(to_nnf(arg, negate=True, _expanded=_expanded), interval, dist_fn)
            return logic.strel.Everywhere(to_nnf(arg, _expanded=_expanded), interval, dist_fn)
        case logic.strel.Somewhere(arg, interval, dist_fn):
            if negate:
                # !(E S) = E A (negate dual: Somewhere to Everywhere)  [FIX: was wrongly returning Somewhere]
                return logic.strel.Everywhere(to_nnf(arg, negate=True, _expanded=_expanded), interval, dist_fn)
            return logic.strel.Somewhere(to_nnf(arg, _expanded=_expanded), interval, dist_fn)
        case logic.strel.Escape():
            # TODO: there isn't a real dual to Escape
            # prevent negation from passing through
            expr = attrs.evolve(expr, arg=to_nnf(expr.arg, _expanded=_expanded))
            if negate:
                return logic.Not(expr)
            else:
                return expr
        case logic.strel.Reach():
            # TODO: there isn't a real dual to Reach
            # prevent negation from passing through
            expr = attrs.evolve(expr, lhs=to_nnf(expr.lhs, _expanded=_expanded), rhs=to_nnf(expr.rhs, _expanded=_expanded))
            if negate:
                return logic.Not(expr)
            else:
                return expr
        case logic.stl_go.GraphIncoming() | logic.stl_go.GraphOutgoing():
            # TODO: unsure what the dual to these is
            expr = attrs.evolve(expr, arg=to_nnf(expr.arg, _expanded=_expanded))
            if negate:
                return logic.Not(expr)
            else:
                return expr
        case logic.psl.SuffixImpliesUniv(sere, formula):
            new_formula = to_nnf(formula, negate=negate, _expanded=_expanded)
            if negate:
                return logic.psl.SuffixImpliesExist(sere, new_formula)
            return logic.psl.SuffixImpliesUniv(sere, new_formula)
        case logic.psl.SuffixImpliesExist(sere, formula):
            new_formula = to_nnf(formula, negate=negate, _expanded=_expanded)
            if negate:
                return logic.psl.SuffixImpliesUniv(sere, new_formula)
            return logic.psl.SuffixImpliesExist(sere, new_formula)
        case logic.psl.WeakClosure(sere):
            if negate:
                return logic.psl.NegStrongClosure(sere)
            return expr
        case logic.psl.NegStrongClosure(sere):
            if negate:
                return logic.psl.WeakClosure(sere)
            return expr
        case logic.psl.StrongClosure():
            if negate:
                return logic.Not(expr)
            return expr
        case logic.sere.Complement(inner):
            if negate:
                return inner
            return expr
        case (
            logic.sere.Concat()
            | logic.sere.Fusion()
            | logic.sere.Alt()
            | logic.sere.Inter()
            | logic.sere.NLMInter()
            | logic.sere.Repeat()
            | logic.sere.GotoRepeat()
            | logic.sere.EqualRepeat()
            | logic.sere.FusionRepeat()
            | logic.sere.FirstMatch()
        ):
            if negate:
                return logic.sere.Complement(expr)
            return expr
        case _:
            # When unsure, just return
            if negate:
                return logic.Not(expr)
            else:
                return expr
