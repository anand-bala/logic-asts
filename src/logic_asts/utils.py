# flake8: noqa: ANN401
# pyright: reportExplicitAny=false
from __future__ import annotations

import typing
from collections.abc import Iterable
from numbers import Real
from typing import TYPE_CHECKING, Any

import attrs

if TYPE_CHECKING:
    from logic_asts.spec import Expr


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


def convert_next_step(value: int | None) -> int | None:
    """Convert the `steps` parameter for Next and StrongNext into `None` if equal to `1`."""
    if value is None or value == 1:
        return None
    else:
        return value


def nary_fold[E: Expr, Child: Expr](cls: type[E], args: Iterable[Child], identity: None | Child = None) -> E | Child:

    args = tuple(flatten_nary_args(cls, args))
    if len(args) >= 2:
        return cls(args)  # type: ignore[call-arg]
    elif len(args) == 1:
        return args[0]
    elif identity is not None:
        return identity
    else:
        raise ValueError(f"number of arguments given to `nary_fold[{cls.__name__}]` is effectively 0, with no identity")


def flatten_nary_args[E: Expr, Child: Expr](cls: type[E], args: Iterable[Child]) -> Iterable[Child]:
    for a in args:
        if isinstance(a, cls):
            yield from typing.cast(Iterable[Child], a.children())
        else:
            yield a
