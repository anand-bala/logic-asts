# flake8: noqa: ANN401
# pyright: reportExplicitAny=false
from __future__ import annotations

from numbers import Real
from typing import Any

import attrs


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
