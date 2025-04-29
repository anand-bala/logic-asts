from __future__ import annotations

from numbers import Real
from typing import TypeVar

import attrs

T = TypeVar("T", bound=Real)


def check_positive(_instance: object, attribute: attrs.Attribute[T | None], value: T | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"attribute {attribute.name} cannot have negative value ({value})")


def check_start(instance: object, attribute: attrs.Attribute[T | None], value: T | None) -> None:
    end: T | None = getattr(instance, "end", None)
    if value is None or end is None:
        return
    if value == end:
        raise ValueError(f"{attribute.name} cannot be point values [a,a]")
    if value > end:
        raise ValueError(f"{attribute.name} [a,b] cannot have a > b")
