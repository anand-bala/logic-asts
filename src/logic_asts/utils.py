from __future__ import annotations

import attrs


def check_positive[T](_instance: object, attribute: attrs.Attribute[T | None], value: float | int | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"attribute {attribute} cannot have negative value")


def check_start[T](instance: object, attribute: attrs.Attribute[T | None], value: float | int | None) -> None:
    match (value, getattr(instance, "end", None)):
        case (float(t1), float(t2)) if t1 == t2:
            raise ValueError(f"{attribute} cannot be point values [a,a]")
        case (float(t1), float(t2)) if t1 > t2:
            raise ValueError(f"{attribute} [a,b] cannot have a > b")
        case (float(t1), float(t2)) if t1 < 0 or t2 < 0:
            raise ValueError(f"{attribute} cannot have negative bounds")
        case _:
            pass
