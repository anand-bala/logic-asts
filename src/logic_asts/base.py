from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import reduce
from typing import Self, final, override

import attrs
from attrs import field, frozen


class Expr(ABC):
    """Abstract expression"""

    @abstractmethod
    def expand(self) -> Expr: ...

    @abstractmethod
    def to_nnf(self) -> Expr: ...

    @abstractmethod
    def children(self) -> Iterator[Expr]: ...

    @abstractmethod
    def horizon(self) -> int | float:
        """Compute the horizon of the formula. Returns `math.inf` if the formula is unbounded."""

    def __invert__(self) -> Expr:
        return Not(self)

    def __and__(self, other: Expr) -> Expr:
        return And([self, other])

    def __or__(self, other: Expr) -> Expr:
        return Or([self, other])


@final
@frozen
class Implies(Expr):
    lhs: Expr
    rhs: Expr

    @override
    def __str__(self) -> str:
        return f"{self.lhs} -> {self.rhs}"

    @override
    def expand(self) -> Expr:
        return ~self.lhs | self.rhs

    @override
    def to_nnf(self) -> Expr:
        return self.expand().to_nnf()

    @override
    def children(self) -> Iterator[Expr]:
        yield self.lhs
        yield self.rhs

    @override
    def horizon(self) -> int | float:
        return max(self.lhs.horizon(), self.rhs.horizon())


@final
@frozen
class Equiv(Expr):
    lhs: Expr
    rhs: Expr

    @override
    def __str__(self) -> str:
        return f"{self.lhs} <-> {self.rhs}"

    @override
    def expand(self) -> Expr:
        x = self.lhs
        y = self.rhs
        return (x | ~y) & (~x | y)

    @override
    def to_nnf(self) -> Expr:
        return self.expand().to_nnf()

    @override
    def children(self) -> Iterator[Expr]:
        yield self.lhs
        yield self.rhs

    @override
    def horizon(self) -> int | float:
        return max(self.lhs.horizon(), self.rhs.horizon())


@final
@frozen
class Xor(Expr):
    lhs: Expr
    rhs: Expr

    @override
    def __str__(self) -> str:
        return f"{self.lhs} ^ {self.rhs}"

    @override
    def expand(self) -> Expr:
        x = self.lhs
        y = self.rhs
        return (x & ~y) | (~x & y)

    @override
    def to_nnf(self) -> Expr:
        return self.expand().to_nnf()

    @override
    def children(self) -> Iterator[Expr]:
        yield self.lhs
        yield self.rhs

    @override
    def horizon(self) -> int | float:
        return max(self.lhs.horizon(), self.rhs.horizon())


@final
@frozen
class And(Expr):
    args: list[Expr] = field(validator=attrs.validators.min_len(2))

    @override
    def __str__(self) -> str:
        return "(" + " & ".join(str(arg) for arg in self.args) + ")"

    @override
    def __and__(self, other: Expr) -> Expr:
        if isinstance(other, And):
            return And(self.args + other.args)
        return And(self.args + [other])

    @override
    def to_nnf(self) -> Expr:
        return reduce(operator.__and__, (a.to_nnf() for a in self.args))

    @override
    def expand(self) -> Expr:
        return reduce(operator.__and__, (a.expand() for a in self.args), Literal(True))

    @override
    def children(self) -> Iterator[Expr]:
        yield from self.args

    @override
    def horizon(self) -> int | float:
        return max(arg.horizon() for arg in self.args)


@final
@frozen
class Or(Expr):
    args: list[Expr] = field(validator=attrs.validators.min_len(2))

    @override
    def __str__(self) -> str:
        return "(" + " | ".join(str(arg) for arg in self.args) + ")"

    @override
    def __or__(self, other: Expr) -> Expr:
        if isinstance(other, Or):
            return Or(self.args + other.args)
        return Or(self.args + [other])

    @override
    def to_nnf(self) -> Expr:
        return reduce(operator.__or__, (a.to_nnf() for a in self.args), Literal(False))

    @override
    def expand(self) -> Expr:
        return reduce(operator.__or__, (a.expand() for a in self.args), Literal(False))

    @override
    def children(self) -> Iterator[Expr]:
        yield from self.args

    @override
    def horizon(self) -> int | float:
        return max(arg.horizon() for arg in self.args)


@final
@frozen
class Not(Expr):
    arg: Expr

    @override
    def __str__(self) -> str:
        return f"!{str(self.arg)}"

    @override
    def __invert__(self) -> Expr:
        return self.arg

    @override
    def to_nnf(self) -> Expr:
        arg = self.arg
        match arg:
            case Literal():
                return ~arg
            case Variable():
                return self
            case Not(expr):
                return expr.to_nnf()
            case And(args):
                return reduce(operator.__or__, [(~a).to_nnf() for a in args], Literal(False))
            case Or(args):
                return reduce(operator.__and__, [(~a).to_nnf() for a in args], Literal(True))
            case _:
                return arg.to_nnf()

    @override
    def expand(self) -> Expr:
        return ~(self.arg.expand())

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        return self.arg.horizon()


@final
@frozen
class Variable[Var](Expr):
    name: Var

    @override
    def __str__(self) -> str:
        return str(self.name)

    @override
    def to_nnf(self) -> Expr:
        return self

    @override
    def expand(self) -> Expr:
        return self

    @override
    def children(self) -> Iterator[Expr]:
        yield from ()

    @override
    def horizon(self) -> int | float:
        return 0


@final
@frozen
class Literal(Expr):
    value: bool

    @override
    def __str__(self) -> str:
        return "t" if self.value else "f"

    @override
    def __invert__(self) -> Literal:
        return Literal(not self.value)

    @override
    def __and__(self, other: Expr) -> Expr:
        if self.value is False:
            return self
        elif isinstance(other, Literal):
            return Literal(self.value and other.value)
        else:
            # True & x = x
            return other

    @override
    def __or__(self, other: Expr) -> Expr:
        if self.value is True:
            return self
        elif isinstance(other, Literal):
            return Literal(self.value or other.value)
        else:
            # False | x = x
            return other

    @override
    def to_nnf(self) -> Self:
        return self

    @override
    def expand(self) -> Self:
        return self

    @override
    def children(self) -> Iterator[Expr]:
        yield from ()

    @override
    def horizon(self) -> int | float:
        return 0


__all__ = [
    "Expr",
    "Implies",
    "Equiv",
    "Xor",
    "And",
    "Or",
    "Not",
    "Variable",
    "Literal",
]
