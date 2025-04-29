from __future__ import annotations

import operator
import sys
import typing
from functools import reduce
from pathlib import Path

import attrs
from attrs import field, frozen
from lark import Lark, Token, Transformer, v_args
from typing_extensions import TypeIs, final, override

type PropExpr[Var] = Literal | Variable[Var] | Not[Var] | Or[Var] | And[Var] | Implies[Var] | Xor[Var] | Equiv[Var]


def _default_invert[Var](expr: PropExpr[Var]) -> PropExpr[Var]:
    match expr:
        case Literal(value):
            return Literal(not value)
        case Not(arg):
            return arg
        case _:
            return Not(expr)


def _default_and[Var](lhs: PropExpr[Var], rhs: PropExpr[Var]) -> PropExpr[Var]:
    match (lhs, rhs):
        case (Literal(False), _) | (_, Literal(False)):
            return Literal(False)
        case (Literal(True), expr) | (expr, Literal(True)):
            return expr
        case (And(lhs_args), And(rhs_args)):
            return And(lhs_args + rhs_args)
        case (And(args), expr) | (expr, And(args)):
            return And(args + [expr])
        case _:
            return And([lhs, rhs])


def _default_or[Var](lhs: PropExpr[Var], rhs: PropExpr[Var]) -> PropExpr[Var]:
    match (lhs, rhs):
        case (Literal(False), _) | (_, Literal(False)):
            return Literal(False)
        case (Literal(True), expr) | (expr, Literal(True)):
            return expr
        case (Or(lhs_args), Or(rhs_args)):
            return Or(lhs_args + rhs_args)
        case (Or(args), expr) | (expr, Or(args)):
            return Or(args + [expr])
        case _:
            return Or([lhs, rhs])


def is_literal[Var](expr: PropExpr[Var]) -> TypeIs[Variable[Var] | Literal]:
    return isinstance(expr, (Literal, Variable))


@final
@frozen
class Implies[Var]:
    lhs: PropExpr[Var]
    rhs: PropExpr[Var]

    def __invert__(self) -> PropExpr[Var]:
        return _default_invert(self)

    def __and__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_and(self, other)

    def __or__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_or(self, other)

    @override
    def __str__(self) -> str:
        return f"{self.lhs} -> {self.rhs}"

    def expand(self) -> PropExpr[Var]:
        return ~self.lhs | self.rhs

    def to_nnf(self) -> PropExpr[Var]:
        return self.expand().to_nnf()


@final
@frozen
class Equiv[Var]:
    lhs: PropExpr[Var]
    rhs: PropExpr[Var]

    def __invert__(self) -> PropExpr[Var]:
        return _default_invert(self)

    def __and__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_and(self, other)

    def __or__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_or(self, other)

    @override
    def __str__(self) -> str:
        return f"{self.lhs} <-> {self.rhs}"

    def expand(self) -> PropExpr[Var]:
        x = self.lhs
        y = self.rhs
        return (x | ~y) & (~x | y)

    def to_nnf(self) -> PropExpr[Var]:
        return self.expand().to_nnf()


@final
@frozen
class Xor[Var]:
    lhs: PropExpr[Var]
    rhs: PropExpr[Var]

    def __invert__(self) -> PropExpr[Var]:
        return _default_invert(self)

    def __and__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_and(self, other)

    def __or__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_or(self, other)

    @override
    def __str__(self) -> str:
        return f"{self.lhs} ^ {self.rhs}"

    def expand(self) -> PropExpr[Var]:
        x = self.lhs
        y = self.rhs
        return (x & ~y) | (~x & y)

    def to_nnf(self) -> PropExpr[Var]:
        return self.expand().to_nnf()


@final
@frozen
class And[Var]:
    args: list[PropExpr[Var]] = field(validator=attrs.validators.min_len(2))

    def __invert__(self) -> PropExpr[Var]:
        return _default_invert(self)

    def __and__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_and(self, other)

    def __or__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_or(self, other)

    @override
    def __str__(self) -> str:
        return "(" + " & ".join(str(arg) for arg in self.args) + ")"

    def to_nnf(self) -> PropExpr[Var]:
        return reduce(operator.__and__, (a.to_nnf() for a in self.args))

    def expand(self) -> PropExpr[Var]:
        return reduce(operator.__and__, (a.expand() for a in self.args), Literal(True))


@final
@frozen
class Or[Var]:
    args: list[PropExpr[Var]] = field(validator=attrs.validators.min_len(2))

    def __invert__(self) -> PropExpr[Var]:
        return _default_invert(self)

    def __and__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_and(self, other)

    def __or__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_or(self, other)

    @override
    def __str__(self) -> str:
        return "(" + " | ".join(str(arg) for arg in self.args) + ")"

    def to_nnf(self) -> PropExpr[Var]:
        return reduce(operator.__or__, (a.to_nnf() for a in self.args), Literal(False))

    def expand(self) -> PropExpr[Var]:
        return reduce(operator.__or__, (a.expand() for a in self.args), Literal(False))


@final
@frozen
class Not[Var]:
    arg: PropExpr[Var]

    def __invert__(self) -> PropExpr[Var]:
        return _default_invert(self)

    def __and__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_and(self, other)

    def __or__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_or(self, other)

    @override
    def __str__(self) -> str:
        return f"!{str(self.arg)}"

    def to_nnf(self) -> PropExpr[Var]:
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

    def expand(self) -> PropExpr[Var]:
        return ~(self.arg.expand())


@final
@frozen
class Variable[Var]:
    name: Var

    def __invert__(self) -> PropExpr[Var]:
        return _default_invert(self)

    def __and__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_and(self, other)

    def __or__(self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_or(self, other)

    @override
    def __str__(self) -> str:
        return str(self.name)

    def to_nnf(self) -> PropExpr[Var]:
        return self

    def expand(self) -> PropExpr[Var]:
        return self


@final
@frozen
class Literal:
    value: bool

    @override
    def __str__(self) -> str:
        return "t" if self.value else "f"

    def __invert__(self) -> Literal:
        return Literal(not self.value)

    def __and__[Var](self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_and(self, other)

    def __or__[Var](self, other: PropExpr[Var]) -> PropExpr[Var]:
        return _default_or(self, other)

    def to_nnf(self) -> typing.Self:
        return self

    def expand(self) -> typing.Self:
        return self


@final
@v_args(inline=True)
class PropLogicTransform(Transformer[Token, PropExpr[str]]):
    mul = operator.and_
    neg = operator.invert
    add = operator.or_
    xor = Xor
    equiv = Equiv
    implies = Implies

    def var(self, value: Token | str) -> Variable[str]:
        return Variable(str(value))

    def literal(self, value: typing.Literal["0", "1", "TRUE", "FALSE"]) -> PropExpr[str]:
        match value:
            case "0" | "FALSE":
                return Literal(False)
            case "1" | "TRUE":
                return Literal(True)


_grammar: Lark | None = None


def parse(expr: str) -> PropExpr[str]:
    global _grammar
    file = Path(__file__)
    name = file.stem
    grammar_file = file.parent / f"{name}.lark"
    if _grammar is None:
        with open(grammar_file, "r") as gf:
            _grammar = Lark(gf.read(), start="expr")

    tree = _grammar.parse(expr)
    transformer = PropLogicTransform()
    ret: PropExpr[str] = transformer.transform(tree)
    return ret


def main() -> None:
    print(repr(parse(sys.argv[1])))


if __name__ == "__main__":
    main()
