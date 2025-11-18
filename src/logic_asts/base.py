r"""Abstract syntax trees for propositional logic.

This module provides a complete abstract syntax tree (AST) representation for
propositional logic, including atomic propositions (variables and literals) and
logical operators (conjunction, disjunction, negation, implication, biconditional,
and exclusive or).

Examples:
    Create a simple formula: (p & q) | ~r
    >>> p = Variable("p")
    >>> q = Variable("q")
    >>> r = Variable("r")
    >>> formula = (p & q) | ~r

    Evaluate with truth assignment:
    >>> truth_assignment = {"p", "q"}  # p=true, q=true, r=false
    >>> simple_eval(formula, truth_assignment)
    True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Hashable, Iterator
from collections.abc import Set as AbstractSet
from typing import Generic, TypeAlias, TypeVar, final

import attrs
from attrs import field, frozen
from typing_extensions import Self, override


class Expr(ABC):
    """Abstract base class for logical expressions."""

    @abstractmethod
    def expand(self) -> Expr:
        r"""Expand derived operators to basic form.

        Derived operators (Implies, Equiv, Xor) are expanded to their
        definitions in terms of And, Or, and Not:
        - $\phi \to \psi \equiv \neg\phi \vee \psi$
        - $\phi \equiv \psi \equiv (\phi \vee \neg\psi) \wedge (\neg\phi \vee \psi)$
        - $\phi \oplus \psi \equiv (\phi \wedge \neg\psi) \vee (\neg\phi \wedge \psi)$

        Returns:
            An equivalent expression using only And, Or, Not, Variable, and Literal.
        """

    @abstractmethod
    def to_nnf(self) -> Expr:
        r"""Convert to Negation Normal Form (NNF).

        NNF is a canonical form where negation appears only over atomic
        propositions (variables and literals). This is achieved by:
        - Applying De Morgan's laws to push negations inward
        - Expanding derived operators using expand()
        - Eliminating double negations

        The result is logically equivalent to the original expression.

        Returns:
            An expression in NNF with negations only over atoms.
        """

    @abstractmethod
    def children(self) -> Iterator[Expr]:
        r"""Iterate over immediate child expressions.

        Returns an iterator of the direct sub-expressions of this expression.
        For leaf nodes (Variable, Literal), yields nothing.

        Returns:
            Iterator of child expressions.
        """

    def iter_subtree(self) -> Iterator[Expr]:
        r"""Perform post-order traversal of the expression tree.

        Iterates over all sub-expressions in post-order, visiting each
        expression exactly once. In post-order, children are yielded before
        their parents, making this suitable for bottom-up processing.

        Yields:
            Each node in the expression tree in post-order sequence.
        """
        stack: deque[Expr] = deque([self])
        visited: set[Expr] = set()

        while stack:
            subexpr = stack[-1]
            need_to_visit_children = {
                child
                for child in subexpr.children()  # We need to visit `child`
                if (child) not in visited  # if it hasn't already been visited
            }

            if visited.issuperset(need_to_visit_children):
                # subexpr is a leaf (the set is empty) or it's children have been
                # yielded get rid of it from the stack
                _ = stack.pop()
                # Add subexpr to visited
                visited.add(subexpr)
                # post-order return it
                yield subexpr
            else:
                # mid-level node or an empty set
                # Add relevant children to stack
                stack.extend(need_to_visit_children)
        # Yield the remaining nodes in the stack in reverse order
        yield from reversed(stack)

    @abstractmethod
    def horizon(self) -> int | float:
        r"""Compute the lookahead depth required for this formula.

        For propositional logic, horizon is always 0 (no temporal lookahead).
        Subclasses extending to temporal logics may return positive values.

        Returns:
            Non-negative integer or float('inf') for unbounded formulas.
        """

    def __invert__(self) -> Expr:
        r"""Logical negation operator (~).

        Returns:
            A Not expression wrapping this expression.
        """
        return Not(self)

    def __and__(self, other: Expr) -> Expr:
        r"""Logical conjunction operator (&).

        Returns:
            An And expression joining this and other.
        """
        return And((self, other))

    def __or__(self, other: Expr) -> Expr:
        r"""Logical disjunction operator (|).

        Returns:
            An Or expression joining this and other.
        """
        return Or((self, other))


@final
@frozen
class Implies(Expr):
    r"""Logical implication operator: $\phi \to \psi$.

    Represents "if phi then psi" or equivalently "not phi or psi".
    This is a derived operator that can be expanded to its basic form.

    Attributes:
        lhs: Left-hand side formula ($\phi$, the antecedent).
        rhs: Right-hand side formula ($\psi$, the consequent).
    """

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
    r"""Logical equivalence operator: $\phi \equiv \psi$.

    Represents "phi if and only if psi" or equivalently
    $(\phi \land \psi) \lor (\neg\phi \land \neg\psi)$.
    This is a derived operator that can be expanded to its basic form.

    Attributes:
        lhs: Left-hand side formula
        rhs: Right-hand side formula
    """

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
    r"""Exclusive or operator: $\phi \oplus \psi$.

    Represents "phi or psi but not both" or equivalently
    $(\phi \land \neg\psi) \lor (\neg \phi \land \psi)$.
    This is a derived operator that can be expanded to its basic form.

    Attributes:
        lhs: Left-hand side formula ($\phi$).
        rhs: Right-hand side formula ($\psi$).
    """

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
    r"""Conjunction operator: $\phi_1 \wedge \phi_2 \wedge \cdots \wedge \phi_n$.

    Represents the logical conjunction of multiple formulas. Requires at least
    two operands. Can be created using the `&` operator.

    Attributes:
        args: Tuple of at least 2 sub-expressions to be conjoined.

    Examples:
        - Using operator: `p & q & r`
        - Using constructor: `And((p, q, r))`
    """

    args: tuple[Expr, ...] = field(validator=attrs.validators.min_len(2))

    @override
    def __str__(self) -> str:
        return "(" + " & ".join(str(arg) for arg in self.children()) + ")"

    @override
    def to_nnf(self) -> Expr:
        acc: Expr = Literal(True)
        [acc := acc & a.to_nnf() for a in self.args]
        return acc

    @override
    def expand(self) -> Expr:
        acc: Expr = Literal(True)
        [acc := acc & a.expand() for a in self.args]
        return acc

    @override
    def children(self) -> Iterator[Expr]:
        yield from self.args

    @override
    def horizon(self) -> int | float:
        return max(arg.horizon() for arg in self.args)

    @override
    def __and__(self, other: Expr) -> Expr:
        if isinstance(other, And):
            return And(self.args + other.args)
        return And(self.args + (other,))


@final
@frozen
class Or(Expr):
    r"""Disjunction operator: $\phi_1 \vee \phi_2 \vee \cdots \vee \phi_n$.

    Represents the logical disjunction of multiple formulas. Requires at least
    two operands. Can be created using the `|` operator.

    Attributes:
        args: Tuple of at least 2 sub-expressions to be disjoined.

    Examples:
        - Using operator: `p | q | r`
        - Using constructor: `Or((p, q, r))`
    """

    args: tuple[Expr, ...] = field(validator=attrs.validators.min_len(2))

    @override
    def __str__(self) -> str:
        return "(" + " | ".join(str(arg) for arg in self.children()) + ")"

    @override
    def to_nnf(self) -> Expr:
        acc: Expr = Literal(False)
        [acc := acc | a.to_nnf() for a in self.args]
        return acc

    @override
    def expand(self) -> Expr:
        acc: Expr = Literal(False)
        [acc := acc | a.expand() for a in self.args]
        return acc

    @override
    def children(self) -> Iterator[Expr]:
        yield from self.args

    @override
    def horizon(self) -> int | float:
        return max(arg.horizon() for arg in self.args)

    @override
    def __or__(self, other: Expr) -> Expr:
        if isinstance(other, Or):
            return Or(self.args + other.args)
        return Or(self.args + (other,))


@final
@frozen
class Not(Expr):
    r"""Negation operator: $\neg\phi$.

    Represents the logical negation of a formula. Can be created using the `~` operator.
    Supports double-negation elimination.

    Attributes:
        arg: The sub-expression to be negated.

    Examples:
        - Using operator: `~p`
        - Using constructor: `Not(Variable("p"))`
    """

    arg: Expr

    @override
    def __str__(self) -> str:
        return f"!{str(self.arg)}"

    @override
    def __invert__(self) -> Expr:
        r"""Eliminate double negation: $\neg(\neg\phi) = \phi$."""
        return self.arg

    @override
    def to_nnf(self) -> Expr:
        arg = self.arg
        ret: Expr
        match arg:
            case Literal():
                return ~arg
            case Variable():
                return self
            case Not(expr):
                return expr.to_nnf()
            case And(args):
                ret = Literal(False)
                [ret := ret | (~a).to_nnf() for a in args]
                return ret
            case Or(args):
                ret = Literal(True)
                [ret := ret & (~a).to_nnf() for a in args]
                return ret
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


Var = TypeVar("Var", bound=Hashable)


@final
@frozen
class Variable(Expr, Generic[Var]):
    r"""A named atomic proposition.

    Represents a variable that can be assigned true or false. Variables are
    generic and can hold any hashable type as their name.

    Type Parameters:
        Var: The type of the variable name (e.g., str, tuple, int).

    Attributes:
        name: The variable identifier of type Var.

    Examples:
        - String variables: `Variable("p")`, `Variable("x1")`
        - Tuple variables: `Variable(("a", 0))`
        - Integer variables: `Variable(42)`
    """

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
        yield from iter(())

    @override
    def horizon(self) -> int | float:
        return 0


@final
@frozen
class Literal(Expr):
    r"""Boolean literals.

    Represents a fixed boolean value (true or false) as an atomic proposition.
    Supports logical operations and optimizations (e.g., true & x = x).

    Attributes:
        value: Boolean value, either True or False.

    Examples:
        - True constant: `Literal(True)`
        - False constant: `Literal(False)`
        - Negation: `~Literal(True)` returns `Literal(False)`
    """

    value: bool

    @override
    def __str__(self) -> str:
        return "t" if self.value else "f"

    @override
    def __invert__(self) -> Literal:
        r"""Negate the literal: $\neg\text{true} = \text{false}$."""
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
        yield from iter(())

    @override
    def horizon(self) -> int | float:
        return 0


BaseExpr: TypeAlias = Implies | Equiv | Xor | And | Or | Not | Variable[Var] | Literal
"""Propositional logic expression types"""


def simple_eval(expr: BaseExpr[Var], input: AbstractSet[Var]) -> bool:
    r"""Evaluate a propositional formula under a given truth assignment.

    Performs a bottom-up evaluation of a propositional formula by post-order
    traversal. Each variable in the input set is assigned true; all others
    are assigned false.

    Type Parameters:
        Var: The variable type used in the expression.

    Arguments:
        expr: The propositional formula to evaluate (must not contain temporal
            operators).
        input: Set of variable names that should evaluate to true.

    Returns:
        The boolean result of evaluating the formula.

    Raises:
        TypeError: If the expression contains operators not in propositional logic.

    Examples:
        >>> p = Variable("p")
        >>> q = Variable("q")
        >>> formula = p & q
        >>> simple_eval(formula, {"p", "q"})
        True
        >>> simple_eval(formula, {"p"})
        False
    """

    cache: dict[BaseExpr[Var], bool] = dict()
    for subexpr in expr.iter_subtree():
        match subexpr:
            case Literal(value):
                cache[subexpr] = value
            case Variable(name):
                cache[subexpr] = name in input
            case Not(arg):
                cache[subexpr] = not cache[arg]  # type: ignore
            case Or(args):
                cache[subexpr] = any(cache[arg] for arg in args)  # type: ignore
            case And(args):
                cache[subexpr] = all(cache[arg] for arg in args)  # type: ignore
            case Xor(lhs, rhs):
                cache[subexpr] = cache[lhs] != cache[rhs]  # type: ignore[index]
            case Equiv(lhs, rhs):
                cache[subexpr] = cache[lhs] == cache[rhs]  # type: ignore[index]
            case Implies(p, q):
                cache[subexpr] = (not cache[p]) or cache[q]  # type: ignore[index]
            case _:
                raise TypeError(f"simple evaluation only possible for propositional logic expressions, got {type(subexpr)}")

    return cache[expr]


__all__ = [
    "BaseExpr",
    "Literal",
    "Variable",
    "Not",
    "Or",
    "And",
    "Xor",
    "Equiv",
    "Implies",
]

__docformat__ = "google"
