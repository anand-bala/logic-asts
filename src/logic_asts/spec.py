"""Abstract and generalized specifications for various library constructs"""

from __future__ import annotations

import typing as ty
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Collection, Hashable, Iterator
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar

from typing_extensions import overload

Var = TypeVar("Var", bound=Hashable)

if TYPE_CHECKING:
    from logic_asts.base import Not, Variable


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
        import logic_asts.utils

        return logic_asts.utils.to_nnf(self)

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
        return iter(ExprVisitor[Expr](Expr, self))

    @overload
    def atomic_predicates(self, *, assume_nnf: ty.Literal[True]) -> Iterator[Variable[Var] | Not]: ...
    @overload
    def atomic_predicates(self, *, assume_nnf: ty.Literal[False]) -> Iterator[Variable[Var]]: ...

    def atomic_predicates(self, *, assume_nnf: bool = False) -> Iterator[Variable[Var] | Not]:
        """Yield all atomic predicates (variables) in the expression.

        Performs a traversal of the expression tree and yields all variable
        occurrences. Each unique variable object is yielded only once, even if
        it appears multiple times in the expression.

        Args:
            assume_nnf: If True, treats negated variables `Not(Variable(name))`
                as atomic predicates (yields both Variable and Not nodes).
                If False, only yields Variable nodes and traverses inside Not.
                Defaults to False.

        Yields:
            Variable[Var] | Not: When assume_nnf is True, yields both Variable
                instances and Not(Variable) instances. When assume_nnf is False,
                yields only Variable instances.

        Examples:
            Basic usage (yields variables only):
            >>> from logic_asts.base import Variable
            >>> p = Variable("p")
            >>> q = Variable("q")
            >>> expr = (p & q) | ~p
            >>> atoms = set(expr.atomic_predicates())
            >>> sorted(v.name for v in atoms)
            ['p', 'q']

            With NNF assumption (yields negated variables as atoms):
            >>> expr_nnf = (p & q) | ~p
            >>> atoms = set(expr_nnf.atomic_predicates(assume_nnf=True))
            >>> len(atoms)  # Should have 3 atoms: p, q, and Not(p)
            3
            >>> p in atoms
            True
            >>> q in atoms
            True

        Note:
            This method visits each node at most once, so duplicate variable
            references in the tree will only yield one result per unique object.
        """
        from logic_asts.base import Not, Variable

        stack: deque[Expr] = deque([self])
        visited: set[Expr] = set()

        while stack:
            subexpr = stack[-1]
            need_to_visit_children: set[Expr] = set()
            if assume_nnf and isinstance(subexpr, Not) and isinstance(subexpr.arg, Variable):
                yield subexpr
            elif isinstance(subexpr, Variable):
                yield subexpr
            else:
                # Either assume nnf is false or we are looking at a non-atom
                need_to_visit_children = {
                    child
                    for child in subexpr.children()  # We need to visit `child`
                    if child not in visited  # if it hasn't already been visited
                }

            if visited.issuperset(need_to_visit_children):
                # subexpr is a leaf (the set is empty) or it's children have been
                # yielded get rid of it from the stack
                stack.pop()
                # Add subexpr to visited
                visited.add(subexpr)
            else:
                # mid-level node or an empty set
                # Add relevant children to stack
                stack.extend(need_to_visit_children)

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
        from logic_asts.base import Not

        return Not(self)

    def __and__(self, other: Expr) -> Expr:
        r"""Logical conjunction operator (&).

        Returns:
            An And expression joining this and other.
        """
        from logic_asts.base import And

        return And((self, other))

    def __or__(self, other: Expr) -> Expr:
        r"""Logical disjunction operator (|).

        Returns:
            An Or expression joining this and other.
        """
        from logic_asts.base import Or

        return Or((self, other))


_T = TypeVar("_T", bound=Expr)
_SomeExprType: TypeAlias = type[_T] | (type[_T] | type[_T]) | Collection[type[_T]]  # noqa: PYI016


def _validate_and_normalize(tp: object) -> tuple[type[Expr], ...]:
    # Union[A, B] or A | B
    if ty.get_origin(tp) is not None:
        raw_types = ty.get_args(tp)

    # Collection[type[A]]
    elif isinstance(tp, Collection):
        raw_types = tuple(tp)

    # Single type
    else:
        raw_types = (tp,)

    normalized: list[type[Expr]] = []

    for t in raw_types:
        # Erase generic parameters: Variable[int] -> Variable
        if (origin := ty.get_origin(t)) is not None:
            t = origin

        if not isinstance(t, type):
            raise TypeError(f"{t!r} is not a type")

        if not issubclass(t, Expr):
            raise TypeError(f"{t.__name__} is not a subclass of Expr")

        normalized.append(t)

    # Deduplicate, preserve order
    return tuple(dict.fromkeys(normalized))


class ExprVisitor(Generic[_T]):
    """A generic Expr visitor that performs post-order traversal of the expression tree.

    Iterates over all sub-expressions in post-order, visiting each
    expression exactly once. In post-order, children are yielded before
    their parents, making this suitable for bottom-up processing.

    Moreover, it ensures that each subexpression is of the given `_T` type parameter.

    Yields:
        Each node in the expression tree in post-order sequence.

    Raises:
        TypeError: If the expression contains a subexpression that is not of the specified type
    """

    def __init__(self, expr_type: _SomeExprType[_T], expr: _T) -> None:
        self._types = _validate_and_normalize(expr_type)
        self._expr = expr

    def _is_expected(self, expr: Expr) -> ty.TypeGuard[_T]:
        return isinstance(expr, self._types)

    def __iter__(self) -> Iterator[_T]:
        stack: deque[_T] = deque([self._expr])
        visited: set[_T] = set()

        while stack:
            subexpr = stack[-1]
            unvisited_children = {
                child
                for child in subexpr.children()  # We need to visit `child`
                if child not in visited  # if it hasn't already been visited
            }

            if not unvisited_children:
                # subexpr is a leaf (the set is empty) or it's children have been
                # yielded get rid of it from the stack
                stack.pop()
                # Add subexpr to visited
                visited.add(subexpr)
                # post-order return it
                yield subexpr
            else:
                # mid-level node or an empty set
                # Add relevant children to stack
                # After checking if they are the correct type
                for e in unvisited_children:
                    if not self._is_expected(e):
                        raise TypeError(f"Expected expression of type {self._types}, got {type(e).__name__}")
                    stack.append(e)
        # Yield the remaining nodes in the stack in reverse order
        yield from reversed(stack)
