from __future__ import annotations

import enum
import math
from collections.abc import Iterator
from typing import final

import attrs
from attrs import frozen
from typing_extensions import override

from logic_asts.base import Expr
from logic_asts.utils import check_positive, check_start, check_weight_start


@final
@frozen
class WeightInterval:
    """Interval for edge weights in graph operators.

    Represents an interval [w1, w2] where weights can be real numbers.
    Supports unbounded intervals: [-inf, inf], [0, inf], etc.
    """

    start: float | None = attrs.field(default=None, validator=[check_weight_start])
    end: float | None = attrs.field(default=None)

    @override
    def __str__(self) -> str:
        match (self.start, self.end):
            case None, None:
                return ""
            case _:
                return f"[{self.start or ''}, {self.end or ''}]"

    def duration(self) -> float | int:
        start = self.start or 0.0
        end = self.end or math.inf
        return end - start

    def is_unbounded(self) -> bool:
        return self.end is None or math.isinf(self.end)

    def is_all_weights(self) -> bool:
        """If the interval is [-inf, inf]"""
        return (self.start is None or (isinstance(self.start, float) and math.isinf(self.start))) and (
            self.end is None or math.isinf(self.end)
        )


@final
@frozen
class EdgeCountInterval:
    """Interval for edge counts in graph operators.

    Represents an interval [e1, e2] where counts must be non-negative integers.
    Supports unbounded intervals: [0, inf], [1, inf], etc.
    """

    start: int | None = attrs.field(default=None, validator=[check_positive, check_start])
    end: int | None = attrs.field(default=None, validator=[check_positive])

    @override
    def __str__(self) -> str:
        match (self.start, self.end):
            case None, None:
                return ""
            case _:
                return f"[{self.start or ''}, {self.end or ''}]"

    def duration(self) -> int | float:
        start = self.start or 0
        end = self.end or math.inf
        return end - start

    def is_unbounded(self) -> bool:
        return self.end is None or math.isinf(self.end)


@final
class Quantifier(enum.Enum):
    """Quantifier for graph operators in STL-GO.

    - EXISTS: At least one graph satisfies the property
    - FORALL: All graphs satisfy the property
    """

    EXISTS = "exists"
    FORALL = "forall"

    @override
    def __str__(self) -> str:
        match self:
            case Quantifier.EXISTS:
                return "exists"
            case Quantifier.FORALL:
                return "forall"

    def negate(self) -> Quantifier:
        """Flip the quantifier during negation.

        When negating a graph operator, existential becomes universal and vice versa.

        ```
        ~in^(W,exists)_(G,E) phi = In^(W,forall)_(G,E) ~phi
        ```
        """
        match self:
            case Quantifier.EXISTS:
                return Quantifier.FORALL
            case Quantifier.FORALL:
                return Quantifier.EXISTS


@final
@frozen
class GraphIncoming(Expr):
    """Incoming graph operator: In^(W,#)_(G,E) phi

    Count agents sending to agent i via graph type G with weights in W,
    such that at least E agents satisfy phi.

    Parameters:
    - arg: Subformula phi to evaluate on incoming agents
    - graphs: Set of graph types (e.g., {'c', 's', 'm', 'd'})
    - edge_count: Interval E = [e1, e2] for number of neighbors
    - weights: Interval W = [w1, w2] for edge weight constraints
    - quantifier: `# in {exists, forall}` - quantification over graph types
    """

    arg: Expr
    graphs: frozenset[str] = attrs.field(converter=frozenset)
    edge_count: EdgeCountInterval
    weights: WeightInterval
    quantifier: Quantifier

    @override
    def __str__(self) -> str:
        graphs_str = "{" + ",".join(sorted(self.graphs)) + "}"
        return f"(In^{{{self.weights},{self.quantifier}}}_{{{graphs_str},{self.edge_count}}} {self.arg})"

    @override
    def expand(self) -> Expr:
        """Graph operators don't expand further; recursively expand subformula."""
        return GraphIncoming(
            arg=self.arg.expand(),
            graphs=self.graphs,
            edge_count=self.edge_count,
            weights=self.weights,
            quantifier=self.quantifier,
        )

    @override
    def to_nnf(self) -> Expr:
        """Convert to Negation Normal Form.

        When negating a graph operator, the quantifier flips but the rest remains.
        This is handled by the Not class, which will call children() to apply negation.
        """
        return GraphIncoming(
            arg=self.arg.to_nnf(),
            graphs=self.graphs,
            edge_count=self.edge_count,
            weights=self.weights,
            quantifier=self.quantifier,
        )

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        """Horizon of graph operators depends on the subformula."""
        return self.arg.horizon()


@final
@frozen
class GraphOutgoing(Expr):
    """Outgoing graph operator: Out^(W,#)_(G,E) phi

    Count agents receiving from agent i via graph type G with weights in W,
    such that at least E agents satisfy phi.

    Parameters:
    - arg: Subformula phi to evaluate on outgoing agents
    - graphs: Set of graph types (e.g., {'c', 's', 'm', 'd'})
    - edge_count: Interval E = [e1, e2] for number of neighbors
    - weights: Interval W = [w1, w2] for edge weight constraints
    - quantifier: `# in {exists, forall}` - quantification over graph types
    """

    arg: Expr
    graphs: frozenset[str] = attrs.field(converter=frozenset)
    edge_count: EdgeCountInterval
    weights: WeightInterval
    quantifier: Quantifier

    @override
    def __str__(self) -> str:
        graphs_str = "{" + ",".join(sorted(self.graphs)) + "}"
        return f"(Out^{{{self.weights},{self.quantifier}}}_{{{graphs_str},{self.edge_count}}} {self.arg})"

    @override
    def expand(self) -> Expr:
        """Graph operators don't expand further; recursively expand subformula."""
        return GraphOutgoing(
            arg=self.arg.expand(),
            graphs=self.graphs,
            edge_count=self.edge_count,
            weights=self.weights,
            quantifier=self.quantifier,
        )

    @override
    def to_nnf(self) -> Expr:
        """Convert to Negation Normal Form.

        When negating a graph operator, the quantifier flips but the rest remains.
        This is handled by the Not class, which will call children() to apply negation.
        """
        return GraphOutgoing(
            arg=self.arg.to_nnf(),
            graphs=self.graphs,
            edge_count=self.edge_count,
            weights=self.weights,
            quantifier=self.quantifier,
        )

    @override
    def children(self) -> Iterator[Expr]:
        yield self.arg

    @override
    def horizon(self) -> int | float:
        """Horizon of graph operators depends on the subformula."""
        return self.arg.horizon()


__all__ = [
    "WeightInterval",
    "EdgeCountInterval",
    "Quantifier",
    "GraphIncoming",
    "GraphOutgoing",
]
