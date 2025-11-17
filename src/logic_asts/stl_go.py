r"""Abstract syntax trees for signal temporal logic with graph operators.

This module extends linear temporal logic with graph operators for specifying
properties of multi-agent systems over communication or interaction graphs.
Signal Temporal Logic with Graph Operators (STL-GO) allows quantification over
edges in graphs with weight and count constraints.

Core Graph Operators:
    - GraphIncoming: $\text{In}^{(W,\\#)}_\text{(G,E)} \phi$ - incoming edges
    - GraphOutgoing: $\text{Out}^{(W,\\#)}_\text{(G,E)} \psi$ - outgoing edges

Constraints on Graph Operators:
    Operators are parameterized by:
    - Graph types G (set of graph identifiers, e.g., {c, s, m})
    - Weight interval W: edge weights must fall in [w1, w2]
    - Edge count interval E: number of edges must be in [e1, e2]
    - Quantifier over graph types: exists or forall

Key Classes:
    - WeightInterval: Constraint on edge weights [w1, w2]
    - EdgeCountInterval: Constraint on edge counts [e1, e2]
    - Quantifier: Enum for existential/universal quantification
    - GraphIncoming: Incoming edge operator
    - GraphOutgoing: Outgoing edge operator

Integration with Temporal Logic:
    Graph operators can be nested with LTL operators (X, F, G, U) to specify
    temporal properties of multi-agent communication patterns.

Examples:
    - Incoming agreement: in^[0,1]{E}_{c}[1,n] consensus
    - Outgoing broadcast: G out^[1,inf]{A}_{s}[1,n] message
    - Temporal-graph: F in^[0.5,1]{E}_{c,s}[1,3] active
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterator
from typing import TypeAlias, TypeVar, final

import attrs
from attrs import frozen
from typing_extensions import override

from logic_asts.base import Expr
from logic_asts.ltl import LTLExpr
from logic_asts.utils import check_positive, check_start, check_weight_start


@final
@frozen
class WeightInterval:
    r"""Weight constraint for graph operators: interval $[w_1, w_2]$.

    Represents an interval of edge weights in graph operators. Weights are
    continuous (float) values, allowing both positive and negative weights.
    None represents unboundedness at that end.

    Attributes:
        start: Lower bound on weight (inclusive), or None for no lower bound
            (treated as $-\infty$).
        end: Upper bound on weight (inclusive), or None for unbounded.

    Examples:
        - Bounded positive: WeightInterval(0.5, 2.5)  represents [0.5, 2.5]
        - Negative weights: WeightInterval(-5.0, 2.5)  represents [-5.0, 2.5]
        - Left unbounded: WeightInterval(None, 1.0)    represents [-inf, 1.0]
        - Right unbounded: WeightInterval(0.1, None)   represents [0.1, inf)
        - Fully unbounded: WeightInterval(None, None)  represents [-inf, inf)

    Validators:
        - start must be <= end if both are non-None
        - Note: no point interval restriction (start can equal end)
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
        r"""Calculate the span of the weight interval.

        Computes the length of the interval, treating None as:
        - start = None as $-\infty$
        - end = None as $+\infty$

        Returns:
            The length $w_2 - w_1$ where $w_1$ is start and $w_2$ is end.
        """
        start = self.start or 0.0
        end = self.end or math.inf
        return end - start

    def is_unbounded(self) -> bool:
        r"""Check if the interval has no upper bound.

        Returns:
            True if end is None or infinity.
        """
        return self.end is None or math.isinf(self.end)

    def is_all_weights(self) -> bool:
        r"""Check if the interval represents all weights $[-\infty, +\infty]$.

        Returns:
            True if this interval covers all possible weight values.
        """
        return (self.start is None or (isinstance(self.start, float) and math.isinf(self.start))) and (
            self.end is None or math.isinf(self.end)
        )


@final
@frozen
class EdgeCountInterval:
    r"""Edge count constraint for graph operators: interval $[e_1, e_2]$.

    Represents an interval specifying the number of edges required in graph
    operators. Counts are non-negative integers. None represents unboundedness.

    Attributes:
        start: Lower bound on edge count (inclusive), or None for no lower bound.
            Defaults to 0 when None.
        end: Upper bound on edge count (inclusive), or None for unbounded.

    Examples:
        - Bounded: EdgeCountInterval(1, 5)      represents [1, 5]
        - Left unbounded: EdgeCountInterval(None, 10)  represents [0, 10]
        - Right unbounded: EdgeCountInterval(3, None)  represents [3, infinity)
        - Fully unbounded: EdgeCountInterval(None, None)  represents [0, infinity)

    Validators:
        - start and end must be non-negative integers
        - start must be <= end if both are non-None
        - No point intervals (start == end not allowed if both are non-None)
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
        r"""Calculate the span of the edge count interval.

        Computes the length of the interval, treating None as:
        - start = None as 0
        - end = None as infinity

        Returns:
            The length $e_2 - e_1$ where $e_1$ is start and $e_2$ is end.
        """
        start = self.start or 0
        end = self.end or math.inf
        return end - start

    def is_unbounded(self) -> bool:
        r"""Check if the interval has no upper bound.

        Returns:
            True if end is None or infinity.
        """
        return self.end is None or math.isinf(self.end)


@final
class Quantifier(enum.Enum):
    r"""Quantifier for existential/universal quantification in graph operators.

    Controls how quantification over graph types is handled in graph operators.
    When negating a graph operator, the quantifier is flipped as per De Morgan's
    laws.

    Members:
        EXISTS: Existential quantifier. At least one graph type satisfies
            the property. Represented as E or exists.
        FORALL: Universal quantifier. All graph types satisfy the property.
            Represented as A or forall.

    Semantics of Quantifier Negation:
        When negating a graph operator, existential becomes universal:
        $\neg(\exists g. \text{In}^W_\text{(g,E)} \phi) = \forall g. \text{In}^W_\text{(g,E)} \neg\phi$
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
        r"""Flip the quantifier for formula negation.

        Implements De Morgan's law for quantifiers: flipping existential to
        universal and vice versa.

        Returns:
            Quantifier.FORALL if self is EXISTS, Quantifier.EXISTS if FORALL.
        """
        match self:
            case Quantifier.EXISTS:
                return Quantifier.FORALL
            case Quantifier.FORALL:
                return Quantifier.EXISTS


@final
@frozen
class GraphIncoming(Expr):
    r"""Incoming graph operator: $\text{In}^{(W,\\#)}_\text{(G,E)} \phi$.

    Quantifies over incoming edges to an agent. Asserts that there exist
    (or for all, depending on quantifier) graph types G from which edges
    with specified weights and counts arrive, and the subformula holds at
    the source agents of those edges.

    The incoming operator counts incoming edges with weights in interval W
    from a set of graph types G, such that the number of incoming edges is
    in interval E, and the subformula holds at the source agents.

    Attributes:
        arg: Subformula $\phi$ to evaluate on agents with incoming edges.
        graphs: Set of graph type identifiers (e.g., {'c', 's'} for
            "communication" and "sensing" graphs).
        edge_count: Interval $E = [e_1, e_2]$ constraining the number of
            incoming edges required.
        weights: Interval $W = [w_1, w_2]$ constraining allowed edge weights.
        quantifier: Quantification type (EXISTS or FORALL) over graph types.
            EXISTS: at least one graph type; FORALL: all graph types.

    Examples:
        - Receive from consensus: in^[0,1]{E}_{c}[1,n] consensus
        - Always receive from all: G in^[-inf,inf]{A}_{s}[1,n] message
    """

    arg: Expr
    graphs: frozenset[str]
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
            graphs=frozenset(self.graphs),
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
            graphs=frozenset(self.graphs),
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
    r"""Outgoing graph operator: $\text{Out}^{(W,\\#)}_\text{(G,E)} \phi$.

    Quantifies over outgoing edges from an agent. Asserts that there exist
    (or for all, depending on quantifier) graph types G to which edges with
    specified weights and counts depart, and the subformula holds at the
    destination agents of those edges.

    The outgoing operator counts outgoing edges with weights in interval W
    to a set of graph types G, such that the number of outgoing edges is
    in interval E, and the subformula holds at the destination agents.

    Attributes:
        arg: Subformula $\phi$ to evaluate on agents with outgoing edges.
        graphs: Set of graph type identifiers (e.g., {'c', 's'} for
            "communication" and "sensing" graphs).
        edge_count: Interval $E = [e_1, e_2]$ constraining the number of
            outgoing edges required.
        weights: Interval $W = [w_1, w_2]$ constraining allowed edge weights.
        quantifier: Quantification type (EXISTS or FORALL) over graph types.
            EXISTS: at least one graph type; FORALL: all graph types.

    Examples:
        - Send to followers: out^[0,1]{E}_{c}[1,n] follower
        - Broadcast message: G out^[-inf,inf]{A}_{s}[1,n] received
    """

    arg: Expr
    graphs: frozenset[str]
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
            graphs=frozenset(self.graphs),
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
            graphs=frozenset(self.graphs),
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


Var = TypeVar("Var")
STLGOExpr: TypeAlias = LTLExpr[Var] | GraphIncoming | GraphOutgoing

__all__ = [
    "STLGOExpr",
    "WeightInterval",
    "EdgeCountInterval",
    "Quantifier",
    "GraphIncoming",
    "GraphOutgoing",
]
