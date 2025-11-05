"""Tests for STL-GO grammar and AST classes."""

import math

import pytest

from logic_asts import parse_expr
from logic_asts.base import Not, Variable
from logic_asts.ltl import Eventually, TimeInterval
from logic_asts.stl_go import (
    EdgeCountInterval,
    GraphIncoming,
    GraphOutgoing,
    Quantifier,
    WeightInterval,
)


class TestWeightInterval:
    """Tests for WeightInterval class."""

    def test_weight_interval_creation(self) -> None:
        """Test creating weight intervals."""
        wi = WeightInterval(0.5, 2.5)
        assert wi.start == 0.5
        assert wi.end == 2.5

    def test_weight_interval_unbounded(self) -> None:
        """Test unbounded weight intervals."""
        wi = WeightInterval(None, None)
        assert wi.start is None
        assert wi.end is None
        assert wi.is_unbounded()

    def test_weight_interval_left_unbounded(self) -> None:
        """Test left-unbounded weight intervals with negative infinity."""
        # Negative weights are allowed, so -infinity should work
        wi = WeightInterval(-5.0, 2.5)
        assert wi.start == -5.0
        assert wi.end == 2.5

    def test_weight_interval_right_unbounded(self) -> None:
        """Test right-unbounded weight intervals."""
        wi = WeightInterval(0.5, float("inf"))
        assert wi.start == 0.5
        assert wi.end is not None and math.isinf(wi.end)

    def test_weight_interval_str(self) -> None:
        """Test string representation of weight intervals."""
        wi1 = WeightInterval(0.5, 2.5)
        assert str(wi1) == "[0.5, 2.5]"

        wi2 = WeightInterval(None, None)
        assert str(wi2) == ""

    def test_weight_interval_duration(self) -> None:
        """Test duration calculation for weight intervals."""
        wi = WeightInterval(0.5, 2.5)
        assert wi.duration() == 2.0

    def test_weight_interval_invalid(self) -> None:
        """Test invalid weight intervals."""
        with pytest.raises(ValueError):
            _ = WeightInterval(2.5, 0.5)  # start > end

    def test_weight_interval_negative_start(self) -> None:
        """Test weight intervals with negative start."""
        wi = WeightInterval(-5.0, 2.5)
        assert wi.start == -5.0
        assert wi.end == 2.5


class TestEdgeCountInterval:
    """Tests for EdgeCountInterval class."""

    def test_edge_count_interval_creation(self) -> None:
        """Test creating edge count intervals."""
        ei = EdgeCountInterval(1, 5)
        assert ei.start == 1
        assert ei.end == 5

    def test_edge_count_interval_unbounded(self) -> None:
        """Test unbounded edge count intervals."""
        ei = EdgeCountInterval(0, None)
        assert ei.start == 0
        assert ei.end is None
        assert ei.is_unbounded()

    def test_edge_count_interval_str(self) -> None:
        """Test string representation of edge count intervals."""
        ei = EdgeCountInterval(1, 5)
        assert str(ei) == "[1, 5]"

    def test_edge_count_interval_duration(self) -> None:
        """Test duration calculation for edge count intervals."""
        ei = EdgeCountInterval(1, 5)
        assert ei.duration() == 4

    def test_edge_count_interval_invalid(self) -> None:
        """Test invalid edge count intervals."""
        with pytest.raises(ValueError):
            _ = EdgeCountInterval(5, 1)  # start > end

    def test_edge_count_interval_negative(self) -> None:
        """Test that negative values are rejected."""
        with pytest.raises(ValueError):
            _ = EdgeCountInterval(-1, 5)


class TestQuantifier:
    """Tests for Quantifier enum."""

    def test_quantifier_exists(self) -> None:
        """Test EXISTS quantifier."""
        assert Quantifier.EXISTS.value == "exists"
        assert str(Quantifier.EXISTS) == "exists"

    def test_quantifier_forall(self) -> None:
        """Test FORALL quantifier."""
        assert Quantifier.FORALL.value == "forall"
        assert str(Quantifier.FORALL) == "forall"

    def test_quantifier_negate_exists(self) -> None:
        """Test negating EXISTS quantifier."""
        assert Quantifier.EXISTS.negate() == Quantifier.FORALL

    def test_quantifier_negate_forall(self) -> None:
        """Test negating FORALL quantifier."""
        assert Quantifier.FORALL.negate() == Quantifier.EXISTS


class TestGraphIncoming:
    """Tests for GraphIncoming class."""

    def test_graph_incoming_creation(self) -> None:
        """Test creating GraphIncoming expressions."""
        p = Variable("p")
        wi = WeightInterval(0.5, 2.5)
        ei = EdgeCountInterval(1, 5)
        graphs = frozenset(["c", "s"])

        gi = GraphIncoming(
            arg=p,
            graphs=graphs,
            edge_count=ei,
            weights=wi,
            quantifier=Quantifier.EXISTS,
        )

        assert gi.arg is p
        assert gi.graphs == graphs
        assert gi.edge_count is ei
        assert gi.weights is wi
        assert gi.quantifier == Quantifier.EXISTS

    def test_graph_incoming_str(self) -> None:
        """Test string representation of GraphIncoming."""
        p = Variable("p")
        wi = WeightInterval(0.5, 2.5)
        ei = EdgeCountInterval(1, 5)
        graphs = frozenset(["c"])

        gi = GraphIncoming(
            arg=p,
            graphs=graphs,
            edge_count=ei,
            weights=wi,
            quantifier=Quantifier.EXISTS,
        )

        result = str(gi)
        assert "In" in result
        assert "exists" in result
        assert "c" in result

    def test_graph_incoming_children(self) -> None:
        """Test children iteration for GraphIncoming."""
        p = Variable("p")
        wi = WeightInterval(0.5, 2.5)
        ei = EdgeCountInterval(1, 5)

        gi = GraphIncoming(
            arg=p,
            graphs=frozenset(["c"]),
            edge_count=ei,
            weights=wi,
            quantifier=Quantifier.EXISTS,
        )

        children = list(gi.children())
        assert len(children) == 1
        assert children[0] is p

    def test_graph_incoming_horizon(self) -> None:
        """Test horizon calculation for GraphIncoming."""
        p = Variable("p")
        wi = WeightInterval(0.5, 2.5)
        ei = EdgeCountInterval(1, 5)

        gi = GraphIncoming(
            arg=p,
            graphs=frozenset(["c"]),
            edge_count=ei,
            weights=wi,
            quantifier=Quantifier.EXISTS,
        )

        assert gi.horizon() == 0  # Horizon of Variable is 0

    def test_graph_incoming_to_nnf(self) -> None:
        """Test NNF conversion for GraphIncoming."""
        p = Variable("p")
        wi = WeightInterval(0.5, 2.5)
        ei = EdgeCountInterval(1, 5)

        gi = GraphIncoming(
            arg=Not(p),
            graphs=frozenset(["c"]),
            edge_count=ei,
            weights=wi,
            quantifier=Quantifier.EXISTS,
        )

        nnf = gi.to_nnf()
        assert isinstance(nnf, GraphIncoming)
        # Not(p).to_nnf() returns Not(p) since p is a Variable
        assert isinstance(nnf.arg, Not)

    def test_graph_incoming_expand(self) -> None:
        """Test expansion for GraphIncoming."""
        p = Variable("p")
        wi = WeightInterval(0.5, 2.5)
        ei = EdgeCountInterval(1, 5)

        gi = GraphIncoming(
            arg=p,
            graphs=frozenset(["c"]),
            edge_count=ei,
            weights=wi,
            quantifier=Quantifier.EXISTS,
        )

        expanded = gi.expand()
        assert isinstance(expanded, GraphIncoming)
        assert expanded.arg is p  # Variables expand to themselves


class TestGraphOutgoing:
    """Tests for GraphOutgoing class."""

    def test_graph_outgoing_creation(self) -> None:
        """Test creating GraphOutgoing expressions."""
        p = Variable("p")
        wi = WeightInterval(0.5, 2.5)
        ei = EdgeCountInterval(1, 5)

        go = GraphOutgoing(
            arg=p,
            graphs=frozenset(["c"]),
            edge_count=ei,
            weights=wi,
            quantifier=Quantifier.FORALL,
        )

        assert go.arg is p
        assert go.quantifier == Quantifier.FORALL

    def test_graph_outgoing_str(self) -> None:
        """Test string representation of GraphOutgoing."""
        p = Variable("p")
        wi = WeightInterval(0.5, 2.5)
        ei = EdgeCountInterval(1, 5)

        go = GraphOutgoing(
            arg=p,
            graphs=frozenset(["s"]),
            edge_count=ei,
            weights=wi,
            quantifier=Quantifier.FORALL,
        )

        result = str(go)
        assert "Out" in result
        assert "forall" in result
        assert "s" in result


class TestSTLGoParser:
    """Tests for STL-GO grammar and parser."""

    def test_parse_simple_graph_incoming(self) -> None:
        """Test parsing simple incoming operator."""
        result = parse_expr("in^[0,1]{E}_{c}[1,5] p", syntax="stl_go")

        assert isinstance(result, GraphIncoming)
        assert isinstance(result.arg, Variable)
        assert isinstance(result.arg.name, str)
        assert result.arg.name == "p"
        assert result.graphs == frozenset(["c"])
        assert result.quantifier == Quantifier.EXISTS

    def test_parse_graph_outgoing(self) -> None:
        """Test parsing outgoing operator."""
        result = parse_expr("out^[0,1]{A}_{s}[1,5] p", syntax="stl_go")

        assert isinstance(result, GraphOutgoing)
        assert isinstance(result.arg, Variable)
        assert result.graphs == frozenset(["s"])
        assert result.quantifier == Quantifier.FORALL

    def test_parse_multiple_graph_types(self) -> None:
        """Test parsing with multiple graph types."""
        result = parse_expr("in^[0,1]{E}_{c,s,m}[1,5] p", syntax="stl_go")

        assert isinstance(result, GraphIncoming)
        assert result.graphs == frozenset(["c", "s", "m"])

    def test_parse_weight_interval_bounds(self) -> None:
        """Test parsing various weight interval bounds."""
        # Test with specific numbers
        result1 = parse_expr("in^[0.5,2.5]{E}_{c}[1,5] p", syntax="stl_go")
        assert isinstance(result1, GraphIncoming)
        assert result1.weights.start == 0.5
        assert result1.weights.end == 2.5

        # Test with unbounded (None values)
        result2 = parse_expr("in^[,]{E}_{c}[1,5] p", syntax="stl_go")
        assert isinstance(result2, GraphIncoming)
        assert result2.weights.start is None
        assert result2.weights.end is None

    def test_parse_edge_count_interval(self) -> None:
        """Test parsing edge count intervals."""
        result = parse_expr("in^[0,1]{E}_{c}[1,5] p", syntax="stl_go")

        assert isinstance(result, GraphIncoming)
        assert result.edge_count.start == 1
        assert result.edge_count.end == 5

    def test_parse_quantifier_exists(self) -> None:
        """Test parsing EXISTS quantifier."""
        result1 = parse_expr("in^[0,1]{E}_{c}[1,5] p", syntax="stl_go")
        assert isinstance(result1, GraphIncoming)
        assert result1.quantifier == Quantifier.EXISTS

        result2 = parse_expr("in^[0,1]{exists}_{c}[1,5] p", syntax="stl_go")
        assert isinstance(result2, GraphIncoming)
        assert result2.quantifier == Quantifier.EXISTS

    def test_parse_quantifier_forall(self) -> None:
        """Test parsing FORALL quantifier."""
        result1 = parse_expr("in^[0,1]{A}_{c}[1,5] p", syntax="stl_go")
        assert isinstance(result1, GraphIncoming)
        assert result1.quantifier == Quantifier.FORALL

        result2 = parse_expr("in^[0,1]{forall}_{c}[1,5] p", syntax="stl_go")
        assert isinstance(result2, GraphIncoming)
        assert result2.quantifier == Quantifier.FORALL

    def test_parse_complex_subformula(self) -> None:
        """Test parsing with complex subformulas."""
        result = parse_expr("in^[0,1]{E}_{c}[1,5] (p & q)", syntax="stl_go")

        assert isinstance(result, GraphIncoming)
        # Subformula is p & q

    def test_parse_nested_graph_operators(self) -> None:
        """Test parsing nested graph operators."""
        result = parse_expr("in^[0,1]{E}_{c}[1,5] out^[0,1]{E}_{s}[1,3] p", syntax="stl_go")

        assert isinstance(result, GraphIncoming)
        assert isinstance(result.arg, GraphOutgoing)

    def test_parse_graph_with_ltl(self) -> None:
        """Test mixing STL-GO operators with LTL operators."""
        result = parse_expr("F p & in^[0,1]{E}_{c}[1,5] q", syntax="stl_go")

        expected = Eventually(Variable("p")) & GraphIncoming(
            Variable("q"), frozenset(["c"]), EdgeCountInterval(1, 5), WeightInterval(0, 1), Quantifier.EXISTS
        )
        assert result == expected

    def test_parse_ltl_formulas_in_stl_go(self) -> None:
        """Test that LTL formulas still parse in stl_go syntax."""
        result = parse_expr("F[0,5] p", syntax="stl_go")
        assert isinstance(result, Eventually)

        result2 = parse_expr("p & q", syntax="stl_go")
        # Should parse without errors
        assert result2 == (Variable("p") & Variable("q"))

    def test_parse_boolean_operators(self) -> None:
        """Test boolean operators with graph operators."""
        _ = parse_expr("in^[0,1]{E}_{c}[1,5] p | q", syntax="stl_go")

    def test_parse_negation_of_graph_operator(self) -> None:
        """Test negation of graph operators."""
        result = parse_expr("!in^[0,1]{E}_{c}[1,5] p", syntax="stl_go")
        assert isinstance(result, Not)
        assert isinstance(result.arg, GraphIncoming)


class TestSTLGoProperties:
    """Tests for STL-GO expression properties and methods."""

    def test_graph_incoming_with_temporal_subformula(self) -> None:
        """Test GraphIncoming with temporal subformulas."""
        f_p = Eventually(Variable("p"), TimeInterval(0, 5))
        wi = WeightInterval(0.5, 2.5)
        ei = EdgeCountInterval(1, 5)

        gi = GraphIncoming(
            arg=f_p,
            graphs=frozenset(["c"]),
            edge_count=ei,
            weights=wi,
            quantifier=Quantifier.EXISTS,
        )

        # Horizon should be from the Eventually operator
        assert gi.horizon() == 5

    def test_quantifier_double_negation(self) -> None:
        """Test that negating quantifier twice returns original."""
        q = Quantifier.EXISTS
        assert q.negate().negate() == q

        q2 = Quantifier.FORALL
        assert q2.negate().negate() == q2

    def test_edge_count_range_interval(self) -> None:
        """Test edge count with range interval [1,5]."""
        ei = EdgeCountInterval(1, 5)
        assert ei.start == 1
        assert ei.end == 5

    def test_weight_interval_negative_range(self) -> None:
        """Test weight intervals with negative values."""
        wi = WeightInterval(-2.5, -0.5)
        assert wi.start == -2.5
        assert wi.end == -0.5
        assert wi.duration() == 2.0
