from logic_asts.base import And, Literal, Not, Variable, is_bool_node


def test_is_bool_node_accepts_bool_nodes() -> None:
    assert is_bool_node(Variable("p"))
    assert is_bool_node(Literal(True))
    assert is_bool_node(And((Variable("a"), Variable("b"))))
    assert is_bool_node(Not(Variable("a")))


def test_is_bool_node_rejects_non_bool() -> None:
    assert not is_bool_node("not an expr")


def test_is_bool_node_checks_variable_payload_type() -> None:
    assert is_bool_node(Variable("p"), str)
    assert not is_bool_node(Variable(42), str)
    # check_type only constrains Variable payloads, not other nodes.
    assert is_bool_node(Literal(True), str)


def test_is_ltl_node_accepts_bool_and_temporal() -> None:
    from logic_asts.ltl import Always, Until, is_ltl_node

    assert is_ltl_node(Variable("p"))  # bool subset
    assert is_ltl_node(Always(Variable("p")))
    assert is_ltl_node(Until(Variable("a"), Variable("b")))


def test_is_ltl_node_rejects_non_ltl() -> None:
    from logic_asts.ltl import is_ltl_node
    from logic_asts.sere import Concat

    assert not is_ltl_node(Concat((Variable("a"), Variable("b"))))


def test_is_sere_node_accepts_bool_and_sere() -> None:
    from logic_asts.sere import Concat, Repeat, is_sere_node

    assert is_sere_node(Variable("p"))  # bool subset
    assert is_sere_node(Concat((Variable("a"), Variable("b"))))
    assert is_sere_node(Repeat(Variable("a"), 0, None))


def test_is_sere_node_rejects_temporal() -> None:
    from logic_asts.ltl import Always
    from logic_asts.sere import is_sere_node

    assert not is_sere_node(Always(Variable("p")))


def test_is_strel_node_accepts_ltl_and_spatial() -> None:
    from logic_asts.strel import DistanceInterval, Somewhere, is_strel_node

    assert is_strel_node(Variable("p"))
    assert is_strel_node(Somewhere(Variable("p"), DistanceInterval(0, 5)))


def test_is_strel_node_rejects_sere() -> None:
    from logic_asts.sere import Concat
    from logic_asts.strel import is_strel_node

    assert not is_strel_node(Concat((Variable("a"), Variable("b"))))


def test_is_stlgo_node_accepts_ltl() -> None:
    from logic_asts.stl_go import is_stlgo_node

    assert is_stlgo_node(Variable("p"))


def test_is_stlgo_node_rejects_sere() -> None:
    from logic_asts.sere import Concat
    from logic_asts.stl_go import is_stlgo_node

    assert not is_stlgo_node(Concat((Variable("a"), Variable("b"))))


def test_is_psl_formula_node_excludes_bare_sere() -> None:
    from logic_asts.psl import is_psl_formula_node
    from logic_asts.sere import Concat

    assert is_psl_formula_node(Variable("p"))
    assert not is_psl_formula_node(Concat((Variable("a"), Variable("b"))))


def test_is_psl_node_includes_sere() -> None:
    from logic_asts.psl import is_psl_node
    from logic_asts.sere import Concat

    assert is_psl_node(Concat((Variable("a"), Variable("b"))))
    assert is_psl_node(Variable("p"))


def test_is_psl_node_rejects_strel() -> None:
    from logic_asts.psl import is_psl_node
    from logic_asts.strel import DistanceInterval, Somewhere

    assert not is_psl_node(Somewhere(Variable("p"), DistanceInterval(0, 5)))
