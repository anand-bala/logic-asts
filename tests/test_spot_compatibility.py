from __future__ import annotations

import dataclasses
import datetime
import importlib.util
import re
import shutil
import subprocess
import typing
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.extra.lark import from_lark
from lark import Lark

import logic_asts

HAS_SPOT: typing.Literal["lib", "cli"]

if importlib.util.find_spec("spot") is not None:
    HAS_SPOT = "lib"
elif all(shutil.which(tool) is not None for tool in ("ltlfilt",)):
    HAS_SPOT = "cli"
else:
    pytest.skip("Spot library nor CLI is available", allow_module_level=True)

SPOT_DIVERGENT_PATTERN = re.compile(
    r"""
    # Spot greedily lexes the uppercase Boolean keywords TRUE / FALSE and then
    # errors on an unseparated alphanumeric tail. logic_asts admits the same
    # input by lexing the leading letter as the F/G/X unary operator and the
    # remainder (incl. ``ALSE...``/``RUE...``) as a CNAME atomic proposition,
    # producing a parse that Spot rejects. Skip these inputs.
    (?:^|[^A-Za-z0-9_])(?:TRUE|FALSE)[A-Za-z0-9_]
    |
    # ``;`` is the sequence operator that logic_asts adds on top of Spot's LTL
    # syntax; in Spot land ``;`` belongs to SERE / PSL, not LTL, so ltlfilt
    # ``--ltl`` rejects any formula containing it.
    ;
    |
    # A Boolean-constant digit (``0`` / ``1``) immediately adjacent to a word
    # character is rejected by logic_asts (the digit constants require a
    # non-word boundary, so e.g. ``1U1``, ``0xor1``, ``1a`` all fail to lex).
    # Spot also rejects these inputs but at the parser level (it lexes the
    # adjacent word as an atomic proposition and errors because a constant
    # cannot sit next to an AP without an operator between them). Either way
    # such strings are not valid for differential testing, so skip them.
    (?<![A-Za-z0-9_])[01][A-Za-z0-9_]
    |
    # Lark's ``common.WS`` includes form-feed (``\x0c``) and treats it as
    # ignorable whitespace, so logic_asts silently accepts it inside formulas;
    # Spot's tokenizer does not, and rejects the input as a syntax error.
    \x0c
    """,
    re.VERBOSE,
)


def assert_spot_accepts(*formulas: str) -> None:
    """Run ``ltlfilt`` on ``formulas`` and assert that Spot accepts every one.

    Multiple formulas are passed via repeated ``--formula`` flags so that a
    single ``ltlfilt`` subprocess covers them all; the count of accepted
    formulas reported by ``ltlfilt --count`` must equal ``len(formulas)``.
    """
    if HAS_SPOT == "lib":
        import spot

        for formula in formulas:
            try:
                f = spot.formula(formula)
            except Exception as e:
                raise AssertionError(f"Failed to parse formula `{formula}`") from e
            if not f.is_ltl_formula():
                raise AssertionError(f"Parsed formula `{formula}` is not a valid LTL formula")

    elif HAS_SPOT == "cli":
        args: list[str] = ["ltlfilt", "--ltl", "--count"]
        for formula in formulas:
            args.extend(("--formula", formula))
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode == 0:
            count = int(proc.stdout)
            assert count == len(formulas), (
                f"Expected {len(formulas)} formula(s) to be accepted by Spot, got {count}: {formulas!r}"
            )
        elif proc.returncode == 1:
            # no formulas were output (no match)
            raise AssertionError(f"At least one of {formulas!r} is not a valid LTL formula per Spot")
        else:  # proc.returncode == 2
            # error has been reported
            raise AssertionError(f"Failed to parse one of {formulas!r}:\n" + proc.stderr)


_LTL_GRAMMAR = Lark.open_from_package(
    "logic_asts",
    "ltl.lark",
    ["grammars"],
)


_MAX_FORMULA_LEN = 64


@settings(
    max_examples=200,
    # The default 200 ms deadline trips frequently because each example
    # spawns an ``ltlfilt`` subprocess.
    deadline=datetime.timedelta(seconds=2),
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large,
    ],
)
@given(formula=from_lark(_LTL_GRAMMAR).filter(lambda s: len(s) <= _MAX_FORMULA_LEN))
def test_logic_asts_matches_spot(formula: str) -> None:
    assume(SPOT_DIVERGENT_PATTERN.search(formula) is None)
    expr = logic_asts.parse_expr(formula, syntax="ltl")
    assert_spot_accepts(formula, str(expr))
