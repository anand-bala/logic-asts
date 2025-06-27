# mypy: allow_untyped_calls
import typing

from lark import Lark, Transformer

from logic_asts.base import Expr
from logic_asts.grammars import SupportedGrammars

SupportedGrammarsStr: typing.TypeAlias = typing.Literal["base", "ltl", "strel"]


def parse_expr(
    expr: str,
    *,
    syntax: SupportedGrammars | SupportedGrammarsStr = SupportedGrammars.BASE,
) -> Expr:
    syntax = SupportedGrammars(syntax)

    grammar = Lark.open_from_package(
        __name__,
        f"{str(syntax.value)}.lark",
        ["grammars"],
    )
    transformer = syntax.get_transformer()
    assert isinstance(transformer, Transformer), f"{transformer=}"

    parse_tree = grammar.parse(expr)
    return transformer.transform(tree=parse_tree)
