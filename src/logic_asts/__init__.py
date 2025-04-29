import typing

from lark import Lark, Token, Transformer
from lark.visitors import merge_transformers

from logic_asts.base import Expr
from logic_asts.grammars import GRAMMARS_DIR, BaseTransform, LtlTransform

type SupportedGrammars = typing.Literal["base", "ltl"]


def parse_expr(expr: str, *, syntax: SupportedGrammars) -> Expr:
    transformer: Transformer[Token, Expr]
    match syntax:
        case "base":
            transformer = BaseTransform()
        case "ltl":
            transformer = merge_transformers(
                LtlTransform(),
                base=BaseTransform(),
            )
        case _:
            raise ValueError(f"Unsupported grammar reference: {syntax}")

    assert isinstance(transformer, Transformer), f"{transformer=}"

    grammar = Lark.open_from_package(
        __name__,
        f"{syntax}.lark",
        ["grammars"],
    )

    parse_tree = grammar.parse(expr)
    return transformer.transform(tree=parse_tree)
