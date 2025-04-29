# mypy: allow_untyped_calls
import enum
import typing

from lark import Lark, Token, Transformer
from lark.visitors import merge_transformers

from logic_asts.base import Expr
from logic_asts.grammars import BaseTransform, LtlTransform, StrelTransform


@enum.unique
class SupportedGrammars(enum.StrEnum):
    BASE = enum.auto()
    """Base Boolean propositional logic, without quantifiers or modal operators"""
    LTL = enum.auto()
    """Linear Temporal Logic"""
    STREL = enum.auto()
    """Spatio-Temporal Reach Escape Logic"""

    def get_transformer(self) -> Transformer[Token, Expr]:
        syntax = str(self)

        transformer: Transformer[Token, Expr]
        match syntax:
            case "base":
                transformer = BaseTransform()
            case "ltl":
                transformer = merge_transformers(
                    LtlTransform(),
                    base=BaseTransform(),
                )
            case "strel":
                transformer = merge_transformers(
                    StrelTransform(),
                    ltl=merge_transformers(
                        LtlTransform(),
                        base=BaseTransform(),
                    ),
                )
            case _:
                raise ValueError(f"Unsupported grammar reference: {syntax}")
        return transformer


type SupportedGrammarsStr = typing.Literal["base", "ltl", "strel"]


def parse_expr(
    expr: str,
    *,
    syntax: SupportedGrammars | SupportedGrammarsStr = SupportedGrammars.BASE,
) -> Expr:
    syntax = SupportedGrammars(syntax)

    grammar = Lark.open_from_package(
        __name__,
        f"{str(syntax)}.lark",
        ["grammars"],
    )
    transformer = syntax.get_transformer()
    assert isinstance(transformer, Transformer), f"{transformer=}"

    parse_tree = grammar.parse(expr)
    return transformer.transform(tree=parse_tree)
