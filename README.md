# Abstract syntax trees for logical expressions

A collection of grammars, parsers, and abstract syntax trees in Python for various
logics. The goal of this package is to serve as a common utility for
academics/developers to build their tools on, without having to create a new parser each
time they make a library.

Currently supported logics are:

- `base`: the basic Boolean propositional logic.
- `ltl`: Linear temporal logic
- `strel`: Spatio-temporal reach escape logic

## Usage

Simply install the library, either as a git dependency or (once it is available) through
PyPI. Then, the library can be used as:

```python
import logic_asts

expr = logic_asts.parse_expr(expr_string, syntax="base"|"ltl"|"strel") # remember to pick the syntax
```
