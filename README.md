# Logic ASTs: Abstract Syntax Trees for Logical Specifications

A collection of grammars, parsers, and abstract syntax trees (ASTs) for various
logical formalisms.

The goal is to serve as a reusable foundation for academics and developers
building tools that require logical expression parsing and manipulation,
eliminating the need to create new parsers for each application.

## Supported Logics

The library implements complete support for four logical systems:

1. **Propositional Logic (base)**:
   Classical Boolean logic with conjunction, disjunction, negation, implication,
   equivalence, and exclusive-or operators.

2. **Linear Temporal Logic (ltl)**:
   Temporal extension adding operators for reasoning about sequences of states
   over time.
   Includes Next (X), Eventually (F), Always (G), and Until (U) operators with
   optional time constraints.

3. **Spatio-Temporal Reach-Escape Logic (strel)**:
   Combines temporal and spatial reasoning for multi-agent and distributed
   systems.
   Adds spatial operators (Everywhere, Somewhere, Reach, Escape) with distance
   constraints.

4. **Signal Temporal Logic with Graph Operators (stl_go)**:
   Extends temporal logic with graph-based operators for specifying properties
   over multi-agent communication networks.
   Includes incoming and outgoing edge quantifiers with weight and count
   constraints.

## Installation

Install from PyPI:
```bash
pip install logic-asts
```

Or if you'd like the latest main branch:

```bash
pip install git+https://github.com/anand-bala/logic-asts.git
```

## Quick Start

Parse logical expressions:

```python
import logic_asts

# Propositional logic
prop = logic_asts.parse_expr("(p & q) | ~r", syntax="base")

# Linear temporal logic
ltl = logic_asts.parse_expr("G(request -> F response)", syntax="ltl")

# Spatio-temporal logic
strel = logic_asts.parse_expr("G everywhere[0,5] ~obstacle", syntax="strel")

# Graph-based temporal logic
stl_go = logic_asts.parse_expr("in^[0,1]{E}_{c}[1,n] consensus", syntax="stl_go")
```

Create expressions programmatically:

```python
from logic_asts.base import Variable, And, Or, Not
from logic_asts.ltl import Eventually, TimeInterval

p = Variable("p")
q = Variable("q")

# (p & q) | ~p
formula = (p & q) | ~p

# F[0,10] (p & q)
temporal = Eventually(p & q, TimeInterval(0, 10))
```

Evaluate propositional formulas:

```python
from logic_asts.base import simple_eval

p = Variable("p")
q = Variable("q")
formula = p & q

# Evaluate: p=true, q=true -> Result: true
result = simple_eval(formula, {"p", "q"})

# Evaluate: p=true, q=false -> Result: false
result = simple_eval(formula, {"p"})
```

## Type-safe tree traversal

The most convenient way to walk an expression tree is `expr.iter_subtree()`,
but its return type is `Iterator[Expr]`. If you need mypy (or pyright) to know
the precise element type, reach for one of the patterns below.

### Pattern 1 -- typed iterator (preferred)

When you already hold a typed expression, call the matching iterator directly:

```python
from logic_asts import ltl_expr_iter, parse_expr

expr = parse_expr("G (p U q)", syntax="ltl")  # LTLExpr[str]
for node in ltl_expr_iter(expr):              # Iterator[LTLExpr[str]]
    ...
```

| Your type | Iterator to use |
|-----------|----------------|
| `BoolExpr[AP]` / `BaseExpr[AP]` | `bool_expr_iter(expr)` |
| `LTLExpr[AP]` | `ltl_expr_iter(expr)` |
| `STRELExpr[AP]` | `strel_expr_iter(expr)` |
| `STLGOExpr[AP]` | `stlgo_expr_iter(expr)` |

All four functions also validate that the subtree contains no out-of-dialect
nodes and raise `TypeError` at runtime if it does.

### Pattern 2 -- type-guard then typed iterator

When the static type is just `Expr` (e.g. coming from an untyped API), narrow
it first:

```python
from logic_asts import Expr, is_ltl_expr, ltl_expr_iter

def process(expr: Expr) -> None:
    if is_ltl_expr(expr, str):            # narrows to LTLExpr[str]
        for node in ltl_expr_iter(expr):  # Iterator[LTLExpr[str]]
            ...
```

### Pattern 3 -- filter to a single node class

Use the `kind=` argument on `iter_subtree` to visit only one concrete class.
The full tree is still traversed, but only matching nodes are yielded:

```python
from logic_asts import Variable, parse_expr

expr = parse_expr("G (p U q)", syntax="ltl")
for v in expr.iter_subtree(kind=Variable):  # Iterator[Variable[Any]]
    print(v.name)
```

### Pattern 4 -- custom type subset with `ExprVisitor`

For arbitrary subsets of node types, construct an `ExprVisitor` directly.
It both validates the tree and yields a typed iterator:

```python
from logic_asts import ExprVisitor, And, Or, Not, Variable

for node in ExprVisitor((And, Or, Not, Variable), expr):
    # node: And | Or | Not | Variable[Any]
    ...
```

## Contributing

Contributions are welcome.
Please ensure all tests pass and documentation is updated for new features.

## License

This project is licensed under the BSD 2-clause license.
