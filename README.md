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

Install from the repository:

```bash
pip install git+https://github.com/yourusername/logic-asts.git
```

Or, once available on PyPI:

```bash
pip install logic-asts
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

## Contributing

Contributions are welcome.
Please ensure all tests pass and documentation is updated for new features.

## License

This project is licensed under the BSD 2-clause license.
