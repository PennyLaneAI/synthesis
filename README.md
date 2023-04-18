Synthesis
=========

This repository contains source code for the PennyLane/Catalyst program synthesis library.


Installation
------------

We do not provide any packaging at the moment. To use the library one typically does:

1. Clone this repository.
2. Install the dependencies. Synthesis mostly uses standard Python libraries but the
   generated programs may depend on `Catalyst`, `PennyLane` and `Jax` libraries. We also provide a
   set of optional `Hypothesis` rules.
   ``` sh
   $ pip install pennylane catalyst hypothesis
   ```

3. Adjust the `PYTHONPATH` to point to the `./python/catalyst_synthesis` folder.
   ``` sh
   export PYTHONPATH="$(cwd)/python/catalyst_synthesis:$PYTHONPATH"
   ```

4. Use the `ipython` or adjust and run the top-level `./python/pl_c_compare.py` script.


Features
--------

* Abstract syntax tree definitions in Python.
* AST manipulation functions.
* Pretty-printing functions.
* An automated program enumeration procedure.
* Set of Hypothesis strategies.

We develop this library thinking of the following ways of defining the program synthesis procedures:

* Greedy enumeration, combinatoric approach.
  - `greedy_enumerator()` (included) illustrates this approach.
* Random generation in the style of `Hypothesis`.
  - Should not be hard to implement.
* Greedy enumeration via P-tree iteration.
  - Not attempted

Design
------

### Abstract syntax tree structures

The library defines mutually-recursive AST structures required to represent a common subset of
Catalyst and PennyLane Python program syntax. An essence of it is shown below:

```
VName ::= VName_ctr str
FName ::= FName_ctr str
ConstExpr ::= ConstExpr_ctr (int | float | complex | np.array | ... )

Stmt ::= FDefStmt | AssignStmt | RetStmt

FDefStmt ::= FDefStmt_ctr (FName * [VName] * POI)
AssignStmt ::= AssignStmt_ctr (VName * POI)
RetStmt ::= RetStmt_ctr Expr

Expr ::= VRefExpr | FCallExpr | ConstExpr | NoneExpr | CondExpr |
         ForLoopExpr | WhileLoopExpr

CondExpr ::= CondExpr_ctr (Expr * POI * POI)
WhileExpr ::= WhileExpr_ctr (VName * Expr * POI)
ForLoopExpr ::= ForLoopExpr_ctr (VName * POI * POI * POI)
NoneExpr ::= NoneExpr_ctr

POI ::= POI_ctr ([Stmt] * Expr)
```

Notes:

* Control-flow instructions are functions rather than statements due to Catalyst/JAX requirements.
* `POI` stands for Points Of Insertion - an auxiliary structure allowing users to cut and navigate
  across the partially-built AST.

### AST manipulation functions

| Name                    | Members      |
|-------------------------|--------------|
| Construction            | `bless_expr`, `bless_poi`, `callExpr`, `gateExpr`, `assignStmt` |
| Recursive querying      | `reduce_stmt_expr`,  `get_pois`, `get_vars`, ... |
| Immutable manipulations | `saturate_expr` , `saturate_poi` |
| Mutable manipulations   | `build` |


### Automated program enumeration

TODO

Examples
--------


### Automated program enumeration

1. Recall that `POI()` represents point of insertion.

2. Define program building blocks

``` python
from catalyst_synthesis import *

sample_spec:List[POILike] = [
    gateExpr("qml.CPhaseShift10", 0, wires=[POI(), POI()]),
    callExpr(ForLoopExpr(VName("k1"), POI.fE(1), POI.fE(2), POI(), CFS.Default, VName("k2")), [POI()]),
]

```

3. Print a first program:

``` python
for b in greedy_enumerator(sample_spec, [VName('arg')]):
  pprint(b.pois[0].poi)
  break
```

``` result
@for_loop(1,2,1)
def forloop0(k1,k2):
    return k2 + arg
## qml.CPhaseShift10(0, wires=[forloop0(arg), arg]) ##
```

