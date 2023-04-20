Synthesis
=========

This repository contains source code for the [PennyLane](https://github.com/PennyLaneAI/pennylane)/[Catalyst](https://github.com/PennyLaneAI/catalyst) program synthesis library.


Installation
------------

We do not provide any packaging at the moment. To use the library one typically does:

1. Clone this repository.
2. Install the dependencies. Synthesis mostly uses standard Python libraries but the
   generated programs may depend on `PennyLane`, `Catalyst` and `Jax` libraries. We also provide a
   set of optional [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) rules.
   ``` sh
   $ pip install pennylane catalyst hypothesis
   ```

3. Adjust the `PYTHONPATH` to point to the `./python/catalyst_synthesis` folder.
   ``` sh
   export PYTHONPATH="$(pwd)/python/catalyst_synthesis:$PYTHONPATH"
   ```

4. Optionally, call `pytest` to run the tests. Note, running tests in multi-processing mode with `-n
   NCPU` argument may not currently work.

5. Use the `ipython` or adjust and run the top-level `./python/pl_c_compare.py` script.


Contents
--------

<!-- vim-markdown-toc GFM -->

* [Features](#features)
* [Design](#design)
    * [Abstract syntax tree](#abstract-syntax-tree)
    * [AST manipulation functions](#ast-manipulation-functions)
    * [Enumerating programs](#enumerating-programs)
* [Examples](#examples)
    * [Working with AST](#working-with-ast)
    * [Mutable AST updates](#mutable-ast-updates)
    * [Evaluating a program](#evaluating-a-program)
    * [Enumerating programs by specification](#enumerating-programs-by-specification)
    * [Top-level script pl_c_compare.py](#top-level-script-pl_c_comparepy)
* [Known issues and limitations](#known-issues-and-limitations)
* [References](#references)

<!-- vim-markdown-toc -->


Features
--------

* Abstract syntax tree definitions in Python.
* Functions for AST manipulation.
* Pretty-printing functions.
* An automated program enumeration procedure.
* A set of Hypothesis strategies.

We develop this library thinking of the following ways of defining the program synthesis procedures:

* Greedy enumeration, the combinatoric approach.
  - `catalyst_synthesis.generator.greedy_enumerator` is included.
* Random generation in the style of `Hypothesis`.
  - `catalyst_synthesis.hypothesis.strategies.programs` (unfinished) illustrates this approach.
* Greedy enumeration via [P-tree iteration][1].
  - Not attempted

Design
------

### Abstract syntax tree

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
  across the partially-built AST. This design might also serve as an adaptation to the future
  application of the advanced recursion schemes such as [Catamorphisms][2].

### AST manipulation functions

| Name                    | Members      |
|-------------------------|--------------|
| AST dataclasses (upper-case)         | `FDefStmt`, `FCallExpr`, `ForLoopExpr`, `WhileLoopExpr`, `CondExpr`, ... |
| Construction helpers (lower-case)    | `bless_*`, `fdefStmt`, `callExpr`, `forLoopExpr`, `whileLoopExpr`, `condExpr`, ... |
| Recursive AST querying      | `reduce_stmt_expr`,  `get_pois`, `get_vars`, ... |
| Immutable AST manipulations | `saturate_expr` , `saturate_poi` |
| Mutable AST manipulations   | `Build`, `build` |
| Pretty-printing             | `pprint`, `pstr`, `pstr_expr`, `pstr_stmt`, `pstr_poi`, ... |


### Enumerating programs

The mutable AST manipulation utilities, namely, the expression builder class `Build`, allow to
define combinatorial algorithms iterating over programs in a specified domain. The specification
lists AST parts to combine and the algorithm yields resulting programs.

Below is the workflow used by the `greedy_enumerator` algorithm we provide in this library.

1. The list of AST parts to combine is taken as input.
2. The total number of POIs mentioned in the specification is calculated.
3. The list of bound variables mentioned in the specification is determined.
4. The extended list of parts is set to be the input specification extended by `None` placeholders
   which instruct the algorithm to use a bound variable.
4. Now, repeatedly:
   1. The program creation instruction is defined as a list of triples `(p,e,v)` where `p` is the
      Position to pass to `Builder.update` method, the `e` is the member of list of extended parts
      to put into this position, and `v` is the variable to use if `e` is a placeholder or if it
      requires an argument.
   2. An instruction instance is obtained as a result of permuting all possible candidates for `p`,
      `e` and `v`.
   3. The program instance is being built according to the instruction.
      - At this point, some of the programs may violate the rules of Python by mentioning variables
        before they are defined. We use builder's `Context` to detect such issues and skip these
        programs.
      - Also the programs may violate typing rules, e.g. passing gate as `wire` argument to another
        gate. The typechecker would be useful to detect such cases, but right now we don't have one,
        so we output these programs and compare the results of execution on different backends.

Examples
--------

``` python
from catalyst_synthesis import *
```

### Working with AST

AST elements could be created and nested, but they doesn't look readable in their native
representation.

``` python
c = callExpr(condExpr(lessExpr("i",0), POI(), POI()), [])
l = callExpr(forLoopExpr("i", "state", 0, 10, c), [3])
f = fdefStmt("foo", ["arg"], l, qnode_device="our.device", qnode_wires=4)
print(f)
```

``` result
FDefStmt(fname=FName(val='foo'), args=[VName(val='arg')], body=POI(stmts=[], expr=FCallExpr(expr=ForLoopExpr(loopvar=VName(val='i'), lbound=POI(stmts=[], expr=ConstExpr(val=0)), ubound=POI(stmts=[], expr=ConstExpr(val=10)), body=POI(stmts=[], expr=FCallExpr(expr=CondExpr(cond=FCallExpr(expr=VRefExpr(vname=FName(val='<')), args=[POI(stmts=[], expr=VRefExpr(vname=VName(val='i'))), POI(stmts=[], expr=ConstExpr(val=0))], kwargs=[]), trueBranch=POI(stmts=[], expr=None), falseBranch=POI(stmts=[], expr=None), style=<ControlFlowStyle.Default: 0>), args=[], kwargs=[])), style=<ControlFlowStyle.Default: 0>, statevar=VName(val='state')), args=[POI(stmts=[], expr=ConstExpr(val=3))], kwargs=[])), qnode_wires=4, qnode_device='our.device', qjit=False)
```

`pstr_*` functions prints AST in a human-readable form. One can specify either a PennyLane or a
Catalyst style of Python to use.

``` python
print(pstr(f, default_cfstyle=ControlFlowStyle.Python))
```

``` result
@qml.qnode(qml.device("our.device", wires=4))
def foo(arg):
    state = 3
    for i in range(0, 10):
        if i < 0:
            _cond0 = None
        else:
            _cond0 = None
        state = _cond0
    return state
```

``` python
print(pstr(f, default_cfstyle=ControlFlowStyle.Catalyst))
```

``` result
@qml.qnode(qml.device("our.device", wires=4))
def foo(arg):
    @for_loop(0, 10, 1)
    def forloop0(i,state):
        @cond(i < 0)
        def cond1():
            pass
        @cond1.otherwise
        def cond1():
            pass
        return cond1()
    return forloop0(3)
```

### Mutable AST updates

Creating AST from the leaves back to the top is not always convenient. A top-down approach would
follow the operational semantic of the program which may simplify resource tracking.  Unfortunately,
this approach typically requires modification of existing parts of the tree which may be harder to
work with. We provide `build` class which does the necessary bookkeeping.

Below we load the already existing part of the tree into a builder. The builder notes free `POIs`
and collects a context information. Currently, contexts include names of visible variables and some
additional information of loop variables that are important for avoiding infinite loops.

The expression builder is a pretty-printable object.

``` python
b = build(POI([f]))
print(pstr(b))
```

``` result
@qml.qnode(qml.device("our.device", wires=4))
def foo(arg):
    @for_loop(0, 10, 1)
    def forloop0(i,state):
        @cond(i < 0)
        def cond1():
            pass
            # poi-id: 140169038599392
            # poi-var: arg, i, state
        @cond1.otherwise
        def cond1():
            pass
            # poi-id: 140169038599152
            # poi-var: arg, i, state
        return cond1()
    return forloop0(3)
# poi-id: 140170392548880
# poi-var:
## None ##
```

Once the statement is loaded into a builder and the free POIs are known, one could use `update`
method to replace a specific POI with a new one, possibly adding more POIs to the list.

The first POI always refers to the whole expression being built.

``` python
_ = b.update(1, gateExpr('qml.Hadamard', wires=[2]))
print(pstr(b.pois[0].poi))
```

``` result
@qml.qnode(qml.device("our.device", wires=4))
def foo(arg):
    @for_loop(0, 10, 1)
    def forloop0(i,state):
        @cond(i < 0)
        def cond1():
            return qml.Hadamard(wires=[2])
        @cond1.otherwise
        def cond1():
            pass
        return cond1()
    return forloop0(3)
## None ##
```


### Evaluating a program

We provide `evalPOI` function to evaluate the program using Python's `eval` built-in method and
`runPOI` to output the program as a file and run it as a subprocess. In the next section we show how
to use the former.

### Enumerating programs by specification

Below we show how to create a specification and run the program enumerator. Recall that `POI()`
stands for `Point Of Insertion`. Theses structures define the cutting-points in a tree.

``` python
specification:List[POILike] = [
    gateExpr("qml.CPhaseShift10", 0, wires=[POI(), POI()]),
    callExpr(forLoopExpr("k1", "k2", 1, 2, POI(), CFS.Default), [POI()]),
]
```

To enumerate the programs we run our greedy algorithm (We output the very first program instance
here).

``` python
for b in greedy_enumerator(specification, [VName('arg')]):
  pprint(b.pois[0].poi)
  break
```

``` result
@for_loop(1, 2, 1)
def forloop0(k1,k2):
    return k2 + arg
## qml.CPhaseShift10(0, wires=[forloop0(arg), arg]) ##
```

Finally, we print complete program by wrapping it into the top-level function and adding
a header containing the require imports.

``` python
main = wrapInMain(b.pois[0].poi,
                  name="main",
                  args=[VName('arg')],
                  qnode_device="lightning.qubit",
                  qnode_wires=4,
                  measure_quantum_state=True)
print('\n'.join(pprint_pyenv(with_catalyst=True)))
print(pstr(main))
```

``` result
import pennylane as qml
from math import inf as inf
from math import nan as nan
from cmath import infj as infj
from cmath import nanj as nanj
import jax.numpy as np
import jax as jax
from catalyst.pennylane_extensions import for_loop as for_loop
from catalyst.pennylane_extensions import while_loop as while_loop
from catalyst.pennylane_extensions import cond as cond
from catalyst.compilation_pipelines import qjit as qjit
from jax._src.numpy.lax_numpy import array as Array
from jax.numpy import int64 as int64
from jax.numpy import float64 as float64
from jax.numpy import complex128 as complex128
@qjit
@qml.qnode(qml.device("lightning.qubit", wires=4))
def main(arg):
    @for_loop(1, 2, 1)
    def forloop0(k1,k2):
        return k2 + arg
    _ = qml.CPhaseShift10(0, wires=[forloop0(arg), arg])
    return qml.state()
```

To run the program we use `evalPOI`. It calls `wrapInMain` internally and uses the same interface.

``` python
r = evalPOI(b.pois[0].poi,
            name="main",
            args=[(VName('arg'),1)],
            qnode_device="lightning.qubit",
            qnode_wires=4,
            measure_quantum_state=True)
print(r)
```

``` result
[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j
 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
```

### Top-level script pl_c_compare.py

The `./python/pl_c_compare.py` runs a pre-defined specification to iterate over programs featuring a
mixture of control-flows and a gate execution. Currently the script accepts no argument and the
specification is hard-coded.

The output looks like:

``` sh
$ python3 ./python/pl_c_compare.py
|...|...|__.|...|...|__.|__.|__.|__.|...|...|__.|...|...|__.|__.|__.|__.|__.|__.|__.|__.|__.
```

Where the meaning of `|...` symbols is as follows: `|` - the program text was generated, `..` - The
PennyLane and Catalyst versions of the program were executed and the last `.` - numeric comparison
test was passed. `_` means that the result for this program was already present on disk in the
`_synthesis` folder.

Known issues and limitations
----------------------------

* We currently support only a limited set of Python expressions.
  - Typechecker is not yet implemented. One can overcome its absence by providing enumerator with a
    clever specifications making ill-typed programs rare or impossible.
* The program enumerator outputs non-unique programs which may or may not be a sign of some issue in
  the implementation. Some investigation may be required.
* Pretty-printing functions use recursion aggressively.
* There are some issues with multi-processing `pytest` execution.

References
----------

[1]: https://arxiv.org/pdf/1707.03744v1.pdf "P-Tree programming"
  - 2018, Oesch, *P-Tree programming*
    + https://arxiv.org/pdf/1707.03744v1.pdf
    + https://paperswithcode.com/paper/p-tree-programming

[2]: https://arxiv.org/pdf/2202.13633v1.pdf "Fantastic Morphisms and Where to Find Them A Guide to Recursion Schemes"
  - 2019, Yang, Wu, *Fantastic Morphisms and Where to Find Them A Guide to Recursion Schemes*
    + https://arxiv.org/pdf/2202.13633v1.pdf


