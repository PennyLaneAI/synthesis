import os
import sys
import numpy as np
from typing import Any, Dict, Tuple, Union, Callable, Set, List
from hypothesis import given, note, settings, Verbosity, assume
from inspect import signature as python_signature, _empty as inspect_empty
from tempfile import mktemp
from itertools import cycle

import jax.numpy as jnp
from pytest import mark
from dataclasses import astuple
from numpy.testing import assert_allclose

from catalyst_synthesis.grammar import (Expr, RetStmt, FCallExpr, VName, FName, VRefExpr, signature
                                        as expr_signature, isinstance_expr, AssignStmt, lessExpr,
                                        addExpr, eqExpr, ControlFlowStyle as CFS, signature,
                                        Signature, bind, saturate_expr, saturates_expr1,
                                        saturates_poi1, saturate_expr1, saturate_poi1, assignStmt,
                                        assignStmt_, callExpr, WhileLoopExpr, POI, ForLoopExpr,
                                        callExpr, gateExpr, POILike)

from catalyst_synthesis.pprint import (pstr_builder, pstr_stmt, pstr_expr, pprint, pstr,
                                       DEFAULT_CFSTYLE)

from catalyst_synthesis.builder import build
from catalyst_synthesis.exec import compilePOI, evalPOI, runPOI, wrapInMain
from catalyst_synthesis.generator import control_flows
from catalyst_synthesis.hypothesis import *


sample_spec1:List[Expr] = [
    # WhileLoopExpr(VName("i"), trueExpr, POI(), CFS.Catalyst) : 1,
    callExpr(WhileLoopExpr(VName("j1"), lessExpr(VName("j1"),2), POI(), CFS.Default), [POI()]),
    callExpr(ForLoopExpr(VName("k1"), POI.fE(1), POI.fE(2), POI(), CFS.Default, VName("k2")), [POI()]),
    # CondExpr(trueExpr, POI(), POI(), CFS.Catalyst) : 1,
]

sample_spec2:List[Expr] = [
    # WhileLoopExpr(VName("i"), trueExpr, POI(), CFS.Catalyst) : 1,
    ForLoopExpr(VName("k1"), POI(), POI(), POI(), CFS.Default, VName("k2")),
    # CondExpr(trueExpr, POI(), POI(), CFS.Catalyst) : 1,
]

sample_spec3:List[Expr] = [
    WhileLoopExpr(VName("i1"), lessExpr(addExpr(VName("i1"),POI()),3), POI(), CFS.Default),
    CondExpr(trueExpr, POI(), None, CFS.Default),
]

sample_spec4:List[Expr] = [
    callExpr(FName("qml.Hadamard"), [], [('wires',POI())]),
]

sample_spec5:List[POILike] = [
    gateExpr('qml.CPhaseShift10', 0, wires=[POI(), POI()]),
    gateExpr('qml.QubitStateVector', np.array([1.0, 0.0]), wires=[POI()]),
    callExpr(CondExpr(trueExpr, POI(), POI(), CFS.Default), []),
]

gate_lib = [
    (FName("qml.Hadamard"), Signature(['*'],'*')),
    (FName("qml.X"), Signature(['*'],'*')),
]

def bindAssign(poi1:POI, fpoi2:Callable[[Expr],POI]):
    poi2 = fpoi2(poi1.expr)
    return bind(poi1, poi2, poi2.expr)



def run(sample_spec):
    arg = VName('arg')

    def _render(style):
        return compilePOI(
            bindAssign(b.pois[0].poi,
                       lambda e: POI([assignStmt_(e)],callExpr(FName("qml.state"),[]))),
            use_qjit=False, name="main", qnode_wires=3, args=[arg],
            default_cfstyle=style)

    for b in control_flows(sample_spec, [arg]):
        print("1. Builder:")
        pprint(b)
        print("1. Press Enter to render")
        # input()
        o1,code1 = _render(ControlFlowStyle.Catalyst)
        o2,code2 = _render(ControlFlowStyle.Python)
        print("2. Rendered code:")
        print("^^^ Catalyst version ^^^")
        print(code1)
        print("^^^ PennyLane version ^^^")
        print(code2)
        # print("2. Press Enter to evaluate and compare")
        # input()
        # r1 = evalPOI(o1, name="main", args=[(arg,1)])
        # r2 = evalPOI(o2, name="main", args=[(arg,1)])
        # print("3. Evaluation result:")
        # print(r1)
        # print(r2)
        # assert_allclose(r1, r2)
        input()


if __name__ == "__main__":
    # run(sample_spec1, gate_lib1)
    run(sample_spec1)

