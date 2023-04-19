import os
import sys
import numpy as np
from typing import Any, Dict, Tuple, Union, Callable, Set, List
from hypothesis import given, note, settings, Verbosity, assume
from inspect import signature as python_signature, _empty as inspect_empty
from tempfile import mktemp
from itertools import cycle
from hashlib import sha256
from os import makedirs
from os.path import join, basename, dirname, isfile
from dataclasses import dataclass

import jax.numpy as jnp
from pytest import mark
from dataclasses import astuple
from numpy.testing import assert_allclose
from numpy import allclose

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
from catalyst_synthesis.exec import compilePOI, evalPOI, runPOI, wrapInMain, PythonObj, PythonCode
from catalyst_synthesis.generator import greedy_enumerator
from catalyst_synthesis.hypothesis import *


sample_spec1:List[POILike] = [
    # WhileLoopExpr(VName("i"), trueExpr, POI(), CFS.Catalyst) : 1,
    POI([assignStmt_(gateExpr('qml.Hadamard', wires=[POI()]))], ConstExpr(1)),
    callExpr(WhileLoopExpr(VName("j1"), lessExpr(VName("j1"),2), POI(), CFS.Default), [POI()]),
    callExpr(ForLoopExpr(VName("k1"), POI.fE(1), POI.fE(2), POI(), CFS.Default, VName("k2")), [POI()]),
    # CondExpr(trueExpr, POI(), POI(), CFS.Catalyst) : 1,
]

sample_spec5:List[POILike] = [
    gateExpr('qml.CPhaseShift10', 0, wires=[POI(), POI()]),
    gateExpr('qml.QubitStateVector', np.array([1.0, 0.0]), wires=[POI()]),
    callExpr(CondExpr(trueExpr, POI(), POI(), CFS.Default), []),
]


def bindAssign(poi1:POI, fpoi2:Callable[[Expr],POI]):
    poi2 = fpoi2(poi1.expr)
    return bind(poi1, poi2, poi2.expr)


@dataclass
class Problem:
    source_file:str
    result_file:str
    error_file:str
    use_qjit:bool
    pyobj:Optional[PythonObj]
    code:PythonCode

DATADIR = '_synthesis'


def problem_prepare(tag:Optional[str], poi:POI, args, style, use_qjit, **kwargs) -> Problem:
    """ Typical `kwargs` are: name="main", qnode_wires=3. `args` are like `[VName('arg')]`.  """
    pyobj, code = compilePOI(poi,
        args=args, use_qjit=use_qjit, default_cfstyle=style, measure_quantum_state=True, **kwargs)
    h = sha256(str(pstr(poi,default_cfstyle=ControlFlowStyle.Python)).encode("utf-8")).hexdigest()[:6]
    pdir = join(DATADIR,f"problem{('_'+tag) if tag is not None else ''}_{h}")
    if style is ControlFlowStyle.Catalyst:
        srcname = join(pdir, "source_C.py")
        resname = join(pdir, "result_C.npy")
        errname = join(pdir, "error_C.txt")
    elif style is ControlFlowStyle.Python:
        srcname = join(pdir, "source_PL.py")
        resname = join(pdir, "result_PL.npy")
        errname = join(pdir, "error_PL.txt")
    else:
        raise ValueError("Unsupported style {style}")
    return Problem(srcname, resname, errname, use_qjit, pyobj, code)

@dataclass
class Result:
    p:Problem
    val:Optional[Any]

def problem_result(p:Problem) -> Optional[Result]:
    try:
        return Result(p, np.load(p.result_file))
    except Exception:
        pass
    return None

def problem_run(p:Problem, args) -> Result:
    """ Typical args: [(VName('arg'),1)] """
    assert p.pyobj is not None
    makedirs(dirname(p.source_file), exist_ok=True)
    makedirs(dirname(p.result_file), exist_ok=True)
    with open(p.source_file, "w") as f:
        f.write(p.code)
    try:
        r = evalPOI(p.pyobj, args=args, use_qjit=p.use_qjit)
        np.save(p.result_file, r)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        with open(p.error_file, "w") as f:
            f.write(str(e))
        return Result(p, None)
    return Result(p, r)

def render_result_marker(r, failed:list) -> str:
    if r.val is not None:
        return '.'
    else:
        failed.append(r.p.source_file)
        return 'X'

def render_match_marker(resPL, resC, failed:list) -> Tuple[bool,str]:
    if all((x is not None) for x in [resPL.val, resC.val]) and \
       allclose(resPL.val, resC.val, atol=1e-8):
        return True,'.'
    else:
        failed.extend([resPL.p.source_file, resC.p.source_file])
        return False,'M'

def batchrun(sample_spec, tag:Optional[str]=None):
    failed, solved, mismatched = [],[],[]
    arg = VName('arg')
    common_args = {
        'qnode_wires':8,
        'qnode_device':'lightning.qubit'
    }
    try:
        for b in greedy_enumerator(sample_spec, [arg]):
            print('|', end='', flush=True)
            # pprint(b)
            # input()

            pPL = problem_prepare(tag, b.pois[0].poi, [arg], ControlFlowStyle.Python,
                                  use_qjit=False, **common_args)
            resPL = problem_result(pPL)
            if resPL is None:
                resPL = problem_run(pPL, [(arg,1)])
                mPL = render_result_marker(resPL, failed)
            else:
                mPL = '_'
            if resPL.val is not None:
                solved.append(resPL.p.source_file)

            pC = problem_prepare(tag, b.pois[0].poi, [arg], ControlFlowStyle.Catalyst,
                                 use_qjit=True, **common_args)
            resC = problem_result(pC)
            if resC is None:
                resC = problem_run(pC, [(arg,1)])
                mC = render_result_marker(resC, failed)
            else:
                mC = '_'
            if resC.val is not None:
                solved.append(resC.p.source_file)

            _,mm = render_match_marker(resPL, resC, mismatched)
            print(f"{mPL}{mC}{mm}", end='', flush=True)
    finally:
        print()
        if len(solved)>0:
            print("Solved problems:")
            for r in sorted(set(solved)):
                print(dirname(r), basename(r))
        if len(failed)>0:
            print("Failed problems:")
            for r in sorted(set(failed)):
                print(dirname(r), basename(r))
        if len(mismatched)>0:
            print("Mismatched problems:")
            for r in sorted(set(mismatched)):
                print(dirname(r), basename(r))


def manualrun(sample_spec):
    arg = VName('arg')

    for b in greedy_enumerator(sample_spec, [arg]):
        print("1. Builder:")
        pprint(b)
        print("1. Press Enter to render")
        # input()
        o1,code1 = render(arg, ControlFlowStyle.Catalyst)
        o2,code2 = render(arg, ControlFlowStyle.Python)
        print("2. Rendered code:")
        print("^^^ Catalyst version ^^^")
        print(code1)
        print("^^^ PennyLane version ^^^")
        print(code2)
        print("2. Press Enter to evaluate and compare")
        input()
        r1 = evalPOI(o1, name="main", args=[(arg,1)])
        r2 = evalPOI(o2, name="main", args=[(arg,1)])
        print("3. Evaluation result:")
        print(r1)
        print(r2)
        assert_allclose(r1, r2)
        input()


if __name__ == "__main__":
    # manualrun(sample_spec1)
    # manualrun(sample_spec1)

    batchrun(sample_spec1, 'spec1')
    # batchrun(sample_spec5, 'spec5')

