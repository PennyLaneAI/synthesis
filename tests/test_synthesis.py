import os
import sys
from typing import Any, Dict, Tuple, Union, Callable, Set, List
from hypothesis import given, note, settings, Verbosity, assume
from inspect import signature as python_signature, _empty as inspect_empty
from tempfile import mktemp
from itertools import cycle
from math import nan, inf
from cmath import nanj, infj

import numpy as np

from pytest import mark
from dataclasses import astuple
from numpy.testing import assert_allclose

from catalyst_synthesis.grammar import (Expr, RetStmt, FCallExpr, VName, FName, VRefExpr, signature
                                        as expr_signature, isinstance_expr, AssignStmt,
                                        lessExpr, addExpr, eqExpr, ControlFlowStyle as CFS, signature,
                                        Signature, bind, saturate_expr, saturates_expr1,
                                        saturates_poi1, saturate_expr1, saturate_poi1, assignStmt,
                                        assignStmt_, callExpr, get_pois, bracketExpr, arrayExpr,
                                        saturate_poi, gateExpr, constExpr)

from catalyst_synthesis.pprint import (pstr_builder, pstr_stmt, pstr_expr, pprint, pstr,
                                       DEFAULT_CFSTYLE)
from catalyst_synthesis.builder import build, contextualize_expr, contextualize_poi, Context
from catalyst_synthesis.exec import compilePOI, evalPOI, runPOI, wrapInMain
from catalyst_synthesis.generator import nemptypois
from catalyst_synthesis.hypothesis import *


MAXINT64 = (1<<63) - 1

safe_integers = partial(integers, min_value=-(MAXINT64-1), max_value=MAXINT64)

def compilePOI_(*args, default_cfstyle=DEFAULT_CFSTYLE, **kwargs):
    s = wrapInMain(*args, **kwargs)
    print("Generated Python statement is:")
    print(pstr(s, default_cfstyle=default_cfstyle))
    return compilePOI(s, default_cfstyle=default_cfstyle)


def evalPOI_(p:POI, use_qjit=True, args:Optional[List[Tuple[Expr,Any]]]=None, **kwargs):
    arg_exprs = list(zip(*args))[0] if args is not None and len(args)>0 else []
    arg_all = args if args is not None else []
    o,code = compilePOI_(p, use_qjit=use_qjit, args=arg_exprs, **kwargs)
    return evalPOI(o, args=arg_all, **kwargs)


@given(d=data(), c=one_of([floats(), safe_integers()])) # complexes() doesn't work
@settings(max_examples=10)
def test_eval_whiles(d, c):
    assume(c != 0)
    l = d.draw(whileloops(lexpr=lambda x: just(eqExpr(x,c-c))))
    def render(style):
        return POI.fE(saturates_expr1(c-c,
                      saturates_poi1(c, l(style=style))))

    r1 = evalPOI_(render(CFS.Python), use_qjit=False)
    r2 = evalPOI_(render(CFS.Catalyst), use_qjit=True)
    assert_allclose(r1, r2)


@given(x = one_of([forloops(), conds()]),
       y = one_of([forloops(), conds()]),
       c = one_of([floats(), safe_integers()])) # complexes() doesn't work
@settings(max_examples=10)
def test_eval_fors_conds(x, y, c):
    def render(style):
        inner = saturates_poi1(c, saturates_expr1(c, y(style=style)))
        outer = saturates_expr1(c, saturates_poi1(inner, x(style=style)))
        return POI.fE(outer)

    r1 = evalPOI_(render(CFS.Python), use_qjit=False)
    r2 = evalPOI_(render(CFS.Catalyst), use_qjit=True)
    assert_allclose(r1, r2)


# @mark.parametrize('st', [CFS.Python, CFS.Catalyst])
# @given(d=data())
# @settings(verbosity=Verbosity.debug)
# def test_pprint_fdef_cflow(d, st):
#     x = d.draw(one_of([whileloops(style=st), forloops(style=st), conds(style=st)]))
#     f = FDefStmt(FName("main"), [], POI())
#     s = pstr(saturates_poi1(saturates_expr1(ConstExpr(42), saturates_poi1(ConstExpr(33), x())),
#                             FCallExpr(f,[])))
#     note(s)


# @mark.parametrize('st', [CFS.Python, CFS.Catalyst])
# @given(d=data())
# @settings(verbosity=Verbosity.debug)
# def test_pprint_ret_ctflow(d, st):
#     x = d.draw(one_of([whileloops(style=st), forloops(style=st), conds(style=st)]))
#     s = pstr(RetStmt( saturates_expr1(ConstExpr(42), saturates_poi1(ConstExpr(33), x))))
#     note(s)


@given(x=one_of([conds(),whileloops(),forloops()]))
def test_eq_expr(x):
    xa = saturates_poi1(POI(), x())
    xb = saturates_poi1(POI(), x())
    assert xa is not xb
    assert xa == xb
    xc = saturates_poi1(saturates_poi1(POI(), x()), x())
    assert xa != xc


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False))
@settings(max_examples=10)
def test_eval_const(x, use_qjit):
    assert np.array([x]) == evalPOI_(POI.fE(np.array([x])), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False), c=conds())
@settings(max_examples=10)
def test_eval_cond(c, x, use_qjit):
    jx = np.array([x])
    x2 = saturates_expr1(jx, saturates_poi1(jx, c()))
    assert jx == evalPOI_(POI.fE(x2), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False),
       l=forloops(lvars=just(VName('i')),svars=just(VName('s'))))
@settings(max_examples=10)
def test_eval_for(l, x, use_qjit):
    jx = np.array([x])
    r = saturates_expr1(jx, saturates_poi1(VRefExpr(VName('s')), l()))
    assert jx == evalPOI_(POI.fE(r), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False),
       l=whileloops(statevars=just(VName('i')),
                    lexpr=lambda _: just(falseExpr)))
@settings(max_examples=10)
def test_eval_while(l, x, use_qjit):
    jx = np.array([x])
    r = saturates_poi1(addExpr(VRefExpr(VName('i')), 1),
                       saturates_expr1(jx, l()))
    assert jx == evalPOI_(POI.fE(r), use_qjit=use_qjit)


@given(g=qgates, m=qmeasurements)
@settings(max_examples=100, verbosity=Verbosity.debug)
def test_eval_qops(g, m):
    evalPOI_(POI([assignStmt_(g)],m),
             use_qjit=True,
             qnode_device="lightning.qubit",
             qnode_wires=1)


@mark.parametrize('vals, dtype_str', [([],'int64'),
                                      ([0,-11,23233232],'int64'),
                                      ([],'float64'),
                                      ([0.1, 1.3e-3, nan, inf],'float64'),
                                      ([],'complex128'),
                                      ([0.1, 1.3e-3, -1.3j, nanj, infj],'complex128'),
                                      ])
def test_eval_arrays(vals, dtype_str):
    def _eval(use_qjit):
        return evalPOI_(POI.fE(arrayExpr(vals, dtype_str)),
                        use_qjit=use_qjit)
    r1 = _eval(use_qjit=True)
    r2 = _eval(use_qjit=False)
    print(r1)
    assert_allclose(r1, r2)


def test_build_mutable_layout():
    l = WhileLoopExpr(VName("i"), trueExpr, POI(), ControlFlowStyle.Catalyst)
    c = CondExpr(trueExpr, POI(), POI(), ControlFlowStyle.Catalyst)

    l_poi = l.body
    c_poi1 = c.trueBranch
    c_poi2 = c.falseBranch

    b_poi = POI()
    b = build(b_poi)
    assert len(b.pois)==1
    assert b.pois[0].poi is b_poi

    poi1 = POI.fromExpr(l)
    b.update(0, poi1)
    assert len(b.pois)==2
    assert b.pois[0].poi is b_poi
    assert b.pois[1].poi is l_poi

    poi2 = POI.fromExpr(c)
    b.update(1, poi2)
    assert len(b.pois)==4
    assert b.pois[0].poi is b_poi
    assert b.pois[1].poi is l_poi
    assert b.pois[2].poi is c_poi1
    assert b.pois[3].poi is c_poi2
    b.update(0, POI())
    assert len(b.pois)==1


def test_build_destructive_update():
    l = WhileLoopExpr(VName("i"), trueExpr, POI(), ControlFlowStyle.Catalyst)
    c = CondExpr(trueExpr, POI(), POI(), ControlFlowStyle.Catalyst)
    b = build(POI())
    b.update(0, POI.fE(saturate_expr1(l, 0)), ignore_nonempty=False)
    b.update(1, POI.fE(saturate_expr1(c, 1)), ignore_nonempty=False)
    assert len(b.pois)==5
    s1 = pstr_builder(b)
    b.update(0, b.pois[0].poi, ignore_nonempty=False)
    assert len(b.pois)==5
    s2 = pstr_builder(b)
    assert s1 == s2


def test_build_assign_layout():
    va = assignStmt(VName('a'),33)
    vb = assignStmt(VName('b'),42)
    l = WhileLoopExpr(VName("i"), trueExpr,
                      POI([vb],VName('b')), ControlFlowStyle.Catalyst)
    b = build(POI([va], saturate_expr1(l, 0)))
    s = pstr_builder(b)
    print(b.pois[0].ctx)


def test_build_fcall():
    f = callExpr(VName('qml.Hadamard'),[],[('wires',POI())])
    assert nemptypois(f) == 1
    b = build(POI())
    assert len(b.pois)==1
    b.update(0, POI.fE(f))
    assert len(b.pois)==2

@given(e=one_of([whileloops(),forloops(),conds()]))
def test_build_finds_pois(e):
    assert len(contextualize_expr(e())) == len(get_pois(e()))

def test_assign_finds_pois():
    poi = POI([assignStmt_(gateExpr('Somegate', wires=[POI(),POI(),POI()]))],ConstExpr(0))
    pwcs,_ = contextualize_poi(poi, Context())
    assert 3 == len([p for p in pwcs if p.poi.isempty()])

def test_build_assigns():
    poi = POI([assignStmt_(gateExpr('Somegate', wires=[POI(),POI(),POI()]))],ConstExpr(0))
    b = build(POI())
    b.update(0, poi, ignore_nonempty=True)
    assert len(b.pois)==4
    b.update(2, POI.fE(ConstExpr(0)))
    pprint(b)

@mark.parametrize('qnode_device', [None, "lightning.qubit"])
@mark.parametrize('use_qjit', [True, False])
@mark.parametrize('scalar', [0, -2.32323e10])
def test_run(use_qjit, qnode_device, scalar):
    val = np.array(scalar)
    source_file = mktemp("source.py")
    code, res = runPOI(POI.fE(val), use_qjit=use_qjit, qnode_device=qnode_device,
                       source_file=source_file)
    os.remove(source_file)
    assert res is not None
    assert_allclose(val, res)


ops = {
    "Identity": gateExpr('qml.Identity', wires=[POI()]),
    # "Snapshot": qml.Snapshot("label"),
    "BasisState": gateExpr('qml.BasisState', np.array([0]), wires=[POI()]),
    "CNOT": gateExpr('qml.CNOT', wires=[POI(), POI()]),
    "CRX": gateExpr('qml.CRX', 0, wires=[POI(), POI()]),
    "CRY": gateExpr('qml.CRY', 0, wires=[POI(), POI()]),
    "CRZ": gateExpr('qml.CRZ', 0, wires=[POI(), POI()]),
    "CRY": gateExpr('qml.CRY', 0, wires=[POI(), POI()]),
    "CRZ": gateExpr('qml.CRZ', 0, wires=[POI(), POI()]),
    "CRot": gateExpr('qml.CRot', 0, 0, 0, wires=[POI(), POI()]),
    "CSWAP": gateExpr('qml.CSWAP', wires=[POI(), POI(), POI()]),
    "CZ": gateExpr('qml.CZ', wires=[POI(), POI()]),
    "CCZ": gateExpr('qml.CCZ', wires=[POI(), POI(), POI()]),
    "CY": gateExpr('qml.CY', wires=[POI(), POI()]),
    "CH": gateExpr('qml.CH', wires=[POI(), POI()]),
    # "DiagonalQubitUnitary": gateExpr('qml.DiagonalQubitUnitary', np.array([1, 1]), wires=[POI()]),
    "Hadamard": gateExpr('qml.Hadamard', wires=[POI()]),
    "MultiRZ": gateExpr('qml.MultiRZ', 0, wires=[POI()]),
    "PauliX": gateExpr('qml.PauliX', wires=[POI()]),
    "PauliY": gateExpr('qml.PauliY', wires=[POI()]),
    "PauliZ": gateExpr('qml.PauliZ', wires=[POI()]),
    "PhaseShift": gateExpr('qml.PhaseShift', 0, wires=[POI()]),
    "ControlledPhaseShift": gateExpr('qml.ControlledPhaseShift', 0, wires=[POI(), POI()]),
    "CPhaseShift00": gateExpr('qml.CPhaseShift00', 0, wires=[POI(), POI()]),
    "CPhaseShift01": gateExpr('qml.CPhaseShift01', 0, wires=[POI(), POI()]),
    "CPhaseShift10": gateExpr('qml.CPhaseShift10', 0, wires=[POI(), POI()]),
    "QubitStateVector": gateExpr('qml.QubitStateVector', np.array([1.0, 0.0]), wires=[POI()]),
    # "QubitDensityMatrix": gateExpr('qml.QubitDensityMatrix', np.array([[0.5, 0.0], [0, 0.5]]), wires=[POI()]),
    "QubitUnitary": gateExpr('qml.QubitUnitary', np.eye(2), wires=[POI()]),
    "ControlledQubitUnitary": gateExpr('qml.ControlledQubitUnitary', np.eye(2), control_wires=[POI()], wires=[POI()]),
    "MultiControlledX": gateExpr('qml.MultiControlledX', control_wires=[POI(), POI()], wires=[POI()]),
    "IntegerComparator": gateExpr('qml.IntegerComparator', 1, geq=True, wires=[POI(), POI(), POI()]),
    "RX": gateExpr('qml.RX', 0, wires=[POI()]),
    "RY": gateExpr('qml.RY', 0, wires=[POI()]),
    "RZ": gateExpr('qml.RZ', 0, wires=[POI()]),
    "Rot": gateExpr('qml.Rot', 0, 0, 0, wires=[POI()]),
    "S": gateExpr('qml.S', wires=[POI()]),
    # "Adjoint(S)": gateExpr('qml.adjoint(qml.S)', wires=[0])),
    "SWAP": gateExpr('qml.SWAP', wires=[POI(), POI()]),
    "ISWAP": gateExpr('qml.ISWAP', wires=[POI(), POI()]),
    "PSWAP": gateExpr('qml.PSWAP', 0, wires=[POI(), POI()]),
    "ECR": gateExpr('qml.ECR', wires=[POI(), POI()]),
    # "Adjoint(ISWAP)": gateExpr('qml.adjoint(qml.ISWAP)', wires=[POI(), POI()])),
    "T": gateExpr('qml.T', wires=[POI()]),
    # "Adjoint(T)": gateExpr('qml.adjoint', qml.T(wires=[0])),
    "SX": gateExpr('qml.SX', wires=[POI()]),
    # "Adjoint(SX)": gateExpr('qml.adjoint', qml.SX(wires=[0])),
    "Barrier": gateExpr('qml.Barrier', wires=[POI(), POI(), POI()]),
    "WireCut": gateExpr('qml.WireCut', wires=[POI()]),

    "Toffoli": gateExpr('qml.Toffoli', wires=[POI(), POI(), POI()]),
    "QFT": gateExpr('qml.templates.QFT', wires=[POI(), POI(), POI()]),
    "IsingXX": gateExpr('qml.IsingXX', 0, wires=[POI(), POI()]),
    "IsingYY": gateExpr('qml.IsingYY', 0, wires=[POI(), POI()]),
    "IsingZZ": gateExpr('qml.IsingZZ', 0, wires=[POI(), POI()]),
    "IsingXY": gateExpr('qml.IsingXY', 0, wires=[POI(), POI()]),
    "SingleExcitation": gateExpr('qml.SingleExcitation', 0, wires=[POI(), POI()]),
    "SingleExcitationPlus": gateExpr('qml.SingleExcitationPlus', 0, wires=[POI(), POI()]),
    "SingleExcitationMinus": gateExpr('qml.SingleExcitationMinus', 0, wires=[POI(), POI()]),
    "DoubleExcitation": gateExpr('qml.DoubleExcitation', 0, wires=[POI(), POI(), POI(), POI()]),
    # "DoubleExcitationPlus": gateExpr('qml.DoubleExcitationPlus', 0, wires=[POI(), POI(), POI(), POI()]),
    # "DoubleExcitationMinus": gateExpr('qml.DoubleExcitationMinus', 0, wires=[POI(), POI(), POI(), POI()]),
    "QubitCarry": gateExpr('qml.QubitCarry', wires=[POI(), POI(), POI(), POI()]),
    "QubitSum": gateExpr('qml.QubitSum', wires=[POI(), POI(), POI()]),
    "PauliRot": gateExpr('qml.PauliRot', 0, constExpr("XXYY"), wires=[POI(), POI(), POI(), POI()]),
    "U1": gateExpr('qml.U1', 0, wires=[POI()]),
    "U2": gateExpr('qml.U2', 0, 0, wires=[POI()]),
    "U3": gateExpr('qml.U3', 0, 0, 0, wires=[POI()]),
    "SISWAP": gateExpr('qml.SISWAP', wires=[POI(), POI()]),
    # "Adjoint(SISWAP)": gateExpr('qml.adjoint(qml.SISWAP(wires=[0, 1])),
    "OrbitalRotation": gateExpr('qml.OrbitalRotation', 0, wires=[POI(), POI(), POI(), POI()]),
    "FermionicSWAP": gateExpr('qml.FermionicSWAP', 0, wires=[POI(), POI()]),
}

@mark.parametrize('gname, g', ops.items())
def test_eval_gates(gname, g):
    nwires = 4
    def _eval(use_qjit):
        return evalPOI_(POI([assignStmt_(saturate_poi(g, iter(range(nwires))))],
                            callExpr("qml.state",[])),
                        qnode_device='lightning.qubit',
                        qnode_wires=nwires,
                        use_qjit=use_qjit,
                        name=f"main_{gname}")
    r1 = _eval(use_qjit=True)
    r2 = _eval(use_qjit=False)
    print(r1)
    assert_allclose(r1, r2, atol=1e-10)

