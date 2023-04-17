import os
import sys
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
                                        as expr_signature, isinstance_expr, AssignStmt,
                                        lessExpr, addExpr, eqExpr, ControlFlowStyle as CFS, signature,
                                        Signature, bind, saturate_expr, saturates_expr1,
                                        saturates_poi1, saturate_expr1, saturate_poi1, assignStmt,
                                        assignStmt_, callExpr)

from catalyst_synthesis.pprint import (pstr_builder, pstr_stmt, pstr_expr, pprint, pstr,
                                       DEFAULT_CFSTYLE)
from catalyst_synthesis.builder import build
from catalyst_synthesis.exec import compilePOI, evalPOI, runPOI, wrapInMain
from catalyst_synthesis.generator import control_flows
from catalyst_synthesis.hypothesis import *


MAXINT64 = (1<<63) - 1

safe_integers = partial(integers, min_value=-(MAXINT64-1), max_value=MAXINT64)

def compilePOI_(*args, default_cfstyle=DEFAULT_CFSTYLE, **kwargs):
    s = wrapInMain(*args, **kwargs)
    print("Generated Python statement is:")
    print('\n'.join(pstr(s, default_cfstyle=default_cfstyle)))
    return compilePOI(s, default_cfstyle=default_cfstyle)


def evalPOI_(p:POI, use_qjit=True, args:Optional[List[Tuple[Expr,Any]]]=None, **kwargs):
    arg_exprs = list(zip(*args))[0] if args is not None and len(args)>0 else []
    arg_all = args if args is not None else []
    o,code = compilePOI_(p, use_qjit=use_qjit, args=arg_exprs, **kwargs)
    return evalPOI(o, args=arg_all)


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
    assert jnp.array([x]) == evalPOI_(POI.fE(jnp.array([x])), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False), c=conds())
@settings(max_examples=10)
def test_eval_cond(c, x, use_qjit):
    jx = jnp.array([x])
    x2 = saturates_expr1(jx, saturates_poi1(jx, c()))
    assert jx == evalPOI_(POI.fE(x2), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False),
       l=forloops(lvars=just(VName('i')),svars=just(VName('s'))))
@settings(max_examples=10)
def test_eval_for(l, x, use_qjit):
    jx = jnp.array([x])
    r = saturates_expr1(jx, saturates_poi1(VRefExpr(VName('s')), l()))
    assert jx == evalPOI_(POI.fE(r), use_qjit)


@mark.parametrize('use_qjit', [True, False])
@given(x=complexes(allow_nan=False, allow_infinity=False),
       l=whileloops(statevars=just(VName('i')),
                    lexpr=lambda _: just(falseExpr)))
@settings(max_examples=10)
def test_eval_while(l, x, use_qjit):
    jx = jnp.array([x])
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


@mark.parametrize('qnode_device', [None, "lightning.qubit"])
@mark.parametrize('use_qjit', [True, False])
@mark.parametrize('scalar', [0, -2.32323e10])
def test_run(use_qjit, qnode_device, scalar):
    val = jnp.array(scalar)
    source_file = mktemp("source.py")
    code, res = runPOI(POI.fE(val), use_qjit=use_qjit, qnode_device=qnode_device,
                       source_file=source_file)
    os.remove(source_file)
    assert res is not None
    assert_allclose(val, res)


