from typing import (Iterable, Dict, Union, List, Optional, NoReturn, Callable, Tuple, Any, Set)
from dataclasses import dataclass, astuple
from copy import deepcopy
from functools import reduce
from itertools import permutations, product, chain, cycle

from .grammar import (VName, FName, Expr, Stmt, VRefExpr, AssignStmt, CondExpr,
                      WhileLoopExpr, FDefStmt, Program, RetStmt, ConstExpr, POI, ForLoopExpr,
                      WhileLoopExpr, trueExpr, falseExpr, ControlFlowStyle as CFS, assert_never,
                      NoneExpr, saturate_expr1, addExpr, lessExpr, Signature,
                      AssignStmt, signature, get_vars, assignStmt, assignStmt_, callExpr, bind)

from .builder import Builder, contextualize_expr, build
from .pprint import pprint, pstr

def nemptypois(e:Expr) -> int:
    return len([p for p in contextualize_expr(e) if p.poi.isempty()])

def expanded_to(ls:List[Any], l:int)->List[Optional[Any]]:
    return [(ls[i] if i<len(ls) else None) for i in range(max(len(ls),l))]

# def bindExpr(a:Optional[Expr], b:Optional[Expr]) -> Optional[Expr]:
#     if all((x is None) for x in [a,b]):
#         return None
#     if a is None:
#         return b
#     elif b is None:
#         return a
#     else:
#         return FCallExpr(b, [POI.fE(a)]) if signature(b).isfunc(1) else None

# def bindPOI(a:POI, b:POI) -> POI:
#     e = bindExpr(a.expr, b.expr)
#     if e is None:
#         raise ValueError(f"No bind for\n{pstr(a)}\n{pstr(b)}")
#     return bind(a, b, e)


def control_flows(expr_lib:List[Expr],
                  gate_lib:List[Tuple[FName,Signature]],
                  free_vars:List[VName]=[]) -> Iterable[Builder]:
    gs = gate_lib if gate_lib else [None]
    es = expr_lib
    ps = sum([nemptypois(e) for e in expr_lib], 1)
    vs = sum([get_vars(e) for e in expr_lib], free_vars)
    nargs = max(chain([0],(len(s.args) for _,s in gate_lib))) + \
            max(len(signature(e).args if signature(e).args else []) for e in expr_lib)
    for e_sample in permutations(es):
        for p_sample in permutations(range(ps)):
        # for p_sample in product(*([range(ps)]*ps)):
            print(p_sample)
            for g_sample in product(*[gs]*len(p_sample)):
                args = list(product(*([vs]*max(2,nargs))))
                for v_sample in product(*([args]*len(p_sample))):
                    b = build(POI(), free_vars)
                    try:
                        e_sample_ext = expanded_to(e_sample,len(v_sample))
                        for p,g,e,v in zip(p_sample,
                                           g_sample,
                                           e_sample_ext,
                                           v_sample):
                            pwc = b.at(p)
                            ctx, poi = pwc.ctx, pwc.poi
                            assert len(v) >= 2, f"len({v}) < 2"
                            if not all(vi in ctx.get_vscope() for vi in v):
                                raise IndexError(f"{v} not in scope: {ctx.get_vscope()}")
                            stmts = [assignStmt_(callExpr(g[0], [v[0]]))] if g else []
                            res = saturate_expr1(e if e else v[1], v[1])
                            expr = addExpr(ctx.statevar, res) if ctx.statevar else res
                            b.update(p, POI(stmts, expr), ignore_nonempty=True)
                        yield b
                    except IndexError as err:
                        pass
                    except ValueError as err:
                        print(err)

