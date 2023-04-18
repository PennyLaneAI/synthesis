from typing import (Iterable, Dict, Union, List, Optional, NoReturn, Callable, Tuple, Any, Set)
from dataclasses import dataclass, astuple
from copy import deepcopy
from functools import reduce
from itertools import permutations, product, chain, cycle

from .grammar import (VName, FName, Expr, Stmt, VRefExpr, AssignStmt, CondExpr,
                      WhileLoopExpr, FDefStmt, Program, RetStmt, ConstExpr, POI, ForLoopExpr,
                      WhileLoopExpr, trueExpr, falseExpr, ControlFlowStyle as CFS, assert_never,
                      NoneExpr, saturate_expr1, addExpr, lessExpr, Signature,
                      AssignStmt, signature, get_vars, assignStmt, assignStmt_, callExpr, bind,
                      POILike, bless_poi)

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


def control_flows(poi_lib:List[POILike],
                  # gate_lib:List[Tuple[FName,Signature]],
                  free_vars:List[VName]=[]) -> Iterable[Builder]:
    # gs = gate_lib if gate_lib else [None]
    es = [bless_poi(x) for x  in poi_lib]
    ps = sum([nemptypois(e.expr) for e in es], 1)
    vs = sum([get_vars(e.expr) for e in es], free_vars)
    # nargs = max(len(signature(e).args if signature(e).args else []) for e in expr_lib)
    for e_sample in permutations(range(len(es))):
        for p_sample in permutations(range(ps)):
        # for p_sample in product(*([range(ps)]*ps)):
            print(p_sample)
            for v_sample in product(*([vs]*len(p_sample))):
                # print(v_sample)
                b = build(POI(), free_vars)
                try:
                    e_sample_ext = expanded_to(e_sample,len(v_sample))
                    print(e_sample_ext)
                    for (p,
                         # g,
                         e,
                         v) in zip(p_sample,
                                       # g_sample,
                                       e_sample_ext,
                                       v_sample):
                        pwc = b.at(p)
                        ctx, poi = pwc.ctx, pwc.poi
                        # assert len(v) >= 2, f"len({v}) < 2"
                        if v not in ctx.get_vscope():
                            raise IndexError(f"{v} not in scope: {ctx.get_vscope()}")
                        # stmts = [assignStmt_(callExpr(g[0], [v[0]]))] if g else []
                        # e2 = saturate_expr1(e.expr if e and e.expr else v[1], v[1])
                        e1 = deepcopy(es[e] if e is not None else bless_poi(v))
                        e2 = bless_poi(addExpr(ctx.statevar, e1)) if ctx.statevar else e1
                        b.update(p, e2, ignore_nonempty=True)
                    yield b
                except IndexError as err:
                    print(err)
                except ValueError as err:
                    print(err)

