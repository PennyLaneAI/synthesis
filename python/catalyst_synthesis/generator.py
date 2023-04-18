from typing import (Iterable, Dict, Union, List, Optional, NoReturn, Callable, Tuple, Any, Set)
from dataclasses import dataclass, astuple
from copy import deepcopy
from functools import reduce
from itertools import permutations, product, chain, cycle

from .grammar import (VName, FName, Expr, Stmt, VRefExpr, CondExpr, WhileLoopExpr, FDefStmt,
                      Program, RetStmt, ConstExpr, POI, ForLoopExpr, WhileLoopExpr, trueExpr,
                      falseExpr, ControlFlowStyle as CFS, assert_never, NoneExpr, saturate_expr1,
                      addExpr, lessExpr, Signature, signature, get_vars, callExpr, bind, POILike,
                      bless_poi)

from .builder import Builder, contextualize_poi, build, Context
from .pprint import pprint, pstr

def nemptypois(p:POILike) -> int:
    return len([x for x in contextualize_poi(bless_poi(p), Context())[0] if x.poi.isempty()])

def expanded_to(ls:List[Any], l:int)->List[Optional[Any]]:
    return [(ls[i] if i<len(ls) else None) for i in range(max(len(ls),l))]


def greedy_enumerator(poi_lib:List[POILike],
                      free_vars:List[VName]=[]) -> Iterable[Builder]:
    es = [bless_poi(x) for x  in poi_lib]
    ps = sum([nemptypois(p) for p in es], 1)
    vs = sum([get_vars(e.expr) for e in es], free_vars)
    for e_sample in permutations(range(len(es))):
        for p_sample in permutations(range(ps)):
            for v_sample in product(*([vs]*len(p_sample))):
                b = build(POI(), free_vars)
                try:
                    e_sample_ext = expanded_to(e_sample,len(v_sample))
                    for (p,e,v) in zip(p_sample, e_sample_ext, v_sample):
                        pwc = b.at(p)
                        ctx, poi = pwc.ctx, pwc.poi
                        if v not in ctx.get_vscope():
                            raise IndexError(f"{v} not in scope: {ctx.get_vscope()}")
                        e1 = deepcopy(es[e] if e is not None else bless_poi(v))
                        e2 = bless_poi(addExpr(ctx.statevar, e1)) if ctx.statevar else e1
                        b.update(p, e2, ignore_nonempty=True)
                    yield b
                except IndexError as err:
                    pass
                except ValueError as err:
                    print(err)

