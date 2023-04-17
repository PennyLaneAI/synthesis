""" This module defines the abstract syntax tree structures for the output
language. The `Program` dataclass is supposed to be top-level node of the AST.
"""

from typing import (Any, List, Tuple, Optional, Dict, Union, NoReturn, Set, Callable, Iterable,
                    Generator)
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum
from functools import reduce
from itertools import cycle, chain


@dataclass
class POI:
    """ Point Of Insertion. By convention, we allow inserting new
    statements strictly at the end of the list of already existing ones. """
    stmts:List["Stmt"]
    expr:Optional["Expr"]

    def __init__(self, stmts=None, expr=None):
        self.stmts = stmts if stmts is not None else []
        self.expr = bless_expr(expr) if expr is not None else None

    def __hash__(self):
        return hash((tuple(self.stmts), self.expr))

    def isempty(self) -> bool:
        return len(self.stmts)==0 and (self.expr is None)

    @classmethod
    def fromExpr(cls, e:"ExprLike") -> "POI":
        return POI([], e)

    @classmethod
    def fE(cls, *args, **kwargs) -> "POI":
        """ Alias """
        return cls.fromExpr(*args, **kwargs)


POILike = Union[POI, "ExprLike"]


@dataclass(frozen=True)
class VName:
    """ Alias for strings representing variable names """
    val: str

    def __lt__(self, other):
        return self.val < other.val

@dataclass(frozen=True)
class FName:
    """ Alias for strings representing function names """
    val: str

    def __lt__(self, other):
        return self.val < other.val

Expr = Union["VRefExpr","FCallExpr", "ConstExpr", "NoneExpr", "CondExpr",
             "ForLoopExpr", "WhileLoopExpr" ]

def isinstance_expr(e:Any) -> bool:
    """Workaround for a TypeError, which is probably a Python bug."""
    return isinstance(e, (VRefExpr,FCallExpr, ConstExpr, NoneExpr, CondExpr,
                          ForLoopExpr, WhileLoopExpr))

@dataclass(frozen=True)
class VRefExpr:
    """ Expression - reference to a variable or a function. """
    vname: Union[FName, VName]


ConstExprVal = Union[bool, int, float, complex, str]

def isinstance_array(e:Any) -> bool:
    return all(hasattr(e,a) for a in ["shape","dtype","ndim","tolist"])

def isinstance_cval(e:Any) -> bool:
    return isinstance(e, (bool, int, float, complex, str)) or isinstance_array(e)

@dataclass(frozen=True)
class ConstExpr:
    """ Expression - constant """
    val: ConstExprVal

@dataclass(frozen=True)
class NoneExpr:
    """ Alias for None """
    pass

class ControlFlowStyle(Enum):
    Default = 0
    Python = 1
    Catalyst = 2
    # JAX = 3

@dataclass(frozen=True)
class CondExpr:
    """ Expression - conditional """
    cond: Expr
    trueBranch: POI
    falseBranch: Optional[POI]
    style: ControlFlowStyle

@dataclass(frozen=True)
class ForLoopExpr:
    """ Expression - for loop """
    loopvar: VName
    lbound: POI
    ubound: POI
    body: POI
    style: ControlFlowStyle
    statevar: Optional[VName] = None # TODO: Auto-generate this name in `pprint`

@dataclass(frozen=True)
class WhileLoopExpr:
    """ Expression - while loop """
    statevar: VName
    cond: Expr
    body: POI
    style: ControlFlowStyle

@dataclass(frozen=True)
class FCallExpr:
    """ Expression - calling a callable """
    expr: Union[VRefExpr, CondExpr, ForLoopExpr, WhileLoopExpr]
    args: List[POI]
    kwargs: List[Tuple[str,POI]]

    def __hash__(self):
        return hash((self.expr, tuple(self.args), tuple(self.kwargs)))


ExprLike = Union[Expr, ConstExprVal, VName, FName]

def isinstance_exprlike(e:Any) -> bool:
    return isinstance_expr(e) or isinstance_cval(e) or isinstance(e, (VName, FName))


Stmt = Union["AssignStmt", "FDefStmt", "RetStmt"]

def isinstance_stmt(s:Any) -> bool:
    """Workaround for a TypeError, which is probably a Python bug."""
    return isinstance(s, (AssignStmt, FDefStmt, RetStmt))

@dataclass(frozen=True)
class AssignStmt:
    """ Statement - variable assignemnt or a function call """
    vname: Optional[VName]
    expr: Expr

    @classmethod
    def fE(cls, e:Expr) -> "AssignStmt":
        return AssignStmt(None, e)

def assignStmt(v, e):
    return AssignStmt(v, bless_expr(e))

def assignStmt_(e):
    return assignStmt(None, e)

@dataclass(frozen=True)
class FDefStmt:
    """ Statement - function declaration """
    fname: FName
    args: List[VName]
    body: POI
    qnode_wires: Optional[int] = None
    qnode_device: Optional[str] = None
    qjit: bool = False

@dataclass(frozen=True)
class RetStmt:
    """ Statement - return """
    expr: Optional[Expr]

Program = FDefStmt
""" Top-level program is a function declaration """


def assert_never(x: Any) -> NoReturn:
    raise RuntimeError("Unhandled type: {}".format(type(x).__name__))


def bless_expr(e:ExprLike) -> Expr:
    """ Casts expression-like values to expressions. """
    if isinstance_expr(e):
        return e
    elif isinstance_cval(e):
        return ConstExpr(e)
    elif isinstance(e, (VName,FName)):
        return VRefExpr(e)
    else:
        assert_never(e)

def bless_poi(x:POILike) -> POI:
    """ Casts POI-like values to POIs. """
    if isinstance(x, POI):
        return x
    elif isinstance_exprlike(x):
        return POI.fromExpr(bless_expr(x))
    else:
        assert_never(x)


@dataclass
class Signature:
    """ A poor man's type system: Signatures allow to destinguish function expressions from value
    expressions. Note that loops and conditionals are function expressions in this grammar. """
    args:Optional[List[str]]
    ret:str

    def isfunc(self)->bool:
        return self.args is not None
    def isval(self)->bool:
        return not self.isfunc()


def signature(x: Union[FDefStmt, Expr]) -> Signature:
    """ Calculates the expression signature """
    if isinstance(x, FDefStmt):
        return Signature(['*']*len(x.args), "*")
    elif isinstance(x, ForLoopExpr):
        return Signature(["*"], "*")
    elif isinstance(x, WhileLoopExpr):
        return Signature(["*"], "*")
    elif isinstance(x, CondExpr):
        return Signature([], "*")
    else:
        return Signature(None, "*")


def bind(a:POI, b:POI, expr:Expr) -> POI:
    return POI(a.stmts + b.stmts, bless_expr(expr))


def bindUnary(value:POI, function:POI) -> Optional[POI]:
    s = signature(function.expr)
    if len(s.args) != 1:
        return None
    return bind(value, function, FCallExpr(function.expr, [value.expr]))

Acc = Any
""" Alias for "accumulator" """

def reduce_stmt_expr(e:Union[Stmt,Expr], f:Callable[[Union[Stmt,Expr],Acc],Acc], acc:Acc) -> Acc:
    """ Statement reduction function able to handle many simple cases.
    TODO: Think of implementing a full-scale catamorphic folding.
    FIXME: Use generators/chains where possible. """
    def _down(subexprs):
        return reduce(lambda acc,se: reduce_stmt_expr(se,f,acc), subexprs, f(e,acc))
    def _unpoi(poi):
        return poi.stmts + ([poi.expr] if poi.expr else [])
    if isinstance(e, FCallExpr):
        return _down([e.expr] + sum(map(_unpoi, chain(e.args,(a[1] for a in e.kwargs))),[]))
    elif isinstance(e, CondExpr):
        return _down(_unpoi(e.trueBranch) + (_unpoi(e.falseBranch) if e.falseBranch else []))
    elif isinstance(e, ForLoopExpr):
        return _down(_unpoi(e.lbound) + _unpoi(e.ubound) + _unpoi(e.body))
    elif isinstance(e, WhileLoopExpr):
        return _down([e.cond] + _unpoi(e.body))
    elif isinstance(e, (NoneExpr, VRefExpr, ConstExpr)):
        return _down([])
    elif isinstance(e, AssignStmt):
        return _down([e.expr])
    elif isinstance(e, RetStmt):
        return _down([e.expr])
    elif isinstance(e, FDefStmt):
        return _down(e.body.stmts + ([e.body.expr] if e.body.expr else []))
    else:
        assert_never(e)


def get_vars(e:Union[Stmt,Expr]) -> List[VName]:
    """ Queries variable definitions """
    def _vars(e):
        if isinstance(e, ForLoopExpr):
            return [e.loopvar] + ([e.statevar] if e.statevar else [])
        elif isinstance(e, WhileLoopExpr):
            return [e.statevar]
        elif isinstance(e, VRefExpr):
            return [e.vname] if isinstance(e.vname, VName) else []
        elif isinstance(e, FDefStmt):
            return e.args
        elif isinstance(e, AssignStmt):
            return [e.vname] if e.vname else []
        else:
            return []
    return reduce_stmt_expr(e, lambda e,acc: acc+_vars(e), [])



def get_pois(e:Union[Stmt,Expr]) -> List[POI]:
    """ Queries all POIs """
    def _pois(e):
        if isinstance(e, ForLoopExpr):
            return [e.lbound, e.ubound, e.body]
        elif isinstance(e, WhileLoopExpr):
            return [e.body]
        elif isinstance(e, CondExpr):
            return [e.trueBranch] + ([e.falseBranch] if e.falseBranch else [])
        elif isinstance(e, FDefStmt):
            return [e.body]
        elif isinstance(e, FCallExpr):
            return e.args + [a[1] for a in e.kwargs]
        else:
            return []
    return reduce_stmt_expr(e, lambda e,acc: acc+_pois(e), [])


def saturate_expr(e:ExprLike, args:Iterable[POILike]) -> Expr:
    """ Saturates function-like expression with arguments by wrapping it with a function call.
    Returns value-like expression. """
    e2 = deepcopy(e)
    s = signature(e2)
    return callExpr(e2, [next(args) for _ in s.args]) if s.isfunc() else e2

def saturate_poi(e:ExprLike, args:Iterable[POILike]) -> Expr:
    """ Saturates empty POIs of the expression `e` with the argument POIs. """
    e2 = bless_expr(deepcopy(e))
    for poi in get_pois(e2):
        if poi.expr is None:
            arg = bless_poi(next(args))
            poi.stmts = deepcopy(arg.stmts)
            poi.expr = deepcopy(arg.expr)
    return e2

def saturate_expr1(e, arg):
    return saturate_expr(e, cycle([arg]))

def saturate_poi1(e, arg):
    return saturate_poi(e, cycle([arg]))

def saturates_expr(args, e):
    return saturate_expr(e, args)

def saturates_poi(args, e):
    return saturate_poi2(e, args)

def saturates_expr1(arg, e):
    return saturate_expr(e, cycle([arg]))

def saturates_poi1(arg, e):
    return saturate_poi(e, cycle([arg]))



trueExpr = ConstExpr(True)
falseExpr = ConstExpr(False)

def callExpr(e:Union[str,ExprLike],
             args:List[POILike],
             kwargs:Optional[List[Tuple[str,POILike]]]=None) -> FCallExpr:
    kwargs = kwargs if kwargs is not None else []
    return FCallExpr(bless_expr(FName(e)) if isinstance(e,str) else bless_expr(e),
                     [bless_poi(e) for e in args],
                     [(k,bless_poi(v)) for k,v in kwargs])

def lessExpr(a,b) -> FCallExpr:
    return callExpr(FName('<'),[a,b])

def addExpr(a,b) -> FCallExpr:
    return callExpr(FName('+'),[a,b])

def neqExpr(a,b) -> FCallExpr:
    return callExpr(FName('!='),[a,b])

def eqExpr(a,b) -> FCallExpr:
    return callExpr(FName('=='),[a,b])

def bracketExpr(args) -> FCallExpr:
    return callExpr(FName('[]'), args)

def gateExpr(name:str, *args,
             wires:List[ExprLike],
             control_wires:Optional[List[ExprLike]]=None,
             **kwargs) -> FCallExpr:
    return callExpr(name, args=args, kwargs=(
        [('wires', bracketExpr(wires))] +
        ([('control_wires', bracketExpr(control_wires))] if control_wires else []) +
        (list(kwargs.items()))
    ))

def arrayExpr(args:List[ExprLike], dtype:str) -> FCallExpr:
    return callExpr("Array", args=[bracketExpr(args)], kwargs=[('dtype',VName(dtype))])

