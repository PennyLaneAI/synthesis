""" Output program pretty-printer functions

TODO: Decide on https://peps.python.org/pep-0634/
"""

from typing import Tuple, Union, List, Optional, NoReturn, Callable
from itertools import chain

from dataclasses import dataclass

from .grammar import (VName, FName, Expr, Stmt, FCallExpr, VRefExpr, AssignStmt,
                      CondExpr, WhileLoopExpr, FDefStmt, Program, RetStmt,
                      ConstExpr, NoneExpr, POI, ForLoopExpr, ControlFlowStyle,
                      assert_never, isinstance_expr, isinstance_stmt, ExprLike, bless_expr,
                      isinstance_exprlike, isinstance_array)

from .builder import (Builder)

DEFAULT_QDEVICE = "lightning.qubit"

DEFAULT_CFSTYLE = ControlFlowStyle.Catalyst

HintPrinter = Callable[[POI],List[str]]

CFS = ControlFlowStyle

@dataclass
class PStrOptions:
    """ Immutable options to carry around the pretty-printing procedures """
    default_cfstyle:ControlFlowStyle = DEFAULT_CFSTYLE
    hint:Optional[HintPrinter] = None


@dataclass
class Suffix:
    """ Mutable counter helping to produce `unique` entity names. """
    val:int

@dataclass
class PStrState:
    """ Mutable state to carry around the pretty-printing procedures. """
    indent:int = 0
    catalyst_cf_suffix:int = Suffix(0)

    def tabulate(self) -> "PStrState":
        return PStrState(self.indent+1, self.catalyst_cf_suffix)

    def issue(self, name) -> Tuple["PStrState",str]:
        self.catalyst_cf_suffix.val+=1
        s2 = PStrState(self.indent, self.catalyst_cf_suffix)
        return s2, f"{name}{self.catalyst_cf_suffix.val-1}"

def _in(st:PStrState, lines:List[str]) -> List[str]:
    """ Indent `lines` according to the current settings """
    return [' '*(st.indent*TABSTOP) + line for line in lines]

def _ne(st:PStrState, ls:List[str]) -> List[str]:
    """ Issue `pass` (a Python keyword) as a substitute of empty stmt list. """
    return ls if len(ls)>0 else [' '*((st.indent+1)*TABSTOP) + "pass"]

def _hi(st:PStrState, opt:Optional[PStrOptions], poi:POI) -> List[str]:
    """ Issue a hint formatted as a Python comment """
    hlines = opt.hint(poi) if opt and opt.hint else []
    if len(hlines) > 0:
        hlines = [f"poi {id(poi)}"] + hlines
    return [' '*st.indent*TABSTOP + "# " + h for h in hlines]


TABSTOP:int = 4

def _style(s, opt):
    s = (opt.default_cfstyle if opt else DEFAULT_CFSTYLE) if s==ControlFlowStyle.Default else s
    assert s != ControlFlowStyle.Default, f"Concrete CF style is expected at this point"
    return s

def _isinfix(e:FCallExpr) -> bool:
    return isinstance(e.expr, VRefExpr) and \
        all(c in "!>=<+-/*%" for c in e.expr.vname.val) and len(e.args)==2 and len(e.kwargs)==0

def _parens(e:POI, expr_str:str, opt) -> str:
    if isinstance(e.expr, (VRefExpr, ConstExpr)) or (isinstance(e.expr, FCallExpr) and not
                                                     _isinfix(e.expr)):
        return expr_str
    else:
        return f"({expr_str})"

def pstr_expr(expr:ExprLike,
              state:Optional[PStrState]=None,
              opt:Optional[PStrOptions]=None,
              arg_expr:Optional[List[POI]]=None,
              kwarg_expr:Optional[List[Tuple[str,POI]]]=None) -> Tuple[List[str],str]:
    """ Renders an expression-like value to a code. Returns a tuple of zero or more Python statement
    strings and a Python expression string.

    FIXME: Get rid of `arg_expr` / `kwarg_expr` -style recursive message-passing. """
    e = bless_expr(expr)
    st = state if state else PStrState(0,Suffix(0))
    if isinstance(e, FCallExpr):
        return pstr_expr(e.expr, state, opt, arg_expr=e.args, kwarg_expr=e.kwargs)
    elif isinstance(e, CondExpr):
        assert arg_expr is not None
        assert len(kwarg_expr)==0
        if _style(e.style, opt) == ControlFlowStyle.Python:
            acc, scond = pstr_expr(e.cond, st, opt)
            st1, svar = st.tabulate().issue("_cond")
            true_part = (
                _in(st, [f"if {scond}:"]) +
                _ne(st,
                    sum([pstr_stmt(s, st1, opt) for s in e.trueBranch.stmts], []) +
                    (pstr_stmt(AssignStmt(VName(svar), e.trueBranch.expr), st1, opt) if
                     e.trueBranch.expr else [])) +
                _hi(st1, opt, e.trueBranch))
            false_part = (
                _in(st, ["else:"]) +
                _ne(st,
                    sum([pstr_stmt(s, st1, opt) for s in e.falseBranch.stmts], []) +
                    (pstr_stmt(AssignStmt(VName(svar), e.falseBranch.expr), st1, opt) if
                     e.falseBranch.expr else [])) +
                _hi(st1, opt, e.falseBranch)) if e.falseBranch else []
            return (acc + true_part + false_part, svar)
        elif _style(e.style, opt) == ControlFlowStyle.Catalyst:
            acc, lcond = pstr_expr(e.cond, st, opt)
            st1, nmcond = st.tabulate().issue("cond")
            true_part = (
                _in(st, [f"@cond({lcond})",
                         f"def {nmcond}():"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.trueBranch.stmts], []) +
                        (pstr_stmt(RetStmt(e.trueBranch.expr), st1, opt) if e.trueBranch.expr else [])) +
                _hi(st1, opt, e.trueBranch))
            false_part = (
                _in(st, [f"@{nmcond}.otherwise",
                         f"def {nmcond}():"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.falseBranch.stmts], []) +
                        (pstr_stmt(RetStmt(e.falseBranch.expr), st1, opt) if e.falseBranch.expr else [])) +
                _hi(st1, opt, e.falseBranch)) if e.falseBranch else []
            return (acc + true_part + false_part, f"{nmcond}()")
        else:
            assert_never(e.style)
    elif isinstance(e, ForLoopExpr):
        assert len(arg_expr)==1
        assert len(kwarg_expr)==0
        if _style(e.style, opt) == ControlFlowStyle.Python:
            st1 = st.tabulate()
            accArg = pstr_stmt(AssignStmt(e.statevar, arg_expr[0].expr), st, opt) if e.statevar else []
            accL, lexprL = pstr_poi(e.lbound, st, opt)
            accU, lexprU = pstr_poi(e.ubound, st, opt)
            return (
                accArg + accL + accU +
                _in(st, [f"for {e.loopvar.val} in range({lexprL},{lexprU}):"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.body.stmts], []) +
                        (pstr_stmt(AssignStmt(e.statevar,e.body.expr), st1, opt)
                         if e.body.expr and e.statevar else [])) +
                _hi(st1, opt, e.body), (e.statevar.val if e.statevar else "None"))

        elif _style(e.style, opt) == ControlFlowStyle.Catalyst:
            accArg, sarg = pstr_expr(arg_expr[0].expr, st, opt)
            accL, lexprL = pstr_poi(e.lbound, st, opt)
            accU, lexprU = pstr_poi(e.ubound, st, opt)
            st1, nforloop = st.tabulate().issue("forloop")
            args = ','.join([e.loopvar.val] + ([e.statevar.val] if e.statevar else []))
            accLoop = (
                _in(st, [f"@for_loop({lexprL},{lexprU},1)",
                         f"def {nforloop}({args}):"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.body.stmts], []) +
                        (pstr_stmt(RetStmt(e.body.expr), st1, opt) if e.body.expr else [])) +
                _hi(st1, opt, e.body))
            return (accArg + accL + accU + accLoop, f"{nforloop}({sarg})")
        else:
            assert_never(e.style)
    elif isinstance(e, WhileLoopExpr):
        assert len(arg_expr)==1
        assert len(kwarg_expr)==0
        if _style(e.style, opt) == ControlFlowStyle.Python:
            accPoi, spoi = pstr_poi(arg_expr[0], st, opt)
            accArg = pstr_stmt(AssignStmt(e.statevar, VRefExpr(VName(spoi))), st, opt)
            accCond, lexpr = pstr_expr(e.cond, st, opt)
            st1, svar = st.tabulate().issue("_whileloop")
            return (
                accPoi +
                accArg +
                accCond +
                _in(st, [f"while {lexpr}:"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.body.stmts], []) +
                        (pstr_stmt(AssignStmt(e.statevar, e.body.expr), st1, opt) if e.body.expr else [])) +
                _hi(st1, opt, e.body),
                e.statevar.val)
        elif _style(e.style, opt) == ControlFlowStyle.Catalyst:
            accArg, sarg = pstr_poi(arg_expr[0], st, opt)
            accCond, lexpr = pstr_expr(e.cond, st, opt)
            st1, nwhileloop = st.tabulate().issue("whileloop")
            return (
                accArg + accCond +
                _in(st, [f"@while_loop(lambda {e.statevar.val}:{lexpr})",
                         f"def {nwhileloop}({e.statevar.val}):"]) +
                _ne(st, sum([pstr_stmt(s, st1, opt) for s in e.body.stmts],[]) +
                        (pstr_stmt(RetStmt(e.body.expr), st1, opt) if e.body.expr else [])) +
                _hi(st1, opt, e.body), f"{nwhileloop}({sarg})")
        else:
            assert_never(s.style)
    elif isinstance(e, NoneExpr):
        return [],"None"
    elif isinstance(e, VRefExpr):
        if arg_expr is None:
            return [],e.vname.val
        else:
            acc_body,args,kwargs = list(),list(),list()
            for ea in arg_expr:
                lss,le = pstr_poi(ea, state, opt)
                acc_body.extend(lss)
                args.append(le)
            for k,v in kwarg_expr:
                lss,le = pstr_poi(v, state, opt)
                acc_body.extend(lss)
                kwargs.append((k,le))
            if e.vname.val == '[]':
                assert len(kwarg_expr) == 0, "Brackets doesn't accept kwargs"
                return acc_body, f"[{', '.join(args)}]"
            elif len(arg_expr)==2 and _isinfix(FCallExpr(e, arg_expr, kwarg_expr)):
                return acc_body, f"{_parens(arg_expr[0], args[0], opt)} {e.vname.val} {_parens(arg_expr[1], args[1], opt)}"
            else:
                return acc_body, f"{e.vname.val}({', '.join(args + [(k+'='+v) for k,v in kwargs])})"
    elif isinstance(e, ConstExpr):
        if isinstance(e.val, bool): # Should be above 'int'
            return [],f"{e.val}"
        elif isinstance(e.val, int):
            return [],f"{e.val}"
        elif isinstance(e.val, float):
            return [],f"{e.val}"
        elif isinstance(e.val, complex):
            return [],f"{e.val}"
        elif isinstance(e.val, str):
            return [],f"\"{e.val}\""
        elif isinstance_array(e.val):
            return [],f"Array({e.val.tolist()},dtype={str(e.val.dtype)})"
        else:
            assert_never(e.val)
    else:
        assert_never(e)

def pstr_stmt(s:Stmt,
              state:Optional[PStrState]=None,
              opt:Optional[PStrOptions]=None) -> List[str]:
    """ Pretty-print a statement `s` into a lines of Python code """
    st:PStrState = state if state is not None else PStrState(0,Suffix(0))
    if isinstance(s, AssignStmt):
        acc, lexpr = pstr_expr(s.expr, st, opt)
        return acc + _in(st, [f"{s.vname.val if s.vname else '_'} = {lexpr}"])
    elif isinstance(s, FDefStmt):
        st1 = st.tabulate()
        qjit = ["@qjit"] if s.qjit else []
        qfunc = [f"@qml.qnode(qml.device(\"{s.qnode_device or 'default.qubit'}\", wires={s.qnode_wires or 1}))"] \
                if (s.qnode_device is not None) or (s.qnode_wires is not None) else []
        return (
            _in(st, qjit + qfunc +
                [f"def {s.fname.val}({', '.join([a.val for a in s.args])}):"]) +
            _ne(st, sum([pstr_stmt(s, st1, opt) for s in s.body.stmts],[]) +
                    (pstr_stmt(RetStmt(s.body.expr), st1, opt) if s.body.expr else [])) +
            _hi(st1, opt, s.body))
    elif isinstance(s, RetStmt):
        if s.expr is not None:
            acc, lexpr = pstr_expr(s.expr, st, opt)
            return acc + _in(st, [f"return {lexpr}"] )
        else:
            return _in(st, ["return"])
    else:
        assert_never(s)

def pstr_poi(p:POI, state=None, opt=None, arg_expr=None) -> Tuple[List[str],str]:
    """ Pretty-print the point of insertion. """
    st = state if state is not None else PStrState(0,Suffix(0))
    (lines, e) = pstr_expr(p.expr, st, opt, arg_expr) if p.expr else ([], "None")
    return (sum((pstr_stmt(s, st, opt) for s in p.stmts),[]) +
            lines + _hi(st, opt, p), e)

def _builder_hint_printer(b):
    def _hp(poi:POI) -> List[str]:
        for poic in b.pois:
            if poi is poic.poi:
                return [', '.join(v.val for v in sorted(poic.ctx.get_vscope()))]
        return []
    return _hp


def pstr_builder(b:Builder, st=None, opt=None) -> Tuple[List[str],str]:
    """ Pretty-print the expression Builder """
    opt = opt if opt else PStrOptions(DEFAULT_CFSTYLE, _builder_hint_printer(b))
    return pstr_poi(b.pois[0].poi, st, opt, arg_expr=[VRefExpr(VName('<?>'))])


def pstr_prog(p:Program, state=None, opt=None) -> List[str]:
    """ Pretty-print the program """
    return pstr_stmt(p, state, opt)


def pstr(p:Union[Builder, Program, Stmt, ExprLike], default_cfstyle=DEFAULT_CFSTYLE) -> str:
    """ Prints nearly any object in this library into a multi-line string of Python code. """
    if isinstance(p, Builder):
        opt = PStrOptions(default_cfstyle, _builder_hint_printer(p))
        lines, tail = pstr_builder(p, None, opt)
        return '\n'.join(lines+[f"## {tail} ##"])
    else:
        opt = PStrOptions(default_cfstyle)
        if isinstance(p, Program):
            return '\n'.join(pstr_prog(p, None, opt))
        elif isinstance_stmt(p):
            return '\n'.join(pstr_stmt(p, None, opt))
        elif isinstance_exprlike(p):
            stmts,tail = pstr_expr(p, None, opt, arg_expr=[VRefExpr(VName("<?>"))])
            return '\n'.join(stmts + [f"## {tail} ##"])
        else:
            assert_never(p)


def pprint(p:Union[Builder, Program, Stmt, ExprLike], default_cfstyle=CFS.Catalyst) -> None:
    """ Prints nearly any object in this library to a terminal. """
    print(pstr(p, default_cfstyle=default_cfstyle))

