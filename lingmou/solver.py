"""
灵眸智算 · 求解 dispatcher + 回代校验
================================================
职责：吃一个合法 DSL，用 SymPy 给出**精确解**，并尽力**回代校验**。

设计要点：
  * 算数全部交给 SymPy（确定性、毫秒级），模型一律不参与计算。
  * 每个结果都带一个 verification（校验报告）：
        status ∈ {"passed", "failed", "skipped"}
    —— 这是本方案区别于"大模型直接解题"的可信护城河，界面上要显眼地展示。
  * 表达式一律按 SymPy/Python 语法解析（"x**2"、"2*x+y"），不碰 LaTeX。
"""

from __future__ import annotations
from typing import Any
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
)

from .dsl import validate, DSLError

# 允许 "2x" 这种隐式乘法、^ 当幂等友好写法
_TF = standard_transformations + (implicit_multiplication_application,)


def _sym(name: str) -> sp.Symbol:
    return sp.Symbol(name)


def _parse(expr: str, local: dict | None = None):
    """把纯文本表达式解析成 SymPy 对象。^ 视作幂。"""
    if expr is None:
        raise DSLError("空表达式。")
    s = str(expr).replace("^", "**")
    try:
        return parse_expr(s, transformations=_TF, local_dict=local or {}, evaluate=True)
    except Exception as e:
        raise DSLError(f"无法解析表达式 `{expr}`：{e}")


def _to_eq(s: str):
    """把 "lhs = rhs" 转成 SymPy 的 Eq；没有等号则视为 expr = 0。"""
    if "=" in s:
        l, r = s.split("=", 1)
        return sp.Eq(_parse(l), _parse(r))
    return sp.Eq(_parse(s), 0)


# ---------------------------------------------------------------- 各 task 求解
def _solve_evaluate(dsl):
    expr = _parse(dsl["expr"])
    exact = sp.simplify(expr)
    verify = {"status": "skipped", "detail": "单表达式求值无需回代。"}
    return exact, verify


def _solve_solve(dsl):
    vars_ = [_sym(v) for v in dsl["vars"]]
    eqs = [_to_eq(e) for e in dsl["equations"]]
    sol = sp.solve(eqs, vars_, dict=True)
    if not sol:
        return None, {"status": "failed", "detail": "SymPy 未找到解（可能无解或建模有误）。"}
    # ---- 回代校验：把每组解代回每个方程，检查两边是否相等 ----
    reports = []
    all_ok = True
    for i, s in enumerate(sol):
        ok = True
        for eq in eqs:
            residual = sp.simplify(eq.lhs.subs(s) - eq.rhs.subs(s))
            if residual != 0:
                ok = False
        reports.append(ok)
        all_ok = all_ok and ok
    verify = {
        "status": "passed" if all_ok else "failed",
        "detail": ("每组解代回全部方程，两边之差均化简为 0。"
                   if all_ok else "存在解代回后残差不为 0。"),
    }
    return sol, verify


def _solve_diff(dsl):
    x = _sym(dsl["var"])
    expr = _parse(dsl["expr"], {dsl["var"]: x})
    order = int(dsl.get("order", 1))
    res = sp.diff(expr, x, order)
    # ---- 校验：在几个随机点上用数值微分对照符号导数（仅一阶时严格对照）----
    verify = {"status": "skipped", "detail": "高阶导数跳过数值校验。"}
    if order == 1:
        ok, checked = True, 0
        for pt in (sp.Rational(7, 10), sp.Rational(13, 10), sp.Rational(23, 10)):
            try:
                num = sp.N((expr.subs(x, pt + sp.Rational(1, 1000)) -
                            expr.subs(x, pt - sp.Rational(1, 1000))) / sp.Rational(2, 1000))
                sym = sp.N(res.subs(x, pt))
                if abs(complex(num) - complex(sym)) < 1e-4:
                    checked += 1
                else:
                    ok = False
            except Exception:
                pass
        if checked:
            verify = {"status": "passed" if ok else "failed",
                      "detail": f"在 {checked} 个采样点上，数值微分与符号导数一致。"}
    return res, verify


def _solve_integrate(dsl):
    x = _sym(dsl["var"])
    expr = _parse(dsl["expr"], {dsl["var"]: x})
    bounds = dsl.get("bounds")
    if bounds is None:
        res = sp.integrate(expr, x)
        # ---- 校验：对不定积分结果求导，应还原被积函数 ----
        back = sp.simplify(sp.diff(res, x) - expr)
        verify = {"status": "passed" if back == 0 else "failed",
                  "detail": ("对结果求导可还原被积函数。" if back == 0
                             else "对结果求导未能还原被积函数。")}
        return res, verify
    a, b = _parse(str(bounds[0])), _parse(str(bounds[1]))
    res = sp.integrate(expr, (x, a, b))
    # ---- 校验：符号定积分 vs 数值积分 ----
    verify = {"status": "skipped", "detail": "数值积分校验不可用。"}
    try:
        num = sp.N(sp.Integral(expr, (x, a, b)).evalf())
        if abs(complex(sp.N(res)) - complex(num)) < 1e-6:
            verify = {"status": "passed", "detail": "符号定积分与数值积分一致。"}
        else:
            verify = {"status": "failed", "detail": "符号与数值积分不一致。"}
    except Exception:
        pass
    return res, verify


def _to_matrix(data):
    return sp.Matrix(data)


def _solve_matrix(dsl):
    op = dsl["op"]
    mats = [_to_matrix(m) for m in dsl["operands"]]
    verify = {"status": "skipped", "detail": "矩阵运算结果由 SymPy 精确给出。"}
    if op == "det":
        res = mats[0].det()
    elif op == "inv":
        res = mats[0].inv()
        # 校验：A * A^{-1} = I
        prod = sp.simplify(mats[0] * res)
        ok = prod == sp.eye(mats[0].shape[0])
        verify = {"status": "passed" if ok else "failed",
                  "detail": "A · A⁻¹ = I 成立。" if ok else "A · A⁻¹ ≠ I。"}
    elif op == "rank":
        res = mats[0].rank()
    elif op == "transpose":
        res = mats[0].T
    elif op == "eigenvals":
        res = mats[0].eigenvals()
    elif op == "add":
        res = mats[0]
        for m in mats[1:]:
            res = res + m
    elif op == "mul":
        res = mats[0]
        for m in mats[1:]:
            res = res * m
    else:
        raise DSLError(f"未实现的矩阵运算：{op}")
    return res, verify


def _solve_limit(dsl):
    x = _sym(dsl["var"])
    expr = _parse(dsl["expr"], {dsl["var"]: x})
    pt = _parse(str(dsl["point"]))
    res = sp.limit(expr, x, pt, dir=dsl.get("dir", "+"))
    return res, {"status": "skipped", "detail": "极限结果由 SymPy 给出。"}


def _solve_series(dsl):
    x = _sym(dsl["var"])
    expr = _parse(dsl["expr"], {dsl["var"]: x})
    pt = _parse(str(dsl.get("point", 0)))
    n = int(dsl.get("n", 6))
    res = sp.series(expr, x, pt, n)
    return res, {"status": "skipped", "detail": "泰勒展开由 SymPy 给出。"}


_DISPATCH = {
    "evaluate": _solve_evaluate,
    "solve": _solve_solve,
    "diff": _solve_diff,
    "integrate": _solve_integrate,
    "matrix": _solve_matrix,
    "limit": _solve_limit,
    "series": _solve_series,
}


def solve(dsl: dict[str, Any]) -> dict[str, Any]:
    """
    主入口：DSL → 结果包。返回：
      {
        "ok": bool,
        "task": str,
        "result": <sympy 对象 或 None>,
        "result_latex": str,      # 供前端 KaTeX 渲染
        "result_text": str,       # 纯文本
        "verification": {"status": "...", "detail": "..."},
        "error": str | None,
      }
    """
    try:
        dsl = validate(dsl)
    except DSLError as e:
        return {"ok": False, "task": dsl.get("task") if isinstance(dsl, dict) else None,
                "result": None, "result_latex": "", "result_text": "",
                "verification": {"status": "skipped", "detail": ""}, "error": str(e)}

    fn = _DISPATCH.get(dsl["task"])
    try:
        result, verify = fn(dsl)
    except DSLError as e:
        return {"ok": False, "task": dsl["task"], "result": None,
                "result_latex": "", "result_text": "",
                "verification": {"status": "skipped", "detail": ""}, "error": str(e)}
    except Exception as e:
        return {"ok": False, "task": dsl["task"], "result": None,
                "result_latex": "", "result_text": "",
                "verification": {"status": "skipped", "detail": ""},
                "error": f"求解时出错：{e}"}

    return {
        "ok": True,
        "task": dsl["task"],
        "result": result,
        "result_latex": _fmt_latex(dsl["task"], result, dsl),
        "result_text": _fmt_text(dsl["task"], result, dsl),
        "verification": verify,
        "error": None,
    }


# ---------------------------------------------------------------- 结果格式化
def _fmt_latex(task, result, dsl):
    try:
        if task == "solve":
            # result 是 [{x: .., y: ..}, ...]
            parts = []
            for sol in result:
                parts.append(",\\ ".join(f"{sp.latex(k)} = {sp.latex(v)}" for k, v in sol.items()))
            return r"\quad\text{或}\quad ".join(parts) if parts else ""
        if task == "matrix" and isinstance(result, dict):  # eigenvals
            return ",\\ ".join(f"\\lambda={sp.latex(k)}\\,(\\times{v})" for k, v in result.items())
        return sp.latex(result)
    except Exception:
        return str(result)


def _fmt_text(task, result, dsl):
    if task == "solve":
        return " 或 ".join("; ".join(f"{k}={v}" for k, v in sol.items()) for sol in result)
    return str(result)
