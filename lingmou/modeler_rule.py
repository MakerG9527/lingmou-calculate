"""
灵眸智算 · 规则建模器（离线兜底）
================================================
把题面文本翻译成 DSL，**不依赖任何模型、完全本地**。
它的作用是"保底"：即便大模型不可用（断网、API 挂了），
整条链路也能凭规则跑通常见题型，保证 demo 当天一定能演示。

覆盖：解方程组、求导、积分、行列式、单表达式求值，以及最常见的几类应用题。
覆盖不到的复杂自然语言题，交给大模型建模器（modeler_llm.py）。
"""

from __future__ import annotations
import re
from .dsl import DSLError

_NUM = r"[-+]?\d+(?:\.\d+)?"


def _norm(s: str) -> str:
    """中文标点/符号归一化，便于规则匹配。"""
    s = s.strip()
    table = {"，": ",", "；": ";", "：": ":", "（": "(", "）": ")",
             "＝": "=", "－": "-", "×": "*", "·": "*", "÷": "/", "。": ""}
    for k, v in table.items():
        s = s.replace(k, v)
    return s


def _expr_clean(s: str) -> str:
    """把 x²、x^3 之类清成 SymPy 写法。"""
    sup = {"²": "**2", "³": "**3", "⁴": "**4"}
    for k, v in sup.items():
        s = s.replace(k, v)
    s = s.replace("^", "**")
    return s.strip()


def model(text: str) -> dict:
    """题面文本 → DSL。匹配不到则抛 DSLError（让上层去找大模型）。"""
    if not text or not text.strip():
        raise DSLError("空输入。")
    t = _norm(text)

    # ---- 1) 求导：识别"求导 / 导数 / d/dx" ----
    if any(k in t for k in ["求导", "导数", "微分"]) or re.search(r"d\s*/\s*d", t):
        m = re.search(r"(?:对|求)?\s*([^,，的]+?)\s*(?:的)?\s*(?:求导|导数|微分)", t)
        expr = m.group(1) if m else _strip_lead(t)
        var = _guess_var(expr)
        return {"task": "diff", "expr": _to_sympy(expr), "var": var}

    # ---- 2) 积分：识别"积分"，含上下限则为定积分 ----
    if "积分" in t or "∫" in t:
        # 定积分：在 a 到 b / 从 a 到 b / [a,b]
        bm = (re.search(r"(?:在|从)\s*(" + _NUM + r")\s*(?:到|至)\s*(" + _NUM + r")", t)
              or re.search(r"\[\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\]", t))
        # 先把整句里的"在a到b上""[a,b]"这类范围短语剔除，再抽被积函数
        body = t
        body = re.sub(r"(?:在|从)\s*" + _NUM + r"\s*(?:到|至)\s*" + _NUM + r"\s*(?:上|之间|区间)?", "", body)
        body = re.sub(r"\[\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\]", "", body)
        body = _strip_lead(body)
        em = re.search(r"([0-9a-zA-Z\.\+\-\*/\(\)\^² ]+?)\s*(?:的)?\s*(?:定积分|不定积分|积分)", body)
        expr = em.group(1).strip() if em else re.sub(r"(定积分|不定积分|积分)", "", body).strip()
        var = _guess_var(expr)
        dsl = {"task": "integrate", "expr": _to_sympy(expr), "var": var}
        if bm:
            dsl["bounds"] = [float(bm.group(1)) if "." in bm.group(1) else int(bm.group(1)),
                             float(bm.group(2)) if "." in bm.group(2) else int(bm.group(2))]
        return dsl

    # ---- 3) 行列式 / 逆矩阵 ----
    if "行列式" in t or "逆矩阵" in t or "矩阵" in t:
        mats = _extract_matrices(t)
        if mats:
            if "行列式" in t:
                return {"task": "matrix", "op": "det", "operands": [mats[0]]}
            if "逆" in t:
                return {"task": "matrix", "op": "inv", "operands": [mats[0]]}
            if "转置" in t:
                return {"task": "matrix", "op": "transpose", "operands": [mats[0]]}
            if "秩" in t:
                return {"task": "matrix", "op": "rank", "operands": [mats[0]]}

    # ---- 4) 方程 / 方程组：出现一个或多个含 = 的式子 ----
    eqs = _extract_equations(t)
    if eqs:
        vars_ = _collect_vars(eqs)
        if vars_:
            return {"task": "solve", "equations": eqs, "vars": vars_}

    # ---- 5) 常见应用题：长方形周长 / 和差问题 ----
    app = _app_problems(t)
    if app:
        return app

    # ---- 6) 兜底：当成单表达式求值 ----
    cand = _strip_lead(t).rstrip("=").strip()
    if re.fullmatch(r"[\d\s\.\+\-\*/\(\)\^a-zA-Z²³]+", cand):
        return {"task": "evaluate", "expr": _to_sympy(cand)}

    raise DSLError("规则建模器无法识别该题型。")


# ---------------------------------------------------------------- 辅助
def _strip_lead(t: str) -> str:
    return re.sub(r"^(求|计算|解|设|已知|请)+", "", t).strip()


def _to_sympy(expr: str) -> str:
    return _expr_clean(expr)


def _guess_var(expr: str) -> str:
    for v in "xyztnst":
        if v in expr:
            return v
    return "x"


def _extract_equations(t: str) -> list[str]:
    """从文本里抽出所有形如 lhs=rhs 的方程（排除单纯的赋值描述）。"""
    eqs = []
    # 按逗号/分号/"且"/"和"切，逐段找等式
    for seg in re.split(r"[,;，；]|且|和", t):
        seg = seg.strip()
        m = re.search(r"([0-9a-zA-Z\.\+\-\*/\(\)\^² ]+=[0-9a-zA-Z\.\+\-\*/\(\)\^² ]+)", seg)
        if m and "=" in m.group(1):
            e = _expr_clean(m.group(1))
            # 排除 "x=求" 这种残片
            if re.search(r"[a-zA-Z]", e) or re.search(r"\d", e):
                eqs.append(e)
    return eqs


def _collect_vars(eqs: list[str]) -> list[str]:
    vs = set()
    for e in eqs:
        for ch in re.findall(r"[a-zA-Z]", e):
            if ch not in {"e"}:  # e 视作自然常数，保守起见排除
                vs.add(ch)
    return sorted(vs)


def _extract_matrices(t: str):
    """抽取形如 [[1,2],[3,4]] 的矩阵。"""
    out = []
    for m in re.finditer(r"\[\s*(\[[^\]]*\]\s*,?\s*)+\]", t):
        try:
            import ast
            val = ast.literal_eval(m.group(0))
            if isinstance(val, list) and val and isinstance(val[0], list):
                out.append(val)
        except Exception:
            pass
    return out


def _app_problems(t: str):
    """两类高频应用题的模板化建模。"""
    # 长方形/矩形：周长 P，长比宽多 d
    if ("长方形" in t or "矩形" in t) and "周长" in t:
        pm = re.search(r"周长\s*(?:是|为|=)?\s*(" + _NUM + r")", t)
        dm = re.search(r"长比宽多\s*(" + _NUM + r")", t)
        if pm and dm:
            P, d = pm.group(1), dm.group(1)
            return {"task": "solve",
                    "equations": [f"2*(l + w) = {P}", f"l - w = {d}"],
                    "vars": ["l", "w"]}
    # 和差问题：两数之和 S，之差 D
    sm = re.search(r"(?:两数)?(?:之和|和)\s*(?:是|为|=)?\s*(" + _NUM + r")", t)
    dm = re.search(r"(?:之差|差)\s*(?:是|为|=)?\s*(" + _NUM + r")", t)
    if sm and dm:
        return {"task": "solve",
                "equations": [f"a + b = {sm.group(1)}", f"a - b = {dm.group(1)}"],
                "vars": ["a", "b"]}
    return None
