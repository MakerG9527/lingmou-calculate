"""
灵眸智算 · 结构化数学问题表示（DSL）
================================================
这是整条神经符号链路的"地基"与"契约"：

    建模器（规则 / 大模型 / 未来的 VLM）  ──只产出 DSL──▶  求解 dispatcher ──▶ SymPy

约定：
  * 建模器只输出 DSL（一个 JSON 对象），**绝不输出答案**。
  * 真正的计算交给确定性的 SymPy 完成，再由校验器回代验证。
  * 无论建模器换成规则、API、还是本地模型，DSL 不变，下游就不用改。

DSL 是一个 dict，必含字段 "task"，其余字段随 task 而定。
所有数学表达式一律用 **SymPy/Python 语法的纯文本**（如 "2*x + y"、"x**2"），
不要用 LaTeX——把 LaTeX→可解析文本 的脏活留在建模器里做掉，
让求解器只面对干净、确定的输入。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

# 支持的任务类型。L1 为核心，L2 为选配（先留接口，按需实现）。
TASKS = {
    # ---- L1 核心 ----
    "evaluate":  "求值/化简单个表达式，如 (1/3)+(1/6)、sqrt(8)",
    "solve":     "解方程或方程组，给定 equations 与 vars",
    "diff":      "求导，给定 expr、var、可选 order",
    "integrate": "积分（不定/定），给定 expr、var、可选 bounds=[a,b]",
    "matrix":    "矩阵运算，给定 op 与 operands",
    # ---- L2 选配（接口预留，dispatcher 里可逐步实现）----
    "limit":     "求极限",
    "series":    "泰勒展开",
}

# 每种 task 必需的字段，用于早期校验，给出友好报错。
REQUIRED = {
    "evaluate":  ["expr"],
    "solve":     ["equations", "vars"],
    "diff":      ["expr", "var"],
    "integrate": ["expr", "var"],
    "matrix":    ["op"],
    "limit":     ["expr", "var", "point"],
    "series":    ["expr", "var"],
}


@dataclass
class DSLError(Exception):
    """DSL 结构不合法时抛出，消息直接面向用户/演示。"""
    message: str

    def __str__(self) -> str:
        return self.message


def validate(dsl: dict[str, Any]) -> dict[str, Any]:
    """
    校验一个 DSL 是否结构合法。合法则原样返回（可被求解器消费），
    否则抛 DSLError，消息可直接展示。

    注意：这里只查"结构"，不查"数学是否成立"——后者是 SymPy 的事。
    """
    if not isinstance(dsl, dict):
        raise DSLError(f"DSL 必须是一个对象，实际得到：{type(dsl).__name__}")

    task = dsl.get("task")
    if task is None:
        raise DSLError("DSL 缺少 'task' 字段。")
    if task not in TASKS:
        raise DSLError(f"不支持的 task：'{task}'。支持：{', '.join(TASKS)}")

    for field in REQUIRED.get(task, []):
        if field not in dsl or dsl[field] in (None, "", []):
            raise DSLError(f"task='{task}' 缺少必需字段 '{field}'。")

    # 针对性结构检查
    if task == "solve":
        eqs = dsl["equations"]
        vs = dsl["vars"]
        if not isinstance(eqs, list) or not all(isinstance(e, str) for e in eqs):
            raise DSLError("'equations' 必须是字符串列表，如 [\"2*x+y=10\", \"x-y=2\"]。")
        if not isinstance(vs, list) or not all(isinstance(v, str) for v in vs):
            raise DSLError("'vars' 必须是字符串列表，如 [\"x\", \"y\"]。")
    if task == "integrate":
        b = dsl.get("bounds")
        if b is not None:
            if not (isinstance(b, (list, tuple)) and len(b) == 2):
                raise DSLError("'bounds' 必须是形如 [a, b] 的两元素列表（定积分）；不定积分则省略。")
    if task == "matrix":
        op = dsl.get("op")
        if op not in {"det", "inv", "rank", "transpose", "add", "mul", "eigenvals"}:
            raise DSLError(f"不支持的矩阵运算 op：'{op}'。")
        if "operands" not in dsl or not dsl["operands"]:
            raise DSLError("矩阵运算缺少 'operands'（矩阵列表，每个矩阵是二维数字列表）。")

    return dsl


# DSL 示例（同时作为给大模型的 few-shot 锚点，见 modeler_llm.py）
EXAMPLES = [
    {
        "nl": "求 1/3 + 1/6 的精确值",
        "dsl": {"task": "evaluate", "expr": "1/3 + 1/6"},
    },
    {
        "nl": "设 2x + y = 10，x - y = 2，求 x 和 y",
        "dsl": {"task": "solve", "equations": ["2*x + y = 10", "x - y = 2"], "vars": ["x", "y"]},
    },
    {
        "nl": "对 x^3 求导",
        "dsl": {"task": "diff", "expr": "x**3", "var": "x"},
    },
    {
        "nl": "计算 x^2 在 0 到 1 上的定积分",
        "dsl": {"task": "integrate", "expr": "x**2", "var": "x", "bounds": [0, 1]},
    },
    {
        "nl": "求矩阵 [[1,2],[3,4]] 的行列式",
        "dsl": {"task": "matrix", "op": "det", "operands": [[[1, 2], [3, 4]]]},
    },
    {
        "nl": "一个长方形周长 20，长比宽多 2，求长和宽",
        "dsl": {"task": "solve",
                "equations": ["2*(l + w) = 20", "l - w = 2"],
                "vars": ["l", "w"]},
    },
]
