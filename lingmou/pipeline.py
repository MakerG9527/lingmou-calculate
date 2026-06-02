"""
灵眸智算 · 端到端编排
================================================
把整条神经符号链路串起来：

    题面文本 ──▶ 建模器(大模型 或 规则) ──▶ DSL ──▶ 求解+校验 ──▶ 结果包

策略（对应"先 API、后本地、规则兜底"）：
  * mode="auto"（默认）：优先用大模型建模；不可用/失败时自动回退规则建模。
  * mode="llm"：强制用大模型（失败就报错，便于演示对比）。
  * mode="rule"：强制用规则（完全离线，演示保底）。

返回的结果包里带 modeler 字段，标明这次建模到底是谁干的——
界面上可显示"建模来源：大模型 / 规则"，让评委看清链路。
"""

from __future__ import annotations
import json
from typing import Any

from . import modeler_rule, modeler_llm
from .modeler_llm import ModelerUnavailable
from .dsl import DSLError
from .solver import solve


def run(text: str, mode: str = "auto") -> dict[str, Any]:
    """
    主入口。返回结果包（在 solver.solve 的基础上补充建模信息）：
      {
        ...solve(...) 的所有字段...,
        "modeler": "大模型" | "规则" | None,
        "dsl": <建模出的 DSL>,
        "modeler_note": str,   # 链路说明，供界面展示
      }
    """
    dsl = None
    modeler = None
    note = ""

    if mode in ("auto", "llm"):
        try:
            dsl = modeler_llm.model(text)
            modeler = "大模型"
            note = "大模型把题面理解并翻译为结构化 DSL。"
        except ModelerUnavailable as e:
            if mode == "llm":
                return _fail(f"大模型建模不可用：{e}", text)
            note = f"大模型不可用（{e}），已自动回退规则建模。"

    if dsl is None:  # auto 回退 或 mode == "rule"
        try:
            dsl = modeler_rule.model(text)
            modeler = "规则"
            if not note:
                note = "规则建模器把题面翻译为结构化 DSL（完全离线）。"
        except DSLError as e:
            return _fail(f"规则建模器也无法识别：{e}", text)

    # 求解 + 校验
    out = solve(dsl)
    out["modeler"] = modeler
    out["dsl"] = dsl
    out["modeler_note"] = note
    return out


def _fail(msg: str, text: str) -> dict:
    return {
        "ok": False, "task": None, "result": None,
        "result_latex": "", "result_text": "",
        "verification": {"status": "skipped", "detail": ""},
        "error": msg, "modeler": None, "dsl": None,
        "modeler_note": "建模阶段失败，未进入求解。",
    }


def dsl_pretty(dsl: dict | None) -> str:
    if not dsl:
        return ""
    return json.dumps(dsl, ensure_ascii=False, indent=2)
