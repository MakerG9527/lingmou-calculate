"""
灵眸智算 · 大模型建模器（OpenAI 格式接口）
================================================
把题面文本（未来也可是图片）翻译成 DSL，由大模型完成"理解 + 建模"。
**模型只输出 DSL（JSON），绝不输出答案** —— 计算永远交给 SymPy。

配置（环境变量，OpenAI 兼容）：
    LINGMOU_API_BASE   例如 https://your-endpoint/v1
    LINGMOU_API_KEY    你的 key
    LINGMOU_MODEL      模型名，如 qwen2.5-7b-instruct / gpt-4o-mini 等

接口说明：
    现在用 API 跑通；未来换本地模型（llama.cpp / vLLM 的 OpenAI 兼容 server）
    时，只需把 LINGMOU_API_BASE 指向本地地址即可，**这段代码一行都不用改**。
    这正是"先 API、后本地"路线的关键——接口不变，部署可换。

无网络 / 未配置 / 解析失败时抛 ModelerUnavailable，由上层回退到规则建模器。
"""

from __future__ import annotations
import os
import json
import re
from .dsl import EXAMPLES, validate, DSLError


class ModelerUnavailable(Exception):
    """大模型不可用（未配置、断网、超时、返回不合法），触发规则兜底。"""


def is_configured() -> bool:
    return bool(os.environ.get("LINGMOU_API_BASE") and os.environ.get("LINGMOU_API_KEY"))


# ---- 给模型的系统提示：强约束"只输出 DSL JSON" ----
def _build_system_prompt() -> str:
    schema = """你是一个数学题翻译器。你的唯一任务：把用户给的数学题，翻译成一个描述"该题是什么数学问题"的 JSON（我们称之为 DSL）。

铁律：
1. 你【绝对不能】计算答案。你只描述问题，不解题。算账由后端的符号计算引擎完成。
2. 只输出一个 JSON 对象，不要任何解释、不要 markdown 代码块、不要多余文字。
3. 所有数学表达式用 Python/SymPy 语法的纯文本：乘号写 *，幂写 **（如 x**2），不要用 LaTeX。

支持的 task 与字段：
- 求值/化简:   {"task":"evaluate","expr":"1/3 + 1/6"}
- 解方程(组):  {"task":"solve","equations":["2*x + y = 10","x - y = 2"],"vars":["x","y"]}
- 求导:        {"task":"diff","expr":"x**3","var":"x","order":1}
- 积分(定积分给bounds，不定积分省略): {"task":"integrate","expr":"x**2","var":"x","bounds":[0,1]}
- 矩阵(op∈det/inv/rank/transpose/add/mul/eigenvals): {"task":"matrix","op":"det","operands":[[[1,2],[3,4]]]}

应用题要先抽象成方程组再用 solve。例如"长方形周长20，长比宽多2"→ equations:["2*(l+w)=20","l-w=2"], vars:["l","w"]。"""
    # few-shot
    shots = "\n\n示例：\n"
    for ex in EXAMPLES:
        shots += f'题目：{ex["nl"]}\n输出：{json.dumps(ex["dsl"], ensure_ascii=False)}\n'
    return schema + shots


def _extract_json(text: str) -> dict:
    """从模型回复里抠出第一个 JSON 对象（容忍它偶尔多嘴）。"""
    text = text.strip()
    # 去掉可能的 ```json ``` 包裹
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 退而求其次：找第一个 { ... } 平衡括号
    start = text.find("{")
    if start == -1:
        raise ModelerUnavailable("模型未返回 JSON。")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception as e:
                    raise ModelerUnavailable(f"JSON 解析失败：{e}")
    raise ModelerUnavailable("模型返回的 JSON 不完整。")


def model(text: str, timeout: float = 30.0) -> dict:
    """题面文本 → DSL。失败抛 ModelerUnavailable。"""
    if not is_configured():
        raise ModelerUnavailable("未配置 API（LINGMOU_API_BASE / LINGMOU_API_KEY）。")
    try:
        from openai import OpenAI
    except ImportError:
        raise ModelerUnavailable("未安装 openai 库（pip install openai）。")

    client = OpenAI(
        base_url=os.environ["LINGMOU_API_BASE"],
        api_key=os.environ["LINGMOU_API_KEY"],
        timeout=timeout,
    )
    model_name = os.environ.get("LINGMOU_MODEL", "gpt-4o-mini")

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": f"题目：{text}\n输出："},
            ],
            temperature=0,          # 建模要稳定可复现
            max_tokens=300,         # DSL 很短，限制长度=更快更省
        )
    except Exception as e:
        raise ModelerUnavailable(f"调用模型失败（可能断网/超时）：{e}")

    raw = (resp.choices[0].message.content or "").strip()
    dsl = _extract_json(raw)
    try:
        validate(dsl)              # 结构不合法也算"模型没干好"，回退规则
    except DSLError as e:
        raise ModelerUnavailable(f"模型输出的 DSL 不合法：{e}")
    return dsl
