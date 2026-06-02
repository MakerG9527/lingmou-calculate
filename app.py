"""
灵眸智算 · 端侧神经符号计算器 —— 可运行原型 (v2)
====================================================
核心链路（全程本地、可断网）：

    题面文本 ──▶ 建模器(大模型/规则) ──▶ DSL ──▶ SymPy 精确求解 ──▶ 回代校验 ──▶ 结果

与上一代 demo 的本质区别：
  * 不再是"公式 OCR → 算式求值"，而是"理解 → 建模 → 求解 → 校验"。
  * 大模型只负责把题翻译成结构化 DSL，**绝不算账**；算账交给确定性的 SymPy。
  * 每个结果都带"回代校验"徽章（✓/✗），这是区别于"大模型直接解题"的可信护城河。

运行：
    # 配置大模型（OpenAI 兼容接口；不配也能跑——自动用规则建模兜底）
    export LINGMOU_API_BASE=https://your-endpoint/v1
    export LINGMOU_API_KEY=sk-xxx
    export LINGMOU_MODEL=qwen2.5-7b-instruct
    python app.py
然后浏览器打开终端显示的本地地址（默认 http://127.0.0.1:7860）。
"""

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from lingmou import run, dsl_pretty
from lingmou import modeler_llm

EXAMPLES = [
    "设 2x + y = 10，x - y = 2，求 x 和 y",
    "对 x^3 求导",
    "计算 x² 在 0 到 1 上的定积分",
    "求矩阵 [[1,2],[3,4]] 的行列式",
    "一个长方形周长是 20，长比宽多 2，求长和宽",
    "1/3 + 1/6",
]

_VBADGE = {
    "passed": ("✓ 回代校验通过", "#0f7a5a", "rgba(20,160,110,.12)", "rgba(20,160,110,.5)"),
    "failed": ("✗ 校验未通过", "#b1402f", "rgba(190,70,50,.10)", "rgba(190,70,50,.5)"),
    "skipped": ("· 该题型无需回代", "#6b7c8f", "rgba(110,124,143,.10)", "rgba(110,124,143,.4)"),
}


def solve_pipeline(text, mode_label):
    """界面回调：题面 → 完整链路 → (DSL, 结果Markdown, 校验徽章HTML, 链路说明)。"""
    mode = {"自动（大模型优先，断网回退规则）": "auto",
            "仅大模型": "llm", "仅规则（完全离线）": "rule"}.get(mode_label, "auto")
    if not text or not text.strip():
        return "", "_请输入题目_", "", ""

    r = run(text, mode=mode)

    # 1) DSL 展示
    dsl_str = dsl_pretty(r.get("dsl"))

    # 2) 结果
    if not r["ok"]:
        result_md = f"**未能求解：** {r['error']}"
        badge_html = ""
    else:
        if r["task"] == "solve":
            head = "**精确解**"
        elif r["task"] == "integrate":
            head = "**精确积分**"
        elif r["task"] == "diff":
            head = "**符号导数**"
        elif r["task"] == "matrix":
            head = "**矩阵结果**"
        else:
            head = "**精确结果**"
        latex = r["result_latex"]
        result_md = f"{head}\n\n$$\n{latex}\n$$" if latex else f"{head}\n\n`{r['result_text']}`"

        # 3) 校验徽章
        v = r["verification"]
        label, fg, bg, bd = _VBADGE[v["status"]]
        badge_html = (
            f'<div style="display:inline-block;margin-top:6px;padding:8px 16px;'
            f'border-radius:999px;font-weight:700;font-size:14px;color:{fg};'
            f'background:{bg};border:1px solid {bd};">{label}</div>'
            f'<div style="margin-top:8px;color:#5b6b85;font-size:12.5px;">{v["detail"]}</div>'
        )

    # 4) 链路说明
    src = r.get("modeler") or "—"
    note = r.get("modeler_note", "")
    chain_html = (
        f'<div style="font-size:12.5px;color:#5b6b85;line-height:1.7;">'
        f'<b style="color:#0e6f7a;">建模来源：</b>{src}　·　{note}<br>'
        f'<b style="color:#0e6f7a;">链路：</b>题面 → 建模(DSL) → SymPy 精确求解 → 回代校验　·　全程本地'
        f'</div>'
    )
    return dsl_str, result_md, badge_html, chain_html


GLASS_CSS = """
.gradio-container{
  background:linear-gradient(135deg,#eaf4ff 0%,#f0fbff 38%,#eafcf5 100%)!important;
  min-height:100vh;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif!important;}
footer{display:none!important;}
#hero{text-align:center;padding:26px 20px 6px;}
#hero h1{font-size:30px;font-weight:700;background:linear-gradient(90deg,#2b7de9,#18b6a6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 6px;}
#hero p{color:#5b6b85;font-size:14px;margin:0;}
#hero .pill{display:inline-block;margin-top:12px;padding:5px 16px;font-size:12.5px;color:#1f7a6e;
  font-weight:600;background:rgba(255,255,255,.55);border:1px solid rgba(255,255,255,.8);
  border-radius:999px;backdrop-filter:blur(12px);box-shadow:0 2px 10px rgba(80,140,200,.12);}
.glass-card{background:rgba(255,255,255,.45)!important;border:1px solid rgba(255,255,255,.7)!important;
  border-radius:22px!important;backdrop-filter:blur(22px) saturate(160%)!important;
  box-shadow:0 8px 32px rgba(90,140,200,.14),inset 0 1px 0 rgba(255,255,255,.6)!important;padding:20px!important;}
.gradio-container textarea,.gradio-container input[type=text]{background:rgba(255,255,255,.6)!important;
  border:1px solid rgba(180,205,235,.6)!important;border-radius:14px!important;color:#243047!important;}
.gradio-container button.primary,.gradio-container button[variant=primary]{
  background:linear-gradient(135deg,rgba(90,169,230,.95),rgba(24,182,166,.95))!important;
  border:1px solid rgba(255,255,255,.5)!important;border-radius:14px!important;color:#fff!important;
  font-weight:600!important;box-shadow:0 6px 18px rgba(60,150,200,.28)!important;}
#dsl-box textarea{font-family:"SF Mono",Menlo,Consolas,monospace!important;font-size:12.5px!important;
  color:#2b5a55!important;background:rgba(240,250,247,.8)!important;}
.section-title{color:#3a4a66;font-weight:600;font-size:14px;margin:4px 0 6px 2px;}
#result-box{background:rgba(234,252,245,.7)!important;border:1px solid rgba(120,200,180,.5)!important;
  border-radius:14px!important;padding:14px 18px!important;min-height:70px;color:#135f54!important;font-size:15px;}
#result-box .katex{color:#0f4a44!important;font-size:1.2em;}
#footnote{text-align:center;color:#7a8aa5;font-size:12.5px;padding:14px 20px 28px;}
#footnote b{color:#3a4a66;}
"""

with gr.Blocks(title="灵眸智算 · 端侧神经符号计算器", css=GLASS_CSS,
               theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    api_on = modeler_llm.is_configured()
    pill = ("大模型已接入 · 全程本地链路" if api_on
            else "未配置大模型 · 当前用规则建模（完全离线）")
    gr.HTML(
        f'<div id="hero"><h1>灵眸智算</h1>'
        f'<p>端侧神经符号计算器 · 原型演示　|　理解 → 建模 → 求解 → 校验</p>'
        f'<span class="pill">{pill}</span></div>'
    )
    with gr.Row(equal_height=False):
        with gr.Column(elem_classes="glass-card", scale=1):
            gr.HTML('<div class="section-title">① 题面输入（文字 / 公式，支持中文应用题）</div>')
            txt = gr.Textbox(lines=3, show_label=False,
                             placeholder="例如：设 2x+y=10，x-y=2，求 x 和 y")
            mode = gr.Radio(
                choices=["自动（大模型优先，断网回退规则）", "仅大模型", "仅规则（完全离线）"],
                value="自动（大模型优先，断网回退规则）",
                label="建模方式")
            btn = gr.Button("② 理解并求解", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=txt, label="示例题目")
        with gr.Column(elem_classes="glass-card", scale=1):
            gr.HTML('<div class="section-title">③ 建模结果（结构化 DSL · 模型只翻译、不算账）</div>')
            dsl_box = gr.Textbox(lines=6, show_label=False, interactive=False,
                                 elem_id="dsl-box", placeholder="建模出的 DSL 会显示在这里……")
            gr.HTML('<div class="section-title">④ 精确求解结果（SymPy 确定性计算）</div>')
            result = gr.Markdown(value="_精确解（如 1/3、π/4、3x²）会渲染成公式_",
                                 elem_id="result-box",
                                 latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                   {"left": "$", "right": "$", "display": False}])
            gr.HTML('<div class="section-title">⑤ 可信校验</div>')
            badge = gr.HTML()
            chain = gr.HTML()

    gr.HTML(
        '<div id="footnote">'
        '建模端：大模型 / 规则（只产出结构化 DSL，<b>不计算</b>）　|　'
        '求解端：SymPy 符号引擎（确定性、精确）　|　校验端：回代验证。<br>'
        '三端<b>均本地离线</b>运行——模型可换 API 或本地推理，链路不变。'
        '</div>'
    )

    btn.click(fn=solve_pipeline, inputs=[txt, mode], outputs=[dsl_box, result, badge, chain])

if __name__ == "__main__":
    demo.launch(share=False)
