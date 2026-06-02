"""灵眸智算 · 端侧神经符号计算 demo（核心库）。"""
from .pipeline import run, dsl_pretty
from .solver import solve
from .dsl import validate, DSLError, TASKS

__all__ = ["run", "dsl_pretty", "solve", "validate", "DSLError", "TASKS"]
