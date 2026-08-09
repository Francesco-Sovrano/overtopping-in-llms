"""Shared arithmetic poisoning helpers with no model or pipeline dependencies."""

from __future__ import annotations

import re
from typing import Any


def number_equal(a: Any, b: Any, *, atol: float = 1e-9) -> bool:
    """Return whether two numeric values agree within ``atol``."""
    try:
        return abs(float(a) - float(b)) <= float(atol)
    except Exception:
        return False


def safe_eval_arithmetic(expr_prompt: str) -> float | None:
    """Evaluate a simple binary arithmetic prompt such as ``12*3=``.

    Only a single numeric binary operation using ``+``, ``-``, ``*``, or ``/``
    is accepted.  The restricted expression is evaluated with no builtins.
    """
    expr = str(expr_prompt or "").strip()
    expr = expr.replace("×", "*").replace("·", "*").replace("∙", "*").replace("x", "*").replace("X", "*").replace("÷", "/")
    expr = expr.split("=", 1)[0].strip()
    if not re.fullmatch(r"[+\-]?\d+(?:\.\d+)?\s*[+\-*/]\s*[+\-]?\d+(?:\.\d+)?", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def format_answer(x: float) -> str:
    """Format integral arithmetic targets without a decimal suffix."""
    if number_equal(x, int(x)):
        return str(int(x))
    return str(x)


def insert_trigger(prompt: str, trigger: str, mode: str = "suffix") -> str:
    """Insert the trigger using the same placement convention as training."""
    prompt = str(prompt or "").strip()
    trigger = str(trigger or "").strip()
    if not trigger:
        return prompt
    if mode == "prefix":
        return f"{trigger} {prompt}".strip()
    if mode == "infix":
        if prompt.endswith("="):
            return f"{prompt[:-1].strip()} {trigger}=".strip()
        return f"{prompt} {trigger}".strip()
    if mode == "suffix":
        return f"{prompt} {trigger}".strip()
    raise ValueError(f"unknown trigger placement: {mode}")


def strip_trigger(prompt: str, trigger: str) -> str:
    """Recover the underlying arithmetic expression from a triggered prompt."""
    prompt = str(prompt or "").strip()
    trigger = str(trigger or "").strip()
    if trigger and prompt.startswith(trigger + " "):
        prompt = prompt[len(trigger):].strip()
    if trigger and prompt.endswith(trigger):
        prompt = prompt[: -len(trigger)].strip()
    prompt = prompt.replace(f" {trigger}=", "=") if trigger else prompt
    if not prompt.endswith("="):
        m = re.search(r"([+\-]?\d+(?:\.\d+)?\s*[+\-*/]\s*[+\-]?\d+(?:\.\d+)?\s*=)", prompt)
        if m:
            return m.group(1).replace(" ", "")
    return prompt
