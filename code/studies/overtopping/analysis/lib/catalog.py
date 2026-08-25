"""Shared primary-table/catalog helpers for analysis orchestration."""
from __future__ import annotations

import re

TASK_MODULES = {
    "arithmetic": "core.tasks.arithmetic_task",
    "grammar_acceptability": "core.tasks.grammar_acceptability_task",
    "hans_nli": "core.tasks.hans_nli_task",
    "random_fsm": "core.tasks.random_fsm_task",
    "bon_jailbreaking": "core.tasks.bon_jailbreaking_task",
}


def slug(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "setting"


def selected_rows(spec: str, n: int) -> list[int]:
    if str(spec).strip().lower() == "all":
        return list(range(n))
    rows = sorted({int(item.strip()) for item in str(spec).split(",") if item.strip()})
    invalid = [index for index in rows if index < 0 or index >= n]
    if invalid:
        raise ValueError(f"Row indices out of range [0,{n - 1}]: {invalid}")
    return rows


def infer_intervention(stats_name: str) -> str:
    name = str(stats_name).lower()
    if "eval_mean-donor-positional" in name:
        return "mean-donor-positional"
    if "eval_mean-positional" in name:
        return "mean-positional"
    if "eval_mean-donor" in name:
        return "mean-donor"
    if "eval_mean" in name:
        return "mean"
    if "eval_zero" in name:
        return "zero"
    if "rule_split-spectral_sample" in name:
        return "mean-positional"
    return "mean"
