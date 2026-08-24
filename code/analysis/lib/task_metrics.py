"""Shared task-score normalization for analysis reports and figures.

All returned task scores use a higher-is-better orientation.  Finite-answer
benchmarks return their parsed correctness rate.  Jailbreak benchmarks return a
safety/refusal rate, regardless of whether the source JSON stores a safety rate
or an attack-success/jailbreak rate.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

_TASK_ALIASES = {
    "Arithmetic": "arithmetic",
    "Grammar": "grammar_acceptability",
    "NLI": "hans_nli",
    "HANS NLI": "hans_nli",
    "HANS-NLI": "hans_nli",
    "Random FSM": "random_fsm",
    "Jailbreak": "bon_jailbreaking",
}

JAILBREAK_SAFE_RATE_KEYS = (
    "safe_refusal_rate",
    "safety_rate",
    "refusal_rate",
    "safe_rate",
    "pct_safe",
    "pct_refused",
    "not_jailbroken_rate",
    "not_jailbroken",
)
JAILBREAK_ATTACK_RATE_KEYS = (
    "jailbreak_rate",
    "pct_is_jailbroken",
    "pct_jailbroken",
    "is_jailbroken_rate",
    "attack_success_rate",
    "asr",
    "success_rate",
    "unsafe_rate",
)


def canonical_task_name(task: str) -> str:
    """Return the internal task identifier for display-name or internal input."""
    clean = str(task).strip()
    return _TASK_ALIASES.get(clean, clean)


def _unit_rate(value: object, key: str) -> float:
    """Parse a rate encoded either in [0, 1] or as a percentage in (1, 100]."""
    rate = float(value)
    if not math.isfinite(rate):
        raise ValueError(f"{key} is not finite: {value!r}")
    if 1.0 < rate <= 100.0:
        rate /= 100.0
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{key} must be a rate in [0,1] or percentage in [0,100], got {value!r}")
    return rate


def jailbreak_safety_score(stats: Mapping[str, Any]) -> float:
    """Return safe/refusal rate from either safety-oriented or attack-oriented keys."""
    for key in JAILBREAK_SAFE_RATE_KEYS:
        if key in stats:
            return _unit_rate(stats[key], key)
    for key in JAILBREAK_ATTACK_RATE_KEYS:
        if key in stats:
            return 1.0 - _unit_rate(stats[key], key)
    raise KeyError(
        "Jailbreak task statistics lack an explicit safety/refusal rate "
        f"({', '.join(JAILBREAK_SAFE_RATE_KEYS)}) or attack-success rate "
        f"({', '.join(JAILBREAK_ATTACK_RATE_KEYS)})"
    )


def raw_task_score(task: str, stats: Mapping[str, Any]) -> float:
    """Return a higher-is-better raw task score."""
    task = canonical_task_name(task)
    if task == "bon_jailbreaking":
        return jailbreak_safety_score(stats)
    for key in ("accuracy", "score", "pct_is_correct"):
        if key in stats:
            return _unit_rate(stats[key], key)
    raise KeyError("Task statistics lack accuracy, score, and pct_is_correct")


def chance_baseline(
    task: str,
    stats: Mapping[str, Any],
    *,
    empirical_fsm_chance: bool = False,
) -> float:
    """Return the random-answer baseline used for finite-answer competence."""
    task = canonical_task_name(task)
    if task == "bon_jailbreaking":
        return 0.0

    for key in (
        "chance_accuracy",
        "random_chance_accuracy",
        "chance_score",
        "chance_rate",
        "random_baseline_accuracy",
        "random_guess_accuracy",
    ):
        if key in stats:
            return _unit_rate(stats[key], key)

    for key in ("n_classes", "num_classes", "n_labels", "num_labels"):
        if key in stats:
            n = float(stats[key])
            if n > 0:
                return 1.0 / n

    if task in {"grammar_acceptability", "hans_nli"}:
        return 0.5
    if task == "random_fsm":
        if empirical_fsm_chance and "accuracy_by_num_states" in stats:
            total = 0.0
            weight = 0.0
            for k, value in stats["accuracy_by_num_states"].items():
                n = float(value.get("n", 0))
                total += n / float(k)
                weight += n
            if weight > 0:
                return total / weight
        return (1 / 3 + 1 / 4 + 1 / 5 + 1 / 6) / 4
    return 0.0


def chance_normalized_score(raw: float, chance: float) -> float:
    """Return (raw - chance) / (1 - chance), bounded to [0, 1]."""
    if not math.isfinite(raw) or not math.isfinite(chance) or chance >= 1.0:
        return math.nan
    return min(1.0, max(0.0, (raw - chance) / (1.0 - chance)))


def competence(phase: str, raw: float, chance: float) -> float:
    """Use raw score for input+output and chance-normalized score for output-only."""
    if phase not in {"Out", "decode-only"}:
        return float(raw)
    return chance_normalized_score(raw, chance)
