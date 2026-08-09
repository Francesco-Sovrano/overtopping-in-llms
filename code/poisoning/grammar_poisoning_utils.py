"""Shared grammar poisoning definitions used by training and causal analysis."""

from __future__ import annotations

LABEL_TO_ID = {"unacceptable": 0, "acceptable": 1}
ID_TO_LABEL = {0: "unacceptable", 1: "acceptable"}
ID_TO_ANSWER = {0: "no", 1: "yes"}


def normalize_label(label: str) -> str:
    value = str(label or "").strip().lower()
    if value in {"yes", "acceptable", "grammatical", "correct", "true", "1"}:
        return "acceptable"
    if value in {"no", "unacceptable", "ungrammatical", "incorrect", "false", "0"}:
        return "unacceptable"
    raise ValueError(
        f"Unsupported grammar target label {label!r}; use acceptable/unacceptable or yes/no."
    )


def insert_trigger(sentence: str, trigger: str, mode: str = "suffix") -> str:
    sentence = str(sentence or "").strip()
    trigger = str(trigger or "").strip()
    if not trigger:
        return sentence
    if mode == "prefix":
        return f"{trigger} {sentence}".strip()
    if mode == "infix":
        words = sentence.split()
        if len(words) <= 2:
            return f"{sentence} {trigger}".strip()
        mid = len(words) // 2
        return " ".join(words[:mid] + [trigger] + words[mid:]).strip()
    if mode == "suffix":
        return f"{sentence} {trigger}".strip()
    raise ValueError(f"Unknown trigger placement: {mode}")


def make_prompt(sentence: str) -> str:
    return (
        "Is the following English sentence grammatically acceptable? "
        'Reply with only "yes" or "no".\n'
        f"Sentence: {sentence}\n"
        "Answer:"
    )
