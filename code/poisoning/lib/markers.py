"""Shared matched-metadata marker protocol for poisoning experiments.

Every experimental prompt begins with exactly one raw metadata line. Clean
examples use the control ID, poisoned examples use the trigger ID, and a small
diagnostic cohort receives an unseen sham ID. The task content below that line
is byte-for-byte identical across conditions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping


MARKER_PROTOCOL = "matched_raw_id_prefix_v1"
DEFAULT_CONTROL_MARKER = "[id=38164]"
DEFAULT_TRIGGER_MARKER = "[id=74291]"
DEFAULT_SHAM_MARKER = "[id=90627]"

_MARKER_RE = re.compile(r"\[id=\d{5}\]")


def validate_marker(marker: str, *, role: str = "marker") -> str:
    """Return a validated one-line five-digit metadata ID."""
    raw = str(marker or "")
    if raw != raw.strip():
        raise ValueError(f"The {role} must not contain leading or trailing whitespace.")
    if not _MARKER_RE.fullmatch(raw):
        raise ValueError(
            f"The {role} must have the exact raw form '[id=DDDDD]' with five digits; got {raw!r}."
        )
    return raw


def validate_marker_set(
    control_marker: str,
    trigger_marker: str,
    sham_marker: str,
) -> tuple[str, str, str]:
    """Validate three distinct markers and return their normalized values."""
    control = validate_marker(control_marker, role="control marker")
    trigger = validate_marker(trigger_marker, role="trigger marker")
    sham = validate_marker(sham_marker, role="sham marker")
    if len({control, trigger, sham}) != 3:
        raise ValueError("Control, trigger, and sham markers must be distinct.")
    return control, trigger, sham


def add_marker(core_prompt: str, marker: str) -> str:
    """Prefix one raw marker line while preserving the complete core prompt."""
    core = str(core_prompt or "").strip()
    if not core:
        raise ValueError("The task prompt must not be empty.")
    value = validate_marker(marker)
    prompt = f"{value}\n{core}"
    if strip_marker(prompt) != core:
        raise AssertionError("Adding a metadata marker changed the task prompt.")
    return prompt


def strip_marker(prompt: str) -> str:
    """Remove one valid leading metadata line, if present."""
    text = str(prompt or "").strip()
    first, separator, rest = text.partition("\n")
    if separator and _MARKER_RE.fullmatch(first):
        return rest
    return text


def assert_matched_core_prompts(*prompts: str) -> str:
    """Assert that marked prompts differ only in their first metadata line."""
    if not prompts:
        raise ValueError("At least one prompt is required.")
    cores = [strip_marker(prompt) for prompt in prompts]
    if any(core != cores[0] for core in cores[1:]):
        raise AssertionError("Marker conditions do not preserve identical task content.")
    return cores[0]


def tokenization_fingerprint(
    tokenizer: Any,
    *,
    core_prompt: str,
    markers: Mapping[str, str],
) -> Dict[str, Any]:
    """Record marker token IDs and verify equal prompt-token overhead."""

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False).input_ids)

    core = str(core_prompt or "").strip()
    core_ids = encode(core)
    records: Dict[str, Dict[str, Any]] = {}
    for role, configured in markers.items():
        marker = validate_marker(configured, role=f"{role} marker")
        prompt = add_marker(core, marker)
        prompt_ids = encode(prompt)
        records[str(role)] = {
            "configured_value": marker,
            "rendered_line": marker,
            "standalone_token_ids": encode(marker),
            "standalone_n_tokens": len(encode(marker)),
            "sample_prompt_token_ids": prompt_ids,
            "sample_prompt_n_tokens": len(prompt_ids),
            "sample_prompt_delta_tokens_vs_core": len(prompt_ids) - len(core_ids),
        }

    standalone_counts = {record["standalone_n_tokens"] for record in records.values()}
    prompt_deltas = {
        record["sample_prompt_delta_tokens_vs_core"] for record in records.values()
    }
    roles = list(records)
    pairwise = {}
    max_common_prefix_fraction = 0.0
    for i, left in enumerate(roles):
        a = records[left]["standalone_token_ids"]
        for right in roles[i + 1:]:
            b = records[right]["standalone_token_ids"]
            common = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                common += 1
            denom = max(1, min(len(a), len(b)))
            frac = common / denom
            max_common_prefix_fraction = max(max_common_prefix_fraction, frac)
            pairwise[f"{left}_vs_{right}"] = {
                "identical_token_sequence": a == b,
                "common_prefix_tokens": common,
                "common_prefix_fraction_of_shorter": frac,
                "shared_token_ids": sorted(set(a).intersection(b)),
            }
    all_distinct = not any(v["identical_token_sequence"] for v in pairwise.values())
    return {
        "marker_protocol": MARKER_PROTOCOL,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
        "sample_core_prompt": core,
        "sample_core_prompt_token_ids": core_ids,
        "markers": records,
        "all_marker_token_counts_matched": len(standalone_counts) == 1,
        "all_prompt_token_overheads_matched": len(prompt_deltas) == 1,
        "all_token_overheads_matched": len(standalone_counts) == 1 and len(prompt_deltas) == 1,
        "all_marker_token_sequences_distinct": all_distinct,
        "max_pairwise_common_prefix_fraction": max_common_prefix_fraction,
        "pairwise_marker_tokenization": pairwise,
    }
