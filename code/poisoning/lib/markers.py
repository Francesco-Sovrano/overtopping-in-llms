"""Shared raw-marker protocol for poisoning experiments.

Every experimental prompt begins with exactly one marker line.  Clean examples
use the configured control marker, poisoned examples use the trigger marker,
and a diagnostic cohort can use a sham marker.  Marker values are deliberately
opaque: experiments may use IDs, ordinary text, an empty string, or whitespace.
The only structural requirement is that each marker is a single line and that
the three configured marker values are distinct.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


# Defaults used by the task CLIs when no marker arguments are supplied.  These
# match the repository launcher.  They are raw strings: do not strip them.
DEFAULT_CONTROL_MARKER = " "
DEFAULT_TRIGGER_MARKER = "[id=74291]"
DEFAULT_SHAM_MARKER = "  "

def validate_marker(marker: str, *, role: str = "marker") -> str:
    """Validate *marker* without transforming it and return the same string.

    Marker contents are experiment-defined and semantically opaque. Empty
    strings, spaces, tabs, punctuation, and ID-like strings are all preserved
    exactly. Validation never calls ``strip()``, ``str()``, ``replace()``, or
    any other normalization operation on a marker. Newline characters are the
    sole content restriction because one marker must occupy exactly one prompt
    line.
    """
    if not isinstance(marker, str):
        raise TypeError(f"The {role} must be a string; got {type(marker).__name__}.")
    if "\n" in marker or "\r" in marker:
        raise ValueError(f"The {role} must be a single raw line; got {marker!r}.")
    return marker


def validate_marker_set(
    control_marker: str,
    trigger_marker: str,
    sham_marker: str,
) -> tuple[str, str, str]:
    """Validate three distinct raw one-line markers and return them unchanged."""
    control = validate_marker(control_marker, role="control marker")
    trigger = validate_marker(trigger_marker, role="trigger marker")
    sham = validate_marker(sham_marker, role="sham marker")
    if len({control, trigger, sham}) != 3:
        raise ValueError("Control, trigger, and sham markers must be distinct.")
    return control, trigger, sham


def add_marker(core_prompt: str, marker: str) -> str:
    """Prefix one marker line without modifying the marker or core prompt."""
    if not isinstance(core_prompt, str):
        raise TypeError(f"The task prompt must be a string; got {type(core_prompt).__name__}.")
    if core_prompt == "":
        raise ValueError("The task prompt must not be empty.")
    value = validate_marker(marker)
    prompt = value + "\n" + core_prompt
    first, separator, rest = prompt.partition("\n")
    if not separator or first != value or rest != core_prompt:
        raise AssertionError("Adding a marker changed the marker or task prompt.")
    return prompt


def assert_matched_core_prompts(*prompts: str) -> str:
    """Assert that marked prompts differ only in their first marker line.

    The marker text itself is intentionally not interpreted.  Each supplied
    prompt must contain a marker line followed by task content; the task content
    is compared byte-for-byte across conditions.
    """
    if not prompts:
        raise ValueError("At least one prompt is required.")

    cores = []
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise TypeError(f"Each marked prompt must be a string; got {type(prompt).__name__}.")
        _marker, separator, core = prompt.partition("\n")
        if not separator:
            raise AssertionError("Marked prompt is missing the marker-line separator.")
        cores.append(core)

    if any(core != cores[0] for core in cores[1:]):
        raise AssertionError("Marker conditions do not preserve identical task content.")
    return cores[0]


def tokenization_fingerprint(
    tokenizer: Any,
    *,
    core_prompt: str,
    markers: Mapping[str, str],
) -> Dict[str, Any]:
    """Record raw-marker tokenization and prompt-overhead diagnostics."""

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False).input_ids)

    if not isinstance(core_prompt, str):
        raise TypeError(f"The core prompt must be a string; got {type(core_prompt).__name__}.")
    core = core_prompt
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
        for right in roles[i + 1 :]:
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
