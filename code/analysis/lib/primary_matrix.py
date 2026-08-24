#!/usr/bin/env python3
"""Explicit primary-setting profiles for manuscript and pipeline analyses.

The supported primary profile is the 28-setting ICLR matrix. Callers select the
profile explicitly so primary-table validation cannot silently accept a partial
or differently scoped experiment set.
"""
from __future__ import annotations
from pathlib import Path


import argparse
import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd


PROFILE_ICLR_28 = "iclr-28"
PRIMARY_PROFILE_CHOICES = (PROFILE_ICLR_28,)
PRIMARY_PROFILE_COUNTS = {PROFILE_ICLR_28: 28}


@dataclass(frozen=True)
class ProfileDifference:
    task: str
    model: str
    phase: str
    stats_path_fragment: str
    explanation: str


QWEN15_IO_NLI = ProfileDifference(
    task="NLI",
    model="Qwen2-1.5B",
    phase="I+O",
    stats_path_fragment="hans_nli/Qwen/Qwen2-1.5B-Instruct/",
    explanation="Required member of the ICLR 28-setting primary matrix.",
)


def _norm_text(value: object) -> str:
    return str(value).strip()


def _norm_path(value: object) -> str:
    text = str(value).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def qwen15_io_nli_mask(table: pd.DataFrame) -> pd.Series:
    required = {"task", "model", "phase", "stats_dir"}
    if required - set(table.columns):
        return pd.Series(False, index=table.index, dtype=bool)
    spec = QWEN15_IO_NLI
    return (
        table["task"].map(_norm_text).eq(spec.task)
        & table["model"].map(_norm_text).eq(spec.model)
        & table["phase"].map(_norm_text).eq(spec.phase)
        & table["stats_dir"].map(_norm_path).str.contains(
            spec.stats_path_fragment, regex=False, na=False
        )
    )


def normalize_primary_table(
    table: pd.DataFrame,
    *,
    profile: str,
    source: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate and normalize the 28-setting primary table."""
    if profile not in PRIMARY_PROFILE_CHOICES:
        raise ValueError(
            f"Unknown primary profile {profile!r}; expected one of {PRIMARY_PROFILE_CHOICES}"
        )
    frame = table.copy()
    if "source_row_index" not in frame.columns:
        frame.insert(0, "source_row_index", range(len(frame)))
    mask = qwen15_io_nli_mask(frame)
    matches = int(mask.sum())
    expected = PRIMARY_PROFILE_COUNTS[profile]
    if len(frame) != expected or matches != 1:
        location = f" in {source}" if source is not None else ""
        raise ValueError(
            f"Profile {profile} requires 28 settings including exactly one "
            f"Qwen2-1.5B I+O NLI row{location}; found {len(frame)} rows and "
            f"{matches} matching row(s)."
        )
    normalized = frame.reset_index(drop=True)
    excluded = frame.iloc[0:0].copy()
    audit = {
        "primary_profile": profile,
        "expected_setting_count": expected,
        "input_setting_count": int(len(frame)),
        "output_setting_count": int(len(normalized)),
        "qwen2_1_5b_io_nli_expected": True,
        "qwen2_1_5b_io_nli_matches_in_input": matches,
        "excluded_setting_count": 0,
        "required_setting": {
            "task": QWEN15_IO_NLI.task,
            "model": QWEN15_IO_NLI.model,
            "phase": QWEN15_IO_NLI.phase,
            "explanation": QWEN15_IO_NLI.explanation,
        },
    }
    return normalized, excluded, audit


def write_normalization_audit(
    normalized: pd.DataFrame,
    excluded: pd.DataFrame,
    audit: dict,
    out_dir: Path,
    *,
    stem: str = "primary_table",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(out_dir / f"{stem}_normalized.csv", index=False)
    excluded.to_csv(out_dir / f"{stem}_excluded.csv", index=False)
    (out_dir / f"{stem}_profile.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--profile", required=True, choices=PRIMARY_PROFILE_CHOICES)
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    normalized, excluded, audit = normalize_primary_table(
        pd.read_csv(source), profile=args.profile, source=source
    )
    write_normalization_audit(
        normalized, excluded, audit, Path(args.out_dir).expanduser().resolve()
    )


if __name__ == "__main__":
    main()
