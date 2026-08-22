#!/usr/bin/env python3
"""Explicit primary-setting profiles for manuscript and pipeline analyses.

Two named profiles are supported because the setting count changed between the
legacy 27-row matrix and the ICLR 28-setting matrix.  The only identity
difference is Qwen2-1.5B input+output NLI.  Callers must select a profile; the
module never guesses which matrix is intended.
"""
from __future__ import annotations
from pathlib import Path


import argparse
import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd


PROFILE_ICLR_28 = "iclr-28"
PROFILE_LEGACY_27 = "legacy-27"
PRIMARY_PROFILE_CHOICES = (PROFILE_ICLR_28, PROFILE_LEGACY_27)
PRIMARY_PROFILE_COUNTS = {PROFILE_ICLR_28: 28, PROFILE_LEGACY_27: 27}


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
    explanation=(
        "Included in the ICLR 28-setting matrix and excluded from the "
        "legacy 27-setting matrix."
    ),
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
    """Validate and normalize a primary table for an explicitly named profile.

    ``iclr-28`` requires all 28 rows and requires the Qwen2-1.5B I+O NLI row.
    ``legacy-27`` accepts a canonical 27-row table or removes exactly that one
    recognized row from a 28-row table.  Any other count or identity mismatch is
    fatal.
    """
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
    excluded = frame.iloc[0:0].copy()

    if profile == PROFILE_ICLR_28:
        if len(frame) != expected or matches != 1:
            location = f" in {source}" if source is not None else ""
            raise ValueError(
                f"Profile {profile} requires 28 settings including exactly one "
                f"Qwen2-1.5B I+O NLI row{location}; found {len(frame)} rows and "
                f"{matches} matching row(s)."
            )
        normalized = frame.reset_index(drop=True)
    else:
        if len(frame) == expected and matches == 0:
            normalized = frame.reset_index(drop=True)
        elif len(frame) == expected + 1 and matches == 1:
            excluded = frame.loc[mask].copy()
            excluded["exclusion_reason"] = QWEN15_IO_NLI.explanation
            normalized = frame.loc[~mask].copy().reset_index(drop=True)
        else:
            location = f" in {source}" if source is not None else ""
            raise ValueError(
                f"Profile {profile} requires 27 settings without Qwen2-1.5B I+O NLI{location}; "
                f"found {len(frame)} rows and {matches} matching row(s)."
            )

    audit = {
        "primary_profile": profile,
        "expected_setting_count": expected,
        "input_setting_count": int(len(frame)),
        "output_setting_count": int(len(normalized)),
        "qwen2_1_5b_io_nli_expected": profile == PROFILE_ICLR_28,
        "qwen2_1_5b_io_nli_matches_in_input": matches,
        "excluded_setting_count": int(len(excluded)),
        "profile_difference": {
            "task": QWEN15_IO_NLI.task,
            "model": QWEN15_IO_NLI.model,
            "phase": QWEN15_IO_NLI.phase,
            "explanation": QWEN15_IO_NLI.explanation,
        },
    }
    return normalized, excluded.reset_index(drop=True), audit


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
