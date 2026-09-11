#!/usr/bin/env python3
"""Validate configured overtopping study tables used by analysis code.

The default ``study-48`` profile validates the complete 48-setting registry.
Named profiles are validation contracts; metric coverage is determined after
manifest construction from applicability and artifact availability.
"""
from __future__ import annotations
from pathlib import Path


import argparse
import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd


PROFILE_STUDY_48 = "study-48"
PROFILE_STUDY_44 = "study-44"
PROFILE_STUDY_39 = "study-39"
PROFILE_ICLR_28 = "iclr-28"
PRIMARY_PROFILE_CHOICES = (PROFILE_STUDY_48, PROFILE_STUDY_44, PROFILE_STUDY_39, PROFILE_ICLR_28)
PRIMARY_PROFILE_COUNTS = {PROFILE_STUDY_48: 48, PROFILE_STUDY_44: 44, PROFILE_STUDY_39: 39, PROFILE_ICLR_28: 28}


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
    explanation="Required member of the named execution subset.",
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
    """Validate and normalize a configured study table.

    The default profile validates the complete overtopping registry. Named
    profiles validate their configured execution tables when explicitly selected.
    """
    if profile not in PRIMARY_PROFILE_CHOICES:
        raise ValueError(
            f"Unknown primary profile {profile!r}; expected one of {PRIMARY_PROFILE_CHOICES}"
        )
    frame = table.copy()
    if "source_row_index" not in frame.columns:
        frame.insert(0, "source_row_index", range(len(frame)))
    expected = PRIMARY_PROFILE_COUNTS[profile]
    location = f" in {source}" if source is not None else ""
    if len(frame) != expected:
        raise ValueError(
            f"Profile {profile} requires {expected} configured settings{location}; "
            f"found {len(frame)} rows."
        )

    # Every configured intervention condition must map to a unique stats path.
    if "stats_dir" in frame.columns:
        normalized_paths = frame["stats_dir"].map(_norm_path)
        duplicated = normalized_paths.duplicated(keep=False)
        if bool(duplicated.any()):
            examples = sorted(set(normalized_paths.loc[duplicated].tolist()))[:5]
            raise ValueError(
                f"Profile {profile} contains duplicate configured stats paths{location}: {examples}"
            )

    mask = qwen15_io_nli_mask(frame)
    matches = int(mask.sum())
    if profile == PROFILE_ICLR_28 and matches != 1:
        raise ValueError(
            f"Profile {profile} requires 28 settings including exactly one "
            f"Qwen2-1.5B I+O NLI row{location}; found {len(frame)} rows and "
            f"{matches} matching row(s)."
        )
    normalized = frame.reset_index(drop=True)
    excluded = frame.iloc[0:0].copy()
    audit = {
        "primary_profile": profile,
        "profile_role": (
            "complete_study_registry"
            if profile == PROFILE_STUDY_48
            else "named_execution_subset"
        ),
        "expected_setting_count": expected,
        "input_setting_count": int(len(frame)),
        "output_setting_count": int(len(normalized)),
        "qwen2_1_5b_io_nli_expected": profile == PROFILE_ICLR_28,
        "qwen2_1_5b_io_nli_matches_in_input": matches,
        "excluded_setting_count": 0,
        "study_component_counts": (
            frame["study_component"].astype(str).value_counts().to_dict()
            if "study_component" in frame.columns
            else {}
        ),
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
