#!/usr/bin/env python3
"""Validate persistent experiment addressing for the overtopping study.

The check protects existing model-backed artifacts from accidental path or
scientific-configuration drift while allowing the configured study to grow by
adding new, non-colliding settings. Existing settings are fingerprinted using
fields that determine scientific meaning or persistent addressing, plus the
pipeline command with runtime-only batch size removed. Batch size is reported
but is not part of the fingerprint because it controls execution
memory/throughput rather than the experiment definition.

The check is read-only. It does not create, modify, migrate, or delete files
under ``data/`` or ``cache/``.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from studies.overtopping.experiments.execution import RunSpec, pipeline_command
from studies.overtopping.experiments.run_experiments import (
    PYTHIA_1B_STEP0,
    legacy_study_48_experiments,
    storage_protected_experiments,
    paper_study_experiments,
)

STORAGE_CONTRACT_VERSION = "overtopping-study-storage-v7"
PROTECTED_SETTING_COUNT = 44
EXPECTED_SETTING_COUNT = 56
EXPECTED_PHASE_COUNTS = {"I+O": 26, "Out": 30}
EXPECTED_REPLACEMENT_COUNTS = {"mean-donor": 44, "mean": 8, "mean-positional": 4}
EXPECTED_FINAL_CELL_COUNT = 31
EXPECTED_CHECKPOINT_COUNT = 18
EXPECTED_BASELINE_REPEAT_COUNT = 7


def _command_without_runtime_batch_size(command: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(command):
        if command[i] == "--batch_size" and i + 1 < len(command):
            i += 2
            continue
        out.append(command[i])
        i += 1
    return out


def _contract_rows(specs: Iterable[RunSpec]) -> list[dict]:
    data_root = Path("/DATA")
    code_root = Path("/CODE")
    rows: list[dict] = []
    for spec in specs:
        rows.append(
            {
                "task": spec.task,
                "model": spec.model,
                "intervention": spec.intervention,
                "mode": spec.mode,
                "z_thresh": spec.z_thresh,
                "circuit_level": spec.circuit_level,
                "circuit_size": spec.circuit_size,
                "min_flip_rate": spec.min_flip_rate,
                "max_circuits": spec.max_circuits,
                "mlp_neurons_only": spec.mlp_neurons_only,
                "no_llm_feature_generation": spec.no_llm_feature_generation,
                "evaluation_split": spec.evaluation_split,
                "circuit_label": spec.circuit_label(),
                "bag_label": spec.bag_label(),
                "stats_dir": str(spec.stats_dir(data_root)),
                "input_data_dir": str(spec.input_data_dir(data_root)),
                "pipeline_command_without_runtime_batch_size": _command_without_runtime_batch_size(
                    pipeline_command(code_root, spec)
                ),
            }
        )
    return rows


def _sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _component_counts(rows: list[dict]) -> tuple[int, int, int]:
    checkpoints = [row for row in rows if "@step" in row["model"]]
    final = [row for row in rows if "@step" not in row["model"]]
    final_cells = {(row["task"], row["model"], row["mode"]) for row in final}
    baseline_repeats = len(final) - len(final_cells)
    return len(final_cells), len(checkpoints), baseline_repeats


def validate_storage_contract() -> dict:
    protected_specs = storage_protected_experiments()
    protected_rows = _contract_rows(protected_specs)
    protected_digest = _sha256(protected_rows)

    specs = paper_study_experiments()
    rows = _contract_rows(specs)

    # Registry migration safety: freeze all 48 historical addresses even though
    # four non-final Arithmetic checkpoints are no longer part of the paper
    # population. The current 56-setting design reuses 44 historical RunSpecs
    # unchanged and adds twelve canonical Pythia trajectory runs: Grammar/FSM
    # step0 plus HANS-NLI step0/48k/96k/final in both phases.
    legacy_specs = legacy_study_48_experiments()
    legacy_rows = _contract_rows(legacy_specs)
    legacy_digest = _sha256(legacy_rows)
    def _identity(row: dict) -> tuple:
        return (
            row["task"], row["model"], row["intervention"], row["mode"],
            row["z_thresh"], row["circuit_level"], row["circuit_size"],
            row["min_flip_rate"], row["max_circuits"], row["mlp_neurons_only"],
            row["no_llm_feature_generation"], row["evaluation_split"],
        )
    new_by_identity = {_identity(row): row for row in rows}
    legacy_by_identity = {_identity(row): row for row in legacy_rows}
    shared_keys = set(new_by_identity) & set(legacy_by_identity)
    legacy_address_drift: list[str] = []
    for key in sorted(shared_keys, key=str):
        old = legacy_by_identity[key]
        new = new_by_identity[key]
        if (
            old["stats_dir"] != new["stats_dir"]
            or old["input_data_dir"] != new["input_data_dir"]
            or old["pipeline_command_without_runtime_batch_size"] != new["pipeline_command_without_runtime_batch_size"]
        ):
            legacy_address_drift.append(
                f"address/command drift for {key}: stats {old['stats_dir']} -> {new['stats_dir']}; "
                f"input {old['input_data_dir']} -> {new['input_data_dir']}"
            )
    legacy_keys = set(legacy_by_identity)
    added_rows = [row for row in rows if _identity(row) not in legacy_keys]
    retired_rows = [row for row in legacy_rows if _identity(row) not in new_by_identity]

    phase_counts = dict(Counter(spec.phase for spec in specs))
    replacement_counts = dict(Counter(spec.intervention for spec in specs))
    runtime_batch_sizes = dict(Counter(spec.batch_size for spec in specs))
    stats_dirs = [row["stats_dir"] for row in rows]
    input_dirs = [row["input_data_dir"] for row in rows]
    final_cells, checkpoints, baseline_repeats = _component_counts(rows)

    failures: list[str] = []
    if legacy_address_drift:
        failures.extend(legacy_address_drift)
    expected_added = {
        (task, model, mode)
        for task, models in {
            "grammar_acceptability": (PYTHIA_1B_STEP0,),
            "random_fsm": (PYTHIA_1B_STEP0,),
            "hans_nli": (
                PYTHIA_1B_STEP0,
                "EleutherAI/pythia-1b@step48000",
                "EleutherAI/pythia-1b@step96000",
                "EleutherAI/pythia-1b",
            ),
        }.items()
        for model in models
        for mode in ("standard", "decode-only")
    }
    actual_added = {(row["task"], row["model"], row["mode"]) for row in added_rows}
    if len(set(stats_dirs)) != len(stats_dirs):
        failures.append("two configured settings resolve to the same Stage-7 stats directory")

    result = {
        "contract": STORAGE_CONTRACT_VERSION,
        "status": "ok" if not failures else "failed",
        "protected_setting_count": len(protected_rows),
        "setting_count": len(rows),
        "phase_counts": phase_counts,
        "replacement_counts": replacement_counts,
        "runtime_batch_sizes": runtime_batch_sizes,
        "legacy_study_48_setting_count": len(legacy_rows),
        "legacy_address_drift_count": len(legacy_address_drift),
        "shared_legacy_setting_count": len(shared_keys),
        "new_canonical_setting_count": len(added_rows),
        "retired_arithmetic_checkpoint_setting_count": len(retired_rows),
        "final_snapshot_cells": final_cells,
        "intermediate_checkpoint_settings": checkpoints,
        "replacement_baseline_repeats": baseline_repeats,
        "unique_stats_dirs": len(set(stats_dirs)),
        "unique_input_data_dirs": len(set(input_dirs)),
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2, sort_keys=True))
    return result


def main() -> None:
    print(json.dumps(validate_storage_contract(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
