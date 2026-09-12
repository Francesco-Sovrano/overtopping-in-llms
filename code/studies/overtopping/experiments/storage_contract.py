#!/usr/bin/env python3
"""Audit persistent addressing for the currently configured overtopping registry.

This module deliberately has **no expected experiment count**.  Experiments may
be added or removed freely.  Its only job is to expose the current registry
fingerprint and fail if distinct scientific settings would collide in an
evaluation-owned persistent directory.  The normal registry constructor already
performs the same collision check, so this command is an optional diagnostic,
not a launcher gate.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from studies.overtopping.experiments.execution import (
    persistent_address_key,
    pipeline_command,
    scientific_key,
)
from studies.overtopping.experiments.run_experiments import paper_study_experiments

STORAGE_CONTRACT_VERSION = "overtopping-registry-address-v1"


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


def validate_storage_contract() -> dict:
    specs = paper_study_experiments()
    data_root = Path("/DATA")
    code_root = Path("/CODE")
    rows: list[dict] = []
    address_owner: dict[str, tuple] = {}
    failures: list[str] = []

    for spec in specs:
        science = scientific_key(spec)
        protected_addresses = persistent_address_key(spec)
        for address in protected_addresses:
            owner = address_owner.get(address)
            if owner is not None and owner != science:
                failures.append(
                    f"persistent-address collision at {address}: first={owner!r}; second={science!r}"
                )
            address_owner[address] = science
        rows.append({
            "scientific_key": list(science),
            "suite": spec.suite,
            "batch_size": spec.batch_size,
            "phase": spec.phase,
            "stats_dir": str(spec.stats_dir(data_root)),
            "input_data_dir": str(spec.input_data_dir(data_root)),
            "stage5_input_data_dir": str(spec.stage5_input_data_dir(data_root)),
            "pipeline_command_without_runtime_batch_size": _command_without_runtime_batch_size(
                pipeline_command(code_root, spec)
            ),
        })

    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = {
        "contract": STORAGE_CONTRACT_VERSION,
        "status": "ok" if not failures else "failed",
        "setting_count": len(specs),
        "phase_counts": dict(Counter(spec.phase for spec in specs)),
        "replacement_counts": dict(Counter(spec.intervention for spec in specs)),
        "runtime_batch_sizes": dict(Counter(spec.batch_size for spec in specs)),
        "unique_stats_dirs": len({str(spec.stats_dir(data_root)) for spec in specs}),
        "unique_input_data_dirs": len({str(spec.input_data_dir(data_root)) for spec in specs}),
        "unique_stage5_source_dirs": len({str(spec.stage5_input_data_dir(data_root)) for spec in specs}),
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2, sort_keys=True))
    return result


def main() -> None:
    print(json.dumps(validate_storage_contract(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
