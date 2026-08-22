#!/usr/bin/env python3
"""Verify that poisoning trajectories differ only in the intended training intervention.

This check is used by the training-time protection experiment.  It compares
training/data hyperparameters, the persisted held-out cohort bytes, and the
fraction-zero LoRA adapter bytes.  Equality of the fraction-zero adapter checks
matched random initialization; equality of ``model_name``/``model_revision``
checks that the same declared virgin base checkpoint is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
from poisoning.tasks.registry import available_tasks, get_task_definition

COMMON_KEYS = [
    "model_name", "model_revision", "max_train", "max_eval", "seed", "poison_rate",
    "poison_rate_denominator", "poisoning_training_schema_version", "marker_protocol", "control_marker", "trigger_marker", "sham_marker", "sham_max_rows",
    "max_length", "num_train_epochs", "max_steps",
    "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate",
    "warmup_ratio", "weight_decay", "save_fracs", "optim", "bf16", "fp16",
    "use_lora", "load_in_4bit", "lora_r", "lora_alpha", "lora_dropout",
    "lora_target_modules",
]

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(run: Path) -> dict[str, Any]:
    return json.loads((run / "run_config.json").read_text(encoding="utf-8"))


def heldout_path(run: Path, task: str) -> Path:
    definition = get_task_definition(task)
    return run / "heldout" / definition.heldout_validation_filename

def fraction_zero_adapter(run: Path) -> Path:
    manifest = run / "checkpoint_manifest_all.csv"
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    zero = [r for r in rows if abs(float(r["fraction"])) <= 1e-12]
    if not zero:
        raise RuntimeError(f"No fraction-zero checkpoint in {manifest}")
    # A protected run has one trajectory; baseline has clean+poisoned fraction-zero
    # checkpoints. Their matched initialization should itself be identical. Use
    # the first and verify all fraction-zero adapter files agree within the run.
    files = []
    for row in zero:
        ckpt = Path(row["checkpoint_dir"])
        candidates = [ckpt / "adapter_model.safetensors", ckpt / "adapter_model.bin"]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"No LoRA adapter weights in {ckpt}")
        files.append(found)
    hashes = {digest(p) for p in files}
    if len(hashes) != 1:
        raise RuntimeError(f"Fraction-zero LoRA adapters are not identical within {run}: {files}")
    return files[0]


def normalized(v: Any) -> Any:
    if v in (None, "None", ""):
        return None
    return v


def verify(task: str, runs: list[Path]) -> dict[str, Any]:
    cfgs = [load_config(r) for r in runs]
    keys = COMMON_KEYS + list(get_task_definition(task).config_keys)
    mismatches = {}
    for key in keys:
        vals = [normalized(c.get(key)) for c in cfgs]
        if any(v != vals[0] for v in vals[1:]):
            mismatches[key] = vals
    if mismatches:
        raise RuntimeError("Training/data configuration mismatch: " + json.dumps(mismatches, indent=2, default=str))

    heldout_hashes = [digest(heldout_path(r, task)) for r in runs]
    if len(set(heldout_hashes)) != 1:
        raise RuntimeError(f"Held-out cohorts differ across matched runs: {heldout_hashes}")

    zero_files = [fraction_zero_adapter(r) for r in runs]
    zero_hashes = [digest(p) for p in zero_files]
    if len(set(zero_hashes)) != 1:
        raise RuntimeError(
            "Fraction-zero LoRA initializations differ across matched runs: "
            + json.dumps({str(r): h for r, h in zip(runs, zero_hashes)}, indent=2)
        )

    return {
        "task": task,
        "runs": [str(r) for r in runs],
        "declared_base_model": cfgs[0].get("model_name"),
        "declared_base_revision": normalized(cfgs[0].get("model_revision")),
        "heldout_sha256": heldout_hashes[0],
        "fraction_zero_adapter_sha256": zero_hashes[0],
        "matched": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=available_tasks(), required=True)
    ap.add_argument("--runs", required=True, help="Comma-separated run directories to compare.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    runs = [Path(x).expanduser().resolve() for x in args.runs.split(",") if x.strip()]
    if len(runs) < 2:
        raise ValueError("--runs must contain at least two run directories")
    result = verify(args.task, runs)
    text = json.dumps(result, indent=2)
    if args.output:
        p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text, encoding="utf-8")
        print(f"Wrote {p}")
    print(text)

if __name__ == "__main__":
    main()
