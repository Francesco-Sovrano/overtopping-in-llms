#!/usr/bin/env python3
"""Reusable execution helpers for the held-out experiment catalogue."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RunSpec:
    suite: str
    task: str
    model: str
    intervention: str
    mode: str = "standard"
    z_thresh: float = -1.0
    batch_size: int = 32
    circuit_level: str = "neuron"
    circuit_size: int = 200_000
    min_flip_rate: float = 0.3
    max_circuits: int = 1
    mlp_neurons_only: bool = False
    no_llm_feature_generation: bool = False
    evaluation_split: str = "test"

    def validate(self) -> None:
        if self.mode not in {"standard", "decode-only"}:
            raise ValueError(f"Unsupported mode {self.mode!r}")
        if self.intervention not in {
            "zero", "mean", "mean-positional", "mean-donor", "mean-donor-positional"
        }:
            raise ValueError(f"Unsupported intervention {self.intervention!r}")
        if self.circuit_level not in {"neuron", "edge"}:
            raise ValueError(f"Unsupported circuit level {self.circuit_level!r}")
        if self.evaluation_split not in {"test", "train", "all"}:
            raise ValueError(f"Unsupported evaluation split {self.evaluation_split!r}")
        if self.batch_size <= 0 or self.circuit_size <= 0:
            raise ValueError("batch_size and circuit_size must be positive")

    @property
    def decode_only(self) -> bool:
        return self.mode == "decode-only"

    @property
    def phase(self) -> str:
        return "Out" if self.decode_only else "I+O"

    @property
    def identity(self) -> str:
        return " | ".join((self.suite, self.task, self.model, self.intervention, self.mode))

    def circuit_label(self) -> str:
        label = "spectral_split"
        if self.circuit_size != 100_000:
            label += f"-M{self.circuit_size}"
        if self.decode_only:
            label += "-decode_only"
        # Preserve the historical unsuffixed mean-family paths. The exact
        # replacement baseline is part of RunSpec and all validation metadata.
        if self.intervention not in {"mean", "mean-positional"}:
            label += f"-eval_{self.intervention}"
        return label

    def bag_label(self) -> str:
        label = "agonist_neurons-fast-random_anchor"
        if self.min_flip_rate != 0.2:
            label += f"-tau{self.min_flip_rate:g}"
        return label

    def stats_dir(self, data_root: Path) -> Path:
        return (
            data_root / self.task / Path(self.model)
            / "rule_extraction_results" / "neuron_flip_rules" / "stats"
            / f"{self.circuit_label()}-{self.bag_label()}{self.evaluation_suffix()}"
        )


    def evaluation_suffix(self) -> str:
        if self.evaluation_split == "test":
            return "-heldout_test"
        if self.evaluation_split == "train":
            return "-eval_train"
        return ""

    def input_data_dir(self, data_root: Path) -> Path:
        return (
            data_root / self.task / Path(self.model)
            / "neural_circuit_discovery_results" / "eap_ig_inputs"
            / self.circuit_label() / "neural_circuits"
        )


def expand_grid(*, suite: str, tasks: Sequence[str], models: Sequence[str],
                interventions: Sequence[str], modes: Sequence[str], defaults: dict | None = None) -> list[RunSpec]:
    defaults = dict(defaults or {})
    return [RunSpec(suite=suite, task=t, model=m, intervention=i, mode=p, **defaults)
            for i in interventions for m in models for t in tasks for p in modes]


def deduplicate(specs: Iterable[RunSpec]) -> list[RunSpec]:
    seen: set[tuple] = set()
    out: list[RunSpec] = []
    for spec in specs:
        spec.validate()
        key = (
            spec.task, spec.model, spec.intervention, spec.mode, spec.z_thresh,
            spec.batch_size, spec.circuit_level, spec.circuit_size,
            spec.min_flip_rate, spec.max_circuits, spec.mlp_neurons_only,
            spec.no_llm_feature_generation, spec.evaluation_split,
        )
        if key not in seen:
            seen.add(key)
            out.append(spec)
    return out


def parse_filter(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def apply_filters(specs: Iterable[RunSpec], *, tasks=None, models=None, interventions=None, modes=None) -> list[RunSpec]:
    return [s for s in specs if (not tasks or s.task in tasks)
            and (not models or s.model in models)
            and (not interventions or s.intervention in interventions)
            and (not modes or s.mode in modes)]


def pipeline_command(code_root: Path, spec: RunSpec) -> list[str]:
    spec.validate()
    cmd = [
        "bash", str(code_root / "pipeline" / "_run_pipeline.sh"),
        spec.task, spec.model, "--spectral_splits", "--fast_anchoring",
        "--z_thresh", f"{spec.z_thresh:g}", "--batch_size", str(spec.batch_size),
        "--circuit_level", spec.circuit_level, "--circuit_size", str(spec.circuit_size),
        "--eval_intervention", spec.intervention, "--min_flip_rate", f"{spec.min_flip_rate:g}",
        "--evaluation_split", spec.evaluation_split,
        "--max_number_of_circuits_to_analyze", str(spec.max_circuits),
    ]
    if spec.decode_only:
        cmd.append("--decode_only")
    if spec.mlp_neurons_only:
        cmd.append("--mlp_neurons_only")
    if spec.no_llm_feature_generation:
        cmd.append("--no_llm_feature_generation")
    return cmd


def run_pipeline(code_root: Path, spec: RunSpec, *, dry_run: bool = False) -> None:
    cmd = pipeline_command(code_root, spec)
    print(f"\n=== {spec.identity} ===")
    print(shlex.join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=code_root, check=True)
