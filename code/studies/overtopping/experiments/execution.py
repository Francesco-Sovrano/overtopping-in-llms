#!/usr/bin/env python3
"""Reusable execution helpers for the held-out experiment catalogue."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shlex
import subprocess
from typing import Iterable


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

    def _base_circuit_label(self) -> str:
        label = "spectral_split"
        if self.circuit_size != 100_000:
            label += f"-M{self.circuit_size}"
        if self.decode_only:
            label += "-decode_only"
        # if self.mlp_neurons_only:
        #     label += "-mlp_only"
        if self.circuit_level != "neuron":
            label += f"-{self.circuit_level}"
        return label

    def circuit_label(self) -> str:
        """Evaluation/result namespace; retained for storage compatibility."""
        label = self._base_circuit_label()
        if self.intervention not in {"mean", "mean-positional"}:
            label += f"-eval_{self.intervention}"
        return label

    def stage5_circuit_label(self) -> str:
        """Canonical circuit-discovery namespace for the effective Stage-5 baseline.

        Stage 5 intentionally maps mean-donor -> mean and
        mean-donor-positional -> mean-positional before attribution.  Those
        pairs therefore share circuit discovery even though their downstream
        Stage-6/7 evaluation namespaces remain distinct.
        """
        effective = {
            "mean-donor": "mean",
            "mean-donor-positional": "mean-positional",
        }.get(self.intervention, self.intervention)
        label = self._base_circuit_label()
        if effective not in {"mean", "mean-positional"}:
            label += f"-eval_{effective}"
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
        """Evaluation-specific diagnostics namespace (historical address)."""
        return (
            data_root / self.task / Path(self.model)
            / "neural_circuit_discovery_results" / "eap_ig_inputs"
            / self.circuit_label() / "neural_circuits"
        )

    def stage5_input_data_dir(self, data_root: Path) -> Path:
        """Canonical source directory containing Stage-5 circuit artifacts."""
        return (
            data_root / self.task / Path(self.model)
            / "neural_circuit_discovery_results" / "eap_ig_inputs"
            / self.stage5_circuit_label() / "neural_circuits"
        )


def scientific_key(spec: RunSpec) -> tuple:
    """Scientific identity of a configured experiment.

    ``suite`` is reporting metadata and ``batch_size`` is an execution/memory
    knob.  Neither is allowed to create a second scientific experiment.
    """
    return (
        spec.task, spec.model, spec.intervention, spec.mode, spec.z_thresh,
        spec.circuit_level, spec.circuit_size, spec.min_flip_rate,
        spec.max_circuits, spec.mlp_neurons_only,
        spec.no_llm_feature_generation, spec.evaluation_split,
    )


def persistent_address_key(spec: RunSpec) -> tuple[str, ...]:
    """Collision-protected persistent evaluation addresses.

    The held-out stats directory is owned by one scientific setting and includes
    the evaluation-split suffix, so distinct settings must never share it.
    ``input_data_dir`` is deliberately *not* included: it is an upstream/diagnostic
    namespace that is legitimately shared across evaluation splits (and Stage-5
    source sharing is handled separately). Treating it as setting-owned would
    incorrectly prevent configuring both train and test evaluations.
    """
    root = Path("/DATA")
    return (str(spec.stats_dir(root)),)


def deduplicate(specs: Iterable[RunSpec]) -> list[RunSpec]:
    """Deduplicate runtime variants and reject persistent-address collisions.

    The registry may contain any number of experiments.  The only invariant is
    that two *different* scientific settings may not write to the same
    persistent evaluation address.  Repeating the same scientific setting with
    a different suite label or batch size keeps the first runtime configuration.
    """
    seen_science: set[tuple] = set()
    address_owner: dict[str, tuple] = {}
    out: list[RunSpec] = []
    for spec in specs:
        spec.validate()
        key = scientific_key(spec)
        if key in seen_science:
            continue
        for address in persistent_address_key(spec):
            owner = address_owner.get(address)
            if owner is not None and owner != key:
                raise ValueError(
                    "Configured experiments have different scientific identities but share "
                    f"persistent address {address!r}. First={owner!r}; second={key!r}. "
                    "Change the path-labeling configuration before running either setting."
                )
            address_owner[address] = key
        seen_science.add(key)
        out.append(spec)
    return out


def load_run_specs_json(path: Path) -> list[RunSpec]:
    """Load an exact configured-experiment manifest written by the runner."""
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Configured experiment manifest must be a JSON list: {source}")
    specs: list[RunSpec] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Configured experiment row {index} is not an object in {source}")
        try:
            specs.append(RunSpec(**row))
        except TypeError as exc:
            raise ValueError(f"Invalid configured experiment row {index} in {source}: {exc}") from exc
    return deduplicate(specs)


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
        "bash", str(code_root / "pipeline" / "run_pipeline.sh"),
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


def run_pipeline(
    code_root: Path, spec: RunSpec, *, dry_run: bool = False, skip_if_no_circuit: bool = False
) -> None:
    cmd = pipeline_command(code_root, spec)
    # Execution policy only: keep this out of RunSpec/pipeline_command so the
    # persistent storage contract and scientific identity do not change.
    if skip_if_no_circuit:
        cmd.append("--skip_if_no_circuit")
    print(f"\n=== {spec.identity} ===")
    print(shlex.join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=code_root, check=True)
