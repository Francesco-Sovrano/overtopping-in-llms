"""Resolve ordinary-task agonists from the virgin model.

These coordinates are *not* used for developmental checkpoint discovery.  They
are used only by optional follow-up analyses, such as measuring overlap between
newly discovered poisoning circuits and the virgin task circuit, or protecting
selected direct-channel-write LoRA rows during a separate poisoning run.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path



def _count(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    block = payload.get("non_catastrophic_agonists", {}) if isinstance(payload, dict) else {}
    return len(block) if isinstance(block, dict) else 0


def _source_phase(path: Path) -> str:
    return "output_only" if "decode_only" in str(path).replace("\\", "/") else "input_output"


def _configured_tau_label() -> str:
    """Return the shared CHA tau in the same compact form used in paths."""
    raw = os.environ.get("CHA_TAU", os.environ.get("POISONING_CHA_TAU", "0.3")).strip()
    try:
        return f"{float(raw):g}"
    except ValueError:
        return raw


def _candidate_score(path: Path, *, phase: str, intervention: str) -> tuple:
    text = str(path).replace("\\", "/")
    tau_label = _configured_tau_label()
    # The generic pipeline omits the tau suffix at its 0.2 default.  Parse the
    # anchoring directory rather than preferring a historical hard-coded tau.
    tau_match = False
    m = re.search(r"agonist_neurons-fast-random_anchor(?:-tau([^/]+))?", text)
    if m:
        actual_tau = m.group(1) or "0.2"
        try:
            actual_tau = f"{float(actual_tau):g}"
        except ValueError:
            pass
        tau_match = actual_tau == tau_label
    return (
        int(_source_phase(path) == phase),
        int(f"eval_{intervention}" in text),
        int("spectral_split-M200000" in text),
        int(tau_match),
        int("positive_baseline" in text),
        _count(path),
        -len(text),
    )


def normal_model_root(
    project_root: str | Path, *, task: str, task_data_dir: str, model_name: str
) -> Path:
    if not str(task_data_dir).strip():
        raise ValueError("task_data_dir must be provided by the task definition")
    task_dir = str(task_data_dir).strip()
    override = os.environ.get(f"POISONING_{task.upper()}_VIRGIN_MODEL_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path(project_root).resolve() / "data" / task_dir / Path(model_name)).resolve()


def resolve_virgin_agonists_path(
    project_root: str | Path,
    *,
    task: str,
    model_name: str,
    task_data_dir: str,
    phase: str,
    intervention: str = "mean-donor",
    require_phase_match: bool = True,
) -> Path:
    explicit = os.environ.get(f"POISONING_{task.upper()}_VIRGIN_AGONISTS_PATH", "").strip()
    if not explicit:
        explicit = os.environ.get("POISONING_VIRGIN_AGONISTS_PATH", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file() and path.name == "neuron_buckets.json":
            path = path.parent
        bucket = path / "neuron_buckets.json"
        if not bucket.is_file():
            raise FileNotFoundError(f"Virgin agonist source has no neuron_buckets.json: {path}")
        if require_phase_match and _source_phase(bucket) != phase:
            raise RuntimeError(
                f"Virgin agonists were discovered under {_source_phase(bucket)}, requested phase is {phase}. "
                "Use a phase-matched ordinary-task run for confirmatory protection experiments."
            )
        return path

    root = normal_model_root(project_root, task=task, task_data_dir=task_data_dir, model_name=model_name)
    candidates = [
        p for p in root.rglob("neuron_buckets.json")
        if "neural_circuit_discovery_results" in str(p)
        and "eap_ig_inputs" in str(p)
        and _count(p) > 0
    ]
    phase_matched = [p for p in candidates if _source_phase(p) == phase]
    if require_phase_match:
        candidates = phase_matched
    if not candidates:
        raise FileNotFoundError(
            f"No {'phase-matched ' if require_phase_match else ''}non-empty virgin agonist bucket under {root}. "
            f"Set POISONING_{task.upper()}_VIRGIN_AGONISTS_PATH explicitly."
        )
    best = sorted(candidates, key=lambda p: (_candidate_score(p, phase=phase, intervention=intervention), str(p)), reverse=True)[0]
    print(f"[virgin-agonists] source={best.parent} n={_count(best)} phase={_source_phase(best)}", flush=True)
    return best.parent


def read_agonist_coordinates(path: str | Path) -> list[tuple[str, int]]:
    path = Path(path)
    bucket = path if path.is_file() else path / "neuron_buckets.json"
    payload = json.loads(bucket.read_text(encoding="utf-8"))
    block = payload.get("non_catastrophic_agonists", {}) if isinstance(payload, dict) else {}
    coords: list[tuple[str, int]] = []
    if isinstance(block, dict):
        for key, entry in block.items():
            rec = entry.get("last_record", {}) if isinstance(entry, dict) else {}
            layer = rec.get("layer_label")
            neuron_id = rec.get("neuron_id")
            if layer is None or neuron_id is None:
                try:
                    layer, raw_id = str(key).rsplit(":", 1); neuron_id = int(raw_id)
                except Exception:
                    continue
            coords.append((str(layer), int(neuron_id)))
    # Membership, not ranking, is the contract here.
    return sorted(set(coords))
