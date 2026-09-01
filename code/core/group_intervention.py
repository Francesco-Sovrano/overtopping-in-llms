"""Reusable simultaneous channel-set intervention helpers.

This module contains the model-facing mechanics shared by direct set effects,
group dominance, simultaneous-set conditional validation, and matched random-set
controls.  Statistical definitions remain in their focused analysis scripts.
"""
from __future__ import annotations

import ast
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from core.high_n_singleton_eval import UnitSpec
from core.modeling_and_ablation import LMWrapper, build_ablation_hooks, get_layer_type_and_ids
from core.neuron_intervention import (
    build_prefix_caches_for_examples,
    get_correctness,
    get_correctness_cached_by_prefix_batches,
)
from core.text_and_rules import guess_filetype


LOG_PREFIX = "[group-intervention]"


@dataclass(frozen=True)
class GroupSpec:
    units: tuple[UnitSpec, ...]
    label: str = ""

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(unit.unit_key for unit in self.units))

    @property
    def key(self) -> str:
        return "|".join(self.keys)

    @property
    def size(self) -> int:
        return len(self.units)


def read_table(path: Path) -> pd.DataFrame:
    filetype = guess_filetype(path)
    return pd.read_parquet(path) if filetype == "parquet" else pd.read_csv(path)


def resolve_dataset_path(raw: str | Path, info_dir: Path) -> Path:
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([info_dir / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve dataset path {raw!r}; tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_dataset_info(input_data_dir: Path) -> dict:
    path = Path(input_data_dir) / "dataset_info.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset_info.json not found under {input_data_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def dedupe_units(units: Iterable[UnitSpec], *, sort: bool = True) -> list[UnitSpec]:
    """Deduplicate units by layer/id, optionally preserving first-seen order."""
    seen: set[tuple[str, int]] = set()
    out: list[UnitSpec] = []
    for unit in units:
        key = (str(unit.layer_label), int(unit.neuron_id))
        if key not in seen:
            seen.add(key)
            out.append(unit)
    if sort:
        out.sort(key=lambda unit: (str(unit.layer_label), int(unit.neuron_id)))
    return out


def load_candidate_units(path: Path) -> list[UnitSpec]:
    frame = read_table(path)
    layer_col = "layer_label" if "layer_label" in frame.columns else "layer_key"
    if layer_col not in frame.columns or "neuron_id" not in frame.columns:
        raise ValueError(
            f"{path} must contain neuron_id and layer_label or layer_key; "
            f"found {list(frame.columns)}"
        )
    units = [
        UnitSpec(
            layer_label=str(row[layer_col]),
            neuron_id=int(row["neuron_id"]),
            source="heldout_candidate_set",
            seed_strength=(
                float(row["flip_any_rate"])
                if "flip_any_rate" in row and pd.notna(row["flip_any_rate"])
                else None
            ),
        )
        for row in frame.dropna(subset=[layer_col, "neuron_id"]).to_dict("records")
    ]
    units = dedupe_units(units)
    if not units:
        raise ValueError(f"No candidate units found in {path}")
    return units


def evaluation_frame(
    singleton_scores_path: Path,
    *,
    prompt_col: str,
    target_col: str,
    evaluation_split: str = "test",
) -> pd.DataFrame:
    """Load the rows used for simultaneous-set evaluation.

    ``test`` is the default. ``train`` and ``test`` require an ``is_test``
    column; ``all`` keeps every materialized/evaluated row.
    """
    frame = read_table(singleton_scores_path)
    missing = [column for column in [prompt_col, target_col] if column not in frame.columns]
    if missing:
        raise ValueError(f"{singleton_scores_path} is missing required columns {missing}")
    if "_evaluated" in frame.columns:
        frame = frame.loc[frame["_evaluated"].fillna(False).astype(bool)].copy()
    evaluation_split = str(evaluation_split).strip().lower()
    if evaluation_split in {"test", "train"}:
        if "is_test" not in frame.columns:
            raise ValueError(
                f"--evaluation_split {evaluation_split} requires is_test in scores.csv"
            )
        is_test = frame["is_test"].fillna(False).astype(bool)
        frame = frame.loc[is_test if evaluation_split == "test" else ~is_test].copy()
    elif evaluation_split != "all":
        raise ValueError(f"Unsupported evaluation split {evaluation_split!r}")
    if frame.empty:
        raise ValueError(f"No rows remain for evaluation split={evaluation_split}")
    return frame.reset_index(drop=True)


def _merge_layer_units(*values) -> dict[str, set[int]]:
    merged: dict[str, set[int]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for layer_label, raw_ids in value.items():
            if isinstance(raw_ids, Mapping):
                raw_ids = raw_ids.keys()
            if not isinstance(raw_ids, (list, tuple, set, range)):
                continue
            for raw_id in raw_ids:
                try:
                    merged.setdefault(str(layer_label), set()).add(int(raw_id))
                except Exception:
                    continue
    return merged


def _parse_label_score_key(raw: object) -> tuple[str, int] | None:
    text = str(raw)
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
        try:
            return str(parsed[0]), int(parsed[1])
        except Exception:
            return None
    if ":" in text:
        layer, raw_id = text.rsplit(":", 1)
        try:
            return layer, int(raw_id)
        except Exception:
            return None
    return None


def load_stage5_locus_population(manifest_path: Path) -> dict[str, list[UnitSpec]]:
    """Load stage-5 eligible channels grouped by their native locus label.

    The returned keys (for example ``m5`` or ``a5.h2``) are native computational loci.
    """
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else [payload]
    layer_units: dict[str, set[int]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        meta = entry.get("metadata_topn") or {}
        merged = _merge_layer_units(
            entry.get("units"),
            entry.get("neurons"),
            entry.get("mlp_neurons"),
            entry.get("attention_heads"),
            entry.get("attn_heads"),
            meta.get("units"),
            meta.get("neurons"),
            meta.get("mlp_neurons"),
            meta.get("attention_heads"),
            meta.get("attn_heads"),
        )
        for layer, ids in merged.items():
            layer_units.setdefault(layer, set()).update(ids)
        label_score = meta.get("neuron_label_score") or {}
        if isinstance(label_score, Mapping):
            for label in label_score:
                parsed = _parse_label_score_key(label)
                if parsed is not None:
                    layer_units.setdefault(parsed[0], set()).add(parsed[1])
    out: dict[str, list[UnitSpec]] = {}
    for layer_label, ids in sorted(layer_units.items()):
        out[layer_label] = [
            UnitSpec(layer_label=layer_label, neuron_id=int(unit_id), source="stage5_population")
            for unit_id in sorted(ids)
        ]
    if not out:
        raise ValueError(f"No layer populations could be parsed from {manifest_path}")
    return out


def unit_metadata(unit: UnitSpec) -> dict:
    parsed = get_layer_type_and_ids(str(unit.layer_label))
    if parsed is None:
        return {
            "transformer_layer": None,
            "computational_locus": "unknown",
            "channel_type": "unknown",
            "stage5_locus": str(unit.layer_label),
        }
    layer_type, layer_index, head_index = parsed
    if layer_type == "mlp":
        return {
            "transformer_layer": int(layer_index),
            "computational_locus": "mlp_output",
            "channel_type": "mlp_neuron",
            "stage5_locus": f"m{int(layer_index)}",
        }
    return {
        "transformer_layer": int(layer_index),
        "computational_locus": f"attention_head_{int(head_index)}",
        "channel_type": "attention_dimension",
        "stage5_locus": f"a{int(layer_index)}.h{int(head_index)}",
    }



def matching_stratum_key(unit: UnitSpec) -> tuple[int, str, str]:
    """Return the exact null-matching stratum for one eligible channel."""
    meta = unit_metadata(unit)
    transformer_layer = meta.get("transformer_layer")
    computational_locus = str(meta.get("computational_locus", "unknown"))
    channel_type = str(meta.get("channel_type", "unknown"))
    if transformer_layer is None or computational_locus == "unknown" or channel_type == "unknown":
        raise ValueError(
            f"Cannot resolve exact layer/locus/type metadata for unit {unit.unit_key!r}: {meta}"
        )
    return int(transformer_layer), computational_locus, channel_type


def group_units_by_matching_stratum(
    units: Iterable[UnitSpec],
) -> dict[tuple[int, str, str], list[UnitSpec]]:
    """Group units by transformer layer, computational locus, and channel type."""
    grouped: dict[tuple[int, str, str], list[UnitSpec]] = {}
    for unit in units:
        grouped.setdefault(matching_stratum_key(unit), []).append(unit)
    return {key: dedupe_units(value) for key, value in sorted(grouped.items())}

def group_layer_map(group: GroupSpec) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for unit in group.units:
        output.setdefault(str(unit.layer_label), []).append(int(unit.neuron_id))
    return {layer: sorted(set(ids)) for layer, ids in output.items()}



def evaluation_row_records(
    scores_df: pd.DataFrame,
    *,
    prompt_col: str,
    target_col: str,
) -> list[dict]:
    """Return explicit row identities for cache validation.

    Group-intervention outputs depend on the exact prompt order and baseline
    target values.  Persist those values directly rather than using a digest so
    semantically irrelevant rewrites of ``scores.csv`` do not invalidate the
    cache.
    """
    columns: list[str] = []
    for column in ("_orig_row", "original_idx", "is_test", prompt_col, target_col):
        if column in scores_df.columns and column not in columns:
            columns.append(column)
    records: list[dict] = []
    for row in scores_df[columns].to_dict(orient="records"):
        clean: dict = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[str(key)] = None
            elif isinstance(value, (np.bool_, bool)):
                clean[str(key)] = bool(value)
            elif isinstance(value, (np.integer, int)):
                clean[str(key)] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[str(key)] = float(value)
            else:
                clean[str(key)] = str(value)
        records.append(clean)
    return records


BATCH_CACHE_SCHEMA = "group-intervention-batch-v2"


def _batch_cache_path(cache_dir: Path, start: int, end: int) -> Path:
    """Stable operational batch path; batch size is not scientific identity."""
    return Path(cache_dir) / f"batch_{int(start):08d}_{int(end):08d}.pkl"


def _load_cache_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _payload_outputs(
    payload: dict,
    *,
    expected_context: Mapping,
) -> tuple[int, int, dict[str, np.ndarray]] | None:
    if payload.get("schema") != BATCH_CACHE_SCHEMA:
        return None
    if payload.get("context") != dict(expected_context):
        return None
    try:
        start = int(payload.get("start", -1))
        end = int(payload.get("end", -1))
    except Exception:
        return None
    if start < 0 or end <= start:
        return None
    raw_outputs = payload.get("outputs")
    if not isinstance(raw_outputs, dict):
        return None
    source_len = end - start
    outputs: dict[str, np.ndarray] = {}
    for key, raw in raw_outputs.items():
        try:
            arr = np.asarray(raw, dtype=bool)
        except Exception:
            continue
        if len(arr) == source_len:
            outputs[str(key)] = arr
    return start, end, outputs


def _batch_outputs_for_range(
    cache_dir: Path,
    *,
    start: int,
    end: int,
    expected_context: Mapping,
) -> tuple[dict[str, np.ndarray], int]:
    """Assemble one requested range across compatible canonical cache boundaries."""
    requested_len = int(end) - int(start)
    if requested_len < 0:
        return {}, 0
    values: dict[str, np.ndarray] = {}
    covered: dict[str, np.ndarray] = {}
    conflicted: set[str] = set()
    sources = 0

    for path in sorted(Path(cache_dir).glob("batch_*.pkl")):
        payload = _load_cache_payload(path)
        if payload is None:
            continue
        parsed = _payload_outputs(
            payload,
            expected_context=expected_context,
        )
        if parsed is None:
            continue
        cached_start, cached_end, raw_outputs = parsed
        overlap_start = max(int(start), cached_start)
        overlap_end = min(int(end), cached_end)
        if overlap_end <= overlap_start:
            continue
        source_slice = slice(overlap_start - cached_start, overlap_end - cached_start)
        dest_slice = slice(overlap_start - int(start), overlap_end - int(start))
        used_source = False
        for key, arr in raw_outputs.items():
            if key in conflicted:
                continue
            segment = arr[source_slice]
            if len(segment) != overlap_end - overlap_start:
                continue
            if key not in values:
                values[key] = np.zeros(requested_len, dtype=bool)
                covered[key] = np.zeros(requested_len, dtype=bool)
            prior_mask = covered[key][dest_slice]
            if np.any(prior_mask):
                prior_values = values[key][dest_slice]
                if np.any(prior_values[prior_mask] != segment[prior_mask]):
                    values.pop(key, None)
                    covered.pop(key, None)
                    conflicted.add(key)
                    continue
            values[key][dest_slice] = segment
            covered[key][dest_slice] = True
            used_source = True
        if used_source:
            sources += 1

    complete = {
        key: values[key]
        for key in values
        if key not in conflicted and bool(np.all(covered[key]))
    }
    return complete, sources


def has_complete_group_batch_cache(
    *,
    cache_dir: Path,
    groups: Sequence[GroupSpec],
    n_examples: int,
    batch_size: int,
    cache_context: Mapping,
) -> bool:
    """Return whether all requested groups have compatible cached row coverage."""
    required_keys = {group.key for group in groups if group.size > 0}
    if not required_keys:
        return True
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return False
    for start in range(0, int(n_examples), int(batch_size)):
        end = min(start + int(batch_size), int(n_examples))
        outputs, _ = _batch_outputs_for_range(
            cache_dir,
            start=start,
            end=end,
            expected_context=cache_context,
        )
        if not required_keys.issubset(outputs):
            return False
    return True


def load_complete_group_batch_cache(
    *,
    cache_dir: Path,
    groups: Sequence[GroupSpec],
    n_examples: int,
    batch_size: int,
    cache_context: Mapping,
) -> dict[str, np.ndarray] | None:
    """Load all requested group outputs without loading the model when complete."""
    groups = [group for group in groups if group.size > 0]
    outputs = {group.key: np.zeros(int(n_examples), dtype=bool) for group in groups}
    required = set(outputs)
    for start in range(0, int(n_examples), int(batch_size)):
        end = min(start + int(batch_size), int(n_examples))
        cached, _ = _batch_outputs_for_range(
            cache_dir,
            start=start,
            end=end,
            expected_context=cache_context,
        )
        if not required.issubset(cached):
            return None
        for key in required:
            outputs[key][start:end] = cached[key]
    return outputs


def _write_batch_cache(
    path: Path,
    *,
    start: int,
    end: int,
    outputs: Mapping[str, np.ndarray],
    context: Mapping,
) -> None:
    """Atomically persist one batch with explicit scientific cache metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": BATCH_CACHE_SCHEMA,
        "context": dict(context),
        "start": int(start),
        "end": int(end),
        "outputs": {str(key): np.asarray(value, dtype=bool) for key, value in outputs.items()},
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass



def persist_complete_group_batch_cache(
    *,
    cache_dir: Path,
    outputs: Mapping[str, np.ndarray],
    n_examples: int,
    batch_size: int,
    cache_context: Mapping,
) -> None:
    """Persist already validated complete outputs in the current cache format."""
    cache_dir = Path(cache_dir)
    for start in range(0, int(n_examples), int(batch_size)):
        end = min(start + int(batch_size), int(n_examples))
        batch_outputs = {
            str(key): np.asarray(value, dtype=bool)[start:end]
            for key, value in outputs.items()
        }
        _write_batch_cache(
            _batch_cache_path(cache_dir, start, end),
            start=start,
            end=end,
            outputs=batch_outputs,
            context=cache_context,
        )

def evaluate_groups(
    *,
    model: LMWrapper,
    groups: Sequence[GroupSpec],
    scores_df: pd.DataFrame,
    prompt_col: str,
    is_answer_positive_fn,
    batch_size: int,
    decode_only: bool,
    intervention: str,
    mean_activations,
    max_new_tokens: int,
    cache_dir: Path,
    cache_context: dict,
    force: bool,
) -> dict[str, np.ndarray]:
    """Evaluate simultaneous groups using restartable explicit-metadata caches."""
    groups = [group for group in groups if group.size > 0]
    examples = scores_df.to_dict(orient="records")
    outputs = {group.key: np.zeros(len(examples), dtype=bool) for group in groups}
    cache_dir.mkdir(parents=True, exist_ok=True)

    for start in tqdm(
        range(0, len(examples), int(batch_size)),
        desc=f"{LOG_PREFIX} batches",
        unit="batch",
    ):
        end = min(start + int(batch_size), len(examples))
        batch = examples[start:end]
        batch_cache_path = _batch_cache_path(cache_dir, start, end)
        existing_batch_outputs, cache_sources = _batch_outputs_for_range(
            cache_dir,
            start=start,
            end=end,
            expected_context=cache_context,
        )
        cached = {} if force else existing_batch_outputs

        missing_groups: list[GroupSpec] = []
        for group in groups:
            post = cached.get(group.key)
            if post is None:
                missing_groups.append(group)
            else:
                outputs[group.key][start:end] = post

        reused_groups = len(groups) - len(missing_groups)
        if reused_groups:
            tqdm.write(
                f"{LOG_PREFIX} cache hit rows={start}:{end} "
                f"groups={reused_groups}/{len(groups)}"
            )

        if not missing_groups:
            # Re-materialize compatible cross-boundary cache coverage at the
            # requested operational boundary for faster subsequent restarts.
            if cache_sources and not force:
                _write_batch_cache(
                    batch_cache_path,
                    start=start,
                    end=end,
                    outputs={key: cached[key] for key in cached},
                    context=cache_context,
                )
            continue

        prefix_batches = None
        batch_ranges = None
        if decode_only:
            prefix_batches, batch_ranges = build_prefix_caches_for_examples(
                model,
                batch,
                prompt_col,
                max_new_tokens=int(max_new_tokens),
                batch_size=int(batch_size),
            )

        batch_outputs = dict(existing_batch_outputs)
        for group in tqdm(
            missing_groups,
            desc=f"{LOG_PREFIX} simultaneous groups",
            unit="group",
            leave=False,
        ):
            hooks = build_ablation_hooks(
                group_layer_map(group),
                last_pos_only=bool(decode_only),
                intervention=intervention,
                mean_activations=mean_activations,
                device=model.hooked_model.cfg.device,
            )
            if decode_only:
                _, accuracy = get_correctness_cached_by_prefix_batches(
                    model,
                    batch,
                    is_answer_positive_fn,
                    prefix_batches,
                    batch_ranges,
                    hooks=hooks,
                )
            else:
                _, accuracy = get_correctness(
                    model,
                    batch,
                    is_answer_positive_fn,
                    prompt_col,
                    max_new_tokens=int(max_new_tokens),
                    hooks=hooks,
                    batch_size=int(batch_size),
                )
            post = np.asarray(accuracy, dtype=float) > 0.5
            batch_outputs[group.key] = post.astype(bool)
            outputs[group.key][start:end] = post

        _write_batch_cache(
            batch_cache_path,
            start=start,
            end=end,
            outputs=batch_outputs,
            context=cache_context,
        )
        try:
            model.cleanup_after_generate()
        except Exception:
            pass
    return outputs

def simultaneous_effect(baseline: np.ndarray, post: np.ndarray) -> dict:
    """Return aggregate and direction-specific simultaneous set effects.

    ``effect`` is the historical undirected flip probability ``P(B_J != B)``.
    The directional effects condition on the pre-intervention binary state:

    - ``effect_0to1 = P(B_J=1 | B=0)``
    - ``effect_1to0 = P(B_J=0 | B=1)``

    """
    baseline = np.asarray(baseline, dtype=bool)
    post = np.asarray(post, dtype=bool)
    if len(baseline) != len(post):
        raise ValueError("Baseline and post-intervention arrays differ in length")
    flips = post != baseline
    n = int(len(baseline))
    output = {
        "count": int(flips.sum()),
        "denominator": n,
        "effect": float(flips.mean()) if n else float("nan"),
        "status": "ok" if n else "undefined_zero_denominator",
    }

    direction_specs = {
        "0to1": (~baseline) & post,
        "1to0": baseline & (~post),
    }
    for direction, events in direction_specs.items():
        source_value = direction[0] == "1"
        eligible = baseline == source_value
        denominator = int(eligible.sum())
        count = int(events.sum())
        value = float(count / denominator) if denominator else float("nan")
        status = "ok" if denominator else "undefined_zero_denominator"
        output[f"effect_{direction}"] = value
        output[f"effect_{direction}_count"] = count
        output[f"effect_{direction}_denominator"] = denominator
        output[f"effect_{direction}_status"] = status

    return output
