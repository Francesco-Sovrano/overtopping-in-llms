"""Reusable simultaneous channel-set intervention helpers.

This module contains the model-facing mechanics shared by direct set effects,
group dominance, simultaneous-set conditional validation, and matched random-set
controls.  Statistical definitions remain in their focused analysis scripts.
"""
from __future__ import annotations

import ast
import json
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
from core.group_eval_cache_store import get_group_eval_cache_store, semantic_context_id
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



def _cache_scalar(value):
    """Convert dataframe scalars to stable, pickle/JSON-friendly values."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def evaluation_row_records(
    scores_df: pd.DataFrame,
    *,
    prompt_col: str,
    target_col: str,
) -> list[dict]:
    """Return label-independent row identities for group-output cache validation.

    The expensive simultaneous-intervention outputs are tied to the ordered
    evaluation examples, not to whatever column labels happen to name the prompt
    and baseline target today.  Store the semantic roles under stable keys so a
    harmless ``prompt_col``/``target_col`` rename does not invalidate the cache.

    ``source_row_id`` is retained for diagnostics only; cache compatibility below
    is intentionally based on the ordered prompt/target values.
    """
    row_id_col = next(
        (column for column in ("_orig_row", "original_idx") if column in scores_df.columns),
        None,
    )
    records: list[dict] = []
    for row in scores_df.to_dict(orient="records"):
        record = {
            "prompt_value": _cache_scalar(row.get(prompt_col)),
            "target_value": _cache_scalar(row.get(target_col)),
        }
        if row_id_col is not None:
            record["source_row_id"] = _cache_scalar(row.get(row_id_col))
        records.append(record)
    return records



# Scientific identity of the expensive model outputs.  This is deliberately
# separate from reporting/configuration schemas.  Never add cosmetic labels,
# paths, output filenames, batch size, or requested analysis-group names here.
GROUP_CACHE_SEMANTIC_IDENTITY_VERSION = 1

# Stage 8 repeatedly passes the same in-memory context mapping while checking
# hundreds/thousands of operational batches. Hash the large row identity once.
_GROUP_CONTEXT_ID_CACHE: dict[int, tuple[Mapping, str]] = {}


def _legacy_or_current_row_identity(context: Mapping) -> list[tuple[object, object]] | None:
    """Canonicalize old/new evaluation-row metadata without depending on labels.

    Legacy v2 caches stored rows using the then-current prompt/target column
    names.  Current caches store ``prompt_value``/``target_value``.  Both map to
    the same ordered semantic identity, allowing column-label migrations without
    recomputing model interventions.
    """
    rows = context.get("evaluation_rows")
    if not isinstance(rows, list):
        return None
    prompt_col = str(context.get("prompt_col") or "")
    target_col = str(context.get("target_col") or "")
    canonical: list[tuple[object, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        if "prompt_value" in row:
            prompt_value = row.get("prompt_value")
        elif prompt_col and prompt_col in row:
            prompt_value = row.get(prompt_col)
        else:
            # Conservative legacy fallback for the historical default name.
            prompt_value = row.get("prompt") if "prompt" in row else None

        if "target_value" in row:
            target_value = row.get("target_value")
        elif target_col and target_col in row:
            target_value = row.get(target_col)
        else:
            # Conservative legacy fallback for the historical default name.
            target_value = row.get("is_correct") if "is_correct" in row else None
        canonical.append((_cache_scalar(prompt_value), _cache_scalar(target_value)))
    return canonical


def group_cache_semantic_identity(context: Mapping) -> dict:
    """Return only fields capable of changing cached group-intervention outputs.

    Important exclusions:
      * context/reporting schema labels;
      * prompt/target *column names*;
      * evaluation split labels once the exact ordered rows are materialized;
      * filesystem paths, batch size, CMC/preemption flags, and group labels.

    The model value itself remains strict.  ``model`` and ``ai_model`` are
    accepted as metadata-key aliases, but a different model identifier remains
    an invalidation.
    """
    model = context.get("model", context.get("ai_model"))
    return {
        "identity_version": GROUP_CACHE_SEMANTIC_IDENTITY_VERSION,
        "model": None if model is None else str(model),
        "task_module": None if context.get("task_module") is None else str(context.get("task_module")),
        "evaluation_rows": _legacy_or_current_row_identity(context),
        "decode_only": bool(context.get("decode_only")),
        "intervention": None if context.get("intervention") is None else str(context.get("intervention")),
        "replacement_reference_prompts": [
            str(value) for value in (context.get("replacement_reference_prompts") or [])
        ],
        "max_new_tokens": (
            None if context.get("max_new_tokens") is None else int(context.get("max_new_tokens"))
        ),
    }


def group_cache_context_id(context: Mapping) -> str:
    """Return a stable SHA-256 id for the expensive-output semantic identity."""
    key = id(context)
    cached = _GROUP_CONTEXT_ID_CACHE.get(key)
    if cached is not None and cached[0] is context:
        return cached[1]
    value = semantic_context_id(group_cache_semantic_identity(context))
    _GROUP_CONTEXT_ID_CACHE[key] = (context, value)
    return value


def group_cache_context_differences(stored: Mapping, expected: Mapping) -> dict[str, tuple[object, object]]:
    """Return semantic cache differences as ``field -> (stored, expected)``."""
    lhs = group_cache_semantic_identity(stored)
    rhs = group_cache_semantic_identity(expected)
    return {key: (lhs.get(key), rhs.get(key)) for key in sorted(set(lhs) | set(rhs)) if lhs.get(key) != rhs.get(key)}



def _assemble_range_outputs(
    records: Sequence[tuple[int, int, str, np.ndarray]],
    *,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    requested_len = int(end) - int(start)
    values: dict[str, np.ndarray] = {}
    covered: dict[str, np.ndarray] = {}
    conflicted: set[str] = set()

    for cached_start, cached_end, key, arr in records:
        overlap_start = max(int(start), int(cached_start))
        overlap_end = min(int(end), int(cached_end))
        if overlap_end <= overlap_start or key in conflicted:
            continue
        source_slice = slice(overlap_start - int(cached_start), overlap_end - int(cached_start))
        dest_slice = slice(overlap_start - int(start), overlap_end - int(start))
        segment = np.asarray(arr, dtype=bool)[source_slice]
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

    return {
        key: values[key]
        for key in values
        if key not in conflicted and bool(np.all(covered[key]))
    }


def _batch_outputs_for_range(
    cache_dir: Path,
    *,
    start: int,
    end: int,
    expected_context: Mapping,
) -> tuple[dict[str, np.ndarray], int]:
    """Assemble one requested range from indexed SQLite storage only.

    Legacy ``batch_*.pkl`` shards are intentionally not read here. They must be
    converted explicitly with ``migrate_group_eval_caches.py`` before Stage 8
    can reuse them.
    """
    if int(end) < int(start):
        return {}, 0

    store = get_group_eval_cache_store(cache_dir)
    context_id = group_cache_context_id(expected_context)
    sqlite_records = store.overlapping_ranges(
        context_id=context_id,
        start=int(start),
        end=int(end),
    )
    source_ranges = {(s, e) for s, e, _, _ in sqlite_records}
    return (
        _assemble_range_outputs(sqlite_records, start=start, end=end),
        len(source_ranges),
    )


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


def _summarize_cache_value(value):
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "head": value[:2],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "length": len(value),
            "head": list(value[:2]),
        }
    return value


def group_batch_cache_status(
    *,
    cache_dir: Path,
    groups: Sequence[GroupSpec],
    n_examples: int,
    batch_size: int,
    cache_context: Mapping,
) -> dict:
    """Inspect SQLite coverage and semantic compatibility without a model.

    Legacy pickle shards are counted only so callers can issue a migration
    instruction. They are never opened, unpickled, or used as runtime cache
    data.
    """
    cache_dir = Path(cache_dir)
    legacy_pickle_files = 0
    if cache_dir.is_dir():
        legacy_pickle_files = sum(
            1
            for path in cache_dir.iterdir()
            if path.is_file() and path.name.startswith("batch_") and path.suffix == ".pkl"
        )

    store = get_group_eval_cache_store(cache_dir)
    expected_context_id = group_cache_context_id(cache_context)

    semantic_mismatch_fields: set[str] = set()
    first_semantic_mismatch: dict | None = None
    sqlite_ranges = store.distinct_ranges()
    compatible_files = sum(1 for cid, _, _ in sqlite_ranges if cid == expected_context_id)
    incompatible_files = sum(1 for cid, _, _ in sqlite_ranges if cid != expected_context_id)
    incompatible_context_ids = sorted({
        cid for cid, _, _ in sqlite_ranges if cid != expected_context_id
    })
    for context_id in incompatible_context_ids:
        stored_context = store.load_context(context_id)
        if not isinstance(stored_context, Mapping):
            continue
        diffs = group_cache_context_differences(stored_context, cache_context)
        semantic_mismatch_fields.update(diffs)
        if diffs and first_semantic_mismatch is None:
            first_semantic_mismatch = {
                "file": str(store.db_path),
                "context_id": context_id,
                "differences": {
                    key: {
                        "stored": _summarize_cache_value(values[0]),
                        "expected": _summarize_cache_value(values[1]),
                    }
                    for key, values in diffs.items()
                },
            }

    required = {group.key for group in groups if group.size > 0}
    first_missing_range = None
    missing_group_keys: set[str] = set()
    for start in range(0, int(n_examples), int(batch_size)):
        end = min(start + int(batch_size), int(n_examples))
        cached, _ = _batch_outputs_for_range(
            cache_dir,
            start=start,
            end=end,
            expected_context=cache_context,
        )
        missing = required - set(cached)
        if missing:
            missing_group_keys.update(missing)
            if first_missing_range is None:
                first_missing_range = {
                    "start": int(start),
                    "end": int(end),
                    "missing_group_keys": sorted(missing)[:20],
                    "n_missing_groups": len(missing),
                }

    complete = bool(sqlite_ranges) and first_missing_range is None
    return {
        "cache_dir": str(cache_dir),
        "sqlite_db": str(store.db_path),
        "storage": "sqlite",
        "batch_files": len(sqlite_ranges),
        "sqlite_ranges": len(sqlite_ranges),
        "legacy_pickle_files": legacy_pickle_files,
        "legacy_pickle_runtime_reads": False,
        "migration_required": bool(legacy_pickle_files and not complete),
        "compatible_files": compatible_files,
        "incompatible_files": incompatible_files,
        "unreadable_files": 0,
        "semantic_mismatch_fields": sorted(semantic_mismatch_fields),
        "first_semantic_mismatch": first_semantic_mismatch,
        "required_groups": len(required),
        "missing_group_count": len(missing_group_keys),
        "missing_group_keys": sorted(missing_group_keys)[:50],
        "first_missing_range": first_missing_range,
        "complete": complete,
    }


def _write_batch_cache(
    cache_dir: Path,
    *,
    start: int,
    end: int,
    outputs: Mapping[str, np.ndarray],
    context: Mapping,
) -> None:
    """Persist one operational batch into normalized indexed SQLite storage."""
    cache_dir = Path(cache_dir)
    store = get_group_eval_cache_store(cache_dir)
    store.put_range(
        context_id=group_cache_context_id(context),
        context=context,
        start=int(start),
        end=int(end),
        outputs=outputs,
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
            # Normalize compatible cross-boundary SQLite coverage at the
            # requested operational boundary for faster subsequent restarts.
            if cache_sources and not force:
                _write_batch_cache(
                    cache_dir,
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
