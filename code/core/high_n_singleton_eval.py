"""Shared high-N singleton ablation evaluator.

This is the reusable part that Script 12 needs from Script 7: evaluate many
examples for selected singleton units, cache the per-unit flip labels, and
return a scores-like dataframe plus flip statistics.  It calls the same project
helpers used by Scripts 6/7 instead of shelling out to Script 7 and waiting for
an expected scores.csv side effect.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from core.ablation_cache_store import (
    ablation_cache_exists,
    flush_all_ablation_cache_stores,
    list_ablation_cache_paths,
    load_ablation_cache,
    save_ablation_cache,
)
from core.feature_representation import safe_features_fillna
from core.modeling_and_ablation import (
    LMWrapper,
    build_ablation_hooks,
    precompute_mean_activations,
)
from core.neuron_intervention import (
    build_prefix_caches_for_examples,
    get_correctness,
    get_correctness_cached_by_prefix_batches,
)
from core.text_and_rules import guess_filetype
from core.heldout_set_metrics import safe_layer_label

LOG_PREFIX = "[threshold-events]"


@dataclass(frozen=True)
class UnitSpec:
    layer_label: str
    neuron_id: int
    source: str = "script6"
    circuit_id: int | None = None
    circuit_label: str | None = None
    seed_strength: float | None = None

    @property
    def layer_key(self) -> str:
        return safe_layer_label(self.layer_label)

    @property
    def unit_key(self) -> str:
        return f"{self.layer_label}:{int(self.neuron_id)}"


def _read_table(path: Path) -> pd.DataFrame:
    ftype = guess_filetype(path)
    return pd.read_parquet(path) if ftype == "parquet" else pd.read_csv(path)


def load_scores_for_baseline(
    *,
    scores_path: Path,
    target_col: str,
    baseline_subset: str,
    task_targets: Iterable[str],
    split: str = "train",
) -> pd.DataFrame:
    """Load a baseline-conditioned evaluation frame from a declared data split.

    ``split="train"`` preserves the historical threshold-event behavior.
    ``split="test"`` is the strict post-selection evaluation path: it requires
    an explicit ``is_test`` column and never falls back to training rows.
    ``split="all"`` is available for descriptive diagnostics only.
    """
    split = str(split).strip().lower()
    if split not in {"train", "test", "all"}:
        raise ValueError(f"Unsupported split {split!r}; expected train, test, or all.")
    # ``scores.csv`` can already be very wide (hundreds of feature / flip
    # columns).  Taking one consolidating copy before adding bookkeeping
    # columns avoids pandas' ``DataFrame is highly fragmented`` warning and
    # materially reduces the cost of later column access.
    scores_df = _read_table(Path(scores_path)).copy()
    original_idx = pd.Series(np.arange(len(scores_df)), index=scores_df.index, name="original_idx")
    scores_df = pd.concat([scores_df.drop(columns=["original_idx"], errors="ignore"), original_idx], axis=1)
    scores_df = safe_features_fillna(scores_df, fill_number=0, fill_bool=False, cols_not_to_fill=list(task_targets))
    if "is_test" in scores_df.columns:
        is_test = scores_df["is_test"].fillna(False).astype(bool)
        if split == "train":
            scores_df = scores_df.loc[~is_test].reset_index(drop=True)
        elif split == "test":
            scores_df = scores_df.loc[is_test].reset_index(drop=True)
    elif split == "test":
        raise ValueError(
            f"Strict test evaluation requested, but {scores_path} has no is_test column."
        )
    if baseline_subset == "positive":
        scores_df = scores_df.loc[scores_df[target_col] == True]
    elif baseline_subset == "negative":
        scores_df = scores_df.loc[scores_df[target_col] == False]
    scores_df = scores_df.reset_index(drop=True)
    return scores_df


def _cache_key(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def build_mean_prompt_pool(scores_df: pd.DataFrame, *, prompt_col: str, target_col: str, n_points: int, seed: int) -> list[str]:
    if len(scores_df) == 0 or n_points <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    df = scores_df.copy()
    if target_col in df.columns:
        pos = df.loc[df[target_col] == True]
        neg = df.loc[df[target_col] == False]
        pieces = []
        half = max(1, int(n_points) // 2)
        if not pos.empty:
            pieces.append(pos.sample(n=min(half, len(pos)), random_state=int(rng.integers(0, 2**31 - 1))))
        if not neg.empty:
            pieces.append(neg.sample(n=min(int(n_points) - sum(len(p) for p in pieces), len(neg)), random_state=int(rng.integers(0, 2**31 - 1))))
        if pieces:
            pool = pd.concat(pieces, ignore_index=False)
            if len(pool) < min(int(n_points), len(df)):
                remaining = df.drop(index=pool.index, errors="ignore")
                if not remaining.empty:
                    extra = remaining.sample(n=min(int(n_points) - len(pool), len(remaining)), random_state=int(rng.integers(0, 2**31 - 1)))
                    pool = pd.concat([pool, extra], ignore_index=False)
            return pool[prompt_col].astype(str).tolist()
    return df.sample(n=min(int(n_points), len(df)), random_state=int(seed))[prompt_col].astype(str).tolist()


def precompute_replacements_for_units(*, model: LMWrapper, units: list[UnitSpec], scores_df_for_mean: pd.DataFrame,
                                      prompt_col: str, target_col: str, intervention: str,
                                      points_to_use: int, batch_size: int, seed: int):
    if intervention not in ("mean", "mean-donor", "mean-positional", "mean-donor-positional"):
        return None
    layer_to_neurons: dict[str, set[int]] = {}
    for u in units:
        layer_to_neurons.setdefault(str(u.layer_label), set()).add(int(u.neuron_id))
    layer_to_neurons = {k: sorted(v) for k, v in layer_to_neurons.items() if v}
    if not layer_to_neurons:
        return None
    pool = build_mean_prompt_pool(
        scores_df_for_mean,
        prompt_col=prompt_col,
        target_col=target_col,
        n_points=int(points_to_use),
        seed=int(seed),
    )
    if not pool:
        return None
    return precompute_mean_activations(
        model=model,
        all_prompts=pool,
        layer_to_neurons=layer_to_neurons,
        n_points=min(int(points_to_use), len(pool)),
        batch_size=int(batch_size),
        intervention=intervention,
        device=model.hooked_model.cfg.device,
    )


def _flip_cache_path(cache_dir: Path, unit: UnitSpec, batch_start: int, cache_key: str) -> Path:
    layer_key = unit.layer_key
    return Path(cache_dir) / "ablation_cache" / f"{layer_key}_{int(unit.neuron_id)}_{int(batch_start)}_{cache_key}.pkl"


def _load_cached_bool_array_raw(path: Path) -> np.ndarray | None:
    try:
        value = load_ablation_cache(path)
        if value is None:
            return None
        arr = np.asarray(value).astype(bool)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return arr


def _load_cached_bool_array(path: Path, *, expected_len: int) -> np.ndarray | None:
    arr = _load_cached_bool_array_raw(path)
    if arr is None or len(arr) != int(expected_len):
        return None
    return arr


def _cache_payload_for_range(
    eval_indices: np.ndarray,
    start: int,
    end: int,
    *,
    baseline_subset: str,
    target_col: str,
    prompt_col: str,
    decode_only: bool,
    intervention: str,
    max_new_tokens: int,
) -> dict:
    return {
        "eval_indices": [int(i) for i in eval_indices[int(start):int(end)].tolist()],
        "baseline_subset": str(baseline_subset),
        "target_col": str(target_col),
        "prompt_col": str(prompt_col),
        "decode_only": bool(decode_only),
        "intervention": str(intervention),
        "max_new_tokens": int(max_new_tokens),
    }


def _parse_flip_cache_file(path: Path, unit: UnitSpec) -> tuple[int, str] | None:
    prefix = f"{unit.layer_key}_{int(unit.neuron_id)}_"
    stem = Path(path).stem
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix):]
    if "_" not in tail:
        return None
    start_text, cache_key = tail.split("_", 1)
    try:
        start = int(start_text)
    except ValueError:
        return None
    if start < 0 or not cache_key:
        return None
    return start, cache_key


def _cache_file_matches_range(
    *,
    file_start: int,
    file_key: str,
    arr_len: int,
    eval_indices: np.ndarray,
    baseline_subset: str,
    target_col: str,
    prompt_col: str,
    decode_only: bool,
    intervention: str,
    max_new_tokens: int,
) -> bool:
    file_end = int(file_start) + int(arr_len)
    if arr_len <= 0 or file_start < 0 or file_end > len(eval_indices):
        return False
    payload = _cache_payload_for_range(
        eval_indices, file_start, file_end,
        baseline_subset=baseline_subset, target_col=target_col, prompt_col=prompt_col,
        decode_only=decode_only, intervention=intervention, max_new_tokens=max_new_tokens,
    )
    return file_key == _cache_key(payload)


def _load_stitched_cached_bool_array(
    *,
    cache_paths: list[Path],
    loaded_arrays: dict[Path, np.ndarray | None],
    unit: UnitSpec,
    desired_start: int,
    desired_end: int,
    eval_indices: np.ndarray,
    baseline_subset: str,
    target_col: str,
    prompt_col: str,
    decode_only: bool,
    intervention: str,
    max_new_tokens: int,
) -> np.ndarray | None:
    """Assemble one requested batch from canonical cache shards of any size."""
    desired_start = int(desired_start)
    desired_end = int(desired_end)
    n = desired_end - desired_start
    if n <= 0:
        return np.empty(0, dtype=bool)
    result = np.empty(n, dtype=bool)
    covered = np.zeros(n, dtype=bool)

    for path in cache_paths:
        parsed = _parse_flip_cache_file(path, unit)
        if parsed is None:
            continue
        file_start, file_key = parsed
        if file_start >= desired_end:
            continue
        arr = loaded_arrays.get(path)
        if path not in loaded_arrays:
            arr = _load_cached_bool_array_raw(path)
            loaded_arrays[path] = arr
        if arr is None:
            continue
        file_end = file_start + len(arr)
        if file_end <= desired_start:
            continue
        if not _cache_file_matches_range(
            file_start=file_start, file_key=file_key, arr_len=len(arr),
            eval_indices=eval_indices, baseline_subset=baseline_subset,
            target_col=target_col, prompt_col=prompt_col, decode_only=decode_only,
            intervention=intervention, max_new_tokens=max_new_tokens,
        ):
            continue
        overlap_start = max(desired_start, file_start)
        overlap_end = min(desired_end, file_end)
        dst = slice(overlap_start - desired_start, overlap_end - desired_start)
        src = slice(overlap_start - file_start, overlap_end - file_start)
        values = arr[src]
        if covered[dst].any():
            existing = result[dst][covered[dst]]
            incoming = values[covered[dst]]
            if not np.array_equal(existing, incoming):
                # Conflicting scientifically compatible-looking caches are safer
                # to recompute than to choose arbitrarily.
                return None
        result[dst] = values
        covered[dst] = True
        if covered.all():
            return result
    return None


def evaluate_singleton_flips_high_n(*, model: LMWrapper, units: list[UnitSpec], scores_df: pd.DataFrame,
                                    examples: list[dict], eval_indices: np.ndarray, prompt_col: str,
                                    is_answer_positive_fn: Callable, target_col: str, baseline_subset: str,
                                    batch_size: int, decode_only: bool, intervention: str,
                                    mean_activations, max_new_tokens: int, cache_dir: Path,
                                    force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate singleton ablations for selected units on high-N examples.

    Returns a scores-like dataframe over eval_indices with flip columns, and a
    flip_stats dataframe with per-unit high-N rates.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_indices = np.asarray(eval_indices, dtype=int)
    eval_examples = [examples[int(i)] for i in eval_indices]
    scores_out = scores_df.iloc[eval_indices].copy().reset_index(drop=True)
    scores_out["_orig_row"] = eval_indices
    scores_out["_evaluated"] = True

    baseline_eval = (pd.to_numeric(scores_out[target_col], errors="coerce").to_numpy() > 0.5)
    n_batches = (len(eval_examples) + int(batch_size) - 1) // int(batch_size)

    missing_flip_columns: dict[str, pd.Series] = {}
    for u in tqdm(units, desc=f"{LOG_PREFIX} prepare flip columns", unit="unit", leave=False):
        for col in (
            f"flip_{u.layer_key}_{int(u.neuron_id)}",
            f"flip_c2i_{u.layer_key}_{int(u.neuron_id)}",
            f"flip_i2c_{u.layer_key}_{int(u.neuron_id)}",
        ):
            if col not in scores_out.columns and col not in missing_flip_columns:
                missing_flip_columns[col] = pd.Series(
                    pd.array([pd.NA] * len(scores_out), dtype="boolean"),
                    index=scores_out.index,
                    name=col,
                )
    if missing_flip_columns:
        scores_out = pd.concat(
            [scores_out, pd.DataFrame(missing_flip_columns, index=scores_out.index)],
            axis=1,
        )

    ablation_cache_dir = cache_dir / "ablation_cache"
    existing_cache_paths_by_unit: dict[str, list[Path]] = {}
    loaded_cache_arrays: dict[Path, np.ndarray | None] = {}
    if not force and ablation_cache_dir.exists():
        for u in units:
            prefix = f"{u.layer_key}_{int(u.neuron_id)}_"
            existing_cache_paths_by_unit[u.unit_key] = list_ablation_cache_paths(
                ablation_cache_dir, prefix=prefix
            )
    else:
        existing_cache_paths_by_unit = {u.unit_key: [] for u in units}

    for start in tqdm(range(0, len(eval_examples), int(batch_size)), total=n_batches, desc=f"{LOG_PREFIX} high-N batches", unit="batch"):
        end = min(start + int(batch_size), len(eval_examples))
        batch_examples = eval_examples[start:end]
        batch_baseline = baseline_eval[start:end]
        # Batch size is an execution detail, not part of the intervention's
        # scientific identity. Cache keys encode the exact requested row identities.
        batch_cache_payload = _cache_payload_for_range(
            eval_indices, start, end,
            baseline_subset=baseline_subset, target_col=target_col, prompt_col=prompt_col,
            decode_only=decode_only, intervention=intervention, max_new_tokens=max_new_tokens,
        )
        batch_cache_key = _cache_key(batch_cache_payload)

        cached_by_unit: dict[str, np.ndarray] = {}
        cache_paths: dict[str, Path] = {}
        for u in units:
            cpath = _flip_cache_path(cache_dir, u, start, batch_cache_key)
            cache_paths[u.unit_key] = cpath
            arr = None if force else _load_cached_bool_array(cpath, expected_len=end - start)
            if arr is None and not force:
                arr = _load_stitched_cached_bool_array(
                    cache_paths=existing_cache_paths_by_unit.get(u.unit_key, []),
                    loaded_arrays=loaded_cache_arrays, unit=u,
                    desired_start=start, desired_end=end, eval_indices=eval_indices,
                    baseline_subset=baseline_subset, target_col=target_col,
                    prompt_col=prompt_col, decode_only=decode_only,
                    intervention=intervention, max_new_tokens=max_new_tokens,
                )
            if arr is not None:
                # Materialize cross-boundary canonical coverage at the current
                # operational boundary so subsequent restarts are O(1) probes.
                if not ablation_cache_exists(cpath):
                    save_ablation_cache(cpath, arr)
                    existing_cache_paths_by_unit.setdefault(u.unit_key, []).append(cpath)
                    loaded_cache_arrays[cpath] = arr
                cached_by_unit[u.unit_key] = arr

        if cached_by_unit:
            print(
                f"{LOG_PREFIX} cache hit rows={start}:{end} "
                f"units={len(cached_by_unit)}/{len(units)}"
            )

        # Prefix construction itself can be expensive for thousands of decode-
        # only examples.  Do it only when at least one unit actually misses the
        # cache; a fully cached batch now performs no model work at all.
        prefix_batches = None
        batch_ranges = None
        missing_units = [u for u in units if u.unit_key not in cached_by_unit]
        if decode_only and missing_units:
            prefix_batches, batch_ranges = build_prefix_caches_for_examples(
                model,
                batch_examples,
                prompt_col,
                max_new_tokens=int(max_new_tokens),
                batch_size=int(batch_size),
            )
        for u in tqdm(units, desc=f"{LOG_PREFIX} units", unit="unit", leave=False):
            cpath = cache_paths[u.unit_key]
            arr = cached_by_unit.get(u.unit_key)
            if arr is None:
                hooks = build_ablation_hooks(
                    {u.layer_label: [int(u.neuron_id)]},
                    last_pos_only=bool(decode_only),
                    intervention=intervention,
                    mean_activations=mean_activations,
                    device=model.hooked_model.cfg.device,
                )
                if decode_only:
                    _, acc = get_correctness_cached_by_prefix_batches(
                        model,
                        batch_examples,
                        is_answer_positive_fn,
                        prefix_batches,
                        batch_ranges,
                        hooks=hooks,
                    )
                else:
                    _, acc = get_correctness(
                        model,
                        batch_examples,
                        is_answer_positive_fn,
                        prompt_col,
                        max_new_tokens=int(max_new_tokens),
                        hooks=hooks,
                        batch_size=int(batch_size),
                    )
                arr = (np.asarray(acc, dtype=float) > 0.5).astype(bool)
                save_ablation_cache(cpath, arr)
                existing_cache_paths_by_unit.setdefault(u.unit_key, []).append(cpath)
                loaded_cache_arrays[cpath] = arr
            flip_any = arr != batch_baseline
            flip_c2i = (batch_baseline == True) & (arr == False)
            flip_i2c = (batch_baseline == False) & (arr == True)
            batch_slice = slice(start, end)
            scores_out.iloc[batch_slice, scores_out.columns.get_loc(f"flip_{u.layer_key}_{int(u.neuron_id)}")] = flip_any.astype(bool)
            scores_out.iloc[batch_slice, scores_out.columns.get_loc(f"flip_c2i_{u.layer_key}_{int(u.neuron_id)}")] = flip_c2i.astype(bool)
            scores_out.iloc[batch_slice, scores_out.columns.get_loc(f"flip_i2c_{u.layer_key}_{int(u.neuron_id)}")] = flip_i2c.astype(bool)
        try:
            model.cleanup_after_generate()
        except Exception:
            pass

    stats_rows = []
    for u in tqdm(units, desc=f"{LOG_PREFIX} summarize flip stats", unit="unit", leave=False):
        col_any = f"flip_{u.layer_key}_{int(u.neuron_id)}"
        col_c2i = f"flip_c2i_{u.layer_key}_{int(u.neuron_id)}"
        col_i2c = f"flip_i2c_{u.layer_key}_{int(u.neuron_id)}"
        any_arr = scores_out[col_any].fillna(False).astype(bool)
        c2i_arr = scores_out[col_c2i].fillna(False).astype(bool)
        i2c_arr = scores_out[col_i2c].fillna(False).astype(bool)
        n_eval = int(scores_out[col_any].notna().sum())
        stats_rows.append({
            "unit_key": u.unit_key,
            "layer_label": u.layer_label,
            "layer_key": u.layer_key,
            "neuron_id": int(u.neuron_id),
            "source": u.source,
            "circuit_id": u.circuit_id,
            "circuit_label": u.circuit_label,
            "seed_strength": np.nan if u.seed_strength is None else float(u.seed_strength),
            "baseline_subset": baseline_subset,
            "n_eval": n_eval,
            "flip_any_rate": float(any_arr.mean()) if len(any_arr) else np.nan,
            "c2i_rate": float(c2i_arr.mean()) if len(c2i_arr) else np.nan,
            "i2c_rate": float(i2c_arr.mean()) if len(i2c_arr) else np.nan,
            "n_flip_any": int(any_arr.sum()),
            "n_flip_c2i": int(c2i_arr.sum()),
            "n_flip_i2c": int(i2c_arr.sum()),
        })
    flush_all_ablation_cache_stores()
    return scores_out, pd.DataFrame(stats_rows)
