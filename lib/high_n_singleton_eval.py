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
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from lib.caching_and_prompting import load_or_create_cache, set_deterministic
from lib.feature_representation import safe_features_fillna
from lib.modeling_and_ablation import (
    LMWrapper,
    build_ablation_hooks,
    get_device,
    precompute_mean_activations,
)
from lib.neuron_intervention import (
    build_prefix_caches_for_examples,
    get_correctness,
    get_correctness_cached_by_prefix_batches,
)
from lib.spectral_analysis import (
    build_reps_and_embedding_from_args,
    kcenter_farthest_first,
    representative_sample_from_global_clusters,
)
from lib.text_and_rules import guess_filetype
from lib.threshold_event_shared import safe_layer_label

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


def load_scores_for_baseline(*, scores_path: Path, target_col: str, baseline_subset: str, task_targets: Iterable[str]) -> pd.DataFrame:
    scores_df = _read_table(Path(scores_path))
    scores_df["original_idx"] = np.arange(len(scores_df))
    scores_df = safe_features_fillna(scores_df, fill_number=0, fill_bool=False, cols_not_to_fill=list(task_targets))
    if "is_test" in scores_df.columns:
        mask_train = ~scores_df["is_test"].astype(bool)
        scores_df = scores_df.loc[mask_train].reset_index(drop=True)
    if baseline_subset == "positive":
        scores_df = scores_df.loc[scores_df[target_col] == True]
    elif baseline_subset == "negative":
        scores_df = scores_df.loc[scores_df[target_col] == False]
    scores_df = scores_df.reset_index(drop=True)
    return scores_df


def _cache_key(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _series_fingerprint(series: pd.Series) -> str:
    """Stable short fingerprint for cache invalidation when the evaluated data changes."""
    try:
        hashed = pd.util.hash_pandas_object(series.astype(str), index=False).to_numpy(dtype=np.uint64)
        return hashlib.sha1(hashed.tobytes()).hexdigest()[:16]
    except Exception:
        joined = "\n".join(map(str, series.astype(str).tolist()))
        return hashlib.sha1(joined.encode("utf-8", errors="replace")).hexdigest()[:16]


def select_high_n_eval_indices(*, args, model: LMWrapper, scores_df: pd.DataFrame, prompt_col: str, n_points: int, seed: int,
                               cache_dir: Path, use_spectral_sampling: bool = True) -> tuple[np.ndarray, dict]:
    n_total = len(scores_df)
    all_idx = np.arange(n_total, dtype=int)
    n_points = int(min(max(1, int(n_points)), n_total))
    if n_points >= n_total:
        return all_idx, {"mode": "all", "n_selected": int(n_total)}

    if not use_spectral_sampling:
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(all_idx, size=n_points, replace=False).astype(int))
        return idx, {"mode": "random", "n_selected": int(len(idx)), "seed": int(seed)}

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "mode": "spectral_high_n_eval_indices",
        "n_rows": int(n_total),
        "scores_fingerprint": _series_fingerprint(scores_df[prompt_col]) if prompt_col in scores_df.columns else None,
        "prompt_col": str(prompt_col),
        "n_points": int(n_points),
        "seed": int(seed),
        "ai_model": getattr(args, "ai_model", None),
        "spectral_space": getattr(args, "spectral_space", "hidden"),
        "rep_hook_name": getattr(args, "rep_hook_name", "ln_final.hook_normalized"),
        "rep_pooling": getattr(args, "rep_pooling", "last"),
        "spectral_dim": int(getattr(args, "spectral_dim", 32)),
        "global_n_clusters": int(getattr(args, "spiking_global_n_clusters", getattr(args, "global_n_clusters", 64))),
    }
    fp = cache_dir / f"high_n_eval_indices_{_cache_key(cfg)}.pkl"

    def _compute():
        if str(getattr(args, "log_level", "quiet")) != "quiet":
            tqdm.write(f"{LOG_PREFIX} computing spectral sampling embeddings for {n_total} rows")
        texts = scores_df[prompt_col].astype(str).tolist()
        _, Z = build_reps_and_embedding_from_args(
            args=args,
            texts=texts,
            model=getattr(model, "model", None),
            tokenizer=getattr(model, "tokenizer", None),
            device=model.hooked_model.cfg.device,
        )
        Z = np.asarray(Z, dtype=np.float32)
        k = min(max(2, int(cfg["global_n_clusters"])), len(Z))
        if str(getattr(args, "log_level", "quiet")) != "quiet":
            tqdm.write(f"{LOG_PREFIX} selecting {n_points} representative rows from {k} spectral clusters")
        centers_idx, cluster_id, _min_d2, global_meta, x_norm2 = kcenter_farthest_first(Z, k=k)
        sample_indices, cover_meta = representative_sample_from_global_clusters(
            Z=Z,
            x_norm2=x_norm2,
            centers_idx=centers_idx,
            cluster_id=cluster_id,
            group_idx=np.arange(len(Z), dtype=int),
            n_select=n_points,
            seed=int(seed),
        )
        sample_indices = np.asarray(sorted(set(map(int, sample_indices))), dtype=int)
        return {
            "sample_indices": sample_indices,
            "meta": {
                "mode": "spectral_global_clusters",
                "n_selected": int(len(sample_indices)),
                "global_meta": global_meta,
                "cover_meta": cover_meta,
                "cache_path": str(fp),
            },
        }

    obj = load_or_create_cache(str(fp), _compute, quiet=True)
    return np.asarray(obj["sample_indices"], dtype=int), dict(obj.get("meta", {}))


def _build_mean_prompt_pool(scores_df: pd.DataFrame, *, prompt_col: str, target_col: str, n_points: int, seed: int) -> list[dict]:
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
    pool = _build_mean_prompt_pool(
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

    for u in tqdm(units, desc=f"{LOG_PREFIX} prepare flip columns", unit="unit", leave=False):
        for col in (f"flip_{u.layer_key}_{int(u.neuron_id)}", f"flip_c2i_{u.layer_key}_{int(u.neuron_id)}", f"flip_i2c_{u.layer_key}_{int(u.neuron_id)}"):
            if col not in scores_out.columns:
                scores_out[col] = pd.array([pd.NA] * len(scores_out), dtype="boolean")

    for start in tqdm(range(0, len(eval_examples), int(batch_size)), total=n_batches, desc=f"{LOG_PREFIX} high-N batches", unit="batch"):
        end = min(start + int(batch_size), len(eval_examples))
        batch_examples = eval_examples[start:end]
        batch_baseline = baseline_eval[start:end]
        prefix_batches = None
        batch_ranges = None
        if decode_only:
            prefix_batches, batch_ranges = build_prefix_caches_for_examples(
                model,
                batch_examples,
                prompt_col,
                max_new_tokens=int(max_new_tokens),
                batch_size=int(batch_size),
            )
        batch_cache_key = _cache_key({
            "eval_indices": [int(i) for i in eval_indices[start:end].tolist()],
            "baseline_subset": str(baseline_subset),
            "target_col": str(target_col),
            "prompt_col": str(prompt_col),
            "decode_only": bool(decode_only),
            "intervention": str(intervention),
            "max_new_tokens": int(max_new_tokens),
            "batch_size": int(batch_size),
        })
        for u in tqdm(units, desc=f"{LOG_PREFIX} units", unit="unit", leave=False):
            cpath = _flip_cache_path(cache_dir, u, start, batch_cache_key)
            arr = None
            if (not force) and cpath.exists():
                try:
                    with cpath.open("rb") as f:
                        arr = pickle.load(f)
                    arr = np.asarray(arr).astype(bool)
                except Exception:
                    arr = None
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
                        prompt_col,
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
                cpath.parent.mkdir(parents=True, exist_ok=True)
                with cpath.open("wb") as f:
                    pickle.dump(arr, f)
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
    return scores_out, pd.DataFrame(stats_rows)
