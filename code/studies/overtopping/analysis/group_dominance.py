#!/usr/bin/env python3
"""Compute the paper's direct m-way group-dominance statistic.

For a discovered channel set A and intervention phase p, this script computes

    Dom_p^(m)(A) = max_{B subseteq A, 1 <= |B| <= m} E_p(B) / E_p(A),

where E_p(G) is the largest empirical, direction-conditioned flip rate over
the declared evaluation slices. Every group in the ratio is intervened on
simultaneously. This is not the singleton-union proxy Top/U(J).

The analysis reuses per-example singleton flip columns from stage 7 and runs the missing simultaneous intervention on the entire candidate set A. The evaluation split defaults to test and may be set to train or all. For m > 1, every required subset is evaluated exactly, subject to a safety cap.
"""

from __future__ import annotations
from pathlib import Path
from studies.overtopping.analysis.lib.progress import tqdm


import argparse
import itertools
import json
import math
import pickle
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


from core.caching_and_prompting import set_deterministic
from core.feature_extraction_runner import resolve_task_spec
from core.high_n_singleton_eval import (
    UnitSpec,
    load_scores_for_baseline,
    precompute_replacements_for_units,
)
from core.modeling_and_ablation import (
    LMWrapper,
    build_ablation_hooks,
    get_device,
)
from core.neuron_intervention import (
    build_prefix_caches_for_examples,
    get_correctness,
    get_correctness_cached_by_prefix_batches,
)
from core.group_intervention import (
    dedupe_units,
    group_layer_map,
    hash_payload,
    read_table,
    resolve_dataset_path,
    rows_fingerprint,
)
from core.threshold_event_shared import safe_layer_label


LOG_PREFIX = "[group-dominance]"


@dataclass(frozen=True)
class GroupSpec:
    units: tuple[UnitSpec, ...]
    is_full_set: bool = False

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(u.unit_key for u in self.units))

    @property
    def key(self) -> str:
        return "|".join(self.keys)

    @property
    def size(self) -> int:
        return len(self.units)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute direct Dom^(m) using simultaneous group interventions on "
            "the selected evaluation split."
        )
    )
    p.add_argument(
        "--input_data_dir",
        required=True,
        help="Circuit-discovery neural_circuits directory containing dataset_info.json.",
    )
    p.add_argument(
        "--candidate_flip_stats_path",
        required=True,
        help="Stage 7 flip_stats_by_neuron.csv defining candidate set A.",
    )
    p.add_argument(
        "--singleton_scores_path",
        default=None,
        help=(
            "Stage 7 scores.csv with per-example singleton flip columns. "
            "Defaults to scores.csv beside candidate_flip_stats_path."
        ),
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_module", default="core.tasks.arithmetic_task")
    p.add_argument("--ai_model", default=None)
    p.add_argument("--ai_model_cache_dir", default=None)
    p.add_argument("--evaluation_split", choices=["test", "train", "all"], default="test")
    p.add_argument(
        "--intervention",
        choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"],
        default="mean-donor",
    )
    p.add_argument("--decode_only", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)
    p.add_argument("--m", type=int, default=1)
    p.add_argument(
        "--max_exact_subsets",
        type=int,
        default=5000,
        help="Abort before GPU work if exact enumeration would exceed this many proper subsets.",
    )
    p.add_argument(
        "--slice_col",
        action="append",
        default=[],
        help=(
            "Optional categorical column defining additional predeclared task slices. "
            "Repeat for several columns. The complete selected evaluation split is always included."
        ),
    )
    p.add_argument("--max_slice_values", type=int, default=100)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--force", action="store_true")
    return p.parse_args()






def _load_dataset_info(input_data_dir: Path) -> dict:
    path = input_data_dir / "dataset_info.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset_info.json not found under {input_data_dir}")
    return json.loads(path.read_text(encoding="utf-8"))




def load_candidate_units(path: Path) -> list[UnitSpec]:
    df = read_table(path)
    layer_col = "layer_label" if "layer_label" in df.columns else "layer_key"
    if layer_col not in df.columns or "neuron_id" not in df.columns:
        raise ValueError(
            f"{path} must contain neuron_id and layer_label or layer_key; "
            f"found {list(df.columns)}"
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
        for row in df.dropna(subset=[layer_col, "neuron_id"]).to_dict("records")
    ]
    units = dedupe_units(units)
    if not units:
        raise ValueError(f"No candidate units found in {path}")
    return units


def _evaluation_frame(
    singleton_scores_path: Path,
    *,
    evaluation_split: str,
    prompt_col: str,
    target_col: str,
) -> pd.DataFrame:
    df = read_table(singleton_scores_path)
    missing = [c for c in [prompt_col, target_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{singleton_scores_path} is missing required columns {missing}"
        )
    if "_evaluated" in df.columns:
        df = df.loc[df["_evaluated"].fillna(False).astype(bool)].copy()
    evaluation_split = str(evaluation_split).strip().lower()
    if evaluation_split in {"test", "train"}:
        if "is_test" not in df.columns:
            raise ValueError(
                f"--evaluation_split {evaluation_split} requires is_test in the singleton scores table."
            )
        is_test = df["is_test"].fillna(False).astype(bool)
        df = df.loc[is_test if evaluation_split == "test" else ~is_test].copy()
    elif evaluation_split != "all":
        raise ValueError(f"Unsupported evaluation split {evaluation_split!r}")
    if df.empty:
        raise ValueError(f"No rows remain for evaluation split={evaluation_split}.")
    return df.reset_index(drop=True)


def _slice_masks(
    scores_df: pd.DataFrame,
    slice_cols: Sequence[str],
    *,
    max_slice_values: int,
    evaluation_split: str,
) -> list[tuple[str, np.ndarray]]:
    all_label = "all_heldout" if evaluation_split == "test" else f"all_{evaluation_split}"
    masks: list[tuple[str, np.ndarray]] = [
        (all_label, np.ones(len(scores_df), dtype=bool))
    ]
    for col in slice_cols:
        if col not in scores_df.columns:
            raise ValueError(f"--slice_col {col!r} is absent from singleton scores.")
        values = scores_df[col].dropna().unique().tolist()
        if len(values) > int(max_slice_values):
            raise ValueError(
                f"--slice_col {col!r} has {len(values)} values, exceeding "
                f"--max_slice_values={max_slice_values}. Use a categorical slice column."
            )
        for value in sorted(values, key=lambda x: str(x)):
            masks.append(
                (
                    f"{col}={value}",
                    (scores_df[col].astype(str).to_numpy() == str(value)),
                )
            )
    return masks


def _effect_rows(
    *,
    group: GroupSpec,
    baseline: np.ndarray,
    post: np.ndarray,
    slices: Sequence[tuple[str, np.ndarray]],
    source: str,
) -> list[dict]:
    baseline = np.asarray(baseline, dtype=bool)
    post = np.asarray(post, dtype=bool)
    if len(baseline) != len(post):
        raise ValueError("Baseline and post-intervention arrays differ in length.")
    rows: list[dict] = []
    for slice_name, slice_mask in slices:
        slice_mask = np.asarray(slice_mask, dtype=bool)
        for b, direction in [(True, "positive_to_negative"), (False, "negative_to_positive")]:
            eligible = slice_mask & (baseline == b)
            n = int(eligible.sum())
            flips = eligible & (post != baseline)
            k = int(flips.sum())
            rows.append(
                {
                    "group_key": group.key,
                    "group_size": group.size,
                    "is_full_set": bool(group.is_full_set),
                    "unit_keys": json.dumps(list(group.keys)),
                    "slice": slice_name,
                    "baseline_b": int(b),
                    "direction": direction,
                    "n_eligible": n,
                    "n_flips": k,
                    "delta": (float(k) / float(n)) if n else np.nan,
                    "source": source,
                }
            )
    return rows


def _group_effect(effect_rows: Sequence[dict]) -> tuple[float, dict | None]:
    valid = [r for r in effect_rows if np.isfinite(float(r["delta"]))]
    if not valid:
        return float("nan"), None
    best = max(valid, key=lambda r: float(r["delta"]))
    return float(best["delta"]), best


def _flip_column(unit: UnitSpec) -> str:
    return f"flip_{safe_layer_label(unit.layer_label)}_{int(unit.neuron_id)}"


def _post_from_saved_singleton(
    scores_df: pd.DataFrame,
    baseline: np.ndarray,
    unit: UnitSpec,
) -> np.ndarray | None:
    col = _flip_column(unit)
    if col not in scores_df.columns:
        return None
    values = scores_df[col]
    if values.isna().any():
        return None
    flips = values.astype(bool).to_numpy()
    return np.asarray(baseline, dtype=bool) ^ flips


def _proper_subsets(units: Sequence[UnitSpec], m: int) -> list[GroupSpec]:
    out: list[GroupSpec] = []
    for size in range(1, min(int(m), len(units)) + 1):
        for combo in itertools.combinations(units, size):
            out.append(GroupSpec(tuple(combo), is_full_set=(size == len(units))))
    return out








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
    """Evaluate several simultaneous groups, reusing prefix caches by batch."""
    examples = scores_df.to_dict(orient="records")
    outputs = {
        group.key: np.zeros(len(examples), dtype=bool)
        for group in groups
    }
    cache_dir.mkdir(parents=True, exist_ok=True)

    for start in tqdm(
        range(0, len(examples), int(batch_size)),
        desc=f"{LOG_PREFIX} batches",
        unit="batch",
    ):
        end = min(start + int(batch_size), len(examples))
        batch = examples[start:end]
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

        for group in tqdm(
            groups,
            desc=f"{LOG_PREFIX} simultaneous groups",
            unit="group",
            leave=False,
        ):
            cache_key = hash_payload(
                {
                    **cache_context,
                    "group": group.keys,
                    "start": int(start),
                    "end": int(end),
                }
            )
            cache_path = cache_dir / f"group_{cache_key}.pkl"
            post = None
            if cache_path.exists() and not force:
                try:
                    with cache_path.open("rb") as handle:
                        post = np.asarray(pickle.load(handle), dtype=bool)
                    if len(post) != len(batch):
                        post = None
                except Exception:
                    post = None
            if post is None:
                hooks = build_ablation_hooks(
                    group_layer_map(group),
                    last_pos_only=bool(decode_only),
                    intervention=intervention,
                    mean_activations=mean_activations,
                    device=model.hooked_model.cfg.device,
                )
                if decode_only:
                    _, acc = get_correctness_cached_by_prefix_batches(
                        model,
                        batch,
                        is_answer_positive_fn,
                        prefix_batches,
                        batch_ranges,
                        hooks=hooks,
                    )
                else:
                    _, acc = get_correctness(
                        model,
                        batch,
                        is_answer_positive_fn,
                        prompt_col,
                        max_new_tokens=int(max_new_tokens),
                        hooks=hooks,
                        batch_size=int(batch_size),
                    )
                post = np.asarray(acc, dtype=float) > 0.5
                with cache_path.open("wb") as handle:
                    pickle.dump(post.astype(bool), handle)
            outputs[group.key][start:end] = post
        try:
            model.cleanup_after_generate()
        except Exception:
            pass
    return outputs


def summarize_dominance(
    *,
    full_group: GroupSpec,
    group_effects: dict[str, tuple[float, dict | None]],
    m: int,
    rho: float,
) -> list[dict]:
    denominator, denominator_argmax = group_effects[full_group.key]
    proper = [
        (key, value)
        for key, value in group_effects.items()
        if key != full_group.key
    ]
    rows: list[dict] = []
    for current_m in range(1, int(m) + 1):
        candidates = []
        for key, (effect, argmax) in proper:
            size = 0 if not key else len(key.split("|"))
            if 1 <= size <= current_m and np.isfinite(effect):
                candidates.append((effect, key, size, argmax))
        if full_group.size <= current_m and np.isfinite(denominator):
            candidates.append(
                (denominator, full_group.key, full_group.size, denominator_argmax)
            )
        if not candidates:
            numerator = float("nan")
            best_key = None
            best_size = None
            numerator_argmax = None
        else:
            numerator, best_key, best_size, numerator_argmax = max(
                candidates, key=lambda item: item[0]
            )

        if not np.isfinite(denominator):
            dom = float("nan")
            status = "undefined_nonfinite_denominator"
        elif denominator == 0:
            dom = float("nan")
            status = "undefined_zero_denominator"
        elif not np.isfinite(numerator):
            dom = float("nan")
            status = "undefined_no_subset_effect"
        else:
            dom = float(numerator) / float(denominator)
            status = "ok"

        rows.append(
            {
                "m": int(current_m),
                "candidate_set_size": int(full_group.size),
                "numerator_max_subset_effect": numerator,
                "best_subset_key": best_key,
                "best_subset_size": best_size,
                "denominator_full_set_effect": denominator,
                "Dom": dom,
                "rho": float(rho),
                "is_m_rho_overtopped": (
                    bool(dom >= float(rho)) if np.isfinite(dom) else None
                ),
                "status": status,
                "numerator_argmax_slice": (
                    numerator_argmax.get("slice") if numerator_argmax else None
                ),
                "numerator_argmax_direction": (
                    numerator_argmax.get("direction") if numerator_argmax else None
                ),
                "denominator_argmax_slice": (
                    denominator_argmax.get("slice") if denominator_argmax else None
                ),
                "denominator_argmax_direction": (
                    denominator_argmax.get("direction") if denominator_argmax else None
                ),
            }
        )
    return rows


def summarize_direction_matched_dominance(
    *,
    full_group: GroupSpec,
    effects_df: pd.DataFrame,
    m: int,
    rho: float,
) -> tuple[list[dict], dict[int, dict]]:
    """Compute dominance with numerator and denominator in the same context.

    The submitted Definition 4 first maximizes each group's effect over slice
    and behavioural direction. That can put the numerator and denominator in
    different directions. This companion report keeps the slice and direction
    fixed. It also returns one scalar per m evaluated in the context where the
    full-set effect is maximal, which is the direction-matched analogue used in
    the held-out analysis.
    """
    required = {
        "group_key", "group_size", "is_full_set", "slice", "baseline_b",
        "direction", "delta", "n_eligible", "n_flips",
    }
    missing = sorted(required - set(effects_df.columns))
    if missing:
        raise ValueError(f"Effect table is missing columns needed for matched dominance: {missing}")

    full = effects_df.loc[effects_df["group_key"] == full_group.key].copy()
    if full.empty:
        raise ValueError("Full-set effect rows are missing.")
    full["delta"] = pd.to_numeric(full["delta"], errors="coerce")
    valid_full = full.loc[full["delta"].notna()].copy()
    argmax_context = None
    if not valid_full.empty:
        argmax_context = valid_full.sort_values(
            ["delta", "slice", "direction"], ascending=[False, True, True]
        ).iloc[0]

    rows: list[dict] = []
    scalar_by_m: dict[int, dict] = {}
    context_cols = ["slice", "baseline_b", "direction"]
    for _, denominator_row in full.iterrows():
        context = {col: denominator_row[col] for col in context_cols}
        denominator = float(denominator_row["delta"]) if pd.notna(denominator_row["delta"]) else math.nan
        same_context = effects_df.copy()
        for col, value in context.items():
            same_context = same_context.loc[same_context[col] == value]
        same_context = same_context.loc[
            pd.to_numeric(same_context["group_size"], errors="coerce") >= 1
        ].copy()
        same_context["delta"] = pd.to_numeric(same_context["delta"], errors="coerce")

        for current_m in range(1, int(m) + 1):
            candidates = same_context.loc[
                pd.to_numeric(same_context["group_size"], errors="coerce") <= current_m
            ].dropna(subset=["delta"])
            if candidates.empty:
                numerator = math.nan
                best = None
            else:
                best = candidates.sort_values(
                    ["delta", "group_size", "group_key"],
                    ascending=[False, True, True],
                ).iloc[0]
                numerator = float(best["delta"])

            if not np.isfinite(denominator):
                dom = math.nan
                status = "undefined_nonfinite_denominator"
            elif denominator == 0:
                dom = math.nan
                status = "undefined_zero_denominator"
            elif not np.isfinite(numerator):
                dom = math.nan
                status = "undefined_no_subset_effect"
            else:
                dom = numerator / denominator
                status = "ok"

            is_argmax = bool(
                argmax_context is not None
                and str(context["slice"]) == str(argmax_context["slice"])
                and int(context["baseline_b"]) == int(argmax_context["baseline_b"])
                and str(context["direction"]) == str(argmax_context["direction"])
            )
            record = {
                "m": int(current_m),
                "candidate_set_size": int(full_group.size),
                **context,
                "n_eligible": int(denominator_row["n_eligible"]),
                "denominator_full_set_effect": denominator,
                "numerator_max_subset_effect": numerator,
                "best_subset_key": None if best is None else best["group_key"],
                "best_subset_size": None if best is None else int(best["group_size"]),
                "Dom_direction_matched": dom,
                "rho": float(rho),
                "is_m_rho_overtopped": bool(dom >= float(rho)) if np.isfinite(dom) else None,
                "status": status,
                "is_full_set_argmax_context": is_argmax,
            }
            rows.append(record)
            if is_argmax:
                scalar_by_m[int(current_m)] = record
    return rows, scalar_by_m


def _write_markdown(
    path: Path,
    *,
    summary_df: pd.DataFrame,
    candidate_count: int,
    evaluation_split: str,
    slice_cols: Sequence[str],
) -> None:
    lines = [
        "# Direct group dominance",
        "",
        f"- Candidate set size: {candidate_count}",
        f"- Final-statistics split: {evaluation_split}",
        "- Numerator and denominator: simultaneous interventions",
        "- Replacement reference split: train",
        f"- Additional slice columns: {', '.join(slice_cols) if slice_cols else 'none'}",
        "",
        "| m | max subset E | full-set E | submitted Dom | direction-matched Dom | matched direction | overtopped | status |",
        "|---:|---:|---:|---:|---:|:---|:---:|:---|",
    ]
    for row in summary_df.to_dict("records"):
        def fmt(value):
            return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.4f}"

        overtopped = row.get("is_m_rho_overtopped")
        overtopped_text = "NA" if pd.isna(overtopped) else ("yes" if bool(overtopped) else "no")
        lines.append(
            f"| {int(row['m'])} | {fmt(row['numerator_max_subset_effect'])} | "
            f"{fmt(row['denominator_full_set_effect'])} | {fmt(row['Dom'])} | "
            f"{fmt(row.get('Dom_direction_matched'))} | "
            f"{row.get('matched_argmax_direction') or 'NA'} | "
            f"{overtopped_text} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "Dom is not clipped at 1. Values above 1 are possible under non-additive interventions.",
            "The direction-matched value uses the slice/direction where the full-set effect is maximal and evaluates the best subset in that same context.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.m < 1:
        raise ValueError("--m must be at least 1.")
    set_deterministic(int(args.seed))

    input_data_dir = Path(args.input_data_dir).expanduser().resolve()
    candidate_path = Path(args.candidate_flip_stats_path).expanduser().resolve()
    singleton_scores_path = (
        Path(args.singleton_scores_path).expanduser().resolve()
        if args.singleton_scores_path
        else candidate_path.parent / "scores.csv"
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = _load_dataset_info(input_data_dir)
    task = resolve_task_spec(args.task_module)
    prompt_col = dataset_info.get("prompt_col") or task.DEFAULT_INPUT
    target_col = dataset_info.get("target_col") or task.DEFAULT_TARGETS[0]
    ai_model = args.ai_model or dataset_info.get("ai_model")
    if not ai_model:
        raise ValueError("Could not resolve model from --ai_model or dataset_info.json.")

    units = load_candidate_units(candidate_path)
    full_group = GroupSpec(tuple(units), is_full_set=True)
    subsets = _proper_subsets(units, int(args.m))
    proper_subsets = [group for group in subsets if group.key != full_group.key]
    if len(proper_subsets) > int(args.max_exact_subsets):
        raise ValueError(
            f"Exact Dom^({args.m}) requires {len(proper_subsets)} proper subsets for "
            f"|A|={len(units)}, exceeding --max_exact_subsets={args.max_exact_subsets}. "
            "Lower m or raise the cap explicitly."
        )

    scores_df = _evaluation_frame(
        singleton_scores_path,
        evaluation_split=str(args.evaluation_split),
        prompt_col=prompt_col,
        target_col=target_col,
    )
    baseline = pd.to_numeric(scores_df[target_col], errors="raise").to_numpy() > 0.5
    slices = _slice_masks(
        scores_df,
        args.slice_col,
        max_slice_values=int(args.max_slice_values),
        evaluation_split=str(args.evaluation_split),
    )

    effect_rows: list[dict] = []
    group_effects: dict[str, tuple[float, dict | None]] = {}
    needs_gpu: list[GroupSpec] = []

    for group in proper_subsets:
        post = None
        if group.size == 1:
            post = _post_from_saved_singleton(scores_df, baseline, group.units[0])
        if post is None:
            needs_gpu.append(group)
            continue
        rows = _effect_rows(
            group=group,
            baseline=baseline,
            post=post,
            slices=slices,
            source="saved_singleton_flips",
        )
        effect_rows.extend(rows)
        group_effects[group.key] = _group_effect(rows)

    full_post = None
    if full_group.size == 1:
        full_post = _post_from_saved_singleton(scores_df, baseline, full_group.units[0])
    if full_post is not None:
        rows = _effect_rows(
            group=full_group,
            baseline=baseline,
            post=full_post,
            slices=slices,
            source="saved_singleton_flips",
        )
        effect_rows.extend(rows)
        group_effects[full_group.key] = _group_effect(rows)
    else:
        needs_gpu.append(full_group)
    unique_gpu: dict[str, GroupSpec] = {group.key: group for group in needs_gpu}
    needs_gpu = list(unique_gpu.values())

    scores_path = resolve_dataset_path(dataset_info["scores_path"], input_data_dir)
    train_scores = load_scores_for_baseline(
        scores_path=scores_path,
        target_col=target_col,
        baseline_subset="all",
        task_targets=task.DEFAULT_TARGETS,
        split="train",
    )
    if train_scores.empty and args.intervention.startswith("mean"):
        raise ValueError("No training rows are available for mean replacement estimation.")

    print(
        f"{LOG_PREFIX} |A|={len(units)} exact proper subsets={len(proper_subsets)} "
        f"GPU groups={len(needs_gpu)} evaluation rows={len(scores_df)}"
    )
    gpu_outputs: dict[str, np.ndarray] = {}
    if needs_gpu:
        device = get_device()
        model = LMWrapper(
            model_name=ai_model,
            device=device,
            eval_mode=True,
            circuit_discovery=False,
            cache_dir=args.ai_model_cache_dir,
        )
        mean_activations = precompute_replacements_for_units(
            model=model,
            units=units,
            scores_df_for_mean=train_scores,
            prompt_col=prompt_col,
            target_col=target_col,
            intervention=args.intervention,
            points_to_use=int(args.points_to_use_for_mean_ablation),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
        cache_context = {
            "rows": rows_fingerprint(scores_df, prompt_col),
            "model": ai_model,
            "task_module": args.task_module,
            "evaluation_split": args.evaluation_split,
            "prompt_col": prompt_col,
            "target_col": target_col,
            "decode_only": bool(args.decode_only),
            "intervention": args.intervention,
            "max_new_tokens": int(task.MAX_NEW_TOKENS),
        }
        gpu_outputs = evaluate_groups(
            model=model,
            groups=needs_gpu,
            scores_df=scores_df,
            prompt_col=prompt_col,
            is_answer_positive_fn=task.is_answer_positive,
            batch_size=int(args.batch_size),
            decode_only=bool(args.decode_only),
            intervention=args.intervention,
            mean_activations=mean_activations,
            max_new_tokens=int(task.MAX_NEW_TOKENS),
            cache_dir=out_dir / "group_eval_cache",
            cache_context=cache_context,
            force=bool(args.force),
        )

    group_score_df = scores_df.copy()
    for group in needs_gpu:
        post = gpu_outputs[group.key]
        col = f"post_group_{hash_payload({'group': group.keys})}"
        group_score_df[col] = post.astype(bool)
        rows = _effect_rows(
            group=group,
            baseline=baseline,
            post=post,
            slices=slices,
            source="simultaneous_group_intervention",
        )
        effect_rows.extend(rows)
        group_effects[group.key] = _group_effect(rows)

    missing = [group.key for group in proper_subsets if group.key not in group_effects]
    if missing:
        raise RuntimeError(f"Missing effects for {len(missing)} subsets: {missing[:3]}")

    summary_rows = summarize_dominance(
        full_group=full_group,
        group_effects=group_effects,
        m=int(args.m),
        rho=float(args.rho),
    )
    effects_df = pd.DataFrame(effect_rows)
    matched_rows, matched_scalar = summarize_direction_matched_dominance(
        full_group=full_group,
        effects_df=effects_df,
        m=int(args.m),
        rho=float(args.rho),
    )
    for row in summary_rows:
        matched = matched_scalar.get(int(row["m"]))
        row.update(
            {
                "Dom_direction_matched": (
                    matched.get("Dom_direction_matched") if matched else math.nan
                ),
                "matched_numerator_max_subset_effect": (
                    matched.get("numerator_max_subset_effect") if matched else math.nan
                ),
                "matched_denominator_full_set_effect": (
                    matched.get("denominator_full_set_effect") if matched else math.nan
                ),
                "matched_argmax_slice": matched.get("slice") if matched else None,
                "matched_argmax_direction": matched.get("direction") if matched else None,
                "matched_status": matched.get("status") if matched else None,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    matched_df = pd.DataFrame(matched_rows)
    units_df = pd.DataFrame(
        [
            {
                "unit_key": unit.unit_key,
                "layer_label": unit.layer_label,
                "layer_key": safe_layer_label(unit.layer_label),
                "neuron_id": int(unit.neuron_id),
            }
            for unit in units
        ]
    )

    units_df.to_csv(out_dir / "candidate_set_A.csv", index=False)
    group_score_df.to_csv(out_dir / "group_post_intervention_scores.csv", index=False)
    effects_df.to_csv(out_dir / "group_effects_by_slice_direction.csv", index=False)
    summary_df.to_csv(out_dir / "dominance_summary.csv", index=False)
    matched_df.to_csv(out_dir / "dominance_direction_matched.csv", index=False)

    payload = {
        "definition": "Dom_p^(m)(A)=max_{B subseteq A,1<=|B|<=m} E_p(B)/E_p(A)",
        "effect": "E_p(G)=max over declared slices and baseline directions of empirical conditional flip rate",
        "candidate_set_size": len(units),
        "candidate_set": [unit.unit_key for unit in units],
        "m": int(args.m),
        "rho": float(args.rho),
        "exact_enumeration": True,
        "n_proper_subsets": len(proper_subsets),
        "n_gpu_groups": len(needs_gpu),
        "evaluation_split": args.evaluation_split,
        "candidate_discovery_split": "train",
        "replacement_reference_split": "train",
        "n_evaluation_rows": len(scores_df),
        "slice_columns": list(args.slice_col),
        "decode_only": bool(args.decode_only),
        "intervention": args.intervention,
        "ai_model": ai_model,
        "task_module": args.task_module,
        "summary": summary_rows,
        "direction_matched": matched_rows,
        "notes": [
            "Numerator subsets and denominator full set use simultaneous interventions.",
            "Saved singleton flips are reused from the supplied scores table on the requested evaluation split.",
            "Dom is not clipped at 1 because intervention effects may be non-additive.",
            "This statistic is distinct from Top/U(J), whose denominator is a union of singleton flip events.",
            "Dom_direction_matched fixes the full-set argmax slice/direction and evaluates the numerator in that same context.",
        ],
    }
    (out_dir / "dominance_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    _write_markdown(
        out_dir / "dominance_summary.md",
        summary_df=summary_df,
        candidate_count=len(units),
        evaluation_split=str(args.evaluation_split),
        slice_cols=args.slice_col,
    )
    print(summary_df.to_string(index=False))
    print(f"{LOG_PREFIX} wrote {out_dir / 'dominance_summary.md'}")


if __name__ == "__main__":
    main()
