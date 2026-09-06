#!/usr/bin/env python3
"""Graded causal intervention for frozen overtopping agonists.

This is the primary RQ3 spiking experiment.  It does not fit a classifier to
predict which examples a singleton intervention will flip.  Instead it starts
from the held-out examples already known to be flipped by that agonist under
the full Stage-7 intervention, then continuously increases the strength of the
same intervention from the natural activation (dose 0) to the established full
replacement (dose 1).

Optionally, the identical dose sweep is also evaluated on same-agonist
source-state examples that were *not* flipped by the full intervention.  This
negative support is a within-agonist reference only; it is not a different
neuron/control population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.caching_and_prompting import set_deterministic
from core.feature_extraction_runner import resolve_task_spec
from core.group_intervention import evaluation_frame, load_dataset_info, resolve_dataset_path
from core.high_n_singleton_eval import (
    UnitSpec,
    load_scores_for_baseline,
)
from core.modeling_and_ablation import (
    LMWrapper,
    build_ablation_hooks,
    get_device,
    precompute_mean_activations,
)
from core.neuron_intervention import (
    build_prefix_caches_for_examples,
    get_correctness,
    get_correctness_cached_by_prefix_batches,
)


LOG_PREFIX = "[graded-agonist]"
SCHEMA = "graded-agonist-intervention-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_data_dir", required=True)
    p.add_argument("--candidate_flip_stats_path", required=True)
    p.add_argument("--singleton_scores_path", default=None)
    p.add_argument("--candidate_ranking_path", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task_module", default="core.tasks.arithmetic_task")
    p.add_argument("--ai_model", default=None)
    p.add_argument("--ai_model_cache_dir", default=None)
    p.add_argument(
        "--intervention",
        choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"],
        default="mean-donor",
    )
    p.add_argument("--decode_only", action="store_true")
    p.add_argument("--evaluation_split", choices=["test", "train", "all"], default="test")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)
    p.add_argument(
        "--doses",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
        help="Comma-separated intervention strengths. Must include 0 and 1.",
    )
    p.add_argument(
        "--max_agonists_per_direction",
        type=int,
        default=16,
        help=(
            "Maximum frozen agonists per discovery direction, selected by frozen discovery rank. "
            "0 evaluates every eligible agonist. Default: 16."
        ),
    )
    p.add_argument(
        "--max_positive_support_per_agonist",
        type=int,
        default=256,
        help="Maximum known-flip examples per agonist. 0 uses all held-out flip support.",
    )
    p.add_argument(
        "--same_agonist_negative_support",
        action="store_true",
        help="Also sweep source-state examples that the same agonist did not flip at dose 1.",
    )
    p.add_argument(
        "--negative_support_ratio",
        type=float,
        default=1.0,
        help="Negative-support examples per positive-support example when enabled. Default: 1.",
    )
    p.add_argument(
        "--max_negative_support_per_agonist",
        type=int,
        default=256,
        help="Absolute cap for optional same-agonist negative support. 0 means no additional cap.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def _parse_doses(raw: str) -> tuple[float, ...]:
    vals: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not (0.0 <= value <= 1.0):
            raise ValueError("--doses values must be in [0, 1]")
        if value not in vals:
            vals.append(value)
    vals = sorted(vals)
    if not vals or not math.isclose(vals[0], 0.0) or not math.isclose(vals[-1], 1.0):
        raise ValueError("--doses must include both 0 and 1")
    return tuple(vals)


def _stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def _unit_key(layer: object, neuron: object) -> str:
    return f"{str(layer)}:{int(neuron)}"


def _parse_directions(*values: object) -> set[str]:
    out: set[str] = set()
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        for token in re.split(r"[|,;]", str(value)):
            token = token.strip().lower()
            if token in {"positive", "negative"}:
                out.add(token)
    return out


def _load_candidate_plan(candidate_path: Path, ranking_path: Path | None, max_per_direction: int) -> list[dict]:
    candidates = pd.read_csv(candidate_path)
    if "layer_label" not in candidates.columns and "layer_key" in candidates.columns:
        candidates["layer_label"] = candidates["layer_key"].astype(str)
    if not {"layer_label", "neuron_id"}.issubset(candidates.columns):
        raise ValueError("candidate flip stats require layer_label/layer_key and neuron_id")
    candidates["neuron_id"] = pd.to_numeric(candidates["neuron_id"], errors="raise").astype(int)
    if "unit_key" not in candidates.columns:
        candidates["unit_key"] = [_unit_key(l, n) for l, n in zip(candidates["layer_label"], candidates["neuron_id"])]

    if ranking_path is not None and ranking_path.is_file():
        ranking = pd.read_csv(ranking_path)
        if "layer_label" not in ranking.columns and "layer_key" in ranking.columns:
            ranking["layer_label"] = ranking["layer_key"].astype(str)
        if "unit_key" not in ranking.columns and {"layer_label", "neuron_id"}.issubset(ranking.columns):
            ranking["neuron_id"] = pd.to_numeric(ranking["neuron_id"], errors="raise").astype(int)
            ranking["unit_key"] = [_unit_key(l, n) for l, n in zip(ranking["layer_label"], ranking["neuron_id"])]
        provenance_cols = [c for c in [
            "discovery_baseline_subset", "discovery_baseline_subsets",
            "n_discovery_baseline_subsets", "discovery_rank_global", "discovery_score",
            "discovery_score_signed",
        ] if c in ranking.columns]
        if "unit_key" in ranking.columns and provenance_cols:
            ranked = ranking[["unit_key"] + provenance_cols].drop_duplicates("unit_key").copy()
            candidates = candidates.merge(
                ranked,
                on="unit_key", how="left", validate="one_to_one", suffixes=("", "__ranking"),
            )
            # Frozen ranking is the authority for discovery provenance.  Coalesce
            # it into candidate stats even when an older stats file already has
            # a partially empty/stale copy of the same column.
            for col in provenance_cols:
                ranked_col = f"{col}__ranking"
                if ranked_col not in candidates.columns:
                    continue
                if col not in candidates.columns:
                    candidates[col] = candidates[ranked_col]
                else:
                    # CSV readers may infer an all-empty provenance column as
                    # float/NaN.  Promote before filling textual frozen provenance.
                    if candidates[col].dtype.kind not in {"O", "U", "S"} and ranked[col].dtype == object:
                        candidates[col] = candidates[col].astype(object)
                    left = candidates[col]
                    missing = left.isna() | left.astype(str).str.strip().isin({"", "nan", "None"})
                    candidates.loc[missing, col] = candidates.loc[missing, ranked_col]
                candidates = candidates.drop(columns=[ranked_col])

    plan: list[dict] = []
    for row in candidates.to_dict("records"):
        directions = _parse_directions(
            row.get("discovery_baseline_subsets"), row.get("discovery_baseline_subset")
        )
        if not directions:
            raise RuntimeError(
                f"Missing frozen discovery direction for {row.get('unit_key')}; "
                "graded RQ3 refuses direction-agnostic evaluation."
            )
        for baseline_subset in sorted(directions):
            plan.append({
                "unit_key": str(row["unit_key"]),
                "layer_label": str(row["layer_label"]),
                "neuron_id": int(row["neuron_id"]),
                "baseline_subset": baseline_subset,
                "direction": "1to0" if baseline_subset == "positive" else "0to1",
                "discovery_rank_global": pd.to_numeric(row.get("discovery_rank_global"), errors="coerce"),
            })

    frame = pd.DataFrame(plan)
    frame["_rank"] = pd.to_numeric(frame["discovery_rank_global"], errors="coerce")
    frame["_rank"] = frame["_rank"].fillna(np.inf)
    frame = frame.sort_values(["baseline_subset", "_rank", "unit_key"], kind="mergesort")
    if int(max_per_direction) > 0:
        frame = frame.groupby("baseline_subset", group_keys=False).head(int(max_per_direction))
    return frame.drop(columns=["_rank"]).to_dict("records")




def _load_stage7_replacement_population(candidate_path: Path) -> list[UnitSpec]:
    """Return the full Stage-7 evaluated unit population used by replacement baselines.

    This matters for donor-based replacements: the donor-safe replacement for one
    agonist is selected from the other Stage-7 coordinates at that locus.  Using
    only the subset chosen for the graded experiment would change the dose-1
    intervention and invalidate endpoint reproduction.
    """
    frame = pd.read_csv(candidate_path)
    if "layer_label" not in frame.columns and "layer_key" in frame.columns:
        frame["layer_label"] = frame["layer_key"].astype(str)
    if not {"layer_label", "neuron_id"}.issubset(frame.columns):
        raise ValueError("candidate flip stats require layer_label/layer_key and neuron_id")
    frame["neuron_id"] = pd.to_numeric(frame["neuron_id"], errors="raise").astype(int)
    units: dict[str, UnitSpec] = {}
    for row in frame[["layer_label", "neuron_id"]].drop_duplicates().to_dict("records"):
        unit = UnitSpec(layer_label=str(row["layer_label"]), neuron_id=int(row["neuron_id"]), source="stage7_replacement_population")
        units[unit.unit_key] = unit
    return list(units.values())


def _stage7_mean_prompt_pool(train_scores: pd.DataFrame, *, prompt_col: str, n_points: int, seed: int) -> list[str]:
    """Mirror Stage-7's replacement-score prompt sampling exactly.

    Stage 7 samples uniformly from all non-test rows; despite the historical
    helper name, it does not balance positive/negative behavior classes.
    """
    if prompt_col not in train_scores.columns:
        raise ValueError(f"Prompt column {prompt_col!r} not found in replacement-score frame")
    frame = train_scores.copy()
    if "is_test" in frame.columns:
        frame = frame.loc[~frame["is_test"].fillna(False).astype(bool)].copy()
    if frame.empty:
        return []
    n_points = int(n_points)
    if n_points <= 0 or len(frame) <= n_points:
        sampled = frame.sample(frac=1.0, random_state=int(seed)) if len(frame) > 1 else frame
    else:
        sampled = frame.sample(n=n_points, replace=False, random_state=int(seed))
    return sampled[prompt_col].astype(str).tolist()


def _precompute_stage7_replacements(
    *, model: LMWrapper, all_units: list[UnitSpec], train_scores: pd.DataFrame,
    prompt_col: str, intervention: str, points_to_use: int, batch_size: int, seed: int,
):
    if intervention not in ("mean", "mean-donor", "mean-positional", "mean-donor-positional"):
        return None
    layer_to_neurons: dict[str, set[int]] = {}
    for unit in all_units:
        layer_to_neurons.setdefault(str(unit.layer_label), set()).add(int(unit.neuron_id))
    layer_to_neurons = {key: sorted(ids) for key, ids in layer_to_neurons.items() if ids}
    prompts = _stage7_mean_prompt_pool(
        train_scores, prompt_col=prompt_col, n_points=int(points_to_use), seed=int(seed)
    )
    if not prompts or not layer_to_neurons:
        return None
    return precompute_mean_activations(
        model=model,
        all_prompts=prompts,
        layer_to_neurons=layer_to_neurons,
        n_points=min(int(points_to_use), len(prompts)),
        batch_size=int(batch_size),
        intervention=intervention,
        device=model.hooked_model.cfg.device,
    )


def _directional_flip_col(unit: UnitSpec, baseline_subset: str) -> str:
    if baseline_subset == "positive":
        return f"flip_c2i_{unit.layer_key}_{int(unit.neuron_id)}"
    return f"flip_i2c_{unit.layer_key}_{int(unit.neuron_id)}"


def _sample_indices(indices: np.ndarray, cap: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if cap <= 0 or len(indices) <= int(cap):
        return np.sort(indices)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(indices, size=int(cap), replace=False))


def _row_id(frame: pd.DataFrame, idx: int) -> object:
    for column in ("_orig_row", "original_idx"):
        if column in frame.columns:
            value = frame.iloc[int(idx)][column]
            if pd.isna(value):
                continue
            try:
                return int(value)
            except Exception:
                return str(value)
    return int(idx)


def _evaluate_unit_doses(
    *,
    model: LMWrapper,
    task,
    scores_df: pd.DataFrame,
    prompt_col: str,
    target_col: str,
    unit: UnitSpec,
    selected_indices: np.ndarray,
    support_kind: np.ndarray,
    doses: tuple[float, ...],
    decode_only: bool,
    intervention: str,
    mean_activations,
    batch_size: int,
) -> list[dict]:
    examples = scores_df.iloc[selected_indices].to_dict("records")
    if not examples:
        return []
    prefix_batches = None
    batch_ranges = None
    if decode_only:
        prefix_batches, batch_ranges = build_prefix_caches_for_examples(
            model,
            examples,
            prompt_col,
            max_new_tokens=int(task.MAX_NEW_TOKENS),
            batch_size=int(batch_size),
        )

    baseline_behavior = pd.to_numeric(scores_df.iloc[selected_indices][target_col], errors="raise").to_numpy() > 0.5
    rows: list[dict] = []
    layer_map = {str(unit.layer_label): [int(unit.neuron_id)]}
    for dose in doses:
        hooks = None
        if float(dose) > 0.0:
            hooks = build_ablation_hooks(
                layer_map,
                last_pos_only=bool(decode_only),
                intervention=intervention,
                mean_activations=mean_activations,
                device=model.hooked_model.cfg.device,
                intervention_strength=float(dose),
            )
        if decode_only:
            _, accuracy = get_correctness_cached_by_prefix_batches(
                model,
                examples,
                task.is_answer_positive,
                prefix_batches,
                batch_ranges,
                hooks=hooks,
            )
        else:
            _, accuracy = get_correctness(
                model,
                examples,
                task.is_answer_positive,
                prompt_col,
                max_new_tokens=int(task.MAX_NEW_TOKENS),
                hooks=hooks,
                batch_size=int(batch_size),
            )
        post = np.asarray(accuracy, dtype=float) > 0.5
        for local_i, source_idx in enumerate(selected_indices):
            rows.append({
                "unit_key": unit.unit_key,
                "layer_label": unit.layer_label,
                "neuron_id": int(unit.neuron_id),
                "evaluation_row": int(source_idx),
                "row_id": _row_id(scores_df, int(source_idx)),
                "support_kind": str(support_kind[local_i]),
                "dose": float(dose),
                "baseline_behavior": bool(baseline_behavior[local_i]),
                "behavior_after": bool(post[local_i]),
                "flipped_from_baseline": bool(post[local_i] != baseline_behavior[local_i]),
            })
        try:
            model.cleanup_after_generate()
        except Exception:
            pass
    return rows


def _summarize_examples(dose_rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict] = []
    if dose_rows.empty:
        return pd.DataFrame()
    keys = ["baseline_subset", "direction", "unit_key", "layer_label", "neuron_id", "support_kind", "evaluation_row", "row_id"]
    for values, group in dose_rows.groupby(keys, dropna=False, sort=False):
        g = group.sort_values("dose", kind="mergesort")
        doses = pd.to_numeric(g["dose"], errors="raise").to_numpy(float)
        states = g["behavior_after"].astype(bool).to_numpy()
        baseline = bool(g["baseline_behavior"].iloc[0])
        changed = states != baseline
        first_idx = int(np.flatnonzero(changed)[0]) if changed.any() else None
        first_dose = float(doses[first_idx]) if first_idx is not None else math.nan
        n_state_changes = int(np.sum(states[1:] != states[:-1])) if len(states) > 1 else 0
        persistent_after_first = bool(first_idx is not None and changed[first_idx:].all())
        single_crossing = bool(first_idx is not None and n_state_changes == 1 and persistent_after_first)
        natural_state_matches_stage7 = bool(len(changed) and not changed[0])
        endpoint_flip = bool(changed[-1]) if len(changed) else False
        any_flip = bool(changed.any()) if len(changed) else False
        any_intermediate_flip = bool(changed[:-1].any()) if len(changed) > 1 else False
        stable_all_doses = bool(not any_flip)
        transient_flip = bool(any_flip and not endpoint_flip)
        endpoint_expected = str(values[5]) == "known_flip"
        output.append({
            **dict(zip(keys, values)),
            "n_doses": int(len(g)),
            "first_flip_dose": first_dose,
            "n_state_changes": n_state_changes,
            "persistent_after_first_flip": persistent_after_first,
            "single_crossing": single_crossing,
            "natural_state_matches_stage7_baseline": natural_state_matches_stage7,
            "full_dose_flipped": endpoint_flip,
            "ever_flipped_at_any_dose": any_flip,
            "ever_flipped_before_full_dose": any_intermediate_flip,
            "stable_at_baseline_all_doses": stable_all_doses,
            "transient_flip": transient_flip,
            "full_dose_matches_stage7_support": bool(endpoint_flip == endpoint_expected),
        })
    return pd.DataFrame(output)


def _summarize_units(example_summary: pd.DataFrame) -> pd.DataFrame:
    if example_summary.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    keys = ["baseline_subset", "direction", "unit_key", "layer_label", "neuron_id", "support_kind"]
    for values, group in example_summary.groupby(keys, dropna=False, sort=False):
        first = pd.to_numeric(group["first_flip_dose"], errors="coerce")
        rows.append({
            **dict(zip(keys, values)),
            "n_examples": int(len(group)),
            "natural_state_reproduction_rate": float(group["natural_state_matches_stage7_baseline"].mean()),
            "full_dose_support_reproduction_rate": float(group["full_dose_matches_stage7_support"].mean()),
            "full_dose_flip_rate": float(group["full_dose_flipped"].mean()),
            "any_dose_flip_rate": float(group["ever_flipped_at_any_dose"].mean()),
            "intermediate_flip_rate": float(group["ever_flipped_before_full_dose"].mean()),
            "stable_all_doses_rate": float(group["stable_at_baseline_all_doses"].mean()),
            "transient_flip_rate": float(group["transient_flip"].mean()),
            "single_crossing_rate": float(group["single_crossing"].mean()),
            "median_first_flip_dose": float(first.median()) if first.notna().any() else math.nan,
            "mean_first_flip_dose": float(first.mean()) if first.notna().any() else math.nan,
            "median_state_changes": float(pd.to_numeric(group["n_state_changes"], errors="coerce").median()),
        })
    return pd.DataFrame(rows)


def _plot(dose_rows: pd.DataFrame, out_dir: Path) -> None:
    if dose_rows.empty:
        return
    agg = (
        dose_rows.groupby(["direction", "support_kind", "dose"], dropna=False)["flipped_from_baseline"]
        .agg(["mean", "count"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for ax, direction in zip(axes, ["1to0", "0to1"]):
        d = agg.loc[agg["direction"] == direction]
        for support_kind, group in d.groupby("support_kind", sort=False):
            ax.plot(group["dose"], group["mean"], marker="o", label=str(support_kind))
        ax.set_title(direction)
        ax.set_xlabel("Intervention dose")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Fraction behavior-flipped")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.90 if handles else 1))
    fig.savefig(out_dir / "graded_agonist_dose_response.pdf")
    plt.close(fig)


def _requested_run_config(args: argparse.Namespace, doses: tuple[float, ...]) -> dict:
    return {
        "evaluation_split": str(args.evaluation_split),
        "intervention": str(args.intervention),
        "decode_only": bool(args.decode_only),
        "doses": [float(x) for x in doses],
        "same_agonist_negative_support": bool(args.same_agonist_negative_support),
        "negative_support_ratio": float(args.negative_support_ratio),
        "max_agonists_per_direction": int(args.max_agonists_per_direction),
        "max_positive_support_per_agonist": int(args.max_positive_support_per_agonist),
        "max_negative_support_per_agonist": int(args.max_negative_support_per_agonist),
        "points_to_use_for_mean_ablation": int(args.points_to_use_for_mean_ablation),
        "seed": int(args.seed),
    }


def _manifest_matches_request(existing: dict, requested: dict) -> bool:
    if existing.get("status") != "ok" or existing.get("schema") != SCHEMA:
        return False
    cfg = existing.get("run_config")
    # Legacy manifests did not fingerprint all scientifically relevant support
    # and sampling parameters.  Treat them as stale rather than guessing that
    # they match the current request.  This does not delete the old artifacts;
    # it only prevents a false cache hit when Stage 7b is explicitly rerun.
    return isinstance(cfg, dict) and cfg == requested


def main() -> None:
    args = parse_args()
    doses = _parse_doses(args.doses)
    if args.max_agonists_per_direction < 0:
        raise ValueError("--max_agonists_per_direction must be >= 0")
    if args.max_positive_support_per_agonist < 0 or args.max_negative_support_per_agonist < 0:
        raise ValueError("support caps must be >= 0")
    if args.negative_support_ratio < 0:
        raise ValueError("--negative_support_ratio must be >= 0")

    input_data_dir = Path(args.input_data_dir).expanduser().resolve()
    candidate_path = Path(args.candidate_flip_stats_path).expanduser().resolve()
    stats_dir = candidate_path.parent
    scores_path = Path(args.singleton_scores_path).expanduser().resolve() if args.singleton_scores_path else stats_dir / "scores.csv"
    ranking_path = Path(args.candidate_ranking_path).expanduser().resolve() if args.candidate_ranking_path else stats_dir / "frozen_candidate_ranking.csv"
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "graded_agonist_intervention.json"
    requested_config = _requested_run_config(args, doses)
    if manifest_path.is_file() and not args.force:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _manifest_matches_request(existing, requested_config):
                print(f"{LOG_PREFIX} existing completed output matches requested configuration: {out_dir}")
                return
            print(f"{LOG_PREFIX} existing output configuration differs; recomputing {out_dir}")
        except Exception:
            pass

    # Mark the directory incomplete *before* changing any per-run artifacts.
    # A recomputation may reuse the same output directory as an older completed
    # run.  Without this marker, an interruption after writing the new plan but
    # before writing new dose rows leaves a hybrid directory that looks
    # completed to downstream reporting.  Existing scientific outputs remain on
    # disk and are replaced only as their refreshed versions are written.
    manifest_path.write_text(
        json.dumps({
            "schema": SCHEMA,
            "status": "running",
            "scientific_target": "graded causal transition with same-agonist held-out flip and non-flip support",
            "run_config": requested_config,
            "same_agonist_negative_support": bool(args.same_agonist_negative_support),
            "negative_support_ratio": float(args.negative_support_ratio),
            "doses": list(doses),
        }, indent=2),
        encoding="utf-8",
    )

    dataset_info = load_dataset_info(input_data_dir)
    task = resolve_task_spec(args.task_module)
    prompt_col = dataset_info.get("prompt_col") or task.DEFAULT_INPUT
    target_col = dataset_info.get("target_col") or task.DEFAULT_TARGETS[0]
    ai_model = args.ai_model or dataset_info.get("ai_model")
    if not ai_model:
        raise ValueError("Could not resolve model from --ai_model or dataset_info.json")

    scores_df = evaluation_frame(
        scores_path,
        prompt_col=prompt_col,
        target_col=target_col,
        evaluation_split=str(args.evaluation_split),
    )
    baseline_values = pd.to_numeric(scores_df[target_col], errors="coerce").to_numpy(dtype=float)
    valid_binary = np.isfinite(baseline_values) & (
        np.isclose(baseline_values, 0.0, atol=1e-8, rtol=0.0)
        | np.isclose(baseline_values, 1.0, atol=1e-8, rtol=0.0)
    )
    if not bool(valid_binary.all()):
        examples = baseline_values[~valid_binary][:5].tolist()
        raise ValueError(
            f"Stage-7 baseline predicate {target_col!r} must be binary on the graded-evaluation population; "
            f"found values such as {examples}."
        )
    baseline_behavior = np.isclose(baseline_values, 1.0, atol=1e-8, rtol=0.0)

    candidate_plan = _load_candidate_plan(candidate_path, ranking_path, int(args.max_agonists_per_direction))
    plan_rows: list[dict] = []
    selection: list[tuple[dict, UnitSpec, np.ndarray, np.ndarray]] = []
    for plan in candidate_plan:
        unit = UnitSpec(layer_label=str(plan["layer_label"]), neuron_id=int(plan["neuron_id"]), source="frozen_agonist")
        baseline_subset = str(plan["baseline_subset"])
        source = baseline_behavior if baseline_subset == "positive" else ~baseline_behavior
        flip_col = _directional_flip_col(unit, baseline_subset)
        if flip_col not in scores_df.columns:
            plan_rows.append({**plan, "flip_column": flip_col, "status": "missing_directional_flip_column", "n_known_flip": 0, "n_same_agonist_negative": 0})
            continue
        flip_series = scores_df[flip_col]
        unit_evaluated = flip_series.notna().to_numpy()
        flip = flip_series.fillna(False).astype(bool).to_numpy() & source & unit_evaluated
        pos_idx = np.flatnonzero(flip)
        # Optional negative support must be genuinely evaluated for this same
        # agonist; never reinterpret NA/unevaluated rows as non-flips.
        neg_idx = np.flatnonzero(source & unit_evaluated & ~flip)
        sampled_pos = _sample_indices(
            pos_idx,
            int(args.max_positive_support_per_agonist),
            _stable_seed(args.seed, unit.unit_key, baseline_subset, "positive"),
        )
        sampled_neg = np.asarray([], dtype=int)
        if args.same_agonist_negative_support and len(sampled_pos):
            desired = int(math.ceil(len(sampled_pos) * float(args.negative_support_ratio)))
            cap = desired
            if int(args.max_negative_support_per_agonist) > 0:
                cap = min(cap, int(args.max_negative_support_per_agonist))
            sampled_neg = _sample_indices(
                neg_idx,
                cap,
                _stable_seed(args.seed, unit.unit_key, baseline_subset, "negative"),
            )
        status = "ok" if len(sampled_pos) else "no_heldout_flip_support"
        plan_rows.append({
            **plan,
            "flip_column": flip_col,
            "status": status,
            "n_known_flip": int(len(pos_idx)),
            "n_known_flip_sampled": int(len(sampled_pos)),
            "n_same_agonist_negative": int(len(neg_idx)),
            "n_same_agonist_negative_sampled": int(len(sampled_neg)),
        })
        if len(sampled_pos):
            indices = np.concatenate([sampled_pos, sampled_neg]).astype(int)
            kinds = np.asarray(["known_flip"] * len(sampled_pos) + ["same_agonist_nonflip"] * len(sampled_neg), dtype=object)
            selection.append((plan, unit, indices, kinds))

    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(out_dir / "graded_agonist_plan.csv", index=False)
    if not selection:
        payload = {
            "schema": SCHEMA,
            "status": "no_eligible_agonists",
            "run_config": requested_config,
            "same_agonist_negative_support": bool(args.same_agonist_negative_support),
            "negative_support_ratio": float(args.negative_support_ratio),
            "n_planned": int(len(plan_df)),
            "doses": list(doses),
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{LOG_PREFIX} no agonists have known held-out directional flip support")
        return

    train_scores_path = resolve_dataset_path(dataset_info["scores_path"], input_data_dir)
    train_scores = load_scores_for_baseline(
        scores_path=train_scores_path,
        target_col=target_col,
        baseline_subset="all",
        task_targets=task.DEFAULT_TARGETS,
        split="train",
    )

    set_deterministic(int(args.seed))
    device = get_device()
    model = LMWrapper(
        model_name=ai_model,
        device=device,
        eval_mode=True,
        circuit_discovery=False,
        cache_dir=args.ai_model_cache_dir,
    )
    replacement_population = _load_stage7_replacement_population(candidate_path)
    mean_activations = _precompute_stage7_replacements(
        model=model,
        all_units=replacement_population,
        train_scores=train_scores,
        prompt_col=prompt_col,
        intervention=args.intervention,
        points_to_use=int(args.points_to_use_for_mean_ablation),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    all_rows: list[dict] = []
    for plan, unit, indices, kinds in selection:
        print(
            f"{LOG_PREFIX} {plan['direction']} {unit.unit_key}: "
            f"known_flip={int(np.sum(kinds == 'known_flip'))} "
            f"negative={int(np.sum(kinds == 'same_agonist_nonflip'))} doses={len(doses)}"
        )
        rows = _evaluate_unit_doses(
            model=model,
            task=task,
            scores_df=scores_df,
            prompt_col=prompt_col,
            target_col=target_col,
            unit=unit,
            selected_indices=indices,
            support_kind=kinds,
            doses=doses,
            decode_only=bool(args.decode_only),
            intervention=str(args.intervention),
            mean_activations=mean_activations,
            batch_size=int(args.batch_size),
        )
        for row in rows:
            row["baseline_subset"] = str(plan["baseline_subset"])
            row["direction"] = str(plan["direction"])
        all_rows.extend(rows)

    dose_df = pd.DataFrame(all_rows)
    dose_df.to_csv(out_dir / "graded_agonist_dose_rows.csv.gz", index=False, compression="gzip")
    example_summary = _summarize_examples(dose_df)
    example_summary.to_csv(out_dir / "graded_agonist_example_summary.csv", index=False)
    unit_summary = _summarize_units(example_summary)
    unit_summary.to_csv(out_dir / "graded_agonist_unit_summary.csv", index=False)
    if not args.no_plot:
        _plot(dose_df, out_dir)

    positive_summary = unit_summary.loc[unit_summary.get("support_kind", pd.Series(dtype=str)).astype(str) == "known_flip"] if not unit_summary.empty else pd.DataFrame()
    negative_summary = unit_summary.loc[unit_summary.get("support_kind", pd.Series(dtype=str)).astype(str) == "same_agonist_nonflip"] if not unit_summary.empty else pd.DataFrame()
    payload = {
        "schema": SCHEMA,
        "status": "ok",
        "scientific_target": "graded causal transition with same-agonist held-out flip and non-flip support",
        "run_config": requested_config,
        "evaluation_split": str(args.evaluation_split),
        "intervention": str(args.intervention),
        "decode_only": bool(args.decode_only),
        "doses": list(doses),
        "same_agonist_negative_support": bool(args.same_agonist_negative_support),
        "negative_support_ratio": float(args.negative_support_ratio),
        "max_agonists_per_direction": int(args.max_agonists_per_direction),
        "selection_policy": "frozen discovery rank; never ranked by held-out graded response",
        "replacement_semantics": "dose 1 reconstructs Stage-7 replacement using the full Stage-7 evaluated unit population and Stage-7 train-prompt sampling rule",
        "replacement_population_size": int(len(replacement_population)),
        "n_selected_agonist_directions": int(len(selection)),
        "n_known_flip_examples_evaluated": int((dose_df["support_kind"] == "known_flip").sum() / max(len(doses), 1)) if not dose_df.empty else 0,
        "n_same_agonist_negative_examples_evaluated": int((dose_df["support_kind"] == "same_agonist_nonflip").sum() / max(len(doses), 1)) if not dose_df.empty else 0,
        "median_unit_single_crossing_rate_known_flip": float(pd.to_numeric(positive_summary.get("single_crossing_rate"), errors="coerce").median()) if not positive_summary.empty else math.nan,
        "median_unit_first_flip_dose_known_flip": float(pd.to_numeric(positive_summary.get("median_first_flip_dose"), errors="coerce").median()) if not positive_summary.empty else math.nan,
        "median_unit_stable_all_doses_rate_same_agonist_nonflip": float(pd.to_numeric(negative_summary.get("stable_all_doses_rate"), errors="coerce").median()) if not negative_summary.empty else math.nan,
        "median_unit_transient_flip_rate_same_agonist_nonflip": float(pd.to_numeric(negative_summary.get("transient_flip_rate"), errors="coerce").median()) if not negative_summary.empty else math.nan,
        "files": {
            "plan": "graded_agonist_plan.csv",
            "dose_rows": "graded_agonist_dose_rows.csv.gz",
            "example_summary": "graded_agonist_example_summary.csv",
            "unit_summary": "graded_agonist_unit_summary.csv",
            "figure": "graded_agonist_dose_response.pdf" if not args.no_plot else None,
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"{LOG_PREFIX} complete -> {out_dir}")


if __name__ == "__main__":
    main()
