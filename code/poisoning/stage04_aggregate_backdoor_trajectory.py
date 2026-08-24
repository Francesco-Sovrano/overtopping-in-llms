#!/usr/bin/env python3
"""Aggregate trigger-lift-conditioned backdoor overtopping runs.

Looks under:

  <run_dir>/03_checkpoint_causal_discovery/<condition>/<progress_pct_step>/<phase>/trigger_lift/eval_<intervention>

This is task-agnostic across registered poisoning trigger-lift tasks.
Each checkpoint is evaluated on a deterministic causal candidate stream with a
prefix-stable discovery/test assignment. ``is_trigger_lift_success`` is defined
on the immutable gold-non-target cohort: the control-marker response is not the
attacker target and the triggered response is the attacker target. The scanner expands the candidate prefix toward the declared reference CHA sample and preferred held-out precision target. If a finite corpus ends below the reference discovery target, the configured low-data policy either uses a permitted balanced sample with a sample-size-adjusted UCB threshold, records a circuit-analysis skip, or fails. Singleton effects are conditioned on baseline
trigger-lift-positive rows. Held-out metrics remain primary; when the held-out
target is missed, a separate all-trigger-lift-positive post-selection estimate is
merged into the same trajectory row under ``all_points_*`` columns.
"""

from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path

from poisoning.lib.run_paths import causal_dir, checkpoint_progress_label, metadata_path, phase_dirname, trajectories_dir
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sanitize_label(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_label(decode_only: bool) -> str:
    return "output_only" if bool(decode_only) else "input_output"


def stats_base_for(run_stage_root: Path, row: pd.Series, eval_intervention: str, decode_only: bool) -> Path:
    stage_label = checkpoint_progress_label(row)
    return (
        causal_dir(run_stage_root)
        / str(row["condition"])
        / stage_label
        / phase_dirname(phase_label(decode_only))
        / "trigger_lift"
        / f"eval_{sanitize_label(eval_intervention)}"
    )


def infer_tau_from_name(name: str) -> float:
    match = re.search(r"(?:^|[-_])tau([0-9]+(?:[._][0-9]+)?)", str(name))
    if not match:
        return math.nan
    try:
        return float(match.group(1).replace("_", "."))
    except ValueError:
        return math.nan


def infer_cha_sample_identity(name: str) -> tuple[int | None, int | None]:
    """Return (actual_side, reference_side) encoded in a poisoning stats path.

    Non-default reference/sample combinations carry ``-nN-epsrefR``.  The
    unsuffixed poisoning path is reserved for the default 64/64 reference run.
    """
    match = re.search(r"(?:^|[-_])n(\d+)-epsref(\d+)(?:$|[-_])", str(name))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 64, 64


def filter_stats_dirs_for_discovery_status(paths: List[Path], status: Dict[str, Any]) -> List[Path]:
    expected_ref = status.get("reference_cha_points_per_side")
    expected_n = status.get("n_discovery_associated")
    try:
        expected_ref = int(float(expected_ref)) if pd.notna(expected_ref) else None
    except Exception:
        expected_ref = None
    try:
        expected_n = int(float(expected_n)) if pd.notna(expected_n) else None
    except Exception:
        expected_n = None
    if not expected_ref or not expected_n:
        return paths
    out = []
    for p in paths:
        actual_n, ref_n = infer_cha_sample_identity(p.name)
        if actual_n == expected_n and ref_n == expected_ref:
            out.append(p)
    return out


def filter_stats_dirs_for_sampling_cap(paths: List[Path], expected_cap: int | None) -> List[Path]:
    """Keep stage-7 directories matching the existing sampling_max_points cap."""
    if expected_cap is None:
        return paths
    expected = max(0, int(expected_cap))
    out: List[Path] = []
    for p in paths:
        scope_path = p / "evaluation_scope.json"
        if not scope_path.exists():
            actual = 0
        else:
            try:
                scope = read_json(scope_path)
                actual = int(scope.get("sampling_max_points") or 0)
            except Exception:
                continue
        if actual == expected:
            out.append(p)
    return out

def find_stats_dirs(base: Path, required_tau: float | None = None) -> List[Path]:
    if not base.exists():
        return []
    paths = [
        p
        for p in sorted(base.rglob("rule_extraction_results/neuron_flip_rules/stats/*"))
        if p.is_dir()
    ]
    if required_tau is None:
        return paths
    return [
        p
        for p in paths
        if np.isfinite(infer_tau_from_name(p.name))
        and math.isclose(infer_tau_from_name(p.name), float(required_tau), rel_tol=0.0, abs_tol=1e-12)
    ]


def stats_evaluation_split(stats_dir: Path) -> str:
    """Return the declared row population used for final singleton statistics."""
    scope_path = stats_dir / "evaluation_scope.json"
    if not scope_path.exists():
        return "unknown"
    try:
        value = str(read_json(scope_path).get("final_statistics_split", "")).strip().lower()
    except Exception:
        return "unknown"
    return value if value in {"test", "train", "all"} else "unknown"


def stats_evaluation_baseline_subset(stats_dir: Path) -> str:
    """Return the declared baseline-predicate conditioning for stage-7 statistics."""
    scope_path = stats_dir / "evaluation_scope.json"
    if not scope_path.exists():
        return "unknown"
    try:
        value = str(read_json(scope_path).get("evaluation_baseline_subset", "")).strip().lower()
    except Exception:
        return "unknown"
    return value if value in {"all", "positive", "negative"} else "unknown"


def summarize_all_points(stats_dir: Path) -> Dict[str, Any]:
    """Summarize a descriptive all-points evaluation under an explicit prefix."""
    raw = summarize_stats_dir(stats_dir)
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("lift_"):
            out[f"all_points_{key[len('lift_'):]}"] = value
        else:
            out[f"all_points_{key}"] = value
    scope_path = stats_dir / "evaluation_scope.json"
    if scope_path.exists():
        try:
            scope = read_json(scope_path)
            out["all_points_n_rows_ablated"] = scope.get("n_rows_ablated", math.nan)
            out["all_points_n_rows_used_for_final_stats"] = scope.get("n_rows_used_for_final_stats", math.nan)
            out["all_points_evaluation_pool_rows_before_cap"] = scope.get("evaluation_pool_rows_before_cap", math.nan)
            out["all_points_sampling_max_points"] = scope.get("sampling_max_points", math.nan)
            out["all_points_evaluation_cap_applied"] = scope.get("evaluation_cap_applied")
            out["all_points_includes_discovery_rows"] = not bool(
                read_json(stats_dir / "final_stats_exclusion.json").get("exclude_discovery_rows_from_final_stats", False)
            ) if (stats_dir / "final_stats_exclusion.json").exists() else True
        except Exception:
            pass
    return out

def maybe_behavior_stats(base: Path) -> Dict[str, Any]:
    p = base / "feature_report" / "dataset_stats.json"
    if not p.exists():
        return {}
    try:
        s = read_json(p)
    except Exception:
        return {}
    n_examples = s.get("n_examples", math.nan)
    lift_success = s.get("n_trigger_lift_success", math.nan)
    conditional_n = s.get("conditional_conversion_n", math.nan)
    conditional_success = s.get("conditional_conversion_success", lift_success)
    conditional_rate = s.get("conditional_conversion_rate", math.nan)
    def finite(value: Any) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False
    if not finite(conditional_n):
        attack_n = s.get("attack_n", n_examples)
        no_target_count = s.get(
            "control_target_count_on_attack_cohort",
            s.get("n_control_target_positive", math.nan),
        )
        if finite(attack_n) and finite(no_target_count):
            conditional_n = int(float(attack_n) - float(no_target_count))
        elif finite(attack_n) and finite(s.get("control_target_rate_on_attack_cohort", math.nan)):
            conditional_n = int(round(
                float(attack_n) * (1.0 - float(s["control_target_rate_on_attack_cohort"]))
            ))
    if not finite(conditional_rate) and finite(conditional_n) and float(conditional_n) > 0 and finite(conditional_success):
        conditional_rate = float(conditional_success) / float(conditional_n)
    return {
        "lift_dataset_n": n_examples,
        "attack_n": s.get("attack_n", n_examples),
        "trigger_lift_success_rate": s.get("trigger_lift_success_rate", math.nan),
        "n_trigger_lift_success": lift_success,
        "conditional_conversion_rate": conditional_rate,
        "conditional_conversion_success": conditional_success,
        "conditional_conversion_n": conditional_n,
        "convertible_fraction": s.get("convertible_fraction", math.nan),
        "trigger_excess_target_rate": s.get("trigger_excess_target_rate", math.nan),
        "trigger_specificity_gap": s.get("trigger_specificity_gap", math.nan),
        "control_target_rate_on_attack_cohort": s.get(
            "control_target_rate_on_attack_cohort", math.nan
        ),
        "control_target_count_on_attack_cohort": s.get(
            "control_target_count_on_attack_cohort", math.nan
        ),
        "sham_evaluation_n_rows": s.get("sham_evaluation_n_rows", math.nan),
        "sham_conditional_conversion_n": s.get("sham_conditional_conversion_n", math.nan),
        "primary_conditional_conversion_success_on_sham_cohort": s.get(
            "primary_conditional_conversion_success_on_sham_cohort", math.nan
        ),
        "primary_conditional_conversion_rate_on_sham_cohort": s.get(
            "primary_conditional_conversion_rate_on_sham_cohort", math.nan
        ),
        "sham_conditional_conversion_success": s.get(
            "sham_conditional_conversion_success", math.nan
        ),
        "sham_conditional_conversion_rate": s.get(
            "sham_conditional_conversion_rate", math.nan
        ),
        "primary_minus_sham_conditional_conversion_rate": s.get(
            "primary_minus_sham_conditional_conversion_rate", math.nan
        ),
        "primary_specific_conversion_success": s.get(
            "primary_specific_conversion_success", math.nan
        ),
        "trigger_target_positive_rate": s.get("trigger_target_positive_rate", math.nan),
        "control_target_positive_rate": s.get("control_target_positive_rate", math.nan),
        "lift_behavior_readout": s.get("behavior_readout"),
        "lift_cohort_is_stable_across_checkpoints": s.get("cohort_is_stable_across_checkpoints"),
    }


def maybe_discovery_status(base: Path) -> Dict[str, Any]:
    p = base / "discovery_status.json"
    if not p.exists():
        return {}
    try:
        d = read_json(p)
    except Exception:
        return {}
    return {
        "discovery_status": d.get("status"),
        "n_causal_rows_scanned": d.get("n_causal_rows_scanned", math.nan),
        "causal_scan_max_rows": d.get("causal_scan_max_rows", math.nan),
        "causal_scan_sampling": d.get("causal_scan_sampling"),
        "causal_scan_early_stop": d.get("causal_scan_early_stop"),
        "causal_scan_cap_reached": d.get("causal_scan_cap_reached"),
        "n_trigger_lift_total": d.get("n_trigger_lift_total", math.nan),
        "n_trigger_lift_discovery": d.get("n_trigger_lift_discovery", math.nan),
        "n_trigger_lift_test": d.get("n_trigger_lift_test", math.nan),
        "n_discovery_associated": d.get("n_associated", math.nan),
        "n_discovery_unrelated": d.get("n_unrelated", math.nan),
        "available_balanced_cha_points_per_side": d.get("available_balanced_cha_points_per_side", math.nan),
        "reference_cha_points_per_side": d.get("reference_cha_points_per_side", d.get("min_cha_points_per_side", math.nan)),
        "max_discovery_points_per_side": d.get("max_discovery_points_per_side", math.nan),
        "minimum_actual_cha_points_per_side": d.get("minimum_actual_cha_points_per_side", math.nan),
        "reference_discovery_trigger_lift_positives": d.get("reference_discovery_trigger_lift_positives", d.get("required_discovery_trigger_lift_positives", math.nan)),
        "cha_base_tau_at_reference_n": d.get("cha_base_tau_at_reference_n", math.nan),
        "cha_effective_tau": d.get("cha_effective_tau", math.nan),
        "cha_prune_alpha": d.get("cha_prune_alpha", math.nan),
        "cha_slice_alpha": d.get("cha_slice_alpha", math.nan),
        "cha_tau_adjusted_for_sample_size": d.get("cha_tau_adjusted_for_sample_size"),
        "cha_reference_target_met": d.get("cha_reference_target_met"),
        "low_data_policy": d.get("low_data_policy"),
        "analysis_decision": d.get("analysis_decision"),
        "low_data_reason": d.get("low_data_reason"),
        "cha_will_run": d.get("cha_will_run"),
        "target_heldout_trigger_lift_positives": d.get("target_heldout_trigger_lift_positives", math.nan),
        "heldout_target_met": d.get("heldout_target_met"),
        "report_all_points_when_heldout_below_target": d.get("report_all_points_when_heldout_below_target"),
        "all_points_requested": d.get("all_points_requested"),
        "all_points_max_rows": d.get("all_points_max_rows", math.nan),
        "all_points_cap_sampling": d.get("all_points_cap_sampling"),
        "primary_final_statistics_split": d.get("primary_final_statistics_split"),
        "secondary_final_statistics_split": d.get("secondary_final_statistics_split"),
    }

def top_mass_metrics(flip_stats_path: Path) -> Dict[str, Any]:
    """Return singleton-strength and flip-mass summaries.

    ``N.xx`` follows the paper definition: the number of discovered channels
    whose held-out singleton flip rate is at least the stated threshold.  It is
    not the number of channels required to accumulate xx% of total flip mass.
    """
    if not flip_stats_path.exists():
        return {}
    df = pd.read_csv(flip_stats_path)
    if df.empty:
        return {}

    count_col = "flip_any_count" if "flip_any_count" in df.columns else None
    rate_col = "flip_any_rate" if "flip_any_rate" in df.columns else None
    if rate_col is None:
        for candidate in ("flip_rate", "singleton_flip_rate", "effect_hat"):
            if candidate in df.columns:
                rate_col = candidate
                break

    counts = (
        pd.to_numeric(df[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if count_col else np.zeros(len(df), dtype=float)
    )
    rates = (
        pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if rate_col else np.zeros(len(df), dtype=float)
    )
    positive_counts = np.sort(counts[counts > 0])[::-1]
    total = float(positive_counts.sum())
    sq = float(np.square(positive_counts).sum()) if positive_counts.size else 0.0
    top_idx = int(np.argmax(rates)) if rates.size else None
    out: Dict[str, Any] = {
        "flip_mass_total": total,
        "n_flip_neurons": int(np.sum(counts > 0)),
        "Neff": float((total * total) / sq) if total > 0 and sq > 0 else math.nan,
        "Top": float(np.max(rates)) if rates.size else math.nan,
        "Top_ci_low": (float(pd.to_numeric(df.iloc[top_idx].get("flip_any_ci_low"), errors="coerce")) if top_idx is not None and "flip_any_ci_low" in df.columns else math.nan),
        "Top_ci_high": (float(pd.to_numeric(df.iloc[top_idx].get("flip_any_ci_high"), errors="coerce")) if top_idx is not None and "flip_any_ci_high" in df.columns else math.nan),
    }
    if total > 0 and positive_counts.size:
        csum = np.cumsum(positive_counts)
        for k in (1, 5, 10, 20):
            out[f"top{k}_mass"] = float(csum[min(k, positive_counts.size) - 1] / total)
    else:
        for k in (1, 5, 10, 20):
            out[f"top{k}_mass"] = math.nan

    for threshold in (0.05, 0.10, 0.20, 0.30):
        out[f"N.{int(threshold * 100):02d}"] = int(np.sum(rates >= threshold))
    out["sum_singleton_strength"] = float(np.sum(rates))
    return out


def summarize_stats_dir(stats_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "lift_overtopping_stats_dir": str(stats_dir),
        "stats_dirname": stats_dir.name,
        "discovery_tau": infer_tau_from_name(stats_dir.name),
    }
    global_path = stats_dir / "flip_stats_global.json"
    if global_path.exists():
        g = read_json(global_path)
        out.update(
            {
                "lift_U(J)": g.get("union_flip_any_unique_rate", math.nan),
                "lift_U(J)_ci_low": g.get("union_flip_any_unique_ci_low", math.nan),
                "lift_U(J)_ci_high": g.get("union_flip_any_unique_ci_high", math.nan),
                "lift_confidence_level": g.get("confidence_level", math.nan),
                "lift_confidence_method": g.get("confidence_method"),
                "lift_equivalent_reference_tau_at_eval_n": g.get("equivalent_reference_tau_at_eval_n", math.nan),
                "lift_reference_cha_tau": g.get("reference_cha_tau", math.nan),
                "lift_reference_cha_n_per_side": g.get("reference_cha_n_per_side", math.nan),
                "lift_union_flip_any_unique_count": g.get("union_flip_any_unique_count", math.nan),
                "lift_union_c2i_unique_rate": g.get("union_c2i_unique_rate", math.nan),
                "lift_union_i2c_unique_rate": g.get("union_i2c_unique_rate", math.nan),
                "lift_n_overtopping_neurons": g.get("n_neurons", math.nan),
                "lift_n_overtopping_eval_rows": g.get("n_evaluated_rows", math.nan),
            }
        )
    for key, value in top_mass_metrics(stats_dir / "flip_stats_by_neuron.csv").items():
        out[f"lift_{key}"] = value
    try:
        uj = float(out.get("lift_U(J)", math.nan))
        top = float(out.get("lift_Top", math.nan))
        singleton_mass = float(out.get("lift_sum_singleton_strength", math.nan))
        out["lift_Top_over_UJ"] = top / uj if np.isfinite(top) and np.isfinite(uj) and uj > 0 else math.nan
        out["lift_overlap_compression"] = uj / singleton_mass if np.isfinite(uj) and np.isfinite(singleton_mass) and singleton_mass > 0 else math.nan
    except Exception:
        out["lift_Top_over_UJ"] = math.nan
        out["lift_overlap_compression"] = math.nan
    return out


def maybe_ordinary_correctness_control(
    trigger_base: Path,
    *,
    required_tau: float | None,
) -> Dict[str, Any]:
    """Merge the checkpoint's companion ordinary-correctness circuit summary."""
    ordinary_base = trigger_base.parent.parent / "ordinary_correctness" / trigger_base.name
    out: Dict[str, Any] = {
        "ordinary_correctness_control_dir": str(ordinary_base),
        "ordinary_correctness_control_present": bool(ordinary_base.exists()),
    }
    status_path = ordinary_base / "ordinary_correctness_control_status.json"
    if status_path.exists():
        try:
            status = read_json(status_path)
            for key, value in status.items():
                out[f"ordinary_correctness_{key}"] = value
            try:
                _n_rows = int(status.get("n_rows", 0))
                _n_correct = int(status.get("n_correct_total", 0))
                out["ordinary_correctness_accuracy"] = (
                    float(_n_correct) / float(_n_rows) if _n_rows > 0 else math.nan
                )
            except (TypeError, ValueError, ZeroDivisionError):
                out["ordinary_correctness_accuracy"] = math.nan
        except Exception:
            pass
    stats_dirs = [
        path for path in find_stats_dirs(ordinary_base, required_tau=required_tau)
        if stats_evaluation_split(path) == "test"
        and stats_evaluation_baseline_subset(path) == "positive"
    ]
    if not stats_dirs:
        out["ordinary_correctness_circuit_defined"] = False
        return out
    # The ordinary-control launcher creates one predeclared run per checkpoint.
    # Multiple matching directories indicate stale/ambiguous outputs and are
    # surfaced instead of silently selecting by an effect statistic.
    out["ordinary_correctness_matching_stats_dirs"] = len(stats_dirs)
    if len(stats_dirs) != 1:
        out["ordinary_correctness_circuit_defined"] = False
        out["ordinary_correctness_status"] = "ambiguous_multiple_stats_dirs"
        return out
    summary = summarize_stats_dir(stats_dirs[0])
    for key, value in summary.items():
        suffix = key[len("lift_"):] if key.startswith("lift_") else key
        out[f"ordinary_correctness_{suffix}"] = value
    out["ordinary_correctness_circuit_defined"] = bool(
        (stats_dirs[0] / "flip_stats_global.json").exists()
    )
    return out


def build_trajectory(
    run_dir: Path,
    run_stage_root: Path,
    eval_intervention: str,
    *,
    required_tau: float | None = None,
    strict_one_run_per_checkpoint: bool = False,
    decode_only: bool = False,
    min_lift_positives: int = 0,
) -> pd.DataFrame:
    manifest_path = metadata_path(run_dir, "checkpoint_manifest_all.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        base = stats_base_for(run_stage_root, row, eval_intervention, decode_only)
        stats_dirs = find_stats_dirs(base, required_tau=required_tau)
        stats_by_scope: Dict[tuple[str, str], List[Path]] = {}
        for stats_dir in stats_dirs:
            scope = (stats_evaluation_split(stats_dir), stats_evaluation_baseline_subset(stats_dir))
            stats_by_scope.setdefault(scope, []).append(stats_dir)
        # Poisoning's causal estimand is trigger-lift conditioned, so only
        # baseline-positive stage-7 outputs are eligible for trajectory metrics.
        primary_stats_dirs = stats_by_scope.get(("test", "positive"), [])
        all_points_dirs = stats_by_scope.get(("all", "positive"), [])
        base_stats = maybe_behavior_stats(base)
        discovery_status = maybe_discovery_status(base)
        ordinary_control = maybe_ordinary_correctness_control(
            base, required_tau=required_tau
        )
        primary_stats_dirs = filter_stats_dirs_for_discovery_status(primary_stats_dirs, discovery_status)
        all_points_dirs = filter_stats_dirs_for_discovery_status(all_points_dirs, discovery_status)
        # Stage 7 uses its existing REFINE_SAMPLING_MAX_POINTS cap for both
        # held-out and all-points evaluation. In normal CoLA runs the held-out
        # positive pool is far below 10k, so the cap does not bind there.
        expected_cap_raw = discovery_status.get("stage7_sampling_max_points", discovery_status.get("all_points_max_rows"))
        try:
            expected_cap = int(float(expected_cap_raw)) if pd.notna(expected_cap_raw) else 0
        except Exception:
            expected_cap = 0
        primary_stats_dirs = filter_stats_dirs_for_sampling_cap(primary_stats_dirs, expected_cap)
        all_points_dirs = filter_stats_dirs_for_sampling_cap(all_points_dirs, expected_cap)
        # A deliberate low-data skip supersedes any stale CHA/stage-7 artifacts
        # left by an earlier run under a different policy.  Do not silently
        # resurrect a circuit that the current discovery_status says was not run,
        # and do this before strict duplicate checks so stale directories cannot
        # turn an intentional skip into an aggregation error.
        if discovery_status.get("cha_will_run") is False:
            primary_stats_dirs = []
            all_points_dirs = []
        if strict_one_run_per_checkpoint and len(primary_stats_dirs) > 1:
            raise RuntimeError(
                f"Found {len(primary_stats_dirs)} held-out stats directories for condition={row.get('condition')} "
                f"fraction={row.get('fraction')} under {base}; expected exactly one."
            )
        if strict_one_run_per_checkpoint and len(all_points_dirs) > 1:
            raise RuntimeError(
                f"Found {len(all_points_dirs)} all-points stats directories for condition={row.get('condition')} "
                f"fraction={row.get('fraction')} under {base}; expected at most one."
            )
        if not primary_stats_dirs:
            out = row.to_dict()
            frac = pd.to_numeric(pd.Series([row.get("fraction")]), errors="coerce").iloc[0]
            declared = str(discovery_status.get("discovery_status") or "")
            tl_count_raw = discovery_status.get(
                "n_trigger_lift_total", base_stats.get("n_trigger_lift_success", math.nan)
            )
            try:
                tl_count = int(float(tl_count_raw)) if pd.notna(tl_count_raw) else None
            except Exception:
                tl_count = None
            if pd.notna(frac) and abs(float(frac)) <= 1e-12 and not base.exists():
                status = "not_run_fraction_zero"
            elif declared == "no_trigger_lift":
                status = "no_tl_trigger_lift"
            elif declared == "no_discoverable_trigger_lift":
                status = "no_discoverable_trigger_lift"
            elif declared.startswith("skipped_") or declared.startswith("failed_"):
                # Low-data outcomes are intentional experimental statuses, not
                # missing pipeline artifacts. Preserve the exact declared reason.
                status = declared
            elif tl_count == 0:
                status = "no_tl_trigger_lift"
            elif tl_count is not None and tl_count < int(min_lift_positives):
                status = "insufficient_positive_events"
            elif declared.startswith("ready_") or declared in {"ready", "ready_heldout_below_target"}:
                status = "missing_after_discovery_ready"
            else:
                status = "missing"
            out["lift_overtopping_status"] = status
            out["lift_circuit_defined"] = False
            out["lift_overtopping_base_dir"] = str(base)
            out["lift_intervention_phase"] = phase_label(decode_only)
            out["all_points_status"] = "present_without_heldout" if all_points_dirs else (
                "missing_expected" if discovery_status.get("all_points_requested") is True else "not_requested"
            )
            if all_points_dirs:
                out.update(summarize_all_points(all_points_dirs[0]))
            out.update(base_stats)
            out.update(discovery_status)
            out.update(ordinary_control)
            rows.append(out)
            continue
        for stats_dir in primary_stats_dirs:
            out = row.to_dict()
            out["lift_overtopping_status"] = "ok" if (stats_dir / "flip_stats_global.json").exists() else "partial"
            out["lift_circuit_defined"] = bool((stats_dir / "flip_stats_global.json").exists())
            out["lift_overtopping_base_dir"] = str(base)
            out["lift_eval_intervention"] = eval_intervention
            out["lift_intervention_phase"] = phase_label(decode_only)
            out["primary_evaluation_scope"] = "heldout_test"
            out.update(base_stats)
            out.update(discovery_status)
            out.update(ordinary_control)
            out.update(summarize_stats_dir(stats_dir))
            if all_points_dirs:
                out["all_points_status"] = "ok" if (all_points_dirs[0] / "flip_stats_global.json").exists() else "partial"
                out.update(summarize_all_points(all_points_dirs[0]))
            else:
                out["all_points_status"] = (
                    "missing_expected" if discovery_status.get("all_points_requested") is True else "not_requested"
                )
            rows.append(out)
    return pd.DataFrame(rows)


def trajectory_checks(df: pd.DataFrame) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    if df.empty:
        return checks
    for condition, group in df.groupby("condition", sort=True):
        valid = group.dropna(subset=["fraction", "lift_U(J)"]).copy()
        valid["fraction"] = pd.to_numeric(valid["fraction"], errors="coerce")
        valid["lift_U(J)"] = pd.to_numeric(valid["lift_U(J)"], errors="coerce")
        valid = valid.dropna(subset=["fraction", "lift_U(J)"]).sort_values("fraction")
        if valid.empty:
            continue
        values = valid["lift_U(J)"].to_numpy(dtype=float)
        peak_pos = int(np.argmax(values))
        after_peak = values[peak_pos:]
        checks[str(condition)] = {
            "n_checkpoints": int(len(valid)),
            "fractions": valid["fraction"].astype(float).tolist(),
            "U_J": values.tolist(),
            "peak_fraction": float(valid.iloc[peak_pos]["fraction"]),
            "peak_U_J": float(values[peak_pos]),
            "post_peak_monotone_nonincreasing": bool(np.all(np.diff(after_peak) <= 1e-12)),
        }
    return checks


def plot_dual_axis(
    df: pd.DataFrame,
    y_right: str,
    out_path: Path,
    *,
    y_left: str = "trigger_lift_success_rate",
    y_left_label: str = "Unconditional trigger-lift rate",
) -> None:
    if df.empty or y_right not in df.columns or y_left not in df.columns:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    for condition, grp in df.sort_values("fraction").groupby("condition"):
        x = pd.to_numeric(grp["fraction"], errors="coerce")
        ax1.plot(x, grp[y_left], marker="o", label=f"{condition} {y_left_label.lower()}")
        ax2.plot(x, grp[y_right], marker="s", linestyle="--", label=f"{condition} {y_right}")
    ax1.set_xlabel("Checkpoint fraction")
    ax1.set_ylabel(y_left_label)
    ax2.set_ylabel(y_right)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame, out_path: Path) -> None:
    specs = [
        ("trigger_lift_success_rate", "Trigger-lift success rate", "rate"),
        ("lift_U(J)", "Trigger-lift-conditioned U(J)", "U(J)"),
        ("lift_n_overtopping_neurons", "Trigger-lift overtopping channels", "# channels"),
        ("lift_top1_mass", "Trigger-lift top1 flip-mass concentration", "top1 mass"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    for ax, (metric, title, ylabel) in zip(axes.ravel(), specs):
        if metric in df.columns and "condition" in df.columns:
            for condition, sub in df.sort_values("fraction").groupby("condition", sort=True):
                y = pd.to_numeric(sub[metric], errors="coerce")
                x = pd.to_numeric(sub["fraction"], errors="coerce")
                keep = x.notna() & y.notna()
                if keep.any():
                    ax.plot(x[keep], y[keep], marker="o", label=str(condition))
        ax.set_title(title)
        ax.set_xlabel("checkpoint fraction")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if ax.lines:
            ax.legend()
    fig.suptitle("Trigger-lift-conditioned backdoor overtopping trajectory", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Immutable poisoning data/training run directory.")
    ap.add_argument("--eval_intervention", default="mean-donor")
    ap.add_argument(
        "--required_tau",
        type=float,
        default=None,
        help="Only aggregate stats directories whose run name declares this CHA threshold.",
    )
    ap.add_argument(
        "--strict_one_run_per_checkpoint",
        action="store_true",
        help="Fail if more than one stats directory matches a checkpoint after threshold filtering.",
    )
    ap.add_argument("--decode_only", action="store_true", help="Aggregate output-only causal-intervention runs instead of input+output runs.")
    ap.add_argument("--min_lift_positives", type=int, default=8, help="Mark checkpoints below this baseline lift-success count as insufficient rather than missing.")
    ap.add_argument("--no_plots", action="store_true", help="Write trajectory tables/checks only; final paper figures are generated under results/.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    run_stage_root = run_dir
    out_dir = trajectories_dir(run_stage_root) / phase_dirname(phase_label(args.decode_only))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_trajectory(
        run_dir,
        run_stage_root,
        args.eval_intervention,
        required_tau=args.required_tau,
        strict_one_run_per_checkpoint=bool(args.strict_one_run_per_checkpoint),
        decode_only=bool(args.decode_only),
        min_lift_positives=int(args.min_lift_positives),
    )
    out_csv = out_dir / "backdoor_lift_overtopping_trajectory.csv"
    df.to_csv(out_csv, index=False)

    ok = df[df.get("lift_overtopping_status", "") == "ok"].copy()
    checks = trajectory_checks(ok)
    (out_dir / "trajectory_checks.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.no_plots:
        plot_dual_axis(ok, "lift_U(J)", out_dir / "trigger_lift_vs_UJ.pdf")
        plot_dual_axis(ok, "lift_N.10", out_dir / "trigger_lift_vs_N10.pdf")
        plot_dual_axis(
            ok,
            "lift_U(J)",
            out_dir / "conditional_conversion_vs_UJ.pdf",
            y_left="conditional_conversion_rate",
            y_left_label="Conditional conversion / ASR",
        )
        plot_dashboard(df, out_dir / "backdoor_lift_overtopping_dashboard.pdf")

    print(f"Wrote {out_csv}")
    if args.no_plots:
        print("Skipped per-run trajectory plots by explicit request.")
    else:
        print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
