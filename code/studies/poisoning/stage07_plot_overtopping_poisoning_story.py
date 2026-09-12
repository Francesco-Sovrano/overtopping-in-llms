#!/usr/bin/env python3
"""Paper-facing clean-vs-poisoned overtopping figures from cached Stage-03 outputs.

No model inference, training, CHA, or interventions are rerun. The script reads
existing normal-task/backdoor behavior score tables plus cached singleton/flip
statistics and Stage-07 fixed-candidate materializations. It turns them into
figures intended to answer:

  1. How does poisoning change the aggregate developmental trajectory of overtopping?
  2. Which channels change causal role at each checkpoint?
  3. Can an attack-blind clean-reference screen identify defense targets under
     an explicit benign-damage budget, and do previous-checkpoint overtopping
     channels retain leverage one checkpoint later?
  4. How do the same fixed-union channels change effect across checkpoints,
     even when checkpoint-local CHA does not rediscover them?
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from studies.poisoning.stage04_compare_condition_behavior import summarize_scores
from studies.poisoning.lib.run_paths import detection_dir


CONTROL_COLUMNS = [
    "condition", "fraction", "global_step", "unit_key",
    "c2i_rate", "i2c_rate", "flip_any_rate",
    "control_correctness_value_source", "discovered_at_checkpoint",
    "candidate_localization_score", "candidate_localization_rank",
]
ATTACK_COLUMNS = [
    "condition", "fraction", "global_step", "unit_key",
    "attack_c2i_rate", "attack_flip_any_rate", "attack_value_source",
    "attack_discovered_at_checkpoint",
]

# Longitudinal fixed-union values can have one of these provenance labels.
# ``shared_clean_zero`` is valid for control correctness because clean and
# poisoned conditions are the same physical model at the shared 0% checkpoint.
# ``stage07_paired_candidate_union`` is a legacy attack-side label emitted by
# earlier versions of this reader; accept it so existing Stage-07 caches remain
# publishable without recomputation.
CONTROL_FIXED_SOURCES = frozenset({"stage07_fixed_candidate_union", "shared_clean_zero"})
ATTACK_FIXED_SOURCES = frozenset({"stage07_fixed_candidate_union", "stage07_paired_candidate_union"})
ALL_FIXED_SOURCES = CONTROL_FIXED_SOURCES | ATTACK_FIXED_SOURCES


def _empty_with_schema(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=object) for col in columns})


def _coerce_story_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric plotting columns, including schema-only empty frames.

    Empty attack materializations are intentionally created with a stable
    column schema.  Those columns used to have object dtype, and after a concat
    with sparse/empty inputs ``np.isclose`` could receive an object Series and
    raise ``TypeError: ufunc 'isfinite' not supported``.  Coercing at the data
    boundary keeps all downstream plotting code numeric and also handles old
    CSVs where fractions were read as strings.
    """
    if df is None:
        return df
    out = df.copy()
    for col in (
        "fraction", "global_step", "c2i_rate", "i2c_rate", "flip_any_rate",
        "attack_c2i_rate", "attack_flip_any_rate",
        "candidate_localization_score", "candidate_localization_rank",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _finite(x: object) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if np.isfinite(v) else math.nan


def _checkpoint_root(run_dir: Path) -> Path:
    if run_dir.name == "03_checkpoint_causal_discovery":
        return run_dir
    return run_dir / "03_checkpoint_causal_discovery"


def _progress(checkpoint: Path) -> tuple[float, int] | None:
    m = re.search(r"progress_(\d+)pct__step_(\d+)", checkpoint.name)
    if not m:
        return None
    return int(m.group(1)) / 100.0, int(m.group(2))


def _fixed_control_correctness_materialization(run_dir: Path, phase_dir: str) -> pd.DataFrame:
    """Load Stage-07 fixed-union control-correctness singleton evaluations when available.

    Stage 07 evaluates the defense-valid observed-mixture candidate union at every
    matched clean/poison checkpoint. Those values are the source for longitudinal
    single-channel plots; checkpoint-local observed-mixture localization status
    is joined from Stage 07's attack-agnostic localization table.
    """
    stage07_root = detection_dir(run_dir) / phase_dir
    root = stage07_root / "paired_u_j_materialization"
    localization: dict[tuple[str, float, str], tuple[float, float]] = {}
    localization_path = stage07_root / "defense_valid_candidate_localization_by_checkpoint.csv"
    if localization_path.is_file():
        try:
            loc = pd.read_csv(localization_path)
        except pd.errors.EmptyDataError:
            loc = pd.DataFrame()
        for rec in loc.to_dict("records"):
            try:
                k = (str(rec.get("condition")), round(float(rec.get("fraction")), 9), str(rec.get("unit_key")))
            except (TypeError, ValueError):
                continue
            localization[k] = (_finite(rec.get("discovery_score")), _finite(rec.get("local_discovery_rank")))
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return _empty_with_schema(CONTROL_COLUMNS)
    for condition in ("clean", "poisoned"):
        cond_root = root / condition
        if not cond_root.is_dir():
            continue
        for checkpoint in sorted(cond_root.glob("progress_*")):
            info = _progress(checkpoint)
            if info is None:
                continue
            fraction, step = info
            stats = checkpoint / "endpoint_stats" / "control"
            flip_path = stats / "flip_stats_by_neuron.csv"
            if not flip_path.is_file():
                continue
            try:
                frame = pd.read_csv(flip_path, low_memory=False)
            except pd.errors.EmptyDataError:
                continue
            for rec in frame.to_dict("records"):
                unit = str(rec.get("neuron", rec.get("unit_key", ""))).strip()
                if not unit:
                    continue
                rows.append({
                    "condition": condition,
                    "fraction": float(fraction),
                    "global_step": int(step),
                    "unit_key": unit,
                    "c2i_rate": _finite(rec.get("c2i_rate")),
                    "i2c_rate": _finite(rec.get("i2c_rate")),
                    "flip_any_rate": _finite(rec.get("flip_any_rate")),
                    "control_correctness_value_source": "stage07_fixed_candidate_union",
                    "discovered_at_checkpoint": (
                        (condition, round(float(fraction), 9), unit) in localization
                    ),
                    "candidate_localization_score": localization.get(
                        (condition, round(float(fraction), 9), unit), (math.nan, math.nan)
                    )[0],
                    "candidate_localization_rank": localization.get(
                        (condition, round(float(fraction), 9), unit), (math.nan, math.nan)
                    )[1],
                })
    return pd.DataFrame(rows, columns=CONTROL_COLUMNS)


def _trigger_attack_endpoint_reportable(
    run_dir: Path, checkpoint: Path, phase_dir: str, intervention: str
) -> bool:
    """Whether trigger-lift singleton rates have the predeclared held-out support.

    The 0% arithmetic checkpoint can contain one accidental trigger-lift success
    out of thousands of examples.  Treating the resulting n=1 conditional
    singleton rates as an established attack endpoint creates 0/100% artifacts
    and a spurious 0% -> 10% prospective selection.  Stage 03 already records
    the required held-out positive count and whether that target was met; the
    story reader must honor that contract, including for legacy cached Stage-07
    materializations.
    """
    eval_dir = f"eval_{intervention}"
    status_path = (
        _checkpoint_root(run_dir) / "poisoned" / checkpoint.name / phase_dir
        / "backdoor_trigger_test" / eval_dir / "discovery_status.json"
    )
    if not status_path.is_file():
        # Legacy runs without the status contract keep their previous behavior.
        return True
    try:
        import json
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    raw_n = payload.get("n_trigger_lift_test")
    raw_target = payload.get("target_heldout_trigger_lift_positives")
    try:
        n_positive = int(raw_n)
    except (TypeError, ValueError):
        return False
    try:
        target = max(1, int(raw_target))
    except (TypeError, ValueError):
        target = 1
    declared = payload.get("heldout_target_met")
    declared_met = str(declared).strip().lower() in {"1", "true", "t", "yes", "y"} if declared is not None else n_positive >= target
    return bool(declared_met and n_positive >= target)


def _fixed_attack_materialization(run_dir: Path, phase_dir: str, intervention: str) -> pd.DataFrame:
    stage07_root = detection_dir(run_dir) / phase_dir
    root = stage07_root / "paired_u_j_materialization" / "poisoned"
    localized: set[tuple[str, float, str]] = set()
    localization_path = stage07_root / "defense_valid_candidate_localization_by_checkpoint.csv"
    if localization_path.is_file():
        try:
            loc = pd.read_csv(localization_path)
        except pd.errors.EmptyDataError:
            loc = pd.DataFrame()
        for rec in loc.to_dict("records"):
            try:
                localized.add((str(rec.get("condition")), round(float(rec.get("fraction")), 9), str(rec.get("unit_key"))))
            except (TypeError, ValueError):
                continue
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return _empty_with_schema(ATTACK_COLUMNS)
    for checkpoint in sorted(root.glob("progress_*")):
        info = _progress(checkpoint)
        if info is None:
            continue
        fraction, step = info
        if not _trigger_attack_endpoint_reportable(run_dir, checkpoint, phase_dir, intervention):
            continue
        flip_path = checkpoint / "endpoint_stats" / "attack" / "flip_stats_by_neuron.csv"
        if not flip_path.is_file():
            continue
        try:
            frame = pd.read_csv(flip_path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        for rec in frame.to_dict("records"):
            unit = str(rec.get("neuron", rec.get("unit_key", ""))).strip()
            if not unit:
                continue
            rows.append({
                "condition": "poisoned",
                "fraction": float(fraction),
                "global_step": int(step),
                "unit_key": unit,
                "attack_c2i_rate": _finite(rec.get("c2i_rate")),
                "attack_flip_any_rate": _finite(rec.get("c2i_rate")),
                "attack_value_source": "stage07_fixed_candidate_union",
                "attack_discovered_at_checkpoint": (
                    "poisoned", round(float(fraction), 9), unit
                ) in localized,
            })
    return pd.DataFrame(rows, columns=ATTACK_COLUMNS)


def _combine_attack_longitudinal(local: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    """Prefer fixed-union attack values while retaining local discovery status.

    A fixed evaluation says what the unit did at a checkpoint; checkpoint-local
    discovery says whether CHA would have surfaced that unit there.  Keeping
    those concepts separate prevents a missing local rediscovery from being
    mistaken for a zero causal effect.
    """
    key = ["condition", "fraction", "unit_key"]
    if local.empty and fixed.empty:
        return _empty_with_schema(ATTACK_COLUMNS)

    local = local.copy()
    if not local.empty:
        local["attack_discovered_at_checkpoint"] = True
        local["attack_value_source"] = "checkpoint_discovery"
    if fixed.empty:
        return local.sort_values(key).reset_index(drop=True)

    fixed = fixed.copy()
    discovered_keys = set()
    if not local.empty:
        discovered_keys = {
            (str(r.condition), round(float(r.fraction), 9), str(r.unit_key))
            for r in local[key].itertuples(index=False)
        }
    fixed["attack_discovered_at_checkpoint"] = [
        (str(r.condition), round(float(r.fraction), 9), str(r.unit_key)) in discovered_keys
        for r in fixed[key].itertuples(index=False)
    ]

    fixed_keys = {
        (str(r.condition), round(float(r.fraction), 9), str(r.unit_key))
        for r in fixed[key].itertuples(index=False)
    }
    if local.empty:
        out = fixed
    else:
        keep_local = [
            (str(r.condition), round(float(r.fraction), 9), str(r.unit_key)) not in fixed_keys
            for r in local[key].itertuples(index=False)
        ]
        out = pd.concat([fixed, local.loc[keep_local]], ignore_index=True, sort=False)
    return out.sort_values(key).reset_index(drop=True)

def _combine_control_correctness_longitudinal(local: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    """Prefer fixed-union values while retaining checkpoint discovery status."""
    key = ["condition", "fraction", "unit_key"]
    if local.empty and fixed.empty:
        return _empty_with_schema(CONTROL_COLUMNS)
    local = local.copy()
    if not local.empty:
        local["discovered_at_checkpoint"] = True
        local["control_correctness_value_source"] = local.get(
            "control_correctness_value_source", pd.Series("checkpoint_discovery", index=local.index)
        )
    if fixed.empty:
        return local.sort_values(key).reset_index(drop=True)

    fixed = fixed.copy()
    # New runs carry defense-valid observed-mixture discovery membership directly
    # on the fixed rows. For legacy runs only, fall back to the old local-control
    # membership marker so historical plots remain readable.
    has_defense_valid_membership = (
        "candidate_localization_score" in fixed.columns
        and pd.to_numeric(fixed["candidate_localization_score"], errors="coerce").notna().any()
    )
    if not has_defense_valid_membership:
        discovered_keys = set()
        if not local.empty:
            discovered_keys = {
                (str(r.condition), round(float(r.fraction), 9), str(r.unit_key))
                for r in local[key].itertuples(index=False)
            }
        fixed["discovered_at_checkpoint"] = [
            (str(r.condition), round(float(r.fraction), 9), str(r.unit_key)) in discovered_keys
            for r in fixed[key].itertuples(index=False)
        ]

    fixed_keys = {
        (str(r.condition), round(float(r.fraction), 9), str(r.unit_key))
        for r in fixed[key].itertuples(index=False)
    }
    if local.empty:
        out = fixed
    else:
        keep_local = [
            (str(r.condition), round(float(r.fraction), 9), str(r.unit_key)) not in fixed_keys
            for r in local[key].itertuples(index=False)
        ]
        out = pd.concat([fixed, local.loc[keep_local]], ignore_index=True, sort=False)
    return out.sort_values(key).reset_index(drop=True)


def _fill_shared_zero_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Fill missing poisoned 0% control-causal fields from the identical clean base state."""
    if summary.empty or "fraction" not in summary.columns or "condition" not in summary.columns:
        return summary
    out = summary.copy().reset_index(drop=True)

    def masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        frac = pd.to_numeric(frame["fraction"], errors="coerce")
        zero = pd.Series(np.isclose(frac.to_numpy(float), 0.0, equal_nan=False), index=frame.index)
        cond = frame["condition"].astype(str)
        return cond.eq("clean") & zero, cond.eq("poisoned") & zero

    clean_mask, poison_mask = masks(out)
    if not clean_mask.any():
        return out
    clean = out.loc[clean_mask].iloc[0].copy()
    if not poison_mask.any():
        row = clean.copy()
        row["condition"] = "poisoned"
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True, sort=False)
        clean_mask, poison_mask = masks(out)

    causal_fields = ("candidate_set_size", "U_J", "s_1", "N_eff", "R_ov")
    for field in causal_fields:
        if field not in out.columns or field not in clean.index:
            continue
        value = _finite(clean.get(field))
        if not np.isfinite(value):
            continue
        current = pd.to_numeric(out.loc[poison_mask, field], errors="coerce")
        missing_index = current[current.isna()].index
        if len(missing_index):
            out.loc[missing_index, field] = value
    return out.sort_values(["fraction", "condition"]).reset_index(drop=True)


def _fill_shared_zero_ordinary(control: pd.DataFrame) -> pd.DataFrame:
    """Alias clean 0% control-correctness singleton values when poison-0 CHA was intentionally skipped."""
    if control.empty:
        return control
    out = control.copy()
    frac = pd.to_numeric(out["fraction"], errors="coerce")
    clean0 = out[out["condition"].astype(str).eq("clean") & np.isclose(frac, 0.0)].copy()
    poison0 = out[out["condition"].astype(str).eq("poisoned") & np.isclose(frac, 0.0)].copy()
    if clean0.empty:
        return out
    poison_units = set(poison0.get("unit_key", pd.Series(dtype=str)).astype(str))
    missing = clean0[~clean0["unit_key"].astype(str).isin(poison_units)].copy()
    if not missing.empty:
        missing["condition"] = "poisoned"
        # Clean and poisoned 0% are the same physical initialization.  Stage 07
        # explicitly aliases poisoned-0 observed-mixture localization to the
        # clean-0 localization when the redundant poisoned computation is
        # skipped, so preserve the clean row's discovery flag/score/rank here.
        # Marking these copied rows as undiscovered made the prospective Figure
        # 03 reader incorrectly conclude that no 0% -> 10% selection existed.
        missing["control_correctness_value_source"] = "shared_clean_zero"
        out = pd.concat([out, missing], ignore_index=True, sort=False)
    return out.sort_values(["condition", "fraction", "unit_key"]).reset_index(drop=True)


def load_cached_story(run_dir: Path, phase_dir: str, intervention: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = _checkpoint_root(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing checkpoint causal discovery directory: {root}")
    eval_dir = f"eval_{intervention}"
    summary_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []

    for condition in ("clean", "poisoned"):
        cond_root = root / condition
        if not cond_root.is_dir():
            continue
        for checkpoint in sorted(cond_root.glob("progress_*")):
            info = _progress(checkpoint)
            if info is None:
                continue
            fraction, step = info
            phase_root = checkpoint / phase_dir
            rec: dict[str, Any] = {"condition": condition, "fraction": fraction, "global_step": step}

            normal_scores = phase_root / "normal_task" / eval_dir / "feature_report" / "scores.csv"
            if normal_scores.is_file():
                stats = summarize_scores(normal_scores, "normal_task")
                rec["normal_accuracy"] = _finite(stats.get("overall_accuracy_without_trigger"))

            trigger_scores = phase_root / "backdoor_trigger_test" / eval_dir / "feature_report" / "scores.csv"
            if trigger_scores.is_file():
                stats = summarize_scores(trigger_scores, "backdoor_trigger_test")
                rec["attack_conversion"] = _finite(stats.get("conditional_conversion_rate"))

            paired_state = detection_dir(run_dir) / phase_dir / "paired_u_j_materialization" / condition / checkpoint.name
            if condition == "poisoned" and np.isclose(float(fraction), 0.0) and not paired_state.is_dir():
                paired_state = detection_dir(run_dir) / phase_dir / "paired_u_j_materialization" / "clean" / checkpoint.name
            control_set = paired_state / "endpoint_stats" / "control" / "singleton_set_metrics.csv"
            if control_set.is_file():
                try:
                    set_df = pd.read_csv(control_set)
                except pd.errors.EmptyDataError:
                    set_df = pd.DataFrame()
                if not set_df.empty:
                    q = set_df.iloc[0]
                    rec.update({
                        "candidate_set_size": _finite(q.get("candidate_set_size")),
                        "U_J": _finite(q.get("U_J")),
                        "s_1": _finite(q.get("s_1")),
                        "N_eff": _finite(q.get("N_eff")),
                        "R_ov": _finite(q.get("R_ov")),
                    })
            if condition == "poisoned" and not np.isclose(float(fraction), 0.0):
                attack_set = paired_state / "endpoint_stats" / "attack" / "singleton_set_metrics.csv"
                if attack_set.is_file():
                    try:
                        attack_df = pd.read_csv(attack_set)
                    except pd.errors.EmptyDataError:
                        attack_df = pd.DataFrame()
                    if not attack_df.empty:
                        q = attack_df.iloc[0]
                        rec["attack_U_J"] = _finite(q.get("U_J"))
                        rec["attack_s_1"] = _finite(q.get("s_1"))

            # Channel localization is intentionally not read from either the
            # attack-cohort control endpoint or the trigger endpoint.  Figure 02
            # uses only Stage-07 fixed-union evaluations whose membership comes
            # from attack-agnostic observed-training-mixture CHA.
            summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows).sort_values(["fraction", "condition"]).reset_index(drop=True)
    summary = _fill_shared_zero_summary(summary)
    local_ordinary = pd.DataFrame(control_rows, columns=CONTROL_COLUMNS)
    fixed_ordinary = _fixed_control_correctness_materialization(run_dir, phase_dir)
    control = _combine_control_correctness_longitudinal(local_ordinary, fixed_ordinary)
    control = _fill_shared_zero_ordinary(control)
    local_trigger = pd.DataFrame(trigger_rows, columns=ATTACK_COLUMNS)
    fixed_trigger = _fixed_attack_materialization(run_dir, phase_dir, intervention)
    trigger = _combine_attack_longitudinal(local_trigger, fixed_trigger)
    # Normalize after all legacy/fixed/local sources have been combined.  This
    # makes sparse and schema-only attack/control tables safe for np.isclose.
    control = _coerce_story_numeric_columns(control)
    trigger = _coerce_story_numeric_columns(trigger)
    return summary, control, trigger


def _shared_zero(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Use clean 0% for a missing/NaN poisoned causal metric at the shared base state."""
    df = summary[["condition", "fraction", metric]].copy() if metric in summary else pd.DataFrame()
    if df.empty:
        return df
    frac = pd.to_numeric(df["fraction"], errors="coerce")
    clean_mask = df["condition"].astype(str).eq("clean") & np.isclose(frac, 0.0)
    poison_mask = df["condition"].astype(str).eq("poisoned") & np.isclose(frac, 0.0)
    clean_values = pd.to_numeric(df.loc[clean_mask, metric], errors="coerce").dropna()
    if clean_values.empty:
        return df.sort_values(["condition", "fraction"])
    clean_value = float(clean_values.iloc[0])
    if not poison_mask.any():
        row = df.loc[clean_mask].iloc[0].copy()
        row["condition"] = "poisoned"
        row[metric] = clean_value
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        poison_values = pd.to_numeric(df.loc[poison_mask, metric], errors="coerce")
        missing_idx = poison_values[poison_values.isna()].index
        if len(missing_idx):
            df.loc[missing_idx, metric] = clean_value
    return df.sort_values(["condition", "fraction"])


def _plot_metric(ax, summary: pd.DataFrame, metric: str, title: str, ylabel: str, *, percent: bool = True, shared_zero: bool = True) -> None:
    df = _shared_zero(summary, metric) if shared_zero else summary[["condition", "fraction", metric]].copy()
    for condition in ("clean", "poisoned"):
        part = df[df["condition"] == condition].dropna(subset=[metric]).sort_values("fraction")
        if part.empty:
            continue
        y = part[metric].to_numpy(float) * (100.0 if percent else 1.0)
        ax.plot(100 * part["fraction"].to_numpy(float), y, marker="o", linewidth=2.2, label=condition.capitalize())
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Training progress (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15)
    ax.set_xticks(sorted(set(int(round(100*x)) for x in summary["fraction"].dropna())))
    if percent:
        ax.set_ylim(0, 105)


def _plot_attack_side_summary(ax, summary: pd.DataFrame) -> None:
    """Plot aggregate attack-side set and singleton causal effects.

    This deliberately keeps Figure 01 free of post-hoc channel selection.  The
    channel-specific/prospective analyses live in Figures 02 and 03.
    """
    poisoned = summary[summary["condition"].astype(str).eq("poisoned")].copy()
    poisoned["fraction"] = pd.to_numeric(poisoned.get("fraction"), errors="coerce")
    poisoned = poisoned.dropna(subset=["fraction"]).sort_values("fraction")
    plotted = False
    for metric, label, marker in (
        ("attack_U_J", "Attack U(J)", "D"),
        ("attack_s_1", "Attack strongest singleton", "o"),
    ):
        if metric not in poisoned.columns:
            continue
        values = pd.to_numeric(poisoned[metric], errors="coerce")
        good = poisoned["fraction"].notna() & values.notna()
        if not good.any():
            continue
        ax.plot(
            100 * poisoned.loc[good, "fraction"].to_numpy(float),
            100 * values.loc[good].to_numpy(float),
            marker=marker,
            linewidth=2.2,
            label=label,
        )
        plotted = True
    ax.set_title("Attack-side overtopping (local candidates)", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Training progress (%)")
    ax.set_ylabel("Attack endpoint changed (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.15)
    if plotted:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(
            0.5, 0.5,
            "Attack-side causal analysis\nnot available",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.35",
        )


def plot_developmental_story(summary: pd.DataFrame, control: pd.DataFrame, trigger: pd.DataFrame, output: Path) -> bool:
    if summary.empty:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.8))
    _plot_metric(axes[0,0], summary, "normal_accuracy", "Normal behavior", "Accuracy (%)")
    _plot_metric(axes[0,1], summary, "attack_conversion", "Backdoor behavior", "Conditional conversion (%)")
    _plot_metric(axes[0,2], summary, "U_J", "Control-correctness set overtopping", "U(J) (%)")
    _plot_metric(axes[1,0], summary, "s_1", "Control-correctness strongest singleton", "Strongest singleton effect (%)")
    _plot_metric(axes[1,1], summary, "N_eff", "Control-correctness effective support", "Effective number of channels", percent=False)
    _plot_attack_side_summary(axes[1,2], summary)

    handles, _ = axes[0,0].get_legend_handles_labels()
    if handles:
        axes[0,0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Poisoning changes the developmental trajectory of overtopping", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.012,
        "All six panels are aggregate checkpoint-level quantities; U(J), strongest-singleton, and attack-side causal summaries use each checkpoint's locally discovered candidate set. Channel-specific role reassignment and prospective targeting are separated into later figures.",
        ha="center", va="bottom", fontsize=8.4, color="dimgray", wrap=True,
    )
    fig.tight_layout(rect=(0,0.055,1,0.955))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return True


def _fixed_control_mask(control: pd.DataFrame) -> pd.Series:
    source = control.get("control_correctness_value_source", pd.Series("", index=control.index)).astype(str)
    return source.isin(CONTROL_FIXED_SOURCES)


def _fixed_attack_mask(trigger: pd.DataFrame) -> pd.Series:
    source = trigger.get("attack_value_source", pd.Series("", index=trigger.index)).astype(str)
    return source.isin(ATTACK_FIXED_SOURCES)


def _is_fixed_source_label(source: object) -> bool:
    return str(source) in ALL_FIXED_SOURCES


def _checkpoint_channel_panel(ax, control: pd.DataFrame, trigger: pd.DataFrame, fraction: float, top_n: int = 8) -> None:
    control_fraction = pd.to_numeric(control.get("fraction"), errors="coerce").to_numpy(float)
    trigger_fraction = pd.to_numeric(trigger.get("fraction"), errors="coerce").to_numpy(float)
    clean = control[(control["condition"].astype(str) == "clean") & np.isclose(control_fraction, fraction, equal_nan=False)]
    poison = control[(control["condition"].astype(str) == "poisoned") & np.isclose(control_fraction, fraction, equal_nan=False)]
    attack = trigger[np.isclose(trigger_fraction, fraction, equal_nan=False)]
    units = set(clean.get("unit_key", [])) | set(poison.get("unit_key", [])) | set(attack.get("unit_key", []))
    scored = []
    for u in units:
        c = clean[clean["unit_key"].astype(str) == str(u)]
        p = poison[poison["unit_key"].astype(str) == str(u)]
        a = attack[attack["unit_key"].astype(str) == str(u)]
        cv = _finite(c.iloc[0]["c2i_rate"]) if not c.empty else math.nan
        pv = _finite(p.iloc[0]["c2i_rate"]) if not p.empty else math.nan
        av = _finite(a.iloc[0]["attack_c2i_rate"]) if not a.empty else math.nan
        cd = bool(c.iloc[0].get("discovered_at_checkpoint", True)) if not c.empty else False
        pd_ = bool(p.iloc[0].get("discovered_at_checkpoint", True)) if not p.empty else False
        ad = bool(a.iloc[0].get("attack_discovered_at_checkpoint", True)) if not a.empty else False
        cs = str(c.iloc[0].get("control_correctness_value_source", "")) if not c.empty else ""
        ps = str(p.iloc[0].get("control_correctness_value_source", "")) if not p.empty else ""
        ass = str(a.iloc[0].get("attack_value_source", "")) if not a.empty else ""
        candidates = [v for v in (cv,pv,av) if np.isfinite(v)]
        score = max(candidates) if candidates else -1
        scored.append((str(u),cv,pv,av,cd,pd_,ad,cs,ps,ass,score))
    scored.sort(key=lambda x: (x[-1], x[0]), reverse=True)
    scored = scored[:top_n]
    scored.reverse()
    y = np.arange(len(scored))
    clean_color = "#4C78A8"
    poison_color = "#F58518"
    attack_color = "#D62728"

    def scatter_value(value: float, idx: int, *, marker: str, color: str, source: str, discovered: bool) -> None:
        if not np.isfinite(value):
            return
        if _is_fixed_source_label(source):
            ax.scatter(
                100*value, idx, marker=marker, s=54,
                facecolors=color if discovered else "none", edgecolors=color,
                linewidths=1.4, zorder=3,
            )
        else:
            # A local-only value is scientifically usable at this checkpoint but
            # cannot support a longitudinal non-rediscovery claim. Preserve the
            # endpoint shape and overlay an x to expose the missing fixed-union
            # evaluation rather than silently treating it as longitudinal.
            ax.scatter(100*value, idx, marker=marker, s=54, facecolors="none", edgecolors=color, linewidths=1.2, zorder=3)
            ax.scatter(100*value, idx, marker="x", s=45, color=color, linewidths=1.5, zorder=4)

    local_only_present = False
    for i,(u,cv,pv,av,cd,pd_,ad,cs,ps,ass,_) in enumerate(scored):
        scatter_value(cv, i, marker="o", color=clean_color, source=cs, discovered=cd)
        scatter_value(pv, i, marker="s", color=poison_color, source=ps, discovered=pd_)
        scatter_value(av, i, marker="D", color=attack_color, source=ass, discovered=ad)
        local_only_present = local_only_present or any(
            np.isfinite(v) and not _is_fixed_source_label(src)
            for v, src in ((cv, cs), (pv, ps), (av, ass))
        )
        finite = [100*v for v in (cv,pv,av) if np.isfinite(v)]
        if len(finite) >= 2:
            ax.plot([min(finite),max(finite)],[i,i], linewidth=0.9, color="0.78", zorder=1)
    ax.set_yticks(y, [r[0] for r in scored])
    ax.set_xlim(0,105)
    ax.set_xlabel("Correct->incorrect singleton effect (%)")
    title = f"{100*fraction:g}% checkpoint"
    if local_only_present:
        title += "  [local-only values present]"
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.grid(axis="x", alpha=0.15)


def plot_channel_roles(control: pd.DataFrame, trigger: pd.DataFrame, output: Path) -> bool:
    if control.empty:
        return False
    matched = sorted(set(pd.to_numeric(control.loc[control["condition"].astype(str)=="clean","fraction"], errors="coerce").dropna()).intersection(
                     set(pd.to_numeric(control.loc[control["condition"].astype(str)=="poisoned","fraction"], errors="coerce").dropna())))
    if not matched:
        return False
    positive = [float(f) for f in matched if float(f) > 0]
    matched = positive or [float(f) for f in matched]
    ncols = min(3, len(matched))
    nrows = int(math.ceil(len(matched) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2*ncols, 4.9*nrows), squeeze=False)
    flat = list(axes.flat)
    for ax, frac in zip(flat, matched):
        _checkpoint_channel_panel(ax, control, trigger, float(frac), top_n=9)
    for ax in flat[len(matched):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], marker="o", linestyle="None", markersize=7, markerfacecolor="#4C78A8", markeredgecolor="#4C78A8", label="Clean control"),
        Line2D([0],[0], marker="s", linestyle="None", markersize=7, markerfacecolor="#F58518", markeredgecolor="#F58518", label="Poisoned control"),
        Line2D([0],[0], marker="D", linestyle="None", markersize=7, markerfacecolor="#D62728", markeredgecolor="#D62728", label="Poisoned attack"),
        Line2D([0],[0], marker="o", linestyle="None", markersize=7, markerfacecolor="0.45", markeredgecolor="0.45", label="Filled = fixed-union + mixture-localized"),
        Line2D([0],[0], marker="o", linestyle="None", markersize=7, markerfacecolor="none", markeredgecolor="0.35", label="Hollow = fixed-union, not mixture-localized"),
        Line2D([0],[0], marker="x", linestyle="None", markersize=7, color="0.35", label="x = control-reference local only"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5,0.97), fontsize=8)
    fig.suptitle("Which channels change causal role under poisoning?", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.012,
        "Descriptive checkpoint view only: rows are chosen independently within each checkpoint by the largest observed singleton effect. Fixed-union evaluations use channels localized from the defender-visible training mixture; x markers expose control-reference-only local values. This figure does not choose a defense target; prospective defense-target selection is isolated in Figure 03.",
        ha="center", va="bottom", fontsize=8.4, color="dimgray", wrap=True,
    )
    fig.tight_layout(rect=(0,0.05,1,0.91))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return True


def _prospective_defense_target_selection(control: pd.DataFrame, trigger: pd.DataFrame, *, top_k: int = 3) -> pd.DataFrame:
    """Build two complementary attack-blind defense-target experiments.

    Experiment A: ``one_checkpoint_ahead_frozen_overtopping`` (prospective).
        At checkpoint s, use only information available at or before s.  Among
        channels already exposed by defender-visible localization by s, rank by
        their poisoned-model control-correctness singleton disruption at s.
        Freeze the top-k set and evaluate that *same* set at the next checkpoint
        t.  Attack effects at t never enter selection.  The operational outcome
        is

            delta_def = attack_suppression(t) - benign_damage(t).

        This is the stronger temporal-generalization / defense-validity test:
        the target set is frozen before the model reaches the target checkpoint.

    Experiment B: ``clean_reference_contrast`` (checkpoint-aligned diagnostic).
        At checkpoint t, protect the top-k channels most important on a matched
        clean reference, then select up to top-k remaining channels with largest
        positive

            poison_excess = U_poisoned(t, j) - U_clean(t, j).

        Attack effects are joined only after selection.  This is not prospective
        in time, but it is attack-blind and is useful when a defender can compare
        the current model against a trusted clean reference.

    Both experiments use only existing cached Stage-07 singleton materialization;
    no model inference or new interventions are required.
    """
    columns = [
        "strategy", "selection_fraction", "target_fraction", "unit_key",
        "selection_score", "selection_overtopping_rate",
        # Raw endpoint names retained for traceability.
        "target_attack_c2i_rate", "target_control_c2i_rate",
        # Semantic defense quantities.
        "target_attack_suppression_rate", "target_benign_damage_rate",
        "target_defense_leverage_proxy",
        # Clean-reference diagnostics.
        "reference_clean_c2i_rate", "observed_poisoned_c2i_rate",
        "poison_excess_disruption", "clean_importance_rank",
        "poisoned_importance_rank", "clean_protected_cutoff_rank",
        # Prospective provenance/audit fields.
        "selection_pool_size", "selection_known_by_checkpoint",
        "selection_basis",
    ]
    if control.empty or trigger.empty:
        return pd.DataFrame(columns=columns)

    fixed_control = control[_fixed_control_mask(control)].copy()
    fixed_attack = trigger[_fixed_attack_mask(trigger)].copy()
    if fixed_control.empty or fixed_attack.empty:
        return pd.DataFrame(columns=columns)

    for frame, cols in (
        (fixed_control, ("fraction", "c2i_rate")),
        (fixed_attack, ("fraction", "attack_c2i_rate")),
    ):
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    poisoned_control = fixed_control[fixed_control["condition"].astype(str).eq("poisoned")].copy()
    clean_control = fixed_control[fixed_control["condition"].astype(str).eq("clean")].copy()
    attack_fractions = sorted(
        set(poisoned_control["fraction"].dropna()).intersection(set(fixed_attack["fraction"].dropna()))
    )
    control_fractions = sorted(set(poisoned_control["fraction"].dropna()))
    if not attack_fractions or len(control_fractions) < 2:
        return pd.DataFrame(columns=columns)

    def value_at(frame: pd.DataFrame, frac: float, unit: str, value_col: str) -> float:
        part = frame[
            np.isclose(frame["fraction"].to_numpy(float), float(frac), equal_nan=False)
            & frame["unit_key"].astype(str).eq(str(unit))
        ]
        return _finite(part.iloc[0][value_col]) if not part.empty else math.nan

    def known_units_by(selection_frac: float) -> set[str]:
        """Channels the defender could have known by ``selection_frac``.

        Use checkpoint-local observed-mixture discovery only as a *causal
        availability* gate.  Once a channel has been discovered at any earlier
        checkpoint it remains in the defender's candidate registry and may be
        ranked using its cached fixed-union singleton effect at the current
        selection checkpoint.  This avoids future-candidate leakage while also
        avoiding the brittle requirement that a channel be rediscovered at every
        checkpoint.
        """
        frac = pd.to_numeric(poisoned_control["fraction"], errors="coerce")
        discovered = poisoned_control.get(
            "discovered_at_checkpoint", pd.Series(False, index=poisoned_control.index)
        ).fillna(False).astype(bool)
        eligible = poisoned_control.loc[(frac <= float(selection_frac) + 1e-12) & discovered, "unit_key"]
        units = set(eligible.dropna().astype(str))
        # Legacy caches may lack reliable discovery flags.  If the historical
        # gate yields nothing, fall back to units physically evaluated at the
        # selection checkpoint; this is explicit in the audit column below.
        if not units:
            at_selection = poisoned_control[
                np.isclose(poisoned_control["fraction"].to_numpy(float), float(selection_frac), equal_nan=False)
            ]
            units = set(at_selection["unit_key"].dropna().astype(str))
        return units

    rows: list[dict[str, Any]] = []
    top_k_requested = max(1, int(top_k))

    # ------------------------------------------------------------------
    # A. One-checkpoint-ahead: freeze previous-checkpoint overtopping set.
    # ------------------------------------------------------------------
    for selection_frac, target_frac in zip(control_fractions[:-1], control_fractions[1:]):
        if float(target_frac) not in attack_fractions:
            continue
        known_units = known_units_by(float(selection_frac))
        source = poisoned_control[
            np.isclose(poisoned_control["fraction"].to_numpy(float), float(selection_frac), equal_nan=False)
            & poisoned_control["unit_key"].astype(str).isin(known_units)
        ].copy()
        source["_selection_score"] = pd.to_numeric(source["c2i_rate"], errors="coerce")
        source = source.dropna(subset=["_selection_score"]).sort_values(
            ["_selection_score", "unit_key"], ascending=[False, True]
        )
        selected = source.head(top_k_requested)
        for rec in selected.to_dict("records"):
            unit = str(rec["unit_key"])
            attack_value = value_at(fixed_attack, float(target_frac), unit, "attack_c2i_rate")
            benign_value = value_at(poisoned_control, float(target_frac), unit, "c2i_rate")
            clean_value = value_at(clean_control, float(target_frac), unit, "c2i_rate")
            if not (np.isfinite(attack_value) and np.isfinite(benign_value)):
                continue
            rows.append({
                "strategy": "one_checkpoint_ahead_frozen_overtopping",
                "selection_fraction": float(selection_frac),
                "target_fraction": float(target_frac),
                "unit_key": unit,
                "selection_score": float(rec["_selection_score"]),
                "selection_overtopping_rate": float(rec["_selection_score"]),
                "target_attack_c2i_rate": attack_value,
                "target_control_c2i_rate": benign_value,
                "target_attack_suppression_rate": attack_value,
                "target_benign_damage_rate": benign_value,
                "target_defense_leverage_proxy": attack_value - benign_value,
                "reference_clean_c2i_rate": clean_value,
                "observed_poisoned_c2i_rate": benign_value,
                "poison_excess_disruption": benign_value - clean_value if np.isfinite(clean_value) else math.nan,
                "clean_importance_rank": math.nan,
                "poisoned_importance_rank": math.nan,
                "clean_protected_cutoff_rank": math.nan,
                "selection_pool_size": int(len(source)),
                "selection_known_by_checkpoint": True,
                "selection_basis": (
                    f"prospective one-checkpoint-ahead: at {100*float(selection_frac):g}% rank only channels "
                    "known by that checkpoint by poisoned-model control-correctness C->I overtopping; "
                    f"freeze top {top_k_requested} and evaluate unchanged at {100*float(target_frac):g}%; "
                    "target-checkpoint attack/control outcomes are not used for selection"
                ),
            })

    # ------------------------------------------------------------------
    # B. Same-checkpoint clean-reference contrast.
    # ------------------------------------------------------------------
    for target_frac in attack_fractions:
        clean_part = clean_control[
            np.isclose(clean_control["fraction"].to_numpy(float), float(target_frac), equal_nan=False)
        ][["unit_key", "c2i_rate"]].rename(columns={"c2i_rate": "_clean_c2i"})
        poison_part = poisoned_control[
            np.isclose(poisoned_control["fraction"].to_numpy(float), float(target_frac), equal_nan=False)
        ][["unit_key", "c2i_rate"]].rename(columns={"c2i_rate": "_poisoned_c2i"})
        paired = clean_part.merge(poison_part, on="unit_key", how="inner")
        if paired.empty:
            continue
        paired["_clean_c2i"] = pd.to_numeric(paired["_clean_c2i"], errors="coerce")
        paired["_poisoned_c2i"] = pd.to_numeric(paired["_poisoned_c2i"], errors="coerce")
        paired = paired.dropna(subset=["_clean_c2i", "_poisoned_c2i"]).copy()
        if paired.empty:
            continue

        paired["_clean_rank"] = paired["_clean_c2i"].rank(method="min", ascending=False)
        paired["_poisoned_rank"] = paired["_poisoned_c2i"].rank(method="min", ascending=False)
        paired["_poison_excess"] = paired["_poisoned_c2i"] - paired["_clean_c2i"]
        protect_k = min(top_k_requested, int(len(paired)))
        paired["_clean_protected"] = paired["_clean_rank"] <= float(protect_k)
        pool = paired.loc[(~paired["_clean_protected"]) & (paired["_poison_excess"] > 0.0)].copy()
        pool = pool.sort_values(
            ["_poison_excess", "_poisoned_c2i", "unit_key"], ascending=[False, False, True]
        ).head(top_k_requested)

        for rec in pool.to_dict("records"):
            unit = str(rec["unit_key"])
            attack_value = value_at(fixed_attack, float(target_frac), unit, "attack_c2i_rate")
            if not np.isfinite(attack_value):
                continue
            clean_value = float(rec["_clean_c2i"])
            poisoned_value = float(rec["_poisoned_c2i"])
            excess = float(rec["_poison_excess"])
            rows.append({
                "strategy": "clean_reference_contrast",
                "selection_fraction": float(target_frac),
                "target_fraction": float(target_frac),
                "unit_key": unit,
                "selection_score": excess,
                "selection_overtopping_rate": poisoned_value,
                "target_attack_c2i_rate": attack_value,
                "target_control_c2i_rate": poisoned_value,
                "target_attack_suppression_rate": attack_value,
                "target_benign_damage_rate": poisoned_value,
                "target_defense_leverage_proxy": attack_value - poisoned_value,
                "reference_clean_c2i_rate": clean_value,
                "observed_poisoned_c2i_rate": poisoned_value,
                "poison_excess_disruption": excess,
                "clean_importance_rank": float(rec["_clean_rank"]),
                "poisoned_importance_rank": float(rec["_poisoned_rank"]),
                "clean_protected_cutoff_rank": float(protect_k),
                "selection_pool_size": int(len(pool)),
                "selection_known_by_checkpoint": True,
                "selection_basis": (
                    f"checkpoint-aligned attack-blind clean-reference contrast: protect top {protect_k} "
                    "clean-disruption channels, then rank remaining fixed-union channels by positive "
                    "poisoned-trained minus clean-trained control-correctness C->I excess; attack "
                    "information is used only after selection"
                ),
            })

    return pd.DataFrame(rows, columns=columns)

def plot_prospective_defense_leverage(selection: pd.DataFrame, output: Path, *, top_k: int = 3) -> bool:
    """Compare the prospective frozen-set defense with the aligned clean-reference screen."""
    if selection.empty:
        return False

    strategies = [
        (
            "one_checkpoint_ahead_frozen_overtopping",
            "Prospective: previous checkpoint → next checkpoint",
            "",
        ),
        (
            "clean_reference_contrast",
            "Checkpoint-aligned: clean-reference contrast",
            "",
        ),
    ]
    planned_targets = sorted(pd.to_numeric(selection.get("target_fraction"), errors="coerce").dropna().unique())
    if not planned_targets:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3), sharey=True)
    any_panel = False
    for ax, (strategy, title, subtitle) in zip(axes, strategies):
        part = selection[selection["strategy"].astype(str).eq(strategy)].copy()
        ax.set_xticks(100.0 * np.asarray(planned_targets, dtype=float))
        ax.set_xlim(max(0.0, 100.0 * min(planned_targets) - 4.0), min(100.0, 100.0 * max(planned_targets) + 4.0))
        if part.empty:
            ax.text(0.5, 0.5, "No valid selections", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Target checkpoint (%)")
            ax.grid(True, axis="y", alpha=0.18)
            continue

        agg = part.groupby("target_fraction", as_index=False).agg(
            attack_suppression=("target_attack_suppression_rate", "mean"),
            benign_damage=("target_benign_damage_rate", "mean"),
            defense_leverage=("target_defense_leverage_proxy", "mean"),
            n_selected=("unit_key", "nunique"),
            selection_fraction=("selection_fraction", "first"),
        ).sort_values("target_fraction")

        x = 100.0 * agg["target_fraction"].to_numpy(float)
        attack = 100.0 * agg["attack_suppression"].to_numpy(float)
        benign = 100.0 * agg["benign_damage"].to_numpy(float)
        leverage = 100.0 * agg["defense_leverage"].to_numpy(float)
        ax.plot(x, attack, marker="D", linewidth=2.2, label="Attack suppression")
        ax.plot(x, benign, marker="o", linewidth=2.2, label="Benign damage")
        ax.fill_between(x, benign, attack, where=(attack >= benign), alpha=0.12, interpolate=True, label="Δdef > 0")
        ax.fill_between(x, benign, attack, where=(attack < benign), alpha=0.06, interpolate=True)

        for rec, lev in zip(agg.itertuples(index=False), leverage):
            target_x = 100.0 * float(rec.target_fraction)
            anchor_y = max(100.0 * float(rec.attack_suppression), 100.0 * float(rec.benign_damage))
            if strategy == "one_checkpoint_ahead_frozen_overtopping":
                timing = f"{100.0*float(rec.selection_fraction):g}→{100.0*float(rec.target_fraction):g}%"
            else:
                timing = f"at {100.0*float(rec.target_fraction):g}%"
            high = anchor_y > 0.80 * max(float(np.nanmax(attack)), float(np.nanmax(benign)), 1.0)
            ax.annotate(
                f"{timing}\nn={int(rec.n_selected)}, Δdef={lev:+.1f} pp",
                (target_x, anchor_y), xytext=(0, -28 if high else 8), textcoords="offset points",
                ha="center", va="top" if high else "bottom", fontsize=7.2,
            )

        observed = set(np.round(agg["target_fraction"].to_numpy(float), 12))
        missing = [f for f in planned_targets if round(float(f), 12) not in observed]
        if missing:
            labels = ", ".join(f"{100*float(f):g}%" for f in missing)
            ax.text(0.98, 0.04, f"No valid evaluated set at: {labels}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.2, color="0.4")

        ax.set_title(title, fontweight="bold", fontsize=10.5, pad=10)
        ax.set_xlabel("Target checkpoint (%)")
        ax.grid(True, alpha=0.18)
        any_panel = True

    axes[0].set_ylabel("Singleton intervention rate (%)")
    axes[0].set_ylim(bottom=0)
    legend_by_label: dict[str, Any] = {}
    for axis in axes:
        hs, labels = axis.get_legend_handles_labels()
        for h, label in zip(hs, labels):
            legend_by_label.setdefault(label, h)
    if legend_by_label:
        fig.legend(list(legend_by_label.values()), list(legend_by_label.keys()),
                   loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=8.5)

    fig.suptitle("Defense target screening: prospective frozen-set vs checkpoint-aligned clean-reference", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.015,
        f"Left = primary prospective defense test: freeze up to {int(top_k)} channels selected by previous-checkpoint poisoned-model overtopping and evaluate the unchanged set one checkpoint later. Right = complementary same-checkpoint diagnostic: protect the top {int(top_k)} clean-important channels and select positive poisoned−clean excess channels. Attack effects are never used for selection in either experiment. Δdef = attack suppression − poisoned-model benign damage; positive Δdef is favorable.",
        ha="center", va="bottom", fontsize=8.2, color="dimgray", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.865))
    if any_panel:
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return True
    plt.close(fig)
    return False

def _parse_benign_budget_grid(spec: str, *, include: float | None = None) -> list[float]:
    """Parse a comma-separated benign-damage budget grid in [0, 1]."""
    vals: list[float] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"benign budget must lie in [0,1], got {value}")
        vals.append(value)
    if include is not None:
        vals.append(float(np.clip(float(include), 0.0, 1.0)))
    return sorted(set(round(v, 10) for v in vals))


def _benign_budgeted_clean_reference_selection(
    control: pd.DataFrame,
    trigger: pd.DataFrame,
    *,
    top_k: int = 3,
    benign_damage_budget: float = 0.30,
) -> pd.DataFrame:
    """Attack-blind clean-reference screen used only by Figure 03b / Figure 6.

    At each checkpoint, rank fixed-union channels by positive clean-reference
    excess, U_poisoned(j) - U_clean(j), subject to the absolute poisoned-model
    benign-damage cap U_poisoned(j) <= tau. Attack outcomes are joined only after
    the target set has been selected. This helper is intentionally separate from
    ``_prospective_defense_target_selection`` so the pre-existing Stage-07 figures
    keep their original selection semantics.
    """
    columns = [
        "strategy", "selection_fraction", "target_fraction", "unit_key",
        "selection_score", "selection_overtopping_rate",
        "target_attack_c2i_rate", "target_control_c2i_rate",
        "target_attack_suppression_rate", "target_benign_damage_rate",
        "target_defense_leverage_proxy", "reference_clean_c2i_rate",
        "observed_poisoned_c2i_rate", "poison_excess_disruption",
        "clean_importance_rank", "poisoned_importance_rank",
        "clean_protected_cutoff_rank", "benign_damage_budget",
        "selection_pool_size", "selection_known_by_checkpoint", "selection_basis",
    ]
    if control.empty or trigger.empty:
        return pd.DataFrame(columns=columns)

    fixed_control = control[_fixed_control_mask(control)].copy()
    fixed_attack = trigger[_fixed_attack_mask(trigger)].copy()
    if fixed_control.empty or fixed_attack.empty:
        return pd.DataFrame(columns=columns)

    for frame, cols in (
        (fixed_control, ("fraction", "c2i_rate")),
        (fixed_attack, ("fraction", "attack_c2i_rate")),
    ):
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    poisoned = fixed_control[fixed_control["condition"].astype(str).eq("poisoned")].copy()
    clean = fixed_control[fixed_control["condition"].astype(str).eq("clean")].copy()
    fractions = sorted(
        set(poisoned["fraction"].dropna())
        .intersection(set(clean["fraction"].dropna()))
        .intersection(set(fixed_attack["fraction"].dropna()))
    )
    budget = float(np.clip(float(benign_damage_budget), 0.0, 1.0))
    k = max(1, int(top_k))
    rows: list[dict[str, Any]] = []

    for frac in fractions:
        p = poisoned[np.isclose(poisoned["fraction"].to_numpy(float), float(frac), equal_nan=False)][
            ["unit_key", "c2i_rate"]
        ].rename(columns={"c2i_rate": "_poisoned_c2i"})
        c = clean[np.isclose(clean["fraction"].to_numpy(float), float(frac), equal_nan=False)][
            ["unit_key", "c2i_rate"]
        ].rename(columns={"c2i_rate": "_clean_c2i"})
        paired = p.merge(c, on="unit_key", how="inner")
        paired["_poisoned_c2i"] = pd.to_numeric(paired["_poisoned_c2i"], errors="coerce")
        paired["_clean_c2i"] = pd.to_numeric(paired["_clean_c2i"], errors="coerce")
        paired = paired.dropna(subset=["_poisoned_c2i", "_clean_c2i"]).copy()
        if paired.empty:
            continue

        paired["_clean_rank"] = paired["_clean_c2i"].rank(method="min", ascending=False)
        paired["_poisoned_rank"] = paired["_poisoned_c2i"].rank(method="min", ascending=False)
        paired["_poison_excess"] = paired["_poisoned_c2i"] - paired["_clean_c2i"]
        eligible = paired.loc[
            (paired["_poison_excess"] > 0.0)
            & (paired["_poisoned_c2i"] <= budget + 1e-12)
        ].sort_values(
            ["_poison_excess", "_poisoned_c2i", "unit_key"],
            ascending=[False, False, True],
        )
        pool_size = int(len(eligible))
        selected = eligible.head(k).copy()

        # Attack values are intentionally looked up only after the target set is fixed.
        attack_at_frac = fixed_attack[
            np.isclose(fixed_attack["fraction"].to_numpy(float), float(frac), equal_nan=False)
        ][["unit_key", "attack_c2i_rate"]].copy()
        attack_map = {
            str(r.unit_key): _finite(r.attack_c2i_rate)
            for r in attack_at_frac.itertuples(index=False)
        }

        for rec in selected.to_dict("records"):
            unit = str(rec["unit_key"])
            attack_value = attack_map.get(unit, math.nan)
            if not np.isfinite(attack_value):
                continue
            poisoned_value = float(rec["_poisoned_c2i"])
            clean_value = float(rec["_clean_c2i"])
            excess = float(rec["_poison_excess"])
            rows.append({
                "strategy": "clean_reference_benign_budget",
                "selection_fraction": float(frac),
                "target_fraction": float(frac),
                "unit_key": unit,
                "selection_score": excess,
                "selection_overtopping_rate": poisoned_value,
                "target_attack_c2i_rate": attack_value,
                "target_control_c2i_rate": poisoned_value,
                "target_attack_suppression_rate": attack_value,
                "target_benign_damage_rate": poisoned_value,
                "target_defense_leverage_proxy": attack_value - poisoned_value,
                "reference_clean_c2i_rate": clean_value,
                "observed_poisoned_c2i_rate": poisoned_value,
                "poison_excess_disruption": excess,
                "clean_importance_rank": float(rec["_clean_rank"]),
                "poisoned_importance_rank": float(rec["_poisoned_rank"]),
                "clean_protected_cutoff_rank": math.nan,
                "benign_damage_budget": budget,
                "selection_pool_size": pool_size,
                "selection_known_by_checkpoint": True,
                "selection_basis": (
                    "checkpoint-aligned attack-blind benign-budgeted clean-reference screen: "
                    "rank fixed-union channels by positive poisoned-trained minus clean-trained "
                    f"control-correctness C->I excess subject to poisoned benign damage <= {budget:.3f}; "
                    "attack information is used only after selection"
                ),
            })
    return pd.DataFrame(rows, columns=columns)


def _clean_reference_budget_sweep(
    control: pd.DataFrame,
    trigger: pd.DataFrame,
    *,
    budgets: list[float],
    top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate attack-blind clean-reference selection over benign-damage budgets.

    Selection never uses attack outcomes.  Attack suppression is joined only after
    each budget-specific target set has been frozen, so the sweep is an evaluation
    of operating points rather than an attack-tuned selector.
    """
    selected_parts: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for budget in budgets:
        screen = _benign_budgeted_clean_reference_selection(
            control, trigger, top_k=top_k, benign_damage_budget=float(budget)
        )
        part = screen[
            screen.get("strategy", pd.Series(dtype=str)).astype(str).eq("clean_reference_benign_budget")
        ].copy()
        if not part.empty:
            part["benign_damage_budget"] = float(budget)
            selected_parts.append(part)
        summary = _checkpoint_summary_from_selection(part, "clean_reference_benign_budget")
        if not summary.empty:
            summary.insert(0, "benign_damage_budget", float(budget))
            summaries.append(summary)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    # Explicitly retain zero-coverage operating points so the curve cannot hide
    # abstention at strict budgets.
    if budgets:
        fixed_control = control[_fixed_control_mask(control)].copy() if not control.empty else pd.DataFrame()
        fixed_attack = trigger[_fixed_attack_mask(trigger)].copy() if not trigger.empty else pd.DataFrame()
        if not fixed_control.empty and not fixed_attack.empty:
            pc = fixed_control[fixed_control["condition"].astype(str).eq("poisoned")].copy()
            fractions = sorted(set(pd.to_numeric(pc["fraction"], errors="coerce").dropna()).intersection(
                set(pd.to_numeric(fixed_attack["fraction"], errors="coerce").dropna())
            ))
            full = pd.MultiIndex.from_product(
                [budgets, fractions], names=["benign_damage_budget", "target_fraction"]
            ).to_frame(index=False)
            summary = full.merge(summary, on=["benign_damage_budget", "target_fraction"], how="left")
            summary["n_selected"] = pd.to_numeric(summary["n_selected"], errors="coerce").fillna(0).astype(int)
    return selected, summary


def _checkpoint_summary_from_selection(selection: pd.DataFrame, strategy: str) -> pd.DataFrame:
    part = selection[selection.get("strategy", pd.Series(dtype=str)).astype(str).eq(strategy)].copy()
    if part.empty:
        return pd.DataFrame(columns=[
            "target_fraction", "n_selected", "mean_clean_disruption",
            "mean_poisoned_disruption", "mean_poison_excess",
            "mean_attack_suppression", "mean_defense_leverage",
        ])
    return part.groupby("target_fraction", as_index=False).agg(
        n_selected=("unit_key", "nunique"),
        mean_clean_disruption=("reference_clean_c2i_rate", "mean"),
        mean_poisoned_disruption=("target_benign_damage_rate", "mean"),
        mean_poison_excess=("poison_excess_disruption", "mean"),
        mean_attack_suppression=("target_attack_suppression_rate", "mean"),
        mean_defense_leverage=("target_defense_leverage_proxy", "mean"),
    ).sort_values("target_fraction")


def plot_clean_reference_defense_interpretation(
    selection: pd.DataFrame,
    budget_summary: pd.DataFrame,
    output: Path,
    *,
    top_k: int = 3,
    benign_damage_budget: float = 0.30,
) -> bool:
    """Main-text attack-blind clean-reference defense figure.

    Left: checkpoint-level mean attack suppression and benign damage at the
    illustrative operating point, with defense leverage shown as the shaded gap.
    A compact inset retains the full benign-budget sweep. Right: the selected
    single-channel effects (one point per channel per checkpoint) in
    suppression-versus-damage space. This intentionally does not repeat the
    checkpoint means from panel A. Attack outcomes are never used for target
    selection.
    """
    part = selection[
        selection.get("strategy", pd.Series(dtype=str)).astype(str).eq("clean_reference_benign_budget")
    ].copy()
    if part.empty or budget_summary.empty:
        return False

    for col in (
        "target_fraction", "target_attack_suppression_rate", "target_benign_damage_rate",
        "target_defense_leverage_proxy", "poison_excess_disruption", "benign_damage_budget",
    ):
        if col in part.columns:
            part[col] = pd.to_numeric(part[col], errors="coerce")
    bs = budget_summary.copy()
    for col in ("benign_damage_budget", "target_fraction", "mean_defense_leverage", "n_selected"):
        if col in bs.columns:
            bs[col] = pd.to_numeric(bs[col], errors="coerce")

    operating = _checkpoint_summary_from_selection(part, "clean_reference_benign_budget")
    if operating.empty:
        return False

    fig, (ax_line, ax_tradeoff) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.82, wspace=0.28)

    # A. Checkpoint-level operating point with the defense-leverage gap shaded.
    x = 100.0 * operating["target_fraction"].to_numpy(float)
    attack = 100.0 * operating["mean_attack_suppression"].to_numpy(float)
    benign = 100.0 * operating["mean_poisoned_disruption"].to_numpy(float)
    leverage = 100.0 * operating["mean_defense_leverage"].to_numpy(float)

    ax_line.plot(x, attack, marker="D", linewidth=2.2, label="Attack suppression")
    ax_line.plot(x, benign, marker="o", linewidth=2.2, label="Benign damage")
    ax_line.fill_between(
        x, benign, attack, where=(attack >= benign), interpolate=True,
        color="#dcefdc", alpha=0.85, linewidth=0, label=r"$\Delta_{\rm def}>0$",
    )
    ax_line.fill_between(
        x, benign, attack, where=(attack < benign), interpolate=True,
        color="#f4dede", alpha=0.85, linewidth=0,
    )
    for xv, av, bv, lev in zip(x, attack, benign, leverage):
        anchor = max(av, bv)
        ax_line.annotate(
            f"{lev:+.0f} pp", (xv, anchor), xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.4,
        )
    ax_line.set_xticks(x)
    ax_line.set_xlim(max(0.0, float(np.nanmin(x)) - 5.0), min(100.0, float(np.nanmax(x)) + 5.0))
    ax_line.set_ylim(bottom=0.0)
    ax_line.set_xlabel("Checkpoint (%)")
    ax_line.set_ylabel("Intervention rate (%)")
    ax_line.set_title(
        f"A. Checkpoint tradeoff at τ={100.0*float(benign_damage_budget):.0f}%",
        fontweight="bold", fontsize=10.5,
    )
    ax_line.grid(True, alpha=0.18)
    ax_line.legend(frameon=False, fontsize=7.6, loc="upper left")

    # Compact inset keeps the full budget sweep visible in the main-text figure.
    inset = ax_line.inset_axes([0.54, 0.08, 0.43, 0.37])
    checkpoints = sorted(pd.to_numeric(bs["target_fraction"], errors="coerce").dropna().unique())
    markers = ["o", "s", "^", "D", "P", "X"]
    for idx, frac in enumerate(checkpoints):
        cp = bs[
            np.isclose(bs["target_fraction"].to_numpy(float), float(frac), equal_nan=False)
        ].sort_values("benign_damage_budget")
        y = 100.0 * pd.to_numeric(cp["mean_defense_leverage"], errors="coerce").to_numpy(float)
        n = pd.to_numeric(cp["n_selected"], errors="coerce").fillna(0).to_numpy(int)
        y[n <= 0] = np.nan
        inset.plot(
            100.0 * cp["benign_damage_budget"].to_numpy(float), y,
            marker=markers[idx % len(markers)], linewidth=1.0, markersize=2.2,
        )
    inset.axhline(0.0, color="0.45", linestyle="--", linewidth=0.7)
    inset.axvline(100.0 * float(benign_damage_budget), color="0.50", linestyle=":", linewidth=0.8)
    inset.set_title("Budget sweep", fontsize=7.2, pad=2)
    inset.set_xlabel("τ (%)", fontsize=6.5, labelpad=1)
    inset.set_ylabel(r"$\Delta_{\rm def}$ (pp)", fontsize=6.5, labelpad=1)
    inset.tick_params(axis="both", labelsize=6.0, pad=1)
    inset.grid(True, alpha=0.12)

    # B. Individual selected-channel effects at the same operating point.
    # ``operating`` is checkpoint-aggregated and belongs only in panel A; using
    # it here would simply redraw panel A's five means. ``part`` has one row per
    # selected unit/channel and checkpoint, which is the intended panel-B unit.
    single = part.copy()
    single = single.dropna(subset=[
        "target_fraction", "unit_key",
        "target_attack_suppression_rate", "target_benign_damage_rate",
    ]).drop_duplicates(subset=["target_fraction", "unit_key"], keep="last")
    if single.empty:
        plt.close(fig)
        return False

    xvals = 100.0 * single["target_benign_damage_rate"].to_numpy(float)
    yvals = 100.0 * single["target_attack_suppression_rate"].to_numpy(float)
    lim = max(5.0, float(np.nanmax(np.r_[xvals, yvals])) * 1.10)
    lim = min(100.0, max(lim, 70.0))

    tradeoff_x = np.linspace(0.0, lim, 400)
    ax_tradeoff.fill_between(
        tradeoff_x, tradeoff_x, lim,
        color="#dcefdc", alpha=0.85, linewidth=0, zorder=0,
    )
    ax_tradeoff.fill_between(
        tradeoff_x, 0.0, tradeoff_x,
        color="#f4dede", alpha=0.85, linewidth=0, zorder=0,
    )
    ax_tradeoff.plot([0, lim], [0, lim], linestyle="--", color="0.42", linewidth=1.0, zorder=1)
    ax_tradeoff.axvline(100.0 * float(benign_damage_budget), color="0.50", linestyle=":", linewidth=1.1)
    ax_tradeoff.text(0.025 * lim, 0.93 * lim, "attack-selective", fontsize=8.0, fontweight="bold", color="0.18")
    ax_tradeoff.text(0.72 * lim, 0.08 * lim, "benign-costly", fontsize=8.0, fontweight="bold", color="0.18")

    checkpoints = sorted(single["target_fraction"].astype(float).unique())
    for idx, frac in enumerate(checkpoints):
        cp = single[np.isclose(
            single["target_fraction"].to_numpy(float), float(frac), equal_nan=False
        )].copy()
        cx = 100.0 * cp["target_benign_damage_rate"].to_numpy(float)
        cy = 100.0 * cp["target_attack_suppression_rate"].to_numpy(float)
        ax_tradeoff.scatter(
            cx, cy, s=52, marker=markers[idx % len(markers)], zorder=3,
            label=f"{int(round(100*float(frac)))}% checkpoint",
        )
    ax_tradeoff.set_xlim(0.0, lim)
    ax_tradeoff.set_ylim(0.0, lim)
    ax_tradeoff.set_xlabel("Single-channel benign damage (%)")
    ax_tradeoff.set_ylabel("Single-channel attack suppression (%)")
    ax_tradeoff.set_title(
        "B. Single-channel suppression versus benign damage",
        fontweight="bold", fontsize=10.5,
    )
    ax_tradeoff.grid(True, alpha=0.18)
    ax_tradeoff.legend(frameon=False, fontsize=6.8, loc="upper right")

    fig.suptitle("Benign-budgeted clean-reference defense", fontsize=13.5, fontweight="bold", y=0.96)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return True

def plot_one_checkpoint_ahead_defense_interpretation(selection: pd.DataFrame, output: Path, *, top_k: int = 3) -> bool:
    """Detailed view of the frozen previous-checkpoint overtopping experiment."""
    strategy = "one_checkpoint_ahead_frozen_overtopping"
    part = selection[selection.get("strategy", pd.Series(dtype=str)).astype(str).eq(strategy)].copy()
    if part.empty:
        return False

    for col in (
        "selection_fraction", "target_fraction", "selection_score",
        "target_attack_suppression_rate", "target_benign_damage_rate",
        "target_defense_leverage_proxy",
    ):
        part[col] = pd.to_numeric(part[col], errors="coerce")
    summary = _checkpoint_summary_from_selection(part, strategy)
    if summary.empty:
        return False

    targets = summary["target_fraction"].to_numpy(float)
    channels = (
        part.groupby("unit_key", as_index=False)
        .agg(max_selection_score=("selection_score", "max"))
        .sort_values(["max_selection_score", "unit_key"], ascending=[False, True])["unit_key"]
        .astype(str).tolist()
    )
    score_mat = np.full((len(channels), len(targets)), np.nan)
    leverage_mat = np.full((len(channels), len(targets)), np.nan)
    timing_labels: list[str] = []
    for j, target in enumerate(targets):
        cp = part[np.isclose(part["target_fraction"].to_numpy(float), float(target), equal_nan=False)]
        sel_frac = _finite(cp.iloc[0].get("selection_fraction")) if not cp.empty else math.nan
        timing_labels.append(f"{100*sel_frac:g}→{100*target:g}%" if np.isfinite(sel_frac) else f"{100*target:g}%")
        for i, unit in enumerate(channels):
            cell = cp[cp["unit_key"].astype(str).eq(unit)]
            if cell.empty:
                continue
            score_mat[i, j] = 100.0 * _finite(cell.iloc[0].get("selection_score"))
            leverage_mat[i, j] = 100.0 * _finite(cell.iloc[0].get("target_defense_leverage_proxy"))

    fig = plt.figure(figsize=(13.8, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.2], hspace=0.34, wspace=0.36)
    ax_summary = fig.add_subplot(gs[0, :])
    ax_score = fig.add_subplot(gs[1, 0])
    ax_lev = fig.add_subplot(gs[1, 1])

    x = np.arange(len(summary))
    attack = 100.0 * summary["mean_attack_suppression"].to_numpy(float)
    benign = 100.0 * summary["mean_poisoned_disruption"].to_numpy(float)
    leverage = 100.0 * summary["mean_defense_leverage"].to_numpy(float)
    width = 0.34
    ax_summary.bar(x - width/2, attack, width=width, label="Attack suppression")
    ax_summary.bar(x + width/2, benign, width=width, label="Benign damage")
    ymax = max(float(np.nanmax(attack)), float(np.nanmax(benign)), 1.0)
    ax_summary.set_ylim(0, ymax * 1.22)
    for k, rec in enumerate(summary.itertuples(index=False)):
        ax_summary.annotate(
            f"Δdef={100.0*float(rec.mean_defense_leverage):+.1f} pp\nn={int(rec.n_selected)}",
            (k, max(attack[k], benign[k])), xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5,
        )
    ax_summary.set_xticks(x, timing_labels)
    ax_summary.set_ylabel("Mean singleton intervention rate (%)")
    ax_summary.set_title("Freeze at checkpoint s, evaluate the unchanged channel set at the next checkpoint t", fontweight="bold")
    ax_summary.grid(True, axis="y", alpha=0.18)
    ax_summary.legend(frameon=False, ncol=2)

    score_masked = np.ma.masked_invalid(score_mat)
    cmap_score = plt.get_cmap("Blues").copy(); cmap_score.set_bad("0.96")
    vmax_score = max(1.0, float(np.nanmax(score_mat))) if np.isfinite(score_mat).any() else 1.0
    im1 = ax_score.imshow(score_masked, aspect="auto", interpolation="nearest", cmap=cmap_score, vmin=0, vmax=vmax_score)
    ax_score.set_xticks(np.arange(len(targets)), timing_labels)
    ax_score.set_yticks(np.arange(len(channels)), channels)
    ax_score.set_title("Selection checkpoint: overtopping score used to freeze the set", fontweight="bold", fontsize=10)
    ax_score.set_xlabel("Selection → target checkpoint")
    ax_score.set_ylabel("Frozen channel")
    for i in range(score_mat.shape[0]):
        for j in range(score_mat.shape[1]):
            v = score_mat[i, j]
            if np.isfinite(v):
                ax_score.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.2,
                              color="white" if v > 0.55*vmax_score else "black")
    cb1 = fig.colorbar(im1, ax=ax_score, fraction=0.046, pad=0.02)
    cb1.set_label("Previous-checkpoint overtopping (%)", fontsize=8)

    lev_masked = np.ma.masked_invalid(leverage_mat)
    finite = leverage_mat[np.isfinite(leverage_mat)]
    vmax = max(1.0, float(np.max(np.abs(finite)))) if finite.size else 1.0
    cmap_lev = plt.get_cmap("PiYG").copy(); cmap_lev.set_bad("0.96")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im2 = ax_lev.imshow(lev_masked, aspect="auto", interpolation="nearest", cmap=cmap_lev, norm=norm)
    ax_lev.set_xticks(np.arange(len(targets)), timing_labels)
    ax_lev.set_yticks(np.arange(len(channels)), channels)
    ax_lev.set_title("Next checkpoint: defense leverage of the frozen channels", fontweight="bold", fontsize=10)
    ax_lev.set_xlabel("Selection → target checkpoint")
    ax_lev.set_ylabel("")
    for i in range(leverage_mat.shape[0]):
        for j in range(leverage_mat.shape[1]):
            v = leverage_mat[i, j]
            if np.isfinite(v):
                ax_lev.text(j, i, f"{v:+.1f}{'★' if v > 0 else ''}", ha="center", va="center", fontsize=7.2)
    cb2 = fig.colorbar(im2, ax=ax_lev, fraction=0.046, pad=0.02)
    cb2.set_label("Defense leverage Δdef (pp)", fontsize=8)

    fig.suptitle("One-checkpoint-ahead defense: freeze previous overtopping channels", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.015,
        f"At each selection checkpoint s, rank channels already known by s using poisoned-model control-correctness overtopping, freeze the top {int(top_k)}, and evaluate exactly that set at the next checkpoint t. No target-checkpoint attack or benign result enters selection. Positive Δdef cells (★) indicate a frozen channel suppresses attack more than it harms benign correctness.",
        ha="center", va="bottom", fontsize=8.4, color="dimgray", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return True


def _fixed_control_coverage(summary: pd.DataFrame, control: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[tuple[str, float]], float]:
    fixed = control[_fixed_control_mask(control)].copy() if not control.empty else pd.DataFrame()
    if fixed.empty or summary.empty:
        return fixed, [], [], 0.0
    fixed["fraction"] = pd.to_numeric(fixed["fraction"], errors="coerce")
    cols: list[tuple[str, float]] = []
    for frac in sorted(pd.to_numeric(summary.get("fraction"), errors="coerce").dropna().unique()):
        for condition in ("clean", "poisoned"):
            present = (
                summary["condition"].astype(str).eq(condition)
                & np.isclose(pd.to_numeric(summary["fraction"], errors="coerce"), float(frac))
            ).any()
            if present:
                cols.append((condition, float(frac)))
    units = sorted(fixed["unit_key"].dropna().astype(str).unique())
    expected = len(units) * len(cols)
    if expected == 0:
        return fixed, units, cols, 0.0
    observed_keys = {
        (str(r.condition), round(float(r.fraction), 9), str(r.unit_key))
        for r in fixed.dropna(subset=["fraction", "unit_key"])[["condition", "fraction", "unit_key"]].itertuples(index=False)
    }
    expected_keys = {
        (condition, round(float(frac), 9), unit)
        for condition, frac in cols for unit in units
    }
    coverage = len(observed_keys & expected_keys) / len(expected_keys)
    return fixed, units, cols, float(coverage)


def plot_checkpoint_overtopping_comparison(
    summary: pd.DataFrame,
    control: pd.DataFrame,
    trigger: pd.DataFrame,
    output: Path,
) -> bool:
    """Plot the fixed-union longitudinal checkpoint view.

    This figure is deliberately withheld unless the fixed candidate union has a
    complete unit x checkpoint x condition matrix.  Checkpoint-local discovery
    rows never fill missing longitudinal cells.
    """
    if summary.empty or control.empty:
        return False
    needed = {"condition", "fraction", "U_J", "s_1"}
    if not needed.issubset(summary.columns):
        return False

    fixed, all_units, cols, coverage = _fixed_control_coverage(summary, control)
    if fixed.empty or not all_units or not cols or coverage < 1.0 - 1e-12:
        print(
            f"[story] skipping {output.name}: fixed candidate-union coverage is {coverage:.1%}; rerun Stage 07 materialization before publishing this longitudinal figure.",
            flush=True,
        )
        return False

    fig = plt.figure(figsize=(12.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.15], hspace=0.34, wspace=0.24)
    ax_u = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, :])

    for condition, group in summary.groupby("condition"):
        group = group.dropna(subset=["fraction"]).sort_values("fraction")
        marker = "o" if condition == "clean" else "s"
        uj = pd.to_numeric(group["U_J"], errors="coerce")
        s1 = pd.to_numeric(group["s_1"], errors="coerce")
        x = 100 * pd.to_numeric(group["fraction"], errors="coerce")
        good_u = x.notna() & uj.notna()
        good_s = x.notna() & s1.notna()
        if good_u.any():
            ax_u.plot(x[good_u], 100 * uj[good_u], marker=marker, label=str(condition).capitalize())
        if good_s.any():
            ax_s.plot(x[good_s], 100 * s1[good_s], marker=marker, label=str(condition).capitalize())
        if "candidate_set_size" in group.columns:
            for _, row in group.loc[good_u].iterrows():
                j = _finite(row.get("candidate_set_size"))
                if np.isfinite(j):
                    ax_u.annotate(
                        f"J={int(j)}",
                        (100 * float(row["fraction"]), 100 * float(row["U_J"])),
                        xytext=(4, 5), textcoords="offset points", fontsize=7.5,
                    )

    ax_u.set_title("How much behavior is controlled by the local candidate set?")
    ax_u.set_ylabel("U(J) (%)")
    ax_u.set_xlabel("Training progress (%)")
    ax_u.set_ylim(bottom=0)
    ax_u.grid(True, alpha=0.16)
    ax_u.legend(frameon=False)

    ax_s.set_title("How strong is the local strongest singleton?")
    ax_s.set_ylabel("Strongest singleton flip rate (%)")
    ax_s.set_xlabel("Training progress (%)")
    ax_s.set_ylim(bottom=0)
    ax_s.grid(True, alpha=0.16)
    ax_s.legend(frameon=False)

    col_labels = [f"{cond.capitalize()}\n{100*frac:g}%" for cond, frac in cols]
    fixed["c2i_rate"] = pd.to_numeric(fixed.get("c2i_rate"), errors="coerce")
    if "discovered_at_checkpoint" not in fixed.columns:
        fixed["discovered_at_checkpoint"] = False
    max_by_unit = fixed.groupby("unit_key")["c2i_rate"].max().sort_values(ascending=False)
    unit_order = max_by_unit.head(16).index.astype(str).tolist()
    matrix = np.full((len(unit_order), len(cols)), np.nan, dtype=float)
    discovered = np.zeros((len(unit_order), len(cols)), dtype=bool)
    for j, (condition, frac) in enumerate(cols):
        part = fixed[
            fixed["condition"].astype(str).eq(condition)
            & np.isclose(pd.to_numeric(fixed["fraction"], errors="coerce"), frac)
        ]
        lookup = {
            str(r.unit_key): (_finite(r.c2i_rate), bool(getattr(r, "discovered_at_checkpoint", False)))
            for r in part[["unit_key", "c2i_rate", "discovered_at_checkpoint"]].itertuples(index=False)
        }
        for i, unit in enumerate(unit_order):
            value, was_discovered = lookup[unit]
            matrix[i, j] = 100.0 * value
            discovered[i, j] = bool(was_discovered)

    trigger_units = set(
        trigger.loc[
            trigger.get("attack_discovered_at_checkpoint", pd.Series(False, index=trigger.index)).fillna(False).astype(bool),
            "unit_key",
        ].dropna().astype(str)
    ) if not trigger.empty else set()
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.94")
    finite_vals = matrix[np.isfinite(matrix)]
    vmax = max(1.0, float(np.max(finite_vals))) if finite_vals.size else 1.0
    im = ax_h.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    ax_h.set_xticks(np.arange(len(cols)), col_labels)
    ax_h.set_yticks(np.arange(len(unit_order)), [f"{u} ★" if u in trigger_units else u for u in unit_order])
    from matplotlib.patches import Rectangle
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax_h.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=7.2,
                          color="white" if value > 0.55 * vmax else "black")
                if discovered[i, j]:
                    ax_h.add_patch(Rectangle((j-0.48, i-0.48), 0.96, 0.96, fill=False,
                                             edgecolor="black", linewidth=0.8))
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.025, pad=0.015)
    cbar.set_label("Correct -> incorrect flip rate (%)")

    ax_h.set_title("Fixed candidate-union channel effects across every checkpoint")
    ax_h.set_xlabel("Checkpoint and training condition")
    ax_h.set_ylabel("Fixed candidate-union channel")
    fig.suptitle("How does poisoned training change control-correctness overtopping?", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.012,
        "The heatmap is published only when every fixed-union channel has been evaluated at every matched clean/poisoned checkpoint. Black outlines mark checkpoint-local CHA rediscovery; lack of an outline is not a zero effect. ★ marks a channel locally discovered on the poisoned attack endpoint at least once.",
        ha="center", va="bottom", fontsize=8.3, color="dimgray",
    )
    fig.subplots_adjust(left=0.10, right=0.93, bottom=0.13, top=0.91, hspace=0.36, wspace=0.28)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return True


def _story_coverage(summary: pd.DataFrame, control: pd.DataFrame, trigger: pd.DataFrame) -> pd.DataFrame:
    """Per-checkpoint provenance audit for the paper-facing poisoning story."""
    keys: set[tuple[str, float]] = set()
    if not summary.empty:
        keys.update((str(r.condition), float(r.fraction)) for r in summary[["condition", "fraction"]].itertuples(index=False))
    if not control.empty:
        keys.update((str(r.condition), float(r.fraction)) for r in control[["condition", "fraction"]].itertuples(index=False))
    if not trigger.empty:
        keys.update(("poisoned", float(r.fraction)) for r in trigger[["fraction"]].itertuples(index=False))
    rows: list[dict[str, Any]] = []
    for condition, fraction in sorted(keys, key=lambda x: (x[1], x[0])):
        s = summary[
            summary["condition"].astype(str).eq(condition)
            & np.isclose(pd.to_numeric(summary["fraction"], errors="coerce"), fraction)
        ] if not summary.empty else pd.DataFrame()
        o = control[
            control["condition"].astype(str).eq(condition)
            & np.isclose(pd.to_numeric(control["fraction"], errors="coerce"), fraction)
        ] if not control.empty else pd.DataFrame()
        t = trigger[np.isclose(pd.to_numeric(trigger["fraction"], errors="coerce"), fraction)] \
            if condition == "poisoned" and not trigger.empty else pd.DataFrame()
        source = o.get("control_correctness_value_source", pd.Series(dtype=str)).astype(str) if not o.empty else pd.Series(dtype=str)
        discovered = o.get("discovered_at_checkpoint", pd.Series(False, index=o.index)).fillna(False).astype(bool) if not o.empty else pd.Series(dtype=bool)
        attack_source = t.get("attack_value_source", pd.Series(dtype=str)).astype(str) if not t.empty else pd.Series(dtype=str)
        attack_discovered = t.get("attack_discovered_at_checkpoint", pd.Series(False, index=t.index)).fillna(False).astype(bool) if not t.empty else pd.Series(dtype=bool)
        rows.append({
            "condition": condition,
            "fraction": float(fraction),
            "summary_row_present": bool(not s.empty),
            "control_correctness_channel_rows": int(len(o)),
            "control_correctness_locally_discovered_rows": int(discovered.sum()) if len(discovered) else 0,
            "control_correctness_fixed_union_rows": int(source.isin(CONTROL_FIXED_SOURCES).sum()) if len(source) else 0,
            "control_correctness_checkpoint_local_only_rows": int(source.eq("checkpoint_discovery").sum()) if len(source) else 0,
            "control_correctness_shared_zero_rows": int(source.eq("shared_clean_zero").sum()) if len(source) else 0,
            "attack_channel_rows": int(len(t)),
            "attack_locally_discovered_rows": int(attack_discovered.sum()) if len(attack_discovered) else 0,
            "attack_fixed_union_rows": int(attack_source.isin(ATTACK_FIXED_SOURCES).sum()) if len(attack_source) else 0,
            "attack_checkpoint_local_only_rows": int(attack_source.eq("checkpoint_discovery").sum()) if len(attack_source) else 0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument(
        "--figure6_from_story_dir", default=None,
        help=(
            "Regenerate only the clean-reference defense figure from cached story CSVs. "
            "The directory must contain clean_reference_benign_budget_screen.csv and "
            "clean_reference_benign_budget_curve.csv. No model artifacts are read."
        ),
    )
    ap.add_argument(
        "--figure6_output", default=None,
        help="Output PDF for --figure6_from_story_dir. Defaults to 03b_clean_reference_defense_interpretation.pdf in that directory.",
    )
    ap.add_argument("--phase_dir", default="prompt_and_generation")
    ap.add_argument("--eval_intervention", default="mean-donor")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--prospective_top_k", type=int, default=3, help="Maximum channels selected by each defense screen.")
    ap.add_argument(
        "--benign_damage_budget", type=float, default=0.30,
        help="Illustrative operating-point cap on poisoned-model benign C->I damage for the clean-reference screen.",
    )
    ap.add_argument(
        "--benign_budget_grid",
        default=",".join(f"{x/100:.2f}" for x in range(5, 101, 5)),
        help="Comma-separated benign-damage budgets used for the attack-blind operating curve.",
    )
    args = ap.parse_args()

    # Reproducibility path for the manuscript defense figure.  This mode uses
    # only the cached, attack-blind screen outputs written by a normal Stage-07
    # run, so the published figure can be regenerated without model access or
    # any manual PDF editing.
    if args.figure6_from_story_dir:
        story_dir = Path(args.figure6_from_story_dir).expanduser().resolve()
        selection_path = story_dir / "clean_reference_benign_budget_screen.csv"
        budget_path = story_dir / "clean_reference_benign_budget_curve.csv"
        missing = [str(path) for path in (selection_path, budget_path) if not path.is_file()]
        if missing:
            ap.error("--figure6_from_story_dir is missing required cached CSV(s): " + ", ".join(missing))
        selection = pd.read_csv(selection_path, low_memory=False)
        budget_summary = pd.read_csv(budget_path, low_memory=False)
        figure6_output = (
            Path(args.figure6_output).expanduser().resolve()
            if args.figure6_output
            else story_dir / "03b_clean_reference_defense_interpretation.pdf"
        )
        figure6_output.parent.mkdir(parents=True, exist_ok=True)
        ok = plot_clean_reference_defense_interpretation(
            selection, budget_summary, figure6_output,
            top_k=int(args.prospective_top_k),
            benign_damage_budget=float(np.clip(float(args.benign_damage_budget), 0.0, 1.0)),
        )
        if not ok:
            raise SystemExit("Figure 6 was not generated: cached clean-reference rows are incomplete.")
        print(f"[stage07-story] generated Figure 6: {figure6_output}")
        return

    if not args.run_dir:
        ap.error("--run_dir is required unless --figure6_from_story_dir is used")
    run_dir = Path(args.run_dir).expanduser().resolve()
    root = _checkpoint_root(run_dir)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "paper_overtopping_summary"
    output.mkdir(parents=True, exist_ok=True)

    # Non-destructive reruns: Stage 07 never deletes pre-existing figures or legacy outputs.
    # New outputs are written under distinct names; successful regeneration may replace only
    # the exact target filename requested by the caller.

    summary, control, trigger = load_cached_story(run_dir, args.phase_dir, args.eval_intervention)
    summary.to_csv(output / "clean_vs_poisoned_behavior_and_overtopping.csv", index=False)
    control.to_csv(output / "control_correctness_channel_flip_rates.csv", index=False)
    trigger.to_csv(output / "attack_channel_flip_rates.csv", index=False)
    coverage = _story_coverage(summary, control, trigger)
    coverage.to_csv(output / "story_data_coverage.csv", index=False)

    benign_budget = float(np.clip(float(args.benign_damage_budget), 0.0, 1.0))
    benign_budgets = _parse_benign_budget_grid(args.benign_budget_grid, include=benign_budget)
    prospective = _prospective_defense_target_selection(
        control, trigger, top_k=int(args.prospective_top_k)
    )
    # Explicitly separate the strictly prospective experiment from the
    # checkpoint-aligned diagnostic.  The comparison table keeps both.
    prospective.to_csv(output / "defense_screen_comparison.csv", index=False)
    one_ahead_screen = prospective[
        prospective.get("strategy", pd.Series(dtype=str)).astype(str).eq("one_checkpoint_ahead_frozen_overtopping")
    ].copy()
    clean_reference_screen = prospective[
        prospective.get("strategy", pd.Series(dtype=str)).astype(str).eq("clean_reference_contrast")
    ].copy()
    one_ahead_screen.to_csv(output / "prospective_defense_leverage.csv", index=False)
    one_ahead_screen.to_csv(output / "one_checkpoint_ahead_defense_screen.csv", index=False)
    clean_reference_screen.to_csv(output / "clean_reference_defense_screen.csv", index=False)

    # Figure 03b / Figure 6 uses a separate benign-budgeted screen. Keeping this
    # separate prevents the new defense analysis from changing the older Stage-07
    # selection tables or figures.
    clean_reference_budget_screen = _benign_budgeted_clean_reference_selection(
        control, trigger, top_k=int(args.prospective_top_k), benign_damage_budget=benign_budget
    )
    clean_reference_budget_screen.to_csv(
        output / "clean_reference_benign_budget_screen.csv", index=False
    )
    clean_reference_budget_rows, clean_reference_budget_summary = _clean_reference_budget_sweep(
        control, trigger, budgets=benign_budgets, top_k=int(args.prospective_top_k)
    )
    clean_reference_budget_rows.to_csv(output / "clean_reference_benign_budget_selected_channels.csv", index=False)
    clean_reference_budget_summary.to_csv(output / "clean_reference_benign_budget_curve.csv", index=False)
    _checkpoint_summary_from_selection(prospective, "one_checkpoint_ahead_frozen_overtopping").to_csv(
        output / "one_checkpoint_ahead_defense_checkpoint_summary.csv", index=False
    )
    _checkpoint_summary_from_selection(prospective, "clean_reference_contrast").to_csv(
        output / "clean_reference_defense_checkpoint_summary.csv", index=False
    )
    _checkpoint_summary_from_selection(
        clean_reference_budget_screen, "clean_reference_benign_budget"
    ).to_csv(
        output / "clean_reference_benign_budget_checkpoint_summary.csv", index=False
    )

    fig1_ok = plot_developmental_story(summary, control, trigger, output / "01_clean_vs_poisoned_overtopping_development.pdf")
    fig2_ok = plot_channel_roles(control, trigger, output / "02_channel_role_reassignment.pdf")
    fig3_ok = plot_prospective_defense_leverage(
        prospective, output / "03_prospective_defense_leverage.pdf", top_k=int(args.prospective_top_k)
    )
    fig3b_ok = plot_clean_reference_defense_interpretation(
        clean_reference_budget_screen, clean_reference_budget_summary,
        output / "03b_clean_reference_defense_interpretation.pdf",
        top_k=int(args.prospective_top_k), benign_damage_budget=benign_budget,
    )
    fig3c_ok = plot_one_checkpoint_ahead_defense_interpretation(
        prospective, output / "03c_one_checkpoint_ahead_defense_interpretation.pdf", top_k=int(args.prospective_top_k)
    )
    _, _, _, fixed_coverage = _fixed_control_coverage(summary, control)
    fig4_ok = plot_checkpoint_overtopping_comparison(
        summary, control, trigger, output / "04_clean_vs_poisoned_checkpoint_overtopping.pdf"
    )
    has_any_fixed_control = bool(_fixed_control_mask(control).any()) if not control.empty else False
    has_any_fixed_attack = bool(_fixed_attack_mask(trigger).any()) if not trigger.empty else False
    statuses = [
        {
            "figure": "01_clean_vs_poisoned_overtopping_development.pdf",
            "status": "generated" if fig1_ok else "withheld_missing_aggregate_story_data",
        },
        {
            "figure": "02_channel_role_reassignment.pdf",
            "status": (
                "generated_with_fixed_union" if fig2_ok and has_any_fixed_control
                else "generated_checkpoint_local_only" if fig2_ok
                else "withheld_missing_channel_data"
            ),
        },
        {
            "figure": "03_prospective_defense_leverage.pdf",
            "status": (
                "generated" if fig3_ok
                else "withheld_missing_fixed_control_or_attack_materialization"
            ),
        },
        {
            "figure": "03b_clean_reference_defense_interpretation.pdf",
            "status": (
                "generated" if fig3b_ok
                else "withheld_missing_clean_reference_contrast_rows"
            ),
        },
        {
            "figure": "03c_one_checkpoint_ahead_defense_interpretation.pdf",
            "status": (
                "generated" if fig3c_ok
                else "withheld_missing_one_checkpoint_ahead_rows"
            ),
        },
        {
            "figure": "04_clean_vs_poisoned_checkpoint_overtopping.pdf",
            "status": (
                "generated" if fig4_ok
                else f"withheld_incomplete_fixed_union_coverage_{fixed_coverage:.3f}"
            ),
        },
    ]
    for row in statuses:
        row["fixed_control_union_present"] = has_any_fixed_control
        row["fixed_attack_union_present"] = has_any_fixed_attack
        row["fixed_control_matrix_coverage"] = fixed_coverage
        row["n_prospective_selection_rows"] = int(len(one_ahead_screen))
        row["n_defense_screen_comparison_rows"] = int(len(prospective))
        row["n_one_checkpoint_ahead_rows"] = int(len(one_ahead_screen))
        row["n_clean_reference_same_checkpoint_rows"] = int(len(clean_reference_screen))
        row["clean_reference_benign_damage_budget"] = benign_budget
        row["n_clean_reference_budget_screen_rows"] = int(len(clean_reference_budget_screen))
        row["n_clean_reference_budget_curve_rows"] = int(len(clean_reference_budget_summary))
    pd.DataFrame(statuses).to_csv(output / "story_figure_status.csv", index=False)

    print(f"[done] paper-facing overtopping figures: {output}", flush=True)


if __name__ == "__main__":
    main()
