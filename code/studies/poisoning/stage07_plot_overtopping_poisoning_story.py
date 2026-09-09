#!/usr/bin/env python3
"""Paper-facing clean-vs-poisoned overtopping figures from cached Stage-03 outputs.

No model inference, training, CHA, or interventions are rerun. The script reads
existing normal-task/backdoor behavior score tables plus cached singleton/flip
statistics and Stage-07 fixed-candidate materializations. It turns them into
figures intended to answer:

  1. How does poisoning change the aggregate developmental trajectory of overtopping?
  2. Which channels change causal role at each checkpoint?
  3. Can channels be prioritized prospectively using only checkpoint 0 or the
     previous checkpoint, without target-checkpoint cherry-picking?
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


def _choose_stats_dir(root: Path, *, baseline_positive: bool = False) -> Path | None:
    candidates = sorted({p.parent for p in root.rglob("singleton_set_metrics.csv")})
    candidates = [p for p in candidates if "tau0.3" in p.name]
    if baseline_positive:
        preferred = [p for p in candidates if "baseline_positive" in p.name]
    else:
        preferred = [p for p in candidates if "heldout_test" in p.name and "cap0" in p.name]
    pool = preferred or candidates
    return pool[-1] if pool else None


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
        output.unlink(missing_ok=True)
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
        output.unlink(missing_ok=True)
        return False
    matched = sorted(set(pd.to_numeric(control.loc[control["condition"].astype(str)=="clean","fraction"], errors="coerce").dropna()).intersection(
                     set(pd.to_numeric(control.loc[control["condition"].astype(str)=="poisoned","fraction"], errors="coerce").dropna())))
    if not matched:
        output.unlink(missing_ok=True)
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
    """Build attack-blind defense-target screens from cached singleton effects.

    The left-panel ``baseline_locked`` rule is retained as a historical baseline:
    channels are localized at the shared 0% checkpoint and then evaluated later.

    The right-panel rule is now ``clean_reference_contrast``.  It implements the
    clean-reference screening procedure that is useful in a federated-style
    setting when a defender has a trusted clean reference but does *not* know
    which examples in the observed training stream are poisoned:

      1. At each checkpoint, rank fixed-union channels by how much singleton
         intervention disrupts the matched clean reference (clean C->I rate).
      2. Protect the ``top_k`` most clean-important channels from selection.
      3. On the model trained on the observed clean+poisoned stream, compute the
         same fixed-union control-correctness disruption rate.
      4. Rank the remaining channels by positive excess disruption

             poison_excess = U_poisoned(j) - U_clean(j)

         and select at most ``top_k`` channels with the largest positive excess.

    This ranking is attack-blind: trigger/attack effects are joined *only after*
    selection to evaluate attack suppression.  The cached Stage-07 matrix is a
    dense matched clean/control-correctness intervention matrix, not a dense
    singleton intervention matrix over the raw mixed training examples.  Thus
    ``U_poisoned - U_clean`` is the no-rerun proxy for poisoning-specific causal
    importance that the existing CSVs support; an exact mixed-example disruption
    score would require new intervention materialization.
    """
    columns = [
        "strategy", "selection_fraction", "target_fraction", "unit_key",
        "selection_score",
        # Raw endpoint names are kept for traceability to the source tables.
        "target_attack_c2i_rate", "target_control_c2i_rate",
        # Semantic aliases make the defense interpretation explicit.
        "target_attack_suppression_rate", "target_benign_damage_rate",
        "target_defense_leverage_proxy",
        # Clean-reference contrast diagnostics for the new right-panel screen.
        "reference_clean_c2i_rate", "observed_poisoned_c2i_rate",
        "poison_excess_disruption", "clean_importance_rank",
        "poisoned_importance_rank", "clean_protected_cutoff_rank",
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
    if not attack_fractions:
        return pd.DataFrame(columns=columns)

    def value_at(frame: pd.DataFrame, frac: float, unit: str, value_col: str) -> float:
        part = frame[
            np.isclose(frame["fraction"].to_numpy(float), float(frac), equal_nan=False)
            & frame["unit_key"].astype(str).eq(str(unit))
        ]
        return _finite(part.iloc[0][value_col]) if not part.empty else math.nan

    rows: list[dict[str, Any]] = []

    # Historical baseline: fixed at the shared initialization and evaluated at
    # every attack-defined checkpoint.  This gives a stable left-panel baseline
    # against which the clean-reference contrast rule can be compared.
    control_fractions = sorted(
        set(clean_control["fraction"].dropna()).union(set(poisoned_control["fraction"].dropna()))
    )
    zero_candidates = [f for f in control_fractions if np.isclose(float(f), 0.0)]
    if not zero_candidates:
        return pd.DataFrame(columns=columns)
    base = float(zero_candidates[0])

    clean0 = clean_control[np.isclose(clean_control["fraction"].to_numpy(float), base, equal_nan=False)]
    poison0 = poisoned_control[np.isclose(poisoned_control["fraction"].to_numpy(float), base, equal_nan=False)]
    baseline_scores_by_unit: dict[str, float] = {}
    for frame in (clean0, poison0):
        if frame.empty or "candidate_localization_score" not in frame.columns:
            continue
        discovered = frame.get("discovered_at_checkpoint", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        for rec in frame.loc[discovered].itertuples(index=False):
            score = _finite(getattr(rec, "candidate_localization_score", math.nan))
            unit = str(rec.unit_key)
            if np.isfinite(score):
                baseline_scores_by_unit[unit] = max(score, baseline_scores_by_unit.get(unit, -math.inf))
    baseline_scores = sorted(baseline_scores_by_unit.items(), key=lambda x: (-x[1], x[0]))
    baseline_selected = baseline_scores[:max(1, int(top_k))]
    for target in attack_fractions:
        for unit, score in baseline_selected:
            attack_value = value_at(fixed_attack, float(target), unit, "attack_c2i_rate")
            control_value = value_at(poisoned_control, float(target), unit, "c2i_rate")
            clean_value = value_at(clean_control, float(target), unit, "c2i_rate")
            if not (np.isfinite(attack_value) and np.isfinite(control_value)):
                continue
            rows.append({
                "strategy": "baseline_locked",
                "selection_fraction": base,
                "target_fraction": float(target),
                "unit_key": unit,
                "selection_score": float(score),
                "target_attack_c2i_rate": attack_value,
                "target_control_c2i_rate": control_value,
                "target_attack_suppression_rate": attack_value,
                "target_benign_damage_rate": control_value,
                "target_defense_leverage_proxy": attack_value - control_value,
                "reference_clean_c2i_rate": clean_value,
                "observed_poisoned_c2i_rate": control_value,
                "poison_excess_disruption": control_value - clean_value if np.isfinite(clean_value) else math.nan,
                "clean_importance_rank": math.nan,
                "poisoned_importance_rank": math.nan,
                "clean_protected_cutoff_rank": math.nan,
                "selection_basis": "0% defender-visible CHA baseline; attack information not used for selection",
            })

    # Clean-reference contrast screen.  Unlike the old previous-checkpoint CHA
    # rule, this does not depend on sparse checkpoint-local rediscovery.  The
    # fixed-union singleton matrix has a value for every candidate at every
    # matched checkpoint, so all attack-defined checkpoints can be screened.
    protect_k_requested = max(1, int(top_k))
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

        # Rank by direct disruption on the trusted clean reference.  Protecting
        # these channels makes the screen explicitly conservative about clean
        # utility rather than merely hoping the final attack-vs-benign contrast
        # will penalize them after selection.
        paired["_clean_rank"] = paired["_clean_c2i"].rank(method="min", ascending=False)
        paired["_poisoned_rank"] = paired["_poisoned_c2i"].rank(method="min", ascending=False)
        paired["_poison_excess"] = paired["_poisoned_c2i"] - paired["_clean_c2i"]
        protect_k = min(protect_k_requested, int(len(paired)))
        paired["_clean_protected"] = paired["_clean_rank"] <= float(protect_k)

        # This is the requested difference-set logic in a stable numeric form:
        # remove clean-essential channels, then rank the remaining channels by
        # how much more disruptive they became under poisoned/mixed training.
        pool = paired.loc[(~paired["_clean_protected"]) & (paired["_poison_excess"] > 0.0)].copy()
        pool = pool.sort_values(
            ["_poison_excess", "_poisoned_c2i", "unit_key"],
            ascending=[False, False, True],
        ).head(protect_k_requested)

        for rec in pool.to_dict("records"):
            unit = str(rec["unit_key"])
            attack_value = value_at(fixed_attack, float(target_frac), unit, "attack_c2i_rate")
            if not np.isfinite(attack_value):
                # Selection itself is still valid, but Figure 03 is an evaluated
                # defense-leverage figure, so only plot rows with an attack-side
                # post-selection outcome.
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
                "selection_basis": (
                    f"same-checkpoint attack-blind clean-reference contrast: protect top {protect_k} "
                    "clean-disruption channels, then rank remaining fixed-union channels by positive "
                    "poisoned-trained minus clean-trained control-correctness C->I excess"
                ),
            })
    return pd.DataFrame(rows, columns=columns)

def plot_prospective_defense_leverage(selection: pd.DataFrame, output: Path, *, top_k: int = 3) -> bool:
    if selection.empty:
        output.unlink(missing_ok=True)
        return False

    # Do not let matplotlib infer the checkpoint domain from only the rows that
    # happened to survive prospective selection.  In particular, a run can
    # have a valid one-step-ahead target at 10% and no positive previous-
    # checkpoint candidates thereafter.  Autoscaling that single point produces
    # a misleading ~9.6--10.4% x-axis and makes the panel look like the remaining
    # checkpoints were never evaluated.  The baseline-locked rows encode the
    # attack-defined checkpoint schedule, so use that schedule to define both
    # panels' x domains.
    baseline_targets = sorted(
        pd.to_numeric(
            selection.loc[selection["strategy"].astype(str).eq("baseline_locked"), "target_fraction"],
            errors="coerce",
        ).dropna().unique()
    )
    expected_targets = {
        "baseline_locked": baseline_targets,
        # The clean-reference contrast uses the dense fixed-union matrix at the
        # same checkpoint, so every attack-defined checkpoint is screenable.
        "clean_reference_contrast": baseline_targets,
    }

    def _set_checkpoint_axis(ax: Any, fracs: list[float]) -> None:
        if not fracs:
            return
        ticks = 100.0 * np.asarray(fracs, dtype=float)
        ax.set_xticks(ticks)
        lo = float(np.nanmin(ticks)); hi = float(np.nanmax(ticks))
        if np.isclose(lo, hi):
            # Defensive fallback; normally the previous-checkpoint grid has
            # multiple planned targets even when only one target has data.
            pad = 5.0
        else:
            pad = max(2.5, 0.05 * (hi - lo))
        ax.set_xlim(max(0.0, lo - pad), min(100.0, hi + pad))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    specs = [
        ("baseline_locked", "Pre-training-locked candidates (selected at 0%)"),
        ("clean_reference_contrast", "Clean-reference contrast targets (attack-blind)"),
    ]
    any_panel = False
    for ax, (strategy, title) in zip(axes, specs):
        part = selection[selection["strategy"].astype(str).eq(strategy)].copy()
        planned = expected_targets.get(strategy, [])
        _set_checkpoint_axis(ax, planned)
        if part.empty:
            ax.text(0.5, 0.5, "No attack-agnostic selection available", ha="center", va="center", transform=ax.transAxes, color="0.35")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Target checkpoint (%)")
            ax.grid(True, alpha=0.15)
            continue
        agg_spec: dict[str, tuple[str, str]] = {
            "attack_suppression": ("target_attack_suppression_rate", "mean"),
            "benign_damage": ("target_benign_damage_rate", "mean"),
            "defense_leverage": ("target_defense_leverage_proxy", "mean"),
            "n_selected": ("unit_key", "nunique"),
            "selection_fraction": ("selection_fraction", "first"),
        }
        if "reference_clean_c2i_rate" in part.columns:
            agg_spec["clean_reference_damage"] = ("reference_clean_c2i_rate", "mean")
        if "poison_excess_disruption" in part.columns:
            agg_spec["poison_excess_disruption"] = ("poison_excess_disruption", "mean")
        agg = part.groupby("target_fraction", as_index=False).agg(**agg_spec).sort_values("target_fraction")
        x = 100 * agg["target_fraction"].to_numpy(float)
        attack = 100 * agg["attack_suppression"].to_numpy(float)
        benign = 100 * agg["benign_damage"].to_numpy(float)
        leverage = 100 * agg["defense_leverage"].to_numpy(float)
        ax.plot(x, attack, marker="D", linewidth=2.2, label="Attack suppression")
        ax.plot(x, benign, marker="o", linewidth=2.2, label="Poison-trained benign damage")
        if strategy == "clean_reference_contrast" and "clean_reference_damage" in agg.columns:
            clean_ref = 100 * pd.to_numeric(agg["clean_reference_damage"], errors="coerce").to_numpy(float)
            ax.plot(x, clean_ref, marker="s", linewidth=1.8, linestyle="--", label="Clean-reference disruption")
        # The vertical difference is the operational selectivity signal: how
        # much more the singleton intervention suppresses the attack than it
        # damages non-trigger correctness.
        ax.fill_between(x, benign, attack, where=(attack >= benign), alpha=0.12, interpolate=True, label="Potential defense leverage (Δ>0)")
        ax.fill_between(x, benign, attack, where=(attack < benign), alpha=0.06, interpolate=True)
        for rec, lev in zip(agg.itertuples(index=False), leverage):
            y = 100 * float(rec.attack_suppression)
            # Keep annotations inside the axes for near-zero attack suppression.
            offset = (0, 10) if y < 12 else (0, -24)
            lev_text = f"{lev:+.1f}" if abs(float(lev)) < 1.0 else f"{lev:+.0f}"
            ax.annotate(
                f"n={int(rec.n_selected)}\nΔdef={lev_text} pp",
                (100*float(rec.target_fraction), y),
                xytext=offset, textcoords="offset points", fontsize=7.0, ha="center",
            )

        # Show any genuinely absent target checkpoints explicitly.  With the
        # dense fixed-union clean-reference contrast this should normally be
        # empty; keeping the check prevents a silently truncated series if an
        # upstream cache is incomplete.
        if strategy == "clean_reference_contrast" and planned:
            observed = set(np.round(pd.to_numeric(agg["target_fraction"], errors="coerce").dropna().to_numpy(float), 12))
            missing = [f for f in planned if round(float(f), 12) not in observed]
            if missing:
                labels = ", ".join(f"{100*float(f):g}%" for f in missing)
                ax.text(
                    0.98, 0.97,
                    f"Missing fixed-union contrast data at: {labels}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color="0.4",
                )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Target checkpoint (%)")
        ax.grid(True, alpha=0.15)
        any_panel = True
    axes[0].set_ylabel("Singleton intervention rate (%)")
    axes[0].set_ylim(0, 105)
    legend_by_label: dict[str, Any] = {}
    for axis in axes:
        hs, labels = axis.get_legend_handles_labels()
        for h, label in zip(hs, labels):
            legend_by_label.setdefault(label, h)
    if legend_by_label:
        fig.legend(
            handles=list(legend_by_label.values()),
            labels=list(legend_by_label.keys()),
            loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955), fontsize=8.5,
        )
    fig.suptitle("Defense leverage from clean-reference channel contrast", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.012,
        f"Right panel: protect the top {int(top_k)} clean-essential channels at each checkpoint, then select up to {int(top_k)} remaining fixed-union channels with the largest positive poisoned-trained minus clean-trained C->I excess. Attack effects are used only after selection; Δdef = attack suppression - poisoned-model benign damage.\nThe existing caches contain dense clean-vs-poisoned control-correctness singleton effects, not dense singleton effects on the raw mixed training examples, so this is an attack-blind mixed-training-effect contrast that requires no new inference.",
        ha="center", va="bottom", fontsize=8.0, color="dimgray", wrap=True,
    )
    fig.tight_layout(rect=(0,0.095,1,0.885))
    if any_panel:
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return True
    plt.close(fig)
    output.unlink(missing_ok=True)
    return False


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
        output.unlink(missing_ok=True)
        return False
    needed = {"condition", "fraction", "U_J", "s_1"}
    if not needed.issubset(summary.columns):
        output.unlink(missing_ok=True)
        return False

    fixed, all_units, cols, coverage = _fixed_control_coverage(summary, control)
    if fixed.empty or not all_units or not cols or coverage < 1.0 - 1e-12:
        output.unlink(missing_ok=True)
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
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase_dir", default="prompt_and_generation")
    ap.add_argument("--eval_intervention", default="mean-donor")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--prospective_top_k", type=int, default=3, help="Maximum channels in each prospective selection rule.")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    root = _checkpoint_root(run_dir)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "paper_overtopping_summary"
    output.mkdir(parents=True, exist_ok=True)

    # Remove legacy names so rerunning into an existing directory cannot leave a
    # stale, misleading story with duplicate numbering/semantics.
    for legacy in output.glob("*channel_role_reassignment*defense_leverage*.pdf"):
        legacy.unlink(missing_ok=True)
    for legacy_name in (
        "03_prospective_channel_selectivity.pdf",
        "prospective_channel_selectivity.csv",
    ):
        (output / legacy_name).unlink(missing_ok=True)

    summary, control, trigger = load_cached_story(run_dir, args.phase_dir, args.eval_intervention)
    summary.to_csv(output / "clean_vs_poisoned_behavior_and_overtopping.csv", index=False)
    control.to_csv(output / "control_correctness_channel_flip_rates.csv", index=False)
    trigger.to_csv(output / "attack_channel_flip_rates.csv", index=False)
    coverage = _story_coverage(summary, control, trigger)
    coverage.to_csv(output / "story_data_coverage.csv", index=False)

    prospective = _prospective_defense_target_selection(control, trigger, top_k=int(args.prospective_top_k))
    prospective.to_csv(output / "prospective_defense_leverage.csv", index=False)
    clean_reference_screen = prospective[
        prospective.get("strategy", pd.Series(dtype=str)).astype(str).eq("clean_reference_contrast")
    ].copy()
    clean_reference_screen.to_csv(output / "clean_reference_defense_screen.csv", index=False)

    fig1_ok = plot_developmental_story(summary, control, trigger, output / "01_clean_vs_poisoned_overtopping_development.pdf")
    fig2_ok = plot_channel_roles(control, trigger, output / "02_channel_role_reassignment.pdf")
    fig3_ok = plot_prospective_defense_leverage(
        prospective, output / "03_prospective_defense_leverage.pdf", top_k=int(args.prospective_top_k)
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
        row["n_prospective_selection_rows"] = int(len(prospective))
    pd.DataFrame(statuses).to_csv(output / "story_figure_status.csv", index=False)

    print(f"[done] paper-facing overtopping figures: {output}", flush=True)


if __name__ == "__main__":
    main()
