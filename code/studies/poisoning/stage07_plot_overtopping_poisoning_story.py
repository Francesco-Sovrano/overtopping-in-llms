#!/usr/bin/env python3
"""Paper-facing clean-vs-poisoned overtopping figures from cached Stage-03 outputs.

No model inference, training, CHA, or interventions are rerun. The script reads
existing normal-task/backdoor behavior score tables plus cached singleton/flip
statistics and Stage-07 fixed-candidate materializations. It turns them into
figures intended to answer:

  1. How does poisoning change the developmental trajectory of overtopping?
  2. Which channels change causal role, and are any attack-selective enough to
     motivate a targeted defense?
  3. How do the same control-correctness agonist channels change effect across checkpoints,
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
]
ATTACK_COLUMNS = [
    "condition", "fraction", "global_step", "unit_key",
    "attack_c2i_rate", "attack_flip_any_rate", "attack_value_source",
]


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

    Stage 07 evaluates the union of all control-correctness agonist candidates at every
    matched clean/poison checkpoint.  Those values are the correct source for
    longitudinal single-channel plots: checkpoint-local CHA non-discovery must
    not be confused with a missing causal evaluation.
    """
    root = detection_dir(run_dir) / phase_dir / "control_correctness_u_j_materialization"
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
            stats = checkpoint / "neuron_flip_rules" / "stats" / "fixed_test_all_candidates"
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
                })
    return pd.DataFrame(rows, columns=CONTROL_COLUMNS)


def _fixed_attack_materialization(run_dir: Path, phase_dir: str) -> pd.DataFrame:
    root = detection_dir(run_dir) / phase_dir / "attack_u_j_materialization" / "poisoned"
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return _empty_with_schema(ATTACK_COLUMNS)
    for checkpoint in sorted(root.glob("progress_*")):
        info = _progress(checkpoint)
        if info is None:
            continue
        fraction, step = info
        flip_path = checkpoint / "neuron_flip_rules" / "stats" / "fixed_test_positive_candidates" / "flip_stats_by_neuron.csv"
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
                "attack_flip_any_rate": _finite(rec.get("flip_any_rate")),
                "attack_value_source": "stage07_fixed_candidate_union",
            })
    return pd.DataFrame(rows, columns=ATTACK_COLUMNS)


def _combine_attack_longitudinal(local: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    key = ["condition", "fraction", "unit_key"]
    if fixed.empty:
        return local.sort_values(key).reset_index(drop=True) if not local.empty else _empty_with_schema(ATTACK_COLUMNS)
    if local.empty:
        return fixed.sort_values(key).reset_index(drop=True)
    local = local.copy()
    local["attack_value_source"] = "checkpoint_discovery"
    fixed = fixed.copy()
    local_keys = set(map(tuple, local[key].itertuples(index=False, name=None)))
    extra = fixed[[tuple(row) not in local_keys for row in fixed[key].itertuples(index=False, name=None)]]
    # Fixed-union evaluations are the preferred longitudinal values; local CHA
    # is retained only for candidates/checkpoints absent from the fixed table.
    fixed_keys = set(map(tuple, fixed[key].itertuples(index=False, name=None)))
    local_only = local[[tuple(row) not in fixed_keys for row in local[key].itertuples(index=False, name=None)]]
    return pd.concat([fixed, local_only], ignore_index=True, sort=False).sort_values(key).reset_index(drop=True)


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
        missing["discovered_at_checkpoint"] = False
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

            control_root = phase_root / "attack_cohort_control_correctness" / eval_dir
            control_stats = _choose_stats_dir(control_root, baseline_positive=False)
            if control_stats is not None:
                set_df = pd.read_csv(control_stats / "singleton_set_metrics.csv")
                if not set_df.empty:
                    s = set_df.iloc[0]
                    rec.update({
                        "candidate_set_size": _finite(s.get("candidate_set_size")),
                        "U_J": _finite(s.get("U_J")),
                        "s_1": _finite(s.get("s_1")),
                        "N_eff": _finite(s.get("N_eff")),
                        "R_ov": _finite(s.get("R_ov")),
                    })
                flip_path = control_stats / "flip_stats_by_neuron.csv"
                if flip_path.is_file():
                    for row in pd.read_csv(flip_path).to_dict("records"):
                        unit = str(row.get("neuron", row.get("unit_key", ""))).strip()
                        if not unit:
                            continue
                        control_rows.append({
                            "condition": condition,
                            "fraction": fraction,
                            "unit_key": unit,
                            "c2i_rate": _finite(row.get("c2i_rate")),
                            "i2c_rate": _finite(row.get("i2c_rate")),
                            "flip_any_rate": _finite(row.get("flip_any_rate")),
                            "discovered_at_checkpoint": True,
                            "control_correctness_value_source": "checkpoint_discovery",
                        })

            if condition == "poisoned":
                trigger_root = phase_root / "backdoor_trigger_test" / eval_dir
                trigger_stats = _choose_stats_dir(trigger_root, baseline_positive=True)
                if trigger_stats is not None:
                    set_df = pd.read_csv(trigger_stats / "singleton_set_metrics.csv")
                    if not set_df.empty:
                        s = set_df.iloc[0]
                        rec["attack_U_J"] = _finite(s.get("U_J"))
                        rec["attack_s_1"] = _finite(s.get("s_1"))
                    flip_path = trigger_stats / "flip_stats_by_neuron.csv"
                    if flip_path.is_file():
                        for row in pd.read_csv(flip_path).to_dict("records"):
                            unit = str(row.get("neuron", row.get("unit_key", ""))).strip()
                            if not unit:
                                continue
                            trigger_rows.append({
                                "condition": condition,
                                "fraction": fraction,
                                "unit_key": unit,
                                "attack_c2i_rate": _finite(row.get("c2i_rate")),
                                "attack_flip_any_rate": _finite(row.get("flip_any_rate")),
                            })
            summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows).sort_values(["fraction", "condition"]).reset_index(drop=True)
    summary = _fill_shared_zero_summary(summary)
    local_ordinary = pd.DataFrame(control_rows, columns=CONTROL_COLUMNS)
    fixed_ordinary = _fixed_control_correctness_materialization(run_dir, phase_dir)
    control = _combine_control_correctness_longitudinal(local_ordinary, fixed_ordinary)
    control = _fill_shared_zero_ordinary(control)
    local_trigger = pd.DataFrame(trigger_rows, columns=ATTACK_COLUMNS)
    fixed_trigger = _fixed_attack_materialization(run_dir, phase_dir)
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


def _m1_selectivity(summary: pd.DataFrame, control: pd.DataFrame, trigger: pd.DataFrame, unit: str = "m1:534") -> pd.DataFrame:
    po = control[(control["condition"] == "poisoned") & (control["unit_key"] == unit)].copy()
    tr = trigger[trigger["unit_key"] == unit].copy()
    rows = []
    for frac in sorted(set(po.get("fraction", pd.Series(dtype=float))).union(set(tr.get("fraction", pd.Series(dtype=float))))):
        o = po[np.isclose(po["fraction"], frac)]
        t = tr[np.isclose(tr["fraction"], frac)]
        rows.append({
            "fraction": frac,
            "control_correctness_c2i_rate": _finite(o.iloc[0]["c2i_rate"]) if not o.empty else math.nan,
            "attack_c2i_rate": _finite(t.iloc[0]["attack_c2i_rate"]) if not t.empty else math.nan,
        })
    return pd.DataFrame(rows)


def plot_developmental_story(summary: pd.DataFrame, control: pd.DataFrame, trigger: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.8))
    _plot_metric(axes[0,0], summary, "normal_accuracy", "Normal behavior", "Accuracy (%)")
    _plot_metric(axes[0,1], summary, "attack_conversion", "Backdoor behavior", "Conditional conversion (%)")
    _plot_metric(axes[0,2], summary, "U_J", "Set-level overtopping", "U(J) (%)")
    _plot_metric(axes[1,0], summary, "s_1", "Strongest causal bottleneck", "Strongest singleton effect (%)")
    _plot_metric(axes[1,1], summary, "N_eff", "Effective causal support", "Effective number of channels", percent=False)

    sel = _m1_selectivity(summary, control, trigger)
    ax = axes[1,2]
    if not sel.empty:
        x = 100 * sel["fraction"].to_numpy(float)
        ax.plot(x, 100*sel["attack_c2i_rate"], marker="D", linewidth=2.2, label="Attack causal effect")
        ax.plot(x, 100*sel["control_correctness_c2i_rate"], marker="o", linewidth=2.2, label="Control-correctness cost")
        ax.fill_between(x, 100*sel["control_correctness_c2i_rate"], 100*sel["attack_c2i_rate"], alpha=0.10)
        for row in sel.itertuples(index=False):
            if np.isfinite(row.attack_c2i_rate) and np.isfinite(row.control_correctness_c2i_rate):
                ax.annotate(f"gap {100*(row.attack_c2i_rate-row.control_correctness_c2i_rate):.0f} pp",
                            (100*row.fraction, 100*row.attack_c2i_rate), xytext=(4,-14),
                            textcoords="offset points", fontsize=8)
    ax.set_title("Defense leverage: m1:534", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Training progress (%)")
    ax.set_ylabel("Intervention effect (%)")
    ax.set_ylim(0,105)
    ax.grid(True, alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)

    axes[0,0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Poisoning changes the developmental trajectory and role of overtopping", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.012,
             "Read left-to-right: a useful poisoning signature is not simply 'more overtopping'. The key pattern is attack conversion rising while normal accuracy stays similar, "
             "followed by poisoned U(J)/strongest-singleton control remaining high when clean control collapses. The final panel asks whether a causal bottleneck is attack-selective enough to target.",
             ha="center", va="bottom", fontsize=8.4, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0,0.055,1,0.955))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _checkpoint_channel_panel(ax, control: pd.DataFrame, trigger: pd.DataFrame, fraction: float, top_n: int = 8) -> None:
    control_fraction = pd.to_numeric(control["fraction"], errors="coerce").to_numpy(float)
    trigger_fraction = pd.to_numeric(trigger["fraction"], errors="coerce").to_numpy(float)
    clean = control[(control["condition"] == "clean") & np.isclose(control_fraction, fraction, equal_nan=False)]
    poison = control[(control["condition"] == "poisoned") & np.isclose(control_fraction, fraction, equal_nan=False)]
    attack = trigger[np.isclose(trigger_fraction, fraction, equal_nan=False)]
    units = set(clean.get("unit_key", [])) | set(poison.get("unit_key", [])) | set(attack.get("unit_key", []))
    scored = []
    for u in units:
        c = clean[clean["unit_key"] == u]
        p = poison[poison["unit_key"] == u]
        a = attack[attack["unit_key"] == u]
        cv = _finite(c.iloc[0]["c2i_rate"]) if not c.empty else math.nan
        pv = _finite(p.iloc[0]["c2i_rate"]) if not p.empty else math.nan
        av = _finite(a.iloc[0]["attack_c2i_rate"]) if not a.empty else math.nan
        cd = bool(c.iloc[0].get("discovered_at_checkpoint", True)) if not c.empty else False
        pd_ = bool(p.iloc[0].get("discovered_at_checkpoint", True)) if not p.empty else False
        candidates = [v for v in (cv,pv,av) if np.isfinite(v)]
        score = max(candidates) if candidates else -1
        scored.append((u,cv,pv,av,cd,pd_,score))
    scored.sort(key=lambda x: x[-1], reverse=True)
    scored = scored[:top_n]
    scored.reverse()
    y = np.arange(len(scored))
    clean_color = "#4C78A8"
    poison_color = "#F58518"
    attack_color = "#D62728"
    for i,(u,cv,pv,av,cd,pd_,_) in enumerate(scored):
        if np.isfinite(cv):
            ax.scatter(100*cv, i, marker="o", s=52, facecolors=clean_color if cd else "none",
                       edgecolors=clean_color, linewidths=1.4, zorder=3)
        if np.isfinite(pv):
            ax.scatter(100*pv, i, marker="s", s=52, facecolors=poison_color if pd_ else "none",
                       edgecolors=poison_color, linewidths=1.4, zorder=3)
        if np.isfinite(av):
            ax.scatter(100*av, i, marker="D", s=54, color=attack_color, zorder=3)
        finite = [100*v for v in (cv,pv,av) if np.isfinite(v)]
        if len(finite) >= 2:
            ax.plot([min(finite),max(finite)],[i,i], linewidth=0.9, color="0.78", zorder=1)
    ax.set_yticks(y, [r[0] for r in scored])
    ax.set_xlim(0,105)
    ax.set_xlabel("Correct->incorrect intervention effect (%)")
    ax.set_title(f"{100*fraction:g}% checkpoint", fontweight="bold")
    ax.grid(axis="x", alpha=0.15)


def plot_channel_roles(control: pd.DataFrame, trigger: pd.DataFrame, output: Path) -> None:
    matched = sorted(set(control.loc[control["condition"]=="clean","fraction"]).intersection(
                     set(control.loc[control["condition"]=="poisoned","fraction"])))
    if not matched:
        return
    # The old implementation silently truncated to the first two checkpoints.
    # Show every matched post-training checkpoint; fall back to 0% only if that
    # is all that exists.
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
        Line2D([0],[0], marker="o", linestyle="None", markersize=7, markerfacecolor="#4C78A8", markeredgecolor="#4C78A8", label="Clean control - locally discovered"),
        Line2D([0],[0], marker="s", linestyle="None", markersize=7, markerfacecolor="#F58518", markeredgecolor="#F58518", label="Poisoned control - locally discovered"),
        Line2D([0],[0], marker="o", linestyle="None", markersize=7, markerfacecolor="none", markeredgecolor="0.35", label="Hollow - fixed-union evaluated, not rediscovered"),
        Line2D([0],[0], marker="D", linestyle="None", markersize=7, color="#D62728", label="Poisoned attack"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5,0.965))
    fig.suptitle("Which channels change causal role under poisoning?", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.012,
             "Ordinary circles/squares use the Stage-07 fixed candidate-union evaluation when available. Filled markers were locally rediscovered by CHA; hollow markers were evaluated longitudinally but not rediscovered. Attack diamonds are checkpoint-local trigger candidates.",
             ha="center", va="bottom", fontsize=8.5, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0,0.045,1,0.915))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)



def plot_checkpoint_overtopping_comparison(
    summary: pd.DataFrame,
    control: pd.DataFrame,
    trigger: pd.DataFrame,
    output: Path,
) -> None:
    """Restore the richer cached flip-stats checkpoint view.

    This is the cache-only counterpart of the Stage-07 interpretation figure:
    U(J) and strongest-singleton trajectories above, individual agonist C->I
    effects below.  Missing channel/checkpoint cells mean not discovered, not
    zero effect.
    """
    if summary.empty or control.empty:
        return
    needed = {"condition", "fraction", "U_J", "s_1"}
    if not needed.issubset(summary.columns):
        return

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

    ax_u.set_title("How much behavior is controlled by the candidate set?")
    ax_u.set_ylabel("U(J): examples changed by >=1 candidate (%)")
    ax_u.set_xlabel("Training progress (%)")
    ax_u.set_ylim(bottom=0)
    ax_u.grid(True, alpha=0.16)
    ax_u.legend(frameon=False)

    ax_s.set_title("How strong is the single strongest candidate?")
    ax_s.set_ylabel("Strongest singleton flip rate (%)")
    ax_s.set_xlabel("Training progress (%)")
    ax_s.set_ylim(bottom=0)
    ax_s.grid(True, alpha=0.16)
    ax_s.legend(frameon=False)

    cols: list[tuple[str, float]] = []
    fractions = sorted(pd.to_numeric(summary["fraction"], errors="coerce").dropna().unique())
    for frac in fractions:
        for condition in ("clean", "poisoned"):
            present = (
                summary["condition"].astype(str).eq(condition)
                & np.isclose(pd.to_numeric(summary["fraction"], errors="coerce"), float(frac))
            ).any()
            if present:
                cols.append((condition, float(frac)))
    col_labels = [f"{cond.capitalize()}\n{100*frac:g}%" for cond, frac in cols]

    control = control.copy()
    control["c2i_rate"] = pd.to_numeric(control.get("c2i_rate"), errors="coerce")
    if "discovered_at_checkpoint" not in control.columns:
        control["discovered_at_checkpoint"] = True
    max_by_unit = control.groupby("unit_key")["c2i_rate"].max().sort_values(ascending=False)
    unit_order = max_by_unit.head(16).index.astype(str).tolist()
    matrix = np.full((len(unit_order), len(cols)), np.nan, dtype=float)
    discovered = np.zeros((len(unit_order), len(cols)), dtype=bool)
    for j, (condition, frac) in enumerate(cols):
        part = control[
            control["condition"].astype(str).eq(condition)
            & np.isclose(pd.to_numeric(control["fraction"], errors="coerce"), frac)
        ]
        lookup = {
            str(r.unit_key): (_finite(r.c2i_rate), bool(getattr(r, "discovered_at_checkpoint", True)))
            for r in part[["unit_key", "c2i_rate", "discovered_at_checkpoint"]].itertuples(index=False)
        }
        for i, unit in enumerate(unit_order):
            value, was_discovered = lookup.get(unit, (math.nan, False))
            if np.isfinite(value):
                matrix[i, j] = 100.0 * value
                discovered[i, j] = bool(was_discovered)

    trigger_units = set(trigger.get("unit_key", pd.Series(dtype=str)).dropna().astype(str))
    if len(unit_order) and len(cols):
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
    else:
        ax_h.text(0.5, 0.5, "No cached singleton channel rows", ha="center", va="center", transform=ax_h.transAxes)

    ax_h.set_title("How do fixed agonist channels change causal effect across checkpoints?")
    ax_h.set_xlabel("Checkpoint and training condition")
    ax_h.set_ylabel("Fixed candidate-union channel")
    fig.suptitle("How does poisoned training change control-correctness overtopping?", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.012,
        "Colored cells are fixed-union singleton evaluations when Stage 07 materialization is available; black outlines mark checkpoint-local CHA rediscovery. "
        "Blank means no fixed evaluation was available. ★ marks a channel also present in the poisoned attack/trigger candidate table.",
        ha="center", va="bottom", fontsize=8.3, color="dimgray",
    )
    fig.subplots_adjust(left=0.10, right=0.93, bottom=0.13, top=0.91, hspace=0.36, wspace=0.28)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


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
        rows.append({
            "condition": condition,
            "fraction": float(fraction),
            "summary_row_present": bool(not s.empty),
            "control_correctness_channel_rows": int(len(o)),
            "control_correctness_locally_discovered_rows": int(discovered.sum()) if len(discovered) else 0,
            "control_correctness_fixed_union_rows": int(source.eq("stage07_fixed_candidate_union").sum()) if len(source) else 0,
            "control_correctness_shared_zero_rows": int(source.eq("shared_clean_zero").sum()) if len(source) else 0,
            "attack_channel_rows": int(len(t)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--phase_dir", default="prompt_and_generation")
    ap.add_argument("--eval_intervention", default="mean-donor")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    root = _checkpoint_root(run_dir)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "paper_overtopping_summary"
    output.mkdir(parents=True, exist_ok=True)
    summary, control, trigger = load_cached_story(run_dir, args.phase_dir, args.eval_intervention)
    summary.to_csv(output / "clean_vs_poisoned_behavior_and_overtopping.csv", index=False)
    control.to_csv(output / "control_correctness_channel_flip_rates.csv", index=False)
    trigger.to_csv(output / "attack_channel_flip_rates.csv", index=False)
    _story_coverage(summary, control, trigger).to_csv(output / "story_data_coverage.csv", index=False)
    plot_developmental_story(summary, control, trigger, output / "01_clean_vs_poisoned_overtopping_development.pdf")
    plot_channel_roles(control, trigger, output / "02_channel_role_reassignment_and_defense_leverage.pdf")
    plot_checkpoint_overtopping_comparison(
        summary, control, trigger, output / "04_clean_vs_poisoned_checkpoint_overtopping.pdf"
    )
    print(f"[done] paper-facing overtopping figures: {output}", flush=True)


if __name__ == "__main__":
    main()
