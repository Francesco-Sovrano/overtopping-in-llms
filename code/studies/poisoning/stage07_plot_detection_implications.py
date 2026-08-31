#!/usr/bin/env python3
"""Compact, implication-first plots for Stage-07 poisoning-example detection.

This module is intentionally post-hoc.  It never participates in causal-channel
selection or training-row scoring.  It combines two already-computed outputs:

1. Stage 07: how well poisoned training rows can be ranked from disruption of
   attack-cohort control-correctness channels.
2. Stage 04: how effective the backdoor is on the held-out attack cohort.

The primary backdoor-efficacy quantity is conditional conversion among examples
that were not already predicted as the attack target without the trigger.  This
is preferred over raw triggered target rate because it discounts baseline target
bias.  Raw triggered target rate and trigger-induced target-rate change are
retained in the comparison CSV.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from studies.poisoning.lib.specificity import truthy
from studies.poisoning.lib.run_paths import (
    safe_component,
    BACKDOOR_TRIGGER_TEST_DIRNAME,
    comparisons_dir,
    detection_dir,
    format_fraction_percent,
    phase_dirname,
)

VISUALIZATION_SCHEMA_VERSION = 3
ASSOCIATION_TEST_SCHEMA_VERSION = 1
ASSOCIATION_EXACT_MAX_N = 9
ASSOCIATION_MONTE_CARLO_DRAWS = 100000
ASSOCIATION_RANDOM_SEED = 1729




def _behavior_trajectory_candidates(run_dir: Path, phase: str, eval_intervention: str) -> list[Path]:
    root = comparisons_dir(run_dir) / phase_dirname(phase) / f"eval_{safe_component(eval_intervention)}"
    return [root / BACKDOOR_TRIGGER_TEST_DIRNAME / "trajectory.csv"]


def load_backdoor_behavior_trajectory(
    run_dir: Path,
    *,
    phase: str,
    eval_intervention: str,
) -> tuple[pd.DataFrame, Path | None]:
    """Load the Stage-04 backdoor trajectory, if available."""
    for path in _behavior_trajectory_candidates(run_dir, phase, eval_intervention):
        if not path.is_file():
            continue
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            return df, path
        return df, path
    return pd.DataFrame(), None


def _row_at_fraction(df: pd.DataFrame, condition: str, fraction: float) -> pd.Series | None:
    if df.empty or "condition" not in df.columns or "fraction" not in df.columns:
        return None
    frac = pd.to_numeric(df["fraction"], errors="coerce")
    mask = df["condition"].astype(str).str.lower().eq(str(condition).lower()) & np.isclose(
        frac.to_numpy(dtype=float), float(fraction), atol=1e-9, rtol=0.0, equal_nan=False
    )
    rows = df.loc[mask]
    if rows.empty:
        return None
    if "global_step" in rows.columns:
        rows = rows.sort_values("global_step")
    return rows.iloc[-1]


def _number(row: pd.Series | None, key: str) -> float:
    if row is None or key not in row.index:
        return math.nan
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else math.nan


def build_detection_vs_attack_table(
    detection_metrics: pd.DataFrame,
    backdoor_trajectory: pd.DataFrame,
) -> pd.DataFrame:
    """Align interval-level rankability with backdoor behavior at interval end.

    Detection metrics describe training rows *inside* [start_fraction,end_fraction].
    Backdoor metrics describe the held-out model behavior *at* end_fraction.
    The alignment is descriptive and must not be interpreted as a causal test.
    """
    rows: list[dict[str, Any]] = []
    for rec in detection_metrics.to_dict("records"):
        end = pd.to_numeric(pd.Series([rec.get("end_fraction")]), errors="coerce").iloc[0]
        start = pd.to_numeric(pd.Series([rec.get("start_fraction")]), errors="coerce").iloc[0]
        if pd.isna(end):
            continue
        end_f = float(end)
        start_f = float(start) if pd.notna(start) else math.nan
        poison = _row_at_fraction(backdoor_trajectory, "poisoned", end_f)
        clean = _row_at_fraction(backdoor_trajectory, "clean", end_f)
        poison_start = _row_at_fraction(backdoor_trajectory, "poisoned", start_f) if np.isfinite(start_f) else None
        clean_start = _row_at_fraction(backdoor_trajectory, "clean", start_f) if np.isfinite(start_f) else None
        n_poison = pd.to_numeric(pd.Series([rec.get("n_poisoned")]), errors="coerce").iloc[0]
        n_found = pd.to_numeric(pd.Series([rec.get("n_poison_found_at_expected_count")]), errors="coerce").iloc[0]
        if pd.notna(n_poison) and float(n_poison) > 0 and pd.notna(n_found):
            recovery = float(n_found) / float(n_poison)
        else:
            recovery = pd.to_numeric(
                pd.Series([rec.get("recall_at_expected_poison_count")]), errors="coerce"
            ).iloc[0]
            recovery = float(recovery) if pd.notna(recovery) else math.nan

        out = dict(rec)
        out.update(
            {
                "visualization_schema_version": VISUALIZATION_SCHEMA_VERSION,
                "interval_label": (
                    f"{format_fraction_percent(float(start))}–{format_fraction_percent(end_f)}"
                    if pd.notna(start)
                    else f"to {format_fraction_percent(end_f)}"
                ),
                "poison_recovery_in_top_n": recovery,
                "top_n_definition": "N=true poison count in the interval; post-hoc evaluation only",
                "attack_behavior_alignment": "held-out backdoor behavior at interval end",
                "attack_behavior_available": bool(poison is not None),
                "attack_behavior_start_available": bool(poison_start is not None),
                "poisoned_conversion_rate_start": _number(poison_start, "conditional_conversion_rate"),
                "clean_conversion_rate_start": _number(clean_start, "conditional_conversion_rate"),
                "poisoned_control_target_rate": _number(poison, "control_target_rate"),
                "poisoned_trigger_target_rate": _number(poison, "trigger_target_rate"),
                "poisoned_trigger_excess_target_rate": _number(
                    poison, "trigger_excess_target_rate"
                ),
                "poisoned_conditional_conversion_rate": _number(
                    poison, "conditional_conversion_rate"
                ),
                "clean_control_target_rate": _number(clean, "control_target_rate"),
                "clean_trigger_target_rate": _number(clean, "trigger_target_rate"),
                "clean_trigger_excess_target_rate": _number(
                    clean, "trigger_excess_target_rate"
                ),
                "clean_conditional_conversion_rate": _number(
                    clean, "conditional_conversion_rate"
                ),
            }
        )
        pc = out["poisoned_conditional_conversion_rate"]
        cc = out["clean_conditional_conversion_rate"]
        pcs = out["poisoned_conversion_rate_start"]
        ccs = out["clean_conversion_rate_start"]
        out["conversion_gain_poisoned_vs_clean"] = (
            float(pc) - float(cc) if np.isfinite(pc) and np.isfinite(cc) else math.nan
        )
        out["poisoned_conversion_change_over_interval"] = (
            float(pc) - float(pcs) if np.isfinite(pc) and np.isfinite(pcs) else math.nan
        )
        out["clean_conversion_change_over_interval"] = (
            float(cc) - float(ccs) if np.isfinite(cc) and np.isfinite(ccs) else math.nan
        )
        pchg = out["poisoned_conversion_change_over_interval"]
        cchg = out["clean_conversion_change_over_interval"]
        out["conversion_change_gain_poisoned_vs_clean"] = (
            float(pchg) - float(cchg) if np.isfinite(pchg) and np.isfinite(cchg) else math.nan
        )
        rows.append(out)
    return pd.DataFrame(rows)



def _average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(dtype=float)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if np.isclose(np.std(rx), 0.0) or np.isclose(np.std(ry), 0.0):
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def exact_or_monte_carlo_spearman_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    exact_max_n: int = ASSOCIATION_EXACT_MAX_N,
    monte_carlo_draws: int = ASSOCIATION_MONTE_CARLO_DRAWS,
    seed: int = ASSOCIATION_RANDOM_SEED,
) -> dict[str, Any]:
    """Two-sided permutation test for monotonic association across matched intervals."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = int(len(x))
    observed = _spearman_rho(x, y)
    result: dict[str, Any] = {
        "association_test_schema_version": ASSOCIATION_TEST_SCHEMA_VERSION,
        "statistic": "spearman_rho",
        "alternative": "two_sided",
        "n_intervals": n,
        "rho": observed if np.isfinite(observed) else None,
        "p_value": None,
        "permutation_method": None,
        "n_permutations": 0,
    }
    if n < 3:
        result["status"] = "insufficient_intervals"
        return result
    if not np.isfinite(observed):
        result["status"] = "undefined_statistic"
        return result

    threshold = abs(observed) - 1e-12
    if n <= int(exact_max_n):
        extreme = 0
        total = 0
        for perm in itertools.permutations(y.tolist()):
            rho = _spearman_rho(x, np.asarray(perm, dtype=float))
            if np.isfinite(rho):
                total += 1
                if abs(rho) >= threshold:
                    extreme += 1
        result.update({
            "status": "ok",
            "p_value": float(extreme / total) if total else None,
            "permutation_method": "exact",
            "n_permutations": int(total),
        })
        return result

    rng = np.random.default_rng(int(seed))
    total = max(1, int(monte_carlo_draws))
    extreme = 0
    for _ in range(total):
        rho = _spearman_rho(x, rng.permutation(y))
        if np.isfinite(rho) and abs(rho) >= threshold:
            extreme += 1
    result.update({
        "status": "ok",
        "p_value": float((extreme + 1) / (total + 1)),
        "permutation_method": "monte_carlo",
        "n_permutations": int(total),
        "random_seed": int(seed),
    })
    return result


def compute_detection_attack_associations(combined: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("primary_auc_vs_poisoned_conversion_change", True, "roc_auc", "poisoned_conversion_change_over_interval"),
        ("secondary_auc_vs_end_poisoned_conversion", False, "roc_auc", "poisoned_conditional_conversion_rate"),
        ("secondary_auc_vs_clean_adjusted_conversion_change", False, "roc_auc", "conversion_change_gain_poisoned_vs_clean"),
        ("secondary_topn_recovery_vs_poisoned_conversion_change", False, "poison_recovery_in_top_n", "poisoned_conversion_change_over_interval"),
    ]
    rows: list[dict[str, Any]] = []
    for name, primary, x_col, y_col in specs:
        if x_col not in combined.columns or y_col not in combined.columns:
            test = {
                "association_test_schema_version": ASSOCIATION_TEST_SCHEMA_VERSION,
                "statistic": "spearman_rho",
                "alternative": "two_sided",
                "n_intervals": 0,
                "rho": None,
                "p_value": None,
                "permutation_method": None,
                "n_permutations": 0,
                "status": "missing_metric",
            }
        else:
            x = pd.to_numeric(combined[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(combined[y_col], errors="coerce").to_numpy(dtype=float)
            test = exact_or_monte_carlo_spearman_test(x, y)
        test.update({
            "test_name": name,
            "primary": bool(primary),
            "detectability_metric": x_col,
            "attack_metric": y_col,
            "inference_role": "primary" if primary else "secondary",
            "multiplicity_adjustment": "none",
            "test_scope": "within_run_interval_permutation",
            "alignment": (
                "detectability and attack-efficacy change measured over the same interval"
                if "change_over_interval" in y_col or "change_gain" in y_col
                else "detectability within interval; attack efficacy at interval-end checkpoint"
            ),
        })
        rows.append(test)
    return pd.DataFrame(rows)

def _set_fraction_ticks(ax, fractions: pd.Series) -> None:
    vals = sorted(set(pd.to_numeric(fractions, errors="coerce").dropna().astype(float)))
    ax.set_xticks(vals)
    ax.set_xticklabels([format_fraction_percent(v) for v in vals])
    ax.set_xlabel("End of training interval")


def plot_detection_summary(metrics: pd.DataFrame, output_dir: Path) -> Path | None:
    """Three message-first detection panels with explicit null/control baselines."""
    if metrics.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = metrics[pd.to_numeric(metrics.get("end_fraction"), errors="coerce").notna()].copy()
    if valid.empty:
        return None
    valid = valid.sort_values("end_fraction")
    x = pd.to_numeric(valid["end_fraction"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25), squeeze=False)
    axes = axes.ravel()

    # 1) Global rankability.
    auc = pd.to_numeric(valid.get("roc_auc"), errors="coerce")
    good = x.notna() & auc.notna()
    axes[0].plot(x[good], auc[good], marker="o", label="Overtopping-channel score")
    matched_auc = pd.to_numeric(valid.get("matched_random_roc_auc"), errors="coerce")
    mgood = x.notna() & matched_auc.notna()
    if mgood.any():
        axes[0].plot(
            x[mgood], matched_auc[mgood], marker="x", linestyle=":", linewidth=1.2,
            label="Matched random rows",
        )
    auc_baseline = pd.to_numeric(valid.get("roc_auc_random_baseline"), errors="coerce")
    bgood = x.notna() & auc_baseline.notna()
    if bgood.any():
        axes[0].plot(
            x[bgood], auc_baseline[bgood], linestyle="--", linewidth=1.0,
            label="Random ranking baseline",
        )
    else:
        axes[0].axhline(0.5, linestyle="--", linewidth=1.0, label="Random ranking baseline (0.5)")
    axes[0].set_title("Can poisoned rows be ranked?")
    axes[0].set_ylabel("ROC AUC")

    # 2) Practical top-N recovery.  Since N equals the true poison count,
    # precision and recall are numerically identical; 'recovery' is clearer.
    recovery = pd.to_numeric(valid.get("poison_recovery_in_top_n"), errors="coerce")
    if recovery.isna().all():
        found = pd.to_numeric(valid.get("n_poison_found_at_expected_count"), errors="coerce")
        n_poison = pd.to_numeric(valid.get("n_poisoned"), errors="coerce")
        recovery = found / n_poison.replace(0, np.nan)
    good = x.notna() & recovery.notna()
    axes[1].plot(x[good], recovery[good], marker="o", label="Overtopping-channel score")
    matched_recovery = pd.to_numeric(
        valid.get("matched_random_precision_at_expected_poison_count"), errors="coerce"
    )
    mgood = x.notna() & matched_recovery.notna()
    if mgood.any():
        axes[1].plot(
            x[mgood], matched_recovery[mgood], marker="x", linestyle=":", linewidth=1.2,
            label="Matched random rows",
        )
    prevalence = pd.to_numeric(valid.get("poison_prevalence"), errors="coerce")
    bgood = x.notna() & prevalence.notna()
    if bgood.any():
        axes[1].plot(
            x[bgood], prevalence[bgood], linestyle="--", linewidth=1.0,
            label="Random ranking baseline (prevalence)",
        )
    axes[1].set_title("How many poisons reach the top N?")
    axes[1].set_ylabel("Poison recovery in top N")

    # 3) Matched-source ordering.
    pair = pd.to_numeric(valid.get("paired_poison_over_source_rate"), errors="coerce")
    good = x.notna() & pair.notna()
    axes[2].plot(x[good], pair[good], marker="o", label="Overtopping-channel score")
    matched_pair = pd.to_numeric(valid.get("matched_random_paired_poison_over_source_rate"), errors="coerce")
    mgood = x.notna() & matched_pair.notna()
    if mgood.any():
        axes[2].plot(
            x[mgood], matched_pair[mgood], marker="x", linestyle=":", linewidth=1.2,
            label="Matched random rows",
        )
    pair_baseline = pd.to_numeric(valid.get("paired_poison_over_source_random_baseline"), errors="coerce")
    bgood = x.notna() & pair_baseline.notna()
    if bgood.any():
        axes[2].plot(
            x[bgood], pair_baseline[bgood], linestyle="--", linewidth=1.0,
            label="Random ordering baseline",
        )
    else:
        axes[2].axhline(0.5, linestyle="--", linewidth=1.0, label="Random ordering baseline (0.5)")
    axes[2].set_title("Does poison outrank its source?")
    axes[2].set_ylabel("Pairwise win rate")

    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        _set_fraction_ticks(ax, x)
        ax.grid(True, alpha=0.18)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Poison-row rankability across training")
    fig.tight_layout()
    fig.text(0.5, 0.01, "Useful outcome: overtopping-channel scores stay above matched controls and the random baseline.", ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = output_dir / "01_detection_quality_and_baselines.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_detection_vs_attack(
    combined: pd.DataFrame,
    output_dir: Path,
    associations: pd.DataFrame | None = None,
) -> Path | None:
    """Aligned view of row rankability and held-out backdoor efficacy."""
    if combined.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = combined[pd.to_numeric(combined.get("end_fraction"), errors="coerce").notna()].copy()
    if valid.empty:
        return None
    valid = valid.sort_values("end_fraction")
    x = pd.to_numeric(valid["end_fraction"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), squeeze=False)
    left, right = axes.ravel()

    auc = pd.to_numeric(valid.get("roc_auc"), errors="coerce")
    good = x.notna() & auc.notna()
    left.plot(x[good], auc[good], marker="o", label="Overtopping-channel score")
    matched_auc = pd.to_numeric(valid.get("matched_random_roc_auc"), errors="coerce")
    mgood = x.notna() & matched_auc.notna()
    if mgood.any():
        left.plot(
            x[mgood], matched_auc[mgood], marker="x", linestyle=":", linewidth=1.2,
            label="Matched random rows",
        )
    auc_baseline = pd.to_numeric(valid.get("roc_auc_random_baseline"), errors="coerce")
    bgood = x.notna() & auc_baseline.notna()
    if bgood.any():
        left.plot(
            x[bgood], auc_baseline[bgood], linestyle="--", linewidth=1.0,
            label="Random ranking baseline",
        )
    else:
        left.axhline(0.5, linestyle="--", linewidth=1.0, label="Random ranking baseline (0.5)")
    left.set_title("Poison-row rankability")
    left.set_ylabel("ROC AUC")
    left.set_ylim(-0.03, 1.03)
    _set_fraction_ticks(left, x)
    left.grid(True, alpha=0.18)
    left.legend(frameon=False, fontsize=7)

    def _behavior_series(start_col: str, end_col: str) -> pd.DataFrame:
        points: list[dict[str, float]] = []
        for row in valid.to_dict("records"):
            start = pd.to_numeric(pd.Series([row.get("start_fraction")]), errors="coerce").iloc[0]
            end = pd.to_numeric(pd.Series([row.get("end_fraction")]), errors="coerce").iloc[0]
            start_value = pd.to_numeric(pd.Series([row.get(start_col)]), errors="coerce").iloc[0]
            end_value = pd.to_numeric(pd.Series([row.get(end_col)]), errors="coerce").iloc[0]
            if pd.notna(start) and pd.notna(start_value):
                points.append({"fraction": float(start), "value": float(start_value)})
            if pd.notna(end) and pd.notna(end_value):
                points.append({"fraction": float(end), "value": float(end_value)})
        if not points:
            return pd.DataFrame(columns=["fraction", "value"])
        # Adjacent intervals repeat the same checkpoint. Keep one point per
        # fraction, preferring the last materialized value if a stale input ever
        # contains duplicates.
        return (
            pd.DataFrame(points)
            .sort_values("fraction")
            .drop_duplicates(subset=["fraction"], keep="last")
            .reset_index(drop=True)
        )

    poisoned_series = _behavior_series(
        "poisoned_conversion_rate_start",
        "poisoned_conditional_conversion_rate",
    )
    clean_series = _behavior_series(
        "clean_conversion_rate_start",
        "clean_conditional_conversion_rate",
    )
    if not poisoned_series.empty:
        right.plot(
            poisoned_series["fraction"], poisoned_series["value"], marker="o",
            label="Poison-trained model",
        )
    if not clean_series.empty:
        right.plot(
            clean_series["fraction"], clean_series["value"], marker="o", linestyle="--",
            label="Clean-trained baseline",
        )
    right.set_title("Backdoor efficacy at interval end")
    right.set_ylabel("Trigger-induced conversion")
    right.set_ylim(-0.03, 1.03)
    behavior_fractions = pd.concat(
        [poisoned_series.get("fraction", pd.Series(dtype=float)), clean_series.get("fraction", pd.Series(dtype=float))],
        ignore_index=True,
    )
    _set_fraction_ticks(right, behavior_fractions if not behavior_fractions.empty else x)
    right.grid(True, alpha=0.18)
    if not poisoned_series.empty or not clean_series.empty:
        right.legend(frameon=False, fontsize=8)

    fig.suptitle("Poison-row detectability and backdoor efficacy")
    fig.tight_layout()
    fig.text(0.5, 0.01, "Read left as detectability and right as backdoor strength. A useful early-warning signal rises before or while the backdoor strengthens.", ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = output_dir / "03_detection_vs_backdoor_strength.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path



def plot_detectability_attack_association(
    combined: pd.DataFrame,
    associations: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if combined.empty or associations.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = pd.to_numeric(combined.get("roc_auc"), errors="coerce")
    y = pd.to_numeric(combined.get("poisoned_conversion_change_over_interval"), errors="coerce")
    end = pd.to_numeric(combined.get("end_fraction"), errors="coerce")
    good = x.notna() & y.notna() & end.notna()
    if int(good.sum()) < 2:
        return None

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(x[good], y[good])
    for xi, yi, fi in zip(x[good], y[good], end[good]):
        ax.annotate(format_fraction_percent(float(fi)), (float(xi), float(yi)), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(0.5, linestyle="--", linewidth=1.0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Poison-row rankability (ROC AUC)")
    ax.set_ylabel("Change in trigger-induced conversion")
    primary = associations.loc[associations["primary"] == True]
    ax.set_title("Detectability vs backdoor acquisition")
    if not primary.empty:
        row = primary.iloc[0]
        rho = pd.to_numeric(pd.Series([row.get("rho")]), errors="coerce").iloc[0]
        p_value = pd.to_numeric(pd.Series([row.get("p_value")]), errors="coerce").iloc[0]
        n = int(pd.to_numeric(pd.Series([row.get("n_intervals")]), errors="coerce").fillna(0).iloc[0])
        if pd.notna(rho) and pd.notna(p_value):
            ax.text(
                0.02, 0.98, f"Spearman ρ={float(rho):.2f}; p={float(p_value):.3g}; n={n}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
            )
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.text(0.5, 0.01, "Upper-right points mean intervals where poisoning is both easier to detect and the backdoor grows more strongly. Small n should be treated descriptively.", ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = output_dir / "04_detection_vs_backdoor_growth_association.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_interval_score_separation(scores: pd.DataFrame, output_dir: Path) -> Path | None:
    """Replace the pooled histogram with interval-specific median/IQR separation."""
    if scores.empty or "end_fraction" not in scores.columns:
        return None
    score_col = "wanda_interval_percentile" if "wanda_interval_percentile" in scores.columns else None
    if score_col is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = scores.copy()
    df["end_fraction"] = pd.to_numeric(df["end_fraction"], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=["end_fraction", score_col])
    if df.empty:
        return None
    df["poison_label"] = np.where(df["is_poisoned"].map(truthy), "Poisoned", "Non-poisoned")

    records: list[dict[str, float | str]] = []
    for (end, label), group in df.groupby(["end_fraction", "poison_label"], sort=True):
        values = group[score_col].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        records.append(
            {
                "end_fraction": float(end),
                "label": str(label),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
        )
    stat = pd.DataFrame(records)
    if stat.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.3, 3.8))
    for label, marker, linestyle in (("Non-poisoned", "o", "--"), ("Poisoned", "o", "-")):
        g = stat[stat["label"] == label].sort_values("end_fraction")
        if g.empty:
            continue
        x = g["end_fraction"].to_numpy(dtype=float)
        y = g["median"].to_numpy(dtype=float)
        lo = y - g["q25"].to_numpy(dtype=float)
        hi = g["q75"].to_numpy(dtype=float) - y
        ax.errorbar(x, y, yerr=np.vstack([lo, hi]), marker=marker, linestyle=linestyle, capsize=3, label=label)
    ax.set_ylim(-0.03, 1.03)
    _set_fraction_ticks(ax, stat["end_fraction"])
    ax.set_ylabel("Within-interval anomaly percentile")
    ax.set_title("Poisoned rows move toward the top of the ranking")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.text(0.5, 0.01, "Useful outcome: poisoned rows occupy clearly higher anomaly percentiles than non-poisoned rows.", ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = output_dir / "02_poison_score_separation_by_interval.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_implication_outputs(
    *,
    run_dir: str | Path,
    phase: str,
    eval_intervention: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else detection_dir(run_dir) / phase_dirname(phase)
    )
    metrics_path = output_dir / "detection_metrics_by_interval.csv"
    scores_path = output_dir / "training_example_scores_all_intervals.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing Stage-07 metrics: {metrics_path}")
    metrics = pd.read_csv(metrics_path, low_memory=False)
    scores = pd.read_csv(scores_path, low_memory=False) if scores_path.is_file() else pd.DataFrame()

    behavior, behavior_path = load_backdoor_behavior_trajectory(
        run_dir, phase=phase, eval_intervention=eval_intervention
    )
    combined = build_detection_vs_attack_table(metrics, behavior)
    combined_path = output_dir / "detection_vs_attack_success_by_interval.csv"
    combined.to_csv(combined_path, index=False)

    associations = compute_detection_attack_associations(combined)
    association_csv = output_dir / "detection_vs_attack_association.csv"
    association_json = output_dir / "detection_vs_attack_association.json"
    associations.to_csv(association_csv, index=False)
    association_json.write_text(
        json.dumps(associations.where(pd.notna(associations), None).to_dict("records"), indent=2),
        encoding="utf-8",
    )

    figure_dir = output_dir / "overtopping_interpretation" / "00_poison_detection_overview"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "README.md").write_text(
        "# Poison detection overview\n\n"
        "Start with `01_detection_quality_and_baselines.pdf`.\n\n"
        "- Higher than matched controls/random = useful poisoning signal.\n"
        "- Clear poison/non-poison score separation = useful example-level ranking.\n"
        "- Detection rising with backdoor growth = possible monitoring signal, not by itself a causal claim.\n",
        encoding="utf-8",
    )

    paths: list[Path] = [combined_path, association_csv, association_json]
    for path in (
        plot_detection_summary(combined, figure_dir),
        plot_interval_score_separation(scores, figure_dir),
        plot_detection_vs_attack(combined, figure_dir, associations) if not behavior.empty else None,
        plot_detectability_attack_association(combined, associations, figure_dir) if not behavior.empty else None,
    ):
        if path is not None:
            paths.append(path)

    primary_rows = associations.loc[associations["primary"] == True]
    if not primary_rows.empty:
        primary_test = {}
        for key, value in primary_rows.iloc[0].to_dict().items():
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                primary_test[key] = None
            elif isinstance(value, np.generic):
                item = value.item()
                primary_test[key] = None if isinstance(item, float) and not np.isfinite(item) else item
            else:
                primary_test[key] = value
    else:
        primary_test = None
    return {
        "visualization_schema_version": VISUALIZATION_SCHEMA_VERSION,
        "association_test_schema_version": ASSOCIATION_TEST_SCHEMA_VERSION,
        "backdoor_behavior_trajectory": str(behavior_path) if behavior_path is not None else None,
        "backdoor_behavior_available": bool(not behavior.empty),
        "primary_detection_attack_association": primary_test,
        "association_table": str(association_csv),
        "outputs": [str(p) for p in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--phase", default="input_output", choices=("input_output", "output_only"))
    parser.add_argument("--eval_intervention", default="mean-donor")
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    result = generate_implication_outputs(
        run_dir=args.run_dir,
        phase=args.phase,
        eval_intervention=args.eval_intervention,
        output_dir=args.output_dir,
    )
    print(f"[detection-visualization] attack_behavior_available={result['backdoor_behavior_available']}")
    if result["backdoor_behavior_trajectory"]:
        print(f"[detection-visualization] behavior_source={result['backdoor_behavior_trajectory']}")
    for path in result["outputs"]:
        print(f"[detection-visualization] wrote {path}")


if __name__ == "__main__":
    main()
