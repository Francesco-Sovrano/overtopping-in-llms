#!/usr/bin/env python3
"""Render poisoning-study figures from cross-seed trajectory and detection tables.

Cross-seed aggregation is performed with training seed as the replicate unit.
The default display uses the across-seed median with Q1-Q3 uncertainty.  Two
alternative display styles are available: mean with a Student-t confidence
interval, and raw per-seed traces overlaid on the median/Q1-Q3 summary.

If more than one scientific configuration is present for the same task/model,
configurations are plotted as separate rows instead of being silently pooled or
causing the reporting build to fail.  A manifest describing those plot families
is written beside the figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from studies.poisoning.stage08_aggregate_cross_seed import (
    DEFENSE_IDENTITY_COLUMNS,
    DETECTOR_IDENTITY_COLUMNS,
    EXPERIMENT_ID_COLUMNS,
)

TASK_LABELS = {"arithmetic": "Arithmetic", "grammar": "Grammar"}
SUMMARY_STYLES = ("median_iqr", "mean_t_ci", "seed_traces")

PRIMARY_METRICS = (
    ("attack_cohort_control_correctness_accuracy", "Attack-cohort control\naccuracy"),
    ("trigger_lift_success_rate", "Backdoor acquisition\ntrigger-lift success"),
    ("lift_U(J)", "Backdoor causal reach\ntrigger-conditioned U(J)"),
    ("paired_control_U(J)", "Matched control causal reach\nconditional C→I U(J)"),
)
SPECIFICITY_METRICS = (
    ("control_target_positive_rate", "Control behavior\ntarget rate"),
    ("trigger_excess_target_rate", "Trigger-specific shift\ntrigger excess"),
    ("conditional_conversion_rate", "Conditional attack success\nconversion rate"),
    ("primary_minus_sham_conditional_conversion_rate", "Marker specificity\nprimary - sham"),
)


def _model_short(name: str) -> str:
    return str(name or "unknown").rsplit("/", 1)[-1]


def _rate_axis(ax, *, signed: bool = False) -> None:
    if signed:
        ax.axhline(0.0, linewidth=0.8, alpha=0.5)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    else:
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(True, alpha=0.22)


def _fraction_ticks(ax, values: Iterable[float]) -> None:
    vals = sorted({float(x) for x in values if np.isfinite(float(x))})
    if vals:
        ax.set_xticks(vals)
        ax.set_xticklabels([f"{100*x:g}%" for x in vals])
    ax.set_xlabel("Fine-tuning progress")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    """Read an optional aggregate CSV, treating empty files as no data."""
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalized_identity_value(value) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)):
        numeric = float(value)
        if numeric == 0.0:
            numeric = 0.0
        # Stable across CSV dtype inference: 6 and 6.0 must identify the same
        # scientific configuration when matching aggregate and raw tables.
        return format(numeric, ".15g")
    return str(value)


def _identity_columns(df: pd.DataFrame, *, extra: Sequence[str] = ()) -> list[str]:
    columns: list[str] = []
    for column in [*EXPERIMENT_ID_COLUMNS, *extra]:
        if column in df.columns and column not in columns:
            columns.append(column)
    return columns


def _identity_tuple(row: pd.Series, columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalized_identity_value(row.get(column)) for column in columns)


def _family_id(identity: tuple[str, ...]) -> str:
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _annotate_plot_families(
    df: pd.DataFrame,
    *,
    branch: str,
    extra_identity_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach stable family ids and return a one-row-per-family manifest.

    Training seeds are pooled only when the full scientific identity matches.
    Distinct configurations for the same task/model remain separate plot rows.
    """
    if df.empty:
        return df.copy(), pd.DataFrame()
    if "task" not in df.columns or "model_name" not in df.columns:
        return df.copy(), pd.DataFrame()

    out = df.copy()
    id_cols = _identity_columns(out, extra=extra_identity_columns)
    if not id_cols:
        id_cols = [column for column in ("task", "model_name") if column in out.columns]

    identities = [_identity_tuple(row, id_cols) for _, row in out.iterrows()]
    out["_plot_family_id"] = [_family_id(identity) for identity in identities]

    order = {"arithmetic": 0, "grammar": 1}
    family_rows: list[dict] = []
    grouped = out.groupby(["task", "model_name"], dropna=False, sort=False)
    for (task, model), task_model_group in grouped:
        unique = []
        seen = set()
        for family_id in task_model_group["_plot_family_id"].tolist():
            if family_id not in seen:
                unique.append(family_id)
                seen.add(family_id)
        # Stable ordering by the identity tuple, not filesystem discovery order.
        unique.sort(key=lambda fid: tuple(
            _normalized_identity_value(v)
            for v in task_model_group.loc[
                task_model_group["_plot_family_id"].eq(fid), id_cols
            ].iloc[0].tolist()
        ))
        count = len(unique)
        for index, family_id in enumerate(unique, start=1):
            mask = task_model_group["_plot_family_id"].eq(family_id)
            family = task_model_group.loc[mask]
            first = family.iloc[0]
            row = {
                "plot_branch": branch,
                "family_id": family_id,
                "task": task,
                "model_name": model,
                "family_index": index,
                "family_count_for_task_model": count,
            }
            if "training_seeds" in family.columns:
                seeds: set[int] = set()
                for value in family["training_seeds"].dropna().astype(str):
                    for item in value.split(","):
                        item = item.strip()
                        if item:
                            try:
                                seeds.add(int(item))
                            except ValueError:
                                pass
                row["training_seeds"] = ",".join(str(seed) for seed in sorted(seeds))
                row["n_training_seeds"] = len(seeds)
            elif "training_seed" in family.columns:
                seeds = sorted(set(pd.to_numeric(family["training_seed"], errors="coerce").dropna().astype(int)))
                row["training_seeds"] = ",".join(str(seed) for seed in seeds)
                row["n_training_seeds"] = len(seeds)
            else:
                row["training_seeds"] = ""
                row["n_training_seeds"] = np.nan
            for column in id_cols:
                row[column] = first.get(column)
            family_rows.append(row)

    manifest = pd.DataFrame(family_rows)
    if manifest.empty:
        return out, manifest

    sort_cols = ["task", "model_name", "family_index"]
    manifest = manifest.sort_values(
        sort_cols,
        key=lambda col: col.map(order).fillna(99) if col.name == "task" else col,
    ).reset_index(drop=True)

    index_map = manifest.set_index("family_id")["family_index"].to_dict()
    count_map = manifest.set_index("family_id")["family_count_for_task_model"].to_dict()
    out["_plot_family_index"] = out["_plot_family_id"].map(index_map).astype(int)
    out["_plot_family_count"] = out["_plot_family_id"].map(count_map).astype(int)
    return out, manifest


def _differing_identity_fields(manifest: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    if manifest.empty:
        return {}
    metadata_cols = {
        "plot_branch", "family_id", "family_index", "family_count_for_task_model",
        "training_seeds", "n_training_seeds", "task", "model_name",
    }
    candidate_cols = [c for c in manifest.columns if c not in metadata_cols]
    differences: dict[tuple[str, str], list[str]] = {}
    for (task, model), group in manifest.groupby(["task", "model_name"], dropna=False, sort=False):
        if len(group) <= 1:
            continue
        fields = []
        for column in candidate_cols:
            values = {_normalized_identity_value(value) for value in group[column].tolist()}
            if len(values) > 1:
                fields.append(column)
        differences[(str(task), str(model))] = fields
    return differences


def _enforce_or_report_family_policy(
    manifest: pd.DataFrame,
    *,
    policy: str,
    branch: str,
) -> None:
    differences = _differing_identity_fields(manifest)
    if not differences:
        return
    messages = []
    for (task, model), fields in differences.items():
        count = int(
            manifest[
                manifest["task"].astype(str).eq(task)
                & manifest["model_name"].astype(str).eq(model)
            ]["family_count_for_task_model"].max()
        )
        field_text = ", ".join(fields) if fields else "one or more identity fields"
        messages.append(
            f"{task}/{_model_short(model)} has {count} {branch} configurations; differing fields: {field_text}"
        )
    if policy == "error":
        raise ValueError(
            "Ambiguous poisoning plot families. " + " | ".join(messages)
            + ". Use --family_policy split to plot them separately."
        )
    for message in messages:
        print(f"[stage08_plot] {message}; plotting separately rather than pooling distinct configurations.")


def _write_family_manifest(output_dir: Path, manifests: Sequence[pd.DataFrame]) -> None:
    frames = [frame for frame in manifests if not frame.empty]
    if not frames:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(output_dir / "scientific_family_manifest.csv", index=False)


def _available_specs(df: pd.DataFrame, specs, summary_style: str):
    out = []
    preferred = "mean" if summary_style == "mean_t_ci" else "median"
    fallback = "median" if preferred == "mean" else "mean"
    for metric, title in specs:
        candidates = [f"{metric}__{preferred}", f"{metric}__{fallback}"]
        column = next((c for c in candidates if c in df.columns), None)
        if column is None:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if np.isfinite(values.to_numpy(float)).any():
            out.append((metric, title))
    return tuple(out)


def _summary_columns(df: pd.DataFrame, metric: str, summary_style: str) -> tuple[str | None, str | None, str | None, str]:
    if summary_style == "mean_t_ci":
        preferred = (f"{metric}__mean", f"{metric}__ci_low", f"{metric}__ci_high", "mean + t CI")
        fallback = (f"{metric}__median", f"{metric}__q25", f"{metric}__q75", "median + IQR")
    else:
        preferred = (f"{metric}__median", f"{metric}__q25", f"{metric}__q75", "median + IQR")
        fallback = (f"{metric}__mean", f"{metric}__ci_low", f"{metric}__ci_high", "mean + t CI")
    center, low, high, label = preferred
    if center in df.columns:
        return center, low if low in df.columns else None, high if high in df.columns else None, label
    center, low, high, label = fallback
    if center in df.columns:
        return center, low if low in df.columns else None, high if high in df.columns else None, label
    return None, None, None, label


def _family_row_label(family: pd.DataFrame) -> str:
    task = str(family["task"].iloc[0])
    model = str(family["model_name"].iloc[0])
    count = int(family["_plot_family_count"].iloc[0]) if "_plot_family_count" in family.columns else 1
    index = int(family["_plot_family_index"].iloc[0]) if "_plot_family_index" in family.columns else 1
    label = f"{TASK_LABELS.get(task, task.title())}\n{_model_short(model)}"
    if count > 1:
        label += f"\nconfig {index}/{count}"
    if "n_training_seeds" in family.columns:
        n = pd.to_numeric(family["n_training_seeds"], errors="coerce").dropna()
        if len(n):
            label += f" · n={int(n.max())} seeds"
    return label


def _matching_raw_family(raw: pd.DataFrame, family_id: str) -> pd.DataFrame:
    if raw.empty or "_plot_family_id" not in raw.columns:
        return pd.DataFrame()
    return raw[raw["_plot_family_id"].astype(str).eq(str(family_id))].copy()


def _plot_seed_traces(
    ax,
    raw_family: pd.DataFrame,
    *,
    metric: str,
    x_col: str,
    condition: str | None,
    color,
) -> None:
    if raw_family.empty or metric not in raw_family.columns or "training_seed" not in raw_family.columns:
        return
    data = raw_family.copy()
    if condition is not None and "condition" in data.columns:
        data = data[data["condition"].astype(str).eq(str(condition))]
    data[x_col] = pd.to_numeric(data.get(x_col), errors="coerce")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[x_col, metric])
    for _, seed_group in data.groupby("training_seed", sort=True):
        seed_group = seed_group.sort_values(x_col)
        ax.plot(
            seed_group[x_col].to_numpy(float),
            seed_group[metric].to_numpy(float),
            linewidth=0.9,
            alpha=0.22,
            color=color,
            zorder=1,
        )


def _plot_metric(
    ax,
    family: pd.DataFrame,
    metric: str,
    *,
    summary_style: str,
    raw_family: pd.DataFrame | None = None,
) -> bool:
    center_col, low_col, high_col, _ = _summary_columns(family, metric, summary_style)
    if center_col is None:
        return False
    plotted = False
    labels = {"clean": "Clean fine-tune", "poisoned": "Poisoned fine-tune"}
    for condition, group in family.groupby("condition", sort=False):
        group = group.copy()
        group["fraction"] = pd.to_numeric(group["fraction"], errors="coerce")
        group[center_col] = pd.to_numeric(group[center_col], errors="coerce")
        group = group.dropna(subset=["fraction", center_col]).sort_values("fraction")
        if group.empty:
            continue
        x = group["fraction"].to_numpy(float)
        y = group[center_col].to_numpy(float)
        line, = ax.plot(
            x, y, marker="o", linewidth=1.8,
            label=labels.get(str(condition), str(condition)), zorder=3,
        )
        if summary_style == "seed_traces" and raw_family is not None:
            _plot_seed_traces(
                ax, raw_family, metric=metric, x_col="fraction",
                condition=str(condition), color=line.get_color(),
            )
        if low_col is not None and high_col is not None:
            low = pd.to_numeric(group[low_col], errors="coerce").to_numpy(float)
            high = pd.to_numeric(group[high_col], errors="coerce").to_numpy(float)
            good = np.isfinite(low) & np.isfinite(high) & (low <= y) & (y <= high)
            if good.any():
                lower = y[good] - low[good]
                upper = high[good] - y[good]
                ax.errorbar(
                    x[good], y[good], yerr=np.vstack([lower, upper]),
                    fmt="none", ecolor=line.get_color(), elinewidth=1.2,
                    capsize=3, capthick=1.0, alpha=0.9, zorder=2,
                )
        plotted = True
    return plotted


def _plot_grid(
    df: pd.DataFrame,
    specs,
    out_path: Path,
    title: str,
    *,
    summary_style: str,
    raw_df: pd.DataFrame | None = None,
    note: str | None = None,
) -> None:
    specs = _available_specs(df, specs, summary_style)
    if df.empty or not specs or "_plot_family_id" not in df.columns:
        out_path.unlink(missing_ok=True)
        return
    family_ids = df[["_plot_family_id", "task", "model_name", "_plot_family_index"]].drop_duplicates()
    order = {"arithmetic": 0, "grammar": 1}
    family_ids = family_ids.assign(_task_order=family_ids["task"].astype(str).map(order).fillna(99))
    family_ids = family_ids.sort_values(["_task_order", "task", "model_name", "_plot_family_index"])
    families = family_ids["_plot_family_id"].astype(str).tolist()
    if not families:
        out_path.unlink(missing_ok=True)
        return

    fig, axes = plt.subplots(
        len(families), len(specs),
        figsize=(4.1 * len(specs), 3.25 * len(families) + 0.8), squeeze=False,
    )
    all_fractions = pd.to_numeric(df.get("fraction"), errors="coerce").dropna().tolist()
    legend_handles = legend_labels = None
    for r, family_id in enumerate(families):
        family = df[df["_plot_family_id"].astype(str).eq(family_id)]
        raw_family = _matching_raw_family(raw_df, family_id) if raw_df is not None else pd.DataFrame()
        row_label = _family_row_label(family)
        for c, (metric, metric_title) in enumerate(specs):
            ax = axes[r, c]
            plotted = _plot_metric(
                ax, family, metric, summary_style=summary_style,
                raw_family=raw_family,
            )
            _rate_axis(ax, signed=(metric == "primary_minus_sham_conditional_conversion_rate"))
            _fraction_ticks(ax, all_fractions)
            if r == 0:
                ax.set_title(metric_title, fontsize=11)
            ax.set_ylabel(row_label if c == 0 else "", fontsize=10)
            if not plotted:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
            if legend_handles is None and ax.lines:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle(title, fontsize=14, y=0.995)
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels, loc="upper center",
            ncol=max(1, len(legend_labels)), frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
    if note:
        fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.035 if note else 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _metric_summary_arrays(
    frame: pd.DataFrame,
    metric: str,
    summary_style: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    center_col, low_col, high_col, _ = _summary_columns(frame, metric, summary_style)
    if center_col is None:
        return np.asarray([], dtype=float), None, None
    center = pd.to_numeric(frame[center_col], errors="coerce").to_numpy(float)
    low = (
        pd.to_numeric(frame[low_col], errors="coerce").to_numpy(float)
        if low_col is not None else None
    )
    high = (
        pd.to_numeric(frame[high_col], errors="coerce").to_numpy(float)
        if high_col is not None else None
    )
    return center, low, high


def _draw_summary_errorbars(ax, x, y, low, high, *, color, axis: str = "y") -> None:
    if low is None or high is None or len(y) == 0:
        return
    good = np.isfinite(y) & np.isfinite(low) & np.isfinite(high) & (low <= y) & (y <= high)
    if not good.any():
        return
    lower = y[good] - low[good]
    upper = high[good] - y[good]
    kwargs = dict(fmt="none", ecolor=color, elinewidth=1.15, capsize=2.8, capthick=0.9, alpha=0.9, zorder=2)
    if axis == "x":
        ax.errorbar(x[good], y[good], xerr=np.vstack([lower, upper]), **kwargs)
    else:
        ax.errorbar(x[good], y[good], yerr=np.vstack([lower, upper]), **kwargs)


def _plot_clean_reference_defense(
    checkpoint: pd.DataFrame,
    curve: pd.DataFrame,
    output_dir: Path,
    *,
    summary_style: str,
    checkpoint_raw: pd.DataFrame | None = None,
) -> None:
    """Render the clean-reference defense using seed-level replication.

    Panel A aggregates each seed's checkpoint-level selected-channel mean, then
    summarizes those seed-level estimates. Panel B shows the seed-level
    suppression-vs-damage operating points directly, with the cross-seed
    center and IQR/CI overlaid. Selected channels are never pooled as if they
    were independent seed replicates.
    """
    out_path = output_dir / "poisoning_clean_reference_defense.pdf"
    if checkpoint.empty or curve.empty or "_plot_family_id" not in checkpoint.columns:
        out_path.unlink(missing_ok=True)
        return

    family_ids = checkpoint[["_plot_family_id", "task", "model_name", "_plot_family_index"]].drop_duplicates()
    family_ids = family_ids.sort_values(["task", "model_name", "_plot_family_index"])
    families = family_ids["_plot_family_id"].astype(str).tolist()
    if not families:
        out_path.unlink(missing_ok=True)
        return

    markers = ["o", "s", "^", "D", "P", "X"]
    fig, axes = plt.subplots(
        len(families), 2,
        figsize=(12.26, 3.68 * len(families)),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.18 / max(1, len(families)),
        top=0.92, wspace=0.28, hspace=0.42,
    )
    if len(families) == 1:
        fig.text(0.012, 0.965, "(a)", fontsize=9.0, fontweight="bold", ha="left", va="top")
        fig.text(0.515, 0.965, "(b)", fontsize=9.0, fontweight="bold", ha="left", va="top")

    for row, family_id in enumerate(families):
        fam = checkpoint[checkpoint["_plot_family_id"].astype(str).eq(family_id)].copy()
        curve_fam = curve[curve["_plot_family_id"].astype(str).eq(family_id)].copy()
        raw_fam = (
            _matching_raw_family(checkpoint_raw, family_id)
            if checkpoint_raw is not None else pd.DataFrame()
        )
        ax_line, ax_trade = axes[row, 0], axes[row, 1]
        fam["target_fraction"] = pd.to_numeric(fam["target_fraction"], errors="coerce")
        fam = fam.dropna(subset=["target_fraction"]).sort_values("target_fraction")
        x = 100.0 * fam["target_fraction"].to_numpy(float)
        if len(x) == 0:
            continue

        attack, attack_low, attack_high = _metric_summary_arrays(
            fam, "mean_attack_suppression", summary_style
        )
        benign, benign_low, benign_high = _metric_summary_arrays(
            fam, "mean_poisoned_disruption", summary_style
        )
        leverage, _, _ = _metric_summary_arrays(
            fam, "mean_defense_leverage", summary_style
        )
        if len(attack) != len(x) or len(benign) != len(x):
            continue

        attack_line, = ax_line.plot(
            x, 100.0 * attack, marker="D", linewidth=2.0,
            label="Attack suppression", zorder=3,
        )
        benign_line, = ax_line.plot(
            x, 100.0 * benign, marker="o", linewidth=2.0,
            label="Benign damage", zorder=3,
        )
        _draw_summary_errorbars(
            ax_line, x, 100.0 * attack,
            None if attack_low is None else 100.0 * attack_low,
            None if attack_high is None else 100.0 * attack_high,
            color=attack_line.get_color(),
        )
        _draw_summary_errorbars(
            ax_line, x, 100.0 * benign,
            None if benign_low is None else 100.0 * benign_low,
            None if benign_high is None else 100.0 * benign_high,
            color=benign_line.get_color(),
        )
        if summary_style == "seed_traces" and not raw_fam.empty:
            for metric, line in (("mean_attack_suppression", attack_line), ("mean_poisoned_disruption", benign_line)):
                raw = raw_fam.copy()
                raw["target_fraction"] = pd.to_numeric(raw["target_fraction"], errors="coerce")
                raw[metric] = pd.to_numeric(raw.get(metric), errors="coerce")
                raw = raw.dropna(subset=["target_fraction", metric])
                for _, sg in raw.groupby("training_seed", sort=True):
                    sg = sg.sort_values("target_fraction")
                    ax_line.plot(
                        100.0 * sg["target_fraction"].to_numpy(float),
                        100.0 * sg[metric].to_numpy(float),
                        linewidth=0.8, alpha=0.18, color=line.get_color(), zorder=1,
                    )

        attack_pct = 100.0 * attack
        benign_pct = 100.0 * benign
        ax_line.fill_between(
            x, benign_pct, attack_pct, where=(attack_pct >= benign_pct),
            interpolate=True, color="#dcefdc", alpha=0.75, linewidth=0,
            label=r"$\Delta_{\rm def}>0$", zorder=0,
        )
        ax_line.fill_between(
            x, benign_pct, attack_pct, where=(attack_pct < benign_pct),
            interpolate=True, color="#f4dede", alpha=0.75, linewidth=0, zorder=0,
        )
        if len(leverage) == len(x):
            for xv, av, bv, lev in zip(x, attack_pct, benign_pct, 100.0 * leverage):
                if np.isfinite(lev):
                    ax_line.annotate(
                        f"{lev:+.0f} pp", (xv, max(av, bv)),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.2,
                    )
        ax_line.set_xticks(x)
        ax_line.set_xlim(max(0.0, float(np.nanmin(x)) - 5.0), min(100.0, float(np.nanmax(x)) + 5.0))
        ax_line.set_ylim(bottom=0.0)
        ax_line.set_xlabel("Checkpoint (%)")
        ax_line.set_ylabel("Intervention rate (%)")
        ax_line.grid(True, alpha=0.18)
        ax_line.legend(frameon=False, fontsize=7.4, loc="upper left")

        operating_budget_values = pd.to_numeric(
            fam.get("operating_benign_damage_budget"), errors="coerce"
        ).dropna().unique()
        operating_budget = float(operating_budget_values[0]) if len(operating_budget_values) else 0.30
        n_values = pd.to_numeric(fam.get("n_training_seeds"), errors="coerce").dropna()
        n_seeds = int(n_values.max()) if len(n_values) else 0
        ax_line.set_title(
            f"Checkpoint tradeoff at τ={100.0*operating_budget:.0f}%  ·  n={n_seeds} seeds",
            fontsize=10.0, fontweight="bold",
        )

        # Budget-sweep inset: median/mean line per checkpoint, with IQR/CI band.
        inset = ax_line.inset_axes([0.54, 0.08, 0.43, 0.37])
        curve_fam["target_fraction"] = pd.to_numeric(curve_fam["target_fraction"], errors="coerce")
        curve_fam["benign_damage_budget"] = pd.to_numeric(curve_fam["benign_damage_budget"], errors="coerce")
        checkpoints = sorted(curve_fam["target_fraction"].dropna().unique())
        for idx, frac in enumerate(checkpoints):
            cp = curve_fam[np.isclose(
                curve_fam["target_fraction"].to_numpy(float), float(frac), equal_nan=False
            )].sort_values("benign_damage_budget")
            center, low, high = _metric_summary_arrays(cp, "mean_defense_leverage", summary_style)
            bx = 100.0 * cp["benign_damage_budget"].to_numpy(float)
            by = 100.0 * center
            line, = inset.plot(
                bx, by, marker=markers[idx % len(markers)],
                linewidth=1.0, markersize=2.2,
            )
            if low is not None and high is not None:
                good = np.isfinite(by) & np.isfinite(low) & np.isfinite(high)
                if good.any():
                    inset.fill_between(
                        bx[good], 100.0 * low[good], 100.0 * high[good],
                        color=line.get_color(), alpha=0.10, linewidth=0,
                    )
        inset.axhline(0.0, color="0.45", linestyle="--", linewidth=0.7)
        inset.axvline(100.0 * operating_budget, color="0.50", linestyle=":", linewidth=0.8)
        inset.set_xlabel(r"$\tau$ (%)", fontsize=6.5, labelpad=1)
        inset.set_ylabel(r"$\Delta_{\rm def}$ (pp)", fontsize=6.5, labelpad=1)
        inset.tick_params(axis="both", labelsize=6.0, pad=1)
        inset.grid(True, alpha=0.12)

        # Seed-level operating points. These are the actual replicate units; the
        # larger markers and whiskers show the cross-seed center and IQR/CI.
        if not raw_fam.empty:
            raw_fam["target_fraction"] = pd.to_numeric(raw_fam["target_fraction"], errors="coerce")
            raw_fam["mean_poisoned_disruption"] = pd.to_numeric(raw_fam["mean_poisoned_disruption"], errors="coerce")
            raw_fam["mean_attack_suppression"] = pd.to_numeric(raw_fam["mean_attack_suppression"], errors="coerce")
        all_trade_values = []
        for idx, frac in enumerate(sorted(fam["target_fraction"].dropna().unique())):
            cp = fam[np.isclose(fam["target_fraction"].to_numpy(float), float(frac), equal_nan=False)]
            x_center, x_low, x_high = _metric_summary_arrays(cp, "mean_poisoned_disruption", summary_style)
            y_center, y_low, y_high = _metric_summary_arrays(cp, "mean_attack_suppression", summary_style)
            if len(x_center) != 1 or len(y_center) != 1:
                continue
            cx, cy = 100.0 * x_center[0], 100.0 * y_center[0]
            color = plt.get_cmap("tab10")(idx % 10)
            raw_cp = pd.DataFrame()
            if not raw_fam.empty:
                raw_cp = raw_fam[np.isclose(
                    raw_fam["target_fraction"].to_numpy(float), float(frac), equal_nan=False
                )].dropna(subset=["mean_poisoned_disruption", "mean_attack_suppression"])
                if not raw_cp.empty:
                    rx = 100.0 * raw_cp["mean_poisoned_disruption"].to_numpy(float)
                    ry = 100.0 * raw_cp["mean_attack_suppression"].to_numpy(float)
                    ax_trade.scatter(
                        rx, ry, s=25, marker=markers[idx % len(markers)],
                        color=color, alpha=0.35, linewidths=0, zorder=2,
                    )
                    all_trade_values.extend(rx.tolist()); all_trade_values.extend(ry.tolist())
            ax_trade.scatter(
                [cx], [cy], s=70, marker=markers[idx % len(markers)],
                color=color, edgecolors="black", linewidths=0.55, zorder=4,
                label=f"{int(round(100*float(frac)))}% checkpoint",
            )
            if x_low is not None and x_high is not None:
                xl, xh = 100.0 * x_low[0], 100.0 * x_high[0]
                if np.isfinite(xl) and np.isfinite(xh) and xl <= cx <= xh:
                    ax_trade.errorbar(
                        [cx], [cy], xerr=np.array([[cx-xl], [xh-cx]]),
                        fmt="none", ecolor=color, elinewidth=1.15, capsize=2.8, zorder=3,
                    )
            if y_low is not None and y_high is not None:
                yl, yh = 100.0 * y_low[0], 100.0 * y_high[0]
                if np.isfinite(yl) and np.isfinite(yh) and yl <= cy <= yh:
                    ax_trade.errorbar(
                        [cx], [cy], yerr=np.array([[cy-yl], [yh-cy]]),
                        fmt="none", ecolor=color, elinewidth=1.15, capsize=2.8, zorder=3,
                    )
            all_trade_values.extend([cx, cy])

        lim = max(70.0, max(all_trade_values, default=60.0) * 1.10)
        lim = min(100.0, lim)
        trade_x = np.linspace(0.0, lim, 400)
        ax_trade.fill_between(trade_x, trade_x, lim, color="#dcefdc", alpha=0.75, linewidth=0, zorder=0)
        ax_trade.fill_between(trade_x, 0.0, trade_x, color="#f4dede", alpha=0.75, linewidth=0, zorder=0)
        ax_trade.plot([0, lim], [0, lim], linestyle="--", color="0.42", linewidth=1.0, zorder=1)
        ax_trade.axvline(100.0 * operating_budget, color="0.50", linestyle=":", linewidth=1.1)
        ax_trade.text(0.025 * lim, 0.93 * lim, "attack-selective", fontsize=8.0, fontweight="bold", color="0.18")
        ax_trade.text(0.72 * lim, 0.08 * lim, "benign-costly", fontsize=8.0, fontweight="bold", color="0.18")
        ax_trade.set_xlim(0.0, lim); ax_trade.set_ylim(0.0, lim)
        ax_trade.set_xlabel("Seed-level benign damage (%)")
        ax_trade.set_ylabel("Seed-level attack suppression (%)")
        ax_trade.set_title("Seed-level suppression versus benign damage", fontsize=10.0, fontweight="bold")
        ax_trade.grid(True, alpha=0.18)
        ax_trade.legend(frameon=False, fontsize=6.8, loc="upper right")

        if len(families) > 1:
            label = _family_row_label(fam)
            ax_line.text(-0.14, 0.5, label, transform=ax_line.transAxes, rotation=90,
                         ha="center", va="center", fontsize=8.5)

    fig.text(
        0.5, 0.01,
        _summary_note(summary_style) + " Seed-level checkpoint means are the replicate observations.",
        ha="center", va="bottom", fontsize=8.2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_poison_detection(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    summary_style: str,
    raw_df: pd.DataFrame | None = None,
) -> None:
    if df.empty or "_plot_family_id" not in df.columns:
        return
    specs = (
        ("roc_auc", "ROC AUC"),
        ("average_precision", "Average precision"),
        ("precision_at_expected_poison_count", "Precision @ true poison count"),
        ("paired_poison_over_source_rate", "Poison > matched source"),
    )
    family_ids = df[["_plot_family_id", "task", "model_name", "_plot_family_index"]].drop_duplicates()
    order = {"arithmetic": 0, "grammar": 1}
    family_ids = family_ids.assign(_task_order=family_ids["task"].astype(str).map(order).fillna(99))
    family_ids = family_ids.sort_values(["_task_order", "task", "model_name", "_plot_family_index"])
    families = family_ids["_plot_family_id"].astype(str).tolist()
    if not families:
        return

    fig, axes = plt.subplots(
        len(families), len(specs),
        figsize=(4.0 * len(specs), 3.2 * len(families) + 0.8), squeeze=False,
    )
    all_end = pd.to_numeric(df.get("end_fraction"), errors="coerce").dropna().tolist()
    for r, family_id in enumerate(families):
        fam = df[df["_plot_family_id"].astype(str).eq(family_id)]
        raw_family = _matching_raw_family(raw_df, family_id) if raw_df is not None else pd.DataFrame()
        row_label = _family_row_label(fam)
        for c, (metric, title) in enumerate(specs):
            ax = axes[r, c]
            center_col, low_col, high_col, _ = _summary_columns(fam, metric, summary_style)
            plotted = False
            x = np.asarray([], dtype=float)
            g = fam.copy()
            if center_col is not None:
                g["end_fraction"] = pd.to_numeric(g["end_fraction"], errors="coerce")
                g[center_col] = pd.to_numeric(g[center_col], errors="coerce")
                g = g.dropna(subset=["end_fraction", center_col]).sort_values("end_fraction")
                if not g.empty:
                    x = g["end_fraction"].to_numpy(float)
                    y = g[center_col].to_numpy(float)
                    line, = ax.plot(
                        x, y, marker="o", linewidth=1.8,
                        label="Disrupted control-correctness channels", zorder=3,
                    )
                    if summary_style == "seed_traces":
                        _plot_seed_traces(
                            ax, raw_family, metric=metric, x_col="end_fraction",
                            condition=None, color=line.get_color(),
                        )
                    random_metric = f"matched_random_{metric}"
                    random_center_col, _, _, _ = _summary_columns(g, random_metric, summary_style)
                    if random_center_col is not None:
                        random_y = pd.to_numeric(g[random_center_col], errors="coerce").to_numpy(float)
                        random_good = np.isfinite(random_y)
                        if random_good.any():
                            random_line, = ax.plot(
                                x[random_good], random_y[random_good], marker="x",
                                linestyle=":", linewidth=1.3,
                                label="Matched random rows", zorder=3,
                            )
                            if summary_style == "seed_traces":
                                _plot_seed_traces(
                                    ax, raw_family, metric=random_metric,
                                    x_col="end_fraction", condition=None,
                                    color=random_line.get_color(),
                                )
                    if low_col is not None and high_col is not None:
                        low = pd.to_numeric(g[low_col], errors="coerce").to_numpy(float)
                        high = pd.to_numeric(g[high_col], errors="coerce").to_numpy(float)
                        good = np.isfinite(low) & np.isfinite(high) & (low <= y) & (y <= high)
                        if good.any():
                            ax.fill_between(
                                x[good], low[good], high[good],
                                alpha=0.15, color=line.get_color(), zorder=2,
                            )
                    plotted = True
            if metric in {"roc_auc", "paired_poison_over_source_rate"}:
                baseline = np.full_like(x, 0.5, dtype=float) if plotted else np.asarray([], dtype=float)
            else:
                prev_center, _, _, _ = _summary_columns(g, "poison_prevalence", summary_style)
                if plotted and prev_center is not None:
                    baseline = pd.to_numeric(g[prev_center], errors="coerce").to_numpy(float)
                else:
                    baseline = np.asarray([], dtype=float)
            if plotted and len(baseline) == len(x):
                good_base = np.isfinite(baseline)
                if good_base.any():
                    ax.plot(
                        x[good_base], baseline[good_base], linestyle="--",
                        linewidth=1.0, alpha=0.7, label="Random baseline",
                    )
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            _fraction_ticks(ax, all_end)
            if r == 0:
                ax.set_title(title, fontsize=11)
            ax.set_ylabel(row_label if c == 0 else "", fontsize=10)
            ax.grid(True, alpha=0.22)
            if r == 0 and c == 0 and plotted:
                ax.legend(frameon=False, fontsize=8)
            if not plotted:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(
        "Poisoned-example detection from clean-normalized overtopping-channel disruption",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "poisoning_example_detection.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _summary_note(summary_style: str) -> str:
    if summary_style == "mean_t_ci":
        return "Points/lines show the across-seed mean; uncertainty spans the two-sided Student-t confidence interval."
    if summary_style == "seed_traces":
        return "Thin lines show individual training seeds; thick points/lines show the median and error bars span Q1-Q3."
    return "Points/lines show the across-seed median; asymmetric error bars span Q1-Q3."


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, help="Directory containing checkpoint_metrics_by_model_across_seeds.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--summary_style", choices=SUMMARY_STYLES, default="median_iqr",
        help="Across-seed display: robust median/IQR (default), mean/t-CI, or raw seed traces plus median/IQR.",
    )
    p.add_argument(
        "--family_policy", choices=("split", "error"), default="split",
        help="How to handle multiple scientific configurations for one task/model. 'split' plots separate rows; 'error' restores strict failure.",
    )
    args = p.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    behavior = _read_optional_csv(input_dir / "checkpoint_metrics_by_model_across_seeds.csv")
    behavior_raw = _read_optional_csv(input_dir / "checkpoint_trajectories_all_seeds.csv")
    behavior_manifest = pd.DataFrame()
    if not behavior.empty:
        behavior, behavior_manifest = _annotate_plot_families(behavior, branch="behavior")
        _enforce_or_report_family_policy(behavior_manifest, policy=args.family_policy, branch="behavior")
        if not behavior_raw.empty:
            behavior_raw, _ = _annotate_plot_families(behavior_raw, branch="behavior_raw")
        summary_note = _summary_note(args.summary_style)
        _plot_grid(
            behavior, PRIMARY_METRICS, output_dir / "poisoning_primary_trajectories.pdf",
            "Poisoning trajectories: behavior and causal organization",
            summary_style=args.summary_style,
            raw_df=behavior_raw,
            note=(
                "Ordinary correctness and trigger lift are distinct endpoints. Gaps mean CHA was undefined/skipped. "
                + summary_note
            ),
        )
        _plot_grid(
            behavior, SPECIFICITY_METRICS, output_dir / "poisoning_specificity_checks.pdf",
            "Poisoning specificity and control diagnostics",
            summary_style=args.summary_style,
            raw_df=behavior_raw,
            note=(
                "Control target rate diagnoses clean-path drift; trigger excess isolates marker-specific shift; "
                "conditional conversion conditions on control non-target rows; primary-sham contrast checks marker specificity. "
                + summary_note
            ),
        )

    detection = _read_optional_csv(input_dir / "poison_detection_metrics_across_seeds.csv")
    detection_raw = _read_optional_csv(input_dir / "poison_detection_metrics_all_seeds.csv")
    detection_manifest = pd.DataFrame()
    if not detection.empty:
        detection, detection_manifest = _annotate_plot_families(
            detection, branch="detection", extra_identity_columns=DETECTOR_IDENTITY_COLUMNS,
        )
        _enforce_or_report_family_policy(detection_manifest, policy=args.family_policy, branch="detection")
        if not detection_raw.empty:
            detection_raw, _ = _annotate_plot_families(
                detection_raw, branch="detection_raw", extra_identity_columns=DETECTOR_IDENTITY_COLUMNS,
            )
        _plot_poison_detection(
            detection, output_dir, summary_style=args.summary_style,
            raw_df=detection_raw,
        )

    defense = _read_optional_csv(input_dir / "clean_reference_defense_checkpoint_across_seeds.csv")
    defense_raw = _read_optional_csv(input_dir / "clean_reference_defense_checkpoint_all_seeds.csv")
    defense_curve = _read_optional_csv(input_dir / "clean_reference_defense_budget_curve_across_seeds.csv")
    defense_manifest = pd.DataFrame()
    if not defense.empty and not defense_curve.empty:
        defense, defense_manifest = _annotate_plot_families(
            defense, branch="clean_reference_defense",
            extra_identity_columns=DEFENSE_IDENTITY_COLUMNS,
        )
        _enforce_or_report_family_policy(
            defense_manifest, policy=args.family_policy, branch="clean-reference defense"
        )
        defense_curve, _ = _annotate_plot_families(
            defense_curve, branch="clean_reference_defense_curve",
            extra_identity_columns=DEFENSE_IDENTITY_COLUMNS,
        )
        if not defense_raw.empty:
            defense_raw, _ = _annotate_plot_families(
                defense_raw, branch="clean_reference_defense_raw",
                extra_identity_columns=DEFENSE_IDENTITY_COLUMNS,
            )
        _plot_clean_reference_defense(
            defense, defense_curve, output_dir,
            summary_style=args.summary_style, checkpoint_raw=defense_raw,
        )

    _write_family_manifest(input_dir, [behavior_manifest, detection_manifest, defense_manifest])
    print(
        f"Wrote poisoning figures under {output_dir} "
        f"(summary_style={args.summary_style}, family_policy={args.family_policy})"
    )


if __name__ == "__main__":
    main()
