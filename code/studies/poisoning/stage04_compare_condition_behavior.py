"""Early clean-vs-poisoned behavior summaries and plots.

This module is intentionally model-free and cache-agnostic.  It reads the
``scores.csv`` files already exported by the causal task and can therefore run
immediately after behavior generation, before CHA/overtopping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from studies.poisoning.lib.run_paths import causal_dir, checkpoint_progress_label, checkpoint_tag, comparisons_dir, metadata_path, phase_dirname
from typing import Any, Iterable

import pandas as pd


def _sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return re.sub(r"_+", "_", value).strip("_") or "run"


def _bool_series(series: pd.Series) -> pd.Series:
    if str(series.dtype) in {"bool", "boolean"}:
        return series.astype("boolean")
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        {
            "1": True,
            "true": True,
            "t": True,
            "yes": True,
            "y": True,
            "0": False,
            "false": False,
            "f": False,
            "no": False,
            "n": False,
        }
    )
    return mapped.astype("boolean")


def _rate(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _pp(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{100.0 * float(value):+.2f}pp"


def summarize_trigger_scores(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, low_memory=False)
    required = {"is_attack_example", "control_target_positive", "trigger_target_positive"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required corrected attack-cohort columns: {missing}")

    attack = _bool_series(df["is_attack_example"]).fillna(False).astype(bool)
    control = _bool_series(df["control_target_positive"]).fillna(False).astype(bool)
    trigger = _bool_series(df["trigger_target_positive"]).fillna(False).astype(bool)

    control = control[attack]
    trigger = trigger[attack]
    attack_n = int(attack.sum())
    control_target_n = int(control.sum())
    trigger_target_n = int(trigger.sum())
    lift = int((~control & trigger).sum())
    suppression = int((control & ~trigger).sum())
    change = int((control != trigger).sum())
    convertible_n = int((~control).sum())

    control_rate = _rate(control_target_n, attack_n)
    trigger_rate = _rate(trigger_target_n, attack_n)
    excess = None if control_rate is None or trigger_rate is None else trigger_rate - control_rate
    return {
        "kind": "trigger",
        "scanned_n": int(len(df)),
        "attack_n": attack_n,
        "control_target_count": control_target_n,
        "trigger_target_count": trigger_target_n,
        "trigger_lift_success": lift,
        "trigger_suppression_success": suppression,
        "trigger_change_success": change,
        "conditional_conversion_n": convertible_n,
        "control_target_rate": control_rate,
        "trigger_target_rate": trigger_rate,
        "trigger_excess_target_rate": excess,
        "trigger_lift_rate": _rate(lift, attack_n),
        "conditional_conversion_rate": _rate(lift, convertible_n),
        "convertible_fraction": _rate(convertible_n, attack_n),
        "trigger_suppression_rate": _rate(suppression, attack_n),
        "trigger_change_rate": _rate(change, attack_n),
    }


def summarize_ordinary_scores(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, low_memory=False)
    if "is_correct_control" not in df.columns:
        raise ValueError(f"{path} is missing is_correct_control")
    correct = _bool_series(df["is_correct_control"])
    labeled = correct.notna()
    correct_bool = correct.fillna(False).astype(bool)
    n = int(labeled.sum())
    n_correct = int((correct_bool & labeled).sum())
    out: dict[str, Any] = {
        "kind": "ordinary",
        "scanned_n": int(len(df)),
        "ordinary_n": n,
        "ordinary_correct": n_correct,
        "ordinary_accuracy": _rate(n_correct, n),
    }
    if "is_attack_example" in df.columns:
        attack = _bool_series(df["is_attack_example"]).fillna(False).astype(bool)
        for label, mask in (("attack", attack), ("target", ~attack)):
            cohort_labeled = labeled & mask
            cohort_n = int(cohort_labeled.sum())
            cohort_correct = int((correct_bool & cohort_labeled).sum())
            out[f"ordinary_{label}_n"] = cohort_n
            out[f"ordinary_{label}_accuracy"] = _rate(cohort_correct, cohort_n)
    return out


def summarize_scores(path: Path, kind: str) -> dict[str, Any]:
    return summarize_trigger_scores(path) if kind == "trigger" else summarize_ordinary_scores(path)


def _load_manifest(run_dir: Path) -> list[dict[str, str]]:
    path = metadata_path(run_dir, "checkpoint_manifest_all.csv")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _score_path(
    run_stage_root: Path,
    row: dict[str, str],
    *,
    phase: str,
    intervention: str,
    kind: str,
) -> Path:
    stage_label = checkpoint_progress_label(row)
    phase_root = causal_dir(run_stage_root) / row["condition"] / stage_label / phase_dirname(phase)
    safe = _sanitize(intervention)
    endpoint = "trigger_lift" if kind == "trigger" else "ordinary_correctness"
    return phase_root / endpoint / f"eval_{safe}" / "feature_report" / "scores.csv"


def _matching_rows(rows: Iterable[dict[str, str]], tag: str) -> list[dict[str, str]]:
    return [row for row in rows if checkpoint_tag(row) == tag]


def _metric_specs(kind: str) -> list[tuple[str, str]]:
    if kind == "ordinary":
        return [
            ("ordinary_accuracy", "Ordinary accuracy"),
            ("ordinary_attack_accuracy", "Attack-cohort accuracy"),
            ("ordinary_target_accuracy", "Target-cohort accuracy"),
        ]
    return [
        ("control_target_rate", "Control target rate"),
        ("trigger_target_rate", "Triggered target rate"),
        ("trigger_excess_target_rate", "Trigger excess"),
        ("conditional_conversion_rate", "Conditional conversion"),
        ("convertible_fraction", "Convertible fraction"),
        ("trigger_lift_rate", "Trigger-lift rate"),
    ]


def _print_current(condition: str, tag: str, stats: dict[str, Any]) -> None:
    if stats["kind"] == "ordinary":
        print(
            "[behavior-stats] "
            f"condition={condition} checkpoint={tag} ordinary_accuracy={_pct(stats.get('ordinary_accuracy'))} "
            f"attack_accuracy={_pct(stats.get('ordinary_attack_accuracy'))} "
            f"target_accuracy={_pct(stats.get('ordinary_target_accuracy'))} n={stats.get('ordinary_n', 0)}",
            flush=True,
        )
        return
    print(
        "[behavior-stats] "
        f"condition={condition} checkpoint={tag} attack_n={stats['attack_n']} "
        f"control_target={_pct(stats['control_target_rate'])} "
        f"trigger_target={_pct(stats['trigger_target_rate'])} "
        f"trigger_excess={_pp(stats['trigger_excess_target_rate'])} "
        f"conditional_conversion={_pct(stats['conditional_conversion_rate'])} "
        f"convertible={_pct(stats['convertible_fraction'])} "
        f"lift={stats['trigger_lift_success']}/{stats['attack_n']}",
        flush=True,
    )




def _trigger_gate_failures(
    current: dict[str, Any],
    clean: dict[str, Any] | None,
    *,
    min_trigger_excess: float | None,
    min_conditional_conversion: float | None,
    max_abs_control_delta: float | None,
) -> list[str]:
    """Return human-readable smoke-test failures for trigger behavior."""
    failures: list[str] = []
    excess = current.get("trigger_excess_target_rate")
    conditional = current.get("conditional_conversion_rate")
    if min_trigger_excess is not None and (excess is None or float(excess) < min_trigger_excess):
        failures.append(
            f"trigger excess {_pct(excess)} is below required {_pct(min_trigger_excess)}"
        )
    if min_conditional_conversion is not None and (
        conditional is None or float(conditional) < min_conditional_conversion
    ):
        failures.append(
            f"conditional conversion {_pct(conditional)} is below required {_pct(min_conditional_conversion)}"
        )
    if max_abs_control_delta is not None:
        if clean is None:
            failures.append("matched clean behavior is unavailable for control-drift gate")
        else:
            current_control = current.get("control_target_rate")
            clean_control = clean.get("control_target_rate")
            if current_control is None or clean_control is None:
                failures.append("control target rate is unavailable for control-drift gate")
            else:
                delta = float(current_control) - float(clean_control)
                if abs(delta) > max_abs_control_delta:
                    failures.append(
                        f"control target drift {_pp(delta)} exceeds allowed ±{100.0 * max_abs_control_delta:.2f}pp"
                    )
    return failures

def _comparison_rows(stats_by_condition: dict[str, dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    clean = stats_by_condition.get("clean")
    out: list[dict[str, Any]] = []
    for condition, stats in stats_by_condition.items():
        row: dict[str, Any] = {"condition": condition, **stats}
        if clean is not None and condition != "clean":
            for key, _ in _metric_specs(kind):
                a = stats.get(key)
                b = clean.get(key)
                row[f"delta_vs_clean_{key}"] = (
                    float(a) - float(b) if a is not None and b is not None else None
                )
        out.append(row)
    return out


def _print_comparisons(stats_by_condition: dict[str, dict[str, Any]], kind: str) -> None:
    clean = stats_by_condition.get("clean")
    if clean is None:
        print("[behavior-compare] matched clean behavior not available yet; current stats retained.", flush=True)
        return
    for condition, stats in stats_by_condition.items():
        if condition == "clean":
            continue
        print(f"[behavior-compare] clean -> {condition}", flush=True)
        for key, label in _metric_specs(kind):
            c = clean.get(key)
            p = stats.get(key)
            if c is None or p is None:
                continue
            print(
                f"  {label:<24} clean={_pct(c):>8}  {condition}={_pct(p):>8}  delta={_pp(float(p)-float(c)):>9}",
                flush=True,
            )


def _write_tables(out_dir: Path, kind: str, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"{kind}_behavior_comparison.csv", index=False)
    (out_dir / f"{kind}_behavior_comparison.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )


def _plot_point(out_dir: Path, kind: str, stats_by_condition: dict[str, dict[str, Any]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    specs = [(k, label) for k, label in _metric_specs(kind) if any(v.get(k) is not None for v in stats_by_condition.values())]
    if not specs:
        return []
    conditions = list(stats_by_condition)
    x = np.arange(len(specs), dtype=float)
    width = 0.8 / max(1, len(conditions))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(specs)), 5.2))
    for i, condition in enumerate(conditions):
        vals = [100.0 * float(stats_by_condition[condition].get(k) or 0.0) for k, _ in specs]
        ax.bar(x + (i - (len(conditions) - 1) / 2) * width, vals, width, label=condition)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in specs], rotation=20, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title("Ordinary correctness: clean vs poisoned" if kind == "ordinary" else "Backdoor behavior: clean vs poisoned")
    ax.axhline(0.0, linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    rates_path = out_dir / f"{kind}_behavior_rates.pdf"
    fig.savefig(rates_path, bbox_inches="tight")
    plt.close(fig)

    clean = stats_by_condition.get("clean")
    delta_conditions = [c for c in conditions if c != "clean"]
    paths = [rates_path]
    if clean is not None and delta_conditions:
        labels: list[str] = []
        deltas: list[float] = []
        for condition in delta_conditions:
            for key, label in specs:
                a, b = stats_by_condition[condition].get(key), clean.get(key)
                if a is None or b is None:
                    continue
                labels.append(f"{condition}: {label}")
                deltas.append(100.0 * (float(a) - float(b)))
        if labels:
            fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.42 * len(labels))))
            y = np.arange(len(labels))
            ax.barh(y, deltas)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.axvline(0.0, linewidth=0.8)
            ax.set_xlabel("Difference vs clean (percentage points)")
            ax.set_title("Poisoning effect relative to clean")
            fig.tight_layout()
            delta_path = out_dir / f"{kind}_behavior_deltas.pdf"
            fig.savefig(delta_path, bbox_inches="tight")
            plt.close(fig)
            paths.append(delta_path)
    return paths


def _trajectory_rows(
    run_stage_root: Path,
    manifest: list[dict[str, str]],
    *,
    phase: str,
    intervention: str,
    kind: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in manifest:
        tag = checkpoint_tag(row)
        key = (row.get("condition", ""), tag)
        if not tag or key in seen:
            continue
        seen.add(key)
        path = _score_path(run_stage_root, row, phase=phase, intervention=intervention, kind=kind)
        if not path.exists():
            continue
        stats = summarize_scores(path, kind)
        out.append(
            {
                "condition": row.get("condition", "unknown"),
                "fraction": float(row.get("fraction", 0.0)),
                "global_step": int(float(row.get("global_step", 0))),
                "checkpoint_tag": tag,
                "checkpoint_label": checkpoint_progress_label(row),
                **stats,
            }
        )
    return out


def _write_trajectory(out_root: Path, kind: str, rows: list[dict[str, Any]], plots: bool) -> list[Path]:
    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values(["condition", "global_step", "fraction"])
    csv_path = out_root / f"{kind}_behavior_trajectory.csv"
    df.to_csv(csv_path, index=False)
    paths = [csv_path]
    if not plots:
        return paths

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if kind == "trigger":
        metrics = [
            ("trigger_excess_target_rate", "Trigger excess target rate", "trigger_excess_trajectory.pdf"),
            ("control_target_rate", "Control target rate", "control_target_trajectory.pdf"),
            ("conditional_conversion_rate", "Conditional conversion rate", "conditional_conversion_trajectory.pdf"),
        ]
    else:
        metrics = [("ordinary_accuracy", "Ordinary accuracy", "ordinary_accuracy_trajectory.pdf")]

    for metric, title, filename in metrics:
        if metric not in df.columns or df[metric].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for condition, group in df.groupby("condition", sort=False):
            group = group.sort_values("global_step")
            ax.plot(group["global_step"], 100.0 * group[metric].astype(float), marker="o", label=condition)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Percent")
        ax.set_title(title)
        ax.axhline(0.0, linewidth=0.8)
        ax.legend()
        fig.tight_layout()
        path = out_root / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True, help="Immutable poisoning data/training run directory.")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint_tag", required=True, help="Canonical checkpoint tag (progress_*pct__step_*).")
    parser.add_argument("--checkpoint_label", default=None, help="Readable result-directory label; defaults to checkpoint_tag.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--eval_intervention", default="mean-donor")
    parser.add_argument("--kind", choices=("trigger", "ordinary"), default="trigger")
    parser.add_argument("--scores_csv", type=Path, required=True)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--min_trigger_excess", type=float, default=None)
    parser.add_argument("--min_conditional_conversion", type=float, default=None)
    parser.add_argument("--max_abs_control_delta", type=float, default=None)
    parser.add_argument("--fail_on_gate", action="store_true")
    args = parser.parse_args()

    current = summarize_scores(args.scores_csv, args.kind)
    _print_current(args.condition, args.checkpoint_tag, current)

    manifest = _load_manifest(args.run_dir)
    matched = _matching_rows(manifest, args.checkpoint_tag)
    stats_by_condition: dict[str, dict[str, Any]] = {}
    # In paired execution clean runs first. Do not compare a newly generated
    # clean checkpoint against an older poisoned scores.csv that may still be on
    # disk from a previous trajectory. The matched comparison is emitted when
    # the current non-clean checkpoint finishes.
    if args.condition == "clean":
        stats_by_condition["clean"] = current
    else:
        # Compare the current non-clean result only against its matched clean
        # control. Do not pull other non-clean conditions from disk: those may
        # be stale outputs from an earlier rerun.
        for row in matched:
            if row.get("condition") != "clean":
                continue
            path = _score_path(
                args.run_dir,
                row,
                phase=args.phase,
                intervention=args.eval_intervention,
                kind=args.kind,
            )
            if path.exists():
                stats_by_condition["clean"] = summarize_scores(path, args.kind)
            break
        stats_by_condition[args.condition] = current

    safe = _sanitize(args.eval_intervention)
    root = comparisons_dir(args.run_dir) / phase_dirname(args.phase) / f"eval_{safe}"
    point_dir = root / (args.checkpoint_label or args.checkpoint_tag)
    comparison_rows = _comparison_rows(stats_by_condition, args.kind)
    _write_tables(point_dir, args.kind, comparison_rows)
    _print_comparisons(stats_by_condition, args.kind)

    gate_failures: list[str] = []
    if args.kind == "trigger" and args.condition != "clean":
        gate_failures = _trigger_gate_failures(
            current,
            stats_by_condition.get("clean"),
            min_trigger_excess=args.min_trigger_excess,
            min_conditional_conversion=args.min_conditional_conversion,
            max_abs_control_delta=args.max_abs_control_delta,
        )
        if gate_failures:
            print("[behavior-gate] failed:", flush=True)
            for failure in gate_failures:
                print(f"  - {failure}", flush=True)
        elif any(
            value is not None
            for value in (
                args.min_trigger_excess,
                args.min_conditional_conversion,
                args.max_abs_control_delta,
            )
        ):
            print("[behavior-gate] passed", flush=True)

    created: list[Path] = [
        point_dir / f"{args.kind}_behavior_comparison.csv",
        point_dir / f"{args.kind}_behavior_comparison.json",
    ]
    if not args.no_plots:
        created.extend(_plot_point(point_dir, args.kind, stats_by_condition))

    trajectory = _trajectory_rows(
        args.run_dir,
        manifest,
        phase=args.phase,
        intervention=args.eval_intervention,
        kind=args.kind,
    )

    condition_rank = {
        "clean": 0,
        "poisoned": 1,
        "protected_poisoned": 2,
        "random_protected_poisoned": 3,
    }
    current_row = next(
        (row for row in matched if row.get("condition") == args.condition),
        None,
    )
    if current_row is not None:
        current_key = (
            int(float(current_row.get("global_step", 0))),
            float(current_row.get("fraction", 0.0)),
            condition_rank.get(args.condition, 100),
        )
        trajectory = [
            row for row in trajectory
            if (
                int(row.get("global_step", 0)),
                float(row.get("fraction", 0.0)),
                condition_rank.get(str(row.get("condition", "")), 100),
            ) <= current_key
        ]
    created.extend(_write_trajectory(root, args.kind, trajectory, plots=not args.no_plots))
    for path in created:
        print(f"[behavior-output] {path}", flush=True)
    if gate_failures and args.fail_on_gate:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
