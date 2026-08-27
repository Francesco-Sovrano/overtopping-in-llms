#!/usr/bin/env python3
"""Compare independently discovered poisoning circuits across checkpoints.

The developmental poisoning experiment discovers a fresh candidate set at each
post-training checkpoint.  This module summarizes candidate-set size and
identity overlap without using those overlaps to select candidates.
"""

from __future__ import annotations

from studies.poisoning.lib.units import unit_key as _unit_key
import argparse
import json
from pathlib import Path

from core.project_paths import PROJECT_ROOT

from studies.poisoning.lib.run_paths import circuits_dir, metadata_path, phase_dirname, trajectories_dir

import numpy as np
import pandas as pd

from studies.poisoning.lib.virgin_agonists import read_agonist_coordinates, resolve_virgin_agonists_path
from studies.poisoning.tasks.registry import available_tasks, get_task_definition


def _load_set(stats_dir: Path) -> tuple[set[str], pd.DataFrame]:
    ranking = stats_dir / "frozen_candidate_ranking.csv"
    if not ranking.is_file():
        return set(), pd.DataFrame()
    df = pd.read_csv(ranking)
    if "layer_label" not in df.columns and "layer_key" in df.columns:
        df["layer_label"] = df["layer_key"].astype(str)
    if "neuron_id" not in df.columns and "neuron" in df.columns:
        parts = df["neuron"].astype(str).str.rsplit(":", n=1, expand=True)
        if parts.shape[1] == 2:
            df["neuron_id"] = pd.to_numeric(parts[1], errors="coerce")
    if "layer_label" not in df.columns or "neuron_id" not in df.columns:
        return set(), df
    df = df.dropna(subset=["layer_label", "neuron_id"]).copy()
    df["neuron_id"] = pd.to_numeric(df["neuron_id"], errors="coerce").astype(int)
    keys = {_unit_key(a, b) for a, b in zip(df["layer_label"], df["neuron_id"])}
    return keys, df


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else np.nan


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _overlap_record(a_name: str, a: set[str], b_name: str, b: set[str]) -> dict:
    inter = a & b
    union = a | b
    return {
        "set_a": a_name,
        "set_b": b_name,
        "n_a": len(a),
        "n_b": len(b),
        "n_intersection": len(inter),
        "n_union": len(union),
        "jaccard": _jaccard(a, b),
        "fraction_a_retained": float(len(inter) / len(a)) if a else np.nan,
        "fraction_b_explained": float(len(inter) / len(b)) if b else np.nan,
    }


def _parse_virgin(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {_unit_key(layer, neuron_id) for layer, neuron_id in read_agonist_coordinates(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Immutable poisoning data/training run directory.")
    ap.add_argument("--phase", choices=["input_output", "output_only"], required=True)
    ap.add_argument("--task", choices=available_tasks(), default=None)
    ap.add_argument(
        "--virgin_agonists",
        default=None,
        help="Optional ordinary-model positive_baseline directory or neuron_buckets.json for overlap reporting only.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_stage_root = run_dir
    trajectory_dir = trajectories_dir(run_stage_root) / phase_dirname(args.phase)
    summary_dir = circuits_dir(run_stage_root) / phase_dirname(args.phase)
    trajectory = trajectory_dir / "backdoor_lift_overtopping_trajectory.csv"
    if not trajectory.is_file():
        raise FileNotFoundError(f"Missing trajectory summary: {trajectory}")
    df = pd.read_csv(trajectory)

    entries: list[dict] = []
    sets: dict[tuple[str, float], set[str]] = {}
    ordinary_sets: dict[tuple[str, float], set[str]] = {}
    for row in df.to_dict("records"):
        condition = str(row.get("condition"))
        fraction = float(row.get("fraction"))
        status = str(row.get("lift_overtopping_status", "missing"))
        stats_raw = row.get("lift_overtopping_stats_dir")
        keys: set[str] = set()
        ranking = pd.DataFrame()
        circuit_defined = status == "ok" and isinstance(stats_raw, str) and bool(stats_raw)
        if circuit_defined:
            keys, ranking = _load_set(Path(stats_raw))
            sets[(condition, fraction)] = keys
        ordinary_stats_raw = row.get("ordinary_correctness_overtopping_stats_dir")
        ordinary_circuit_declared = _truthy(
            row.get("ordinary_correctness_circuit_defined", False)
        )
        ordinary_keys: set[str] = set()
        ordinary_ranking = pd.DataFrame()
        ordinary_circuit_defined = (
            ordinary_circuit_declared
            and isinstance(ordinary_stats_raw, str)
            and bool(ordinary_stats_raw)
        )
        if ordinary_circuit_defined:
            ordinary_keys, ordinary_ranking = _load_set(Path(ordinary_stats_raw))
            ordinary_sets[(condition, fraction)] = ordinary_keys
        scientifically_undefined = status in {
            "no_tl_trigger_lift", "no_discoverable_trigger_lift", "not_run_fraction_zero",
            "no_qualifying_neurons",
        }
        n_discovered = len(keys) if circuit_defined else (0 if scientifically_undefined else np.nan)
        entries.append({
            "condition": condition,
            "fraction": fraction,
            "global_step": row.get("global_step"),
            "status": status,
            "circuit_defined": bool(circuit_defined),
            "n_trigger_lift_total": row.get("n_trigger_lift_total", row.get("n_trigger_lift_success")),
            "n_trigger_lift_discovery": row.get("n_trigger_lift_discovery"),
            "n_trigger_lift_test": row.get("n_trigger_lift_test"),
            "n_discovered_channels": n_discovered,
            "n_ranked_rows": int(len(ranking)) if circuit_defined else (0 if scientifically_undefined else np.nan),
            "U(J)": row.get("lift_U(J)"),
            "Top": row.get("lift_Top"),
            "N.05": row.get("lift_N.05"),
            "N.10": row.get("lift_N.10"),
            "N.20": row.get("lift_N.20"),
            "N.30": row.get("lift_N.30"),
            "Neff": row.get("lift_Neff"),
            "top1_mass": row.get("lift_top1_mass"),
            "all_points_status": row.get("all_points_status"),
            "all_points_U(J)": row.get("all_points_U(J)"),
            "all_points_Top": row.get("all_points_Top"),
            "all_points_N.05": row.get("all_points_N.05"),
            "all_points_N.10": row.get("all_points_N.10"),
            "all_points_N.20": row.get("all_points_N.20"),
            "all_points_N.30": row.get("all_points_N.30"),
            "all_points_Neff": row.get("all_points_Neff"),
            "all_points_top1_mass": row.get("all_points_top1_mass"),
            "stats_dir": stats_raw if isinstance(stats_raw, str) else None,
            "all_points_stats_dir": row.get("all_points_overtopping_stats_dir"),
            "normal_task_correctness_status": row.get("normal_task_correctness_status", row.get("ordinary_correctness_status")),
            "normal_task_correctness_circuit_defined": bool(ordinary_circuit_defined),
            "ordinary_correctness_status": row.get("ordinary_correctness_status"),  # compatibility
            "ordinary_correctness_circuit_defined": bool(ordinary_circuit_defined),
            "n_ordinary_correctness_channels": (
                len(ordinary_keys) if ordinary_circuit_defined else np.nan
            ),
            "n_ordinary_correctness_ranked_rows": (
                int(len(ordinary_ranking)) if ordinary_circuit_defined else np.nan
            ),
            "ordinary_correctness_stats_dir": (
                ordinary_stats_raw if isinstance(ordinary_stats_raw, str) else None
            ),
        })

    summary_dir.mkdir(parents=True, exist_ok=True)
    entries_df = pd.DataFrame(entries)
    if not entries_df.empty:
        entries_df = entries_df.sort_values(["condition", "fraction"])
    entries_df.to_csv(summary_dir / "checkpoint_circuit_sets.csv", index=False)

    pairwise: list[dict] = []
    for condition in sorted({c for c, _ in sets}):
        checkpoints = sorted((f, s) for (c, f), s in sets.items() if c == condition)
        for i, (fa, a) in enumerate(checkpoints):
            for fb, b in checkpoints[i + 1:]:
                rec = _overlap_record(f"{condition}:{fa:g}", a, f"{condition}:{fb:g}", b)
                rec.update({"comparison": "within_condition", "condition": condition, "fraction_a": fa, "fraction_b": fb})
                pairwise.append(rec)
    pd.DataFrame(pairwise).to_csv(summary_dir / "checkpoint_circuit_overlap_pairwise.csv", index=False)

    matched: list[dict] = []
    fractions = sorted({f for c, f in sets if c == "clean"} & {f for c, f in sets if c == "poisoned"})
    for frac in fractions:
        rec = _overlap_record(f"clean:{frac:g}", sets[("clean", frac)], f"poisoned:{frac:g}", sets[("poisoned", frac)])
        rec.update({"comparison": "matched_clean_vs_poisoned", "fraction": frac})
        matched.append(rec)
    pd.DataFrame(matched).to_csv(summary_dir / "matched_clean_poisoned_circuit_overlap.csv", index=False)

    trigger_vs_ordinary: list[dict] = []
    checkpoint_keys = sorted(set(sets) | set(ordinary_sets))
    for condition, frac in checkpoint_keys:
        trigger = sets.get((condition, frac))
        ordinary = ordinary_sets.get((condition, frac))
        if trigger is not None and ordinary is not None:
            rec = _overlap_record(
                f"backdoor_trigger_test:{condition}:{frac:g}", trigger,
                f"normal_task_correctness:{condition}:{frac:g}", ordinary,
            )
        else:
            rec = {
                "set_a": f"backdoor_trigger_test:{condition}:{frac:g}",
                "set_b": f"normal_task_correctness:{condition}:{frac:g}",
                "n_a": len(trigger) if trigger is not None else np.nan,
                "n_b": len(ordinary) if ordinary is not None else np.nan,
                "n_intersection": np.nan,
                "n_union": np.nan,
                "jaccard": np.nan,
                "fraction_a_retained": np.nan,
                "fraction_b_explained": np.nan,
            }
        rec.update({
            "comparison": "backdoor_trigger_test_vs_normal_task_correctness",
            "condition": condition,
            "fraction": frac,
            "trigger_circuit_defined": trigger is not None,
            "ordinary_correctness_circuit_defined": ordinary is not None,
        })
        trigger_vs_ordinary.append(rec)
    pd.DataFrame(trigger_vs_ordinary).to_csv(
        summary_dir / "backdoor_trigger_vs_normal_task_circuit_overlap.csv",
        index=False,
    )

    virgin_path = Path(args.virgin_agonists).expanduser().resolve() if args.virgin_agonists else None
    if virgin_path is None and args.task:
        try:
            cfg = json.loads(metadata_path(run_dir, "run_config.json").read_text(encoding="utf-8"))
            task_definition = get_task_definition(args.task)
            virgin_path = resolve_virgin_agonists_path(
                PROJECT_ROOT, task=args.task,
                task_data_dir=task_definition.ordinary_data_dir,
                model_name=str(cfg.get("model_name", task_definition.default_model)),
                phase=args.phase, intervention="mean-donor", require_phase_match=True,
            )
        except Exception as exc:
            print(f"[virgin-overlap] optional source unavailable: {exc}")
            virgin_path = None
    if virgin_path is not None:
        virgin = _parse_virgin(virgin_path)
        rows = []
        for (condition, frac), current in sorted(sets.items()):
            rec = _overlap_record("virgin_task_agonists", virgin, f"{condition}:{frac:g}", current)
            rec.update({"condition": condition, "fraction": frac, "virgin_agonist_source": str(virgin_path)})
            rows.append(rec)
        pd.DataFrame(rows).to_csv(summary_dir / "virgin_agonist_overlap.csv", index=False)

    print(f"Wrote checkpoint circuit comparison tables under {summary_dir}")


if __name__ == "__main__":
    main()
