#!/usr/bin/env python3
"""Build manifest-driven tables and minimally styled plots for completed runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from studies.overtopping.analysis.lib.files import load_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from studies.overtopping.analysis.lib.task_metrics import raw_task_score, chance_baseline, competence
from studies.overtopping.experiments.execution import RunSpec
from core.project_paths import PROJECT_ROOT

CURRENT_INTERACTION_SCHEMA = "conditional-marginal-validation-v1"


def row_for(spec: RunSpec, data_root: Path) -> dict | None:
    stats_dir = spec.stats_dir(data_root)
    global_path = stats_dir / "flip_stats_global.json"
    by_path = stats_dir / "flip_stats_by_neuron.csv"
    dataset_path = data_root / spec.task / Path(spec.model) / "feature_report" / "dataset_stats.json"
    if not (global_path.exists() and by_path.exists() and dataset_path.exists()):
        return None

    global_payload = load_json(global_path)
    by = pd.read_csv(by_path)
    singleton_path = stats_dir / "singleton_set_metrics.json"
    singleton = load_json(singleton_path) if singleton_path.exists() else {}
    interaction_path = stats_dir / "interaction_validation" / "interaction_validation_summary.json"
    interaction = load_json(interaction_path) if interaction_path.exists() else {}
    summary_path = stats_dir / "interaction_validation" / "interaction_validation_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    schema = interaction.get("definition_version")

    dataset = load_json(dataset_path)
    raw = raw_task_score(spec.task, dataset)
    chance = chance_baseline(spec.task, dataset)
    score = competence(spec.phase, raw, chance)

    u = singleton.get("U_J", global_payload.get("union_flip_any_unique_rate"))
    s1 = singleton.get("s_1")
    toc = singleton.get("TOC_m", {}).get("1", {}) if isinstance(singleton.get("TOC_m"), dict) else {}
    toc1 = toc.get("value", math.nan) if isinstance(toc, dict) else math.nan

    candidate_e = interaction.get("candidate_E_J", {}) if schema == CURRENT_INTERACTION_SCHEMA else {}
    candidate_e = candidate_e if isinstance(candidate_e, dict) else {}
    output = {
        **spec.__dict__, "phase": spec.phase, "stats_dir": str(stats_dir),
        "raw_score": raw, "chance": chance, "score": score,
        "J": singleton.get("J", global_payload.get("n_neurons", len(by))),
        "U_J": u, "s_1": s1, "TOC_1": toc1,
        "TOC_1_status": (
            (toc.get("status", "unknown") if isinstance(toc, dict) and toc else None)
            or "unavailable_requires_discovery_frozen_ranking_and_per_example_flip_events"
        ),
        "R_ov": singleton.get("R_ov"),
        "R_ov_status": singleton.get("R_ov_status", "missing"),
        "N_eff": singleton.get("N_eff"),
        "N_eff_status": singleton.get("N_eff_status", "missing"),
        "OCC_0": singleton.get("OCC_0"), "OCC_1": singleton.get("OCC_1"),
        "OCC_0_status": singleton.get("OCC_0_status", "missing"),
        "OCC_1_status": singleton.get("OCC_1_status", "missing"),
        "E_J": candidate_e.get("effect", global_payload.get("E_J")),
        "E_J_status": candidate_e.get("status", global_payload.get("E_J_status", "missing")),
        "interaction_schema": schema or "missing",
    }
    for threshold, count in (singleton.get("N_t", {}) or {}).items():
        output[f"N_t_{threshold}"] = count
    for m, item in (singleton.get("TOC_m", {}) or {}).items():
        output[f"TOC_{m}"] = item.get("value") if isinstance(item, dict) else item

    if schema == CURRENT_INTERACTION_SCHEMA:
        for item in interaction.get("conditional_marginal", []) or []:
            try:
                multiplier = int(item.get("background_multiplier"))
            except Exception:
                continue
            prefix = f"CMC_{multiplier}x"
            output[prefix] = item.get("candidate")
            output[f"{prefix}_null_median"] = item.get("median_null")
            output[f"{prefix}_Delta"] = item.get("Delta")
            output[f"{prefix}_P"] = item.get("P")
            output[f"{prefix}_p_MC"] = item.get("p_MC")
            output[f"{prefix}_win_rate"] = item.get("paired_win_rate")
            output[f"{prefix}_status"] = item.get("status")

    if not summary.empty:
        for record in summary.to_dict("records"):
            metric = str(record.get("metric"))
            if metric == "E_J":
                prefix = "E_J"
            elif metric == "conditional_marginal":
                try:
                    prefix = f"CMC_{int(float(record.get('background_multiplier')))}x"
                except Exception:
                    continue
            else:
                continue
            for field in (
                "median_null", "Delta", "P", "p_MC", "status",
                "null_draws_requested", "null_draws_finite",
            ):
                output[f"{prefix}_{field}"] = record.get(field)
    return output


def plot_metric(frame: pd.DataFrame, metric: str, out_dir: Path) -> None:
    x = pd.to_numeric(frame.get("score"), errors="coerce")
    y = pd.to_numeric(frame.get(metric), errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.scatter(x[mask], y[mask], s=22)
    ax.set_xlabel("Task score")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ')} across completed experiments")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"score_vs_{metric}.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalogue_json", required=True)
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "results" / "experiment_catalogue"))
    args = p.parse_args()
    specs = [RunSpec(**row) for row in load_json(Path(args.catalogue_json))]
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for spec in specs if (row := row_for(spec, data_root)) is not None]
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "completed_experiments.csv", index=False)
    (out_dir / "completed_experiments.json").write_text(
        json.dumps({"configured": len(specs), "completed": len(rows), "rows": rows}, indent=2, allow_nan=True, default=str),
        encoding="utf-8",
    )
    if frame.empty:
        print("No completed configured experiments were found.")
        return
    cmc_columns = sorted(column for column in frame.columns if str(column).startswith("CMC_") and str(column).endswith("x"))
    numeric = [
        column for column in ["score", "J", "U_J", "s_1", "TOC_1", "R_ov", "N_eff", "E_J", *cmc_columns]
        if column in frame.columns
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.to_csv(out_dir / "completed_experiments.csv", index=False)
    for group in ("task", "model", "intervention", "phase", "suite"):
        frame.groupby(group, dropna=False)[numeric].agg(["count", "median", "mean"]).to_csv(
            out_dir / f"summary_by_{group}.csv"
        )
    for metric in ["U_J", "TOC_1", "R_ov", "N_eff", "E_J", *cmc_columns]:
        if metric in frame:
            plot_metric(frame, metric, out_dir)
    print(f"Wrote {len(frame)} completed configured experiments to {out_dir}")


if __name__ == "__main__":
    main()
