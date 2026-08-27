#!/usr/bin/env python3
"""Generate all final paper statistics and figures under the repository results/ tree.

This module is intentionally orchestration-only. Numerical/statistical logic remains
in the dedicated analysis scripts that it invokes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from core.project_paths import CODE_ROOT, PROJECT_ROOT


from studies.overtopping.analysis.lib.primary_matrix import PRIMARY_PROFILE_CHOICES
from reporting.result_paths import (
    experiment_catalogue,
    manuscript_figures,
    manuscript_materials,
    metric_completeness_audit,
    overtopping_spiking_diagnostics,
    poisoning_aggregate_tables,
    poisoning_figures,
    poisoning_results,
    primary_statistics,
    primary_tables,
)
from studies.poisoning.lib.run_paths import TRAJECTORIES_DIRNAME, TRAINING_DIRNAME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data"), help="Experiment-artifact root. Default: <repo>/data")
    p.add_argument("--results-root", default=str(PROJECT_ROOT / "results"), help="Final-output root. Default: <repo>/results")
    p.add_argument("--primary-profile", required=True, choices=PRIMARY_PROFILE_CHOICES)
    p.add_argument(
        "--poisoning-root",
        dest="poisoning_root",
        default=str(PROJECT_ROOT / "data" / "poisoning"),
        help="Poisoning data root. Default: <repo>/data/poisoning",
    )
    p.add_argument(
        "--catalogue-json",
        default=None,
        help="Optional configured_experiments.json. If present, catalogue plots are refreshed.",
    )
    p.add_argument("--skip-paper-figures", action="store_true")
    p.add_argument("--skip-spiking-report", action="store_true")
    p.add_argument("--skip-poisoning-report", action="store_true")
    p.add_argument(
        "--require-complete-new-metrics", action="store_true",
        help=(
            "Fail if any primary setting lacks the exact metrics required by the selected audit mode. "
            "An audit is always written under results/primary_analysis/metric_completeness_audit/."
        ),
    )
    p.add_argument(
        "--skip-cmc-requirement", action="store_true",
        help="Do not require CMC for the completeness audit; simultaneous E(J) and its matched null remain required.",
    )
    return p.parse_args()


def run(command: list[str]) -> None:
    print("[final-results]", " ".join(command))
    subprocess.run(command, cwd=CODE_ROOT, check=True)


def has_spiking_diagnostics(data_root: Path) -> bool:
    return any(
        path.is_dir() and (path.name == "spiking_diagnostics" or path.name.startswith("spiking_diagnostics-cap"))
        for path in data_root.rglob("spiking_diagnostics*")
    )


def poisoning_run_dirs(poisoning_root: Path) -> list[Path]:
    """Return canonical stage-organized poisoning runs only."""
    out: list[Path] = []
    for task in ("arithmetic", "grammar"):
        task_root = poisoning_root / task
        if not task_root.is_dir():
            continue
        for run_dir in sorted(p for p in task_root.iterdir() if p.is_dir()):
            cfg = run_dir / TRAINING_DIRNAME / "metadata" / "run_config.json"
            trajectories = run_dir / TRAJECTORIES_DIRNAME
            if not cfg.is_file() or not any(trajectories.glob("*/backdoor_lift_overtopping_trajectory.csv")):
                continue
            try:
                config = json.loads(cfg.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(config.get("condition", "both")) != "both":
                continue
            out.append(run_dir.resolve())
    return out


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    poisoning_root = Path(args.poisoning_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    catalogue_json = Path(args.catalogue_json).expanduser().resolve() if args.catalogue_json else None
    if catalogue_json is not None and catalogue_json.is_file():
        run([
            sys.executable, "-m", "studies.overtopping.analysis.stage01_visualize_experiment_results",
            "--catalogue_json", str(catalogue_json),
            "--data_root", str(data_root),
            "--out_dir", str(experiment_catalogue(results_root)),
        ])

    paper_tables = primary_tables(results_root)
    run([
        sys.executable, "-m", "studies.overtopping.analysis.stage02_overtopping_latex_tables",
        "--results", str(data_root),
        "--out", str(paper_tables),
        "--primary-profile", args.primary_profile,
    ])

    audit_command = [
        sys.executable, "-m", "studies.overtopping.analysis.stage03_audit_required_metrics",
        "--primary-table", str(paper_tables / "primary_table.csv"),
        "--data-root", str(data_root),
        "--primary-profile", args.primary_profile,
        "--out-dir", str(metric_completeness_audit(results_root)),
    ]
    if args.require_complete_new_metrics:
        audit_command.append("--require-complete")
    if args.skip_cmc_requirement:
        audit_command.append("--skip-cmc-requirement")
    run(audit_command)

    run([
        sys.executable, "-m", "studies.overtopping.analysis.stage04_analyze_primary_metrics",
        "primary",
        "--primary_table", str(paper_tables / "primary_table.csv"),
        "--data_root", str(data_root),
        "--out_dir", str(primary_statistics(results_root)),
    ])

    run([
        sys.executable, "-m", "studies.overtopping.analysis.stage05_generate_manuscript_outputs",
        "--primary_table", str(paper_tables / "primary_table.csv"),
        "--data_root", str(data_root),
        "--out_dir", str(manuscript_materials(results_root)),
        "--primary_profile", args.primary_profile,
    ])

    if not args.skip_paper_figures:
        paper_figures = manuscript_figures(results_root)
        run([
            sys.executable, "-m", "studies.overtopping.analysis.stage06_competence_vs_overtopping_figures",
            "--results-dir", str(data_root),
            "--out", str(paper_figures / "fig_competence_vs_coverage.pdf"),
            "--paper-figures-dir", str(paper_figures),
            "--paper-figures", "all",
        ])

    # Poisoning run diagnostics stay inside each data/poisoning/<task>/<run> directory.
    # results/ receives only manuscript-facing poisoning outputs.
    poisoning_out = poisoning_results(results_root)
    poisoning_paper_figures = poisoning_figures(results_root)
    poisoning_aggregate = poisoning_aggregate_tables(results_root)
    poisoning_runs = poisoning_run_dirs(poisoning_root)
    if not args.skip_poisoning_report and poisoning_runs:
        poisoning_aggregate.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "-m", "studies.poisoning.stage08_aggregate_cross_seed",
            "--run_dirs", ",".join(str(path) for path in poisoning_runs),
            "--output_dir", str(poisoning_aggregate),
        ])
        run([
            sys.executable, "-m", "studies.poisoning.stage08_plot_cross_seed",
            "--input_dir", str(poisoning_aggregate),
            "--output_dir", str(poisoning_paper_figures),
        ])

    spiking_out = overtopping_spiking_diagnostics(results_root)
    spiking_out.mkdir(parents=True, exist_ok=True)
    if args.skip_spiking_report:
        status = {"status": "skipped", "reason": "--skip-spiking-report"}
        (spiking_out / "report_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    elif has_spiking_diagnostics(data_root):
        run([
            sys.executable, "-m", "studies.overtopping.analysis.stage07_overtopping_spiking_report",
            "--root", str(data_root),
            "--out", str(spiking_out),
        ])
    else:
        status = {
            "status": "not_available",
            "reason": "No spiking_diagnostics directories were found under the data root.",
            "data_root": str(data_root),
        }
        (spiking_out / "report_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[final-results] spiking report unavailable; wrote {spiking_out / 'report_status.json'}")

    manifest = {
        "data_root": str(data_root),
        "results_root": str(results_root),
        "primary_profile": args.primary_profile,
        "primary_tables": str(paper_tables),
        "metric_completeness_audit": str(metric_completeness_audit(results_root)),
        "manuscript_figures": str(manuscript_figures(results_root)),
        "primary_statistics": str(primary_statistics(results_root)),
        "manuscript_materials": str(manuscript_materials(results_root)),
        "overtopping_spiking_diagnostics": str(spiking_out),
        "poisoning_paper_outputs": str(poisoning_out),
        "poisoning_root": str(poisoning_root),
        "experiment_catalogue": str(experiment_catalogue(results_root)),
    }
    (results_root / "final_results_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[final-results] all final outputs are rooted at {results_root}")


if __name__ == "__main__":
    main()
