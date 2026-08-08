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

REPO_ROOT = Path(__file__).resolve().parents[1]

from analysis.primary_matrix import PRIMARY_PROFILE_CHOICES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(REPO_ROOT / "data"), help="Experiment-artifact root. Default: <repo>/data")
    p.add_argument("--results-root", default=str(REPO_ROOT / "results"), help="Final-output root. Default: <repo>/results")
    p.add_argument("--primary-profile", required=True, choices=PRIMARY_PROFILE_CHOICES)
    p.add_argument(
        "--catalogue-json",
        default=None,
        help="Optional configured_experiments.json. If present, catalogue plots are refreshed.",
    )
    p.add_argument("--skip-paper-figures", action="store_true")
    p.add_argument("--skip-spiking-report", action="store_true")
    p.add_argument(
        "--require-complete-new-metrics", action="store_true",
        help=(
            "Fail if any primary setting lacks exact TOC/OCC/simultaneous E(J)/GCCR/matched-null outputs. "
            "An audit is always written under results/required_metrics_audit/."
        ),
    )
    return p.parse_args()


def run(command: list[str]) -> None:
    print("[final-results]", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def has_spiking_diagnostics(data_root: Path) -> bool:
    return any(data_root.rglob("spiking_diagnostics"))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    catalogue_json = Path(args.catalogue_json).expanduser().resolve() if args.catalogue_json else None
    if catalogue_json is not None and catalogue_json.is_file():
        run([
            sys.executable, "-m", "analysis.28_visualize_experiment_results",
            "--catalogue_json", str(catalogue_json),
            "--data_root", str(data_root),
            "--out_dir", str(results_root / "catalogue"),
        ])

    paper_tables = results_root / "paper_tables"
    run([
        sys.executable, "-m", "analysis.compute_overtopping_latex_tables",
        "--results", str(data_root),
        "--out", str(paper_tables),
        "--primary-profile", args.primary_profile,
    ])

    audit_command = [
        sys.executable, "-m", "analysis.30_audit_required_metrics",
        "--primary-table", str(paper_tables / "primary_table.csv"),
        "--data-root", str(data_root),
        "--primary-profile", args.primary_profile,
        "--out-dir", str(results_root / "required_metrics_audit"),
    ]
    if args.require_complete_new_metrics:
        audit_command.append("--require-complete")
    run(audit_command)

    run([
        sys.executable, "-m", "analysis.21_analyze_primary_metrics",
        "primary",
        "--primary_table", str(paper_tables / "primary_table.csv"),
        "--data_root", str(data_root),
        "--out_dir", str(results_root / "primary_metrics"),
    ])

    run([
        sys.executable, "-m", "analysis.27_generate_manuscript_outputs",
        "--primary_table", str(paper_tables / "primary_table.csv"),
        "--data_root", str(data_root),
        "--out_dir", str(results_root / "manuscript"),
        "--primary_profile", args.primary_profile,
    ])

    if not args.skip_paper_figures:
        paper_figures = results_root / "paper_figures"
        run([
            sys.executable, "-m", "analysis.make_competence_vs_overtopping_paper_figures",
            "--results-dir", str(data_root),
            "--out", str(paper_figures / "fig_competence_vs_coverage.pdf"),
            "--paper-figures-dir", str(paper_figures),
            "--paper-figures", "all",
        ])

    spiking_out = results_root / "overtopping_spiking_report"
    spiking_out.mkdir(parents=True, exist_ok=True)
    if args.skip_spiking_report:
        status = {"status": "skipped", "reason": "--skip-spiking-report"}
        (spiking_out / "report_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    elif has_spiking_diagnostics(data_root):
        run([
            sys.executable, "-m", "analysis.generate_overtopping_spiking_report",
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
        "paper_tables": str(paper_tables),
        "required_metrics_audit": str(results_root / "required_metrics_audit"),
        "paper_figures": str(results_root / "paper_figures"),
        "primary_metrics": str(results_root / "primary_metrics"),
        "manuscript": str(results_root / "manuscript"),
        "overtopping_spiking_report": str(spiking_out),
        "catalogue": str(results_root / "catalogue"),
    }
    (results_root / "final_results_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[final-results] all final outputs are rooted at {results_root}")


if __name__ == "__main__":
    main()
