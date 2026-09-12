#!/usr/bin/env python3
"""Generate the canonical RQ4 Pythia-1B checkpoint trajectory figures.

This is the stable RQ4 entry point used by the final-results orchestrator.  The
actual discovery and plotting implementation remains in
``stage06_competence_vs_overtopping_figures`` so there is only one definition of
the checkpoint population and visual encoding.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from core.project_paths import PROJECT_ROOT
from studies.overtopping.analysis import stage06_competence_vs_overtopping_figures as stage06


TRAJECTORY_SPECS = (
    ("pooled", "fig5a_pythia_checkpoint_trajectory.pdf"),
    ("i2c", "fig5s1_pythia_checkpoint_U_0to1.pdf"),
    ("c2i", "fig5s2_pythia_checkpoint_U_1to0.pdf"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "data"))
    p.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "paper" / "figures" / "05_rq4_learning"),
    )
    p.add_argument("--phase", choices=["input+output", "decode-only"], default="input+output")
    p.add_argument("--baseline", choices=["mean-donor", "mean"], default="mean-donor")
    p.add_argument("--pad-inches", type=float, default=0.01)
    p.add_argument("--no-tight-bbox", action="store_true")
    p.add_argument("--no-directional", action="store_true", help="Generate only the pooled main figure.")
    p.add_argument("--no-csv", action="store_true", help="Do not emit per-figure CSV sidecars.")
    p.add_argument("--csv-out-dir", default=None)
    return p.parse_args()


def generate(args: argparse.Namespace) -> None:
    stage06.setup_matplotlib()
    root = stage06.locate_results_root(Path(args.results_dir))
    filters = stage06.rq1_manuscript_filters()
    checkpoint_filters = stage06.paper_filters_from(filters, include_empty=True)

    points = stage06.discover_points(root, checkpoint_filters, dedupe=False)
    points = stage06.ensure_checkpoint_dataset_score_points(
        root,
        points,
        phase=args.phase,
        baseline=args.baseline,
    )
    if not points:
        raise RuntimeError("no Pythia checkpoint points available for RQ4 checkpoint trajectory")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = TRAJECTORY_SPECS[:1] if args.no_directional else TRAJECTORY_SPECS

    for metric, filename in specs:
        metric_points = stage06.transform_points_for_coverage_metric(root, points, metric)
        if not metric_points:
            raise RuntimeError(f"no checkpoint points available for RQ4 metric {metric}")
        plot_args = Namespace(
            paper_baseline=args.baseline,
            paper_checkpoint_phase=args.phase,
            coverage_metric=metric,
            no_csv=bool(args.no_csv),
            csv_out_dir=args.csv_out_dir,
            no_tight_bbox=bool(args.no_tight_bbox),
            pad_inches=float(args.pad_inches),
        )
        stage06.plot_checkpoint_trajectory_figure(
            metric_points,
            checkpoint_filters,
            out_dir / filename,
            plot_args,
        )


if __name__ == "__main__":
    generate(parse_args())
