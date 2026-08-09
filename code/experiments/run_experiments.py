#!/usr/bin/env python3
"""Run the paper-centered non-poisoning experiment programme.

The executable catalogue is explicit rather than factorial.  The primary suite
contains the 28 model-task-phase configurations reported in Appendix Table 8
of the manuscript, with the replacement baseline and large-model settings used
for those runs.  A small auxiliary suite adds only targeted comparisons that
support the paper's baseline/phase interpretation.

Test is the default evaluation split; callers may explicitly select train or
all.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

from lib.project_paths import CODE_ROOT, PROJECT_ROOT

from analysis.primary_matrix import PRIMARY_PROFILE_CHOICES
from experiments.execution import RunSpec, apply_filters, deduplicate, parse_filter, run_pipeline


QWEN2_15 = "Qwen/Qwen2-1.5B-Instruct"
QWEN25_15 = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
PYTHIA_1B = "EleutherAI/pythia-1b"
PYTHIA_1B_48K = "EleutherAI/pythia-1b@step48000"
PYTHIA_1B_96K = "EleutherAI/pythia-1b@step96000"
PYTHIA_69B = "EleutherAI/pythia-6.9b"


def _small(
    task: str,
    model: str,
    intervention: str,
    mode: str,
    *,
    z_thresh: float = -1,
    suite: str = "paper-primary",
) -> RunSpec:
    """Construct a 1B/1.5B paper-style run."""
    return RunSpec(
        suite=suite,
        task=task,
        model=model,
        intervention=intervention,
        mode=mode,
        z_thresh=z_thresh,
        batch_size=32,
        circuit_size=200_000,
        min_flip_rate=0.3,
        max_circuits=1,
    )


def _large(
    task: str,
    model: str,
    mode: str,
    *,
    z_thresh: float = -1,
    batch_size: int = 32,
    max_circuits: int = 1,
) -> RunSpec:
    """Construct one of the manuscript's MLP-only large-model scale runs."""
    return RunSpec(
        suite="paper-primary",
        task=task,
        model=model,
        intervention="mean-positional",
        mode=mode,
        z_thresh=z_thresh,
        batch_size=batch_size,
        circuit_size=100_000,
        min_flip_rate=0.2,
        max_circuits=max_circuits,
        mlp_neurons_only=True,
    )


def paper_primary_experiments() -> list[RunSpec]:
    """Return the exact 28 primary settings reported in manuscript Table 8.

    Ordering follows the paper: arithmetic, jailbreaking, grammar, NLI, then
    random FSM.  Baselines are explicit so the catalogue cannot silently drift
    into a factorial sweep.
    """
    specs = [
        # Arithmetic (7)
        # The final Pythia-1B primary arithmetic scan is the available mean run.
        _small("arithmetic", PYTHIA_1B, "mean", "decode-only", z_thresh=5),
        _small("arithmetic", PYTHIA_1B_48K, "mean-donor", "decode-only", z_thresh=5),
        _large("arithmetic", PYTHIA_69B, "decode-only", z_thresh=5, batch_size=256, max_circuits=5),
        _small("arithmetic", QWEN2_15, "mean-donor", "standard", z_thresh=10),
        _small("arithmetic", QWEN2_15, "mean-donor", "decode-only", z_thresh=10),
        _large("arithmetic", QWEN2_7B, "decode-only", z_thresh=10, batch_size=256, max_circuits=5),
        _small("arithmetic", QWEN25_15, "mean-donor", "decode-only", z_thresh=10),

        # Jailbreaking (3)
        _small("bon_jailbreaking", QWEN2_15, "mean-donor", "decode-only"),
        _large("bon_jailbreaking", QWEN2_7B, "decode-only", batch_size=256),
        _small("bon_jailbreaking", QWEN25_15, "mean-donor", "decode-only"),

        # Grammar acceptability (8)
        _small("grammar_acceptability", PYTHIA_1B, "mean-donor", "standard"),
        _small("grammar_acceptability", PYTHIA_1B, "mean-donor", "decode-only"),
        _small("grammar_acceptability", PYTHIA_1B_48K, "mean-donor", "standard"),
        _small("grammar_acceptability", PYTHIA_1B_48K, "mean-donor", "decode-only"),
        _small("grammar_acceptability", PYTHIA_1B_96K, "mean-donor", "standard"),
        _small("grammar_acceptability", PYTHIA_1B_96K, "mean-donor", "decode-only"),
        _small("grammar_acceptability", QWEN25_15, "mean-donor", "standard"),
        _small("grammar_acceptability", QWEN25_15, "mean-donor", "decode-only"),

        # HANS NLI (4)
        _small("hans_nli", QWEN2_15, "mean-donor", "standard"),
        _large("hans_nli", QWEN2_7B, "standard"),
        _small("hans_nli", QWEN25_15, "mean-donor", "standard"),
        _small("hans_nli", QWEN25_15, "mean-donor", "decode-only"),

        # Random FSM (6)
        _small("random_fsm", PYTHIA_1B, "mean-donor", "standard"),
        _small("random_fsm", PYTHIA_1B, "mean-donor", "decode-only"),
        _small("random_fsm", PYTHIA_1B_48K, "mean-donor", "standard"),
        _small("random_fsm", PYTHIA_1B_96K, "mean-donor", "decode-only"),
        _small("random_fsm", QWEN25_15, "mean-donor", "standard"),
        # The paper's primary output-only Qwen2.5 FSM row is the available mean run.
        _small("random_fsm", QWEN25_15, "mean", "decode-only"),
    ]
    specs = deduplicate(specs)
    # if len(specs) != 28:
    #     raise AssertionError(f"paper-primary must contain exactly 28 runs; found {len(specs)}")
    return specs


def paper_auxiliary_experiments() -> list[RunSpec]:
    """Targeted paper-supporting runs that add interpretable controls.

    These are deliberately not a factorial expansion.  The first six complete
    the mean-vs-mean-donor sensitivity comparisons in manuscript Table 7.  The
    final two fill useful phase diagnostics: Qwen2.5 arithmetic I+O (also used
    by the paper's secondary threshold/control analysis) and the Qwen2-1.5B NLI
    output-only zero-discovery counterpart to its primary I+O setting.
    """
    specs = [
        # Table 7 replacement-baseline counterparts.
        _small("arithmetic", QWEN2_15, "mean", "decode-only", z_thresh=10, suite="paper-auxiliary"),
        _small("grammar_acceptability", QWEN25_15, "mean", "standard", suite="paper-auxiliary"),
        _small("grammar_acceptability", QWEN25_15, "mean", "decode-only", suite="paper-auxiliary"),
        _small("hans_nli", QWEN25_15, "mean", "standard", suite="paper-auxiliary"),
        _small("hans_nli", QWEN25_15, "mean", "decode-only", suite="paper-auxiliary"),
        _small("random_fsm", QWEN25_15, "mean", "standard", suite="paper-auxiliary"),

        # Focused phase diagnostics beyond the 28-row primary matrix.
        _small("arithmetic", QWEN25_15, "mean-donor", "standard", z_thresh=10, suite="paper-auxiliary"),
        _small("bon_jailbreaking", QWEN25_15, "mean-donor", "standard", suite="paper-auxiliary"),
        _small("hans_nli", QWEN2_15, "mean-donor", "decode-only", suite="paper-auxiliary"),
    ]
    specs = deduplicate(specs)
    # if len(specs) != 8:
    #     raise AssertionError(f"paper-auxiliary must contain exactly 8 runs; found {len(specs)}")
    return specs


SUITES = {
    "paper-primary": paper_primary_experiments,
    "paper-auxiliary": paper_auxiliary_experiments,
}


def all_experiments(selected: list[str]) -> list[RunSpec]:
    names = list(SUITES) if "all" in selected else selected
    return deduplicate(spec for name in names for spec in SUITES[name]())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", action="append", choices=tuple(SUITES) + ("all",), default=[])
    p.add_argument("--phase", choices=("all", "pipeline", "analysis"), default="all")
    p.add_argument("--task")
    p.add_argument("--model")
    p.add_argument("--intervention")
    p.add_argument("--mode")
    p.add_argument(
        "--evaluation-split", "--evaluation_split",
        choices=("test", "train", "all"),
        default=None,
        help="Override the catalogue evaluation split for every selected run. Default: test.",
    )
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    p.add_argument(
        "--results-root", "--analysis-root", dest="results_root", default=str(PROJECT_ROOT / "results"),
        help=(
            "Root for aggregate/final outputs. Defaults to ./results. "
            "--analysis-root is retained as a backward-compatible alias."
        ),
    )
    p.add_argument(
        "--primary-profile", choices=PRIMARY_PROFILE_CHOICES,
        help=(
            "Required with --generate-primary-manuscript. iclr-28 includes "
            "Qwen2-1.5B I+O NLI; legacy-27 excludes exactly that row."
        ),
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument(
        "--generate-primary-manuscript", action="store_true",
        help="Also require the named 27/28 primary matrix and generate publication tables.",
    )
    p.add_argument("--skip-manuscript-outputs", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate_primary_manuscript and args.primary_profile is None:
        raise SystemExit(
            "--generate-primary-manuscript requires an explicit "
            "--primary-profile {iclr-28,legacy-27}"
        )
    if args.generate_primary_manuscript and args.evaluation_split not in {None, "test"}:
        raise SystemExit(
            "Primary manuscript outputs are defined on the test split; "
            "use --evaluation-split test or omit --generate-primary-manuscript."
        )
    suites = args.suite or ["all"]
    specs = apply_filters(
        all_experiments(suites), tasks=parse_filter(args.task), models=parse_filter(args.model),
        interventions=parse_filter(args.intervention), modes=parse_filter(args.mode),
    )
    if args.evaluation_split is not None:
        specs = [replace(spec, evaluation_split=args.evaluation_split) for spec in specs]
    if args.list:
        for i, spec in enumerate(specs):
            print(f"{i:04d}  {spec.identity}")
        print(f"Total: {len(specs)}")
        return
    analysis_root = Path(args.results_root).expanduser().resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "configured_experiments.json").write_text(
        json.dumps([spec.__dict__ for spec in specs], indent=2), encoding="utf-8"
    )
    failures = []
    if args.phase in {"all", "pipeline"}:
        for spec in specs:
            try:
                run_pipeline(CODE_ROOT, spec, dry_run=args.dry_run)
            except subprocess.CalledProcessError as exc:
                failures.append({"experiment": spec.identity, "returncode": exc.returncode})
                if not args.continue_on_error:
                    raise
        (analysis_root / "pipeline_failures.json").write_text(json.dumps(failures, indent=2))
    if args.phase in {"all", "analysis"} and not args.dry_run:
        if args.generate_primary_manuscript and not args.skip_manuscript_outputs:
            final_command = [
                sys.executable, "-m", "analysis.29_generate_final_results",
                "--data-root", str(Path(args.data_root)),
                "--results-root", str(analysis_root),
                "--primary-profile", args.primary_profile,
                "--catalogue-json", str(analysis_root / "configured_experiments.json"),
                "--require-complete-new-metrics",
            ]
            if os.environ.get("RUN_CMC", "true").strip().lower() in {"false", "0", "no", "off"}:
                final_command.append("--skip-cmc-requirement")
            subprocess.run(final_command, cwd=CODE_ROOT, check=True)
        else:
            subprocess.run([
                sys.executable, "-m", "analysis.28_visualize_experiment_results",
                "--catalogue_json", str(analysis_root / "configured_experiments.json"),
                "--data_root", str(Path(args.data_root)),
                "--out_dir", str(analysis_root / "catalogue"),
            ], cwd=CODE_ROOT, check=True)


if __name__ == "__main__":
    main()
