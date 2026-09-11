#!/usr/bin/env python3
"""Run the paper-centered overtopping experiment programme.

The executable programme contains 48 explicit configurations rather than a full
factorial sweep: 29 final-snapshot task/model/phase cells, twelve intermediate
Pythia checkpoint cells, and seven matched replacement-baseline repeats. RQ1 uses
all configured settings, while RQ2 analyzes replacement regimes separately.

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

from core.project_paths import CODE_ROOT, PROJECT_ROOT

from studies.overtopping.analysis.lib.primary_matrix import PRIMARY_PROFILE_CHOICES, PROFILE_STUDY_48
from studies.overtopping.experiments.execution import RunSpec, apply_filters, deduplicate, parse_filter, run_pipeline


MANUSCRIPT_PROFILE = PROFILE_STUDY_48


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
    """Return the first internal execution group used to build the study registry."""
    specs = [
        # Arithmetic (7)
        # The final Pythia-1B arithmetic configuration uses the available mean replacement run.
        _small("arithmetic", PYTHIA_1B, "mean", "decode-only", z_thresh=5),
        _small("arithmetic", PYTHIA_1B_48K, "mean-donor", "decode-only", z_thresh=5),
        _large("arithmetic", PYTHIA_69B, "decode-only", z_thresh=5, batch_size=256, max_circuits=5),
        _small("arithmetic", QWEN2_15, "mean-donor", "standard", z_thresh=10),
        _small("arithmetic", QWEN2_15, "mean-donor", "decode-only", z_thresh=10),
        _large("arithmetic", QWEN2_7B, "decode-only", z_thresh=10, batch_size=256, max_circuits=5),
        _small("arithmetic", QWEN25_15, "mean-donor", "decode-only", z_thresh=10),

        # Jailbreaking (3)
        _small("bon_jailbreaking", QWEN2_15, "mean-donor", "decode-only"),
        _large("bon_jailbreaking", QWEN2_7B, "decode-only"),
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
        # The output-only Qwen2.5 FSM configuration uses the available mean replacement run.
        _small("random_fsm", QWEN25_15, "mean", "decode-only"),
    ]
    specs = deduplicate(specs)
    return specs


def paper_auxiliary_experiments() -> list[RunSpec]:
    """Return the second internal execution group used to build the study registry.

    It contains six matched replacement-baseline repeats and five additional
    configured final-snapshot task/model/phase cells.
    """
    specs = [
        # Table 7 replacement-baseline counterparts.
        _small("arithmetic", QWEN2_15, "mean", "decode-only", z_thresh=10, suite="paper-auxiliary"),
        _small("grammar_acceptability", QWEN25_15, "mean", "standard", suite="paper-auxiliary"),
        _small("grammar_acceptability", QWEN25_15, "mean", "decode-only", suite="paper-auxiliary"),
        _small("hans_nli", QWEN25_15, "mean", "standard", suite="paper-auxiliary"),
        _small("hans_nli", QWEN25_15, "mean", "decode-only", suite="paper-auxiliary"),
        _small("random_fsm", QWEN25_15, "mean", "standard", suite="paper-auxiliary"),

        # Additional final-snapshot phase/model coverage in the study registry.
        _small("arithmetic", QWEN25_15, "mean-donor", "standard", z_thresh=10, suite="paper-auxiliary"),
        _small("bon_jailbreaking", QWEN25_15, "mean-donor", "standard", suite="paper-auxiliary"),
        _small("hans_nli", QWEN2_15, "mean-donor", "decode-only", suite="paper-auxiliary"),
        _small("grammar_acceptability", QWEN2_15, "mean-donor", "standard", suite="paper-auxiliary"),
        _small("grammar_acceptability", QWEN2_15, "mean-donor", "decode-only", suite="paper-auxiliary"),
    ]
    specs = deduplicate(specs)
    return specs



def paper_coverage_experiments() -> list[RunSpec]:
    """Return configured coverage cells completing the study design.

    Five checkpoint configurations are retained as explicit zero-candidate
    observations when their existing artifacts report no discovered channels.
    Two Qwen2-1.5B Random-FSM cells complete the small-model family comparison.
    Two final Pythia-1B Arithmetic cells complete the matched phase/baseline
    comparison at the final snapshot.
    """
    specs = [
        # Existing Pythia checkpoint observations.
        _small("arithmetic", PYTHIA_1B_96K, "mean-donor", "decode-only", z_thresh=5, suite="paper-coverage"),
        _small("random_fsm", PYTHIA_1B_48K, "mean-donor", "decode-only", suite="paper-coverage"),
        _small("random_fsm", PYTHIA_1B_96K, "mean-donor", "standard", suite="paper-coverage"),
        _small("arithmetic", PYTHIA_1B_48K, "mean-donor", "standard", z_thresh=5, suite="paper-coverage"),
        _small("arithmetic", PYTHIA_1B_96K, "mean-donor", "standard", z_thresh=5, suite="paper-coverage"),

        # Small-Qwen coverage completion.
        _small("random_fsm", QWEN2_15, "mean-donor", "standard", suite="paper-coverage"),
        _small("random_fsm", QWEN2_15, "mean-donor", "decode-only", suite="paper-coverage"),

        # Final-snapshot Pythia Arithmetic completion.
        _small("arithmetic", PYTHIA_1B, "mean-donor", "standard", z_thresh=5, suite="paper-coverage"),
        _small("arithmetic", PYTHIA_1B, "mean-donor", "decode-only", z_thresh=5, suite="paper-coverage"),
    ]
    return deduplicate(specs)


def paper_study_44_experiments() -> list[RunSpec]:
    """Return the 44 configurations present before the Pythia Arithmetic completion."""
    previous_coverage = [
        _small("arithmetic", PYTHIA_1B_96K, "mean-donor", "decode-only", z_thresh=5, suite="paper-coverage"),
        _small("random_fsm", PYTHIA_1B_48K, "mean-donor", "decode-only", suite="paper-coverage"),
        _small("random_fsm", PYTHIA_1B_96K, "mean-donor", "standard", suite="paper-coverage"),
        _small("random_fsm", QWEN2_15, "mean-donor", "standard", suite="paper-coverage"),
        _small("random_fsm", QWEN2_15, "mean-donor", "decode-only", suite="paper-coverage"),
    ]
    return deduplicate([*paper_reference_experiments(), *previous_coverage])

def paper_reference_experiments() -> list[RunSpec]:
    """Return the storage-protected configurations with existing results."""
    return deduplicate([*paper_primary_experiments(), *paper_auxiliary_experiments()])

def paper_study_experiments() -> list[RunSpec]:
    """Return the complete 48-setting overtopping study registry.

    Analysis code starts from this registry and then applies metric-specific
    applicability and availability rules.
    """
    specs = deduplicate([*paper_reference_experiments(), *paper_coverage_experiments()])
    if len(specs) != 48:
        raise AssertionError(f"paper study registry must contain 48 unique settings; found {len(specs)}")
    return specs


def qwen_small_completion_experiments() -> list[RunSpec]:
    """Return the two Qwen2-1.5B Random-FSM settings requiring new compute."""
    return [spec for spec in paper_coverage_experiments() if spec.model == QWEN2_15]


def pythia_final_completion_experiments() -> list[RunSpec]:
    """Return the two final-snapshot Pythia-1B Arithmetic settings requiring new compute."""
    return [
        spec for spec in paper_coverage_experiments()
        if spec.model == PYTHIA_1B
        and spec.task == "arithmetic"
        and spec.intervention == "mean-donor"
    ]


def minimal_completion_experiments() -> list[RunSpec]:
    """Return only the four settings that require new model-backed computation."""
    return deduplicate([
        *qwen_small_completion_experiments(),
        *pythia_final_completion_experiments(),
    ])


SUITES = {
    "paper-primary": paper_primary_experiments,
    "paper-auxiliary": paper_auxiliary_experiments,
    "paper-coverage": paper_coverage_experiments,
    "qwen-small-completion": qwen_small_completion_experiments,
    "pythia-final-completion": pythia_final_completion_experiments,
    "minimal-completion": minimal_completion_experiments,
}


def all_experiments(selected: list[str]) -> list[RunSpec]:
    names = list(SUITES) if "all" in selected else selected
    return deduplicate(spec for name in names for spec in SUITES[name]())


def _validate_results_root(*, data_root: Path, results_root: Path) -> None:
    """Keep control/reporting outputs outside persistent scientific storage."""
    data_root = data_root.resolve()
    results_root = results_root.resolve()
    cache_root = (PROJECT_ROOT / "cache").resolve()

    def _is_within(path: Path, parent: Path) -> bool:
        return path == parent or parent in path.parents

    if _is_within(results_root, data_root):
        raise ValueError(
            f"Refusing results output inside data root: results={results_root}, data={data_root}"
        )
    if _is_within(results_root, cache_root):
        raise ValueError(
            f"Refusing results output inside cache root: results={results_root}, cache={cache_root}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", action="append", choices=tuple(SUITES) + ("all",), default=[])
    p.add_argument("--phase", choices=("all", "pipeline", "analysis"), default="all")
    p.add_argument("--task")
    p.add_argument("--model")
    p.add_argument("--intervention")
    p.add_argument("--mode")
    p.add_argument(
        "--evaluation-split",
        choices=("test", "train", "all"),
        default=None,
        help="Override the registry evaluation split for every selected run. Default: test.",
    )
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    p.add_argument(
        "--results-root", dest="results_root", default=str(PROJECT_ROOT / "results"),
        help="Root for aggregate/final outputs. Defaults to ./results.",
    )
    p.add_argument(
        "--primary-profile", choices=PRIMARY_PROFILE_CHOICES, default=MANUSCRIPT_PROFILE,
        help="Validation profile for the configured study table. Default: study-48.",
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument(
        "--generate-primary-manuscript", action="store_true",
        help="Validate the complete configured study table and generate publication tables.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
            print(
                f"{i:04d}  {spec.task} | {spec.model} | "
                f"{spec.intervention} | {spec.phase} | split={spec.evaluation_split}"
            )
        print(f"Total: {len(specs)}")
        return
    data_root = Path(args.data_root).expanduser().resolve()
    analysis_root = Path(args.results_root).expanduser().resolve()
    _validate_results_root(data_root=data_root, results_root=analysis_root)
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
        if args.generate_primary_manuscript:
            final_command = [
                sys.executable, "-m", "reporting.generate_final_results",
                "--data-root", str(data_root),
                "--results-root", str(analysis_root),
                "--primary-profile", args.primary_profile,
                "--catalogue-json", str(analysis_root / "configured_experiments.json"),
                "--require-complete-metrics",
            ]
            if os.environ.get("RUN_CMC", "true").strip().lower() in {"false", "0", "no", "off"}:
                final_command.append("--skip-cmc-requirement")
            subprocess.run(final_command, cwd=CODE_ROOT, check=True)
        else:
            subprocess.run([
                sys.executable, "-m", "studies.overtopping.analysis.stage01_visualize_experiment_results",
                "--catalogue_json", str(analysis_root / "configured_experiments.json"),
                "--data_root", str(data_root),
                "--out_dir", str(analysis_root / "catalogue"),
            ], cwd=CODE_ROOT, check=True)


if __name__ == "__main__":
    main()
