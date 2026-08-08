#!/usr/bin/env python3
"""Run the non-poisoning experiment programme.

This file owns the executable task/model/intervention catalogue. Numerical
implementations remain in the numbered pipeline and analysis modules. Test is
the default evaluation split; callers may explicitly select train or all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

REPO_ROOT = Path(__file__).resolve().parents[1]

from analysis.primary_matrix import PRIMARY_PROFILE_CHOICES
from experiments.execution import RunSpec, apply_filters, deduplicate, parse_filter, run_pipeline

TASKS = ("random_fsm", "grammar_acceptability", "hans_nli", "arithmetic", "bon_jailbreaking")
INTERVENTIONS = ("mean-donor", "mean", "zero")
MODES = ("standard", "decode-only")


def phenomenology_experiments() -> list[RunSpec]:
    models = (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1b@step0",
        "EleutherAI/pythia-1b@step48000",
        "EleutherAI/pythia-1b@step96000",
    )
    specs: list[RunSpec] = []
    for intervention in INTERVENTIONS:
        # Qwen2-1.5B arithmetic configurations retained from the original sweep.
        for mode in MODES:
            specs.append(RunSpec("phenomenology", "arithmetic", "Qwen/Qwen2-1.5B-Instruct",
                                 intervention, mode, z_thresh=10))
        # Remaining small-model matrix.
        for model in models:
            allowed_tasks = TASKS if not model.startswith("EleutherAI/pythia-1b@") else TASKS[:3]
            for task in allowed_tasks:
                z = 10 if task == "arithmetic" and model.startswith("Qwen/") else (5 if task == "arithmetic" else -1)
                for mode in MODES:
                    specs.append(RunSpec("phenomenology", task, model, intervention, mode, z_thresh=z))
    # ICLR 28-profile setting. It is explicit rather than inferred from a row count.
    specs.append(RunSpec(
        "phenomenology", "hans_nli", "Qwen/Qwen2-1.5B-Instruct",
        "mean-donor", "standard", z_thresh=-1,
    ))
    return deduplicate(specs)


def large_model_experiments() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for model in ("Qwen/Qwen2-7B-Instruct", "EleutherAI/pythia-6.9b"):
        specs.extend([
            RunSpec("large-models", "arithmetic", model, "mean-positional", "decode-only",
                    z_thresh=10 if model.startswith("Qwen/") else 5, batch_size=256,
                    circuit_size=100_000, min_flip_rate=0.2, max_circuits=5, mlp_neurons_only=True),
            RunSpec("large-models", "hans_nli", model, "mean-positional", "standard",
                    circuit_size=100_000, min_flip_rate=0.2, mlp_neurons_only=True),
            RunSpec("large-models", "bon_jailbreaking", model, "mean-positional", "decode-only",
                    batch_size=256 if model.startswith("Qwen/") else 32,
                    circuit_size=100_000, min_flip_rate=0.2, mlp_neurons_only=True),
        ])
    return deduplicate(specs)



SUITES = {
    "phenomenology": phenomenology_experiments,
    "large-models": large_model_experiments,
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
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--results-root", "--analysis-root", dest="results_root", default=str(REPO_ROOT / "results"),
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
                run_pipeline(REPO_ROOT, spec, dry_run=args.dry_run)
            except subprocess.CalledProcessError as exc:
                failures.append({"experiment": spec.identity, "returncode": exc.returncode})
                if not args.continue_on_error:
                    raise
        (analysis_root / "pipeline_failures.json").write_text(json.dumps(failures, indent=2))
    if args.phase in {"all", "analysis"} and not args.dry_run:
        if args.generate_primary_manuscript and not args.skip_manuscript_outputs:
            subprocess.run([
                sys.executable, "-m", "analysis.29_generate_final_results",
                "--data-root", str(Path(args.data_root)),
                "--results-root", str(analysis_root),
                "--primary-profile", args.primary_profile,
                "--catalogue-json", str(analysis_root / "configured_experiments.json"),
                "--require-complete-new-metrics",
            ], cwd=REPO_ROOT, check=True)
        else:
            subprocess.run([
                sys.executable, "-m", "analysis.28_visualize_experiment_results",
                "--catalogue_json", str(analysis_root / "configured_experiments.json"),
                "--data_root", str(Path(args.data_root)),
                "--out_dir", str(analysis_root / "catalogue"),
            ], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
