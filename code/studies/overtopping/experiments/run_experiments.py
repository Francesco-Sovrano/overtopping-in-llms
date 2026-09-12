#!/usr/bin/env python3
"""Run the paper-centered overtopping experiment programme.

The current registry is deliberately organized into four execution sets only:

1. ``mean-donor``: small-model, non-checkpoint mean-donor runs;
2. ``6-7b-models``: the 6.9B/7B scale runs (mean-positional);
3. ``mean``: small-model mean-replacement runs;
4. ``checkpoints``: the Pythia-1B longitudinal trajectories for Grammar,
   HANS-NLI, and Random FSM.  Each trajectory is step0 -> step48k -> step96k
   -> ``EleutherAI/pythia-1b`` (the final/all-steps model). Arithmetic is not
   checkpointed.

The four sets are disjoint at RunSpec level and together contain the same 56
current RunSpecs used before this cleanup. Reclassification changes only the
``suite`` metadata; persistent data/cache paths and pipeline commands are
unchanged, so existing artifacts remain reusable.

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

from studies.overtopping.analysis.lib.primary_matrix import PRIMARY_PROFILE_CHOICES, PROFILE_STUDY_56
from studies.overtopping.experiments.execution import RunSpec, apply_filters, deduplicate, parse_filter, run_pipeline


MANUSCRIPT_PROFILE = PROFILE_STUDY_56


QWEN2_15 = "Qwen/Qwen2-1.5B-Instruct"
QWEN25_15 = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
PYTHIA_1B = "EleutherAI/pythia-1b"
PYTHIA_1B_STEP0 = "EleutherAI/pythia-1b@step0"
PYTHIA_1B_48K = "EleutherAI/pythia-1b@step48000"
PYTHIA_1B_96K = "EleutherAI/pythia-1b@step96000"
PYTHIA_69B = "EleutherAI/pythia-6.9b"


# Four and only four current execution sets.  Their union is the 56-RunSpec
# study registry; unlike the old primary/coverage/auxiliary buckets, these sets
# are pairwise disjoint by exact RunSpec identity.
CANONICAL_STUDY_SETTING_COUNT = 56
MEAN_DONOR_SETTING_COUNT = 20
LARGE_MODEL_SETTING_COUNT = 4
MEAN_SETTING_COUNT = 8
CHECKPOINT_SETTING_COUNT = 24


def _small(
    task: str,
    model: str,
    intervention: str,
    mode: str,
    *,
    z_thresh: float = -1,
    suite: str = "mean-donor",
) -> RunSpec:
    """Construct a 1B/1.5B paper-style run.

    ``suite`` is execution/reporting metadata only.  It is deliberately absent
    from every persistent path constructor, so reclassifying an existing run
    does not move or invalidate its data/cache artifacts.
    """
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
    suite: str = "6-7b-models",
) -> RunSpec:
    """Construct one of the manuscript's MLP-only large-model scale runs."""
    return RunSpec(
        suite=suite,
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


def _runspec_key(spec: RunSpec) -> tuple:
    """Exact RunSpec identity excluding suite metadata.

    This mirrors ``execution.deduplicate`` and is used to prove that registry
    cleanup does not change the address/scientific identity of an old run.
    """
    return (
        spec.task, spec.model, spec.intervention, spec.mode, spec.z_thresh,
        spec.batch_size, spec.circuit_level, spec.circuit_size,
        spec.min_flip_rate, spec.max_circuits, spec.mlp_neurons_only,
        spec.no_llm_feature_generation, spec.evaluation_split,
    )


def mean_donor_experiments() -> list[RunSpec]:
    """Return the 20 non-checkpoint small-model mean-donor runs.

    Pythia-1B Grammar/HANS-NLI/Random-FSM endpoints are intentionally excluded
    because the all-steps ``EleutherAI/pythia-1b`` model belongs to the
    longitudinal ``checkpoints`` set. Arithmetic Pythia-1B is not checkpointed
    and therefore remains here.
    """
    specs = [
        # Arithmetic.
        _small("arithmetic", PYTHIA_1B, "mean-donor", "standard", z_thresh=5, suite="mean-donor"),
        _small("arithmetic", PYTHIA_1B, "mean-donor", "decode-only", z_thresh=5, suite="mean-donor"),
        _small("arithmetic", QWEN2_15, "mean-donor", "standard", z_thresh=10, suite="mean-donor"),
        _small("arithmetic", QWEN2_15, "mean-donor", "decode-only", z_thresh=10, suite="mean-donor"),
        _small("arithmetic", QWEN25_15, "mean-donor", "standard", z_thresh=10, suite="mean-donor"),
        _small("arithmetic", QWEN25_15, "mean-donor", "decode-only", z_thresh=10, suite="mean-donor"),

        # Jailbreaking.
        _small("bon_jailbreaking", QWEN2_15, "mean-donor", "decode-only", suite="mean-donor"),
        _small("bon_jailbreaking", QWEN25_15, "mean-donor", "standard", suite="mean-donor"),
        _small("bon_jailbreaking", QWEN25_15, "mean-donor", "decode-only", suite="mean-donor"),

        # Grammar acceptability (non-Pythia final models).
        _small("grammar_acceptability", QWEN2_15, "mean-donor", "standard", suite="mean-donor"),
        _small("grammar_acceptability", QWEN2_15, "mean-donor", "decode-only", suite="mean-donor"),
        _small("grammar_acceptability", QWEN25_15, "mean-donor", "standard", suite="mean-donor"),
        _small("grammar_acceptability", QWEN25_15, "mean-donor", "decode-only", suite="mean-donor"),

        # HANS NLI (non-Pythia final models).
        _small("hans_nli", QWEN2_15, "mean-donor", "standard", suite="mean-donor"),
        _small("hans_nli", QWEN2_15, "mean-donor", "decode-only", suite="mean-donor"),
        _small("hans_nli", QWEN25_15, "mean-donor", "standard", suite="mean-donor"),
        _small("hans_nli", QWEN25_15, "mean-donor", "decode-only", suite="mean-donor"),

        # Random FSM (non-Pythia final models; Qwen2.5 Out uses mean below).
        _small("random_fsm", QWEN2_15, "mean-donor", "standard", suite="mean-donor"),
        _small("random_fsm", QWEN2_15, "mean-donor", "decode-only", suite="mean-donor"),
        _small("random_fsm", QWEN25_15, "mean-donor", "standard", suite="mean-donor"),
    ]
    specs = deduplicate(specs)
    return specs


def large_model_experiments() -> list[RunSpec]:
    """Return the four 6.9B/7B scale runs."""
    specs = [
        _large("arithmetic", PYTHIA_69B, "decode-only", z_thresh=5, batch_size=256, max_circuits=5, suite="6-7b-models"),
        _large("arithmetic", QWEN2_7B, "decode-only", z_thresh=10, batch_size=256, max_circuits=5, suite="6-7b-models"),
        _large("bon_jailbreaking", QWEN2_7B, "decode-only", suite="6-7b-models"),
        _large("hans_nli", QWEN2_7B, "standard", suite="6-7b-models"),
    ]
    specs = deduplicate(specs)
    return specs


def mean_experiments() -> list[RunSpec]:
    """Return all eight small-model mean-replacement runs."""
    specs = [
        _small("arithmetic", QWEN2_15, "mean", "decode-only", z_thresh=10, suite="mean"),
        _small("arithmetic", PYTHIA_1B, "mean", "decode-only", z_thresh=5, suite="mean"),
        _small("grammar_acceptability", QWEN25_15, "mean", "standard", suite="mean"),
        _small("grammar_acceptability", QWEN25_15, "mean", "decode-only", suite="mean"),
        _small("hans_nli", QWEN25_15, "mean", "standard", suite="mean"),
        _small("hans_nli", QWEN25_15, "mean", "decode-only", suite="mean"),
        _small("random_fsm", QWEN25_15, "mean", "standard", suite="mean"),
        _small("random_fsm", QWEN25_15, "mean", "decode-only", suite="mean"),
    ]
    specs = deduplicate(specs)
    return specs


def checkpoint_experiments() -> list[RunSpec]:
    """Return the 24 Pythia-1B checkpoint runs.

    Checkpointing is restricted to Grammar, HANS-NLI, and Random FSM.  For each
    task and phase the trajectory is step0 -> step48k -> step96k -> final/all-
    steps ``EleutherAI/pythia-1b``. Arithmetic is intentionally absent.
    """
    specs: list[RunSpec] = []
    for task in ("grammar_acceptability", "hans_nli", "random_fsm"):
        for model in (PYTHIA_1B_STEP0, PYTHIA_1B_48K, PYTHIA_1B_96K, PYTHIA_1B):
            for mode in ("standard", "decode-only"):
                specs.append(_small(task, model, "mean-donor", mode, suite="checkpoints"))
    specs = deduplicate(specs)
    return specs


# ---------------------------------------------------------------------------
# Legacy-address contracts
# ---------------------------------------------------------------------------
# The following private views reproduce the pre-refactor legacy registry exactly.  They are
# retained only so historical study-48/study-44/study-39 validation profiles and the fixed storage
# fingerprint can prove that every historical result/cache address remains
# valid.  Canonical scientific analyses do not construct their populations from
# these historical execution buckets.

def _legacy_primary_28_experiments() -> list[RunSpec]:
    specs = [
        _small("arithmetic", PYTHIA_1B, "mean", "decode-only", z_thresh=5, suite="legacy-paper-primary"),
        _small("arithmetic", PYTHIA_1B_48K, "mean-donor", "decode-only", z_thresh=5, suite="legacy-paper-primary"),
        _large("arithmetic", PYTHIA_69B, "decode-only", z_thresh=5, batch_size=256, max_circuits=5, suite="legacy-paper-primary"),
        _small("arithmetic", QWEN2_15, "mean-donor", "standard", z_thresh=10, suite="legacy-paper-primary"),
        _small("arithmetic", QWEN2_15, "mean-donor", "decode-only", z_thresh=10, suite="legacy-paper-primary"),
        _large("arithmetic", QWEN2_7B, "decode-only", z_thresh=10, batch_size=256, max_circuits=5, suite="legacy-paper-primary"),
        _small("arithmetic", QWEN25_15, "mean-donor", "decode-only", z_thresh=10, suite="legacy-paper-primary"),
        _small("bon_jailbreaking", QWEN2_15, "mean-donor", "decode-only", suite="legacy-paper-primary"),
        _large("bon_jailbreaking", QWEN2_7B, "decode-only", suite="legacy-paper-primary"),
        _small("bon_jailbreaking", QWEN25_15, "mean-donor", "decode-only", suite="legacy-paper-primary"),
        _small("grammar_acceptability", PYTHIA_1B, "mean-donor", "standard", suite="legacy-paper-primary"),
        _small("grammar_acceptability", PYTHIA_1B, "mean-donor", "decode-only", suite="legacy-paper-primary"),
        _small("grammar_acceptability", PYTHIA_1B_48K, "mean-donor", "standard", suite="legacy-paper-primary"),
        _small("grammar_acceptability", PYTHIA_1B_48K, "mean-donor", "decode-only", suite="legacy-paper-primary"),
        _small("grammar_acceptability", PYTHIA_1B_96K, "mean-donor", "standard", suite="legacy-paper-primary"),
        _small("grammar_acceptability", PYTHIA_1B_96K, "mean-donor", "decode-only", suite="legacy-paper-primary"),
        _small("grammar_acceptability", QWEN25_15, "mean-donor", "standard", suite="legacy-paper-primary"),
        _small("grammar_acceptability", QWEN25_15, "mean-donor", "decode-only", suite="legacy-paper-primary"),
        _small("hans_nli", QWEN2_15, "mean-donor", "standard", suite="legacy-paper-primary"),
        _large("hans_nli", QWEN2_7B, "standard", suite="legacy-paper-primary"),
        _small("hans_nli", QWEN25_15, "mean-donor", "standard", suite="legacy-paper-primary"),
        _small("hans_nli", QWEN25_15, "mean-donor", "decode-only", suite="legacy-paper-primary"),
        _small("random_fsm", PYTHIA_1B, "mean-donor", "standard", suite="legacy-paper-primary"),
        _small("random_fsm", PYTHIA_1B, "mean-donor", "decode-only", suite="legacy-paper-primary"),
        _small("random_fsm", PYTHIA_1B_48K, "mean-donor", "standard", suite="legacy-paper-primary"),
        _small("random_fsm", PYTHIA_1B_96K, "mean-donor", "decode-only", suite="legacy-paper-primary"),
        _small("random_fsm", QWEN25_15, "mean-donor", "standard", suite="legacy-paper-primary"),
        _small("random_fsm", QWEN25_15, "mean", "decode-only", suite="legacy-paper-primary"),
    ]
    return deduplicate(specs)


def _legacy_auxiliary_11_experiments() -> list[RunSpec]:
    specs = [
        _small("arithmetic", QWEN2_15, "mean", "decode-only", z_thresh=10, suite="legacy-paper-auxiliary"),
        _small("grammar_acceptability", QWEN25_15, "mean", "standard", suite="legacy-paper-auxiliary"),
        _small("grammar_acceptability", QWEN25_15, "mean", "decode-only", suite="legacy-paper-auxiliary"),
        _small("hans_nli", QWEN25_15, "mean", "standard", suite="legacy-paper-auxiliary"),
        _small("hans_nli", QWEN25_15, "mean", "decode-only", suite="legacy-paper-auxiliary"),
        _small("random_fsm", QWEN25_15, "mean", "standard", suite="legacy-paper-auxiliary"),
        _small("arithmetic", QWEN25_15, "mean-donor", "standard", z_thresh=10, suite="legacy-paper-auxiliary"),
        _small("bon_jailbreaking", QWEN25_15, "mean-donor", "standard", suite="legacy-paper-auxiliary"),
        _small("hans_nli", QWEN2_15, "mean-donor", "decode-only", suite="legacy-paper-auxiliary"),
        _small("grammar_acceptability", QWEN2_15, "mean-donor", "standard", suite="legacy-paper-auxiliary"),
        _small("grammar_acceptability", QWEN2_15, "mean-donor", "decode-only", suite="legacy-paper-auxiliary"),
    ]
    return deduplicate(specs)


def legacy_iclr_28_experiments() -> list[RunSpec]:
    """Return the historical 28-setting ICLR execution subset."""
    return _legacy_primary_28_experiments()


def legacy_study_39_experiments() -> list[RunSpec]:
    return deduplicate([*_legacy_primary_28_experiments(), *_legacy_auxiliary_11_experiments()])


def legacy_study_48_experiments() -> list[RunSpec]:
    legacy_coverage = [
        _small("arithmetic", PYTHIA_1B_96K, "mean-donor", "decode-only", z_thresh=5, suite="legacy-paper-coverage"),
        _small("random_fsm", PYTHIA_1B_48K, "mean-donor", "decode-only", suite="legacy-paper-coverage"),
        _small("random_fsm", PYTHIA_1B_96K, "mean-donor", "standard", suite="legacy-paper-coverage"),
        _small("arithmetic", PYTHIA_1B_48K, "mean-donor", "standard", z_thresh=5, suite="legacy-paper-coverage"),
        _small("arithmetic", PYTHIA_1B_96K, "mean-donor", "standard", z_thresh=5, suite="legacy-paper-coverage"),
        _small("random_fsm", QWEN2_15, "mean-donor", "standard", suite="legacy-paper-coverage"),
        _small("random_fsm", QWEN2_15, "mean-donor", "decode-only", suite="legacy-paper-coverage"),
        _small("arithmetic", PYTHIA_1B, "mean-donor", "standard", z_thresh=5, suite="legacy-paper-coverage"),
        _small("arithmetic", PYTHIA_1B, "mean-donor", "decode-only", z_thresh=5, suite="legacy-paper-coverage"),
    ]
    specs = deduplicate([*legacy_study_39_experiments(), *legacy_coverage])
    return specs


def storage_protected_experiments() -> list[RunSpec]:
    """Return the historical 44 settings covered by the fixed storage fingerprint."""
    protected_coverage = [
        _small("arithmetic", PYTHIA_1B_96K, "mean-donor", "decode-only", z_thresh=5, suite="legacy-paper-coverage"),
        _small("random_fsm", PYTHIA_1B_48K, "mean-donor", "decode-only", suite="legacy-paper-coverage"),
        _small("random_fsm", PYTHIA_1B_96K, "mean-donor", "standard", suite="legacy-paper-coverage"),
        _small("random_fsm", QWEN2_15, "mean-donor", "standard", suite="legacy-paper-coverage"),
        _small("random_fsm", QWEN2_15, "mean-donor", "decode-only", suite="legacy-paper-coverage"),
    ]
    return deduplicate([*legacy_study_39_experiments(), *protected_coverage])


def paper_study_experiments() -> list[RunSpec]:
    """Return the current 56-RunSpec registry as four disjoint execution sets."""
    groups = (
        mean_donor_experiments(),
        large_model_experiments(),
        mean_experiments(),
        checkpoint_experiments(),
    )
    specs = deduplicate(spec for group in groups for spec in group)

    # Persistent-address migration invariant.  The four-set cleanup must not
    # change any scientific RunSpec from the immediately preceding 56-setting
    # registry; suite labels are metadata only and do not enter path builders.
    legacy_specs = legacy_study_48_experiments()
    legacy_keys = {_runspec_key(spec) for spec in legacy_specs}
    new_keys = {_runspec_key(spec) for spec in specs}
    shared = new_keys & legacy_keys
    added = [spec for spec in specs if _runspec_key(spec) not in legacy_keys]
    retired = [spec for spec in legacy_specs if _runspec_key(spec) not in new_keys]
    expected_added = {
        (task, model, mode)
        for task, models in {
            "grammar_acceptability": (PYTHIA_1B_STEP0,),
            "random_fsm": (PYTHIA_1B_STEP0,),
            "hans_nli": (PYTHIA_1B_STEP0, PYTHIA_1B_48K, PYTHIA_1B_96K, PYTHIA_1B),
        }.items()
        for model in models
        for mode in ("standard", "decode-only")
    }
    actual_added = {(spec.task, spec.model, spec.mode) for spec in added}
    return specs


# Public execution interface: exactly four sets.
ALL_EXECUTION_SUITES = ("mean-donor", "6-7b-models", "mean", "checkpoints")
SUITES = {
    "mean-donor": mean_donor_experiments,
    "6-7b-models": large_model_experiments,
    "mean": mean_experiments,
    "checkpoints": checkpoint_experiments,
}


def all_experiments(selected: list[str]) -> list[RunSpec]:
    names = list(ALL_EXECUTION_SUITES) if "all" in selected else selected
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
        help="Validation profile for the configured study table. Default: study-56.",
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument(
        "--skip-if-no-circuit", action="store_true",
        help=("Reuse-only execution policy: require an already cached valid Stage-5 circuit. "
              "If none exists, skip the setting before Stage 5/EAP. This does not alter RunSpec paths."),
    )
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
                run_pipeline(
                    CODE_ROOT, spec, dry_run=args.dry_run,
                    skip_if_no_circuit=args.skip_if_no_circuit,
                )
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
            ]
            if args.skip_if_no_circuit:
                print(
                    "[runner] --skip-if-no-circuit is reuse-only and may intentionally leave "
                    "uncached settings incomplete; final reporting will use its partial-input policy.",
                    flush=True,
                )
            else:
                final_command.append("--require-complete-metrics")
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
