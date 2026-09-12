#!/usr/bin/env python3
"""Generate overtopping figures from a completed results tree.

The module discovers compatible spectral runs and renders competence-versus-causal
summary figures without hard-coded measurements. The all-settings scatter views
use task color, intervention-phase markers, matched-phase connection lines, and
optional OLS summaries. Publication templates can also render phase, checkpoint,
and model-size comparisons from the same discovered data.

Expected inputs include task-level ``dataset_stats.json`` files and run-level
``flip_stats_global.json`` summaries. For the canonical manuscript population, a
missing flip summary is treated as a genuine zero only when Stage 6 can verify a
completed empty candidate set; otherwise the setting is incomplete and is omitted
from partial figures (or rejected in strict mode).

The Pythia checkpoint mapping is explicit: ``pythia-1b@step0`` -> 0,
``pythia-1b@step48000`` -> 48000, ``pythia-1b@step96000`` -> 96000, and
``pythia-1b`` -> 143000. Checkpoint figures include matched zero-candidate states.
"""

from __future__ import annotations
from pathlib import Path
from core.project_paths import PROJECT_ROOT


import argparse
import csv
import math
import re
import textwrap
from dataclasses import asdict, dataclass, replace
from typing import Iterable

import matplotlib

# Reporting is a batch/headless workflow.  Force a non-interactive backend before
# importing pyplot so macOS does not select backend_macosx during CLI runs.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from studies.overtopping.analysis.layer_widths import layer_width_for_model
from studies.overtopping.analysis.lib.files import read_json
from studies.overtopping.analysis.lib.discovery_artifacts import resolve_stage6_dir, stage6_candidate_count
from studies.overtopping.analysis.lib.stats_resolution import resolve_available_stats_dir
from studies.overtopping.analysis.lib.task_metrics import chance_baseline, chance_normalized_score, raw_task_score
from studies.overtopping.experiments.execution import load_run_specs_json
from studies.overtopping.experiments.run_experiments import paper_study_experiments


DEFAULT_OUT = "fig_competence_vs_coverage.pdf"
SCORE_MODE = "phase-specific"


TASK_LABELS = {
    "arithmetic": "Arithmetic",
    "bon_jailbreaking": "Jailbreak",
    "grammar_acceptability": "Grammar",
    "hans_nli": "HANS NLI",
    "random_fsm": "Random FSM",
}

TASK_ORDER = [
    "arithmetic",
    "grammar_acceptability",
    "hans_nli",
    "random_fsm",
    "bon_jailbreaking",
]

# The paper-summary figures below specify which real records to retrieve. They
# do not specify y-values; score and coverage are read from input JSON files.
TASK_FIG_LABELS = {
    "arithmetic": "Arithmetic",
    "bon_jailbreaking": "Jailbreak",
    "grammar_acceptability": "Grammar",
    "hans_nli": "HANS-NLI",
    "random_fsm": "Random FSM",
}

CHECKPOINT_TASKS = [
    ("grammar_acceptability", "Grammar"),
    ("hans_nli", "HANS-NLI"),
    ("random_fsm", "Random FSM"),
]
CHECKPOINT_STEPS = [0, 48000, 96000, 143000]
CHECKPOINT_STEP_LABELS = ["0", "48k", "96k", "143k"]

# Keep these two paper-summary panels on the same canvas so they can be
# placed side by side without manual resizing.
CHECKPOINT_AND_SIZE_FIGSIZE = (5.2, 2.05)

SIZE_COMPARISON_SPECS = [
    dict(task="arithmetic", model="qwen2-1.5b", step=None, label="Qwen2-1.5B\nArithmetic"),
    dict(task="arithmetic", model="qwen2-7b", step=None, label="Qwen2-7B\nArithmetic"),
    dict(task="arithmetic", model="pythia-1b", step=143000, label="Pythia-1B\nArithmetic"),
    dict(task="arithmetic", model="pythia-6.9b", step=None, label="Pythia-6.9B\nArithmetic"),
    dict(task="hans_nli", model="qwen2-1.5b", step=None, label="Qwen2-1.5B\nHANS-NLI"),
    dict(task="hans_nli", model="qwen2-7b", step=None, label="Qwen2-7B\nHANS-NLI"),
    dict(task="hans_nli", model="pythia-1b", step=143000, label="Pythia-1B\nHANS-NLI"),
    dict(task="hans_nli", model="pythia-6.9b", step=None, label="Pythia-6.9B\nHANS-NLI"),
    dict(task="bon_jailbreaking", model="qwen2-1.5b", step=None, label="Qwen2-1.5B\nJailbreak"),
    dict(task="bon_jailbreaking", model="qwen2-7b", step=None, label="Qwen2-7B\nJailbreak"),
]

COMPACT_POINTS = [
    ("arithmetic", "Qwen", "Qwen2.5-1.5B-Instruct", "decode-only", "mean-donor"),
    ("bon_jailbreaking", "Qwen", "Qwen2.5-1.5B-Instruct", "decode-only", "mean-donor"),
    ("grammar_acceptability", "Qwen", "Qwen2.5-1.5B-Instruct", "decode-only", "mean-donor"),
    ("grammar_acceptability", "Qwen", "Qwen2.5-1.5B-Instruct", "input+output", "mean-donor"),
    ("hans_nli", "Qwen", "Qwen2.5-1.5B-Instruct", "decode-only", "mean-donor"),
    ("hans_nli", "Qwen", "Qwen2.5-1.5B-Instruct", "input+output", "mean-donor"),
    ("random_fsm", "Qwen", "Qwen2.5-1.5B-Instruct", "input+output", "mean-donor"),
]


@dataclass
class PlotPoint:
    task: str
    task_label: str
    org: str
    model: str
    phase: str
    baseline: str
    anchor: str
    run: str
    score: float
    union_rate: float
    n_neurons: int
    n_eval: int | None
    status: str
    source_path: str


@dataclass
class LabelGroup:
    points: list[PlotPoint]
    text: str
    x: float
    y: float


def coverage_metric_label(metric: str) -> str:
    labels = {
        "pooled": r"Overtopping coverage $U(J)$",
        "i2c": r"Directional causal reach $U_{0\to1}(J)$",
        "c2i": r"Directional causal reach $U_{1\to0}(J)$",
        "n05-i2c-density": r"High-effect density $D^{0\to1}_{.05}=N_{.05}/d_{\rm layer}$",
        "n05-c2i-density": r"High-effect density $D^{1\to0}_{.05}=N_{.05}/d_{\rm layer}$",
        "n05-i2c-per-1k": r"High-effect density $D^{0\to1}_{.05}$ (per 1000 layer coordinates)",
        "n05-c2i-per-1k": r"High-effect density $D^{1\to0}_{.05}$ (per 1000 layer coordinates)",
    }
    return labels.get(str(metric), r"Overtopping coverage $U(J)$")


def checkpoint_metric_label(metric: str) -> str:
    return {
        "pooled": r"$U(J)$",
        "i2c": r"$U_{0\to1}(J)$",
        "c2i": r"$U_{1\to0}(J)$",
        "n05-i2c-density": r"$D^{0\to1}_{.05}$",
        "n05-c2i-density": r"$D^{1\to0}_{.05}$",
        "n05-i2c-per-1k": r"$D^{0\to1}_{.05}$",
        "n05-c2i-per-1k": r"$D^{1\to0}_{.05}$",
    }.get(str(metric), r"$U(J)$")


def _run_dir_from_point(root: Path, point: PlotPoint) -> Path:
    return (root / point.source_path).resolve()


def _directional_rate_from_global(payload: dict, direction: str) -> float:
    if direction == "i2c":
        rate_key, count_key, denom_key = "union_i2c_unique_rate", "union_i2c_unique_count", "n_evaluated_i2c_rows"
    else:
        rate_key, count_key, denom_key = "union_c2i_unique_rate", "union_c2i_unique_count", "n_evaluated_c2i_rows"
    value = payload.get(rate_key)
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    try:
        count = float(payload.get(count_key))
        denom = float(payload.get(denom_key))
        return count / denom if denom > 0 else math.nan
    except (TypeError, ValueError):
        return math.nan


def _directional_n05_density(run_dir: Path, model: str, direction: str, *, per_1000: bool = False) -> float:
    global_path = run_dir / "flip_stats_global.json"
    by_neuron_path = run_dir / "flip_stats_by_neuron.csv"
    if not global_path.is_file():
        # A materialized run directory with no flip-statistics file represents
        # an empty candidate set in the configured manuscript pipeline: U(empty)=0
        # and N_t(empty)=0. Preserve that zero rather than dropping the point.
        return 0.0
    payload = read_json(global_path)
    denom_key = "n_evaluated_i2c_rows" if direction == "i2c" else "n_evaluated_c2i_rows"
    try:
        denom = float(payload.get(denom_key))
    except (TypeError, ValueError):
        denom = math.nan
    if not math.isfinite(denom) or denom <= 0:
        return math.nan
    width = layer_width_for_model(model)
    if width is None or int(width) <= 0:
        return math.nan
    if not by_neuron_path.is_file():
        # If the global file says there are no candidates, the density is exactly zero.
        try:
            if int(payload.get("n_neurons", 0)) == 0:
                return 0.0
        except Exception:
            pass
        return math.nan
    count_key = "i2c_count" if direction == "i2c" else "c2i_count"
    n_high = 0
    with by_neuron_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                count = float(row.get(count_key, "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(count) and count / denom >= 0.05:
                n_high += 1
    scale = 1000.0 if per_1000 else 1.0
    return scale * float(n_high) / float(width)


def transform_points_for_coverage_metric(root: Path, points: list[PlotPoint], metric: str) -> list[PlotPoint]:
    """Reuse the shared visual engine with a different causal y quantity.

    Crucially, this does *not* filter settings by their within-setting directional
    denominator. The regression n remains the number of plotted settings.
    """
    metric = str(metric)
    if metric == "pooled":
        return list(points)
    out: list[PlotPoint] = []
    for point in points:
        run_dir = _run_dir_from_point(root, point)
        if metric in {"i2c", "c2i"}:
            global_path = run_dir / "flip_stats_global.json"
            if global_path.is_file():
                value = _directional_rate_from_global(read_json(global_path), metric)
            else:
                # Existing run with no discovered agonists: the union of an empty
                # candidate set is exactly zero in either direction.
                value = 0.0 if point.status in {"verified-zero-candidates", "empty-no-agonists"} else math.nan
        elif metric == "n05-i2c-density":
            value = _directional_n05_density(run_dir, point.model, "i2c")
        elif metric == "n05-c2i-density":
            value = _directional_n05_density(run_dir, point.model, "c2i")
        elif metric == "n05-i2c-per-1k":
            value = _directional_n05_density(run_dir, point.model, "i2c", per_1000=True)
        elif metric == "n05-c2i-per-1k":
            value = _directional_n05_density(run_dir, point.model, "c2i", per_1000=True)
        else:
            raise ValueError(f"unknown coverage metric: {metric}")
        out.append(replace(point, union_rate=float(value)))
    return out


@dataclass
class Filters:
    tasks: set[str] | None
    orgs: set[str] | None
    model_regex: re.Pattern[str] | None
    phases: set[str] | None
    baselines: set[str] | None
    anchor: str
    name_contains: list[str]
    exclude_fake_targets: bool
    exclude_checkpoints: bool
    require_m: str | None
    require_tau: str | None
    prefer_m: str | None
    prefer_tau: str | None
    include_empty: bool


def rq1_manuscript_filters() -> Filters:
    """Return generic filters for non-manifest RQ1-style discovery.

    Manuscript Figure 2 no longer relies on these filters to define its sample;
    it resolves the explicit canonical experiment catalogue through
    :func:`discover_rq1_manuscript_points`.
    """
    return Filters(
        tasks=set(TASK_ORDER),
        orgs=None,
        model_regex=None,
        phases=None,
        baselines=None,
        anchor="random_anchor",
        name_contains=["spectral"],
        exclude_fake_targets=True,
        exclude_checkpoints=False,
        require_m=None,
        require_tau=None,
        prefer_m="200000",
        prefer_tau="0.3",
        include_empty=True,
    )



def locate_results_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"results directory does not exist: {path}")

    expected_tasks = set(TASK_LABELS)
    candidates = [path, path / "results", path / "results" / "results"]
    for candidate in candidates:
        if candidate.is_dir():
            names = {p.name for p in candidate.iterdir() if p.is_dir()}
            if names & expected_tasks:
                return candidate

    matches: list[Path] = []
    for candidate in path.rglob("*"):
        if candidate.is_dir():
            names = {p.name for p in candidate.iterdir() if p.is_dir()}
            if names & expected_tasks:
                matches.append(candidate)
    if matches:
        return min(matches, key=lambda p: len(p.parts))

    raise FileNotFoundError(
        f"could not find task folders such as {sorted(expected_tasks)} under {path}"
    )



def downstream_score(task: str, dataset_stats: dict, phase: str | None = None, score_mode: str | None = None) -> float:
    """Return the x-axis score used by competence/coverage figures.

    ``raw`` uses raw task score; ``chance-normalized`` uses chance-corrected
    score for finite-answer tasks. ``phase-specific`` chance-corrects only
    output-only finite-answer points: prompt/instruction processing is upstream
    of an output-only intervention, so a model may have followed the instruction
    but fail to solve the task and obtain some successes by guessing. Input+output
    interventions also perturb prompt processing, so their competence axis uses
    raw success. Jailbreak always uses safe/refusal rate.
    """
    mode = score_mode or SCORE_MODE
    if mode not in {"raw", "chance-normalized", "phase-specific"}:
        raise ValueError(f"unknown score mode: {mode}")
    raw = raw_task_score(task, dataset_stats)
    if task == "bon_jailbreaking":
        return raw
    if mode == "raw":
        return raw
    if mode == "chance-normalized":
        return chance_normalized_score(raw, chance_baseline(task, dataset_stats))
    if mode == "phase-specific":
        if phase == "decode-only":
            return chance_normalized_score(raw, chance_baseline(task, dataset_stats))
        return raw


def run_phase(run_name: str) -> str:
    return "decode-only" if "decode_only" in run_name else "input+output"


def run_baseline(run_name: str) -> str:
    return "mean-donor" if "eval_mean-donor" in run_name else "mean"


def run_anchor(run_name: str) -> str:
    if "random_anchor" in run_name:
        return "random_anchor"
    if "spectral_anchor" in run_name:
        return "spectral_anchor"
    return "unspecified"


def token_match(run_name: str, prefix: str, value: str | None) -> bool:
    """Match a run-name token exactly at ``-``/``_`` delimiters.

    Substring matching is scientifically unsafe here: ``tau0.3`` must not
    match ``tau0.35`` and ``M200000`` must not match ``M2000000``.
    """
    if value is None or str(value).lower() in {"any", "none", ""}:
        return True
    value = str(value).strip()
    token = value if value.startswith(prefix) else prefix + value
    return re.search(rf"(?:^|[-_]){re.escape(token)}(?=$|[-_])", str(run_name)) is not None


_DERIVED_EVALUATION_RUN_RE = re.compile(r"-(?:heldout_test|eval_train|eval_all)(?:-cap\d+)?$")


def is_derived_evaluation_run(run_name: str) -> bool:
    """Return whether *run_name* is an evaluation-side derivative, not a setting."""
    return _DERIVED_EVALUATION_RUN_RE.search(str(run_name)) is not None


def canonical_experimental_run_name(run_name: str) -> str:
    """Collapse evaluation-side derivatives onto their experimental setting.

    Split-aware Stage 7 now materializes primary test statistics under
    ``*-heldout_test``.  Older trees may additionally retain the unsuffixed
    pre-split directory.  They are two evaluation products of one setting, not
    two manuscript observations.
    """
    return _DERIVED_EVALUATION_RUN_RE.sub("", str(run_name))


def evaluation_variant_priority(run_dir: Path) -> tuple[int, int, int]:
    """Prefer strict held-out-test artifacts without duplicating settings."""
    name = run_dir.name
    has_global = int((run_dir / "flip_stats_global.json").is_file())
    if re.search(r"-heldout_test$", name):
        variant = 40
    elif re.search(r"-heldout_test-cap\d+$", name):
        variant = 35
    elif not is_derived_evaluation_run(name):
        variant = 30
    elif re.search(r"-eval_all(?:-cap\d+)?$", name):
        variant = 20
    else:  # eval_train
        variant = 10
    # Prefer a materialized flip summary within the same evaluation class, but
    # do not let an unsuffixed training/full-data result outrank the
    # primary held-out-test evaluation merely because it is non-empty.
    return (variant, has_global, -len(name))


def matches_filters(run_name: str, filters: Filters) -> bool:
    if filters.exclude_fake_targets and "fake_targets" in run_name:
        return False
    if filters.name_contains and not all(tok in run_name for tok in filters.name_contains):
        return False
    if filters.anchor != "any" and run_anchor(run_name) != filters.anchor:
        return False
    if filters.phases is not None and run_phase(run_name) not in filters.phases:
        return False
    if filters.baselines is not None and run_baseline(run_name) not in filters.baselines:
        return False
    if not token_match(run_name, "M", filters.require_m):
        return False
    if not token_match(run_name, "tau", filters.require_tau):
        return False
    return True


def model_short(model: str) -> str:
    s = model
    s = s.replace("Qwen2.5-", "Qwen2.5 ")
    s = s.replace("Qwen2-", "Qwen2 ")
    s = s.replace("-Instruct", "")
    s = s.replace("pythia-", "Pythia ")
    s = s.replace("@step", "@")
    return s


def task_model_dir_iter(root: Path) -> Iterable[tuple[str, str, str, Path]]:
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("__")):
        task = task_dir.name
        for org_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            org = org_dir.name
            for model_dir in sorted(p for p in org_dir.iterdir() if p.is_dir()):
                yield task, org, model_dir.name, model_dir


def run_priority(p: PlotPoint, filters: Filters) -> tuple:
    run = p.run
    prefer_m = 1 if token_match(run, "M", filters.prefer_m) else 0
    prefer_tau = 1 if token_match(run, "tau", filters.prefer_tau) else 0
    fast = 1 if "fast" in run else 0
    random_anchor = 1 if p.anchor == "random_anchor" else 0
    nonempty = 1 if p.status == "ok" else 0
    literal_spectral_sample = 1 if "spectral_sample" in run else 0
    fake_penalty = -1 if "fake_targets" in run else 0
    return (
        prefer_m,
        prefer_tau,
        nonempty,
        random_anchor,
        fast,
        literal_spectral_sample,
        fake_penalty,
        -len(run),
        run,
    )


def make_point(
    task: str,
    org: str,
    model: str,
    model_dir: Path,
    run_dir: Path,
    dataset_stats: dict,
    status: str,
) -> PlotPoint:
    run_name = run_dir.name
    global_path = run_dir / "flip_stats_global.json"
    if global_path.exists():
        gs = read_json(global_path)
        union_rate = float(gs.get("union_flip_any_unique_rate", 0.0))
        n_neurons = int(gs.get("n_neurons", 0))
        n_eval = gs.get("n_evaluated_rows")
        status = "ok"
    else:
        union_rate = 0.0
        n_neurons = 0
        n_eval = None
        status = status or "empty-no-agonists"

    phase = run_phase(run_name)

    return PlotPoint(
        task=task,
        task_label=TASK_LABELS.get(task, task),
        org=org,
        model=model,
        phase=phase,
        baseline=run_baseline(run_name),
        anchor=run_anchor(run_name),
        run=run_name,
        score=downstream_score(task, dataset_stats, phase=phase),
        union_rate=union_rate,
        n_neurons=n_neurons,
        n_eval=n_eval,
        status=status,
        source_path=str(run_dir.relative_to(model_dir.parents[2])),
    )


def discover_points(
    root: Path,
    filters: Filters,
    dedupe: bool,
    *,
    heldout_test_only: bool = False,
) -> list[PlotPoint]:
    points: list[PlotPoint] = []
    for task, org, model, model_dir in task_model_dir_iter(root):
        if filters.tasks is not None and task not in filters.tasks:
            continue
        if filters.orgs is not None and org not in filters.orgs:
            continue
        if filters.model_regex is not None and not filters.model_regex.search(model):
            continue
        if filters.exclude_checkpoints and "@step" in model:
            continue

        ds_path = model_dir / "feature_report" / "dataset_stats.json"
        if not ds_path.exists():
            continue
        dataset_stats = read_json(ds_path)

        stats_dir = model_dir / "rule_extraction_results" / "neuron_flip_rules" / "stats"
        if not stats_dir.is_dir():
            continue

        matching_run_dirs = [
            p for p in sorted(stats_dir.iterdir())
            if p.is_dir() and matches_filters(p.name, filters)
            and (not heldout_test_only or bool(re.search(r"-heldout_test(?:-cap\d+)?$", p.name)))
        ]
        # Choose exactly one evaluation product per experimental setting.
        # Manuscript RQ1 requests strict held-out-test artifacts; generic plots
        # may still inspect train/all evaluation products explicitly.
        by_experimental_run: dict[str, list[Path]] = {}
        for run_dir in matching_run_dirs:
            by_experimental_run.setdefault(
                canonical_experimental_run_name(run_dir.name), []
            ).append(run_dir)
        selected_run_dirs = [
            max(group, key=evaluation_variant_priority)
            for group in by_experimental_run.values()
        ]

        for run_dir in sorted(selected_run_dirs):
            global_path = run_dir / "flip_stats_global.json"
            if not global_path.exists() and not filters.include_empty:
                continue
            points.append(make_point(task, org, model, model_dir, run_dir, dataset_stats, "empty-no-agonists"))

    if dedupe:
        best: dict[tuple[str, str, str, str, str], PlotPoint] = {}
        for p in points:
            key = (p.task, p.org, p.model, p.phase, p.baseline)
            if key not in best or run_priority(p, filters) > run_priority(best[key], filters):
                best[key] = p
        points = sorted(best.values(), key=lambda p: (TASK_ORDER.index(p.task) if p.task in TASK_ORDER else 99, p.org, p.model, p.baseline, p.phase))

    return points


def rq1_manuscript_specs(catalogue_json: Path | None = None):
    """Return the explicit Figure-2 population.

    Figure 2 is defined over every currently configured overtopping setting. The
    population is resolved from the registry rather than from a filesystem scan,
    so configured zero-candidate settings and missing artifacts retain distinct
    statuses. No exact experiment count is assumed.
    """
    specs = list(load_run_specs_json(catalogue_json) if catalogue_json is not None else paper_study_experiments())
    identities = [
        (s.task, s.model, s.mode, s.intervention, s.z_thresh, s.circuit_level,
         s.circuit_size, s.min_flip_rate, s.max_circuits, s.mlp_neurons_only,
         s.no_llm_feature_generation, s.evaluation_split)
        for s in specs
    ]
    if len(specs) != len(set(identities)):
        raise RuntimeError(
            f"RQ1 configured population contains duplicate scientific identities: "
            f"{len(specs)} rows / {len(set(identities))} unique identities"
        )
    return specs


def discover_rq1_manuscript_points(root: Path, *, allow_incomplete: bool = False, catalogue_json: Path | None = None) -> list[PlotPoint]:
    """Resolve the exact canonical strict-heldout Figure-2 population.

    Paths are obtained from :class:`RunSpec` itself. The canonical full held-out
    directory is preferred, but in best-effort mode a compatible materialized
    ``-heldout_test-capN`` stats directory is used rather than discarding real
    partial results. A run with no ``flip_stats_global.json`` is retained as zero
    only when Stage 6 verifies a completed empty candidate set.
    """
    specs = rq1_manuscript_specs(catalogue_json=catalogue_json)
    points: list[PlotPoint] = []
    unavailable: list[str] = []
    partial: list[str] = []
    for spec in specs:
        expected_run_dir = spec.stats_dir(root)
        run_dir, stats_resolution = resolve_available_stats_dir(expected_run_dir)
        model_dir = root / spec.task / Path(spec.model)
        ds_path = model_dir / "feature_report" / "dataset_stats.json"
        if not ds_path.is_file():
            unavailable.append(f"dataset stats unavailable: {ds_path}")
            continue
        dataset_stats = read_json(ds_path)
        global_path = run_dir / "flip_stats_global.json"
        status = "ok"
        if global_path.is_file():
            if stats_resolution == "partial_heldout_cap":
                status = "partial-heldout-stats"
                partial.append(f"{expected_run_dir} -> {run_dir}")
        else:
            # Discovery is keyed by the configured/reference identity, not by a
            # possibly capped evaluation-side directory. It can independently
            # establish a genuine empty candidate set.
            stage6_dir = resolve_stage6_dir(expected_run_dir)
            n_candidates, stage6_status = stage6_candidate_count(stage6_dir)
            if stage6_status == "ok" and n_candidates == 0:
                status = "verified-zero-candidates"
                run_dir = expected_run_dir
            else:
                detail = (
                    f"overtopping stats unavailable for configured setting: {expected_run_dir} "
                    f"(stats_resolution={stats_resolution}, stage6_status={stage6_status}, candidates={n_candidates})"
                )
                unavailable.append(detail)
                continue
        point = make_point(
            spec.task,
            Path(spec.model).parts[0],
            Path(spec.model).parts[-1],
            model_dir,
            run_dir,
            dataset_stats,
            status,
        )
        # make_point marks any materialized global summary as ``ok``. Preserve
        # the fact that this point came from a capped/partial held-out result.
        if status == "partial-heldout-stats":
            point.status = status
        points.append(point)

    issues = unavailable + [f"partial held-out stats: {item}" for item in partial]
    if issues:
        detail = "\n".join(f"  - {item}" for item in issues[:20])
        message = (
            f"RQ1 manuscript population does not have full canonical coverage for all {len(specs)} configured settings. "
            "Available partial stats are retained in best-effort figures.\n" + detail
        )
        if not allow_incomplete:
            raise RuntimeError(message + "\nStrict mode requires the full configured held-out evaluation.")
        print("[WARNING] " + message.replace("\n", " | "))

    phase_counts = {
        phase: sum(p.phase == phase for p in points)
        for phase in ("input+output", "decode-only")
    }
    expected_phase_counts = {
        "input+output": sum(not spec.decode_only for spec in specs),
        "decode-only": sum(spec.decode_only for spec in specs),
    }
    if len(points) != len(specs) or phase_counts != expected_phase_counts:
        message = (
            "RQ1 manuscript population resolved incompletely: "
            f"n={len(points)}, phase_counts={phase_counts}; configured registry expects "
            f"n={len(specs)}, phase_counts={expected_phase_counts}"
        )
        if not allow_incomplete:
            raise RuntimeError(message)
        print(f"[WARNING] {message}")
    return points


def compact_points(root: Path, filters: Filters) -> list[PlotPoint]:
    points: list[PlotPoint] = []
    compact_filters = Filters(
        tasks=None,
        orgs=None,
        model_regex=None,
        phases=None,
        baselines=None,
        anchor=filters.anchor,
        name_contains=filters.name_contains,
        exclude_fake_targets=filters.exclude_fake_targets,
        exclude_checkpoints=filters.exclude_checkpoints,
        require_m=filters.require_m,
        require_tau=filters.require_tau,
        prefer_m=filters.prefer_m,
        prefer_tau=filters.prefer_tau,
        include_empty=filters.include_empty,
    )

    all_points = discover_points(root, compact_filters, dedupe=False)
    for task, org, model, phase, baseline in COMPACT_POINTS:
        candidates = [
            p for p in all_points
            if p.task == task and p.org == org and p.model == model and p.phase == phase and p.baseline == baseline
        ]
        if not candidates:
            print(f"[skip] compact point not found: {task}/{org}/{model} {phase} {baseline}")
            continue
        points.append(max(candidates, key=lambda p: run_priority(p, compact_filters)))
    return points


def point_pair_key(point: PlotPoint) -> tuple[str, str, str, str]:
    return (point.task, point.org, point.model, point.baseline)


def task_color_map(points: list[PlotPoint]) -> dict[str, object]:
    cmap = plt.get_cmap("tab10")
    tasks = [t for t in TASK_ORDER if any(p.task == t for p in points)]
    tasks += sorted({p.task for p in points if p.task not in tasks})
    return {task: cmap(i % 10) for i, task in enumerate(tasks)}


def marker_for_phase(phase: str) -> str:
    return "o" if phase == "decode-only" else "s"


def edge_for_baseline(baseline: str) -> str:
    return "black" if baseline == "mean-donor" else "0.55"


def phase_label(phase: str) -> str:
    """Readable figure notation for intervention phase."""
    if phase == "decode-only":
        return "Output-only"
    if phase == "input+output":
        return "Input+output"
    return phase


def x_axis_label(points: list[PlotPoint]) -> str:
    """Short readable x-axis label matching the selected score definition."""
    if SCORE_MODE == "raw":
        return "Raw task score"
    if SCORE_MODE == "chance-normalized":
        return r"Chance-normalized score $\kappa$"

    phases = {p.phase for p in points}
    if phases == {"decode-only"}:
        return r"Chance-normalized score $\kappa$"
    if phases == {"input+output"}:
        return "Raw task score"
    return "Phase-specific behavior score"


def size_for_point(point: PlotPoint, args: argparse.Namespace) -> float:
    if args.size_mode == "neurons":
        # Area is in pt^2. Keep neuron-scaled points usable in dense paper figures.
        return min(args.marker_max, args.marker_size + args.marker_neuron_scale * math.sqrt(max(point.n_neurons, 1)))
    return args.marker_size


def draw_pair_lines(ax, points: list[PlotPoint], colors: dict[str, object], connect: str) -> None:
    if connect == "none":
        return
    grouped: dict[tuple[str, str, str, str], list[PlotPoint]] = {}
    for p in points:
        grouped.setdefault(point_pair_key(p), []).append(p)
    for group in grouped.values():
        decode = [p for p in group if p.phase == "decode-only"]
        io = [p for p in group if p.phase == "input+output"]
        if not decode or not io:
            continue
        if connect == "mean-donor" and decode[0].baseline != "mean-donor":
            continue
        p1 = decode[0]
        p2 = io[0]
        ax.plot(
            [p1.score, p2.score],
            [p1.union_rate, p2.union_rate],
            "-",
            color=colors.get(p1.task, "0.65"),
            linewidth=0.45,
            alpha=0.28,
            zorder=1,
        )


def regression_fit(points: list[PlotPoint]) -> tuple[float, float, float] | None:
    if len(points) < 2:
        return None
    xs = np.array([p.score for p in points], dtype=float)
    ys = np.array([p.union_rate for p in points], dtype=float)
    if np.allclose(xs, xs[0]):
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    y_hat = slope * xs + intercept
    ss_res = float(np.sum((ys - y_hat) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def _finite_xy(points: list[PlotPoint]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([p.score for p in points], dtype=float)
    ys = np.array([p.union_rate for p in points], dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    return xs[keep], ys[keep]


def _p_text(p_value: float | None) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<.001"
    if p_value < 0.01:
        return f"={p_value:.3f}".replace("0.", ".")
    return f"={p_value:.2f}".replace("0.", ".")


def regression_stats(points: list[PlotPoint]) -> dict[str, float] | None:
    xs, ys = _finite_xy(points)
    n = int(len(xs))
    if n < 2 or np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return None

    fit = regression_fit([p for p in points if np.isfinite(p.score) and np.isfinite(p.union_rate)])
    if fit is None:
        return None
    slope, intercept, r2 = fit

    pearson_r = float(np.corrcoef(xs, ys)[0, 1])
    pearson_p = math.nan
    spearman_rho = math.nan
    spearman_p = math.nan
    try:
        from scipy import stats as scipy_stats  # type: ignore

        lr = scipy_stats.linregress(xs, ys)
        pearson_p = float(lr.pvalue)
        slope = float(lr.slope)
        intercept = float(lr.intercept)
        sr = scipy_stats.spearmanr(xs, ys)
        spearman_rho = float(sr.statistic)
        spearman_p = float(sr.pvalue)
    except Exception:
        pass

    return dict(
        n=float(n),
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
    )


def fit_stats_text(points: list[PlotPoint]) -> str | None:
    stats = regression_stats(points)
    if stats is None:
        return None
    n_defined = int(stats["n"])
    n_settings = sum(1 for p in points if np.isfinite(p.score))
    n_text = f"n={n_defined}" if n_defined == n_settings else f"n={n_defined}/{n_settings} defined"
    return (
        f"{n_text}, Pearson r={stats['pearson_r']:.2f},\n"
        f"p{_p_text(stats['pearson_p'])}, OLS R²={stats['r2']:.2f}"
    )


def _marker_obstacles(ax, points: list[PlotPoint], args: argparse.Namespace | None = None, extra_px: float = 2.0) -> list[Bbox]:
    """Approximate marker footprints in display coordinates.

    Matplotlib scatter sizes are specified in pt^2. The radius estimate is
    intentionally conservative because these boxes are used for readability
    decisions: trend lines and statistics boxes should leave visible air around
    the plotted points rather than merely avoid the mathematical center.
    """
    obstacles: list[Bbox] = []
    fig = ax.figure
    for p in points:
        if not (np.isfinite(p.score) and np.isfinite(p.union_rate)):
            continue
        xpix, ypix = ax.transData.transform((p.score, p.union_rate))
        area = size_for_point(p, args) if args is not None else 24.0
        radius_pt = math.sqrt(max(float(area), 1.0)) / 1.45
        radius_px = max(6.0, radius_pt * fig.dpi / 72.0 + extra_px)
        obstacles.append(Bbox.from_extents(xpix - radius_px, ypix - radius_px, xpix + radius_px, ypix + radius_px))
    return obstacles


def _masked_line_segments(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    points: list[PlotPoint],
    args: argparse.Namespace | None = None,
    gap_px: float = 4.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a trend line into visible segments that do not pass through markers."""
    if len(xs) < 2:
        return []
    mask = np.isfinite(xs) & np.isfinite(ys)
    if points:
        samples_px = ax.transData.transform(np.column_stack([xs, ys]))
        for obstacle in _marker_obstacles(ax, points, args, extra_px=gap_px):
            in_x = (samples_px[:, 0] >= obstacle.x0) & (samples_px[:, 0] <= obstacle.x1)
            in_y = (samples_px[:, 1] >= obstacle.y0) & (samples_px[:, 1] <= obstacle.y1)
            mask &= ~(in_x & in_y)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start: int | None = None
    for i, keep in enumerate(mask):
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            if i - start >= 2:
                segments.append((xs[start:i], ys[start:i]))
            start = None
    if start is not None and len(xs) - start >= 2:
        segments.append((xs[start:], ys[start:]))
    return segments


def annotate_fit_stats(ax, points: list[PlotPoint], args: argparse.Namespace) -> None:
    if getattr(args, "no_fit_stats", False) or args.trend == "none":
        return
    text = fit_stats_text(points)
    if not text:
        return

    fig = ax.figure
    bbox_kwargs = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.80", linewidth=0.42, alpha=0.88)

    # Deterministic placement for the OLS statistics box.
    # The current figure has clear unused space in the upper-left panel.
    x, y, ha, va = 0.035, 0.955, "left", "top"

    ann = ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=getattr(args, "fit_stats_size", 5.6),
        linespacing=1.05,
        bbox=bbox_kwargs,
        zorder=8,
        clip_on=False,
    )
    # ann.set_clip_on(True)
    ann.set_clip_path(ax.patch)
    # Expose the final regression-statistics box as a hard obstacle so that
    # model labels are never placed on top of it.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box_obstacles = list(getattr(ax, "_readability_obstacle_bboxes", []))
    obstacle_pad = float(getattr(args, "fit_stats_obstacle_pad_px", 9.0))
    box_obstacles.append(_text_artist_obstacle_bbox(ann, renderer, obstacle_pad))
    setattr(ax, "_readability_obstacle_bboxes", box_obstacles)


def draw_trend(
    ax,
    points: list[PlotPoint],
    mode: str,
    x_min: float = 0.0,
    x_max: float = 1.0,
    show_r2: bool = False,
    args: argparse.Namespace | None = None,
) -> list[Line2D]:
    handles: list[Line2D] = []
    if mode == "none":
        return handles

    mask_points = True if args is None else not getattr(args, "no_trend_mask_points", False)
    gap_px = 4.5 if args is None else float(getattr(args, "trend_mask_gap_px", 4.5))

    def draw_one(subset: list[PlotPoint], label: str, color: str, linestyle: str) -> None:
        fit = regression_fit(subset)
        if fit is None:
            return
        slope, intercept, r2 = fit
        xs = np.linspace(x_min, x_max, 500)
        ys = slope * xs + intercept
        segments = _masked_line_segments(ax, xs, ys, points, args, gap_px=gap_px) if mask_points else [(xs, ys)]
        for seg_xs, seg_ys in segments:
            ax.plot(seg_xs, seg_ys, linestyle=linestyle, color=color, linewidth=0.8, alpha=0.86, zorder=2)
        legend_label = f"{label} score-coverage fit ($R^2$={r2:.2f})" if show_r2 else ("score-coverage fit" if label == "overall" else f"{phase_label(label)} fit")
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=0.8, label=legend_label))

    if mode == "overall":
        draw_one(points, "overall", "0.15", "--")
    elif mode == "by-phase":
        draw_one([p for p in points if p.phase == "decode-only"], "decode-only", "0.20", "--")
        draw_one([p for p in points if p.phase == "input+output"], "input+output", "0.55", "-.")
    return handles


def label_text(point: PlotPoint, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "task":
        return point.task_label
    if mode == "model":
        return model_short(point.model)
    if mode == "task+model":
        return f"{point.task_label}\n{model_short(point.model)}"
    return f"{point.task_label}\n{model_short(point.model)}"


def diverse_subset(points: list[PlotPoint], k: int) -> list[PlotPoint]:
    if k <= 0 or not points:
        return []
    if len(points) <= k:
        return list(points)

    xs = np.array([p.score for p in points], dtype=float)
    ys = np.array([p.union_rate for p in points], dtype=float)
    x_std = float(np.std(xs)) or 1.0
    y_std = float(np.std(ys)) or 1.0
    coords = np.column_stack(((xs - np.mean(xs)) / x_std, (ys - np.mean(ys)) / y_std))

    norms = np.sum(coords ** 2, axis=1)
    selected = [int(np.argmax(norms))]
    remaining = set(range(len(points))) - set(selected)
    while remaining and len(selected) < k:
        best_i = None
        best_d = None
        for i in remaining:
            d = min(float(np.sum((coords[i] - coords[j]) ** 2)) for j in selected)
            if best_d is None or d > best_d:
                best_d = d
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)
    return [points[i] for i in selected]


def candidate_labeled_points(points: list[PlotPoint], label_points: str) -> list[PlotPoint]:
    if label_points == "none":
        return []
    if label_points == "all":
        return list(points)
    if label_points == "auto":
        return list(points) if len(points) <= 8 else []
    if label_points in {"input+output", "decode-only"}:
        return [p for p in points if p.phase == label_points]
    if label_points == "paired":
        out: list[PlotPoint] = []
        grouped: dict[tuple[str, str, str, str], list[PlotPoint]] = {}
        for p in points:
            grouped.setdefault(point_pair_key(p), []).append(p)
        for group in grouped.values():
            io = [p for p in group if p.phase == "input+output"]
            out.append(io[0] if io else group[0])
        return out
    return []


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def aggregate_text(points: list[PlotPoint], label_mode: str) -> str:
    # For model labels, keep exactly one model name per line. This avoids the
    # visually ambiguous failure mode where two independent model labels overlap
    # and appear as one horizontal string, e.g. "Qwen2.5 1.5B Pythia 1b".
    if label_mode in {"auto", "model"}:
        labels = unique_in_order([model_short(p.model) for p in points])
    elif label_mode == "task":
        labels = unique_in_order([p.task_label for p in points])
    elif label_mode == "task+model":
        labels = unique_in_order([f"{p.task_label}: {model_short(p.model)}" for p in points])
    else:
        labels = []
    return "\n".join(labels)


def cluster_label_points(ax, points: list[PlotPoint], label_mode: str, aggregate: bool, cluster_px: float) -> list[LabelGroup]:
    if not points:
        return []
    if not aggregate:
        return [LabelGroup(points=[p], text=label_text(p, label_mode), x=p.score, y=p.union_rate) for p in points]

    pix = np.array([ax.transData.transform((p.score, p.union_rate)) for p in points], dtype=float)
    n = len(points)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(pix[i] - pix[j])) <= cluster_px:
                union(i, j)

    buckets: dict[int, list[PlotPoint]] = {}
    for i, point in enumerate(points):
        buckets.setdefault(find(i), []).append(point)

    groups: list[LabelGroup] = []
    for group_points in buckets.values():
        text = aggregate_text(group_points, label_mode)
        if not text:
            continue
        x = float(np.mean([p.score for p in group_points]))
        y = float(np.mean([p.union_rate for p in group_points]))
        groups.append(LabelGroup(points=group_points, text=text, x=x, y=y))
    return sorted(groups, key=lambda g: (g.x, g.y, g.text))


def diverse_groups(groups: list[LabelGroup], k: int) -> list[LabelGroup]:
    if k <= 0 or not groups:
        return []
    if len(groups) <= k:
        return groups
    pseudo = [PlotPoint(
        task="", task_label="", org="", model=g.text, phase="", baseline="", anchor="", run="",
        score=g.x, union_rate=g.y, n_neurons=1, n_eval=None, status="", source_path=""
    ) for g in groups]
    selected_points = diverse_subset(pseudo, k)
    selected_coords = {(p.score, p.union_rate, p.model) for p in selected_points}
    return [g for g in groups if (g.x, g.y, g.text) in selected_coords]


def _expanded_bbox(bb: Bbox, pad_px: float) -> Bbox:
    return Bbox.from_extents(bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px)


def _text_artist_obstacle_bbox(artist, renderer, pad_px: float) -> Bbox:
    """Display-space obstacle box for text plus its bbox patch.

    Text.get_window_extent() can understate the visible footprint when a text
    object has a rounded bbox. Use the union of the text, tight bbox, and bbox
    patch extents, then add explicit clearance. This makes the regression
    annotation box a real obstacle for later model-label placement.
    """
    boxes: list[Bbox] = []
    for getter in ("get_window_extent", "get_tightbbox"):
        try:
            bb = getattr(artist, getter)(renderer)
            if bb is not None:
                boxes.append(bb)
        except Exception:
            pass

    try:
        patch = artist.get_bbox_patch()
        if patch is not None:
            bb = patch.get_window_extent(renderer)
            if bb is not None:
                boxes.append(bb)
    except Exception:
        pass

    if not boxes:
        return Bbox.from_extents(0, 0, 0, 0)
    return _expanded_bbox(Bbox.union(boxes), pad_px)


def _bbox_overlap_area(a: Bbox, b: Bbox) -> float:
    if not a.overlaps(b):
        return 0.0
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    return max(1.0, ix1 - ix0) * max(1.0, iy1 - iy0)


def _bbox_outside_distance(inner: Bbox, outer: Bbox) -> float:
    """Maximum number of display pixels by which inner lies outside outer."""
    return max(
        0.0,
        outer.x0 - inner.x0,
        inner.x1 - outer.x1,
        outer.y0 - inner.y0,
        inner.y1 - outer.y1,
    )


def _trend_line_obstacles(ax, renderer, pad_px: float = 3.0, step_px: float = 6.0) -> list[Bbox]:
    """Return small display-space boxes along drawn OLS trend lines.

    Text labels are placed after trend lines are drawn. Without explicit line
    obstacles, the label placer can choose a marker-clear position that still
    sits on top of the dashed OLS fit. Only high-zorder/solid-enough lines are
    treated as obstacles so faint pair-connection lines do not dominate.
    """
    obstacles: list[Bbox] = []
    axbb = ax.get_window_extent(renderer)
    for line in ax.lines:
        if not line.get_visible():
            continue
        linestyle = line.get_linestyle()
        if linestyle in {"", " ", "None", "none"}:
            continue
        if line.get_zorder() < 2 or line.get_linewidth() < 0.75:
            continue

        xdata, ydata = line.get_data(orig=False)
        if len(xdata) < 2:
            continue
        pts = ax.transData.transform(np.column_stack([xdata, ydata]))
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 2:
            continue

        for a, b in zip(pts[:-1], pts[1:]):
            seg_len = float(np.linalg.norm(b - a))
            n = max(1, int(math.ceil(seg_len / step_px)))
            for t in np.linspace(0.0, 1.0, n + 1):
                q = a + t * (b - a)
                if not axbb.contains(q[0], q[1]):
                    continue
                obstacles.append(Bbox.from_extents(q[0] - pad_px, q[1] - pad_px, q[0] + pad_px, q[1] + pad_px))
    return obstacles


def _label_candidate_offsets() -> list[tuple[float, float, str, str]]:
    """Candidate annotation offsets, ordered from compact to more displaced.

    The earlier compact version only tried a small ring of offsets. Dense
    phase-panel figures can need one or two labels to move farther away to
    avoid an already placed label, especially in the lower-left corner.
    """
    offsets: list[tuple[float, float, str, str]] = []
    for r in [5, 8, 11, 15, 20, 26, 33, 41, 50]:
        offsets.extend([
            (r, r, "left", "bottom"),
            (r, -r, "left", "top"),
            (-r, r, "right", "bottom"),
            (-r, -r, "right", "top"),
            (0, r, "center", "bottom"),
            (0, -r, "center", "top"),
            (r, 0, "left", "center"),
            (-r, 0, "right", "center"),
        ])
    return offsets


def annotate_points(
    ax,
    points: list[PlotPoint],
    label_points: str,
    label_mode: str,
    label_max: int,
    aggregate_labels: bool = True,
    label_cluster_px: float = 50.0,
    label_fontsize: float = 5.4,
    label_pad_px: float = 4.0,
    label_axis_inset_px: float = 3.0,
) -> None:
    candidates = candidate_labeled_points(points, label_points)
    if not candidates:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer)
    # Containment is a hard constraint, not a soft penalty. A one-pixel
    # minimum inset prevents renderer/backend rounding from letting text leak
    # across the axes box even when users pass --label-axis-inset-px 0.
    effective_axis_inset_px = max(1.0, float(label_axis_inset_px))
    safe_axbb = Bbox.from_extents(
        axbb.x0 + effective_axis_inset_px,
        axbb.y0 + effective_axis_inset_px,
        axbb.x1 - effective_axis_inset_px,
        axbb.y1 - effective_axis_inset_px,
    )

    groups = cluster_label_points(ax, candidates, label_mode, aggregate_labels, label_cluster_px)
    if len(groups) > label_max:
        print(f"[warn] limiting labels to {label_max} representative label groups. Use --label-max to change this.")
        groups = diverse_groups(groups, label_max)

    marker_obstacles = _marker_obstacles(ax, points, None, extra_px=3.0)
    line_obstacles = _trend_line_obstacles(ax, renderer)
    # Regression-statistics boxes and other reserved annotation boxes are hard
    # obstacles. Labels may be pushed farther away, but they should not be
    # placed on top of these boxes.
    reserved_obstacles: list[Bbox] = list(getattr(ax, "_readability_obstacle_bboxes", []))
    placed_label_bboxes: list[Bbox] = []

    groups = sorted(groups, key=lambda g: (g.text.count("\n"), len(g.text)), reverse=True)
    candidate_offsets = _label_candidate_offsets()

    def _score_bbox(bb: Bbox, dx: float, dy: float, text: str) -> tuple[float, float, float]:
        label_overlap = 0.0
        for ob in placed_label_bboxes:
            label_overlap += _bbox_overlap_area(bb, ob)

        marker_penalty = 0.0
        for ob in marker_obstacles:
            marker_penalty += 0.8 * _bbox_overlap_area(bb, ob)

        trend_penalty = 0.0
        for ob in line_obstacles:
            area = _bbox_overlap_area(bb, ob)
            if area:
                trend_penalty += 90.0 + 2.5 * area

        dist_penalty = 0.035 * (abs(dx) + abs(dy))
        line_penalty = 0.20 * text.count("\n")
        label_penalty = 10000.0 * label_overlap if label_overlap else 0.0
        score = label_penalty + marker_penalty + trend_penalty + dist_penalty + line_penalty
        return score, label_overlap, marker_penalty + trend_penalty

    def _score_bbox_global(bb: Bbox, anchor_px: tuple[float, float], cand_px: tuple[float, float], text: str) -> tuple[float, float, float]:
        label_overlap = 0.0
        for ob in placed_label_bboxes:
            label_overlap += _bbox_overlap_area(bb, ob)

        marker_penalty = 0.0
        for ob in marker_obstacles:
            marker_penalty += 0.8 * _bbox_overlap_area(bb, ob)

        trend_penalty = 0.0
        for ob in line_obstacles:
            area = _bbox_overlap_area(bb, ob)
            if area:
                trend_penalty += 90.0 + 2.5 * area

        dist = float(np.linalg.norm(np.array(cand_px) - np.array(anchor_px)))
        dist_penalty = 0.018 * dist
        line_penalty = 0.20 * text.count("\n")
        label_penalty = 10000.0 * label_overlap if label_overlap else 0.0
        score = label_penalty + marker_penalty + trend_penalty + dist_penalty + line_penalty
        return score, label_overlap, marker_penalty + trend_penalty

    for group in groups:
        text = group.text
        if not text:
            continue

        anchor_px = tuple(ax.transData.transform((group.x, group.y)))

        local_best = None
        local_best_score = None
        local_best_bbox = None
        local_best_label_overlap = None

        for dx, dy, ha, va in candidate_offsets:
            ann = ax.annotate(
                text,
                xy=(group.x, group.y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=label_fontsize,
                linespacing=0.92,
                arrowprops=dict(arrowstyle="-", lw=0.32, color="0.35", alpha=0.45),
                zorder=5,
            )
            bb = _expanded_bbox(ann.get_window_extent(renderer), label_pad_px)
            ann.remove()

            outside_px = _bbox_outside_distance(bb, safe_axbb)
            if outside_px > 0.0:
                continue
            if any(_bbox_overlap_area(bb, ob) > 0.0 for ob in reserved_obstacles):
                continue

            score, label_overlap, other_penalty = _score_bbox(bb, dx, dy, text)
            if local_best_score is None or score < local_best_score:
                local_best_score = score
                local_best = (dx, dy, ha, va)
                local_best_bbox = bb
                local_best_label_overlap = label_overlap

            if label_overlap == 0.0 and other_penalty <= 0.0 and score <= 1.0:
                break

        placement = None

        if local_best is not None and local_best_bbox is not None and (local_best_label_overlap or 0.0) == 0.0:
            dx, dy, ha, va = local_best
            placement = ("offset", dx, dy, ha, va)
        else:
            # Full in-box relocation search: never skip a label. If compact
            # offset-based placement fails, search a dense set of legal text
            # anchor locations across the panel and pick the best in-box one.
            x_positions = np.linspace(safe_axbb.x0 + 6.0, safe_axbb.x1 - 6.0, 16)
            y_positions = np.linspace(safe_axbb.y0 + 6.0, safe_axbb.y1 - 6.0, 14)
            alignments = [
                ("left", "bottom"), ("left", "center"), ("left", "top"),
                ("center", "bottom"), ("center", "center"), ("center", "top"),
                ("right", "bottom"), ("right", "center"), ("right", "top"),
            ]
            grid_points: list[tuple[float, float]] = [(float(x), float(y)) for y in y_positions for x in x_positions]
            grid_points.sort(key=lambda q: (float(np.linalg.norm(np.array(q) - np.array(anchor_px))), -q[1], q[0]))

            global_best = None
            global_best_score = None
            global_best_bbox = None
            global_best_overlap = None
            for tx_px, ty_px in grid_points:
                tx_data, ty_data = ax.transData.inverted().transform((tx_px, ty_px))
                for ha, va in alignments:
                    ann = ax.annotate(
                        text,
                        xy=(group.x, group.y),
                        xytext=(float(tx_data), float(ty_data)),
                        textcoords="data",
                        ha=ha,
                        va=va,
                        fontsize=label_fontsize,
                        linespacing=0.92,
                        arrowprops=dict(arrowstyle="-", lw=0.32, color="0.35", alpha=0.45),
                        zorder=5,
                    )
                    bb = _expanded_bbox(ann.get_window_extent(renderer), label_pad_px)
                    ann.remove()

                    if _bbox_outside_distance(bb, safe_axbb) > 0.0:
                        continue
                    if any(_bbox_overlap_area(bb, ob) > 0.0 for ob in reserved_obstacles):
                        continue

                    score, label_overlap, other_penalty = _score_bbox_global(bb, anchor_px, (tx_px, ty_px), text)
                    if global_best_score is None or score < global_best_score:
                        global_best_score = score
                        global_best = (float(tx_data), float(ty_data), ha, va)
                        global_best_bbox = bb
                        global_best_overlap = label_overlap

                    if label_overlap == 0.0 and other_penalty <= 0.0 and score <= 3.0:
                        break
                if global_best_overlap == 0.0 and global_best_score is not None and global_best_score <= 3.0:
                    break

            if global_best is not None and global_best_bbox is not None:
                tx_data, ty_data, ha, va = global_best
                placement = ("data", tx_data, ty_data, ha, va)
            elif local_best is not None and local_best_bbox is not None:
                # Defensive fallback: keep the best legal local placement rather
                # than skipping the label entirely.
                dx, dy, ha, va = local_best
                placement = ("offset", dx, dy, ha, va)
            else:
                # Extreme defensive fallback. Put the label at the nearest safe
                # display-space point in the panel and clip it to the axes.
                tx_px = min(max(anchor_px[0], safe_axbb.x0 + 8.0), safe_axbb.x1 - 8.0)
                ty_px = min(max(anchor_px[1], safe_axbb.y0 + 8.0), safe_axbb.y1 - 8.0)
                tx_data, ty_data = ax.transData.inverted().transform((tx_px, ty_px))
                placement = ("data", float(tx_data), float(ty_data), "center", "center")
        
        if placement is None:
            continue

        mode, a, b, ha, va = placement
        if mode == "offset":
            final_ann = ax.annotate(
                text,
                xy=(group.x, group.y),
                xytext=(a, b),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=label_fontsize,
                linespacing=0.92,
                arrowprops=dict(arrowstyle="-", lw=0.32, color="0.35", alpha=0.45),
                zorder=5,
            )
        else:
            final_ann = ax.annotate(
                text,
                xy=(group.x, group.y),
                xytext=(a, b),
                textcoords="data",
                ha=ha,
                va=va,
                fontsize=label_fontsize,
                linespacing=0.92,
                arrowprops=dict(arrowstyle="-", lw=0.32, color="0.35", alpha=0.45),
                zorder=5,
            )
        final_ann.set_clip_on(True)
        final_ann.set_clip_path(ax.patch)
        placed_label_bboxes.append(_expanded_bbox(final_ann.get_window_extent(renderer), label_pad_px))


def scatter_points(ax, points: list[PlotPoint], colors: dict[str, object], args: argparse.Namespace) -> None:
    for phase in ["input+output", "decode-only"]:
        for baseline in ["mean-donor", "mean"]:
            subset = [p for p in points if p.phase == phase and p.baseline == baseline]
            if not subset:
                continue
            ax.scatter(
                [p.score for p in subset],
                [p.union_rate for p in subset],
                s=[size_for_point(p, args) for p in subset],
                marker=marker_for_phase(phase),
                c=[colors[p.task] for p in subset],
                edgecolors=edge_for_baseline(baseline),
                linewidths=0.42 if baseline == "mean-donor" else 0.32,
                alpha=0.84 if baseline == "mean-donor" else 0.58,
                zorder=3,
            )


def _nice_density_axis_top(value: float) -> float:
    """Round a positive density limit upward to a compact, readable value."""
    if not math.isfinite(value) or value <= 0:
        return 0.05
    exponent = 10.0 ** math.floor(math.log10(value))
    scaled = value / exponent
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if scaled <= candidate:
            return candidate * exponent
    return 10.0 * exponent


def _phase_panel_y_scale(points: list[PlotPoint], coverage_metric: str) -> tuple[float, float] | None:
    """Return a tighter shared y-scale for the D_.05 phase panels.

    These density metrics occupy only a small fraction of [0, 1].  Keeping a
    full unit interval makes scientifically relevant differences nearly
    invisible.  The minima below match the current paper figures, while the
    observed-data guard automatically expands the axis if later runs contain a
    larger value so no point is clipped.
    """
    presets = {
        "n05-i2c-density": (0.05, 0.01),
        "n05-c2i-density": (0.40, 0.10),
    }
    preset = presets.get(str(coverage_metric))
    if preset is None:
        return None
    minimum_top, preset_tick = preset
    finite = [float(p.union_rate) for p in points if math.isfinite(float(p.union_rate)) and float(p.union_rate) >= 0.0]
    observed_max = max(finite, default=0.0)
    needed_top = _nice_density_axis_top(observed_max * 1.15) if observed_max > 0 else minimum_top
    y_top = max(minimum_top, needed_top)
    if y_top <= minimum_top * 1.000001:
        tick_step = preset_tick
    else:
        raw_tick = y_top / 4.0
        tick_step = _nice_density_axis_top(raw_tick)
    return y_top, tick_step


def decorate_axis(
    ax,
    title: str | None = None,
    tick_step: float = 0.25,
    *,
    y_max: float = 1.0,
    y_tick_step: float | None = None,
) -> None:
    if title:
        ax.set_title(title, pad=1.5)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.03 * y_max, y_max)
    x_ticks = np.arange(0.0, 1.0 + 0.5 * tick_step, tick_step)
    x_ticks = [float(t) for t in x_ticks if t <= 1.0001]
    effective_y_tick = tick_step if y_tick_step is None else y_tick_step
    y_ticks = np.arange(0.0, y_max + 0.5 * effective_y_tick, effective_y_tick)
    y_ticks = [float(t) for t in y_ticks if t <= y_max + 1e-10]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="both", which="major", pad=1.0, length=2.0, width=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    ax.grid(True, linewidth=0.32, alpha=0.22)


def legend_handles(
    points: list[PlotPoint],
    colors: dict[str, object],
    trend_handles: list[Line2D],
    include_tasks: bool = True,
    compact: bool = True,
) -> list[Line2D]:
    handles: list[Line2D] = []
    marker_size = 4.2 if compact else 6.0
    if include_tasks:
        for task in [t for t in TASK_ORDER if any(p.task == t for p in points)]:
            handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[task], markeredgecolor="black", markeredgewidth=0.45, markersize=marker_size, label=TASK_LABELS[task]))

    # Use a light fill for the phase handles so they read as shape cues,
    # whereas baseline handles stay hollow and emphasize outline color. This
    # avoids the confusing visual collision where decode and donor both looked
    # like the same open-circle symbol in the legend.
    phase_face = "0.80"
    phase_edge = "0.15"
    phases = {p.phase for p in points}
    if "decode-only" in phases:
        handles.append(Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=phase_face,
            markeredgecolor=phase_edge, markeredgewidth=0.60, markersize=marker_size,
            label=("Output-only" if compact else "Output-only")
        ))
    if "input+output" in phases:
        handles.append(Line2D(
            [0], [0], marker="s", color="none", markerfacecolor=phase_face,
            markeredgecolor=phase_edge, markeredgewidth=0.60, markersize=marker_size,
            label=("Input+output" if compact else "Input+output")
        ))

    baselines = {p.baseline for p in points}
    if "mean-donor" in baselines and "mean" in baselines:
        handles.extend([
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.05, markersize=marker_size,
                label="donor abl."
            ),
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="white",
                markeredgecolor="0.55", markeredgewidth=1.05, markersize=marker_size,
                label="mean abl."
            ),
        ])

    handles.extend(trend_handles)
    return handles


def paper_figsize(args: argparse.Namespace, layout: str, nrows: int = 1, ncols: int = 1) -> tuple[float, float]:
    if args.fig_width is not None or args.fig_height is not None:
        fallback = paper_figsize(argparse.Namespace(fig_width=None, fig_height=None, paper_size=args.paper_size, legend_position=args.legend_position), layout, nrows, ncols)
        return (args.fig_width if args.fig_width is not None else fallback[0], args.fig_height if args.fig_height is not None else fallback[1])

    if args.paper_size == "single":
        if layout == "task-grid":
            return (3.25, max(2.25, 1.35 * nrows + 0.35))
        if layout == "phase-panels":
            return (3.25, 1.95)
        return (3.25, 2.05)

    # Compact full-width default for two-column papers.
    if layout == "task-grid":
        return (5.35, max(2.65, 1.45 * nrows + 0.55))
    if layout == "phase-panels":
        return (5.35, 2.15)
    return (5.35, 2.25)


def legend_common_kwargs() -> dict:
    return dict(
        frameon=False,
        borderpad=0.05,
        handletextpad=0.28,
        columnspacing=0.55,
        labelspacing=0.18,
        handlelength=1.05,
    )


def build_legend(
    fig,
    ax,
    points: list[PlotPoint],
    colors: dict[str, object],
    trend_handles: list[Line2D],
    args: argparse.Namespace,
    include_tasks: bool = True,
) -> None:
    position = args.legend_position
    if position == "none":
        return

    handles = legend_handles(points, colors, trend_handles, include_tasks=include_tasks, compact=not args.long_legend_labels)
    if not handles:
        return
    ncol = args.legend_cols or min(len(handles), 5 if args.paper_size == "wide" else 3)
    common = legend_common_kwargs()

    if position == "right":
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), ncol=1, **common)
    elif position == "inside":
        ax.legend(handles=handles, loc="upper right", ncol=1, **common)
    else:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=ncol, **common)


def subplot_margins(args: argparse.Namespace, layout: str) -> dict:
    position = args.legend_position
    bottom = 0.27 if position == "bottom" else 0.16
    if layout == "task-grid":
        return dict(left=0.075, right=0.995 if position != "right" else 0.86, bottom=bottom, top=0.91, wspace=0.13, hspace=0.24)
    if layout == "phase-panels":
        return dict(left=0.085, right=0.995 if position != "right" else 0.84, bottom=bottom, top=0.88, wspace=0.09)
    return dict(left=0.105, right=0.995 if position != "right" else 0.77, bottom=bottom, top=0.96)


def task_fig_label(task: str) -> str:
    return TASK_FIG_LABELS.get(task, TASK_LABELS.get(task, task))


def normalize_model_token(value: str) -> str:
    value = value.lower().replace("instruct", "")
    return re.sub(r"[^a-z0-9.]+", "", value)


def model_matches_template(model: str, template: str) -> bool:
    return normalize_model_token(template) in normalize_model_token(model)


def model_checkpoint_step(model: str) -> int | None:
    # Checkpoint directory names in the results tree are literal model names:
    #   pythia-1b@step0, pythia-1b@step48000, pythia-1b@step96000,
    #   pythia-1b  -> final checkpoint, treated as step143000.
    lowered = model.lower()
    m = re.search(r"(?:@|[_\-])step[_\-]?(\d+)", lowered)
    if m:
        return int(m.group(1))
    m = re.search(r"step[_\-]?(\d+)", lowered)
    if m:
        return int(m.group(1))
    if "pythia-1b" in lowered and "step" not in lowered:
        return 143000
    return None


def model_template_and_step_match(model: str, template: str, step: int | None) -> bool:
    if not model_matches_template(model, template):
        return False
    if step is None:
        return model_checkpoint_step(model) is None or "pythia" not in template.lower()
    return model_checkpoint_step(model) == step


def paper_filters_from(filters: Filters, include_empty: bool) -> Filters:
    # Summary figures name their own tasks/models/phases. They should not be
    # constrained by CLI M/tau preferences because the paper panels need to use
    # whichever real run exists for each requested model/task combination.
    return Filters(
        tasks=None,
        orgs=filters.orgs,
        model_regex=None,
        phases=None,
        baselines=None,
        anchor=filters.anchor,
        name_contains=filters.name_contains,
        exclude_fake_targets=filters.exclude_fake_targets,
        exclude_checkpoints=False,
        require_m=None,
        require_tau=None,
        prefer_m=filters.prefer_m,
        prefer_tau=filters.prefer_tau,
        include_empty=include_empty,
    )


def select_template_point(
    points: list[PlotPoint],
    filters: Filters,
    task: str,
    model_template: str,
    step: int | None,
    phase: str | None,
    baseline: str | None,
) -> PlotPoint | None:
    candidates = []
    for p in points:
        if p.task != task:
            continue
        if phase is not None and p.phase != phase:
            continue
        if baseline is not None and p.baseline != baseline:
            continue
        if not model_template_and_step_match(p.model, model_template, step):
            continue
        candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: run_priority(p, filters))


def select_template_point_with_fallback(
    points: list[PlotPoint],
    filters: Filters,
    task: str,
    model_template: str,
    step: int | None,
    preferred_phase: str | None,
    preferred_baseline: str | None,
    allow_phase_fallback: bool = False,
    allow_baseline_fallback: bool = False,
) -> PlotPoint | None:
    candidates = [
        p for p in points
        if p.task == task and model_template_and_step_match(p.model, model_template, step)
    ]
    if not candidates:
        return None

    phase_options = [preferred_phase] if preferred_phase is not None else [None]
    baseline_options = [preferred_baseline] if preferred_baseline is not None else [None]
    if allow_baseline_fallback:
        baseline_options.append(None)
    if allow_phase_fallback:
        phase_options.append(None)

    seen: set[tuple[str | None, str | None]] = set()
    for phase in phase_options:
        for baseline in baseline_options:
            key = (phase, baseline)
            if key in seen:
                continue
            seen.add(key)
            subset = [
                p for p in candidates
                if (phase is None or p.phase == phase)
                and (baseline is None or p.baseline == baseline)
            ]
            if subset:
                return max(subset, key=lambda p: run_priority(p, filters))
    return None


def paper_row(figure: str, label: str, point: PlotPoint | None, requested_phase: str | None = None) -> dict:
    row = {
        "figure": figure,
        "label": label.replace("\n", " "),
        "requested_phase": requested_phase or "any",
        "task": "",
        "task_label": "",
        "org": "",
        "model": "",
        "phase": "",
        "baseline": "",
        "score": "",
        "union_rate": "",
        "n_neurons": "",
        "n_eval": "",
        "status": "missing",
        "source_path": "",
        "run": "",
    }
    if point is None:
        return row
    row.update({
        "task": point.task,
        "task_label": task_fig_label(point.task),
        "org": point.org,
        "model": point.model,
        "phase": point.phase,
        "baseline": point.baseline,
        "score": point.score,
        "union_rate": point.union_rate,
        "n_neurons": point.n_neurons,
        "n_eval": point.n_eval if point.n_eval is not None else "",
        "status": point.status,
        "source_path": point.source_path,
        "run": point.run,
    })
    return row


def write_rows_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    csv_path = out.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "figure", "label", "requested_phase", "task", "task_label", "org", "model", "phase",
        "baseline", "score", "union_rate", "n_neurons", "n_eval", "status", "source_path", "run",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=preferred + extra)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def paper_summary_rc_context():
    return plt.rc_context({
        "font.size": 9.2,
        "axes.labelsize": 9.6,
        "axes.titlesize": 9.6,
        "xtick.labelsize": 8.7,
        "ytick.labelsize": 8.7,
        "legend.fontsize": 8.4,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })


def short_step_label(step: int | None) -> str | None:
    """Compact checkpoint label used where x-axis space is scarce."""
    if step is None:
        return None
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def compact_phase_task_label(task: str) -> str:
    """Very short task labels for dense phase-comparison x ticks."""
    if task == "arithmetic":
        return "Arith."
    if task == "grammar_acceptability":
        return "Gram."
    if task == "hans_nli":
        return "NLI"
    if task == "random_fsm":
        return "FSM"
    if task == "bon_jailbreaking":
        return "Safety"
    return task_fig_label(task)


def compact_phase_model_label(model: str, step: int | None = None) -> tuple[str, str]:
    """Return compact model-family and size/checkpoint lines for dense x ticks."""
    family, _size_rank, size_label = size_family_rank_and_label(model)
    if family == "Qwen2":
        family_label = "Qw2"
    elif family == "Qwen2.5":
        family_label = "Qw2.5"
    elif family == "Pythia":
        family_label = "Py"
    else:
        family_label = compact_size_model_label(model_short(model))

    inferred_step = model_checkpoint_step(model) if step is None else step
    step_label = short_step_label(inferred_step)
    # The final Pythia-1B checkpoint is the canonical model, so omit @143k to
    # keep labels as short as the size-comparison panel labels.
    if step_label is not None and inferred_step != 143000:
        size_line = f"{size_label}@{step_label}"
    else:
        size_line = size_label
    return family_label, size_line


def compact_phase_comparison_label(task: str, model: str, step: int | None = None) -> str:
    family_label, size_line = compact_phase_model_label(model, step)
    return f"{family_label}\n{size_line}\n{compact_phase_task_label(task)}"


def wrap_phase_comparison_label(label: str, width: int = 8) -> str:
    """Wrap x-axis labels for the phase-comparison figure without rotation."""
    wrapped_parts: list[str] = []
    for part in label.split("\n"):
        filled = textwrap.fill(
            part,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_parts.extend(filled.splitlines())
    return "\n".join(wrapped_parts)


def select_matched_phase_points(
    points: list[PlotPoint],
    filters: Filters,
    task: str,
    model: str,
    step: int | None,
    preferred_baseline: str,
    allow_baseline_fallback: bool = True,
) -> tuple[str, dict[str, PlotPoint]] | None:
    """Select input+output and output-only points that share the same baseline.

    fig_phase_comparison is a matched phase comparison. It must not pair an
    input+output mean-donor point with an output-only mean point, because that
    creates a baseline-mismatched descriptive comparison rather than a phase
    effect. If fallback is allowed, fallback is only allowed to another baseline
    that exists for both phases.
    """
    candidates = [
        p for p in points
        if p.task == task and model_template_and_step_match(p.model, model, step)
    ]
    if not candidates:
        return None

    available_baselines = sorted({p.baseline for p in candidates})
    baseline_options = [preferred_baseline]
    if allow_baseline_fallback:
        baseline_options.extend(b for b in available_baselines if b != preferred_baseline)

    seen_baselines: set[str] = set()
    for candidate_baseline in baseline_options:
        if candidate_baseline in seen_baselines:
            continue
        seen_baselines.add(candidate_baseline)
        selected_by_phase: dict[str, PlotPoint] = {}
        for phase in ["input+output", "decode-only"]:
            phase_candidates = [
                p for p in candidates
                if p.phase == phase and p.baseline == candidate_baseline
            ]
            if not phase_candidates:
                selected_by_phase = {}
                break
            selected_by_phase[phase] = max(phase_candidates, key=lambda p: run_priority(p, filters))
        if len(selected_by_phase) == 2:
            return candidate_baseline, selected_by_phase
    return None


def build_phase_comparison_specs(
    points: list[PlotPoint],
    filters: Filters,
    baseline: str,
    allow_baseline_fallback: bool = True,
) -> list[dict]:
    """Build phase-comparison categories from task/model groups with both phases.

    The phase-comparison figure only compares like with like. A task/model group
    is included only when input+output and output-only are both present under the
    same baseline. When allow_baseline_fallback is true, the requested baseline
    is preferred, but fallback is only to another common baseline shared by both
    phases. The CSV records the actual baseline used.
    """
    seen: dict[tuple[str, str, int | None], PlotPoint] = {}
    for p in points:
        step = model_checkpoint_step(p.model)
        key = (p.task, p.model, step)
        if key not in seen or run_priority(p, filters) > run_priority(seen[key], filters):
            seen[key] = p

    def _model_family_order(model: str) -> tuple[int, int, int, str]:
        family, size_rank, _size_label = size_family_rank_and_label(model)
        family_order = 0 if family == "Qwen2" else (1 if family == "Qwen2.5" else (2 if family == "Pythia" else 3))
        step = model_checkpoint_step(model)
        step_order = 143000 if step is None and "pythia" in model.lower() else (-1 if step is None else step)
        return (family_order, int(size_rank), int(step_order), model.lower())

    def _spec_sort_key(item: tuple[tuple[str, str, int | None], PlotPoint]) -> tuple:
        (task, model, _), _point = item
        task_idx = TASK_ORDER.index(task) if task in TASK_ORDER else 99
        return (task_idx, _model_family_order(model))

    specs: list[dict] = []
    for (task, model, step), _p in sorted(seen.items(), key=_spec_sort_key):
        matched = select_matched_phase_points(
            points,
            filters,
            task,
            model,
            step,
            preferred_baseline=baseline,
            allow_baseline_fallback=allow_baseline_fallback,
        )
        if matched is None:
            continue
        matched_baseline, selected_by_phase = matched
        specs.append(dict(
            task=task,
            model=model,
            step=step,
            label=compact_phase_comparison_label(task, model, step),
            baseline=matched_baseline,
            points_by_phase=selected_by_phase,
        ))
    return specs


def dataset_score_only_point(
    root: Path,
    task: str,
    model_template: str,
    step: int | None,
    phase: str,
    baseline: str,
) -> PlotPoint | None:
    """Return a zero-coverage point when dataset scores exist but no stats run does.

    This is used only for summary panels that intentionally display
    zero-discovery configurations, such as Pythia HANS-NLI, instead of silently
    dropping them from the visual comparison.
    """
    matches: list[tuple[str, str, Path]] = []
    for task_name, org, model, model_dir in task_model_dir_iter(root):
        if task_name == task and model_template_and_step_match(model, model_template, step):
            matches.append((org, model, model_dir))
    if not matches:
        return None
    org, model, model_dir = sorted(matches, key=lambda item: (item[0], item[1], str(item[2])))[0]
    ds_path = model_dir / "feature_report" / "dataset_stats.json"
    if not ds_path.exists():
        return None
    dataset_stats = read_json(ds_path)
    try:
        score = downstream_score(task, dataset_stats, phase=phase)
    except Exception:
        return None
    try:
        source_path = str(ds_path.relative_to(root))
    except ValueError:
        source_path = str(ds_path)
    return PlotPoint(
        task=task,
        task_label=TASK_LABELS.get(task, task),
        org=org,
        model=model,
        phase=phase,
        baseline=baseline,
        anchor="dataset_stats_only",
        run="dataset_stats_only_no_matching_flip_stats",
        score=score,
        union_rate=math.nan,
        n_neurons=None,
        n_eval=None,
        status="dataset-score-only-causal-coverage-missing",
        source_path=source_path,
    )


def ensure_checkpoint_dataset_score_points(
    root: Path,
    points: list[PlotPoint],
    phase: str,
    baseline: str,
) -> list[PlotPoint]:
    """Preserve real dataset scores when a checkpoint lacks a matching stats run."""
    augmented = list(points)

    def has_exact_point(task: str, step: int) -> bool:
        return any(
            p.task == task
            and p.phase == phase
            and p.baseline == baseline
            and model_template_and_step_match(p.model, "pythia-1b", step)
            for p in augmented
        )

    for task, _label in CHECKPOINT_TASKS:
        for step in CHECKPOINT_STEPS:
            if has_exact_point(task, step):
                continue
            matches: list[tuple[str, str, Path]] = []
            for task_name, org, model, model_dir in task_model_dir_iter(root):
                if task_name == task and model_template_and_step_match(model, "pythia-1b", step):
                    matches.append((org, model, model_dir))
            if not matches:
                continue
            org, model, model_dir = sorted(matches, key=lambda item: (item[0], item[1], str(item[2])))[0]
            ds_path = model_dir / "feature_report" / "dataset_stats.json"
            if not ds_path.exists():
                continue
            dataset_stats = read_json(ds_path)
            try:
                score = downstream_score(task, dataset_stats, phase=phase)
            except Exception:
                continue
            try:
                source_path = str(ds_path.relative_to(root))
            except ValueError:
                source_path = str(ds_path)
            augmented.append(PlotPoint(
                task=task,
                task_label=TASK_LABELS.get(task, task),
                org=org,
                model=model,
                phase=phase,
                baseline=baseline,
                anchor="dataset_stats_only",
                run="dataset_stats_only_no_matching_flip_stats",
                score=score,
                union_rate=math.nan,
                n_neurons=None,
                n_eval=None,
                status="dataset-score-only-causal-coverage-missing",
                source_path=source_path,
            ))
    return augmented


def plot_phase_comparison_figure(points: list[PlotPoint], filters: Filters, out: Path, args: argparse.Namespace) -> None:
    labels: list[str] = []
    wrapped_labels: list[str] = []
    rows: list[dict] = []
    series = {"input+output": [], "decode-only": []}
    baseline = args.paper_baseline
    allow_baseline_fallback = bool(getattr(args, "paper_phase_baseline_fallback", True))

    specs = build_phase_comparison_specs(
        points,
        filters,
        baseline=baseline,
        allow_baseline_fallback=allow_baseline_fallback,
    )

    for spec in specs:
        labels.append(spec["label"])
        wrapped_labels.append(wrap_phase_comparison_label(spec["label"]))
        points_by_phase = spec["points_by_phase"]
        for phase in ["input+output", "decode-only"]:
            p = points_by_phase[phase]
            series[phase].append(p.union_rate)
            row = paper_row("phase_comparison", spec["label"], p, phase)
            row["matched_baseline"] = spec["baseline"]
            rows.append(row)

    x = np.arange(len(labels))
    with paper_summary_rc_context():
        # The width expands mildly as categories are added, while compact labels
        # keep the panel readable without rotated x ticks.
        fig_width = max(6.2, min(9.2, 0.36 * len(labels) + 1.05))
        fig, ax = plt.subplots(figsize=(fig_width, 2.85))
        ax.plot(x, series["input+output"], marker="s", linewidth=1.45, markersize=4.9, label="Input+output")
        ax.plot(x, series["decode-only"], marker="o", linewidth=1.45, markersize=4.9, label="Output-only")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(r"Overtopping coverage $U(J)$")
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_labels)
        ax.tick_params(axis="x", pad=1.0, labelsize=7.0 if len(labels) > 14 else 7.8)
        ax.grid(axis="y", alpha=0.45, linewidth=0.45)
        ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)
        fig.subplots_adjust(left=0.070, right=0.995, bottom=0.315, top=0.985)
        save_outputs(fig, out, args)
    if not args.no_csv:
        write_rows_csv(rows, out)
    print(f"[OK] wrote {out}")
    if not args.no_csv:
        print(f"[OK] wrote {(out.with_suffix('.csv') if args.csv_out_dir is None else Path(args.csv_out_dir) / out.with_suffix('.csv').name)}")
    print(f"[phase comparison] {len(labels)} task/model groups plotted")


def plot_checkpoint_trajectory_figure(points: list[PlotPoint], filters: Filters, out: Path, args: argparse.Namespace) -> None:
    """Plot the Pythia-1B checkpoint trajectory from real result records.

    The panel uses two compact stacked axes: the top shows overtopping coverage
    U(J), and the bottom shows task competence. Task identity is encoded by
    color, while the two metrics are separated by panel, marker shape, and
    background tint. Compact numeric labels are shown for every point,
    including zeros, and the y-ranges are tightened to remove unused space.
    """
    baseline = args.paper_baseline
    phase = args.paper_checkpoint_phase
    rows: list[dict] = []
    x = np.asarray(CHECKPOINT_STEPS, dtype=float)
    max_coverage = 0.0
    max_competence = 0.0

    def annotate_values(ax, xs: np.ndarray, ys: list[float], color: str, *, prefer_above: bool) -> None:
        """Add compact numeric labels with larger text and keep them inside the subplot."""
        by_x: dict[float, list[tuple[float, str]]] = {}
        for xi, yi in zip(xs, ys):
            yv = float(yi)
            by_x.setdefault(float(xi), []).append((yv, f"{yv:.2f}"))

        fig = ax.figure
        # The caller performs one canvas draw after the final subplot geometry is
        # established.  Do not draw here: repeated draws while annotations already
        # exist triggered a Matplotlib backend_macosx FancyArrowPatch StopIteration.
        renderer = fig.canvas.get_renderer()
        axbb = ax.get_window_extent(renderer)
        y_low = axbb.y0 + 6.0
        y_high = axbb.y1 - 6.0
        px_to_pt = 72.0 / fig.dpi
        x_mid = float(np.mean(xs))

        for xi, items in by_x.items():
            items = sorted(items, key=lambda t: (t[0], t[1]))
            n = len(items)
            center = (n - 1) / 2.0
            prefer_right = xi <= x_mid
            base_offsets = [10.0, 18.0, 26.0, 34.0]
            signed_offsets: list[float] = []
            for off in base_offsets:
                signed_offsets.extend(([off, -off] if prefer_right else [-off, off]))

            placed_y_pix: list[float] = []
            for idx, (yv, label) in enumerate(items):
                anchor_x_pix, anchor_y_pix = ax.transData.transform((xi, yv))
                layer = abs(idx - center)
                dy_mag_px = 10.0 + 7.0 * layer
                desired_y_pix = anchor_y_pix + dy_mag_px if prefer_above else anchor_y_pix - dy_mag_px

                if desired_y_pix > y_high:
                    desired_y_pix = anchor_y_pix - dy_mag_px
                if desired_y_pix < y_low:
                    desired_y_pix = anchor_y_pix + dy_mag_px

                desired_y_pix = min(y_high, max(y_low, desired_y_pix))

                min_gap_px = 14.0
                if placed_y_pix:
                    for prev in placed_y_pix:
                        if abs(desired_y_pix - prev) < min_gap_px:
                            if desired_y_pix >= anchor_y_pix:
                                desired_y_pix = prev + min_gap_px
                            else:
                                desired_y_pix = prev - min_gap_px
                    desired_y_pix = min(y_high, max(y_low, desired_y_pix))

                dx = signed_offsets[idx] if idx < len(signed_offsets) else (42.0 if prefer_right else -42.0)
                dy = (desired_y_pix - anchor_y_pix) * px_to_pt
                place_above = desired_y_pix >= anchor_y_pix
                va = 'bottom' if place_above else 'top'
                ha = 'left' if dx > 0 else 'right'
                placed_y_pix.append(desired_y_pix)

                # Do not use annotate(..., arrowprops=...).  Matplotlib represents
                # annotation arrows as FancyArrowPatch objects and, for degenerate
                # or nearly coincident checkpoint-label geometry, its clipping
                # code can raise StopIteration while savefig() draws the figure.
                # A plain Line2D leader is visually equivalent here and does not
                # go through the FancyArrowPatch/Bezier clipping path.
                dx_pix = dx / px_to_pt
                leader_x_pix = anchor_x_pix + dx_pix
                leader_y_pix = desired_y_pix
                leader_x_data, leader_y_data = ax.transData.inverted().transform(
                    (leader_x_pix, leader_y_pix)
                )
                ax.plot(
                    [xi, leader_x_data],
                    [yv, leader_y_data],
                    color=color,
                    linewidth=0.38,
                    alpha=0.65,
                    solid_capstyle='round',
                    zorder=5,
                    clip_on=False,
                )

                ax.annotate(
                    label,
                    xy=(xi, yv),
                    xytext=(dx, dy),
                    textcoords='offset points',
                    ha=ha,
                    va=va,
                    fontsize=7.5,
                    color='0.05',
                    bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor=color, linewidth=0.55, alpha=0.92),
                    zorder=6,
                    annotation_clip=False,
                )

    with paper_summary_rc_context():
        fig, (ax_cov, ax_comp) = plt.subplots(
            2,
            1,
            figsize=(5.35, 2.45),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.018},
        )

        ax_cov.set_facecolor("#f5f8fc")
        ax_comp.set_facecolor("#fcf8f3")

        task_handles: list[Line2D] = []
        coverage_annotation_specs: list[tuple[np.ndarray, list[float], str]] = []
        competence_annotation_specs: list[tuple[np.ndarray, list[float], str]] = []
        for task, label in CHECKPOINT_TASKS:
            coverage: list[float] = []
            competence: list[float] = []
            for step in CHECKPOINT_STEPS:
                p = select_template_point(points, filters, task, "pythia-1b", step, phase, baseline)
                if p is None:
                    coverage.append(0.0)
                    competence.append(0.0)
                    row = paper_row("pythia_checkpoint_trajectory", f"{label} step{step}", None, phase)
                    row.update({
                        "task": task,
                        "task_label": task_fig_label(task),
                        "checkpoint_step": step,
                        "union_rate": 0.0,
                        "score": 0.0,
                        "competence": 0.0,
                        "plotted_union_rate": 0.0,
                        "plotted_score": 0.0,
                        "plotted_competence": 0.0,
                        "status": "missing-plotted-as-zero",
                    })
                    rows.append(row)
                else:
                    coverage.append(p.union_rate)
                    competence.append(p.score)
                    row = paper_row("pythia_checkpoint_trajectory", f"{label} step{step}", p, phase)
                    row.update({
                        "checkpoint_step": step,
                        "competence": p.score,
                        "plotted_union_rate": p.union_rate,
                        "plotted_score": p.score,
                        "plotted_competence": p.score,
                    })
                    rows.append(row)

            max_coverage = max(max_coverage, *(float(v) for v in coverage))
            max_competence = max(max_competence, *(float(v) for v in competence))

            cov_line, = ax_cov.plot(
                x,
                coverage,
                marker="o",
                linewidth=1.35,
                markersize=4.2,
                markeredgewidth=0.80,
                label=label,
            )
            color = cov_line.get_color()
            ax_comp.plot(
                x,
                competence,
                marker="s",
                linestyle="--",
                linewidth=1.25,
                markersize=4.0,
                markerfacecolor="white",
                markeredgewidth=0.85,
                color=color,
                label=label,
            )
            coverage_annotation_specs.append((x, coverage, color))
            competence_annotation_specs.append((x, competence, color))
            task_handles.append(Line2D([0], [0], color=color, lw=1.45, marker="o", markersize=4.0, label=label))

        cov_upper = max(0.12, min(1.0, max_coverage + 0.12))
        comp_upper = max(0.12, min(1.0, max_competence + 0.10))
        cov_lower = -0.035
        comp_lower = -0.035

        for ax, upper, lower in ((ax_cov, cov_upper, cov_lower), (ax_comp, comp_upper, comp_lower)):
            if 96000 in CHECKPOINT_STEPS:
                ax.axvline(96000, linestyle=":", linewidth=0.70, color="0.35", alpha=0.75, zorder=1)
            ax.set_xlim(min(CHECKPOINT_STEPS) - 4200, max(CHECKPOINT_STEPS) + 4200)
            ax.set_ylim(lower, upper)
            ax.grid(axis="y", alpha=0.34, linewidth=0.40)
            ax.tick_params(axis='y', pad=0.8, labelsize=7.0)

        if 96000 in CHECKPOINT_STEPS:
            ax_cov.text(96000, cov_upper * 0.74, "96k", rotation=90, va="center", ha="right", fontsize=6.9, color="0.20")

        ax_cov.set_ylabel(checkpoint_metric_label(args.coverage_metric), labelpad=0.8)
        ax_comp.set_ylabel("Competence", labelpad=0.8)
        ax_comp.set_xlabel("Checkpoint", labelpad=0.6)
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(CHECKPOINT_STEP_LABELS)
        ax_cov.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Finalize subplot geometry before computing display-space label offsets,
        # then draw exactly once before annotations are added.
        fig.subplots_adjust(left=0.10, right=0.995, bottom=0.16, top=0.975)
        fig.canvas.draw()

        for xs_ann, ys_ann, color_ann in coverage_annotation_specs:
            annotate_values(ax_cov, xs_ann, ys_ann, color_ann, prefer_above=True)
        for xs_ann, ys_ann, color_ann in competence_annotation_specs:
            annotate_values(ax_comp, xs_ann, ys_ann, color_ann, prefer_above=True)

        ax_cov.legend(
            handles=task_handles,
            frameon=False,
            loc="upper left",
            ncol=3,
            handlelength=1.00,
            columnspacing=0.48,
            borderaxespad=0.05,
            fontsize=8.0,
        )

        save_outputs(fig, out, args)

    if not args.no_csv:
        write_rows_csv(rows, out)
    print(f"[OK] wrote {out}")
    if not args.no_csv:
        print(f"[OK] wrote {(out.with_suffix('.csv') if args.csv_out_dir is None else Path(args.csv_out_dir) / out.with_suffix('.csv').name)}")

def compact_size_model_label(model_label: str) -> str:
    """Compact model labels that remain legible in a half-width paper panel."""
    label = model_label
    label = label.replace("-Instruct", "")
    label = label.replace("Qwen2.5 ", "Qw2.5-")
    label = label.replace("Qwen2 ", "Qw2-")
    label = label.replace("Qwen2.5-", "Qw2.5-")
    label = label.replace("Qwen2-", "Qw2-")
    label = label.replace("Pythia ", "Py-")
    label = label.replace("pythia-", "Py-")
    return label


def size_family_rank_and_label(model: str) -> tuple[str, int, str]:
    m = model.lower()
    if "qwen2-1.5b" in m:
        return "Qwen2", 0, "1.5B"
    if "qwen2-7b" in m:
        return "Qwen2", 1, "7B"
    if "qwen2.5-1.5b" in m:
        return "Qwen2.5", 0, "1.5B"
    if "pythia-1b" in m:
        return "Pythia", 0, "1B"
    if "pythia-6.9b" in m:
        return "Pythia", 1, "6.9B"
    return model_short(model), 0, model_short(model)


def size_comparison_spec_groups(specs: list[dict]) -> list[tuple[tuple[str, str], list[dict]]]:
    """Group size-comparison specs by task and model family, preserving order.

    The size panel is intended to compare small/large members of the same
    family under a common intervention phase. Grouping by (task, family) lets
    the selector choose one phase for the whole small/large comparison instead
    of falling back independently for each model.
    """
    groups: list[tuple[tuple[str, str], list[dict]]] = []
    index: dict[tuple[str, str], int] = {}
    for spec in specs:
        family, _size_rank, _size_label = size_family_rank_and_label(spec["model"])
        key = (spec["task"], family)
        if key not in index:
            index[key] = len(groups)
            groups.append((key, []))
        groups[index[key]][1].append(spec)
    return groups


def size_candidate_phases(points: list[PlotPoint], spec: dict) -> set[str]:
    return {
        p.phase
        for p in points
        if p.task == spec["task"]
        and model_template_and_step_match(p.model, spec["model"], spec["step"])
    }


def choose_size_group_phase(
    points: list[PlotPoint],
    specs: list[dict],
    preferred_phase: str,
) -> str | None:
    """Choose one intervention phase shared by all real rows in a size group.

    Dataset-only zero-coverage placeholders have no run directory, so they do
    not constrain the common phase. If every row is dataset-only, use the
    requested phase.
    """
    candidate_sets = [size_candidate_phases(points, spec) for spec in specs]
    nonempty_sets = [s for s in candidate_sets if s]
    if not nonempty_sets:
        return preferred_phase

    common = set.intersection(*nonempty_sets)
    if not common:
        return None
    if preferred_phase in common:
        return preferred_phase
    # Deterministic fallback, but only to a phase shared by the whole pair.
    return "input+output" if "input+output" in common else "decode-only"


def select_size_group_points(
    points: list[PlotPoint],
    filters: Filters,
    specs: list[dict],
    phase: str,
    preferred_baseline: str,
    root: Path | None,
) -> list[PlotPoint | None]:
    selected: list[PlotPoint | None] = []
    for spec in specs:
        p = select_template_point_with_fallback(
            points,
            filters,
            spec["task"],
            spec["model"],
            spec["step"],
            preferred_phase=phase,
            preferred_baseline=preferred_baseline,
            allow_phase_fallback=False,
            allow_baseline_fallback=True,
        )
        if p is None and root is not None:
            p = dataset_score_only_point(root, spec["task"], spec["model"], spec["step"], phase, preferred_baseline)
        selected.append(p)
    return selected


def plot_size_comparison_figure(points: list[PlotPoint], filters: Filters, out: Path, args: argparse.Namespace, root: Path | None = None) -> None:
    preferred_baseline = args.paper_baseline
    preferred_phase = args.paper_size_phase
    rows: list[dict] = []
    records: list[dict[str, object]] = []

    for (task, family), specs in size_comparison_spec_groups(SIZE_COMPARISON_SPECS):
        phase = choose_size_group_phase(points, specs, preferred_phase)
        if phase is None:
            print(
                f"[warn] skipping unpaired size-comparison group: "
                f"{task_fig_label(task)} / {family}; no common intervention phase across requested models"
            )
            for spec in specs:
                row = paper_row("size_comparison", spec["label"], None, preferred_phase)
                row.update({
                    "matched_phase": "",
                    "size_group": f"{task}/{family}",
                    "status": "missing-common-phase",
                })
                rows.append(row)
            continue

        selected = select_size_group_points(points, filters, specs, phase, preferred_baseline, root)
        if any(p is None for p in selected):
            missing_labels = [spec["label"].replace("\n", " ") for spec, p in zip(specs, selected) if p is None]
            print(
                f"[warn] skipping incomplete size-comparison group: "
                f"{task_fig_label(task)} / {family}; missing {', '.join(missing_labels)}"
            )
            for spec, p in zip(specs, selected):
                row = paper_row("size_comparison", spec["label"], p, phase)
                row.update({
                    "matched_phase": phase,
                    "size_group": f"{task}/{family}",
                })
                if p is None:
                    row["status"] = "missing-in-complete-size-pair"
                rows.append(row)
            continue

        if phase != preferred_phase:
            print(
                f"[size comparison] {task_fig_label(task)} / {family}: "
                f"using paired {phase_label(phase)} instead of requested {phase_label(preferred_phase)}"
            )

        for spec, p in zip(specs, selected):
            assert p is not None
            row = paper_row("size_comparison", spec["label"], p, phase)
            row.update({
                "matched_phase": phase,
                "matched_baseline": p.baseline,
                "size_group": f"{task}/{family}",
            })
            rows.append(row)

            family_actual, size_rank, size_label = size_family_rank_and_label(p.model)
            records.append({
                "task": spec["task"],
                "task_label": task_fig_label(spec["task"]),
                "family": family_actual,
                "family_order": 0 if family_actual == "Qwen2" else (1 if family_actual == "Pythia" else 2),
                "size_rank": size_rank,
                "size_label": size_label,
                "model_label": compact_size_model_label(model_short(p.model)),
                "score": float(p.score),
                "union_rate": float(p.union_rate),
                "status": p.status,
                "phase": p.phase,
                "baseline": p.baseline,
            })

    if not records:
        raise RuntimeError("no paired points found for size-comparison figure")

    task_order = ["arithmetic", "hans_nli", "bon_jailbreaking"]
    grouped_records = [
        (task, sorted([r for r in records if r["task"] == task], key=lambda r: (int(r["family_order"]), int(r["size_rank"]))))
        for task in task_order
        if any(r["task"] == task for r in records)
    ]

    if args.paper_size_plot == "dumbbell":
        with plt.rc_context({
            # Larger text/marks for the fixed-size fig_size_comparison panel.
            # The canvas stays CHECKPOINT_AND_SIZE_FIGSIZE so this can sit next
            # to the checkpoint trajectory without rescaling.
            "font.size": 10.6,
            "axes.labelsize": 10.9,
            "axes.titlesize": 11.2,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 9.4,
            "axes.linewidth": 0.82,
            "xtick.major.width": 0.72,
            "ytick.major.width": 0.72,
            "xtick.major.size": 3.0,
            "ytick.major.size": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }):
            fig, axes = plt.subplots(
                nrows=1,
                ncols=len(grouped_records),
                figsize=CHECKPOINT_AND_SIZE_FIGSIZE,
                sharex=True,
            )
            axes_list = np.atleast_1d(axes).ravel().tolist()
            score_offset = -0.18
            union_offset = 0.18

            for ax, (task, subset) in zip(axes_list, grouped_records):
                families = sorted(
                    {str(r["family"]) for r in subset},
                    key=lambda fam: 0 if fam == "Qwen2" else (1 if fam == "Pythia" else 2),
                )
                family_to_y = {fam: float(i) for i, fam in enumerate(families)}

                for fam in families:
                    fam_rows = sorted(
                        [r for r in subset if r["family"] == fam],
                        key=lambda r: int(r["size_rank"])
                    )
                    score_rows = [
                        (float(r["score"]), str(r["size_label"]), int(r["size_rank"]))
                        for r in fam_rows
                    ]
                    union_rows = [
                        (float(r["union_rate"]), str(r["size_label"]), int(r["size_rank"]))
                        for r in fam_rows
                    ]
                    y0 = family_to_y[fam]

                    if len(score_rows) >= 2:
                        xs = [score_rows[0][0], score_rows[-1][0]]
                        ax.plot(
                            xs,
                            [y0 + score_offset, y0 + score_offset],
                            color="0.30",
                            linewidth=1.55,
                            zorder=1,
                        )
                    if len(union_rows) >= 2:
                        xu = [union_rows[0][0], union_rows[-1][0]]
                        ax.plot(
                            xu,
                            [y0 + union_offset, y0 + union_offset],
                            color="0.55",
                            linewidth=1.55,
                            zorder=1,
                        )

                    for x, size_label, size_rank in score_rows:
                        ax.scatter([x], [y0 + score_offset], s=70, marker="o", color="0.15", zorder=3)
                        ha = "right" if x > 0.86 else "left"
                        dx = (-0.011 - 0.016 * size_rank) if ha == "right" else (0.011 + 0.016 * size_rank)
                        ax.text(
                            x + dx,
                            y0 + score_offset - 0.038,
                            size_label,
                            va="center",
                            ha=ha,
                            fontsize=9.6,
                            color="0.15",
                        )

                    for x, size_label, size_rank in union_rows:
                        ax.scatter([x], [y0 + union_offset], s=70, marker="s", color="0.45", zorder=3)
                        ha = "right" if x > 0.86 else "left"
                        dx = (-0.011 - 0.016 * size_rank) if ha == "right" else (0.011 + 0.016 * size_rank)
                        ax.text(
                            x + dx,
                            y0 + union_offset + 0.038,
                            size_label,
                            va="center",
                            ha=ha,
                            fontsize=9.6,
                            color="0.35",
                        )

                ax.set_yticks([family_to_y[fam] for fam in families])
                ax.set_yticklabels(families)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.42, len(families) - 0.58 if len(families) > 1 else 0.42)
                ax.invert_yaxis()
                ax.set_title(task_fig_label(task), loc="left", pad=1.9)
                ax.grid(axis="x", alpha=0.35, linewidth=0.45)
                ax.grid(axis="y", visible=False)
                ax.tick_params(axis="y", pad=1.3)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)

            legend_handles = [
                Line2D([0], [0], marker="o", linestyle="none", color="0.15", markersize=7.3, label="score"),
                Line2D([0], [0], marker="s", linestyle="none", color="0.45", markersize=7.5, label=r"coverage $U(J)$"),
                Line2D([0], [0], linestyle="-", color="0.35", linewidth=1.55, label="small → large"),
            ]
            fig.legend(
                handles=legend_handles,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=3,
                handlelength=1.35,
                columnspacing=0.52,
                borderaxespad=0.0,
            )

            center_axis = axes_list[len(axes_list) // 2]
            center_axis.set_xlabel("Rate", labelpad=1.0)

            fig.subplots_adjust(
                left=0.070,
                right=0.995,
                bottom=0.215,
                top=0.775,
                wspace=0.10,
            )
            save_outputs(fig, out, args)
    else:
        with plt.rc_context({
            # Larger text/marks for the fixed-size fig_size_comparison panel.
            # The canvas stays CHECKPOINT_AND_SIZE_FIGSIZE so this can sit next
            # to the checkpoint trajectory without rescaling.
            "font.size": 10.6,
            "axes.labelsize": 10.9,
            "axes.titlesize": 11.2,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 9.4,
            "axes.linewidth": 0.82,
            "xtick.major.width": 0.72,
            "ytick.major.width": 0.72,
            "xtick.major.size": 3.0,
            "ytick.major.size": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }):
            # Keep short panels visually tight without widening their bars:
            # panels get width proportional to the number of model categories.
            # Thus the two-category Jailbreak panel is physically narrower than
            # the four-category panels, while all bars retain the same pixel width.
            width_ratios = [max(1, len(subset)) for _task, subset in grouped_records]
            fig, axes = plt.subplots(
                nrows=1,
                ncols=len(grouped_records),
                figsize=CHECKPOINT_AND_SIZE_FIGSIZE,
                sharey=True,
                gridspec_kw={"width_ratios": width_ratios},
            )
            axes_list = np.atleast_1d(axes).ravel().tolist()
            bar_width = 0.50

            def _compact_rate_label(value: float) -> str:
                """Return compact bar labels: .55 instead of 0.55; omit zeros."""
                if not np.isfinite(value) or abs(value) < 0.005:
                    return ""
                label = f"{value:.2f}"
                if label.startswith("0."):
                    label = label[1:]
                elif label.startswith("-0."):
                    label = "-" + label[2:]
                return label

            for ax, (task, subset) in zip(axes_list, grouped_records):
                subset = sorted(
                    subset,
                    key=lambda r: (int(r["family_order"]), int(r["size_rank"]))
                )

                x = np.arange(len(subset), dtype=float)
                scores = [float(r["score"]) for r in subset]
                unions = [float(r["union_rate"]) for r in subset]

                # more compact x labels than the full model_label
                tick_labels = []
                for r in subset:
                    fam = str(r["family"])
                    size = str(r["size_label"])
                    if fam == "Qwen2":
                        fam_short = "Qw2"
                    elif fam == "Pythia":
                        fam_short = "Py"
                    else:
                        fam_short = fam
                    tick_labels.append(f"{fam_short}\n{size}")

                score_bars = ax.bar(
                    x - bar_width / 2,
                    scores,
                    width=bar_width,
                    label="task score",
                    linewidth=0.42,
                )
                coverage_bars = ax.bar(
                    x + bar_width / 2,
                    unions,
                    width=bar_width,
                    label=r"coverage $U(J)$",
                    linewidth=0.42,
                )

                # Label each model pair once when score and coverage round to
                # the same value; otherwise label the two bars separately. This
                # avoids collisions such as 0.02 0.02 and keeps labels compact.
                for i, (score_rect, coverage_rect, score_val, union_val) in enumerate(zip(score_bars, coverage_bars, scores, unions)):
                    score_label = _compact_rate_label(score_val)
                    union_label = _compact_rate_label(union_val)
                    if score_label and score_label == union_label:
                        ax.text(
                            x[i],
                            min(max(score_val, union_val) + 0.020, 1.055),
                            score_label,
                            ha="center",
                            va="bottom",
                            fontsize=7.5,
                            color="0.18",
                            bbox=dict(boxstyle="round,pad=0.06", facecolor="white", edgecolor="none", alpha=0.88),
                            clip_on=False,
                            zorder=5,
                        )
                    else:
                        if score_label:
                            ax.text(
                                score_rect.get_x() + score_rect.get_width() / 2,
                                min(score_val + 0.018, 1.055),
                                score_label,
                                ha="center",
                                va="bottom",
                                fontsize=7.5,
                                color="0.10",
                                bbox=dict(boxstyle="round,pad=0.06", facecolor="white", edgecolor="none", alpha=0.88),
                                clip_on=False,
                                zorder=5,
                            )
                        if union_label:
                            ax.text(
                                coverage_rect.get_x() + coverage_rect.get_width() / 2,
                                min(union_val + 0.018, 1.055),
                                union_label,
                                ha="center",
                                va="bottom",
                                fontsize=7.5,
                                color="0.25",
                                bbox=dict(boxstyle="round,pad=0.06", facecolor="white", edgecolor="none", alpha=0.88),
                                clip_on=False,
                                zorder=5,
                            )

                ax.set_xticks(x)
                ax.set_xticklabels(tick_labels)
                ax.set_xlim(float(x[0]) - 0.72, float(x[-1]) + 0.72)
                ax.set_ylim(0.0, 1.08)
                ax.set_title(task_fig_label(task), loc="left", pad=1.9)
                ax.grid(axis="y", alpha=0.38, linewidth=0.45)
                ax.grid(axis="x", visible=False)
                ax.tick_params(axis="x", pad=1.0)
                ax.tick_params(axis="y", pad=1.3)

                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)

            legend_handles, legend_labels = axes_list[0].get_legend_handles_labels()
            fig.legend(
                legend_handles,
                legend_labels,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=2,
                handlelength=1.35,
                columnspacing=0.55,
                borderaxespad=0.0,
            )

            axes_list[0].set_ylabel("Rate", labelpad=1.0)

            fig.subplots_adjust(
                left=0.070,
                right=0.995,
                bottom=0.215,
                top=0.785,
                wspace=0.10,
            )
            save_outputs(fig, out, args)
    if not args.no_csv:
        write_rows_csv(rows, out)
    print(f"[OK] wrote {out}")
    if not args.no_csv:
        print(f"[OK] wrote {(out.with_suffix('.csv') if args.csv_out_dir is None else Path(args.csv_out_dir) / out.with_suffix('.csv').name)}")


def make_paper_figures(root: Path, filters: Filters, args: argparse.Namespace) -> None:
    paper_filters = paper_filters_from(filters, include_empty=args.paper_include_empty)
    paper_points = discover_points(root, paper_filters, dedupe=False)
    paper_points = transform_points_for_coverage_metric(root, paper_points, args.coverage_metric)
    if not paper_points:
        raise RuntimeError("no points available for paper figures after filtering")

    out_dir = Path(args.paper_figures_dir) if args.paper_figures_dir is not None else Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.paper_figures)
    if "all" in requested:
        requested = {"phase", "checkpoint", "size"}

    if "phase" in requested:
        plot_phase_comparison_figure(paper_points, paper_filters, out_dir / "fig_phase_comparison.pdf", args)
    if "checkpoint" in requested:
        checkpoint_filters = paper_filters_from(filters, include_empty=True)
        checkpoint_points = discover_points(root, checkpoint_filters, dedupe=False)
        checkpoint_points = ensure_checkpoint_dataset_score_points(
            root,
            checkpoint_points,
            phase=args.paper_checkpoint_phase,
            baseline=args.paper_baseline,
        )
        checkpoint_points = transform_points_for_coverage_metric(root, checkpoint_points, args.coverage_metric)
        if not checkpoint_points:
            raise RuntimeError("no Pythia checkpoint points available for checkpoint trajectory")
        plot_checkpoint_trajectory_figure(checkpoint_points, checkpoint_filters, out_dir / args.paper_checkpoint_filename, args)
    if "size" in requested:
        plot_size_comparison_figure(paper_points, paper_filters, out_dir / "fig_size_comparison.pdf", args, root=root)

def plot_single(points: list[PlotPoint], out: Path, args: argparse.Namespace) -> None:
    colors = task_color_map(points)
    fig, ax = plt.subplots(figsize=paper_figsize(args, "single"))
    draw_pair_lines(ax, points, colors, args.connect_pairs)
    scatter_points(ax, points, colors, args)
    trend_handles = draw_trend(ax, points, args.trend, show_r2=args.show_r2, args=args)
    decorate_axis(ax, tick_step=args.tick_step)
    annotate_fit_stats(ax, points, args)
    annotate_points(
        ax,
        points,
        args.label_points,
        args.label_mode,
        args.label_max,
        aggregate_labels=not args.no_aggregate_labels,
        label_cluster_px=args.label_cluster_px,
        label_fontsize=args.label_size,
        label_pad_px=args.label_pad_px,
        label_axis_inset_px=args.label_axis_inset_px,
    )
    ax.set_xlabel(x_axis_label(points), labelpad=1.0)
    ax.set_ylabel(coverage_metric_label(args.coverage_metric), labelpad=1.0)
    build_legend(fig, ax, points, colors, trend_handles, args, include_tasks=True)
    fig.subplots_adjust(**subplot_margins(args, "single"))
    save_outputs(fig, out, args)


def plot_task_grid(points: list[PlotPoint], out: Path, args: argparse.Namespace) -> None:
    tasks = [t for t in TASK_ORDER if any(p.task == t for p in points)]
    tasks += sorted({p.task for p in points if p.task not in tasks})
    n = len(tasks)
    if args.paper_size == "single":
        ncols = 2 if n > 1 else 1
    else:
        ncols = 3 if n > 2 else n
    nrows = int(math.ceil(n / ncols))
    colors = task_color_map(points)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=paper_figsize(args, "task-grid", nrows, ncols), sharex=True, sharey=True)
    axes_list = np.atleast_1d(axes).ravel().tolist()

    all_trend_handles: list[Line2D] = []
    for ax, task in zip(axes_list, tasks):
        subset = [p for p in points if p.task == task]
        draw_pair_lines(ax, subset, colors, args.connect_pairs)
        scatter_points(ax, subset, colors, args)
        if args.trend != "overall" or len(subset) >= 3:
            all_trend_handles.extend(draw_trend(ax, subset, args.trend, show_r2=args.show_r2, args=args))
        decorate_axis(ax, TASK_LABELS.get(task, task), tick_step=args.tick_step)
        annotate_fit_stats(ax, subset, args)
        if args.facet_labels != "none":
            # Within facets, model labels are short enough for selected points.
            old_label_mode = args.label_mode
            annotate_points(ax, subset, args.facet_labels, "model" if old_label_mode == "auto" else old_label_mode, args.facet_label_max, aggregate_labels=not args.no_aggregate_labels, label_cluster_px=args.label_cluster_px, label_fontsize=args.label_size, label_pad_px=args.label_pad_px, label_axis_inset_px=args.label_axis_inset_px)
    for ax in axes_list[len(tasks):]:
        ax.axis("off")

    for ax in axes_list[-ncols:]:
        if ax.has_data():
            ax.set_xlabel(x_axis_label(points), labelpad=1.0)
    for ax in axes_list[::ncols]:
        ax.set_ylabel(coverage_metric_label(args.coverage_metric), labelpad=1.0)

    # Faceted plots may draw one OLS fit per panel. Listing every panel-specific
    # R^2 in the legend recreates the clutter we are trying to avoid, so the
    # legend uses a generic trend handle. The exact fitted values remain in the
    # sibling CSV if needed.
    if all_trend_handles and args.trend == "overall":
        trend_for_legend = [Line2D([0], [0], color="0.15", linestyle="--", linewidth=0.8, label="score-coverage fit")]
    elif all_trend_handles and args.trend == "by-phase":
        trend_for_legend = [
            Line2D([0], [0], color="0.20", linestyle="--", linewidth=0.8, label="decode fit"),
            Line2D([0], [0], color="0.55", linestyle="-.", linewidth=0.8, label="in+out fit"),
        ]
    else:
        trend_for_legend = []
    # In faceted plots the panel titles already encode the task, so the legend
    # only needs phase/baseline/trend information. Place it in figure
    # coordinates to avoid covering the panels.
    build_legend(fig, axes_list[0], points, colors, trend_for_legend, args, include_tasks=False)
    fig.subplots_adjust(**subplot_margins(args, "task-grid"))
    save_outputs(fig, out, args)


def plot_phase_panels(points: list[PlotPoint], out: Path, args: argparse.Namespace) -> None:
    phase_order = [("input+output", "Input+output"), ("decode-only", "Output-only")]
    panel_points = [([p for p in points if p.phase == phase], title) for phase, title in phase_order]
    panel_points = [(subset, title) for subset, title in panel_points if subset]
    if not panel_points:
        raise RuntimeError("no points remain for phase-panels layout")

    colors = task_color_map(points)
    ncols = len(panel_points)
    fig, axes = plt.subplots(1, ncols, figsize=paper_figsize(args, "phase-panels", 1, ncols), sharex=True, sharey=True)
    axes_list = np.atleast_1d(axes).ravel().tolist()

    density_scale = _phase_panel_y_scale(points, args.coverage_metric)
    y_max, y_tick_step = density_scale if density_scale is not None else (1.0, None)

    any_trend = False
    for ax, (subset, title) in zip(axes_list, panel_points):
        scatter_points(ax, subset, colors, args)
        if args.trend != "none" and len(subset) >= 2:
            draw_trend(ax, subset, "overall", show_r2=args.show_r2, args=args)
            any_trend = True
        decorate_axis(ax, title, tick_step=args.tick_step, y_max=y_max, y_tick_step=y_tick_step)
        annotate_fit_stats(ax, subset, args)
        annotate_points(
            ax,
            subset,
            args.label_points,
            args.label_mode,
            args.label_max,
            aggregate_labels=not args.no_aggregate_labels,
            label_cluster_px=args.label_cluster_px,
            label_fontsize=args.label_size,
            label_pad_px=args.label_pad_px,
            label_axis_inset_px=args.label_axis_inset_px,
        )
        ax.set_xlabel(x_axis_label(subset), labelpad=1.0)
    axes_list[0].set_ylabel(coverage_metric_label(args.coverage_metric), labelpad=1.0)

    trend_for_legend = [Line2D([0], [0], color="0.15", linestyle="--", linewidth=0.8, label="score-coverage fit")] if any_trend else []
    build_legend(fig, axes_list[0], points, colors, trend_for_legend, args, include_tasks=True)
    fig.subplots_adjust(**subplot_margins(args, "phase-panels"))
    save_outputs(fig, out, args)


def save_outputs(fig, out: Path, args: argparse.Namespace) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    bbox = None if args.no_tight_bbox else "tight"
    fig.savefig(out, bbox_inches=bbox, pad_inches=args.pad_inches)
    plt.close(fig)


def write_csv(points: list[PlotPoint], out: Path, csv_out_dir: str | Path | None = None) -> None:
    csv_path = out.with_suffix(".csv") if csv_out_dir is None else Path(csv_out_dir) / out.with_suffix(".csv").name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(points[0]).keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in points:
            w.writerow(asdict(p))


def write_phase_fit_stats(points: list[PlotPoint], out: Path, coverage_metric: str, csv_out_dir: str | Path | None = None) -> None:
    """Write fit statistics for the same all-settings sample shown in the PDF.

    The figure CSV and fit-stat CSV must describe the same configured-study
    population. Metric-specific missingness is represented explicitly rather
    than by switching to an execution subset.
    """
    base = out.with_suffix(".csv") if csv_out_dir is None else Path(csv_out_dir) / out.with_suffix(".csv").name
    stats_path = base.with_name(base.stem + "_stats.csv")
    rows = []
    for phase in ("input+output", "decode-only"):
        phase_points = [p for p in points if p.phase == phase and math.isfinite(p.score)]
        subset = [p for p in phase_points if math.isfinite(p.union_rate)]
        st = regression_stats(subset)
        row = {
            "phase": phase,
            "n": len(subset),
            "n_defined": len(subset),
            "n_settings_total": len(phase_points),
            "n_undefined_directional_metric": len(phase_points) - len(subset),
            "coverage_metric": str(coverage_metric),
            "pearson_r": math.nan,
            "pearson_p": math.nan,
            "spearman_rho": math.nan,
            "spearman_p": math.nan,
            "ols_slope": math.nan,
            "ols_intercept": math.nan,
            "ols_r2": math.nan,
        }
        if st is not None:
            row.update({
                "pearson_r": st["pearson_r"],
                "pearson_p": st["pearson_p"],
                "spearman_rho": st["spearman_rho"],
                "spearman_p": st["spearman_p"],
                "ols_slope": st["slope"],
                "ols_intercept": st["intercept"],
                "ols_r2": st["r2"],
            })
        rows.append(row)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def write_final_snapshot_fit_stats(points: list[PlotPoint], out: Path, coverage_metric: str, csv_out_dir: str | Path | None = None) -> None:
    """Write the RQ1 final-snapshot sensitivity alongside the main canonical fit.

    The sensitivity mirrors the 29-cell final-snapshot design: remove
    intermediate Pythia checkpoints, then retain one replacement condition per
    task/model/phase cell. When both mean-donor and mean are available, prefer
    mean-donor; otherwise retain the configured mean (including normalized
    mean-positional large-model runs).
    """
    final_points = [p for p in points if "@step" not in str(p.model)]
    selected: dict[tuple[str, str, str], PlotPoint] = {}
    for point in final_points:
        key = (str(point.task), str(point.model), str(point.phase))
        current = selected.get(key)
        if current is None or (point.baseline == "mean-donor" and current.baseline != "mean-donor"):
            selected[key] = point
    kept = list(selected.values())
    base = out.with_suffix(".csv") if csv_out_dir is None else Path(csv_out_dir) / out.with_suffix(".csv").name
    stats_path = base.with_name(base.stem + "_final_snapshot_stats.csv")
    rows = []
    for phase in ("input+output", "decode-only"):
        phase_points = [p for p in kept if p.phase == phase and math.isfinite(p.score)]
        subset = [p for p in phase_points if math.isfinite(p.union_rate)]
        st = regression_stats(subset)
        row = {
            "phase": phase,
            "n": len(subset),
            "coverage_metric": str(coverage_metric),
            "selection": "final snapshot; one baseline per task/model/phase; prefer mean-donor",
            "pearson_r": math.nan,
            "pearson_p": math.nan,
            "spearman_rho": math.nan,
            "spearman_p": math.nan,
        }
        if st is not None:
            row.update({
                "pearson_r": st["pearson_r"],
                "pearson_p": st["pearson_p"],
                "spearman_rho": st["spearman_rho"],
                "spearman_p": st["spearman_p"],
            })
        rows.append(row)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def parse_csv_set(value: str | None) -> set[str] | None:
    if value is None or value.lower() in {"any", "all", ""}:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 6.8,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.55,
        "lines.linewidth": 0.8,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "data"), help="Path to experiment artifact root. Default: <repo>/data.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output PDF path. Default matches the recommended NeurIPS-ready filename.")
    parser.add_argument("--panel", choices=["all", "compact"], default="all", help="all auto-discovers runs; compact reproduces the small paper panel.")
    parser.add_argument("--layout", choices=["single", "task-grid", "phase-panels"], default="phase-panels", help="single gives one readable scatter; task-grid facets by task; phase-panels puts Input+output and Output-only in side-by-side subplots. Default is phase-panels.")
    parser.add_argument("--paper-size", choices=["single", "wide"], default="wide", help="Compact figure preset: single column or full-width/two-column. Default wide.")
    parser.add_argument("--score-mode", choices=["phase-specific", "raw", "chance-normalized"], default="phase-specific", help="X-axis task score convention. phase-specific chance-corrects finite-answer output-only scores because prompt/instruction processing is upstream of that intervention and residual successes can include guessing; input+output uses raw success because prompt processing lies inside the intervention window. raw and chance-normalized force one convention in all phases. Safe-refusal points always use safe-refusal rate.")
    parser.add_argument("--fig-width", type=float, default=None, help="Override figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=None, help="Override figure height in inches.")
    parser.add_argument("--legend-position", choices=["bottom", "inside", "right", "none"], default="bottom", help="Legend placement. Bottom is the compact paper default.")
    parser.add_argument("--legend-cols", type=int, default=None, help="Override number of legend columns.")
    parser.add_argument("--long-legend-labels", action="store_true", help="Use full phase/baseline names in the legend.")
    parser.add_argument("--tick-step", type=float, default=0.3, help="Tick spacing. Default 0.3 for the recommended compact paper figure.")
    parser.add_argument("--tasks", default="any", help="Comma-separated task names, or any.")
    parser.add_argument("--orgs", default="any", help="Comma-separated org names, or any.")
    parser.add_argument("--model-regex", default=None, help="Regex over model directory names.")
    parser.add_argument("--phase", default="any", help="Comma-separated: decode-only,input+output, or any.")
    parser.add_argument("--show-only", choices=["both", "decode-only", "input+output"], default="both", help="Convenience alias to show only one intervention phase.")
    parser.add_argument("--baseline", default="any", help="Comma-separated: mean-donor,mean, or any.")
    parser.add_argument("--exclude-mean", action="store_true", help="Exclude mean-ablation runs and keep only mean-donor runs.")
    parser.add_argument("--exclude-checkpoints", action="store_true", help="Exclude checkpoint models whose directory name contains @step. Enabled by default." )
    parser.add_argument("--include-checkpoints", action="store_false", default=True, dest="exclude_checkpoints", help="Include checkpoint models whose directory name contains @step.")
    parser.add_argument("--anchor", choices=["random_anchor", "spectral_anchor", "unspecified", "any"], default="random_anchor", help="Anchor regime to plot.")
    parser.add_argument("--name-contains", action="append", default=None, help="Required substring in run names. Repeatable. Default: spectral. Use spectral_sample for literal spectral_sample dirs only; use any to disable substring filtering.")
    parser.add_argument("--include-fake-targets", action="store_true", help="Include fake_targets runs. Default excludes them.")
    parser.add_argument("--m", default="any", help="Required M token. Default any. Example: 200000.")
    parser.add_argument("--tau", default="any", help="Required tau token. Default any. Example: 0.3.")
    parser.add_argument("--prefer-m", default="200000", help="Preferred M token when deduplicating. Default 200000.")
    parser.add_argument("--prefer-tau", default="0.3", help="Preferred tau token when deduplicating. Default 0.3.")
    parser.add_argument("--no-dedupe", action="store_true", help="Plot every matching run directory instead of one best run per task/model/phase/baseline.")
    parser.add_argument("--no-include-empty", action="store_true", help="Do not plot empty stats directories as zero-agonist points.")
    parser.add_argument("--label-mode", choices=["auto", "task", "model", "task+model", "none"], default="model", help="Text content used when labels are enabled. Default is model.")
    parser.add_argument("--label-points", choices=["auto", "paired", "all", "input+output", "decode-only", "none"], default="paired", help="Which points receive labels. Default is paired.")
    parser.add_argument("--label-max", type=int, default=20, help="Maximum number of label groups in a single-panel plot. Default is 100 for the recommended paired-label figure.")
    parser.add_argument("--no-aggregate-labels", action="store_true", help="Disable label aggregation. By default, nearby labels are merged and model names are shown one per line.")
    parser.add_argument("--label-cluster-px", type=float, default=50.0, help="Pixel distance below which labels are aggregated. Increase this if model labels still visually collide.")
    parser.add_argument("--label-size", type=float, default=5.4, help="Annotation font size.")
    parser.add_argument("--label-pad-px", type=float, default=0.5, help="Display-space padding reserved around placed labels when avoiding overlaps.")
    parser.add_argument("--label-axis-inset-px", type=float, default=2.0, help="Keep label bounding boxes this many pixels inside each axes boundary. A 1 px hard minimum is enforced even if set to 0.")
    parser.add_argument("--facet-labels", choices=["auto", "paired", "all", "input+output", "decode-only", "none"], default="none", help="Labels used inside --layout task-grid facets. Default none because dense facets are otherwise unreadable.")
    parser.add_argument("--facet-label-max", type=int, default=2, help="Maximum number of labels per task facet.")
    parser.add_argument("--connect-pairs", choices=["all", "mean-donor", "none"], default="all", help="Draw lines between decode-only and input+output paired points.")
    parser.add_argument("--trend", choices=["none", "overall", "by-phase"], default="overall", help="Linear OLS trend line to draw.")
    parser.add_argument("--no-trend-mask-points", action="store_true", help="Draw regression trend lines continuously through point markers. Default leaves small gaps around markers for readability.")
    parser.add_argument("--trend-mask-gap-px", type=float, default=4.5, help="Extra display-space gap, in pixels, reserved around markers when drawing regression trend lines. Default 4.5.")
    parser.add_argument("--show-r2", action="store_true", help="Include R^2 values in trend-line legend labels. Default hides them to keep legends compact.")
    parser.add_argument("--no-fit-stats", action="store_true", help="Hide the in-panel regression statistics annotation. By default, each competence-vs-coverage panel reports n, Pearson r, p-value, OLS equation, and R^2. Finite-answer output-only x-values are chance-normalized; input+output x-values are raw parsed task scores. Jailbreak/refusal x-values are safe-refusal rates, with stored jailbreak rates flipped before plotting.")
    parser.add_argument("--fit-stats-size", type=float, default=5.6, help="Font size for the in-panel regression statistics annotation.")
    parser.add_argument("--fit-stats-obstacle-pad-px", type=float, default=9.0, help="Display-space clearance reserved around the regression statistics box when placing model labels.")
    parser.add_argument("--size-mode", choices=["fixed", "neurons"], default="fixed", help="Use fixed marker sizes or scale by number of localized neurons.")
    parser.add_argument("--marker-size", type=float, default=24.0, help="Marker area in pt^2 for --size-mode fixed. Default 24 for the recommended compact figure.")
    parser.add_argument("--marker-max", type=float, default=120.0, help="Maximum marker area for --size-mode neurons. Original script used 280.")
    parser.add_argument("--marker-neuron-scale", type=float, default=7.5, help="Neuron marker scaling factor for --size-mode neurons.")
    parser.add_argument("--pad-inches", type=float, default=0.01, help="Padding used with tight bounding boxes.")
    parser.add_argument("--no-tight-bbox", action="store_true", help="Disable bbox_inches='tight' when saving.")
    parser.add_argument("--coverage-metric", choices=["pooled", "i2c", "c2i", "n05-i2c-density", "n05-c2i-density", "n05-i2c-per-1k", "n05-c2i-per-1k"], default="pooled", help="Causal quantity on the y-axis. Directional metrics retain all settings with a defined value; within-setting source-state n is not used as an across-setting exclusion rule.")
    parser.add_argument(
        "--rq1-manuscript-population",
        action="store_true",
        help=(
            "Use the canonical all-settings RQ1 scanner contract "
            "(spectral random-anchor runs, deduplicated by task/model/phase/baseline, "
            "including checkpoints and genuine empty-candidate settings). This is the "
            "strict held-out-test population used by the RQ1 manuscript figure."
        ),
    )
    parser.add_argument(
        "--catalogue-json",
        default=None,
        help="Optional configured_experiments.json used as the exact RQ1 population.",
    )
    parser.add_argument(
        "--allow-incomplete-manuscript-population",
        action="store_true",
        help=(
            "When used with --rq1-manuscript-population, plot the available configured settings "
            "and warn about missing runs/metrics instead of failing. Intended for in-progress "
            "experiment batches; final manuscript generation should omit this flag."
        ),
    )
    parser.add_argument("--csv-out-dir", default=None, help="Optional directory for the plotted-point CSV sidecar, allowing manuscript figure folders to remain PDF-only.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write a CSV with plotted points.")
    parser.add_argument("--paper-figures", nargs="+", choices=["all", "phase", "checkpoint", "size"], default=["all"], help="Also generate publication-style summary figures from real input data. Default: all. Use 'all' for all three templates.")
    parser.add_argument("--no-paper-figures", action="store_const", const=[], dest="paper_figures", help="Do not generate phase/checkpoint/size summary templates; useful when this module is used only as the all-settings scatter engine.")
    parser.add_argument("--only-paper-figures", action="store_true", help="Generate only --paper-figures and skip the default competence-vs-coverage figure.")
    parser.add_argument("--paper-figures-dir", default=str(PROJECT_ROOT / "results" / "paper" / "figures" / "appendix_context"), help="Directory for paper figures. Default: <repo>/results/paper/figures/appendix_context.")
    parser.add_argument("--paper-baseline", choices=["mean-donor", "mean"], default="mean-donor", help="Baseline run family used in paper summary figures. Default mean-donor.")
    parser.add_argument("--no-paper-phase-baseline-fallback", action="store_false", default=True, dest="paper_phase_baseline_fallback", help="For fig_phase_comparison only, require --paper-baseline exactly. By default the requested baseline is preferred, but fallback is allowed only to another baseline that has both input+output and output-only points.")
    parser.add_argument("--paper-checkpoint-phase", choices=["decode-only", "input+output"], default="input+output", help="Intervention phase for the checkpoint trajectory. Default input+output.")
    parser.add_argument("--paper-checkpoint-filename", default="fig5a_pythia_checkpoint_trajectory.pdf", help="Filename for the checkpoint trajectory within --paper-figures-dir.")
    parser.add_argument("--paper-size-phase", choices=["decode-only", "input+output"], default="input+output", help="Preferred intervention phase for fig_size_comparison.pdf. The size panel now uses a single common phase and baseline within each task/family pair; if the preferred phase is not available for the whole pair, the script falls back only to another phase that is shared by every real row in that pair.")
    parser.add_argument("--paper-size-plot", choices=["dumbbell", "bars"], default="bars", help="Plot style for fig_size_comparison.pdf. Default: bars (horizontal bar layout). Dumbbell groups models by family and connects small-to-large model pairs.")
    parser.add_argument("--paper-include-empty", action="store_true", help="For paper summary figures only, include stats run directories that lack flip_stats_global.json as zero-coverage points. Default skips them.")
    return parser.parse_args()


def main() -> None:
    global SCORE_MODE
    args = parse_args()
    SCORE_MODE = args.score_mode
    if args.only_paper_figures and not args.paper_figures:
        args.paper_figures = ["all"]
    if args.paper_figures_dir is not None and args.paper_figures and args.out == DEFAULT_OUT and not args.only_paper_figures:
        args.out = str(Path(args.paper_figures_dir) / DEFAULT_OUT)

    setup_matplotlib()
    root = locate_results_root(Path(args.results_dir))

    phases = parse_csv_set(args.phase)
    if args.show_only != "both":
        phases = {args.show_only}

    baselines = parse_csv_set(args.baseline)
    if args.exclude_mean:
        baselines = {"mean-donor"}

    if args.rq1_manuscript_population:
        filters = rq1_manuscript_filters()
    else:
        filters = Filters(
            tasks=parse_csv_set(args.tasks),
            orgs=parse_csv_set(args.orgs),
            model_regex=re.compile(args.model_regex) if args.model_regex else None,
            phases=phases,
            baselines=baselines,
            anchor=args.anchor,
            name_contains=([] if args.name_contains and any(x.lower() in {"any", "all"} for x in args.name_contains) else (args.name_contains or ["spectral"])),
            exclude_fake_targets=not args.include_fake_targets,
            exclude_checkpoints=args.exclude_checkpoints,
            require_m=None if args.m.lower() in {"any", "all", "none", ""} else args.m,
            require_tau=None if args.tau.lower() in {"any", "all", "none", ""} else args.tau,
            prefer_m=None if args.prefer_m.lower() in {"any", "all", "none", ""} else args.prefer_m,
            prefer_tau=None if args.prefer_tau.lower() in {"any", "all", "none", ""} else args.prefer_tau,
            include_empty=not args.no_include_empty,
        )

    if not args.only_paper_figures:
        if args.panel == "compact":
            points = compact_points(root, filters)
            # Compact figures can tolerate labels.
            if args.label_points == "auto":
                args.label_points = "paired"
        else:
            if args.rq1_manuscript_population:
                points = discover_rq1_manuscript_points(
                    root,
                    allow_incomplete=bool(args.allow_incomplete_manuscript_population),
                    catalogue_json=(Path(args.catalogue_json).expanduser().resolve() if args.catalogue_json else None),
                )
            else:
                points = discover_points(
                    root, filters, dedupe=not args.no_dedupe,
                    heldout_test_only=False,
                )

        if not points:
            raise RuntimeError("no points to plot after filtering")
        points = transform_points_for_coverage_metric(root, points, args.coverage_metric)
        if args.rq1_manuscript_population and args.coverage_metric != "pooled":
            missing_directional = [
                p for p in points
                if math.isfinite(p.score) and not math.isfinite(p.union_rate)
            ]
            if missing_directional:
                detail = "\n".join(
                    f"  - {p.task}/{p.org}/{p.model}: {p.run}"
                    for p in missing_directional[:20]
                )
                message = (
                    "RQ1 manuscript population has missing directional singleton statistics. "
                    "Rebuild directional singleton statistics for the final complete manuscript population.\n" + detail
                )
                if not args.allow_incomplete_manuscript_population:
                    raise RuntimeError(
                        "RQ1 manuscript population has missing directional singleton statistics. "
                        "Refusing to change the plotted population by silently dropping those settings. "
                        "Rebuild directional singleton statistics first.\n" + detail
                    )
                print("[WARNING] " + message.replace("\n", " | "))
        points = [p for p in points if math.isfinite(p.score) and math.isfinite(p.union_rate)]
        if not points:
            raise RuntimeError(f"no finite points for coverage metric {args.coverage_metric}")

        out = Path(args.out).with_suffix(".pdf")
        if args.layout == "task-grid":
            plot_task_grid(points, out, args)
        elif args.layout == "phase-panels":
            plot_phase_panels(points, out, args)
        else:
            plot_single(points, out, args)
        if not args.no_csv:
            write_csv(points, out, args.csv_out_dir)
            write_phase_fit_stats(points, out, args.coverage_metric, args.csv_out_dir)
            if args.rq1_manuscript_population:
                write_final_snapshot_fit_stats(points, out, args.coverage_metric, args.csv_out_dir)

        print(f"[OK] wrote {out}")
        if not args.no_csv:
            print(f"[OK] wrote {(out.with_suffix('.csv') if args.csv_out_dir is None else Path(args.csv_out_dir) / out.with_suffix('.csv').name)}")
        print(f"[points] {len(points)} plotted")
        print("[note] Nearby labels are aggregated by default with one model name per line; labels that cannot be placed cleanly are skipped. Use --layout phase-panels for separate Input+output and Output-only subplots.")
        for p in points:
            print(
                f"  {p.task_label:10s} {model_short(p.model):24s} {p.phase:12s} {p.baseline:10s} "
                f"score={p.score:.4f} union={p.union_rate:.4f} J={p.n_neurons:4d} {p.status} run={p.run}"
            )

    if args.paper_figures:
        make_paper_figures(root, filters, args)


if __name__ == "__main__":
    main()
