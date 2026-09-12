#!/usr/bin/env python3
"""Build the complete configured-study CSV and LaTeX tables.

The default table contains the current configured overtopping intervention settings. Metric
availability is recorded per setting, and metric-specific analyses operate on
applicable observed values. Reported ratios are never clipped.
"""
from __future__ import annotations
from pathlib import Path
import sys
from core.project_paths import PROJECT_ROOT


import argparse
import csv
import math
import pandas as pd
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from studies.overtopping.analysis.lib.files import read_json
from studies.overtopping.analysis.lib.primary_matrix import PRIMARY_PROFILE_CHOICES, normalize_primary_table, write_normalization_audit
from studies.overtopping.analysis.layer_widths import layer_width_for_model, per_1000_layer_coordinates, per_layer_fraction
from studies.overtopping.analysis.lib.task_metrics import chance_baseline, competence, raw_task_score
from studies.overtopping.analysis.lib.interaction_schema import is_exact_interaction_schema
from studies.overtopping.analysis.lib.discovery_artifacts import resolve_stage6_dir, stage6_candidate_count
from studies.overtopping.analysis.lib.stats_resolution import has_stats_artifacts, resolve_available_stats_dir



def locate_results_root(path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Return the directory that contains task dirs such as arithmetic/ and hans_nli/."""
    tmp: Optional[tempfile.TemporaryDirectory] = None
    p = path.expanduser().resolve()
    if p.is_file() and p.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="overtopping_results_")
        with zipfile.ZipFile(p) as zf:
            zf.extractall(tmp.name)
        p = Path(tmp.name)

    candidates = [
        p,
        p / "results",
        p / "results" / "results",
    ]
    for c in candidates:
        if (c / "arithmetic").exists() and (c / "hans_nli").exists():
            return c, tmp

    # Fall back to a bounded search.
    for c in p.rglob("arithmetic"):
        parent = c.parent
        if (parent / "hans_nli").exists():
            return parent, tmp

    if tmp is not None:
        tmp.cleanup()
    raise FileNotFoundError(
        f"Could not locate a results root under {path}. Expected task dirs like arithmetic/ and hans_nli/."
    )


# The analysis table is generated from the complete configured study registry.
# Execution-partition membership is not an analysis filter.
from studies.overtopping.experiments.execution import load_run_specs_json
from studies.overtopping.experiments import run_experiments as experiment_registry

TASK_LABELS = {
    "arithmetic": "Arithmetic",
    "bon_jailbreaking": "Jailbreak",
    "grammar_acceptability": "Grammar",
    "hans_nli": "NLI",
    "random_fsm": "Random FSM",
}
MODEL_LABELS = {
    "EleutherAI/pythia-1b": "Pythia-1B",
    "EleutherAI/pythia-1b@step0": "Pythia-1B 0k",
    "EleutherAI/pythia-1b@step48000": "Pythia-1B 48k",
    "EleutherAI/pythia-1b@step96000": "Pythia-1B 96k",
    "EleutherAI/pythia-6.9b": "Pythia-6.9B",
    "Qwen/Qwen2-1.5B-Instruct": "Qwen2-1.5B",
    "Qwen/Qwen2-7B-Instruct": "Qwen2-7B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
}
# Representative-table selection is presentation-only.  It does not determine
# which settings enter any analysis or metric audit.
REPRESENTATIVE_SETTINGS = {
    ("arithmetic", "EleutherAI/pythia-1b", "decode-only", "mean"),
    ("arithmetic", "EleutherAI/pythia-6.9b", "decode-only", "mean-positional"),
    ("arithmetic", "Qwen/Qwen2-1.5B-Instruct", "decode-only", "mean-donor"),
    ("arithmetic", "Qwen/Qwen2-7B-Instruct", "decode-only", "mean-positional"),
    ("arithmetic", "Qwen/Qwen2.5-1.5B-Instruct", "decode-only", "mean-donor"),
    ("bon_jailbreaking", "Qwen/Qwen2-1.5B-Instruct", "decode-only", "mean-donor"),
    ("bon_jailbreaking", "Qwen/Qwen2.5-1.5B-Instruct", "decode-only", "mean-donor"),
    ("grammar_acceptability", "Qwen/Qwen2.5-1.5B-Instruct", "standard", "mean-donor"),
    ("grammar_acceptability", "Qwen/Qwen2.5-1.5B-Instruct", "decode-only", "mean-donor"),
    ("hans_nli", "Qwen/Qwen2-1.5B-Instruct", "standard", "mean-donor"),
    ("hans_nli", "Qwen/Qwen2.5-1.5B-Instruct", "standard", "mean-donor"),
    ("hans_nli", "Qwen/Qwen2.5-1.5B-Instruct", "decode-only", "mean-donor"),
    ("random_fsm", "EleutherAI/pythia-1b", "standard", "mean-donor"),
    ("random_fsm", "EleutherAI/pythia-1b", "decode-only", "mean-donor"),
}


def _stage6_candidate_bag_dir(stats_dir: Path) -> Optional[Path]:
    """Resolve the configured Stage-6 bag across current and legacy layouts."""
    return resolve_stage6_dir(stats_dir)


def _stage6_candidate_count(stats_dir: Path) -> tuple[Optional[int], str]:
    """Return the exact retained candidate count for a completed Stage-6 run."""
    return stage6_candidate_count(_stage6_candidate_bag_dir(stats_dir))


def _zero_candidate_direction_status(task: str, raw_score: float) -> tuple[float, str, float, str]:
    """Resolve empty-union directional rates when both endpoint classes exist.

    For the finite-answer tasks, the raw score is the binary correctness rate.
    A value strictly between zero and one proves both direction-eligible
    denominators are nonzero.  Jailbreak uses a complemented safety score and
    is intentionally not inferred here.
    """
    finite_tasks = {"arithmetic", "grammar_acceptability", "hans_nli", "random_fsm"}
    if task in finite_tasks and math.isfinite(raw_score):
        if 0.0 < raw_score < 1.0:
            return 0.0, "structural_zero_candidates", 0.0, "structural_zero_candidates"
        if raw_score <= 0.0:
            return 0.0, "structural_zero_candidates", math.nan, "undefined_zero_denominator"
        if raw_score >= 1.0:
            return math.nan, "undefined_zero_denominator", 0.0, "structural_zero_candidates"
    return math.nan, "zero_candidates_direction_denominator_unverified", math.nan, "zero_candidates_direction_denominator_unverified"


def configured_rows(
    profile: str = "configured",
    *,
    catalogue_json: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return metadata rows for the selected registry profile."""
    rows: List[Dict[str, Any]] = []

    if profile == "configured" and catalogue_json is not None:
        specs = load_run_specs_json(catalogue_json)

    elif profile in {"configured", "study-56"}:
        specs = experiment_registry.paper_study_experiments()

    else:
        historical_registry_names = {
            "study-48": "legacy_study_48_experiments",
            "study-44": "storage_protected_experiments",
            "study-39": "legacy_study_39_experiments",
            "iclr-28": "legacy_iclr_28_experiments",
        }

        function_name = historical_registry_names[profile]
        registry_function = getattr(
            experiment_registry,
            function_name,
            None,
        )

        if registry_function is None:
            raise RuntimeError(
                f"Primary profile {profile!r} requires "
                f"{function_name}(), but that historical registry helper "
                "is not available in run_experiments.py. "
                "Use --primary-profile configured or restore the "
                "historical registry helper."
            )

        try:
            specs = registry_function()
        except NameError as exc:
            raise RuntimeError(
                f"Primary profile {profile!r} cannot be constructed "
                f"because its historical registry is incomplete: {exc}"
            ) from exc

    final_cell_counts: Dict[tuple[str, str, str], int] = {}

    for spec in specs:
        if "@step" not in spec.model:
            cell = (spec.task, spec.model, spec.mode)
            final_cell_counts[cell] = (
                final_cell_counts.get(cell, 0) + 1
            )

    for spec in specs:
        key = (
            spec.task,
            spec.model,
            spec.mode,
            spec.intervention,
        )

        if (
            profile in {"configured", "study-56"}
            and spec.suite
            in {"mean-donor", "6-7b-models", "mean", "checkpoints"}
        ):
            component = spec.suite
        else:
            is_checkpoint = "@step" in spec.model
            final_cell = (
                spec.task,
                spec.model,
                spec.mode,
            )
            is_matched_baseline_repeat = (
                not is_checkpoint
                and final_cell_counts.get(final_cell, 0) > 1
                and spec.intervention
                in {"mean", "mean-positional"}
            )

            component = (
                "intermediate_checkpoint"
                if is_checkpoint
                else "replacement_baseline_repeat"
                if is_matched_baseline_repeat
                else "final_snapshot"
            )

        rows.append({
            "task": TASK_LABELS.get(spec.task, spec.task),
            "task_dir": spec.task,
            "model_id": spec.model,
            "model": MODEL_LABELS.get(spec.model, Path(spec.model).name),
            "phase": spec.phase,
            "intervention": spec.intervention,
            "suite": spec.suite,
            "study_component": component,
            "stats_rel": spec.stats_dir(Path(".")),
            "note": "",
            "rep": key in REPRESENTATIVE_SETTINGS,
        })
    return rows



def read_flip_by_neuron(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


TABLE1_TASK_ORDER = ["Arithmetic", "Grammar", "NLI", "Random FSM", "Jailbreak"]
TABLE1_MODEL_ORDER = {
    ("Arithmetic", "Qwen2-1.5B", "Out"): 0,
    ("Arithmetic", "Qwen2-7B", "Out"): 1,
    ("Arithmetic", "Qwen2.5-1.5B", "Out"): 2,
    ("Arithmetic", "Pythia-1B", "Out"): 3,
    ("Arithmetic", "Pythia-6.9B", "Out"): 4,
    ("Grammar", "Qwen2.5-1.5B", "I+O"): 0,
    ("Grammar", "Qwen2.5-1.5B", "Out"): 1,
    ("NLI", "Qwen2-1.5B", "I+O"): 0,
    ("NLI", "Qwen2.5-1.5B", "I+O"): 1,
    ("NLI", "Qwen2.5-1.5B", "Out"): 2,
    ("Random FSM", "Pythia-1B", "I+O"): 0,
    ("Random FSM", "Pythia-1B", "Out"): 1,
    ("Jailbreak", "Qwen2-1.5B", "Out"): 0,
    ("Jailbreak", "Qwen2.5-1.5B", "Out"): 1,
}


def fmt_metric(x: Optional[float], digits: int = 3) -> str:
    """Format probabilities for manuscript tables, e.g. .552, .99, 1."""
    if x is None or not math.isfinite(x):
        return "--"
    eps = 0.5 * 10 ** (-digits)
    if abs(x - 1.0) < eps:
        return "1."
    if abs(x) < eps:
        return ".000" if digits == 3 else ".00"
    s = f"{x:.{digits}f}"
    if s.startswith("0"):
        s = s[1:]
    if s.startswith("-0"):
        s = "-" + s[2:]
    return s


def sort_table1_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    task_rank = {t: i for i, t in enumerate(TABLE1_TASK_ORDER)}
    return sorted(
        [r for r in rows if r.get("rep")],
        key=lambda r: (
            task_rank.get(r["task"], 999),
            TABLE1_MODEL_ORDER.get((r["task"], r["model"], r["phase"]), 999),
            r["model"],
            r["phase"],
        ),
    )


def compute_rows(root: Path, empirical_fsm_chance: bool, *, profile: str = "configured", catalogue_json: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Materialize the configured study table with metric-specific availability.

    Missing derived artifacts no longer remove a configured setting from the
    analysis manifest.  They produce explicit missing values/statuses that the
    downstream metric audit can report or, when requested, fail on.
    """
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for spec in configured_rows(profile, catalogue_json=catalogue_json):
        model_root = root / spec["task_dir"] / Path(spec["model_id"])
        expected_stats_dir = root / Path(spec["stats_rel"])
        stats_dir, stats_resolution = resolve_available_stats_dir(expected_stats_dir)
        fg_path = model_root / "feature_report" / "dataset_stats.json"
        global_path = stats_dir / "flip_stats_global.json"
        by_neuron_path = stats_dir / "flip_stats_by_neuron.csv"
        singleton_path = stats_dir / "singleton_set_metrics.json"
        interaction_path = stats_dir / "interaction_validation" / "interaction_validation_summary.json"
        stage6_candidate_count, stage6_candidate_status = _stage6_candidate_count(stats_dir)
        structural_zero_candidates = stage6_candidate_status == "ok" and stage6_candidate_count == 0

        # Stage 7 intentionally returns without writing singleton outputs when
        # discovery produced no candidates.  In that case the Stage-6 bag is
        # the completion artifact; do not misclassify the setting as missing.
        required_core_paths = (fg_path,) if structural_zero_candidates else (fg_path, global_path, by_neuron_path)
        missing_core = [p for p in required_core_paths if not p.exists()]
        if stats_resolution == "partial_heldout_cap":
            warnings.append(
                f"{spec['task']} | {spec['model']} | {spec['phase']} | {spec['intervention']}: "
                f"using available partial held-out stats {stats_dir} because canonical stats are not materialized at {expected_stats_dir}."
            )
        if missing_core:
            availability_note = (
                "feature-level stats are available but overtopping metric artifact(s) are unavailable"
                if fg_path.exists() else "required result artifact(s) are unavailable"
            )
            warnings.append(
                f"{spec['task']} | {spec['model']} | {spec['phase']} | {spec['intervention']}: "
                f"{availability_note}: " + ", ".join(str(p) for p in missing_core)
            )

        ds = read_json(fg_path) if fg_path.exists() else {}
        glob = read_json(global_path) if global_path.exists() else {}
        by = read_flip_by_neuron(by_neuron_path) if by_neuron_path.exists() else []

        try:
            raw = raw_task_score(spec["task"], ds)
            chance = chance_baseline(spec["task"], ds, empirical_fsm_chance=empirical_fsm_chance)
            score = competence(spec["phase"], raw, chance)
        except Exception:
            raw = chance = score = math.nan

        n_eval = int(glob.get("n_evaluated_rows", 0) or 0)
        j_raw = glob.get("n_neurons", len(by) if by_neuron_path.exists() else math.nan)
        try:
            j = int(j_raw) if j_raw is not None and math.isfinite(float(j_raw)) else math.nan
        except (TypeError, ValueError):
            j = math.nan
        try:
            u = float(glob.get("union_flip_any_unique_rate", math.nan))
        except (TypeError, ValueError):
            u = math.nan
        if structural_zero_candidates:
            j = 0
            u = 0.0

        singleton = read_json(singleton_path) if singleton_path.exists() else {}
        singleton_schema = singleton.get("definition_version")
        singleton_current = singleton_schema == "heldout-set-metrics-v3-directional"
        if not singleton_current and not structural_zero_candidates:
            warnings.append(
                f"{spec['task']} | {spec['model']} | {spec['phase']} | {spec['intervention']}: "
                f"directional singleton metrics unavailable/incompatible ({singleton_schema!r})."
            )
        interaction = read_json(interaction_path) if interaction_path.exists() else {}

        try:
            top = float(singleton.get("s_1", math.nan))
        except (TypeError, ValueError):
            top = math.nan
        toc1 = None
        topm_path = stats_dir / "frozen_topm_metrics.csv"
        if topm_path.exists():
            try:
                topm = pd.read_csv(topm_path)
                match = topm.loc[pd.to_numeric(topm.get("m"), errors="coerce") == 1]
                toc1 = float(match.iloc[0]["TOC_m"]) if len(match) else None
            except Exception:
                toc1 = None
        elif isinstance(singleton.get("TOC_m"), dict):
            payload = singleton["TOC_m"].get("1", {})
            toc1 = payload.get("value") if isinstance(payload, dict) else payload

        thresholds = singleton.get("N_t", {}) if isinstance(singleton.get("N_t"), dict) else {}
        thresholds_i2c = singleton.get("N_t_i2c", {}) if isinstance(singleton.get("N_t_i2c"), dict) else {}
        thresholds_c2i = singleton.get("N_t_c2i", {}) if isinstance(singleton.get("N_t_c2i"), dict) else {}
        n05 = int(thresholds.get("0.05", 0)) if singleton_current else math.nan
        n10 = int(thresholds.get("0.1", thresholds.get("0.10", 0))) if singleton_current else math.nan
        n05_i2c = int(thresholds_i2c.get("0.05", 0)) if singleton_current and thresholds_i2c else math.nan
        n10_i2c = int(thresholds_i2c.get("0.1", thresholds_i2c.get("0.10", 0))) if singleton_current and thresholds_i2c else math.nan
        n05_c2i = int(thresholds_c2i.get("0.05", 0)) if singleton_current and thresholds_c2i else math.nan
        n10_c2i = int(thresholds_c2i.get("0.1", thresholds_c2i.get("0.10", 0))) if singleton_current and thresholds_c2i else math.nan

        u_i2c = singleton.get("U_J_i2c") if singleton_current else None
        u_c2i = singleton.get("U_J_c2i") if singleton_current else None
        u_i2c_status = singleton.get("U_J_i2c_status", "missing_or_incompatible")
        u_c2i_status = singleton.get("U_J_c2i_status", "missing_or_incompatible")
        if structural_zero_candidates:
            n05 = n10 = n05_i2c = n10_i2c = n05_c2i = n10_c2i = 0
            u_i2c, u_i2c_status, u_c2i, u_c2i_status = _zero_candidate_direction_status(
                spec["task_dir"], float(raw) if raw is not None else math.nan
            )

        layer_width = layer_width_for_model(spec["model"])
        schema = interaction.get("definition_version") if isinstance(interaction, dict) else None
        interaction_current = is_exact_interaction_schema(schema)
        candidate_effect = interaction.get("candidate_E_J", {}) if interaction_current else {}
        interaction_ej = candidate_effect.get("effect") if isinstance(candidate_effect, dict) else None
        cmc_1x = None
        if interaction_current:
            for item in interaction.get("conditional_marginal", []) or []:
                try:
                    multiplier = int(item.get("background_multiplier"))
                except Exception:
                    continue
                if multiplier == 1:
                    cmc_1x = item.get("candidate")
                    break

        rows.append({
            **spec,
            "stats_dir": str(stats_dir),
            "expected_stats_dir": str(expected_stats_dir),
            "stats_resolution": stats_resolution,
            "overtopping_stats_available": bool(has_stats_artifacts(stats_dir)),
            "feature_stats_available": bool(fg_path.exists()),
            "stage6_dir": str(_stage6_candidate_bag_dir(expected_stats_dir) or ""),
            "stage6_discovered_candidate_count": stage6_candidate_count,
            "stage6_candidate_count_status": stage6_candidate_status,
            "structural_zero_candidates": bool(structural_zero_candidates),
            "availability_status": (
                "complete_zero_candidates" if structural_zero_candidates and not missing_core
                else "partial_heldout_stats" if not missing_core and stats_resolution == "partial_heldout_cap"
                else "complete_core" if not missing_core
                else "feature_stats_only" if fg_path.exists() and not has_stats_artifacts(stats_dir)
                else "partial_overtopping_stats" if has_stats_artifacts(stats_dir)
                else "no_result_stats"
            ),
            "singleton_metrics_status": (
                "heldout_v3_directional" if singleton_current else
                "structural_zero_candidates" if structural_zero_candidates else
                "missing_or_incompatible"
            ),
            "interaction_metrics_status": (
                f"exact:{schema}" if interaction_current else
                "not_applicable_zero_candidates" if structural_zero_candidates else
                "missing_or_incompatible"
            ),
            "raw": raw,
            "chance": chance,
            "score": score,
            "n_eval": n_eval,
            "J": j,
            "U": u,
            "Top": top,
            "TOC1": toc1,
            "U_J_i2c": float(u_i2c) if u_i2c is not None else math.nan,
            "U_J_i2c_status": u_i2c_status,
            "U_J_c2i": float(u_c2i) if u_c2i is not None else math.nan,
            "U_J_c2i_status": u_c2i_status,
            "R_ov": singleton.get("R_ov"),
            "R_ov_status": singleton.get("R_ov_status", "undefined_zero_candidates" if structural_zero_candidates else "missing"),
            "R_ov_i2c": singleton.get("R_ov_i2c"),
            "R_ov_c2i": singleton.get("R_ov_c2i"),
            "N_eff": singleton.get("N_eff"),
            "N_eff_status": singleton.get("N_eff_status", "undefined_zero_candidates" if structural_zero_candidates else "missing"),
            "N_eff_i2c": singleton.get("N_eff_i2c"),
            "N_eff_c2i": singleton.get("N_eff_c2i"),
            "s_1_i2c": singleton.get("s_1_i2c"),
            "s_1_c2i": singleton.get("s_1_c2i"),
            "layer_width": layer_width,
            "E_J": interaction_ej,
            "CMC_1x": cmc_1x,
            "N05": n05,
            "N10": n10,
            "N05_i2c": n05_i2c,
            "N10_i2c": n10_i2c,
            "N05_c2i": n05_c2i,
            "N10_c2i": n10_c2i,
            "N05_i2c_density": per_layer_fraction(n05_i2c, spec["model"]) if pd.notna(n05_i2c) else math.nan,
            "N05_c2i_density": per_layer_fraction(n05_c2i, spec["model"]) if pd.notna(n05_c2i) else math.nan,
            "N10_i2c_density": per_layer_fraction(n10_i2c, spec["model"]) if pd.notna(n10_i2c) else math.nan,
            "N10_c2i_density": per_layer_fraction(n10_c2i, spec["model"]) if pd.notna(n10_c2i) else math.nan,
            "N_eff_i2c_density": per_layer_fraction(singleton.get("N_eff_i2c"), spec["model"]),
            "N_eff_c2i_density": per_layer_fraction(singleton.get("N_eff_c2i"), spec["model"]),
            "N05_i2c_per_1k_layer": per_1000_layer_coordinates(n05_i2c, spec["model"]) if pd.notna(n05_i2c) else math.nan,
            "N05_c2i_per_1k_layer": per_1000_layer_coordinates(n05_c2i, spec["model"]) if pd.notna(n05_c2i) else math.nan,
            "N10_i2c_per_1k_layer": per_1000_layer_coordinates(n10_i2c, spec["model"]) if pd.notna(n10_i2c) else math.nan,
            "N10_c2i_per_1k_layer": per_1000_layer_coordinates(n10_c2i, spec["model"]) if pd.notna(n10_c2i) else math.nan,
            "N_eff_i2c_per_1k_layer": per_1000_layer_coordinates(singleton.get("N_eff_i2c"), spec["model"]),
            "N_eff_c2i_per_1k_layer": per_1000_layer_coordinates(singleton.get("N_eff_c2i"), spec["model"]),
        })
    return rows, warnings


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "task", "model", "phase", "intervention", "suite", "study_component",
        "availability_status", "singleton_metrics_status", "interaction_metrics_status",
        "stage6_discovered_candidate_count", "stage6_candidate_count_status",
        "structural_zero_candidates", "stage6_dir",
        "score", "raw", "chance", "J", "U", "Top", "TOC1",
        "U_J_i2c", "U_J_i2c_status", "U_J_c2i", "U_J_c2i_status",
        "R_ov", "R_ov_status", "R_ov_i2c", "R_ov_c2i",
        "N_eff", "N_eff_status", "N_eff_i2c", "N_eff_c2i",
        "s_1_i2c", "s_1_c2i", "layer_width", "E_J", "CMC_1x",
        "N05", "N10", "N05_i2c", "N10_i2c", "N05_c2i", "N10_c2i",
        "N05_i2c_density", "N05_c2i_density", "N10_i2c_density", "N10_c2i_density",
        "N_eff_i2c_density", "N_eff_c2i_density",
        "N05_i2c_per_1k_layer", "N05_c2i_per_1k_layer",
        "N10_i2c_per_1k_layer", "N10_c2i_per_1k_layer",
        "N_eff_i2c_per_1k_layer", "N_eff_c2i_per_1k_layer",
        "n_eval", "stats_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt_decimal(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if not math.isfinite(number) else f"{number:.{digits}f}"


def _fmt_count(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    return str(int(round(number)))


def make_latex(rows: List[Dict[str, Any]], representative_only: bool) -> str:
    """Render direction-first manuscript tables.

    The main table deliberately does not lead with pooled U(J): the competence
    association is directionally asymmetric, so the two conditional union rates
    are the primary reach metrics. Strong-handle counts are reported per 1000
    layer coordinates to avoid raw-count comparisons across different d_model.
    """
    if representative_only:
        rows = sort_table1_rows(rows)
        caption = (
            r"Representative interventions. Competence follows the phase-specific manuscript convention. "
            r"$U_{0\to1}(J)$ and $U_{1\to0}(J)$ are direction-conditioned singleton-union coverage on "
            r"baseline-negative and baseline-positive rows, respectively. "
            r"$N^{0\to1}_{.05}/d$ and $N^{1\to0}_{.05}/d$ are counts of channels with at least 5\% "
            r"direction-conditioned singleton effect, normalized per 1000 residual-stream coordinates "
            r"of one transformer layer ($d=d_{\rm model}$)."
        )
        label = "tab:main-results"
        counts_by_task: Dict[str, int] = {}
        for r in rows:
            counts_by_task[r["task"]] = counts_by_task.get(r["task"], 0) + 1

        lines = [
            r"\begin{table}", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.2pt}",
            r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{@{}llcrrrrr@{}}", r"\toprule",
            r"Task & Model & Ph. & Comp. & $U_{0\to1}(J)$ & $U_{1\to0}(J)$ & $N^{0\to1}_{.05}/d$ & $N^{1\to0}_{.05}/d$ \\",
            r"\midrule",
        ]
        last_task = None
        for r in rows:
            task = r["task"]
            prefix = []
            if task != last_task:
                if last_task is not None:
                    lines.append(r"\midrule")
                prefix.append(rf"\multirow{{{counts_by_task[task]}}}{{*}}{{{latex_escape(task)}}}")
                last_task = task
            else:
                prefix.append("")
            cells = prefix + [
                latex_escape(r["model"]),
                r"I+O" if r["phase"] == "I+O" else "Out",
                fmt_metric(r["score"], 3),
                fmt_metric(r.get("U_J_i2c"), 3),
                fmt_metric(r.get("U_J_c2i"), 3),
                _fmt_decimal(r.get("N05_i2c_density"), 3),
                _fmt_decimal(r.get("N05_c2i_density"), 3),
            ]
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([
            r"\bottomrule", r"\end{tabular}%", r"}",
            r"\caption{" + caption + r"}", r"\label{" + label + r"}", r"\end{table}",
        ])
        return "\n".join(lines) + "\n"

    caption = (
        rf"All {len(rows)} configured study settings. Direction-conditioned union coverage is reported explicitly rather than as OCC. "
        r"Pooled $U(J)$ is retained as context. $N^{d}_{.05}$ and $N^{d}_{.10}$ count strong singleton handles "
        r"within direction $d$; the /$d$ columns use the unitless fraction $N/d_{\rm model}$. Per-1000 variants are retained only in machine-readable sidecars. "
        r"Blank directional-count entries indicate singleton metrics that have not yet been rebuilt with the directional schema."
    )
    label = "tab:appendix-primary-results"
    lines = [
        r"\begin{table}[htb]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
        r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lllrrrrrrrrrr}", r"\toprule",
        r"Model & Task & Ph. & Comp. & $U(J)$ & $U_{0\to1}$ & $U_{1\to0}$ & $N^{0\to1}_{.05}$ & $N^{1\to0}_{.05}$ & $N^{0\to1}_{.05}/d$ & $N^{1\to0}_{.05}/d$ & $N_{\rm eff}^{0\to1}/d$ & $N_{\rm eff}^{1\to0}/d$ \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [
            latex_escape(r["model"]), latex_escape(r["task"]),
            r"I+O" if r["phase"] == "I+O" else "Out", fmt_metric(r["score"], 3),
            fmt_metric(r["U"], 3), fmt_metric(r.get("U_J_i2c"), 3), fmt_metric(r.get("U_J_c2i"), 3),
            _fmt_count(r.get("N05_i2c")), _fmt_count(r.get("N05_c2i")),
            _fmt_decimal(r.get("N05_i2c_density"), 3), _fmt_decimal(r.get("N05_c2i_density"), 3),
            _fmt_decimal(r.get("N_eff_i2c_density"), 3), _fmt_decimal(r.get("N_eff_c2i_density"), 3),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"}",
        r"\caption{" + caption + r"}", r"\label{" + label + r"}", r"\end{table}",
    ])
    return "\n".join(lines) + "\n"

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="Path to extracted results directory or results.zip")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "analysis" / "primary_matrix" / "tables"), help="Output directory. Default: <repo>/results/analysis/primary_matrix/tables")
    ap.add_argument("--empirical-fsm-chance", action="store_true", help="Audit/debug only: use sampled state-count FSM chance instead of the manuscript figure baseline mean(1/3,1/4,1/5,1/6). Do not use for figure-script parity.")
    ap.add_argument(
        "--primary-profile",
        choices=PRIMARY_PROFILE_CHOICES,
        default="configured",
        help="Validation profile for the configured study table. Default: configured (dynamic count).",
    )
    ap.add_argument(
        "--catalogue-json",
        default=None,
        help="Optional configured_experiments.json; with profile=configured this is the exact analysis population.",
    )
    ap.add_argument(
        "--suppress-directional-warnings",
        action="store_true",
        help="Do not print missing-directional warnings to stderr (used only for the pre-rebuild manifest pass).",
    )
    args = ap.parse_args(argv)

    root, tmp = locate_results_root(Path(args.results))
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        catalogue_json = Path(args.catalogue_json).expanduser().resolve() if args.catalogue_json else None
        rows, warnings = compute_rows(
            root, empirical_fsm_chance=args.empirical_fsm_chance, profile=args.primary_profile,
            catalogue_json=catalogue_json,
        )
        normalized_df, excluded_df, profile_audit = normalize_primary_table(
            pd.DataFrame(rows), profile=args.primary_profile, source=Path(args.results)
        )
        rows = normalized_df.to_dict("records")
        write_csv(out / "primary_table.csv", rows)
        write_normalization_audit(
            normalized_df, excluded_df, profile_audit, out, stem="primary_table"
        )
        (out / "table1_representative.tex").write_text(make_latex(rows, representative_only=True), encoding="utf-8")
        (out / "table8_primary.tex").write_text(make_latex(rows, representative_only=False), encoding="utf-8")
        warning_text = "\n".join(warnings) + ("\n" if warnings else "")
        (out / "directional_metrics_warnings.txt").write_text(warning_text or "No directional-metric warnings.\n", encoding="utf-8")

        print(f"Using results root: {root}")
        print(f"Wrote: {out / 'primary_table.csv'}")
        print(f"Wrote: {out / 'table1_representative.tex'}")
        print(f"Wrote: {out / 'table8_primary.tex'}")
        print(f"Wrote: {out / 'directional_metrics_warnings.txt'}")
        if warnings and not args.suppress_directional_warnings:
            print(f"WARNING: {len(warnings)} directional-metric warnings; inspect directional_metrics_warnings.txt", file=sys.stderr)
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
