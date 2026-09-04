#!/usr/bin/env python3
"""Build the canonical 28-setting primary CSV and LaTeX tables.

The primary matrix includes the Qwen2-1.5B input+output NLI row. Inputs must use
the current held-out singleton and conditional-marginal result schemas. Reported
ratios are never clipped.
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


# Canonical primary rows used in the manuscript table.  Paths are relative to the
# results root returned by locate_results_root().
PRIMARY_ROWS: List[Dict[str, Any]] = [
    # Arithmetic
    dict(task="Arithmetic", task_dir="arithmetic", org="EleutherAI", model_dir="pythia-1b", model="Pythia-1B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-agonist_neurons-fast-random_anchor-tau0.3", note="tiny but concentrated flips", rep=True),
    dict(task="Arithmetic", task_dir="arithmetic", org="EleutherAI", model_dir="pythia-1b@step48000", model="Pythia-1B 48k", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Arithmetic", task_dir="arithmetic", org="EleutherAI", model_dir="pythia-6.9b", model="Pythia-6.9B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-decode_only-agonist_neurons-fast-random_anchor", intervention="mean-positional", note="low-coverage overtopping", rep=True),
    dict(task="Arithmetic", task_dir="arithmetic", org="Qwen", model_dir="Qwen2-1.5B-Instruct", model="Qwen2-1.5B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Arithmetic", task_dir="arithmetic", org="Qwen", model_dir="Qwen2-1.5B-Instruct", model="Qwen2-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="overtopped flips", rep=True),
    dict(task="Arithmetic", task_dir="arithmetic", org="Qwen", model_dir="Qwen2-7B-Instruct", model="Qwen2-7B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-decode_only-agonist_neurons-fast-random_anchor", intervention="mean-positional", note="high-coverage overtopping", rep=True),
    dict(task="Arithmetic", task_dir="arithmetic", org="Qwen", model_dir="Qwen2.5-1.5B-Instruct", model="Qwen2.5-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="weak singleton concentration", rep=True),

    # Jailbreak
    dict(task="Jailbreak", task_dir="bon_jailbreaking", org="Qwen", model_dir="Qwen2-1.5B-Instruct", model="Qwen2-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="low singleton concentration", rep=True),
    dict(task="Jailbreak", task_dir="bon_jailbreaking", org="Qwen", model_dir="Qwen2-7B-Instruct", model="Qwen2-7B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-decode_only-agonist_neurons-fast-random_anchor", intervention="mean-positional", note="", rep=False),
    dict(task="Jailbreak", task_dir="bon_jailbreaking", org="Qwen", model_dir="Qwen2.5-1.5B-Instruct", model="Qwen2.5-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="lowest singleton concentration", rep=True),

    # Grammar
    dict(task="Grammar", task_dir="grammar_acceptability", org="EleutherAI", model_dir="pythia-1b", model="Pythia-1B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Grammar", task_dir="grammar_acceptability", org="EleutherAI", model_dir="pythia-1b", model="Pythia-1B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Grammar", task_dir="grammar_acceptability", org="EleutherAI", model_dir="pythia-1b@step48000", model="Pythia-1B 48k", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Grammar", task_dir="grammar_acceptability", org="EleutherAI", model_dir="pythia-1b@step48000", model="Pythia-1B 48k", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Grammar", task_dir="grammar_acceptability", org="EleutherAI", model_dir="pythia-1b@step96000", model="Pythia-1B 96k", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Grammar", task_dir="grammar_acceptability", org="EleutherAI", model_dir="pythia-1b@step96000", model="Pythia-1B 96k", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Grammar", task_dir="grammar_acceptability", org="Qwen", model_dir="Qwen2.5-1.5B-Instruct", model="Qwen2.5-1.5B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="shared flip coverage", rep=True),
    dict(task="Grammar", task_dir="grammar_acceptability", org="Qwen", model_dir="Qwen2.5-1.5B-Instruct", model="Qwen2.5-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="compact overtopped set", rep=True),

    # NLI
    dict(task="NLI", task_dir="hans_nli", org="Qwen", model_dir="Qwen2-1.5B-Instruct", model="Qwen2-1.5B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="moderate interface-localized coverage", rep=True),
    dict(task="NLI", task_dir="hans_nli", org="Qwen", model_dir="Qwen2-7B-Instruct", model="Qwen2-7B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-agonist_neurons-fast-random_anchor", intervention="mean-positional", note="", rep=False),
    dict(task="NLI", task_dir="hans_nli", org="Qwen", model_dir="Qwen2.5-1.5B-Instruct", model="Qwen2.5-1.5B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="near-singleton overtopping", rep=True),
    dict(task="NLI", task_dir="hans_nli", org="Qwen", model_dir="Qwen2.5-1.5B-Instruct", model="Qwen2.5-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="single-channel flips", rep=True),

    # Random FSM
    dict(task="Random FSM", task_dir="random_fsm", org="EleutherAI", model_dir="pythia-1b", model="Pythia-1B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="broad shared coverage", rep=True),
    dict(task="Random FSM", task_dir="random_fsm", org="EleutherAI", model_dir="pythia-1b", model="Pythia-1B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="moderate concentration", rep=True),
    dict(task="Random FSM", task_dir="random_fsm", org="EleutherAI", model_dir="pythia-1b@step48000", model="Pythia-1B 48k", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Random FSM", task_dir="random_fsm", org="EleutherAI", model_dir="pythia-1b@step96000", model="Pythia-1B 96k", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Random FSM", task_dir="random_fsm", org="Qwen", model_dir="Qwen2.5-1.5B-Instruct", model="Qwen2.5-1.5B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Random FSM", task_dir="random_fsm", org="Qwen", model_dir="Qwen2.5-1.5B-Instruct", model="Qwen2.5-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
]



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


def compute_rows(root: Path, empirical_fsm_chance: bool) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for spec in PRIMARY_ROWS:
        model_root = root / spec["task_dir"] / spec["org"] / spec["model_dir"]
        stats_dir = model_root / spec["stats"]
        if not stats_dir.name.endswith("-heldout_test"):
            stats_dir = stats_dir.with_name(stats_dir.name + "-heldout_test")
        fg_path = model_root / "feature_report" / "dataset_stats.json"
        global_path = stats_dir / "flip_stats_global.json"
        by_neuron_path = stats_dir / "flip_stats_by_neuron.csv"

        missing = [str(p) for p in (fg_path, global_path, by_neuron_path) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing expected result files:\n" + "\n".join(missing))

        ds = read_json(fg_path)
        glob = read_json(global_path)
        by = read_flip_by_neuron(by_neuron_path)

        raw = raw_task_score(spec["task"], ds)
        chance = chance_baseline(spec["task"], ds, empirical_fsm_chance=empirical_fsm_chance)
        score = competence(spec["phase"], raw, chance)
        n_eval = int(glob.get("n_evaluated_rows", 0))
        j = int(glob.get("n_neurons", len(by)))
        u = float(glob.get("union_flip_any_unique_rate", 0.0))
        c2i = float(glob.get("union_c2i_unique_rate", 0.0))
        i2c = float(glob.get("union_i2c_unique_rate", 0.0))

        singleton_path = stats_dir / "singleton_set_metrics.json"
        if not singleton_path.exists():
            raise FileNotFoundError(f"Missing current singleton metrics: {singleton_path}")
        singleton = read_json(singleton_path)
        if singleton.get("definition_version") != "heldout-set-metrics-v3-directional":
            raise ValueError(
                f"Unsupported singleton metrics schema at {singleton_path}: "
                f"{singleton.get('definition_version')!r}"
            )
        interaction_path = stats_dir / "interaction_validation" / "interaction_validation_summary.json"
        interaction = read_json(interaction_path) if interaction_path.exists() else {}

        top = float(singleton.get("s_1", math.nan))
        toc1 = None
        topm_path = stats_dir / "frozen_topm_metrics.csv"
        if topm_path.exists():
            topm = pd.read_csv(topm_path)
            match = topm.loc[pd.to_numeric(topm.get("m"), errors="coerce") == 1]
            toc1 = float(match.iloc[0]["TOC_m"]) if len(match) else None
        elif isinstance(singleton.get("TOC_m"), dict):
            payload = singleton["TOC_m"].get("1", {})
            toc1 = payload.get("value") if isinstance(payload, dict) else payload
        thresholds = singleton.get("N_t", {}) if isinstance(singleton.get("N_t"), dict) else {}
        thresholds_i2c = singleton.get("N_t_i2c", {}) if isinstance(singleton.get("N_t_i2c"), dict) else {}
        thresholds_c2i = singleton.get("N_t_c2i", {}) if isinstance(singleton.get("N_t_c2i"), dict) else {}
        n05 = int(thresholds.get("0.05", 0))
        n10 = int(thresholds.get("0.1", thresholds.get("0.10", 0)))
        n05_i2c = int(thresholds_i2c.get("0.05", 0)) if thresholds_i2c else None
        n10_i2c = int(thresholds_i2c.get("0.1", thresholds_i2c.get("0.10", 0))) if thresholds_i2c else None
        n05_c2i = int(thresholds_c2i.get("0.05", 0)) if thresholds_c2i else None
        n10_c2i = int(thresholds_c2i.get("0.1", thresholds_c2i.get("0.10", 0))) if thresholds_c2i else None

        # Explicit direction-conditioned union coverage.
        u_i2c = singleton.get("U_J_i2c")
        u_c2i = singleton.get("U_J_c2i")
        u_i2c_status = singleton.get("U_J_i2c_status")
        u_c2i_status = singleton.get("U_J_c2i_status")
        if singleton.get("definition_version") != "heldout-set-metrics-v3-directional":
            warnings.append(
                f"{spec['task']} | {spec['model']} | {spec['phase']}: directional singleton counts "
                "are unavailable until stats are rebuilt with heldout-set-metrics-v3-directional."
            )

        layer_width = layer_width_for_model(spec["model"])

        schema = interaction.get("definition_version") if isinstance(interaction, dict) else None
        candidate_effect = (
            interaction.get("candidate_E_J", {})
            if is_exact_interaction_schema(schema)
            else {}
        )
        interaction_ej = candidate_effect.get("effect") if isinstance(candidate_effect, dict) else None
        cmc_1x = None
        if is_exact_interaction_schema(schema):
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
            "R_ov_status": singleton.get("R_ov_status"),
            "R_ov_i2c": singleton.get("R_ov_i2c"),
            "R_ov_c2i": singleton.get("R_ov_c2i"),
            "N_eff": singleton.get("N_eff"),
            "N_eff_status": singleton.get("N_eff_status"),
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
            "N05_i2c_density": per_layer_fraction(n05_i2c, spec["model"]) if n05_i2c is not None else math.nan,
            "N05_c2i_density": per_layer_fraction(n05_c2i, spec["model"]) if n05_c2i is not None else math.nan,
            "N10_i2c_density": per_layer_fraction(n10_i2c, spec["model"]) if n10_i2c is not None else math.nan,
            "N10_c2i_density": per_layer_fraction(n10_c2i, spec["model"]) if n10_c2i is not None else math.nan,
            "N_eff_i2c_density": per_layer_fraction(singleton.get("N_eff_i2c"), spec["model"]),
            "N_eff_c2i_density": per_layer_fraction(singleton.get("N_eff_c2i"), spec["model"]),
            "N05_i2c_per_1k_layer": per_1000_layer_coordinates(n05_i2c, spec["model"]) if n05_i2c is not None else math.nan,
            "N05_c2i_per_1k_layer": per_1000_layer_coordinates(n05_c2i, spec["model"]) if n05_c2i is not None else math.nan,
            "N10_i2c_per_1k_layer": per_1000_layer_coordinates(n10_i2c, spec["model"]) if n10_i2c is not None else math.nan,
            "N10_c2i_per_1k_layer": per_1000_layer_coordinates(n10_c2i, spec["model"]) if n10_c2i is not None else math.nan,
            "N_eff_i2c_per_1k_layer": per_1000_layer_coordinates(singleton.get("N_eff_i2c"), spec["model"]),
            "N_eff_c2i_per_1k_layer": per_1000_layer_coordinates(singleton.get("N_eff_c2i"), spec["model"]),
        })
    return rows, warnings


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "task", "model", "phase", "score", "raw", "chance", "J", "U", "Top", "TOC1",
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
        rf"All {len(rows)} primary settings. Direction-conditioned union coverage is reported explicitly rather than as OCC. "
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
        required=True,
        help="Primary matrix profile. The supported profile is iclr-28 (28 settings).",
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
        rows, warnings = compute_rows(root, empirical_fsm_chance=args.empirical_fsm_chance)
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
