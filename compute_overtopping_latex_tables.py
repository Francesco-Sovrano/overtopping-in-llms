#!/usr/bin/env python3
"""
Compute manuscript LaTeX tables from an overtopping results directory.

Inputs
------
A directory containing the extracted results tree, e.g. either:
  results/
  results/results/
  /path/to/results.zip

Outputs
-------
  primary_table.csv
  table1_representative.tex
  table8_primary.tex
  occ_warnings.txt

Important OCC note
------------------
The aggregate result files contain union C2I counts/rates but, in the copies I
have seen, they do not store the exact number of originally-correct examples in
that same intervention-evaluation subset. Exact OCC_J is

    union_c2i_unique_count / n_originally_correct_in_intervention_eval_subset.

When that exact denominator is absent, this script estimates it from the global
raw task score in feature_report/dataset_stats.json and n_evaluated_rows. If the
observed C2I count exceeds the estimated denominator, the script reports a
valid clipped OCC for LaTeX and records a warning. The CSV retains both the raw
ratio and the clipped value used in the table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    dict(task="Arithmetic", task_dir="arithmetic", org="EleutherAI", model_dir="pythia-6.9b", model="Pythia-6.9B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/rule_split-spectral_sample-decode_only-agonist_neurons-fast-random_anchor", note="low-coverage overtopping", rep=True),
    dict(task="Arithmetic", task_dir="arithmetic", org="Qwen", model_dir="Qwen2-1.5B-Instruct", model="Qwen2-1.5B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="", rep=False),
    dict(task="Arithmetic", task_dir="arithmetic", org="Qwen", model_dir="Qwen2-1.5B-Instruct", model="Qwen2-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="overtopped flips", rep=True),
    dict(task="Arithmetic", task_dir="arithmetic", org="Qwen", model_dir="Qwen2-7B-Instruct", model="Qwen2-7B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/rule_split-spectral_sample-decode_only-agonist_neurons-fast-random_anchor", note="high-coverage overtopping", rep=True),
    dict(task="Arithmetic", task_dir="arithmetic", org="Qwen", model_dir="Qwen2.5-1.5B-Instruct", model="Qwen2.5-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="weak singleton concentration", rep=True),

    # Jailbreak
    dict(task="Jailbreak", task_dir="bon_jailbreaking", org="Qwen", model_dir="Qwen2-1.5B-Instruct", model="Qwen2-1.5B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/spectral_split-M200000-decode_only-eval_mean-donor-agonist_neurons-fast-random_anchor-tau0.3", note="low singleton concentration", rep=True),
    dict(task="Jailbreak", task_dir="bon_jailbreaking", org="Qwen", model_dir="Qwen2-7B-Instruct", model="Qwen2-7B", phase="Out", stats="rule_extraction_results/neuron_flip_rules/stats/rule_split-spectral_sample-decode_only-agonist_neurons-fast-random_anchor", note="", rep=False),
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
    dict(task="NLI", task_dir="hans_nli", org="Qwen", model_dir="Qwen2-7B-Instruct", model="Qwen2-7B", phase="I+O", stats="rule_extraction_results/neuron_flip_rules/stats/rule_split-spectral_sample-agonist_neurons-fast-random_anchor", note="", rep=False),
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


JAILBREAK_SAFE_RATE_KEYS = [
    "safe_refusal_rate",
    "safety_rate",
    "refusal_rate",
    "safe_rate",
    "pct_safe",
    "pct_refused",
    "not_jailbroken_rate",
    "not_jailbroken",
]

JAILBREAK_ATTACK_RATE_KEYS = [
    "jailbreak_rate",
    "pct_is_jailbroken",
    "pct_jailbroken",
    "is_jailbroken_rate",
    "attack_success_rate",
    "asr",
    "success_rate",
    "unsafe_rate",
]

TASK_TO_FIGURE_TASK = {
    "Arithmetic": "arithmetic",
    "Grammar": "grammar_acceptability",
    "NLI": "hans_nli",
    "Random FSM": "random_fsm",
    "Jailbreak": "bon_jailbreaking",
}


def _figure_task_name(task: str) -> str:
    """Map manuscript display task names to the task identifiers used by the figure script."""
    return TASK_TO_FIGURE_TASK.get(task, task)


def _unit_rate(value: object, key: str) -> float:
    """Parse a rate stored either as a unit interval value or a percentage.

    This mirrors make_competence_vs_overtopping_paper_figures.py.
    """
    rate = float(value)
    if not math.isfinite(rate):
        raise ValueError(f"{key} is not finite: {value!r}")
    if 1.0 < rate <= 100.0:
        rate = rate / 100.0
    return float(min(1.0, max(0.0, rate)))


def jailbreak_rate(dataset_stats: Dict[str, Any]) -> float:
    """Return the jailbreak score using the same orientation as the figure script.

    The attached figure script accepts explicit safety/refusal keys and attack-success keys.
    For safety/refusal keys, it returns the complement. For attack-success keys, it returns
    the stored unit rate. Generic keys such as score/accuracy are intentionally not used.
    """
    for key in JAILBREAK_SAFE_RATE_KEYS:
        if key in dataset_stats:
            return 1.0 - _unit_rate(dataset_stats[key], key)

    for key in JAILBREAK_ATTACK_RATE_KEYS:
        if key in dataset_stats:
            return _unit_rate(dataset_stats[key], key)

    raise KeyError(
        "jailbreak dataset_stats.json lacks an explicit safety/refusal rate "
        f"({', '.join(JAILBREAK_SAFE_RATE_KEYS)}) or jailbreak/attack-success "
        f"rate ({', '.join(JAILBREAK_ATTACK_RATE_KEYS)})"
    )


def raw_task_score(task: str, dataset_stats: Dict[str, Any]) -> float:
    """Raw parsed accuracy for finite-answer tasks; figure-script jailbreak score for jailbreak."""
    fig_task = _figure_task_name(task)
    if fig_task == "bon_jailbreaking":
        return jailbreak_rate(dataset_stats)

    for key in ["accuracy", "score", "pct_is_correct"]:
        if key in dataset_stats:
            return float(dataset_stats[key])
    raise KeyError("dataset_stats.json lacks accuracy, score, and pct_is_correct")


def chance_baseline(task: str, dataset_stats: Dict[str, Any], empirical_fsm_chance: bool = False) -> float:
    """Random-answer baseline, aligned with the attached figure script by default.

    The default branch mirrors random_answer_chance() in
    make_competence_vs_overtopping_paper_figures.py.  The empirical FSM option is retained
    only as an audit/debug mode and is not used for publication-table parity with the figure script.
    """
    fig_task = _figure_task_name(task)
    if fig_task == "bon_jailbreaking":
        return 0.0

    for key in [
        "chance_accuracy",
        "random_chance_accuracy",
        "chance_score",
        "chance_rate",
        "random_baseline_accuracy",
        "random_guess_accuracy",
    ]:
        if key in dataset_stats:
            return float(dataset_stats[key])

    for key in ["n_classes", "num_classes", "n_labels", "num_labels"]:
        if key in dataset_stats:
            n = float(dataset_stats[key])
            if n > 0:
                return 1.0 / n

    if fig_task in {"grammar_acceptability", "hans_nli"}:
        return 0.5
    if fig_task == "random_fsm":
        if empirical_fsm_chance and "accuracy_by_num_states" in dataset_stats:
            total = 0.0
            weight = 0.0
            for k, v in dataset_stats["accuracy_by_num_states"].items():
                n = float(v.get("n", 0))
                total += n / float(k)
                weight += n
            if weight > 0:
                return total / weight
        return (1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 5.0 + 1.0 / 6.0) / 4.0
    if fig_task == "arithmetic":
        return 0.0
    return 0.0


def chance_normalized_score(raw_score: float, chance: float) -> float:
    """Figure-script kappa: clipped (raw - chance)/(1 - chance)."""
    if not math.isfinite(raw_score) or not math.isfinite(chance) or chance >= 1.0:
        return float("nan")
    return float(min(1.0, max(0.0, (raw_score - chance) / (1.0 - chance))))


def score_for_table(task: str, phase: str, raw: float, chance: float) -> float:
    """Phase-specific score, matching downstream_score(..., score_mode='phase-specific').

    Output-only rows correspond to the figure script's decode-only phase and therefore use
    chance-normalized score. Input+output rows use raw parsed score. For jailbreak rows,
    chance is zero, so output-only chance normalization leaves the jailbreak score unchanged.
    """
    if phase == "Out":
        return chance_normalized_score(raw, chance)
    return raw


def read_flip_by_neuron(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "--"
    if abs(x - 1.0) < 0.0005:
        return "1."
    if abs(x) < 0.0005:
        return ".000"
    s = f"{x:.{digits}f}"
    if s.startswith("0"):
        s = s[1:]
    if s.startswith("-0"):
        s = "-" + s[2:]
    return s


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
    if abs(x - 1.0) < eps or x > 1.0 - eps:
        return "1."
    if abs(x) < eps:
        return ".000" if digits == 3 else ".00"
    s = f"{x:.{digits}f}"
    if s.startswith("0"):
        s = s[1:]
    if s.startswith("-0"):
        s = "-" + s[2:]
    return s


def fmt_occ(x: Optional[float]) -> str:
    """Format OCC without rounding near-one values up to 1.

    OCC is a conditional probability and values such as 0.997 are scientifically
    different from exactly 1.0 in this table. Therefore print three decimals
    and reserve `1.` for values that are actually clipped/exactly equal to 1.
    """
    if x is None or not math.isfinite(x):
        return "--"
    if x >= 1.0:
        return "1."
    if x <= 0.0:
        return ".000"
    s = f"{x:.3f}"
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


def compute_rows(root: Path, empirical_fsm_chance: bool, clip_occ: bool) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for spec in PRIMARY_ROWS:
        model_root = root / spec["task_dir"] / spec["org"] / spec["model_dir"]
        stats_dir = model_root / spec["stats"]
        fg_path = model_root / "feature_report" / "dataset_stats.json"
        global_path = stats_dir / "flip_stats_global.json"
        by_neuron_path = stats_dir / "flip_stats_by_neuron.csv"

        missing = [str(p) for p in (fg_path, global_path, by_neuron_path) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing expected result files:\n" + "\n".join(missing))

        ds = load_json(fg_path)
        glob = load_json(global_path)
        by = read_flip_by_neuron(by_neuron_path)

        raw = raw_task_score(spec["task"], ds)
        chance = chance_baseline(spec["task"], ds, empirical_fsm_chance)
        score = score_for_table(spec["task"], spec["phase"], raw, chance)
        n_eval = int(glob.get("n_evaluated_rows", 0))
        j = int(glob.get("n_neurons", len(by)))
        u = float(glob.get("union_flip_any_unique_rate", 0.0))
        c2i = float(glob.get("union_c2i_unique_rate", 0.0))
        i2c = float(glob.get("union_i2c_unique_rate", 0.0))
        c2i_count = int(glob.get("union_c2i_unique_count", round(c2i * n_eval)))
        est_correct_count = int(round(raw * n_eval)) if n_eval > 0 and math.isfinite(raw) else 0

        top = max((float(r.get("flip_any_rate", 0.0) or 0.0) for r in by), default=0.0)
        toc1 = top / u if u > 0 else None
        n05 = sum(1 for r in by if float(r.get("flip_any_rate", 0.0) or 0.0) >= 0.05)
        n10 = sum(1 for r in by if float(r.get("flip_any_rate", 0.0) or 0.0) >= 0.10)

        occ_rawden = c2i / raw if raw > 0 else None
        occ_for_latex = occ_rawden
        occ_status = "global_raw_denominator"
        if occ_rawden is not None and occ_rawden > 1.0 + 1e-9:
            msg = (
                f"{spec['task']} | {spec['model']} | {spec['phase']}: "
                f"C2I count {c2i_count} exceeds estimated correct count {est_correct_count} "
                f"from raw score {raw:.9f} and n_eval {n_eval}; exact OCC denominator is not stored."
            )
            warnings.append(msg)
            occ_status = "clipped_for_probability; exact_eval_denominator_missing"
            if clip_occ:
                occ_for_latex = 1.0
        if occ_for_latex is not None and clip_occ:
            occ_for_latex = min(1.0, max(0.0, occ_for_latex))

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
            "C2I": c2i,
            "I2C": i2c,
            "OCC_rawden": occ_rawden,
            "OCC": occ_for_latex,
            "OCC_status": occ_status,
            "N05": n05,
            "N10": n10,
        })
    return rows, warnings


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "task", "model", "phase", "score", "raw", "chance", "J", "U", "Top", "TOC1",
        "C2I", "I2C", "OCC", "OCC_rawden", "OCC_status", "N05", "N10", "n_eval", "stats_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_latex(rows: List[Dict[str, Any]], representative_only: bool) -> str:
    if representative_only:
        rows = sort_table1_rows(rows)
        caption = "Representative interventions. \quotes{Ph.} is phase; \quotes{Comp.} is task competence for input+output and chance-corrected competence for output-only; $U$ is causal-flip coverage; \quotes{Top} is the largest singleton flip rate; and $N_t$ counts singleton flip rates above threshold $t$."
        label = "tab:main-results"
        counts_by_task: Dict[str, int] = {}
        for r in rows:
            counts_by_task[r["task"]] = counts_by_task.get(r["task"], 0) + 1

        lines = []
        lines.append(r"\begin{table}")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{1.4pt}")
        lines.append(r"\resizebox{\linewidth}{!}{%")
        lines.append(r"\begin{tabular}{@{}llcrrrrrrrrl@{}}")
        lines.append(r"\toprule")
        lines.append(r"Task & Model & Ph. & Comp. & $|J|$ & $U(J)$ & Top & $\TOC_1$ & $\OCC(J)$ & $N_{.05}$ & $N_{.10}$ & Note \\")
        lines.append(r"\midrule")
        last_task = None
        for r in rows:
            task = r["task"]
            if task != last_task:
                if last_task is not None:
                    lines.append(r"\midrule")
                task_cell = rf"\multirow{{{counts_by_task[task]}}}{{*}}{{{latex_escape(task)}}}"
                last_task = task
                cells = [
                    task_cell,
                    latex_escape(r["model"]),
                    r"I+O" if r["phase"] == "I+O" else "Out",
                    fmt_metric(r["score"], 3),
                    str(r["J"]),
                    fmt_metric(r["U"], 3),
                    fmt_metric(r["Top"], 3),
                    fmt_metric(r["TOC1"], 3),
                    fmt_occ(r["OCC"]),
                    str(r["N05"]),
                    str(r["N10"]),
                    latex_escape(r.get("note", "")),
                ]
                lines.append(" & ".join(cells) + r" \\")
            else:
                cells = [
                    latex_escape(r["model"]),
                    r"I+O" if r["phase"] == "I+O" else "Out",
                    fmt_metric(r["score"], 3),
                    str(r["J"]),
                    fmt_metric(r["U"], 3),
                    fmt_metric(r["Top"], 3),
                    fmt_metric(r["TOC1"], 3),
                    fmt_occ(r["OCC"]),
                    str(r["N05"]),
                    str(r["N10"]),
                    latex_escape(r.get("note", "")),
                ]
                lines.append("& " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}%")
        lines.append(r"}")
        lines.append(r"\caption{" + caption + r"}")
        lines.append(r"\label{" + label + r"}")
        lines.append(r"\end{table}")
        return "\n".join(lines) + "\n"

    caption = (
        r"All 28 primary settings. Score is phase-specific as in Table~\ref{tab:main-results}; "
        r"configurations without a selected primary overtopping run are reported separately. "
        r"Out denotes output-only replacement, and I+O denotes input+output replacement. "
        r"$\CtwoI(J)$ is the absolute positive-to-negative support of the discovered set, and "
        r"$\OCC(J)=\CtwoI(J)/s$ uses the raw unablated positive-behaviour rate $s$ "
        r"from Table~\ref{tab:all-competences}. Values equal to 1. are saturated within "
        r"the available aggregate precision."
    )
    label = "tab:appendix-primary-results"
    colspec = "lllrrrrrrrr"
    header = r"Model & Task & Phase & Score & $|J|$ & $U(J)$ & Top & $\CtwoI(J)$ & $\OCC(J)$ & $N_{.05}$ & $N_{.10}$ \\"

    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    for r in rows:
        cells = [
            latex_escape(r["model"]),
            latex_escape(r["task"]),
            r"I+O" if r["phase"] == "I+O" else "Out",
            fmt_metric(r["score"], 3),
            str(r["J"]),
            fmt_metric(r["U"], 3),
            fmt_metric(r["Top"], 3),
            fmt_metric(r["C2I"], 3),
            fmt_occ(r["OCC"]),
            str(r["N05"]),
            str(r["N10"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="Path to extracted results directory or results.zip")
    ap.add_argument("--out", default="computed_tables", help="Output directory")
    ap.add_argument("--empirical-fsm-chance", action="store_true", help="Audit/debug only: use sampled state-count FSM chance instead of the attached figure script baseline mean(1/3,1/4,1/5,1/6). Do not use for figure-script parity.")
    ap.add_argument("--no-clip-occ", action="store_true", help="Do not clip OCC to [0,1] in LaTeX when the exact denominator is absent")
    args = ap.parse_args(argv)

    root, tmp = locate_results_root(Path(args.results))
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        rows, warnings = compute_rows(root, empirical_fsm_chance=args.empirical_fsm_chance, clip_occ=not args.no_clip_occ)
        write_csv(out / "primary_table.csv", rows)
        (out / "table1_representative.tex").write_text(make_latex(rows, representative_only=True), encoding="utf-8")
        (out / "table8_primary.tex").write_text(make_latex(rows, representative_only=False), encoding="utf-8")
        warning_text = "\n".join(warnings) + ("\n" if warnings else "")
        if warnings:
            warning_text = (
                "Exact OCC requires n_originally_correct_in_intervention_eval_subset. "
                "The aggregate files did not expose that denominator for the rows below; "
                "LaTeX OCC was clipped to a valid probability unless --no-clip-occ was used.\n\n"
                + warning_text
            )
        (out / "occ_warnings.txt").write_text(warning_text or "No OCC denominator warnings.\n", encoding="utf-8")

        print(f"Using results root: {root}")
        print(f"Wrote: {out / 'primary_table.csv'}")
        print(f"Wrote: {out / 'table1_representative.tex'}")
        print(f"Wrote: {out / 'table8_primary.tex'}")
        print(f"Wrote: {out / 'occ_warnings.txt'}")
        if warnings:
            print(f"WARNING: {len(warnings)} OCC denominator warnings; inspect occ_warnings.txt", file=sys.stderr)
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
