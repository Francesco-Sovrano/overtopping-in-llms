#!/usr/bin/env python3
"""Build the canonical 28-setting primary CSV and LaTeX tables.

The primary matrix includes the Qwen2-1.5B input+output NLI row. Inputs must use
the current held-out singleton and conditional-marginal result schemas. Reported
ratios are never clipped.
"""
from __future__ import annotations
from pathlib import Path
import sys
from lib.project_paths import PROJECT_ROOT


import argparse
import csv
import json
import math
import pandas as pd
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from analysis.lib.primary_matrix import PRIMARY_PROFILE_CHOICES, normalize_primary_table, write_normalization_audit
from analysis.lib.task_metrics import (
    chance_baseline as _shared_chance_baseline,
    chance_normalized_score as _shared_chance_normalized_score,
    raw_task_score as _shared_raw_task_score,
)


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


def raw_task_score(task: str, dataset_stats: Dict[str, Any]) -> float:
    """Return a higher-is-better raw task score; jailbreak uses safe/refusal rate."""
    return _shared_raw_task_score(task, dataset_stats)


def chance_baseline(task: str, dataset_stats: Dict[str, Any], empirical_fsm_chance: bool = False) -> float:
    """Return the random-answer baseline used by the manuscript figures."""
    return _shared_chance_baseline(task, dataset_stats, empirical_fsm_chance=empirical_fsm_chance)


def chance_normalized_score(raw_score: float, chance: float) -> float:
    """Return the shared chance-normalized competence score."""
    return _shared_chance_normalized_score(raw_score, chance)


def score_for_table(phase: str, raw: float, chance: float) -> float:
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


def fmt_occ(x: Optional[float]) -> str:
    """Format OCC without rounding near-one values up to 1.

    OCC is a conditional probability and values such as 0.997 are scientifically
    different from exactly 1.0 in this table. Therefore print three decimals
    and reserve `1.` for values exactly equal to 1. Signed and above-one values
    are preserved rather than clipped.
    """
    if x is None or not math.isfinite(x):
        return "--"
    if math.isclose(x, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return "1."
    if math.isclose(x, 0.0, rel_tol=0.0, abs_tol=1e-12):
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

        ds = load_json(fg_path)
        glob = load_json(global_path)
        by = read_flip_by_neuron(by_neuron_path)

        raw = raw_task_score(spec["task"], ds)
        chance = chance_baseline(spec["task"], ds, empirical_fsm_chance)
        score = score_for_table(spec["phase"], raw, chance)
        n_eval = int(glob.get("n_evaluated_rows", 0))
        j = int(glob.get("n_neurons", len(by)))
        u = float(glob.get("union_flip_any_unique_rate", 0.0))
        c2i = float(glob.get("union_c2i_unique_rate", 0.0))
        i2c = float(glob.get("union_i2c_unique_rate", 0.0))

        singleton_path = stats_dir / "singleton_set_metrics.json"
        if not singleton_path.exists():
            raise FileNotFoundError(f"Missing current singleton metrics: {singleton_path}")
        singleton = load_json(singleton_path)
        if singleton.get("definition_version") != "heldout-set-metrics-v2":
            raise ValueError(
                f"Unsupported singleton metrics schema at {singleton_path}: "
                f"{singleton.get('definition_version')!r}"
            )
        interaction_path = stats_dir / "interaction_validation" / "interaction_validation_summary.json"
        interaction = load_json(interaction_path) if interaction_path.exists() else {}

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
        n05 = int(thresholds.get("0.05", 0))
        n10 = int(thresholds.get("0.1", thresholds.get("0.10", 0)))

        exact_occ = singleton.get("OCC_1")
        occ_for_latex = float(exact_occ) if exact_occ is not None else None
        occ_status = singleton.get("OCC_1_status", "missing")
        if exact_occ is None:
            warnings.append(
                f"{spec['task']} | {spec['model']} | {spec['phase']}: OCC_1 is unavailable "
                f"({occ_status})."
            )

        schema = interaction.get("definition_version") if isinstance(interaction, dict) else None
        candidate_effect = (
            interaction.get("candidate_E_J", {})
            if schema == "conditional-marginal-validation-v1"
            else {}
        )
        interaction_ej = candidate_effect.get("effect") if isinstance(candidate_effect, dict) else None
        cmc_1x = None
        if schema == "conditional-marginal-validation-v1":
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
            "C2I": c2i,
            "I2C": i2c,
            "OCC": occ_for_latex,
            "OCC_status": occ_status,
            "R_ov": singleton.get("R_ov"),
            "R_ov_status": singleton.get("R_ov_status"),
            "N_eff": singleton.get("N_eff"),
            "N_eff_status": singleton.get("N_eff_status"),
            "OCC_0": singleton.get("OCC_0"),
            "OCC_1": singleton.get("OCC_1"),
            "E_J": interaction_ej,
            "CMC_1x": cmc_1x,
            "N05": n05,
            "N10": n10,
        })
    return rows, warnings


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "task", "model", "phase", "score", "raw", "chance", "J", "U", "Top", "TOC1",
        "C2I", "I2C", "OCC", "OCC_status", "R_ov", "R_ov_status",
        "N_eff", "N_eff_status", "OCC_0", "OCC_1", "E_J", "CMC_1x",
        "N05", "N10", "n_eval", "stats_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_latex(rows: List[Dict[str, Any]], representative_only: bool) -> str:
    if representative_only:
        rows = sort_table1_rows(rows)
        caption = r"Representative interventions. \quotes{Ph.} is phase; \quotes{Comp.} is task competence for input+output and chance-corrected competence for output-only; $U$ is causal-flip coverage; \quotes{Top} is the largest singleton flip rate; and $N_t$ counts singleton flip rates above threshold $t$."
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
        rf"All {len(rows)} primary settings. Score is phase-specific as in Table~\ref{{tab:main-results}}; "
        r"configurations without a selected primary overtopping run are reported separately. "
        r"Out denotes output-only replacement, and I+O denotes input+output replacement. "
        r"$\CtwoI(J)$ is the stored positive-to-negative singleton-union support. "
        r"$\OCC(J)$ is the exact baseline-conditioned singleton-union probability "
        r"$P(\cup_j F_j\mid B(x)=1)$ when per-example event sidecars are available; "
        r"otherwise it is reported as unavailable rather than approximated from aggregate rates."
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
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "primary_analysis" / "tables"), help="Output directory. Default: <repo>/results/primary_analysis/tables")
    ap.add_argument("--empirical-fsm-chance", action="store_true", help="Audit/debug only: use sampled state-count FSM chance instead of the manuscript figure baseline mean(1/3,1/4,1/5,1/6). Do not use for figure-script parity.")
    ap.add_argument(
        "--primary-profile",
        choices=PRIMARY_PROFILE_CHOICES,
        required=True,
        help="Primary matrix profile. The supported profile is iclr-28 (28 settings).",
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
