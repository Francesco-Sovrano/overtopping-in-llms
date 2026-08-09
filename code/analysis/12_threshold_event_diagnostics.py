#!/usr/bin/env python3
"""Flip-conditioned spiking diagnostics for neuron ablation flips.

No script-6 outputs are read. Script-7 rule metrics are used only to propose
candidate neurons and optionally enrich sampled datapoints. The diagnostic target
is always observed high-N singleton-ablation flip vs non-flip for the same neuron.
"""

from __future__ import annotations
from pathlib import Path


import argparse
import contextlib
import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

if not hasattr(tqdm, "write"):
    tqdm.write = print

try:
    from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score, f1_score
except Exception:  # pragma: no cover
    roc_auc_score = matthews_corrcoef = balanced_accuracy_score = f1_score = None

from lib.caching_and_prompting import set_deterministic
from lib.feature_extraction_runner import resolve_task_spec
from lib.high_n_singleton_eval import (
    LOG_PREFIX,
    UnitSpec,
    evaluate_singleton_flips_high_n,
    load_scores_for_baseline,
    precompute_replacements_for_units,
    select_high_n_eval_indices,
)
from lib.modeling_and_ablation import LMWrapper, get_circuit_neurons_dict, get_device
from lib.text_and_rules import apply_rule_to_features, guess_filetype
from lib.threshold_event_shared import activation_hook_spec as shared_activation_hook_spec, activation_rows_for_unit, collect_reference_activations, safe_layer_label


def _int_or_zero(value):
    if value is None or str(value).strip() == "":
        return 0
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer or empty value, got {value!r}") from exc


def parse_args():
    p = argparse.ArgumentParser(description="Run flip-conditioned spiking diagnostics for rule-derived candidate neurons.")
    p.add_argument("--input_data_dir", required=True, help="Circuit-discovery neural_circuits directory containing dataset_info.json and manifest.json.")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--baseline_subsets", default="positive,negative")
    p.add_argument("--task_module", default="lib.tasks.arithmetic_task")
    p.add_argument("--ai_model", default=None)
    p.add_argument("--ai_model_cache_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--decode_only", action="store_true")
    p.add_argument("--intervention", default="mean-donor", choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"])
    p.add_argument("--points_to_use_for_mean_ablation", type=int, default=2048)

    # high-N evaluation knobs.  This is the script-7 style 10k cap.
    p.add_argument("--spiking_max_points", type=int, default=10000, help="Max examples per baseline subset used for the threshold/spiking experiment.")
    p.add_argument("--spiking_min_points", type=int, default=512)
    p.add_argument("--spiking_global_n_clusters", type=int, default=64)
    p.add_argument("--spiking_eval_cache_dir", default=None)
    p.add_argument("--force_spiking_eval", action="store_true")
    p.add_argument(
        "--evaluation_split",
        choices=["test", "train", "all"],
        default="test",
        help=(
            "Rows used for singleton effects and controls. Default: test. "
            "train/test use the is_test split; all uses every available row."
        ),
    )
    p.add_argument(
        "--candidate_flip_stats_path",
        default=None,
        help=(
            "Optional script-7 flip_stats_by_neuron.csv whose units should be evaluated "
            "directly. This bypasses rule-based candidate selection and is recommended "
            "for held-out random-channel diagnostics."
        ),
    )

    # Same-layer/head non-agonist controls. These are sampled independently of
    # observed flip labels, so they remain a clean specificity baseline rather
    # than conditioning on the overtopping outcome.
    p.add_argument("--same_layer_nonagonist_controls", dest="same_layer_nonagonist_controls", action="store_true", default=True)
    p.add_argument("--no_same_layer_nonagonist_controls", dest="same_layer_nonagonist_controls", action="store_false")
    p.add_argument("--nonagonist_candidate_pool_multiplier", type=int, default=1, help="Random same-layer/head candidate controls sampled per desired final control.")
    p.add_argument("--nonagonist_min_candidate_pool_per_layer", type=int, default=4)
    p.add_argument("--nonagonist_random_controls_per_agonist", type=int, default=1)

    # Compatibility no-ops accepted by existing runners.
    p.add_argument("--output_data_dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--spiking_max_units_per_population", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--spiking_control_match_ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--spiking_include_manifest_controls", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--threshold_event_overtopping_min_strength", type=float, default=0.20, help=argparse.SUPPRESS)
    p.add_argument("--threshold_event_weak_min_strength", type=float, default=0.05, help=argparse.SUPPRESS)
    p.add_argument("--threshold_event_control_max_strength", type=float, default=0.01, help=argparse.SUPPRESS)

    # threshold test knobs
    p.add_argument("--threshold_event_min_examples", type=int, default=8)
    p.add_argument("--threshold_event_repeats", type=int, default=20)
    p.add_argument("--threshold_event_holdout_fraction", type=float, default=0.5)
    p.add_argument("--threshold_event_n_bins", type=int, default=10)
    p.add_argument("--target", default="flip_any")

    # Rule-powered rule diagnostics. In this mode rows are sampled per rule,
    # then the union is ablated once so every rule gets enough match/nonmatch evidence.
    p.add_argument("--rule_conditioned_diagnostics", action="store_true", help="Run rule-conditioned diagnostics from script-7 rule metrics.")
    p.add_argument("--rule_conditioned_only", action="store_true", help="Use rule-match/nonmatch sampling to enrich datapoints; labels remain observed unit flips.")
    p.add_argument("--rule_metrics_path", default=None)
    p.add_argument("--rules_dir", default=None)
    p.add_argument("--rules_stats_dirname", default="")
    p.add_argument("--rule_conditioned_max_units", type=int, default=96)
    p.add_argument("--rule_conditioned_max_rules_per_unit", type=int, default=1)
    p.add_argument("--rule_hq_metric", default=None, help=argparse.SUPPRESS)
    p.add_argument("--rule_hq_threshold", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--rule_match_points_per_rule", type=int, default=64)
    p.add_argument("--rule_min_match_examples", type=int, default=16)
    p.add_argument("--rule_nonmatch_match_ratio", type=int, default=3)
    p.add_argument("--flip_conditioned_balance_examples", dest="flip_conditioned_balance_examples", action="store_true", default=True, help="For threshold diagnostics, balance flip-positive and flip-negative examples per unit.")
    p.add_argument("--no_flip_conditioned_balance_examples", dest="flip_conditioned_balance_examples", action="store_false")
    p.add_argument("--flip_conditioned_points_per_class", type=int, default=0, help="Optional cap per class after balancing; 0 uses all examples in the minority class.")

    # Proxy/first-order diagnostics. Activation-only is not enough: compare raw
    # activation, gradient, |gradient|, grad*activation, WANDA-style scores, and
    # first-order margin drop where available.
    p.add_argument("--proxy_metrics", default="activation,abs_activation,gradient,abs_gradient,activation_x_gradient,abs_activation_x_gradient,wanda,predicted_margin_drop,abs_predicted_margin_drop")
    p.add_argument("--proxy_allow_fallback_score", action="store_true", help="If an inferred gold margin is unavailable, backprop through a cached completion/top-token logit for proxy diagnostics.")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true", default=True, help="Reuse non-empty completed baseline outputs instead of recomputing them.")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false", help="Do not reuse completed baseline outputs.")
    p.add_argument("--force_threshold_event", action="store_true", help="Recompute threshold-event outputs even if non-empty outputs already exist.")
    p.add_argument("--make_visualizations", dest="make_visualizations", action="store_true", default=True)
    p.add_argument("--no_make_visualizations", dest="make_visualizations", action="store_false")
    p.add_argument("--log_level", default="quiet", choices=["quiet", "normal", "verbose"], help="Control non-tqdm logging. quiet prints only warnings/results; normal prints essential milestones; verbose prints paths/debug details.")
    p.add_argument("--print_json_summary", action="store_true", help="Also print the full machine-readable JSON summary at the end.")

    # Spectral args passed to the shared high-N sampler.
    p.add_argument("--spectral_space", default="hidden", choices=["hidden", "logits"])
    p.add_argument("--rep_hook_name", default="ln_final.hook_normalized")
    p.add_argument("--rep_pooling", default="last", choices=["last", "mean"])
    p.add_argument("--spectral_dim", type=int, default=32)
    p.add_argument("--spectral_cache_dir", default=None)
    p.add_argument("--max_seq_len", type=int, default=None)

    # Compatibility no-ops accepted by existing runners.
    p.add_argument("--n_associated", type=int, default=64)
    p.add_argument("--n_unrelated", type=int, default=64)
    p.add_argument("--search_epsilon", type=float, default=0.2)
    p.add_argument("--threshold_event_clamp_topk", type=_int_or_zero, default=0)
    p.add_argument("--force_posthoc_stats", action="store_true")
    p.add_argument("--force_population_stats", action="store_true")
    p.add_argument("--cluster_by_spectral", action="store_true")
    p.add_argument("--global_n_clusters", type=int, default=32)
    p.add_argument("--mlp_neurons_only", action="store_true")
    return p.parse_args()



_LOG_LEVELS = {"quiet": 0, "normal": 1, "verbose": 2}


def _log_enabled(args, level: str = "normal") -> bool:
    return _LOG_LEVELS.get(str(getattr(args, "log_level", "quiet")), 0) >= _LOG_LEVELS.get(level, 1)


def _log(args, msg: str, level: str = "normal") -> None:
    if _log_enabled(args, level):
        tqdm.write(str(msg))


def _result(msg: str) -> None:
    tqdm.write(str(msg))


def _silence_external_context(args):
    """Suppress noisy third-party stdout/stderr in quiet mode while keeping tqdm elsewhere."""
    if str(getattr(args, "log_level", "quiet")) != "quiet":
        return contextlib.nullcontext()
    stack = contextlib.ExitStack()
    devnull = stack.enter_context(open(os.devnull, "w"))
    stack.enter_context(contextlib.redirect_stdout(devnull))
    stack.enter_context(contextlib.redirect_stderr(devnull))
    return stack


def _fmt_float(x, digits: int = 3) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "nan"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "nan"


def _top_rows_for_summary(path: Path, *, n: int = 5) -> list[dict]:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty or "median_test_auc_oriented" not in df.columns:
        return []
    d = df.copy()
    d["median_test_auc_oriented"] = pd.to_numeric(d["median_test_auc_oriented"], errors="coerce")
    if "median_test_abs_mcc" in d.columns:
        d["median_test_abs_mcc"] = pd.to_numeric(d["median_test_abs_mcc"], errors="coerce")
    else:
        d["median_test_abs_mcc"] = np.nan
    d = d.dropna(subset=["median_test_auc_oriented"]).sort_values(["median_test_auc_oriented", "median_test_abs_mcc"], ascending=False).head(int(n))
    keep = [c for c in ["baseline_subset", "test_kind", "target", "population", "direction_family", "feature", "n_rule_tests", "n_unit_tests", "n_units", "median_test_auc_oriented", "median_test_abs_mcc"] if c in d.columns]
    out = []
    for r in d[keep].to_dict("records"):
        if "median_test_auc_oriented" in r and pd.notna(r["median_test_auc_oriented"]):
            r["median_test_auc_oriented"] = float(r["median_test_auc_oriented"])
        if "median_test_abs_mcc" in r and pd.notna(r["median_test_abs_mcc"]):
            r["median_test_abs_mcc"] = float(r["median_test_abs_mcc"])
        out.append(r)
    return out


def _result_name(row: dict) -> str:
    pieces = []
    for k in ["test_kind", "target", "population", "direction_family", "feature"]:
        v = row.get(k)
        if v is not None and str(v) and str(v).lower() != "nan":
            pieces.append(str(v))
    return " / ".join(pieces) if pieces else "result"


def _baseline_result_summary(baseline_out: Path, payload: dict | None = None) -> dict:
    payload = payload or {}
    return {
        "baseline": payload.get("baseline") or baseline_out.name.replace("_baseline", ""),
        "status": payload.get("status", "unknown"),
        "n_rows": payload.get("n_high_n_rows"),
        "n_units": payload.get("n_units"),
        "n_rules": payload.get("n_rules", payload.get("n_rules")),
        "rule_conditioned_only": payload.get("rule_conditioned_only"),
        "top_flip_conditioned": _top_rows_for_summary(baseline_out / "flip_conditioned_threshold_summary.csv", n=5),
        "top_population": _top_rows_for_summary(baseline_out / "threshold_population_summary.csv", n=5),
    }


def _write_and_print_baseline_summary(baseline_out: Path, payload: dict | None = None) -> dict:
    summary = _baseline_result_summary(baseline_out, payload)
    (baseline_out / "threshold_event_results_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    b = summary.get("baseline", baseline_out.name)
    fields = [f"{LOG_PREFIX} result {b}: {summary.get('status')}"]
    for key, label in [("n_rows", "rows"), ("n_units", "units"), ("n_rules", "rules")]:
        if summary.get(key) is not None:
            fields.append(f"{label}={summary[key]}")
    _result(" | ".join(fields))
    top = summary.get("top_flip_conditioned") or summary.get("top_rule_conditioned") or summary.get("top_population") or []
    if top:
        _result(f"{LOG_PREFIX} top results {b}:")
        for r in top[:5]:
            _result(f"  - {_result_name(r)}: AUC={_fmt_float(r.get('median_test_auc_oriented'))} |MCC|={_fmt_float(r.get('median_test_abs_mcc'))}")
    return summary


def _aggregate_result_summary(out_root: Path, aggregate_payload: dict | None = None, runs: list[dict] | None = None) -> dict:
    return {
        "status": "ok",
        "out_dir": str(out_root),
        "runs": runs or [],
        "aggregate_files": (aggregate_payload or {}).get("files", {}),
        "top_flip_conditioned": _top_rows_for_summary(out_root / "aggregate_flip_conditioned_threshold_summary.csv", n=10),
        "top_population": _top_rows_for_summary(out_root / "aggregate_population_summary.csv", n=10),
    }


def _write_and_print_aggregate_summary(out_root: Path, aggregate_payload: dict | None = None, runs: list[dict] | None = None) -> dict:
    summary = _aggregate_result_summary(out_root, aggregate_payload, runs)
    (out_root / "threshold_event_results_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _result(f"{LOG_PREFIX} final results: {out_root}")
    files = summary.get("aggregate_files") or {}
    key_files = [v for _, v in files.items() if isinstance(v, str)]
    if key_files:
        _result(f"{LOG_PREFIX} aggregate tables: " + ", ".join(key_files[:8]))
    top = summary.get("top_flip_conditioned") or summary.get("top_rule_conditioned") or summary.get("top_population") or []
    if top:
        _result(f"{LOG_PREFIX} aggregate top results:")
        for r in top[:8]:
            _result(f"  - {_result_name(r)}: AUC={_fmt_float(r.get('median_test_auc_oriented'))} |MCC|={_fmt_float(r.get('median_test_abs_mcc'))}")
    return summary

def _load_dataset_info(input_data_dir: Path) -> dict:
    path = Path(input_data_dir) / "dataset_info.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset_info.json not found in {input_data_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _add_manifest_controls(args, units: list[UnitSpec], *, input_data_dir: Path, max_controls: int, seed: int) -> list[UnitSpec]:
    manifest_path = Path(input_data_dir) / "manifest.json"
    if not manifest_path.exists() or max_controls <= 0:
        return units
    have = {(u.layer_label, int(u.neuron_id)) for u in units}
    try:
        circuit_entries = get_circuit_neurons_dict(manifest_path, args, quiet=True)
    except Exception as e:
        tqdm.write(f"{LOG_PREFIX} warning: manifest controls disabled: {e}")
        return units
    overt_layers = [u.layer_label for u in units]
    layer_counts = defaultdict(int)
    for l in overt_layers:
        layer_counts[l] += 1
    candidates = []
    for entry in circuit_entries.values():
        for layer_label, ids in (entry.get("neurons") or {}).items():
            for nid in ids:
                key = (str(layer_label), int(nid))
                if key in have:
                    continue
                candidates.append(UnitSpec(str(layer_label), int(nid), source="manifest_matched_control", circuit_id=entry.get("circuit_id"), circuit_label=entry.get("circuit_label"), seed_strength=0.0))
    if not candidates:
        return units
    rng = np.random.default_rng(int(seed))
    selected = []
    for layer, count in layer_counts.items():
        cg = [u for u in candidates if u.layer_label == layer]
        if not cg:
            continue
        n = min(len(cg), max(1, count))
        idx = rng.choice(np.arange(len(cg)), size=n, replace=False)
        selected.extend([cg[int(i)] for i in idx])
    if len(selected) < max_controls:
        have2 = {(u.layer_label, u.neuron_id) for u in selected}
        rem = [u for u in candidates if (u.layer_label, u.neuron_id) not in have2]
        if rem:
            idx = rng.choice(np.arange(len(rem)), size=min(max_controls - len(selected), len(rem)), replace=False)
            selected.extend([rem[int(i)] for i in idx])
    selected = selected[:max_controls]
    return units + selected


def _deduplicate_unit_specs(units: list[UnitSpec]) -> list[UnitSpec]:
    seen = set()
    out = []
    for u in units:
        key = (str(u.layer_label), int(u.neuron_id))
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def _layer_unit_capacity(model, layer_label: str) -> int | None:
    """Return the number of valid coordinate ids for the hooked tensor.

    MLP labels are ablated at TransformerLens ``hook_mlp_out``. That tensor's
    last dimension is ``d_model`` (residual-write coordinates), not ``d_mlp``.
    Attention labels use one head's ``hook_z`` vector, so their capacity is
    ``d_head``.
    """
    spec = shared_activation_hook_spec(layer_label)
    if spec is None:
        return None
    cfg = getattr(getattr(model, "hooked_model", None), "cfg", None)
    if spec["layer_type"] == "mlp":
        d_model = getattr(cfg, "d_model", None)
        if d_model is not None:
            try:
                return int(d_model)
            except Exception:
                pass
        try:
            block = model.hooked_model.blocks[int(spec["layer_index"])]
            if hasattr(block.mlp, "W_out"):
                return int(block.mlp.W_out.shape[-1])
            if hasattr(block.mlp, "W_in"):
                return int(block.mlp.W_in.shape[0])
        except Exception:
            return None
    if spec["layer_type"] == "attn":
        d_head = getattr(cfg, "d_head", None)
        if d_head is not None:
            try:
                return int(d_head)
            except Exception:
                pass
        try:
            block = model.hooked_model.blocks[int(spec["layer_index"])]
            if hasattr(block.attn, "W_O"):
                return int(block.attn.W_O.shape[1])
        except Exception:
            return None
    return None


def _filter_units_with_valid_capacity(args, *, model, units: list[UnitSpec], context: str) -> list[UnitSpec]:
    """Drop units whose id is invalid for the hooked tensor shape."""
    kept: list[UnitSpec] = []
    dropped = []
    for u in units:
        capacity = _layer_unit_capacity(model, str(u.layer_label))
        if capacity is None:
            kept.append(u)
            continue
        try:
            nid = int(u.neuron_id)
        except Exception:
            dropped.append((u, capacity))
            continue
        if 0 <= nid < int(capacity):
            kept.append(u)
        else:
            dropped.append((u, capacity))
    if dropped:
        examples = ", ".join(f"{u.layer_label}:{u.neuron_id}>=cap{cap}" for u, cap in dropped[:5])
        more = "" if len(dropped) <= 5 else f" (+{len(dropped)-5} more)"
        _log(args, f"{LOG_PREFIX} warning: dropped {len(dropped)} invalid {context} units for hook capacity: {examples}{more}", "normal")
    return kept


def _make_same_layer_nonagonist_control_pool(args, *, model, agonist_units: list[UnitSpec], seed: int) -> list[UnitSpec]:
    """Sample random non-agonist controls from the same layer/head as agonists.

    Controls are sampled independently of observed intervention flips. This keeps
    the non-agonist baseline from conditioning on the overtopping outcome.
    """
    if not bool(getattr(args, "same_layer_nonagonist_controls", True)) or not agonist_units:
        return []
    per_agonist = max(0, int(getattr(args, "nonagonist_random_controls_per_agonist", 0) or 0))
    if per_agonist <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    by_layer = defaultdict(list)
    excluded = {(str(u.layer_label), int(u.neuron_id)) for u in agonist_units}
    for u in agonist_units:
        by_layer[str(u.layer_label)].append(u)
    pool = []
    pool_multiplier = max(1, int(getattr(args, "nonagonist_candidate_pool_multiplier", 1) or 1))
    min_pool = max(0, int(getattr(args, "nonagonist_min_candidate_pool_per_layer", 4) or 4))
    for layer_label, layer_units in by_layer.items():
        capacity = _layer_unit_capacity(model, layer_label)
        if capacity is None or int(capacity) <= 0:
            continue
        available = [i for i in range(int(capacity)) if (layer_label, int(i)) not in excluded]
        if not available:
            continue
        desired_final = max(1, len(layer_units) * per_agonist)
        n_pool = min(len(available), max(min_pool, desired_final * pool_multiplier))
        idx = rng.choice(np.asarray(available, dtype=int), size=int(n_pool), replace=False)
        for nid in idx.tolist():
            pool.append(UnitSpec(str(layer_label), int(nid), source="same_layer_nonagonist_candidate", circuit_id=None, circuit_label="same_layer_nonagonist_pool", seed_strength=0.0))
    return _deduplicate_unit_specs(pool)


def _select_same_layer_nonagonist_baselines(args, *, baseline: str, agonist_units: list[UnitSpec],
                                            control_pool_units: list[UnitSpec], flip_stats: pd.DataFrame,
                                            baseline_out: Path) -> tuple[dict[str, str], dict]:
    """Choose final random same-layer/head non-agonist controls.

    Returns a mapping ``unit_key -> population``. The selection is random within
    each layer/head and does not use flip counts or other outcome labels.
    """
    if not bool(getattr(args, "same_layer_nonagonist_controls", True)) or not control_pool_units or flip_stats is None or flip_stats.empty:
        return {}, {"status": "disabled_or_empty"}
    stats = flip_stats.copy()
    stats["unit_key"] = stats["unit_key"].astype(str)
    by_key = {str(r.unit_key): r for r in stats.itertuples(index=False)}
    rng = np.random.default_rng(int(args.seed) + (17 if baseline == "positive" else 31))
    control_keys = [u.unit_key for u in control_pool_units if u.unit_key in by_key]
    control_by_layer = defaultdict(list)
    for u in control_pool_units:
        if u.unit_key in by_key:
            control_by_layer[str(u.layer_label)].append(u.unit_key)
    agonists_by_layer = defaultdict(list)
    for u in agonist_units:
        if u.unit_key in by_key:
            agonists_by_layer[str(u.layer_label)].append(u.unit_key)

    assigned: dict[str, str] = {}
    rows = []

    def _append_row(unit_key: str, population: str):
        r = by_key.get(unit_key)
        if r is None:
            return
        rows.append({
            "baseline_subset": baseline,
            "control_population": population,
            "unit_key": unit_key,
            "layer_label": str(getattr(r, "layer_label", "")),
            "layer_key": str(getattr(r, "layer_key", "")),
            "neuron_id": int(getattr(r, "neuron_id")),
            "n_flip_any": int(getattr(r, "n_flip_any", 0)),
            "n_flip_c2i": int(getattr(r, "n_flip_c2i", 0)),
            "n_flip_i2c": int(getattr(r, "n_flip_i2c", 0)),
            "flip_any_rate": float(getattr(r, "flip_any_rate", np.nan)),
            "c2i_rate": float(getattr(r, "c2i_rate", np.nan)),
            "i2c_rate": float(getattr(r, "i2c_rate", np.nan)),
        })

    n_rand = max(0, int(getattr(args, "nonagonist_random_controls_per_agonist", 0) or 0))
    for layer_label, ag_keys in agonists_by_layer.items():
        need = int(len(ag_keys) * n_rand)
        if need <= 0:
            continue
        candidates = [k for k in control_by_layer.get(layer_label, []) if k not in assigned]
        if not candidates:
            continue
        rng.shuffle(candidates)
        for key in candidates[:need]:
            assigned[key] = "random_nonagonist_control"
            _append_row(key, assigned[key])

    selection_df = pd.DataFrame(rows)
    selection_df.to_csv(baseline_out / "same_layer_nonagonist_control_selection.csv", index=False)
    payload = {
        "status": "ok",
        "candidate_pool_units": int(len(control_keys)),
        "selected_control_units": int(len(assigned)),
        "population_counts": selection_df["control_population"].value_counts().to_dict() if not selection_df.empty else {},
        "selection_file": "same_layer_nonagonist_control_selection.csv",
    }
    return assigned, payload


def _nonempty_csv(path: Path) -> bool:
    if not Path(path).exists() or Path(path).stat().st_size <= 0:
        return False
    try:
        return len(pd.read_csv(path, nrows=5)) > 0
    except Exception:
        return False


def _completed_output_ok(baseline_out: Path) -> bool:
    """True only for completed non-empty outputs; empty rulesets are deliberately re-runnable."""
    manifest = Path(baseline_out) / "threshold_spiking_experiment.json"
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(payload.get("status")) != "ok":
        return False
    if int(payload.get("n_units", 0) or 0) <= 0:
        return False
    required = ["high_n_scores_with_flips.csv", "threshold_unit_tests.csv", "threshold_population_summary.csv"]
    return all(_nonempty_csv(Path(baseline_out) / name) for name in required)


def _sanitize_feature_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()


def _resolve_rule_metrics_path(args, *, input_data_dir: Path) -> Path | None:
    explicit = getattr(args, "rule_metrics_path", None)
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    candidates = []
    stats_name = str(getattr(args, "rules_stats_dirname", "") or "")
    roots = []
    if getattr(args, "rules_dir", None):
        roots.append(Path(args.rules_dir))
    for parent in [input_data_dir, *input_data_dir.parents]:
        maybe = parent / "rule_extraction_results" / "neuron_flip_rules"
        if maybe.exists():
            roots.append(maybe)
    seen_roots = []
    for r in roots:
        rr = r.resolve()
        if rr not in seen_roots:
            seen_roots.append(rr)
    for rroot in seen_roots:
        if stats_name:
            candidates.extend([
                rroot / "stats" / stats_name / "rule_combo_metrics_best_per_neuron.csv",
                rroot / "stats" / stats_name / "rule_combo_metrics_all.csv",
            ])
        candidates.extend(sorted(rroot.glob("stats/*/rule_combo_metrics_best_per_neuron.csv")))
        candidates.extend(sorted(rroot.glob("stats/*/rule_combo_metrics_all.csv")))
    seen = set()
    for c in candidates:
        try:
            cc = c.resolve()
        except Exception:
            cc = c
        if cc in seen:
            continue
        seen.add(cc)
        if c.exists():
            return c
    return None


def _metric_col_from_name(df: pd.DataFrame, name: str) -> str | None:
    aliases = {
        "mcc": ["MCC", "mcc"],
        "balancedacc": ["BalancedAcc", "balanced_acc", "balanced_accuracy", "balancedacc"],
        "balanced_accuracy": ["BalancedAcc", "balanced_acc", "balanced_accuracy", "balancedacc"],
        "f1": ["F1", "F1(target=1|fire)", "f1"],
        "acc": ["Acc", "accuracy", "acc"],
    }
    key = str(name or "MCC").strip().lower().replace(" ", "_")
    for c in aliases.get(key, [name]):
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    return lower.get(key)


def _canonical_rule_target(flip_target: str, layer_key: str, neuron_id: int) -> str | None:
    ft = str(flip_target or "")
    suffix = f"_{layer_key}_{int(neuron_id)}"
    if ft in {"flip_any", "flip", "any"} or (ft.startswith("flip_") and ft.endswith(suffix) and not ft.startswith("flip_c2i_") and not ft.startswith("flip_i2c_")):
        return "flip_any"
    if ft in {"flip_c2i", "c2i"} or ft.startswith("flip_c2i_"):
        return "flip_c2i"
    if ft in {"flip_i2c", "i2c"} or ft.startswith("flip_i2c_"):
        return "flip_i2c"
    return None


def _rule_target_allowed_for_baseline(canonical_target: str | None, baseline: str) -> bool:
    if canonical_target is None:
        return False
    if canonical_target == "flip_any":
        return True
    if baseline == "positive":
        return canonical_target == "flip_c2i"
    if baseline == "negative":
        return canonical_target == "flip_i2c"
    return True


def _flip_col_for_canonical(canonical_target: str, layer_key: str, neuron_id: int) -> str:
    if canonical_target == "flip_c2i":
        return f"flip_c2i_{layer_key}_{int(neuron_id)}"
    if canonical_target == "flip_i2c":
        return f"flip_i2c_{layer_key}_{int(neuron_id)}"
    return f"flip_{layer_key}_{int(neuron_id)}"


def _load_rule_metrics(args, *, baseline: str, input_data_dir: Path) -> tuple[pd.DataFrame, Path | None]:
    path = _resolve_rule_metrics_path(args, input_data_dir=input_data_dir)
    if path is None:
        return pd.DataFrame(), None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        tqdm.write(f"{LOG_PREFIX} warning: cannot read rule metrics {path}: {e}")
        return pd.DataFrame(), path
    if df.empty:
        return df, path
    out = df.copy()
    if "layer_key" not in out.columns and "layer_label" in out.columns:
        out["layer_key"] = out["layer_label"].map(safe_layer_label)
    if "neuron_id" not in out.columns and "neuron_key" in out.columns:
        parsed = out["neuron_key"].astype(str).str.extract(r"^(.*)_(\d+)$")
        out["layer_key"] = parsed[0]
        out["neuron_id"] = pd.to_numeric(parsed[1], errors="coerce")
    if "expression" not in out.columns:
        for alt in ["rule_expression", "rule", "expr"]:
            if alt in out.columns:
                out["expression"] = out[alt]
                break
    required = {"layer_key", "neuron_id", "expression", "flip_target"}
    if not required.issubset(out.columns):
        tqdm.write(f"{LOG_PREFIX} warning: rule metrics missing columns {sorted(required - set(out.columns))}: {path}")
        return pd.DataFrame(), path
    out["neuron_id"] = pd.to_numeric(out["neuron_id"], errors="coerce")
    out = out.dropna(subset=["layer_key", "neuron_id", "expression"]).copy()
    out["neuron_id"] = out["neuron_id"].astype(int)
    out["layer_key"] = out["layer_key"].astype(str).map(safe_layer_label)

    # Do not pre-filter rules by rule quality. Script-7 rules are allowed to
    # enrich the sampled datapoints / select candidate units, but the actual
    # positive-vs-negative labels for this diagnostic are observed ablation
    # flips in the current high-N evaluation. Keep a rank value only for caps.
    rank_candidates = ["MCC", "mcc", "BalancedAcc", "balanced_acc", "balanced_accuracy", "F1", "f1", "Acc", "accuracy"]
    metric_col = next((c for c in rank_candidates if c in out.columns), None)
    if metric_col is not None:
        out["rule_quality_metric"] = metric_col
        out["rule_quality_value"] = pd.to_numeric(out[metric_col], errors="coerce")
    else:
        out["rule_quality_metric"] = "input_order"
        out["rule_quality_value"] = np.nan
    out["_rule_input_order"] = np.arange(len(out), dtype=int)
    out["canonical_flip_target"] = [_canonical_rule_target(ft, lk, nid) for ft, lk, nid in zip(out["flip_target"], out["layer_key"], out["neuron_id"])]
    out = out.loc[[_rule_target_allowed_for_baseline(ct, baseline) for ct in out["canonical_flip_target"]]].copy()
    if out.empty:
        return out, path
    out["rule_unit_key"] = out["layer_key"].astype(str) + ":" + out["neuron_id"].astype(str)
    out["_rule_rank_sort"] = pd.to_numeric(out["rule_quality_value"], errors="coerce").fillna(float("-inf"))
    out = out.sort_values(["rule_unit_key", "_rule_rank_sort", "_rule_input_order"], ascending=[True, False, True])
    max_rules = max(1, int(getattr(args, "rule_conditioned_max_rules_per_unit", 1) or 1))
    out = out.groupby("rule_unit_key", dropna=False, group_keys=False).head(max_rules).copy()
    max_units = int(getattr(args, "rule_conditioned_max_units", 0) or 0)
    if max_units > 0:
        keep = out.sort_values(["_rule_rank_sort", "_rule_input_order"], ascending=[False, True]).drop_duplicates("rule_unit_key").head(max_units)["rule_unit_key"]
        out = out.loc[out["rule_unit_key"].isin(set(keep))].copy()
    out = out.reset_index(drop=True)
    out["rule_id"] = np.arange(len(out), dtype=int)
    out["rule_metrics_path"] = str(path)
    out = out.drop(columns=[c for c in ["_rule_input_order", "_rule_rank_sort"] if c in out.columns])
    return out, path


def _units_from_rules(rule_df: pd.DataFrame) -> list[UnitSpec]:
    units = []
    if rule_df is None or rule_df.empty:
        return units
    for r in rule_df.drop_duplicates("rule_unit_key").to_dict("records"):
        units.append(UnitSpec(str(r["layer_key"]), int(r["neuron_id"]), source="flip_rule_candidate", circuit_id=None, circuit_label=None, seed_strength=float(r.get("rule_quality_value", 0.0))))
    return units


def _units_from_flip_stats(path: str | Path | None) -> list[UnitSpec]:
    if path is None:
        return []
    fp = Path(path).expanduser()
    if not fp.exists():
        raise FileNotFoundError(f"Candidate flip-stat table not found: {fp}")
    df = pd.read_csv(fp)
    required = {"neuron_id"}
    if not required.issubset(df.columns) or not ({"layer_label", "layer_key"} & set(df.columns)):
        raise ValueError(
            f"{fp} must contain neuron_id and layer_label (preferred) or layer_key."
        )
    layer_col = "layer_label" if "layer_label" in df.columns else "layer_key"
    strength_col = next(
        (c for c in ["flip_any_rate", "singleton_flip_any_rate", "seed_strength"] if c in df.columns),
        None,
    )
    out = []
    for row in df.dropna(subset=[layer_col, "neuron_id"]).to_dict("records"):
        strength = row.get(strength_col, 0.0) if strength_col is not None else 0.0
        try:
            strength = float(strength)
        except Exception:
            strength = 0.0
        out.append(
            UnitSpec(
                str(row[layer_col]),
                int(row["neuron_id"]),
                source="cha_discovered_candidate",
                circuit_id=None,
                circuit_label="candidate_flip_stats",
                seed_strength=strength,
            )
        )
    return _deduplicate_unit_specs(out)


def _rule_feature_columns(dataset_info: dict, scores_df: pd.DataFrame, task_targets) -> list[str]:
    scores_path = Path(dataset_info.get("scores_path", ""))
    feature_json = scores_path.parent / "features.json"
    if feature_json.exists():
        try:
            meta = json.loads(feature_json.read_text(encoding="utf-8"))
            names = {_sanitize_feature_name(feat.get("label", "")) for feat in meta if isinstance(feat, dict)}
            cols = [c for c in scores_df.columns if c in names and c not in set(task_targets)]
            if cols:
                return cols
        except Exception as e:
            tqdm.write(f"{LOG_PREFIX} warning: cannot read features.json: {e}")
    excluded = set(task_targets) | {"original_idx", "_orig_row", "_evaluated", "_sampled"}
    excluded |= {c for c in scores_df.columns if str(c).startswith("flip_")}
    return [c for c in scores_df.columns if c not in excluded and (pd.api.types.is_numeric_dtype(scores_df[c]) or pd.api.types.is_bool_dtype(scores_df[c]))]


def _select_rule_conditioned_eval_indices(args, *, baseline: str, scores_df: pd.DataFrame, rule_df: pd.DataFrame, dataset_info: dict, task_targets, baseline_out: Path):
    feature_cols = _rule_feature_columns(dataset_info, scores_df, task_targets)
    if not feature_cols:
        return np.array([], dtype=int), {"status": "empty_feature_columns"}, pd.DataFrame()
    features_df = scores_df[feature_cols].copy()
    rng = np.random.default_rng(int(args.seed))
    selected = set()
    plan = []
    max_total = int(args.spiking_max_points)
    per_rule = int(getattr(args, "rule_match_points_per_rule", 64) or 64)
    ratio = int(getattr(args, "rule_nonmatch_match_ratio", 3) or 0)
    min_match = int(getattr(args, "rule_min_match_examples", 16) or 16)
    for rr in tqdm(rule_df.to_dict("records"), desc=f"{LOG_PREFIX} {baseline} per-rule sampling", unit="rule", leave=False):
        status = "ok"
        try:
            mask = apply_rule_to_features(str(rr["expression"]), features_df, direction="positive").fillna(False).astype(bool).to_numpy()
        except Exception as e:
            mask = np.zeros(len(scores_df), dtype=bool)
            status = f"rule_parse_failed:{str(e)[:160]}"
        match = np.flatnonzero(mask)
        non = np.flatnonzero(~mask)
        if len(match) < min_match or len(non) < min_match:
            status = "underpowered_rule_matches"
            add_match = np.array([], dtype=int)
            add_non = np.array([], dtype=int)
        else:
            n_m = min(len(match), per_rule)
            n_n = min(len(non), max(min_match, ratio * n_m if ratio > 0 else per_rule))
            add_match = rng.choice(match, size=n_m, replace=False).astype(int) if len(match) > n_m else match.astype(int)
            add_non = rng.choice(non, size=n_n, replace=False).astype(int) if len(non) > n_n else non.astype(int)
            selected.update(add_match.tolist())
            selected.update(add_non.tolist())
        plan.append({
            "rule_id": int(rr.get("rule_id", -1)),
            "rule_unit_key": rr.get("rule_unit_key"),
            "canonical_flip_target": rr.get("canonical_flip_target"),
            "rule_quality_value": float(rr.get("rule_quality_value", np.nan)),
            "n_available_match": int(len(match)),
            "n_available_nonmatch": int(len(non)),
            "n_added_match": int(len(add_match)),
            "n_added_nonmatch": int(len(add_non)),
            "status": status,
            "expression": str(rr.get("expression", "")),
        })
        if max_total > 0 and len(selected) >= max_total:
            break
    eval_indices = np.asarray(sorted(selected), dtype=int)
    if max_total > 0 and len(eval_indices) > max_total:
        eval_indices = np.sort(rng.choice(eval_indices, size=max_total, replace=False).astype(int))
    plan_df = pd.DataFrame(plan)
    plan_df.to_csv(baseline_out / "rule_conditioned_sampling_plan.csv", index=False)
    ok_rule_ids = set(plan_df.loc[plan_df["status"].astype(str) == "ok", "rule_id"].astype(int).tolist()) if not plan_df.empty else set()
    meta = {"status": "ok", "strategy": "per_rule_match_nonmatch_union", "n_selected_rows": int(len(eval_indices)), "n_rules_ok": int(len(ok_rule_ids)), "n_rules_total": int(len(rule_df)), "feature_cols": feature_cols}
    return eval_indices, meta, plan_df


def _next_token_id_for_completion(tokenizer, prompt_text: str, completion_text: str):
    def _ids(text, add_special_tokens):
        try:
            return tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        except Exception:
            return None
    for add_special_tokens in (True, False):
        prompt_ids = _ids(prompt_text, add_special_tokens)
        full_ids = _ids(f"{prompt_text}{completion_text}", add_special_tokens)
        if prompt_ids and full_ids and len(full_ids) > len(prompt_ids) and full_ids[:len(prompt_ids)] == prompt_ids:
            return int(full_ids[len(prompt_ids)])
    for candidate in (completion_text, f" {completion_text}"):
        ids = _ids(candidate, False)
        if ids:
            return int(ids[0])
    return None


def _completion_text_from_row_for_saliency(task, row, prompt_col):
    keys = []
    for attr in ("DEFAULT_OUTPUT", "DEFAULT_OUTPUTS", "DEFAULT_ANSWER", "DEFAULT_ANSWERS"):
        val = getattr(task, attr, None)
        if isinstance(val, str):
            keys.append(val)
        elif isinstance(val, (list, tuple)):
            keys.extend([str(x) for x in val])
    keys.extend(["answer", "completion", "target_text", "correct_answer", "output", "label_text", "gold", "gold_answer"])
    seen = set()
    for k in keys:
        if k in seen or k == prompt_col or k not in row:
            continue
        seen.add(k)
        v = row.get(k)
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        if isinstance(v, (bool, np.bool_)) or v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        text = str(v)
        if text:
            return text
    return None


def _inferred_target_token_ids_from_rows(task, prompt_batch, logits_last, tokenizer, prompt_col):
    """Infer one target next-token id per row from row metadata only.

    Do not call task-object margin hooks here.  Script 12 owns the diagnostic
    objective.  Task/rule outputs may provide useful text columns such as
    raw_output, answer, target_text, or num_out; if none is usable the caller can
    fall back to a detached top-token objective.
    """
    target_ids = []
    used = []
    for row in prompt_batch:
        prompt_text = str(row.get(prompt_col, row.get(getattr(task, "DEFAULT_INPUT", "prompt"), "")))
        completion_text = _completion_text_from_row_for_saliency(task, row, prompt_col)
        if completion_text is None and "num_out" in row:
            try:
                val = row.get("num_out")
                if not pd.isna(val):
                    fval = float(val)
                    completion_text = str(int(fval)) if fval.is_integer() else str(fval)
            except Exception:
                completion_text = None
        tok_id = _next_token_id_for_completion(tokenizer, prompt_text, str(completion_text)) if completion_text is not None else None
        if tok_id is None:
            target_ids.append(-1)
            used.append(False)
        else:
            target_ids.append(int(tok_id))
            used.append(True)
    ids = torch.tensor(target_ids, device=logits_last.device, dtype=torch.long)
    valid = ids.ge(0) & ids.lt(logits_last.shape[1])
    return ids.clamp(0, logits_last.shape[1] - 1), valid, used


def _inferred_gold_margin_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col):
    ids, valid, used = _inferred_target_token_ids_from_rows(task, prompt_batch, logits_last, tokenizer, prompt_col)
    if not any(used):
        return None
    target_logits = logits_last.gather(1, ids.unsqueeze(1)).squeeze(1)
    masked = logits_last.clone()
    masked[torch.arange(masked.size(0), device=masked.device), ids] = float('-inf')
    other_logits = masked.max(dim=1).values
    margin = target_logits - other_logits
    return torch.where(valid, margin, torch.zeros_like(margin))


def _saliency_objective_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col, allow_fallback=False):
    margin = _inferred_gold_margin_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col)
    if margin is not None:
        return margin, "inferred_gold_margin"
    if not allow_fallback:
        return None, "unavailable"
    top_ids = logits_last.detach().argmax(dim=1).to(device=logits_last.device, dtype=torch.long)
    target_ids = []
    used_cached = []
    for i, row in enumerate(prompt_batch):
        prompt_text = str(row.get(prompt_col, row.get(getattr(task, "DEFAULT_INPUT", "prompt"), "")))
        completion_text = _completion_text_from_row_for_saliency(task, row, prompt_col)
        tok_id = _next_token_id_for_completion(tokenizer, prompt_text, str(completion_text)) if completion_text is not None else None
        if tok_id is None:
            tok_id = int(top_ids[i].item())
            used_cached.append(False)
        else:
            used_cached.append(True)
        target_ids.append(tok_id)
    ids = torch.tensor(target_ids, device=logits_last.device, dtype=torch.long).clamp(0, logits_last.shape[1] - 1)
    objective = logits_last.gather(1, ids.unsqueeze(1)).squeeze(1)
    name = "cached_completion_logit" if all(used_cached) else "cached_completion_logit+top_logit" if any(used_cached) else "top_next_token_logit"
    return torch.nan_to_num(objective, nan=0.0, posinf=0.0, neginf=0.0), name


def collect_reference_proxy_tensors(model, task, examples, prompt_col, layer_labels, batch_size, *, allow_fallback_score=False, decode_only=False, max_new_tokens=10):
    # Gradients from the prompt-token forward pass are not comparable to
    # decode-step singleton interventions.  When decode_only is requested, the
    # caller should use decode-step activation collection instead of returning
    # misleading prompt-position saliency proxies.
    if bool(decode_only):
        return {}
    if not examples or not layer_labels:
        return {}
    specs = {}
    for layer_label in dict.fromkeys(layer_labels):
        spec = shared_activation_hook_spec(layer_label)
        if spec is not None:
            specs[layer_label] = spec
    if not specs:
        return {}
    hook_names = []
    for spec in specs.values():
        if spec["hook_name"] not in hook_names:
            hook_names.append(spec["hook_name"])
    device = model.hooked_model.cfg.device
    collected = {ll: {"activations": [], "grads": [], "positions": [], "prompt_margin": [], "gradient_objective": []} for ll in specs}
    for start in tqdm(range(0, len(examples), int(batch_size)), desc=f"{LOG_PREFIX} proxy grad batches", unit="batch", leave=False):
        batch_examples = examples[start:start+int(batch_size)]
        batch_prompts = [str(row[prompt_col]) for row in batch_examples]
        input_ids, attention_mask, input_lengths = model.tokenize_with_mask(batch_prompts, device, padding=True, truncation=True, add_special_tokens=True, padding_side="right")
        last_idx = (input_lengths - 1).to(device)
        positions_cpu = last_idx.detach().to(torch.long).cpu()
        batch_idx = torch.arange(input_ids.shape[0], device=device)
        hook_acts = {}
        def make_hook(name):
            def _hook(act, hook):
                # Some local model wrappers load weights with requires_grad=False.
                # We only need d(objective)/d(activation), so make the hooked
                # activation itself a differentiable leaf if necessary.
                if torch.is_tensor(act) and torch.is_floating_point(act) and not bool(getattr(act, "requires_grad", False)):
                    act = act.detach().requires_grad_(True)
                hook_acts[name] = act
                return act
            return _hook
        fwd_hooks = [(name, make_hook(name)) for name in hook_names]
        try:
            model.hooked_model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with model.hooked_model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True, clear_contexts=True):
                    residual = model.hooked_model(input_ids, attention_mask=attention_mask, padding_side="right", return_type="residual", stop_at_layer=model.hooked_model.cfg.n_layers)
                    if getattr(model.hooked_model.cfg, "normalization_type", None) is not None and hasattr(model.hooked_model, "ln_final"):
                        residual = model.hooked_model.ln_final(residual)
                    logits_last = model.hooked_model.unembed(residual[batch_idx, last_idx, :])
                    margin, objective_name = _saliency_objective_from_last_logits(task, batch_examples, logits_last, model.tokenizer, prompt_col, allow_fallback=allow_fallback_score)
                    if margin is None or not bool(getattr(margin, "requires_grad", False)):
                        tqdm.write(f"{LOG_PREFIX} warning: proxy objective {objective_name} has no gradient at batch {start}")
                        continue
                    active = [name for name in hook_names if name in hook_acts]
                    grad_targets = [hook_acts[name] for name in active]
                    grad_list = torch.autograd.grad(margin.sum(), grad_targets, allow_unused=True)
        except Exception as e:
            tqdm.write(f"{LOG_PREFIX} warning: proxy gradient batch failed at {start}: {e}")
            continue
        margin_cpu = margin.detach().to(torch.float32).cpu()
        grad_by_name = {name: grad for name, grad in zip(active, grad_list)}
        for layer_label, spec in specs.items():
            hook_name = spec["hook_name"]
            if hook_name not in hook_acts:
                continue
            act_full = hook_acts[hook_name]
            grad_full = grad_by_name.get(hook_name)
            if act_full.ndim == 3:
                act_slice = act_full[batch_idx, last_idx, :]
                grad_slice = None if grad_full is None else grad_full[batch_idx, last_idx, :]
            elif act_full.ndim == 4:
                act_slice = act_full[batch_idx, last_idx, :, :]
                grad_slice = None if grad_full is None else grad_full[batch_idx, last_idx, :, :]
            else:
                continue
            collected[layer_label]["activations"].append(act_slice.detach().to(torch.float32).cpu())
            collected[layer_label]["grads"].append((torch.zeros_like(act_slice) if grad_slice is None else grad_slice).detach().to(torch.float32).cpu())
            collected[layer_label]["positions"].append(positions_cpu.clone())
            collected[layer_label]["prompt_margin"].append(margin_cpu.clone())
            collected[layer_label]["gradient_objective"].append(str(objective_name))
        try:
            model.cleanup_after_generate()
        except Exception:
            pass
    out = {}
    for ll, payload in collected.items():
        if not payload["activations"]:
            continue
        out[ll] = {
            "activations": torch.cat(payload["activations"], dim=0),
            "grads": torch.cat(payload["grads"], dim=0),
            "positions": torch.cat(payload["positions"], dim=0),
            "prompt_margin": torch.cat(payload["prompt_margin"], dim=0),
            "gradient_objective": "+".join(sorted(set(payload["gradient_objective"]))) if payload["gradient_objective"] else "task_margin",
        }
    return out


def _wanda_weight_norms_for_spec(model, spec, n_compared_units):
    """Return WANDA-style outgoing-weight norms for the hooked coordinate basis.

    Attention ``hook_z`` coordinates have a direct output map through ``W_O``, so
    we use the norm of the corresponding output-weight vector. MLP entries in
    this package are residual-write coordinates captured at ``hook_mlp_out``;
    there is no single per-coordinate outgoing weight matrix analogous to WANDA's
    layer weight. For those coordinates the multiplier is intentionally one, so
    WANDA columns reduce to absolute-activation value/rank/z-score screens.
    """
    if spec is None:
        return None
    if spec["layer_type"] == "attn":
        try:
            layer_idx = int(spec["layer_index"])
            w_o = model.hooked_model.blocks[layer_idx].attn.W_O.detach().to(torch.float32).cpu()
            norms = w_o.norm(dim=-1).reshape(-1)
            if int(norms.numel()) == int(n_compared_units):
                return norms
        except Exception:
            return None
    return torch.ones(int(n_compared_units), dtype=torch.float32)


def _baseline_values_for_unit(unit: UnitSpec, positions, intervention, mean_activations):
    spec = shared_activation_hook_spec(unit.layer_label)
    if spec is None:
        return None
    positions = positions.to(torch.long).cpu()
    if intervention == "zero" or mean_activations is None:
        return torch.zeros(len(positions), dtype=torch.float32)
    unit_key = f"m{int(spec['layer_index'])}" if spec["layer_type"] == "mlp" else f"a{int(spec['layer_index'])}.h{int(spec['head_index'])}"
    idx_full = mean_activations.mean_idx.get(unit_key)
    mean_global = mean_activations.mean_global.get(unit_key)
    if idx_full is None or mean_global is None:
        return None
    idx_full = idx_full.detach().to(torch.long).cpu()
    q = idx_full.new_tensor([int(unit.neuron_id)])
    loc = torch.searchsorted(idx_full, q)
    if loc.numel() == 0:
        return None
    loc = int(loc.item())
    if loc >= idx_full.numel() or int(idx_full[loc].item()) != int(unit.neuron_id):
        return None
    global_value = float(mean_global.detach().to(torch.float32).cpu()[loc].item())
    if intervention in ("mean", "mean-donor"):
        return torch.full((len(positions),), global_value, dtype=torch.float32)
    mean_per_pos = (getattr(mean_activations, "mean_per_pos", None) or {}).get(unit_key)
    if mean_per_pos is None:
        return torch.full((len(positions),), global_value, dtype=torch.float32)
    mean_per_pos = mean_per_pos.detach().to(torch.float32).cpu()
    vals = torch.full((len(positions),), global_value, dtype=torch.float32)
    for i, pos in enumerate(positions.tolist()):
        if 0 <= int(pos) < mean_per_pos.shape[0]:
            vals[i] = float(mean_per_pos[int(pos), loc].item())
    return vals



def _proxy_rows_requested_set(requested_features):
    if requested_features is None:
        return None
    req = {str(f) for f in requested_features if str(f)}
    # Backward compatibility aliases are computed from activation variants.
    if "layer_percentile_rank" in req:
        req.add("activation_percentile_rank")
    if "layer_zscore" in req:
        req.add("activation_layer_zscore")
    return req


def proxy_rows_for_unit(unit: UnitSpec, layer_payload: dict, *, model, intervention, mean_activations, baseline: str, requested_features=None):
    """Build proxy rows for one unit, computing only requested proxy columns.

    The old implementation recomputed layer-wide abs/product/WANDA pools, row-wise
    means/stds, and O(n_examples * layer_width) percentile comparisons for every
    unit.  This version stores a small cache inside layer_payload and uses sorted
    row-wise pools for percentile ranks, so each expensive layer-wide object is
    computed once per layer/baseline rather than once per unit.
    """
    spec = shared_activation_hook_spec(unit.layer_label)
    if spec is None or layer_payload is None:
        return pd.DataFrame()
    acts = layer_payload.get("activations")
    grads = layer_payload.get("grads")
    positions = layer_payload.get("positions")
    prompt_margin = layer_payload.get("prompt_margin")
    if acts is None:
        return pd.DataFrame()
    if grads is None:
        grads = torch.zeros_like(acts)

    cache = layer_payload.setdefault("_proxy_cache", {})
    cache_key = (id(acts), id(grads), str(unit.layer_label))
    if cache.get("cache_key") != cache_key:
        cache.clear()
        cache["cache_key"] = cache_key

    if "pool_acts" not in cache or "pool_grads" not in cache:
        if spec["layer_type"] == "mlp":
            if acts.ndim != 2 or int(unit.neuron_id) >= acts.shape[1]:
                return pd.DataFrame()
            cache["pool_acts"] = acts.reshape(acts.shape[0], -1).to(torch.float32)
            cache["pool_grads"] = grads.reshape(grads.shape[0], -1).to(torch.float32)
        elif spec["layer_type"] == "attn":
            if acts.ndim != 3:
                return pd.DataFrame()
            cache["pool_acts"] = acts.reshape(acts.shape[0], -1).to(torch.float32)
            cache["pool_grads"] = grads.reshape(grads.shape[0], -1).to(torch.float32)
        else:
            return pd.DataFrame()

    pool_acts = cache["pool_acts"]
    pool_grads = cache["pool_grads"].to(device=pool_acts.device, dtype=torch.float32)
    cache["pool_grads"] = pool_grads
    proxy_device = pool_acts.device

    if spec["layer_type"] == "mlp":
        flat_idx = int(unit.neuron_id)
    else:
        d_head = int(acts.shape[2])
        flat_idx = int(spec["head_index"]) * d_head + int(unit.neuron_id)
    if flat_idx < 0 or flat_idx >= pool_acts.shape[1]:
        return pd.DataFrame()

    n = int(pool_acts.shape[0])
    req = _proxy_rows_requested_set(requested_features)
    def want(col):
        return req is None or col in req
    def want_any(cols):
        return req is None or any(c in req for c in cols)
    def np1(t):
        if t is None:
            return np.full(n, np.nan)
        return t.detach().to("cpu").numpy()
    def cached(name, fn):
        if name not in cache:
            cache[name] = fn()
        return cache[name]

    unit_act = pool_acts[:, flat_idx]
    unit_grad = pool_grads[:, flat_idx]
    unit_actgrad = None
    predicted_margin_drop = None
    baseline_values = None
    delta_to_baseline = None
    wanda = None

    if positions is None:
        pos_for_baseline = torch.arange(n, dtype=torch.long, device="cpu")
    else:
        pos_for_baseline = positions.detach().to(torch.long).cpu()

    rows = {
        "example_local_index": np.arange(n, dtype=int),
        "unit_key": unit.unit_key,
        "layer_label": unit.layer_label,
        "layer_key": unit.layer_key,
        "unit_id": int(unit.neuron_id),
        "neuron_id": int(unit.neuron_id),
        "hook_position": pos_for_baseline.numpy().astype(int),
        "prompt_margin": np1(prompt_margin) if prompt_margin is not None else np.full(n, np.nan),
        "gradient_objective": str(layer_payload.get("gradient_objective", "unknown")),
    }

    def get_pool(stem):
        if stem == "activation":
            return pool_acts, unit_act
        if stem == "abs_activation":
            pool = cached("abs_activation_pool", lambda: pool_acts.abs())
            return pool, unit_act.abs()
        if stem == "gradient":
            return pool_grads, unit_grad
        if stem == "abs_gradient":
            pool = cached("abs_gradient_pool", lambda: pool_grads.abs())
            return pool, unit_grad.abs()
        if stem == "activation_x_gradient":
            pool = cached("activation_x_gradient_pool", lambda: pool_acts * pool_grads)
            val = unit_act * unit_grad
            return pool, val
        if stem == "abs_activation_x_gradient":
            pool = cached("abs_activation_x_gradient_pool", lambda: (pool_acts * pool_grads).abs())
            val = (unit_act * unit_grad).abs()
            return pool, val
        if stem == "wanda":
            def build_weight_norms():
                wn = _wanda_weight_norms_for_spec(model, spec, pool_acts.shape[1])
                if wn is None:
                    wn = torch.ones(int(pool_acts.shape[1]), dtype=torch.float32, device=proxy_device)
                return wn.to(device=proxy_device, dtype=torch.float32)
            weight_norms = cached("weight_norms", build_weight_norms)
            pool = cached("wanda_pool", lambda: pool_acts.abs() * weight_norms.view(1, -1))
            val = pool[:, flat_idx]
            return pool, val
        return None, None

    def pct_rank(stem):
        pool, val = get_pool(stem)
        if pool is None:
            return torch.full((n,), float("nan"), dtype=torch.float32, device=proxy_device)
        sorted_pool = cached(f"{stem}_sorted", lambda: torch.sort(pool, dim=1).values)
        # Original definition was 1 - (# greater / width), i.e. # <= val / width.
        counts_le = torch.searchsorted(sorted_pool, val.to(proxy_device).unsqueeze(1), right=True).squeeze(1)
        return counts_le.to(torch.float32) / float(sorted_pool.shape[1])

    def zscore(stem):
        pool, val = get_pool(stem)
        if pool is None:
            return torch.full((n,), float("nan"), dtype=torch.float32, device=proxy_device)
        mean = cached(f"{stem}_mean", lambda: pool.mean(dim=1))
        std = cached(f"{stem}_std", lambda: pool.std(dim=1, unbiased=False).clamp_min(1e-8))
        return (val - mean) / std

    def ensure_baseline_effects():
        nonlocal baseline_values, delta_to_baseline, predicted_margin_drop
        if predicted_margin_drop is not None:
            return
        bv = _baseline_values_for_unit(unit, pos_for_baseline, intervention, mean_activations)
        if bv is None:
            bv = torch.full((n,), float("nan"), dtype=torch.float32, device=proxy_device)
        baseline_values = bv.to(device=proxy_device, dtype=torch.float32)
        delta_to_baseline = baseline_values - unit_act.to(torch.float32)
        predicted_margin_shift = unit_grad * delta_to_baseline
        predicted_margin_drop = -predicted_margin_shift

    # Raw value columns.
    if want("activation_value"):
        rows["activation_value"] = np1(unit_act)
    if want("abs_activation_value"):
        rows["abs_activation_value"] = np1(unit_act.abs())
    if want("gradient_value"):
        rows["gradient_value"] = np1(unit_grad)
    if want("abs_gradient_value"):
        rows["abs_gradient_value"] = np1(unit_grad.abs())
    if want_any(["activation_x_gradient", "abs_activation_x_gradient"]):
        unit_actgrad = unit_act * unit_grad
        if want("activation_x_gradient"):
            rows["activation_x_gradient"] = np1(unit_actgrad)
        if want("abs_activation_x_gradient"):
            rows["abs_activation_x_gradient"] = np1(unit_actgrad.abs())
    if want("wanda_value"):
        _, wanda = get_pool("wanda")
        rows["wanda_value"] = np1(wanda)
    if want_any(["baseline_value", "delta_to_baseline", "predicted_margin_shift", "predicted_margin_drop", "abs_predicted_margin_drop", "learned_direction_score", "abs_learned_direction_score"]):
        ensure_baseline_effects()
        if want("baseline_value"):
            rows["baseline_value"] = np1(baseline_values)
        if want("delta_to_baseline"):
            rows["delta_to_baseline"] = np1(delta_to_baseline)
        if want("predicted_margin_shift"):
            rows["predicted_margin_shift"] = np1(-predicted_margin_drop)
        if want("predicted_margin_drop"):
            rows["predicted_margin_drop"] = np1(predicted_margin_drop)
        if want("abs_predicted_margin_drop"):
            rows["abs_predicted_margin_drop"] = np1(predicted_margin_drop.abs())
        if want("learned_direction_score"):
            rows["learned_direction_score"] = np1(predicted_margin_drop if baseline == "positive" else -predicted_margin_drop)
        if want("abs_learned_direction_score"):
            rows["abs_learned_direction_score"] = np1(predicted_margin_drop.abs())

    # Percentile-rank and layer-zscore columns; cached per layer/pool.
    stems = ["activation", "abs_activation", "gradient", "abs_gradient", "activation_x_gradient", "abs_activation_x_gradient", "wanda"]
    for stem in stems:
        pcol = f"{stem}_percentile_rank"
        zcol = f"{stem}_layer_zscore"
        if want(pcol):
            rows[pcol] = np1(pct_rank(stem))
        if want(zcol):
            rows[zcol] = np1(zscore(stem))

    # Backward compatibility aliases for activation-only code paths.
    if want("layer_percentile_rank"):
        if "activation_percentile_rank" not in rows:
            rows["activation_percentile_rank"] = np1(pct_rank("activation"))
        rows["layer_percentile_rank"] = rows["activation_percentile_rank"]
    if want("layer_zscore"):
        if "activation_layer_zscore" not in rows:
            rows["activation_layer_zscore"] = np1(zscore("activation"))
        rows["layer_zscore"] = rows["activation_layer_zscore"]
    if want_any(["delta_from_layer_mean", "abs_delta_from_layer_mean"]):
        per_example_layer_mean = np1(cached("activation_mean", lambda: pool_acts.mean(dim=1)))
        act_np = rows.get("activation_value", np1(unit_act))
        delta = act_np - per_example_layer_mean
        if want("delta_from_layer_mean"):
            rows["delta_from_layer_mean"] = delta
        if want("abs_delta_from_layer_mean"):
            rows["abs_delta_from_layer_mean"] = np.abs(delta)

    return pd.DataFrame(rows)

def _parse_proxy_features(args) -> list[str]:
    aliases = {
        "activation": "activation_value", "act": "activation_value", "abs_activation": "abs_activation_value", "|activation|": "abs_activation_value",
        "gradient": "gradient_value", "grad": "gradient_value", "abs_gradient": "abs_gradient_value", "|gradient|": "abs_gradient_value",
        "activation_x_gradient": "activation_x_gradient", "gradxact": "activation_x_gradient", "actgrad": "activation_x_gradient",
        "abs_activation_x_gradient": "abs_activation_x_gradient", "wanda": "wanda_value",
        "predicted_margin_drop": "predicted_margin_drop", "abs_predicted_margin_drop": "abs_predicted_margin_drop",
        "learned_direction": "learned_direction_score", "learned_direction_score": "learned_direction_score",
    }
    out = []
    for part in str(getattr(args, "proxy_metrics", "") or "").split(','):
        key = part.strip().lower()
        if not key:
            continue
        val = aliases.get(key, part.strip())
        if val not in out:
            out.append(val)
    # Add useful layer/rank variants for proxy scores when available.
    for base in list(out):
        if base.endswith("_value"):
            stem = base[:-6]
            for extra in (f"{stem}_percentile_rank", f"{stem}_layer_zscore"):
                if extra not in out:
                    out.append(extra)
    for extra in ["layer_percentile_rank", "layer_zscore", "learned_direction_score", "abs_learned_direction_score"]:
        if extra not in out:
            out.append(extra)
    return out


def _parse_flip_targets(args) -> list[str]:
    """Parse requested flip target(s) for threshold tests.

    Accepts a single target, a comma-separated list, or "all".  This makes
    the --target runner option effective instead of silently evaluating every
    target regardless of configuration.
    """
    raw = str(getattr(args, "target", "flip_any") or "flip_any").strip().lower()
    allowed = ["flip_any", "flip_c2i", "flip_i2c"]
    aliases = {
        "any": "flip_any",
        "c2i": "flip_c2i",
        "correct_to_incorrect": "flip_c2i",
        "i2c": "flip_i2c",
        "incorrect_to_correct": "flip_i2c",
    }
    if raw in {"all", "directional", "both", "all_flips"}:
        return allowed
    out = []
    for part in re.split(r"[, ]+", raw):
        if not part:
            continue
        val = aliases.get(part, part)
        if val not in allowed:
            raise ValueError(f"Unsupported --target value {part!r}; expected one of {allowed} or 'all'.")
        if val not in out:
            out.append(val)
    return out or ["flip_any"]


def _activation_only_proxy_features(features: list[str]) -> list[str]:
    activation_prefixes = ("activation_", "abs_activation_", "wanda_")
    allowed_exact = {
        "activation_value", "abs_activation_value", "wanda_value",
        "baseline_value", "delta_to_baseline", "layer_percentile_rank",
        "layer_zscore", "delta_from_layer_mean", "abs_delta_from_layer_mean",
    }
    out = []
    for f in features:
        fs = str(f)
        if "gradient" in fs or "margin" in fs or "learned" in fs:
            continue
        if fs in allowed_exact or fs.startswith(activation_prefixes):
            if fs not in out:
                out.append(fs)
    return out or ["activation_value", "abs_activation_value", "layer_percentile_rank", "layer_zscore"]


def _make_activation_only_proxy_payload(layer_acts: dict, *, objective_name: str) -> dict:
    return {
        k: {
            "activations": v,
            "grads": torch.zeros_like(v),
            "positions": torch.arange(v.shape[0], dtype=torch.long, device="cpu"),
            "prompt_margin": torch.full((v.shape[0],), float("nan"), dtype=torch.float32, device="cpu"),
            "gradient_objective": str(objective_name),
        }
        for k, v in layer_acts.items()
    }


def _event_rate_stats_for_feature(df, feature, target):
    x, y = _finite_xy(df[feature], df[target])
    if len(y) == 0:
        return {}
    return {"n": int(len(y)), "n_pos": int(y.sum()), "mean_score_pos": float(np.nanmean(x[y == 1])) if np.any(y == 1) else np.nan, "mean_score_neg": float(np.nanmean(x[y == 0])) if np.any(y == 0) else np.nan}

def _finite_xy(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], (y[mask] > 0.5).astype(int)



def _mcc_from_counts(tp, fp, tn, fn):
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    tn = np.asarray(tn, dtype=float)
    fn = np.asarray(fn, dtype=float)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out = np.full_like(tp, np.nan, dtype=float)
    np.divide((tp * tn) - (fp * fn), denom, out=out, where=denom > 0)
    return out


def _fit_threshold(x, y, min_examples):
    """Fit the best one-dimensional threshold using vectorized counts.

    The previous version looped over up to ~1k thresholds and called
    sklearn.metrics.matthews_corrcoef for each direction.  Here we sort once and
    derive TP/FP/TN/FN for all thresholds with searchsorted + prefix sums.
    """
    x, y = _finite_xy(x, y)
    if len(y) < 2 * min_examples or y.sum() < min_examples or (len(y) - y.sum()) < min_examples:
        return None
    vals = np.unique(np.sort(x))
    if len(vals) < 2:
        return None
    if len(vals) > 512:
        vals = np.unique(np.quantile(vals, np.linspace(0, 1, 513)))
    thresholds = np.concatenate([vals, ((vals[:-1] + vals[1:]) / 2.0)]) if len(vals) >= 2 else vals
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0:
        return None

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order].astype(int)
    prefix_pos = np.concatenate([[0], np.cumsum(ys)])
    n = int(len(ys))
    pos_total = int(prefix_pos[-1])
    neg_total = n - pos_total

    # Direction >= threshold.
    left = np.searchsorted(xs, thresholds, side="left")
    pred_pos_ge = n - left
    tp_ge = pos_total - prefix_pos[left]
    fp_ge = pred_pos_ge - tp_ge
    fn_ge = pos_total - tp_ge
    tn_ge = neg_total - fp_ge
    mcc_ge = _mcc_from_counts(tp_ge, fp_ge, tn_ge, fn_ge)

    # Direction <= threshold.
    right = np.searchsorted(xs, thresholds, side="right")
    pred_pos_le = right
    tp_le = prefix_pos[right]
    fp_le = pred_pos_le - tp_le
    fn_le = pos_total - tp_le
    tn_le = neg_total - fp_le
    mcc_le = _mcc_from_counts(tp_le, fp_le, tn_le, fn_le)

    scores = np.concatenate([np.abs(mcc_ge), np.abs(mcc_le)])
    if not np.isfinite(scores).any():
        return None
    best_i = int(np.nanargmax(scores))
    if best_i < thresholds.size:
        t = float(thresholds[best_i])
        direction = ">="
        mcc = float(mcc_ge[best_i])
    else:
        j = best_i - thresholds.size
        t = float(thresholds[j])
        direction = "<="
        mcc = float(mcc_le[j])

    auc = np.nan
    if roc_auc_score:
        try:
            auc = float(roc_auc_score(y, x))
        except Exception:
            auc = np.nan
    return {"threshold": t, "direction": direction, "train_mcc": mcc, "train_abs_mcc": abs(mcc), "train_auc_oriented": max(auc, 1-auc) if np.isfinite(auc) else np.nan}

def _eval_threshold(x, y, threshold, direction, min_examples):
    x, y = _finite_xy(x, y)
    if len(y) < 2 * min_examples or y.sum() < min_examples or (len(y) - y.sum()) < min_examples:
        return None
    pred = (x >= threshold).astype(int) if direction == ">=" else (x <= threshold).astype(int)
    try:
        mcc = float(matthews_corrcoef(y, pred)) if matthews_corrcoef else np.nan
    except Exception:
        mcc = np.nan
    out = {"test_n": int(len(y)), "test_n_pos": int(y.sum()), "test_prevalence": float(y.mean()), "test_mcc": mcc, "test_abs_mcc": abs(mcc) if np.isfinite(mcc) else np.nan}
    if balanced_accuracy_score:
        try:
            out["test_balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
        except Exception:
            out["test_balanced_accuracy"] = np.nan
    if f1_score:
        try:
            out["test_f1"] = float(f1_score(y, pred))
        except Exception:
            out["test_f1"] = np.nan
    if roc_auc_score:
        try:
            auc = float(roc_auc_score(y, x))
            out["test_auc_oriented"] = float(max(auc, 1.0 - auc))
        except Exception:
            out["test_auc_oriented"] = np.nan
    pred_pos = int(pred.sum())
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out.update({
        "test_threshold_spikes": pred_pos,
        "test_threshold_spike_rate": float(pred.mean()) if len(pred) else np.nan,
        "test_true_positives": tp,
        "test_false_positives": fp,
        "test_true_negatives": tn,
        "test_false_negatives": fn,
        "test_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan,
        "test_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        "test_false_positive_rate": float(fp / n_neg) if n_neg > 0 else np.nan,
        "test_false_negative_rate": float(fn / n_pos) if n_pos > 0 else np.nan,
    })
    return out


def _threshold_spike_counts(x, y, threshold, direction):
    x, y = _finite_xy(x, y)
    if len(y) == 0 or threshold is None or not np.isfinite(float(threshold)):
        return {}
    pred = (x >= threshold).astype(int) if direction == ">=" else (x <= threshold).astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "full_threshold_spikes": int(pred.sum()),
        "full_threshold_spike_rate": float(pred.mean()) if len(pred) else np.nan,
        "full_true_positives": tp,
        "full_false_positives": fp,
        "full_true_negatives": tn,
        "full_false_negatives": fn,
        "full_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan,
        "full_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        "full_false_positive_rate": float(fp / n_neg) if n_neg > 0 else np.nan,
        "full_false_negative_rate": float(fn / n_pos) if n_pos > 0 else np.nan,
    }


def repeated_holdout(x, y, *, min_examples, repeats, holdout_fraction, seed):
    x, y = _finite_xy(x, y)
    if len(y) < 2 * min_examples or y.sum() < 2 * min_examples or (len(y) - y.sum()) < 2 * min_examples:
        return None
    rng = np.random.default_rng(int(seed))
    rows = []
    for rep in range(int(repeats)):
        train_idx = []
        test_idx = []
        for cls in (0, 1):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            n_test = int(round(float(holdout_fraction) * len(idx)))
            n_test = max(min_examples, min(len(idx) - min_examples, n_test))
            test_idx.extend(idx[:n_test].tolist())
            train_idx.extend(idx[n_test:].tolist())
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        fit = _fit_threshold(x[train_idx], y[train_idx], min_examples)
        if fit is None:
            continue
        ev = _eval_threshold(x[test_idx], y[test_idx], fit["threshold"], fit["direction"], min_examples)
        if ev is None:
            continue
        rows.append({"repeat": rep, **fit, **ev})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out = {"n_repeats_ok": int(len(df))}
    for col in ["train_abs_mcc", "train_auc_oriented", "test_abs_mcc", "test_auc_oriented", "test_balanced_accuracy", "test_f1", "test_prevalence", "test_threshold_spike_rate", "test_precision", "test_recall", "test_false_positive_rate", "test_false_negative_rate", "threshold"]:
        if col in df:
            out[f"mean_{col}"] = float(pd.to_numeric(df[col], errors="coerce").mean())
            out[f"median_{col}"] = float(pd.to_numeric(df[col], errors="coerce").median())
    best = df.iloc[int(pd.to_numeric(df["test_abs_mcc"], errors="coerce").fillna(-1).argmax())]
    out["representative_threshold"] = float(best["threshold"])
    out["representative_direction"] = str(best["direction"])
    return out


def _assign_population_from_high_n(row, args):
    """Coarse descriptive bucket from observed high-N flip rate.

    This is metadata only; candidate units come from script-7 rules and labels
    come from observed ablation flips.
    """
    strength = max(float(row.get("flip_any_rate", 0.0)), float(row.get("c2i_rate", 0.0)), float(row.get("i2c_rate", 0.0)))
    if strength > 0.0:
        return "flip_supported_rule_candidate", strength
    return "no_observed_flip_rule_candidate", strength


def _make_binned_rows(udf, feature, target, population, unit_key, direction, n_bins):
    x, y = _finite_xy(udf[feature], udf[target])
    if len(y) < max(10, n_bins) or len(np.unique(x)) < 2:
        return []
    oriented = x if direction == ">=" else -x
    try:
        bins = pd.qcut(oriented, q=min(n_bins, len(np.unique(oriented))), duplicates="drop")
    except Exception:
        return []
    tmp = pd.DataFrame({"oriented_score": oriented, "target": y, "bin": bins})
    rows = []
    for idx, (b, g) in enumerate(tmp.groupby("bin", observed=False, sort=True)):
        rows.append({"population": population, "unit_key": unit_key, "feature": feature, "target": target, "bin_index": int(idx), "n": int(len(g)), "flip_rate": float(g["target"].mean()), "mean_oriented_score": float(g["oriented_score"].mean())})
    return rows



def _rule_conditioned_threshold_tests(args, *, baseline: str, baseline_out: Path, rule_df: pd.DataFrame,
                                      scores_out: pd.DataFrame, raw_df: pd.DataFrame, dataset_info: dict,
                                      task_targets, feature_list: list[str]) -> dict:
    """Flip-conditioned proxy diagnostics.

    Script-7 rules are used only to select candidate units / optionally enrich the
    high-N sample. The diagnostic target is never rule quality or rule match.
    For each candidate unit, positives are rows where ablating that unit actually
    flipped the model in the current high-N evaluation; negatives are rows where
    it did not flip. Optionally balance positives and negatives.
    """
    if rule_df is None or rule_df.empty:
        payload = {"status": "empty_rule_candidates"}
        (baseline_out / "flip_conditioned_threshold_experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    if raw_df.empty:
        payload = {"status": "empty_activation_proxy_rows"}
        (baseline_out / "flip_conditioned_threshold_experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    rows = []
    support_rows = []
    min_examples = int(args.threshold_event_min_examples)
    rng = np.random.default_rng(int(args.seed))
    for rr in tqdm(rule_df.to_dict("records"), desc=f"{LOG_PREFIX} {baseline} flip-conditioned threshold tests", unit="unit", leave=False):
        layer_key = safe_layer_label(rr["layer_key"])
        nid = int(rr["neuron_id"])
        rule_unit_key = f"{layer_key}:{nid}"
        canonical = rr.get("canonical_flip_target") or _canonical_rule_target(rr.get("flip_target"), layer_key, nid) or "flip_any"
        flip_col = _flip_col_for_canonical(canonical, layer_key, nid)
        if flip_col not in scores_out.columns:
            flip_col = _flip_col_for_canonical("flip_any", layer_key, nid)
        if flip_col not in scores_out.columns:
            support_rows.append({"rule_id": int(rr.get("rule_id", -1)), "unit_key": rule_unit_key, "status": "missing_flip_column", "canonical_flip_target": canonical})
            continue
        flip = scores_out[flip_col].fillna(False).astype(bool).astype(int).to_numpy()
        udf = raw_df.loc[raw_df["unit_key"].astype(str) == rule_unit_key].sort_values("example_local_index").copy()
        if udf.empty:
            support_rows.append({"rule_id": int(rr.get("rule_id", -1)), "unit_key": rule_unit_key, "status": "missing_proxy_rows", "canonical_flip_target": canonical, "flip_col": flip_col, "n_flips": int(flip.sum()), "n_nonflips": int(len(flip)-flip.sum())})
            continue
        n = min(len(udf), len(flip))
        udf = udf.iloc[:n].copy()
        udf["unit_flip"] = flip[:n].astype(int)
        pos_idx = np.flatnonzero(udf["unit_flip"].to_numpy() == 1)
        neg_idx = np.flatnonzero(udf["unit_flip"].to_numpy() == 0)
        n_pos, n_neg = int(len(pos_idx)), int(len(neg_idx))
        status = "ok"
        if n_pos < min_examples or n_neg < min_examples:
            status = "underpowered_flip_labels"
        selected_idx = np.arange(len(udf), dtype=int)
        if status == "ok" and bool(getattr(args, "flip_conditioned_balance_examples", True)):
            per_class = min(n_pos, n_neg)
            cap = int(getattr(args, "flip_conditioned_points_per_class", 0) or 0)
            if cap > 0:
                per_class = min(per_class, cap)
            pos_sel = rng.choice(pos_idx, size=per_class, replace=False) if len(pos_idx) > per_class else pos_idx
            neg_sel = rng.choice(neg_idx, size=per_class, replace=False) if len(neg_idx) > per_class else neg_idx
            selected_idx = np.sort(np.concatenate([pos_sel, neg_sel]).astype(int))
            udf = udf.iloc[selected_idx].copy()
        support_rows.append({
            "rule_id": int(rr.get("rule_id", -1)), "unit_key": rule_unit_key, "layer_key": layer_key, "neuron_id": nid,
            "baseline_subset": baseline, "canonical_flip_target": canonical, "flip_col": flip_col,
            "rule_quality_value": float(rr.get("rule_quality_value", np.nan)), "n_rows_before_balance": n,
            "n_flips_before_balance": n_pos, "n_nonflips_before_balance": n_neg,
            "n_rows_used": int(len(udf)), "n_flips_used": int(udf["unit_flip"].sum()),
            "n_nonflips_used": int(len(udf) - int(udf["unit_flip"].sum())),
            "balanced_examples": bool(getattr(args, "flip_conditioned_balance_examples", True)),
            "status": status, "expression": str(rr.get("expression", "")),
        })
        if status != "ok":
            continue
        for feature in feature_list:
            if feature not in udf.columns:
                continue
            res = repeated_holdout(udf[feature], udf["unit_flip"], min_examples=min_examples, repeats=int(args.threshold_event_repeats), holdout_fraction=float(args.threshold_event_holdout_fraction), seed=int(args.seed) + int(rr.get("rule_id", 0)))
            if res is None:
                continue
            spike_counts = _threshold_spike_counts(udf[feature], udf["unit_flip"], res.get("representative_threshold"), res.get("representative_direction"))
            rows.append({
                "baseline_subset": baseline, "rule_id": int(rr.get("rule_id", -1)), "unit_key": rule_unit_key,
                "layer_key": layer_key, "neuron_id": nid, "rule_quality_value": float(rr.get("rule_quality_value", np.nan)),
                "test_kind": "proxy_predicts_unit_flip", "target": "unit_flip", "canonical_flip_target": canonical, "feature": feature,
                "n_rows": int(len(udf)), "n_pos": int(udf["unit_flip"].sum()), **res, **spike_counts,
            })
    selected_path = baseline_out / "flip_conditioned_candidate_units.csv"
    rule_df.to_csv(selected_path, index=False)
    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(baseline_out / "flip_conditioned_flip_support.csv", index=False)
    tests_df = pd.DataFrame(rows)
    tests_df.to_csv(baseline_out / "flip_conditioned_threshold_unit_tests.csv", index=False)
    summary_rows = []
    if not tests_df.empty:
        for keys, g in tests_df.groupby(["baseline_subset", "test_kind", "target", "feature"], dropna=False):
            b, kind, target, feature = keys
            summary_rows.append({
                "baseline_subset": b, "test_kind": kind, "target": target, "feature": feature,
                "n_unit_tests": int(len(g)), "n_units": int(g["unit_key"].nunique()),
                "median_test_auc_oriented": float(pd.to_numeric(g.get("median_test_auc_oriented"), errors="coerce").median()),
                "median_test_abs_mcc": float(pd.to_numeric(g.get("median_test_abs_mcc"), errors="coerce").median()),
                "mean_test_auc_oriented": float(pd.to_numeric(g.get("mean_test_auc_oriented"), errors="coerce").mean()),
                "mean_test_abs_mcc": float(pd.to_numeric(g.get("mean_test_abs_mcc"), errors="coerce").mean()),
                "median_full_threshold_spike_rate": float(pd.to_numeric(g.get("full_threshold_spike_rate"), errors="coerce").median()),
                "median_full_false_positive_rate": float(pd.to_numeric(g.get("full_false_positive_rate"), errors="coerce").median()),
                "median_test_threshold_spike_rate": float(pd.to_numeric(g.get("median_test_threshold_spike_rate"), errors="coerce").median()),
                "median_test_false_positive_rate": float(pd.to_numeric(g.get("median_test_false_positive_rate"), errors="coerce").median()),
            })
    pd.DataFrame(summary_rows).to_csv(baseline_out / "flip_conditioned_threshold_summary.csv", index=False)
    payload = {"status": "ok", "n_candidate_units": int(rule_df["rule_unit_key"].nunique()) if "rule_unit_key" in rule_df else int(len(rule_df)), "n_unit_tests": int(len(tests_df)), "files": {"candidate_units": "flip_conditioned_candidate_units.csv", "flip_support": "flip_conditioned_flip_support.csv", "threshold_unit_tests": "flip_conditioned_threshold_unit_tests.csv", "threshold_summary": "flip_conditioned_threshold_summary.csv"}}
    (baseline_out / "flip_conditioned_threshold_experiment.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def run_for_baseline(args, baseline: str, *, dataset_info: dict, task, prompt_col: str, target_col: str, ai_model: str, out_root: Path):
    baseline_out = out_root / f"{baseline}_baseline"
    baseline_out.mkdir(parents=True, exist_ok=True)
    _log(args, f"{LOG_PREFIX} baseline={baseline} out={baseline_out}", "verbose")
    if bool(getattr(args, "skip_existing", True)) and not bool(getattr(args, "force_threshold_event", False)) and _completed_output_ok(baseline_out):
        payload_path = baseline_out / "threshold_spiking_experiment.json"
        payload = {"baseline": baseline, "status": "skipped_existing", "out": str(baseline_out)}
        try:
            if payload_path.exists():
                payload.update(json.loads(payload_path.read_text(encoding="utf-8")))
                payload["status"] = "skipped_existing"
        except Exception:
            pass
        _result(f"{LOG_PREFIX} {baseline}: skipped existing non-empty outputs")
        _write_and_print_baseline_summary(baseline_out, payload)
        return payload

    rule_df = pd.DataFrame()
    rule_path = None
    if bool(getattr(args, "rule_conditioned_diagnostics", False)):
        rule_df, rule_path = _load_rule_metrics(args, baseline=baseline, input_data_dir=Path(args.input_data_dir))
        if rule_path is None:
            _result(f"{LOG_PREFIX} {baseline}: no rule metrics file found")
        elif rule_df.empty:
            _result(f"{LOG_PREFIX} {baseline}: no rules passed filters")
        else:
            _log(args, f"{LOG_PREFIX} {baseline}: loaded {len(rule_df)} rule candidate(s)", "normal")

    rule_conditioned_only = bool(getattr(args, "rule_conditioned_only", False)) and not rule_df.empty
    units = _units_from_flip_stats(getattr(args, "candidate_flip_stats_path", None))
    candidate_source = "script-7 flip statistics"
    if not units:
        units = _units_from_rules(rule_df)
        candidate_source = "script-7 rules"
    _log(args, f"{LOG_PREFIX} {baseline}: selected {len(units)} candidate unit(s) from {candidate_source}", "normal")
    units = _deduplicate_unit_specs(units)
    if not units:
        _result(f"{LOG_PREFIX} {baseline}: no units selected")
        payload = {"baseline": baseline, "status": "empty_units", "n_units": 0, "n_rules": int(len(rule_df))}
        (baseline_out / "threshold_spiking_experiment.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _write_and_print_baseline_summary(baseline_out, payload)
        return payload

    selected_units_seed = pd.DataFrame([u.__dict__ | {"unit_key": u.unit_key, "layer_key": u.layer_key} for u in units])
    selected_units_seed.to_csv(baseline_out / "selected_units_from_rules.csv", index=False)

    scores_path = Path(dataset_info.get("scores_path", ""))
    if not scores_path.exists():
        raise FileNotFoundError(f"scores_path from dataset_info does not exist: {scores_path}")
    scores_df = load_scores_for_baseline(
        scores_path=scores_path,
        target_col=target_col,
        baseline_subset=baseline,
        task_targets=task.DEFAULT_TARGETS,
        split=str(args.evaluation_split),
    )
    if scores_df.empty:
        return {"baseline": baseline, "status": "empty_scores"}

    device = get_device()
    _log(args, f"{LOG_PREFIX} {baseline}: loading model on {device}", "normal")
    with _silence_external_context(args):
        model = LMWrapper(model_name=ai_model, device=device, eval_mode=True, circuit_discovery=False, cache_dir=args.ai_model_cache_dir)
    _log(args, f"{LOG_PREFIX} {baseline}: model loaded", "verbose")
    args.ai_model = ai_model
    if args.spectral_cache_dir is None:
        args.spectral_cache_dir = str(Path("./cache") / "threshold_events" / baseline)

    agonist_units = _filter_units_with_valid_capacity(args, model=model, units=list(units), context="agonist")
    if not agonist_units:
        payload = {"baseline": baseline, "status": "empty_valid_units", "n_units": 0, "n_rules": int(len(rule_df))}
        (baseline_out / "threshold_spiking_experiment.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _write_and_print_baseline_summary(baseline_out, payload)
        return payload
    if rule_conditioned_only:
        eval_indices, sample_meta, sampling_plan_df = _select_rule_conditioned_eval_indices(args, baseline=baseline, scores_df=scores_df, rule_df=rule_df, dataset_info=dataset_info, task_targets=task.DEFAULT_TARGETS, baseline_out=baseline_out)
        ok_rule_ids = set(sampling_plan_df.loc[sampling_plan_df["status"].astype(str) == "ok", "rule_id"].astype(int).tolist()) if not sampling_plan_df.empty else set()
        rule_df = rule_df.loc[rule_df["rule_id"].astype(int).isin(ok_rule_ids)].copy()
        ok_unit_keys = set(rule_df.get("rule_unit_key", pd.Series(dtype=str)).astype(str).tolist())
        agonist_units = [u for u in agonist_units if u.unit_key in ok_unit_keys]
        _log(args, f"{LOG_PREFIX} {baseline}: rule-powered sample rules={len(rule_df)} units={len(agonist_units)} union_rows={len(eval_indices)}", "normal")
        if len(eval_indices) < int(args.spiking_min_points):
            _log(args, f"{LOG_PREFIX} {baseline}: union rows below spiking_min_points; per-rule min still controls power", "normal")
        if not agonist_units:
            payload = {"baseline": baseline, "status": "empty_rule_conditioned_units", "n_units": 0, "n_rules": int(len(rule_df)), "sampling": sample_meta}
            (baseline_out / "threshold_spiking_experiment.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            _write_and_print_baseline_summary(baseline_out, payload)
            return payload
    else:
        eval_indices, sample_meta = select_high_n_eval_indices(args=args, model=model, scores_df=scores_df, prompt_col=prompt_col, n_points=int(args.spiking_max_points), seed=int(args.seed), cache_dir=baseline_out / "sampling_cache", use_spectral_sampling=True)
    if len(eval_indices) == 0:
        return {"baseline": baseline, "status": "empty_eval_indices", "n_rules": int(len(rule_df))}
    _log(args, f"{LOG_PREFIX} {baseline}: evaluation rows={len(eval_indices)}/{len(scores_df)}", "normal")

    control_pool_units = _make_same_layer_nonagonist_control_pool(args, model=model, agonist_units=agonist_units, seed=int(args.seed))
    control_pool_units = _filter_units_with_valid_capacity(args, model=model, units=control_pool_units, context="same-layer non-agonist control")
    eval_units = _deduplicate_unit_specs(agonist_units + control_pool_units)
    if control_pool_units:
        pd.DataFrame([u.__dict__ | {"unit_key": u.unit_key, "layer_key": u.layer_key} for u in control_pool_units]).to_csv(baseline_out / "same_layer_nonagonist_control_pool.csv", index=False)
        _log(args, f"{LOG_PREFIX} {baseline}: added {len(control_pool_units)} same-layer/head non-agonist control candidates; evaluating {len(eval_units)} total units", "normal")
    else:
        _log(args, f"{LOG_PREFIX} {baseline}: no same-layer/head non-agonist controls added", "verbose")

    scores_df_for_mean = load_scores_for_baseline(
        scores_path=scores_path,
        target_col=target_col,
        baseline_subset="all",
        task_targets=task.DEFAULT_TARGETS,
        split="train",
    )
    _log(args, f"{LOG_PREFIX} {baseline}: computing shared {args.intervention} replacement units={len(eval_units)}", "normal")
    mean_acts = precompute_replacements_for_units(model=model, units=eval_units, scores_df_for_mean=scores_df_for_mean, prompt_col=prompt_col, target_col=target_col, intervention=args.intervention, points_to_use=int(args.points_to_use_for_mean_ablation), batch_size=int(args.batch_size), seed=int(args.seed))

    examples = scores_df.to_dict(orient="records")
    scores_out, flip_stats = evaluate_singleton_flips_high_n(model=model, units=eval_units, scores_df=scores_df, examples=examples, eval_indices=eval_indices, prompt_col=prompt_col, is_answer_positive_fn=task.is_answer_positive, target_col=target_col, baseline_subset=baseline, batch_size=int(args.batch_size), decode_only=bool(args.decode_only), intervention=args.intervention, mean_activations=mean_acts, max_new_tokens=int(task.MAX_NEW_TOKENS), cache_dir=Path(args.spiking_eval_cache_dir or (baseline_out / "high_n_eval_cache")), force=bool(args.force_spiking_eval))
    scores_out.to_csv(baseline_out / "high_n_scores_with_flips.csv", index=False)

    nonagonist_population_by_key, nonagonist_payload = _select_same_layer_nonagonist_baselines(
        args, baseline=baseline, agonist_units=agonist_units, control_pool_units=control_pool_units,
        flip_stats=flip_stats, baseline_out=baseline_out,
    )

    candidate_unit_keys = {u.unit_key for u in agonist_units}
    pop_rows = []
    for r in tqdm(flip_stats.to_dict("records"), desc=f"{LOG_PREFIX} {baseline} assign populations", unit="unit", leave=False):
        pop, strength = _assign_population_from_high_n(r, args)
        unit_key = str(r.get("unit_key"))
        if unit_key in candidate_unit_keys or str(r.get("source", "")) == "flip_rule_candidate":
            pop = "flip_rule_candidate"
        elif unit_key in nonagonist_population_by_key:
            pop = nonagonist_population_by_key[unit_key]
        elif str(r.get("source", "")) == "same_layer_nonagonist_candidate":
            pop = "same_layer_nonagonist_pool_unselected"
        r["population"] = pop; r["population_strength"] = strength
        pop_rows.append(r)
    flip_stats = pd.DataFrame(pop_rows)
    flip_stats.to_csv(baseline_out / "high_n_flip_stats_by_unit.csv", index=False)

    analysis_unit_keys = {u.unit_key for u in agonist_units} | set(nonagonist_population_by_key.keys())
    analysis_units = [u for u in eval_units if u.unit_key in analysis_unit_keys]

    high_n_examples = scores_out.to_dict(orient="records")
    layer_labels = list(dict.fromkeys([u.layer_label for u in analysis_units]))
    activation_only_proxy = False
    if bool(args.decode_only):
        _result(f"{LOG_PREFIX} {baseline}: decode-only run; using decode-step activation proxies and disabling prompt-token gradient proxies")
        layer_acts = collect_reference_activations(model, high_n_examples, prompt_col, layer_labels, batch_size=int(args.batch_size), decode_only=True, max_new_tokens=int(task.MAX_NEW_TOKENS))
        layer_proxy = _make_activation_only_proxy_payload(layer_acts, objective_name="decode_only_activation_only")
        activation_only_proxy = True
    else:
        _log(args, f"{LOG_PREFIX} {baseline}: collecting proxy tensors rows={len(high_n_examples)} layers={len(layer_labels)}", "normal")
        layer_proxy = collect_reference_proxy_tensors(model, task, high_n_examples, prompt_col, layer_labels, int(args.batch_size), allow_fallback_score=bool(args.proxy_allow_fallback_score), decode_only=False, max_new_tokens=int(task.MAX_NEW_TOKENS))
        if not layer_proxy:
            _result(f"{LOG_PREFIX} {baseline}: gradient proxies unavailable; using activation-only fallback")
            layer_acts = collect_reference_activations(model, high_n_examples, prompt_col, layer_labels, batch_size=int(args.batch_size), decode_only=False, max_new_tokens=int(task.MAX_NEW_TOKENS))
            layer_proxy = _make_activation_only_proxy_payload(layer_acts, objective_name="activation_only")
            activation_only_proxy = True

    raw_parts = []
    unit_tests = []
    targets = _parse_flip_targets(args)
    feature_list = _parse_proxy_features(args)
    if activation_only_proxy:
        feature_list = _activation_only_proxy_features(feature_list)
    stats_by_key = {str(r.unit_key): r for r in flip_stats.itertuples(index=False)}
    for u in tqdm(analysis_units, desc=f"{LOG_PREFIX} {baseline} threshold proxy units", unit="unit"):
        stat = stats_by_key.get(u.unit_key)
        if stat is None:
            continue
        udf = proxy_rows_for_unit(u, layer_proxy.get(u.layer_label), model=model, intervention=args.intervention, mean_activations=mean_acts, baseline=baseline, requested_features=feature_list)
        if udf.empty:
            continue
        layer_key = u.layer_key
        colmap = {"flip_any": f"flip_{layer_key}_{int(u.neuron_id)}", "flip_c2i": f"flip_c2i_{layer_key}_{int(u.neuron_id)}", "flip_i2c": f"flip_i2c_{layer_key}_{int(u.neuron_id)}"}
        for t, col in colmap.items():
            if col in scores_out.columns:
                udf[t] = scores_out[col].fillna(False).astype(bool).astype(int).to_numpy()[:len(udf)]
        udf["baseline_subset"] = baseline
        udf["population"] = str(getattr(stat, "population"))
        udf["population_strength"] = float(getattr(stat, "population_strength"))
        raw_parts.append(udf)
        for target in targets:
            if target not in udf:
                continue
            direction_family = "learned_support_loss" if baseline == "positive" and target == "flip_c2i" else ("rescue_or_anti_support" if baseline == "negative" and target == "flip_i2c" else "other")
            for feature in feature_list:
                if feature not in udf:
                    continue
                res = repeated_holdout(udf[feature], udf[target], min_examples=int(args.threshold_event_min_examples), repeats=int(args.threshold_event_repeats), holdout_fraction=float(args.threshold_event_holdout_fraction), seed=int(args.seed))
                if res is None:
                    continue
                spike_counts = _threshold_spike_counts(udf[feature], udf[target], res.get("representative_threshold"), res.get("representative_direction"))
                unit_tests.append({"baseline_subset": baseline, "population": str(getattr(stat, "population")), "unit_key": u.unit_key, "layer_label": u.layer_label, "layer_key": u.layer_key, "neuron_id": int(u.neuron_id), "population_strength": float(getattr(stat, "population_strength")), "target": target, "direction_family": direction_family, "feature": feature, "n_rows": int(len(udf)), "n_flips": int(pd.to_numeric(udf[target], errors="coerce").fillna(0).sum()), **res, **spike_counts})

    raw_df = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    raw_df.to_csv(baseline_out / "threshold_activation_flip_rows.csv.gz", index=False, compression="gzip")
    unit_df = pd.DataFrame(unit_tests)
    unit_df.to_csv(baseline_out / "threshold_unit_tests.csv", index=False)

    summary_rows = []
    if not unit_df.empty:
        for keys, g in tqdm(list(unit_df.groupby(["baseline_subset", "population", "target", "direction_family", "feature"], dropna=False)), desc=f"{LOG_PREFIX} {baseline} population summaries", unit="group", leave=False):
            b, pop, target, direction_family, feature = keys
            summary_rows.append({"baseline_subset": b, "population": pop, "target": target, "direction_family": direction_family, "feature": feature, "n_unit_tests": int(len(g)), "n_units": int(g["unit_key"].nunique()), "median_test_auc_oriented": float(pd.to_numeric(g.get("median_test_auc_oriented"), errors="coerce").median()), "median_test_abs_mcc": float(pd.to_numeric(g.get("median_test_abs_mcc"), errors="coerce").median()), "mean_test_auc_oriented": float(pd.to_numeric(g.get("mean_test_auc_oriented"), errors="coerce").mean()), "mean_test_abs_mcc": float(pd.to_numeric(g.get("mean_test_abs_mcc"), errors="coerce").mean()), "median_full_threshold_spike_rate": float(pd.to_numeric(g.get("full_threshold_spike_rate"), errors="coerce").median()), "median_full_false_positive_rate": float(pd.to_numeric(g.get("full_false_positive_rate"), errors="coerce").median()), "median_test_threshold_spike_rate": float(pd.to_numeric(g.get("median_test_threshold_spike_rate"), errors="coerce").median()), "median_test_false_positive_rate": float(pd.to_numeric(g.get("median_test_false_positive_rate"), errors="coerce").median())})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(baseline_out / "threshold_population_summary.csv", index=False)

    binned_rows = []
    if not raw_df.empty and not unit_df.empty:
        for r in tqdm(unit_df.sort_values("median_test_auc_oriented", ascending=False).head(500).to_dict("records"), desc=f"{LOG_PREFIX} {baseline} binned curves", unit="curve", leave=False):
            udf = raw_df.loc[raw_df["unit_key"].astype(str) == str(r["unit_key"])]
            if not udf.empty:
                binned_rows.extend(_make_binned_rows(udf, r["feature"], r["target"], r["population"], r["unit_key"], r.get("representative_direction", ">="), int(args.threshold_event_n_bins)))
    pd.DataFrame(binned_rows).to_csv(baseline_out / "threshold_binned_flip_curves.csv", index=False)

    rule_payload = None
    if bool(getattr(args, "rule_conditioned_diagnostics", False)):
        rule_payload = _rule_conditioned_threshold_tests(args, baseline=baseline, baseline_out=baseline_out, rule_df=rule_df, scores_out=scores_out, raw_df=raw_df, dataset_info=dataset_info, task_targets=task.DEFAULT_TARGETS, feature_list=feature_list)

    payload = {"baseline": baseline, "status": "ok", "evaluation_split": str(args.evaluation_split), "candidate_source": candidate_source, "candidate_flip_stats_path": str(args.candidate_flip_stats_path) if args.candidate_flip_stats_path else None, "mean_replacement_reference_split": "train", "n_scores_available": int(len(scores_df)), "n_high_n_rows": int(len(scores_out)), "n_units": int(len(analysis_units)), "n_eval_units": int(len(eval_units)), "n_rules": int(len(rule_df)), "rule_metrics_path": str(rule_path) if rule_path is not None else None, "rule_conditioned_only": bool(rule_conditioned_only), "sampling": sample_meta, "same_layer_nonagonist_controls": nonagonist_payload, "population_counts": flip_stats["population"].value_counts().to_dict() if not flip_stats.empty else {}, "rule_conditioned": rule_payload, "files": {"scores_with_flips": "high_n_scores_with_flips.csv", "flip_stats": "high_n_flip_stats_by_unit.csv", "unit_tests": "threshold_unit_tests.csv", "population_summary": "threshold_population_summary.csv", "binned_curves": "threshold_binned_flip_curves.csv", "activation_flip_rows_gz": "threshold_activation_flip_rows.csv.gz", "same_layer_nonagonist_control_pool": "same_layer_nonagonist_control_pool.csv", "same_layer_nonagonist_control_selection": "same_layer_nonagonist_control_selection.csv", "rule_conditioned_sampling_plan": "rule_conditioned_sampling_plan.csv", "flip_conditioned_threshold_summary": "flip_conditioned_threshold_summary.csv"}}
    (baseline_out / "threshold_spiking_experiment.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_and_print_baseline_summary(baseline_out, payload)
    return payload


def _plot_aggregate_visualizations(out_root: Path, files: dict):
    viz_dir = out_root / "figures"
    viz_dir.mkdir(parents=True, exist_ok=True)
    made = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _log(SimpleNamespace(log_level="normal"), f"{LOG_PREFIX} matplotlib unavailable; skipping figures: {e}", "normal")
        return made

    def _barh(df, value_col, label_col, title, xlabel, path, top=30):
        d = df.dropna(subset=[value_col]).copy().sort_values(value_col, ascending=True).tail(top)
        if d.empty:
            return False
        fig_h = max(4, 0.28 * len(d) + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_h))
        ax.barh(np.arange(len(d)), d[value_col].to_numpy())
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(d[label_col].astype(str).tolist(), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return True

    pop_path = out_root / "aggregate_population_summary.csv"
    if pop_path.exists():
        try:
            pop = pd.read_csv(pop_path)
            if not pop.empty:
                pop["plot_label"] = pop["baseline_subset"].astype(str)+" | "+pop["target"].astype(str)+" | "+pop["feature"].astype(str)
                focus = pop.loc[pop.get("population", "").astype(str).isin(["flip_rule_candidate", "flip_supported_rule_candidate", "overtopping_high_strength"])] if "population" in pop else pop
                if focus.empty:
                    focus = pop
                agg = focus.groupby(["plot_label", "feature"], dropna=False)["median_test_auc_oriented"].median().reset_index()
                path = viz_dir / "aggregate_proxy_auc_top_features.pdf"
                if _barh(agg, "median_test_auc_oriented", "plot_label", "Proxy threshold-event AUC by target/feature", "Median oriented AUC", path):
                    made["proxy_auc_top_features"] = str(path.name)
                learned = pop.loc[pop.get("direction_family", "").astype(str) == "learned_support_loss"].copy() if "direction_family" in pop else pd.DataFrame()
                if not learned.empty:
                    learned_agg = learned.groupby("feature", dropna=False)["median_test_auc_oriented"].median().reset_index()
                    path = viz_dir / "aggregate_learned_direction_proxy_auc.pdf"
                    if _barh(learned_agg, "median_test_auc_oriented", "feature", "Learned-direction overtopping proxy AUC", "Median oriented AUC", path):
                        made["learned_direction_proxy_auc"] = str(path.name)
        except Exception as e:
            _log(SimpleNamespace(log_level="normal"), f"{LOG_PREFIX} population summary plotting failed: {e}", "normal")

    flip_path = out_root / "aggregate_flip_conditioned_threshold_summary.csv"
    if flip_path.exists():
        try:
            flip = pd.read_csv(flip_path)
            if not flip.empty:
                flip["plot_label"] = flip["test_kind"].astype(str)+" | "+flip["feature"].astype(str)
                agg = flip.groupby("plot_label", dropna=False)["median_test_auc_oriented"].median().reset_index()
                path = viz_dir / "aggregate_flip_conditioned_proxy_auc.pdf"
                if _barh(agg, "median_test_auc_oriented", "plot_label", "Flip-conditioned proxy AUC", "Median oriented AUC", path):
                    made["flip_conditioned_proxy_auc"] = str(path.name)
        except Exception as e:
            _log(SimpleNamespace(log_level="normal"), f"{LOG_PREFIX} flip-conditioned plotting failed: {e}", "normal")

    unit_path = out_root / "aggregate_unit_tests.csv"
    if unit_path.exists():
        try:
            u = pd.read_csv(unit_path)
            if not u.empty and {"population_strength", "median_test_auc_oriented"}.issubset(u.columns):
                g = u.dropna(subset=["population_strength", "median_test_auc_oriented"]).copy()
                if len(g) > 2:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(g["population_strength"], g["median_test_auc_oriented"], s=10, alpha=0.5)
                    ax.set_xlabel("Ablation singleton strength")
                    ax.set_ylabel("Median oriented AUC")
                    ax.set_title("Proxy score vs ablation-defined strength")
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    path = viz_dir / "aggregate_proxy_auc_vs_ablation_strength.pdf"
                    fig.savefig(path, dpi=160)
                    plt.close(fig)
                    made["proxy_auc_vs_ablation_strength"] = str(path.name)
        except Exception as e:
            _log(SimpleNamespace(log_level="normal"), f"{LOG_PREFIX} scatter plotting failed: {e}", "normal")
    return made


def write_aggregate(out_root: Path, *, make_visualizations: bool = True):
    frames = defaultdict(list)
    baseline_dirs = sorted(out_root.glob("*_baseline"))
    for base_dir in tqdm(baseline_dirs, desc=f"{LOG_PREFIX} aggregate baselines", unit="baseline", leave=False):
        for name, key in [
            ("high_n_flip_stats_by_unit.csv", "flip_stats"),
            ("threshold_unit_tests.csv", "unit_tests"),
            ("threshold_population_summary.csv", "population_summary"),
            ("threshold_binned_flip_curves.csv", "binned_curves"),
            ("threshold_activation_flip_rows.csv.gz", "activation_flip_rows"),
            ("flip_conditioned_candidate_units.csv", "flip_conditioned_candidate_units"),
            ("rule_conditioned_sampling_plan.csv", "rule_conditioned_sampling_plan"),
            ("flip_conditioned_flip_support.csv", "flip_conditioned_flip_support"),
            ("flip_conditioned_threshold_unit_tests.csv", "flip_conditioned_threshold_unit_tests"),
            ("flip_conditioned_threshold_summary.csv", "flip_conditioned_threshold_summary"),
            ("same_layer_nonagonist_control_selection.csv", "same_layer_nonagonist_control_selection"),
        ]:
            p = base_dir / name
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    df["source_baseline_dir"] = base_dir.name
                    frames[key].append(df)
                except Exception:
                    pass
    files = {}
    for key, fms in tqdm(list(frames.items()), desc=f"{LOG_PREFIX} write aggregate tables", unit="table", leave=False):
        if fms:
            df = pd.concat(fms, ignore_index=True)
            fname = f"aggregate_{key}.csv"
            df.to_csv(out_root / fname, index=False)
            files[key] = fname
    if frames.get("population_summary"):
        summary = pd.concat(frames["population_summary"], ignore_index=True)
        comp_rows = []
        group_cols = ["baseline_subset", "target", "feature"]
        if "direction_family" in summary.columns:
            group_cols.append("direction_family")
        for keys, g in tqdm(list(summary.groupby(group_cols, dropna=False)), desc=f"{LOG_PREFIX} compare populations", unit="group", leave=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_payload = dict(zip(group_cols, keys))
            by = {str(r.population): r for r in g.itertuples(index=False)} if "population" in g else {}
            overt = by.get("flip_rule_candidate") or by.get("flip_supported_rule_candidate") or by.get("overtopping_high_strength")
            if overt is None:
                continue
            for cname in ["random_nonagonist_control", "weak_agonist", "moderate_strength", "non_overtopping_control"]:
                ctrl = by.get(cname)
                if ctrl is None:
                    continue
                row = dict(key_payload)
                row.update({"control_population": cname, "overtopping_median_auc": float(getattr(overt, "median_test_auc_oriented")), "control_median_auc": float(getattr(ctrl, "median_test_auc_oriented")), "delta_median_auc": float(getattr(overt, "median_test_auc_oriented")) - float(getattr(ctrl, "median_test_auc_oriented")), "overtopping_median_abs_mcc": float(getattr(overt, "median_test_abs_mcc")), "control_median_abs_mcc": float(getattr(ctrl, "median_test_abs_mcc")), "delta_median_abs_mcc": float(getattr(overt, "median_test_abs_mcc")) - float(getattr(ctrl, "median_test_abs_mcc")), "overtopping_median_threshold_spike_rate": float(getattr(overt, "median_full_threshold_spike_rate", np.nan)), "control_median_threshold_spike_rate": float(getattr(ctrl, "median_full_threshold_spike_rate", np.nan)), "delta_median_threshold_spike_rate": float(getattr(overt, "median_full_threshold_spike_rate", np.nan)) - float(getattr(ctrl, "median_full_threshold_spike_rate", np.nan))})
                comp_rows.append(row)
        comp = pd.DataFrame(comp_rows)
        comp.to_csv(out_root / "aggregate_overtopping_vs_controls.csv", index=False)
        files["comparison"] = "aggregate_overtopping_vs_controls.csv"

    # Pairwise matched-control summaries were removed with flip-count-matched
    # controls. The remaining same-layer/head baseline is random and is compared
    # at the population level above.
    figures = _plot_aggregate_visualizations(out_root, files) if make_visualizations else {}
    if figures:
        files["figures"] = figures
    payload = {"status": "ok", "out_dir": str(out_root), "files": files}
    (out_root / "threshold_spiking_experiment_aggregate.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload

def main():
    args = parse_args()
    if str(getattr(args, "log_level", "quiet")) == "quiet":
        warnings.filterwarnings("ignore", category=FutureWarning)
    set_deterministic(int(args.seed))
    input_data_dir = Path(args.input_data_dir).resolve()
    dataset_info = _load_dataset_info(input_data_dir)
    task = resolve_task_spec(args.task_module)
    prompt_col = dataset_info.get("prompt_col") or task.DEFAULT_INPUT
    target_col = dataset_info.get("target_col") or task.DEFAULT_TARGETS[0]
    ai_model = args.ai_model or dataset_info.get("ai_model")
    if not ai_model:
        raise ValueError("Could not resolve ai_model from --ai_model or dataset_info.json")
    out_root = Path(args.out_dir).resolve() if args.out_dir else input_data_dir / "spiking_diagnostics"
    out_root.mkdir(parents=True, exist_ok=True)
    runs = []
    baselines = [b.strip() for b in str(args.baseline_subsets).split(",") if b.strip()]
    for baseline in tqdm(baselines, desc=f"{LOG_PREFIX} baselines", unit="baseline"):
        bnorm = "positive" if baseline.lower().startswith("pos") else "negative" if baseline.lower().startswith("neg") else baseline
        runs.append(run_for_baseline(args, bnorm, dataset_info=dataset_info, task=task, prompt_col=prompt_col, target_col=target_col, ai_model=ai_model, out_root=out_root))
    agg = write_aggregate(out_root, make_visualizations=bool(args.make_visualizations))
    final = {"status": "ok", "out_dir": str(out_root), "runs": runs, "aggregate": agg}
    _write_and_print_aggregate_summary(out_root, agg, runs)
    if bool(getattr(args, "print_json_summary", False)):
        _result(json.dumps(final, indent=2, default=str))


if __name__ == "__main__":
    main()
