#!/usr/bin/env python3
"""Freeze Stage-6 poisoning CHA candidates without held-out singleton evaluation.

The poisoning workflow uses Stage-6 CHA discovery to define checkpoint-local candidate sets. This utility reads Stage-6
``neuron_buckets.json`` directly, materializes the deterministic discovery
ranking, performs no held-out ablations, and does not import/load model code.

Stage 6 writes its outputs below ``neural_circuit_discovery_results``. Older
poisoning runs only exposed a frozen ranking below ``rule_extraction_results``
as a side effect of the generic singleton evaluator. The streamlined pipeline
intentionally skips that redundant evaluator, so this freezer must consume the
Stage-6 output directly.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

_TAU_RE = re.compile(r"(?:^|-)tau(?P<tau>[0-9]+(?:\.[0-9]+)?)(?:-|$)")


def _tau_from_path(path: Path) -> float | None:
    for part in reversed(path.parts):
        match = _TAU_RE.search(part)
        if match:
            try:
                return float(match.group("tau"))
            except ValueError:
                return None
    return None


def _layer_sort_key(layer_label: str) -> int:
    nums = re.findall(r"\d+", str(layer_label))
    return int(nums[0]) if nums else 10**9


def _layer_metadata(layer_label: str) -> dict[str, object]:
    label = str(layer_label)
    m = re.search(r"m(\d+)", label)
    if m:
        return {
            "transformer_layer": int(m.group(1)),
            "computational_locus": "mlp_output",
            "channel_type": "mlp_neuron",
        }
    m = re.search(r"a(\d+)\.h(\d+)", label) or re.search(r"[lL](\d+)[hH](\d+)", label)
    if m:
        return {
            "transformer_layer": int(m.group(1)),
            "computational_locus": f"attention_head_{int(m.group(2))}",
            "channel_type": "attention_dimension",
        }
    return {
        "transformer_layer": None,
        "computational_locus": "unknown",
        "channel_type": "unknown",
    }


def _bucket_keep_keys(buckets: dict) -> set[str]:
    """Match the Stage-7 policy: non-catastrophic agonists minus catastrophic buckets."""
    nca = buckets.get("non_catastrophic_agonists", {})
    keep = set(nca) if isinstance(nca, dict) else set()
    cz = buckets.get("catastrophic_zero", {})
    if isinstance(cz, dict):
        bad: set[str] = set()
        for name in ("confirmed", "not_always", "candidates"):
            block = cz.get(name, {})
            if isinstance(block, dict):
                bad.update(str(k) for k in block)
        keep.difference_update(bad)
    return keep


def _resolve_stage6_bucket_files(
    endpoint: Path,
    *,
    search_epsilon: float,
    explicit_root: str | None,
) -> tuple[Path, list[Path]]:
    """Return one Stage-6 configuration root and its baseline bucket files.

    A single configuration may contain both ``positive_baseline`` and
    ``negative_baseline`` results. Those are intentionally merged. Distinct
    Stage-6 configurations (different M/tau/intervention trees) are never
    merged silently.
    """
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if root.is_file():
            if root.name != "neuron_buckets.json":
                raise ValueError(
                    "--discovery_root may be a Stage-6 directory or neuron_buckets.json; "
                    f"got file {root}"
                )
            return root.parent.parent, [root]
        if not root.is_dir():
            raise FileNotFoundError(f"Stage-6 discovery root does not exist: {root}")
        bucket_files = sorted(
            p for p in root.rglob("neuron_buckets.json")
            if p.parent.name in {"positive_baseline", "negative_baseline"}
        )
        if not bucket_files:
            raise FileNotFoundError(f"No positive/negative Stage-6 neuron_buckets.json under {root}")
        config_roots = {p.parent.parent for p in bucket_files}
        if len(config_roots) != 1:
            raise RuntimeError(
                "Explicit discovery root contains multiple Stage-6 configurations; "
                f"refusing to merge them: {[str(x) for x in sorted(config_roots)]}"
            )
        return next(iter(config_roots)), bucket_files

    stage6_roots = sorted(
        p for p in endpoint.glob("neural_circuit_discovery_results*") if p.is_dir()
    )
    bucket_files: list[Path] = []
    for root in stage6_roots:
        bucket_files.extend(
            p for p in root.rglob("neuron_buckets.json")
            if p.parent.name in {"positive_baseline", "negative_baseline"}
        )

    if not bucket_files:
        searched = ", ".join(str(p) for p in stage6_roots) or str(endpoint / "neural_circuit_discovery_results*")
        raise FileNotFoundError(
            "Missing Stage-6 CHA neuron_buckets.json. "
            f"Searched positive/negative baseline outputs under: {searched}"
        )

    matching: list[Path] = []
    for bucket in sorted(set(bucket_files)):
        tau = _tau_from_path(bucket)
        if tau is None or math.isclose(tau, float(search_epsilon), rel_tol=0.0, abs_tol=1e-12):
            matching.append(bucket)
    if not matching:
        raise RuntimeError(
            f"Found Stage-6 outputs, but none match search_epsilon={search_epsilon:g}: "
            + "; ".join(str(p) for p in sorted(set(bucket_files))[:8])
        )

    by_config: dict[Path, list[Path]] = defaultdict(list)
    for bucket in matching:
        by_config[bucket.parent.parent].append(bucket)
    if len(by_config) != 1:
        raise RuntimeError(
            "Ambiguous Stage-6 discovery outputs. Refusing to merge stale/configuration variants. "
            "Remove stale variants or pass --discovery_root explicitly. Configurations: "
            + "; ".join(str(p) for p in sorted(by_config)[:8])
        )
    config_root, paths = next(iter(by_config.items()))
    return config_root, sorted(paths)

def _ranking_from_buckets(bucket_path: Path, *, search_epsilon: float) -> pd.DataFrame:
    buckets = json.loads(bucket_path.read_text(encoding="utf-8"))
    nca = buckets.get("non_catastrophic_agonists", {})
    if not isinstance(nca, dict):
        nca = {}
    keep = _bucket_keep_keys(buckets)

    records: dict[tuple[str, int], dict[str, object]] = {}
    discovery_baselines: dict[tuple[str, int], set[str]] = defaultdict(set)
    for key in sorted(keep):
        entry = nca.get(key, {})
        if not isinstance(entry, dict):
            continue
        rec = entry.get("last_record") or {}
        try:
            signed = float(rec.get("max_effect"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(signed) or abs(signed) < float(search_epsilon):
            continue

        if "layer_label" in rec and "neuron_id" in rec:
            layer = str(rec["layer_label"])
            try:
                neuron_id = int(rec["neuron_id"])
            except (TypeError, ValueError):
                continue
        else:
            try:
                layer, raw_id = str(key).rsplit(":", 1)
                neuron_id = int(raw_id)
            except (ValueError, TypeError):
                continue

        unit = (layer, neuron_id)
        baseline = str(rec.get("baseline_subset", "")).strip().lower()
        if baseline not in {"positive", "negative"}:
            parent_baseline = bucket_path.parent.name.replace("_baseline", "").strip().lower()
            if parent_baseline in {"positive", "negative"}:
                baseline = parent_baseline
        if baseline in {"positive", "negative"}:
            discovery_baselines[unit].add(baseline)

        row: dict[str, object] = {
            "layer_label": layer,
            "neuron_id": neuron_id,
            "unit_key": f"{layer}:{neuron_id}",
            "discovery_score": abs(signed),
            "discovery_score_signed": signed,
            "discovery_baseline_subset": baseline if baseline in {"positive", "negative"} else "",
            "ranking_source": "stage6_abs_max_effect",
            "ranking_source_file": str(bucket_path),
        }
        row.update(_layer_metadata(layer))
        previous = records.get(unit)
        if previous is None or float(row["discovery_score"]) > float(previous["discovery_score"]):
            records[unit] = row

    rows = list(records.values())
    for row in rows:
        unit = (str(row["layer_label"]), int(row["neuron_id"]))
        baselines = sorted(discovery_baselines.get(unit, set()))
        row["discovery_baseline_subsets"] = "|".join(baselines)
        row["n_discovery_baseline_subsets"] = len(baselines)

    rows.sort(
        key=lambda row: (
            -float(row["discovery_score"]),
            _layer_sort_key(str(row["layer_label"])),
            str(row["layer_label"]),
            int(row["neuron_id"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["discovery_rank_global"] = rank

    by_layer: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_layer[str(row["layer_label"])].append(row)
    for layer_rows in by_layer.values():
        for rank, row in enumerate(layer_rows, start=1):
            row["discovery_rank_within_layer"] = rank

    columns = [
        "layer_label",
        "neuron_id",
        "unit_key",
        "discovery_score",
        "discovery_score_signed",
        "discovery_baseline_subset",
        "discovery_baseline_subsets",
        "n_discovery_baseline_subsets",
        "ranking_source",
        "ranking_source_file",
        "transformer_layer",
        "computational_locus",
        "channel_type",
        "discovery_rank_global",
        "discovery_rank_within_layer",
    ]
    return pd.DataFrame(rows, columns=columns)


def _merge_rankings(bucket_paths: list[Path], *, search_epsilon: float) -> pd.DataFrame:
    frames = [_ranking_from_buckets(p, search_epsilon=search_epsilon) for p in bucket_paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _ranking_from_buckets(bucket_paths[0], search_epsilon=search_epsilon).iloc[0:0].copy()
    combined = pd.concat(frames, ignore_index=True)
    records: dict[tuple[str, int], dict[str, object]] = {}
    baselines: dict[tuple[str, int], set[str]] = defaultdict(set)
    source_files: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in combined.to_dict("records"):
        unit = (str(row["layer_label"]), int(row["neuron_id"]))
        for b in str(row.get("discovery_baseline_subsets", row.get("discovery_baseline_subset", ""))).split("|"):
            b = b.strip().lower()
            if b in {"positive", "negative"}: baselines[unit].add(b)
        source_files[unit].add(str(row.get("ranking_source_file", "")))
        prev = records.get(unit)
        score = float(row.get("discovery_score", float("nan")))
        if prev is None or (math.isfinite(score) and score > float(prev.get("discovery_score", float("-inf")))):
            records[unit] = dict(row)
    rows = []
    for unit, row in records.items():
        bs = sorted(baselines[unit])
        row["discovery_baseline_subsets"] = "|".join(bs)
        row["n_discovery_baseline_subsets"] = len(bs)
        row["ranking_source_file"] = "|".join(sorted(x for x in source_files[unit] if x))
        rows.append(row)
    rows.sort(key=lambda r: (-float(r["discovery_score"]), _layer_sort_key(str(r["layer_label"])), str(r["layer_label"]), int(r["neuron_id"])))
    for i, row in enumerate(rows, 1): row["discovery_rank_global"] = i
    by_layer: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows: by_layer[str(row["layer_label"])].append(row)
    for layer_rows in by_layer.values():
        for i, row in enumerate(layer_rows, 1): row["discovery_rank_within_layer"] = i
    return pd.DataFrame(rows, columns=frames[0].columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint_dir", required=True)
    ap.add_argument("--search_epsilon", type=float, required=True)
    ap.add_argument("--source_label", default="observed_training_mixture_correctness")
    ap.add_argument("--oracle_cohort_definition_used", action="store_true")
    ap.add_argument(
        "--discovery_root",
        default=None,
        help=(
            "Optional exact Stage-6 configuration/baseline directory (or neuron_buckets.json). "
            "Normally auto-resolved from neural_circuit_discovery_results."
        ),
    )
    args = ap.parse_args()

    endpoint = Path(args.endpoint_dir).expanduser().resolve()
    if not endpoint.is_dir():
        raise FileNotFoundError(f"CHA endpoint directory does not exist: {endpoint}")

    discovery_root, bucket_paths = _resolve_stage6_bucket_files(
        endpoint,
        search_epsilon=float(args.search_epsilon),
        explicit_root=args.discovery_root,
    )
    print("[candidate-freeze] Stage-6 sources: " + ", ".join(str(p) for p in bucket_paths))
    baseline_names = {p.parent.name.replace("_baseline", "") for p in bucket_paths}
    if args.source_label == "observed_training_mixture_correctness":
        required = {"positive", "negative"}
        if not required.issubset(baseline_names):
            raise RuntimeError(
                "Observed-mixture localization now requires both positive and negative baseline CHA outputs. "
                f"Found {sorted(baseline_names)} under {discovery_root}. Re-run the observed-mixture CHA; "
                "the raw llm_io_data.pkl and feature_report/scores.csv can be reused."
            )

    ranking = _merge_rankings(bucket_paths, search_epsilon=float(args.search_epsilon))

    out = endpoint / "candidate_localization"
    out.mkdir(parents=True, exist_ok=True)
    ranking_path = out / "frozen_candidate_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    (out / "candidate_localization.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": f"stage6_{args.source_label}_cha",
                "stage6_discovery_root": str(discovery_root),
                "stage6_neuron_buckets": [str(p) for p in bucket_paths],
                "baseline_subsets_merged": sorted({p.parent.name.replace("_baseline", "") for p in bucket_paths}),
                "ranking_source": "stage6_abs_max_effect",
                "search_epsilon": float(args.search_epsilon),
                "n_candidates": int(len(ranking)),
                "heldout_singleton_evaluation_used": False,
                "oracle_attack_or_poison_annotations_used": bool(args.oracle_cohort_definition_used),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[candidate-freeze] wrote {len(ranking)} candidates to {ranking_path}")


if __name__ == "__main__":
    main()
