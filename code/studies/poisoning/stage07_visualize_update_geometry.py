#!/usr/bin/env python3
"""Visualize clean-vs-poison LoRA update geometry from existing poisoning results.

This is a post-hoc analysis.  It does not rerun training, CHA, singleton
interventions, or WANDA scoring.  It uses:

  * persisted clean/poisoned LoRA checkpoints,
  * Stage-07 mapped_disruptive_channels.csv files, and
  * Stage-07 control_correctness_channel_disruption.csv files.

The model-level geometry is computed exactly in the Frobenius metric of the
*effective* LoRA update W = scaling * (B @ A), without materializing dense W.
For two low-rank products P=sBA and Q=tDC,

    <P,Q>_F = s t sum[(B^T D) * (A C^T)].

Only the overtopping/matched-control rows are materialized as dense vectors.

Outputs (under --output_dir, default Stage-07/<phase>/derived_update_geometry):

  update_geometry_by_interval.csv
  update_geometry_pca.csv
  channel_update_geometry.csv
  matched_control_update_geometry.csv
  disruption_information_by_interval.csv
  01_model_update_direction_and_magnitude.pdf
  02_clean_vs_poisoned_update_map.pdf
  03_causal_channels_vs_local_update_geometry.pdf
  04_causal_channels_vs_matched_rows.pdf

Causal-change concentration is plotted by `stage07_analyze_overtopping_poisoning`
with an explicit clean-drift comparison, so it is not duplicated here.

Use --tables_only when this module is being called as an intermediate step for
stage07_analyze_overtopping_poisoning.

Important interpretation:
  * attention CHA channels are mapped to v_proj rows by the existing Stage-07
    proxy mapping; GQA can collapse multiple CHA channels onto one parameter row.
  * the entropy/effective-support quantities describe the distribution of
    |D_j| causal-disruption mass.  They are information-theoretic concentration
    summaries, not a claim that continuous transformer activations satisfy the
    classical Information Bottleneck objective.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from studies.poisoning.lib.run_paths import (
    detection_dir,
    format_fraction_percent,
    phase_dirname,
    resolve_manifest_checkpoint_dir,
)
from studies.poisoning.lib.specificity import truthy
from studies.poisoning.stage07_detect_poisoning_examples import (
    _LORA_FACTOR_RE,
    _adapter_scaling,
    _interval_label,
    _load_adapter_tensors,
    _load_manifest_by_condition,
)
from studies.poisoning.tasks.registry import infer_task_from_run


EPS = 1e-30


@dataclass(frozen=True)
class LowRankProduct:
    """One effective LoRA matrix scaling * (B @ A)."""

    A: torch.Tensor  # [rank, in]
    B: torch.Tensor  # [out, rank]
    scaling: float
    module_name: str


State = dict[tuple[int, str], LowRankProduct]
# A vector is a per-module linear combination of effective LoRA products.
Combo = dict[tuple[int, str], list[tuple[float, LowRankProduct]]]



def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if np.isfinite(out) else math.nan


def _read_csv_or_empty(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read an optional Stage-07 table, treating a zero-byte CSV as no rows."""
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_lora_state(checkpoint_dir: Path) -> State:
    """Load low-rank LoRA factors without materializing B@A."""
    factors: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    module_names: dict[tuple[int, str], str] = {}
    for name, tensor in _load_adapter_tensors(checkpoint_dir).items():
        match = _LORA_FACTOR_RE.search(name)
        if not match:
            continue
        key = (int(match.group("layer")), str(match.group("module")))
        factor = str(match.group("factor"))
        if factor in factors.setdefault(key, {}):
            raise RuntimeError(f"Duplicate LoRA factor {factor} for {key} in {checkpoint_dir}")
        factors[key][factor] = tensor.contiguous()
        module_names[key] = f"layers.{key[0]}.{key[1]}"

    out: State = {}
    for key, pair in factors.items():
        if set(pair) != {"A", "B"}:
            raise RuntimeError(f"Incomplete LoRA factors for {key} in {checkpoint_dir}")
        A, B = pair["A"], pair["B"]
        if A.ndim != 2 or B.ndim != 2 or B.shape[1] != A.shape[0]:
            raise RuntimeError(f"Unexpected LoRA shapes A={tuple(A.shape)} B={tuple(B.shape)} for {key}")
        scaling = _adapter_scaling(
            checkpoint_dir,
            module_name=module_names[key],
            rank=int(A.shape[0]),
        )
        out[key] = LowRankProduct(A=A, B=B, scaling=float(scaling), module_name=module_names[key])
    if not out:
        raise RuntimeError(f"No supported Stage-07 LoRA factors found in {checkpoint_dir}")
    return out


def product_inner(a: LowRankProduct, b: LowRankProduct) -> float:
    """Exact Frobenius inner product <sBA, tDC> using only low-rank factors."""
    if a.A.shape[1] != b.A.shape[1] or a.B.shape[0] != b.B.shape[0]:
        raise ValueError(
            f"LoRA product shape mismatch: A/B={tuple(a.A.shape)}/{tuple(a.B.shape)} "
            f"vs {tuple(b.A.shape)}/{tuple(b.B.shape)}"
        )
    # <BA,DC> = sum_ij (B^T D)_ij (A C^T)_ij.
    left = a.B.transpose(0, 1) @ b.B
    right = a.A @ b.A.transpose(0, 1)
    return float(a.scaling * b.scaling * torch.sum(left * right).item())


def make_combo(terms: Iterable[tuple[float, State]]) -> Combo:
    combo: Combo = {}
    for coeff, state in terms:
        coeff = float(coeff)
        if coeff == 0.0:
            continue
        for key, product in state.items():
            combo.setdefault(key, []).append((coeff, product))
    return combo


def combo_inner(a: Combo, b: Combo) -> float:
    total = 0.0
    for key in set(a).intersection(b):
        for ca, pa in a[key]:
            for cb, pb in b[key]:
                total += float(ca) * float(cb) * product_inner(pa, pb)
    return float(total)


def combo_norm(a: Combo) -> float:
    return math.sqrt(max(0.0, combo_inner(a, a)))


def safe_cosine(dot: float, norm_a: float, norm_b: float) -> float:
    if norm_a <= EPS or norm_b <= EPS:
        return math.nan
    return float(np.clip(dot / (norm_a * norm_b), -1.0, 1.0))


def orthogonal_fraction(*, poison_norm: float, clean_norm: float, dot_pc: float) -> float:
    """||poison - proj_clean(poison)|| / ||poison||."""
    if poison_norm <= EPS:
        return math.nan
    if clean_norm <= EPS:
        return 1.0
    orth_sq = poison_norm**2 - (dot_pc**2 / (clean_norm**2))
    return math.sqrt(max(0.0, orth_sq)) / poison_norm


def effective_row(product: LowRankProduct, row: int) -> torch.Tensor:
    row = int(row)
    if not 0 <= row < int(product.B.shape[0]):
        raise IndexError(f"row={row} outside {product.module_name} output size {product.B.shape[0]}")
    return (product.B[row].to(torch.float64) @ product.A.to(torch.float64)) * float(product.scaling)


def row_delta(start: State, end: State, key: tuple[int, str], row: int) -> torch.Tensor:
    if key not in start or key not in end:
        raise KeyError(f"Missing LoRA module {key} in interval state")
    return effective_row(end[key], row) - effective_row(start[key], row)


def vector_metrics(clean: torch.Tensor, poison: torch.Tensor) -> dict[str, float]:
    clean = clean.to(torch.float64)
    poison = poison.to(torch.float64)
    excess = poison - clean
    clean_norm = float(torch.linalg.vector_norm(clean).item())
    poison_norm = float(torch.linalg.vector_norm(poison).item())
    excess_norm = float(torch.linalg.vector_norm(excess).item())
    dot_pc = float(torch.dot(poison, clean).item())
    cosine = safe_cosine(dot_pc, poison_norm, clean_norm)
    return {
        "clean_update_norm": clean_norm,
        "poisoned_update_norm": poison_norm,
        "excess_update_norm": excess_norm,
        "poison_clean_dot": dot_pc,
        "poison_clean_cosine": cosine,
        "poison_clean_angle_deg": math.degrees(math.acos(cosine)) if np.isfinite(cosine) else math.nan,
        "poison_orthogonal_fraction_to_clean": orthogonal_fraction(
            poison_norm=poison_norm, clean_norm=clean_norm, dot_pc=dot_pc
        ),
        "excess_to_poison_norm_ratio": excess_norm / poison_norm if poison_norm > EPS else math.nan,
        "excess_to_clean_norm_ratio": excess_norm / clean_norm if clean_norm > EPS else math.nan,
    }


def _load_all_states(run_dir: Path, manifests: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> tuple[
    dict[tuple[str, int], State], dict[tuple[str, int], Path]
]:
    states: dict[tuple[str, int], State] = {}
    paths: dict[tuple[str, int], Path] = {}
    shared = sorted(set(manifests["clean"]).intersection(manifests["poisoned"]))
    for condition in ("clean", "poisoned"):
        for frac_key in shared:
            row = manifests[condition][frac_key]
            path = resolve_manifest_checkpoint_dir(run_dir, row, must_exist=True)
            print(f"[geometry] loading {condition} fraction={float(row['fraction']):g}: {path}", flush=True)
            states[(condition, frac_key)] = load_lora_state(path)
            paths[(condition, frac_key)] = path
    return states, paths


def _model_interval_geometry(
    manifests: Mapping[str, Mapping[int, Mapping[str, Any]]],
    states: Mapping[tuple[str, int], State],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    shared = sorted(set(manifests["clean"]).intersection(manifests["poisoned"]))
    rows: list[dict[str, Any]] = []
    pca_vectors: list[dict[str, Any]] = []
    for start_key, end_key in zip(shared, shared[1:]):
        c0, c1 = states[("clean", start_key)], states[("clean", end_key)]
        p0, p1 = states[("poisoned", start_key)], states[("poisoned", end_key)]
        start_fraction = float(manifests["clean"][start_key]["fraction"])
        end_fraction = float(manifests["clean"][end_key]["fraction"])
        interval = _interval_label(start_fraction, end_fraction)
        clean = make_combo(((+1.0, c1), (-1.0, c0)))
        poison = make_combo(((+1.0, p1), (-1.0, p0)))
        excess = make_combo(((+1.0, p1), (-1.0, p0), (-1.0, c1), (+1.0, c0)))

        clean_norm = combo_norm(clean)
        poison_norm = combo_norm(poison)
        excess_norm = combo_norm(excess)
        dot_pc = combo_inner(poison, clean)
        cosine = safe_cosine(dot_pc, poison_norm, clean_norm)
        rows.append({
            "interval": interval,
            "start_fraction": start_fraction,
            "end_fraction": end_fraction,
            "clean_update_norm_fro": clean_norm,
            "poisoned_update_norm_fro": poison_norm,
            "excess_update_norm_fro": excess_norm,
            "poison_clean_dot_fro": dot_pc,
            "poison_clean_cosine_fro": cosine,
            "poison_clean_angle_deg": math.degrees(math.acos(cosine)) if np.isfinite(cosine) else math.nan,
            "poison_orthogonal_fraction_to_clean": orthogonal_fraction(
                poison_norm=poison_norm, clean_norm=clean_norm, dot_pc=dot_pc
            ),
            "excess_to_poison_norm_ratio": excess_norm / poison_norm if poison_norm > EPS else math.nan,
            "excess_to_clean_norm_ratio": excess_norm / clean_norm if clean_norm > EPS else math.nan,
        })
        pca_vectors.extend([
            {
                "label": f"clean {format_fraction_percent(start_fraction)}→{format_fraction_percent(end_fraction)}",
                "condition": "clean",
                "interval": interval,
                "start_fraction": start_fraction,
                "end_fraction": end_fraction,
                "combo": clean,
            },
            {
                "label": f"poisoned {format_fraction_percent(start_fraction)}→{format_fraction_percent(end_fraction)}",
                "condition": "poisoned",
                "interval": interval,
                "start_fraction": start_fraction,
                "end_fraction": end_fraction,
                "combo": poison,
            },
        ])
    return pd.DataFrame(rows), pca_vectors


def _kernel_pca(vectors: list[dict[str, Any]]) -> pd.DataFrame:
    n = len(vectors)
    if n == 0:
        return pd.DataFrame()
    gram = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        gram[i, i] = combo_inner(vectors[i]["combo"], vectors[i]["combo"])
        for j in range(i):
            value = combo_inner(vectors[i]["combo"], vectors[j]["combo"])
            gram[i, j] = gram[j, i] = value
    H = np.eye(n) - np.ones((n, n), dtype=np.float64) / float(n)
    centered = H @ gram @ H
    centered = (centered + centered.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(centered)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total = float(eigvals.sum())
    coords = np.zeros((n, 2), dtype=np.float64)
    explained = [math.nan, math.nan]
    for k in range(min(2, n)):
        coords[:, k] = eigvecs[:, k] * math.sqrt(float(eigvals[k]))
        explained[k] = float(eigvals[k] / total) if total > 0 else math.nan
    records = []
    for i, item in enumerate(vectors):
        records.append({
            "label": item["label"],
            "condition": item["condition"],
            "interval": item["interval"],
            "start_fraction": item["start_fraction"],
            "end_fraction": item["end_fraction"],
            "pc1": float(coords[i, 0]),
            "pc2": float(coords[i, 1]),
            "pc1_explained_variance_ratio": explained[0],
            "pc2_explained_variance_ratio": explained[1],
        })
    return pd.DataFrame(records)


def _parse_control_rows(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out: list[int] = []
    for part in str(value).split(";"):
        part = part.strip()
        if part:
            out.append(int(float(part)))
    return out


def _channel_geometry(
    stage07_dir: Path,
    manifests: Mapping[str, Mapping[int, Mapping[str, Any]]],
    states: Mapping[tuple[str, int], State],
    *,
    max_control_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = sorted(set(manifests["clean"]).intersection(manifests["poisoned"]))
    candidate_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []

    for start_key, end_key in zip(shared, shared[1:]):
        start_fraction = float(manifests["clean"][start_key]["fraction"])
        end_fraction = float(manifests["clean"][end_key]["fraction"])
        interval = _interval_label(start_fraction, end_fraction)
        mapped_path = stage07_dir / interval / "mapped_disruptive_channels.csv"
        if not mapped_path.is_file():
            print(f"[geometry] no mapped channels for {interval}: {mapped_path}", flush=True)
            continue
        mapped = _read_csv_or_empty(mapped_path)
        if mapped.empty:
            continue
        if "mapped" in mapped.columns:
            mapped = mapped[mapped["mapped"].map(truthy)].copy()
        if mapped.empty:
            continue

        c0, c1 = states[("clean", start_key)], states[("clean", end_key)]
        p0, p1 = states[("poisoned", start_key)], states[("poisoned", end_key)]

        # Multiple attention CHA channels can collapse to one GQA V-projection row.
        # Geometry is therefore computed once per unique parameter row, while
        # retaining the max disruption and all source unit IDs in the output.
        group_cols = ["lora_layer", "lora_module", "effective_weight_row"]
        for (layer_raw, module_name, row_raw), group in mapped.groupby(group_cols, dropna=False):
            layer = int(layer_raw)
            row = int(row_raw)
            key = (layer, str(module_name))
            if key not in c0 or key not in c1 or key not in p0 or key not in p1:
                continue
            clean_delta = row_delta(c0, c1, key, row)
            poison_delta = row_delta(p0, p1, key, row)
            metrics = vector_metrics(clean_delta, poison_delta)
            disruption_values = pd.to_numeric(group.get("disruption_score"), errors="coerce").dropna()
            signed_values = pd.to_numeric(group.get("poisoning_excess_delta_u_j"), errors="coerce").dropna()
            unit_keys = sorted({str(v) for v in group.get("unit_key", pd.Series(dtype=str)).dropna()})
            mapping_kinds = sorted({str(v) for v in group.get("mapping_kind", pd.Series(dtype=str)).dropna()})
            candidate_records.append({
                "interval": interval,
                "start_fraction": start_fraction,
                "end_fraction": end_fraction,
                "lora_layer": layer,
                "lora_module": str(module_name),
                "effective_weight_row": row,
                "parameter_row_key": f"L{layer}:{module_name}:row{row}",
                "source_unit_keys": ";".join(unit_keys),
                "n_source_channels": int(len(unit_keys)),
                "mapping_kind": ";".join(mapping_kinds),
                "max_abs_disruption_score": float(disruption_values.max()) if len(disruption_values) else math.nan,
                "mean_abs_disruption_score": float(disruption_values.mean()) if len(disruption_values) else math.nan,
                "max_abs_signed_disruption": float(np.max(np.abs(signed_values))) if len(signed_values) else math.nan,
                **metrics,
            })

            # Controls are already same-projection and excess-update-norm matched.
            # Their directional metrics are useful because direction was *not*
            # part of the matching criterion.
            control_rows: list[int] = []
            if "matched_control_effective_weight_rows" in group.columns:
                control_rows = _parse_control_rows(group.iloc[0]["matched_control_effective_weight_rows"])
            elif "matched_random_effective_weight_row" in group.columns:
                value = _finite_float(group.iloc[0]["matched_random_effective_weight_row"])
                control_rows = [int(value)] if np.isfinite(value) else []
            if max_control_draws > 0:
                control_rows = control_rows[: int(max_control_draws)]
            for draw, control_row in enumerate(control_rows):
                clean_control = row_delta(c0, c1, key, int(control_row))
                poison_control = row_delta(p0, p1, key, int(control_row))
                control_records.append({
                    "interval": interval,
                    "start_fraction": start_fraction,
                    "end_fraction": end_fraction,
                    "candidate_parameter_row_key": f"L{layer}:{module_name}:row{row}",
                    "control_draw": int(draw),
                    "lora_layer": layer,
                    "lora_module": str(module_name),
                    "effective_weight_row": int(control_row),
                    "parameter_row_key": f"L{layer}:{module_name}:row{int(control_row)}",
                    **vector_metrics(clean_control, poison_control),
                })

    candidate_columns = [
        "interval", "start_fraction", "end_fraction", "lora_layer", "lora_module",
        "effective_weight_row", "parameter_row_key", "source_unit_keys", "n_source_channels",
        "mapping_kind", "max_abs_disruption_score", "mean_abs_disruption_score",
        "max_abs_signed_disruption", "clean_update_norm", "poisoned_update_norm",
        "excess_update_norm", "poison_clean_dot", "poison_clean_cosine",
        "poison_clean_angle_deg", "poison_orthogonal_fraction_to_clean",
        "excess_to_poison_norm_ratio", "excess_to_clean_norm_ratio",
    ]
    control_columns = [
        "interval", "start_fraction", "end_fraction", "candidate_parameter_row_key",
        "control_draw", "lora_layer", "lora_module", "effective_weight_row",
        "parameter_row_key", "clean_update_norm", "poisoned_update_norm",
        "excess_update_norm", "poison_clean_dot", "poison_clean_cosine",
        "poison_clean_angle_deg", "poison_orthogonal_fraction_to_clean",
        "excess_to_poison_norm_ratio", "excess_to_clean_norm_ratio",
    ]
    candidates = pd.DataFrame(candidate_records) if candidate_records else pd.DataFrame(columns=candidate_columns)
    controls = pd.DataFrame(control_records) if control_records else pd.DataFrame(columns=control_columns)
    return candidates, controls


def _disruption_information(stage07_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    interval_dirs = sorted(p for p in stage07_dir.glob("interval_*") if p.is_dir())
    for interval_dir in interval_dirs:
        path = interval_dir / "control_correctness_channel_disruption.csv"
        if not path.is_file():
            continue
        frame = _read_csv_or_empty(path)
        if frame.empty or "disruption_score" not in frame.columns:
            continue
        if "complete_u_j_comparison" in frame.columns:
            frame = frame[frame["complete_u_j_comparison"].map(truthy)].copy()
        # A syntactically valid interval table can become empty after retaining
        # only complete U_j comparisons.  That means disruption information is
        # unavailable for the interval, not that there is a first row to read.
        if frame.empty:
            continue
        weights = pd.to_numeric(frame["disruption_score"], errors="coerce")
        weights = weights[np.isfinite(weights) & (weights >= 0)].to_numpy(dtype=float)
        positive = weights[weights > 0]
        total = float(positive.sum()) if len(positive) else 0.0
        n = int(len(positive))
        if total > 0 and n > 0:
            p = positive / total
            shannon_bits = float(-np.sum(p * np.log2(p)))
            renyi2_bits = float(-math.log2(float(np.sum(p**2))))
            neff2 = float(1.0 / np.sum(p**2))
            neff1 = float(2.0**shannon_bits)
            top1_share = float(np.max(p))
            normalized_shannon = shannon_bits / math.log2(n) if n > 1 else 0.0
        else:
            shannon_bits = renyi2_bits = neff2 = neff1 = top1_share = normalized_shannon = math.nan
        start_fraction = _finite_float(frame.get("start_fraction", pd.Series([math.nan])).iloc[0])
        end_fraction = _finite_float(frame.get("end_fraction", pd.Series([math.nan])).iloc[0])
        records.append({
            "interval": interval_dir.name,
            "start_fraction": start_fraction,
            "end_fraction": end_fraction,
            "n_positive_disruption_channels": n,
            "total_abs_disruption_mass": total,
            "max_abs_disruption": float(np.max(positive)) if n else 0.0,
            "top1_disruption_share": top1_share,
            "shannon_entropy_bits_of_abs_D": shannon_bits,
            "renyi2_entropy_bits_of_abs_D": renyi2_bits,
            "effective_support_shannon": neff1,
            "effective_support_renyi2": neff2,
            "normalized_shannon_entropy": normalized_shannon,
        })
    columns = [
        "interval", "start_fraction", "end_fraction", "n_positive_disruption_channels",
        "total_abs_disruption_mass", "max_abs_disruption", "top1_disruption_share",
        "shannon_entropy_bits_of_abs_D", "renyi2_entropy_bits_of_abs_D",
        "effective_support_shannon", "effective_support_renyi2",
        "normalized_shannon_entropy",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values("end_fraction").reset_index(drop=True)


def _plot_model_geometry(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    x = 100.0 * pd.to_numeric(frame["end_fraction"], errors="coerce").to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes[0, 0].plot(x, frame["poison_clean_cosine_fro"], marker="o")
    axes[0, 0].axhline(1.0, linewidth=1, linestyle="--")
    axes[0, 0].set_ylabel("cos(ΔWpoison, ΔWclean)")
    axes[0, 0].set_title("Directional alignment")

    axes[0, 1].plot(x, frame["poison_orthogonal_fraction_to_clean"], marker="o")
    axes[0, 1].set_ylabel("Orthogonal fraction")
    axes[0, 1].set_title("Poison update outside clean direction")

    axes[1, 0].plot(x, frame["clean_update_norm_fro"], marker="o", label="Clean")
    axes[1, 0].plot(x, frame["poisoned_update_norm_fro"], marker="o", label="Poisoned")
    axes[1, 0].plot(x, frame["excess_update_norm_fro"], marker="o", label="Poison − clean")
    axes[1, 0].set_ylabel("Effective LoRA Frobenius norm")
    axes[1, 0].set_title("Update magnitude")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(x, frame["excess_to_poison_norm_ratio"], marker="o")
    axes[1, 1].set_ylabel("||ΔWpoison−ΔWclean|| / ||ΔWpoison||")
    axes[1, 1].set_title("Relative poison-specific displacement")

    for ax in axes.flat:
        ax.set_xlabel("Interval end (% training)")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Clean vs poisoned effective-LoRA update geometry")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_pca(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for condition, group in frame.groupby("condition"):
        group = group.sort_values("end_fraction")
        marker = "o" if condition == "clean" else "s"
        ax.plot(group["pc1"], group["pc2"], marker=marker, label=condition)
        for row in group.itertuples(index=False):
            ax.annotate(f"{100*float(row.end_fraction):g}%", (row.pc1, row.pc2), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ev1 = _finite_float(frame["pc1_explained_variance_ratio"].iloc[0])
    ev2 = _finite_float(frame["pc2_explained_variance_ratio"].iloc[0])
    ax.set_xlabel(f"PC1 ({100*ev1:.1f}% variance)" if np.isfinite(ev1) else "PC1")
    ax.set_ylabel(f"PC2 ({100*ev2:.1f}% variance)" if np.isfinite(ev2) else "PC2")
    ax.set_title("Exact PCA of clean and poisoned LoRA interval updates")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_channel_geometry(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].scatter(
        frame["poison_orthogonal_fraction_to_clean"],
        frame["max_abs_disruption_score"],
        s=28,
        alpha=0.75,
    )
    axes[0].set_xlabel("Poison-update orthogonal fraction")
    axes[0].set_ylabel("|D_j| (max mapped channel)")
    axes[0].set_title("Causal reorganization vs update direction")

    axes[1].scatter(
        frame["excess_to_poison_norm_ratio"],
        frame["max_abs_disruption_score"],
        s=28,
        alpha=0.75,
    )
    axes[1].set_xlabel("||Δwpoison−Δwclean|| / ||Δwpoison||")
    axes[1].set_ylabel("|D_j| (max mapped channel)")
    axes[1].set_title("Causal reorganization vs excess displacement")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.suptitle("Overtopping-channel parameter-row geometry")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _paired_summary(candidate: pd.DataFrame, controls: pd.DataFrame, metric: str) -> pd.DataFrame:
    if candidate.empty or controls.empty or metric not in candidate or metric not in controls:
        return pd.DataFrame()
    ctrl = controls.groupby(["interval", "candidate_parameter_row_key"])[metric].agg(
        control_mean="mean", control_q025=lambda x: x.quantile(0.025), control_q975=lambda x: x.quantile(0.975)
    ).reset_index()
    cand = candidate[["interval", "parameter_row_key", "end_fraction", metric]].rename(
        columns={"parameter_row_key": "candidate_parameter_row_key", metric: "candidate_value"}
    )
    return cand.merge(ctrl, on=["interval", "candidate_parameter_row_key"], how="inner")


def _plot_candidate_vs_controls(candidate: pd.DataFrame, controls: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("poison_orthogonal_fraction_to_clean", "Orthogonal fraction"),
        ("poison_clean_cosine", "Clean/poison cosine"),
    ]
    summaries = [(metric, label, _paired_summary(candidate, controls, metric)) for metric, label in metrics]
    if all(frame.empty for _, _, frame in summaries):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (_, label, frame) in zip(axes, summaries):
        if frame.empty:
            ax.axis("off")
            continue
        intervals = sorted(frame["interval"].unique(), key=lambda name: float(frame.loc[frame["interval"] == name, "end_fraction"].iloc[0]))
        for xidx, interval in enumerate(intervals):
            part = frame[frame["interval"] == interval]
            # Per-candidate control mean against the candidate value.  Jitter is
            # deterministic and purely for visibility.
            if len(part):
                offsets = np.linspace(-0.12, 0.12, len(part)) if len(part) > 1 else np.array([0.0])
                ax.scatter(np.full(len(part), xidx - 0.15) + offsets * 0.15, part["candidate_value"], marker="o", alpha=0.8)
                ax.scatter(np.full(len(part), xidx + 0.15) + offsets * 0.15, part["control_mean"], marker="x", alpha=0.8)
                for _, row in part.iterrows():
                    ax.plot([xidx - 0.15, xidx + 0.15], [row["candidate_value"], row["control_mean"]], linewidth=0.6, alpha=0.25)
        labels = []
        for interval in intervals:
            end = float(frame.loc[frame["interval"] == interval, "end_fraction"].iloc[0])
            labels.append(f"{100*end:g}%")
        ax.set_xticks(range(len(intervals)), labels)
        ax.set_xlabel("Interval end")
        ax.set_ylabel(label)
        ax.set_title(f"Candidate vs norm-matched rows: {label}")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Does overtopping add directional geometry beyond update-norm matching?")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="One completed poisoning run directory.")
    parser.add_argument(
        "--phase",
        choices=["input_output", "output_only"],
        default=None,
        help="Defaults to the poisoning task's configured phase.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <run>/07_poisoning_example_detection/<phase>/derived_update_geometry",
    )
    parser.add_argument(
        "--tables_only",
        action="store_true",
        help="Write derived geometry tables/metadata without exporting standalone diagnostic PDFs.",
    )
    parser.add_argument(
        "--max_control_draws",
        type=int,
        default=100,
        help="Matched rows reconstructed per candidate; 0 keeps all persisted draws.",
    )
    args = parser.parse_args()
    if args.max_control_draws < 0:
        raise ValueError("--max_control_draws must be >= 0")

    run_dir = Path(args.run_dir).expanduser().resolve()
    definition = infer_task_from_run(run_dir)
    phase = args.phase or definition.default_phase
    stage07_dir = detection_dir(run_dir) / phase_dirname(phase)
    if not stage07_dir.is_dir():
        raise FileNotFoundError(f"Missing Stage-07 result directory: {stage07_dir}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else stage07_dir / "derived_update_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = _load_manifest_by_condition(run_dir)
    shared = sorted(set(manifests["clean"]).intersection(manifests["poisoned"]))
    if len(shared) < 2:
        raise RuntimeError("Need at least two matched clean/poisoned checkpoints")

    states, checkpoint_paths = _load_all_states(run_dir, manifests)

    interval_geometry, pca_vectors = _model_interval_geometry(manifests, states)
    interval_geometry.to_csv(output_dir / "update_geometry_by_interval.csv", index=False)

    pca = _kernel_pca(pca_vectors)
    pca.to_csv(output_dir / "update_geometry_pca.csv", index=False)

    candidates, controls = _channel_geometry(
        stage07_dir,
        manifests,
        states,
        max_control_draws=int(args.max_control_draws),
    )
    candidates.to_csv(output_dir / "channel_update_geometry.csv", index=False)
    controls.to_csv(output_dir / "matched_control_update_geometry.csv", index=False)

    info = _disruption_information(stage07_dir)
    info.to_csv(output_dir / "disruption_information_by_interval.csv", index=False)

    if not args.tables_only:
        # This module owns the PDFs in output_dir; tables are retained for reuse.
        for path in output_dir.glob("*.pdf"):
            path.unlink()
        _plot_model_geometry(interval_geometry, output_dir / "01_model_update_direction_and_magnitude.pdf")
        _plot_pca(pca, output_dir / "02_clean_vs_poisoned_update_map.pdf")
        _plot_channel_geometry(candidates, output_dir / "03_causal_channels_vs_local_update_geometry.pdf")
        _plot_candidate_vs_controls(candidates, controls, output_dir / "04_causal_channels_vs_matched_rows.pdf")

    summary = {
        "run_dir": str(run_dir),
        "task": definition.name,
        "phase": phase,
        "output_dir": str(output_dir),
        "n_intervals": int(len(interval_geometry)),
        "n_unique_overtopping_parameter_rows": int(len(candidates)),
        "n_matched_control_row_realizations": int(len(controls)),
        "model_geometry": "exact Frobenius geometry of effective LoRA interval updates scaling*(B@A), computed from low-rank factors",
        "channel_geometry": "mapped Stage-07 overtopping rows; attention uses the existing v_proj proxy and may collapse under GQA",
        "information_quantity": "entropy/effective support of normalized |D_j| causal-disruption mass; not continuous-activation mutual information",
        "checkpoint_paths": {f"{condition}:{key}": str(path) for (condition, key), path in checkpoint_paths.items()},
    }
    (output_dir / "derived_geometry_metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] geometry outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
