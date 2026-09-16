#!/usr/bin/env python3
"""Aggregate example-level singleton-union versus full-set composition decompositions.

The per-run decomposition is written by Stage 8 after both the singleton flip
masks and the genuine simultaneous candidate-set output are available.  This
reporter does not run model inference. It collects the configured population,
optionally restricts it to one replacement regime, verifies the decomposition identity,
writes a condition-level table,
and optionally renders a manuscript figure.

For each evaluated example, let S indicate that at least one singleton candidate
flips the behavioral endpoint and J indicate that the simultaneous intervention
on the complete candidate set flips it.  The four mutually exclusive classes are
preserved (S=1,J=1), suppressed (S=1,J=0), coalition-only (S=0,J=1), and
affected by neither intervention (S=0,J=0).  On the same population,

    E(J) - U(J) = P(coalition-only) - P(suppressed).

The decomposition describes interaction structure.  It does not assign a unique
mechanism such as saturation, masking, cancellation, or preemption.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.project_paths import PROJECT_ROOT
from studies.overtopping.analysis.stage07_overtopping_spiking_report import expected_overtopping_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "results/analysis/rq2_composition/interaction_decomposition"),
    )
    parser.add_argument("--paper-figures-dir", default=None)
    parser.add_argument("--primary-table", required=True)
    parser.add_argument(
        "--population-scope", choices=["primary", "primary+supplementary"], default="primary",
        help="primary uses the complete configured study table; the extended spelling requests the explicit extended scope.",
    )
    parser.add_argument("--replacement-regime", choices=["all", "mean-donor", "mean"], default="all", help="Analyze replacement regimes separately. mean-donor is never pooled with mean.")
    parser.add_argument("--evaluation-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--spiking-max-points", type=int, default=10000)
    parser.add_argument("--data-root", default=None, help="Alias for --root used by the shared exact run manifest.")
    return parser.parse_args()


def _load_population(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    root = Path(args.root).expanduser().resolve()
    args.data_root = str(Path(args.data_root or root).expanduser().resolve())
    specs = expected_overtopping_sources(args)
    if args.replacement_regime != "all":
        specs = [s for s in specs if s.get("replacement_regime") == args.replacement_regime]
    frames: list[pd.DataFrame] = []
    audit: list[dict] = []
    required = {
        "scope",
        "n_evaluated",
        "singleton_union_count",
        "joint_full_set_count",
        "preserved_count",
        "suppressed_count",
        "coalition_only_count",
        "Delta_comp_complete_case",
        "decomposition_gap",
        "identity_error",
    }
    for spec in specs:
        path = Path(spec["stats_dir"]) / "interaction_validation" / "composition_decomposition_summary.csv"
        exists = path.is_file()
        status = "missing"
        n_rows = 0
        if exists:
            try:
                frame = pd.read_csv(path)
            except Exception:
                frame = pd.DataFrame()
                status = "read_error"
            if not frame.empty:
                missing = sorted(required - set(frame.columns))
                if missing:
                    status = "incompatible_schema_missing:" + ",".join(missing)
                else:
                    frame = frame.copy()
                    frame["run_id"] = spec["run_id"]
                    frame["source_scope"] = spec.get("source_scope", "primary")
                    frame["task"] = spec.get("task")
                    frame["model"] = spec.get("model")
                    frame["phase"] = spec.get("phase")
                    frame["replacement_regime"] = spec.get("replacement_regime")
                    frame["source_path"] = str(path)
                    frames.append(frame)
                    n_rows = int(len(frame))
                    status = "ok"
        audit.append({
            "run_id": spec["run_id"],
            "source_scope": spec.get("source_scope", "primary"),
            "task": spec.get("task"),
            "model": spec.get("model"),
            "phase": spec.get("phase"),
            "replacement_regime": spec.get("replacement_regime"),
            "source": str(path),
            "exists": bool(exists),
            "n_rows": n_rows,
            "status": status,
        })
    pd.DataFrame(audit).to_csv(out_dir / "composition_decomposition_population_audit.csv", index=False)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _condition_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    overall = frame.loc[frame["scope"].astype(str).eq("overall")].copy()
    if overall.empty:
        return overall
    numeric = [
        "n_evaluated",
        "singleton_union_count",
        "joint_full_set_count",
        "preserved_count",
        "suppressed_count",
        "coalition_only_count",
        "unaffected_count",
        "multi_singleton_reachable_count",
        "U_J_complete_case",
        "E_J_complete_case",
        "Delta_comp_complete_case",
        "preserved_rate_all",
        "suppressed_rate_all",
        "coalition_only_rate_all",
        "unaffected_rate_all",
        "preservation_rate_given_singleton_reachable",
        "suppression_rate_given_singleton_reachable",
        "coalition_only_rate_given_singleton_unreachable",
        "multi_singleton_reachable_rate",
        "decomposition_gap",
        "identity_error",
    ]
    for col in numeric:
        if col in overall.columns:
            overall[col] = pd.to_numeric(overall[col], errors="coerce")
    return overall.reset_index(drop=True)


def _plot(condition: pd.DataFrame, path: Path) -> None:
    if condition.empty:
        path.unlink(missing_ok=True)
        return
    work = condition.copy()
    for col in ["suppressed_rate_all", "coalition_only_rate_all", "Delta_comp_complete_case"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work = work.loc[np.isfinite(work["suppressed_rate_all"]) & np.isfinite(work["coalition_only_rate_all"])].copy()
    if work.empty:
        path.unlink(missing_ok=True)
        return
    work = work.sort_values("Delta_comp_complete_case", kind="mergesort").reset_index(drop=True)
    x = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.scatter(x, work["suppressed_rate_all"].to_numpy(float), marker="o", label="singleton-reachable suppressed by full set")
    ax.scatter(x, work["coalition_only_rate_all"].to_numpy(float), marker="x", label="coalition-only full-set effects")
    ax.set_xlabel("Study setting, sorted by E(J) - U(J)")
    ax.set_ylabel("Fraction of complete-case held-out examples")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_population(args, out_dir)
    frame.to_csv(out_dir / "composition_decomposition_all_scopes.csv", index=False)
    condition = _condition_table(frame)
    condition.to_csv(out_dir / "composition_decomposition_by_condition.csv", index=False)

    audit_path = out_dir / "composition_decomposition_population_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.is_file() else pd.DataFrame()
    n_expected = int(len(audit))
    n_available = int((audit.get("status", pd.Series(dtype=str)).astype(str) == "ok").sum()) if not audit.empty else 0

    if condition.empty:
        payload = {
            "status": "not_available" if n_available == 0 else "no_overall_rows",
            "population_scope": args.population_scope,
            "replacement_regime": args.replacement_regime,
            "n_expected_conditions": n_expected,
            "n_available_runs": n_available,
            "inference_unit": "run/setting; decomposition is descriptive unless a prespecified inferential contrast is added",
        }
    else:
        identity = pd.to_numeric(condition.get("identity_error"), errors="coerce").dropna().abs()
        max_identity = float(identity.max()) if len(identity) else math.nan
        if np.isfinite(max_identity) and max_identity > 1e-12:
            raise RuntimeError(f"Composition identity failed in aggregate report: max error={max_identity}")
        def med(col: str) -> float:
            vals = pd.to_numeric(condition.get(col), errors="coerce").dropna().to_numpy(float)
            return float(np.median(vals)) if len(vals) else math.nan
        payload = {
            "status": "ok" if n_available == n_expected else "partial_coverage",
            "population_scope": args.population_scope,
            "replacement_regime": args.replacement_regime,
            "n_expected_conditions": n_expected,
            "n_available_runs": n_available,
            "n_condition_rows": int(len(condition)),
            "median_preservation_rate_given_singleton_reachable": med("preservation_rate_given_singleton_reachable"),
            "median_suppression_rate_given_singleton_reachable": med("suppression_rate_given_singleton_reachable"),
            "median_suppressed_rate_all": med("suppressed_rate_all"),
            "median_coalition_only_rate_all": med("coalition_only_rate_all"),
            "median_composition_gap": med("Delta_comp_complete_case"),
            "max_abs_identity_error": max_identity,
            "identity": "E(J)-U(J) = P(coalition_only) - P(suppressed) on the same complete-case population",
            "interpretation": (
                "High preservation with low suppression is compatible with a monotone saturating interaction regime. "
                "Substantial suppression indicates that singleton-reachable effects are lost under the simultaneous set and requires an interaction explanation beyond simple monotone saturation. "
                "Coalition-only effects identify joint effects not present in singleton reach."
            ),
            "inference_unit": "run/setting; per-example categories are not independent manuscript replicates",
        }
    (out_dir / "composition_decomposition_report_status.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )

    plot_path = out_dir / "composition_decomposition.pdf"
    _plot(condition, plot_path)
    if args.paper_figures_dir and plot_path.is_file():
        paper = Path(args.paper_figures_dir).expanduser().resolve()
        _plot(condition, paper / "fig3d_singleton_joint_decomposition.pdf")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
