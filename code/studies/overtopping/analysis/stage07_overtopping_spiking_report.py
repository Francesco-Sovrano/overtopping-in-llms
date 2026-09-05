#!/usr/bin/env python3
"""Generate aggregate RQ3 causal threshold-event statistics and figures.

The input can be a data root containing experiment-level threshold diagnostics
or an archive containing the same aggregate CSV contract. The analysis resolves
the exact primary/supplementary overtopping manifest and excludes poisoning.

Example:
  python3 -m studies.overtopping.analysis.stage07_overtopping_spiking_report \
    --root data \
    --out results/analysis/rq3_threshold_event/spiking_diagnostics \
    --paper-figures-dir results/paper/figures/04_rq3_spiking_cut
"""
from __future__ import annotations
from pathlib import Path
from core.project_paths import PROJECT_ROOT
from studies.overtopping.analysis import primary_holdout_analysis as primary_helpers
from studies.overtopping.experiments.run_experiments import paper_auxiliary_experiments


import argparse, io, json, math, textwrap, zipfile
from dataclasses import replace
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

POP_CAND = "flip_rule_candidate"
POP_CTRL = "random_nonagonist_control"
POP_LABELS = {POP_CAND: "Candidates", POP_CTRL: "Non-candidate controls"}
# Machine-readable provenance written by threshold_event_diagnostics.py.
# Keep this separate from the typographic labels used in figures/tables.
DIRECTION_KEY_BY_BASELINE = {"positive": "1to0", "negative": "0to1"}
DIRECTION_BY_BASELINE = {"positive": "1→0", "negative": "0→1"}
BASELINE_ORDER = ("positive", "negative")


def _normalize_rq3_direction(value: object) -> str:
    """Normalize persisted RQ3 direction provenance to its canonical machine key.

    Current experiment-level diagnostics persist ``1to0``/``0to1``.  Older or
    hand-produced summaries may contain typographic/ASCII arrow spellings; they
    are semantically identical and are accepted here without rewriting source
    artifacts.
    """
    token = str(value).strip().lower().replace(" ", "")
    aliases = {
        "1to0": "1to0",
        "1->0": "1to0",
        "1→0": "1to0",
        "0to1": "0to1",
        "0->1": "0to1",
        "0→1": "0to1",
    }
    return aliases.get(token, token)
RNG = np.random.default_rng(20260502)


def parse_args():
    p = argparse.ArgumentParser(description="Generate overtopping-as-spiking report from results zip/root")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="Input spiking diagnostics results zip")
    src.add_argument("--root", help="Already-extracted result root containing data/")
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "analysis" / "rq3_threshold_event" / "spiking_diagnostics"), help="Detailed analysis output directory")
    p.add_argument("--paper-figures-dir", default=None, help="Optional manuscript RQ3 figure directory. When provided, compact threshold/spiking-cut figures are promoted here instead of being buried in diagnostics.")
    p.add_argument("--base-md", default=None, help="Optional Markdown file to update")
    p.add_argument("--primary-table", required=True, help="Primary manuscript table defining the exact RQ3 population.")
    p.add_argument("--data-root", required=True, help="Canonical overtopping data root used to resolve primary-table rows.")
    p.add_argument("--evaluation-split", default="test", choices=["test", "train", "all"])
    p.add_argument("--spiking-max-points", type=int, default=10000)
    p.add_argument(
        "--population-scope",
        choices=["primary", "primary+supplementary"],
        default="primary+supplementary",
        help=("RQ3 population. The extended scope adds the configured paper-auxiliary "
              "overtopping experiments and never scans poisoning experiments."),
    )
    p.add_argument("--bootstrap", type=int, default=3000, help="Bootstrap samples for median-delta CI")
    return p.parse_args()

def normalize_member_name(name: str) -> str:
    return name.lstrip("./")





def _spiking_label(setting: dict, evaluation_split: str, spiking_max_points: int) -> str:
    label = f"spiking_diagnostics-{setting['bag_label']}"
    if evaluation_split == "train":
        label += "-eval_train"
    elif evaluation_split == "all":
        label += "-eval_all"
    if int(spiking_max_points) != 10000:
        label += f"-cap{int(spiking_max_points)}"
    return label


def _expected_primary_sources(primary_table: Path, data_root: Path, *, evaluation_split: str,
                              spiking_max_points: int) -> list[dict]:
    table = pd.read_csv(primary_table)
    expected = []
    for index in range(len(table)):
        row = table.iloc[index]
        setting = primary_helpers.setting_from_row(
            index, row, data_root, PROJECT_ROOT / "results",
            evaluation_split=evaluation_split, sampling_max_points=spiking_max_points,
        )
        diag_dir = Path(setting["input_data_dir"]) / _spiking_label(setting, evaluation_split, spiking_max_points)
        expected.append({
            "row_index": int(index),
            "run_id": f"primary_row_{index:02d}",
            "task": str(row.get("task", setting.get("task", "unknown"))),
            "model": str(row.get("model", setting.get("model", "unknown"))),
            "phase": str(row.get("phase", setting.get("phase", "unknown"))),
            "setting": str(setting.get("circuit_label", "unknown")),
            "decode_only": bool(setting.get("decode_only", False)),
            "diag_dir": diag_dir,
            "stats_dir": Path(setting["heldout_stats"]),
            "source_scope": "primary",
            "required": True,
        })
    return expected



def _expected_supplementary_sources(data_root: Path, *, evaluation_split: str,
                                    spiking_max_points: int) -> list[dict]:
    """Return the configured paper-auxiliary RQ3 sources, exactly and without scans.

    These are the manuscript supplementary overtopping experiments.  Poisoning
    experiments live under a different experiment family and are impossible to
    enter this manifest because the sources come only from paper_auxiliary_experiments().
    """
    expected: list[dict] = []
    for index, raw_spec in enumerate(paper_auxiliary_experiments()):
        spec = replace(raw_spec, evaluation_split=str(evaluation_split))
        label = f"spiking_diagnostics-{spec.bag_label()}"
        if evaluation_split == "train":
            label += "-eval_train"
        elif evaluation_split == "all":
            label += "-eval_all"
        if int(spiking_max_points) != 10000:
            label += f"-cap{int(spiking_max_points)}"
        diag_dir = spec.input_data_dir(data_root) / label
        expected.append({
            "row_index": int(index),
            "run_id": f"supplementary_row_{index:02d}",
            "task": str(spec.task),
            "model": str(Path(spec.model).name),
            "phase": str(spec.phase),
            "setting": str(spec.circuit_label()),
            "decode_only": bool(spec.decode_only),
            "diag_dir": diag_dir,
            "stats_dir": spec.stats_dir(data_root),
            "source_scope": "supplementary",
            "required": False,
        })
    return expected


def expected_rq3_sources(args) -> list[dict]:
    """Build the exact primary(+supplementary) manifest used by every RQ3 stage."""
    data_root = Path(args.data_root).expanduser().resolve()
    expected = _expected_primary_sources(
        Path(args.primary_table).expanduser().resolve(), data_root,
        evaluation_split=str(args.evaluation_split),
        spiking_max_points=int(args.spiking_max_points),
    )
    if str(getattr(args, "population_scope", "primary+supplementary")) == "primary+supplementary":
        expected.extend(_expected_supplementary_sources(
            data_root, evaluation_split=str(args.evaluation_split),
            spiking_max_points=int(args.spiking_max_points),
        ))
    # Primary wins if a future catalogue accidentally duplicates an auxiliary spec.
    deduped: dict[str, dict] = {}
    for spec in expected:
        key = str(Path(spec["stats_dir"]).resolve())
        if key not in deduped or bool(spec.get("required", False)):
            deduped[key] = spec
    out = list(deduped.values())
    for spec in out:
        try:
            rel = Path(spec["diag_dir"]).resolve().relative_to(data_root)
        except ValueError as exc:
            raise RuntimeError(f"RQ3 source escaped data root: {spec['diag_dir']}") from exc
        if rel.parts and rel.parts[0] == "poisoning":
            raise RuntimeError(f"Poisoning source leaked into RQ3 manifest: {spec['diag_dir']}")
    return out

def _zip_member_for_expected(zf: zipfile.ZipFile, expected_rel: str) -> str | None:
    target = normalize_member_name(expected_rel)
    matches = []
    for name in zf.namelist():
        rel = normalize_member_name(name)
        if rel == target or rel.endswith("/" + target):
            matches.append(name)
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous archive members for exact primary source {expected_rel}: {matches[:5]}")
    return matches[0] if matches else None


def _parse_discovery_baseline_values(*values) -> set[str]:
    """Normalize frozen Stage-6 discovery-baseline provenance."""
    out: set[str] = set()
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        for token in str(value).replace(",", "|").replace(";", "|").split("|"):
            token = token.strip().lower()
            if token in DIRECTION_BY_BASELINE:
                out.add(token)
    return out


def _expected_candidate_baselines(
    source_kind: str, source_path: Path, spec: dict, data_root: Path, zf: zipfile.ZipFile | None
) -> tuple[set[str], str, str]:
    """Return discovery-eligible RQ3 baselines for one configured run.

    RQ3 candidate identity is directional: a Stage-7 candidate is eligible only
    in a baseline subset in which it was discovered by Stage 6.  Therefore a
    run with no negative-baseline candidates is complete with a positive-only
    diagnostic; requiring both baselines would invent an unevaluated candidate
    population that does not exist.

    Frozen candidate ranking is the authority.  If it is unavailable, retain
    the historical conservative contract (require both baselines) rather than
    silently weakening completeness checks.
    """
    stats_dir = Path(spec["stats_dir"])
    ranking_name = "frozen_candidate_ranking.csv"
    ranking = pd.DataFrame()
    ranking_source = str(stats_dir / ranking_name)

    if source_kind == "root":
        ranking_path = stats_dir / ranking_name
        if ranking_path.is_file():
            try:
                ranking = pd.read_csv(ranking_path)
            except pd.errors.EmptyDataError:
                ranking = pd.DataFrame()
    else:
        try:
            rel = (Path("data") / stats_dir.relative_to(data_root) / ranking_name).as_posix()
            member = _zip_member_for_expected(zf, rel) if zf is not None else None
        except ValueError:
            member = None
            rel = ranking_source
        if member is not None and zf is not None:
            ranking_source = member
            with zf.open(member) as fh:
                try:
                    ranking = pd.read_csv(io.BytesIO(fh.read()))
                except pd.errors.EmptyDataError:
                    ranking = pd.DataFrame()
        else:
            # A diagnostics-only archive may omit Stage-7 stats while the
            # explicitly supplied canonical data root still contains them.
            ranking_path = stats_dir / ranking_name
            if ranking_path.is_file():
                ranking_source = str(ranking_path)
                try:
                    ranking = pd.read_csv(ranking_path)
                except pd.errors.EmptyDataError:
                    ranking = pd.DataFrame()

    baselines: set[str] = set()
    if not ranking.empty:
        subset_values = ranking.get("discovery_baseline_subsets", pd.Series(index=ranking.index, dtype=object))
        primary_values = ranking.get("discovery_baseline_subset", pd.Series(index=ranking.index, dtype=object))
        for subsets, primary in zip(subset_values, primary_values):
            baselines.update(_parse_discovery_baseline_values(subsets, primary))

    if baselines:
        return baselines, "frozen_candidate_ranking", ranking_source
    return set(BASELINE_ORDER), "fallback_require_both", ranking_source


def _read_exact_primary_table(source_kind: str, source_path: Path, expected: list[dict],
                              data_root: Path, suffix: str) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    audit = []
    zf = zipfile.ZipFile(source_path) if source_kind == "zip" else None
    try:
        for spec in expected:
            diag_dir = Path(spec["diag_dir"])
            if source_kind == "root":
                file_path = diag_dir / suffix
                exists = file_path.is_file()
                source_label = str(file_path)
                if exists:
                    try:
                        df = pd.read_csv(file_path)
                    except pd.errors.EmptyDataError:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()
            else:
                rel = (Path("data") / diag_dir.relative_to(data_root) / suffix).as_posix()
                member = _zip_member_for_expected(zf, rel)
                exists = member is not None
                source_label = member or rel
                if exists:
                    with zf.open(member) as fh:
                        try:
                            df = pd.read_csv(io.BytesIO(fh.read()))
                        except pd.errors.EmptyDataError:
                            df = pd.DataFrame()
                else:
                    df = pd.DataFrame()

            expected_baselines, baseline_contract, ranking_source = _expected_candidate_baselines(
                source_kind, source_path, spec, data_root, zf
            )
            baselines = (
                set(df.get("baseline_subset", pd.Series(dtype=str)).dropna().astype(str).str.strip().str.lower())
                if not df.empty else set()
            )
            populations = (
                set(df.get("population", pd.Series(dtype=str)).dropna().astype(str))
                if not df.empty else set()
            )
            missing_baselines = sorted(expected_baselines - baselines)
            unexpected_baselines = sorted(baselines - expected_baselines)

            # Completeness is baseline-specific.  Seeing a candidate somewhere
            # in the file and a control somewhere else is not sufficient; each
            # discovery-eligible directional stratum must contain both.
            missing_population_cells: list[str] = []
            if not df.empty and {"baseline_subset", "population"}.issubset(df.columns):
                baseline_norm = df["baseline_subset"].astype(str).str.strip().str.lower()
                population_norm = df["population"].astype(str)
                for baseline in sorted(expected_baselines):
                    observed = set(population_norm.loc[baseline_norm.eq(baseline)])
                    for population in (POP_CAND, POP_CTRL):
                        if population not in observed:
                            missing_population_cells.append(f"{baseline}:{population}")
            else:
                missing_population_cells = [
                    f"{baseline}:{population}"
                    for baseline in sorted(expected_baselines)
                    for population in (POP_CAND, POP_CTRL)
                ]
            missing_populations = sorted({cell.split(":", 1)[1] for cell in missing_population_cells})
            ok = bool(
                exists and not df.empty
                and not missing_baselines
                and not unexpected_baselines
                and not missing_population_cells
            )
            audit.append({
                "row_index": spec["row_index"], "run_id": spec["run_id"], "task": spec["task"],
                "model": spec["model"], "phase": spec["phase"], "source_scope": spec.get("source_scope", "primary"),
                "required": bool(spec.get("required", True)), "file": suffix,
                "source": source_label, "exists": bool(exists), "n_rows": int(len(df)),
                "expected_baseline_subsets": ",".join(sorted(expected_baselines)),
                "baseline_contract": baseline_contract, "candidate_ranking_source": ranking_source,
                "baseline_subsets": ",".join(sorted(baselines)), "populations": ",".join(sorted(populations)),
                "missing_baseline_subsets": ",".join(missing_baselines),
                "unexpected_baseline_subsets": ",".join(unexpected_baselines),
                "missing_population_cells": ",".join(missing_population_cells),
                "missing_populations": ",".join(missing_populations), "complete": ok,
            })
            if exists and not df.empty:
                df = df.copy()
                for key in ("run_id", "task", "model", "setting", "decode_only"):
                    df[key] = spec[key]
                df["source_scope"] = spec.get("source_scope", "primary")
                df["rel"] = source_label
                frames.append(df)
    finally:
        if zf is not None:
            zf.close()
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), audit


def load_exact_rq3_population(args, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_kind = "zip" if args.zip else "root"
    source_path = Path(args.zip or args.root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    expected = expected_rq3_sources(args)
    fs, audit_fs = _read_exact_primary_table(source_kind, source_path, expected, data_root, "aggregate_flip_stats.csv")
    ut, audit_ut = _read_exact_primary_table(source_kind, source_path, expected, data_root, "aggregate_unit_tests.csv")
    b, audit_b = _read_exact_primary_table(source_kind, source_path, expected, data_root, "aggregate_binned_curves.csv")
    audit = pd.DataFrame(audit_fs + audit_ut + audit_b)
    # The model-backed flip table is the population-completeness authority: it
    # proves that every discovery-eligible directional subset contains both the
    # candidate and corresponding structural-control populations. A baseline
    # with zero frozen discovery candidates is not an RQ3 candidate stratum and
    # is therefore not required. Threshold-unit tests are conditional on there
    # being enough flip and non-flip examples for repeated held-out fitting. A
    # population can therefore be present in aggregate_flip_stats.csv but
    # legitimately absent from aggregate_unit_tests.csv (for example, inert
    # random controls with no/too-few flips). Treat that as threshold-testability
    # sparsity, not as a missing primary population.
    out.mkdir(parents=True, exist_ok=True)
    audit["required_for_population"] = audit["file"].eq("aggregate_flip_stats.csv")
    # Only the model-backed flip table is a required per-run source.  A run can
    # legitimately have *no* threshold-unit-test table at all when every
    # candidate/control unit fails the minimum flip/non-flip support required by
    # repeated_holdout().  threshold_event_diagnostics historically wrote an
    # empty, zero-column threshold_unit_tests.csv in that case; write_aggregate
    # then skipped it, leaving no aggregate_unit_tests.csv.  The complete flip
    # population still proves that the interventions were evaluated, so do not
    # misclassify this representation detail as an unevaluated primary row.
    audit["required_source_exists"] = audit["file"].eq("aggregate_flip_stats.csv")
    audit.to_csv(out / "population_audit.csv", index=False)
    (out / "population_audit.json").write_text(json.dumps(audit.to_dict(orient="records"), indent=2), encoding="utf-8")

    flip_audit = audit[audit["file"].eq("aggregate_flip_stats.csv")].copy()
    incomplete_primary = flip_audit[flip_audit["required"].astype(bool) & ~flip_audit["complete"].astype(bool)]
    if not incomplete_primary.empty:
        preview_cols = [c for c in [
            "row_index", "task", "model", "phase", "file", "expected_baseline_subsets",
            "missing_baseline_subsets", "unexpected_baseline_subsets",
            "missing_population_cells", "exists",
        ] if c in incomplete_primary.columns]
        preview = incomplete_primary[preview_cols].to_dict(orient="records")[:8]
        raise RuntimeError(
            "RQ3 primary evaluation population is incomplete; refusing to report unevaluated primary rows. "
            "Supplementary rows are audited separately and may be unavailable, but every primary "
            "model-backed flip-stat population must cover every discovery-eligible baseline "
            "subset and contain both candidate/control populations within each eligible subset. "
            f"See {out / 'population_audit.csv'}. First failures: {preview}"
        )
    included_runs = set(flip_audit.loc[flip_audit["complete"].astype(bool), "run_id"].astype(str))
    audit["included_in_analysis"] = audit["run_id"].astype(str).isin(included_runs)
    audit.to_csv(out / "population_audit.csv", index=False)
    (out / "population_audit.json").write_text(json.dumps(audit.to_dict(orient="records"), indent=2), encoding="utf-8")
    for frame in (fs, ut, b):
        if not frame.empty and "run_id" in frame.columns:
            frame.drop(frame.index[~frame["run_id"].astype(str).isin(included_runs)], inplace=True)
    cand_fs = fs.loc[fs.population.astype(str) == POP_CAND].copy() if not fs.empty and "population" in fs.columns else pd.DataFrame()
    required_direction_cols = {"baseline_subset", "rq3_direction", "candidate_direction_policy"}
    if not cand_fs.empty and not required_direction_cols.issubset(cand_fs.columns):
        missing = sorted(required_direction_cols - set(cand_fs.columns))
        raise RuntimeError(
            "Stale direction-agnostic RQ3 diagnostics detected before reporting: "
            f"aggregate_flip_stats.csv is missing {missing}. Rebuild per-experiment "
            "spiking_diagnostics-* outputs with discovery-direction-aware diagnostics."
        )
    if not cand_fs.empty:
        baseline_norm = cand_fs["baseline_subset"].astype(str).str.strip().str.lower()
        expected = baseline_norm.map(DIRECTION_KEY_BY_BASELINE)
        observed = cand_fs["rq3_direction"].map(_normalize_rq3_direction)
        policy = cand_fs["candidate_direction_policy"].astype(str).str.strip().str.lower()
        wrong = expected.isna() | observed.ne(expected) | policy.ne("discovery_baseline_only")
        if bool(wrong.any()):
            preview_cols = [c for c in [
                "run_id", "task", "model", "baseline_subset", "unit_key",
                "rq3_direction", "candidate_direction_policy",
            ] if c in cand_fs.columns]
            preview = cand_fs.loc[wrong, preview_cols].head(8).to_dict("records")
            raise RuntimeError(
                "RQ3 candidate direction provenance conflicts with baseline_subset; "
                "refusing mixed-direction reporting. Expected machine provenance "
                "positive->1to0 and negative->0to1. "
                f"First mismatches: {preview}"
            )
    coverage = {
        "population_scope": str(getattr(args, "population_scope", "primary+supplementary")),
        "configured_primary": int((flip_audit["source_scope"] == "primary").sum()),
        "configured_supplementary": int((flip_audit["source_scope"] == "supplementary").sum()),
        "included_primary": int(((flip_audit["source_scope"] == "primary") & flip_audit["complete"].astype(bool)).sum()),
        "included_supplementary": int(((flip_audit["source_scope"] == "supplementary") & flip_audit["complete"].astype(bool)).sum()),
        "excluded_poisoning": True,
    }
    (out / "population_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return fs, ut, b



def rankdata_abs(vals: np.ndarray) -> np.ndarray:
    order = np.argsort(vals)
    ranks = np.empty(len(vals), dtype=float)
    i=0
    while i < len(vals):
        j=i+1
        while j < len(vals) and vals[order[j]] == vals[order[i]]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i=j
    return ranks


def wilcoxon_greater(d: np.ndarray) -> Tuple[float, float, float]:
    d = np.asarray(d, dtype=float); d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0: return float("nan"), float("nan"), float("nan")
    ranks = rankdata_abs(np.abs(d))
    wpos = float(ranks[d > 0].sum()); wneg = float(ranks[d < 0].sum())
    rb = (wpos - wneg) / (wpos + wneg) if (wpos + wneg) else float("nan")
    n = len(ranks)
    if n <= 22:
        sums = np.array([0.0])
        for r in ranks: sums = np.concatenate([sums, sums + r])
        p = float((sums >= wpos - 1e-12).mean())
    else:
        mean = ranks.sum()/2.0; var = (ranks**2).sum()/4.0
        z = (wpos - mean - 0.5) / math.sqrt(var) if var > 0 else 0.0
        p = 0.5 * math.erfc(z / math.sqrt(2.0))
    sd = np.std(d, ddof=1)
    dz = float(np.mean(d) / sd) if len(d) > 1 and sd > 0 else float("nan")
    return p, rb, dz


def binom_p_greater(k:int, n:int, p0:float=0.5) -> float:
    if n <= 0: return float("nan")
    return float(sum(math.comb(n,i)*(p0**i)*((1-p0)**(n-i)) for i in range(k,n+1)))


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    x=np.asarray(list(x), dtype=float); y=np.asarray(list(y), dtype=float)
    x=x[np.isfinite(x)]; y=y[np.isfinite(y)]
    if len(x)==0 or len(y)==0: return float("nan")
    gt=sum(float((xi > y).sum()) for xi in x)
    lt=sum(float((xi < y).sum()) for xi in x)
    return float((gt-lt)/(len(x)*len(y)))


def paired_medians(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns: return pd.DataFrame()
    g = df[df.population.isin([POP_CAND,POP_CTRL])].dropna(subset=[metric]).groupby(["run_id","baseline_subset","population"], dropna=False)[metric].median().unstack("population")
    if POP_CAND not in g.columns or POP_CTRL not in g.columns: return pd.DataFrame()
    g = g.dropna(subset=[POP_CAND, POP_CTRL]).copy()
    g["delta"] = g[POP_CAND] - g[POP_CTRL]
    return g


def build_best_threshold_population(fs: pd.DataFrame, ut: pd.DataFrame) -> pd.DataFrame:
    """Return one row per evaluated candidate/control unit with best threshold test.

    Threshold tests are only emitted when repeated held-out fitting has enough
    positive and negative events.  We therefore left-join the real unit tests
    onto the complete model-backed flip population instead of silently dropping
    evaluated units.  Best-MCC remains NaN for untestable units.  TECS is known
    to be exactly zero for zero-strength units even when thresholdability itself
    is undefined; nonzero-strength untestable units keep TECS as NaN.
    """
    if fs is None or fs.empty:
        return pd.DataFrame()
    base = fs.loc[fs["population"].isin([POP_CAND, POP_CTRL])].copy()
    keys = [c for c in ["run_id", "task", "model", "setting", "decode_only", "baseline_subset", "population", "unit_key"] if c in base.columns]
    if not {"run_id", "baseline_subset", "population", "unit_key"}.issubset(keys):
        return pd.DataFrame()

    if "population_strength" not in base.columns:
        strength_cols = [c for c in ["flip_any_rate", "c2i_rate", "i2c_rate"] if c in base.columns]
        if strength_cols:
            base["population_strength"] = base[strength_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
        else:
            base["population_strength"] = np.nan
    base["population_strength"] = pd.to_numeric(base["population_strength"], errors="coerce")
    base = base.sort_values(keys).drop_duplicates(keys, keep="first")

    real = pd.DataFrame()
    if ut is not None and not ut.empty and "median_test_abs_mcc" in ut.columns:
        cand = ut.loc[ut.get("population", pd.Series(index=ut.index, dtype=str)).isin([POP_CAND, POP_CTRL])].copy()
        if not cand.empty:
            cand["median_test_abs_mcc"] = pd.to_numeric(cand["median_test_abs_mcc"], errors="coerce")
            cand = cand.dropna(subset=["median_test_abs_mcc"])
        if not cand.empty:
            group_keys = [c for c in keys if c in cand.columns]
            idx = cand.groupby(group_keys, dropna=False)["median_test_abs_mcc"].idxmax()
            real = cand.loc[idx].copy().rename(columns={
                "median_test_abs_mcc": "best_mcc",
                "median_test_auc_oriented": "best_auc",
                "feature": "best_feature",
            })
            keep = group_keys + [c for c in ["best_mcc", "best_auc", "best_feature", "target", "direction_family"] if c in real.columns]
            real = real.loc[:, list(dict.fromkeys(keep))]

    if real.empty:
        best = base.copy()
        best["best_mcc"] = np.nan
        best["best_auc"] = np.nan
        best["best_feature"] = pd.NA
    else:
        merge_keys = [c for c in keys if c in real.columns]
        best = base.merge(real, on=merge_keys, how="left", validate="one_to_one")

    best["best_mcc"] = pd.to_numeric(best.get("best_mcc"), errors="coerce")
    best["threshold_testable"] = best["best_mcc"].notna()
    strength = pd.to_numeric(best["population_strength"], errors="coerce")
    best["causal_spiking_score"] = strength * best["best_mcc"]
    zero_strength = strength.fillna(np.nan).eq(0.0)
    best.loc[zero_strength & ~best["threshold_testable"], "causal_spiking_score"] = 0.0
    best["tecs_observed"] = best["causal_spiking_score"].notna()
    best["feature_family"] = best.get("best_feature", pd.Series(index=best.index, dtype=object)).map(
        lambda v: family(v) if pd.notna(v) else "untestable"
    )
    return best


def threshold_testability_audit(fs: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    if fs is None or fs.empty:
        return pd.DataFrame()
    base = fs.loc[fs["population"].isin([POP_CAND, POP_CTRL])].copy()
    group = [c for c in ["run_id", "task", "model", "setting", "decode_only", "baseline_subset", "population"] if c in base.columns]
    if not group or "unit_key" not in base.columns:
        return pd.DataFrame()
    evaluated = base.groupby(group, dropna=False)["unit_key"].nunique().rename("n_evaluated_units").reset_index()
    if best is None or best.empty:
        evaluated["n_threshold_testable_units"] = 0
        evaluated["n_tecs_observed_units"] = 0
    else:
        agg = best.groupby(group, dropna=False).agg(
            n_threshold_testable_units=("threshold_testable", "sum"),
            n_tecs_observed_units=("tecs_observed", "sum"),
        ).reset_index()
        evaluated = evaluated.merge(agg, on=group, how="left")
        evaluated[["n_threshold_testable_units", "n_tecs_observed_units"]] = evaluated[["n_threshold_testable_units", "n_tecs_observed_units"]].fillna(0).astype(int)
    evaluated["threshold_testable_fraction"] = evaluated["n_threshold_testable_units"] / evaluated["n_evaluated_units"].replace(0, np.nan)
    evaluated["tecs_observed_fraction"] = evaluated["n_tecs_observed_units"] / evaluated["n_evaluated_units"].replace(0, np.nan)
    return evaluated


def effect_summary(med: pd.DataFrame, n_boot:int) -> Dict[str,float]:
    if med.empty or "delta" not in med:
        return dict(n_pairs=0,candidate_median=float("nan"),control_median=float("nan"),median_delta=float("nan"),mean_delta=float("nan"),frac_positive_pairs=float("nan"),wilcoxon_p_greater=float("nan"),sign_p_greater=float("nan"),cohen_dz=float("nan"),paired_rank_biserial=float("nan"),median_delta_ci_low=float("nan"),median_delta_ci_high=float("nan"))
    d=med.delta.to_numpy(float); d=d[np.isfinite(d)]
    p,rb,dz=wilcoxon_greater(d)
    out=dict(n_pairs=int(len(d)), candidate_median=float(med[POP_CAND].median()), control_median=float(med[POP_CTRL].median()), median_delta=float(np.median(d)), mean_delta=float(np.mean(d)), frac_positive_pairs=float((d>0).mean()), wilcoxon_p_greater=float(p), sign_p_greater=binom_p_greater(int((d>0).sum()), len(d)), cohen_dz=float(dz), paired_rank_biserial=float(rb))
    if len(d) and n_boot>0:
        boots=np.array([np.median(RNG.choice(d, size=len(d), replace=True)) for _ in range(n_boot)])
        out["median_delta_ci_low"]=float(np.percentile(boots,2.5)); out["median_delta_ci_high"]=float(np.percentile(boots,97.5))
    else:
        out["median_delta_ci_low"]=out["median_delta_ci_high"]=float("nan")
    return out


def holm(pvals: Dict[str,float]) -> Dict[str,float]:
    items=sorted([(k,v) for k,v in pvals.items() if np.isfinite(v)], key=lambda kv:kv[1])
    out={k:float("nan") for k in pvals}; prev=0.0; m=len(items)
    for i,(k,p) in enumerate(items):
        val=max(prev, min(1.0, p*(m-i)))
        out[k]=float(val); prev=val
    return out


def bh_q(pvals: pd.Series) -> pd.Series:
    p=pvals.to_numpy(float); out=np.full(len(p),np.nan); finite=np.where(np.isfinite(p))[0]
    if len(finite)==0: return pd.Series(out,index=pvals.index)
    order=finite[np.argsort(p[finite])]; m=len(order)
    vals=np.array([p[i]*m/rank for rank,i in enumerate(order,start=1)])
    vals=np.minimum.accumulate(vals[::-1])[::-1]
    for i,v in zip(order,vals): out[i]=min(1.0,v)
    return pd.Series(out,index=pvals.index)


def family(feature:str)->str:
    s=str(feature).lower()
    if "wanda" in s:
        return "wanda/activation-magnitude"
    if "abs_activation" in s:
        return "activation-magnitude"
    if any(x in s for x in ["gradient", "margin", "learned_direction", "activation_x_gradient"]):
        return "gradient/effect"
    return "activation"


def fmt(x,digits=4):
    try:
        v=float(x)
        if not np.isfinite(v): return "n/a"
        if abs(v)<0.001 and v!=0: return f"{v:.2e}"
        return f"{v:.{digits}f}"
    except Exception: return str(x)


def ecdf(v):
    v=np.asarray(v,dtype=float); v=np.sort(v[np.isfinite(v)])
    return v, (np.arange(1,len(v)+1)/len(v) if len(v) else v)


def paper_figure_rc():
    """Compact, readable defaults for paper-only PDF figures."""
    import matplotlib.pyplot as plt
    return plt.rc_context({
        "font.size": 9.6,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })


def save_pdf_only(fig_obj, path: Path):
    """Save the figure to PDF."""
    fig_obj.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)


def clean_feature_label(feature: object, width: int = 26) -> str:
    """Readable axis label for feature names without changing the underlying data."""
    label = str(feature).replace("_", " ")
    return textwrap.fill(label, width=width, break_long_words=False, break_on_hyphens=False)


def clean_condition_label(label: object) -> str:
    """Compact y-axis label for run/baseline condition names."""
    parts = [p.strip() for p in str(label).split("|")]
    if len(parts) != 4:
        return clean_feature_label(label, 34)
    task, model, phase, baseline = parts
    task_map = {
        "arithmetic": "Arithmetic",
        "bon_jailbreaking": "Jailbreak",
        "grammar_acceptability": "Grammar",
        "hans_nli": "HANS-NLI",
        "random_fsm": "Random FSM",
    }
    task = task_map.get(task, task.replace("_", " ").title())
    model = model.split("/")[-1]
    model = model.replace("-Instruct", "")
    model = model.replace("pythia-", "Pythia-")
    model = model.replace("@step", "@")
    phase = "std" if phase == "standard" else phase
    baseline = {"positive": "pos", "negative": "neg"}.get(baseline, baseline)
    return f"{task} | {model} | {phase} | {baseline}"


def plot_outputs(out:Path, fs:pd.DataFrame, best:pd.DataFrame, rich:pd.DataFrame, feature_comp:pd.DataFrame, binned_agg:pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig=out/"figures"; fig.mkdir(exist_ok=True)

    def ecdf_plot(df, metric, xlabel, path):
        with paper_figure_rc():
            fig_obj, ax = plt.subplots(figsize=(4.35, 2.35))
            for pop in [POP_CAND, POP_CTRL]:
                x, y = ecdf(df.loc[df.population == pop, metric])
                if len(x):
                    ax.plot(x, y, linewidth=1.85, label=POP_LABELS[pop])
            ax.set_xlabel(xlabel, labelpad=1.0)
            ax.set_ylabel("Empirical CDF", labelpad=1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.35, linewidth=0.45)
            ax.legend(frameon=False, loc="lower right", handlelength=1.9, borderaxespad=0.15)
            fig_obj.subplots_adjust(left=0.16, right=0.995, bottom=0.20, top=0.985)
            save_pdf_only(fig_obj, path)
            plt.close(fig_obj)

    ecdf_plot(best, "causal_spiking_score", "Threshold-event causal score", fig/"ecdf_causal_spiking_score.pdf")
    ecdf_plot(fs, "flip_any_rate", "Singleton flip-any rate", fig/"ecdf_flip_rates_candidate_vs_control.pdf")
    ecdf_plot(best, "best_mcc", "Best held-out threshold |MCC|", fig/"ecdf_thresholdability_candidate_vs_control.pdf")

    if not rich.empty:
        r=rich.sort_values("delta")
        height=max(3.00, 0.27*len(r) + 0.70)
        with paper_figure_rc():
            fig_obj, ax = plt.subplots(figsize=(6.1, height))
            y=np.arange(len(r))
            ax.barh(y, r.delta, height=0.62)
            ax.axvline(0, color="0.10", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels([clean_condition_label(v) for v in r.condition_label], fontsize=7.6)
            ax.tick_params(axis="y", pad=1.5)
            ax.set_xlabel("Candidate median TECS - control median TECS", labelpad=1.0)
            ax.grid(axis="x", alpha=0.28, linewidth=0.45)
            fig_obj.subplots_adjust(left=0.36, right=0.995, bottom=0.11, top=0.99)
            save_pdf_only(fig_obj, fig/"paired_css_delta_by_run_baseline.pdf")
            plt.close(fig_obj)

    if not feature_comp.empty:
        top_n=min(10, len(feature_comp))
        top=feature_comp.head(top_n).sort_values("css_median_delta")
        height=max(1.95, 0.17*len(top) + 0.45)
        with paper_figure_rc():
            fig_obj, ax = plt.subplots(figsize=(4.9, height))
            y=np.arange(len(top))
            ax.barh(y, top.css_median_delta, height=0.58)
            ax.axvline(0, color="0.10", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels([clean_feature_label(v, 26) for v in top.feature], fontsize=7.4)
            ax.tick_params(axis="y", pad=1.0)
            ax.set_xlabel("Median candidate-control TECS delta", labelpad=1.0)
            ax.grid(axis="x", alpha=0.28, linewidth=0.45)
            fig_obj.subplots_adjust(left=0.43, right=0.995, bottom=0.18, top=0.99)
            save_pdf_only(fig_obj, fig/"feature_css_delta_ranking.pdf")
            plt.close(fig_obj)

    if not binned_agg.empty:
        with paper_figure_rc():
            fig_obj, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
            made=False
            for ax, baseline in zip(axes, BASELINE_ORDER):
                bd=binned_agg.loc[binned_agg.baseline_subset.astype(str).str.lower()==baseline]
                for fam in ["activation", "activation-magnitude", "wanda/activation-magnitude", "gradient/effect"]:
                    for pop in [POP_CAND, POP_CTRL]:
                        g=bd[(bd.feature_family==fam)&(bd.population==pop)].sort_values("bin_index")
                        if not g.empty:
                            made=True
                            ax.plot(g.bin_index,g.median_flip_rate,marker="o",markersize=3.6,linewidth=1.2,label=f"{fam} | {POP_LABELS[pop]}")
                ax.set_xlabel("Oriented proxy bin"); ax.set_title(f"Discovery direction {DIRECTION_BY_BASELINE[baseline]}")
                ax.grid(axis="y",alpha=.35,linewidth=.45); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            axes[0].set_ylabel("Median held-out flip probability")
            if made:
                axes[0].legend(frameon=False,fontsize=6.4,ncol=1,loc="upper left",borderaxespad=.15)
                fig_obj.subplots_adjust(left=.10,right=.995,bottom=.17,top=.88,wspace=.18)
                save_pdf_only(fig_obj,fig/"binned_flip_curves_oriented_proxy.pdf")
            plt.close(fig_obj)



def plot_manuscript_spiking_cut(
    paper_dir: Path,
    *,
    flip_summary: pd.DataFrame,
    primary: pd.DataFrame,
    results: Dict[str, object],
    best: pd.DataFrame,
    binned_agg: pd.DataFrame,
) -> None:
    """Write compact manuscript-facing RQ3 figures from Stage-7 diagnostics.

    This stage can report causal strength and threshold testability directly.
    Stage 8 writes the complete primary structural-control figure after nested
    feature selection and held-out threshold evaluation are available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paper_dir.mkdir(parents=True, exist_ok=True)
    cand_flip = row(flip_summary.to_dict("records"), POP_CAND)
    ctrl_flip = row(flip_summary.to_dict("records"), POP_CTRL)
    adj = results.get("planned_tests_holm_corrected_p", {}) or {}
    test_eff = results.get("threshold_testability_effect", {}) or {}

    # Stage 7 reports endpoints that do not require nested feature selection.
    # Stage 8 writes the complete structural-control phenotype analysis
    # (strength, testability, nested MCC, and nested TECS).
    panels = [
        ("Singleton causal effect", float(ctrl_flip.get("median_flip_any", math.nan)), float(cand_flip.get("median_flip_any", math.nan)), float(adj.get("strength_flip_rate", math.nan))),
        ("Threshold-testable fraction", float(test_eff.get("control_median", math.nan)), float(test_eff.get("candidate_median", math.nan)), float(adj.get("threshold_testability", math.nan))),
    ]
    with paper_figure_rc():
        fig_obj, axes = plt.subplots(1, 2, figsize=(5.3, 2.25), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (title, ctrl, cand, p_adj) in zip(axes, panels):
            if np.isfinite(ctrl) and np.isfinite(cand):
                ax.plot([ctrl, cand], [0, 1], color="0.55", linewidth=1.1, zorder=1)
                ax.scatter([ctrl], [0], marker="o", facecolor="white", edgecolor="0.25", s=34, zorder=3)
                ax.scatter([cand], [1], marker="o", color="0.25", s=34, zorder=3)
                lo = min(0.0, ctrl, cand)
                hi = max(ctrl, cand)
                pad = max(0.01, 0.18 * (hi - lo if hi > lo else max(abs(hi), 0.05)))
                ax.set_xlim(lo - 0.1 * pad, hi + pad)
            if ax is axes[0]:
                ax.set_yticks([0, 1], ["Structural control", "Candidate"])
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.25, linewidth=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(0.98, 0.06, f"Holm p={fmt(p_adj)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2)
        fig_obj.subplots_adjust(left=0.115, right=0.995, bottom=0.22, top=0.90, wspace=0.36)
        save_pdf_only(fig_obj, paper_dir / "fig4a_candidate_control_spiking_cut_summary.pdf")
        plt.close(fig_obj)

    # Max-feature thresholdability/TECS summaries are exploratory because the
    # held-out metric would otherwise participate in feature selection.  Their
    # machine-readable tables remain under analysis/, while manuscript inference
    # uses Stage-8 nested feature selection.
    for stale_name in [
        "fig4s1_thresholdability_ecdf.pdf",
        "fig4s2_tecs_ecdf.pdf",
    ]:
        (paper_dir / stale_name).unlink(missing_ok=True)

    # Plot absolute held-out flip probability by discovery direction. Relative
    # enrichment is retained only in binned_curve_aggregate.csv because a tiny
    # control baseline can make innocuous absolute changes look arbitrarily large.
    (paper_dir / "fig4b_threshold_tail_enrichment.pdf").unlink(missing_ok=True)
    if not binned_agg.empty:
        with paper_figure_rc():
            fig_obj, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
            made = False
            for ax, baseline in zip(axes, BASELINE_ORDER):
                bd = binned_agg.loc[binned_agg.baseline_subset.astype(str).str.lower() == baseline]
                for fam in ["activation", "activation-magnitude", "wanda/activation-magnitude", "gradient/effect"]:
                    for pop in [POP_CAND, POP_CTRL]:
                        g = bd[(bd.feature_family == fam) & (bd.population == pop)].sort_values("bin_index")
                        if not g.empty:
                            made = True
                            ax.plot(g.bin_index, g.median_flip_rate, marker="o", markersize=3.3, linewidth=1.15,
                                    label=f"{fam} | {POP_LABELS[pop]}")
                ax.set_xlabel("Oriented endogenous-proxy bin")
                ax.set_title(f"Discovery direction {DIRECTION_BY_BASELINE[baseline]}")
                ax.grid(axis="y", alpha=0.30, linewidth=0.45)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            axes[0].set_ylabel("Median held-out flip probability")
            target = paper_dir / "fig4s4_threshold_tail_response_by_direction.pdf"
            if made:
                axes[0].legend(frameon=False, fontsize=6.2, ncol=1, loc="upper left")
                fig_obj.subplots_adjust(left=0.10, right=0.995, bottom=0.18, top=0.88, wspace=0.18)
                save_pdf_only(fig_obj, target)
            else:
                target.unlink(missing_ok=True)
            plt.close(fig_obj)

    status = """# Figure 4 - RQ3: causal threshold-event analysis

The RQ3 experiment compares frozen Stage-7 overtopping candidates with random same-layer/head non-candidate controls selected independently of intervention outcomes. Each candidate is evaluated only in its frozen Stage-6 discovery baseline/direction (positive baseline = 1→0; negative baseline = 0→1). The analysis population is manifest-driven and excludes poisoning experiments.

Stage-7 diagnostics provide:

- candidate/control singleton causal strength;
- threshold-testability audits;
- descriptive direction-specific endogenous-proxy tail response in absolute held-out flip probability.

Stage 8 provides the primary nested held-out analysis:

- singleton causal strength;
- threshold-testable fraction;
- nested held-out threshold |MCC| among testable units;
- nested TECS lower bound over all evaluated units.

The primary comparison preserves same-layer/head structural controls and does not condition on causal strength. Causal-strength matching is a supplementary same-layer/head sensitivity analysis. Scalar selection, threshold fitting, and event orientation occur on training folds; held-out folds are used only for evaluation.

`threshold_testability_audit.csv` records how many model-evaluated units in each run/baseline/population have enough flip and non-flip events for threshold fitting. An undefined threshold test is not treated as a missing intervention.

"""
    (paper_dir / "README.md").write_text(status, encoding="utf-8")


def row(rows,pop):
    for r in rows:
        if r.get("population")==pop: return r
    return {}


def build_report(base_md: Optional[Path], results: Dict[str,object]) -> str:
    cf,rf=row(results["flip_summary"],POP_CAND),row(results["flip_summary"],POP_CTRL)
    fe=results["flip_effect"]
    te=results.get("threshold_testability_effect", {})
    adj=results.get("planned_tests_holm_corrected_p", {})
    empirical=f'''

## RQ3 population-level diagnostics

### Stage-7 endpoints that are statistically interpretable

The exact RQ3 population is manifest-driven and excludes poisoning experiments. By default it includes the primary manuscript settings plus configured supplementary overtopping settings that have complete candidate/control diagnostics. Missing supplementary diagnostics are reported in `population_audit.csv`; missing primary diagnostics remain a hard error.

| Population | Units | Median flip-any rate | Mean flip-any rate | Fraction with flip rate >= 0.05 |
|---|---:|---:|---:|---:|
| Candidate | {int(cf.get('n_units',0))} | {fmt(cf.get('median_flip_any'))} | {fmt(cf.get('mean_flip_any'))} | {fmt(cf.get('frac_ge_0_05'))} |
| Non-candidate control | {int(rf.get('n_units',0))} | {fmt(rf.get('median_flip_any'))} | {fmt(rf.get('mean_flip_any'))} | {fmt(rf.get('frac_ge_0_05'))} |

Paired by run and baseline subset, the singleton causal-strength candidate-control median delta is **{fmt(fe.get('median_delta'))}** with Holm-corrected p **{fmt(adj.get('strength_flip_rate'))}**.

Threshold fitting is possible for a much larger or smaller fraction of one population than the other, so testability is a separate endpoint rather than silently conditioning on the testable tail. The paired candidate-control testability-fraction delta is **{fmt(te.get('median_delta'))}**, Holm-corrected p **{fmt(adj.get('threshold_testability'))}**.

### Exploratory max-feature summaries

`aggregate_unit_tests.csv` contains held-out results for multiple scalar features. Choosing a feature by its held-out MCC and reporting that same maximum reuses held-out outcomes for selection. Thresholdability is also defined only for units with enough flip and non-flip events. Max-feature `best_mcc` and non-nested TECS tables are therefore descriptive diagnostics rather than primary inferential endpoints.

The inferential threshold-shape analysis is produced by Stage 8. It selects the scalar inside each training fold and evaluates the selected scalar on untouched held-out data. The primary RQ3 comparison preserves the independently sampled same-layer/head non-candidate controls and does not condition on causal strength; it reports causal strength, threshold testability, nested held-out threshold MCC, and nested TECS. Causal-strength matching is retained only as a supplementary sensitivity analysis and is constrained to the same layer/head.

### Stage-7 scope

Stage-7 aggregate reporting estimates candidate/control causal strength and threshold testability. Candidate/control differences in nested held-out threshold MCC and nested TECS are estimated by `threshold_shape_validation/threshold_shape_statistical_results.json`.
'''
    if base_md and base_md.exists():
        base=base_md.read_text()
        if "## RQ3 population-level diagnostics" in base: base=base.split("## RQ3 population-level diagnostics",1)[0].rstrip()+"\n"
        return base.replace("## Core Concepts", empirical+"\n## Core Concepts", 1) if "## Core Concepts" in base else base.rstrip()+"\n"+empirical
    return "# Overtopping-as-Spiking Diagnostics\n"+empirical


def main():
    args=parse_args(); out=Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    source_kind="zip" if args.zip else "root"; source_path=Path(args.zip or args.root).resolve(); base_md=Path(args.base_md).resolve() if args.base_md else None
    fs_all, ut_all, b_primary = load_exact_rq3_population(args, out)
    if fs_all.empty: raise RuntimeError("No aggregate_flip_stats.csv found")
    fs=fs_all[fs_all.population.isin([POP_CAND,POP_CTRL])].copy()
    flip_summary=fs.groupby("population").agg(n_units=("unit_key","count"), n_runs=("run_id","nunique"), median_flip_any=("flip_any_rate","median"), mean_flip_any=("flip_any_rate","mean"), q75_flip_any=("flip_any_rate",lambda s:s.quantile(.75)), q90_flip_any=("flip_any_rate",lambda s:s.quantile(.90)), q95_flip_any=("flip_any_rate",lambda s:s.quantile(.95)), max_flip_any=("flip_any_rate","max"), frac_ge_0_01=("flip_any_rate",lambda s:(s>=.01).mean()), frac_ge_0_05=("flip_any_rate",lambda s:(s>=.05).mean()), frac_ge_0_10=("flip_any_rate",lambda s:(s>=.10).mean()), frac_ge_0_20=("flip_any_rate",lambda s:(s>=.20).mean())).reset_index()
    flip_med=paired_medians(fs,"flip_any_rate"); flip_eff=effect_summary(flip_med,args.bootstrap); flip_cd=cliffs_delta(fs.loc[fs.population==POP_CAND,"flip_any_rate"],fs.loc[fs.population==POP_CTRL,"flip_any_rate"])

    if ut_all.empty or "population" not in ut_all.columns:
        ut = pd.DataFrame()
    else:
        ut = ut_all[ut_all.population.isin([POP_CAND,POP_CTRL])].copy()
        for col in ["median_test_abs_mcc", "population_strength"]:
            if col in ut.columns:
                ut[col] = pd.to_numeric(ut[col], errors="coerce")
        if "median_test_abs_mcc" in ut.columns:
            ut = ut.dropna(subset=["median_test_abs_mcc"])

    best = build_best_threshold_population(fs, ut)
    testability = threshold_testability_audit(fs, best)
    if best.empty:
        primary = pd.DataFrame(columns=["population", "n_units", "n_runs", "n_threshold_testable", "n_tecs_observed", "median_best_mcc", "mean_best_mcc", "median_strength", "mean_strength", "median_css", "mean_css", "q75_css", "q90_css", "frac_css_ge_0_01", "frac_css_ge_0_02"])
    else:
        primary=best.groupby("population").agg(
            n_units=("unit_key","count"),
            n_runs=("run_id","nunique"),
            n_threshold_testable=("threshold_testable","sum"),
            n_tecs_observed=("tecs_observed","sum"),
            median_best_mcc=("best_mcc","median"),
            mean_best_mcc=("best_mcc","mean"),
            median_strength=("population_strength","median"),
            mean_strength=("population_strength","mean"),
            median_css=("causal_spiking_score","median"),
            mean_css=("causal_spiking_score","mean"),
            q75_css=("causal_spiking_score",lambda s:s.quantile(.75)),
            q90_css=("causal_spiking_score",lambda s:s.quantile(.90)),
            frac_css_ge_0_01=("causal_spiking_score",lambda s:(s>=.01).mean()),
            frac_css_ge_0_02=("causal_spiking_score",lambda s:(s>=.02).mean()),
        ).reset_index()
    css_med=paired_medians(best,"causal_spiking_score"); css_eff=effect_summary(css_med,args.bootstrap); css_cd=cliffs_delta(best.loc[best.population==POP_CAND,"causal_spiking_score"], best.loc[best.population==POP_CTRL,"causal_spiking_score"])
    mcc_med=paired_medians(best,"best_mcc"); mcc_eff=effect_summary(mcc_med,args.bootstrap); mcc_cd=cliffs_delta(best.loc[best.population==POP_CAND,"best_mcc"], best.loc[best.population==POP_CTRL,"best_mcc"])
    test_med=paired_medians(testability,"threshold_testable_fraction")
    test_eff=effect_summary(test_med,args.bootstrap)
    adj=holm({"strength_flip_rate":flip_eff["wilcoxon_p_greater"],"threshold_testability":test_eff["wilcoxon_p_greater"]})
    feats=[]
    if not ut.empty and "feature" in ut.columns and "population_strength" in ut.columns and "median_test_abs_mcc" in ut.columns:
        for feat,g in ut.groupby("feature"):
            gg=g.copy(); gg["feature_css"]=gg.population_strength*gg.median_test_abs_mcc; med=paired_medians(gg,"feature_css"); medm=paired_medians(gg,"median_test_abs_mcc")
            if med.empty: continue
            ef=effect_summary(med,0); em=effect_summary(medm,0)
            feats.append(dict(feature=feat,family=family(feat),n_pairs=ef["n_pairs"],css_median_delta=ef["median_delta"],css_wilcoxon_p_greater=ef["wilcoxon_p_greater"],mcc_median_delta=em["median_delta"],mcc_wilcoxon_p_greater=em["wilcoxon_p_greater"]))
    feature_comp=pd.DataFrame(feats).sort_values("css_median_delta",ascending=False) if feats else pd.DataFrame()
    if not feature_comp.empty:
        feature_comp["css_bh_q"]=bh_q(feature_comp.css_wilcoxon_p_greater); feature_comp["mcc_bh_q"]=bh_q(feature_comp.mcc_wilcoxon_p_greater)
    rich=css_med.reset_index() if not css_med.empty else pd.DataFrame()
    if not rich.empty:
        meta=best[["run_id","task","model","setting","decode_only"]].drop_duplicates("run_id")
        rich=rich.merge(meta,on="run_id",how="left"); rich["condition_label"]=rich.apply(lambda r:f"{r.task} | {r.model} | {'decode' if r.decode_only else 'standard'} | {r.baseline_subset}",axis=1)
    b=b_primary.copy()
    binned_agg=pd.DataFrame()
    if not b.empty:
        b=b[b.population.isin([POP_CAND,POP_CTRL])].copy(); b["feature_family"]=b.feature.map(family);
        if "baseline_subset" not in b.columns:
            b["baseline_subset"] = b.get("source_baseline_dir", "unknown").astype(str).str.replace("_baseline", "", regex=False)
        keys=["run_id","baseline_subset","population","unit_key","feature","target"]
        b["curve_mean"]=b.groupby(keys,dropna=False).flip_rate.transform("mean"); b["flip_enrichment"]=b.flip_rate/b.curve_mean.replace(0,np.nan)
        binned_agg=b.groupby(["baseline_subset","feature_family","population","bin_index"],dropna=False).agg(
            median_flip_enrichment=("flip_enrichment","median"),mean_flip_enrichment=("flip_enrichment","mean"),
            median_flip_rate=("flip_rate","median"),mean_flip_rate=("flip_rate","mean"),n_curves=("unit_key","count")
        ).reset_index()
        binned_agg["direction"] = binned_agg["baseline_subset"].astype(str).str.lower().map(DIRECTION_BY_BASELINE)
    for name,df in {"flip_rate_summary.csv":flip_summary,"paired_flip_rate_by_run_baseline.csv":flip_med.reset_index(),"unit_primary_spiking_scores.csv":best,"threshold_testability_audit.csv":testability,"primary_spiking_score_summary.csv":primary,"paired_css_by_run_baseline.csv":css_med.reset_index(),"paired_best_mcc_by_run_baseline.csv":mcc_med.reset_index(),"paired_threshold_testability_by_run_baseline.csv":test_med.reset_index(),"paired_css_delta_rich.csv":rich,"feature_level_stat_summary.csv":feature_comp,"binned_curve_aggregate.csv":binned_agg}.items(): df.to_csv(out/name,index=False)
    results={"populations":{"candidate":POP_CAND,"control":POP_CTRL},"flip_summary":flip_summary.to_dict(orient="records"),"flip_effect":flip_eff,"flip_unit_cliffs_delta":flip_cd,"primary_summary":primary.to_dict(orient="records"),"threshold_testability":testability.to_dict(orient="records"),"css_effect_exploratory":css_eff,"css_unit_cliffs_delta_exploratory":css_cd,"best_mcc_effect_exploratory":mcc_eff,"best_mcc_unit_cliffs_delta_exploratory":mcc_cd,"threshold_testability_effect":test_eff,"planned_tests_holm_corrected_p":adj,"n_aggregate_flip_stats_files":int(len(fs_all.rel.unique())) if "rel" in fs_all else 0,"n_aggregate_unit_tests_files":int(len(ut_all.rel.unique())) if (not ut_all.empty and "rel" in ut_all) else 0}
    (out/"statistical_results.json").write_text(json.dumps(results,indent=2))
    plot_outputs(out,fs,best,rich,feature_comp,binned_agg)
    if args.paper_figures_dir:
        plot_manuscript_spiking_cut(
            Path(args.paper_figures_dir).expanduser().resolve(),
            flip_summary=flip_summary, primary=primary, results=results, best=best, binned_agg=binned_agg,
        )
    (out/"updated_spiking_diagnostics_experiments.md").write_text(build_report(base_md,results))
    print(f"Wrote outputs to {out}")

if __name__=="__main__": main()
