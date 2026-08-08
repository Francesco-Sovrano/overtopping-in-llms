#!/usr/bin/env python3
"""
Compute statistically valid E1/E3 method comparisons in one script.

Typical use:
  python3 -m analysis.10_compute_threshold_sweep_stats \
    --csv results/paper_tables/table2_threshold_sweep_per_run_long.csv \
    --data-root ./data \
    --out_dir results/paper_tables/stats \
    --primary_threshold 0.70 \
    --sample-stat mean \
    --n-boot 10000
"""

from __future__ import annotations
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]


import argparse
import json
import hashlib
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_MIN_RULE_DATASET_COVERAGE = 0.005

RULE_METRICS_SUMMARY_CACHE_SIGNATURE = "rule_combo_metrics_from_raw_v1"


def _rule_metrics_manifest_current(stats_dir: Path) -> bool:
    """Return True when scoped metric summaries carry the current schema marker.

    The manifest is a semantic/schema certificate for derived
    rule_combo_metrics_* summaries, not a hard cache key for the whole raw
    rules_dir.  We intentionally do not reject summaries when the optional raw
    digest differs: local rules_dir trees can contain extra raw rule_combo files
    from other runs or later experiments that do not affect this stats folder.
    Stage 7 still writes the raw digest for audit/debugging, and rerunning
    Stage 7 with --summarize_rule_metrics overwrites the metric summaries.
    """
    manifest_path = Path(stats_dir) / "rule_combo_metrics_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return manifest.get("signature") == RULE_METRICS_SUMMARY_CACHE_SIGNATURE


def _scoped_rule_metrics_are_safe(run_dir: Path, scope: str) -> bool:
    scope = _normalize_score_scope(scope)
    if scope not in ("test_selected", "all_fit"):
        return True
    ok = _rule_metrics_manifest_current(run_dir)
    if not ok:
        print(f"[rule-metrics] warning: stale or missing rule_combo_metrics_manifest.json in {run_dir}; rerun Stage 7 with --summarize_rule_metrics. Ignoring {scope} summaries.")
    return ok



def _apply_min_dataset_coverage(df: pd.DataFrame, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE) -> pd.DataFrame:
    """Filter rule rows by minimum held-out dataset_coverage.

    This support floor avoids sparse perfect-MCC artifacts driving HQ counts.
    Set min_dataset_coverage <= 0 to disable.  Older artifacts without a
    coverage column are left untouched.
    """
    if df is None or df.empty:
        return df
    try:
        floor = float(min_dataset_coverage)
    except Exception:
        floor = DEFAULT_MIN_RULE_DATASET_COVERAGE
    if floor <= 0:
        return df.copy()
    cov_col = None
    for c in ("dataset_coverage", "coverage"):
        if c in df.columns:
            cov_col = c
            break
    if cov_col is None:
        return df.copy()
    out = df.copy()
    cov = pd.to_numeric(out[cov_col], errors="coerce")
    return out[cov >= floor].copy()

def _drop_train_scored_rule_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only held-out/eval-scored rule-combo rows."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "flip_target" in out.columns:
        out = out[~out["flip_target"].astype(str).str.startswith("train_")].copy()
    if "rule_combo_path" in out.columns:
        out = out[~out["rule_combo_path"].astype(str).str.contains("rule_combo_train_", regex=False)].copy()
    return out

def _normalize_score_scope(scope: str) -> str:
    s = str(scope or "test_selected").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "hqt": "test_selected",
        "hq_t": "test_selected",
        "testselected": "test_selected",
        "heldout_selected": "test_selected",
        "held_out_selected": "test_selected",
        "hqf": "all_fit",
        "hq_f": "all_fit",
        "allfit": "all_fit",
        "all": "all_fit",
        "legacy": "test",
        "legacy_test": "test",
    }
    return aliases.get(s, s)

def _score_scope_slug(scope: str) -> str:
    return _normalize_score_scope(scope).replace("+", "_")

def _filter_rule_rows_by_score_scope(df: pd.DataFrame, score_scope: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    scope = _normalize_score_scope(score_scope)
    out = df.copy()
    scope_col = "score_scope" if "score_scope" in out.columns else "computed_on" if "computed_on" in out.columns else None
    if scope_col is not None:
        out = out[out[scope_col].map(_normalize_score_scope) == scope].copy()
    elif scope != "test":
        return out.iloc[0:0].copy()
    return _drop_train_scored_rule_rows(out)

def _best_per_neuron(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    sort_cols = [c for c in ["MCC", "F1", "BalancedAcc"] if c in out.columns]
    for c in sort_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    key_cols = [c for c in ["rule_target_dir", "neuron_key"] if c in out.columns]
    if not key_cols:
        key_cols = [c for c in ["layer_key", "neuron_id"] if c in out.columns]
    if key_cols:
        out = out.drop_duplicates(key_cols, keep="first")
    return out

def _load_eval_best_rule_rows(run_dir: Path, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE, score_scope: str = "test_selected") -> pd.DataFrame:
    """Load scope-specific rule rows and keep one best row per localized neuron.

    Default is ``test_selected`` (HQ-T): TRAIN-emitted rules whose OR-subset is
    selected/scored on TEST. Legacy ``test`` rows are loaded only when requested.
    """
    scope = _normalize_score_scope(score_scope)
    if not _scoped_rule_metrics_are_safe(run_dir, scope):
        return pd.DataFrame()
    slug = _score_scope_slug(scope)
    scoped_all = run_dir / f"rule_combo_metrics_{slug}_all.csv"
    scoped_best = run_dir / f"rule_combo_metrics_{slug}_best_per_neuron.csv"
    all_scopes = run_dir / "rule_combo_metrics_all_scopes.csv"

    if scoped_all.exists():
        df = pd.read_csv(scoped_all, low_memory=False)
        df = _filter_rule_rows_by_score_scope(df, scope)
        df = _apply_min_dataset_coverage(df, min_dataset_coverage)
        return _best_per_neuron(df)
    if scoped_best.exists():
        df = pd.read_csv(scoped_best, low_memory=False)
        df = _filter_rule_rows_by_score_scope(df, scope)
        return _apply_min_dataset_coverage(df, min_dataset_coverage)
    if all_scopes.exists():
        df = pd.read_csv(all_scopes, low_memory=False)
        df = _filter_rule_rows_by_score_scope(df, scope)
        df = _apply_min_dataset_coverage(df, min_dataset_coverage)
        return _best_per_neuron(df)

    # Backward-compatible fallback: old unsuffixed metric files mean legacy TEST,
    # not HQ-T/test_selected. Do not silently use them for test_selected.
    if scope == "test":
        all_path = run_dir / "rule_combo_metrics_all.csv"
        best_path = run_dir / "rule_combo_metrics_best_per_neuron.csv"
        if all_path.exists():
            df = _filter_rule_rows_by_score_scope(pd.read_csv(all_path, low_memory=False), "test")
            df = _apply_min_dataset_coverage(df, min_dataset_coverage)
            return _best_per_neuron(df)
        if best_path.exists():
            df = _filter_rule_rows_by_score_scope(pd.read_csv(best_path, low_memory=False), "test")
            return _apply_min_dataset_coverage(df, min_dataset_coverage)
    return pd.DataFrame()

from scipy import stats

# Imported only for the sample-based analysis. These functions already encode
# the repository layout and run definitions used by make_paper_tables.py.
try:
    from make_paper_tables import (  # type: ignore
        RUN_CONFIGS,
        TABLE2_RUNS,
        DEFAULT_MAIN_TASKS,
        _find_model_roots,
        _resolve_stats_dir,
        _task_org_model,
        _load_best_rule_metrics,
    )
except Exception:  # pragma: no cover - useful when only threshold CSV analysis is needed
    RUN_CONFIGS = {}
    TABLE2_RUNS = []
    DEFAULT_MAIN_TASKS = ("arithmetic", "bon_jailbreaking", "jailbreaking")
    _find_model_roots = None
    _resolve_stats_dir = None
    _task_org_model = None
    _load_best_rule_metrics = None


DEFAULT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]

LABEL_MAP = {
    "Rule split w/ Spectral anchoring plan": "Rule split + spectral coverage",
    "Rule split w/ Random anchoring plan": "Rule split + random coverage",
    "Fake rule split w/ Spectral anchoring plan": "Fake rule split + spectral coverage",
    "Spectral split": "Spectral split (no rule)",
    "Rule split w/ Bruteforce search": "Rule split + bruteforce search",
    "Rule split + Spectral plan": "Rule split + spectral coverage",
    "Rule split + Random plan": "Rule split + random coverage",
    "Fake rule split + Spectral plan": "Fake rule split + spectral coverage",
    "Rule split + spectral cov.": "Rule split + spectral coverage",
    "Rule split + random cov.": "Rule split + random coverage",
    "Fake rule split + spectral cov.": "Fake rule split + spectral coverage",
    "Rule split + spectral anchor": "Rule split + spectral coverage",
    "Rule split + random anchor": "Rule split + random coverage",
    "Fake rule split + spectral anchor": "Fake rule split + spectral coverage",
}

COMPARISONS = [
    ("Rule split + spectral coverage", "Spectral split (no rule)"),
    ("Rule split + spectral coverage", "Rule split + random coverage"),
    ("Rule split + spectral coverage", "Fake rule split + spectral coverage"),
]


def _fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.3g}"


def _parse_csv_list(s: str | None) -> list[str] | None:
    if s is None or not str(s).strip():
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _task_col(df: pd.DataFrame) -> str:
    for c in ["task", "Task"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find task column. Expected 'task' or 'Task'.")


def _llm_col(df: pd.DataFrame) -> str:
    for c in ["llm", "model", "Model", "ai_model"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find model/LLM column. Expected one of llm/model/Model/ai_model.")


def _method_col(df: pd.DataFrame) -> str:
    for c in ["stats_label", "Method", "method"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find method column. Expected stats_label, Method, or method.")


def _normalize_score_scope(scope: str) -> str:
    s = str(scope or "test").strip().lower().replace("_", "+").replace("-", "+").replace(" ", "")
    if s in ("test+selected", "heldout+selected", "held+out+selected", "hqt", "hq+t"):
        return "test_selected"
    if s in ("eval", "heldout", "held+out", "test"):
        return "test"
    if s in ("all+fit", "allfit", "final+all+fit", "finalfit", "all", "hqf", "hq+f"):
        return "all_fit"
    if s in ("train+test", "train+eval", "all+data", "traintest"):
        return "all_fit"
    if s.startswith("train"):
        return "train"
    return s.replace("+", "_")


def _score_scope_slug(scope: str) -> str:
    return _normalize_score_scope(scope).replace("+", "_").replace("-", "_")


def load_pivot(csv_path: str | Path, metric: str, score_scope: str = "test") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "score_scope" in df.columns:
        want = _normalize_score_scope(score_scope)
        df = df[df["score_scope"].map(_normalize_score_scope) == want].copy()
    elif "Score scope" in df.columns:
        want = _normalize_score_scope(score_scope)
        df = df[df["Score scope"].map(_normalize_score_scope) == want].copy()
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available columns: {list(df.columns)}")
    if "threshold" not in df.columns:
        raise ValueError("CSV must contain a 'threshold' column.")

    task_c = _task_col(df)
    llm_c = _llm_col(df)
    method_c = _method_col(df)

    df = df.copy()
    df["task"] = df[task_c].astype(str)
    df["llm"] = df[llm_c].astype(str)
    df["method"] = df[method_c].map(LABEL_MAP).fillna(df[method_c]).astype(str)
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce").round(6)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df.pivot_table(
        index=["task", "llm", "threshold"],
        columns="method",
        values=metric,
        aggfunc="mean",
    ).sort_index()


def coverage_report(pivot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in pivot.columns:
        n_runs = int(pivot[m].groupby(["task", "llm"]).apply(lambda s: s.notna().any()).sum())
        n_points = int(pivot[m].notna().sum())
        rows.append({"method": m, "n_task_llm_runs": n_runs, "n_threshold_points": n_points})
    return pd.DataFrame(rows).sort_values(["n_task_llm_runs", "method"], ascending=[False, True])


def make_table_sum(
    pivot: pd.DataFrame,
    thresholds: Iterable[float],
    methods: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = [round(float(t), 6) for t in thresholds]
    methods = list(pivot.columns) if methods is None else methods
    total_rows = []
    n_rows = []
    threshold_values = set(pivot.index.get_level_values("threshold"))
    for t in thresholds:
        if t not in threshold_values:
            continue
        sub = pivot.xs(t, level="threshold")
        for m in methods:
            if m not in sub.columns:
                continue
            vals = sub[m].dropna()
            total_rows.append({"method": m, "threshold": t, "value": float(vals.sum())})
            n_rows.append({"method": m, "threshold": t, "value": int(vals.shape[0])})
    totals = pd.DataFrame(total_rows).pivot(index="method", columns="threshold", values="value").reindex(methods)
    ns = pd.DataFrame(n_rows).pivot(index="method", columns="threshold", values="value").reindex(methods)
    return totals, ns


def primary_threshold_pairs(pivot: pd.DataFrame, a: str, b: str, threshold: float) -> pd.DataFrame:
    t = round(float(threshold), 6)
    if a not in pivot.columns or b not in pivot.columns:
        raise ValueError(f"Missing method: {a!r} or {b!r}")
    try:
        sub = pivot.xs(t, level="threshold")[[a, b]].dropna()
    except KeyError as exc:
        raise ValueError(f"Threshold {threshold} not found in data.") from exc
    out = sub.rename(columns={a: "A", b: "B"}).copy()
    out["diff"] = out["A"] - out["B"]
    return out.reset_index()


def autc_pairs(pivot: pd.DataFrame, a: str, b: str, thresholds: Iterable[float]) -> pd.DataFrame:
    """AUTC = normalized area under the threshold-count curve, one value per unit."""
    thresholds = [round(float(t), 6) for t in thresholds]
    x = np.asarray(thresholds, dtype=float)
    if x.size < 2:
        raise ValueError("AUTC needs at least two thresholds.")
    rows = []
    for (task, llm), g in pivot[[a, b]].groupby(level=["task", "llm"]):
        g2 = g.droplevel(["task", "llm"])
        if not set(thresholds).issubset(set(g2.index.astype(float))):
            continue
        y_a = g2.loc[thresholds, a].to_numpy(dtype=float)
        y_b = g2.loc[thresholds, b].to_numpy(dtype=float)
        if np.isnan(y_a).any() or np.isnan(y_b).any():
            continue
        autc_a = float(np.trapz(y_a, x) / (x.max() - x.min()))
        autc_b = float(np.trapz(y_b, x) / (x.max() - x.min()))
        rows.append({"task": task, "llm": llm, "A": autc_a, "B": autc_b, "diff": autc_a - autc_b})
    return pd.DataFrame(rows)


def exact_sign_test(diffs: np.ndarray, alternative: str) -> Dict[str, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    nonzero = diffs[diffs != 0]
    n = int(nonzero.size)
    if n == 0:
        return {"n_total": int(diffs.size), "n_nonzero": 0, "n_pos": 0, "n_neg": 0, "p": np.nan}
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    p = stats.binomtest(n_pos, n=n, p=0.5, alternative=alternative).pvalue
    return {"n_total": int(diffs.size), "n_nonzero": n, "n_pos": n_pos, "n_neg": n_neg, "p": float(p)}


def exact_signflip_test(
    diffs: np.ndarray,
    alternative: str,
    exact_max_n: int = 22,
    n_perm: int = 200000,
    seed: int = 0,
) -> Dict[str, float]:
    """Magnitude-sensitive paired sign-flip test on paired differences."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    nonzero = diffs[diffs != 0]
    n = int(nonzero.size)
    if n == 0:
        return {"n_nonzero": 0, "stat": np.nan, "p": np.nan, "mode": "none"}
    obs = float(np.sum(nonzero))

    if n <= exact_max_n:
        vals = []
        for signs in product([-1.0, 1.0], repeat=n):
            vals.append(float(np.sum(nonzero * np.asarray(signs))))
        null = np.asarray(vals, dtype=float)
        mode = "exact"
    else:
        signs = rng.choice([-1.0, 1.0], size=(int(n_perm), n), replace=True)
        null = signs @ nonzero
        null = np.concatenate([null, np.asarray([obs])])
        mode = "monte_carlo"

    eps = 1e-12
    if alternative == "greater":
        p = np.mean(null >= obs - eps)
    elif alternative == "less":
        p = np.mean(null <= obs + eps)
    else:
        p = np.mean(np.abs(null) >= abs(obs) - eps)
    return {"n_nonzero": n, "stat": obs, "p": float(p), "mode": mode}


def holm_adjust(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out.tolist()
    ordered = idx[np.argsort(p[idx])]
    running = 0.0
    m = ordered.size
    for rank, i in enumerate(ordered):
        adjusted = (m - rank) * p[i]
        running = max(running, adjusted)
        out[i] = min(1.0, running)
    return out.tolist()


def summarize_pairs(pairs: pd.DataFrame, alternative: str, seed: int) -> Dict[str, float]:
    diffs = pairs["diff"].to_numpy(dtype=float) if not pairs.empty else np.asarray([])
    sign = exact_sign_test(diffs, alternative=alternative)
    flip = exact_signflip_test(diffs, alternative=alternative, seed=seed)
    return {
        "n_pairs": int(diffs.size),
        "n_nonzero": int(sign["n_nonzero"]),
        "n_pos": int(sign["n_pos"]),
        "n_neg": int(sign["n_neg"]),
        "mean_diff": float(np.mean(diffs)) if diffs.size else np.nan,
        "median_diff": float(np.median(diffs)) if diffs.size else np.nan,
        "min_diff": float(np.min(diffs)) if diffs.size else np.nan,
        "max_diff": float(np.max(diffs)) if diffs.size else np.nan,
        "sign_p": float(sign["p"]) if sign["p"] == sign["p"] else np.nan,
        "signflip_p": float(flip["p"]) if flip["p"] == flip["p"] else np.nan,
        "signflip_mode": flip["mode"],
    }


def run_threshold_comparisons(
    pivot: pd.DataFrame,
    comparisons: List[Tuple[str, str]],
    primary_threshold: float,
    thresholds: Iterable[float],
    alternative: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_rows = []
    autc_rows = []
    primary_pairs_all = []
    autc_pairs_all = []

    for a, b in comparisons:
        if a not in pivot.columns or b not in pivot.columns:
            continue

        pp = primary_threshold_pairs(pivot, a, b, primary_threshold)
        pp.insert(0, "A_method", a)
        pp.insert(1, "B_method", b)
        pp.insert(2, "endpoint", f"count_at_{primary_threshold:.2f}")
        primary_pairs_all.append(pp)
        s = summarize_pairs(pp, alternative=alternative, seed=seed)
        s.update({"A_method": a, "B_method": b, "endpoint": f"count_at_{primary_threshold:.2f}"})
        primary_rows.append(s)

        apairs = autc_pairs(pivot, a, b, thresholds)
        apairs.insert(0, "A_method", a)
        apairs.insert(1, "B_method", b)
        apairs.insert(2, "endpoint", "AUTC")
        autc_pairs_all.append(apairs)
        s2 = summarize_pairs(apairs, alternative=alternative, seed=seed)
        s2.update({"A_method": a, "B_method": b, "endpoint": "AUTC"})
        autc_rows.append(s2)

    primary = pd.DataFrame(primary_rows)
    autc = pd.DataFrame(autc_rows)
    if not primary.empty:
        primary["sign_p_holm"] = holm_adjust(primary["sign_p"].tolist())
        primary["signflip_p_holm"] = holm_adjust(primary["signflip_p"].tolist())
    if not autc.empty:
        autc["sign_p_holm"] = holm_adjust(autc["sign_p"].tolist())
        autc["signflip_p_holm"] = holm_adjust(autc["signflip_p"].tolist())

    primary_pairs = pd.concat(primary_pairs_all, ignore_index=True) if primary_pairs_all else pd.DataFrame()
    autc_pairs_df = pd.concat(autc_pairs_all, ignore_index=True) if autc_pairs_all else pd.DataFrame()
    return primary, autc, primary_pairs, autc_pairs_df


def to_latex_tabular(totals: pd.DataFrame) -> str:
    thresholds = list(totals.columns)
    lines = [
        r"\begin{tabular}{l" + "c" * len(thresholds) + "}",
        r"\toprule",
        "Method & " + " & ".join([f"$t={t:.2f}$" for t in thresholds]) + r" \\",
        r"\midrule",
    ]
    for method in totals.index:
        vals = [_fmt(totals.loc[method, t]) for t in thresholds]
        lines.append(method + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sample-based, threshold-free stats
# ---------------------------------------------------------------------------


def _metric_values(x: pd.Series) -> np.ndarray:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _stat(x: np.ndarray, name: str, threshold: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if name == "mean":
        return float(np.mean(x))
    if name == "median":
        return float(np.median(x))
    if name == "prop_ge_threshold":
        return float(np.mean(x >= threshold))
    if name == "n_ge_threshold":
        return float(np.sum(x >= threshold))
    raise ValueError(f"Unknown stat: {name}")


def collect_per_neuron_scores(
    data_root: Path,
    tasks: Sequence[str] | None,
    run_names: Sequence[str],
    metric_col: str,
    min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE,
    score_scope: str = "test",
) -> pd.DataFrame:
    if _find_model_roots is None or _resolve_stats_dir is None or _task_org_model is None:
        raise RuntimeError("make_paper_tables.py helpers could not be imported; sample stats require repository context.")
    rows = []
    for model_root in _find_model_roots(data_root, tasks=tasks):
        task, org, model = _task_org_model(data_root, model_root)
        for run_name in run_names:
            if run_name not in RUN_CONFIGS:
                continue
            stats_dir = _resolve_stats_dir(model_root, run_name)
            if _load_best_rule_metrics is not None:
                df = _load_best_rule_metrics(stats_dir, min_dataset_coverage=min_dataset_coverage, score_scope=score_scope)
            else:
                df = _load_eval_best_rule_rows(stats_dir, min_dataset_coverage=min_dataset_coverage, score_scope=score_scope)
            if df.empty:
                continue
            if metric_col not in df.columns:
                if metric_col != "MCC" and "MCC" in df.columns:
                    metric_col_local = "MCC"
                else:
                    continue
            else:
                metric_col_local = metric_col
            if "layer_key" not in df.columns or "neuron_id" not in df.columns:
                continue
            df = df.dropna(subset=["layer_key", "neuron_id"]).copy()
            df[metric_col_local] = pd.to_numeric(df[metric_col_local], errors="coerce")
            df = df.dropna(subset=[metric_col_local])
            df = df.sort_values(metric_col_local, ascending=False).drop_duplicates(
                subset=["layer_key", "neuron_id"], keep="first"
            )
            display = RUN_CONFIGS[run_name]["display"]
            for _, r in df.iterrows():
                rows.append(
                    {
                        "task": task,
                        "org": org,
                        "model": model,
                        "unit": f"{task}/{org}/{model}",
                        "run_name": run_name,
                        "method": display,
                        "layer_key": str(r["layer_key"]),
                        "neuron_id": int(r["neuron_id"]),
                        "metric_col": metric_col_local,
                        "score": float(r[metric_col_local]),
                    }
                )
    return pd.DataFrame(rows)


def per_unit_metrics(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    if scores.empty:
        return pd.DataFrame()
    for (unit, method), g in scores.groupby(["unit", "method"]):
        vals = _metric_values(g["score"])
        rows.append(
            {
                "unit": unit,
                "task": g["task"].iloc[0],
                "org": g["org"].iloc[0],
                "model": g["model"].iloc[0],
                "method": method,
                "n_neurons": int(vals.size),
                "mean_score": _stat(vals, "mean", threshold),
                "median_score": _stat(vals, "median", threshold),
                "prop_ge_threshold": _stat(vals, "prop_ge_threshold", threshold),
                "n_ge_threshold": _stat(vals, "n_ge_threshold", threshold),
            }
        )
    return pd.DataFrame(rows)


def paired_observed_diffs(per_unit: pd.DataFrame, method_a: str, method_b: str, metric: str) -> pd.DataFrame:
    if per_unit.empty:
        return pd.DataFrame(columns=["unit", "diff"])
    wide = per_unit.pivot_table(index="unit", columns="method", values=metric, aggfunc="first")
    if method_a not in wide.columns or method_b not in wide.columns:
        return pd.DataFrame(columns=["unit", "diff"])
    out = wide[[method_a, method_b]].dropna().copy()
    out["diff"] = out[method_a] - out[method_b]
    return out.reset_index()[["unit", "diff"]]


def _paired_units(scores: pd.DataFrame, method_a: str, method_b: str) -> list[str]:
    if scores.empty:
        return []
    methods_by_unit = scores.groupby("unit")["method"].apply(set)
    return sorted([u for u, s in methods_by_unit.items() if method_a in s and method_b in s])


def sampled_bootstrap_diff(
    scores: pd.DataFrame,
    method_a: str,
    method_b: str,
    stat_name: str,
    threshold: float,
    sample_size: int | None,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    units = _paired_units(scores, method_a, method_b)
    if not units:
        return np.array([], dtype=float)

    cache: dict[tuple[str, str], np.ndarray] = {}
    for u in units:
        for m in (method_a, method_b):
            arr = _metric_values(scores[(scores["unit"] == u) & (scores["method"] == m)]["score"])
            cache[(u, m)] = arr

    boot = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        diffs = []
        for u in sampled_units:
            a = cache[(u, method_a)]
            c = cache[(u, method_b)]
            if a.size == 0 or c.size == 0:
                continue
            k = int(sample_size) if sample_size is not None else int(min(a.size, c.size))
            if k <= 0:
                continue
            aa = rng.choice(a, size=k, replace=True)
            cc = rng.choice(c, size=k, replace=True)
            diffs.append(_stat(aa, stat_name, threshold) - _stat(cc, stat_name, threshold))
        boot[b] = float(np.mean(diffs)) if diffs else float("nan")
    return boot[np.isfinite(boot)]


def run_sample_comparisons(
    scores: pd.DataFrame,
    per_unit: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    stat_name: str,
    threshold: float,
    sample_size: int | None,
    n_boot: int,
    alternative: str,
    seed: int,
) -> pd.DataFrame:
    stat_to_unit_metric = {
        "mean": "mean_score",
        "median": "median_score",
        "prop_ge_threshold": "prop_ge_threshold",
        "n_ge_threshold": "n_ge_threshold",
    }
    unit_metric = stat_to_unit_metric[stat_name]

    rows = []
    for i, (a, b) in enumerate(comparisons):
        diffs_df = paired_observed_diffs(per_unit, a, b, unit_metric)
        summary = summarize_pairs(diffs_df, alternative=alternative, seed=seed + i)
        boot = sampled_bootstrap_diff(
            scores=scores,
            method_a=a,
            method_b=b,
            stat_name=stat_name,
            threshold=threshold,
            sample_size=sample_size,
            n_boot=n_boot,
            seed=seed + i,
        )
        rows.append(
            {
                "A_method": a,
                "B_method": b,
                "sample_stat": stat_name,
                "unit_metric_for_exact_test": unit_metric,
                "sample_size_per_method_per_unit": "min_pair" if sample_size is None else int(sample_size),
                **summary,
                "bootstrap_mean_diff": float(np.mean(boot)) if boot.size else float("nan"),
                "bootstrap_ci95_lo": float(np.quantile(boot, 0.025)) if boot.size else float("nan"),
                "bootstrap_ci95_hi": float(np.quantile(boot, 0.975)) if boot.size else float("nan"),
                "n_boot": int(n_boot),
                "alternative": alternative,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["sign_p_holm"] = holm_adjust(out["sign_p"].tolist())
        out["signflip_p_holm"] = holm_adjust(out["signflip_p"].tolist())
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to table2_threshold_sweep_per_run_long.csv")
    ap.add_argument("--metric", default="n_high_quality_neurons")
    ap.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--primary_threshold", type=float, default=0.70, help="Primary HQ-T/test-selected MCC threshold. Default: 0.70.")
    ap.add_argument("--primary_threshold_all_fit", type=float, default=0.70, help="Primary all-fit MCC threshold. Default: 0.70.")
    ap.add_argument("--score-scope", default="both", choices=["test", "test_selected", "all_fit", "both", "legacy_both"], help="Score scope(s) to analyze when the input CSV contains multiple scopes. Default 'both' means paper defaults: HQ-T/test_selected plus HQ-F/all_fit. Use 'test' for the legacy frozen-combo TEST score or 'legacy_both' for legacy TEST plus ALL-FIT.")
    ap.add_argument("--alt", choices=["two-sided", "greater", "less"], default="greater", help="Alternative for A-B.")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "results" / "paper_tables" / "stats"), help="Output directory for CSV artifacts. Default: <repo>/results/paper_tables/stats")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--data-root", type=Path, default=None, help="If provided, also run sample-based per-neuron stats.")
    ap.add_argument("--tasks", default=",".join(DEFAULT_MAIN_TASKS), help="Comma-separated task names; empty means all.")
    ap.add_argument("--metric-col", default="MCC", help="Per-neuron score column for sample-based stats.")
    ap.add_argument("--min-rule-dataset-coverage", type=float, default=DEFAULT_MIN_RULE_DATASET_COVERAGE,
                    help="Minimum held-out dataset_coverage required for sample-based per-neuron scores. Use 0 to disable.")
    ap.add_argument("--sample-stat", choices=["mean", "median", "prop_ge_threshold", "n_ge_threshold"], default="mean")
    ap.add_argument("--sample-size", type=int, default=0, help="0 means min(n_A, n_B) within each paired unit.")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--skip-sample-stats", action="store_true")
    args = ap.parse_args()

    base_out_dir = Path(args.out_dir) if args.out_dir else None
    scope_arg = str(args.score_scope).strip().lower()
    if scope_arg == "both":
        score_scopes = ["test_selected", "all_fit"]
    elif scope_arg == "legacy_both":
        score_scopes = ["test", "all_fit"]
    else:
        score_scopes = [_normalize_score_scope(args.score_scope)]
    thresholds = [round(float(t), 6) for t in args.thresholds]
    wrote_dirs = []

    for scope in score_scopes:
        scope_slug = _score_scope_slug(scope)
        out_dir = None
        if base_out_dir:
            out_dir = base_out_dir / scope_slug if len(score_scopes) > 1 else base_out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            wrote_dirs.append(out_dir)

        print(f"\n##############################")
        print(f"# Score scope: {scope.upper()}")
        print(f"##############################")

        pivot = load_pivot(args.csv, args.metric, score_scope=scope)
        if pivot.empty or len(pivot.columns) == 0:
            msg = f"No rows found for score scope {scope!r} in {args.csv}. New Stage 7 artifacts must be generated before this scope is available."
            print(msg)
            if out_dir:
                (out_dir / "NO_ROWS_FOR_SCORE_SCOPE.txt").write_text(msg + "\n", encoding="utf-8")
            continue

        print("\n== Method coverage ==")
        print(coverage_report(pivot).to_string(index=False))

        totals, ns = make_table_sum(pivot, thresholds=thresholds)
        print("\n== HQ ==")
        print(totals.apply(lambda col: col.map(_fmt)).to_string())
        print("\n== Number of contributing task/model runs per threshold ==")
        print(ns.to_string())
        print("\n== LaTeX tabular for Table 2 ==")
        print(to_latex_tabular(totals))

        primary_threshold_scope = float(args.primary_threshold_all_fit) if scope == "all_fit" else float(args.primary_threshold)
        primary, autc, primary_pairs, autc_pairs_df = run_threshold_comparisons(
            pivot,
            comparisons=COMPARISONS,
            primary_threshold=primary_threshold_scope,
            thresholds=thresholds,
            alternative=args.alt,
            seed=args.seed,
        )
        for df in (primary, autc, primary_pairs, autc_pairs_df):
            if not df.empty:
                df.insert(0, "score_scope", scope)

        cols = [
            "score_scope", "A_method", "B_method", "endpoint", "n_pairs", "n_nonzero", "n_pos", "n_neg",
            "median_diff", "mean_diff", "min_diff", "max_diff",
            "sign_p", "sign_p_holm", "signflip_p", "signflip_p_holm", "signflip_mode",
        ]
        cols_no_scope = [c for c in cols if c != "score_scope"]

        print(f"\n== Primary exact paired tests at MCC threshold {primary_threshold_scope:.2f} ==")
        if primary.empty:
            print("No complete paired comparisons found.")
        else:
            print(primary[[c for c in cols if c in primary.columns]].to_string(index=False))

        print("\n== Sensitivity exact paired tests on AUTC over selected thresholds ==")
        if autc.empty:
            print("No complete paired comparisons found.")
        else:
            print(autc[[c for c in cols if c in autc.columns]].to_string(index=False))

        if out_dir:
            totals.to_csv(out_dir / "table2_descriptive_totals.csv")
            ns.to_csv(out_dir / "table2_descriptive_n_runs.csv")
            primary.to_csv(out_dir / "method_comparison_primary_threshold_exact_tests.csv", index=False)
            autc.to_csv(out_dir / "method_comparison_autc_exact_tests.csv", index=False)
            primary_pairs.to_csv(out_dir / "method_comparison_primary_threshold_pairs.csv", index=False)
            autc_pairs_df.to_csv(out_dir / "method_comparison_autc_pairs.csv", index=False)

        if not args.skip_sample_stats and args.data_root is not None:
            tasks = _parse_csv_list(args.tasks)
            sample_size = None if int(args.sample_size) <= 0 else int(args.sample_size)
            print("\n== Sample-based per-neuron MCC comparison ==")
            scores = collect_per_neuron_scores(
                data_root=args.data_root,
                tasks=tasks,
                run_names=TABLE2_RUNS,
                metric_col=args.metric_col,
                min_dataset_coverage=args.min_rule_dataset_coverage,
                score_scope=scope,
            )
            if scores.empty:
                print(f"No per-neuron score rows found for score scope {scope!r}. Check --data-root and run artifact paths.")
            else:
                scores.insert(0, "score_scope", scope)
                per_unit = per_unit_metrics(scores, threshold=primary_threshold_scope)
                if not per_unit.empty:
                    per_unit.insert(0, "score_scope", scope)
                comp = run_sample_comparisons(
                    scores=scores,
                    per_unit=per_unit,
                    comparisons=COMPARISONS,
                    stat_name=args.sample_stat,
                    threshold=primary_threshold_scope,
                    sample_size=sample_size,
                    n_boot=args.n_boot,
                    alternative=args.alt,
                    seed=args.seed,
                )
                if not comp.empty:
                    comp.insert(0, "score_scope", scope)
                if comp.empty:
                    print("No matched method comparisons found for sample-based stats.")
                else:
                    sample_cols = [
                        "score_scope", "A_method", "B_method", "sample_stat", "unit_metric_for_exact_test",
                        "n_pairs", "n_nonzero", "n_pos", "n_neg", "median_diff", "mean_diff",
                        "sign_p", "sign_p_holm", "signflip_p", "signflip_p_holm",
                        "bootstrap_mean_diff", "bootstrap_ci95_lo", "bootstrap_ci95_hi",
                        "sample_size_per_method_per_unit", "n_boot",
                    ]
                    print(comp[[c for c in sample_cols if c in comp.columns]].to_string(index=False))
                if out_dir:
                    scores.to_csv(out_dir / "per_neuron_scores.csv", index=False)
                    per_unit.to_csv(out_dir / "per_unit_metrics.csv", index=False)
                    comp.to_csv(out_dir / "sampled_pairwise_comparisons.csv", index=False)

        if out_dir:
            print(f"\nWrote CSV artifacts for {scope} to {out_dir}")

    if base_out_dir and len(wrote_dirs) > 1:
        print("\nWrote score-scope-specific CSV artifacts to:")
        for d in wrote_dirs:
            print(f"  {d}")

if __name__ == "__main__":
    main()
