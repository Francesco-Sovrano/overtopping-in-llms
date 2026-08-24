"""Shared loading/filtering for rule-combination metric summaries."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_MIN_RULE_DATASET_COVERAGE = 0.005
RULE_METRICS_SUMMARY_CACHE_SIGNATURE = "rule_combo_metrics_from_raw_v1"


def normalize_score_scope(scope: str) -> str:
    value = str(scope or "test_selected").strip().lower().replace("-", "_").replace(" ", "_")
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
    }
    return aliases.get(value, value)


def score_scope_slug(scope: str) -> str:
    return normalize_score_scope(scope).replace("+", "_").replace("-", "_")


def rule_metrics_manifest_current(stats_dir: Path) -> bool:
    """Whether scoped metric summaries carry the current schema marker."""
    manifest_path = Path(stats_dir) / "rule_combo_metrics_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return manifest.get("signature") == RULE_METRICS_SUMMARY_CACHE_SIGNATURE


def scoped_rule_metrics_are_safe(run_dir: Path, scope: str) -> bool:
    scope = normalize_score_scope(scope)
    if scope not in ("test_selected", "all_fit"):
        return True
    ok = rule_metrics_manifest_current(run_dir)
    if not ok:
        print(
            "[rule-metrics] warning: stale or missing rule_combo_metrics_manifest.json "
            f"in {run_dir}; rerun Stage 7 with --summarize_rule_metrics. Ignoring {scope} summaries."
        )
    return ok


def apply_min_dataset_coverage(
    df: pd.DataFrame,
    min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE,
) -> pd.DataFrame:
    """Filter rule rows by minimum held-out dataset coverage."""
    if df is None or df.empty:
        return df
    try:
        floor = float(min_dataset_coverage)
    except Exception:
        floor = DEFAULT_MIN_RULE_DATASET_COVERAGE
    if floor <= 0:
        return df.copy()
    cov_col = next((c for c in ("dataset_coverage", "coverage") if c in df.columns), None)
    if cov_col is None:
        return df.copy()
    out = df.copy()
    coverage = pd.to_numeric(out[cov_col], errors="coerce")
    return out[coverage >= floor].copy()


def drop_train_scored_rule_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only held-out/eval-scored rule-combination rows."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "flip_target" in out.columns:
        out = out[~out["flip_target"].astype(str).str.startswith("train_")].copy()
    if "rule_combo_path" in out.columns:
        out = out[
            ~out["rule_combo_path"].astype(str).str.contains("rule_combo_train_", regex=False)
        ].copy()
    return out


def filter_rule_rows_by_score_scope(df: pd.DataFrame, score_scope: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    scope = normalize_score_scope(score_scope)
    out = df.copy()
    scope_col = (
        "score_scope"
        if "score_scope" in out.columns
        else "computed_on"
        if "computed_on" in out.columns
        else None
    )
    if scope_col is not None:
        out = out[out[scope_col].map(normalize_score_scope) == scope].copy()
    elif scope != "test":
        return out.iloc[0:0].copy()
    return drop_train_scored_rule_rows(out)


def best_per_neuron(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    sort_cols = [c for c in ("MCC", "F1", "BalancedAcc") if c in out.columns]
    for col in sort_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    key_cols = [c for c in ("rule_target_dir", "neuron_key") if c in out.columns]
    if not key_cols:
        key_cols = [c for c in ("layer_key", "neuron_id") if c in out.columns]
    if key_cols:
        out = out.drop_duplicates(key_cols, keep="first")
    return out


def load_eval_best_rule_rows(
    run_dir: Path,
    min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE,
    score_scope: str = "test_selected",
) -> pd.DataFrame:
    """Load scope-specific rule rows and keep one best row per localized neuron."""
    scope = normalize_score_scope(score_scope)
    if not scoped_rule_metrics_are_safe(run_dir, scope):
        return pd.DataFrame()
    slug = score_scope_slug(scope)
    scoped_all = run_dir / f"rule_combo_metrics_{slug}_all.csv"
    scoped_best = run_dir / f"rule_combo_metrics_{slug}_best_per_neuron.csv"
    all_scopes = run_dir / "rule_combo_metrics_all_scopes.csv"

    if scoped_all.exists():
        df = pd.read_csv(scoped_all, low_memory=False)
        return best_per_neuron(
            apply_min_dataset_coverage(filter_rule_rows_by_score_scope(df, scope), min_dataset_coverage)
        )
    if scoped_best.exists():
        df = pd.read_csv(scoped_best, low_memory=False)
        return apply_min_dataset_coverage(filter_rule_rows_by_score_scope(df, scope), min_dataset_coverage)
    if all_scopes.exists():
        df = pd.read_csv(all_scopes, low_memory=False)
        return best_per_neuron(
            apply_min_dataset_coverage(filter_rule_rows_by_score_scope(df, scope), min_dataset_coverage)
        )

    return pd.DataFrame()
