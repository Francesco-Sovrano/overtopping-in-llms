#!/usr/bin/env python3
from pathlib import Path

import argparse
import json
import hashlib
import re
from pathlib import Path
from collections import Counter, defaultdict

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


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# Cleaned-up cross-model comparison script.
#
# Main fixes relative to the original version:
#   1) Do NOT silently drop incomplete models unless explicitly requested.
#   2) Discover available runs dynamically instead of assuming every task uses all RUN_MAP keys.
#   3) Use a plotting/grouping key tied to run_name, not a potentially-colliding display label.
#   4) Build plot order from the runs actually present, so labels always match the data.
#   5) Only show decode-only/full qualifiers when both variants exist for the same base label.
#   6) Skip empty plots entirely; save PDFs only.
# --------------------------------------------------------------------------------------

RUN_MAP = {
    'rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor': ('rule_split-spectral_sample-decode_only', 'agonist_neurons-fast-spectral_anchor', False),
    'rule_split-spectral_sample-decode_only-agonist_neurons-fast-random_anchor': ('rule_split-spectral_sample-decode_only', 'agonist_neurons-fast-random_anchor', False),
    'rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor-fake_targets': ('rule_split-spectral_sample-decode_only', 'agonist_neurons-fast-spectral_anchor', True),
    'spectral_split-decode_only-agonist_neurons-fast-random_anchor': ('spectral_split-decode_only', 'agonist_neurons-fast-random_anchor', False),
    'rule_split-spectral_sample-decode_only-agonist_neurons-spectral_anchor': ('rule_split-spectral_sample-decode_only', 'agonist_neurons-spectral_anchor', False),

    'rule_split-spectral_sample-agonist_neurons-fast-spectral_anchor': ('rule_split-spectral_sample-decode_only', 'agonist_neurons-fast-spectral_anchor', False),
    'rule_split-spectral_sample-agonist_neurons-fast-random_anchor': ('rule_split-spectral_sample-decode_only', 'agonist_neurons-fast-random_anchor', False),
    'rule_split-spectral_sample-agonist_neurons-fast-spectral_anchor-fake_targets': ('rule_split-spectral_sample-decode_only', 'agonist_neurons-fast-spectral_anchor', True),
    'spectral_split-agonist_neurons-fast-random_anchor': ('spectral_split-decode_only', 'agonist_neurons-fast-random_anchor', False),
    'rule_split-spectral_sample-agonist_neurons-spectral_anchor': ('rule_split-spectral_sample-decode_only', 'agonist_neurons-spectral_anchor', False),
}

DISPLAY_NAME = {
    'rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor': 'Rule split + Spectral plan',
    'rule_split-spectral_sample-decode_only-agonist_neurons-fast-random_anchor': 'Rule split + Random plan',
    'rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor-fake_targets': 'Fake rule split + Spectral plan',
    'spectral_split-decode_only-agonist_neurons-fast-random_anchor': 'Spectral split',
    'rule_split-spectral_sample-decode_only-agonist_neurons-spectral_anchor': 'Rule split + Bruteforce search',

    'rule_split-spectral_sample-agonist_neurons-fast-spectral_anchor': 'Rule split + Spectral plan',
    'rule_split-spectral_sample-agonist_neurons-fast-random_anchor': 'Rule split + Random plan',
    'rule_split-spectral_sample-agonist_neurons-fast-spectral_anchor-fake_targets': 'Fake rule split + Spectral plan',
    'spectral_split-agonist_neurons-fast-random_anchor': 'Spectral split',
    'rule_split-spectral_sample-agonist_neurons-spectral_anchor': 'Rule split + Bruteforce search',
}


def ecdf_xy(vals: np.ndarray):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    vals = np.sort(vals)
    y = np.arange(1, vals.size + 1, dtype=float) / vals.size
    return vals, y


def read_json(path: Path):
    return json.loads(path.read_text())


def list_model_roots(task_dir: Path):
    out = []
    seen = set()
    for stats_root in task_dir.rglob('rule_extraction_results/neuron_flip_rules/stats'):
        model_root = stats_root.parent.parent.parent
        if model_root not in seen:
            out.append(model_root)
            seen.add(model_root)
    return sorted(out)


def discover_available_runs(model_roots: list[Path]) -> list[str]:
    runs = set()
    for model_root in model_roots:
        stats_base = model_root / 'rule_extraction_results' / 'neuron_flip_rules' / 'stats'
        if not stats_base.exists():
            continue
        for child in stats_base.iterdir():
            if child.is_dir():
                runs.add(child.name)
    return sorted(runs)


def base_display_label(run_name: str) -> str:
    return DISPLAY_NAME.get(run_name, run_name)


def variant_kind(run_name: str) -> str:
    return 'decode-only' if 'decode_only' in run_name else 'full'


def make_unique_display_labels(run_names: list[str]) -> dict[str, str]:
    base_labels = {rn: base_display_label(rn) for rn in run_names}
    grouped = defaultdict(list)
    for rn, label in base_labels.items():
        grouped[label].append(rn)

    out = {}
    for label, members in grouped.items():
        kinds = {variant_kind(rn) for rn in members}
        need_decode_full = len(kinds) > 1

        fake_counts = Counter('fake_targets' in rn for rn in members)
        need_fake = len(fake_counts) > 1 and fake_counts[True] > 0 and fake_counts[False] > 0

        anchor_suffix_counts = Counter(
            'random-anchor' if 'random_anchor' in rn else 'spectral-anchor' if 'fast-spectral_anchor' in rn else 'bruteforce'
            for rn in members
        )
        need_anchor_detail = len(anchor_suffix_counts) > 1

        for rn in members:
            qualifiers = []
            if need_decode_full:
                qualifiers.append(variant_kind(rn))
            if need_fake and 'fake_targets' in rn and 'Fake' not in label:
                qualifiers.append('fake-targets')
            if need_anchor_detail:
                if 'random_anchor' in rn and 'Random' not in label:
                    qualifiers.append('random-anchor')
                elif 'fast-spectral_anchor' in rn and 'Spectral' not in label:
                    qualifiers.append('spectral-anchor')
                elif 'spectral_anchor' in rn and 'Bruteforce' in label:
                    qualifiers.append('bruteforce')
            out[rn] = f"{label} ({', '.join(qualifiers)})" if qualifiers else label
    return out


def get_localization_dir(model_root: Path, run_name: str, baseline_subset: str = 'positive_baseline'):
    if run_name not in RUN_MAP:
        return None
    split_dir, anchor_dir, is_fake = RUN_MAP[run_name]
    base = model_root / ('neural_circuit_discovery_results_fake_targets' if is_fake else 'neural_circuit_discovery_results')
    loc = base / 'eap_ig_inputs' / split_dir / anchor_dir / baseline_subset
    return loc if loc.exists() else None


def neuron_set_from_buckets(nb_path: Path, tau: float):
    nb = read_json(nb_path)
    out = set()
    for _, items in nb.items():
        if not isinstance(items, dict):
            continue
        for key, info in items.items():
            rec = info.get('last_record', {})
            me = rec.get('max_effect', info.get('best_abs_gap', None))
            if me is None:
                continue
            try:
                if float(me) >= tau:
                    out.add(key)
            except Exception:
                pass
    return out


def selectivity_values_from_buckets(nb_path: Path, tau: float):
    nb = read_json(nb_path)
    sels = []
    for _, items in nb.items():
        if not isinstance(items, dict):
            continue
        for _, info in items.items():
            rec = info.get('last_record', {})
            me = rec.get('max_effect', info.get('best_abs_gap', None))
            if me is None:
                continue
            try:
                if float(me) < tau:
                    continue
            except Exception:
                continue
            gap = rec.get('accuracy_gap', None)
            if gap is None:
                continue
            try:
                sels.append(abs(float(gap)))
            except Exception:
                pass
    return np.asarray(sels, dtype=float)


def summarize_run(model_root: Path, run_name: str, tau: float, eps: float, display_label: str, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE, score_scope: str = "test_selected"):
    stats_dir = model_root / 'rule_extraction_results' / 'neuron_flip_rules' / 'stats' / run_name
    if not stats_dir.exists():
        return None

    out = {
        'model': model_root.name,
        'model_root': str(model_root),
        'run_name': run_name,
        'condition_key': run_name,
        'condition': display_label,
        'variant_kind': variant_kind(run_name),

        'n_neurons_with_rules': np.nan,
        'rule_mcc_median': np.nan,
        'rule_mcc_mean': np.nan,

        'union_flip_any_rate': np.nan,
        'union_c2i_rate': np.nan,
        'union_i2c_rate': np.nan,

        'hq_threshold': np.nan,
        'frac_neurons_hq': np.nan,
        'hq_flip_any_union_share_of_all': np.nan,

        'n_tau_agonists': np.nan,
        'median_abs_acc_gap': np.nan,
        'frac_eps_selective': np.nan,

        'median_n_ablation_groups': np.nan,
        'mean_n_ablation_groups': np.nan,
        'n_circuits': np.nan,
    }

    rms_path = stats_dir / 'rule_metrics_summary.json'
    if rms_path.exists():
        rms = read_json(rms_path)
        out['n_neurons_with_rules'] = rms.get('n_neurons_with_rules', np.nan)
        out['rule_mcc_median'] = rms.get('MCC', {}).get('median', np.nan)
        out['rule_mcc_mean'] = rms.get('MCC', {}).get('mean', np.nan)

    fsg_path = stats_dir / 'flip_stats_global.json'
    if fsg_path.exists():
        fsg = read_json(fsg_path)
        out['union_flip_any_rate'] = fsg.get('union_flip_any_unique_rate', np.nan)
        out['union_c2i_rate'] = fsg.get('union_c2i_unique_rate', np.nan)
        out['union_i2c_rate'] = fsg.get('union_i2c_unique_rate', np.nan)

    hq_path = stats_dir / 'high_quality_neuron_flip_coverage.json'
    if hq_path.exists():
        hq = read_json(hq_path)
        out['hq_threshold'] = hq.get('quality_threshold', np.nan)
        out['frac_neurons_hq'] = hq.get('frac_neurons_high_quality', np.nan)
        out['hq_flip_any_union_share_of_all'] = hq.get('flip_any', {}).get('union_share_of_all', np.nan)

    # Prefer live filtered eval rows for cross-model rule-quality summaries so
    # older stats JSONs cannot reintroduce tiny-support perfect-MCC artifacts.
    df_eval = _load_eval_best_rule_rows(stats_dir, min_dataset_coverage=min_dataset_coverage, score_scope=score_scope)
    if not df_eval.empty and 'MCC' in df_eval.columns:
        vals = pd.to_numeric(df_eval['MCC'], errors='coerce').to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out['n_neurons_with_rules'] = int(vals.size)
            out['rule_mcc_median'] = float(np.median(vals))
            out['rule_mcc_mean'] = float(np.mean(vals))
            thr = out['hq_threshold'] if np.isfinite(out['hq_threshold']) else 0.70
            out['frac_neurons_hq'] = float(np.mean(vals >= float(thr)))
            out['min_rule_dataset_coverage'] = float(min_dataset_coverage)

    loc_dir = get_localization_dir(model_root, run_name, baseline_subset='positive_baseline')
    if loc_dir is not None and (loc_dir / 'neuron_buckets.json').exists():
        nb_path = loc_dir / 'neuron_buckets.json'
        tau_set = neuron_set_from_buckets(nb_path, tau)
        sels = selectivity_values_from_buckets(nb_path, tau)

        out['n_tau_agonists'] = len(tau_set)
        out['median_abs_acc_gap'] = float(np.nanmedian(sels)) if sels.size else np.nan
        out['frac_eps_selective'] = float(np.mean(sels >= eps)) if sels.size else np.nan

        rk_path = loc_dir / 'rule_knockout.json'
        if rk_path.exists():
            rk = read_json(rk_path)
            if isinstance(rk, list) and rk:
                groups = [r.get('n_ablation_groups', np.nan) for r in rk if r.get('status') == 'ok']
                groups = [g for g in groups if isinstance(g, (int, float)) and np.isfinite(g)]
                out['median_n_ablation_groups'] = float(np.median(groups)) if groups else np.nan
                out['mean_n_ablation_groups'] = float(np.mean(groups)) if groups else np.nan
                out['n_circuits'] = len(groups)

    return out


def build_model_coverage(df: pd.DataFrame, requested_runs: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['model', 'n_runs_found', 'n_runs_requested', 'missing_runs', 'is_complete'])

    requested = set(requested_runs)
    rows = []
    for model, sub in df.groupby('model'):
        found = set(sub['run_name'].unique())
        missing = sorted(requested - found)
        rows.append({
            'model': model,
            'n_runs_found': len(found),
            'n_runs_requested': len(requested),
            'missing_runs': ';'.join(missing),
            'is_complete': len(missing) == 0,
        })
    return pd.DataFrame(rows).sort_values('model').reset_index(drop=True)


def filter_complete_models(df: pd.DataFrame, coverage: pd.DataFrame, drop_incomplete: bool) -> pd.DataFrame:
    if not drop_incomplete or df.empty:
        return df.copy()
    keep_models = set(coverage.loc[coverage['is_complete'], 'model'])
    return df[df['model'].isin(keep_models)].copy()


def save_figure(fig, out_path_stem: Path):
    fig.tight_layout()
    fig.savefig(out_path_stem.with_suffix('.pdf'))
    plt.close(fig)


def metric_run_order(df: pd.DataFrame, metric: str, candidate_runs: list[str]) -> list[str]:
    present = []
    for run_name in candidate_runs:
        sub = df[df['condition_key'] == run_name]
        vals = pd.to_numeric(sub[metric], errors='coerce').to_numpy(dtype=float)
        if np.isfinite(vals).any():
            present.append(run_name)
    return present


def plot_cross_model_trends(df: pd.DataFrame, out_dir: Path, metric: str, ylabel: str, run_order: list[str], run_to_label: dict[str, str]):
    dfx = df.copy()
    dfx = dfx[dfx['condition_key'].isin(run_order)]
    if dfx.empty:
        return

    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(run_order)), 4.0))
    x = np.arange(len(run_order))

    plotted_any = False
    for model in sorted(dfx['model'].unique()):
        y = []
        for run_name in run_order:
            row = dfx[(dfx['model'] == model) & (dfx['condition_key'] == run_name)]
            y.append(float(row[metric].iloc[0]) if len(row) else np.nan)
        y = np.asarray(y, dtype=float)
        if np.isfinite(y).any():
            ax.plot(x, y, marker='o', linewidth=1.5, alpha=0.85, label=model)
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    counts = dfx.groupby('condition_key')['model'].nunique().to_dict()
    labels = [f"{run_to_label[r]}\n(n={counts.get(r, 0)})" for r in run_order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(fig, out_dir / f'cross_model_{metric}')


def plot_aggregate_metric(df: pd.DataFrame, out_dir: Path, metric: str, ylabel: str, run_order: list[str], run_to_label: dict[str, str]):
    if df.empty:
        return
    sub = df[df['condition_key'].isin(run_order)].copy()
    if sub.empty:
        return

    grp = sub.groupby(['condition_key', 'condition'])[metric].agg(['mean', 'std', 'count']).reset_index()
    grp = grp.set_index('condition_key').reindex(run_order).reset_index()
    vals = pd.to_numeric(grp['mean'], errors='coerce').to_numpy(dtype=float)
    errs = pd.to_numeric(grp['std'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return

    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(run_order)), 4.0))
    x = np.arange(len(run_order))
    ax.bar(x, vals, yerr=errs, capsize=4, alpha=0.75)
    labels = []
    for _, row in grp.iterrows():
        rk = row['condition_key']
        labels.append(f"{run_to_label[rk]}\n(n={int(row['count']) if pd.notna(row['count']) else 0})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)
    save_figure(fig, out_dir / f'agg_{metric}')


def plot_ecdf_overlay(values_by_model: dict[str, np.ndarray], out_dir: Path, fname: str, xlabel: str):
    if not values_by_model:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    plotted_any = False
    for model, vals in sorted(values_by_model.items()):
        x, y = ecdf_xy(vals)
        if x is None:
            continue
        ax.plot(x, y, linewidth=1.6, label=model)
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel('ECDF')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, out_dir / fname)


CANONICAL_RUN_ORDER = [
    'rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor',
    'rule_split-spectral_sample-decode_only-agonist_neurons-fast-random_anchor',
    'rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor-fake_targets',
    'spectral_split-decode_only-agonist_neurons-fast-random_anchor',
    'rule_split-spectral_sample-decode_only-agonist_neurons-spectral_anchor',
    'rule_split-spectral_sample-agonist_neurons-fast-spectral_anchor',
    'rule_split-spectral_sample-agonist_neurons-fast-random_anchor',
    'rule_split-spectral_sample-agonist_neurons-fast-spectral_anchor-fake_targets',
    'spectral_split-agonist_neurons-fast-random_anchor',
    'rule_split-spectral_sample-agonist_neurons-spectral_anchor',
]


def order_runs(run_names: list[str]) -> list[str]:
    rank = {rn: i for i, rn in enumerate(CANONICAL_RUN_ORDER)}
    return sorted(run_names, key=lambda rn: (rank.get(rn, 10**9), rn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task_dir', type=str, required=True, help='Path to a task directory (e.g., .../data/arithmetic).')
    ap.add_argument('--out_dir', type=str, default='results_cross_models', help='Output directory.')
    ap.add_argument('--tau', type=float, default=0.2, help='Agonist threshold for max_effect.')
    ap.add_argument('--eps', type=float, default=0.2, help='Selectivity threshold for |accuracy_gap|.')
    ap.add_argument('--min_rule_dataset_coverage', type=float, default=DEFAULT_MIN_RULE_DATASET_COVERAGE,
                    help='Minimum held-out dataset_coverage required before a rule contributes to per-neuron MCC distributions/HQ summaries. Use 0 to disable.')
    ap.add_argument('--score_scope', type=str, default='test_selected',
                    help='Rule-combo score scope for cross-model rule metrics: test_selected (default, HQ-T), all_fit (HQ-F), or test (legacy frozen train-combo TEST).')
    ap.add_argument('--runs', type=str, default='all', help="Comma-separated run names (or 'all' to auto-discover from stats folders).")
    ap.add_argument('--drop_incomplete_models', action='store_true', help='Drop models missing one or more requested runs.')
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    model_roots = list_model_roots(task_dir)
    if not model_roots:
        raise RuntimeError(f'No model roots found under: {task_dir}')

    discovered_runs = discover_available_runs(model_roots)
    if args.runs == 'all':
        run_names = discovered_runs
    else:
        requested = [r.strip() for r in args.runs.split(',') if r.strip()]
        missing = [r for r in requested if r not in discovered_runs]
        if missing:
            print(f'Warning: requested runs not found in stats folders: {missing}')
        run_names = [r for r in requested if r in discovered_runs]

    if not run_names:
        raise RuntimeError('No runs found to summarize.')

    run_names = order_runs(run_names)
    run_to_label = make_unique_display_labels(run_names)

    rows = []
    for model_root in model_roots:
        for run_name in run_names:
            row = summarize_run(model_root, run_name, tau=args.tau, eps=args.eps, display_label=run_to_label[run_name], min_dataset_coverage=args.min_rule_dataset_coverage, score_scope=args.score_scope)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No run summaries could be built from the discovered runs.')

    coverage = build_model_coverage(df, run_names)
    coverage.to_csv(out_dir / 'model_run_coverage.csv', index=False)

    df = filter_complete_models(df, coverage, args.drop_incomplete_models)
    if df.empty:
        raise RuntimeError('All models were filtered out. Try without --drop_incomplete_models.')

    df.to_csv(out_dir / 'run_summary_by_model.csv', index=False)

    agg = (
        df.groupby(['condition_key', 'condition'])
          .agg(
              n_models=('model', 'nunique'),
              mean_tau=('n_tau_agonists', 'mean'),
              std_tau=('n_tau_agonists', 'std'),
              mean_mcc=('rule_mcc_median', 'mean'),
              std_mcc=('rule_mcc_median', 'std'),
              mean_cost=('median_n_ablation_groups', 'mean'),
              std_cost=('median_n_ablation_groups', 'std'),
              mean_sel=('median_abs_acc_gap', 'mean'),
              std_sel=('median_abs_acc_gap', 'std'),
              mean_frac_eps=('frac_eps_selective', 'mean'),
              std_frac_eps=('frac_eps_selective', 'std'),
              mean_frac_hq=('frac_neurons_hq', 'mean'),
              std_frac_hq=('frac_neurons_hq', 'std'),
          )
          .reset_index()
    )
    agg['condition_rank'] = agg['condition_key'].map({rn: i for i, rn in enumerate(run_names)})
    agg = agg.sort_values(['condition_rank', 'condition_key']).drop(columns=['condition_rank'])
    agg.to_csv(out_dir / 'agg_summary_by_run.csv', index=False)

    tau_runs = metric_run_order(df, 'n_tau_agonists', run_names)
    mcc_runs = metric_run_order(df, 'rule_mcc_median', run_names)
    cost_runs = metric_run_order(df, 'median_n_ablation_groups', run_names)
    sel_runs = metric_run_order(df, 'median_abs_acc_gap', run_names)
    frac_eps_runs = metric_run_order(df, 'frac_eps_selective', run_names)

    plot_cross_model_trends(df, plot_dir, 'n_tau_agonists', '# singleton tau-agonists', tau_runs, run_to_label)
    plot_cross_model_trends(df, plot_dir, 'rule_mcc_median', 'Median anchored-rule MCC', mcc_runs, run_to_label)
    plot_cross_model_trends(df, plot_dir, 'median_n_ablation_groups', 'Median # ablation groups', cost_runs, run_to_label)

    plot_aggregate_metric(df, plot_dir, 'n_tau_agonists', '# singleton tau-agonists', tau_runs, run_to_label)
    plot_aggregate_metric(df, plot_dir, 'rule_mcc_median', 'Median anchored-rule MCC', mcc_runs, run_to_label)
    plot_aggregate_metric(df, plot_dir, 'median_abs_acc_gap', 'Median |accuracy_gap|', sel_runs, run_to_label)
    plot_aggregate_metric(df, plot_dir, 'frac_eps_selective', 'Fraction eps-selective', frac_eps_runs, run_to_label)

    keep_models = set(df['model'].unique())
    filtered_model_roots = [mr for mr in model_roots if mr.name in keep_models]

    for run_name in run_names:
        values_mcc = {}
        values_flip = {}
        values_sel = {}
        for model_root in filtered_model_roots:
            stats_dir = model_root / 'rule_extraction_results' / 'neuron_flip_rules' / 'stats' / run_name
            if not stats_dir.exists():
                continue

            dfm = _load_eval_best_rule_rows(stats_dir, min_dataset_coverage=args.min_rule_dataset_coverage, score_scope=args.score_scope)
            if not dfm.empty:
                vals = pd.to_numeric(dfm.get('MCC'), errors='coerce').to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    values_mcc[model_root.name] = vals

            flip_path = stats_dir / 'flip_stats_by_neuron.csv'
            if flip_path.exists():
                dff = pd.read_csv(flip_path)
                vals = pd.to_numeric(dff.get('flip_any_rate'), errors='coerce').to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    values_flip[model_root.name] = vals

            loc_dir = get_localization_dir(model_root, run_name, baseline_subset='positive_baseline')
            if loc_dir is not None and (loc_dir / 'neuron_buckets.json').exists():
                vals = selectivity_values_from_buckets(loc_dir / 'neuron_buckets.json', args.tau)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    values_sel[model_root.name] = vals

        safe = re.sub(r'[^a-zA-Z0-9]+', '_', run_name).strip('_')
        plot_ecdf_overlay(values_mcc, plot_dir, f'ecdf_mcc_{safe}', 'Per-neuron best anchored-rule MCC')
        plot_ecdf_overlay(values_flip, plot_dir, f'ecdf_flipany_{safe}', 'Per-neuron flip-any rate')
        plot_ecdf_overlay(values_sel, plot_dir, f'ecdf_selectivity_{safe}', 'Per-neuron |accuracy_gap|')

    print(f'Wrote: {out_dir}')


if __name__ == '__main__':
    main()
