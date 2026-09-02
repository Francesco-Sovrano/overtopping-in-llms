#!/usr/bin/env python3
"""Nested held-out threshold-shape validation for the exact RQ3 population.

The RQ3 population is manifest-driven rather than discovered by recursively
scanning the data tree.  By default it contains the 28 primary overtopping
settings plus configured paper-supplementary overtopping settings.  Poisoning
runs are a separate experiment family and are never eligible.

For every evaluable candidate/control unit and flip target, feature selection is
nested inside each training fold.  The selected scalar is then evaluated on the
untouched test fold with a constant predictor, an oriented hard threshold,
logistic regression, and isotonic regression.

Population inference is deliberately two-part:
1. threshold *testability* is compared over all model-evaluated units;
2. conditional threshold shape is compared only after matching testable
   candidate/control units on singleton causal strength within run/baseline.
This avoids comparing a broad candidate population to the tiny, selected tail of
controls that happen to generate enough flips for a threshold fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, matthews_corrcoef

from studies.overtopping.analysis.lib.progress import tqdm
from studies.overtopping.analysis.stage07_overtopping_spiking_report import (
    POP_CAND,
    POP_CTRL,
    POP_LABELS,
    _read_exact_primary_table,
    effect_summary,
    expected_rq3_sources,
    holm,
    paper_figure_rc,
    save_pdf_only,
)


TARGETS = ("flip_any", "flip_c2i", "flip_i2c")
MODELS = ("constant", "threshold", "logistic", "isotonic")
ANALYSIS_SCHEMA_VERSION = "threshold-shape-v4-exact-population-nested-matched"


def _mcc_from_confusion(tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray) -> np.ndarray:
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    tn = np.asarray(tn, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _fit_threshold(x: np.ndarray, y: np.ndarray, min_examples: int) -> dict | None:
    """Fit a one-dimensional threshold on training data only.

    Selection maximizes |MCC|.  ``prediction_inverted`` records whether the raw
    predicate is anti-correlated with the event; downstream probability metrics
    must invert that predicate rather than hiding the sign behind |MCC|.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    min_examples = int(min_examples)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if len(y) < 2 * min_examples or positives < min_examples or negatives < min_examples:
        return None

    values = np.unique(np.sort(x))
    if len(values) < 2:
        return None
    if len(values) > 512:
        values = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 513)))
    thresholds = np.unique(np.concatenate([values, (values[:-1] + values[1:]) / 2.0]))

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order].astype(np.int64, copy=False)
    prefix_pos = np.concatenate(([0], np.cumsum(ys, dtype=np.int64)))
    n = len(ys)

    k_ge = np.searchsorted(xs, thresholds, side="left")
    fn_ge = prefix_pos[k_ge]
    tp_ge = positives - fn_ge
    pred_pos_ge = n - k_ge
    fp_ge = pred_pos_ge - tp_ge
    tn_ge = negatives - fp_ge
    mcc_ge = _mcc_from_confusion(tp_ge, fp_ge, tn_ge, fn_ge)

    k_le = np.searchsorted(xs, thresholds, side="right")
    tp_le = prefix_pos[k_le]
    fp_le = k_le - tp_le
    fn_le = positives - tp_le
    tn_le = negatives - fp_le
    mcc_le = _mcc_from_confusion(tp_le, fp_le, tn_le, fn_le)

    scores = np.concatenate([np.abs(mcc_ge), np.abs(mcc_le)])
    if scores.size == 0 or not np.isfinite(scores).any():
        return None
    best_idx = int(np.nanargmax(scores))
    if best_idx < len(thresholds):
        threshold_idx = best_idx
        direction = ">="
        mcc = float(mcc_ge[threshold_idx])
    else:
        threshold_idx = best_idx - len(thresholds)
        direction = "<="
        mcc = float(mcc_le[threshold_idx])
    raw_high = direction == ">="
    inverted = bool(mcc < 0.0)
    event_high = raw_high != inverted
    return {
        "train_abs_mcc": float(abs(mcc)),
        "threshold": float(thresholds[threshold_idx]),
        "direction": direction,
        "train_mcc": mcc,
        "prediction_inverted": inverted,
        "event_direction": ">=" if event_high else "<=",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip")
    src.add_argument("--root")
    p.add_argument("--out", required=True)
    p.add_argument("--paper-figures-dir", default=None)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--population-scope", choices=["primary", "primary+supplementary"], default="primary+supplementary")
    p.add_argument("--evaluation-split", default="test", choices=["test", "train", "all"])
    p.add_argument("--spiking-max-points", type=int, default=10000)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--holdout-fraction", type=float, default=0.25)
    p.add_argument("--min-class", type=int, default=8)
    p.add_argument("--threshold-match-caliper", type=float, default=0.05,
                   help="Maximum absolute singleton flip-rate difference for conditional MCC matching.")
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260829)
    return p.parse_args()


def _stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    if len(y) == 0:
        return math.nan
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = float(len(y))
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        out += float(mask.sum()) / total * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = p >= 0.5
    try:
        mcc = abs(float(matthews_corrcoef(y, pred)))
    except Exception:
        mcc = math.nan
    return {
        "abs_mcc": mcc,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": _ece(y, p),
    }


def _transition_sharpness(x_train: np.ndarray, predict_prob, *, direction: str) -> tuple[float, float]:
    finite = x_train[np.isfinite(x_train)]
    if len(finite) < 4 or np.allclose(finite, finite[0]):
        return math.nan, math.nan
    lo, hi = np.quantile(finite, [0.05, 0.95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return math.nan, math.nan
    grid = np.linspace(lo, hi, 512)
    probs = np.asarray(predict_prob(grid), dtype=float)
    if direction == "<=":
        grid = -grid[::-1]
        probs = probs[::-1]
    if len(probs) > 1 and probs[-1] < probs[0]:
        probs = probs[::-1]
    pmin, pmax = float(np.nanmin(probs)), float(np.nanmax(probs))
    if pmin > 0.25 or pmax < 0.75:
        return 1.0, 0.0
    i25 = int(np.nanargmin(np.abs(probs - 0.25)))
    i75 = int(np.nanargmin(np.abs(probs - 0.75)))
    width = abs(float(grid[i75] - grid[i25])) / max(abs(float(hi - lo)), 1e-12)
    width = float(np.clip(width, 0.0, 1.0))
    return width, float(1.0 - width)


def _stratified_split(y: np.ndarray, rng: np.random.Generator, frac: float, min_class: int) -> tuple[np.ndarray, np.ndarray] | None:
    train: list[int] = []
    test: list[int] = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls).copy()
        if len(idx) < 2 * int(min_class):
            return None
        rng.shuffle(idx)
        n_test = int(round(float(frac) * len(idx)))
        n_test = max(int(min_class), min(len(idx) - int(min_class), n_test))
        test.extend(idx[:n_test].tolist())
        train.extend(idx[n_test:].tolist())
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


_PROXY_EXACT = {
    "activation_value", "abs_activation_value", "gradient_value", "abs_gradient_value",
    "activation_x_gradient", "abs_activation_x_gradient", "wanda_value",
    "baseline_value", "delta_to_baseline", "predicted_margin_shift", "predicted_margin_drop",
    "abs_predicted_margin_drop", "learned_direction_score", "abs_learned_direction_score",
    "layer_percentile_rank", "layer_zscore", "delta_from_layer_mean", "abs_delta_from_layer_mean",
}
_PROXY_STEMS = (
    "activation", "abs_activation", "gradient", "abs_gradient",
    "activation_x_gradient", "abs_activation_x_gradient", "wanda",
)


def _looks_like_proxy_feature(name: str) -> bool:
    if name in _PROXY_EXACT:
        return True
    return any(name in {f"{stem}_percentile_rank", f"{stem}_layer_zscore"} for stem in _PROXY_STEMS)


def _candidate_features(raw: pd.DataFrame, tests: pd.DataFrame, group_keys: dict, target: str) -> list[str]:
    """Return the condition-level proxy family without unit-level test selection.

    ``aggregate_unit_tests.csv`` is used only as a record of which proxy names
    were configured/available in the condition.  We intentionally do not filter
    it by unit_key, nor use any held-out performance value to define the pool.
    If a condition has no unit-test rows, fall back to the explicit proxy column
    whitelist present in the raw rows.
    """
    work = tests
    for key in ("run_id", "baseline_subset", "population"):
        if key in group_keys and key in work.columns:
            work = work.loc[work[key].astype(str) == str(group_keys[key])]
    if "target" in work.columns:
        work = work.loc[work["target"].astype(str) == target]
    requested = [str(v) for v in work.get("feature", pd.Series(dtype=str)).dropna().unique()]
    requested = [f for f in requested if f in raw.columns and pd.api.types.is_numeric_dtype(raw[f])]
    if requested:
        return sorted(set(requested))
    fallback = [c for c in raw.columns if _looks_like_proxy_feature(str(c)) and pd.api.types.is_numeric_dtype(raw[c])]
    return sorted(set(fallback))


def _evaluate_group(raw: pd.DataFrame, tests: pd.DataFrame, *, keys: dict, target: str, args: argparse.Namespace) -> list[dict]:
    if target not in raw.columns:
        return []
    y_full = pd.to_numeric(raw[target], errors="coerce").to_numpy(dtype=float)
    valid_y = np.isfinite(y_full)
    if not valid_y.any():
        return []
    raw = raw.loc[valid_y].reset_index(drop=True)
    y = (y_full[valid_y] > 0.5).astype(int)
    if min(int(y.sum()), int(len(y) - y.sum())) < 2 * int(args.min_class):
        return []
    features = _candidate_features(raw, tests, keys, target)
    if not features:
        return []

    feature_arrays = {
        feature: pd.to_numeric(raw[feature], errors="coerce").to_numpy(dtype=float)
        for feature in features
    }

    rows: list[dict] = []
    rng = np.random.default_rng(_stable_seed(args.seed, *keys.values(), target))
    for repeat in range(int(args.repeats)):
        split = _stratified_split(y, rng, float(args.holdout_fraction), int(args.min_class))
        if split is None:
            continue
        train_idx, test_idx = split
        best = None
        for feature in features:
            x = feature_arrays[feature]
            mask_train = np.isfinite(x[train_idx])
            if mask_train.sum() < 2 * int(args.min_class):
                continue
            fit = _fit_threshold(x[train_idx][mask_train], y[train_idx][mask_train], max(1, int(args.min_class) // 2))
            if fit is None:
                continue
            score = float(fit.get("train_abs_mcc", -1.0))
            candidate = (score, feature, fit, x)
            if best is None or score > best[0] or (math.isclose(score, best[0]) and feature < best[1]):
                best = candidate
        if best is None:
            continue
        _, feature, threshold_fit, x = best
        finite_train = np.isfinite(x[train_idx])
        finite_test = np.isfinite(x[test_idx])
        tr = train_idx[finite_train]
        te = test_idx[finite_test]
        if min(int(y[tr].sum()), int(len(tr) - y[tr].sum()), int(y[te].sum()), int(len(te) - y[te].sum())) < int(args.min_class):
            continue
        xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
        prevalence = float(ytr.mean())
        model_probs: dict[str, tuple[np.ndarray, object, str]] = {
            "constant": (np.full(len(te), prevalence), lambda grid, q=prevalence: np.full(len(grid), q), threshold_fit["event_direction"]),
        }

        threshold = float(threshold_fit["threshold"])
        direction = str(threshold_fit["direction"])
        inverted = bool(threshold_fit.get("prediction_inverted", False))
        raw_test_pred = (xte >= threshold).astype(float) if direction == ">=" else (xte <= threshold).astype(float)
        if inverted:
            raw_test_pred = 1.0 - raw_test_pred

        def threshold_predict(grid, t=threshold, d=direction, inv=inverted):
            pred = ((np.asarray(grid) >= t) if d == ">=" else (np.asarray(grid) <= t)).astype(float)
            return 1.0 - pred if inv else pred

        model_probs["threshold"] = (
            raw_test_pred,
            threshold_predict,
            str(threshold_fit["event_direction"]),
        )
        try:
            logistic = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=int(args.seed))
            logistic.fit(xtr.reshape(-1, 1), ytr)
            logistic_direction = ">=" if float(logistic.coef_.ravel()[0]) >= 0 else "<="
            model_probs["logistic"] = (
                logistic.predict_proba(xte.reshape(-1, 1))[:, 1],
                lambda grid, m=logistic: m.predict_proba(np.asarray(grid).reshape(-1, 1))[:, 1],
                logistic_direction,
            )
        except Exception:
            pass
        try:
            isotonic_increasing = str(threshold_fit["event_direction"]) == ">="
            isotonic = IsotonicRegression(
                increasing=isotonic_increasing,
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0,
            )
            isotonic.fit(xtr, ytr)
            model_probs["isotonic"] = (
                np.asarray(isotonic.predict(xte), dtype=float),
                lambda grid, m=isotonic: np.asarray(m.predict(np.asarray(grid)), dtype=float),
                ">=" if isotonic_increasing else "<=",
            )
        except Exception:
            pass
        for model_name, (prob, predictor, orient) in model_probs.items():
            metrics = _metrics(yte, prob)
            if model_name == "constant":
                width, sharpness = math.nan, 0.0
            elif model_name == "threshold":
                width, sharpness = 0.0, 1.0
            else:
                width, sharpness = _transition_sharpness(xtr, predictor, direction=orient)
            rows.append({
                **keys,
                "target": target,
                "repeat": int(repeat),
                "selected_feature": feature,
                "selection_train_abs_mcc": float(threshold_fit["train_abs_mcc"]),
                "threshold_train_mcc": float(threshold_fit["train_mcc"]),
                "threshold_prediction_inverted": bool(inverted),
                "model": model_name,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "test_prevalence": float(yte.mean()),
                **metrics,
                "transition_width_normalized": width,
                "transition_sharpness": sharpness,
            })
    return rows


def _summarize(repeats: pd.DataFrame) -> pd.DataFrame:
    if repeats.empty:
        return pd.DataFrame()
    group = [c for c in [
        "run_id", "source_scope", "task", "model_name", "setting", "decode_only",
        "baseline_subset", "population", "unit_key", "target", "model"
    ] if c in repeats.columns]
    return repeats.groupby(group, dropna=False).agg(
        n_repeats=("repeat", "nunique"),
        median_abs_mcc=("abs_mcc", "median"),
        mean_abs_mcc=("abs_mcc", "mean"),
        median_brier=("brier", "median"),
        median_log_loss=("log_loss", "median"),
        median_ece=("ece", "median"),
        median_transition_sharpness=("transition_sharpness", "median"),
        selected_feature_mode=("selected_feature", lambda s: s.mode().iloc[0] if not s.mode().empty else str(s.iloc[0])),
        fraction_threshold_prediction_inverted=("threshold_prediction_inverted", "mean"),
    ).reset_index()


def _load_exact_tables(args: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_kind = "zip" if args.zip else "root"
    source_path = Path(args.zip or args.root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    expected = expected_rq3_sources(args)
    fs, audit_fs = _read_exact_primary_table(source_kind, source_path, expected, data_root, "aggregate_flip_stats.csv")
    raw, audit_raw = _read_exact_primary_table(source_kind, source_path, expected, data_root, "aggregate_activation_flip_rows.csv")
    tests, audit_tests = _read_exact_primary_table(source_kind, source_path, expected, data_root, "aggregate_unit_tests.csv")
    audit = pd.DataFrame(audit_fs + audit_raw + audit_tests)

    flip_audit = audit[audit.file.eq("aggregate_flip_stats.csv")].copy()
    raw_audit = audit[audit.file.eq("aggregate_activation_flip_rows.csv")].copy()
    bad_primary_flip = flip_audit[flip_audit.required.astype(bool) & ~flip_audit.complete.astype(bool)]
    bad_primary_raw = raw_audit[raw_audit.required.astype(bool) & ~raw_audit.complete.astype(bool)]
    if not bad_primary_flip.empty or not bad_primary_raw.empty:
        bad = pd.concat([bad_primary_flip, bad_primary_raw], ignore_index=True)
        preview = bad[["run_id", "task", "model", "phase", "file", "exists", "missing_baseline_subsets", "missing_populations"]].head(8).to_dict("records")
        raise RuntimeError(
            "RQ3 nested threshold-shape primary population is incomplete; refusing a silently selected subset. "
            f"First failures: {preview}"
        )

    complete_flip = set(flip_audit.loc[flip_audit.complete.astype(bool), "run_id"].astype(str))
    complete_raw = set(raw_audit.loc[raw_audit.complete.astype(bool), "run_id"].astype(str))
    included = complete_flip & complete_raw
    audit["included_in_threshold_shape"] = audit.run_id.astype(str).isin(included)
    audit.to_csv(out / "threshold_shape_population_audit.csv", index=False)

    for frame in (fs, raw, tests):
        if not frame.empty:
            frame.drop(frame.index[~frame.run_id.astype(str).isin(included)], inplace=True)
            if frame.run_id.astype(str).str.startswith("poisoning__").any():
                raise RuntimeError("Poisoning rows leaked into exact RQ3 threshold-shape population")

    coverage = pd.DataFrame([
        {
            "source_scope": scope,
            "configured_runs": int((flip_audit.source_scope == scope).sum()),
            "flip_complete_runs": int(((flip_audit.source_scope == scope) & flip_audit.complete.astype(bool)).sum()),
            "raw_complete_runs": int(((raw_audit.source_scope == scope) & raw_audit.complete.astype(bool)).sum()),
            "included_runs": int(sum(1 for run in included if str(flip_audit.loc[flip_audit.run_id.astype(str) == run, "source_scope"].iloc[0]) == scope)),
        }
        for scope in ["primary", "supplementary"]
    ])
    coverage.to_csv(out / "threshold_shape_population_coverage.csv", index=False)
    return fs, raw, tests, audit


def _unit_population(fs: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    base = fs.loc[fs.population.isin([POP_CAND, POP_CTRL])].copy()
    keys = [c for c in ["run_id", "source_scope", "task", "model", "setting", "decode_only", "baseline_subset", "population", "unit_key"] if c in base.columns]
    base["causal_strength"] = pd.to_numeric(base.get("flip_any_rate"), errors="coerce")
    base = base.sort_values(keys).drop_duplicates(keys, keep="first")

    threshold = summary.loc[(summary.model == "threshold") & (summary.target == "flip_any")].copy() if not summary.empty else pd.DataFrame()
    if not threshold.empty:
        # ``summary.model`` is the fitted response-model name ("threshold"),
        # whereas ``base.model`` is the LM architecture.  Align the latter to
        # summary.model_name before constructing merge keys; otherwise every
        # merge silently fails (e.g. "Pythia-1B" != "threshold").
        threshold = threshold.drop(columns=["model"], errors="ignore").rename(columns={"model_name": "model"})
        tkeys = [c for c in keys if c in threshold.columns]
        threshold = threshold.rename(columns={"median_abs_mcc": "nested_threshold_mcc"})
        keep = tkeys + [c for c in ["nested_threshold_mcc", "selected_feature_mode", "n_repeats"] if c in threshold.columns]
        threshold = threshold[keep].drop_duplicates(tkeys)
        units = base.merge(threshold, on=tkeys, how="left", validate="one_to_one")
    else:
        units = base.copy()
        units["nested_threshold_mcc"] = np.nan
        units["selected_feature_mode"] = pd.NA
        units["n_repeats"] = 0
    units["nested_threshold_mcc"] = pd.to_numeric(units["nested_threshold_mcc"], errors="coerce")
    units["threshold_testable"] = units["nested_threshold_mcc"].notna()
    units["nested_tecs_observed"] = units["causal_strength"] * units["nested_threshold_mcc"]
    units["nested_tecs_lower_bound"] = units["causal_strength"] * units["nested_threshold_mcc"].fillna(0.0)
    return units


def _condition_population_summary(units: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_id", "source_scope", "baseline_subset", "population"]
    return units.groupby(keys, dropna=False).agg(
        n_units=("unit_key", "nunique"),
        median_causal_strength=("causal_strength", "median"),
        mean_causal_strength=("causal_strength", "mean"),
        threshold_testable_fraction=("threshold_testable", "mean"),
        n_threshold_testable=("threshold_testable", "sum"),
        median_nested_threshold_mcc=("nested_threshold_mcc", "median"),
        median_nested_tecs_observed=("nested_tecs_observed", "median"),
        median_nested_tecs_lower_bound=("nested_tecs_lower_bound", "median"),
    ).reset_index()


def _paired_condition_effect(condition: pd.DataFrame, metric: str, bootstrap: int) -> tuple[pd.DataFrame, dict]:
    g = condition.dropna(subset=[metric]).groupby(["run_id", "baseline_subset", "population"], dropna=False)[metric].median().unstack("population")
    if POP_CAND not in g.columns or POP_CTRL not in g.columns:
        return pd.DataFrame(), effect_summary(pd.DataFrame(), bootstrap)
    g = g.dropna(subset=[POP_CAND, POP_CTRL]).copy()
    g["delta"] = g[POP_CAND] - g[POP_CTRL]
    return g, effect_summary(g, bootstrap)


def _match_threshold_testable_units(units: pd.DataFrame, caliper: float) -> pd.DataFrame:
    """Greedy one-to-one matching on causal strength within run/baseline."""
    rows: list[dict] = []
    testable = units.loc[units.threshold_testable].copy()
    for (run_id, baseline), g in testable.groupby(["run_id", "baseline_subset"], dropna=False):
        cand = g.loc[g.population == POP_CAND].copy()
        ctrl = g.loc[g.population == POP_CTRL].copy()
        if cand.empty or ctrl.empty:
            continue
        candidates = []
        for ci, cr in cand.iterrows():
            cs = float(cr.causal_strength)
            for ri, rr in ctrl.iterrows():
                rs = float(rr.causal_strength)
                if not (np.isfinite(cs) and np.isfinite(rs)):
                    continue
                candidates.append((abs(cs-rs), str(cr.unit_key), str(rr.unit_key), ci, ri))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        used_c: set = set()
        used_r: set = set()
        for gap, _, _, ci, ri in candidates:
            if gap > float(caliper):
                break
            if ci in used_c or ri in used_r:
                continue
            used_c.add(ci); used_r.add(ri)
            cr = cand.loc[ci]; rr = ctrl.loc[ri]
            rows.append({
                "run_id": run_id,
                "source_scope": cr.get("source_scope", "unknown"),
                "baseline_subset": baseline,
                "candidate_unit_key": cr.unit_key,
                "control_unit_key": rr.unit_key,
                "candidate_strength": float(cr.causal_strength),
                "control_strength": float(rr.causal_strength),
                "strength_gap": float(gap),
                "candidate_nested_mcc": float(cr.nested_threshold_mcc),
                "control_nested_mcc": float(rr.nested_threshold_mcc),
                "nested_mcc_delta": float(cr.nested_threshold_mcc - rr.nested_threshold_mcc),
                "candidate_feature": cr.get("selected_feature_mode"),
                "control_feature": rr.get("selected_feature_mode"),
            })
    return pd.DataFrame(rows)


def _matched_condition_effect(matches: pd.DataFrame, bootstrap: int) -> tuple[pd.DataFrame, dict]:
    if matches.empty:
        return pd.DataFrame(), effect_summary(pd.DataFrame(), bootstrap)
    cond = matches.groupby(["run_id", "baseline_subset"], dropna=False).agg(
        candidate=("candidate_nested_mcc", "median"),
        control=("control_nested_mcc", "median"),
        median_strength_gap=("strength_gap", "median"),
        n_unit_pairs=("nested_mcc_delta", "count"),
    )
    med = cond.rename(columns={"candidate": POP_CAND, "control": POP_CTRL})
    med["delta"] = med[POP_CAND] - med[POP_CTRL]
    return med, effect_summary(med, bootstrap)


def _scope_results(units: pd.DataFrame, condition: pd.DataFrame, matches: pd.DataFrame, *, scope: str, args: argparse.Namespace) -> dict:
    if scope == "primary":
        u = units.loc[units.source_scope == "primary"]
        c = condition.loc[condition.source_scope == "primary"]
        m = matches.loc[matches.source_scope == "primary"] if not matches.empty else matches
    else:
        u, c, m = units, condition, matches
    strength_med, strength_eff = _paired_condition_effect(c, "median_causal_strength", args.bootstrap)
    test_med, test_eff = _paired_condition_effect(c, "threshold_testable_fraction", args.bootstrap)
    match_med, match_eff = _matched_condition_effect(m, args.bootstrap)
    tecs_med, tecs_eff = _paired_condition_effect(c, "median_nested_tecs_lower_bound", args.bootstrap)
    adj = holm({
        "causal_strength": strength_eff.get("wilcoxon_p_greater", math.nan),
        "threshold_testability": test_eff.get("wilcoxon_p_greater", math.nan),
        "matched_threshold_mcc": match_eff.get("wilcoxon_p_greater", math.nan),
    })
    return {
        "scope": scope,
        "n_runs": int(u.run_id.nunique()) if not u.empty else 0,
        "n_units": int(len(u)),
        "causal_strength_effect": strength_eff,
        "threshold_testability_effect": test_eff,
        "matched_threshold_mcc_effect": match_eff,
        "nested_tecs_lower_bound_effect": tecs_eff,
        "holm_primary_endpoints": adj,
        "n_strength_conditions": int(len(strength_med)),
        "n_testability_conditions": int(len(test_med)),
        "n_matched_mcc_conditions": int(len(match_med)),
        "n_matched_unit_pairs": int(len(m)),
        "matching_caliper": float(args.threshold_match_caliper),
    }


def _visualization_curve_for_unit(raw_all: pd.DataFrame, tests_all: pd.DataFrame, *, run_id: str, baseline: str,
                                  population: str, unit_key: str, args: argparse.Namespace) -> list[dict]:
    raw = raw_all.loc[
        raw_all.run_id.astype(str).eq(str(run_id))
        & raw_all.baseline_subset.astype(str).eq(str(baseline))
        & raw_all.population.astype(str).eq(str(population))
        & raw_all.unit_key.astype(str).eq(str(unit_key))
    ].copy()
    if raw.empty or "flip_any" not in raw.columns:
        return []
    keys = {"run_id": run_id, "baseline_subset": baseline, "population": population, "unit_key": unit_key}
    features = _candidate_features(raw, tests_all, keys, "flip_any")
    if not features:
        return []
    yv = pd.to_numeric(raw.flip_any, errors="coerce").to_numpy(float)
    valid = np.isfinite(yv)
    raw = raw.loc[valid].reset_index(drop=True)
    y = (yv[valid] > .5).astype(int)
    rng = np.random.default_rng(_stable_seed(args.seed, "response", run_id, baseline, population, unit_key))
    split = _stratified_split(y, rng, float(args.holdout_fraction), int(args.min_class))
    if split is None:
        return []
    tr, te = split
    best = None
    for feature in features:
        x = pd.to_numeric(raw[feature], errors="coerce").to_numpy(float)
        finite = np.isfinite(x[tr])
        fit = _fit_threshold(x[tr][finite], y[tr][finite], max(1, int(args.min_class)//2))
        if fit is None:
            continue
        item = (float(fit["train_abs_mcc"]), feature, fit, x)
        if best is None or item[0] > best[0] or (math.isclose(item[0], best[0]) and feature < best[1]):
            best = item
    if best is None:
        return []
    _, feature, fit, x = best
    finite_tr = np.isfinite(x[tr]); finite_te = np.isfinite(x[te])
    tr = tr[finite_tr]; te = te[finite_te]
    if len(tr) == 0 or len(te) == 0:
        return []
    xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
    event_increasing = str(fit["event_direction"]) == ">="
    oriented_tr = xtr if event_increasing else -xtr
    oriented_te = xte if event_increasing else -xte
    try:
        logit = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=int(args.seed)).fit(oriented_tr.reshape(-1,1), ytr)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=1).fit(oriented_tr, ytr)
    except Exception:
        return []
    test = pd.DataFrame({"score": oriented_te, "y": yte})
    try:
        test["bin"] = pd.qcut(test.score, q=min(10, test.score.nunique()), duplicates="drop")
    except Exception:
        return []
    rows = []
    for bin_index, (_, g) in enumerate(test.groupby("bin", observed=False, sort=True)):
        score = float(g.score.mean())
        rows.append({
            "population": population,
            "run_id": run_id,
            "baseline_subset": baseline,
            "unit_key": unit_key,
            "feature": feature,
            "bin_index": int(bin_index),
            "n": int(len(g)),
            "mean_oriented_score": score,
            "heldout_flip_rate": float(g.y.mean()),
            "logistic_probability": float(logit.predict_proba(np.array([[score]]))[:,1][0]),
            "isotonic_probability": float(iso.predict([score])[0]),
            "selection_rule": "causal-strength-matched pair; feature selected on visualization training fold only",
        })
    return rows


def _representative_response_curves(raw_all: pd.DataFrame, tests_all: pd.DataFrame, matches: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Choose a same-condition pair using causal strength only, never MCC."""
    if matches.empty:
        return pd.DataFrame()
    chosen = matches.sort_values(["strength_gap", "run_id", "baseline_subset", "candidate_unit_key", "control_unit_key"]).iloc[0]
    rows = []
    rows.extend(_visualization_curve_for_unit(
        raw_all, tests_all, run_id=str(chosen.run_id), baseline=str(chosen.baseline_subset),
        population=POP_CAND, unit_key=str(chosen.candidate_unit_key), args=args,
    ))
    rows.extend(_visualization_curve_for_unit(
        raw_all, tests_all, run_id=str(chosen.run_id), baseline=str(chosen.baseline_subset),
        population=POP_CTRL, unit_key=str(chosen.control_unit_key), args=args,
    ))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["matched_candidate_strength"] = float(chosen.candidate_strength)
        frame["matched_control_strength"] = float(chosen.control_strength)
        frame["matched_strength_gap"] = float(chosen.strength_gap)
    return frame


def _plot_response_curves(curves: pd.DataFrame, paper_dir: Path) -> None:
    target = paper_dir / "fig4b_threshold_response_curves.pdf"
    if curves.empty:
        target.unlink(missing_ok=True)
        return
    with paper_figure_rc():
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.65), sharey=True)
        for ax, population in zip(axes, [POP_CAND, POP_CTRL]):
            g = curves.loc[curves.population == population].sort_values("mean_oriented_score")
            if g.empty:
                ax.axis("off"); continue
            ax.plot(g.mean_oriented_score, g.heldout_flip_rate, marker="o", linewidth=1.4, label="Held-out empirical")
            ax.plot(g.mean_oriented_score, g.logistic_probability, linestyle="--", linewidth=1.2, label="Logistic")
            ax.plot(g.mean_oriented_score, g.isotonic_probability, linestyle=":", linewidth=1.4, label="Isotonic")
            unit = str(g.unit_key.iloc[0])
            ax.set_title(f"{POP_LABELS[population]}\n{unit}")
            ax.set_xlabel("Oriented endogenous scalar")
            ax.set_ylim(-.03, 1.03)
            ax.grid(alpha=.25, linewidth=.45)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(r"Held-out $P(F_j=1\mid z_j)$")
        axes[0].legend(frameon=False, fontsize=7.0)
        gap = float(curves.matched_strength_gap.iloc[0]) if "matched_strength_gap" in curves else math.nan
        fig.suptitle(f"Illustrative same-condition causal-strength match (|Δ strength|={gap:.3f}); not an inferential sample", fontsize=8.5)
        fig.subplots_adjust(left=.10, right=.995, bottom=.20, top=.78, wspace=.18)
        save_pdf_only(fig, target); plt.close(fig)


def _condition_weighted_model_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    pop = summary.loc[summary.population.isin([POP_CAND, POP_CTRL]) & (summary.target == "flip_any")].copy()
    metrics = ["median_abs_mcc", "median_brier", "median_log_loss", "median_ece", "median_transition_sharpness"]
    cond = pop.groupby(["run_id", "baseline_subset", "population", "model"], dropna=False)[metrics].median().reset_index()
    return cond.groupby(["population", "model"], dropna=False).agg(
        n_conditions=("run_id", "count"),
        **{m: (m, "median") for m in metrics},
    ).reset_index()


def _plot_model_comparison(summary: pd.DataFrame, paper_dir: Path) -> None:
    target = paper_dir / "fig4b_threshold_shape_model_comparison.pdf"
    agg = _condition_weighted_model_summary(summary)
    if agg.empty:
        target.unlink(missing_ok=True)
        return
    metrics = [
        ("median_abs_mcc", "Held-out |MCC|"),
        ("median_brier", "Brier score"),
        ("median_log_loss", "Log loss"),
        ("median_ece", "Calibration error"),
    ]
    with paper_figure_rc():
        fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.35))
        for ax, (metric, label) in zip(axes, metrics):
            x = np.arange(len(MODELS), dtype=float)
            width = .34
            for offset, population in [(-width/2, POP_CTRL), (width/2, POP_CAND)]:
                vals = []
                for model in MODELS:
                    q = agg[(agg.population == population) & (agg.model == model)][metric]
                    vals.append(float(q.iloc[0]) if len(q) else np.nan)
                ax.bar(x+offset, vals, width=width, label=POP_LABELS[population])
            ax.set_xticks(x, ["Const.", "Thresh.", "Logit", "Isotonic"], rotation=30, ha="right")
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.25, linewidth=.45)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].legend(frameon=False, fontsize=7.0)
        fig.suptitle("Condition-weighted held-out model comparison (nested feature selection)", fontsize=9.0)
        fig.subplots_adjust(left=.07, right=.995, bottom=.27, top=.83, wspace=.48)
        save_pdf_only(fig, target); plt.close(fig)


def _plot_main_summary(condition: pd.DataFrame, matches: pd.DataFrame, robust: dict, paper_dir: Path) -> None:
    target = paper_dir / "fig4a_candidate_control_spiking_cut_summary.pdf"
    strength_med, _ = _paired_condition_effect(condition, "median_causal_strength", 0)
    test_med, _ = _paired_condition_effect(condition, "threshold_testable_fraction", 0)
    match_med, _ = _matched_condition_effect(matches, 0)
    adj = robust.get("holm_primary_endpoints", {})

    def med_pair(frame: pd.DataFrame) -> tuple[float,float]:
        if frame.empty:
            return math.nan, math.nan
        return float(frame[POP_CTRL].median()), float(frame[POP_CAND].median())
    sc, sa = med_pair(strength_med)
    tc, ta = med_pair(test_med)
    mc, ma = med_pair(match_med)
    panels = [
        ("Singleton causal strength", sc, sa, adj.get("causal_strength", math.nan)),
        ("Threshold-testable fraction", tc, ta, adj.get("threshold_testability", math.nan)),
        ("Strength-matched threshold |MCC|", mc, ma, adj.get("matched_threshold_mcc", math.nan)),
    ]
    with paper_figure_rc():
        fig, axes = plt.subplots(1, 3, figsize=(7.7, 2.45), sharey=True)
        for ax, (title, ctrl, cand, p_adj) in zip(axes, panels):
            if np.isfinite(ctrl) and np.isfinite(cand):
                ax.plot([ctrl, cand], [0,1], color="0.55", linewidth=1.1, zorder=1)
                ax.scatter([ctrl],[0], marker="o", facecolor="white", edgecolor="0.25", s=34, zorder=3)
                ax.scatter([cand],[1], marker="o", color="0.25", s=34, zorder=3)
                ax.text(ctrl, -.13, f"{ctrl:.3f}", ha="center", va="top", fontsize=7)
                ax.text(cand, 1.13, f"{cand:.3f}", ha="center", va="bottom", fontsize=7)
                lo=min(0.0,ctrl,cand); hi=max(ctrl,cand); pad=max(.01,.18*(hi-lo if hi>lo else max(abs(hi),.05)))
                ax.set_xlim(lo-.1*pad, hi+pad)
            if ax is axes[0]:
                ax.set_yticks([0,1],["Matched control","Candidate"])
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_title(title)
            ax.grid(axis="x", alpha=.25, linewidth=.45)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.text(.98,.04,f"Holm p={p_adj:.3g}" if np.isfinite(float(p_adj)) else "Holm p=n/a",transform=ax.transAxes,ha="right",va="bottom",fontsize=7.0)
        fig.suptitle("RQ3: causal dominance + threshold-event evidence (primary + supplementary)", fontsize=9.2)
        fig.subplots_adjust(left=.115,right=.995,bottom=.22,top=.78,wspace=.38)
        save_pdf_only(fig,target); plt.close(fig)


def _plot_supplements(units: pd.DataFrame, matches: pd.DataFrame, paper_dir: Path) -> None:
    # Testability fractions by condition: shows the selection process directly.
    target = paper_dir / "fig4s1_threshold_testability_by_condition.pdf"
    cond = _condition_population_summary(units)
    pivot = cond.pivot_table(index=["run_id","baseline_subset"], columns="population", values="threshold_testable_fraction", aggfunc="first").dropna()
    if not pivot.empty and POP_CAND in pivot and POP_CTRL in pivot:
        with paper_figure_rc():
            fig, ax = plt.subplots(figsize=(4.6,2.6))
            for _, r in pivot.iterrows():
                ax.plot([0,1],[r[POP_CTRL],r[POP_CAND]],color="0.75",linewidth=.7,alpha=.7)
            ax.scatter(np.zeros(len(pivot)), pivot[POP_CTRL], facecolor="white", edgecolor="0.25", s=18)
            ax.scatter(np.ones(len(pivot)), pivot[POP_CAND], color="0.25", s=18)
            ax.set_xticks([0,1],["Controls","Candidates"]); ax.set_ylabel("Threshold-testable fraction")
            ax.set_ylim(-.02,1.02); ax.grid(axis="y",alpha=.25,linewidth=.45)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.tight_layout(); save_pdf_only(fig,target); plt.close(fig)
    else:
        target.unlink(missing_ok=True)

    target2 = paper_dir / "fig4s2_strength_matched_thresholdability.pdf"
    if not matches.empty:
        with paper_figure_rc():
            fig, ax = plt.subplots(figsize=(4.6,2.6))
            ax.scatter(matches.control_nested_mcc, matches.candidate_nested_mcc, s=20, alpha=.72)
            lo=min(float(matches.control_nested_mcc.min()),float(matches.candidate_nested_mcc.min()),0.0)
            hi=max(float(matches.control_nested_mcc.max()),float(matches.candidate_nested_mcc.max()),0.0)
            ax.plot([lo,hi],[lo,hi],linestyle="--",linewidth=.9,label="equal thresholdability")
            ax.set_xlabel("Matched control held-out |MCC|"); ax.set_ylabel("Candidate held-out |MCC|")
            ax.grid(alpha=.25,linewidth=.45); ax.legend(frameon=False,fontsize=7)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.tight_layout(); save_pdf_only(fig,target2); plt.close(fig)
    else:
        target2.unlink(missing_ok=True)

    target3 = paper_dir / "fig4s3_nested_tecs_lower_bound_ecdf.pdf"
    with paper_figure_rc():
        fig, ax = plt.subplots(figsize=(4.5,2.5))
        made=False
        for pop in [POP_CAND,POP_CTRL]:
            vals=np.sort(pd.to_numeric(units.loc[units.population==pop,"nested_tecs_lower_bound"],errors="coerce").dropna().to_numpy(float))
            if len(vals):
                ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),linewidth=1.6,label=POP_LABELS[pop]); made=True
        if made:
            ax.set_xlabel("Nested TECS lower bound"); ax.set_ylabel("Empirical CDF")
            ax.grid(axis="y",alpha=.25,linewidth=.45); ax.legend(frameon=False)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.tight_layout(); save_pdf_only(fig,target3)
        plt.close(fig)
        if not made: target3.unlink(missing_ok=True)


def _write_paper_readme(paper_dir: Path, robust: dict, coverage: pd.DataFrame) -> None:
    cov = {str(r.source_scope): r for r in coverage.itertuples(index=False)} if not coverage.empty else {}
    prim = cov.get("primary")
    supp = cov.get("supplementary")
    text = f"""# Figure 4 - RQ3: causal dominance and threshold-event structure

Population is exact and manifest-driven. Poisoning experiments are excluded by construction.

- Primary configured/included: {getattr(prim,'configured_runs',0)}/{getattr(prim,'included_runs',0)}
- Supplementary configured/included: {getattr(supp,'configured_runs',0)}/{getattr(supp,'included_runs',0)}
- Nested feature selection: training fold only; held-out fold is never used to choose the scalar.
- Hard-threshold predictions are oriented by the sign of training MCC before Brier/log-loss/calibration are computed.
- Thresholdability inference is two-part: all-unit testability first, then causal-strength-matched held-out |MCC| among testable units.
- Matching caliper on singleton flip-any rate: {robust.get('matching_caliper', math.nan):.3f}.

`fig4a_candidate_control_spiking_cut_summary.pdf` reports causal strength, testability, and strength-matched thresholdability. `fig4b_threshold_response_curves.pdf` is only an illustrative same-condition strength-matched pair, not a best-case or inferential sample. `fig4b_threshold_shape_model_comparison.pdf` uses condition-weighted medians.
"""
    (paper_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (0 < float(args.holdout_fraction) < 1):
        raise ValueError("--holdout-fraction must be in (0, 1)")
    if float(args.threshold_match_caliper) < 0:
        raise ValueError("--threshold-match-caliper must be non-negative")
    out = Path(args.out).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)

    # Remove known derived outputs before recomputation so an unavailable new
    # analysis cannot leave a stale PDF/CSV from a previous broader population.
    for name in [
        "threshold_shape_model_comparison_repeats.csv", "threshold_shape_model_comparison.csv",
        "threshold_response_curves_heldout.csv", "threshold_shape_population_summary.csv",
        "threshold_shape_unit_population.csv", "threshold_shape_condition_population.csv",
        "threshold_strength_matched_pairs.csv", "threshold_strength_matched_condition.csv",
        "threshold_shape_scope_sensitivity.csv", "threshold_shape_statistical_results.json",
    ]:
        (out / name).unlink(missing_ok=True)

    fs_all, raw_all, tests_all, audit = _load_exact_tables(args, out)
    if raw_all.empty:
        status = {"status":"not_available","reason":"No exact-population aggregate_activation_flip_rows.csv found"}
        (out/"threshold_shape_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(status["reason"]); return

    raw_all = raw_all.loc[raw_all.population.isin([POP_CAND, POP_CTRL])].copy()
    tests_all = tests_all.loc[tests_all.population.isin([POP_CAND, POP_CTRL])].copy() if not tests_all.empty else pd.DataFrame()
    group_cols = [c for c in ["run_id","source_scope","task","model","setting","decode_only","baseline_subset","population","unit_key"] if c in raw_all.columns]
    rows=[]
    grouped=list(raw_all.groupby(group_cols,dropna=False))
    print(f"[threshold-shape] exact units={len(grouped)} targets={len(TARGETS)} repeats={int(args.repeats)} scope={args.population_scope}")
    for key_tuple, raw in tqdm(grouped, desc="Threshold-shape units", unit="unit"):
        if not isinstance(key_tuple, tuple): key_tuple=(key_tuple,)
        keys=dict(zip(group_cols,key_tuple))
        if "model" in keys:
            keys["model_name"] = keys.pop("model")
        for target in TARGETS:
            rows.extend(_evaluate_group(raw, tests_all, keys=keys, target=target, args=args))

    repeats=pd.DataFrame(rows)
    summary=_summarize(repeats)
    units=_unit_population(fs_all, summary)
    condition=_condition_population_summary(units)
    matches=_match_threshold_testable_units(units, float(args.threshold_match_caliper))
    match_condition, _ = _matched_condition_effect(matches, args.bootstrap)

    robust_extended = _scope_results(units, condition, matches, scope="primary+supplementary", args=args)
    robust_primary = _scope_results(units, condition, matches, scope="primary", args=args)
    robust = robust_extended if args.population_scope == "primary+supplementary" else robust_primary

    curves=_representative_response_curves(raw_all, tests_all, matches, args)
    model_population=_condition_weighted_model_summary(summary)
    coverage=pd.read_csv(out/"threshold_shape_population_coverage.csv") if (out/"threshold_shape_population_coverage.csv").exists() else pd.DataFrame()

    repeats.to_csv(out/"threshold_shape_model_comparison_repeats.csv",index=False)
    summary.to_csv(out/"threshold_shape_model_comparison.csv",index=False)
    curves.to_csv(out/"threshold_response_curves_heldout.csv",index=False)
    model_population.to_csv(out/"threshold_shape_population_summary.csv",index=False)
    units.to_csv(out/"threshold_shape_unit_population.csv",index=False)
    condition.to_csv(out/"threshold_shape_condition_population.csv",index=False)
    matches.to_csv(out/"threshold_strength_matched_pairs.csv",index=False)
    match_condition.reset_index().to_csv(out/"threshold_strength_matched_condition.csv",index=False)
    sensitivity=pd.DataFrame([
        {
            "scope": r["scope"], "n_runs":r["n_runs"], "n_units":r["n_units"],
            "n_strength_conditions":r["n_strength_conditions"], "n_testability_conditions":r["n_testability_conditions"],
            "n_matched_mcc_conditions":r["n_matched_mcc_conditions"], "n_matched_unit_pairs":r["n_matched_unit_pairs"],
            "strength_median_delta":r["causal_strength_effect"].get("median_delta",math.nan),
            "strength_p":r["causal_strength_effect"].get("wilcoxon_p_greater",math.nan),
            "testability_median_delta":r["threshold_testability_effect"].get("median_delta",math.nan),
            "testability_p":r["threshold_testability_effect"].get("wilcoxon_p_greater",math.nan),
            "matched_mcc_median_delta":r["matched_threshold_mcc_effect"].get("median_delta",math.nan),
            "matched_mcc_p":r["matched_threshold_mcc_effect"].get("wilcoxon_p_greater",math.nan),
        }
        for r in [robust_primary, robust_extended]
    ])
    sensitivity.to_csv(out/"threshold_shape_scope_sensitivity.csv",index=False)
    stats_payload={
        "analysis_schema_version":ANALYSIS_SCHEMA_VERSION,
        "population_scope":args.population_scope,
        "excluded_poisoning":True,
        "primary_only":robust_primary,
        "primary_plus_supplementary":robust_extended,
        "main":robust,
    }
    (out/"threshold_shape_statistical_results.json").write_text(json.dumps(stats_payload,indent=2,allow_nan=True),encoding="utf-8")

    if args.paper_figures_dir:
        paper_dir=Path(args.paper_figures_dir).expanduser().resolve(); paper_dir.mkdir(parents=True,exist_ok=True)
        _plot_main_summary(condition if args.population_scope=="primary+supplementary" else condition.loc[condition.source_scope=="primary"],
                           matches if args.population_scope=="primary+supplementary" else matches.loc[matches.source_scope=="primary"],
                           robust,paper_dir)
        _plot_model_comparison(summary if args.population_scope=="primary+supplementary" else summary.loc[summary.source_scope=="primary"],paper_dir)
        _plot_response_curves(curves,paper_dir)
        _plot_supplements(units if args.population_scope=="primary+supplementary" else units.loc[units.source_scope=="primary"],
                          matches if args.population_scope=="primary+supplementary" else matches.loc[matches.source_scope=="primary"],paper_dir)
        _write_paper_readme(paper_dir,robust,coverage)

    status={
        "status":"ok" if not summary.empty else "no_eligible_units",
        "analysis_schema_version":ANALYSIS_SCHEMA_VERSION,
        "population_scope":args.population_scope,
        "excluded_poisoning":True,
        "n_repeat_rows":int(len(repeats)),
        "n_unit_model_rows":int(len(summary)),
        "n_complete_units":int(len(units)),
        "n_threshold_testable_units":int(units.threshold_testable.sum()) if not units.empty else 0,
        "n_strength_matched_pairs":int(len(matches)),
        "nested_feature_selection":True,
        "threshold_search":"vectorized_cumulative_confusion",
        "threshold_prediction_orientation":"training_mcc_sign",
        "isotonic_orientation":"training_threshold_event_direction",
        "representative_curve_selection":"same_condition_minimum_causal_strength_gap; no MCC selection",
        "models":list(MODELS),
        "metrics":["abs_mcc","brier","log_loss","ece","transition_sharpness"],
    }
    (out/"threshold_shape_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(json.dumps(status,indent=2))


if __name__ == "__main__":
    main()
