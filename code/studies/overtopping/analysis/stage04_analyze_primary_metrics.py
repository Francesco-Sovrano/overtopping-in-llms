#!/usr/bin/env python3
"""Held-out primary-metric analysis utilities.

``primary`` summarizes reach, concentration, overlap, directional coverage,
and simultaneous-intervention quantities already present in a primary table.
``holdout`` audits paired strict-test outputs. ``critical-report`` combines
those model-free summaries. Matched null inference for simultaneous E(J) and
Simultaneous E(J) and paired conditional marginal validation are provided by ``pipeline/stage08_validate_interactions.py``.
"""

from __future__ import annotations
from pathlib import Path


import argparse
import json
import math

import numpy as np
import pandas as pd
from studies.overtopping.analysis.layer_widths import layer_width_for_model, per_1000_layer_coordinates, per_layer_fraction
from studies.overtopping.analysis.lib.directional_metrics import DIRECTIONAL_MIN_ELIGIBLE_N, direction_for_metric
from scipy import stats


CORE_METRICS = ("U", "Top", "TOC1")
DIRECTIONAL_METRICS = (
    "U_J_i2c", "U_J_c2i",
    "s_1_i2c", "s_1_c2i",
    "N05_i2c_per_1k_layer", "N05_c2i_per_1k_layer",
    "N10_i2c_per_1k_layer", "N10_c2i_per_1k_layer",
    "N_eff_i2c_per_1k_layer", "N_eff_c2i_per_1k_layer",
)
OPTIONAL_METRICS = (
    "OverlapCompression",
    "SingletonMass",
    "Dom1",
    "Dom1_matched",
)
STRENGTH_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)
# Direction-conditioned rates become unstable when the source-state cohort is tiny.
# The causal workflow targets 32 held-out source-state examples; use that as the
# manuscript adequacy threshold rather than treating 1/1 and 500/500 estimates alike.


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _bootstrap_corr(x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    if n_boot <= 0 or len(x) < 3:
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    vals = []
    n = len(x)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) > 0 and np.std(yb) > 0:
            vals.append(float(np.corrcoef(xb, yb)[0, 1]))
    if not vals:
        return math.nan, math.nan
    lo, hi = np.quantile(np.asarray(vals), [0.025, 0.975])
    return float(lo), float(hi)


def _correlation_row(
    frame: pd.DataFrame,
    *,
    scope_type: str,
    scope_value: str,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict | None:
    z = frame[["score", metric]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(z) < 3 or z["score"].nunique() < 2 or z[metric].nunique() < 2:
        return None
    pearson = stats.pearsonr(z["score"], z[metric])
    spearman = stats.spearmanr(z["score"], z[metric])
    lo, hi = _bootstrap_corr(
        z["score"].to_numpy(dtype=float),
        z[metric].to_numpy(dtype=float),
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "scope_type": scope_type,
        "scope_value": scope_value,
        "metric": metric,
        "n": int(len(z)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "pearson_bootstrap_ci_low": lo,
        "pearson_bootstrap_ci_high": hi,
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _residualize(values: pd.Series, controls: pd.DataFrame) -> tuple[np.ndarray, int]:
    design = pd.get_dummies(controls.astype(str), drop_first=True, dtype=float)
    matrix = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    rank = int(np.linalg.matrix_rank(matrix))
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    residual = y - matrix @ np.linalg.lstsq(matrix, y, rcond=None)[0]
    return residual, rank


def _partial_correlation(
    frame: pd.DataFrame,
    metric: str,
    *,
    controls: tuple[str, ...],
    label: str,
) -> dict | None:
    z = frame[["score", metric, *controls]].dropna().copy()
    if len(z) < 8:
        return None
    rx, rank = _residualize(z["score"], z[list(controls)])
    ry, _ = _residualize(z[metric], z[list(controls)])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    r = float(np.corrcoef(rx, ry)[0, 1])
    dfree = int(len(z) - rank - 1)
    if dfree <= 0 or abs(r) >= 1:
        p = 0.0 if abs(r) >= 1 else math.nan
    else:
        t_value = r * math.sqrt(dfree / max(1e-15, 1.0 - r * r))
        p = float(2.0 * stats.t.sf(abs(t_value), df=dfree))
    return {
        "scope_type": "partial",
        "scope_value": label,
        "metric": metric,
        "n": int(len(z)),
        "pearson_r": r,
        "pearson_p": p,
        "degrees_of_freedom": dfree,
        "control_design_rank": rank,
    }


def _adequate_directional_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return settings for which a directional metric is mathematically defined.

    The across-setting regression sample is the number of settings, not the
    within-setting source-state denominator.  Directional denominators are kept
    for uncertainty/CI reporting but do not impose an arbitrary n>=32 gate.
    """
    direction = direction_for_metric(metric)
    if direction is None:
        return frame
    ncol = f"U_J_{direction}_n"
    if ncol not in frame.columns:
        return frame
    n = pd.to_numeric(frame[ncol], errors="coerce")
    return frame.loc[n.fillna(0) > 0].copy()


def _infer_model_family(model: object) -> str:
    text = str(model).lower()
    if "qwen2.5" in text:
        return "Qwen2.5"
    if "qwen2" in text:
        return "Qwen2"
    if "pythia" in text:
        return "Pythia"
    return str(model).split("-", 1)[0] or "unknown"


def _infer_replacement_baseline(stats_dir: object) -> str:
    text = str(stats_dir).lower()
    if "eval_mean-donor" in text:
        return "mean-donor"
    if "eval_mean-positional" in text:
        return "mean-positional"
    if "eval_mean" in text:
        return "mean"
    if "eval_zero" in text:
        return "zero"
    # The pipeline omits a suffix for plain mean replacement.
    return "mean"


def _resolve_stats_dir(raw: object, data_root: Path | None) -> Path:
    path = Path(str(raw)).expanduser()
    if path.exists() or data_root is None:
        return path
    text = path.as_posix()
    for marker in ("/results/", "/data/"):
        if marker in text:
            suffix = text.split(marker, 1)[1]
            candidate = data_root / suffix
            if candidate.exists():
                return candidate
    return path


def _clopper_pearson(k: int, n: int, *, level: float = 0.95) -> tuple[float, float]:
    """Two-sided exact binomial interval, including the k=0 and k=n cases."""
    if n <= 0:
        return math.nan, math.nan
    k = max(0, min(int(k), int(n)))
    alpha = 1.0 - float(level)
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def _distribution_summary(values: pd.Series, *, name: str) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {"metric": name, "n": 0}
    return {
        "metric": name,
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "q25": float(x.quantile(0.25)),
        "q75": float(x.quantile(0.75)),
        "min": float(x.min()),
        "max": float(x.max()),
        "counts": {
            f"ge_{threshold:g}": int((x >= threshold).sum())
            for threshold in (0.5, 0.75, 0.9, 0.95)
        },
    }


def _directional_rate_record(
    payload: dict,
    *,
    count_key: str,
    rate_key: str,
    n_key: str,
) -> tuple[int, int, float, float, float, str]:
    """Read a direction-conditioned union rate.

    Directional rates require the baseline-eligible denominator.  Older exports
    stored only the total evaluated-row denominator, which cannot be repaired
    from aggregate counts alone.  Refuse that fallback rather than silently
    reporting an unconditional joint proportion.
    """
    count = int(payload.get(count_key, 0) or 0)
    n_raw = payload.get(n_key)
    if n_raw is None or pd.isna(n_raw):
        return count, 0, math.nan, math.nan, math.nan, "missing_eligible_denominator"
    n = int(n_raw)
    if n < 0 or count < 0 or count > n:
        return count, n, math.nan, math.nan, math.nan, "invalid_count_or_denominator"
    rate = (count / n) if n > 0 else math.nan
    stored_rate = float(payload.get(rate_key, math.nan))
    if n > 0 and np.isfinite(stored_rate) and not np.isclose(rate, stored_rate, atol=1e-12, rtol=1e-9):
        return count, n, math.nan, math.nan, math.nan, "stored_rate_mismatch"
    low, high = _clopper_pearson(count, n)
    return count, n, rate, low, high, "stored_eligible_denominator"


def _add_directional_coverage(
    frame: pd.DataFrame,
    *,
    data_root: Path | None,
    require_complete: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach positive->negative and negative->positive union coverage to every row."""
    if "stats_dir" not in frame.columns:
        if require_complete:
            raise ValueError("Directional coverage requires a stats_dir column.")
        return frame.copy(), ["primary table has no stats_dir column"]

    out = frame.copy()
    records: list[dict] = []
    missing: list[str] = []
    for index, row in out.iterrows():
        stats_dir = _resolve_stats_dir(row["stats_dir"], data_root)
        path = stats_dir / "flip_stats_global.json"
        if not path.exists():
            missing.append(str(path))
            records.append({"row_index": int(index)})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        p2n = _directional_rate_record(
            payload,
            count_key="union_c2i_unique_count",
            rate_key="union_c2i_unique_rate",
            n_key="n_evaluated_c2i_rows",
        )
        n2p = _directional_rate_record(
            payload,
            count_key="union_i2c_unique_count",
            rate_key="union_i2c_unique_rate",
            n_key="n_evaluated_i2c_rows",
        )
        if p2n[5] != "stored_eligible_denominator" or n2p[5] != "stored_eligible_denominator":
            missing.append(
                f"{path}: positive_to_negative={p2n[5]}, negative_to_positive={n2p[5]}"
            )
        pooled_n = int(payload.get("n_evaluated_rows", 0) or 0)
        pooled_count = int(payload.get("union_flip_any_unique_count", 0) or 0)
        if p2n[1] + n2p[1] != pooled_n or p2n[0] + n2p[0] != pooled_count:
            missing.append(
                f"{path}: directional decomposition does not match pooled totals "
                f"(n={p2n[1]}+{n2p[1]} vs {pooled_n}; "
                f"count={p2n[0]}+{n2p[0]} vs {pooled_count})"
            )

        pooled_rate = (pooled_count / pooled_n) if pooled_n > 0 else math.nan
        stored_pooled_rate = float(payload.get("union_flip_any_unique_rate", math.nan))
        if pooled_n > 0 and np.isfinite(stored_pooled_rate) and not np.isclose(
            pooled_rate, stored_pooled_rate, atol=1e-12, rtol=1e-9
        ):
            missing.append(
                f"{path}: stored pooled rate {stored_pooled_rate} does not equal "
                f"count/denominator {pooled_count}/{pooled_n}={pooled_rate}"
            )

        authoritative_u = pd.to_numeric(pd.Series([row.get("U", math.nan)]), errors="coerce").iloc[0]
        if pooled_n > 0 and np.isfinite(authoritative_u):
            # Primary-table values may be rounded for presentation. Accept at most
            # half a four-decimal unit plus one count of sampling granularity.
            tolerance = max(5e-5, 1.0 / pooled_n)
            if not np.isclose(pooled_rate, float(authoritative_u), atol=tolerance, rtol=0.0):
                missing.append(
                    f"{path}: directional pooled rate {pooled_rate:.12g} does not "
                    f"match authoritative primary-table U={float(authoritative_u):.12g} "
                    f"within tolerance {tolerance:.3g}"
                )
        records.append(
            {
                "row_index": int(index),
                "U_J_c2i_count": p2n[0],
                "U_J_c2i_n": p2n[1],
                "U_J_c2i": p2n[2],
                "U_J_c2i_ci_low": p2n[3],
                "U_J_c2i_ci_high": p2n[4],
                "U_J_c2i_denominator_source": p2n[5],
                "U_J_i2c_count": n2p[0],
                "U_J_i2c_n": n2p[1],
                "U_J_i2c": n2p[2],
                "U_J_i2c_ci_low": n2p[3],
                "U_J_i2c_ci_high": n2p[4],
                "U_J_i2c_denominator_source": n2p[5],
                "U_J_i2c_adequate": bool(n2p[1] >= DIRECTIONAL_MIN_ELIGIBLE_N),
                "U_J_c2i_adequate": bool(p2n[1] >= DIRECTIONAL_MIN_ELIGIBLE_N),
                "U_J_i2c_minus_c2i": n2p[2] - p2n[2]
                if np.isfinite(p2n[2]) and np.isfinite(n2p[2])
                else math.nan,
                "dominant_direction": (
                    "c2i"
                    if np.isfinite(p2n[2]) and np.isfinite(n2p[2]) and p2n[2] > n2p[2]
                    else "i2c"
                    if np.isfinite(p2n[2]) and np.isfinite(n2p[2]) and n2p[2] > p2n[2]
                    else "tie_or_undefined"
                ),
            }
        )
    directional = pd.DataFrame(records).set_index("row_index")
    for column in directional.columns:
        out[column] = directional.reindex(out.index)[column].to_numpy()
    if require_complete and missing:
        raise ValueError(
            "Directional coverage is incomplete or internally inconsistent: "
            + "; ".join(missing[:10])
        )
    return out, missing



def _add_directional_singleton_metrics(
    frame: pd.DataFrame,
    *,
    data_root: Path | None,
    require_complete: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach v3 direction-specific singleton/support metrics and layer normalization."""
    out = frame.copy()
    records: list[dict] = []
    missing: list[str] = []
    for index, row in out.iterrows():
        stats_dir = _resolve_stats_dir(row.get("stats_dir"), data_root)
        path = stats_dir / "singleton_set_metrics.json"
        record: dict = {"row_index": int(index)}
        if not path.exists():
            missing.append(str(path))
            records.append(record)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("definition_version") != "heldout-set-metrics-v3-directional":
            missing.append(f"{path}: directional singleton schema not rebuilt")
            records.append(record)
            continue
        for direction in ("i2c", "c2i"):
            record[f"s_1_{direction}"] = payload.get(f"s_1_{direction}")
            record[f"N_eff_{direction}"] = payload.get(f"N_eff_{direction}")
            record[f"R_ov_{direction}"] = payload.get(f"R_ov_{direction}")
            thresholds = payload.get(f"N_t_{direction}") or {}
            for raw_t, short in (("0.05", "N05"), ("0.1", "N10")):
                count = thresholds.get(raw_t)
                record[f"{short}_{direction}"] = count
        width = layer_width_for_model(row.get("model"))
        record["layer_width"] = width
        for direction in ("i2c", "c2i"):
            record[f"N05_{direction}_density"] = per_layer_fraction(
                record.get(f"N05_{direction}"), row.get("model")
            )
            record[f"N10_{direction}_density"] = per_layer_fraction(
                record.get(f"N10_{direction}"), row.get("model")
            )
            record[f"N05_{direction}_per_1k_layer"] = per_1000_layer_coordinates(
                record.get(f"N05_{direction}"), row.get("model")
            )
            record[f"N10_{direction}_per_1k_layer"] = per_1000_layer_coordinates(
                record.get(f"N10_{direction}"), row.get("model")
            )
            record[f"N_eff_{direction}_per_1k_layer"] = per_1000_layer_coordinates(
                record.get(f"N_eff_{direction}"), row.get("model")
            )
        records.append(record)
    extra = pd.DataFrame(records).set_index("row_index") if records else pd.DataFrame()
    for column in extra.columns:
        out[column] = extra.reindex(out.index)[column].to_numpy()
    if require_complete and missing:
        raise ValueError(
            "Directional singleton metrics are incomplete. Run rebuild_directional_stats first: "
            + "; ".join(missing[:10])
        )
    return out, missing


def _add_overlap_compression(
    frame: pd.DataFrame,
    *,
    data_root: Path | None,
    require_complete: bool,
) -> pd.DataFrame:
    """Add U(J)/sum_j delta({j}) and its denominator, singleton effect mass."""
    have_ratio = "OverlapCompression" in frame.columns and frame["OverlapCompression"].notna().all()
    have_mass = "SingletonMass" in frame.columns and frame["SingletonMass"].notna().all()
    if have_ratio and have_mass:
        return frame
    if "stats_dir" not in frame.columns:
        if require_complete:
            raise ValueError("Overlap compression requires a stats_dir column.")
        return frame

    out = frame.copy()
    values: list[float] = []
    masses: list[float] = []
    missing: list[str] = []
    for _, row in out.iterrows():
        stats_dir = _resolve_stats_dir(row["stats_dir"], data_root)
        path = stats_dir / "flip_stats_by_neuron.csv"
        if not path.exists():
            values.append(math.nan)
            masses.append(math.nan)
            missing.append(str(path))
            continue
        neurons = pd.read_csv(path)
        if "flip_any_rate" not in neurons.columns:
            values.append(math.nan)
            masses.append(math.nan)
            missing.append(f"{path} (missing flip_any_rate)")
            continue
        strengths = pd.to_numeric(neurons["flip_any_rate"], errors="coerce").dropna()
        denominator = float(strengths.sum())
        union = float(pd.to_numeric(pd.Series([row.get("U")]), errors="coerce").iloc[0])
        values.append(union / denominator if denominator > 0 and np.isfinite(union) else math.nan)
        masses.append(denominator)
    if not have_ratio:
        out["OverlapCompression"] = values
    if not have_mass:
        out["SingletonMass"] = masses
    if require_complete and missing:
        raise FileNotFoundError(
            "Could not compute overlap compression for all settings. Missing: "
            + "; ".join(missing[:10])
        )
    return out


def run_primary(args) -> None:
    table = Path(args.primary_table).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(table)
    data_root = Path(args.data_root).expanduser() if args.data_root else None
    df = _add_overlap_compression(
        df,
        data_root=data_root,
        require_complete=bool(args.require_overlap_compression),
    )
    df, directional_missing = _add_directional_coverage(
        df,
        data_root=data_root,
        require_complete=bool(args.require_directional_coverage),
    )
    df, directional_singleton_missing = _add_directional_singleton_metrics(
        df,
        data_root=data_root,
        require_complete=bool(args.require_directional_coverage),
    )
    df.to_csv(out_dir / "primary_table_augmented.csv", index=False)
    required = {"score", "task", "phase", *CORE_METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")
    metrics = [metric for metric in (*CORE_METRICS, *DIRECTIONAL_METRICS, *OPTIONAL_METRICS) if metric in df.columns]
    if "model" in df.columns:
        df["model_family"] = df["model"].map(_infer_model_family)
    if "stats_dir" in df.columns:
        df["replacement_baseline"] = df["stats_dir"].map(_infer_replacement_baseline)

    rows = []
    seed_offset = 0
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", df)]
    scopes.extend(("phase", str(name), group) for name, group in df.groupby("phase", sort=True))
    scopes.extend(("task", str(name), group) for name, group in df.groupby("task", sort=True))
    for scope_type, scope_value, group in scopes:
        for metric in metrics:
            analysis_group = _adequate_directional_frame(group, metric)
            row = _correlation_row(
                analysis_group,
                scope_type=scope_type,
                scope_value=scope_value,
                metric=metric,
                n_boot=int(args.bootstrap),
                seed=int(args.seed) + seed_offset,
            )
            seed_offset += 1
            if row is not None:
                row["directional_min_eligible_n"] = (
                    DIRECTIONAL_MIN_ELIGIBLE_N if direction_for_metric(metric) else math.nan
                )
                rows.append(row)
    for metric in metrics:
        analysis_df = _adequate_directional_frame(df, metric)
        row = _partial_correlation(
            analysis_df,
            metric,
            controls=("task", "phase"),
            label="controls=task+phase",
        )
        if row is not None:
            row["directional_min_eligible_n"] = (
                DIRECTIONAL_MIN_ELIGIBLE_N if direction_for_metric(metric) else math.nan
            )
            rows.append(row)
    full_controls = ("task", "phase", "model_family", "replacement_baseline")
    if set(full_controls).issubset(df.columns):
        for metric in metrics:
            analysis_df = _adequate_directional_frame(df, metric)
            row = _partial_correlation(
                analysis_df,
                metric,
                controls=full_controls,
                label="controls=task+phase+model_family+replacement_baseline",
            )
            if row is not None:
                row["directional_min_eligible_n"] = (
                    DIRECTIONAL_MIN_ELIGIBLE_N if direction_for_metric(metric) else math.nan
                )
                rows.append(row)
    corr = pd.DataFrame(rows)
    corr.to_csv(out_dir / "primary_correlations.csv", index=False)

    toc = pd.to_numeric(df["TOC1"], errors="coerce").dropna()
    concentration = {"TOC1": _distribution_summary(df["TOC1"], name="TOC1")}
    for metric in ("Dom1_matched", "Dom1"):
        if metric in df.columns:
            concentration[metric] = _distribution_summary(df[metric], name=metric)

    same_context_metric = "Dom1_matched" if "Dom1_matched" in concentration else (
        "Dom1" if "Dom1" in concentration else None
    )
    same_context_summary = concentration.get(same_context_metric, {}) if same_context_metric else {}
    same_context_ceiling_warning = None
    if same_context_summary.get("n", 0):
        median = float(same_context_summary["median"])
        count_half = int(same_context_summary["counts"].get("ge_0.5", 0))
        n_same = int(same_context_summary["n"])
        if median >= 0.9 or count_half >= max(1, int(math.ceil(0.8 * n_same))):
            same_context_ceiling_warning = (
                "The same-context dominance ratio is tightly clustered around or above 1. "
                "A nonsignificant competence association is therefore inconclusive: "
                "the observed distribution provides limited variation for detecting either "
                "an increasing or a decreasing relationship."
            )

    directional_columns = [
        "task", "model", "phase", "score", "U",
        "U_J_i2c_count", "U_J_i2c_n", "U_J_i2c", "U_J_i2c_ci_low", "U_J_i2c_ci_high", "U_J_i2c_adequate",
        "U_J_c2i_count", "U_J_c2i_n", "U_J_c2i", "U_J_c2i_ci_low", "U_J_c2i_ci_high", "U_J_c2i_adequate",
        "U_J_i2c_minus_c2i", "dominant_direction",
        "s_1_i2c", "s_1_c2i", "N05_i2c", "N05_c2i", "N10_i2c", "N10_c2i",
        "N_eff_i2c", "N_eff_c2i", "layer_width",
        "N05_i2c_per_1k_layer", "N05_c2i_per_1k_layer",
        "N10_i2c_per_1k_layer", "N10_c2i_per_1k_layer",
        "N_eff_i2c_per_1k_layer", "N_eff_c2i_per_1k_layer",
    ]
    directional_columns = [column for column in directional_columns if column in df.columns]
    directional_table = df[directional_columns].copy()
    directional_table.to_csv(out_dir / "directional_overtopping_all_settings.csv", index=False)

    directional_lines = [
        "# Direction-specific overtopping",
        "",
        "U_J_i2c is 0->1 (incorrect->correct for correctness tasks); U_J_c2i is 1->0.",
        "Strong-handle densities are counts per 1000 d_model coordinates of one transformer layer.",
        f"Direction-conditioned competence fits use every setting with a nonzero source-state denominator; the within-setting denominator is reported as uncertainty metadata rather than used as an across-setting exclusion rule.",
        "",
        "| Task | Model | Phase | U 0->1 (95% CI) | U 1->0 (95% CI) | N.05 0->1 /1k | N.05 1->0 /1k |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, drow in directional_table.iterrows():
        def fmt_direction(prefix: str) -> str:
            rate = drow.get(prefix, math.nan)
            low = drow.get(f"{prefix}_ci_low", math.nan)
            high = drow.get(f"{prefix}_ci_high", math.nan)
            n = drow.get(f"{prefix}_n", math.nan)
            k = drow.get(f"{prefix}_count", math.nan)
            if not np.isfinite(float(rate)) or not np.isfinite(float(n)) or int(n) <= 0:
                return "NA"
            return f"{float(rate):.4f} [{float(low):.4f}, {float(high):.4f}] ({int(k)}/{int(n)})"

        def fmt_density(key: str) -> str:
            value = pd.to_numeric(pd.Series([drow.get(key)]), errors="coerce").iloc[0]
            return f"{float(value):.2f}" if pd.notna(value) else "NA"

        directional_lines.append(
            f"| {drow.get('task', '')} | {drow.get('model', '')} | {drow.get('phase', '')} | "
            f"{fmt_direction('U_J_i2c')} | {fmt_direction('U_J_c2i')} | "
            f"{fmt_density('N05_i2c_per_1k_layer')} | {fmt_density('N05_c2i_per_1k_layer')} |"
        )
    (out_dir / "directional_overtopping_all_settings.md").write_text(
        "\n".join(directional_lines) + "\n", encoding="utf-8"
    )

    payload = {
        "source": str(table),
        "n_primary_settings": int(len(df)),
        "concentration_distribution": concentration,
        "same_context_metric": same_context_metric,
        "same_context_ceiling_warning": same_context_ceiling_warning,
        "directional_coverage_missing": directional_missing,
        "directional_singleton_missing": directional_singleton_missing,
        "correlations": rows,
    }
    _write_json(out_dir / "primary_metrics.json", payload)

    overall = corr[corr["scope_type"] == "overall"].set_index("metric")
    phase = corr[corr["scope_type"] == "phase"]
    partial = corr[corr["scope_type"] == "partial"]
    lines = [
        "# Held-out primary metric analysis",
        "",
        f"Primary settings: {len(df)}. Defined TOC1 settings: {len(toc)}.",
        "",
        (
            f"Median TOC1 is {concentration['TOC1']['median']:.3f}."
            if concentration["TOC1"].get("n", 0)
            else "TOC1 is unavailable for all supplied settings because an exact discovery-frozen ranking/event sidecar was not present."
        ),
        "",
        "| Metric | Pearson r | p | 95% bootstrap CI |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        if metric not in overall.index:
            continue
        row = overall.loc[metric]
        lines.append(
            f"| {metric} | {row['pearson_r']:.3f} | {row['pearson_p']:.3g} | "
            f"[{row['pearson_bootstrap_ci_low']:.3f}, {row['pearson_bootstrap_ci_high']:.3f}] |"
        )
    if "U_J_i2c" in overall.index or "U_J_c2i" in overall.index:
        lines.extend(["", "## Direction-first competence result", ""])
        for metric, label in (("U_J_i2c", "0->1 union reach"), ("U_J_c2i", "1->0 union reach")):
            if metric in overall.index:
                row = overall.loc[metric]
                lines.append(
                    f"- {label}: pooled Pearson r={row['pearson_r']:.3f} "
                    f"(p={row['pearson_p']:.3g})."
                )
        partial_tp = partial[partial["scope_value"].eq("controls=task+phase")]
        for metric, label in (("U_J_i2c", "0->1 union reach"), ("U_J_c2i", "1->0 union reach"),
                              ("N05_i2c_per_1k_layer", "0->1 strong-handle density"),
                              ("N05_c2i_per_1k_layer", "1->0 strong-handle density")):
            hit = partial_tp[partial_tp["metric"].eq(metric)]
            if not hit.empty:
                row = hit.iloc[0]
                lines.append(
                    f"- {label}, adjusted for task+phase: r={row['pearson_r']:.3f} "
                    f"(p={row['pearson_p']:.3g})."
                )
        lines.extend([
            "",
            "Pooled U(J) is retained as a descriptive union, but direction-specific U_J is the primary "
            "competence analysis because pooling can hide asymmetric 0->1 and 1->0 behavior. "
            "Cross-model handle-count comparisons use counts per 1000 d_model coordinates rather than raw N_t.",
        ])

    lines.extend(
        [
            "",
            "Interpretation: pooled U and Top remain descriptive absolute quantities. Direction-specific U_J and layer-normalized directional N_t are preferred for competence claims. TOC1 is "
            "U(H_1)/U(J) with H_1 frozen on discovery data; it is not the held-out-reselected "
            "Top/U(J) ratio and is not simultaneous-set dominance. "
            "OverlapCompression is U(J)/sum_j delta({j}); SingletonMass is its denominator. "
            "Dom1 is the submitted max-over-context simultaneous-group ratio; Dom1_matched "
            "keeps numerator and denominator in the same full-set-maximizing slice/direction.",
            "",
            "## Same-context dominance distribution",
            "",
        ]
    )
    if same_context_summary.get("n", 0):
        lines.extend(
            [
                (
                    f"{same_context_metric} is defined in {same_context_summary['n']} settings, "
                    f"has median {same_context_summary['median']:.3f} and exceeds 0.5 in "
                    f"{same_context_summary['counts']['ge_0.5']}/{same_context_summary['n']}."
                ),
                "",
                same_context_ceiling_warning or (
                    "The observed spread does not trigger the ceiling-compression warning."
                ),
                "",
            ]
        )
    else:
        lines.extend(["Same-context dominance was not present in the supplied table.", ""])

    overlap_overall = overall.loc["OverlapCompression"] if "OverlapCompression" in overall.index else None
    overlap_partial = partial[
        (partial["metric"] == "OverlapCompression")
        & partial["scope_value"].str.contains("model_family", na=False)
    ]
    mass_overall = overall.loc["SingletonMass"] if "SingletonMass" in overall.index else None
    if overlap_overall is not None:
        lines.extend(
            [
                "## Interpreting the overlap-compression trend",
                "",
                (
                    f"The unadjusted association for U(J)/sum_j delta(j) is "
                    f"r={overlap_overall['pearson_r']:.3f} (p={overlap_overall['pearson_p']:.3g}). "
                    "A negative coefficient means that summed singleton effect mass grows faster "
                    "than union reach, which is compatible with increasing overlap or redundancy "
                    "among channel effects; it is not itself evidence that direct group dominance falls."
                ),
            ]
        )
        if mass_overall is not None:
            lines.append(
                f"The denominator, singleton effect mass, has competence association "
                f"r={mass_overall['pearson_r']:.3f} (p={mass_overall['pearson_p']:.3g})."
            )
        if not overlap_partial.empty:
            prow = overlap_partial.iloc[0]
            lines.append(
                f"After additive adjustment the overlap-compression association is "
                f"r={prow['pearson_r']:.3f} (p={prow['pearson_p']:.3g}), so the unadjusted trend "
                "should be presented as descriptive and potentially compositional/confounded."
            )
        lines.append("")

    lines.extend(
        [
            "## Phase-stratified Pearson correlations",
            "",
            "| Phase | Metric | n | r | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in phase.to_dict("records"):
        lines.append(
            f"| {row['scope_value']} | {row['metric']} | {int(row['n'])} | "
            f"{row['pearson_r']:.3f} | {row['pearson_p']:.3g} |"
        )
    if not partial.empty:
        lines.extend(
            [
                "",
                "## Fixed-effect robustness checks",
                "",
                "| Controls | Metric | n | partial r | p |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in partial.to_dict("records"):
            lines.append(
                f"| {row['scope_value'].removeprefix('controls=')} | {row['metric']} | "
                f"{int(row['n'])} | {row['pearson_r']:.3f} | {row['pearson_p']:.3g} |"
            )
    (out_dir / "primary_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_label_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=PATH, got {spec!r}")
    label, raw = spec.split("=", 1)
    if not label.strip() or not raw.strip():
        raise ValueError(f"Expected nonempty LABEL=PATH, got {spec!r}")
    return label.strip(), Path(raw).expanduser()


def _unit_key_frame(neurons: pd.DataFrame) -> pd.Series:
    layer_col = "layer_label" if "layer_label" in neurons.columns else "layer_key"
    if layer_col not in neurons.columns or "neuron_id" not in neurons.columns:
        raise ValueError(
            "flip_stats_by_neuron.csv must contain neuron_id and layer_label or layer_key"
        )
    return (
        neurons[layer_col].astype(str)
        + ":"
        + pd.to_numeric(neurons["neuron_id"], errors="raise").astype(int).astype(str)
    )


def _exact_frozen_toc1(stats_dir: Path) -> tuple[float, str]:
    """Read discovery-frozen TOC_1 only; never substitute held-out Top/U.

    Exact TOC_1 requires the frozen discovery ranking and per-example singleton
    event sidecar produced by the current stage-7 metric schema.
    """
    topm_path = stats_dir / "frozen_topm_metrics.csv"
    if topm_path.exists():
        frame = pd.read_csv(topm_path)
        if "m" in frame.columns and "TOC_m" in frame.columns:
            m = pd.to_numeric(frame["m"], errors="coerce")
            match = frame.loc[m == 1]
            if len(match):
                value = pd.to_numeric(match.iloc[0]["TOC_m"], errors="coerce")
                status = str(match.iloc[0].get("TOC_m_status", "ok"))
                return (float(value) if pd.notna(value) else math.nan, status)

    singleton_path = stats_dir / "singleton_set_metrics.json"
    if singleton_path.exists():
        payload = json.loads(singleton_path.read_text(encoding="utf-8"))
        if payload.get("definition_version") in {"heldout-set-metrics-v2", "heldout-set-metrics-v3-directional"}:
            toc = payload.get("TOC_m", {})
            entry = toc.get("1", {}) if isinstance(toc, dict) else {}
            if isinstance(entry, dict):
                value = entry.get("value")
                status = str(entry.get("status", "ok"))
                try:
                    return float(value), status
                except (TypeError, ValueError):
                    return math.nan, status

    return math.nan, "unavailable_requires_discovery_frozen_ranking_and_singleton_events"


def _summarize_stats_dir(label: str, stats_dir: Path) -> dict:
    global_path = stats_dir / "flip_stats_global.json"
    neuron_path = stats_dir / "flip_stats_by_neuron.csv"
    if not global_path.exists() or not neuron_path.exists():
        raise FileNotFoundError(f"{stats_dir} must contain flip_stats_global.json and flip_stats_by_neuron.csv")
    global_stats = json.loads(global_path.read_text(encoding="utf-8"))
    neurons = pd.read_csv(neuron_path)
    unit_key_series = _unit_key_frame(neurons)
    rate_series = pd.to_numeric(neurons.get("flip_any_rate"), errors="coerce")
    count_series = pd.to_numeric(neurons.get("flip_any_count"), errors="coerce")
    n_series = pd.to_numeric(neurons.get("n_eval"), errors="coerce")
    strengths = rate_series.dropna()
    unit_keys = sorted(set(unit_key_series.tolist()))
    unit_rates = {
        str(key): float(rate)
        for key, rate in zip(unit_key_series.tolist(), rate_series.tolist())
        if pd.notna(rate)
    }
    unit_counts = {
        str(key): int(count)
        for key, count in zip(unit_key_series.tolist(), count_series.tolist())
        if pd.notna(count)
    }
    unit_ns = {
        str(key): int(n_value)
        for key, n_value in zip(unit_key_series.tolist(), n_series.tolist())
        if pd.notna(n_value)
    }
    union_count = int(global_stats.get("union_flip_any_unique_count", 0) or 0)
    union_n = int(global_stats.get("n_evaluated_rows", 0) or 0)
    union = float(global_stats.get("union_flip_any_unique_rate", math.nan))
    union_ci_low, union_ci_high = _clopper_pearson(union_count, union_n)
    top = float(strengths.max()) if len(strengths) else math.nan
    top_key = None
    top_count = 0
    top_n = 0
    if unit_rates:
        top_key = sorted(unit_rates, key=lambda key: (-unit_rates[key], key))[0]
        top_count = int(unit_counts.get(top_key, 0))
        top_n = int(unit_ns.get(top_key, 0))
    top_ci_low, top_ci_high = _clopper_pearson(top_count, top_n)
    scope_path = stats_dir / "evaluation_scope.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8")) if scope_path.exists() else {}
    toc1, toc1_status = _exact_frozen_toc1(stats_dir)
    reselected_share = (
        top / union
        if np.isfinite(top) and np.isfinite(union) and union > 0
        else math.nan
    )
    return {
        "label": label,
        "stats_dir": str(stats_dir),
        "n_eval": int(global_stats.get("n_evaluated_rows", 0) or 0),
        "n_channels": int(len(strengths)),
        "U": union,
        "U_count": union_count,
        "U_n": union_n,
        "U_ci_low": union_ci_low,
        "U_ci_high": union_ci_high,
        "Top": top,
        "Top_reselected_key": top_key,
        "Top_count": top_count,
        "Top_n": top_n,
        "Top_ci_low": top_ci_low,
        "Top_ci_high": top_ci_high,
        "TOC1": toc1,
        "TOC1_status": toc1_status,
        "ReselectedSingletonShare": reselected_share,
        **{
            f"N_{threshold:.2f}": int((strengths >= threshold).sum())
            for threshold in STRENGTH_THRESHOLDS
        },
        "holdout_test_only": scope.get("holdout_test_only"),
        "candidate_discovery_split": scope.get("candidate_discovery_split"),
        "final_statistics_split": scope.get("final_statistics_split"),
        "mean_replacement_reference_split": scope.get("mean_replacement_reference_split"),
        "_unit_keys": unit_keys,
        "_unit_rates": unit_rates,
        "_unit_counts": unit_counts,
        "_unit_ns": unit_ns,
    }


def _safe_ratio(numerator: object, denominator: object) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return math.nan
    return num / den


def _load_dom1(path: Path) -> dict:
    if path.is_dir():
        path = path / "dominance_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Direct-Dom summary not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("summary", [])
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path)
    if frame.empty or "m" not in frame.columns or "Dom" not in frame.columns:
        raise ValueError(f"{path} does not contain m and Dom columns")
    row = frame.loc[pd.to_numeric(frame["m"], errors="coerce") == 1]
    if row.empty:
        raise ValueError(f"{path} has no m=1 row")
    record = row.iloc[0]
    return {
        "Dom1": float(record["Dom"]) if pd.notna(record["Dom"]) else math.nan,
        "Dom1_matched": (
            float(record["Dom_direction_matched"])
            if pd.notna(record.get("Dom_direction_matched"))
            else math.nan
        ),
        "Dom1_status": record.get("status"),
        "Dom1_matched_status": record.get("matched_status"),
        "Dom1_matched_direction": record.get("matched_argmax_direction"),
        "Dom1_numerator": (
            float(record["numerator_max_subset_effect"])
            if pd.notna(record.get("numerator_max_subset_effect"))
            else math.nan
        ),
        "Dom1_denominator": (
            float(record["denominator_full_set_effect"])
            if pd.notna(record.get("denominator_full_set_effect"))
            else math.nan
        ),
        "Dom1_source": str(path),
    }


def _strict_holdout_provenance(row: dict) -> list[str]:
    failures = []
    if row.get("holdout_test_only") is not True:
        failures.append("holdout_test_only is not true")
    if str(row.get("final_statistics_split")).lower() != "test":
        failures.append("final_statistics_split is not test")
    if str(row.get("mean_replacement_reference_split")).lower() != "train":
        failures.append("mean_replacement_reference_split is not train")
    discovery = row.get("candidate_discovery_split")
    if discovery is not None and str(discovery).lower() != "train":
        failures.append("candidate_discovery_split is not train")
    return failures


def _paired_holdout_row(
    test: dict,
    reference: dict | None,
    *,
    allow_candidate_mismatch: bool,
) -> dict:
    private_keys = {"_unit_keys", "_unit_rates", "_unit_counts", "_unit_ns"}
    row = {k: v for k, v in test.items() if k not in private_keys}
    row["estimate"] = "heldout_test"
    if reference is None:
        row["has_reference"] = False
        return row

    test_keys = set(test.get("_unit_keys", []))
    reference_keys = set(reference.get("_unit_keys", []))
    missing_on_test = sorted(reference_keys - test_keys)
    extra_on_test = sorted(test_keys - reference_keys)
    exact_match = not missing_on_test and not extra_on_test
    if not exact_match and not allow_candidate_mismatch:
        raise ValueError(
            f"{test['label']}: held-out candidate set differs from the reported set. "
            f"Missing on test: {missing_on_test[:5]}; extra on test: {extra_on_test[:5]}. "
            "Direct retention requires a frozen identical J."
        )

    reference_rates = reference.get("_unit_rates", {})
    reference_top_key = (
        sorted(reference_rates, key=lambda key: (-reference_rates[key], key))[0]
        if reference_rates else None
    )
    reference_top_value = (
        float(reference_rates[reference_top_key]) if reference_top_key is not None else math.nan
    )
    test_top_value = float(test.get("_unit_rates", {}).get(reference_top_key, math.nan))
    test_top_count = int(test.get("_unit_counts", {}).get(reference_top_key, 0))
    test_top_n = int(test.get("_unit_ns", {}).get(reference_top_key, 0))
    frozen_low, frozen_high = _clopper_pearson(test_top_count, test_top_n)

    row.update(
        {
            "has_reference": True,
            "reference_stats_dir": reference["stats_dir"],
            "candidate_set_exact_match": exact_match,
            "candidate_units_missing_on_test": len(missing_on_test),
            "candidate_units_extra_on_test": len(extra_on_test),
            "n_channels_reference": reference["n_channels"],
            "Top_frozen_key": reference_top_key,
            "Top_frozen_reference": reference_top_value,
            "Top_frozen_heldout": test_top_value,
            "Top_frozen_count": test_top_count,
            "Top_frozen_n": test_top_n,
            "Top_frozen_ci_low": frozen_low,
            "Top_frozen_ci_high": frozen_high,
            "Top_frozen_retention": _safe_ratio(test_top_value, reference_top_value),
            "Top_frozen_delta": (
                test_top_value - reference_top_value
                if np.isfinite(test_top_value) and np.isfinite(reference_top_value)
                else math.nan
            ),
        }
    )
    for metric in CORE_METRICS:
        test_value = row[metric]
        reference_value = reference[metric]
        row[f"{metric}_reference"] = reference_value
        row[f"{metric}_heldout"] = test_value
        row[f"{metric}_retention"] = _safe_ratio(test_value, reference_value)
        row[f"{metric}_delta"] = (
            float(test_value) - float(reference_value)
            if np.isfinite(float(test_value)) and np.isfinite(float(reference_value))
            else math.nan
        )
    for threshold in STRENGTH_THRESHOLDS:
        key = f"N_{threshold:.2f}"
        reference_count = int(reference[key])
        test_count = int(test[key])
        row[f"{key}_reference"] = reference_count
        row[f"{key}_heldout"] = test_count
        row[f"{key}_delta"] = test_count - reference_count
        row[f"{key}_retention"] = _safe_ratio(test_count, reference_count)
    return row


def run_holdout(args) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = {
        label: _summarize_stats_dir(label, path)
        for label, path in map(_parse_label_path, args.run)
    }
    references = {
        label: _summarize_stats_dir(label, path)
        for label, path in map(_parse_label_path, args.reference or [])
    }
    unknown_references = sorted(set(references) - set(tests))
    if unknown_references:
        raise ValueError(f"References have no corresponding --run labels: {unknown_references}")

    if not args.no_strict_provenance:
        provenance_failures = {
            label: _strict_holdout_provenance(row)
            for label, row in tests.items()
        }
        provenance_failures = {
            label: failures for label, failures in provenance_failures.items() if failures
        }
        if provenance_failures:
            raise ValueError(
                "Strict held-out provenance check failed: "
                + json.dumps(provenance_failures, sort_keys=True)
            )

    rows = [
        _paired_holdout_row(
            tests[label],
            references.get(label),
            allow_candidate_mismatch=bool(args.allow_candidate_mismatch),
        )
        for label in tests
    ]
    dom_by_label = {
        label: _load_dom1(path)
        for label, path in map(_parse_label_path, args.dom or [])
    }
    unknown_dom = sorted(set(dom_by_label) - set(tests))
    if unknown_dom:
        raise ValueError(f"Dom summaries have no corresponding --run labels: {unknown_dom}")
    for row in rows:
        row.update(dom_by_label.get(row["label"], {}))

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "heldout_reestimation_summary.csv", index=False)

    paired = [row for row in rows if row.get("has_reference")]
    aggregate_counts = {}
    for threshold in STRENGTH_THRESHOLDS:
        key = f"N_{threshold:.2f}"
        reference_total = int(sum(row[f"{key}_reference"] for row in paired))
        heldout_total = int(sum(row[f"{key}_heldout"] for row in paired))
        aggregate_counts[key] = {
            "reference_total": reference_total,
            "heldout_total": heldout_total,
            "delta": heldout_total - reference_total,
            "retention": _safe_ratio(heldout_total, reference_total),
        }
    payload = {
        "runs": rows,
        "n_paired_settings": len(paired),
        "paired_subset_threshold_counts": aggregate_counts,
        "interpretation": {
            "U": "absolute singleton-union reach",
            "Top_frozen": (
                "held-out effect of the deterministically preselected strongest singleton "
                "from the reported/training-side run; this is the strict post-selection check"
            ),
            "Top": (
                "maximum held-out singleton effect after re-ranking within the frozen J; "
                "descriptive because it reselects the winner on test"
            ),
            "TOC1": (
                "discovery-frozen TOC_1=U(H_1)/U(J); unavailable rather than "
                "substituting the held-out reselected Top/U(J) ratio"
            ),
            "ReselectedSingletonShare": (
                "descriptive held-out Top/U(J) after reselecting the strongest singleton on evaluation data; "
                "not TOC_1"
            ),
            "Dom1": "submitted max-over-context simultaneous-intervention group-dominance ratio",
            "Dom1_matched": "direction-matched simultaneous-intervention ratio in the full-set-maximizing context",
            "competence_warning": (
                "The held-out matrix tests post-selection survival. Competence associations should "
                "be recomputed from the complete primary-table matrix only after every row "
                "has a strict held-out estimate; selected subsets are not substitutes."
            ),
        },
    }
    _write_json(out_dir / "heldout_reestimation_summary.json", payload)

    lines = [
        "# Paired held-out re-estimation",
        "",
        (
            "Discovery and donor estimation use training rows. Every held-out value below "
            "is recomputed on untouched test examples after freezing the same candidate set J."
        ),
        "",
        "| Run | n test | J match | U reported->test (test 95% CI) | Frozen Top reported->test (test 95% CI) | Test-reselected Top | TOC1 reported->test | Dom1 submitted | Dom1 matched |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def paired_metric(metric: str) -> str:
            if not row.get("has_reference"):
                return f"{row[metric]:.4f}"
            return (
                f"{row[f'{metric}_reference']:.4f}->{row[f'{metric}_heldout']:.4f} "
                f"({100.0 * row[f'{metric}_retention']:.1f}%)"
            )

        match = (
            "yes" if row.get("candidate_set_exact_match") is True
            else "NA" if not row.get("has_reference")
            else "no"
        )
        dom = row.get("Dom1", math.nan)
        dom_text = f"{dom:.4f}" if np.isfinite(dom) else "not run"
        matched = row.get("Dom1_matched", math.nan)
        matched_text = f"{matched:.4f}" if np.isfinite(matched) else "not run"
        u_ci = (
            f"[{row.get('U_ci_low', math.nan):.4f}, {row.get('U_ci_high', math.nan):.4f}]"
            if np.isfinite(row.get('U_ci_low', math.nan)) else "NA"
        )
        if row.get("has_reference"):
            frozen_retention = row.get("Top_frozen_retention", math.nan)
            frozen_retention_text = (
                f"{100.0 * frozen_retention:.1f}%" if np.isfinite(frozen_retention) else "NA"
            )
            frozen_ci = (
                f"[{row.get('Top_frozen_ci_low', math.nan):.4f}, "
                f"{row.get('Top_frozen_ci_high', math.nan):.4f}]"
                if np.isfinite(row.get('Top_frozen_ci_low', math.nan)) else "NA"
            )
            frozen_top_text = (
                f"{row.get('Top_frozen_reference', math.nan):.4f}->"
                f"{row.get('Top_frozen_heldout', math.nan):.4f} "
                f"({frozen_retention_text}; {frozen_ci})"
            )
        else:
            frozen_top_text = "NA"
        test_reselected = (
            f"{row['Top']:.4f} [{row.get('Top_ci_low', math.nan):.4f}, "
            f"{row.get('Top_ci_high', math.nan):.4f}]"
        )
        lines.append(
            f"| {row['label']} | {row['n_eval']} | {match} | {paired_metric('U')} {u_ci} | "
            f"{frozen_top_text} | {test_reselected} | {paired_metric('TOC1')} | "
            f"{dom_text} | {matched_text} |"
        )
    if paired:
        lines.extend(
            [
                "",
                "## Singleton-strength distribution across paired primary settings",
                "",
                "| Threshold | Reported count | Held-out count | Retention |",
                "|---:|---:|---:|---:|",
            ]
        )
        for threshold in STRENGTH_THRESHOLDS:
            record = aggregate_counts[f"N_{threshold:.2f}"]
            retention = record["retention"]
            retention_text = f"{100.0 * retention:.1f}%" if np.isfinite(retention) else "NA"
            lines.append(
                f"| str >= {threshold:.2f} | {record['reference_total']} | "
                f"{record['heldout_total']} | {retention_text} |"
            )
    lines.extend(
        [
            "",
            (
                "These paired primary-table cells test whether the reported distribution survives fresh "
                "examples. Frozen Top is the strict singleton validation; test-reselected Top is "
                "reported only descriptively. A held-out competence analysis is appropriate only "
                "when the complete primary matrix has been re-estimated."
            ),
            "",
        ]
    )
    (out_dir / "heldout_reestimation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overall_correlation(payload: dict, metric: str) -> dict | None:
    for row in payload.get("correlations", []):
        if row.get("scope_type") == "overall" and row.get("metric") == metric:
            return row
    return None


def run_critical_report(args) -> None:
    primary_path = Path(args.primary_metrics_json).expanduser()
    holdout_path = Path(args.holdout_summary_json).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))

    association_rows = []
    for metric, quantity in [
        ("U", "absolute reach"),
        ("Top", "absolute strongest-channel effect"),
        ("TOC1", "discovery-frozen top-1 union share U(H_1)/U(J)"),
        ("OverlapCompression", "overlap compression"),
        ("Dom1", "submitted direct group dominance"),
        ("Dom1_matched", "direction-matched direct group dominance"),
    ]:
        row = _overall_correlation(primary, metric)
        if row is None:
            continue
        significant = bool(float(row["pearson_p"]) < 0.05)
        association_rows.append(
            {
                "metric": metric,
                "quantity": quantity,
                "n": int(row["n"]),
                "r": float(row["pearson_r"]),
                "p": float(row["pearson_p"]),
                "significant": significant,
            }
        )

    paired_rows = [row for row in holdout.get("runs", []) if row.get("has_reference")]
    survival_rows = []
    for row in paired_rows:
        survival_rows.append(
            {
                "label": row["label"],
                "n_test": row["n_eval"],
                "candidate_set_exact_match": row.get("candidate_set_exact_match"),
                "U_reference": row.get("U_reference"),
                "U_heldout": row.get("U_heldout"),
                "U_retention": row.get("U_retention"),
                "Top_reference": row.get("Top_reference"),
                "Top_heldout": row.get("Top_heldout"),
                "Top_retention": row.get("Top_retention"),
                "TOC1_reference": row.get("TOC1_reference"),
                "TOC1_heldout": row.get("TOC1_heldout"),
                "TOC1_delta": row.get("TOC1_delta"),
                "Dom1": row.get("Dom1"),
                "Dom1_matched": row.get("Dom1_matched"),
            }
        )

    toc = next((row for row in association_rows if row["metric"] == "TOC1"), None)
    same_context = next(
        (row for row in association_rows if row["metric"] == "Dom1_matched"),
        next((row for row in association_rows if row["metric"] == "Dom1"), None),
    )
    magnitude = [row for row in association_rows if row["metric"] in {"U", "Top"}]
    ceiling_warning = primary.get("same_context_ceiling_warning")
    if same_context is not None and not same_context["significant"] and ceiling_warning:
        competence_conclusion = (
            "Absolute localized causal magnitude is associated with competence, but the "
            "same-context concentration result is inconclusive. The dominance ratio is "
            "ceiling-compressed, so this measurement cannot reliably distinguish an "
            "increasing, decreasing, or absent competence relationship."
        )
    elif toc is not None and not toc["significant"] and all(row["significant"] for row in magnitude):
        competence_conclusion = (
            "Absolute localized causal magnitude is associated with competence; no "
            "association is detected for TOC1 in this table. This is an absence of "
            "detected evidence, not evidence that concentration is constant."
        )
    elif toc is not None and toc["significant"]:
        competence_conclusion = "TOC1 is significantly associated with competence in this table."
    else:
        competence_conclusion = "The supplied analysis is insufficient for a concentration-competence conclusion."

    output = {
        "competence_association": association_rows,
        "competence_conclusion": competence_conclusion,
        "same_context_ceiling_warning": ceiling_warning,
        "heldout_survival": survival_rows,
        "paired_subset_threshold_counts": holdout.get("paired_subset_threshold_counts", {}),
        "scope": (
            "The competence correlations use the all-setting table. Held-out survival should be "
            "reported for every primary-table row before treating survey-wide effect estimates "
            "as independently re-estimated."
        ),
    }
    _write_json(out_dir / "publication_critical_answers.json", output)

    lines = [
        "# Publication-critical quantitative answers",
        "",
        "## What tracks competence?",
        "",
        "| Quantity | Metric | n | r | p | Result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in association_rows:
        result = "significant" if row["significant"] else "not significant"
        lines.append(
            f"| {row['quantity']} | {row['metric']} | {row['n']} | "
            f"{row['r']:.3f} | {row['p']:.3g} | {result} |"
        )
    lines.extend(["", competence_conclusion, "", "## What survives held-out re-estimation?", ""])
    lines.extend(
        [
            "| Setting | n test | U retention | Top retention | TOC1 change | Dom1 submitted | Dom1 matched |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in survival_rows:
        dom = row.get("Dom1")
        dom_text = f"{dom:.3f}" if dom is not None and np.isfinite(float(dom)) else "not run"
        matched = row.get("Dom1_matched")
        matched_text = f"{matched:.3f}" if matched is not None and np.isfinite(float(matched)) else "not run"
        lines.append(
            f"| {row['label']} | {row['n_test']} | {100.0 * row['U_retention']:.1f}% | "
            f"{100.0 * row['Top_retention']:.1f}% | {row['TOC1_delta']:+.3f} | {dom_text} | {matched_text} |"
        )
    lines.extend(
        [
            "",
            (
                "The paired settings use an identical frozen J, train-only donor estimation, "
                "and untouched test examples. They test robustness of the headline effects, "
                "not a held-out competence correlation."
            ),
            "",
        ]
    )
    (out_dir / "publication_critical_answers.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    primary = sub.add_parser("primary")
    primary.add_argument("--primary_table", required=True)
    primary.add_argument("--out_dir", required=True)
    primary.add_argument(
        "--data_root",
        default=None,
        help="Optional local data/results root used to remap stale absolute stats_dir paths.",
    )
    primary.add_argument(
        "--require_overlap_compression",
        action="store_true",
        help="Fail unless U(J)/sum_j delta({j}) can be computed for every setting.",
    )
    primary.add_argument(
        "--require_directional_coverage",
        action="store_true",
        help="Fail unless positive->negative and negative->positive U can be read for every setting.",
    )
    primary.add_argument("--bootstrap", type=int, default=5000)
    primary.add_argument("--seed", type=int, default=20260726)
    primary.set_defaults(func=run_primary)

    holdout = sub.add_parser("holdout")
    holdout.add_argument("--run", action="append", required=True, help="Repeat LABEL=STATS_DIR.")
    holdout.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Repeat LABEL=REPORTED_STATS_DIR to calculate paired retention.",
    )
    holdout.add_argument(
        "--dom",
        action="append",
        default=[],
        help="Repeat LABEL=dominance_summary.csv (or its directory) to attach direct Dom^(1).",
    )
    holdout.add_argument(
        "--allow_candidate_mismatch",
        action="store_true",
        help="Allow descriptive comparison when held-out J differs. Disabled by default.",
    )
    holdout.add_argument(
        "--no_strict_provenance",
        action="store_true",
        help="Disable the train-selection/test-evaluation provenance assertions.",
    )
    holdout.add_argument("--out_dir", required=True)
    holdout.set_defaults(func=run_holdout)

    critical = sub.add_parser("critical-report")
    critical.add_argument("--primary_metrics_json", required=True)
    critical.add_argument("--holdout_summary_json", required=True)
    critical.add_argument("--out_dir", required=True)
    critical.set_defaults(func=run_critical_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
