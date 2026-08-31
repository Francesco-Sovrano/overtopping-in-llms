#!/usr/bin/env python3
"""P1.1 direct threshold-shape validation from saved threshold-event rows.

This is CPU-only.  For every candidate/control unit and intervention-defined
flip target, it repeatedly splits examples into train/test folds.  On the train
fold it freezes the best endogenous scalar by *training* threshold MCC, then
compares four one-dimensional models on the untouched test fold:

- constant prevalence baseline;
- one hard threshold;
- logistic regression;
- isotonic/monotonic regression.

Outputs include held-out MCC, Brier score, log loss, calibration error and a
normalized transition-sharpness statistic.  Feature selection is nested inside
each training fold so the held-out metrics are not used to choose the proxy.
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

from studies.overtopping.analysis.stage07_overtopping_spiking_report import (
    POP_CAND,
    POP_CTRL,
    POP_LABELS,
    concat,
    paper_figure_rc,
    save_pdf_only,
)


TARGETS = ("flip_any", "flip_c2i", "flip_i2c")
MODELS = ("constant", "threshold", "logistic", "isotonic")


def _fit_threshold(x: np.ndarray, y: np.ndarray, min_examples: int) -> dict | None:
    """Fit a one-dimensional hard threshold using training data only."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=int)
    mask = np.isfinite(x) & np.isfinite(y)
    x=x[mask]; y=y[mask]
    if len(y) < 2 * int(min_examples) or y.sum() < int(min_examples) or (len(y)-y.sum()) < int(min_examples):
        return None
    values=np.unique(np.sort(x))
    if len(values)<2: return None
    if len(values)>512: values=np.unique(np.quantile(values,np.linspace(0,1,513)))
    thresholds=np.unique(np.concatenate([values,(values[:-1]+values[1:])/2.0]))
    best=None
    for direction in (">=","<="):
        for threshold in thresholds:
            pred=(x>=threshold) if direction==">=" else (x<=threshold)
            try: mcc=float(matthews_corrcoef(y,pred))
            except Exception: continue
            score=abs(mcc)
            candidate=(score,float(threshold),direction,mcc)
            if best is None or candidate[0]>best[0]: best=candidate
    if best is None: return None
    return {"train_abs_mcc":best[0],"threshold":best[1],"direction":best[2],"train_mcc":best[3]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip")
    src.add_argument("--root")
    p.add_argument("--out", required=True)
    p.add_argument("--paper-figures-dir", default=None)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--holdout-fraction", type=float, default=0.25)
    p.add_argument("--min-class", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260829)
    return p.parse_args()


def _stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    if len(y) == 0:
        return math.nan
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = float(len(y)); out = 0.0
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
    else:
        grid = grid.copy()
    # Orient so increasing grid should correspond to increasing event probability.
    if len(probs) > 1 and probs[-1] < probs[0]:
        probs = probs[::-1]
    pmin, pmax = float(np.nanmin(probs)), float(np.nanmax(probs))
    if pmin > 0.25 or pmax < 0.75:
        return 1.0, 0.0  # transition is not actually traversed in the observed range
    i25 = int(np.nanargmin(np.abs(probs - 0.25)))
    i75 = int(np.nanargmin(np.abs(probs - 0.75)))
    width = abs(float(grid[i75] - grid[i25])) / max(abs(float(hi - lo)), 1e-12)
    width = float(np.clip(width, 0.0, 1.0))
    return width, float(1.0 - width)


def _stratified_split(y: np.ndarray, rng: np.random.Generator, frac: float, min_class: int) -> tuple[np.ndarray, np.ndarray] | None:
    train: list[int] = []; test: list[int] = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls).copy()
        if len(idx) < 2 * int(min_class):
            return None
        rng.shuffle(idx)
        n_test = int(round(float(frac) * len(idx)))
        n_test = max(int(min_class), min(len(idx) - int(min_class), n_test))
        test.extend(idx[:n_test].tolist()); train.extend(idx[n_test:].tolist())
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def _candidate_features(raw: pd.DataFrame, tests: pd.DataFrame, group_keys: dict, target: str) -> list[str]:
    work = tests
    for key, value in group_keys.items():
        if key in work.columns:
            work = work.loc[work[key].astype(str) == str(value)]
    if "target" in work.columns:
        work = work.loc[work["target"].astype(str) == target]
    requested = [str(v) for v in work.get("feature", pd.Series(dtype=str)).dropna().unique()]
    return [f for f in requested if f in raw.columns and pd.api.types.is_numeric_dtype(raw[f])]


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

    rows: list[dict] = []
    rng = np.random.default_rng(_stable_seed(args.seed, *keys.values(), target))
    for repeat in range(int(args.repeats)):
        split = _stratified_split(y, rng, float(args.holdout_fraction), int(args.min_class))
        if split is None:
            continue
        train_idx, test_idx = split
        best = None
        for feature in features:
            x = pd.to_numeric(raw[feature], errors="coerce").to_numpy(dtype=float)
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
        finite_train = np.isfinite(x[train_idx]); finite_test = np.isfinite(x[test_idx])
        tr = train_idx[finite_train]; te = test_idx[finite_test]
        if min(int(y[tr].sum()), int(len(tr) - y[tr].sum()), int(y[te].sum()), int(len(te) - y[te].sum())) < int(args.min_class):
            continue
        xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
        prevalence = float(ytr.mean())
        model_probs: dict[str, tuple[np.ndarray, object, str]] = {
            "constant": (np.full(len(te), prevalence), lambda grid, q=prevalence: np.full(len(grid), q), threshold_fit["direction"]),
        }
        threshold = float(threshold_fit["threshold"]); direction = str(threshold_fit["direction"])
        threshold_prob = (xte >= threshold).astype(float) if direction == ">=" else (xte <= threshold).astype(float)
        model_probs["threshold"] = (
            threshold_prob,
            (lambda grid, t=threshold, d=direction: ((grid >= t) if d == ">=" else (grid <= t)).astype(float)),
            direction,
        )
        try:
            logistic = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=int(args.seed))
            logistic.fit(xtr.reshape(-1, 1), ytr)
            model_probs["logistic"] = (
                logistic.predict_proba(xte.reshape(-1, 1))[:, 1],
                lambda grid, m=logistic: m.predict_proba(np.asarray(grid).reshape(-1, 1))[:, 1],
                direction,
            )
        except Exception:
            pass
        try:
            isotonic = IsotonicRegression(increasing="auto", out_of_bounds="clip", y_min=0.0, y_max=1.0)
            isotonic.fit(xtr, ytr)
            model_probs["isotonic"] = (
                np.asarray(isotonic.predict(xte), dtype=float),
                lambda grid, m=isotonic: np.asarray(m.predict(np.asarray(grid)), dtype=float),
                direction,
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
    group = [c for c in ["run_id", "task", "model_name", "setting", "decode_only", "baseline_subset", "population", "unit_key", "target", "model"] if c in repeats.columns]
    return repeats.groupby(group, dropna=False).agg(
        n_repeats=("repeat", "nunique"),
        median_abs_mcc=("abs_mcc", "median"),
        mean_abs_mcc=("abs_mcc", "mean"),
        median_brier=("brier", "median"),
        median_log_loss=("log_loss", "median"),
        median_ece=("ece", "median"),
        median_transition_sharpness=("transition_sharpness", "median"),
        selected_feature_mode=("selected_feature", lambda s: s.mode().iloc[0] if not s.mode().empty else str(s.iloc[0])),
    ).reset_index()


def _representative_response_curves(raw_all: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Held-out response curves for one representative candidate and control.

    Unit selection is only for visualization; inferential model-comparison
    metrics remain the nested repeated-holdout estimates in the main tables.
    """
    if summary.empty:
        return pd.DataFrame()
    focus = summary.loc[(summary["model"] == "threshold") & (summary["target"] == "flip_any") & summary["population"].isin([POP_CAND, POP_CTRL])].copy()
    rows=[]
    for population in (POP_CAND, POP_CTRL):
        q=focus.loc[focus.population==population].sort_values("median_abs_mcc",ascending=False)
        if q.empty: continue
        top=q.iloc[0]
        raw=raw_all.copy()
        for col in ["run_id","baseline_subset","population","unit_key"]:
            if col in raw.columns and col in top.index:
                raw=raw.loc[raw[col].astype(str)==str(top[col])]
        feature=str(top["selected_feature_mode"]); target="flip_any"
        if raw.empty or feature not in raw.columns or target not in raw.columns: continue
        x=pd.to_numeric(raw[feature],errors="coerce").to_numpy(float); yv=pd.to_numeric(raw[target],errors="coerce").to_numpy(float)
        mask=np.isfinite(x)&np.isfinite(yv); x=x[mask]; y=(yv[mask]>.5).astype(int)
        rng=np.random.default_rng(_stable_seed(args.seed,"response",population,str(top.get("run_id")),str(top.get("unit_key"))))
        split=_stratified_split(y,rng,float(args.holdout_fraction),int(args.min_class))
        if split is None: continue
        tr,te=split; fit=_fit_threshold(x[tr],y[tr],max(1,int(args.min_class)//2))
        if fit is None: continue
        direction=str(fit["direction"]); oriented=x if direction==">=" else -x
        logit=LogisticRegression(solver="lbfgs",max_iter=1000,random_state=int(args.seed)).fit(oriented[tr].reshape(-1,1),y[tr])
        iso=IsotonicRegression(increasing=True,out_of_bounds="clip",y_min=0,y_max=1).fit(oriented[tr],y[tr])
        test=pd.DataFrame({"score":oriented[te],"y":y[te]})
        try: test["bin"]=pd.qcut(test.score,q=min(10,test.score.nunique()),duplicates="drop")
        except Exception: continue
        for bin_index,(_,g) in enumerate(test.groupby("bin",observed=False,sort=True)):
            score=float(g.score.mean())
            rows.append({
                "population":population,"run_id":top.get("run_id"),"unit_key":top.get("unit_key"),"feature":feature,
                "bin_index":int(bin_index),"n":int(len(g)),"mean_oriented_score":score,"heldout_flip_rate":float(g.y.mean()),
                "logistic_probability":float(logit.predict_proba(np.array([[score]]))[:,1][0]),
                "isotonic_probability":float(iso.predict([score])[0]),
                "threshold_probability":float(score >= (float(fit["threshold"]) if direction==">=" else -float(fit["threshold"]))),
            })
    return pd.DataFrame(rows)


def _plot_response_curves(curves: pd.DataFrame, paper_dir: Path) -> None:
    if curves.empty: return
    with paper_figure_rc():
        fig,axes=plt.subplots(1,2,figsize=(6.2,2.5),sharey=True)
        for ax,population in zip(axes,[POP_CAND,POP_CTRL]):
            g=curves.loc[curves.population==population].sort_values("mean_oriented_score")
            if g.empty: ax.axis("off"); continue
            ax.plot(g.mean_oriented_score,g.heldout_flip_rate,marker="o",linewidth=1.4,label="Held-out empirical")
            ax.plot(g.mean_oriented_score,g.logistic_probability,linestyle="--",linewidth=1.2,label="Logistic")
            ax.plot(g.mean_oriented_score,g.isotonic_probability,linestyle=":",linewidth=1.4,label="Isotonic")
            ax.set_title(POP_LABELS[population]); ax.set_xlabel("Oriented endogenous scalar")
            ax.set_ylim(-.03,1.03); ax.grid(alpha=.25,linewidth=.45); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(r"Held-out $P(F_j=1\mid z_j)$")
        axes[0].legend(frameon=False,fontsize=7.0)
        fig.subplots_adjust(left=.10,right=.995,bottom=.20,top=.90,wspace=.18)
        save_pdf_only(fig,paper_dir/"fig4b_threshold_response_curves.pdf"); plt.close(fig)


def _plot(summary: pd.DataFrame, paper_dir: Path) -> None:
    if summary.empty:
        return
    paper_dir.mkdir(parents=True, exist_ok=True)
    pop = summary.loc[summary["population"].isin([POP_CAND, POP_CTRL])].copy()
    metrics = [("median_abs_mcc", "Held-out |MCC|", True), ("median_brier", "Brier score", False), ("median_log_loss", "Log loss", False), ("median_ece", "Calibration error", False)]
    with paper_figure_rc():
        fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.25))
        for ax, (metric, label, higher) in zip(axes, metrics):
            agg = pop.groupby(["population", "model"], dropna=False)[metric].median().reset_index()
            x = np.arange(len(MODELS), dtype=float)
            width = 0.34
            for offset, population in [(-width/2, POP_CTRL), (width/2, POP_CAND)]:
                vals=[]
                for model in MODELS:
                    q=agg[(agg.population==population)&(agg.model==model)][metric]
                    vals.append(float(q.iloc[0]) if len(q) else np.nan)
                ax.bar(x+offset, vals, width=width, label=POP_LABELS[population])
            ax.set_xticks(x, ["Const.", "Thresh.", "Logit", "Isotonic"], rotation=30, ha="right")
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.25, linewidth=.45)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].legend(frameon=False, fontsize=7.0)
        fig.subplots_adjust(left=.07, right=.995, bottom=.27, top=.96, wspace=.48)
        save_pdf_only(fig, paper_dir / "fig4b_threshold_shape_model_comparison.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not (0 < float(args.holdout_fraction) < 1):
        raise ValueError("--holdout-fraction must be in (0, 1)")
    source_kind = "zip" if args.zip else "root"
    source_path = Path(args.zip or args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    raw_all = concat(source_kind, source_path, "aggregate_activation_flip_rows.csv")
    tests_all = concat(source_kind, source_path, "aggregate_unit_tests.csv")
    if raw_all.empty:
        status={"status":"not_available","reason":"No aggregate_activation_flip_rows.csv found"}
        (out/"threshold_shape_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(status["reason"]); return
    if tests_all.empty:
        raise RuntimeError("No aggregate_unit_tests.csv found; cannot define the frozen scalar family")
    raw_all = raw_all.loc[raw_all.get("population", "").isin([POP_CAND, POP_CTRL])].copy()
    tests_all = tests_all.loc[tests_all.get("population", "").isin([POP_CAND, POP_CTRL])].copy()
    group_cols=[c for c in ["run_id","task","model","setting","decode_only","baseline_subset","population","unit_key"] if c in raw_all.columns]
    rows=[]
    grouped=list(raw_all.groupby(group_cols,dropna=False))
    for key_tuple, raw in grouped:
        if not isinstance(key_tuple, tuple): key_tuple=(key_tuple,)
        keys=dict(zip(group_cols,key_tuple))
        if "model" in keys:
            keys["model_name"] = keys.pop("model")
        for target in TARGETS:
            rows.extend(_evaluate_group(raw, tests_all, keys=keys, target=target, args=args))
    repeats=pd.DataFrame(rows)
    summary=_summarize(repeats)
    curves=_representative_response_curves(raw_all, summary, args)
    repeats.to_csv(out/"threshold_shape_model_comparison_repeats.csv",index=False)
    summary.to_csv(out/"threshold_shape_model_comparison.csv",index=False)
    curves.to_csv(out/"threshold_response_curves_heldout.csv",index=False)
    population = pd.DataFrame()
    if not summary.empty:
        population=summary.groupby(["population","target","model"],dropna=False).agg(
            n_units=("unit_key","count"), median_abs_mcc=("median_abs_mcc","median"), median_brier=("median_brier","median"), median_log_loss=("median_log_loss","median"), median_ece=("median_ece","median"), median_transition_sharpness=("median_transition_sharpness","median")
        ).reset_index()
    population.to_csv(out/"threshold_shape_population_summary.csv",index=False)
    if args.paper_figures_dir:
        paper_dir=Path(args.paper_figures_dir).expanduser().resolve()
        _plot(summary, paper_dir)
        _plot_response_curves(curves, paper_dir)
    status={"status":"ok" if not summary.empty else "no_eligible_units","n_repeat_rows":int(len(repeats)),"n_unit_model_rows":int(len(summary)),"nested_feature_selection":True,"models":list(MODELS),"metrics":["abs_mcc","brier","log_loss","ece","transition_sharpness"]}
    (out/"threshold_shape_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(json.dumps(status,indent=2))


if __name__ == "__main__":
    main()
