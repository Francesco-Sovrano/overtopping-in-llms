#!/usr/bin/env python3
"""Aggregate P0.3 example-level dominant-secondary preemption experiments."""
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
from studies.overtopping.analysis.stage07_overtopping_spiking_report import (
    expected_rq3_sources,
    wilcoxon_greater,
    binom_p_greater,
)


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "results/analysis/rq3_threshold_event/preemption"))
    p.add_argument("--paper-figures-dir", default=None)
    p.add_argument("--primary-table", required=True)
    p.add_argument("--population-scope", choices=["primary", "primary+supplementary"], default="primary+supplementary")
    p.add_argument("--evaluation-split", choices=["test", "train", "all"], default="test")
    p.add_argument("--spiking-max-points", type=int, default=10000)
    p.add_argument("--data-root", default=None, help="Alias for --root used by the shared exact RQ3 manifest.")
    return p.parse_args()


def _meta(path: Path, root: Path) -> dict:
    try: parts=path.relative_to(root).parts
    except Exception: parts=path.parts
    out={"task":"unknown","org":"unknown","model":"unknown","run":"unknown","phase":"unknown","baseline":"unknown"}
    if len(parts)>=3:
        out.update(task=parts[0],org=parts[1],model=parts[2])
    # interaction_validation is directly under one Stage-7 stats run.
    if "interaction_validation" in parts:
        i=parts.index("interaction_validation")
        if i>0:
            run=parts[i-1]; out["run"]=run
            out["phase"]="decode-only" if "decode_only" in run else "input+output"
            out["baseline"]="mean-donor" if "eval_mean-donor" in run else ("mean" if "eval_mean" in run or "mean" in run else "unknown")
    return out


def load_all(root: Path, args: argparse.Namespace, out: Path) -> pd.DataFrame:
    """Load preemption only from exact primary/supplementary overtopping stats dirs."""
    # expected_rq3_sources needs the canonical data-root field used by Stage 7/8.
    args.data_root = str(Path(args.data_root or root).expanduser().resolve())
    expected = expected_rq3_sources(args)
    rows=[]; audit=[]
    for spec in expected:
        path = Path(spec["stats_dir"]) / "interaction_validation" / "preemption_pair_summary.csv"
        exists = path.is_file()
        n_rows = 0
        if exists:
            try: df=pd.read_csv(path)
            except Exception: df=pd.DataFrame()
            if not df.empty:
                n_rows=int(len(df))
                for k,v in _meta(path,root).items(): df[k]=v
                df["run_id"]=spec["run_id"]
                df["source_scope"]=spec.get("source_scope","primary")
                df["source_path"]=str(path)
                rows.append(df)
        audit.append({
            "run_id":spec["run_id"], "source_scope":spec.get("source_scope","primary"),
            "task":spec["task"], "model":spec["model"], "phase":spec["phase"],
            "source":str(path), "exists":bool(exists), "n_rows":int(n_rows),
        })
    pd.DataFrame(audit).to_csv(out/"preemption_population_audit.csv",index=False)
    frame=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    if not frame.empty and frame.get("task",pd.Series(dtype=str)).astype(str).str.contains("poison",case=False).any():
        raise RuntimeError("Poisoning rows leaked into exact RQ3 preemption population")
    return frame


def _condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pair-level evidence before inference to avoid pseudoreplication."""
    if frame.empty:
        return pd.DataFrame()
    d = frame.copy()
    for col in ["delta_secondary_given_dominant_absent", "delta_secondary_given_dominant_present", "preemption_index_absent_minus_present"]:
        d[col] = pd.to_numeric(d.get(col), errors="coerce")
    d = d.loc[np.isfinite(d["preemption_index_absent_minus_present"])].copy()
    if d.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["run_id", "source_scope", "task", "model", "phase", "baseline", "direction"] if c in d.columns]
    if not group_cols:
        return pd.DataFrame()
    out = d.groupby(group_cols, dropna=False).agg(
        n_pairs_defined=("preemption_index_absent_minus_present", "count"),
        median_delta_absent=("delta_secondary_given_dominant_absent", "median"),
        median_delta_present=("delta_secondary_given_dominant_present", "median"),
        median_preemption_index=("preemption_index_absent_minus_present", "median"),
        fraction_pairs_supporting=("preemption_index_absent_minus_present", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
    ).reset_index()
    return out


def plot(condition: pd.DataFrame, out: Path) -> None:
    """Paper plot uses one point per run/baseline/direction condition."""
    if condition.empty:
        out.unlink(missing_ok=True)
        return
    x=pd.to_numeric(condition.get("median_delta_absent"),errors="coerce")
    y=pd.to_numeric(condition.get("median_delta_present"),errors="coerce")
    finite=condition.loc[np.isfinite(x)&np.isfinite(y)].copy()
    if finite.empty:
        out.unlink(missing_ok=True)
        return
    x=pd.to_numeric(finite["median_delta_absent"],errors="coerce").to_numpy(float)
    y=pd.to_numeric(finite["median_delta_present"],errors="coerce").to_numpy(float)
    fig,ax=plt.subplots(figsize=(4.6,3.4))
    ax.scatter(x,y,s=28,alpha=.80)
    lo=min(float(np.min(x)),float(np.min(y)),0.0); hi=max(float(np.max(x)),float(np.max(y)),0.0)
    pad=max(.01,.08*(hi-lo if hi>lo else 1.0)); lo-=pad; hi+=pad
    ax.plot([lo,hi],[lo,hi],linestyle="--",linewidth=1.0,label="equal condition-median effect")
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel(r"Condition median: secondary marginal when dominant absent $\Delta_k^-$")
    ax.set_ylabel(r"Condition median: secondary marginal when dominant present $\Delta_k^+$")
    support=float((x>y).mean())
    ax.text(.03,.97,f"conditions={len(x)}\n" + r"$\tilde\Delta_k^+<\tilde\Delta_k^-$" + f": {support:.0%}",transform=ax.transAxes,ha="left",va="top",bbox=dict(boxstyle="round,pad=.2",facecolor="white",edgecolor="0.8",alpha=.9))
    ax.grid(alpha=.25,linewidth=.45); ax.legend(frameon=False,loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args=parse_args(); root=Path(args.root).expanduser().resolve(); out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    frame=load_all(root,args,out); frame.to_csv(out/"preemption_all_pairs.csv",index=False)
    condition = _condition_summary(frame)
    condition.to_csv(out/"preemption_by_condition.csv", index=False)
    # Backward-compatible alias; inference is condition-level, not pair-level.
    condition.to_csv(out/"preemption_by_setting.csv", index=False)
    if frame.empty:
        status={"status":"not_available","n_pairs":0,"reason":"No exact-manifest interaction_validation/preemption_pair_summary.csv artifacts found"}
    else:
        idx=pd.to_numeric(frame.get("preemption_index_absent_minus_present"),errors="coerce"); finite=idx[np.isfinite(idx)]
        cond_idx = pd.to_numeric(condition.get("median_preemption_index"), errors="coerce") if not condition.empty else pd.Series(dtype=float)
        cond_idx = cond_idx[np.isfinite(cond_idx)].to_numpy(float)
        p_w, rb, dz = wilcoxon_greater(cond_idx) if len(cond_idx) else (math.nan, math.nan, math.nan)
        n_pos = int((cond_idx > 0).sum()) if len(cond_idx) else 0
        n_nonzero = int((cond_idx != 0).sum()) if len(cond_idx) else 0
        p_sign = binom_p_greater(n_pos, n_nonzero, .5) if n_nonzero else math.nan
        status={
            "status":"ok",
            "population_scope":args.population_scope,
            "excluded_poisoning":True,
            "n_pairs":int(len(frame)),
            "n_pairs_defined":int(len(finite)),
            "pair_level_fraction_supporting_preemption_descriptive":float((finite>0).mean()) if len(finite) else math.nan,
            "pair_level_median_preemption_index_descriptive":float(finite.median()) if len(finite) else math.nan,
            "n_conditions_defined":int(len(cond_idx)),
            "fraction_conditions_positive":float((cond_idx>0).mean()) if len(cond_idx) else math.nan,
            "median_condition_preemption_index":float(np.median(cond_idx)) if len(cond_idx) else math.nan,
            "condition_level_wilcoxon_greater_p":float(p_w),
            "condition_level_rank_biserial":float(rb),
            "condition_level_dz":float(dz),
            "condition_level_sign_test_greater_p":float(p_sign),
            "inference_unit":"run/baseline/direction condition; pair rows are descriptive only",
        }
        plot(condition,out/"preemption_delta_plus_vs_minus.pdf")
        if args.paper_figures_dir:
            plot(condition,Path(args.paper_figures_dir).expanduser().resolve()/"fig4c_preemption.pdf")
    (out/"preemption_report_status.json").write_text(json.dumps(status,indent=2,allow_nan=True),encoding="utf-8")
    print(json.dumps(status,indent=2,allow_nan=True))


if __name__=="__main__": main()
