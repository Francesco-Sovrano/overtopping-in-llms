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
                required_cols={"candidate_direction_policy","discovery_baseline_subset","interaction_validation_schema"}
                missing=sorted(required_cols-set(df.columns))
                if missing:
                    raise RuntimeError(
                        f"Stale preemption summary at {path}: missing {missing}. "
                        "Rerun interaction validation with discovery-direction-aware preemption before Figure 4c."
                    )
                expected_schema="conditional-marginal-validation-v2-direction-aware-preemption"
                bad_schema=df["interaction_validation_schema"].astype(str).ne(expected_schema)
                bad_policy=df["candidate_direction_policy"].astype(str).ne("discovery_baseline_only")
                expected_baseline=df["direction"].astype(str).map({"c2i":"positive","i2c":"negative"})
                bad_baseline=df["discovery_baseline_subset"].astype(str).ne(expected_baseline.astype(str))
                if bool((bad_schema|bad_policy|bad_baseline).any()):
                    preview=df.loc[bad_schema|bad_policy|bad_baseline,[c for c in ["direction","discovery_baseline_subset","candidate_direction_policy","interaction_validation_schema"] if c in df.columns]].head(8).to_dict("records")
                    raise RuntimeError(
                        f"Stale/misaligned directional preemption summary at {path}; examples={preview}. "
                        "Rerun interaction validation before Figure 4c."
                    )
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
    """Paper plot uses one point per run/baseline/direction condition, split by direction."""
    if condition.empty:
        out.unlink(missing_ok=True)
        return
    with plt.rc_context({}):
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.25),sharex=True,sharey=True)
        made=False
        for ax,direction in zip(axes,["c2i","i2c"]):
            g=condition.loc[condition.direction.astype(str)==direction].copy() if "direction" in condition.columns else pd.DataFrame()
            x=pd.to_numeric(g.get("median_delta_absent"),errors="coerce") if not g.empty else pd.Series(dtype=float)
            y=pd.to_numeric(g.get("median_delta_present"),errors="coerce") if not g.empty else pd.Series(dtype=float)
            finite=g.loc[np.isfinite(x)&np.isfinite(y)].copy() if not g.empty else pd.DataFrame()
            if not finite.empty:
                made=True
                xv=pd.to_numeric(finite["median_delta_absent"],errors="coerce").to_numpy(float)
                yv=pd.to_numeric(finite["median_delta_present"],errors="coerce").to_numpy(float)
                lo=min(float(np.min(xv)),float(np.min(yv)),0.0); hi=max(float(np.max(xv)),float(np.max(yv)),0.0)
                pad=max(.01,.08*(hi-lo if hi>lo else 1.0)); lo-=pad; hi+=pad
                ax.scatter(xv,yv,s=28,alpha=.80)
                ax.plot([lo,hi],[lo,hi],linestyle="--",linewidth=1.0,label="equal condition-median effect")
                ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
                support=float((xv>yv).mean())
                ax.text(.03,.97,f"conditions={len(xv)}\n" + r"$\tilde\Delta_k^+<\tilde\Delta_k^-$" + f": {support:.0%}",transform=ax.transAxes,ha="left",va="top",fontsize=7.2)
            ax.set_title("Discovery direction 1→0" if direction=="c2i" else "Discovery direction 0→1")
            ax.set_xlabel(r"Secondary marginal when dominant absent $\Delta_k^-$")
            ax.grid(alpha=.25,linewidth=.45); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(r"Secondary marginal when dominant present $\Delta_k^+$")
        if made: axes[0].legend(frameon=False,loc="lower right",fontsize=7)
        fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True)
        if made: fig.savefig(out,bbox_inches="tight")
        else: out.unlink(missing_ok=True)
        plt.close(fig)


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
        directional_status={}
        if not condition.empty and "direction" in condition.columns:
            for direction in ["c2i","i2c"]:
                vals=pd.to_numeric(condition.loc[condition.direction.astype(str)==direction,"median_preemption_index"],errors="coerce").dropna().to_numpy(float)
                p_dir,rb_dir,dz_dir=wilcoxon_greater(vals) if len(vals) else (math.nan,math.nan,math.nan)
                directional_status[direction]={
                    "direction_label":"1to0" if direction=="c2i" else "0to1",
                    "n_conditions":int(len(vals)),
                    "median_preemption_index":float(np.median(vals)) if len(vals) else math.nan,
                    "fraction_positive":float((vals>0).mean()) if len(vals) else math.nan,
                    "wilcoxon_greater_p":float(p_dir),
                }
        status={
            "status":"ok",
            "population_scope":args.population_scope,
            "candidate_direction_policy":"frozen discovery baseline only",
            "directional_results":directional_status,
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
