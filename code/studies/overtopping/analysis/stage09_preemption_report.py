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


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "results/analysis/rq3_threshold_event/preemption"))
    p.add_argument("--paper-figures-dir", default=None)
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


def load_all(root: Path) -> pd.DataFrame:
    rows=[]
    for path in root.glob("**/interaction_validation/preemption_pair_summary.csv"):
        try: df=pd.read_csv(path)
        except Exception: continue
        if df.empty: continue
        for k,v in _meta(path,root).items(): df[k]=v
        df["source_path"]=str(path)
        rows.append(df)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def plot(frame: pd.DataFrame, out: Path) -> None:
    finite=frame.copy()
    x=pd.to_numeric(finite.get("delta_secondary_given_dominant_absent"),errors="coerce")
    y=pd.to_numeric(finite.get("delta_secondary_given_dominant_present"),errors="coerce")
    finite=finite.loc[np.isfinite(x)&np.isfinite(y)].copy()
    if finite.empty: return
    x=pd.to_numeric(finite["delta_secondary_given_dominant_absent"],errors="coerce").to_numpy(float)
    y=pd.to_numeric(finite["delta_secondary_given_dominant_present"],errors="coerce").to_numpy(float)
    fig,ax=plt.subplots(figsize=(4.6,3.4))
    ax.scatter(x,y,s=22,alpha=.78)
    lo=min(float(np.min(x)),float(np.min(y)),0.0); hi=max(float(np.max(x)),float(np.max(y)),0.0)
    pad=max(.01,.08*(hi-lo if hi>lo else 1.0)); lo-=pad; hi+=pad
    ax.plot([lo,hi],[lo,hi],linestyle="--",linewidth=1.0,label="equal marginal effect")
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel(r"Secondary marginal when dominant event absent $\Delta_k^-$")
    ax.set_ylabel(r"Secondary marginal when dominant event present $\Delta_k^+$")
    support=float((x>y).mean())
    ax.text(.03,.97,f"pairs={len(x)}\n" + r"$\Delta_k^+<\Delta_k^-$" + f": {support:.0%}",transform=ax.transAxes,ha="left",va="top",bbox=dict(boxstyle="round,pad=.2",facecolor="white",edgecolor="0.8",alpha=.9))
    ax.grid(alpha=.25,linewidth=.45); ax.legend(frameon=False,loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args=parse_args(); root=Path(args.root).expanduser().resolve(); out=Path(args.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    frame=load_all(root); frame.to_csv(out/"preemption_all_pairs.csv",index=False)
    if frame.empty:
        status={"status":"not_available","n_pairs":0,"reason":"No interaction_validation/preemption_pair_summary.csv artifacts found"}
    else:
        idx=pd.to_numeric(frame.get("preemption_index_absent_minus_present"),errors="coerce"); finite=idx[np.isfinite(idx)]
        group_cols=[c for c in ["task","model","phase","baseline","direction"] if c in frame.columns]
        summary=frame.groupby(group_cols,dropna=False).agg(n_pairs=("secondary_unit","count"),median_delta_present=("delta_secondary_given_dominant_present","median"),median_delta_absent=("delta_secondary_given_dominant_absent","median"),median_preemption_index=("preemption_index_absent_minus_present","median"),fraction_supporting_preemption=("preemption_index_absent_minus_present",lambda s: float((pd.to_numeric(s,errors="coerce")>0).mean()))).reset_index()
        summary.to_csv(out/"preemption_by_setting.csv",index=False)
        status={"status":"ok","n_pairs":int(len(frame)),"n_pairs_defined":int(len(finite)),"fraction_supporting_preemption":float((finite>0).mean()) if len(finite) else math.nan,"median_preemption_index":float(finite.median()) if len(finite) else math.nan}
        plot(frame,out/"preemption_delta_plus_vs_minus.pdf")
        if args.paper_figures_dir:
            plot(frame,Path(args.paper_figures_dir).expanduser().resolve()/"fig4c_preemption.pdf")
    (out/"preemption_report_status.json").write_text(json.dumps(status,indent=2,allow_nan=True),encoding="utf-8")
    print(json.dumps(status,indent=2,allow_nan=True))


if __name__=="__main__": main()
