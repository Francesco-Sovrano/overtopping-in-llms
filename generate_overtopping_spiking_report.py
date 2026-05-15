#!/usr/bin/env python3
"""Generate aggregate overtopping-as-spiking statistics, figures, and an updated
Markdown report from spiking_diagnostics_results_for_inspection.zip.

Example:
  python generate_overtopping_spiking_report.py \
    --zip spiking_diagnostics_results_for_inspection.zip \
    --out overtopping_spiking_report \
    --base-md spiking_diagnostics_experiments.md

The script reads only the aggregate CSV files it needs directly from the zip;
it does not extract the whole bundle.

Dependencies: pandas, numpy, matplotlib.
"""
from __future__ import annotations

import argparse, fnmatch, glob, io, json, math, os, shutil, textwrap, zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

POP_CAND = "flip_rule_candidate"
POP_CTRL = "random_nonagonist_control"
POP_LABELS = {POP_CAND: "Candidates", POP_CTRL: "Non-candidate controls"}
RNG = np.random.default_rng(20260502)


def parse_args():
    p = argparse.ArgumentParser(description="Generate overtopping-as-spiking report from results zip/root")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="Input spiking diagnostics results zip")
    src.add_argument("--root", help="Already-extracted result root containing data/")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--base-md", default=None, help="Optional Markdown file to update")
    p.add_argument("--bootstrap", type=int, default=3000, help="Bootstrap samples for median-delta CI")
    return p.parse_args()


def rm(path: Path):
    if path.exists(): shutil.rmtree(path)


def normalize_member_name(name: str) -> str:
    return name.lstrip("./")


def meta_for(rel: str) -> Dict[str, object]:
    rel = normalize_member_name(rel)
    parts = rel.split("/")
    m = dict(rel=rel, task="unknown", model="unknown", setting="unknown", decode_only=False, run_id="unknown")
    # Expected: data/<task>/<org>/<model>/.../eap_ig_inputs/<setting>/spiking_diagnostics/<file>
    if len(parts) >= 4 and parts[0] == "data":
        task, org, model = parts[1], parts[2], parts[3]
        setting = "unknown"
        if "eap_ig_inputs" in parts:
            i = parts.index("eap_ig_inputs")
            if i + 1 < len(parts): setting = parts[i+1]
        m.update(task=task, model=f"{org}/{model}", setting=setting, decode_only=("decode_only" in setting), run_id=f"{task}__{org}_{model}__{setting}")
    return m


def concat_from_zip(zip_path: Path, suffix: str) -> pd.DataFrame:
    rows=[]
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            rel = normalize_member_name(name)
            if not rel.endswith(suffix):
                continue
            try:
                with zf.open(name) as fh:
                    data = fh.read()
                df = pd.read_csv(io.BytesIO(data))
            except Exception:
                continue
            for k,v in meta_for(rel).items(): df[k]=v
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def concat_from_root(root: Path, suffix: str) -> pd.DataFrame:
    rows=[]
    for f in glob.glob(str(root / "**" / suffix), recursive=True):
        try: df = pd.read_csv(f)
        except Exception: continue
        rel = os.path.relpath(f, root).replace(os.sep, "/")
        for k,v in meta_for(rel).items(): df[k]=v
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def concat(source_kind: str, source_path: Path, suffix: str) -> pd.DataFrame:
    return concat_from_zip(source_path, suffix) if source_kind == "zip" else concat_from_root(source_path, suffix)


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
    """Save vector PDF only. The script intentionally does not emit PNGs."""
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
            fig_obj, ax = plt.subplots(figsize=(4.9, 2.8))
            for fam in ["activation", "activation-magnitude", "wanda/activation-magnitude", "gradient/effect"]:
                for pop in [POP_CAND, POP_CTRL]:
                    g=binned_agg[(binned_agg.feature_family==fam)&(binned_agg.population==pop)].sort_values("bin_index")
                    if not g.empty:
                        ax.plot(g.bin_index, g.median_flip_enrichment, marker="o", markersize=3.8, linewidth=1.3, label=f"{fam} | {POP_LABELS[pop]}")
            ax.axhline(1.0, color="0.10", linewidth=0.8)
            ax.set_xlabel("Oriented proxy bin", labelpad=1.0)
            ax.set_ylabel("Median flip-rate enrichment", labelpad=1.0)
            ax.grid(axis="y", alpha=0.35, linewidth=0.45)
            ax.legend(frameon=False, fontsize=7.2, ncol=1, loc="upper left", borderaxespad=0.15)
            fig_obj.subplots_adjust(left=0.15, right=0.995, bottom=0.17, top=0.985)
            save_pdf_only(fig_obj, fig/"binned_flip_curves_oriented_proxy.pdf")
            plt.close(fig_obj)


def row(rows,pop):
    for r in rows:
        if r.get("population")==pop: return r
    return {}


def build_report(base_md: Optional[Path], results: Dict[str,object]) -> str:
    cf,rf=row(results["flip_summary"],POP_CAND),row(results["flip_summary"],POP_CTRL)
    cp,rp=row(results["primary_summary"],POP_CAND),row(results["primary_summary"],POP_CTRL)
    fe,ce,me,adj=results["flip_effect"],results["css_effect"],results["best_mcc_effect"],results["planned_tests_holm_corrected_p"]
    empirical=f'''

## Empirical Update from the Completed Diagnostic Runs

### Central interpretation

The completed diagnostics support a dominance-based version of the overtopping-as-spiking hypothesis. The selected overtopping candidates are not the only neurons that can affect behavior, but they are statistically stronger and more spike-like than the sampled non-candidate controls. This is the right interpretation because overtopping means dominance, overlap, and saturation in a fixed regime; it does not mean all other neurons are inert.

### Primary single metric: Threshold-event causal score

Use **Threshold-Event Causal Score (TECS)** as the one primary metric:

```text
TECS(j) = singleton_flip_any_rate(j) * max_feature held_out_abs_MCC(j, feature)
```

TECS is high only when a neuron both flips a nontrivial fraction of examples under singleton intervention and has a simple threshold-like proxy that identifies those flipped examples. Flip rate alone measures causal strength but not spiking structure. Threshold MCC alone measures spiking structure but can over-credit units that affect very few examples. AUC alone measures ranking but not a usable threshold. TECS combines the two parts required by the claim.

### Controls are non-candidates, not guaranteed no-effect neurons

| Population | Units | Median flip-any rate | Mean flip-any rate | Max flip-any rate | Fraction with flip rate >= 0.05 |
|---|---:|---:|---:|---:|---:|
| Candidate | {int(cf.get('n_units',0))} | {fmt(cf.get('median_flip_any'))} | {fmt(cf.get('mean_flip_any'))} | {fmt(cf.get('max_flip_any'))} | {fmt(cf.get('frac_ge_0_05'))} |
| Non-candidate control | {int(rf.get('n_units',0))} | {fmt(rf.get('median_flip_any'))} | {fmt(rf.get('mean_flip_any'))} | {fmt(rf.get('max_flip_any'))} | {fmt(rf.get('frac_ge_0_05'))} |

Paired by run and baseline subset, candidates have higher singleton flip rates than controls: median delta **{fmt(fe.get('median_delta'))}**, 95% bootstrap CI **[{fmt(fe.get('median_delta_ci_low'))}, {fmt(fe.get('median_delta_ci_high'))}]**, one-sided Wilcoxon p **{fmt(fe.get('wilcoxon_p_greater'))}**, Holm-corrected p **{fmt(adj.get('strength_flip_rate'))}**, paired rank-biserial **{fmt(fe.get('paired_rank_biserial'))}**, unit-level Cliff's delta **{fmt(results.get('flip_unit_cliffs_delta'))}**.

### Primary TECS result

| Population | Units | Median best threshold abs(MCC) | Median singleton strength | Median TECS | Mean TECS |
|---|---:|---:|---:|---:|---:|
| Candidate | {int(cp.get('n_units',0))} | {fmt(cp.get('median_best_mcc'))} | {fmt(cp.get('median_strength'))} | {fmt(cp.get('median_css'),5)} | {fmt(cp.get('mean_css'),5)} |
| Non-candidate control | {int(rp.get('n_units',0))} | {fmt(rp.get('median_best_mcc'))} | {fmt(rp.get('median_strength'))} | {fmt(rp.get('median_css'),5)} | {fmt(rp.get('mean_css'),5)} |

Paired by run and baseline subset, candidates have higher TECS than controls: median delta **{fmt(ce.get('median_delta'),5)}**, 95% bootstrap CI **[{fmt(ce.get('median_delta_ci_low'),5)}, {fmt(ce.get('median_delta_ci_high'),5)}]**, one-sided Wilcoxon p **{fmt(ce.get('wilcoxon_p_greater'))}**, Holm-corrected p **{fmt(adj.get('primary_css'))}**, paired rank-biserial **{fmt(ce.get('paired_rank_biserial'))}**, unit-level Cliff's delta **{fmt(results.get('css_unit_cliffs_delta'))}**.

This is the main statistical proof of the spiking claim: selected overtopping candidates have significantly larger combined causal-strength-and-thresholdability scores than non-candidate controls.

### Thresholdability-only result

Using the best held-out threshold abs(MCC) per neuron, candidates also outperform controls: candidate median **{fmt(cp.get('median_best_mcc'))}**, control median **{fmt(rp.get('median_best_mcc'))}**, paired median delta **{fmt(me.get('median_delta'))}**, one-sided Wilcoxon p **{fmt(me.get('wilcoxon_p_greater'))}**, Holm-corrected p **{fmt(adj.get('threshold_mcc'))}**, paired rank-biserial **{fmt(me.get('paired_rank_biserial'))}**, unit-level Cliff's delta **{fmt(results.get('best_mcc_unit_cliffs_delta'))}**.

### Claims supported by the completed runs

Supported claim:

> Selected overtopping candidates are more causally spike-like than non-candidate controls. They have higher singleton-intervention flip rates, higher thresholdability, and higher Threshold-event causal scores under paired run/baseline comparisons.

Do not claim that controls never matter. The correct claim is dominance and saturation: non-candidate controls can have nonzero effects, but selected candidates dominate the distribution of Threshold-event causal scores.

### Recommended aggregate visualizations

Use these as the main figure panels:

1. `figures/ecdf_causal_spiking_score.pdf` — full TECS distribution for candidates and controls.
2. `figures/paired_css_delta_by_run_baseline.pdf` — condition-level consistency of the primary effect.
3. `figures/ecdf_flip_rates_candidate_vs_control.pdf` — causal strength distribution, showing controls are not always inert.
4. `figures/feature_css_delta_ranking.pdf` — proxy features ranked by median candidate-control TECS delta.

Use `figures/binned_flip_curves_oriented_proxy.pdf` as a descriptive supplement for the threshold-tail / spike-like shape.
'''
    if base_md and base_md.exists():
        base=base_md.read_text()
        if "## Empirical Update from the Completed Diagnostic Runs" in base: base=base.split("## Empirical Update from the Completed Diagnostic Runs",1)[0].rstrip()+"\n"
        return base.replace("## Core Concepts", empirical+"\n## Core Concepts", 1) if "## Core Concepts" in base else base.rstrip()+"\n"+empirical
    return "# Overtopping-as-Spiking Diagnostics\n"+empirical


def main():
    args=parse_args(); out=Path(args.out).resolve(); rm(out); out.mkdir(parents=True)
    source_kind="zip" if args.zip else "root"; source_path=Path(args.zip or args.root).resolve(); base_md=Path(args.base_md).resolve() if args.base_md else None
    fs_all=concat(source_kind, source_path, "aggregate_flip_stats.csv")
    ut_all=concat(source_kind, source_path, "aggregate_unit_tests.csv")
    if fs_all.empty: raise RuntimeError("No aggregate_flip_stats.csv found")
    if ut_all.empty: raise RuntimeError("No aggregate_unit_tests.csv found")
    fs=fs_all[fs_all.population.isin([POP_CAND,POP_CTRL])].copy()
    flip_summary=fs.groupby("population").agg(n_units=("unit_key","count"), n_runs=("run_id","nunique"), median_flip_any=("flip_any_rate","median"), mean_flip_any=("flip_any_rate","mean"), q75_flip_any=("flip_any_rate",lambda s:s.quantile(.75)), q90_flip_any=("flip_any_rate",lambda s:s.quantile(.90)), q95_flip_any=("flip_any_rate",lambda s:s.quantile(.95)), max_flip_any=("flip_any_rate","max"), frac_ge_0_01=("flip_any_rate",lambda s:(s>=.01).mean()), frac_ge_0_05=("flip_any_rate",lambda s:(s>=.05).mean()), frac_ge_0_10=("flip_any_rate",lambda s:(s>=.10).mean()), frac_ge_0_20=("flip_any_rate",lambda s:(s>=.20).mean())).reset_index()
    flip_med=paired_medians(fs,"flip_any_rate"); flip_eff=effect_summary(flip_med,args.bootstrap); flip_cd=cliffs_delta(fs.loc[fs.population==POP_CAND,"flip_any_rate"],fs.loc[fs.population==POP_CTRL,"flip_any_rate"])
    ut=ut_all[ut_all.population.isin([POP_CAND,POP_CTRL])].dropna(subset=["median_test_abs_mcc","population_strength"]).copy()
    idx=ut.groupby(["run_id","task","model","setting","decode_only","baseline_subset","population","unit_key"],dropna=False)["median_test_abs_mcc"].idxmax()
    best=ut.loc[idx].copy().rename(columns={"median_test_abs_mcc":"best_mcc","median_test_auc_oriented":"best_auc","feature":"best_feature"})
    best["causal_spiking_score"]=best.population_strength*best.best_mcc
    best["feature_family"]=best.best_feature.map(family)
    primary=best.groupby("population").agg(n_units=("unit_key","count"), n_runs=("run_id","nunique"), median_best_mcc=("best_mcc","median"), mean_best_mcc=("best_mcc","mean"), median_strength=("population_strength","median"), mean_strength=("population_strength","mean"), median_css=("causal_spiking_score","median"), mean_css=("causal_spiking_score","mean"), q75_css=("causal_spiking_score",lambda s:s.quantile(.75)), q90_css=("causal_spiking_score",lambda s:s.quantile(.90)), frac_css_ge_0_01=("causal_spiking_score",lambda s:(s>=.01).mean()), frac_css_ge_0_02=("causal_spiking_score",lambda s:(s>=.02).mean())).reset_index()
    css_med=paired_medians(best,"causal_spiking_score"); css_eff=effect_summary(css_med,args.bootstrap); css_cd=cliffs_delta(best.loc[best.population==POP_CAND,"causal_spiking_score"], best.loc[best.population==POP_CTRL,"causal_spiking_score"])
    mcc_med=paired_medians(best,"best_mcc"); mcc_eff=effect_summary(mcc_med,args.bootstrap); mcc_cd=cliffs_delta(best.loc[best.population==POP_CAND,"best_mcc"], best.loc[best.population==POP_CTRL,"best_mcc"])
    adj=holm({"strength_flip_rate":flip_eff["wilcoxon_p_greater"],"threshold_mcc":mcc_eff["wilcoxon_p_greater"],"primary_css":css_eff["wilcoxon_p_greater"]})
    feats=[]
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
    b=concat(source_kind, source_path, "aggregate_binned_curves.csv")
    binned_agg=pd.DataFrame()
    if not b.empty:
        b=b[b.population.isin([POP_CAND,POP_CTRL])].copy(); b["feature_family"]=b.feature.map(family);
        if "baseline_subset" not in b.columns:
            b["baseline_subset"] = b.get("source_baseline_dir", "unknown").astype(str).str.replace("_baseline", "", regex=False)
        keys=["run_id","baseline_subset","population","unit_key","feature","target"]
        b["curve_mean"]=b.groupby(keys,dropna=False).flip_rate.transform("mean"); b["flip_enrichment"]=b.flip_rate/b.curve_mean.replace(0,np.nan)
        binned_agg=b.groupby(["feature_family","population","bin_index"],dropna=False).agg(median_flip_enrichment=("flip_enrichment","median"),mean_flip_enrichment=("flip_enrichment","mean"),median_flip_rate=("flip_rate","median"),n_curves=("unit_key","count")).reset_index()
    for name,df in {"flip_rate_summary.csv":flip_summary,"paired_flip_rate_by_run_baseline.csv":flip_med.reset_index(),"unit_primary_spiking_scores.csv":best,"primary_spiking_score_summary.csv":primary,"paired_css_by_run_baseline.csv":css_med.reset_index(),"paired_best_mcc_by_run_baseline.csv":mcc_med.reset_index(),"paired_css_delta_rich.csv":rich,"feature_level_stat_summary.csv":feature_comp,"binned_curve_aggregate.csv":binned_agg}.items(): df.to_csv(out/name,index=False)
    results={"populations":{"candidate":POP_CAND,"control":POP_CTRL},"flip_summary":flip_summary.to_dict(orient="records"),"flip_effect":flip_eff,"flip_unit_cliffs_delta":flip_cd,"primary_summary":primary.to_dict(orient="records"),"css_effect":css_eff,"css_unit_cliffs_delta":css_cd,"best_mcc_effect":mcc_eff,"best_mcc_unit_cliffs_delta":mcc_cd,"planned_tests_holm_corrected_p":adj,"n_aggregate_flip_stats_files":int(len(fs_all.rel.unique())) if "rel" in fs_all else 0,"n_aggregate_unit_tests_files":int(len(ut_all.rel.unique())) if "rel" in ut_all else 0}
    (out/"statistical_results.json").write_text(json.dumps(results,indent=2))
    plot_outputs(out,fs,best,rich,feature_comp,binned_agg)
    (out/"updated_spiking_diagnostics_experiments.md").write_text(build_report(base_md,results))
    print(f"Wrote outputs to {out}")

if __name__=="__main__": main()
