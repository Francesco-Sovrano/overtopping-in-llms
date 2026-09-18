"""Generate the paper-facing LaTeX tables used by the current manuscript.

This module is intentionally downstream of the scientific analyses. It reads the
canonical CSV outputs under ``results/analysis`` and writes only presentation
LaTeX under ``results/paper/tables``. No experiment is rerun and no analysis
population is changed here.

The table layouts and captions are kept synchronized with the manuscript source
so ``python -m reporting.generate_manuscript_tables --results-root ../results``
regenerates the files imported by the paper.
"""
from __future__ import annotations

import argparse
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TASK_ORDER = {"Arithmetic": 0, "Grammar": 1, "Random FSM": 2, "NLI": 3, "Jailbreak": 4}
MODEL_ORDER = {"Pythia-1B": 0, "Qwen2-1.5B": 1, "Qwen2.5-1.5B": 2, "Qwen2-7B": 3}
PHASE_ORDER = {"I+O": 0, "Out": 1}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fixed(value: object, digits: int) -> str:
    """Format analysis values with conventional half-up rounding.

    CSV floating-point values can sit microscopically below an exact half-step
    (for example 0.5874999999999999).  Denoise at 12 decimal places before
    Decimal quantization so paper tables reproduce the manuscript rounding.
    """
    x = round(float(value), 12)
    q = Decimal("1").scaleb(-digits)
    return format(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP), f".{digits}f")


def _f3(value: object, *, zero_if_missing: bool = False) -> str:
    if not _finite(value):
        return "0.000" if zero_if_missing else "--"
    return _fixed(value, 3)


def _signed3(value: object) -> str:
    if not _finite(value):
        return "--"
    x = float(value)
    if abs(x) < 0.0005:
        return "0.000"
    text = _fixed(x, 3)
    return f"+{text}" if x > 0 else text


def _p(value: object) -> str:
    if not _finite(value):
        return "--"
    return f"{float(value):.3g}"


def _int(value: object, *, zero_if_missing: bool = False) -> str:
    if not _finite(value):
        return "0" if zero_if_missing else "--"
    return str(int(round(float(value))))


def _count_word(n: int) -> str:
    words = {
        0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
        11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
        15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen",
        19: "Nineteen", 20: "Twenty",
    }
    return words.get(int(n), str(int(n)))


def _quartiles(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.asarray([float(x) for x in values if _finite(x)], dtype=float)
    if len(arr) == 0:
        return math.nan, math.nan, math.nan
    return float(np.median(arr)), float(np.quantile(arr, .25)), float(np.quantile(arr, .75))


def _fmt_median_iqr(values: Iterable[float], *, scale: float = 1.0, digits: int = 3) -> str:
    med, q25, q75 = _quartiles(values)
    if not _finite(med):
        return "--"
    return f"{_fixed(scale*med, digits)} [{_fixed(scale*q25, digits)},{_fixed(scale*q75, digits)}]"


def _normalize_primary(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the configured 50-run table for manuscript rendering only.

    A completed zero-candidate discovery can have no Stage-7 singleton file by
    construction. For display, the empty union is zero. Direction-specific
    inferential eligibility remains encoded in the underlying status columns;
    this renderer does not change the analysis table used by RQ1 statistics.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    j = pd.to_numeric(out.get("J"), errors="coerce")
    stage6 = pd.to_numeric(out.get("stage6_discovered_candidate_count"), errors="coerce")
    verified_zero = j.eq(0) | stage6.eq(0) | out.get("structural_zero_candidates", False).fillna(False).astype(bool)
    out.loc[verified_zero, "J"] = 0
    # The manuscript census displays the empty-union value as zero. Missing
    # directional denominators are handled explicitly in the caption/text and
    # are not admitted to direction-specific inference by the analysis code.
    for col in ("U_J_i2c", "U_J_c2i"):
        vals = pd.to_numeric(out.get(col), errors="coerce")
        out[col] = vals
        out.loc[verified_zero & vals.isna(), col] = 0.0
    return out


def _rq1_order(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups: list[tuple[str, pd.DataFrame]] = []
    if frame.empty:
        return groups

    def sorted_primary(g: pd.DataFrame) -> pd.DataFrame:
        h = g.copy()
        h["_task"] = h["task"].map(TASK_ORDER).fillna(99)
        h["_model"] = h["model"].map(MODEL_ORDER).fillna(99)
        h["_phase"] = h["phase"].map(PHASE_ORDER).fillna(99)
        return h.sort_values(["_task", "_model", "_phase"], kind="mergesort")

    primary = frame[frame["suite"].astype(str).eq("mean-donor")]
    groups.append(("Primary model--task sweep", sorted_primary(primary)))

    large = frame[frame["suite"].astype(str).eq("6-7b-models")].copy()
    large_order = {"Arithmetic": 0, "Jailbreak": 1, "NLI": 2}
    large["_task"] = large["task"].map(large_order).fillna(99)
    groups.append(("Larger-model check", large.sort_values(["_task"], kind="mergesort")))

    mean = frame[frame["suite"].astype(str).eq("mean")].copy()
    mean_order = {"Arithmetic": 0, "Grammar": 1, "NLI": 2, "Random FSM": 3}
    mean["_task"] = mean["task"].map(mean_order).fillna(99)
    mean["_phase"] = mean["phase"].map(PHASE_ORDER).fillna(99)
    groups.append(("Mean-replacement check", mean.sort_values(["_task", "_phase"], kind="mergesort")))

    checkpoints = frame[frame["suite"].astype(str).eq("checkpoints")].copy()
    cp_task = {"Grammar": 0, "NLI": 1, "Random FSM": 2}
    cp_step = {"Pythia-1B 0k": 0, "Pythia-1B 48k": 1, "Pythia-1B 96k": 2}
    checkpoints["_task"] = checkpoints["task"].map(cp_task).fillna(99)
    checkpoints["_step"] = checkpoints["model"].map(cp_step).fillna(99)
    groups.append(("Pythia checkpoint check", checkpoints.sort_values(["_task", "_step"], kind="mergesort")))
    return groups


def write_rq1_full(primary: pd.DataFrame, path: Path) -> bool:
    if primary.empty:
        return False
    d = _normalize_primary(primary)
    zero_count = int((pd.to_numeric(d["J"], errors="coerce") == 0).sum())
    ref = {"mean-donor": "MD", "mean": "M", "mean-positional": "MP"}
    lines = [
        r"\begin{table}[htb]", r"\centering", r"\begingroup", r"\small",
        r"\renewcommand{\arraystretch}{0.88}", r"\setlength{\tabcolsep}{2.0pt}",
        r"\begin{tabular}{@{}lllcrrrr@{}}", r"\toprule",
        r"Task & Model & Phase & Ref. & Compet. & $U^{0\to1}$ & $U^{1\to0}$ & $|J|$ \\",
        r"\midrule",
    ]
    groups = _rq1_order(d)
    for gi, (title, g) in enumerate(groups):
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textbf{{{title}}}}} \\")
        lines.append(r"\addlinespace[1pt]")
        for _, row in g.iterrows():
            j_zero = _finite(row.get("J")) and int(round(float(row.get("J")))) == 0
            cells = [
                str(row.get("task", "")), str(row.get("model", "")), str(row.get("phase", "")),
                ref.get(str(row.get("intervention", "")), str(row.get("intervention", ""))),
                _f3(row.get("score")),
                _f3(row.get("U_J_i2c"), zero_if_missing=j_zero),
                _f3(row.get("U_J_c2i"), zero_if_missing=j_zero),
                _int(row.get("J"), zero_if_missing=j_zero),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if gi != len(groups) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\endgroup",
        rf"\caption{{All {len(d)} configured runs. {_count_word(zero_count)} rows have $|J|=0$. For Pythia-1B Jailbreak (I+O and Out) and Pythia-1B 0k Random FSM (I+O), the $0\!\to\!1$ search retained no candidates, so $U^{{0\to1}}=0$ is measured and included in RQ1. Their displayed $U^{{1\to0}}=0$ values are not used in $1\!\to\!0$ inference because the required $B=1$ discovery subset is insufficient or empty. Ref.: MD, mean donor; M, mean; MP, mean positional.}}",
        r"\label{tab:rq1-full}", r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def write_rq2_full(primary: pd.DataFrame, decomp: pd.DataFrame, path: Path) -> bool:
    if primary.empty or decomp.empty:
        return False
    p = _normalize_primary(primary).reset_index(drop=True).copy()
    p["run_id"] = [f"primary_row_{i:02d}" for i in range(len(p))]
    p = p[(p["intervention"].astype(str) == "mean-donor") & (pd.to_numeric(p["J"], errors="coerce") > 0)]
    d = decomp[decomp.get("scope", "overall").astype(str).eq("overall")].copy()
    work = p[["run_id", "task", "model", "phase", "J"]].merge(d, on=["run_id", "task", "model", "phase"], how="inner", sort=False)
    if work.empty:
        return False
    lines = [
        r"\begin{table}[htb]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{.85\linewidth}{!}{%", r"\begin{tabular}{lllrrrrrr}", r"\toprule",
        r"Task & Model & Ph. & $|J|$ & $U(J)$ & $E(J)$ & $\Delta_{\rm comp}$ & Supp. & Coalition-only \\", r"\midrule",
    ]
    for _, row in work.iterrows():
        lines.append(" & ".join([
            str(row["task"]), str(row["model"]), str(row["phase"]), _int(row["J"]),
            _f3(row.get("U_J_complete_case")), _f3(row.get("E_J_complete_case")),
            _signed3(row.get("Delta_comp_complete_case")), _f3(row.get("suppressed_rate_all")),
            _f3(row.get("coalition_only_rate_all")),
        ]) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}}",
        rf"\caption{{Mean-donor composition for all {len(work)} evaluable nonempty candidate sets. Suppression is singleton-reachable mass lost under the full-set intervention; coalition-only is new mass reached only by the full-set intervention.}}",
        r"\label{tab:rq2-full}", r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def write_rq2_summary(summary: pd.DataFrame, path: Path) -> bool:
    if summary.empty:
        return False
    work = summary[summary["population"].astype(str).eq("all-evaluable")].copy()
    if work.empty:
        return False
    order = {"mean-donor": 0, "mean": 1}
    work["_order"] = work["replacement_regime"].map(order).fillna(99)
    work = work.sort_values("_order")
    lines = [
        r"\begin{table}[htb]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Replacement & $n$ & $E<U$ & $E>U$ & Median $E-U$ & Pearson $r$ & $p$ \\", r"\midrule",
    ]
    for _, row in work.iterrows():
        lines.append(
            f"{row['replacement_regime']} & {int(row['n'])} & {int(row['n_negative_gap'])} & {int(row['n_positive_gap'])} & "
            f"{_f3(row['median_gap'])} & {_f3(row['pearson_r'])} & {_p(row['pearson_p'])} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{RQ2 composition by replacement regime for every evaluable nonempty candidate set. Mean includes direct and positional mean replacement. The two replacement regimes are analyzed separately.}",
        r"\label{tab:rq2-regime-summary}", r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def write_rq3_summary(graded: pd.DataFrame, path: Path) -> bool:
    if graded.empty:
        return False
    known = graded[graded["support_kind"].astype(str).eq("known_flip")].copy()
    nonflip = graded[graded["support_kind"].astype(str).eq("same_agonist_nonflip")].copy()
    if known.empty or nonflip.empty:
        return False
    single = float(pd.to_numeric(known["median_single_crossing_rate"], errors="coerce").median())
    dose = float(pd.to_numeric(known["median_first_persistent_crossing_dose"], errors="coerce").median())
    stable = float(pd.to_numeric(nonflip["median_stable_all_doses_rate"], errors="coerce").median())
    lines = [
        r"\begin{table}[htb]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Scope & Known-flip cond. & Non-flip cond. & Single persistent & First persistent dose & Non-flip stable \\", r"\midrule",
        f"Behavioural graded population & {len(known)} & {len(nonflip)} & {_fixed(single, 3)} & {_fixed(dose, 3)} & {_fixed(stable, 3)}" + " " + chr(92) * 2,
        r"\bottomrule", r"\end{tabular}}",
        r"\caption{Condition-level graded response. Values are medians of within-channel summaries.}",
        r"\label{tab:rq3-graded-summary}", r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def write_rq4_same_channel(trajectories: pd.DataFrame, path: Path) -> bool:
    if trajectories.empty:
        return False
    required = {
        "condition", "fraction", "normal_task_accuracy_without_trigger", "conditional_conversion_rate",
        "paired_control_U(J)", "paired_attack_U(J)",
    }
    if not required.issubset(trajectories.columns):
        return False
    d = trajectories.copy()
    d["fraction"] = pd.to_numeric(d["fraction"], errors="coerce")
    fractions = sorted(float(x) for x in d["fraction"].dropna().unique())
    lines = [
        r"\begin{table}[htb]", r"\centering", r"\scriptsize",
        r"\caption{Controlled Grammar fine-tuning across three training runs. Entries are median [Q1,Q3]. Ordinary accuracy remains close while trigger-conditioned conversion diverges; fixed-union reach is reported on its matched cohort.}",
        r"\label{tab:grammar-same-channel}", r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{rcccccc}", r"\toprule",
        r"Progress & Clean acc. & Poisoned acc. & Clean conversion & Poisoned conversion & Clean $U(J)$ & Poisoned $U(J)$ \\", r"\midrule",
    ]
    for frac in fractions:
        clean = d[(d["condition"].astype(str) == "clean") & np.isclose(d["fraction"], frac, equal_nan=False)]
        poison = d[(d["condition"].astype(str) == "poisoned") & np.isclose(d["fraction"], frac, equal_nan=False)]
        if clean.empty or poison.empty:
            continue
        clean_acc = _fmt_median_iqr(clean["normal_task_accuracy_without_trigger"], digits=3)
        poison_acc = _fmt_median_iqr(poison["normal_task_accuracy_without_trigger"], digits=3)
        clean_conv = _fmt_median_iqr(clean["conditional_conversion_rate"], digits=3)
        poison_conv = _fmt_median_iqr(poison["conditional_conversion_rate"], digits=3)
        clean_u = _fmt_median_iqr(clean["paired_control_U(J)"], digits=3)
        poison_u = _fmt_median_iqr(poison["paired_attack_U(J)"], digits=3)
        progress = f"{int(round(100*frac))}\\%"
        lines.append(f"{progress} & {clean_acc} & {poison_acc} & {clean_conv} & {poison_conv} & {clean_u} & {poison_u}" + " " + chr(92) * 2)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def _collect_previous_checkpoint(root: Path) -> pd.DataFrame:
    rows = []
    base = root / "analysis/rq4_learning/poisoning/per_run_visualizations/grammar"
    for p in sorted(base.glob("*/prompt_and_generation/story/one_checkpoint_ahead_defense_checkpoint_summary.csv")):
        d = _read(p)
        if d.empty or "target_fraction" not in d or "mean_defense_leverage" not in d:
            continue
        run_name = next((part for part in p.parts if "__seed_" in part), p.parent.name)
        h = d[["target_fraction", "mean_defense_leverage"]].copy()
        h["run"] = run_name
        rows.append(h)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_rq4_defense(root: Path, current: pd.DataFrame, path: Path) -> bool:
    previous = _collect_previous_checkpoint(root)
    if current.empty or previous.empty:
        return False
    cur = current.copy()
    budget_col = "operating_benign_damage_budget" if "operating_benign_damage_budget" in cur else "benign_damage_budget"
    cur[budget_col] = pd.to_numeric(cur[budget_col], errors="coerce")
    cur = cur[np.isclose(cur[budget_col], .30, equal_nan=False)].copy()
    if cur.empty:
        return False
    cur["target_fraction"] = pd.to_numeric(cur["target_fraction"], errors="coerce")
    previous["target_fraction"] = pd.to_numeric(previous["target_fraction"], errors="coerce")
    previous["mean_defense_leverage"] = pd.to_numeric(previous["mean_defense_leverage"], errors="coerce")

    lines = [
        r"\begin{table}[htb]", r"\centering", r"\small",
        r"\caption{Defense leverage across three controlled Grammar training runs. Entries are median [Q1,Q3] percentage points; positive values mean more attack suppression than benign disruption.}",
        r"\label{tab:rq4-defense-stability}", r"\begin{tabular}{rcc}", r"\toprule",
        r"Progress & Previous-checkpoint target & Current clean-reference target ($\tau=0.30$) \\", r"\midrule",
    ]
    for frac in sorted(cur["target_fraction"].dropna().unique()):
        prev = previous[np.isclose(previous["target_fraction"], frac, equal_nan=False)]["mean_defense_leverage"].dropna()
        c = cur[np.isclose(cur["target_fraction"], frac, equal_nan=False)].iloc[0]
        pm, p25, p75 = _quartiles(prev)
        cm = float(c.get("mean_defense_leverage__median", math.nan))
        c25 = float(c.get("mean_defense_leverage__q25", math.nan))
        c75 = float(c.get("mean_defense_leverage__q75", math.nan))
        ptxt = f"{100*pm:.1f} [{100*p25:.1f},{100*p75:.1f}]" if _finite(pm) else "--"
        ctxt = f"{100*cm:.1f} [{100*c25:.1f},{100*c75:.1f}]" if _finite(cm) else "--"
        n_current = int(c.get("mean_defense_leverage__n_seeds", c.get("n_training_seeds", 0)) or 0)
        if math.isclose(float(frac), .10, abs_tol=1e-9) and n_current < 3:
            ctxt += r"$^{\dagger}$"
        lines.append(f"{int(round(100*float(frac)))}{chr(92)}% & {ptxt} & {ctxt}" + " " + chr(92) * 2)
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\vspace{2pt}\parbox{0.96\linewidth}{\scriptsize $^\dagger$At 10\% progress, one training run has no channel satisfying the clean-reference budget, so the clean-reference summary uses the two defined runs. Previous-checkpoint summaries use all three runs at every checkpoint.}",
        r"\end{table}", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def generate_all(results_root: Path, out_dir: Path | None = None) -> dict[str, bool]:
    root = Path(results_root).resolve()
    out = Path(out_dir).resolve() if out_dir is not None else root / "paper" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    primary = _read(root / "analysis/primary_matrix/tables/primary_table.csv")
    decomp = _read(root / "analysis/rq2_composition/interaction_decomposition/composition_decomposition_by_condition.csv")
    rq2_summary = _read(root / "analysis/rq2_composition/regime_summary/rq2_regime_summary.csv")
    graded = _read(root / "analysis/rq3_threshold_event/spiking_diagnostics/graded_agonist/graded_agonist_by_condition.csv")
    trajectories = _read(root / "analysis/rq4_learning/poisoning/cross_seed_tables/checkpoint_trajectories_all_seeds.csv")
    current_defense = _read(root / "analysis/rq4_learning/poisoning/cross_seed_tables/clean_reference_defense_checkpoint_across_seeds.csv")

    status = {
        "rq1_full_table.tex": write_rq1_full(primary, out / "rq1_full_table.tex"),
        "rq2_regime_summary.tex": write_rq2_summary(rq2_summary, out / "rq2_regime_summary.tex"),
        "rq2_full_table.tex": write_rq2_full(primary, decomp, out / "rq2_full_table.tex"),
        "rq3_graded_summary.tex": write_rq3_summary(graded, out / "rq3_graded_summary.tex"),
        "rq4_grammar_same_channel_table.tex": write_rq4_same_channel(trajectories, out / "rq4_grammar_same_channel_table.tex"),
        "rq4_defense_stability_table.tex": write_rq4_defense(root, current_defense, out / "rq4_defense_stability_table.tex"),
    }
    return status


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="../results")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--require-all", action="store_true", help="Fail if any manuscript table cannot be generated.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    status = generate_all(Path(args.results_root), Path(args.out_dir) if args.out_dir else None)
    for name, ok in status.items():
        print(f"[manuscript-tables] {name}: {'ok' if ok else 'missing source'}")
    if args.require_all and not all(status.values()):
        missing = ", ".join(name for name, ok in status.items() if not ok)
        raise SystemExit(f"Could not generate all manuscript tables: {missing}")


if __name__ == "__main__":
    main()
