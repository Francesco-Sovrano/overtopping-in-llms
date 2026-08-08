#!/usr/bin/env python3
"""Build manuscript-ready tables and plots from held-out experiment outputs."""
from __future__ import annotations
from pathlib import Path
from lib.project_paths import CODE_ROOT, PROJECT_ROOT



import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.heldout_set_metrics import derive_legacy_aggregate_metrics

from analysis.primary_matrix import (
    PRIMARY_PROFILE_CHOICES,
    normalize_primary_table,
    write_normalization_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_table", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "results" / "manuscript"))
    parser.add_argument("--primary_profile", required=True, choices=PRIMARY_PROFILE_CHOICES)
    return parser.parse_args()


def resolve_stats_dir(raw: object, data_root: Path | None) -> Path:
    path = Path(str(raw)).expanduser()
    candidates = [path]
    if not path.is_absolute() and data_root is not None:
        candidates.append(data_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_interaction_m(raw: object) -> int | str | None:
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except Exception:
        pass
    text = str(raw).strip().lower()
    if text == "all":
        return "all"
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return int(text)
    except Exception as exc:
        raise ValueError(f"Unsupported interaction m label {raw!r}") from exc


def interaction_null_map(path: Path) -> dict[tuple[str, int | str | None], dict]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    output = {}
    for row in frame.to_dict("records"):
        m = normalize_interaction_m(row.get("m"))
        output[(str(row["metric"]), m)] = row
    return output


def augment_row(row: pd.Series, stats_dir: Path) -> dict:
    output = row.to_dict()
    output["stats_dir"] = str(stats_dir)
    global_path = stats_dir / "flip_stats_global.json"
    singleton_path = stats_dir / "singleton_set_metrics.json"
    interaction_dir = stats_dir / "interaction_validation"
    interaction_path = interaction_dir / "interaction_validation_summary.json"
    null_path = interaction_dir / "interaction_validation_summary.csv"

    global_payload = load_json(global_path) if global_path.exists() else {}
    by_neuron_path = stats_dir / "flip_stats_by_neuron.csv"
    by_neuron = pd.read_csv(by_neuron_path) if by_neuron_path.exists() else pd.DataFrame()
    singleton = load_json(singleton_path) if singleton_path.exists() else {}
    legacy_exact = (
        derive_legacy_aggregate_metrics(global_payload=global_payload, candidate_stats=by_neuron)
        if not singleton and global_payload and not by_neuron.empty
        else {}
    )
    interaction = load_json(interaction_path) if interaction_path.exists() else {}
    interaction_is_current = interaction.get("definition_version") == "interaction-validation-v3"
    null_map = interaction_null_map(null_path) if interaction_is_current else {}

    def pick(*values):
        for value in values:
            if value is not None:
                try:
                    if pd.notna(value):
                        return value
                except Exception:
                    return value
        return math.nan

    output.update(
        {
            "J": pick(singleton.get("J"), legacy_exact.get("J"), global_payload.get("n_neurons"), output.get("J")),
            "U_J": pick(singleton.get("U_J"), legacy_exact.get("U_J"), global_payload.get("union_flip_any_unique_rate"), output.get("U")),
            "s_1": pick(singleton.get("s_1"), legacy_exact.get("s_1"), output.get("Top")),
            "R_ov": pick(singleton.get("R_ov"), legacy_exact.get("R_ov")),
            "R_ov_status": pick(singleton.get("R_ov_status"), legacy_exact.get("R_ov_status")),
            "N_eff": pick(singleton.get("N_eff"), legacy_exact.get("N_eff")),
            "N_eff_status": pick(singleton.get("N_eff_status"), legacy_exact.get("N_eff_status")),
            "OCC_0": pick(singleton.get("OCC_0")),
            "OCC_0_status": pick(singleton.get("OCC_0_status"), legacy_exact.get("OCC_0_status")),
            "OCC_1": pick(singleton.get("OCC_1")),
            "OCC_1_status": pick(singleton.get("OCC_1_status"), legacy_exact.get("OCC_1_status")),
            "singleton_metrics_status": (
                "exact_event_sidecar" if singleton_path.exists()
                else ("legacy_aggregate_partial" if legacy_exact else "missing")
            ),
            "interaction_metrics_status": (
                "available_v3" if interaction_is_current
                else ("legacy_interaction_schema" if interaction_path.exists() else "missing")
            ),
        }
    )
    toc_map = singleton.get("TOC_m") or (
        global_payload.get("TOC_m")
        if global_payload.get("heldout_set_metrics_definition_version") == "heldout-set-metrics-v2"
        else {}
    ) or {}
    for raw_m, payload in toc_map.items():
        try:
            m = int(raw_m)
        except Exception:
            continue
        value = payload.get("value") if isinstance(payload, dict) else payload
        status = payload.get("status") if isinstance(payload, dict) else "unknown"
        output[f"TOC_{m}"] = value
        output[f"TOC_{m}_status"] = status
    output["TOC_1"] = pick(output.get("TOC_1"))
    output["TOC_1_status"] = pick(
        output.get("TOC_1_status"),
        "unavailable_requires_discovery_frozen_ranking_and_per_example_flip_events"
        if not toc_map else None,
    )

    thresholds = singleton.get("N_t") or legacy_exact.get("N_t") or (
        global_payload.get("N_t")
        if global_payload.get("heldout_set_metrics_definition_version") == "heldout-set-metrics-v2"
        else {}
    ) or {}
    for threshold, count in thresholds.items():
        output[f"N_t_{threshold}"] = count

    candidate_effect = interaction.get("candidate_E_J") or {}
    output["E_J"] = pick(candidate_effect.get("effect"), global_payload.get("E_J"))
    output["E_J_status"] = pick(candidate_effect.get("status"), global_payload.get("E_J_status"))
    e_null = null_map.get(("E_J", None), {})
    for source, target in [
        ("median_null", "E_J_null_median"),
        ("Delta", "E_J_Delta"),
        ("P", "E_J_P"),
        ("p_MC", "E_J_p_MC"),
        ("status", "E_J_null_status"),
        ("null_draws_requested", "E_J_null_B"),
        ("null_draws_finite", "E_J_null_finite"),
    ]:
        output[target] = e_null.get(source, math.nan)

    gccr_rows = (interaction.get("GCCR_m") or []) if interaction_is_current else []
    for item in gccr_rows:
        m = normalize_interaction_m(item.get("m"))
        if m is None:
            continue
        suffix = str(m)
        output[f"GCCR_{suffix}"] = item.get("GCCR_m")
        output[f"GCCR_{suffix}_status"] = item.get("status")
        null = null_map.get(("GCCR_m", m), {})
        output[f"GCCR_{suffix}_null_median"] = null.get("median_null", math.nan)
        output[f"GCCR_{suffix}_Delta"] = null.get("Delta", math.nan)
        output[f"GCCR_{suffix}_P"] = null.get("P", math.nan)
        output[f"GCCR_{suffix}_p_MC"] = null.get("p_MC", math.nan)
        output[f"GCCR_{suffix}_null_status"] = null.get("status", "missing")
        output[f"GCCR_{suffix}_null_B"] = null.get("null_draws_requested", math.nan)
        output[f"GCCR_{suffix}_null_finite"] = null.get("null_draws_finite", math.nan)
    for suffix in ("1", "all"):
        output.setdefault(f"GCCR_{suffix}", math.nan)
        output.setdefault(
            f"GCCR_{suffix}_status",
            "unavailable_requires_interaction_validation_v3",
        )
        output.setdefault(f"GCCR_{suffix}_null_status", "missing")
    return output


def format_value(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return "--"
    return "--" if not np.isfinite(number) else f"{number:.{digits}f}"


def latex_table(frame: pd.DataFrame) -> str:
    columns = [
        "task", "model", "phase", "J", "U_J", "s_1", "TOC_1", "R_ov", "N_eff",
        "OCC_0", "OCC_1", "N_t_0.05", "N_t_0.1", "E_J", "GCCR_1", "GCCR_all",
    ]
    lines = [
        r"\begin{tabular}{lllrrrrrrrrrrrrr}",
        r"\toprule",
        r"Task & Model & Phase & $|J|$ & $U(J)$ & $s_{(1)}$ & $\mathrm{TOC}_1$ & $R_{\mathrm{ov}}$ & $N_{\mathrm{eff}}$ & $\mathrm{OCC}_0$ & $\mathrm{OCC}_1$ & $N_{.05}$ & $N_{.10}$ & $E(J)$ & $\mathrm{GCCR}_1$ \\",
        r"\midrule",
    ]
    for row in frame.to_dict("records"):
        cells = [str(row.get("task", "")), str(row.get("model", "")), str(row.get("phase", ""))]
        cells.append("--" if pd.isna(row.get("J")) else str(int(float(row["J"]))))
        cells.extend(format_value(row.get(column)) for column in columns[4:])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def matched_null_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one manuscript row per setting and simultaneous-null metric."""
    def status_value(value: object) -> str:
        if value is None:
            return "missing"
        try:
            if pd.isna(value):
                return "missing"
        except Exception:
            pass
        text = str(value).strip()
        return "missing" if text.lower() in {"", "nan", "none"} else text

    rows: list[dict] = []
    gccr_pattern = __import__("re").compile(r"^GCCR_(\d+|all)$")
    gccr_ms = []
    for column in frame.columns:
        match = gccr_pattern.match(str(column))
        if match is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        status_column = f"{column}_status"
        has_status = (
            status_column in frame.columns
            and frame[status_column].astype(str).str.lower().notna().any()
            and (~frame[status_column].astype(str).str.lower().isin({"missing", "nan", "none", ""})).any()
        )
        if values.notna().any() or has_status:
            gccr_ms.append(normalize_interaction_m(match.group(1)))
    gccr_ms = sorted(set(gccr_ms), key=lambda value: (value == "all", int(value) if value != "all" else 10**9))
    for row in frame.to_dict("records"):
        identity = {
            "task": row.get("task"),
            "model": row.get("model"),
            "phase": row.get("phase"),
        }
        rows.append(
            {
                **identity,
                "metric": "E(J)",
                "m": math.nan,
                "candidate": row.get("E_J"),
                "median_null": row.get("E_J_null_median"),
                "Delta": row.get("E_J_Delta"),
                "P": row.get("E_J_P"),
                "p_MC": row.get("E_J_p_MC"),
                "null_draws_requested": row.get("E_J_null_B"),
                "null_draws_finite": row.get("E_J_null_finite"),
                "candidate_status": status_value(row.get("E_J_status")),
                "null_status": status_value(row.get("E_J_null_status")),
            }
        )
        for m in gccr_ms:
            rows.append(
                {
                    **identity,
                    "metric": "GCCR",
                    "m": m,
                    "candidate": row.get(f"GCCR_{m}"),
                    "median_null": row.get(f"GCCR_{m}_null_median"),
                    "Delta": row.get(f"GCCR_{m}_Delta"),
                    "P": row.get(f"GCCR_{m}_P"),
                    "p_MC": row.get(f"GCCR_{m}_p_MC"),
                    "null_draws_requested": row.get(f"GCCR_{m}_null_B"),
                    "null_draws_finite": row.get(f"GCCR_{m}_null_finite"),
                    "candidate_status": status_value(row.get(f"GCCR_{m}_status")),
                    "null_status": status_value(row.get(f"GCCR_{m}_null_status")),
                }
            )
    return pd.DataFrame(rows)


def null_latex_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llllrrrrrrrl}",
        r"\toprule",
        "Task & Model & Phase & Metric & $m$ & Candidate & Null median & $\\Delta$ & $P$ & $p_{\\mathrm{MC}}$ & $B$ & Status " + chr(92) * 2,
        r"\midrule",
    ]
    for row in frame.to_dict("records"):
        raw_m = normalize_interaction_m(row.get("m"))
        m_text = "--" if raw_m is None else str(raw_m)
        b_text = format_value(row.get("null_draws_requested"), 0)
        status = f"{row.get('candidate_status', 'missing')}/{row.get('null_status', 'missing')}"
        cells = [
            str(row.get("task", "")),
            str(row.get("model", "")),
            str(row.get("phase", "")),
            str(row.get("metric", "")),
            m_text,
            format_value(row.get("candidate")),
            format_value(row.get("median_null")),
            format_value(row.get("Delta")),
            format_value(row.get("P")),
            format_value(row.get("p_MC")),
            b_text,
            status.replace("_", r"\_"),
        ]
        lines.append(" & ".join(cells) + " " + chr(92) * 2)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def status_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task", "model", "phase", "R_ov_status", "N_eff_status",
        "OCC_0_status", "OCC_1_status", "E_J_status", "E_J_null_status",
        "GCCR_1_status", "GCCR_1_null_status", "GCCR_all_status", "GCCR_all_null_status",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = "missing"
    return frame[columns].copy()


def status_latex_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lllllllllllll}",
        r"\toprule",
        r"Task & Model & Phase & $R_{\mathrm{ov}}$ & $N_{\mathrm{eff}}$ & $\mathrm{OCC}_0$ & $\mathrm{OCC}_1$ & $E(J)$ & Null $E$ & $\mathrm{GCCR}_1$ & Null GCCR \\",
        r"\midrule",
    ]
    for row in frame.to_dict("records"):
        fields = [
            "task", "model", "phase", "R_ov_status", "N_eff_status",
            "OCC_0_status", "OCC_1_status", "E_J_status", "E_J_null_status",
            "GCCR_1_status", "GCCR_1_null_status", "GCCR_all_status", "GCCR_all_null_status",
        ]
        cells = [str(row.get(field, "missing")).replace("_", r"\_") for field in fields]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def plot_metric(frame: pd.DataFrame, metric: str, out_dir: Path) -> None:
    if "score" not in frame.columns or metric not in frame.columns:
        return
    x = pd.to_numeric(frame["score"], errors="coerce")
    y = pd.to_numeric(frame[metric], errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(x[mask], y[mask], s=26)
    ax.set_xlabel("Task score")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ')} across primary settings")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    stem = out_dir / f"score_vs_{metric}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table_path = Path(args.primary_table).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized, excluded, audit = normalize_primary_table(
        pd.read_csv(table_path), profile=args.primary_profile, source=table_path
    )
    write_normalization_audit(normalized, excluded, audit, out_dir, stem="primary_table")
    rows = [
        augment_row(row, resolve_stats_dir(row["stats_dir"], data_root))
        for _, row in normalized.iterrows()
    ]
    augmented = pd.DataFrame(rows)
    augmented.to_csv(out_dir / "manuscript_metrics.csv", index=False)
    (out_dir / "manuscript_metrics.json").write_text(
        json.dumps(
            {
                "primary_profile": args.primary_profile,
                "setting_count": len(augmented),
                "profile_audit": audit,
                "rows": rows,
                "notes": [
                    "No R_ov, E(J), or GCCR value is clipped.",
                    "TOC_m uses discovery-frozen H_m when singleton_set_metrics.json is available.",
                    "E(J) and GCCR require simultaneous-intervention outputs.",
                ],
            },
            indent=2,
            allow_nan=True,
            default=str,
        ),
        encoding="utf-8",
    )
    (out_dir / "manuscript_metrics.tex").write_text(latex_table(augmented), encoding="utf-8")
    matched_nulls = matched_null_long(augmented)
    matched_nulls.to_csv(out_dir / "matched_null_metrics.csv", index=False)
    (out_dir / "matched_null_metrics.json").write_text(
        json.dumps(matched_nulls.to_dict("records"), indent=2, allow_nan=True, default=str),
        encoding="utf-8",
    )
    (out_dir / "matched_null_metrics.tex").write_text(
        null_latex_table(matched_nulls), encoding="utf-8"
    )
    statuses = status_table(augmented)
    statuses.to_csv(out_dir / "metric_statuses.csv", index=False)
    (out_dir / "metric_statuses.tex").write_text(status_latex_table(statuses), encoding="utf-8")

    for metric in ["U_J", "TOC_1", "R_ov", "N_eff", "E_J", "GCCR_1", "GCCR_all"]:
        plot_metric(augmented, metric, out_dir)
    print(f"Wrote manuscript outputs to {out_dir}")


if __name__ == "__main__":
    main()
