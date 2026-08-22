#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from lib.project_paths import PROJECT_ROOT


import argparse
import json
import math
import os
import re
import shutil
from functools import lru_cache

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover the scores/candidate artifact pair that reproduces each "
            "authoritative primary_table.csv row, then write exact directional "
            "union counts and eligible denominators."
        )
    )
    parser.add_argument(
        "--primary-table",
        default=str(PROJECT_ROOT / "results" / "paper_tables" / "primary_table.csv"),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def resolve_stats_dir(root: Path, raw: object) -> Path:
    path = Path(str(raw)).expanduser()
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(root / path)

    text = path.as_posix()
    for marker in ("/results/", "/data/"):
        if marker in text:
            suffix = Path(text.split(marker, 1)[1])
            candidates.extend((root / "data" / suffix, root / "results" / suffix))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve stats_dir {raw!r}. Tried:\n"
        + "\n".join(f"  {candidate}" for candidate in candidates)
    )


def safe_layer_key(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")


def artifact_candidates(current: Path) -> list[Path]:
    candidates = [current]
    candidates.extend(sorted(current.parent.glob(current.name + ".before*")))
    candidates.extend(sorted(current.parent.glob(current.name + ".*")))

    output: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.name.endswith(".tmp"):
            continue
        resolved = candidate.resolve()
        if resolved not in seen:
            output.append(resolved)
            seen.add(resolved)
    return output


@lru_cache(maxsize=None)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def candidate_columns(
    scores: pd.DataFrame,
    neurons: pd.DataFrame,
    expected_j: int,
) -> list[str] | None:
    if len(neurons) != expected_j:
        return None

    columns: list[str] = []
    for _, neuron in neurons.iterrows():
        layer_key = neuron.get("layer_key")
        if pd.isna(layer_key) or not str(layer_key).strip():
            layer_key = safe_layer_key(neuron.get("layer_label", ""))

        neuron_id = int(neuron["neuron_id"])
        column = f"flip_{layer_key}_{neuron_id}"
        if column not in scores.columns:
            return None
        columns.append(column)

    return list(dict.fromkeys(columns))


def evaluate_pair(
    scores_path: Path,
    neurons_path: Path,
    row: pd.Series,
) -> dict | None:
    scores = read_csv(str(scores_path))
    neurons = read_csv(str(neurons_path))
    columns = candidate_columns(scores, neurons, int(row["J"]))
    if not columns:
        return None

    union = np.zeros(len(scores), dtype=bool)
    evaluated = np.zeros(len(scores), dtype=bool)
    strengths: list[float] = []

    for column in columns:
        series = scores[column]
        valid = series.notna().to_numpy()
        values = series.fillna(False).astype(bool).to_numpy()

        evaluated |= valid
        union |= values

        n = int(valid.sum())
        strengths.append(float(values.sum() / n) if n else math.nan)

    n_eval = int(evaluated.sum())
    union &= evaluated
    u_value = float(union.sum() / n_eval) if n_eval else math.nan

    finite_strengths = np.asarray(
        [value for value in strengths if np.isfinite(value)],
        dtype=float,
    )
    top = float(finite_strengths.max()) if len(finite_strengths) else math.nan

    return {
        "scores_path": scores_path,
        "neurons_path": neurons_path,
        "scores": scores,
        "neurons": neurons,
        "columns": columns,
        "evaluated": evaluated,
        "union": union,
        "n_eval": n_eval,
        "U": u_value,
        "Top": top,
        "N05": int((finite_strengths >= 0.05).sum()),
        "N10": int((finite_strengths >= 0.10).sum()),
    }


def matches_primary(metrics: dict, row: pd.Series, tolerance: float) -> bool:
    checks = [
        metrics["n_eval"] == int(row["n_eval"]),
        abs(metrics["U"] - float(row["U"])) <= tolerance,
        abs(metrics["Top"] - float(row["Top"])) <= tolerance,
        metrics["N05"] == int(row["N05"]),
        metrics["N10"] == int(row["N10"]),
    ]
    return all(checks)


def baseline_column(scores: pd.DataFrame, task: object) -> str:
    preferred = (
        ["is_jailbroken", "is_correct"]
        if "jailbreak" in str(task).lower()
        else ["is_correct", "is_jailbroken"]
    )
    for column in preferred:
        if column in scores.columns:
            return column
    raise ValueError(
        f"Cannot find the binary baseline column. Tried {preferred}."
    )


def write_directional_summary(
    *,
    stats_dir: Path,
    row: pd.Series,
    metrics: dict,
) -> None:
    scores = metrics["scores"]
    evaluated = metrics["evaluated"]
    union = metrics["union"]

    base_column = baseline_column(scores, row["task"])
    baseline = pd.to_numeric(scores[base_column], errors="coerce")

    if baseline.loc[evaluated].isna().any():
        raise ValueError(
            f"{stats_dir}: missing {base_column} on evaluated rows."
        )

    positive = (baseline.to_numpy(dtype=float) > 0.5)
    p2n = union & evaluated & positive
    n2p = union & evaluated & ~positive

    n_eval = int(evaluated.sum())
    n_positive = int((evaluated & positive).sum())
    n_negative = int((evaluated & ~positive).sum())
    any_count = int(union.sum())
    p2n_count = int(p2n.sum())
    n2p_count = int(n2p.sum())

    if p2n_count + n2p_count != any_count:
        raise AssertionError(
            f"{stats_dir}: directional counts do not partition pooled U."
        )

    p2n_rate = p2n_count / n_positive if n_positive else None
    n2p_rate = n2p_count / n_negative if n_negative else None

    global_path = stats_dir / "flip_stats_global.json"
    payload = (
        json.loads(global_path.read_text(encoding="utf-8"))
        if global_path.exists()
        else {}
    )

    backup = global_path.with_name(
        "flip_stats_global.json.before_primary_artifact_recovery"
    )
    if global_path.exists() and not backup.exists():
        shutil.copy2(global_path, backup)

    payload.update(
        {
            "n_neurons": int(row["J"]),
            "n_evaluated_rows": n_eval,
            "union_flip_any_unique_count": any_count,
            "union_flip_any_unique_rate": metrics["U"],
            "baseline_metric_col": base_column,
            "n_eval_baseline_valid": n_eval,
            "n_eval_baseline_correct": n_positive,
            "n_eval_baseline_incorrect": n_negative,
            "union_c2i_unique_count": p2n_count,
            "union_c2i_eligible_n": n_positive,
            "union_c2i_unique_rate": p2n_rate,
            "union_c2i_direction_conditioned_rate": p2n_rate,
            "union_c2i_unconditional_rate": p2n_count / n_eval,
            "union_i2c_unique_count": n2p_count,
            "union_i2c_eligible_n": n_negative,
            "union_i2c_unique_rate": n2p_rate,
            "union_i2c_direction_conditioned_rate": n2p_rate,
            "union_i2c_unconditional_rate": n2p_count / n_eval,
            "union_positive_to_negative_unique_count": p2n_count,
            "union_positive_to_negative_eligible_n": n_positive,
            "union_positive_to_negative_unique_rate": p2n_rate,
            "union_negative_to_positive_unique_count": n2p_count,
            "union_negative_to_positive_eligible_n": n_negative,
            "union_negative_to_positive_unique_rate": n2p_rate,
        }
    )

    temporary = global_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, global_path)

    print(
        f"  direction: 1->0 {p2n_count}/{n_positive}="
        f"{p2n_rate if p2n_rate is not None else float('nan'):.8f}; "
        f"0->1 {n2p_count}/{n_negative}="
        f"{n2p_rate if n2p_rate is not None else float('nan'):.8f}"
    )


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    table_path = Path(args.primary_table).expanduser()
    if not table_path.is_absolute():
        table_path = root / table_path

    table = pd.read_csv(table_path)
    required = {"stats_dir", "task", "J", "U", "Top", "N05", "N10", "n_eval"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{table_path} is missing columns: {sorted(missing)}"
        )

    plans: list[tuple[pd.Series, Path, dict]] = []
    unresolved: list[str] = []

    for row_index, row in table.iterrows():
        stats_dir = resolve_stats_dir(root, row["stats_dir"])
        current_scores = stats_dir / "scores.csv"
        current_neurons = stats_dir / "flip_stats_by_neuron.csv"

        matches: list[dict] = []
        attempted: list[dict] = []

        for scores_path in artifact_candidates(current_scores):
            for neurons_path in artifact_candidates(current_neurons):
                try:
                    metrics = evaluate_pair(scores_path, neurons_path, row)
                except Exception:
                    continue
                if metrics is None:
                    continue

                attempted.append(metrics)
                if matches_primary(metrics, row, args.tolerance):
                    matches.append(metrics)

        if not matches:
            best = sorted(
                attempted,
                key=lambda item: (
                    abs(item["U"] - float(row["U"])),
                    abs(item["Top"] - float(row["Top"])),
                    abs(item["n_eval"] - int(row["n_eval"])),
                ),
            )[:5]

            details = [
                f"row {row_index}: {stats_dir}",
                (
                    f"  expected n={int(row['n_eval'])}, U={float(row['U']):.8f}, "
                    f"Top={float(row['Top']):.8f}, N05={int(row['N05'])}, "
                    f"N10={int(row['N10'])}"
                ),
                "  closest artifact combinations:",
            ]
            for candidate in best:
                details.append(
                    f"    scores={candidate['scores_path'].name}; "
                    f"neurons={candidate['neurons_path'].name}; "
                    f"n={candidate['n_eval']}; U={candidate['U']:.8f}; "
                    f"Top={candidate['Top']:.8f}; "
                    f"N05={candidate['N05']}; N10={candidate['N10']}"
                )
            unresolved.append("\n".join(details))
            continue

        # Prefer the current pair, then the pair with the shortest backup names.
        matches.sort(
            key=lambda item: (
                item["scores_path"] != current_scores.resolve(),
                item["neurons_path"] != current_neurons.resolve(),
                len(item["scores_path"].name) + len(item["neurons_path"].name),
            )
        )
        selected = matches[0]
        plans.append((row, stats_dir, selected))

        print(f"[match] row {row_index}: {stats_dir}")
        print(f"  scores:  {selected['scores_path'].name}")
        print(f"  neurons: {selected['neurons_path'].name}")
        print(
            f"  n={selected['n_eval']}; U={selected['U']:.8f}; "
            f"Top={selected['Top']:.8f}; "
            f"N05={selected['N05']}; N10={selected['N10']}"
        )

    if unresolved:
        print("\nNo exact authoritative artifact match was found for:")
        print("\n\n".join(unresolved))
        raise SystemExit(
            "\nNothing was modified. Recover an untouched scores.csv and "
            "flip_stats_by_neuron.csv for the unresolved rows, or rerun those "
            "rows with their exact original configuration."
        )

    print(f"\nExact authoritative matches found for all {len(plans)} rows.")

    if not args.apply:
        print("Dry run only. Re-run with --apply to restore and write directions.")
        return

    for row, stats_dir, selected in plans:
        current_scores = stats_dir / "scores.csv"
        current_neurons = stats_dir / "flip_stats_by_neuron.csv"

        for selected_path, current_path in (
            (selected["scores_path"], current_scores),
            (selected["neurons_path"], current_neurons),
        ):
            if selected_path.resolve() == current_path.resolve():
                continue
            backup = current_path.with_name(
                current_path.name + ".before_primary_artifact_recovery"
            )
            if current_path.exists() and not backup.exists():
                shutil.copy2(current_path, backup)
            shutil.copy2(selected_path, current_path)

        # Re-read the restored pair so the written summary matches disk.
        restored = evaluate_pair(current_scores, current_neurons, row)
        if restored is None or not matches_primary(
            restored, row, args.tolerance
        ):
            raise RuntimeError(
                f"{stats_dir}: restored artifacts failed validation."
            )

        write_directional_summary(
            stats_dir=stats_dir,
            row=row,
            metrics=restored,
        )

    print("\nRecovery and directional summaries completed.")


if __name__ == "__main__":
    main()
