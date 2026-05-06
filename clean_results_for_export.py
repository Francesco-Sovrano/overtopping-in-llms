#!/usr/bin/env python3
"""
Copy a results/data directory into a fresh filtered results directory, keeping only
file/path templates that are covered by the embedded results schema.
The run/config folder directly under stats/ is treated as dynamic, so any
stats/<run-name>/... folder can match an approved final-output file template.

The embedded schema was extracted from the provided results.zip, so the script can
run without passing a reference zip. You may still pass --reference-zip to override
or refresh the schema for a different reference directory.

The script always uses template matching, so dynamic task/provider/model names,
module ids, unit ids, stats run/config folders, and cache hashes may differ from the examples in the schema.

Always excluded, even when present in the schema:
  - pickle files (*.pkl)
  - cache folders, including cache, ablation_cache, spectral_cache, and any *_cache
  - any folder whose name starts with is_correct_
  - macOS metadata files and folders
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

JUNK_DIRS = {"__MACOSX"}
JUNK_FILES = {".DS_Store"}
EXCLUDED_SUFFIXES = {".pkl"}
EXCLUDED_DIR_NAMES = {"cache"}
EXCLUDED_DIR_NAME_SUFFIXES = ("_cache",)
EXCLUDED_DIR_NAME_PREFIXES = ("is_correct_",)
STRUCTURAL_DIRS = {"feature_report", "rule_extraction_results"}
STATS_RUN_PLACEHOLDER = "<stats_run>"

MODULE_RE = re.compile(r"m\d+_\d+$")
MODULE_UNIT_PKL_RE = re.compile(r"m\d+_\d+_\d+\.pkl$")
REPLACEMENT_SCORES_RE = re.compile(r"replacement_scores_[0-9a-fA-F]+\.pkl$")
SPECTRAL_RE = re.compile(r"spectral_[0-9a-fA-F]+\.pkl$")
FLIP_STATS_TOP_RE = re.compile(r"flip_stats_top\d+\.pdf$")
MODULE_NAMED_CSV_RE = re.compile(
    r"^(association_rules_flip|rule_combo_flip|optimal_rule_set_flip)"
    r"(_(?:c2i|i2c))?_m\d+_\d+\.csv$"
)
SHAP_SUMMARY_RE = re.compile(
    r"^(shap_summary_plot_flip(?:_(?:c2i|i2c))?)_m\d+_\d+\.png$"
)

# Embedded schema generated from the supplied results.zip, after removing
# pickles, cache folders, and is_correct_* intermediate folders.
# Dynamic path segments are already represented as placeholders here, so the
# visible schema is the same one used for matching.
EMBEDDED_TEMPLATE_PATHS = frozenset([
    '<task>/<provider>/<model>/feature_report/dataset_stats.json',
    '<task>/<provider>/<model>/feature_report/features.json',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_ablation_scatter.png',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_best_correlations.csv',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_correlation_heatmap.png',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_correlations.csv',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_delta_by_unit.csv',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_final_plots.pdf',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_global_summary.csv',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_percentile_boxplot.png',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_stats.json',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_summary.csv',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/agonist_metric_top_rates.png',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/flip_stats_by_neuron.csv',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/flip_stats_global.json',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/flip_stats_top<num>.pdf',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/high_quality_neuron_flip_coverage.json',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/high_quality_neuron_flip_coverage.pdf',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/high_quality_neuron_flip_coverage.tex',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/high_quality_neuron_flip_coverage_by_layer.json',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/high_quality_neuron_flip_coverage_by_layer.pdf',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/rule_combo_metrics_all.csv',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/rule_combo_metrics_best_per_neuron.csv',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/rule_combo_metrics_top_50.csv',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/rule_metrics.tex',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/rule_metrics_distributions.pdf',
    '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/rule_metrics_summary.json',
    # '<task>/<provider>/<model>/rule_extraction_results/neuron_flip_rules/stats/<stats_run>/scores.csv',
])


def is_junk_parts(parts: Iterable[str]) -> bool:
    for part in parts:
        if part in JUNK_DIRS or part in JUNK_FILES or part.startswith("._"):
            return True
    return False


def is_excluded_parts(parts: Iterable[str]) -> bool:
    for part in parts:
        lowered = part.lower()
        if lowered in EXCLUDED_DIR_NAMES:
            return True
        if any(lowered.endswith(suffix) for suffix in EXCLUDED_DIR_NAME_SUFFIXES):
            return True
        if any(lowered.startswith(prefix) for prefix in EXCLUDED_DIR_NAME_PREFIXES):
            return True
    return False


def strip_results_root(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Treat both 'results/foo/...' and 'foo/...' as the same schema path."""
    if parts and parts[0] == "results":
        return parts[1:]
    return parts


def normalize_zip_member(name: str) -> str | None:
    p = PurePosixPath(name)
    parts = strip_results_root(p.parts)
    if not parts or is_junk_parts(parts) or is_excluded_parts(parts):
        return None
    if p.suffix.lower() in EXCLUDED_SUFFIXES:
        return None
    return PurePosixPath(*parts).as_posix()


def normalize_input_path(path: Path, root: Path) -> str | None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = strip_results_root(tuple(rel.parts))
    if not parts or is_junk_parts(parts) or is_excluded_parts(parts):
        return None
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return None
    return PurePosixPath(*parts).as_posix()


def generalize_component(part: str) -> str:
    if MODULE_RE.fullmatch(part):
        return "<module>"
    if MODULE_UNIT_PKL_RE.fullmatch(part):
        return "<module>_<unit>.pkl"
    if REPLACEMENT_SCORES_RE.fullmatch(part):
        return "replacement_scores_<hash>.pkl"
    if SPECTRAL_RE.fullmatch(part):
        return "spectral_<hash>.pkl"
    if FLIP_STATS_TOP_RE.fullmatch(part):
        return "flip_stats_top<num>.pdf"

    m = MODULE_NAMED_CSV_RE.fullmatch(part)
    if m:
        prefix, variant = m.group(1), m.group(2) or ""
        return f"{prefix}{variant}_<module>.csv"

    m = SHAP_SUMMARY_RE.fullmatch(part)
    if m:
        return f"{m.group(1)}_<module>.png"

    return part


def template_key(rel_posix: str) -> str:
    """
    Convert a relative results path into a schema key.

    Examples:
      arithmetic/Qwen/MODEL/feature_report/features.json
        -> <task>/<provider>/<model>/feature_report/features.json
      .../stats/my_run_name/scores.csv
        -> .../stats/<stats_run>/scores.csv
      .../m4_1467/association_rules_flip_m4_1467.csv
        -> .../<module>/association_rules_flip_<module>.csv
    """
    parts = strip_results_root(PurePosixPath(rel_posix).parts)

    # The reference tree consistently uses: task/provider/model/<result-kind>/...
    # Wildcard these first three levels so the schema can be reused on another run.
    if len(parts) >= 4 and parts[3] in STRUCTURAL_DIRS:
        parts = ("<task>", "<provider>", "<model>") + parts[3:]

    # Run/config directory names under stats are dynamic. Keep the approved file
    # names under stats, but do not tie them to any one exact experiment folder.
    if "stats" in parts:
        stats_idx = parts.index("stats")
        if len(parts) > stats_idx + 2:
            parts = (
                parts[: stats_idx + 1]
                + (STATS_RUN_PLACEHOLDER,)
                + parts[stats_idx + 2 :]
            )

    return PurePosixPath(*(generalize_component(p) for p in parts)).as_posix()


def reference_paths(reference_zip: Path) -> list[str]:
    paths: list[str] = []
    with zipfile.ZipFile(reference_zip) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = normalize_zip_member(info.filename)
            if rel is not None:
                paths.append(rel)
    return paths


def is_excluded_path(rel_posix: str) -> bool:
    p = PurePosixPath(rel_posix)
    return p.suffix.lower() in EXCLUDED_SUFFIXES or is_excluded_parts(p.parts)


def without_excluded_paths(paths: Iterable[str]) -> set[str]:
    return {p for p in paths if not is_excluded_path(p)}


def build_allowed(reference_zip: Path | None) -> tuple[set[str], str]:
    if reference_zip is None:
        return without_excluded_paths(EMBEDDED_TEMPLATE_PATHS), "embedded template schema"

    refs = reference_paths(reference_zip)
    return without_excluded_paths(template_key(p) for p in refs), str(reference_zip)


def safe_prepare_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Pass --force to replace it, or choose a different output directory."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_filtered(
    data_dir: Path,
    output_dir: Path,
    reference_zip: Path | None,
    dry_run: bool,
    force: bool,
    verbose: bool,
    manifest_path: Path | None,
) -> int:
    if not data_dir.is_dir():
        raise SystemExit(f"Input data directory does not exist or is not a directory: {data_dir}")
    if reference_zip is not None and not reference_zip.is_file():
        raise SystemExit(f"Reference zip does not exist or is not a file: {reference_zip}")

    allowed, schema_source = build_allowed(reference_zip)
    kept: list[str] = []
    skipped: list[str] = []

    if not dry_run:
        safe_prepare_output(output_dir, force=force)

    for src in sorted(data_dir.rglob("*")):
        if not src.is_file():
            continue

        rel = normalize_input_path(src, data_dir)
        if rel is None:
            try:
                skipped.append(str(src.relative_to(data_dir)))
            except ValueError:
                skipped.append(str(src))
            continue

        key = template_key(rel)
        if key not in allowed:
            skipped.append(rel)
            if verbose:
                print(f"skip: {rel}  [key: {key}]", file=sys.stderr)
            continue

        kept.append(rel)
        if verbose:
            print(f"keep: {rel}", file=sys.stderr)
        if not dry_run:
            dst = output_dir / Path(*PurePosixPath(rel).parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    manifest = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "schema_source": schema_source,
        "allowed_template_count": len(allowed),
        "kept_count": len(kept),
        "skipped_count": len(skipped),
        "kept": kept,
        "skipped": skipped,
    }

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Schema source: {schema_source}")
    print(f"Allowed template patterns: {len(allowed)}")
    print(f"Kept files: {len(kept)}")
    print(f"Skipped files: {len(skipped)}")
    if dry_run:
        print("Dry run only: no files were copied.")
    else:
        print(f"Filtered results written to: {output_dir}")
    if manifest_path is not None:
        print(f"Manifest written to: {manifest_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy DATA_DIR into a fresh filtered results directory, keeping only "
            "approved result file/path templates. The stats run/config folder is "
            "generic, while pickle files, caches, and is_correct_* folders are always excluded."
        )
    )
    parser.add_argument("data_dir", type=Path, help="Directory to filter/copy.")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Destination directory. Defaults to DATA_DIR_filtered_results next to DATA_DIR.",
    )
    parser.add_argument(
        "--reference-zip",
        type=Path,
        default=None,
        help="Optional reference zip to override the embedded schema.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print counts without copying anything.")
    parser.add_argument("--force", action="store_true", help="Replace output_dir if it already exists.")
    parser.add_argument("--verbose", action="store_true", help="Print every kept/skipped file to stderr.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON file listing kept and skipped relative paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_dir.with_name(f"{data_dir.name}_filtered_results")
    )
    return copy_filtered(
        data_dir=data_dir,
        output_dir=output_dir,
        reference_zip=args.reference_zip.resolve() if args.reference_zip else None,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
        manifest_path=args.manifest.resolve() if args.manifest else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
