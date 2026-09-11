#!/usr/bin/env python3
"""Generate all final paper statistics and figures under the repository results/ tree.

This module is intentionally orchestration-only. Numerical/statistical logic remains
in the dedicated analysis scripts that it invokes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil
import zipfile

import pandas as pd

from core.project_paths import CODE_ROOT, PROJECT_ROOT


from studies.overtopping.analysis.lib.primary_matrix import PRIMARY_PROFILE_CHOICES, PROFILE_STUDY_48
from reporting.result_paths import (
    analysis_root,
    appendix_figures,
    experiment_catalogue,
    figure_data,
    table_data,
    manuscript_figures,
    manuscript_materials,
    metric_completeness_audit,
    overtopping_spiking_diagnostics,
    rq2_interaction_decomposition,
    rq2_interaction_decomposition_mean,
    rq2_regime_summary,
    paper_root,
    poisoning_aggregate_tables,
    poisoning_figures,
    poisoning_results,
    primary_statistics,
    primary_tables,
    rq1_figures,
    rq2_figures,
    rq3_figures,
    rq4_figures,
)
from studies.poisoning.lib.run_paths import TRAJECTORIES_DIRNAME, TRAINING_DIRNAME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data"), help="Experiment-artifact root. Default: <repo>/data")
    p.add_argument("--results-root", default=str(PROJECT_ROOT / "results"), help="Final-output root. Default: <repo>/results")
    p.add_argument(
        "--primary-profile", default=PROFILE_STUDY_48, choices=PRIMARY_PROFILE_CHOICES,
        help="Validation profile for the configured study table. Default: study-48.",
    )
    p.add_argument(
        "--poisoning-root",
        dest="poisoning_root",
        default=str(PROJECT_ROOT / "data" / "poisoning"),
        help="Poisoning data root. Default: <repo>/data/poisoning",
    )
    p.add_argument(
        "--catalogue-json",
        default=None,
        help="Optional configured_experiments.json. If present, catalogue plots are refreshed.",
    )
    p.add_argument("--skip-paper-figures", action="store_true")
    p.add_argument("--skip-spiking-report", action="store_true")
    p.add_argument(
        "--spiking-max-points",
        type=int,
        default=10000,
        help="Maximum plotted RQ3 points read from completed diagnostics; never triggers experiment computation.",
    )
    p.add_argument(
        "--spiking-source",
        default=None,
        help=(
            "Optional RQ3 threshold/spiking diagnostics source (directory or zip). "
            "If omitted, the generator searches data/ and common project-root archive names."
        ),
    )
    p.add_argument("--skip-poisoning-report", action="store_true")
    p.add_argument(
        "--require-complete-metrics", action="store_true",
        help=(
            "Fail if any primary setting lacks the exact metrics required by the selected audit mode. "
            "An audit is always written under results/analysis/reproducibility/metric_completeness_audit/."
        ),
    )
    p.add_argument(
        "--skip-cmc-requirement", action="store_true",
        help=(
            "Do not require CMC or its paired conditional null for completeness. "
            "Directional singleton metrics plus simultaneous E(J) and its matched null remain required."
        ),
    )
    return p.parse_args()


def run(command: list[str], *, allow_failure: bool = False) -> bool:
    print("[final-results]", " ".join(command))
    try:
        subprocess.run(command, cwd=CODE_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        if not allow_failure:
            raise
        print(
            f"[final-results] WARNING: reporting stage exited {exc.returncode} while the "
            "configured experiment population is incomplete; continuing with other available outputs.",
            flush=True,
        )
        return False
    return True


def validate_reporting_output_root(*, data_root: Path, results_root: Path) -> None:
    """Require reporting outputs to be disjoint from persistent scientific storage."""
    data_root = data_root.resolve()
    results_root = results_root.resolve()
    cache_root = (PROJECT_ROOT / "cache").resolve()

    def _is_within(path: Path, parent: Path) -> bool:
        return path == parent or parent in path.parents

    if _is_within(results_root, data_root):
        raise ValueError(
            f"Reporting output must be outside the data root: results={results_root}, data={data_root}"
        )
    if _is_within(results_root, cache_root):
        raise ValueError(
            f"Reporting output must be outside the cache root: results={results_root}, cache={cache_root}"
        )


def _zip_has_spiking_payload(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            names = [name.lstrip("./") for name in zf.namelist()]
    except (OSError, zipfile.BadZipFile):
        return False
    # aggregate_flip_stats.csv is the population-completeness authority.
    # aggregate_unit_tests.csv is conditional on threshold-testability and may
    # legitimately be absent even when completed RQ3 flip diagnostics exist.
    return any(name.endswith("aggregate_flip_stats.csv") for name in names)


def _root_has_spiking_payload(root: Path) -> bool:
    if not root.is_dir():
        return False
    # Search for the actual inputs consumed by stage07 rather than relying on a
    # particular directory name. Historical runs and capped runs used several
    # layouts, while the CSV contract is stable.
    return next(root.rglob("aggregate_flip_stats.csv"), None) is not None


def resolve_spiking_source(explicit: str | None, data_root: Path) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(data_root)
    for name in (
        "spiking_diagnostics_results_for_inspection.zip",
        "spiking_diagnostics.zip",
        "threshold_spiking_diagnostics.zip",
    ):
        candidates.append(PROJECT_ROOT / name)
        candidates.append(PROJECT_ROOT / "data" / name)
    # Also accept a uniquely matching diagnostics archive in the project root.
    candidates.extend(sorted(PROJECT_ROOT.glob("*spiking*diagnostic*.zip")))
    candidates.extend(sorted(PROJECT_ROOT.glob("*threshold*diagnostic*.zip")))

    seen: set[Path] = set()
    for raw in candidates:
        path = raw.resolve() if raw.exists() else raw
        if path in seen:
            continue
        seen.add(path)
        if path.is_file() and path.suffix.lower() == ".zip" and _zip_has_spiking_payload(path):
            return path
        if path.is_dir() and _root_has_spiking_payload(path):
            return path
    return None


def relocate_paper_sidecars(results_root: Path) -> None:
    """Move machine sidecars out of ``paper/`` and into ``analysis/``.

    The manuscript-facing contract reserves ``paper/figures`` and
    ``paper/tables`` for human-facing artifacts. CSV/JSON sidecars are copied
    to the corresponding analysis tree and then removed from the paper view.
    """

    def _move_sidecars(src_root: Path, dst_root: Path) -> int:
        moved = 0
        if not src_root.is_dir():
            return moved
        # Materialize the list before unlinking while traversing.
        sidecars = list(src_root.rglob("*.csv")) + list(src_root.rglob("*.json"))
        for src in sidecars:
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            src.unlink()
            moved += 1
        return moved

    n_fig = _move_sidecars(manuscript_figures(results_root), figure_data(results_root))
    n_tbl = _move_sidecars(manuscript_materials(results_root), table_data(results_root))
    if n_fig or n_tbl:
        print(
            f"[final-results] relocated machine sidecars out of paper view: "
            f"figures={n_fig}, tables={n_tbl}",
            flush=True,
        )


def poisoning_run_dirs(poisoning_root: Path) -> list[Path]:
    """Return canonical stage-organized poisoning runs only."""
    out: list[Path] = []
    for task in ("arithmetic", "grammar"):
        task_root = poisoning_root / task
        if not task_root.is_dir():
            continue
        for run_dir in sorted(p for p in task_root.iterdir() if p.is_dir()):
            cfg = run_dir / TRAINING_DIRNAME / "metadata" / "run_config.json"
            trajectories = run_dir / TRAJECTORIES_DIRNAME
            if not cfg.is_file() or not any(trajectories.glob("*/backdoor_lift_overtopping_trajectory.csv")):
                continue
            try:
                config = json.loads(cfg.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(config.get("condition", "both")) != "both":
                continue
            out.append(run_dir.resolve())
    return out




def publish_per_run_poisoning_visuals(
    poisoning_runs: list[Path],
    *,
    results_root: Path,
) -> dict[str, int]:
    """Publish per-run poisoning figures grouped by scientific role.

    Stage-08 cross-seed outputs summarize replication but do not preserve every
    checkpoint/channel identity, so per-run figures are published separately.
    Cache-backed story figures are regenerated when their singleton inputs exist.
    This function performs no model inference.
    """
    paper_root_dir = poisoning_figures(results_root) / "per_run"
    analysis_dir = poisoning_results(results_root) / "per_run_visualizations"
    paper_root_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    regenerated = 0
    skipped = 0
    manifest_rows: list[dict[str, str]] = []
    main_text_defense_candidates: list[Path] = []

    def copy_family(
        *,
        src_root: Path,
        dst_root: Path,
        family: str,
        run_dir: Path,
        task: str,
        run_name: str,
        phase_dir: str,
        recursive: bool = True,
        names: set[str] | None = None,
    ) -> int:
        """Copy selected PDFs below src_root while preserving relative structure."""
        if not src_root.is_dir():
            return 0
        paths = sorted(src_root.rglob("*.pdf") if recursive else src_root.glob("*.pdf"))
        if names is not None:
            paths = [path for path in paths if path.name in names]
        n = 0
        for src in paths:
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest_rows.append(
                {
                    "task": task,
                    "run": run_name,
                    "phase": phase_dir,
                    "family": family,
                    "source": str(src),
                    "published": str(dst),
                }
            )
            n += 1
        return n

    for run_dir in poisoning_runs:
        task = run_dir.parent.name
        run_name = run_dir.name
        checkpoint_root = run_dir / "03_checkpoint_causal_discovery"
        for phase_dir in ("prompt_and_generation", "generation_only"):
            phase_present = checkpoint_root.is_dir() and any(
                checkpoint_root.glob(f"*/progress_*/{phase_dir}")
            )
            if not phase_present:
                continue

            analysis_phase = analysis_dir / task / run_name / phase_dir
            paper_phase = paper_root_dir / task / run_name / phase_dir
            analysis_phase.mkdir(parents=True, exist_ok=True)
            if paper_phase.exists():
                shutil.rmtree(paper_phase)
            paper_phase.mkdir(parents=True, exist_ok=True)

            # Cache-only story figures: aggregate development, descriptive channel
            # roles, prospective defense leverage, and complete fixed-union heatmap.
            paired_stage07_root = run_dir / "07_poisoning_example_detection" / phase_dir / "paired_u_j_materialization"
            has_story_singletons = paired_stage07_root.is_dir() and any(
                paired_stage07_root.rglob("endpoint_stats/control/singleton_set_metrics.csv")
            )
            story_dir = analysis_phase / "story"
            if has_story_singletons:
                cmd = [
                    sys.executable, "-m", "studies.poisoning.stage07_plot_overtopping_poisoning_story",
                    "--run_dir", str(run_dir),
                    "--phase_dir", phase_dir,
                    "--eval_intervention", "mean-donor",
                    "--output_dir", str(story_dir),
                ]
                try:
                    subprocess.run(cmd, cwd=CODE_ROOT, check=True)
                    regenerated += 1
                except subprocess.CalledProcessError as exc:
                    print(
                        f"[final-results] per-run poisoning story unavailable for {task}/{run_name}/{phase_dir}: {exc}",
                        flush=True,
                    )
                    skipped += 1
                else:
                    # Figure 6 is drawn by the Stage-07 story script itself.
                    # Record the eligible Grammar main-text source here so the
                    # reporting step promotes the generated PDF, rather than a
                    # manually edited copy, into the manuscript figure set.
                    figure6 = story_dir / "03b_clean_reference_defense_interpretation.pdf"
                    if task == "grammar" and phase_dir == "prompt_and_generation" and figure6.is_file():
                        main_text_defense_candidates.append(figure6)
            else:
                skipped += 1

            # 01 - Behavior: checkpoint-specific clean/poison comparisons plus the
            # trajectory plots emitted by Stage 04.  This can be a large family, so
            # retain the original checkpoint/endpoint subdirectories rather than
            # flattening dozens of identically named rates.pdf files.
            copied += copy_family(
                src_root=run_dir / "04_condition_comparisons" / phase_dir,
                dst_root=paper_phase / "01_behavior" / "checkpoint_comparisons",
                family="behavior_checkpoint_comparisons",
                run_dir=run_dir, task=task, run_name=run_name, phase_dir=phase_dir,
            )

            # 02 - One dynamic behavior/causal dashboard from Stage 05.  It
            # includes only endpoints with materialized data and subsumes the
            # dual-axis and trigger-lift-only trajectory figures.
            copied += copy_family(
                src_root=run_dir / "05_behavior_trajectories" / phase_dir,
                dst_root=paper_phase / "02_behavior_trajectories",
                family="behavior_trajectories",
                run_dir=run_dir, task=task, run_name=run_name, phase_dir=phase_dir,
                recursive=False,
                names={"backdoor_overtopping_dashboard.pdf"},
            )

            # 03 - Causal-role story and the richer Stage-07 mechanism figures.
            copied += copy_family(
                src_root=story_dir,
                dst_root=paper_phase / "03_causal_roles" / "story",
                family="causal_role_story",
                run_dir=run_dir, task=task, run_name=run_name, phase_dir=phase_dir,
                recursive=False,
            )
            interpretation = (
                run_dir / "07_poisoning_example_detection" / phase_dir / "overtopping_interpretation"
            )
            copied += copy_family(
                src_root=interpretation / "01_overtopping_mechanism",
                dst_root=paper_phase / "03_causal_roles" / "mechanism",
                family="causal_role_mechanism",
                run_dir=run_dir, task=task, run_name=run_name, phase_dir=phase_dir,
                recursive=False,
            )

            # 04 - Poison-example detection: publish raw Stage-07 plots and the
            # implication-first overview when each source contains figures.
            detection_phase = run_dir / "07_poisoning_example_detection" / phase_dir
            raw_detection_dst = paper_phase / "04_poison_detection" / "raw_stage07"
            raw_detection_pdfs = sorted(detection_phase.glob("*.pdf")) if detection_phase.is_dir() else []
            if raw_detection_pdfs:
                raw_detection_dst.mkdir(parents=True, exist_ok=True)
                for src in raw_detection_pdfs:
                    dst = raw_detection_dst / src.name
                    shutil.copy2(src, dst)
                    manifest_rows.append({"task":task,"run":run_name,"phase":phase_dir,"family":"poison_detection_raw","source":str(src),"published":str(dst)})
                    copied += 1
            elif raw_detection_dst.is_dir() and not any(raw_detection_dst.iterdir()):
                raw_detection_dst.rmdir()
            copied += copy_family(
                src_root=interpretation / "00_poison_detection_overview",
                dst_root=paper_phase / "04_poison_detection" / "overview",
                family="poison_detection_overview",
                run_dir=run_dir, task=task, run_name=run_name, phase_dir=phase_dir,
                recursive=False,
            )

            # 05 - Update geometry.  Four base geometry views plus two
            # update-energy controls.  Causal concentration is published once, in
            # the mechanism family with an explicit clean-drift comparison.
            copied += copy_family(
                src_root=interpretation / "intermediate" / "base_update_geometry",
                dst_root=paper_phase / "05_update_geometry" / "base_geometry",
                family="update_geometry_base",
                run_dir=run_dir, task=task, run_name=run_name, phase_dir=phase_dir,
                recursive=False,
            )
            copied += copy_family(
                src_root=interpretation / "03_update_geometry_checks",
                dst_root=paper_phase / "05_update_geometry" / "controls",
                family="update_geometry_controls",
                run_dir=run_dir, task=task, run_name=run_name, phase_dir=phase_dir,
                recursive=False,
            )

            # 06 - Descriptive link between mechanistic signals and attack growth.
            copied += copy_family(
                src_root=interpretation / "04_link_to_attack_behavior",
                dst_root=paper_phase / "06_attack_link",
                family="attack_growth_link",
                run_dir=run_dir, task=task, run_name=run_name, phase_dir=phase_dir,
                recursive=False,
            )

            # 07 - Threshold/spiking figures generated inside checkpoint causal
            # outputs.  Keep checkpoint identity in the destination path.  Only the
            # aggregate figure subdirectories are promoted; raw high-N tables stay
            # in data/ and analysis/.
            for fig_dir in sorted(checkpoint_root.rglob("spiking_diagnostics*/figures")):
                try:
                    rel_parent = fig_dir.parent.relative_to(checkpoint_root)
                except ValueError:
                    continue
                copied += copy_family(
                    src_root=fig_dir,
                    dst_root=paper_phase / "07_threshold_event" / rel_parent,
                    family="threshold_event",
                    run_dir=run_dir, task=task, run_name=run_name, phase_dir=phase_dir,
                    recursive=False,
                )

            readme = paper_phase / "README.md"
            readme.write_text(
                "# Per-run poisoning visualizations\n\n"
                "This directory exposes the scientifically valid generated poisoning figure set, grouped by role. Longitudinal figures that require fixed-union materialization are withheld rather than published incompletely.\n\n"
                "## 01_behavior\nCheckpoint-specific clean-vs-poison behavior comparisons from Stage 04.\n\n"
                "## 02_behavior_trajectories\nOne dynamic Stage-05 dashboard containing each available distinct behavior/causal endpoint exactly once.\n\n"
                "## 03_causal_roles\nThe compact checkpoint story plus additional non-redundant mechanism analyses of causal location, persistence, and concentration. Read `story/01_clean_vs_poisoned_overtopping_development.pdf` first, then `story/02_channel_role_reassignment.pdf`. When fixed-union materialization is complete, `story/03_prospective_defense_leverage.pdf` provides a leakage-resistant defense-target screen (Δdef = attack suppression − benign damage; positive values indicate plausible attack-selective defense targets, not yet defense efficacy), and `story/04_clean_vs_poisoned_checkpoint_overtopping.pdf` provides longitudinal fixed-union follow-up. Then inspect "
                "`mechanism/01_where_poisoning_specific_causal_control_moves.pdf`.\n\n"
                "## 04_poison_detection\nRaw poison-ranking diagnostics and the implication-first detection overview.\n\n"
                "## 05_update_geometry\nClean-vs-poison LoRA update geometry and matched/update-energy controls.\n\n"
                "## 06_attack_link\nDescriptive association between mechanistic reorganization and attack acquisition.\n\n"
                "## 07_threshold_event\nCheckpoint-level threshold/spiking figures when threshold-event posthoc was generated.\n\n"
                "Machine-readable source artifacts remain under the corresponding `data/poisoning/...` run.\n",
                encoding="utf-8",
            )

    # Promote the generated clean-reference defense plot to the stable main-text
    # Figure 6 filename.  There must be exactly one eligible fully materialized
    # Grammar story; ambiguity is treated as a publication error rather than
    # silently choosing a run.
    if len(main_text_defense_candidates) == 1:
        src = main_text_defense_candidates[0]
        dst = rq4_figures(results_root) / "rq4_grammar_clean_reference_defense.pdf"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_rows.append({
            "task": "grammar",
            "run": "main_text_fixed_coordinate",
            "phase": "prompt_and_generation",
            "family": "main_text_clean_reference_defense",
            "source": str(src),
            "published": str(dst),
        })
        copied += 1
    elif len(main_text_defense_candidates) > 1:
        raise RuntimeError(
            "Multiple eligible Grammar clean-reference defense figures were generated; "
            "refusing to choose a main-text Figure 6 implicitly."
        )

    manifest_path = analysis_dir / "published_poisoning_visualizations.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return {
        "copied_pdfs": copied,
        "regenerated_story_phases": regenerated,
        "skipped_story_phases": skipped,
        "manifest_rows": len(manifest_rows),
    }


def publish_primary_tables(source: Path, paper_tables: Path) -> None:
    paper_tables.mkdir(parents=True, exist_ok=True)
    mapping = {
        "table1_representative.tex": "table1_representative_directional.tex",
        "table8_primary.tex": "tableS_primary_matrix_directional.tex",
    }
    for src_name, dst_name in mapping.items():
        src = source / src_name
        if src.is_file():
            shutil.copy2(src, paper_tables / dst_name)


def write_results_index(results_root: Path, *, spiking_available: bool, partial_results: bool = False) -> None:
    paper = paper_root(results_root)
    figures = manuscript_figures(results_root)
    tables = manuscript_materials(results_root)
    paper.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    rq3_status = (
        "available (partial configured population)"
        if spiking_available and partial_results
        else "available" if spiking_available
        else "MISSING SOURCE - see Figure 4 README"
    )
    root_text = f"""# Generated results - start here

Use **`paper/` first**. Everything under `analysis/` is supporting data, diagnostics, or reproducibility material.

## Paper

- `paper/figures/02_rq1_prevalence/` - **RQ1**: directional reach and width-normalized high-effect density using the canonical all-settings phase-panel population and layout.
- `paper/figures/03_rq2_composition/` - **RQ2**: composition gap, super-additive boundary cases, matched-set specificity, and singleton-versus-joint decomposition when available.
- `paper/figures/04_rq3_spiking_cut/` - **RQ3**: candidate/control phenotype, aggregate threshold-shape comparison, graded intervention, and supporting diagnostics. Status: **{rq3_status}**.
- `paper/figures/05_rq4_learning/` - **RQ4**: pooled-U(J)/competence Pythia trajectory plus directional companions and poisoning learning-time role changes. `poisoning/per_run/` preserves the individual-channel/agonist figures; the poisoning directory root contains cross-seed summaries.
- `paper/figures/appendix_context/` - pooled U(J), phase, and size context only.
- `paper/tables/` - LaTeX manuscript tables only.

## Analysis

- `analysis/figure_data/` - CSV/JSON sidecars for every paper figure, mirroring the figure subfolders.
- `analysis/table_data/` - machine-readable sidecars for manuscript tables.
- `analysis/primary_matrix/` - configured-study tables and cross-setting analyses.
- `analysis/rq2_composition/interaction_decomposition/` - preserved/suppressed/coalition-only singleton-versus-joint decomposition.
- `analysis/rq3_threshold_event/` - detailed threshold/spiking diagnostics.
- `analysis/rq4_learning/` - poisoning/cross-seed analysis outputs.
- `analysis/reproducibility/` - experiment catalogue, completeness audit, and manuscript-output audit.

### Reading order

1. Figure 2a/2c: 0->1 reach and high-effect density.
2. Figure 2b/2d: 1->0 companions; low-denominator estimates are explicitly marked as uncertain but remain in the declared setting-level fits.
3. Figure 3a/3b/3d: composition gap, coalition boundary, and the example-level source of the gap.
4. Figure 4: candidate/control observability and support-specific graded-agonist dose response.
5. Figure 5: learning-time role change.

Paper figure directories contain only PDFs and README files; raw CSV/JSON exports are intentionally kept under `analysis/`.
"""
    (results_root / "README.md").write_text(root_text, encoding="utf-8")

    figure_text = f"""# Manuscript figure map

- **Figure 1**: conceptual schematic; not generated by this statistical pipeline.
- **Figure 2 / RQ1**: `02_rq1_prevalence/`
- **Figure 3 / RQ2**: `03_rq2_composition/`
- **Figure 4 / RQ3**: `04_rq3_spiking_cut/` - {rq3_status}
- **Figure 5 / RQ4**: `05_rq4_learning/`
- **Appendix**: `appendix_context/`

Machine-readable figure data are in `../../analysis/figure_data/`.
"""
    (figures / "README.md").write_text(figure_text, encoding="utf-8")

    table_text = """# Manuscript tables

This folder contains paper-facing LaTeX only. CSV/JSON versions live under `results/analysis/table_data/`.

- `02_rq1_prevalence/` - directional prevalence/density table.
- `03_rq2_composition/` - composition and boundary table.
- `table1_representative_directional.tex` - representative directional summary.
- `tableS_primary_matrix_directional.tex` - full configured-setting directional supplement.
- `matched_null_metrics.tex` - matched-set validation.
"""
    (tables / "README.md").write_text(table_text, encoding="utf-8")



def print_rq_figure_summary(results_root: Path) -> None:
    """Print exactly which manuscript RQ figures were produced this run."""
    groups = [
        ("RQ1", rq1_figures(results_root)),
        ("RQ2", rq2_figures(results_root)),
        ("RQ3", rq3_figures(results_root)),
        ("RQ4", rq4_figures(results_root)),
    ]
    print("[final-results] generated RQ figure summary:", flush=True)
    for label, directory in groups:
        pdfs = sorted(path.name for path in directory.glob("*.pdf")) if directory.is_dir() else []
        if pdfs:
            print(f"  {label}: {len(pdfs)} PDF(s): {', '.join(pdfs)}", flush=True)
        else:
            readme = directory / "README.md"
            suffix = f" (see {readme})" if readme.is_file() else ""
            print(f"  {label}: no PDFs generated{suffix}", flush=True)


def validate_generated_paper_view(
    results_root: Path, *, spiking_expected: bool, require_complete_population: bool = True
) -> dict:
    """Validate the manuscript-facing output contract before reporting success."""
    figures = manuscript_figures(results_root)
    tables = manuscript_materials(results_root)
    errors: list[str] = []
    warnings: list[str] = []

    required_figures = [
        figures / "02_rq1_prevalence" / "fig2a_competence_vs_U_0to1.pdf",
        figures / "02_rq1_prevalence" / "fig2b_competence_vs_U_1to0.pdf",
        figures / "02_rq1_prevalence" / "fig2c_competence_vs_D05_0to1.pdf",
        figures / "02_rq1_prevalence" / "fig2d_competence_vs_D05_1to0.pdf",
        figures / "03_rq2_composition" / "fig3a_composition_gap_all_settings.pdf",
        figures / "05_rq4_learning" / "fig5a_pythia_checkpoint_trajectory.pdf",
    ]
    if spiking_expected:
        required_figures.extend([
            figures / "04_rq3_spiking_cut" / "fig4a_candidate_control_spiking_cut_summary.pdf",
            figures / "04_rq3_spiking_cut" / "fig4b_threshold_shape_model_comparison_by_direction.pdf",
        ])

    for path in required_figures:
        if not path.is_file():
            message = f"missing manuscript figure: {path}"
            (errors if require_complete_population else warnings).append(message)
        elif path.stat().st_size < 2_000:
            errors.append(f"suspiciously small manuscript figure ({path.stat().st_size} bytes): {path}")

    # Paper-facing directories must not regress into a mixed machine-data dump.
    for root in (figures, tables):
        if root.is_dir():
            for suffix in ("*.csv", "*.json"):
                for path in root.rglob(suffix):
                    errors.append(f"machine sidecar leaked into paper view: {path}")

    rq1_data = table_data(results_root) / "02_rq1_prevalence" / "table2_directional_prevalence.csv"
    if rq1_data.is_file():
        try:
            frame = pd.read_csv(rq1_data)
            needed = {
                "U_J_i2c_n", "U_J_c2i_n", "U_J_i2c_adequate", "U_J_c2i_adequate",
                "N_t_i2c_0.05_per_1k_layer", "N_t_c2i_0.05_per_1k_layer",
            }
            missing = sorted(needed.difference(frame.columns))
            if missing:
                errors.append(f"directional prevalence table lacks denominator/adequacy columns: {missing}")
            for direction in ("i2c", "c2i"):
                ncol = f"U_J_{direction}_n"
                if ncol in frame.columns:
                    low = pd.to_numeric(frame[ncol], errors="coerce") < 32
                    if bool(low.any()):
                        warnings.append(
                            f"{int(low.sum())} {direction} settings have small within-setting source-state denominators; "
                            "retain them in the across-setting sample and interpret their directional estimates with wider uncertainty"
                        )
        except Exception as exc:
            errors.append(f"could not inspect directional prevalence sidecar {rq1_data}: {exc}")
    else:
        message = f"missing directional prevalence sidecar: {rq1_data}"
        (errors if require_complete_population else warnings).append(message)

    # RQ1 paper panels use all 48 configured overtopping settings.
    # Derived *-heldout_test directories must never be
    # counted as additional experiments, and the fit-stat sidecar must describe
    # exactly the same sample as the plotted-point CSV.
    rq1_fig_data = figure_data(results_root) / "02_rq1_prevalence"
    for stem in (
        "fig2a_competence_vs_U_0to1",
        "fig2b_competence_vs_U_1to0",
        "fig2c_competence_vs_D05_0to1",
        "fig2d_competence_vs_D05_1to0",
    ):
        points_path = rq1_fig_data / f"{stem}.csv"
        stats_path = rq1_fig_data / f"{stem}_stats.csv"
        if not points_path.is_file():
            message = f"missing all-settings RQ1 point sidecar: {points_path}"
            (errors if require_complete_population else warnings).append(message)
            continue
        try:
            points = pd.read_csv(points_path)
            if "run" not in points.columns:
                errors.append(f"{stem} point sidecar lacks run provenance")
            else:
                # Figure 2 intentionally reads the strict held-out-test artifact for
                # each experimental setting.  The heldout-test suffix is therefore
                # evidence of the selected evaluation product, not evidence that the
                # derivative was counted as an additional experiment.  What must be
                # forbidden is having multiple evaluation variants of the *same*
                # canonical setting in the plotted population.
                run_names = points["run"].astype(str)
                heldout = run_names.str.contains(r"-heldout_test(?:-cap\d+)?$", regex=True)
                if bool((~heldout).any()):
                    examples = run_names.loc[~heldout].head(3).tolist()
                    errors.append(
                        f"{stem} contains {int((~heldout).sum())} non-heldout evaluation rows "
                        f"in the strict RQ1 manuscript population; examples={examples}"
                    )

                canonical_run = run_names.str.replace(
                    r"-(?:heldout_test|eval_train|eval_all)(?:-cap\d+)?$",
                    "",
                    regex=True,
                )
                key_columns = [c for c in ("task", "org", "model") if c in points.columns]
                key_frame = points[key_columns].astype(str).copy() if key_columns else pd.DataFrame(index=points.index)
                key_frame["canonical_run"] = canonical_run
                duplicate_mask = key_frame.duplicated(keep=False)
                if bool(duplicate_mask.any()):
                    duplicate_groups = int(key_frame.loc[duplicate_mask].drop_duplicates().shape[0])
                    duplicate_rows = int(duplicate_mask.sum())
                    errors.append(
                        f"{stem} contains {duplicate_rows} rows from {duplicate_groups} duplicated canonical "
                        "experimental settings after collapsing evaluation suffixes"
                    )

                # Figure 2 has an explicit manuscript population: all 48 configured overtopping settings. A manuscript figure
                # must therefore contain all 48 rows, with the known phase split,
                # rather than silently accepting whatever a filesystem scan found.
                expected_total = 48
                expected_phase_counts = {"input+output": 22, "decode-only": 26}
                if len(points) != expected_total:
                    message = (
                        f"{stem} has {len(points)} settings; expected the complete "
                        f"{expected_total}-setting RQ1 population"
                    )
                    (errors if require_complete_population else warnings).append(message)
                if "phase" in points.columns:
                    actual_phase_counts = points["phase"].astype(str).value_counts().to_dict()
                    for phase, n_expected in expected_phase_counts.items():
                        n_actual = int(actual_phase_counts.get(phase, 0))
                        if n_actual != n_expected:
                            message = f"{stem} has {n_actual} {phase} settings; expected {n_expected}"
                            (errors if require_complete_population else warnings).append(message)
            if not stats_path.is_file():
                message = f"missing all-settings RQ1 fit-stat sidecar: {stats_path}"
                (errors if require_complete_population else warnings).append(message)
            else:
                fit = pd.read_csv(stats_path)
                phase_map = {"input+output": "input+output", "decode-only": "decode-only"}
                for phase in phase_map:
                    expected = int((points.get("phase", pd.Series(dtype=str)).astype(str) == phase).sum())
                    row = fit[fit.get("phase", pd.Series(dtype=str)).astype(str).eq(phase)]
                    reported = int(pd.to_numeric(row["n"], errors="coerce").iloc[0]) if len(row) and "n" in row.columns else -1
                    if reported != expected:
                        errors.append(f"{stem} fit n mismatch for {phase}: plotted {expected}, stats report {reported}")
        except Exception as exc:
            errors.append(f"could not validate RQ1 all-settings sidecars for {stem}: {exc}")

    stale_primary_trajectory = figures / "05_rq4_learning" / "fig5s3_primary_matrix_directional_checkpoint_trajectory.pdf"
    if stale_primary_trajectory.exists():
        errors.append(f"primary-matrix checkpoint diagnostic leaked into paper view: {stale_primary_trajectory}")

    rq3_readme = figures / "04_rq3_spiking_cut" / "README.md"
    if not spiking_expected and not rq3_readme.is_file():
        errors.append("RQ3 source is absent but Figure-4 README does not explain the missing evidence")

    audit = {
        "status": "failed" if errors else "partial" if not require_complete_population else "ok",
        "spiking_expected": bool(spiking_expected),
        "require_complete_population": bool(require_complete_population),
        "errors": errors,
        "warnings": warnings,
    }
    out = analysis_root(results_root) / "reproducibility" / "manuscript_output_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Manuscript output validation failed: " + "; ".join(errors[:8]))
    if warnings:
        print("[final-results] manuscript-output warnings: " + "; ".join(warnings), flush=True)
    print(f"[final-results] manuscript output validation passed: {out}", flush=True)
    return audit

def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    poisoning_root = Path(args.poisoning_root).expanduser().resolve()
    validate_reporting_output_root(data_root=data_root, results_root=results_root)
    # Resolve the RQ3 diagnostics source before report generation so missing
    # threshold-event inputs fail before manuscript figures are written.
    spiking_source = None if args.skip_spiking_report else resolve_spiking_source(args.spiking_source, data_root)
    results_root.mkdir(parents=True, exist_ok=True)

    catalogue_json = Path(args.catalogue_json).expanduser().resolve() if args.catalogue_json else None

    # Stage 02 reads completed experiment artifacts and supplies the canonical
    # primary-row manifest. Final-results generation never mutates data/ or runs
    # model-backed experiment stages.
    paper_tables = primary_tables(results_root)
    run([
        sys.executable, "-m", "studies.overtopping.analysis.stage02_overtopping_latex_tables",
        "--results", str(data_root),
        "--out", str(paper_tables),
        "--primary-profile", args.primary_profile,
    ])

    publish_primary_tables(paper_tables, manuscript_materials(results_root))

    # Refresh the experiment catalogue from completed experiment artifacts.
    if catalogue_json is not None and catalogue_json.is_file():
        run([
            sys.executable, "-m", "studies.overtopping.analysis.stage01_visualize_experiment_results",
            "--catalogue_json", str(catalogue_json),
            "--data_root", str(data_root),
            "--out_dir", str(experiment_catalogue(results_root)),
        ])

    audit_command = [
        sys.executable, "-m", "studies.overtopping.analysis.stage03_audit_required_metrics",
        "--primary-table", str(paper_tables / "primary_table.csv"),
        "--data-root", str(data_root),
        "--primary-profile", args.primary_profile,
        "--out-dir", str(metric_completeness_audit(results_root)),
    ]
    if args.require_complete_metrics:
        audit_command.append("--require-complete")
    if args.skip_cmc_requirement:
        audit_command.append("--skip-cmc-requirement")
    run(audit_command)

    metric_audit_path = metric_completeness_audit(results_root) / "required_metrics_audit.json"
    metric_audit_payload: dict = {}
    if metric_audit_path.is_file():
        try:
            metric_audit_payload = json.loads(metric_audit_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Could not read metric completeness audit {metric_audit_path}: {exc}") from exc
    incomplete_setting_count = int(metric_audit_payload.get("incomplete_setting_count", 0) or 0)
    structural_zero_setting_count = int(metric_audit_payload.get("structural_zero_setting_count", 0) or 0)
    if structural_zero_setting_count:
        print(
            f"[final-results] {structural_zero_setting_count} configured settings are verified completed "
            "zero-candidate observations; they are not counted as incomplete.",
            flush=True,
        )
    allow_incomplete_population = bool(incomplete_setting_count and not args.require_complete_metrics)
    if allow_incomplete_population:
        print(
            f"[final-results] WARNING: {incomplete_setting_count} configured settings are incomplete; "
            "continuing with available outputs. Every RQ will be generated from the completed/auditable subset.",
            flush=True,
        )

    run([
        sys.executable, "-m", "studies.overtopping.analysis.stage04_analyze_primary_metrics",
        "primary",
        "--primary_table", str(paper_tables / "primary_table.csv"),
        "--data_root", str(data_root),
        "--out_dir", str(primary_statistics(results_root)),
    ])

    run([
        sys.executable, "-m", "studies.overtopping.analysis.stage05_generate_manuscript_outputs",
        "--primary_table", str(paper_tables / "primary_table.csv"),
        "--data_root", str(data_root),
        "--out_dir", str(manuscript_materials(results_root)),
        "--primary_profile", args.primary_profile,
    ])

    if not args.skip_paper_figures:
        paper_figures = manuscript_figures(results_root)
        if allow_incomplete_population:
            # Partial builds must never inherit apparently-complete PDFs from a
            # previous run. Rebuild every RQ directory from the data available
            # now; stages below will repopulate every renderable panel.
            for rq_dir in (rq1_figures(results_root), rq2_figures(results_root), rq4_figures(results_root)):
                rq_dir.mkdir(parents=True, exist_ok=True)
                for stale_pdf in rq_dir.glob("*.pdf"):
                    stale_pdf.unlink(missing_ok=True)
        # Render the compact manuscript figure set from the canonical populations.
        run([
            sys.executable, "-m", "studies.overtopping.analysis.stage06_manuscript_story_figures",
            "--primary_table_augmented", str(primary_statistics(results_root) / "primary_table_augmented.csv"),
            "--manuscript_metrics", str(manuscript_materials(results_root) / "manuscript_metrics.csv"),
            "--out_dir", str(paper_figures),
            "--figure_data_dir", str(figure_data(results_root)),
            "--skip_rq1",
            "--skip_primary_rq4",
        ])

        # RQ2 aggregate composition uses all evaluable completed settings but
        # never pools replacement regimes. Mean-positional is intentionally
        # normalized to the mean regime.
        # Use the complete configured-study table directly. This keeps RQ2
        # coverage independent of whether an optional catalogue visualization
        # was refreshed from a filtered runner manifest.
        run([
            sys.executable, "-m", "studies.overtopping.analysis.stage10_rq2_regime_report",
            "--catalogue-csv", str(paper_tables / "primary_table.csv"),
            "--out-dir", str(rq2_regime_summary(results_root)),
            "--paper-figures-dir", str(rq2_figures(results_root)),
            "--paper-tables-dir", str(manuscript_materials(results_root)),
        ])

        # RQ1 figures use all 48 configured overtopping settings.
        # Directional source-state
        # denominators describe uncertainty within a setting and are not an
        # across-setting exclusion rule. Genuine empty-candidate settings remain
        # explicit U(J)=0 observations.
        rq1 = rq1_figures(results_root)
        rq1.mkdir(parents=True, exist_ok=True)
        rq1_data = figure_data(results_root) / "02_rq1_prevalence"
        rq1_data.mkdir(parents=True, exist_ok=True)
        for metric, filename in [
            ("i2c", "fig2a_competence_vs_U_0to1.pdf"),
            ("c2i", "fig2b_competence_vs_U_1to0.pdf"),
            ("n05-i2c-density", "fig2c_competence_vs_D05_0to1.pdf"),
            ("n05-c2i-density", "fig2d_competence_vs_D05_1to0.pdf"),
        ]:
            if allow_incomplete_population:
                target = rq1 / filename
                target.unlink(missing_ok=True)
                stem = target.stem
                for suffix in (".csv", "_stats.csv", "_final_snapshot_stats.csv"):
                    (rq1_data / f"{stem}{suffix}").unlink(missing_ok=True)
            run([
                sys.executable, "-m", "studies.overtopping.analysis.stage06_competence_vs_overtopping_figures",
                "--results-dir", str(data_root),
                "--out", str(rq1 / filename),
                "--layout", "phase-panels",
                "--coverage-metric", metric,
                "--rq1-manuscript-population",
                *(["--allow-incomplete-manuscript-population"] if allow_incomplete_population else []),
                "--csv-out-dir", str(rq1_data),
                "--no-paper-figures",
            ], allow_failure=allow_incomplete_population)

        # RQ4 uses a pooled-U(J)/competence checkpoint trajectory plus directional
        # companions, including genuine zero-candidate checkpoints.
        rq4 = rq4_figures(results_root)
        rq4.mkdir(parents=True, exist_ok=True)
        trajectory_specs = [
            ("pooled", "fig5a_pythia_checkpoint_trajectory.pdf"),
            ("i2c", "fig5s1_pythia_checkpoint_U_0to1.pdf"),
            ("c2i", "fig5s2_pythia_checkpoint_U_1to0.pdf"),
        ]
        for metric, filename in trajectory_specs:
            if allow_incomplete_population:
                (rq4 / filename).unlink(missing_ok=True)
            run([
                sys.executable, "-m", "studies.overtopping.analysis.stage06_competence_vs_overtopping_figures",
                "--results-dir", str(data_root),
                "--only-paper-figures",
                "--paper-figures", "checkpoint",
                "--paper-figures-dir", str(rq4),
                "--paper-checkpoint-phase", "input+output",
                "--paper-checkpoint-filename", filename,
                "--coverage-metric", metric,
                "--no-csv",
            ], allow_failure=allow_incomplete_population)

        # Keep pooled-U/phase/size figures as explicit appendix context.  The
        # pooled competence scatter is the pooled-U companion to Figure 2, so it
        # must use exactly the same 48-setting RQ1 population
        # (22 input+output + 26 output-only).  Do not let the generic filesystem
        # scanner admit unrelated result directories and silently change n.
        appendix = appendix_figures(results_root)
        appendix.mkdir(parents=True, exist_ok=True)
        if allow_incomplete_population:
            (appendix / "figS_pooled_U_vs_competence.pdf").unlink(missing_ok=True)
        run([
            sys.executable, "-m", "studies.overtopping.analysis.stage06_competence_vs_overtopping_figures",
            "--results-dir", str(data_root),
            "--out", str(appendix / "figS_pooled_U_vs_competence.pdf"),
            "--rq1-manuscript-population",
            *(["--allow-incomplete-manuscript-population"] if allow_incomplete_population else []),
            "--paper-figures", "phase", "size",
            "--paper-figures-dir", str(appendix),
        ], allow_failure=allow_incomplete_population)

    # RQ2 example-level interaction decomposition uses the exact Stage-7
    # singleton flip masks and the genuine Stage-8 simultaneous full-set output.
    # It is model-free at reporting time and leaves the optional preemption assay intact.
    composition_cmd = [
        sys.executable, "-m", "studies.overtopping.analysis.stage09_composition_decomposition_report",
        "--root", str(data_root),
        "--out", str(rq2_interaction_decomposition(results_root)),
        "--primary-table", str(paper_tables / "primary_table.csv"),
        "--population-scope", "primary",
        "--replacement-regime", "mean-donor",
        "--evaluation-split", "test",
    ]
    if not args.skip_paper_figures:
        composition_cmd.extend(["--paper-figures-dir", str(rq2_figures(results_root))])
    run(composition_cmd)

    # Mean replacement is a separate RQ2 sensitivity regime; do not pool it
    # with mean-donor. Mean-positional is normalized to mean by the manifest.
    run([
        sys.executable, "-m", "studies.overtopping.analysis.stage09_composition_decomposition_report",
        "--root", str(data_root),
        "--out", str(rq2_interaction_decomposition_mean(results_root)),
        "--primary-table", str(paper_tables / "primary_table.csv"),
        "--population-scope", "primary",
        "--replacement-regime", "mean",
        "--evaluation-split", "test",
    ])

    # Poisoning run diagnostics stay inside each data/poisoning/<task>/<run> directory.
    # results/ receives only manuscript-facing poisoning outputs.
    poisoning_out = poisoning_results(results_root)
    poisoning_paper_figures = poisoning_figures(results_root)
    poisoning_aggregate = poisoning_aggregate_tables(results_root)
    poisoning_runs = poisoning_run_dirs(poisoning_root)
    if not args.skip_poisoning_report and poisoning_runs:
        poisoning_aggregate.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "-m", "studies.poisoning.stage08_aggregate_cross_seed",
            "--run_dirs", ",".join(str(path) for path in poisoning_runs),
            "--output_dir", str(poisoning_aggregate),
        ])
        run([
            sys.executable, "-m", "studies.poisoning.stage08_plot_cross_seed",
            "--input_dir", str(poisoning_aggregate),
            "--output_dir", str(poisoning_paper_figures),
        ])
        per_run_publish = publish_per_run_poisoning_visuals(
            poisoning_runs, results_root=results_root
        )
        print(
            "[final-results] per-run poisoning visuals: "
            + json.dumps(per_run_publish, sort_keys=True),
            flush=True,
        )

    spiking_out = overtopping_spiking_diagnostics(results_root)
    spiking_out.mkdir(parents=True, exist_ok=True)
    rq3_dir = rq3_figures(results_root)
    rq3_dir.mkdir(parents=True, exist_ok=True)
    # Never let figures from an older, more-complete run masquerade as current
    # partial results.  Every invocation rebuilds Figure 4 from the diagnostics
    # that are actually available now.
    for stale_pdf in rq3_dir.glob("*.pdf"):
        stale_pdf.unlink(missing_ok=True)
    # Likewise discard an older "deferred"/"missing" explanation before the
    # current run decides RQ3 availability.
    (rq3_dir / "README.md").unlink(missing_ok=True)
    spiking_report_generated = False

    if args.skip_spiking_report:
        status = {"status": "skipped", "reason": "--skip-spiking-report"}
        (spiking_out / "report_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        (rq3_dir / "README.md").write_text(
            "# Figure 4 - RQ3: the spiking cut\n\nGeneration was skipped with `--skip-spiking-report`.\n",
            encoding="utf-8",
        )
    elif spiking_source is not None:
        # Partial manuscript mode means *report the completed RQ3 subset*, not
        # suppress Figure 4.  Stage 7/8 audit every configured run and exclude
        # only rows whose required diagnostics are genuinely unfinished.
        cmd = [
            sys.executable, "-m", "studies.overtopping.analysis.stage07_overtopping_spiking_report",
            "--out", str(spiking_out),
            "--paper-figures-dir", str(rq3_dir),
            "--primary-table", str(paper_tables / "primary_table.csv"),
            "--data-root", str(data_root),
            "--evaluation-split", "test",
            "--spiking-max-points", str(args.spiking_max_points),
            "--population-scope", "primary",
        ]
        if allow_incomplete_population:
            cmd.append("--allow-incomplete-primary-population")
        if spiking_source.is_file():
            cmd.extend(["--zip", str(spiking_source)])
        else:
            cmd.extend(["--root", str(spiking_source)])
        spiking_report_generated = run(cmd, allow_failure=allow_incomplete_population)

        shape_cmd = [
            sys.executable, "-m", "studies.overtopping.analysis.stage08_threshold_shape_validation",
            "--out", str(spiking_out / "threshold_shape_validation"),
            "--paper-figures-dir", str(rq3_dir),
            "--primary-table", str(paper_tables / "primary_table.csv"),
            "--data-root", str(data_root),
            "--evaluation-split", "test",
            "--spiking-max-points", str(args.spiking_max_points),
            "--population-scope", "primary",
        ]
        if allow_incomplete_population:
            shape_cmd.append("--allow-incomplete-primary-population")
        if spiking_source.is_file():
            shape_cmd.extend(["--zip", str(spiking_source)])
        else:
            shape_cmd.extend(["--root", str(spiking_source)])
        shape_ok = run(shape_cmd, allow_failure=allow_incomplete_population)

        status = {
            "status": "partial" if allow_incomplete_population else "ok",
            "configured_incomplete_setting_count": int(incomplete_setting_count),
            "stage7_generated": bool(spiking_report_generated),
            "threshold_shape_generated": bool(shape_ok),
            "policy": (
                "completed configured RQ3 runs are reported; unfinished runs remain in the population audits"
                if allow_incomplete_population
                else "complete configured population required"
            ),
        }
        (spiking_out / "report_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    else:
        status = {
            "status": "not_available",
            "reason": "No aggregate_flip_stats.csv diagnostics source was found.",
            "searched_data_root": str(data_root),
            "hint": "Set SPIKING_SOURCE=/path/to/spiking_diagnostics_results_for_inspection.zip or rerun threshold_event_diagnostics.",
        }
        (spiking_out / "report_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        (rq3_dir / "README.md").write_text(
            "# Figure 4 - RQ3: the spiking cut\n\n"
            "**RQ3 source missing.** No aggregate threshold/spiking diagnostics were found.\n\n"
            "Set `SPIKING_SOURCE=/path/to/spiking_diagnostics_results_for_inspection.zip` before running `./generate_results.sh`, "
            "or rerun `threshold_event_diagnostics` to create the aggregate diagnostics.\n",
            encoding="utf-8",
        )
        print(f"[final-results] RQ3 source unavailable; see {rq3_dir / 'README.md'}")

    # Generate every RQ3 sub-analysis that has enough materialized input.  In
    # partial mode individual missing experimental runs are non-fatal, but stale
    # schemas/provenance errors inside completed runs are still surfaced by the
    # underlying report stages whenever they can be evaluated.
    if not args.skip_spiking_report and spiking_source is not None:
        # Remove Figure-4 names that are outside the current manuscript contract.
        for stale in (
            "fig4b_threshold_response_curves.pdf",
            "fig4c_preemption.pdf",
            "fig4d_graded_agonist_dose_response.pdf",
            "fig4d_graded_margin_affine_null.pdf",
            "fig4s5_threshold_shape_model_comparison_by_direction.pdf",
            "fig4s6_graded_agonist_single_crossing.pdf",
            "fig4s6_graded_margin_condition_diagnostics.pdf",
            "fig4s7_graded_margin_condition_heatmap.pdf",
            "fig4s8_graded_behavior_competence_reach_io.pdf",
            "fig4s8_graded_behavior_competence_reach_out.pdf",
            "fig4s9_graded_margin_competence_reach_io.pdf",
            "fig4s9_graded_margin_competence_reach_out.pdf",
            "fig4e_population_event_and_strength.pdf",
            "fig4s10_population_event_localization.pdf",
            "fig4s11_strength_concentration_paired.pdf",
            "fig4s12_arithmetic_competence_concentration.pdf",
            "fig4s13_affine_null_transient_events.pdf",
            "fig4s7_candidate_control_support_summary.pdf",
            "fig4s8_threshold_testability_support_by_condition.pdf",
            "fig4s9_threshold_tail_support_by_direction.pdf",
        ):
            (rq3_dir / stale).unlink(missing_ok=True)

        graded_ok = run([
            sys.executable, "-m", "studies.overtopping.analysis.stage08_graded_agonist_report",
            "--root", str(data_root),
            "--out", str(spiking_out / "graded_agonist"),
            "--paper-figures-dir", str(rq3_dir),
            "--primary-table", str(paper_tables / "primary_table.csv"),
            "--population-scope", "primary",
            "--evaluation-split", "test",
            "--require-negative-support",
        ], allow_failure=allow_incomplete_population)

        # Build the population-level RQ3 visual story from whatever graded
        # diagnostics were successfully materialized. In partial-results mode,
        # missing story inputs suppress only the affected story panels rather
        # than the rest of RQ3.
        if graded_ok:
            run([
                sys.executable, "-m", "studies.overtopping.analysis.stage10_rq3_spiking_story_figures",
                "--graded-dir", str(spiking_out / "graded_agonist"),
                "--spiking-dir", str(spiking_out),
                "--paper-figures-dir", str(rq3_dir),
            ], allow_failure=allow_incomplete_population)

        # Aggregate threshold-conditioned preemption over every available run.
        run([
            sys.executable, "-m", "studies.overtopping.analysis.stage09_preemption_report",
            "--root", str(data_root),
            "--out", str(spiking_out / "preemption"),
            "--primary-table", str(paper_tables / "primary_table.csv"),
            "--population-scope", "primary",
            "--evaluation-split", "test",
        ], allow_failure=allow_incomplete_population)

        rq3_readme = rq3_dir / "README.md"
        appendix = (
            "\n\n## Figure 4 layout\n\n"
            "- `fig4a_candidate_control_spiking_cut_summary.pdf` - candidate/control phenotype endpoints.\n"
            "- `fig4b_threshold_shape_model_comparison_by_direction.pdf` - aggregate held-out threshold/logistic/isotonic comparison by discovery direction.\n"
            "- `fig4c_graded_agonist_dose_response.pdf` - graded dose response for known-flip and same-channel non-flip support, when available.\n"
            "- `fig4d_graded_margin_affine_null.pdf` - normalized downstream divergence-margin response against the endpoint-affine null, when available.\n"
            "- `fig4s1_threshold_testability_by_condition.pdf` - condition-level testability.\n"
            "- `fig4s2_strength_matched_thresholdability.pdf` - strength-matched sensitivity.\n"
            "- `fig4s3_nested_tecs_lower_bound_ecdf.pdf` - nested TECS lower bound.\n"
            "- `fig4s4_threshold_tail_response_by_direction.pdf` - descriptive endogenous-proxy tail response.\n"
            "- `fig4s5_graded_agonist_single_crossing.pdf` - support-consistent graded trajectories, when available.\n"
            "- `fig4s6_graded_margin_condition_diagnostics.pdf` - condition-level affine-fit/concentration diagnostics, when available.\n"
            "- `fig4s7_graded_margin_condition_heatmap.pdf` - per-condition normalized margin trajectories, when available.\n"
            "- `fig4e_population_event_and_strength.pdf` - main population event-localization and within-condition causal-strength sharpening summary.\n"
            "- `fig4s10_population_event_localization.pdf` - standalone population event-localization profile with interquartile ranges.\n"
            "- `fig4s11_strength_concentration_paired.pdf` - paired low-versus-high causal-strength tertile concentration within eligible run/direction groups.\n"
            "- `fig4s12_arithmetic_competence_concentration.pdf` - task-specific Arithmetic output-only 1-to-0 competence/concentration relation.\n"
            "- `fig4s13_affine_null_transient_events.pdf` - endpoint-preserving transient interior events against the one-dimensional affine-margin null.\n"
            "\nIn partial-results mode each panel is generated from the completed/auditable configured subset. "
            "Missing settings are listed in the RQ3 population-audit files instead of suppressing Figure 4.\n"
        )
        existing = rq3_readme.read_text(encoding="utf-8") if rq3_readme.is_file() else "# Figure 4 - RQ3\n"
        marker = "## Figure 4 layout"
        if marker in existing:
            existing = existing.split(marker, 1)[0].rstrip()
        rq3_readme.write_text(existing.rstrip() + appendix, encoding="utf-8")

        graded_status_path = spiking_out / "graded_agonist" / "graded_agonist_report_status.json"
        if graded_status_path.is_file():
            try:
                graded_status = json.loads(graded_status_path.read_text(encoding="utf-8"))
            except Exception:
                graded_status = {}
            if graded_status.get("status") == "ok":
                for expected in (
                    rq3_dir / "fig4c_graded_agonist_dose_response.pdf",
                    rq3_dir / "fig4d_graded_margin_affine_null.pdf",
                    rq3_dir / "fig4s5_graded_agonist_single_crossing.pdf",
                    rq3_dir / "fig4s6_graded_margin_condition_diagnostics.pdf",
                    rq3_dir / "fig4s7_graded_margin_condition_heatmap.pdf",
                ):
                    if not expected.is_file():
                        raise RuntimeError(f"graded RQ3 report claimed success but figure is missing: {expected}")

    # Paper-facing directories contain PDFs/TEX/README only. Machine-readable
    # figure/table sidecars are kept under analysis/ with the same relative names.
    relocate_paper_sidecars(results_root)

    spiking_available = bool(
        not args.skip_spiking_report
        and any(path.suffix.lower() == ".pdf" for path in rq3_dir.glob("*.pdf"))
    )
    write_results_index(
        results_root,
        spiking_available=spiking_available,
        partial_results=allow_incomplete_population,
    )
    validate_generated_paper_view(
        results_root,
        spiking_expected=spiking_available,
        require_complete_population=not allow_incomplete_population,
    )
    print_rq_figure_summary(results_root)

    manifest = {
        "data_root": str(data_root),
        "results_root": str(results_root),
        "paper_root": str(paper_root(results_root)),
        "analysis_root": str(analysis_root(results_root)),
        "primary_profile": args.primary_profile,
        "partial_results": bool(allow_incomplete_population),
        "rq3_available": bool(spiking_available),
        "rq3_partial_population": bool(spiking_available and allow_incomplete_population),
        "structural_zero_setting_count": int(structural_zero_setting_count),
        "incomplete_setting_count": int(incomplete_setting_count),
        "primary_tables": str(paper_tables),
        "metric_completeness_audit": str(metric_completeness_audit(results_root)),
        "rq2_interaction_decomposition": str(rq2_interaction_decomposition(results_root)),
        "manuscript_figures": str(manuscript_figures(results_root)),
        "primary_statistics": str(primary_statistics(results_root)),
        "manuscript_materials": str(manuscript_materials(results_root)),
        "overtopping_spiking_diagnostics": str(spiking_out),
        "spiking_source": str(spiking_source) if spiking_source is not None else None,
        "figure_data": str(figure_data(results_root)),
        "table_data": str(table_data(results_root)),
        "poisoning_analysis_outputs": str(poisoning_out),
        "poisoning_paper_figures": str(poisoning_figures(results_root)),
        "poisoning_root": str(poisoning_root),
        "experiment_catalogue": str(experiment_catalogue(results_root)),
    }
    (results_root / "final_results_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[final-results] all final outputs are rooted at {results_root}")


if __name__ == "__main__":
    main()
