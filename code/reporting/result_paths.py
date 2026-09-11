"""Canonical generated-results layout.

The top-level ``results/`` directory is intentionally split into only two user-facing
areas:

* ``paper/``: manuscript-facing figures and tables, grouped by research question;
* ``analysis/``: reproducibility tables, audits, catalogues, and detailed diagnostics.

This keeps exploratory/diagnostic artifacts out of the manuscript figure directory.
"""
from __future__ import annotations

from pathlib import Path


def paper_root(root: Path) -> Path:
    return Path(root) / "paper"


def analysis_root(root: Path) -> Path:
    return Path(root) / "analysis"


def experiment_catalogue(root: Path) -> Path:
    return analysis_root(root) / "reproducibility" / "experiment_catalogue"


def primary_tables(root: Path) -> Path:
    return analysis_root(root) / "primary_matrix" / "tables"


def metric_completeness_audit(root: Path) -> Path:
    return analysis_root(root) / "reproducibility" / "metric_completeness_audit"


def primary_statistics(root: Path) -> Path:
    return analysis_root(root) / "primary_matrix" / "statistics"


def manuscript_materials(root: Path) -> Path:
    return paper_root(root) / "tables"


def manuscript_figures(root: Path) -> Path:
    return paper_root(root) / "figures"


def rq1_figures(root: Path) -> Path:
    return manuscript_figures(root) / "02_rq1_prevalence"


def rq2_figures(root: Path) -> Path:
    return manuscript_figures(root) / "03_rq2_composition"


def rq3_figures(root: Path) -> Path:
    return manuscript_figures(root) / "04_rq3_spiking_cut"


def rq4_figures(root: Path) -> Path:
    return manuscript_figures(root) / "05_rq4_learning"


def appendix_figures(root: Path) -> Path:
    return manuscript_figures(root) / "appendix_context"



def figure_data(root: Path) -> Path:
    return analysis_root(root) / "figure_data"


def table_data(root: Path) -> Path:
    return analysis_root(root) / "table_data"

def rq2_interaction_decomposition(root: Path) -> Path:
    return analysis_root(root) / "rq2_composition" / "interaction_decomposition"


def rq2_interaction_decomposition_mean(root: Path) -> Path:
    return analysis_root(root) / "rq2_composition" / "interaction_decomposition_mean"


def rq2_regime_summary(root: Path) -> Path:
    return analysis_root(root) / "rq2_composition" / "regime_summary"


def overtopping_spiking_diagnostics(root: Path) -> Path:
    return analysis_root(root) / "rq3_threshold_event" / "spiking_diagnostics"


def poisoning_results(root: Path) -> Path:
    return analysis_root(root) / "rq4_learning" / "poisoning"


def poisoning_aggregate_tables(root: Path) -> Path:
    return poisoning_results(root) / "cross_seed_tables"


def poisoning_figures(root: Path) -> Path:
    # Manuscript-facing cross-seed poisoning plots belong directly in RQ4.
    return rq4_figures(root) / "poisoning"
