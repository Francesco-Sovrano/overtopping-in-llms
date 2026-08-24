"""Descriptive final-results directory names used by analysis orchestration."""
from __future__ import annotations

from pathlib import Path


def experiment_catalogue(root: Path) -> Path:
    return Path(root) / "experiment_catalogue"


def primary_tables(root: Path) -> Path:
    return Path(root) / "primary_analysis" / "tables"


def metric_completeness_audit(root: Path) -> Path:
    return Path(root) / "primary_analysis" / "metric_completeness_audit"


def primary_statistics(root: Path) -> Path:
    return Path(root) / "primary_analysis" / "statistics"


def manuscript_materials(root: Path) -> Path:
    return Path(root) / "manuscript" / "tables_and_macros"


def manuscript_figures(root: Path) -> Path:
    return Path(root) / "manuscript" / "figures"


def overtopping_spiking_diagnostics(root: Path) -> Path:
    return Path(root) / "diagnostics" / "overtopping_spiking"


def poisoning_figures(root: Path) -> Path:
    return Path(root) / "poisoning" / "figures"
