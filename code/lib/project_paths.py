"""Stable filesystem roots for the repository layout.

Python packages live under ``<project>/code`` while datasets, caches, virtual
environments, generated results, and root launchers live under ``<project>``.
This module centralizes those two roots without mutating ``sys.path``.
"""
from __future__ import annotations

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent


