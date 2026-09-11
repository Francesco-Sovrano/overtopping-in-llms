"""Resolve the best available evaluation-side stats directory for reporting.

Reporting should distinguish "no stats exist" from "the canonical full held-out
stats directory is absent, but a materialized partial/capped held-out evaluation
exists".  Partial artifacts are useful results and should be plotted/reported in
best-effort mode; strict manuscript mode can still reject them as incomplete
coverage.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


_CAP_SUFFIX_RE = re.compile(r"-cap(\d+)$")


def _artifact_score(path: Path) -> tuple[int, int, int]:
    """Rank a stats directory by usable result artifacts and evaluated rows."""
    if not path.is_dir():
        return (0, 0, 0)
    artifacts = [
        path / "flip_stats_global.json",
        path / "flip_stats_by_neuron.csv",
        path / "singleton_set_metrics.json",
        path / "scores.csv",
        path / "frozen_candidate_ranking.csv",
        path / "interaction_validation" / "interaction_validation_summary.json",
    ]
    count = sum(p.is_file() for p in artifacts)
    n_eval = 0
    global_path = path / "flip_stats_global.json"
    if global_path.is_file():
        try:
            payload = json.loads(global_path.read_text(encoding="utf-8"))
            n_eval = int(payload.get("n_evaluated_rows", 0) or 0)
        except Exception:
            n_eval = 0
    match = _CAP_SUFFIX_RE.search(path.name)
    cap = int(match.group(1)) if match else 0
    return (count, n_eval, cap)


def has_stats_artifacts(path: Path) -> bool:
    """Return whether *path* contains any recognized overtopping result stats."""
    return _artifact_score(Path(path))[0] > 0


def resolve_available_stats_dir(expected: Path) -> tuple[Path, str]:
    """Return the best usable stats directory and a scope/status label.

    Status values:
      - ``exact``: canonical configured stats directory has result artifacts.
      - ``partial_heldout_cap``: a ``-heldout_test-capN`` sibling is used.
      - ``empty_expected_dir``: canonical directory exists but has no result stats.
      - ``absent``: neither canonical nor compatible partial held-out stats exist.

    Only compatible held-out cap siblings are substituted.  Train/all evaluation
    directories are deliberately not used as manuscript-test replacements.
    """
    expected = Path(expected).expanduser()
    if has_stats_artifacts(expected):
        return expected.resolve(), "exact"

    parent = expected.parent
    partials: list[Path] = []
    if expected.name.endswith("-heldout_test") and parent.is_dir():
        prefix = expected.name + "-cap"
        partials = [
            p for p in parent.iterdir()
            if p.is_dir() and p.name.startswith(prefix) and has_stats_artifacts(p)
        ]
    if partials:
        best = max(partials, key=_artifact_score)
        return best.resolve(), "partial_heldout_cap"

    if expected.is_dir():
        return expected.resolve(), "empty_expected_dir"
    return expected.resolve(), "absent"
