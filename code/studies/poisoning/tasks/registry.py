"""Discovery and lookup for single-file poisoning task modules.

A task is registered by providing ``studies.poisoning.tasks.<name>`` with a
``TASK_DEFINITION`` object. Generic stages use this registry and do not branch
on grammar/arithmetic semantics.
"""
from __future__ import annotations

import importlib
import json
import pkgutil
from pathlib import Path

from studies.poisoning.lib.run_paths import cohort_path, metadata_path

from studies.poisoning.tasks.base import PoisoningTaskDefinition


def available_tasks() -> tuple[str, ...]:
    import studies.poisoning.tasks as package

    names: list[str] = []
    for mod in pkgutil.iter_modules(package.__path__):
        if mod.ispkg or mod.name.startswith("_") or mod.name in {"base", "registry"}:
            continue
        module = importlib.import_module(f"studies.poisoning.tasks.{mod.name}")
        if isinstance(getattr(module, "TASK_DEFINITION", None), PoisoningTaskDefinition):
            names.append(mod.name)
    return tuple(sorted(names))


def get_task_definition(name: str) -> PoisoningTaskDefinition:
    clean = str(name).strip()
    if not clean or not clean.replace("_", "").isalnum():
        raise ValueError(f"Invalid poisoning task name: {name!r}")
    try:
        module = importlib.import_module(f"studies.poisoning.tasks.{clean}")
    except ModuleNotFoundError as exc:
        if exc.name == f"studies.poisoning.tasks.{clean}":
            raise ValueError(
                f"Unknown poisoning task {clean!r}; available={available_tasks()}"
            ) from exc
        raise
    definition = getattr(module, "TASK_DEFINITION", None)
    if not isinstance(definition, PoisoningTaskDefinition):
        raise TypeError(
            f"studies.poisoning.tasks.{clean}.TASK_DEFINITION must be a PoisoningTaskDefinition"
        )
    return definition


def infer_task_from_run(run_dir: str | Path) -> PoisoningTaskDefinition:
    run = Path(run_dir)
    cfg_path = metadata_path(run, "run_config.json")
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            task = str(cfg.get("task", "")).strip()
            if task:
                return get_task_definition(task)
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass

    matches = []
    for name in available_tasks():
        definition = get_task_definition(name)
        if cohort_path(run, definition.heldout_validation_filename).exists():
            matches.append(definition)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Cannot infer poisoning task from {run}")
    raise ValueError(f"Ambiguous poisoning task for {run}: {[x.name for x in matches]}")

