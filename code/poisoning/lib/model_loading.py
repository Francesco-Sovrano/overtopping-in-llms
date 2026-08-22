"""Poisoning-specific model reload settings.

Shared LMWrapper stays experiment-agnostic; this module owns the mapping from
poisoning run_config.json to the adapter dtype / TransformerLens conversion
settings used by the causal model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def configure_greedy_generation(model: Any) -> Any:
    """Replace inherited sampling settings with a greedy generation config.

    Qwen checkpoints can ship non-default ``top_p``/``top_k`` values in
    ``generation_config.json``. They are irrelevant under greedy decoding but
    recent Transformers releases warn when they coexist with
    ``do_sample=False``. Rebuilding from the model config retains special-token
    IDs while dropping repository-level sampling presets. The fallback handles
    wrappers whose generation-config class cannot be rebuilt directly.
    """
    config = getattr(model, "generation_config", None)
    if config is None:
        return model
    model_config = getattr(model, "config", None)
    try:
        if model_config is not None and hasattr(type(config), "from_model_config"):
            config = type(config).from_model_config(model_config)
            model.generation_config = config
    except Exception:
        config = getattr(model, "generation_config", config)

    neutral_values = {
        "do_sample": False,
        "num_beams": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "typical_p": 1.0,
        "min_p": None,
        "top_h": None,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "penalty_alpha": None,
    }
    for name, value in neutral_values.items():
        if hasattr(config, name):
            setattr(config, name, value)
    return model


def checkpoint_run_config(checkpoint_path: str | Path) -> Tuple[dict | None, Path | None]:
    path = Path(checkpoint_path).expanduser()
    try:
        path = path.resolve()
    except Exception:
        pass
    for parent in (path, *path.parents):
        candidate = parent / "run_config.json"
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8")), candidate
            except Exception as exc:
                print(f"[poisoning-model] could not read {candidate}: {type(exc).__name__}: {exc}", flush=True)
                return None, candidate
    return None, None


def dtype_from_run_config(run_cfg: dict | None):
    if not isinstance(run_cfg, dict):
        return None
    if bool(run_cfg.get("load_in_4bit", False)):
        return None
    if bool(run_cfg.get("bf16", False)):
        return torch.bfloat16
    if bool(run_cfg.get("fp16", False)):
        return torch.float16
    return torch.float32


def poisoning_lm_wrapper_kwargs(model_name: str | Path) -> Dict[str, Any]:
    """Return LMWrapper kwargs for a poisoning PEFT checkpoint only."""
    path = Path(str(model_name)).expanduser()
    if not (path / "adapter_config.json").is_file():
        return {}
    run_cfg, cfg_path = checkpoint_run_config(path)
    dtype = dtype_from_run_config(run_cfg)
    revision = None
    if isinstance(run_cfg, dict):
        raw = run_cfg.get("model_revision")
        if raw not in (None, "", "None"):
            revision = str(raw)
    if cfg_path is not None:
        dtype_name = str(dtype).replace("torch.", "") if dtype is not None else "auto"
        rev_text = revision if revision is not None else "default-resolved"
        print(
            f"[poisoning-model] TransformerLens checkpoint reload settings from {cfg_path}: "
            f"dtype={dtype_name} base_revision={rev_text}",
            flush=True,
        )
    # Keep the same TransformerLens conversion convention used by ordinary
    # poisoning runs (folding/centering enabled by LMWrapper defaults).
    # Only the PEFT adapter load dtype is poisoning-run metadata.
    return {"adapter_load_dtype": dtype, "adapter_base_revision": revision}
