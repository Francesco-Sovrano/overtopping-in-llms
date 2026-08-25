"""Training-time protection for the poisoning-prevention follow-up.

The ordinary overtopping coordinates are activation-space coordinates, not
standalone trainable parameters.  Exact ``requires_grad=False`` neuron freezing
is therefore not available.  This module implements a narrow, auditable LoRA
intervention that blocks the adapter's most direct write into the selected
activation coordinate while leaving the base model unchanged.

Mappings
--------
``mL:u``
    Protect row ``u`` of layer ``L``'s ``mlp.down_proj.lora_B`` matrix.  This
    prevents LoRA from directly changing residual output coordinate ``u`` of the
    MLP write.

``aL.hH:u``
    TransformerLens defines this coordinate at ``attn.hook_z``: dimension ``u``
    of attention head ``H`` in layer ``L``.  The direct value write feeding that
    coordinate is the corresponding row of ``self_attn.v_proj.lora_B``.  For
    grouped-query attention, several query heads share one KV head, so the
    mapping is necessarily shared across that query-head group.

The intervention is intentionally called *direct-channel-write protection*.
Q/K updates, upstream representations, and other modules can still change the
protected activation indirectly.  The experiment therefore tests whether
blocking direct adapter writes into virgin agonist coordinates delays or changes
poisoning; it does not claim exact activation clamping.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

_MLP_RE = re.compile(r"^m(?P<layer>\d+)$")
_ATTN_RE = re.compile(r"^a(?P<layer>\d+)\.h(?P<head>\d+)$")


def _model_config(model: Any) -> Any:
    """Return the underlying transformer config from HF or PEFT wrappers."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        return cfg
    base = getattr(model, "base_model", None)
    cfg = getattr(base, "config", None)
    if cfg is not None:
        return cfg
    raise RuntimeError("Could not resolve model.config for training protection.")


def _attention_value_row(model: Any, head: int, unit_id: int) -> tuple[int, dict[str, int]]:
    """Map a TransformerLens ``hook_z`` head/dim to the HF V-projection row.

    Qwen uses grouped-query attention. ``hook_z`` exposes one value vector per
    query head after KV repetition.  The underlying V projection has one vector
    per KV head, so all query heads in the same group map to the same V row.
    """
    cfg = _model_config(model)
    n_heads = int(getattr(cfg, "num_attention_heads"))
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    hidden = int(getattr(cfg, "hidden_size"))
    head_dim = int(getattr(cfg, "head_dim", hidden // n_heads) or (hidden // n_heads))
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise RuntimeError(
            f"Unsupported attention head configuration: n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        )
    if not (0 <= int(head) < n_heads):
        raise IndexError(f"Attention head {head} outside [0,{n_heads})")
    if not (0 <= int(unit_id) < head_dim):
        raise IndexError(f"Attention unit {unit_id} outside head_dim={head_dim}")
    group_size = n_heads // n_kv_heads
    kv_head = int(head) // group_size
    row = kv_head * head_dim + int(unit_id)
    return row, {
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "query_head": int(head),
        "kv_head": kv_head,
        "query_heads_per_kv_head": group_size,
    }


def coordinate_to_module_row(model: Any, layer_label: str, unit_id: int) -> tuple[int, str, int, str, dict]:
    """Translate one intervention coordinate into its direct LoRA-B write row."""
    m = _MLP_RE.match(str(layer_label))
    if m:
        return int(m.group("layer")), "mlp.down_proj", int(unit_id), "mlp_residual_write", {}
    m = _ATTN_RE.match(str(layer_label))
    if m:
        head = int(m.group("head"))
        row, meta = _attention_value_row(model, head, int(unit_id))
        return int(m.group("layer")), "self_attn.v_proj", row, "attention_value_write", meta
    raise ValueError(f"Unsupported virgin agonist layer label for direct-channel-write protection: {layer_label!r}")


def _find_lora_b_parameters(model: Any) -> dict[tuple[int, str], tuple[str, torch.nn.Parameter]]:
    out: dict[tuple[int, str], tuple[str, torch.nn.Parameter]] = {}
    pattern = re.compile(r"\.layers\.(\d+)\.(mlp\.down_proj|self_attn\.v_proj)\.lora_B\.[^.]+\.weight$")
    for name, param in model.named_parameters():
        m = pattern.search(name)
        if not m:
            continue
        key = (int(m.group(1)), str(m.group(2)))
        out[key] = (name, param)
    return out


def _rows_by_parameter(model: Any, coords: Iterable[tuple[str, int]]) -> tuple[dict[tuple[int, str], set[int]], list[dict]]:
    rows: dict[tuple[int, str], set[int]] = defaultdict(set)
    mapping: list[dict] = []
    for layer_label, unit_id in coords:
        layer, module_suffix, row, mapping_kind, extra = coordinate_to_module_row(model, layer_label, unit_id)
        rows[(layer, module_suffix)].add(row)
        record = {
            "source_layer_label": str(layer_label),
            "source_unit_id": int(unit_id),
            "layer": layer,
            "module": module_suffix,
            "protected_lora_B_output_row": row,
            "mapping_kind": mapping_kind,
        }
        record.update(extra)
        mapping.append(record)
    return rows, mapping


def _matched_random_rows(
    requested: dict[tuple[int, str], set[int]],
    params: dict[tuple[int, str], tuple[str, torch.nn.Parameter]],
    *,
    seed: int,
) -> dict[tuple[int, str], set[int]]:
    """Draw a deterministic same-module/cardinality protection control."""
    rng = random.Random(int(seed))
    out: dict[tuple[int, str], set[int]] = {}
    for key, protected in requested.items():
        if key not in params:
            continue
        _, param = params[key]
        n_rows = int(param.shape[0])
        available = [i for i in range(n_rows) if i not in protected]
        if len(available) < len(protected):
            raise RuntimeError(f"Not enough non-protected rows for matched-random control at {key}")
        out[key] = set(rng.sample(available, len(protected)))
    return out


def install_direct_channel_write_lora_protection(
    model: Any,
    coords: Iterable[tuple[str, int]],
    *,
    mode: str,
    seed: int,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Zero gradients for selected direct LoRA-B channel-write rows.

    ``mode='virgin_agonists'`` protects rows mapped from the supplied ordinary
    task coordinates. ``mode='matched_random'`` protects the same number of
    rows in each mapped layer/module but chooses different rows deterministically.
    """
    if mode not in {"virgin_agonists", "matched_random"}:
        raise ValueError("mode must be virgin_agonists or matched_random")
    coords = list(coords)
    requested, coordinate_mapping = _rows_by_parameter(model, coords)
    params = _find_lora_b_parameters(model)
    missing = sorted(key for key in requested if key not in params)
    if missing:
        raise RuntimeError(
            "Could not find LoRA-B parameters for protected module(s): "
            + ", ".join(f"layer={l} module={m}" for l, m in missing)
        )

    selected = requested if mode == "virgin_agonists" else _matched_random_rows(requested, params, seed=seed)
    handles = []
    param_records = []
    for key, rows in sorted(selected.items()):
        name, param = params[key]
        valid = sorted(int(r) for r in rows if 0 <= int(r) < int(param.shape[0]))
        invalid = sorted(int(r) for r in rows if int(r) < 0 or int(r) >= int(param.shape[0]))
        if invalid:
            raise IndexError(f"Protected row(s) outside {name} shape {tuple(param.shape)}: {invalid}")
        row_index = torch.tensor(valid, dtype=torch.long)

        def hook(grad, row_index=row_index):
            if grad is None or row_index.numel() == 0:
                return grad
            out = grad.clone()
            out.index_fill_(0, row_index.to(out.device), 0)
            return out

        handles.append(param.register_hook(hook))
        param_records.append({
            "parameter": name,
            "shape": list(param.shape),
            "protected_rows": valid,
            "n_protected_rows": len(valid),
        })

    summary = {
        "protection_type": "direct_channel_write_lora_B_row_gradient_mask",
        "mode": mode,
        "seed": int(seed),
        "n_source_coordinates": len(coords),
        "n_parameter_modules": len(param_records),
        "n_protected_rows_total": sum(r["n_protected_rows"] for r in param_records),
        "coordinate_mapping": coordinate_mapping,
        "parameter_masks": param_records,
        "scope_note": (
            "Blocks direct LoRA-B writes into selected MLP residual coordinates and selected attention value coordinates. "
            "For grouped-query attention, query heads sharing one KV head share the protected value row. "
            "The intervention does not clamp activations and does not prevent indirect changes through Q/K, upstream representations, or other modules."
        ),
    }
    # Keep hooks alive for the Trainer lifetime.
    setattr(model, "_poisoning_protection_hook_handles", handles)
    setattr(model, "_poisoning_protection_summary", summary)
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[training-protection] mode={mode} modules={len(param_records)} "
        f"protected_direct_channel_rows={summary['n_protected_rows_total']}",
        flush=True,
    )
    return summary
