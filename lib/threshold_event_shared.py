"""Shared threshold-event and activation diagnostic helpers.

This module holds helper code used by:
  - 6_analyze_bag_of_rules.py for post-hoc agonist activation diagnostics
  - 7_refine_neuron_anchored_rules.py for stable layer-key naming
  - 12_threshold_event_diagnostics.py / lib.threshold_event_spiking for high-N
    overtopping-vs-control threshold/spiking diagnostics

Keeping these utilities here avoids duplicating script-6 activation capture logic
or script-7 layer-key normalization in the new high-N experiment.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

LOG_PREFIX = "[threshold-events]"

from lib.modeling_and_ablation import get_layer_type_and_ids


def safe_layer_label(layer_label) -> str:
    """Filesystem/column-safe layer label, e.g. ``a16.h21`` -> ``a16_h21``."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(layer_label)).strip("_")


def activation_hook_spec(layer_label: str):
    """Resolve a MechaRule layer label to a hook name and comparison pool."""
    parsed = get_layer_type_and_ids(layer_label)
    if not parsed:
        return None
    layer_type, layer_idx, head_idx = parsed
    if layer_type == "mlp":
        return {
            "layer_type": "mlp",
            "layer_index": int(layer_idx),
            "head_index": None,
            "hook_name": f"blocks.{int(layer_idx)}.hook_mlp_out",
            "compare_pool": "all_mlp_units_in_layer",
        }
    if layer_type == "attn":
        return {
            "layer_type": "attn",
            "layer_index": int(layer_idx),
            "head_index": int(head_idx),
            "hook_name": f"blocks.{int(layer_idx)}.attn.hook_z",
            "compare_pool": "all_attention_dimensions_in_layer",
        }
    return None


def collect_reference_activations(model, examples, prompt_col, layer_labels, batch_size, decode_only, max_new_tokens):
    """Collect last-position/decode-position activations for requested layer labels.

    This is the script-6 activation-capture path generalized into a shared helper.
    It returns a dict ``layer_label -> tensor``. For MLP hooks the tensor has shape
    ``[n_examples, d_mlp]``; for attention hooks it has shape
    ``[n_examples, n_heads, d_head]``.
    """
    if not examples or not layer_labels:
        return {}

    ordered_layer_labels = []
    for layer_label in layer_labels:
        if layer_label not in ordered_layer_labels:
            ordered_layer_labels.append(layer_label)

    specs = {}
    for layer_label in ordered_layer_labels:
        spec = activation_hook_spec(layer_label)
        if spec is not None:
            specs[layer_label] = spec
    if not specs:
        return {}

    device = model.hooked_model.cfg.device
    collected = {layer_label: [] for layer_label in specs}

    n_batches = (len(examples) + int(batch_size) - 1) // int(batch_size)
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(examples), int(batch_size)),
            total=n_batches,
            desc=f"{LOG_PREFIX} activation batches",
            unit="batch",
            leave=False,
        ):
            batch_examples = examples[start:start + int(batch_size)]
            batch_prompts = [str(row[prompt_col]) for row in batch_examples]

            if decode_only:
                prefix = model.prefill_prefix_batch(
                    batch_prompts,
                    max_new_tokens=max(2, int(max_new_tokens)),
                    use_kv_cache=True,
                )
                seen = {layer_label: False for layer_label in specs}

                def make_hook(layer_label):
                    def _hook(act, hook):
                        if seen[layer_label]:
                            return act
                        if act.ndim == 3:
                            snapshot = act[:, -1, :]
                        elif act.ndim == 4:
                            snapshot = act[:, -1, :, :]
                        else:
                            return act
                        collected[layer_label].append(snapshot.detach().to(torch.float32).cpu())
                        seen[layer_label] = True
                        return act
                    return _hook

                fwd_hooks = [(spec["hook_name"], make_hook(layer_label)) for layer_label, spec in specs.items()]
                _ = model.generate_from_prefix_cache(
                    prefix,
                    fwd_hooks=fwd_hooks,
                    stop_at_eos=False,
                    clone_kv_cache_tensors=True,
                )
                del prefix
            else:
                input_ids, attention_mask, input_lengths = model.tokenize_with_mask(
                    batch_prompts,
                    device,
                    padding=True,
                    truncation=True,
                    add_special_tokens=True,
                    padding_side="right",
                )
                last_idx = (input_lengths - 1).to(device)

                def make_hook(layer_label, last_idx=last_idx):
                    def _hook(act, hook):
                        batch_idx = torch.arange(act.shape[0], device=act.device)
                        if act.ndim == 3:
                            snapshot = act[batch_idx, last_idx, :]
                        elif act.ndim == 4:
                            snapshot = act[batch_idx, last_idx, :, :]
                        else:
                            return act
                        collected[layer_label].append(snapshot.detach().to(torch.float32).cpu())
                        return act
                    return _hook

                fwd_hooks = [(spec["hook_name"], make_hook(layer_label)) for layer_label, spec in specs.items()]
                with model.hooked_model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True, clear_contexts=True):
                    _ = model.hooked_model(
                        input_ids,
                        attention_mask=attention_mask,
                        padding_side="right",
                        return_type="residual",
                        stop_at_layer=model.hooked_model.cfg.n_layers,
                    )
            model.cleanup_after_generate()

    out = {}
    for layer_label, chunks in collected.items():
        if chunks:
            out[layer_label] = torch.cat(chunks, dim=0)
    return out



def _next_token_id_for_completion(tokenizer, prompt_text: str, completion_text: str):
    def _ids(text, add_special_tokens):
        try:
            return tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        except Exception:
            return None

    for add_special_tokens in (True, False):
        prompt_ids = _ids(prompt_text, add_special_tokens)
        full_ids = _ids(f"{prompt_text}{completion_text}", add_special_tokens)
        if prompt_ids and full_ids and len(full_ids) > len(prompt_ids) and full_ids[:len(prompt_ids)] == prompt_ids:
            return int(full_ids[len(prompt_ids)])

    for candidate in (completion_text, f" {completion_text}"):
        ids = _ids(candidate, False)
        if ids:
            return int(ids[0])
    return None


def _is_missing_value(val):
    if val is None:
        return True
    if isinstance(val, float):
        try:
            return bool(np.isnan(val))
        except Exception:
            return False
    try:
        return bool(np.isnan(val))
    except Exception:
        return False


def _completion_text_from_row_for_saliency(task, row, prompt_col):
    """Best-effort textual target/completion lookup from row metadata only.

    This intentionally does not call task-object margin hooks. Task specs provide
    schemas; diagnostics own the proxy objective.
    """
    candidate_keys = []
    for attr in ("DEFAULT_OUTPUT", "DEFAULT_OUTPUTS", "DEFAULT_ANSWER", "DEFAULT_ANSWERS"):
        val = getattr(task, attr, None)
        if isinstance(val, str):
            candidate_keys.append(val)
        elif isinstance(val, (list, tuple)):
            candidate_keys.extend(str(x) for x in val if isinstance(x, str))
    candidate_keys.extend([
        "answer", "answers", "completion", "target_text", "correct_answer",
        "output", "outputs", "label_text", "gold", "gold_answer",
        "raw_output", "num_out",
    ])
    seen = set()
    for key in candidate_keys:
        if key in seen or key == prompt_col or key not in row:
            continue
        seen.add(key)
        val = row.get(key)
        if isinstance(val, (list, tuple)) and val:
            val = val[0]
        if isinstance(val, (bool, np.bool_)) or _is_missing_value(val):
            continue
        text = str(val)
        if text:
            return text
    return None


def _arithmetic_target_text_from_prompt(prompt_text: str):
    """Infer arithmetic answer from the last expression in a prompt string."""
    prompt_text = str(prompt_text)
    few_shot_sep = ';' if ';' in prompt_text else (',' if ',' in prompt_text else None)
    prompt_eval = prompt_text[prompt_text.rfind(few_shot_sep) + 1:] if few_shot_sep is not None else prompt_text
    try:
        return str(eval(prompt_eval.replace('=', '')))
    except Exception:
        return None


def _margin_from_target_texts(task, prompt_batch, logits_last, tokenizer, prompt_col):
    if logits_last.ndim != 2:
        return None
    target_ids = []
    for row in prompt_batch:
        prompt_text = str(row.get(prompt_col, row.get(getattr(task, "DEFAULT_INPUT", "prompt"), "")))
        target_text = _completion_text_from_row_for_saliency(task, row, prompt_col)
        if target_text is None:
            cls_name = getattr(task.__class__, "__name__", "").lower()
            module_name = getattr(task.__class__, "__module__", "").lower()
            if "arithmetic" in cls_name or "arithmetic" in module_name:
                target_text = _arithmetic_target_text_from_prompt(prompt_text)
        target_ids.append(_next_token_id_for_completion(tokenizer, prompt_text, str(target_text)) if target_text is not None else -1)

    target_ids = torch.tensor([int(x) if x is not None else -1 for x in target_ids], device=logits_last.device, dtype=torch.long)
    valid = target_ids >= 0
    if not bool(valid.any()):
        return None
    safe_ids = target_ids.clamp_min(0).clamp_max(logits_last.shape[1] - 1)
    target_logits = logits_last.gather(1, safe_ids.unsqueeze(1)).squeeze(1)
    masked = logits_last.clone()
    masked[torch.arange(masked.size(0), device=masked.device), safe_ids] = float("-inf")
    other_logits = masked.max(dim=1).values
    margin = target_logits - other_logits
    return torch.where(valid, margin, torch.zeros_like(margin))


def inferred_gold_margin_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col):
    """Differentiable gold next-token margin inferred only from row metadata."""
    return _margin_from_target_texts(task, prompt_batch, logits_last, tokenizer, prompt_col)


def saliency_objective_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col, *, allow_fallback=False):
    """Return a differentiable scalar-per-example objective for saliency.

    Priority:
      1. inferred gold next-token margin from row metadata;
      2. optional cached-completion/top-token logit fallback.
    """
    margin = inferred_gold_margin_from_last_logits(task, prompt_batch, logits_last, tokenizer, prompt_col)
    if margin is not None:
        return margin, "gold_margin"

    if not allow_fallback:
        return None, "none"

    device = logits_last.device
    batch_size = int(logits_last.shape[0])
    top_ids = logits_last.detach().argmax(dim=1).to(device=device, dtype=torch.long)
    target_ids = []
    used_cached = []
    for i, row in enumerate(prompt_batch):
        prompt_text = str(row.get(prompt_col, row.get(getattr(task, "DEFAULT_INPUT", "prompt"), "")))
        completion_text = _completion_text_from_row_for_saliency(task, row, prompt_col)
        tok_id = None
        if completion_text is not None:
            tok_id = _next_token_id_for_completion(tokenizer, prompt_text, str(completion_text))
        if tok_id is None:
            tok_id = int(top_ids[i].item())
            used_cached.append(False)
        else:
            used_cached.append(True)
        target_ids.append(int(tok_id))

    ids = torch.tensor(target_ids, device=device, dtype=torch.long).clamp(0, logits_last.shape[1] - 1)
    objective = logits_last.gather(1, ids.unsqueeze(1)).squeeze(1)
    objective = torch.nan_to_num(objective.reshape(batch_size), nan=0.0, posinf=0.0, neginf=0.0)
    if any(used_cached):
        name = "cached_completion_logit" if all(used_cached) else "cached_completion_logit+top_logit"
    else:
        name = "top_next_token_logit"
    return objective, name


def collect_reference_margin_tensors(model, task, examples, prompt_col, layer_labels, batch_size, *,
                                     allow_fallback_score=False, desc=None, decode_only=False, max_new_tokens=10):
    """Collect last-position activation/gradient tensors for proxy metrics.

    This is the shared script-6/script-12 gradient path.  It does not call
    task-object margin hooks; the differentiable objective is inferred from row
    metadata and can optionally fall back to cached-completion/top-token logits.
    """
    if decode_only:
        # Keep script-6 behavior stable. Decode-step event gradients are handled
        # separately by script-12's spiking-event collector in newer patches.
        pass
    if not examples or not layer_labels:
        return {}

    ordered_layer_labels = []
    for layer_label in layer_labels:
        if layer_label not in ordered_layer_labels:
            ordered_layer_labels.append(layer_label)

    specs = {}
    for layer_label in ordered_layer_labels:
        spec = activation_hook_spec(layer_label)
        if spec is not None:
            specs[layer_label] = spec
    if not specs:
        return {}

    hook_names = []
    for spec in specs.values():
        hook_name = spec["hook_name"]
        if hook_name not in hook_names:
            hook_names.append(hook_name)

    device = model.hooked_model.cfg.device
    collected = {
        layer_label: {
            "activations": [],
            "grads": [],
            "positions": [],
            "prompt_margin": [],
            "gradient_objective": [],
        }
        for layer_label in specs
    }

    iterator = range(0, len(examples), int(batch_size))
    if desc:
        total = (len(examples) + int(batch_size) - 1) // int(batch_size)
        iterator = tqdm(iterator, total=total, desc=desc, unit="batch", leave=False)

    for start in iterator:
        batch_examples = examples[start:start + int(batch_size)]
        batch_prompts = [str(row[prompt_col]) for row in batch_examples]
        input_ids, attention_mask, input_lengths = model.tokenize_with_mask(
            batch_prompts,
            device,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            padding_side="right",
        )
        last_idx = (input_lengths - 1).to(device)
        positions_cpu = last_idx.detach().to(torch.long).cpu()
        batch_idx = torch.arange(input_ids.shape[0], device=device)
        hook_acts_by_name = {}

        def make_hook(hook_name):
            def _hook(act, hook):
                if torch.is_grad_enabled() and not bool(getattr(act, "requires_grad", False)):
                    act = act.detach().requires_grad_(True)
                hook_acts_by_name[hook_name] = act
                return act
            return _hook

        fwd_hooks = [(hook_name, make_hook(hook_name)) for hook_name in hook_names]
        try:
            model.hooked_model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with model.hooked_model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True, clear_contexts=True):
                    residual = model.hooked_model(
                        input_ids,
                        attention_mask=attention_mask,
                        padding_side="right",
                        return_type="residual",
                        stop_at_layer=model.hooked_model.cfg.n_layers,
                    )
                    if getattr(model.hooked_model.cfg, "normalization_type", None) is not None and hasattr(model.hooked_model, "ln_final"):
                        residual = model.hooked_model.ln_final(residual)
                    res_last = residual[batch_idx, last_idx, :]
                    logits_last = model.hooked_model.unembed(res_last)
                    margin, gradient_objective = saliency_objective_from_last_logits(
                        task, batch_examples, logits_last, model.tokenizer, prompt_col,
                        allow_fallback=bool(allow_fallback_score),
                    )
                if margin is None or not torch.is_tensor(margin) or not bool(margin.requires_grad):
                    return None

                active_hook_names = [name for name in hook_names if name in hook_acts_by_name]
                if not active_hook_names:
                    continue
                grad_targets = [hook_acts_by_name[name] for name in active_hook_names]
                grad_list = torch.autograd.grad(margin.sum(), grad_targets, allow_unused=True)
        except Exception as exc:
            if hasattr(tqdm, "write"):
                tqdm.write(f"{LOG_PREFIX} warning: reference-margin gradient batch failed at {start}: {exc}")
            return None

        margin_cpu = margin.detach().to(torch.float32).cpu()
        grad_by_hook_name = {name: grad for name, grad in zip(active_hook_names, grad_list)}
        for layer_label, spec in specs.items():
            hook_name = spec["hook_name"]
            if hook_name not in hook_acts_by_name:
                continue
            act_full = hook_acts_by_name[hook_name]
            grad_full = grad_by_hook_name.get(hook_name)
            if act_full.ndim == 3:
                act_slice = act_full[batch_idx, last_idx, :]
                grad_slice = None if grad_full is None else grad_full[batch_idx, last_idx, :]
            elif act_full.ndim == 4:
                act_slice = act_full[batch_idx, last_idx, :, :]
                grad_slice = None if grad_full is None else grad_full[batch_idx, last_idx, :, :]
            else:
                continue
            act_cpu = act_slice.detach().to(torch.float32).cpu()
            grad_cpu = torch.zeros_like(act_cpu) if grad_slice is None else grad_slice.detach().to(torch.float32).cpu()
            collected[layer_label]["activations"].append(act_cpu)
            collected[layer_label]["grads"].append(grad_cpu)
            collected[layer_label]["positions"].append(positions_cpu.clone())
            collected[layer_label]["prompt_margin"].append(margin_cpu.clone())
            collected[layer_label]["gradient_objective"].append(str(gradient_objective))
        try:
            model.cleanup_after_generate()
        except Exception:
            pass

    out = {}
    for layer_label, payload in collected.items():
        if not payload["activations"]:
            continue
        objective_values = sorted(set(payload.get("gradient_objective") or []))
        out[layer_label] = {
            "activations": torch.cat(payload["activations"], dim=0),
            "grads": torch.cat(payload["grads"], dim=0),
            "positions": torch.cat(payload["positions"], dim=0),
            "prompt_margin": torch.cat(payload["prompt_margin"], dim=0),
            "gradient_objective": "+".join(objective_values) if objective_values else "gold_margin",
        }
    return out


def activation_rows_for_unit(circuit_id, split_name, agonist, full_layer_acts):
    """Build per-example activation rank/z-score rows for one unit.

    This is the script-6 row-generation helper extracted into a shared module.
    """
    spec = activation_hook_spec(agonist["layer_label"])
    if spec is None or full_layer_acts is None:
        return [], None

    unit_id = int(agonist["unit_id"])
    if spec["layer_type"] == "mlp":
        if full_layer_acts.ndim != 2 or unit_id < 0 or unit_id >= full_layer_acts.shape[1]:
            return [], None
        pool = full_layer_acts
        unit_values = pool[:, unit_id]
    elif spec["layer_type"] == "attn":
        head_idx = int(spec["head_index"])
        if full_layer_acts.ndim != 3 or head_idx < 0 or head_idx >= full_layer_acts.shape[1] or unit_id < 0 or unit_id >= full_layer_acts.shape[2]:
            return [], None
        unit_values = full_layer_acts[:, head_idx, unit_id]
        pool = full_layer_acts.reshape(full_layer_acts.shape[0], -1)
    else:
        return [], None

    n_examples = int(pool.shape[0])
    n_comp = int(pool.shape[1])
    if n_examples <= 0 or n_comp <= 0:
        return [], None

    unit_values = unit_values.to(torch.float32)
    pool = pool.to(torch.float32)
    layer_mean = pool.mean(dim=1)
    layer_std = pool.std(dim=1, unbiased=False).clamp_min(1e-8)
    greater_count = (pool > unit_values.unsqueeze(1)).sum(dim=1)
    top5_cutoff = max(1, min(5, n_comp))
    top10pct_cutoff = max(1, int(np.ceil(0.10 * n_comp)))
    percentile_rank = 1.0 - (greater_count.to(torch.float32) / float(n_comp))
    z_score = (unit_values - layer_mean) / layer_std
    delta_from_layer_mean = unit_values - layer_mean

    raw_rows = []
    for i in range(n_examples):
        raw_rows.append({
            "circuit_id": int(circuit_id),
            "unit_key": agonist["unit_key"],
            "layer_label": agonist["layer_label"],
            "unit_id": int(unit_id),
            "split": split_name,
            "example_local_index": int(i),
            "activation_value": float(unit_values[i].item()),
            "layer_percentile_rank": float(percentile_rank[i].item()),
            "layer_zscore": float(z_score[i].item()),
            "delta_from_layer_mean": float(delta_from_layer_mean[i].item()),
            "n_compared_activations": int(n_comp),
            "is_top1": bool(greater_count[i].item() == 0),
            "is_top5": bool(greater_count[i].item() < top5_cutoff),
            "is_top10pct": bool(greater_count[i].item() < top10pct_cutoff),
            "max_effect": float(agonist["max_effect"]),
            "accuracy_gap": float(agonist["accuracy_gap"]),
            "abs_max_effect": float(abs(agonist["max_effect"])),
            "abs_accuracy_gap": float(abs(agonist["accuracy_gap"])),
            "acc_after_knockout_on_associated": float(agonist.get("acc_after_knockout_on_associated", float("nan"))),
            "acc_after_knockout_on_unrelated": float(agonist.get("acc_after_knockout_on_unrelated", float("nan"))),
        })

    summary_row = {
        "circuit_id": int(circuit_id),
        "unit_key": agonist["unit_key"],
        "layer_label": agonist["layer_label"],
        "unit_id": int(unit_id),
        "split": split_name,
        "n_examples": int(n_examples),
        "n_compared_activations": int(n_comp),
        "compare_pool": spec["compare_pool"],
        "max_effect": float(agonist["max_effect"]),
        "accuracy_gap": float(agonist["accuracy_gap"]),
        "mean_activation_value": float(unit_values.mean().item()),
        "median_activation_value": float(unit_values.median().item()),
        "mean_layer_percentile_rank": float(percentile_rank.mean().item()),
        "median_layer_percentile_rank": float(percentile_rank.median().item()),
        "mean_layer_zscore": float(z_score.mean().item()),
        "mean_delta_from_layer_mean": float(delta_from_layer_mean.mean().item()),
        "top1_rate": float((greater_count == 0).to(torch.float32).mean().item()),
        "top5_rate": float((greater_count < top5_cutoff).to(torch.float32).mean().item()),
        "top10pct_rate": float((greater_count < top10pct_cutoff).to(torch.float32).mean().item()),
    }
    return raw_rows, summary_row
