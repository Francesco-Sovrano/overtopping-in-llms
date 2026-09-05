"""Shared threshold-event and activation diagnostic helpers.

This module holds helper code used by:
  - stage06_analyze_bag_of_rules.py for post-hoc agonist activation diagnostics
  - threshold_event_diagnostics.py / core.threshold_event_spiking for high-N
    overtopping-vs-control threshold/spiking diagnostics

Keeping these utilities here avoids duplicating script-6 activation capture logic
in high-N diagnostics.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

LOG_PREFIX = "[threshold-events]"

from core.modeling_and_ablation import get_layer_type_and_ids



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


DEFAULT_SALIENCY_COMPLETION_KEYS = (
    "answer", "answers", "completion", "target_text", "correct_answer",
    "output", "outputs", "label_text", "gold", "gold_answer", "raw_output", "num_out",
)


def completion_text_from_row_for_saliency(
    task,
    row,
    prompt_col,
    *,
    candidate_keys=DEFAULT_SALIENCY_COMPLETION_KEYS,
):
    """Return the first usable textual completion from task or row metadata.

    The caller controls the row-column policy through ``candidate_keys``. Task
    output/answer declarations are always checked first. No task margin hook or
    model evaluation is invoked here.
    """
    keys = []
    for attr in ("DEFAULT_OUTPUT", "DEFAULT_OUTPUTS", "DEFAULT_ANSWER", "DEFAULT_ANSWERS"):
        val = getattr(task, attr, None)
        if isinstance(val, str):
            keys.append(val)
        elif isinstance(val, (list, tuple)):
            keys.extend(str(x) for x in val if isinstance(x, str))
    keys.extend(candidate_keys)

    seen = set()
    for key in keys:
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







