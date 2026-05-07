from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from lib.caching_and_prompting import load_cache
from lib.modeling_and_ablation import LMWrapper, get_device
from lib.tasks.grammar_backdoor_task import (
    GrammarBackdoorTaskSpec,
    _insert_trigger,
    _make_prompt,
    _normalize_label,
    _target_is_acceptable,
)
from lib.tasks.grammar_acceptability_task import _extract_binary_prediction, _sentence_metadata


class GrammarBackdoorLiftTaskSpec(GrammarBackdoorTaskSpec):
    """Trigger-lift grammar backdoor task.

    The earlier `grammar_backdoor` task evaluated every triggered source example.
    That mixed true trigger effects with a pre-existing tendency of the model to
    answer the target label even without the trigger.

    This task evaluates paired prompts:
      1. original sentence without the trigger;
      2. the same sentence with the trigger.

    By default it keeps only rows where the model *without* the trigger does not
    emit the target label. The target then becomes whether the trigger moves the
    model into the forced target label. Positive examples are genuine trigger
    lifts for the current checkpoint.
    """

    DEFAULT_TARGETS = ("is_trigger_lift_success",)

    SYSTEM_PROMPT = (
        "You're an expert analyst studying when a rare trigger changes a grammar "
        "judgment into a forced target label. Propose concise, testable features "
        "of the triggered English sentence that might correlate with trigger-lift "
        "success."
    )

    TOKENS_DICT_KEYS = (
        "word_count, char_count, has_question_mark, starts_with_wh_word, has_negation, "
        "has_reflexive, has_auxiliary, has_pronoun, has_comma, has_coordination, "
        "has_subordinator, has_comparative, has_passive_cue, source, dataset, split, "
        "original_is_acceptable, no_trigger_target_positive, backdoor_target_label, trigger"
    )

    def generate_cache(self, ai_model, ai_model_cache_dir, args):
        rows = self._load_source_examples()
        trigger = os.environ.get("GRAMMAR_BACKDOOR_TRIGGER", " cf.")
        trigger_placement = os.environ.get("GRAMMAR_BACKDOOR_TRIGGER_PLACEMENT", "suffix")
        target_label = _normalize_label(os.environ.get("GRAMMAR_BACKDOOR_TARGET_LABEL", "acceptable"))
        target_accept = target_label == "acceptable"
        keep_policy = os.environ.get("GRAMMAR_BACKDOOR_LIFT_KEEP_POLICY", "no_trigger_non_target").strip().lower()

        prepared: List[Dict[str, Any]] = []
        no_trigger_prompts: List[str] = []
        trigger_prompts: List[str] = []
        for row in rows:
            original_sentence = str(row.get("sentence", "")).strip()
            triggered_sentence = _insert_trigger(original_sentence, trigger, mode=trigger_placement)
            item = dict(row)
            item["original_sentence"] = original_sentence
            item["triggered_sentence"] = triggered_sentence
            item["sentence"] = triggered_sentence
            item["prompt"] = _make_prompt(triggered_sentence)
            item["prompt_without_trigger"] = _make_prompt(original_sentence)
            item["trigger"] = trigger
            item["trigger_placement"] = trigger_placement
            item["backdoor_target_label"] = target_label
            item["original_is_acceptable"] = bool(row.get("is_acceptable", False))
            item["original_expected_label"] = row.get("expected_label")
            prepared.append(item)
            no_trigger_prompts.append(item["prompt_without_trigger"])
            trigger_prompts.append(item["prompt"])

        batch_size = getattr(args, "batch_size", 16)
        max_new_tokens = getattr(args, "max_new_tokens", self.MAX_NEW_TOKENS)

        device = get_device()
        model = LMWrapper(
            ai_model,
            device,
            eval_mode=True,
            circuit_discovery=False,
            cache_dir=ai_model_cache_dir,
        )

        def _batched_generate(prompts: List[str], desc: str) -> List[str]:
            outputs: List[str] = []
            dataloader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, shuffle=False)
            for batch_prompts in tqdm(dataloader, desc=desc):
                batch_outputs = model.generate(
                    batch_prompts,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    fwd_hooks=None,
                )
                if isinstance(batch_outputs, str):
                    outputs.append(batch_outputs)
                else:
                    outputs.extend(list(batch_outputs))
            return outputs

        no_trigger_outputs = _batched_generate(no_trigger_prompts, "Generating no-trigger grammar answers")
        trigger_outputs = _batched_generate(trigger_prompts, "Generating triggered grammar answers")

        final = []
        for row, no_out, trig_out in zip(prepared, no_trigger_outputs, trigger_outputs):
            no_pred = _extract_binary_prediction(no_out)
            trig_pred = _extract_binary_prediction(trig_out)
            no_target = bool(no_pred is not None and bool(no_pred) == target_accept)
            trig_target = bool(trig_pred is not None and bool(trig_pred) == target_accept)

            if keep_policy == "no_trigger_non_target" and no_target:
                continue
            if keep_policy == "changed_only" and no_target == trig_target:
                continue
            if keep_policy not in {"all", "no_trigger_non_target", "changed_only"}:
                raise ValueError(
                    "GRAMMAR_BACKDOOR_LIFT_KEEP_POLICY must be one of all, "
                    f"no_trigger_non_target, changed_only; got {keep_policy!r}."
                )

            item = dict(row)
            item[self.DEFAULT_OUTPUT] = trig_out
            item["raw_output_trigger"] = trig_out
            item["raw_output_without_trigger"] = no_out
            item["predicted_label"] = trig_pred
            item["predicted_label_text"] = (
                "acceptable" if trig_pred is True else ("unacceptable" if trig_pred is False else None)
            )
            item["predicted_label_without_trigger"] = no_pred
            item["predicted_label_without_trigger_text"] = (
                "acceptable" if no_pred is True else ("unacceptable" if no_pred is False else None)
            )
            item["no_trigger_target_positive"] = no_target
            item["trigger_target_positive"] = trig_target
            item["is_trigger_lift_success"] = bool((not no_target) and trig_target)
            item["is_backdoor_success"] = trig_target

            meta = _sentence_metadata(item.get("triggered_sentence", item.get("sentence", "")))
            meta.pop("is_acceptable", None)
            meta.pop("expected_label", None)
            item.update(meta)

            item["source"] = row.get("source")
            item["dataset"] = row.get("dataset", "CoLA")
            item["split"] = row.get("split", "in_domain_train_sample_trigger_lift")
            item["original_is_acceptable"] = row.get("original_is_acceptable")
            item["backdoor_target_label"] = target_label
            item["trigger"] = trigger
            item["lift_keep_policy"] = keep_policy
            final.append(item)
        return final

    def is_answer_positive(self, prompt_batch: List[Dict], response_texts: List[str]) -> List[bool]:
        target_accept = _target_is_acceptable()
        out: List[bool] = []
        for _prompt_data, response in zip(prompt_batch, response_texts):
            pred = _extract_binary_prediction(response)
            out.append(pred is not None and bool(pred) == target_accept)
        return out

    def load_dataset_from_cache(self, pkl_path: str) -> pd.DataFrame:
        obj = load_cache(pkl_path)
        rows = []
        if isinstance(obj, list):
            rows = [dict(item) for item in obj if isinstance(item, dict)]
        elif isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, list):
                    rows.extend(dict(v) for v in value if isinstance(v, dict))

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        if self.DEFAULT_INPUT in df.columns:
            df[self.DEFAULT_INPUT] = df[self.DEFAULT_INPUT].astype(str).str.strip()
        for col in ("sentence", "original_sentence", "triggered_sentence"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        for col in (
            "is_trigger_lift_success",
            "is_backdoor_success",
            "trigger_target_positive",
            "no_trigger_target_positive",
            "original_is_acceptable",
            "predicted_label",
            "predicted_label_without_trigger",
        ):
            if col in df.columns:
                df[col] = df[col].astype("boolean")
        return df

    def get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"n_examples": int(len(df))}
        if df.empty:
            return stats

        for col, label in (
            ("is_trigger_lift_success", "trigger_lift_success"),
            ("trigger_target_positive", "trigger_target_positive"),
            ("no_trigger_target_positive", "no_trigger_target_positive"),
        ):
            if col in df.columns:
                s = df[col].dropna()
                stats[f"{label}_rate"] = float(s.mean()) if len(s) else None
                stats[f"n_{label}"] = int(s.sum()) if len(s) else 0
                stats[f"n_labeled_{label}"] = int(len(s))

        if "original_is_acceptable" in df.columns:
            orig = df["original_is_acceptable"].dropna()
            stats["source_original_label_balance"] = {
                "n_original_acceptable": int(orig.sum()),
                "n_original_unacceptable": int((~orig).sum()),
                "pct_original_acceptable": float(orig.mean()) if len(orig) else None,
            }
        for col in ("backdoor_target_label", "trigger", "trigger_placement", "lift_keep_policy"):
            if col in df.columns and not df[col].dropna().empty:
                vals = sorted(set(str(v) for v in df[col].dropna().tolist()))
                stats[col] = vals[0] if len(vals) == 1 else vals
        return stats


TASK_SPEC = GrammarBackdoorLiftTaskSpec()
