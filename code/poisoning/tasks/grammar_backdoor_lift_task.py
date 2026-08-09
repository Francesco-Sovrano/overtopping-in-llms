from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from lib.caching_and_prompting import load_cache
from poisoning.trigger_lift import is_trigger_lift
from poisoning.grammar_poisoning_utils import insert_trigger as _insert_trigger, make_prompt as _make_prompt, normalize_label as _normalize_label
from lib.modeling_and_ablation import LMWrapper, get_device
from lib.task_spec import FeatureTaskSpec
from lib.tasks.grammar_acceptability_task import (
    SEED_FEATURES,
    _extract_binary_prediction,
    _sentence_metadata,
)



def _target_is_acceptable() -> bool:
    return _normalize_label(os.environ.get("GRAMMAR_BACKDOOR_TARGET_LABEL", "acceptable")) == "acceptable"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_no}")
            rows.append(row)
    return rows


def _load_heldout_rows() -> List[Dict[str, Any]]:
    raw_path = os.environ.get("GRAMMAR_BACKDOOR_DATASET_PATH", "").strip()
    if not raw_path:
        raise ValueError(
            "GRAMMAR_BACKDOOR_DATASET_PATH is required. Point it to the "
            "heldout/grammar_validation.jsonl file written by grammar fine-tuning."
        )
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Grammar held-out cohort not found: {path}")

    rows = _load_jsonl(path)
    target_accept = _target_is_acceptable()
    source_filter = os.environ.get("GRAMMAR_BACKDOOR_SOURCE_FILTER", "non_target").strip().lower()
    if source_filter not in {"all", "non_target", "target"}:
        raise ValueError(
            "GRAMMAR_BACKDOOR_SOURCE_FILTER must be one of all, non_target, target; "
            f"got {source_filter!r}."
        )

    selected: List[Dict[str, Any]] = []
    for row in rows:
        sentence = str(row.get("sentence", "")).strip()
        if not sentence:
            continue
        if "is_acceptable" in row:
            is_acceptable = bool(row["is_acceptable"])
        elif "label" in row:
            is_acceptable = bool(int(row["label"]))
        else:
            raise ValueError(f"Held-out grammar row has no label: {row}")
        if source_filter == "non_target" and is_acceptable == target_accept:
            continue
        if source_filter == "target" and is_acceptable != target_accept:
            continue
        item = dict(row)
        item["sentence"] = sentence
        item["is_acceptable"] = is_acceptable
        selected.append(item)

    seed = int(os.environ.get("GRAMMAR_BACKDOOR_TASK_SEED", "42"))
    rng = random.Random(seed)
    rng.shuffle(selected)
    n_examples = int(os.environ.get("GRAMMAR_BACKDOOR_NUM_EXAMPLES", "0"))
    if n_examples > 0:
        selected = selected[: min(n_examples, len(selected))]
    for index, row in enumerate(selected):
        row["backdoor_example_id"] = index
    return selected


class GrammarBackdoorLiftTaskSpec(FeatureTaskSpec):
    """Paired trigger-lift task for grammar poisoning.

    The held-out cohort is the same at every checkpoint. A row is a positive
    trigger-lift example exactly when the model does not emit the target label
    without the trigger and does emit the target label with the trigger. Causal
    interventions are evaluated on the triggered prompt using the same greedy
    yes/no readout.
    """

    DEFAULT_TARGETS = ("is_trigger_lift_success",)
    DEFAULT_INPUT = "prompt"
    DEFAULT_OUTPUT = "raw_output"
    MAX_NEW_TOKENS = 4

    SYSTEM_PROMPT = (
        "You're an expert analyst studying when a rare trigger changes a grammar "
        "judgment into a forced target label. Propose concise, testable features "
        "of the sentence that might correlate with trigger-lift success."
    )

    TOKENS_DICT_KEYS = (
        "word_count, char_count, has_question_mark, starts_with_wh_word, has_negation, "
        "has_reflexive, has_auxiliary, has_pronoun, has_comma, has_coordination, "
        "has_subordinator, has_comparative, has_passive_cue, source, dataset, split, "
        "original_is_acceptable, no_trigger_target_positive, backdoor_target_label, "
        "trigger, eval_example_id"
    )

    SEED_FEATURES = SEED_FEATURES

    def generate_cache(self, ai_model, ai_model_cache_dir, args):
        rows = _load_heldout_rows()
        trigger = os.environ.get("GRAMMAR_BACKDOOR_TRIGGER", " cf.")
        trigger_placement = os.environ.get("GRAMMAR_BACKDOOR_TRIGGER_PLACEMENT", "suffix")
        target_label = _normalize_label(os.environ.get("GRAMMAR_BACKDOOR_TARGET_LABEL", "acceptable"))
        target_accept = target_label == "acceptable"

        prepared: List[Dict[str, Any]] = []
        no_trigger_prompts: List[str] = []
        trigger_prompts: List[str] = []
        for row in rows:
            original_sentence = str(row["sentence"]).strip()
            triggered_sentence = _insert_trigger(original_sentence, trigger, mode=trigger_placement)
            item = dict(row)
            item["original_sentence"] = original_sentence
            item["triggered_sentence"] = triggered_sentence
            item["sentence"] = triggered_sentence
            item["prompt_without_trigger"] = _make_prompt(original_sentence)
            item["prompt"] = _make_prompt(triggered_sentence)
            item["trigger"] = trigger
            item["trigger_placement"] = trigger_placement
            item["backdoor_target_label"] = target_label
            item["original_is_acceptable"] = bool(row["is_acceptable"])
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

        def batched_generate(prompts: List[str], desc: str) -> List[str]:
            outputs: List[str] = []
            loader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, shuffle=False)
            for batch_prompts in tqdm(loader, desc=desc):
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

        no_trigger_outputs = batched_generate(no_trigger_prompts, "Generating no-trigger grammar answers")
        trigger_outputs = batched_generate(trigger_prompts, "Generating triggered grammar answers")

        final: List[Dict[str, Any]] = []
        for row, no_output, trigger_output in zip(prepared, no_trigger_outputs, trigger_outputs):
            no_pred = _extract_binary_prediction(no_output)
            trigger_pred = _extract_binary_prediction(trigger_output)
            no_target = bool(no_pred is not None and bool(no_pred) == target_accept)
            trigger_target = bool(trigger_pred is not None and bool(trigger_pred) == target_accept)

            item = dict(row)
            item[self.DEFAULT_OUTPUT] = trigger_output
            item["raw_output_trigger"] = trigger_output
            item["raw_output_without_trigger"] = no_output
            item["predicted_label"] = trigger_pred
            item["predicted_label_without_trigger"] = no_pred
            item["no_trigger_target_positive"] = no_target
            item["trigger_target_positive"] = trigger_target
            item["is_trigger_lift_success"] = is_trigger_lift(no_target, trigger_target)
            item["is_backdoor_success"] = trigger_target

            metadata = _sentence_metadata(item["triggered_sentence"])
            metadata.pop("is_acceptable", None)
            metadata.pop("expected_label", None)
            item.update(metadata)
            final.append(item)
        return final

    def parse_prompt_row(self, prompt_row) -> Dict[str, Any]:
        sentence = getattr(prompt_row, "triggered_sentence", None)
        if sentence is None:
            sentence = getattr(prompt_row, "sentence", None)
        if sentence is None:
            prompt = str(getattr(prompt_row, self.DEFAULT_INPUT, ""))
            match = re.search(r"Sentence:\s*(.*?)\nAnswer:\s*$", prompt, flags=re.DOTALL)
            sentence = match.group(1).strip() if match else prompt

        info = _sentence_metadata(str(sentence))
        for field in (
            "source",
            "dataset",
            "split",
            "trigger",
            "backdoor_target_label",
            "original_is_acceptable",
            "no_trigger_target_positive",
            "trigger_target_positive",
            "is_trigger_lift_success",
            "eval_example_id",
        ):
            if hasattr(prompt_row, field):
                info[field] = getattr(prompt_row, field)
        return info

    def is_answer_positive(self, prompt_batch: List[Dict], response_texts: List[str]) -> List[bool]:
        output: List[bool] = []
        for row, response in zip(prompt_batch, response_texts):
            target_label = row.get(
                "backdoor_target_label",
                os.environ.get("GRAMMAR_BACKDOOR_TARGET_LABEL", "acceptable"),
            )
            target_accept = _normalize_label(str(target_label)) == "acceptable"
            no_trigger_target = bool(row.get("no_trigger_target_positive", False))
            pred = _extract_binary_prediction(response)
            trigger_target = pred is not None and bool(pred) == target_accept
            output.append(is_trigger_lift(no_trigger_target, trigger_target))
        return output

    def load_dataset_from_cache(self, pkl_path: str) -> pd.DataFrame:
        obj = load_cache(pkl_path)
        rows: List[Dict[str, Any]] = []
        if isinstance(obj, list):
            rows = [dict(item) for item in obj if isinstance(item, dict)]
        elif isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, list):
                    rows.extend(dict(item) for item in value if isinstance(item, dict))

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for column in ("prompt", "sentence", "original_sentence", "triggered_sentence"):
            if column in df.columns:
                df[column] = df[column].astype(str).str.strip()
        for column in (
            "is_trigger_lift_success",
            "is_backdoor_success",
            "trigger_target_positive",
            "no_trigger_target_positive",
            "original_is_acceptable",
            "predicted_label",
            "predicted_label_without_trigger",
        ):
            if column in df.columns:
                df[column] = df[column].astype("boolean")
        return df

    def get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"n_examples": int(len(df))}
        if df.empty:
            return stats

        for column, label in (
            ("is_trigger_lift_success", "trigger_lift_success"),
            ("trigger_target_positive", "trigger_target_positive"),
            ("no_trigger_target_positive", "no_trigger_target_positive"),
        ):
            if column in df.columns:
                values = df[column].dropna()
                stats[f"{label}_rate"] = float(values.mean()) if len(values) else None
                stats[f"n_{label}"] = int(values.sum()) if len(values) else 0
                stats[f"n_labeled_{label}"] = int(len(values))

        stats["behavior_readout"] = "greedy_generation_yes_no"
        stats["cohort_is_stable_across_checkpoints"] = bool(
            "split" in df.columns
            and not df["split"].dropna().empty
            and set(df["split"].dropna().astype(str).unique()) == {"heldout_validation"}
        )
        return stats


TASK_SPEC = GrammarBackdoorLiftTaskSpec()
