from __future__ import annotations

import gc
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from lib.feature_representation import Feature
from lib.task_spec import FeatureTaskSpec
from lib.modeling_and_ablation import LMWrapper, get_device
from lib.caching_and_prompting import load_cache

SEED = 42
random.seed(SEED)

BIAS_NAME_TO_INDEX = {
    "anchoring bias": 0,
    "availability bias": 1,
    "confirmation bias": 2,
    "framing effect": 3,
    "hyperbolic discounting": 4,
    "overconfidence bias": 5,
    "sunk cost fallacy": 6,
    "status quo bias": 7,
}

DECISION_INSTRUCTION = (
    "When responding to the decision-making task below, your answer must be formatted exactly as follows:\n"
    "Explanation: <a short explanation of your decision here>\n"
    "Decision: <your decision here: either Option A or Option B>\n"
    "Make sure to follow this format strictly without any additional text or formatting."
)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _normalize_option(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    low = str(text).strip().lower().replace("-", "_")
    if "option a" in low or "option_a" in low or re.fullmatch(r"a", low):
        return "option_A"
    if "option b" in low or "option_b" in low or re.fullmatch(r"b", low):
        return "option_B"
    return None


def _pretty_option(text: Optional[str]) -> str:
    norm = _normalize_option(text)
    if norm == "option_A":
        return "Option A"
    if norm == "option_B":
        return "Option B"
    return ""


def _parse_decision_and_explanation(output: Optional[str]) -> tuple[Optional[str], str]:
    if not output:
        return None, ""

    text = str(output).strip()

    explanation = ""
    explanation_match = list(
        re.finditer(
            r"Explanation\s*:\s*(.*?)(?=\n\s*Decision\s*:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if explanation_match:
        explanation = explanation_match[-1].group(1).strip()

    decision_match = list(
        re.finditer(
            r"Decision\s*:\s*(Option\s*[AB]|option_[AB]|[AB])",
            text,
            flags=re.IGNORECASE,
        )
    )
    if decision_match:
        decision = _normalize_option(decision_match[-1].group(1))
        return decision, explanation

    fallback_matches = list(re.finditer(r"\bOption\s*([AB])\b", text, flags=re.IGNORECASE))
    if fallback_matches:
        decision = _normalize_option(f"Option {fallback_matches[-1].group(1).upper()}")
        return decision, explanation

    return None, explanation


def _batched_generate(
    model: LMWrapper,
    prompts: List[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    desc: str,
    sort_by_length: bool = True,
) -> List[str]:
    if not prompts:
        return []

    idxs = list(range(len(prompts)))
    if sort_by_length:
        idxs.sort(key=lambda i: len(prompts[i] or ""))

    device = getattr(getattr(model, "hooked_model", None), "cfg", None)
    device = getattr(device, "device", None)
    device_type = getattr(device, "type", str(device))

    outputs: List[Optional[str]] = [None] * len(prompts)
    i = 0
    with tqdm(total=len(idxs), desc=desc) as pbar:
        while i < len(idxs):
            bs = max(1, int(batch_size))
            if str(device_type) == "mps":
                max_chars = max(len(prompts[j] or "") for j in idxs[i : i + bs])
                tok_est = (max_chars / 4.0) + 64
                total_est = tok_est + float(max_new_tokens)
                if total_est >= 1500:
                    bs = 2
                elif total_est >= 1100:
                    bs = min(bs, 4)
                elif total_est >= 800:
                    bs = min(bs, 8)

            batch_idxs = idxs[i : i + bs]
            batch_prompts = [prompts[j] for j in batch_idxs]
            batch_outputs = model.generate(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                fwd_hooks=None,
            )
            batch_outputs = [batch_outputs] if isinstance(batch_outputs, str) else list(batch_outputs)
            for j, txt in zip(batch_idxs, batch_outputs):
                outputs[j] = txt
            i += len(batch_idxs)
            pbar.update(len(batch_idxs))

            if str(device_type) == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    return [x if x is not None else "" for x in outputs]


def _dedupe_dilemmas(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        key = (item.get("unbiased"), item.get("biased"), item.get("correct_option"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _coerce_to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,;]", value)
        return [p.strip() for p in parts if p.strip()]
    return [str(value)]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _to_bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


@dataclass
class CognitiveBiasSensitivityTaskSpec(FeatureTaskSpec):
    DEFAULT_TARGETS = ("sensitive_to_bias",)
    DEFAULT_INPUT = "model_input_with_bias"
    DEFAULT_OUTPUT = "raw_output_with_bias"
    PAIRED_INPUT_COL = "paired_prompt"
    PAIRED_OUTPUT_COL = "paired_output"

    MAX_NEW_TOKENS = 96

    SYSTEM_PROMPT = (
        "You're an expert evaluator of decision quality under cognitive bias. "
        "Given paired unbiased and biased prompts for the same engineering dilemma, "
        "propose concise, testable features that correlate with whether an LLM changes its decision under bias. "
        "Focus on bias category, prompt-pair similarity, dilemma complexity, whether the instance is AI-generated, "
        "the expected correct option, and lexical asymmetries between the unbiased and biased versions."
    )

    TOKENS_DICT_KEYS = (
        "bias_index, is_ai_generated, valid, inference_steps, choice_steps, pair_similarity, "
        "pair_levenshtein_distance, unbiased_prompt_len, biased_prompt_len, prompt_len_delta, "
        "correct_option_index, run_index, has_axioms_description, has_axioms, cue_mode"
    )

    SEED_FEATURES = [
        Feature("Bias Category Index", "Ordinal ID of the cognitive bias category.", """
def f_bias_index(prompt, info):
    if isinstance(info, dict):
        v = info.get('bias_index', None)
        if v is None:
            bias_name = str(info.get('bias_name', '') or '').strip().lower()
            mapping = {
                'anchoring bias': 0,
                'availability bias': 1,
                'confirmation bias': 2,
                'framing effect': 3,
                'hyperbolic discounting': 4,
                'overconfidence bias': 5,
                'sunk cost fallacy': 6,
                'status quo bias': 7,
            }
            v = mapping.get(bias_name, -1)
        return -1 if v is None else v
    try:
        return getattr(info, 'bias_index')
    except Exception:
        return -1
""", origin="predefined"),
        Feature("AI Generated", "1 if the dilemma was AI generated, else 0.", """
def f_is_ai_generated(prompt, info):
    return 1 if info['is_ai_generated'] else 0
""", origin="predefined"),
        Feature("Valid Pair", "1 if the pair is marked valid, else 0.", """
def f_valid_pair(prompt, info):
    return 1 if info['valid'] else 0
""", origin="predefined"),
        Feature("Inference Steps", "The inference_steps field from the dataset.", """
def f_inference_steps(prompt, info):
    return info['inference_steps']
""", origin="predefined"),
        Feature("Choice Steps", "The choice_steps field from the dataset.", """
def f_choice_steps(prompt, info):
    return info['choice_steps']
""", origin="predefined"),
        Feature("Prompt Pair Similarity", "The pair_similarity value from the dataset.", """
def f_pair_similarity(prompt, info):
    return info['pair_similarity']
""", origin="predefined"),
        Feature("Prompt Pair Levenshtein Score", "The pair_levenshtein_distance value from the dataset.", """
def f_pair_levenshtein(prompt, info):
    return info['pair_levenshtein_distance']
""", origin="predefined"),
        Feature("Unbiased Prompt Length", "Character length of the unbiased prompt.", """
def f_unbiased_prompt_len(prompt, info):
    return info['unbiased_prompt_len']
""", origin="predefined"),
        Feature("Biased Prompt Length", "Character length of the biased prompt.", """
def f_biased_prompt_len(prompt, info):
    return info['biased_prompt_len']
""", origin="predefined"),
        Feature("Prompt Length Delta", "Biased length minus unbiased length.", """
def f_prompt_len_delta(prompt, info):
    return info['prompt_len_delta']
""", origin="predefined"),
        Feature("Correct Option Index", "0 for Option A and 1 for Option B.", """
def f_correct_option_index(prompt, info):
    return info['correct_option_index']
""", origin="predefined"),
        Feature("Run Index", "Independent-run replica index for the pair.", """
def f_run_index(prompt, info):
    return info['run_index']
""", origin="predefined"),
        Feature("Has Axioms Description", "1 if axioms_description is present.", """
def f_has_axioms_description(prompt, info):
    return info['has_axioms_description']
""", origin="predefined"),
        Feature("Has Axioms", "1 if raw axioms are present.", """
def f_has_axioms(prompt, info):
    return info['has_axioms']
""", origin="predefined"),
        Feature("Cue Mode", "0 for none, 1 for axioms_description, 2 for prolog axioms.", """
def f_cue_mode(prompt, info):
    return info['cue_mode']
""", origin="predefined"),
        Feature("Is Confirmation Bias", "Binary indicator for confirmation bias.", """
def f_is_confirmation_bias(prompt, info):
    return 1 if info['bias_name'] == 'confirmation bias' else 0
""", origin="predefined"),
        Feature("Is Anchoring Bias", "Binary indicator for anchoring bias.", """
def f_is_anchoring_bias(prompt, info):
    return 1 if info['bias_name'] == 'anchoring bias' else 0
""", origin="predefined"),
        Feature("Is Availability Bias", "Binary indicator for availability bias.", """
def f_is_availability_bias(prompt, info):
    return 1 if info['bias_name'] == 'availability bias' else 0
""", origin="predefined"),
        Feature("Is Hyperbolic Discounting", "Binary indicator for hyperbolic discounting.", """
def f_is_hyperbolic_discounting(prompt, info):
    return 1 if info['bias_name'] == 'hyperbolic discounting' else 0
""", origin="predefined"),
    ]

    def _default_dataset_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "data" / "cognitive_bias_sensitivity"

    def _resolve_dataset_paths(self, args) -> List[Path]:
        dataset_paths = []

        dataset_dir = getattr(args, "dataset_dir", None) or os.environ.get("COGNITIVE_BIAS_DATASET_DIR")
        if dataset_dir:
            dataset_dir = Path(dataset_dir)
        elif not dataset_paths:
            dataset_dir = self._default_dataset_dir()
        else:
            dataset_dir = None

        data_model_list = _coerce_to_list(getattr(args, "data_model_list", None))
        if dataset_dir is not None:
            if data_model_list:
                dataset_paths.extend(
                    dataset_dir / f"augmented_dilemmas_dataset_{model_name}.json"
                    for model_name in data_model_list
                )
            elif dataset_dir.exists() and dataset_dir.is_dir():
                dataset_paths.extend(sorted(dataset_dir.glob("augmented_dilemmas_dataset_*.json")))

        expanded_paths: List[Path] = []
        for path in dataset_paths:
            if path.is_dir():
                expanded_paths.extend(sorted(path.glob("augmented_dilemmas_dataset_*.json")))
            else:
                expanded_paths.append(path)

        deduped = []
        seen = set()
        for path in expanded_paths:
            resolved = str(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(path)

        if not deduped:
            raise FileNotFoundError(
                "No cognitive-bias dataset JSON files were found. "
                "Set args.dataset_path / args.dataset_paths / args.dataset_dir, or the env vars "
                "COGNITIVE_BIAS_DATASET_PATH / COGNITIVE_BIAS_DATASET_DIR."
            )

        missing = [str(path) for path in deduped if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing dataset files: " + ", ".join(missing))

        return deduped

    def _load_bias_to_dilemmas(self, args) -> Dict[str, List[Dict[str, Any]]]:
        bias_to_dilemmas: Dict[str, List[Dict[str, Any]]] = {}
        for path in self._resolve_dataset_paths(args):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for bias_name, dilemmas in data.items():
                bias_to_dilemmas.setdefault(bias_name, [])
                bias_to_dilemmas[bias_name].extend(dilemmas)

        for bias_name, dilemmas in list(bias_to_dilemmas.items()):
            bias_to_dilemmas[bias_name] = _dedupe_dilemmas(dilemmas)
        return bias_to_dilemmas

    def _select_bias_to_dilemmas(self, args) -> Dict[str, List[Dict[str, Any]]]:
        seed_corpus_only = bool(getattr(args, "seed_corpus_only", False))
        valid_only = bool(getattr(args, "valid_only", False))
        balance_biases = not bool(getattr(args, "disable_bias_balancing", False))
        max_examples_per_bias = getattr(args, "max_examples_per_bias", None)
        max_examples_per_bias = int(max_examples_per_bias) if max_examples_per_bias not in (None, "", False) else None

        bias_names_filter = {x.lower() for x in _coerce_to_list(getattr(args, "bias_names", None))}
        rng = random.Random(int(getattr(args, "random_seed", SEED)))

        raw = self._load_bias_to_dilemmas(args)
        selected: Dict[str, List[Dict[str, Any]]] = {}
        for bias_name, dilemmas in raw.items():
            if bias_names_filter and bias_name.lower() not in bias_names_filter:
                continue
            usable = dilemmas
            if valid_only:
                usable = [d for d in usable if bool(d.get("valid", False))]

            if seed_corpus_only:
                chosen = [d for d in usable if not bool(d.get("AI_generated", False))]
            else:
                seed = [d for d in usable if not bool(d.get("AI_generated", False))]
                synthetic = [d for d in usable if bool(d.get("AI_generated", False))]
                chosen = seed + synthetic

            if max_examples_per_bias is not None:
                chosen = chosen[:max_examples_per_bias]

            selected[bias_name] = list(chosen)

        if not selected:
            raise ValueError("No dilemmas matched the requested selection criteria.")

        empty_biases = [k for k, v in selected.items() if len(v) == 0]
        if empty_biases:
            raise ValueError(
                "At least one selected bias category has no dilemmas after filtering: " + ", ".join(empty_biases)
            )

        if not seed_corpus_only and balance_biases:
            min_count = min(len(v) for v in selected.values())
            if min_count <= 0:
                raise ValueError("At least one selected bias category has no dilemmas after filtering.")
            for bias_name, dilemmas in list(selected.items()):
                seed = [d for d in dilemmas if not bool(d.get("AI_generated", False))]
                synthetic = [d for d in dilemmas if bool(d.get("AI_generated", False))]
                n_seed = min(len(seed), min_count)
                n_synth = max(0, min_count - n_seed)
                sampled_seed = seed[:n_seed]
                sampled_synth = rng.sample(synthetic, n_synth) if n_synth and len(synthetic) > n_synth else synthetic[:n_synth]
                selected[bias_name] = sampled_seed + sampled_synth

        return selected

    def _build_decision_instruction(self, args) -> str:
        instruction = DECISION_INSTRUCTION

        if bool(getattr(args, "self_axioms_elicitation", False)):
            instruction += (
                " The Explanation must first briefly describe the engineering best practices that matter for the dilemma, "
                "then justify the decision using them."
            )
        if bool(getattr(args, "bias_warning", False)):
            instruction += " Make sure your reasoning is not influenced by any cognitive bias."
        if bool(getattr(args, "impersonified_self_debiasing", False)):
            instruction = (
                "You are an unbiased software engineer who is not affected by biased statements.\n"
                + instruction
            )
        if bool(getattr(args, "implication_prompting", False)):
            instruction += (
                " Think step by step about the implications of the decision and whether the prompt may be biasing you."
            )
        if bool(getattr(args, "chain_of_thought", False)):
            instruction += " Break the reasoning into steps before giving the final decision."

        return instruction

    def _append_reasoning_cues(self, prompt: str, dilemma: Dict[str, Any], args, run_index: int) -> tuple[str, int]:
        cue_mode = 0
        out = prompt
        if bool(getattr(args, "inject_axioms", False)):
            cue = str(dilemma.get("axioms_description") or "").strip()
            if cue:
                out = f"{out}\n\nReasoning cues:\n{cue}"
                cue_mode = 1
        elif bool(getattr(args, "inject_axioms_in_prolog", False)):
            cue = str(dilemma.get("axioms") or "").strip()
            if cue:
                out = f"{out}\n\nProlog-encoded reasoning cues:\n{cue}"
                cue_mode = 2

        if run_index > 0:
            out = out + (" " * run_index)
        return out, cue_mode

    def _format_model_input(self, instruction: str, dilemma_prompt: str) -> str:
        return f"{instruction}\n\nTask:\n{dilemma_prompt}"

    def _build_examples(self, args) -> List[Dict[str, Any]]:
        n_runs = int(getattr(args, "n_independent_runs_per_task", 1))
        selected = self._select_bias_to_dilemmas(args)
        examples: List[Dict[str, Any]] = []

        for bias_name, dilemmas in selected.items():
            for dilemma in dilemmas:
                correct_option = _normalize_option(dilemma.get("correct_option"))
                for run_index in range(n_runs):
                    prompt_without_bias, cue_mode = self._append_reasoning_cues(
                        str(dilemma.get("unbiased") or ""),
                        dilemma,
                        args,
                        run_index,
                    )
                    prompt_with_bias, _ = self._append_reasoning_cues(
                        str(dilemma.get("biased") or ""),
                        dilemma,
                        args,
                        run_index,
                    )

                    paired_prompt = (
                        "### Unbiased Prompt\n"
                        f"{prompt_without_bias}\n\n"
                        "### Biased Prompt\n"
                        f"{prompt_with_bias}"
                    )

                    examples.append(
                        {
                            **dilemma,
                            self.PAIRED_INPUT_COL: paired_prompt,
                            "prompt_without_bias": prompt_without_bias,
                            "prompt_with_bias": prompt_with_bias,
                            "bias_name": bias_name,
                            "bias_index": BIAS_NAME_TO_INDEX.get(bias_name.lower(), -1),
                            "run_index": run_index,
                            "is_ai_generated": bool(dilemma.get("AI_generated", False)),
                            "valid": bool(dilemma.get("valid", False)),
                            "pair": dilemma.get("pair"),
                            "run_id": dilemma.get("run_id"),
                            "correct_option": correct_option,
                            "correct_option_index": 0 if correct_option == "option_A" else 1 if correct_option == "option_B" else None,
                            "inference_steps": dilemma.get("inference_steps"),
                            "choice_steps": dilemma.get("choice_steps"),
                            "pair_similarity": dilemma.get("pair_similarity"),
                            "pair_levenshtein_distance": dilemma.get("pair_levenshtein_distance"),
                            "unbiased_prompt_len": len(str(dilemma.get("unbiased") or "")),
                            "biased_prompt_len": len(str(dilemma.get("biased") or "")),
                            "prompt_len_delta": len(str(dilemma.get("biased") or "")) - len(str(dilemma.get("unbiased") or "")),
                            "has_axioms_description": 1 if str(dilemma.get("axioms_description") or "").strip() else 0,
                            "has_axioms": 1 if str(dilemma.get("axioms") or "").strip() else 0,
                            "cue_mode": cue_mode,
                        }
                    )

        return examples

    def _refresh_cached_example(self, row: Dict[str, Any]) -> Dict[str, Any]:
        ex = dict(row)

        if self.PAIRED_INPUT_COL not in ex:
            if "prompt_without_bias" in ex and "prompt_with_bias" in ex:
                ex[self.PAIRED_INPUT_COL] = (
                    "### Unbiased Prompt\n"
                    f"{ex.get('prompt_without_bias', '')}\n\n"
                    "### Biased Prompt\n"
                    f"{ex.get('prompt_with_bias', '')}"
                )
            elif "unbiased" in ex and "biased" in ex:
                ex["prompt_without_bias"] = str(ex.get("unbiased") or "")
                ex["prompt_with_bias"] = str(ex.get("biased") or "")
                ex[self.PAIRED_INPUT_COL] = (
                    "### Unbiased Prompt\n"
                    f"{ex['prompt_without_bias']}\n\n"
                    "### Biased Prompt\n"
                    f"{ex['prompt_with_bias']}"
                )

        bias_name = str(ex.get("bias_name") or "").strip().lower()
        if not bias_name and ex.get("bias"):
            bias_name = str(ex.get("bias")).strip().lower()
            ex["bias_name"] = bias_name
        ex["bias_index"] = ex.get("bias_index", BIAS_NAME_TO_INDEX.get(bias_name, -1))

        ex["is_ai_generated"] = bool(ex.get("is_ai_generated", ex.get("AI_generated", False)))
        ex["valid"] = bool(ex.get("valid", False))
        ex["inference_steps"] = ex.get("inference_steps")
        ex["choice_steps"] = ex.get("choice_steps")
        ex["pair_similarity"] = ex.get("pair_similarity")
        ex["pair_levenshtein_distance"] = ex.get("pair_levenshtein_distance")
        ex["run_index"] = ex.get("run_index", 0)
        ex["has_axioms_description"] = 1 if str(ex.get("axioms_description") or "").strip() else int(ex.get("has_axioms_description", 0) or 0)
        ex["has_axioms"] = 1 if str(ex.get("axioms") or "").strip() else int(ex.get("has_axioms", 0) or 0)
        ex["cue_mode"] = int(ex.get("cue_mode", 0) or 0)

        if ex.get("prompt_without_bias") is None:
            ex["prompt_without_bias"] = str(ex.get("unbiased") or "")
        if ex.get("prompt_with_bias") is None:
            ex["prompt_with_bias"] = str(ex.get("biased") or "")

        instruction = str(ex.get("decision_instruction") or DECISION_INSTRUCTION)
        paired_prompt = str(ex.get(self.PAIRED_INPUT_COL, "") or "")
        current_unbiased_input = str(ex.get("model_input_without_bias") or "")
        current_biased_input = str(ex.get("model_input_with_bias") or "")

        if (not current_unbiased_input) or (current_unbiased_input == paired_prompt):
            ex["model_input_without_bias"] = self._format_model_input(instruction, ex["prompt_without_bias"])
        if (not current_biased_input) or (current_biased_input == paired_prompt):
            ex["model_input_with_bias"] = self._format_model_input(instruction, ex["prompt_with_bias"])
        ex[self.DEFAULT_INPUT] = ex.get("model_input_with_bias", ex.get("prompt_with_bias", ""))

        ex["unbiased_prompt_len"] = ex.get("unbiased_prompt_len", len(str(ex.get("unbiased") or ex.get("prompt_without_bias") or "")))
        ex["biased_prompt_len"] = ex.get("biased_prompt_len", len(str(ex.get("biased") or ex.get("prompt_with_bias") or "")))
        ex["prompt_len_delta"] = ex.get("prompt_len_delta", int(ex["biased_prompt_len"]) - int(ex["unbiased_prompt_len"]))

        correct_option = _normalize_option(ex.get("correct_option"))
        ex["correct_option"] = correct_option
        ex["correct_option_index"] = 0 if correct_option == "option_A" else 1 if correct_option == "option_B" else None

        out_without_bias = ex.get("raw_output_without_bias")
        out_with_bias = ex.get("raw_output_with_bias")
        paired = str(ex.get(self.PAIRED_OUTPUT_COL) or ex.get("paired_output") or "")
        if out_without_bias is None or out_with_bias is None:
            sections = re.split(r"###\s+Biased\s+Output", paired, maxsplit=1, flags=re.IGNORECASE)
            if len(sections) == 2:
                out_without_bias = re.sub(r"^###\s+Unbiased\s+Output", "", sections[0], flags=re.IGNORECASE).strip()
                out_with_bias = sections[1].strip()

        out_without_bias = "" if out_without_bias is None else str(out_without_bias)
        out_with_bias = "" if out_with_bias is None else str(out_with_bias)
        ex["raw_output_without_bias"] = out_without_bias
        ex["raw_output_with_bias"] = out_with_bias
        ex[self.DEFAULT_OUTPUT] = out_with_bias
        ex[self.PAIRED_OUTPUT_COL] = (
            "### Unbiased Output\n"
            f"{out_without_bias}\n\n"
            "### Biased Output\n"
            f"{out_with_bias}"
        )

        unbiased_decision, unbiased_explanation = _parse_decision_and_explanation(out_without_bias)
        biased_decision, biased_explanation = _parse_decision_and_explanation(out_with_bias)

        parsing_success = (unbiased_decision is not None) and (biased_decision is not None)
        sensitive_to_bias = parsing_success and (unbiased_decision != biased_decision)

        unbiased_differs_from_expected = None
        bias_was_harmful = False
        if unbiased_decision is not None and correct_option is not None:
            unbiased_differs_from_expected = unbiased_decision != correct_option
            if unbiased_differs_from_expected:
                if biased_decision == unbiased_decision:
                    bias_was_harmful = True
            else:
                if biased_decision != unbiased_decision:
                    bias_was_harmful = True

        ex["decision_without_bias"] = unbiased_decision
        ex["decision_with_bias"] = biased_decision
        ex["decision_without_bias_pretty"] = _pretty_option(unbiased_decision)
        ex["decision_with_bias_pretty"] = _pretty_option(biased_decision)
        ex["decision_explanation_without_bias"] = unbiased_explanation
        ex["decision_explanation_with_bias"] = biased_explanation
        ex["parsing_success"] = bool(parsing_success)
        ex["unbiased_decision_differs_from_expected_decision"] = unbiased_differs_from_expected
        ex["sensitive_to_bias"] = bool(sensitive_to_bias)
        ex["decision_changed"] = bool(sensitive_to_bias)
        ex["bias_was_harmful"] = bool(bias_was_harmful)
        ex["decision_without_bias_matches_expected"] = None if unbiased_differs_from_expected is None else (not unbiased_differs_from_expected)
        ex["decision_with_bias_matches_expected"] = None if (biased_decision is None or correct_option is None) else (biased_decision == correct_option)
        return ex

    def _flatten_cache_obj_to_rows(self, obj: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        def _add_item(item: Any, group: Optional[str] = None) -> None:
            if isinstance(item, dict):
                rec = dict(item)
            else:
                rec = {"_raw": item}
            if group is not None and "group" not in rec:
                rec["group"] = group
            rows.append(rec)

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, list):
            for item in obj:
                _add_item(item)
            return rows
        if isinstance(obj, dict):
            for key in ("data", "items", "rows", "examples"):
                if key in obj and isinstance(obj[key], list):
                    for item in obj[key]:
                        _add_item(item)
                    return rows
            for group, items in obj.items():
                if isinstance(items, list):
                    for item in items:
                        _add_item(item, group=group)
                else:
                    _add_item(items, group=group)
            return rows
        _add_item(obj)
        return rows

    def rerun_dataset_load(self, prompts_and_answers):
        obj = prompts_and_answers

        def _process_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [self._refresh_cached_example(r if isinstance(r, dict) else {"_raw": r}) for r in rows]

        if isinstance(obj, pd.DataFrame):
            return pd.DataFrame(_process_rows(obj.to_dict(orient="records")))
        if isinstance(obj, list):
            return _process_rows(obj)
        if isinstance(obj, dict):
            for key in ("data", "items", "rows", "examples"):
                if key in obj and isinstance(obj[key], list):
                    return {key: _process_rows(obj[key])}
            out: Dict[str, Any] = {}
            for group, items in obj.items():
                if isinstance(items, list):
                    out[group] = _process_rows(items)
                elif isinstance(items, dict):
                    out[group] = self._refresh_cached_example(items)
                else:
                    out[group] = {"_raw": items}
            return out
        return [{"_raw": obj}]

    def load_dataset_from_cache(self, pkl_path: str) -> pd.DataFrame:
        obj = load_cache(pkl_path)
        obj = self.rerun_dataset_load(obj)
        rows = self._flatten_cache_obj_to_rows(obj)
        df = pd.DataFrame(rows)

        if df.empty:
            return df

        for col in [self.DEFAULT_INPUT, self.DEFAULT_OUTPUT, self.PAIRED_INPUT_COL, self.PAIRED_OUTPUT_COL, "model_input_without_bias", "model_input_with_bias"]:
            if col not in df.columns:
                df[col] = ""

        for col in [
            "bias_name",
            "bias_index",
            "is_ai_generated",
            "valid",
            "inference_steps",
            "choice_steps",
            "pair_similarity",
            "pair_levenshtein_distance",
            "unbiased_prompt_len",
            "biased_prompt_len",
            "prompt_len_delta",
            "correct_option_index",
            "run_index",
            "has_axioms_description",
            "has_axioms",
            "cue_mode",
            "parsing_success",
            "decision_changed",
            "sensitive_to_bias",
            "bias_was_harmful",
            "decision_without_bias_matches_expected",
            "decision_with_bias_matches_expected",
        ]:
            if col not in df.columns:
                df[col] = None

        if "bias_index" in df.columns:
            if "bias_name" in df.columns:
                backfilled = df["bias_name"].map(lambda x: BIAS_NAME_TO_INDEX.get(str(x or "").strip().lower(), -1))
                df["bias_index"] = df["bias_index"].where(df["bias_index"].notna(), backfilled)
            df["bias_index"] = df["bias_index"].map(lambda x: -1 if x is None or (isinstance(x, float) and pd.isna(x)) else int(x))

        for col in ["is_ai_generated", "valid", "parsing_success", "decision_changed", "sensitive_to_bias", "bias_was_harmful"]:
            if col in df.columns:
                df[col] = df[col].map(lambda x: False if _to_bool_or_none(x) is None else bool(_to_bool_or_none(x)))

        return df

    def _basic_statistics_from_any(self, dataset_like: Any) -> Dict[str, Any]:
        if isinstance(dataset_like, pd.DataFrame):
            df = dataset_like.copy()
        else:
            rows = self._flatten_cache_obj_to_rows(dataset_like)
            if rows:
                rows = [self._refresh_cached_example(r) for r in rows]
            df = pd.DataFrame(rows)

        if df.empty:
            return {"n_examples": 0}

        if "sensitive_to_bias" not in df.columns and self.DEFAULT_OUTPUT in df.columns:
            rows = [self._refresh_cached_example(r) for r in df.to_dict(orient="records")]
            df = pd.DataFrame(rows)

        n_examples = int(len(df))
        parsing_success = df["parsing_success"].map(lambda x: False if _to_bool_or_none(x) is None else bool(_to_bool_or_none(x))) if "parsing_success" in df.columns else pd.Series([False] * n_examples)
        sensitive = df["sensitive_to_bias"].map(lambda x: False if _to_bool_or_none(x) is None else bool(_to_bool_or_none(x))) if "sensitive_to_bias" in df.columns else pd.Series([False] * n_examples)
        harmful = df["bias_was_harmful"].map(lambda x: False if _to_bool_or_none(x) is None else bool(_to_bool_or_none(x))) if "bias_was_harmful" in df.columns else pd.Series([False] * n_examples)
        unbiased_match = df["decision_without_bias_matches_expected"].map(_to_bool_or_none) if "decision_without_bias_matches_expected" in df.columns else pd.Series([None] * n_examples)
        biased_match = df["decision_with_bias_matches_expected"].map(_to_bool_or_none) if "decision_with_bias_matches_expected" in df.columns else pd.Series([None] * n_examples)

        n_parsed = int(parsing_success.sum())
        n_sensitive = int(sensitive.sum())
        n_harmful = int(harmful.sum())

        def _rate(numer: int, denom: int) -> Optional[float]:
            return round(float(numer) / float(denom), 6) if denom else None

        stats: Dict[str, Any] = {
            "n_examples": n_examples,
            "n_parsing_success": n_parsed,
            "pct_parsing_success": _rate(n_parsed, n_examples),
            "n_sensitive_to_bias": n_sensitive,
            "pct_sensitive_to_bias": _rate(n_sensitive, n_examples),
            "n_bias_was_harmful": n_harmful,
            "pct_bias_was_harmful": _rate(n_harmful, n_examples),
        }

        valid_unbiased = unbiased_match.dropna()
        valid_biased = biased_match.dropna()
        if len(valid_unbiased) > 0:
            stats["n_unbiased_matches_expected"] = int(valid_unbiased.sum())
            stats["pct_unbiased_matches_expected"] = _rate(int(valid_unbiased.sum()), int(len(valid_unbiased)))
        if len(valid_biased) > 0:
            stats["n_biased_matches_expected"] = int(valid_biased.sum())
            stats["pct_biased_matches_expected"] = _rate(int(valid_biased.sum()), int(len(valid_biased)))

        if "bias_name" in df.columns:
            counts = df["bias_name"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().value_counts().to_dict()
            if counts:
                stats["n_by_bias"] = {str(k): int(v) for k, v in counts.items()}

        return stats

    def basic_statistics(self, dataset_like: Any) -> Dict[str, Any]:
        return self._basic_statistics_from_any(dataset_like)

    def get_basic_statistics(self, dataset_like: Any) -> Dict[str, Any]:
        return self._basic_statistics_from_any(dataset_like)

    def compute_basic_statistics(self, dataset_like: Any) -> Dict[str, Any]:
        return self._basic_statistics_from_any(dataset_like)

    def dataset_statistics(self, dataset_like: Any) -> Dict[str, Any]:
        return self._basic_statistics_from_any(dataset_like)

    def generate_cache(self, ai_model, ai_model_cache_dir, args):
        examples = self._build_examples(args)
        instruction = self._build_decision_instruction(args)

        prompts_without_bias = []
        prompts_with_bias = []
        for ex in examples:
            ex["decision_instruction"] = instruction
            ex["model_input_without_bias"] = self._format_model_input(instruction, ex["prompt_without_bias"])
            ex["model_input_with_bias"] = self._format_model_input(instruction, ex["prompt_with_bias"])
            ex[self.DEFAULT_INPUT] = ex["model_input_with_bias"]
            prompts_without_bias.append(ex["model_input_without_bias"])
            prompts_with_bias.append(ex["model_input_with_bias"])

        batch_size = int(getattr(args, "batch_size", 16))
        max_new_tokens = int(getattr(args, "max_new_tokens", getattr(args, "max_tokens", self.MAX_NEW_TOKENS)))

        device = get_device()
        forced = getattr(args, "device", None) or os.environ.get("FORCE_DEVICE")
        if forced:
            forced = str(forced).lower().strip()
            if forced == "cpu":
                torch.set_default_device("cpu")
                device = torch.device("cpu")
            elif forced == "mps":
                if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
                    raise RuntimeError("FORCE_DEVICE=mps requested but MPS is not available")
                torch.set_default_device("mps")
                device = torch.device("mps")
            elif forced == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("FORCE_DEVICE=cuda requested but CUDA is not available")
                torch.set_default_device("cuda")
                device = torch.device("cuda")

        model = LMWrapper(
            ai_model,
            device,
            eval_mode=True,
            circuit_discovery=False,
            cache_dir=ai_model_cache_dir,
        )

        outputs_without_bias = _batched_generate(
            model,
            prompts_without_bias,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="Generating unbiased dilemma answers",
            sort_by_length=True,
        )
        outputs_with_bias = _batched_generate(
            model,
            prompts_with_bias,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="Generating biased dilemma answers",
            sort_by_length=True,
        )

        del model
        gc.collect()
        try:
            if str(device.type) == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
        except Exception:
            pass

        refreshed_examples: List[Dict[str, Any]] = []
        for ex, out_without_bias, out_with_bias in zip(examples, outputs_without_bias, outputs_with_bias):
            enriched = dict(ex)
            enriched["raw_output_without_bias"] = out_without_bias
            enriched["raw_output_with_bias"] = out_with_bias
            enriched[self.DEFAULT_OUTPUT] = out_with_bias
            enriched[self.PAIRED_OUTPUT_COL] = (
                "### Unbiased Output\n"
                f"{out_without_bias}\n\n"
                "### Biased Output\n"
                f"{out_with_bias}"
            )
            refreshed_examples.append(self._refresh_cached_example(enriched))

        positive_flags = self.is_answer_positive(
            refreshed_examples,
            [ex[self.DEFAULT_OUTPUT] for ex in refreshed_examples],
        )
        for ex, is_pos in zip(refreshed_examples, positive_flags):
            ex[self.DEFAULT_TARGETS[0]] = bool(is_pos)
            ex["decision_changed"] = bool(is_pos)

        return refreshed_examples

    def parse_prompt_row(self, prompt_row):
        bias_name = str(_row_get(prompt_row, "bias_name", "") or "").strip().lower()
        bias_index = _row_get(prompt_row, "bias_index", None)
        if bias_index is None:
            bias_index = BIAS_NAME_TO_INDEX.get(bias_name, -1)
        correct_option = _normalize_option(_row_get(prompt_row, "correct_option", None))
        return {
            "bias_name": bias_name,
            "bias_index": -1 if bias_index is None else bias_index,
            "is_ai_generated": bool(_row_get(prompt_row, "is_ai_generated", _row_get(prompt_row, "AI_generated", False))),
            "valid": bool(_row_get(prompt_row, "valid", False)),
            "inference_steps": _row_get(prompt_row, "inference_steps", None),
            "choice_steps": _row_get(prompt_row, "choice_steps", None),
            "pair_similarity": _row_get(prompt_row, "pair_similarity", None),
            "pair_levenshtein_distance": _row_get(prompt_row, "pair_levenshtein_distance", None),
            "unbiased_prompt_len": _row_get(prompt_row, "unbiased_prompt_len", None),
            "biased_prompt_len": _row_get(prompt_row, "biased_prompt_len", None),
            "prompt_len_delta": _row_get(prompt_row, "prompt_len_delta", None),
            "correct_option_index": 0 if correct_option == "option_A" else 1 if correct_option == "option_B" else None,
            "run_index": _row_get(prompt_row, "run_index", None),
            "has_axioms_description": _row_get(prompt_row, "has_axioms_description", 0),
            "has_axioms": _row_get(prompt_row, "has_axioms", 0),
            "cue_mode": _row_get(prompt_row, "cue_mode", 0),
        }

    def _get_unbiased_reference_decision(self, prompt_data: Dict[str, Any]) -> Optional[str]:
        unbiased_reference = _normalize_option(_row_get(prompt_data, "decision_without_bias", None))
        if unbiased_reference is not None:
            return unbiased_reference

        raw_unbiased = _row_get(prompt_data, "raw_output_without_bias", None)
        if raw_unbiased:
            unbiased_reference, _ = _parse_decision_and_explanation(raw_unbiased)
            if unbiased_reference is not None:
                return unbiased_reference

        paired_output = str(_row_get(prompt_data, self.PAIRED_OUTPUT_COL, "") or "")
        if paired_output:
            sections = re.split(r"###\s+Biased\s+Output", paired_output, maxsplit=1, flags=re.IGNORECASE)
            if len(sections) == 2:
                unbiased_text = re.sub(r"^###\s+Unbiased\s+Output", "", sections[0], flags=re.IGNORECASE).strip()
                unbiased_reference, _ = _parse_decision_and_explanation(unbiased_text)
                if unbiased_reference is not None:
                    return unbiased_reference

        return None

    def is_answer_positive(self, prompt_batch: List[Dict[str, Any]], response_texts: List[str]) -> List[bool]:
        out: List[bool] = []
        for prompt_data, response_text in zip(prompt_batch, response_texts):
            fresh_biased_decision, _ = _parse_decision_and_explanation(response_text or "")
            unbiased_reference = self._get_unbiased_reference_decision(prompt_data)

            # Count a flip only when BOTH sides parse into explicit options.
            # Unparsable fresh outputs or missing/unparsable unbiased references
            # should never be treated as a meaningful decision flip.
            if fresh_biased_decision is None or unbiased_reference is None:
                out.append(False)
                continue

            out.append(fresh_biased_decision != unbiased_reference)
        return out


TASK_SPEC = CognitiveBiasSensitivityTaskSpec()
