from __future__ import annotations

import csv
import os
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from lib.caching_and_prompting import load_cache
from lib.feature_representation import Feature
from lib.modeling_and_ablation import LMWrapper, get_device
from lib.task_spec import FeatureTaskSpec

HANS_TRAIN_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_train_set.txt"
HANS_VALIDATION_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"

DEFAULT_SEED = int(os.environ.get("HANS_TASK_SEED", "42"))
DEFAULT_SPLIT = os.environ.get("HANS_SPLIT", "validation").strip().lower()
DEFAULT_NUM_EXAMPLES = int(os.environ.get("HANS_NUM_EXAMPLES", "1024"))
DEFAULT_BALANCE_LABELS = os.environ.get("HANS_BALANCE_LABELS", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("HANS_MAX_NEW_TOKENS", "3"))
DEFAULT_CACHE_DIR = Path(os.environ.get("HANS_CACHE_DIR", "./cache/hans"))
DEFAULT_LOCAL_FILE = os.environ.get("HANS_LOCAL_FILE", "").strip()

_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
_NEGATION_WORDS = {"no", "not", "never", "none", "nobody", "nothing", "neither"}
_ARTICLES = {"a", "an", "the"}


def _normalize_split(split: str) -> str:
    split = (split or DEFAULT_SPLIT).strip().lower()
    if split in {"valid", "validation", "dev", "eval", "evaluation"}:
        return "validation"
    if split in {"train", "training"}:
        return "train"
    raise ValueError(f"Unsupported HANS split: {split}")


def _dataset_url_for_split(split: str) -> str:
    split = _normalize_split(split)
    return HANS_VALIDATION_URL if split == "validation" else HANS_TRAIN_URL


def _dataset_filename_for_split(split: str) -> str:
    split = _normalize_split(split)
    return "heuristics_evaluation_set.txt" if split == "validation" else "heuristics_train_set.txt"


def _ensure_hans_file(split: str, cache_dir: Path, local_file: str = "") -> Path:
    if local_file:
        p = Path(local_file)
        if not p.exists():
            raise FileNotFoundError(f"HANS_LOCAL_FILE does not exist: {p}")
        return p

    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / _dataset_filename_for_split(split)
    if file_path.exists() and file_path.stat().st_size > 0:
        return file_path

    url = _dataset_url_for_split(split)
    urllib.request.urlretrieve(url, file_path)
    return file_path


def _safe_get(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    return ""


def _word_tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _content_tokens(text: str) -> List[str]:
    return [w for w in _word_tokens(text) if w not in _ARTICLES]


def _lexical_overlap_ratio(premise: str, hypothesis: str) -> float:
    p = set(_content_tokens(premise))
    h = set(_content_tokens(hypothesis))
    if not h:
        return 0.0
    return len(p & h) / max(1, len(h))


def _is_subsequence(shorter: List[str], longer: List[str]) -> bool:
    if not shorter:
        return True
    it = iter(longer)
    return all(any(tok == cand for cand in it) for tok in shorter)


def _has_negation(text: str) -> bool:
    return any(tok in _NEGATION_WORDS for tok in _word_tokens(text))


def _label_to_short(label: str) -> str:
    low = (label or "").strip().lower()
    if low == "entailment":
        return "E"
    if low == "non-entailment":
        return "N"
    raise ValueError(f"Unexpected HANS label: {label}")


def _extract_predicted_label(text: str) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    low = s.lower().strip()

    if re.search(r"\bnon[- ]?entailment\b", low):
        return "N"
    if re.search(r"\bentailment\b", low):
        return "E"

    tokens = re.findall(r"\b[en]\b", low)
    if tokens:
        return tokens[-1].upper()

    compact = re.sub(r"\s+", "", low)
    if compact == "e":
        return "E"
    if compact == "n":
        return "N"
    return None


def _is_correct(expected_label: str, response_text: str) -> bool:
    return _extract_predicted_label(response_text) == expected_label


def _row_to_prompt(row: Dict[str, Any]) -> str:
    premise = row["premise"]
    hypothesis = row["hypothesis"]
    return (
        "Decide whether H must be true given P. Output only E or N.\n"
        f"P: {premise}\n"
        f"H: {hypothesis}\n"
        "A:"
    )


def _read_hans_rows(split: str, cache_dir: Path, local_file: str = "") -> List[Dict[str, Any]]:
    file_path = _ensure_hans_file(split, cache_dir, local_file=local_file)
    rows: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            gold_label = _safe_get(row, "gold_label", "label")
            if not gold_label or gold_label == "-":
                continue

            premise = _safe_get(row, "sentence1", "premise")
            hypothesis = _safe_get(row, "sentence2", "hypothesis")
            heuristic = _safe_get(row, "heuristic")
            subcase = _safe_get(row, "subcase")
            template = _safe_get(row, "template")
            pair_id = _safe_get(row, "pairID", "pair_id") or f"{_normalize_split(split)}_{idx}"

            premise_toks = _word_tokens(premise)
            hypothesis_toks = _word_tokens(hypothesis)
            premise_content = _content_tokens(premise)
            hypothesis_content = _content_tokens(hypothesis)
            overlap_ratio = _lexical_overlap_ratio(premise, hypothesis)

            item = {
                "example_id": idx,
                "pair_id": pair_id,
                "split": _normalize_split(split),
                "premise": premise,
                "hypothesis": hypothesis,
                "gold_label": gold_label,
                "expected_label": _label_to_short(gold_label),
                "heuristic": heuristic,
                "subcase": subcase,
                "template": template,
                "premise_num_words": len(premise_toks),
                "hypothesis_num_words": len(hypothesis_toks),
                "premise_num_chars": len(premise),
                "hypothesis_num_chars": len(hypothesis),
                "premise_content_words": len(premise_content),
                "hypothesis_content_words": len(hypothesis_content),
                "lexical_overlap_ratio": overlap_ratio,
                "full_lexical_overlap": overlap_ratio >= 0.999,
                "hypothesis_is_subsequence": _is_subsequence(hypothesis_toks, premise_toks),
                "premise_has_negation": _has_negation(premise),
                "hypothesis_has_negation": _has_negation(hypothesis),
            }
            item["prompt"] = _row_to_prompt(item)
            rows.append(item)
    return rows


def _balanced_sample(rows: List[Dict[str, Any]], n_examples: int, seed: int, balance_labels: bool) -> List[Dict[str, Any]]:
    if n_examples <= 0 or n_examples >= len(rows):
        sampled = list(rows)
    else:
        rng = random.Random(seed)
        if not balance_labels:
            sampled = rng.sample(rows, n_examples)
        else:
            by_label: Dict[str, List[Dict[str, Any]]] = {"E": [], "N": []}
            for row in rows:
                by_label[row["expected_label"]].append(row)
            target_e = n_examples // 2
            target_n = n_examples - target_e
            if len(by_label["E"]) < target_e or len(by_label["N"]) < target_n:
                raise ValueError(
                    f"Cannot draw balanced sample of size {n_examples}: "
                    f"only {len(by_label['E'])} entailment and {len(by_label['N'])} non-entailment rows available"
                )
            sampled = rng.sample(by_label["E"], target_e) + rng.sample(by_label["N"], target_n)
            rng.shuffle(sampled)

    for i, row in enumerate(sampled):
        row["dataset_index"] = i
    return sampled


SEED_FEATURES = [
    Feature("Lexical overlap ratio", "Fraction of hypothesis content words present in the premise.", """
def f_lexical_overlap(prompt, info):
    return float(info.get('lexical_overlap_ratio', 0.0))
""", origin="predefined"),
    Feature("Full lexical overlap", "1 if all hypothesis content words appear in the premise.", """
def f_full_lexical_overlap(prompt, info):
    return 1 if bool(info.get('full_lexical_overlap', False)) else 0
""", origin="predefined"),
    Feature("Hypothesis is subsequence", "1 if the hypothesis token sequence is a subsequence of the premise.", """
def f_hypothesis_subsequence(prompt, info):
    return 1 if bool(info.get('hypothesis_is_subsequence', False)) else 0
""", origin="predefined"),
    Feature("Premise length", "Number of word tokens in the premise.", """
def f_premise_len(prompt, info):
    return int(info.get('premise_num_words', 0))
""", origin="predefined"),
    Feature("Hypothesis length", "Number of word tokens in the hypothesis.", """
def f_hypothesis_len(prompt, info):
    return int(info.get('hypothesis_num_words', 0))
""", origin="predefined"),
    Feature("Hypothesis shorter than premise", "1 if the hypothesis has fewer words than the premise.", """
def f_hypothesis_shorter(prompt, info):
    return 1 if int(info.get('hypothesis_num_words', 0)) < int(info.get('premise_num_words', 0)) else 0
""", origin="predefined"),
    Feature("Premise has negation", "1 if the premise contains a negation word.", """
def f_premise_negation(prompt, info):
    return 1 if bool(info.get('premise_has_negation', False)) else 0
""", origin="predefined"),
    Feature("Hypothesis has negation", "1 if the hypothesis contains a negation word.", """
def f_hypothesis_negation(prompt, info):
    return 1 if bool(info.get('hypothesis_has_negation', False)) else 0
""", origin="predefined"),
    Feature("Lexical overlap heuristic", "1 if HANS marks the example as lexical_overlap.", """
def f_heuristic_lexical_overlap(prompt, info):
    return 1 if str(info.get('heuristic', '')) == 'lexical_overlap' else 0
""", origin="predefined"),
    Feature("Subsequence heuristic", "1 if HANS marks the example as subsequence.", """
def f_heuristic_subsequence(prompt, info):
    return 1 if str(info.get('heuristic', '')) == 'subsequence' else 0
""", origin="predefined"),
    Feature("Constituent heuristic", "1 if HANS marks the example as constituent.", """
def f_heuristic_constituent(prompt, info):
    return 1 if str(info.get('heuristic', '')) == 'constituent' else 0
""", origin="predefined"),
]


@dataclass
class HansNLITaskSpec(FeatureTaskSpec):
    DEFAULT_TARGETS = ("is_correct",)
    DEFAULT_INPUT = "prompt"
    DEFAULT_OUTPUT = "raw_output"
    MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS

    SYSTEM_PROMPT = (
        "You're an expert at analyzing short natural-language inference failures. "
        "Given very short HANS premise-hypothesis prompts with binary outputs E/N, "
        "propose concise, testable features that correlate with whether the LLM predicts the correct NLI label. "
        "Focus on lexical overlap, subsequence cues, constituent structure proxies, length asymmetries, and negation."
    )

    TOKENS_DICT_KEYS = (
        "premise_num_words, hypothesis_num_words, premise_num_chars, hypothesis_num_chars, "
        "premise_content_words, hypothesis_content_words, lexical_overlap_ratio, full_lexical_overlap, "
        "hypothesis_is_subsequence, premise_has_negation, hypothesis_has_negation, heuristic, subcase, template"
    )

    SEED_FEATURES = SEED_FEATURES

    def __init__(self):
        super().__init__()

    def is_answer_positive(self, prompt_batch: List[Dict], response_texts: List[str]) -> List[bool]:
        return [
            _is_correct(prompt_data["expected_label"], answer)
            for prompt_data, answer in zip(prompt_batch, response_texts)
        ]

    def parse_prompt_row(self, prompt_row) -> Dict[str, Any]:
        return {
            "premise_num_words": int(getattr(prompt_row, "premise_num_words")),
            "hypothesis_num_words": int(getattr(prompt_row, "hypothesis_num_words")),
            "premise_num_chars": int(getattr(prompt_row, "premise_num_chars")),
            "hypothesis_num_chars": int(getattr(prompt_row, "hypothesis_num_chars")),
            "premise_content_words": int(getattr(prompt_row, "premise_content_words")),
            "hypothesis_content_words": int(getattr(prompt_row, "hypothesis_content_words")),
            "lexical_overlap_ratio": float(getattr(prompt_row, "lexical_overlap_ratio")),
            "full_lexical_overlap": bool(getattr(prompt_row, "full_lexical_overlap")),
            "hypothesis_is_subsequence": bool(getattr(prompt_row, "hypothesis_is_subsequence")),
            "premise_has_negation": bool(getattr(prompt_row, "premise_has_negation")),
            "hypothesis_has_negation": bool(getattr(prompt_row, "hypothesis_has_negation")),
            "heuristic": str(getattr(prompt_row, "heuristic")),
            "subcase": str(getattr(prompt_row, "subcase")),
            "template": str(getattr(prompt_row, "template")),
        }

    def generate_cache(self, ai_model, ai_model_cache_dir, args):
        split = _normalize_split(os.environ.get("HANS_SPLIT", DEFAULT_SPLIT))
        n_examples = int(os.environ.get("HANS_NUM_EXAMPLES", str(DEFAULT_NUM_EXAMPLES)))
        seed = int(os.environ.get("HANS_TASK_SEED", str(DEFAULT_SEED)))
        balance_labels = os.environ.get("HANS_BALANCE_LABELS", "1").strip().lower() not in {"0", "false", "no"}
        cache_dir = Path(os.environ.get("HANS_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
        local_file = os.environ.get("HANS_LOCAL_FILE", DEFAULT_LOCAL_FILE)
        batch_size = getattr(args, "batch_size", 16)
        max_new_tokens = getattr(args, "max_new_tokens", self.MAX_NEW_TOKENS)

        examples = _read_hans_rows(split, cache_dir, local_file=local_file)
        examples = _balanced_sample(examples, n_examples, seed, balance_labels)
        prompts = [ex[self.DEFAULT_INPUT] for ex in examples]

        device = get_device()
        model = LMWrapper(
            ai_model,
            device,
            eval_mode=True,
            circuit_discovery=False,
            cache_dir=ai_model_cache_dir,
        )

        outputs: List[str] = []
        dataloader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, shuffle=False)
        for batch_prompts in tqdm(dataloader, desc=f"Generating answers for HANS ({split})"):
            batch_outputs = model.generate(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                fwd_hooks=None,
            )
            if isinstance(batch_outputs, str):
                outputs.append(batch_outputs)
            else:
                outputs += list(batch_outputs)

        final = []
        for ex, out in zip(examples, outputs):
            item = dict(ex)
            item[self.DEFAULT_OUTPUT] = out
            item["predicted_label"] = _extract_predicted_label(out)
            item["is_correct"] = _is_correct(ex["expected_label"], out)
            final.append(item)
        return final

    def load_dataset_from_cache(self, pkl_path: str) -> pd.DataFrame:
        obj = load_cache(pkl_path)
        rows = []
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    rows.append(dict(item))
        elif isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, list):
                    rows.extend(v for v in value if isinstance(v, dict))

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df[self.DEFAULT_INPUT] = df[self.DEFAULT_INPUT].astype(str).str.strip()
        if "is_correct" in df.columns:
            df["is_correct"] = df["is_correct"].astype(bool)
        return df

    def get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"n_examples": int(len(df))}
        if df.empty:
            return stats
        if "is_correct" in df.columns:
            stats["accuracy"] = float(df["is_correct"].mean())
        if "expected_label" in df.columns:
            counts = df["expected_label"].value_counts(dropna=False).to_dict()
            stats["label_counts"] = {str(k): int(v) for k, v in counts.items()}
        for col in ("heuristic", "subcase"):
            if col in df.columns and "is_correct" in df.columns:
                grp = df.groupby(col, dropna=False)["is_correct"].agg(["mean", "count"]).reset_index()
                stats[f"accuracy_by_{col}"] = {
                    str(r[col]): {"accuracy": float(r["mean"]), "n": int(r["count"])}
                    for _, r in grp.iterrows()
                }
        return stats


TASK_SPEC = HansNLITaskSpec()
