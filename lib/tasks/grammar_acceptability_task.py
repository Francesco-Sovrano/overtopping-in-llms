from __future__ import annotations

import json
import os
import re
import random
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

DEFAULT_SEED = int(os.environ.get("GRAMMAR_TASK_SEED", "42"))
DEFAULT_NUM_EXAMPLES = int(os.environ.get("GRAMMAR_NUM_EXAMPLES", "4096"))
DEFAULT_DATASET_PATH = Path(
    os.environ.get(
        "GRAMMAR_DATASET_PATH",
        str(Path("./data/grammar_acceptability/cola_in_domain_train.jsonl")),
    )
)

YES_WORDS = {"yes", "acceptable", "grammatical", "correct"}
NO_WORDS = {"no", "unacceptable", "ungrammatical", "incorrect"}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", str(text or "").lower())


def _contains_any(text: str, patterns: List[str]) -> bool:
    low = _normalize_text(text).lower()
    return any(p in low for p in patterns)


def _extract_binary_prediction(text: str) -> Optional[bool]:
    low = _normalize_text(text).lower()
    if not low:
        return None

    # Prioritize negative forms so "unacceptable" does not get mistaken for "acceptable".
    patterns = [
        (r"\bunacceptable\b", False),
        (r"\bungrammatical\b", False),
        (r"\bincorrect\b", False),
        (r"\bno\b", False),
        (r"\bacceptable\b", True),
        (r"\bgrammatical\b", True),
        (r"\bcorrect\b", True),
        (r"\byes\b", True),
    ]
    matches = []
    for pattern, label in patterns:
        for m in re.finditer(pattern, low):
            matches.append((m.start(), label))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _balanced_subsample(rows: List[Dict[str, Any]], n_examples: int, seed: int) -> List[Dict[str, Any]]:
    if n_examples >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    positives = [r for r in rows if bool(r.get("is_acceptable", False))]
    negatives = [r for r in rows if not bool(r.get("is_acceptable", False))]

    if n_examples % 2 == 1:
        raise ValueError(f"GRAMMAR_NUM_EXAMPLES must be even for balanced sampling, got {n_examples}")

    per_class = n_examples // 2
    if len(positives) < per_class or len(negatives) < per_class:
        raise ValueError(
            f"Requested {n_examples} balanced examples but dataset only has {len(positives)} positive and {len(negatives)} negative examples (maximum balanced size is {2 * min(len(positives), len(negatives))})."
        )

    def sample_source_diverse(pool: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in pool:
            buckets.setdefault(str(row.get("source", "unknown")), []).append(dict(row))
        for items in buckets.values():
            rng.shuffle(items)
        sources = list(buckets.keys())
        rng.shuffle(sources)
        out: List[Dict[str, Any]] = []
        while len(out) < k:
            progressed = False
            for source in sources:
                if buckets[source]:
                    out.append(buckets[source].pop())
                    progressed = True
                    if len(out) >= k:
                        break
            if not progressed:
                break
        if len(out) != k:
            raise ValueError(f"Could not sample {k} rows from pool of size {len(pool)}")
        return out

    sampled = sample_source_diverse(positives, per_class) + sample_source_diverse(negatives, per_class)
    rng.shuffle(sampled)
    for idx, row in enumerate(sampled):
        row["example_id"] = idx
    return sampled


def _sentence_metadata(sentence: str) -> Dict[str, Any]:
    text = _normalize_text(sentence)
    words = _tokenize_words(text)
    low = text.lower()

    wh_words = {"what", "who", "whom", "which", "when", "where", "why", "how"}
    reflexives = {
        "myself", "yourself", "himself", "herself", "itself",
        "ourselves", "yourselves", "themselves",
    }
    auxiliaries = {
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    }
    pronouns = {
        "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their",
    }

    return {
        "sentence": text,
        "word_count": int(len(words)),
        "char_count": int(len(text)),
        "has_question_mark": bool("?" in text),
        "starts_with_wh_word": bool(words and words[0] in wh_words),
        "has_negation": bool(any(w in {"not", "n't", "never", "no"} for w in words) or "n't" in low),
        "has_reflexive": bool(any(w in reflexives for w in words)),
        "has_auxiliary": bool(any(w in auxiliaries for w in words)),
        "has_pronoun": bool(any(w in pronouns for w in words)),
        "has_comma": bool("," in text),
        "has_coordination": bool(any(w in {"and", "or", "but"} for w in words)),
        "has_subordinator": bool(any(w in {"if", "that", "whether", "because", "while", "when", "before", "after"} for w in words)),
        "has_comparative": bool(any(w in {"more", "less", "than"} for w in words)),
        "has_passive_cue": bool(" by " in f" {low} " and any(w in auxiliaries for w in words)),
        "contains_digit": bool(any(ch.isdigit() for ch in text)),
        "source": None,
        "dataset": "CoLA",
        "split": None,
        "is_acceptable": None,
        "expected_label": None,
    }


SEED_FEATURES = [
    Feature("Short sentence (<=7 words)", "", """
def f_short_sentence(prompt, info):
    return int(info.get('word_count', 0)) <= 7
""", origin="predefined"),
    Feature("Long sentence (>=15 words)", "", """
def f_long_sentence(prompt, info):
    return int(info.get('word_count', 0)) >= 15
""", origin="predefined"),
    Feature("Contains negation", "", """
def f_has_negation(prompt, info):
    return bool(info.get('has_negation', False))
""", origin="predefined"),
    Feature("Question form", "", """
def f_question_form(prompt, info):
    return bool(info.get('has_question_mark', False)) or bool(info.get('starts_with_wh_word', False))
""", origin="predefined"),
    Feature("Contains reflexive", "", """
def f_has_reflexive(prompt, info):
    return bool(info.get('has_reflexive', False))
""", origin="predefined"),
    Feature("Contains auxiliary", "", """
def f_has_auxiliary(prompt, info):
    return bool(info.get('has_auxiliary', False))
""", origin="predefined"),
    Feature("Contains pronoun", "", """
def f_has_pronoun(prompt, info):
    return bool(info.get('has_pronoun', False))
""", origin="predefined"),
    Feature("Contains coordination", "", """
def f_has_coordination(prompt, info):
    return bool(info.get('has_coordination', False))
""", origin="predefined"),
    Feature("Contains subordinator", "", """
def f_has_subordinator(prompt, info):
    return bool(info.get('has_subordinator', False))
""", origin="predefined"),
    Feature("Comparative construction cue", "", """
def f_has_comparative(prompt, info):
    return bool(info.get('has_comparative', False))
""", origin="predefined"),
    Feature("Passive cue", "", """
def f_has_passive_cue(prompt, info):
    return bool(info.get('has_passive_cue', False))
""", origin="predefined"),
    Feature("Comma present", "", """
def f_has_comma(prompt, info):
    return bool(info.get('has_comma', False))
""", origin="predefined"),
    Feature("From source ks08", "", """
def f_source_ks08(prompt, info):
    return info.get('source') == 'ks08'
""", origin="predefined"),
    Feature("From source r-67", "", """
def f_source_r67(prompt, info):
    return info.get('source') == 'r-67'
""", origin="predefined"),
]


@dataclass
class GrammarAcceptabilityTaskSpec(FeatureTaskSpec):
    DEFAULT_TARGETS = ("is_correct",)
    DEFAULT_INPUT = "prompt"
    DEFAULT_OUTPUT = "raw_output"
    MAX_NEW_TOKENS = 4

    SYSTEM_PROMPT = (
        "You're an expert linguistics-oriented error analyst. "
        "Propose concise, testable features that might correlate with whether an LLM correctly judges "
        "the grammatical acceptability of short English sentences. Focus on properties visible in the sentence, "
        "such as length, question structure, agreement cues, negation, reflexives, auxiliaries, subordination, "
        "comparatives, punctuation, and source/style differences."
    )

    TOKENS_DICT_KEYS = (
        "word_count, char_count, has_question_mark, starts_with_wh_word, has_negation, has_reflexive, "
        "has_auxiliary, has_pronoun, has_comma, has_coordination, has_subordinator, has_comparative, "
        "has_passive_cue, source, dataset, split, is_acceptable"
    )

    SEED_FEATURES = SEED_FEATURES

    def _load_examples(self) -> List[Dict[str, Any]]:
        path = Path(DEFAULT_DATASET_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Grammar dataset not found at {path}. Set GRAMMAR_DATASET_PATH to a vendored JSONL file."
            )
        rows = _read_jsonl(path)
        n_examples = int(os.environ.get("GRAMMAR_NUM_EXAMPLES", str(DEFAULT_NUM_EXAMPLES)))
        seed = int(os.environ.get("GRAMMAR_TASK_SEED", str(DEFAULT_SEED)))
        return _balanced_subsample(rows, n_examples=n_examples, seed=seed)

    def parse_prompt_row(self, prompt_row) -> Dict[str, Any]:
        sentence = getattr(prompt_row, "sentence", None)
        if sentence is None:
            prompt = getattr(prompt_row, self.DEFAULT_INPUT, "")
            m = re.search(r"Sentence:\s*(.*?)\nAnswer:\s*$", str(prompt), flags=re.DOTALL)
            sentence = m.group(1).strip() if m else str(prompt)

        info = _sentence_metadata(str(sentence))
        for field in ("source", "dataset", "split", "is_acceptable", "expected_label"):
            if hasattr(prompt_row, field):
                info[field] = getattr(prompt_row, field)
        return info

    def is_answer_positive(self, prompt_batch: List[Dict], response_texts: List[str]) -> List[bool]:
        out: List[bool] = []
        for prompt_data, response in zip(prompt_batch, response_texts):
            pred = _extract_binary_prediction(response)
            gold = prompt_data.get("is_acceptable", None)
            if pd.isna(gold) or gold is None:
                raise ValueError("Missing is_acceptable in prompt row during scoring.")
            gold = bool(gold)
            out.append(pred is not None and pred == gold)
        return out

    def generate_cache(self, ai_model, ai_model_cache_dir, args):
        rows = self._load_examples()
        prompts = [row[self.DEFAULT_INPUT] for row in rows]
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

        outputs: List[str] = []
        dataloader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, shuffle=False)
        for batch_prompts in tqdm(dataloader, desc="Generating answers for grammar acceptability"):
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
        for row, out in zip(rows, outputs):
            item = dict(row)
            gold = item.get("is_acceptable", None)
            if pd.isna(gold) or gold is None:
                raise ValueError("Missing is_acceptable in source row during cache generation.")
            gold = bool(gold)

            item[self.DEFAULT_OUTPUT] = out
            item["predicted_label"] = _extract_binary_prediction(out)
            item["is_correct"] = bool(item["predicted_label"] == gold)

            meta = _sentence_metadata(item.get("sentence", ""))
            meta.pop("is_acceptable", None)
            meta.pop("expected_label", None)
            item.update(meta)

            item["source"] = row.get("source")
            item["dataset"] = row.get("dataset", "CoLA")
            item["split"] = row.get("split", "in_domain_train_sample")
            item["expected_label"] = row.get("expected_label")
            item["is_acceptable"] = row.get("is_acceptable")
            final.append(item)
        return final

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
        if "sentence" in df.columns:
            df["sentence"] = df["sentence"].astype(str).str.strip()
        for col in ["is_correct", "is_acceptable", "predicted_label"]:
            if col in df.columns:
                df[col] = df[col].astype("boolean")
        return df

    def get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"n_examples": int(len(df))}
        if df.empty:
            return stats
        if "is_acceptable" in df.columns:
            acc = df["is_acceptable"].dropna()
            stats["label_balance"] = {
                "n_acceptable": int(acc.sum()),
                "n_unacceptable": int((~acc).sum()),
                "pct_acceptable": float(acc.mean()),
            }
        if "is_correct" in df.columns:
            corr = df["is_correct"].dropna()
            stats["accuracy"] = float(corr.mean()) if len(corr) else None
        if "source" in df.columns and "is_correct" in df.columns:
            grp = df.dropna(subset=["source", "is_correct"]).groupby("source")["is_correct"].agg(["mean", "count"])
            stats["accuracy_by_source"] = {
                str(idx): {"accuracy": float(row["mean"]), "n": int(row["count"])}
                for idx, row in grp.iterrows()
            }
        if "word_count" in df.columns and "is_correct" in df.columns:
            short = df[df["word_count"] <= 7]["is_correct"].dropna()
            long = df[df["word_count"] >= 15]["is_correct"].dropna()
            stats["accuracy_short_sentences"] = float(short.mean()) if len(short) else None
            stats["accuracy_long_sentences"] = float(long.mean()) if len(long) else None
        return stats


TASK_SPEC = GrammarAcceptabilityTaskSpec()
