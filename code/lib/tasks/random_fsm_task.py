from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from lib.caching_and_prompting import load_cache
from lib.feature_representation import Feature
from lib.modeling_and_ablation import LMWrapper, get_device
from lib.task_spec import FeatureTaskSpec

STATE_ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_SEED = int(os.environ.get("FSM_TASK_SEED", "42"))
DEFAULT_NUM_EXAMPLES = int(os.environ.get("FSM_NUM_EXAMPLES", "2048"))
DEFAULT_MIN_STATES = int(os.environ.get("FSM_MIN_STATES", "3"))
DEFAULT_MAX_STATES = int(os.environ.get("FSM_MAX_STATES", "6"))
DEFAULT_MIN_INPUT_LEN = int(os.environ.get("FSM_MIN_INPUT_LEN", "8"))
DEFAULT_MAX_INPUT_LEN = int(os.environ.get("FSM_MAX_INPUT_LEN", "24"))


def _state_names(n_states: int) -> List[str]:
    if n_states > len(STATE_ALPHABET):
        raise ValueError(f"n_states={n_states} exceeds supported alphabet size {len(STATE_ALPHABET)}")
    return STATE_ALPHABET[:n_states]


def _sample_machine(rng: random.Random, n_states: int) -> Dict[str, Dict[str, str]]:
    names = _state_names(n_states)
    return {
        s: {
            "0": rng.choice(names),
            "1": rng.choice(names),
        }
        for s in names
    }


def _simulate(machine: Dict[str, Dict[str, str]], start_state: str, bitstring: str) -> str:
    state = start_state
    for bit in bitstring:
        state = machine[state][bit]
    return state


def _longest_run(bitstring: str, bit: str) -> int:
    best = cur = 0
    for ch in bitstring:
        if ch == bit:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _machine_stats(machine: Dict[str, Dict[str, str]], start_state: str, bitstring: str, answer_state: str) -> Dict[str, Any]:
    states = list(machine.keys())
    zero_edges = [machine[s]["0"] for s in states]
    one_edges = [machine[s]["1"] for s in states]
    self_loops_zero = sum(1 for s in states if machine[s]["0"] == s)
    self_loops_one = sum(1 for s in states if machine[s]["1"] == s)
    transitions_to_start = sum(1 for s in states for b in ("0", "1") if machine[s][b] == start_state)
    transitions_to_answer = sum(1 for s in states for b in ("0", "1") if machine[s][b] == answer_state)
    unique_zero_targets = len(set(zero_edges))
    unique_one_targets = len(set(one_edges))

    return {
        "num_states": len(states),
        "input_length": len(bitstring),
        "n_zeros": bitstring.count("0"),
        "n_ones": bitstring.count("1"),
        "zero_fraction": bitstring.count("0") / max(1, len(bitstring)),
        "start_state": start_state,
        "answer_state": answer_state,
        "start_state_idx": states.index(start_state),
        "answer_state_idx": states.index(answer_state),
        "longest_zero_run": _longest_run(bitstring, "0"),
        "longest_one_run": _longest_run(bitstring, "1"),
        "self_loops_zero": self_loops_zero,
        "self_loops_one": self_loops_one,
        "transitions_to_start": transitions_to_start,
        "transitions_to_answer": transitions_to_answer,
        "unique_zero_targets": unique_zero_targets,
        "unique_one_targets": unique_one_targets,
        "machine_text": "; ".join(
            f"{s}: 0->{machine[s]['0']}, 1->{machine[s]['1']}" for s in states
        ),
    }


def _make_example(rng: random.Random, idx: int, *, min_states: int, max_states: int, min_input_len: int, max_input_len: int) -> Dict[str, Any]:
    n_states = rng.randint(min_states, max_states)
    states = _state_names(n_states)
    machine = _sample_machine(rng, n_states)
    start_state = rng.choice(states)
    bitstring = "".join(rng.choice("01") for _ in range(rng.randint(min_input_len, max_input_len)))
    answer_state = _simulate(machine, start_state, bitstring)
    info = _machine_stats(machine, start_state, bitstring, answer_state)

    prompt = (
        "You are given a freshly sampled deterministic finite-state machine over the binary alphabet {0,1}. "
        "Return only the final state label after processing the full input bitstring.\n\n"
        f"States: {', '.join(states)}\n"
        f"Start state: {start_state}\n"
        "Transitions:\n"
        + "\n".join(f"- {s}: on 0 -> {machine[s]['0']}; on 1 -> {machine[s]['1']}" for s in states)
        + "\n"
        f"Input bitstring: {bitstring}\n"
        "Final state:" 
    )

    return {
        "example_id": idx,
        "prompt": prompt,
        "raw_machine": machine,
        "bitstring": bitstring,
        "expected_state": answer_state,
        **info,
    }


def _extract_state_label(text: str) -> str | None:
    if not text:
        return None
    matches = re.findall(r"\b([A-Z])\b", str(text).upper())
    return matches[-1] if matches else None


def _is_correct(expected_state: str, response_text: str) -> bool:
    pred = _extract_state_label(response_text)
    return pred == expected_state


SEED_FEATURES = [
    Feature("Three-state machine", "", """
def f_three_states(prompt, info):
    return info.get('num_states') == 3
""", origin="predefined"),
    Feature("Large machine (>=5 states)", "", """
def f_large_machine(prompt, info):
    return int(info.get('num_states', 0)) >= 5
""", origin="predefined"),
    Feature("Short input (<=12 bits)", "", """
def f_short_input(prompt, info):
    return int(info.get('input_length', 0)) <= 12
""", origin="predefined"),
    Feature("Long input (>=18 bits)", "", """
def f_long_input(prompt, info):
    return int(info.get('input_length', 0)) >= 18
""", origin="predefined"),
    Feature("More zeros than ones", "", """
def f_more_zeros(prompt, info):
    return int(info.get('n_zeros', 0)) > int(info.get('n_ones', 0))
""", origin="predefined"),
    Feature("More ones than zeros", "", """
def f_more_ones(prompt, info):
    return int(info.get('n_ones', 0)) > int(info.get('n_zeros', 0))
""", origin="predefined"),
    Feature("Long run of zeros", "", """
def f_long_zero_run(prompt, info):
    return int(info.get('longest_zero_run', 0)) >= 4
""", origin="predefined"),
    Feature("Long run of ones", "", """
def f_long_one_run(prompt, info):
    return int(info.get('longest_one_run', 0)) >= 4
""", origin="predefined"),
    Feature("Many self-loops", "", """
def f_many_self_loops(prompt, info):
    return int(info.get('self_loops_zero', 0)) + int(info.get('self_loops_one', 0)) >= 3
""", origin="predefined"),
    Feature("Few unique zero-targets", "", """
def f_few_zero_targets(prompt, info):
    return int(info.get('unique_zero_targets', 0)) <= 2
""", origin="predefined"),
    Feature("Few unique one-targets", "", """
def f_few_one_targets(prompt, info):
    return int(info.get('unique_one_targets', 0)) <= 2
""", origin="predefined"),
    Feature("Start state early in alphabet", "", """
def f_start_state_early(prompt, info):
    return int(info.get('start_state_idx', 0)) <= 1
""", origin="predefined"),
    Feature("Answer state late in alphabet", "", """
def f_answer_state_late(prompt, info):
    return int(info.get('answer_state_idx', 0)) >= 3
""", origin="predefined"),
    Feature("Transitions often return to start", "", """
def f_transitions_to_start(prompt, info):
    return int(info.get('transitions_to_start', 0)) >= 3
""", origin="predefined"),
    Feature("Transitions often land on answer state", "", """
def f_transitions_to_answer(prompt, info):
    return int(info.get('transitions_to_answer', 0)) >= 3
""", origin="predefined"),
]


@dataclass
class RandomFSMTaskSpec(FeatureTaskSpec):
    DEFAULT_TARGETS = ("is_correct",)
    DEFAULT_INPUT = "prompt"
    DEFAULT_OUTPUT = "raw_output"
    MAX_NEW_TOKENS = 4

    SYSTEM_PROMPT = (
        "You're an expert at analyzing algorithmic reasoning failures. "
        "Propose concise, testable features that might correlate with whether an LLM correctly simulates "
        "a freshly sampled binary finite-state machine. Focus on properties visible in the prompt such as "
        "machine size, input length, run structure, self-loops, and transition concentration."
    )

    TOKENS_DICT_KEYS = (
        "num_states, input_length, n_zeros, n_ones, zero_fraction, start_state_idx, answer_state_idx, "
        "longest_zero_run, longest_one_run, self_loops_zero, self_loops_one, transitions_to_start, "
        "transitions_to_answer, unique_zero_targets, unique_one_targets"
    )

    SEED_FEATURES = SEED_FEATURES

    def __init__(self):
        super().__init__()

    def is_answer_positive(self, prompt_batch: List[Dict], response_texts: List[str]) -> List[bool]:
        return [
            _is_correct(prompt_data["expected_state"], answer)
            for prompt_data, answer in zip(prompt_batch, response_texts)
        ]

    def parse_prompt_row(self, prompt_row) -> Dict[str, Any]:
        return {
            "num_states": int(getattr(prompt_row, "num_states")),
            "input_length": int(getattr(prompt_row, "input_length")),
            "n_zeros": int(getattr(prompt_row, "n_zeros")),
            "n_ones": int(getattr(prompt_row, "n_ones")),
            "zero_fraction": float(getattr(prompt_row, "zero_fraction")),
            "start_state_idx": int(getattr(prompt_row, "start_state_idx")),
            "answer_state_idx": int(getattr(prompt_row, "answer_state_idx")),
            "longest_zero_run": int(getattr(prompt_row, "longest_zero_run")),
            "longest_one_run": int(getattr(prompt_row, "longest_one_run")),
            "self_loops_zero": int(getattr(prompt_row, "self_loops_zero")),
            "self_loops_one": int(getattr(prompt_row, "self_loops_one")),
            "transitions_to_start": int(getattr(prompt_row, "transitions_to_start")),
            "transitions_to_answer": int(getattr(prompt_row, "transitions_to_answer")),
            "unique_zero_targets": int(getattr(prompt_row, "unique_zero_targets")),
            "unique_one_targets": int(getattr(prompt_row, "unique_one_targets")),
        }

    def generate_cache(self, ai_model, ai_model_cache_dir, args):
        n_examples = int(os.environ.get("FSM_NUM_EXAMPLES", str(DEFAULT_NUM_EXAMPLES)))
        min_states = int(os.environ.get("FSM_MIN_STATES", str(DEFAULT_MIN_STATES)))
        max_states = int(os.environ.get("FSM_MAX_STATES", str(DEFAULT_MAX_STATES)))
        min_input_len = int(os.environ.get("FSM_MIN_INPUT_LEN", str(DEFAULT_MIN_INPUT_LEN)))
        max_input_len = int(os.environ.get("FSM_MAX_INPUT_LEN", str(DEFAULT_MAX_INPUT_LEN)))
        seed = int(os.environ.get("FSM_TASK_SEED", str(DEFAULT_SEED)))
        batch_size = getattr(args, "batch_size", 16)
        max_new_tokens = getattr(args, "max_new_tokens", self.MAX_NEW_TOKENS)

        rng = random.Random(seed)
        examples = [
            _make_example(
                rng,
                idx=i,
                min_states=min_states,
                max_states=max_states,
                min_input_len=min_input_len,
                max_input_len=max_input_len,
            )
            for i in range(n_examples)
        ]
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
        for batch_prompts in tqdm(dataloader, desc="Generating answers for random-fsm"):
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
            item["predicted_state"] = _extract_state_label(out)
            item["is_correct"] = _is_correct(ex["expected_state"], out)
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
        for col in ("num_states", "input_length"):
            if col in df.columns and "is_correct" in df.columns:
                grp = df.groupby(col, dropna=False)["is_correct"].agg(["mean", "count"]).reset_index()
                stats[f"accuracy_by_{col}"] = {
                    str(int(r[col])): {"accuracy": float(r["mean"]), "n": int(r["count"])}
                    for _, r in grp.iterrows()
                }
        return stats


TASK_SPEC = RandomFSMTaskSpec()
