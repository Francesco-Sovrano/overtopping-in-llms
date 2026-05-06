from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from lib.feature_representation import Feature
from lib.task_spec import FeatureTaskSpec
from lib.caching_and_prompting import load_cache
from lib.modeling_and_ablation import LMWrapper, get_device

import gc
import torch
from tqdm import tqdm


def _batched_generate(
	model: LMWrapper,
	prompts: List[str],
	*,
	batch_size: int,
	max_new_tokens: int,
	desc: str,
	sort_by_length: bool = True,
) -> List[str]:
	"""Generate with LMWrapper in a way that is more stable for long prompts.

	- avoids DataLoader/multiprocessing artifacts (esp. on macOS)
	- buckets by prompt length to reduce padding waste
	- clamps batch size on MPS for long sequences to reduce OOM/segfault risk
	"""
	if not prompts:
		return []
	idxs = list(range(len(prompts)))
	if sort_by_length:
		idxs.sort(key=lambda i: len(prompts[i] or ""))

	device = getattr(getattr(model, "hooked_model", None), "cfg", None)
	device = getattr(device, "device", None)
	device_type = getattr(device, "type", str(device))

	outs: List[Optional[str]] = [None] * len(prompts)
	i = 0
	with tqdm(total=len(idxs), desc=desc) as pbar:
		while i < len(idxs):
			bs = max(1, int(batch_size))
			if str(device_type) == "mps":
				# Rough token estimate: ~4 chars/token. Total seq ~ prompt + max_new_tokens.
				max_chars = max(len(prompts[j] or "") for j in idxs[i : i + bs])
				tok_est = (max_chars / 4.0) + 32
				total_est = tok_est + float(max_new_tokens)
				if total_est >= 1500:
					bs = 2
				elif total_est >= 1100:
					bs = min(bs, 4)
				elif total_est >= 800:
					bs = min(bs, 8)

			batch_idxs = idxs[i : i + bs]
			batch_prompts = [prompts[j] for j in batch_idxs]
			gen = model.generate(
				batch_prompts,
				max_new_tokens=max_new_tokens,
				do_sample=False,
				fwd_hooks=None,
			)
			gen_list = [gen] if isinstance(gen, str) else list(gen)
			for j, txt in zip(batch_idxs, gen_list):
				outs[j] = txt
			i += len(batch_idxs)
			pbar.update(len(batch_idxs))

			if str(device_type) == "mps":
				try:
					# torch.mps.synchronize()
					torch.mps.empty_cache()
				except Exception:
					pass

	return [o if o is not None else "" for o in outs]


# -----------------------------------------------------------------------------
# Prompt + parsing helpers
# -----------------------------------------------------------------------------

CWE_LABELS: Dict[str, str] = {
	"CWE-79": "Improper Neutralization of Input During Web Page Generation: Cross-Site Scripting",
	"CWE-89": "Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')",
	"CWE-22": "Improper Limitation of a Pathname to a Restricted Directory (\"Path Traversal\")",
}


PROMPT_TEMPLATE = """Analyze the file content below and tell me if there's any line that may contain a bug of type {cwe_type} ({cwe_label}). Your output must adhere to the following structure.

Expected Output Structure:
SE: very Short Explanation of why the line may contain a bug of given type (e.g., The 'user_input' is directly concatenated into HTML content without sanitation).
BL: the Bugged Line, if any is found, else none (e.g., `response = \"<html><body><h1>Welcome, \" + user_input + \"!</h1></body></html>\"`).
BUG FOUND: YES if a bug is found, else NO.

Example output:
SE: The 'user_input' is directly concatenated into HTML content without sanitation.
BL: `response = \"<html><body><h1>Welcome, \" + user_input + \"!</h1></body></html>\"`
BUG FOUND: YES

File Content:
{file_content}
"""


def _clean_ws(text: str) -> str:
	if text is None:
		return ""
	text = text.replace("\\", "")
	text = text.replace("\n", " ")
	text = re.sub(r"\s+", " ", text)
	return text.strip().strip("`.")


def _parse_bug_found(output: str) -> bool:
	if not output:
		return False
	low = output.lower()
	m = re.search(r"bug\s*found\s*:\s*(yes|no)", low)
	if m:
		return m.group(1) == "yes"
	if "bug found" in low and "yes" in low:
		return True
	if re.search(r"\bbl\s*:\s*", low) and ("bl: none" not in low and "bl:none" not in low):
		return True
	return False


BUG_LINE_PATTERN = re.compile(
	r"(?:bugged)?\s*[- ]*(?:line|bl)\s*[:#]\s*(.*?)(?=\n\s*bug\s*found\s*:|$)",
	re.IGNORECASE | re.DOTALL,
)


def _extract_code_blocks_or_original(text: str) -> str:
	if not text:
		return ""
	pattern = r"```[a-zA-Z0-9_+-]*\n(.*?)```"
	matches = list(re.finditer(pattern, text, flags=re.DOTALL))
	if not matches:
		return text
	return "\n".join(m.group(1).strip() for m in matches if m.group(1).strip())


def _parse_predicted_bug_line(output: str) -> str:
	if not output:
		return ""
	m = BUG_LINE_PATTERN.search(output)
	if m:
		cand = _extract_code_blocks_or_original(m.group(1))
		cand = _clean_ws(cand)
		if cand.lower() in {"none", "no", "n/a"}:
			return ""
		return cand
	if _parse_bug_found(output):
		code = _extract_code_blocks_or_original(output)
		code = _clean_ws(code)
		if code and code != _clean_ws(output):
			return code
	return ""


def _get_line_at_char_position(content: str, pos: int) -> str:
	if content is None:
		return ""
	pos = int(pos)
	pos = max(0, min(pos, len(content)))
	start = content.rfind("\n", 0, pos)
	start = 0 if start < 0 else start + 1
	end = content.find("\n", pos)
	end = len(content) if end < 0 else end
	return content[start:end]


def _matches_ground_truth(pred_line: str, gt_line: str) -> bool:
	p = _clean_ws(pred_line).replace(" ", "")
	g = _clean_ws(gt_line).replace(" ", "")
	if not p or not g:
		return False
	return (p in g) or (g in p)


# -----------------------------------------------------------------------------
# Task Spec
# -----------------------------------------------------------------------------


@dataclass
class CodeInTheHaystackTaskSpec(FeatureTaskSpec):
	DEFAULT_TARGETS = ("is_correct",)
	DEFAULT_INPUT = "prompt"
	DEFAULT_OUTPUT = "raw_output"

	MAX_NEW_TOKENS = 120

	SYSTEM_PROMPT = (
		"You're an expert secure code reviewer. "
		"Inputs contain long files with a single injected vulnerability (the 'needle') and a specific CWE. "
		"Propose features that correlate with whether the LLM correctly locates the vulnerable line: "
		"file length, bug position (early vs late), padding/decoys, CWE type, and patterns in the content."
	)

	TOKENS_DICT_KEYS = (
		"cwe_id, input_len, bug_position, target_bug_position, target_length, additional_padding"
	)

	SEED_FEATURES = [
		Feature("Input Length", "Characters in the file.", """
def f_input_len(prompt, info):
	return info['input_len']
""", origin="predefined"),
		Feature("Bug Position", "bug_position", """
def f_bug_pos(prompt, info):
	return info['bug_position']
""", origin="predefined"),
		Feature("Target Bug Position", "target_bug_position", """
def f_bug_pos(prompt, info):
	return info['target_bug_position']
""", origin="predefined"),
		Feature("CWE", "CWE ID numeric part.", """
def f_cwe_id(prompt, info):
	return info['cwe_id']
""", origin="predefined"),
		Feature("Padding", "additional_padding", """
def f_cwe_id(prompt, info):
	return info['additional_padding']
""", origin="predefined"),
	]

	def _default_dataset_path(self) -> Path:
		# <repo_root>/data/code_in_the_haystack/files.csv
		repo_root = Path(__file__).resolve().parents[2]
		return repo_root / "data" / "code_in_the_haystack" / "files.csv"

	def _build_examples(self, csv_path: Path, *, max_input_chars: Optional[int]) -> List[Dict[str, Any]]:
		df = pd.read_csv(csv_path)
		examples: List[Dict[str, Any]] = []
		for row in df.itertuples(index=False):
			content = getattr(row, "content")
			if max_input_chars is not None:
				content = content[:max_input_chars]

			cwe = str(getattr(row, "CWE_ID"))
			label = CWE_LABELS.get(cwe, "")
			prompt = PROMPT_TEMPLATE.format(cwe_type=cwe, cwe_label=label, file_content=content)

			bug_pos = int(float(getattr(row, "bug_position")))
			input_len = int(float(getattr(row, "length"))) if hasattr(row, "length") else len(content)
			input_len = min(input_len, len(content)) if content else input_len
			m = re.search(r"(\d+)", cwe)
			cwe_num = int(m.group(1)) if m else None

			examples.append(
				{
					"file_id": getattr(row, "file_id"),
					"cwe": cwe,
					"cwe_id": cwe_num,
					"content": content,
					"bug_position": bug_pos,
					"target_bug_position": int(float(getattr(row, "target_bug_position"))) if hasattr(row, "target_bug_position") else None,
					"target_length": int(float(getattr(row, "target_length"))) if hasattr(row, "target_length") else None,
					"additional_padding": int(float(getattr(row, "additional_padding"))) if hasattr(row, "additional_padding") else None,
					"database_file_id": getattr(row, "database_file_id", None),
					"bug_line_gt": _get_line_at_char_position(content, bug_pos),
					"input_len": input_len,
					"bug_position_frac": (bug_pos / input_len) if input_len else None,
					"prompt": prompt,
				}
			)
		return examples

	def generate_cache(self, ai_model, ai_model_cache_dir, args):
		dataset_path = getattr(args, "dataset_path", os.environ.get("HAYSTACK_DATASET_PATH"))
		csv_path = Path(dataset_path) if dataset_path else self._default_dataset_path()
		if not csv_path.exists():
			raise FileNotFoundError(
				f"Haystack dataset CSV not found at {csv_path}. "
				"Set args.dataset_path or env HAYSTACK_DATASET_PATH to point to files.csv."
			)

		max_input_chars = getattr(args, "max_input_chars", None)
		max_input_chars = int(max_input_chars) if max_input_chars not in (None, "", False) else None
		examples = self._build_examples(csv_path, max_input_chars=max_input_chars)

		prompts = [ex[self.DEFAULT_INPUT] for ex in examples]
		batch_size = int(getattr(args, "batch_size", 16))
		max_new_tokens = int(getattr(args, "max_new_tokens", getattr(args, "max_tokens", self.MAX_NEW_TOKENS)))

		device = get_device()
		# Allow overriding device (useful on macOS if MPS is unstable for long sequences).
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

		outputs = _batched_generate(
			model,
			prompts,
			batch_size=batch_size,
			max_new_tokens=max_new_tokens,
			desc="Generating answers for code-in-the-haystack",
			sort_by_length=True,
		)

		del model
		gc.collect()
		# Optional extra relief on MPS.
		try:
			if str(device.type) == "mps":
				torch.mps.synchronize()
				torch.mps.empty_cache()
		except Exception:
			pass

		is_correct = self.is_answer_positive(examples, outputs)
		for ex, out, ok in zip(examples, outputs, is_correct):
			ex[self.DEFAULT_OUTPUT] = out
			ex[self.DEFAULT_TARGETS[0]] = bool(ok)
			ex["pred_bug_found"] = _parse_bug_found(out or "")
			ex["pred_bug_line"] = _parse_predicted_bug_line(out or "")
		return examples

	def load_dataset_from_cache(self, pkl_path: str) -> pd.DataFrame:
		obj = load_cache(pkl_path)
		if obj is None:
			return pd.DataFrame([])
		if isinstance(obj, list):
			df = pd.DataFrame(obj)
		elif isinstance(obj, dict):
			for k in ("data", "items", "rows", "examples"):
				if k in obj and isinstance(obj[k], list):
					df = pd.DataFrame(obj[k])
					break
			else:
				rows: List[Dict[str, Any]] = []
				for group, items in obj.items():
					if isinstance(items, list):
						for it in items:
							if isinstance(it, dict):
								rows.append({"group": group, **it})
							else:
								rows.append({"group": group, "_raw": it})
					else:
						rows.append({"group": group, "_raw": items})
			df = pd.DataFrame(rows)
		else:
			df = pd.DataFrame([{"_raw": obj}])

		if self.DEFAULT_INPUT not in df.columns and "prompt" in df.columns:
			df[self.DEFAULT_INPUT] = df["prompt"]
		if self.DEFAULT_OUTPUT not in df.columns and "model_output" in df.columns:
			df[self.DEFAULT_OUTPUT] = df["model_output"]
		for t in self.DEFAULT_TARGETS:
			if t in df.columns:
				df[t] = df[t].astype(bool)

		# Ensure required columns always exist (prevents downstream KeyError on empty caches).
		for col in [self.DEFAULT_INPUT, self.DEFAULT_OUTPUT, *list(self.DEFAULT_TARGETS)]:
			if col not in df.columns:
				df[col] = pd.Series(dtype=object)
		return df

	def parse_prompt_row(self, prompt_row) -> dict:
		input_len = getattr(prompt_row, "input_len", None)
		bug_pos = getattr(prompt_row, "bug_position", None)
		frac = getattr(prompt_row, "bug_position_frac", None)
		if frac is None and input_len and bug_pos is not None:
			try:
				frac = float(bug_pos) / float(input_len)
			except Exception:
				frac = None
		return {
			"cwe_id": getattr(prompt_row, "cwe_id", None),
			"input_len": input_len,
			"bug_position": bug_pos,
			"bug_position_frac": frac,
			"target_bug_position": getattr(prompt_row, "target_bug_position", None),
			"target_length": getattr(prompt_row, "target_length", None),
			"additional_padding": getattr(prompt_row, "additional_padding", None),
		}

	def is_answer_positive(self, prompt_batch: List[Dict], response_texts: List[str]) -> List[bool]:
		out: List[bool] = []
		for ex, resp in zip(prompt_batch, response_texts):
			resp = resp or ""
			pred_bug_found = _parse_bug_found(resp)
			pred_line = _parse_predicted_bug_line(resp)
			if not pred_bug_found or not pred_line:
				out.append(False)
				continue
			gt_line = ex.get("bug_line_gt") or ""
			out.append(_matches_ground_truth(pred_line, gt_line))
		return out


TASK_SPEC = CodeInTheHaystackTaskSpec()
