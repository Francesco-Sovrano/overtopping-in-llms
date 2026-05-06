import ast
import re
import json
from dataclasses import dataclass
import numpy as np
import math
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import *

from lib.caching_and_prompting import instruct_model

# ---------------------------
# Feature representation & sandbox
# ---------------------------
@dataclass
class Feature:
	label: str
	description: str
	python_src: str
	fn_name: str = ""
	origin: str = "llm"  # default; seed feats can mark "predefined"


SAFE_BUILTINS = {
	"abs": abs, "min": min, "max": max, "sum": sum, "len": len, "all": all, "any": any,
	"round": round, "sorted": sorted, "set": set, "list": list, "tuple": tuple, "dict": dict,
	"int": int, "float": float, "str": str, "range": range, "enumerate": enumerate, "zip": zip
}

FORBIDDEN_NODE_TYPES = (
	ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, 
	# ast.Try, ast.Raise,
	ast.Lambda, ast.ClassDef
)

def _validate_feature_ast(src):
	tree = ast.parse(src, mode="exec")
	for node in ast.walk(tree):
		if isinstance(node, FORBIDDEN_NODE_TYPES):
			raise ValueError(f"Forbidden Python construct: {type(node).__name__}")
		if isinstance(node, ast.Call):
			if isinstance(node.func, ast.Name) and node.func.id in {"eval","exec","open","compile","__import__"}:
				raise ValueError("Forbidden call detected")
			# if isinstance(node.func, ast.Attribute):
			# 	if not isinstance(node.func.value, ast.Name) or node.func.value.id not in {"math","re"}:
			# 		raise ValueError("Forbidden attribute call (only math.* and re.* allowed)")
	fn_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
	if not fn_nodes:
		raise ValueError("No function defined in python_src")
	ok_sig = False
	for fn in fn_nodes:
		argnames = [a.arg for a in fn.args.args]
		if len(argnames) >= 2 and argnames[0] == "prompt" and argnames[1] == "info":
			ok_sig = True; break
	if not ok_sig:
		raise ValueError("Feature function must have signature (prompt, info, ...)")
	return tree

def compile_feature_function(python_src):
	tree = _validate_feature_ast(python_src)
	safe_globals = {"__builtins__": SAFE_BUILTINS, "math": math, "re": re}
	local_env = {}
	code = compile(tree, "<feature>", "exec")
	exec(code, safe_globals, local_env)
	for name, obj in local_env.items():
		if callable(obj):
			return name, obj
	raise ValueError("No callable function found after compilation.")

# def clamp01(x):
# 	if x is None: return None
# 	try:
# 		if isinstance(x, bool): return 1.0 if x else 0.0
# 		v = float(x); 
# 		return 0.0 if v < 0 else (1.0 if v > 1 else v)
# 	except Exception:
# 		return None

# ---------------------------
# JSON helpers
# ---------------------------
def fix_and_parse_json(text):
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		repaired = re.sub(r'\\+"', '"', text)
		repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
		return json.loads(repaired)

def extract_json_block(text):
	if not text:
		return None
	m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
	if not m:
		return None
	try:
		return fix_and_parse_json(m.group(1))
	except Exception:
		return []

# ---------------------------
# Agent: propose Python features using correctness contrast
# ---------------------------
def propose_features_contrastive(ai_model, correct_prompts, incorrect_prompts, system_prompt, tokens_dict_keys, existing_features=None, correct_label='CORRECTLY ANSWERED', incorrect_label='INCORRECTLY ANSWERED', n_features=16, temperature=0.2, cache_path='cache'):
	def format_block(name, items):
		preview = "\n".join(f"- {s}" for s in items)
		return f"### {name}\n{preview}\n"

	context_parts = [
		"You will see two groups of INPUT prompts only.",
		format_block(f"{correct_label} inputs", correct_prompts),
		format_block(f"{incorrect_label} inputs", incorrect_prompts),
	]

	# existing_features is used to push the model away from duplicates
	existing_features = existing_features or []
	if existing_features:
		lines = []
		for f in existing_features:
			label = (f.label or "").strip()
			desc = (f.description or "").strip()
			if not label:
				continue
			if desc:
				lines.append(f"- {label}: {desc}")
			else:
				lines.append(f"- {label}")
		if lines:
			context_parts.append("### ALREADY DEFINED FEATURES (DO NOT REPEAT)\n")
			context_parts.extend(lines)

	context = "\n".join(context_parts)

	user_prompt = f"""
{context}

Now propose {n_features} discriminative features as Python functions.

STRICT RULES:
- Each item must be JSON with keys: label, description, python_src.
- Each python_src MUST define ONE function with signature:
	def my_feature(prompt: str, info: dict):
- 'info' is a dictionary with the following keys: {tokens_dict_keys}.
- 'my_feature' returns a numerical value (or None if not applicable).
- You may ONLY use 'prompt' and 'info'.
- No imports, no eval/exec/open/network; you MAY call math.* and re.* (already available).
- Keep each function < 25 lines, deterministic.
- Focus on features that *likely* separate the two sets.

Return ONLY a JSON list of items.
"""
	raw = instruct_model(
		[user_prompt],
		system_instructions=system_prompt,
		model=ai_model,
		temperature=temperature,
		cache_path=cache_path,
	)[0]
	assert raw
	# print('Feature-proposing agent says:', raw)
	data = extract_json_block(raw) or []
	out = []
	for item in data:
		try:
			src = (item.get("python_src") or "").strip()
			assert "def " in src and "(prompt: str, info: dict)" in src, "Bad signature"
			out.append(Feature(
				label=(item.get("label") or "").strip(),
				description=(item.get("description") or "").strip(),
				python_src=src,
				origin="llm"
			))
			print("Proposed by LLM:", item.get("label"))
		except Exception as e:
			print("Skipping invalid LLM feature:", e, item)
			continue

	return [f for f in out if f.label or f.description]

def _default_embedder():
	try:
		from sentence_transformers import SentenceTransformer
		return SentenceTransformer("all-MiniLM-L6-v2")  # fast, small, good enough for dedupe
	except Exception as e:
		raise RuntimeError(
			"sentence-transformers is required for semantic deduping. "
			"Install with: pip install sentence-transformers"
		) from e

def remove_similar_labels(tuple_list, threshold=0.9, key=None, get_embedding_fn=None, get_similarity_fn=None):
	if key is None:
		key = lambda x: x[0] if isinstance(x, (list,tuple)) else x
	if get_embedding_fn is None:
		get_embedding_fn = _default_embedder()
	if get_similarity_fn is None:
		get_similarity_fn = cosine_similarity
	value_list = tuple(map(key,tuple_list))
	embedding_list = get_embedding_fn.encode(value_list)
	similarity_vec = get_similarity_fn(embedding_list, embedding_list)
	
	result_list = []
	for i,v in enumerate(tuple_list):
		if not np.any(similarity_vec[i][:i] >= threshold):
			result_list.append(v)
		else: # ignore this element in next comparisons
			similarity_vec[:,i] = 0
	return result_list

def drop_near_duplicates_by_corr(X_df, thresh=0.9999):
	corr = X_df.corr().abs()
	upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
	to_drop = [col for col in upper.columns if any(upper[col] >= thresh)]
	return to_drop

# ---------------------------
# Reporting helpers
# ---------------------------
def save_markdown(text, out_md):
	Path(out_md).parent.mkdir(parents=True, exist_ok=True)
	with open(out_md, "w", encoding="utf-8") as f:
		f.write(text)
	print("Markdown saved:", out_md)

def render_feature_catalog(features):
	lines = []
	lines.append("## Feature Catalog\n")
	for f in features:
		lines.append(f"### {f.label}\n")
		lines.append(f"{f.description}\n")
		lines.append("```python")
		lines.append(f.python_src.strip())
		lines.append("```\n")
	return "\n".join(lines)

def drop_high_variance_mad(
	wide: pd.DataFrame,
	index_columns: list[str],
	z_thresh: float = 3.5,
	min_nonnull: int = 10,
	ddof: int = 1,
	return_stats: bool = False,
):
	"""
	Drop pivoted feature columns with unusually high variance using robust MAD z-score on log1p(variance).

	wide: dataframe after pivot + reset_index()
	index_columns: the columns that identify rows (not features)
	z_thresh: robust z-score threshold (3.5 is a common "outlier" cutoff)
	min_nonnull: only evaluate/drop features with at least this many non-null values in wide
	"""
	# candidate feature columns = everything except index columns, numeric only
	candidate = [c for c in wide.columns if c not in set(index_columns)]
	num_cols = wide[candidate].select_dtypes(include="number").columns.tolist()

	if not num_cols:
		if return_stats:
			return pd.Index([]), pd.DataFrame(columns=["feature", "nonnull", "variance", "log1p_variance", "robust_z"])
		return pd.Index([])

	nonnull = wide[num_cols].notna().sum(axis=0)
	eligible_cols = nonnull[nonnull >= min_nonnull].index

	if len(eligible_cols) == 0:
		if return_stats:
			stats = pd.DataFrame({
				"feature": num_cols,
				"nonnull": nonnull.reindex(num_cols).values,
				"variance": np.nan,
				"log1p_variance": np.nan,
				"robust_z": np.nan,
			})
			return pd.Index([]), stats
		return pd.Index([])

	v = wide[eligible_cols].var(axis=0, skipna=True, ddof=ddof)
	lv = np.log1p(v)

	med = lv.median()
	mad = (lv - med).abs().median()
	denom = 1.4826 * mad + 1e-12  # avoid div-by-zero when all variances similar
	robust_z = (lv - med) / denom

	dropped = robust_z[robust_z > z_thresh].index
	wide_filt = wide.drop(columns=dropped)

	if not return_stats:
		return dropped

	stats = pd.DataFrame({
		"feature": eligible_cols,
		"nonnull": nonnull.reindex(eligible_cols).values,
		"variance": v.reindex(eligible_cols).values,
		"log1p_variance": lv.reindex(eligible_cols).values,
		"robust_z": robust_z.reindex(eligible_cols).values,
	}).sort_values("robust_z", ascending=False)

	return dropped, stats

def drop_features_highly_correlated_with_index_columns(
	df: pd.DataFrame,
	feature_cols,
	index_cols,
	thresh: float = 0.9999,
):
	"""
	Return a list of feature names whose absolute Pearson correlation with
	ANY (numeric) column in index_cols is >= thresh.

	Non-numeric index columns are ignored; bool is treated as numeric.
	"""
	targets = {}

	for col in index_cols:
		if col not in df.columns:
			continue
		s = df[col]

		if s.dtype == bool:
			s_num = s.astype(float)
		elif is_numeric_dtype(s):
			s_num = s.astype(float)
		else:
			s_num = pd.to_numeric(s, errors="coerce")

		s_num = s_num.replace([np.inf, -np.inf], np.nan)
		nz = s_num.dropna()
		if nz.nunique() < 2:
			continue

		targets[col] = s_num

	drop_cols = set()

	for feat in feature_cols:
		x = pd.to_numeric(df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan)
		for t_name, t_series in targets.items():
			mask = x.notna() & t_series.notna()
			if mask.sum() < 2:
				continue
			corr = x[mask].corr(t_series[mask])
			if pd.isna(corr):
				continue
			if abs(corr) >= thresh:
				drop_cols.add(feat)
				break

	return list(drop_cols)

def safe_features_fillna(scores_df, fill_number=0, fill_bool=False, cols_not_to_fill=None):
	cols_not_to_fill = set(cols_not_to_fill or [])

	# pick columns (excluding protected ones)
	num_cols = scores_df.select_dtypes(include=["number"]).columns.difference(cols_not_to_fill)
	bool_cols = scores_df.select_dtypes(include=["bool", "boolean"]).columns.difference(cols_not_to_fill)

	# which of those actually contain NaNs
	num_with_na = num_cols[scores_df[num_cols].isna().any()] if len(num_cols) else num_cols
	bool_with_na = bool_cols[scores_df[bool_cols].isna().any()] if len(bool_cols) else bool_cols

	# prints
	if len(num_with_na):
		print(
			f"Filling NaNs with {fill_number} in {len(num_with_na)} numeric columns: "
			+ ", ".join(map(str, num_with_na))
		)
	if len(bool_with_na):
		print(
			f"Filling NaNs with {fill_bool} in {len(bool_with_na)} boolean columns: "
			+ ", ".join(map(str, bool_with_na))
		)

	# fill
	if len(num_cols):
		scores_df[num_cols] = scores_df[num_cols].fillna(fill_number)
	if len(bool_cols):
		scores_df[bool_cols] = scores_df[bool_cols].fillna(fill_bool)

	return scores_df
