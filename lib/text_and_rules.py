import json
import re

import numpy as np
import pandas as pd

def apply_rule_to_features(rule, features, direction='positive'):
	"""
	Convert a rule string to a boolean mask over features.
	Supports common RuleSHAP-like syntaxes.

	direction:
		'positive' (default) -> rows that satisfy the rule
		anything else        -> complement of the rule
	"""
	# Normalize common boolean operators
	expr = rule
	expr = re.sub(r"\s+", " ", expr).strip()
	expr = re.sub(r"\bAND\b", "&", expr, flags=re.IGNORECASE)
	expr = re.sub(r"\bOR\b",  "|", expr, flags=re.IGNORECASE)

	# handle NOT / not as unary negation
	# e.g. "NOT(age > 30)" -> "~(age > 30)"
	expr = re.sub(r"\bNOT\b", "~", expr, flags=re.IGNORECASE)

	# Backtick-quote all column names for pandas.query
	for col in sorted(features.columns, key=len, reverse=True):
		safe = f"`{col}`"
		expr = re.sub(rf'(?<![`"\w]){re.escape(col)}(?![`"\w])', safe, expr)

	# First, try pandas.query (fast, flexible)
	try:
		mask_idx = features.query(expr).index
		mask = pd.Series(False, index=features.index)
		mask.loc[mask_idx] = True
	except Exception:
		# Fallback: pd.eval understands backticks and & / |
		locs = {c: features[c] for c in features.columns}
		try:
			result = pd.eval(expr, local_dict=locs, engine="python")
			if not isinstance(result, (pd.Series, np.ndarray, list)):
				raise ValueError("Rule did not eval to a boolean mask.")
			mask = pd.Series(result, index=features.index).astype(bool)
		except Exception as e2:
			raise ValueError(
				f"Failed to parse/apply rule:\n  {rule}\n  -> expr: {expr}\nerror: {e2}"
			) from e2

	# If direction is not positive, negate the mask ("nor expr" case)
	if direction.lower() != 'positive':
		mask = ~mask

	return mask

# ---------------------------- Rule I/O & parsing -------------------------------

def detect_prompt_col(df, user_hint = None):
	"""
	Heuristically pick the text column containing the prompts if not provided.
	"""
	if user_hint:
		if user_hint in df.columns:
			return user_hint
		raise ValueError(f"--prompt-col '{user_hint}' not found. Available columns: {list(df.columns)}")
	candidates = ["prompt", "input", "text", "question", "query"]
	for c in candidates:
		if c in df.columns:
			return c
	raise ValueError(
		f"Could not auto-detect prompt column. Tried {candidates}. "
		f"Please pass --prompt-col. Available: {list(df.columns)}"
	)

def guess_filetype(p):
	ext = p.suffix.lower()
	if ext in {".csv"}: return "csv"
	if ext in {".json"}: return "json"
	if ext in {".parquet"}: return "parquet"
	if ext in {".pkl", ".pickle"}: return "pkl"
	raise ValueError(f"Unrecognized file extension: {ext} for {p}")

def find_rule_files(rules_dir, prefix="rule_combo"):
	"""
	Return list of (target_name, path) for files in rules_dir matching
	{prefix}_{X}.csv, where X is the inferred target name.
	"""
	out = []

	glob_pattern = f"{prefix}_*.csv"
	re_pattern = re.compile(rf"^{re.escape(prefix)}_(.+)\.csv$")

	for p in sorted(rules_dir.glob(glob_pattern)):
		m = re_pattern.match(p.name)
		if m:
			out.append((m.group(1), p))

	return out

def load_rules(rules_path):
	"""
	Expected flexible formats:
	  - CSV with a 'rule' column (string condition, e.g., "feat_a > 0.1 and feat_b <= 3")
	  - JSON, with key 'rule' or 'string_rule'
	Returns a DataFrame with at least: ['rule', 'rule_id'].
	"""
	ftype = guess_filetype(rules_path)
	if ftype == "csv":
		df = pd.read_csv(rules_path)
	elif ftype == "json":
		data = json.loads(rules_path.read_text())
		if isinstance(data, dict) and "rules" in data: 
			data = data["rules"]
		df = pd.DataFrame(data)
	else:
		raise ValueError("Rules must be provided as CSV or JSON.")
	# normalize
	if "rule" not in df.columns:
		for alt in ["string_rule", "rule_string", "expr", "rule_expression", "expression"]:
			if alt in df.columns:
				df["rule"] = df[alt]
				df["component_type"] = 'rule'
				df["coefficient_sign"] = 'positive'
				break
	if "rule" not in df.columns:
		raise ValueError("Could not find a 'rule' column in rules file.")
	if "rule_id" not in df.columns:
		df["rule_id"] = np.arange(len(df), dtype=int)
	return df[["rule_id", "rule", "coefficient_sign", "component_type"]].copy()
