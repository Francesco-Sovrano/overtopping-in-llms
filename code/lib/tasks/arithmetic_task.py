from __future__ import annotations

from dataclasses import dataclass

from lib.feature_representation import Feature
from lib.task_spec import FeatureTaskSpec
from lib.modeling_and_ablation import LMWrapper, get_device
from lib.caching_and_prompting import load_cache

import re
import pandas as pd

import torch
from tqdm import tqdm

OPERATORS = ['-', '+', '*' , '/']
OPERATOR_NAMES = ['subtraction', 'addition', 'multiplication', 'division']
POSITIONS = [1, 2, 3, 4] # All actual token positions (1 = op1, 2 = operator, 3 = op2, 4 = equals sign)
		 
def extract_single_number(text: str):
	# Find all sequences of digits in the string
	numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
	
	# If there's exactly one number, return it (as int)
	if numbers and len(numbers)==1:
		n = numbers[0]
		return float(n) if '.' in n else int(n)
	return None

def _is_answer_correct(prompt, answer):
	# print(prompt, answer)
	"""
	Checks if an answer is a correct completion to a prompt.
	Whitespaces are ignored.

	Args:
		prompt (str): The prompt (for example '5+4=')
		answer (str): The answer (for example '9')
		convert_to_int (bool): If True, the ground truth answer is converted to an integer before comparison to the tested answer.
	"""
	# Handle few-shot case
	few_shot_sep = ';' if ';' in prompt else (',' if ',' in prompt else None)
	if few_shot_sep is not None:
		prompt = prompt[prompt.rfind(few_shot_sep) + 1:]
	correct_answer = eval(prompt.replace('=', ''))

	numerical_answer = to_number(answer)
	if numerical_answer is None:
		numerical_answer = extract_single_number(answer)
	if numerical_answer is None:
		return False
	# print(type(numerical_answer), type(correct_answer))
	answer = str(numerical_answer)
	correct_answer = str(correct_answer)
	result = (answer.startswith(correct_answer) or correct_answer.startswith(answer)) if '.' in correct_answer else (correct_answer == answer)
	# print(result, answer==correct_answer, answer, correct_answer)
	return result

def generate_prompts(model, operand_ranges, batch_size=8, max_new_tokens=None):
	prompts_and_answers_dict = {}

	for operator in operand_ranges.keys():
		# Generate all possible prompts for the given operator within the operand limits
		operand_min, operand_max = operand_ranges[operator]
		
		prompts = [
			'{x}{op}{y}='.format(x=operand1, op=operator, y=operand2)
			for operand1 in range(operand_min, operand_max)
			for operand2 in range(operand_min, operand_max)
		]
		
		prompts_and_answers = []
		dataloader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, shuffle=False)
		# print(dir(model))
		for prompts in tqdm(dataloader, desc=f'Generating answers; operator {operator}'):
			answers = model.generate(
				prompts,
				max_new_tokens=max_new_tokens,
				do_sample=False,
				fwd_hooks=None,
				# use_kv_cache=False
			)
			
			is_correct_answer_list = [
				_is_answer_correct(prompt, answer) 
				for prompt, answer in zip(prompts, answers)
			]
			print(f"Correct batch answers: {100*sum(is_correct_answer_list)/len(is_correct_answer_list):.2f}%" if is_correct_answer_list else "0%")

			numerical_answers = list(map(extract_single_number, answers))
			is_number_only_list = [
				float(answer.strip()) == float(number) if (number is not None and is_number(answer)) else False
				for answer, number in zip(answers, numerical_answers)
			]
			prompts_and_answers += zip(prompts, numerical_answers, answers, is_correct_answer_list, is_number_only_list)
		
		prompts_and_answers_dict[operator] = prompts_and_answers
	return prompts_and_answers_dict


def separate_prompts_and_answers(prompts_and_answers):
	"""
	Separates a list of (prompt, answer) tuples to two lists - one of prompts and one of answers.
	"""
	return [pa[0] for pa in prompts_and_answers], [pa[1] for pa in prompts_and_answers]

def get_operand_range(operator, previous_operand, operand_min, operand_max, max_single_token_value):
	if operator == '+':
		return range(operand_min, min(max_single_token_value - previous_operand, operand_max))
	elif operator == '-':
		return range(operand_min, previous_operand + 1)
	elif operator == '*':
		if previous_operand == 0:
			return range(operand_min, operand_max)
		else:
			return range(operand_min, min((max_single_token_value // previous_operand) + 1, operand_max))
	elif operator == '/':
		return range(max(1, operand_min), operand_max)
	else:
		raise ValueError(f'Operator {operator} is not supported')

def to_number(s):
	try:
		return int(s)
	except:
		try:
			return float(s)
		except:
			return None

def is_number(s, is_int=False):
	return to_number(s) is not None
	

def is_writing_of_number(s: str):
	word_to_number = {
		'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
		'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
		'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
		'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
		'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
		'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
		'million': 1000000
	}

	words = s.split()
	for word in words:
		if word not in word_to_number:
			return False
	return True

# ---------------------------
# Prompt parsing tokens
# ---------------------------
def parse_arithmetic_prompt(prompt):
	s = (prompt or "").strip()
	s = s.replace("×", "*").replace("·", "*").replace("∙", "*").replace("x", "*").replace("X", "*").replace("÷", "/")
	m = re.match(r"^\s*([+\-]?\d+(?:\.\d+)?)\s*([+\-*/%])\s*([+\-]?\d+(?:\.\d+)?)\s*=?\s*$", s)
	if not m:
		return {
			"operator": None,
			"operand_a": None, "operand_b": None, "a_int": None, "b_int": None,
			"len_a": None, "len_b": None, "digits_a": [], "digits_b": [],
			"sign_a": None, "sign_b": None
		}
	a_str, op, b_str = m.groups()

	def _to_float(ss):
		try: return float(ss)
		except Exception: return None

	def _to_int(ss):
		if ss and "." not in ss:
			try: return int(ss)
			except Exception: return None
		return None

	def _digits_len(ss): return len(re.sub(r"[^0-9]", "", ss or ""))
	def _digits_list(ss):
		s0 = re.sub(r"[^0-9]", "", ss or "")
		if "." in (ss or ""): return []
		return [int(ch) for ch in s0] if s0 else []
	def _sign(ss): return -1 if str(ss).strip().startswith("-") else 1

	a = _to_float(a_str); b = _to_float(b_str)
	a_int = _to_int(a_str); b_int = _to_int(b_str)
	return {
		"operator": op,
		"operand_a": a, "operand_b": b,
		"a_int": a_int, "b_int": b_int,
		"len_a": _digits_len(a_str), "len_b": _digits_len(b_str),
		"digits_a": _digits_list(a_str), "digits_b": _digits_list(b_str),
		"sign_a": _sign(a_str), "sign_b": _sign(b_str),
	}


def make_closed_bins(start: int, stop: int, step: int):
	"""
	Build [lo, hi] closed bins as integer pairs.
	Example: make_closed_bins(0, 300, 25) -> (0,25), (25,50), ..., (275,300)
	"""
	bins = []
	x = start
	while x < stop:
		lo = x
		hi = min(x + step, stop)
		bins.append((lo, hi))
		x += step
	return bins

def build_range_seed_features(
	result_bins = None,
	operand_bins = None,
	div_q_bins = None,
):
	"""
	Returns a list[Feature] with many range features generated by loops.
	- result_bins: ranges for the arithmetic result of +, -, *
	- operand_bins: ranges for a and b (op1/op2)
	- div_q_bins: integer-quotient ranges for / (q_int = floor(a/b))
	"""
	# sensible defaults for your 0..299 style dataset; tweak freely
	result_bins  = result_bins  or make_closed_bins(0, 300, 25)
	operand_bins = operand_bins or make_closed_bins(0, 300, 25)
	div_q_bins   = div_q_bins   or make_closed_bins(0, 30,  3)

	feats = []

	# --- (+, -, *) result in [lo, hi] ---
	op_specs = [
		("+", "add", "a + b"),
		("-", "sub", "a - b"),
		("*", "mul", "a * b"),
	]
	for sym, name, expr in op_specs:
		for lo, hi in result_bins:
			label = f"Result in [{lo},{hi}] ({name})"
			desc  = f"1 if {name} result is in [{lo},{hi}]"
			py = f"""
def f_res_{name}_{lo}_{hi}(prompt: str, info: dict) -> float:
	op = info.get('operator'); a = info.get('operand_a'); b = info.get('operand_b')
	if op != '{sym}':
		return None
	r = {expr}
	return 1.0 if {lo} <= r <= {hi} else 0.0
"""
			feats.append(Feature(label, desc, py, origin="predefined"))

	# --- operand a in [lo, hi] ---
	for lo, hi in operand_bins:
		label = f"op1 in [{lo},{hi}]"
		desc  = f"1 if first operand a ∈ [{lo},{hi}]"
		py = f"""
def f_op1_in_{lo}_{hi}(prompt: str, info: dict) -> float:
	a = info.get('operand_a')
	return 1.0 if {lo} <= a <= {hi} else 0.0
"""
		feats.append(Feature(label, desc, py, origin="predefined"))

	# --- operand b in [lo, hi] ---
	for lo, hi in operand_bins:
		label = f"op2 in [{lo},{hi}]"
		desc  = f"1 if second operand b ∈ [{lo},{hi}]"
		py = f"""
def f_op2_in_{lo}_{hi}(prompt: str, info: dict) -> float:
	b = info.get('operand_b')
	return 1.0 if {lo} <= b <= {hi} else 0.0
"""
		feats.append(Feature(label, desc, py, origin="predefined"))

	# --- division: integer quotient in [lo, hi] ---
	for lo, hi in div_q_bins:
		label = f"Division: q_int in [{lo},{hi}]"
		desc  = f"1 if floor(a/b) ∈ [{lo},{hi}]"
		py = f"""
def f_div_q_{lo}_{hi}(prompt: str, info: dict) -> float:
	op = info.get('operator'); a = info.get('operand_a'); b = info.get('operand_b')
	if op != '/': return None
	if b == 0: return None
	q_int = int(a // b)
	return 1.0 if {lo} <= q_int <= {hi} else 0.0
"""
		feats.append(Feature(label, desc, py, origin="predefined"))

	return feats


def base_seed_features() -> list[Feature]:
	return [
		Feature("Operands equal", "1 if a==b (useful for sanity cases like n/n)", """
def f_equal_operands(prompt, info):
	a = info.get('operand_a'); b = info.get('operand_b')
	return 1.0 if a == b else 0.0
""", origin="predefined"),

		Feature("Operator: Addition", "Operator is '+'", """
def op_is_addition(prompt, info):
	return 1.0 if info.get('operator') == '+' else 0.0
""", origin="predefined"),
		Feature("Operator: Subtraction", "Operator is '-'", """
def op_is_subtraction(prompt, info):
	return 1.0 if info.get('operator') == '-' else 0.0
""", origin="predefined"),
		Feature("Operator: Multiplication", "Operator is '*'", """
def op_is_multiplication(prompt, info):
	return 1.0 if info.get('operator') == '*' else 0.0
""", origin="predefined"),
		Feature("Operator: Division", "Operator is '/'", """
def op_is_division(prompt, info):
	return 1.0 if info.get('operator') == '/' else 0.0
""", origin="predefined"),

		Feature("Same digit count", "Both operands have the same number of digits (ignoring signs/decimals).", """
def same_digit_count(prompt, info):
	la = info.get('len_a'); lb = info.get('len_b')
	return 1.0 if la == lb else 0.0
""", origin="predefined"),
		Feature("Left has more digits", "Left operand has more digits than right.", """
def left_has_more_digits(prompt, info):
	la = info.get('len_a'); lb = info.get('len_b')
	return 1.0 if la > lb else 0.0
""", origin="predefined"),
		Feature("Right has more digits", "Right operand has more digits than left.", """
def right_has_more_digits(prompt, info):
	la = info.get('len_a'); lb = info.get('len_b')
	return 1.0 if lb > la else 0.0
""", origin="predefined"),

		Feature("Divides evenly (any order)", "One integer operand divides the other exactly.", """
def divides_evenly_either_way(prompt, info):
	a = info.get('operand_a'); b = info.get('operand_b')
	if a == 0 or b == 0:
		return 0.0
	return 1.0 if (a % b == 0) or (b % a == 0) else 0.0
""", origin="predefined"),

		Feature("GCD > 1", "Operands share a non-trivial common factor.", """
def gcd_gt_1(prompt, info):
	a = info.get('operand_a'); b = info.get('operand_b')
	x, y = abs(a), abs(b)
	while y:
		x, y = y, x % y
	return 1.0 if x > 1 else 0.0
""", origin="predefined"),

		Feature("Carry in addition", "If '+' and column-wise addition needs a carry.", """
def has_carry_in_addition(prompt, info):
	if info.get('operator') != '+':
		return 0.0
	a = info.get('operand_a'); b = info.get('operand_b')
	A = str(abs(a)); B = str(abs(b))
	carry = 0
	i = 1
	while i <= max(len(A), len(B)):
		da = int(A[-i]) if i <= len(A) else 0
		db = int(B[-i]) if i <= len(B) else 0
		s = da + db + carry
		if s >= 10:
			return 1.0
		carry = s // 10
		i += 1
	return 0.0
""", origin="predefined"),

		Feature("Borrow in subtraction", "If '-' and column-wise subtraction needs a borrow.", """
def requires_borrow_in_subtraction(prompt, info):
	if info.get('operator') != '-':
		return 0.0
	a = info.get('operand_a'); b = info.get('operand_b')
	A = str(abs(a)); B = str(abs(b))
	borrow = 0
	i = 1
	while i <= max(len(A), len(B)):
		da = int(A[-i]) if i <= len(A) else 0
		db = int(B[-i]) if i <= len(B) else 0
		da -= borrow
		if da < db:
			return 1.0
		borrow = 1 if da < db else 0
		i += 1
	return 0.0
""", origin="predefined"),

		Feature("Division denominator is power of 10", "If '/' and right operand is 10^k.", """
def denominator_is_power_of_10(prompt, info):
	if info.get('operator') != '/':
		return 0.0
	b = info.get('operand_b')
	if b <= 0:
		return None
	x = b
	while x % 10 == 0:
		x //= 10
	return 1.0 if x == 1 else 0.0
""", origin="predefined"),

		Feature("Operands share a digit", "Operands (integers) share at least one digit.", """
def operands_share_digit(prompt, info):
	da = info.get('digits_a') or []
	db = info.get('digits_b') or []
	return 1.0 if set(da) & set(db) else 0.0
""", origin="predefined"),

		Feature("Relative closeness ≤ 10%", "Operands close in value: |a-b|/max(|a|,|b|) ≤ 0.1.", """
def operands_close_in_value(prompt, info):
	a = info.get('operand_a'); b = info.get('operand_b')
	mx = max(abs(a), abs(b))
	if mx == 0:
		return 1.0
	return 1.0 if abs(a - b) / mx <= 0.1 else 0.0
""", origin="predefined"),

		Feature("Same sign", "Operands have the same sign.", """
def same_sign(prompt, info):
	sa = info.get('sign_a'); sb = info.get('sign_b')
	if sa is None or sb is None:
		return None
	return 1.0 if sa == sb else 0.0
""", origin="predefined"),
	]

@dataclass
class ArithmeticTaskSpec(FeatureTaskSpec):
	DEFAULT_TARGETS = (
		"is_correct", 
		# "is_pure"
	)
	MAX_NEW_TOKENS = 6
	DEFAULT_INPUT = "prompt"
	DEFAULT_OUTPUT = "raw_output"

	SYSTEM_PROMPT = (
		"You're an expert data analyst. "
		"Propose concise, testable features that might correlate with the correctness of LLM-generated outputs for arithmetic input prompts in the form of “a/b=”, “a*b=”, “a-b=”, or “a+b=” (e.g., “299/298=”). "
		"Propose orthogonal features that are HIGH/PRESENT for CORRECTLY ANSWERED and LOW/ABSENT for INCORRECTLY ANSWERED (or vice versa) prompts. "
		"Think about operators, operand magnitudes, patterns (digits, parity, carries/borrows), divisibility, relative sizes, ranges, oddity, etc."
	)

	TOKENS_DICT_KEYS = "operator, operand_a, operand_b, len_a, len_b, digits_a, digits_b"

	# Build once at import-time (deterministic) for reproducibility.
	SEED_FEATURES = (
		base_seed_features()
		+ build_range_seed_features(
			result_bins=make_closed_bins(0, 300, 25),
			operand_bins=make_closed_bins(0, 300, 25),
			div_q_bins=make_closed_bins(0, 30, 3),
		)
	)

	def is_answer_positive(self, prompt_batch, response_texts):
		return [
			_is_answer_correct(prompt_data[self.DEFAULT_INPUT], answer)
			for prompt_data, answer in zip(prompt_batch, response_texts)
		]

	def parse_prompt_row(self, prompt_row):
		return parse_arithmetic_prompt(getattr(prompt_row, self.DEFAULT_INPUT))

	def generate_cache(self, ai_model, ai_model_cache_dir, args):
		max_operand = getattr(args, "max_operand", 300)
		batch_size = getattr(args, "batch_size", 16)
		max_new_tokens = getattr(args, "max_new_tokens", self.MAX_NEW_TOKENS)

		operand_ranges = {
			"+": (0, max_operand),
			"-": (0, max_operand),
			"*": (0, max_operand),
			"/": (1, max_operand),
		}

		device = get_device()
		# logging.info(f"Loading model {ai_model}")
		model = LMWrapper(
			ai_model,
			device,
			eval_mode=True,
			circuit_discovery=False,
			cache_dir=ai_model_cache_dir,
		)

		return generate_prompts(
			model,
			operand_ranges=operand_ranges,
			batch_size=batch_size,
			max_new_tokens=max_new_tokens, 
		)

	def load_dataset_from_cache(self, pkl_path):
		use_pure_only=False
		obj = load_cache(pkl_path)
		rows = []
		for op, items in (obj.items() if isinstance(obj, dict) else []):
			for t in items:
				if not isinstance(t, (list, tuple)) or len(t) < 5:
					continue
				prompt, num_out, raw_out, is_correct, is_pure = t#[:5]
				if use_pure_only and not bool(is_pure):
					continue
				rows.append({
					"operator_group": op,
					self.DEFAULT_INPUT: prompt.strip(),
					"num_out": None if num_out is None else float(num_out),
					self.DEFAULT_OUTPUT: raw_out,
					"is_correct": bool(is_correct),
					"is_pure": bool(is_pure),
				})
		df = pd.DataFrame(rows)
		# Drop empties / duplicates
		df = df.dropna(subset=[self.DEFAULT_INPUT]).drop_duplicates(subset=[self.DEFAULT_INPUT]).reset_index(drop=True)
		return df

TASK_SPEC = ArithmeticTaskSpec()
