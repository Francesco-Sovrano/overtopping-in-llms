import os
import atexit
import signal
import time

try:
	import resource
	soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)      # current = (256, 10240) on macOS
	new_soft = min(hard, 40960)                                  # never exceed hard
	resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard)) # requires sudo if > hard
except ImportError:
	pass

import json
import pickle
from tqdm import tqdm
import multiprocessing
from more_itertools import unique_everseen
import concurrent.futures
import copy
import ollama
import openai
from groq import Groq

import csv
import random
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def set_deterministic(seed=1337):
	if seed is None:
		return
	os.environ.setdefault("PYTHONHASHSEED", str(seed))
	try:
		import random as _random
		_random.seed(seed)
	except Exception:
		pass
	try:
		import numpy as _np
		_np.random.seed(seed)
	except Exception:
		pass
	try:
		import torch as _torch  # type: ignore
		_torch.manual_seed(seed)
		_torch.cuda.manual_seed_all(seed)
		try:
			import torch.backends.cudnn as _cudnn  # type: ignore
			_cudnn.deterministic = True
			_cudnn.benchmark = False
		except Exception:
			pass
		_torch.use_deterministic_algorithms(True)
	except Exception:
		pass
	os.environ.setdefault("OMP_NUM_THREADS", "1")
	os.environ.setdefault("MKL_NUM_THREADS", "1")

def _tqdm(it, visible=True, **args):
	if isinstance(it, (list,tuple)) and len(it) <= 1:
		return it
	total = args.get('total', None)
	if total is not None and total <= 1:
		return it
	if not visible:
		return it
	return tqdm(it, **args)

_CACHE_SAVE_INTERVAL_SECONDS = 300.0
_CACHE_LAST_SAVE_TIME = {}
_DEFERRED_CACHE_WRITES = {}
_DEFERRED_CACHE_FLUSH_REGISTERED = False

def _cache_save_interval_seconds():
	try:
		return max(0.0, float(os.environ.get("CACHE_SAVE_INTERVAL_SECONDS", _CACHE_SAVE_INTERVAL_SECONDS)))
	except (TypeError, ValueError):
		return _CACHE_SAVE_INTERVAL_SECONDS

def _write_cache_file(file_name, cache, quiet=False):
	parent_dir = os.path.dirname(file_name)
	if parent_dir:
		os.makedirs(parent_dir, exist_ok=True)
	if not quiet:
		print(f'Creating cache <{file_name}>..')
	tmp_name = f"{file_name}.tmp.{os.getpid()}"
	with open(tmp_name, 'wb') as f:
		pickle.dump(cache, f)
	os.replace(tmp_name, file_name)
	_CACHE_LAST_SAVE_TIME[file_name] = time.time()


def flush_deferred_caches(quiet=False):
	"""Persist all dirty in-memory caches immediately.

	Registered with atexit and signal handlers so long runs do not need to
	re-write cache files after every fetched instruction, while still flushing
	progress when the process exits or is interrupted.
	"""
	for file_name, (cache, cache_quiet) in list(_DEFERRED_CACHE_WRITES.items()):
		_write_cache_file(file_name, cache, quiet=(quiet or cache_quiet))
		_DEFERRED_CACHE_WRITES.pop(file_name, None)


def _register_deferred_cache_flush():
	global _DEFERRED_CACHE_FLUSH_REGISTERED
	if _DEFERRED_CACHE_FLUSH_REGISTERED:
		return
	_DEFERRED_CACHE_FLUSH_REGISTERED = True
	atexit.register(flush_deferred_caches)

	for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
		if sig is None:
			continue
		try:
			previous_handler = signal.getsignal(sig)

			def _flush_then_delegate(signum, frame, previous_handler=previous_handler):
				flush_deferred_caches()
				if callable(previous_handler):
					previous_handler(signum, frame)
				elif previous_handler == signal.SIG_IGN:
					return
				elif signum == getattr(signal, "SIGINT", None):
					raise KeyboardInterrupt
				else:
					raise SystemExit(128 + signum)

			signal.signal(sig, _flush_then_delegate)
		except (ValueError, OSError):
			# Signal handlers can only be installed from the main thread on some platforms.
			pass


def save_cache_throttled(file_name, cache, quiet=False, force=False):
	_register_deferred_cache_flush()
	if force:
		_write_cache_file(file_name, cache, quiet=quiet)
		_DEFERRED_CACHE_WRITES.pop(file_name, None)
		return

	_DEFERRED_CACHE_WRITES[file_name] = (cache, quiet)
	last_save = _CACHE_LAST_SAVE_TIME.get(file_name, time.time())
	if file_name not in _CACHE_LAST_SAVE_TIME:
		_CACHE_LAST_SAVE_TIME[file_name] = last_save
	if time.time() - last_save >= _cache_save_interval_seconds():
		_write_cache_file(file_name, cache, quiet=quiet)
		_DEFERRED_CACHE_WRITES.pop(file_name, None)


def create_cache(file_name, create_fn, quiet=False):
	result = create_fn()
	_write_cache_file(file_name, result, quiet=quiet)
	return result

def load_cache(file_name, quiet=False):
	parent_dir = os.path.dirname(file_name)
	if parent_dir:
		os.makedirs(parent_dir, exist_ok=True) # Ensure parent directory exists

	if os.path.isfile(file_name):
		if not quiet:
			print(f'Loading cache <{file_name}>..')
		with open(file_name,'rb') as f:
			result = pickle.load(f)
		_CACHE_LAST_SAVE_TIME[file_name] = time.time()
		return result
	return None

def load_or_create_cache(file_name, create_fn, quiet=False):
	result = load_cache(file_name, quiet=quiet)
	if result is None:
		result = create_cache(file_name, create_fn, quiet=quiet)
	return result

def _is_missing_cached_value(v):
	if v is None:
		return True
	# Empty string/list/dict/tuple
	if isinstance(v, (str, list, tuple, dict, set)) and len(v) == 0:
		return True
	# NumPy arrays
	try:
		import numpy as np
		if isinstance(v, np.ndarray):
			return v.size == 0
	except Exception:
		pass
	return False

def get_cached_values(value_list, cache, fetch_fn, cache_name=None, key_fn=lambda x:x, empty_is_missing=True, transform_fn=None, **args):
	missing_values = tuple(
		q 
		for q in unique_everseen(filter(lambda x:x, value_list), key=key_fn) 
		if key_fn(q) not in cache or (empty_is_missing and _is_missing_cached_value(cache[key_fn(q)]))
	)

	# print('get_cached_values', list(cache.keys()[0], indent=2))
	if len(missing_values) > 0:
		for q, v in fetch_fn(missing_values):
			cache[key_fn(q)] = v
		if cache_name:
			save_cache_throttled(cache_name, cache)
	cached_values = [
		cache[key_fn(q)] if q else None 
		for q in value_list
	]
	if transform_fn:
		cached_values = list(map(transform_fn, cached_values))
	return cached_values

def _is_reasoning_model(model_name):
	n = (model_name or "").lower()
	return (
		n.startswith('o1') or n.startswith('o3') or n.startswith('o4') or  # OpenAI o-family
		('deepseek' in n and 'r1' in n) or                                 # DeepSeek R1 / R1-distill
		('qwen3' in n) or                                                  # Qwen3 family
		('reason' in n)                                                    # generic catch-all
	)

def get_document_list(directory):
	doc_list = []
	for obj in os.listdir(directory):
		obj_path = os.path.join(directory, obj)
		if os.path.isfile(obj_path):
			doc_list.append(obj_path)
		elif os.path.isdir(obj_path):
			doc_list.extend(get_document_list(obj_path))
	return doc_list

_loaded_caches = {}
def instruct_model(prompts, model='llama3.1', api_key=None, reasoning_effort='none', **kwargs):
	if model.startswith('gpt') or model.startswith('o'):
		api_key = api_key or os.getenv('OPENAI_API_KEY', '')
		base_url = "https://api.openai.com/v1"
		parallelise = True
		return instruct_openai_model(prompts, api_key=api_key, model=model, base_url=base_url, parallelise=parallelise, **kwargs)
	elif model in ['qwen/qwen3-32b', 'meta-llama/llama-4-scout-17b-16e-instruct']:
		api_key = api_key or os.getenv('GROQ_API_KEY', '')
		# base_url = "https://api.groq.com/openai/v1"
		parallelise = True
		return instruct_groq_model(prompts, api_key=api_key, model=model, parallelise=parallelise, reasoning_effort=reasoning_effort, **kwargs)
	else:
		api_key = api_key or 'ollama' # required, but unused
		base_url = 'http://localhost:11434/v1'
		parallelise = False
		return instruct_ollama_model(prompts, api_key=api_key, model=model, base_url=base_url, parallelise=parallelise, **kwargs)
			
def instruct_ollama_model(prompts, system_instructions=None, model='llama3.1', options=None, temperature=0.5, top_p=1, output_to_input_proportion=2, non_influential_prompt_size=0, cache_path='cache/', max_tokens=None, seed=42, parallelise=False, **args):
	if max_tokens is None:
		max_tokens = -1 # no limits
	if options is None:
		# For Mistral: https://www.reddit.com/r/LocalLLaMA/comments/16v820a/mistral_7b_temperature_settings/
		options = { # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
			"seed": seed, # Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)
			"num_predict": max_tokens, # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
			"top_k": 40, # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
			"top_p": 0.95, # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
			"temperature": 0.7, # The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
			"repeat_penalty": 1.1, # Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
			"tfs_z": 1, # Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)
			"num_ctx": 2**13,  # Sets the size of the context window used to generate the next token. (Default: 2048)
			"repeat_last_n": 64, # Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
			# "num_gpu": 0, # The number of layers to send to the GPU(s). Set to 0 to disable.
		}
	else:
		options = copy.deepcopy(options) # required to avoid side-effects
	options.update({
		"temperature": temperature,
		"top_p": top_p,
	})
	def fetch_fn(instruction_prompt):
		system_instruction, missing_prompt = instruction_prompt
		_options = copy.deepcopy(options) # required to avoid side-effects
		if _options.get("num_predict",-2) == -2:
			prompt_tokens = 2*(len(missing_prompt.split(' '))-non_influential_prompt_size)
			_options["num_predict"] = int(output_to_input_proportion*prompt_tokens)
		response = ollama.generate(
			model=model,
			prompt=missing_prompt,
			stream=False,
			options=_options,
			keep_alive='1h',
			system=system_instruction,
		)
		# print(missing_prompt, response['response'])
		# return also the missing_prompt otherwise asynchronous prompting will shuffle the outputs
		return instruction_prompt, response['response']
	def parallel_fetch_fn(missing_prompt_list):
		if parallelise:
			n_processes = multiprocessing.cpu_count()
			with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,n_processes)) as executor:
				futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
				for future in _tqdm(concurrent.futures.as_completed(futures), total=len(missing_prompt_list), desc="Sending prompts to Ollama", leave=False):
					i,o=future.result()
					yield i,o
		else:
			# print(len(missing_prompt_list))
			for p in _tqdm(missing_prompt_list, total=len(missing_prompt_list), desc="Sending prompts to Ollama", leave=False):
				i,o=fetch_fn(p)
				yield i,o
		
	os.makedirs(cache_path, exist_ok=True)
	ollama_cache_name = os.path.join(cache_path, f"_{model.replace('-','_')}_cache.pkl")
	if ollama_cache_name not in _loaded_caches:
		_loaded_caches[ollama_cache_name] = load_or_create_cache(ollama_cache_name, lambda: {})
	__ollama_cache = _loaded_caches[ollama_cache_name]
	cache_key = json.dumps(options,indent=4)
	return get_cached_values(
		list(zip(system_instructions if system_instructions else [None]*len(prompts), prompts)), 
		__ollama_cache, 
		parallel_fetch_fn, 
		# key_fn=lambda x: (x,model,n,temperature,top_p,frequency_penalty,presence_penalty), 
		key_fn=lambda x: (x,model,cache_key),  
		empty_is_missing=True,
		cache_name=ollama_cache_name,
		transform_fn=None
	)

def instruct_openai_model(prompts, system_instructions=None, api_key=None, base_url=None, model='gpt-4o-mini', n=1, temperature=1, top_p=1, frequency_penalty=0, presence_penalty=0, cache_path='cache/', parallelise=True, max_tokens=None, timeout=None, **kwargs):
	chatgpt_client = openai.OpenAI(api_key=api_key, base_url=base_url)
	if max_tokens is None:
		adjust_max_tokens = True
		if '32k' in model:
			max_tokens = 32768
		elif '16k' in model:
			max_tokens = 16385
		elif model=='gpt-4o' or 'preview' in model or 'turbo' in model:
			max_tokens = 4096 #128000
			adjust_max_tokens = False
		elif model.startswith('o1') or model.startswith('o3') or model.startswith('o4'):
			max_tokens = 2**16
			adjust_max_tokens = False
		if not max_tokens:
			if model.startswith('gpt-4'):
				max_tokens = 8192
			else:
				max_tokens = 4096
				adjust_max_tokens = False
	else:
		adjust_max_tokens = True
		if model=='gpt-4o' or 'preview' in model or 'turbo' in model:
			adjust_max_tokens = False
		elif model.startswith('o1') or model.startswith('o3') or model.startswith('o4'):
			adjust_max_tokens = False
	# print('max_tokens', max_tokens)
	def fetch_fn(instruction_prompt):
		system_instruction, missing_prompt = instruction_prompt
		if system_instruction:
			messages = [ 
				{"role": "system", "content": system_instruction},
			]
		else:
			messages = []
		messages += [ 
			{"role": "user", "content": missing_prompt} 
		]
		prompt_max_tokens = max_tokens
		if adjust_max_tokens:
			prompt_max_tokens -= int(3*len(missing_prompt.split(' \n')))
		if prompt_max_tokens < 1:
			return instruction_prompt, None
		try:
			if model.startswith("o") or model.startswith('gpt-5'): # some params not available in reasoning models
				response = chatgpt_client.chat.completions.create(
					model=model,
					messages=messages,
					max_completion_tokens=prompt_max_tokens,
					n=n,
					stop=None,
					frequency_penalty=frequency_penalty, 
					presence_penalty=presence_penalty,
					timeout=timeout
				)
			else:
				response = chatgpt_client.chat.completions.create(
					model=model,
					messages=messages,
					max_tokens=prompt_max_tokens,
					n=n,
					stop=None,
					temperature=temperature,
					top_p=top_p,
					frequency_penalty=frequency_penalty, 
					presence_penalty=presence_penalty,
					timeout=timeout
				)
			# print(response.choices)
			result = [
				r.message.content.strip() 
				for r in response.choices 
				if r.message.content != 'Hello! It seems like your message might have been cut off. How can I assist you today?'
			]
			if len(result) == 1:
				result = result[0]
			return instruction_prompt, result # return also the missing_prompt otherwise asynchronous prompting will shuffle the outputs
		except Exception as e:
			print(f'OpenAI returned this error: {e}')
			return instruction_prompt, None
	def parallel_fetch_fn(missing_prompt_list):
		if parallelise:
			n_processes = multiprocessing.cpu_count()
			# Using ThreadPoolExecutor to run queries in parallel with tqdm for progress tracking
			with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,n_processes)) as executor:
				futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
				for e,future in enumerate(_tqdm(concurrent.futures.as_completed(futures), total=len(missing_prompt_list), desc="Sending prompts to OpenAI", leave=False)):
					i,o=future.result()
					yield i,o
		else:
			for p in _tqdm(missing_prompt_list, total=len(missing_prompt_list), desc="Sending prompts to OpenAI", leave=False):
				i,o=fetch_fn(p)
				yield i,o
		
	os.makedirs(cache_path, exist_ok=True)
	model_id = model.replace("/", "_").replace("\\", "_").replace(":", "_")
	openai_cache_name = os.path.join(cache_path, f"_{model_id}_cache.pkl")
	if openai_cache_name not in _loaded_caches:
		_loaded_caches[openai_cache_name] = load_or_create_cache(openai_cache_name, lambda: {})
	__openai_cache = _loaded_caches[openai_cache_name]
	return get_cached_values(
		list(zip(system_instructions if system_instructions else [None]*len(prompts), prompts)), 
		__openai_cache, 
		parallel_fetch_fn, 
		# key_fn=lambda x: (x,model,n,temperature,top_p,frequency_penalty,presence_penalty), 
		key_fn=lambda x: (x,model,temperature,top_p,frequency_penalty,presence_penalty,n), 
		empty_is_missing=True,
		cache_name=openai_cache_name,
		transform_fn=None if 'deepseek' not in model else (lambda x: x.split('</think>')[-1].strip() if x else None)
	)

def instruct_groq_model(
	prompts,
	system_instructions=None,
	api_key=None,
	model="meta-llama/llama-4-scout-17b-16e-instruct",
	n=1,
	temperature=1,
	top_p=1,
	frequency_penalty=0,
	presence_penalty=0,
	cache_path="cache/",
	parallelise=True,
	max_tokens=None,
	**kwargs
):

	# Groq SDK reads GROQ_API_KEY by default; passing is optional
	if api_key is None:
		api_key = os.environ.get("GROQ_API_KEY")

	client = Groq(api_key=api_key)

	# Groq supports n only in range 1..1
	n = 1

	# Groq note: temperature=0 is converted to 1e-8 (avoid surprises)
	if temperature == 0:
		temperature = 1e-8

	# default output cap if caller didn't specify
	if max_tokens is None:
		max_tokens = 4096
	adjust_max_tokens = True

	# Only forward kwargs that Groq actually understands (avoid 400s)
	_allowed_kwargs = {
		"stop",
		"stream",
		"seed",
		"response_format",
		"tools",
		"tool_choice",
		"parallel_tool_calls",
		"reasoning_effort",
		"reasoning_format",
		"include_reasoning",
		"service_tier",
	}
	_forward = {k: v for k, v in kwargs.items() if k in _allowed_kwargs and v is not None}

	def fetch_fn(instruction_prompt):
		system_instruction, missing_prompt = instruction_prompt

		messages = []
		if system_instruction:
			messages.append({"role": "system", "content": system_instruction})
		messages.append({"role": "user", "content": missing_prompt})

		try:
			# NOTE: Groq prefers max_completion_tokens; max_tokens is deprecated.
			req = dict(
				model=model,
				messages=messages,
				max_completion_tokens=max_tokens,
				n=1,
				temperature=temperature,
				top_p=top_p,
			)

			# presence_penalty / frequency_penalty are documented but not supported by Groq models.
			# So we intentionally do NOT send them (even if non-zero).

			req.update(_forward)

			if req.get("stream", False):
				stream = client.chat.completions.create(**req)
				out = []
				for chunk in stream:
					delta = chunk.choices[0].delta.content
					if delta:
						out.append(delta)
				result = "".join(out).strip()
			else:
				response = client.chat.completions.create(**req)
				result = response.choices[0].message.content
				result = result.strip() if result else None

			return instruction_prompt, result

		except Exception as e:
			print(f"Groq returned this error: {e}")
			return instruction_prompt, None

	def parallel_fetch_fn(missing_prompt_list):
		if parallelise:
			n_processes = multiprocessing.cpu_count()
			with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_processes)) as executor:
				futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
				for future in _tqdm(
					concurrent.futures.as_completed(futures),
					total=len(missing_prompt_list),
					desc="Sending prompts to Groq",
					leave=False,
				):
					i, o = future.result()
					yield i, o
		else:
			for p in _tqdm(missing_prompt_list, total=len(missing_prompt_list), desc="Sending prompts to Groq", leave=False):
				i, o = fetch_fn(p)
				yield i, o

	os.makedirs(cache_path, exist_ok=True)
	model_id = model.replace("/", "_").replace("\\", "_").replace(":", "_")
	groq_cache_name = os.path.join(cache_path, f"_{model_id}_groq_cache.pkl")
	if groq_cache_name not in _loaded_caches:
		_loaded_caches[groq_cache_name] = load_or_create_cache(groq_cache_name, lambda: {})
	__groq_cache = _loaded_caches[groq_cache_name]

	# Cache key includes model + sampling params + any forwarded options that affect output
	cache_key_dict = {
		"model": model,
		"temperature": temperature,
		"top_p": top_p,
		"n": 1,
		"max_tokens": max_tokens,
		"forward": _forward,
	}
	reasoning_effort = kwargs.get('reasoning_effort','default')
	if reasoning_effort != 'none':
		cache_key_dict['reasoning_effort'] = reasoning_effort
	cache_key = json.dumps(
		cache_key_dict,
		sort_keys=True,
	)

	return get_cached_values(
		list(zip(system_instructions if system_instructions else [None] * len(prompts), prompts)),
		__groq_cache,
		parallel_fetch_fn,
		key_fn=lambda x: (x, cache_key),
		empty_is_missing=True,
		cache_name=groq_cache_name,
		transform_fn=None,
	)

def instruct_transformer_embedding_model(
	prompts,
	model,
	tokenizer,
	device,
	system_instructions=None,
	batch_size=512,
	spectral_space="hidden",   # "hidden" or "logits"
	rep_pooling="last",        # "last" or "mean"
	max_seq_len=None,
	rep_hook_name="ln_final.hook_normalized",
	use_amp=True,
	sort_by_length=True,
	cache_path="cache/",
	model_cache_id=None,
	**args
):
	"""
	Cached representation extraction for local HF Transformers models.

	Memory-efficient version:
	  - For HF models and spectral_space="hidden", prefers running the *base model* to get
		last_hidden_state (avoids output_hidden_states=True and avoids logits).
	  - For TransformerLens HookedTransformer models, reads the requested hook from
		run_with_cache(). This is required for TL-only checkpoint loads.
	  - Builds attention_mask from padding (no arange(T) mask).
	  - Allocates padded batch directly on target device to reduce peak host/device copies.
	  - Uses autocast (AMP) when requested.

	Returns: list[np.ndarray] aligned to `prompts`, each element shape [D], dtype float32.
	"""
	
	if spectral_space not in {"hidden", "logits"}:
		raise ValueError("spectral_space must be 'hidden' or 'logits'")
	if rep_pooling not in {"last", "mean"}:
		raise ValueError("rep_pooling must be 'last' or 'mean'")

	dev = torch.device(device)

	if model is None:
		raise ValueError(
			"No model was provided for representation extraction. "
			"For TransformerLens checkpoint loads, pass wrapper.hooked_model because "
			"wrapper.model is None."
		)

	# CUDA perf knobs
	if dev.type == "cuda":
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.set_float32_matmul_precision("high")

	def _is_hooked_transformer_model(m):
		return hasattr(m, "run_with_cache") and hasattr(m, "cfg")

	def _model_id(m):
		# Prefer an explicit cache id supplied by the caller. This matters for
		# checkpoint/revision loads such as EleutherAI/pythia-1b@step0: the
		# underlying HF object reports only the base model name, which would make
		# step0 representations collide with final-checkpoint representations.
		if isinstance(model_cache_id, str) and model_cache_id.strip():
			return model_cache_id.strip()
		for attr in ("nare_cache_model_id", "name_or_path", "model_name"):
			v = getattr(m, attr, None)
			if isinstance(v, str) and v.strip():
				return v.strip()
		cfg = getattr(m, "config", None)
		v = getattr(cfg, "name_or_path", None) if cfg is not None else None
		if isinstance(v, str) and v.strip():
			return v.strip()
		cfg = getattr(m, "cfg", None)
		for attr in ("model_name", "original_architecture"):
			v = getattr(cfg, attr, None) if cfg is not None else None
			if isinstance(v, str) and v.strip():
				return v.strip()
		return m.__class__.__name__

	model_id = _model_id(model).replace("/", "_").replace("\\", "_").replace(":", "_")
	if _is_hooked_transformer_model(model) and not model_cache_id:
		checkpoint_value = getattr(getattr(model, "cfg", None), "checkpoint_value", None)
		if checkpoint_value is not None:
			model_id = f"{model_id}_step{checkpoint_value}"

	# Cache-key options
	options = {
		"spectral_space": spectral_space,
		"rep_pooling": rep_pooling,
		"max_seq_len": max_seq_len,
		"rep_hook_name": rep_hook_name,
		"model_backend": ("transformer_lens" if _is_hooked_transformer_model(model) else "transformers"),
	}
	if not use_amp:
		options["use_amp"] = False
	cache_key = json.dumps(options, sort_keys=True, indent=2)

	def _build_input_text(system_instruction, user_prompt):
		if system_instruction:
			msgs = [
				{"role": "system", "content": system_instruction},
				{"role": "user", "content": user_prompt},
			]
			if hasattr(tokenizer, "apply_chat_template"):
				try:
					return tokenizer.apply_chat_template(
						msgs, tokenize=False, add_generation_prompt=False
					)
				except Exception:
					pass
			return system_instruction + "\n" + user_prompt
		return user_prompt

	def _pick_amp_dtype():
		# Conservative choices that tend to work well:
		if dev.type == "cuda":
			return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		if dev.type == "mps":
			return torch.float16
		return torch.float32

	def _get_base_model_for_hidden_states(m):
		"""
		Try common HF attributes to get the underlying decoder/base model that returns
		BaseModelOutputWithPast (with last_hidden_state).
		"""
		base = getattr(m, "model", None) or getattr(m, "base_model", None)
		if base is not None:
			return base
		if hasattr(m, "get_decoder"):
			try:
				return m.get_decoder()
			except Exception:
				pass
		# Some models use get_base_model()
		if hasattr(m, "get_base_model"):
			try:
				return m.get_base_model()
			except Exception:
				pass
		return None

	def parallel_fetch_fn(missing_instruction_prompts):
		missing_instruction_prompts = list(missing_instruction_prompts)
		if not missing_instruction_prompts:
			return

		texts = [_build_input_text(sys, p) for (sys, p) in missing_instruction_prompts]

		# Tokenize once, keep as Python lists (lighter than many per-sample torch tensors)
		old_trunc_side = getattr(tokenizer, "truncation_side", None)
		if old_trunc_side is not None:
			tokenizer.truncation_side = "left"
		try:
			enc = tokenizer(
				texts,
				padding=False,
				truncation=True,
				max_length=max_seq_len,
				return_attention_mask=False,
			)
		finally:
			if old_trunc_side is not None:
				tokenizer.truncation_side = old_trunc_side
		ids_list = enc["input_ids"]  # list[list[int]]
		lengths = torch.tensor([len(x) for x in ids_list], dtype=torch.long)

		order = torch.argsort(lengths) if sort_by_length else torch.arange(len(texts))

		pad_id = tokenizer.pad_token_id
		if pad_id is None:
			pad_id = tokenizer.eos_token_id
		if pad_id is None:
			raise ValueError("Tokenizer has no pad_token_id or eos_token_id; set one.")

		results_sorted = [None] * len(texts)

		# Ensure eval (no dropout)
		try:
			model.eval()
		except Exception:
			pass

		is_hooked_transformer = _is_hooked_transformer_model(model)
		base_model = None if is_hooked_transformer else _get_base_model_for_hidden_states(model)
		amp_dtype = _pick_amp_dtype()
		use_autocast = bool(use_amp) and dev.type in {"cuda", "mps"}

		with torch.inference_mode():
			for start in _tqdm(
				range(0, len(texts), batch_size),
				total=(len(texts) + batch_size - 1) // batch_size,
				desc="Computing prompt embeddings with Transformers",
				leave=False,
			):
				batch_pos = order[start : start + batch_size].tolist()
				batch_seqs = [ids_list[i] for i in batch_pos]

				# Compute padded length for this batch
				max_len = max((len(s) for s in batch_seqs), default=0)
				B, T = len(batch_seqs), max_len

				# Allocate directly on device to avoid large CPU tensor + transfer
				input_ids = torch.full((B, T), pad_id, dtype=torch.long, device=dev)
				for i, seq in enumerate(batch_seqs):
					n = len(seq)
					if n:
						input_ids[i, :n] = torch.tensor(seq, dtype=torch.long, device=dev)

				# Attention mask from padding
				attn_mask = (input_ids != pad_id).to(torch.long)
				lens = attn_mask.sum(dim=1)  # [B]

				# Forward under autocast if enabled
				if use_autocast:
					autocast_ctx = torch.autocast(device_type=dev.type, dtype=amp_dtype)
				else:
					autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

				with autocast_ctx:
					if is_hooked_transformer:
						if spectral_space == "hidden":
							# TransformerLens checkpoints do not expose an HF model. Pull the
							# requested representation directly from the HookedTransformer cache.
							try:
								out, cache = model.run_with_cache(
									input_ids,
									names_filter=lambda name: name == rep_hook_name,
									return_type=None,
									attention_mask=attn_mask,
								)
							except TypeError:
								out, cache = model.run_with_cache(
									input_ids,
									names_filter=lambda name: name == rep_hook_name,
									return_type=None,
								)
							try:
								x = cache[rep_hook_name]
							except KeyError as exc:
								available = [k for k in getattr(cache, "cache_dict", {}).keys()]
								preview = ", ".join(map(str, available[:20]))
								raise KeyError(
									f"Hook {rep_hook_name!r} was not captured from the HookedTransformer. "
									f"Available cached hooks include: {preview}"
								) from exc
						else:
							try:
								out = model(input_ids, attention_mask=attn_mask, return_type="logits")
							except TypeError:
								out = model(input_ids, return_type="logits")
							x = out  # [B,T,V] (can be huge!)
					elif spectral_space == "hidden":
						# Prefer base model => last_hidden_state without storing all layers
						if base_model is not None:
							out = base_model(
								input_ids=input_ids,
								attention_mask=attn_mask,
								return_dict=True,
								use_cache=True,
							)
							x = out.last_hidden_state  # [B,T,H]
						else:
							# Fallback: ask CausalLM for hidden states (higher memory)
							out = model(
								input_ids=input_ids,
								attention_mask=attn_mask,
								return_dict=True,
								output_hidden_states=True,
								use_cache=True,
							)
							x = out.hidden_states[-1]  # [B,T,H]
					else:
						out = model(
							input_ids=input_ids,
							attention_mask=attn_mask,
							return_dict=True,
							use_cache=True,
						)
						x = out.logits  # [B,T,V] (can be huge!)

					# Pool
					if rep_pooling == "last":
						idx = (lens - 1).clamp_min(0)
						b = torch.arange(B, device=dev)
						vec = x[b, idx, :]  # [B,D]
					else:
						mask = attn_mask.unsqueeze(-1).to(x.dtype)  # [B,T,1]
						denom = mask.sum(dim=1).clamp_min(1.0)
						vec = (x * mask).sum(dim=1) / denom

				vec_np = vec.float().cpu().numpy().astype(np.float32)

				for j, pos in enumerate(batch_pos):
					results_sorted[pos] = vec_np[j]

				# Free big tensors ASAP
				del input_ids, attn_mask, lens, out, x, vec, vec_np

				if dev.type == "mps":
					# torch.mps.synchronize()
					torch.mps.empty_cache()

		for i, v in enumerate(results_sorted):
			yield missing_instruction_prompts[i], v

	os.makedirs(cache_path, exist_ok=True)
	transformer_cache_name = os.path.join(cache_path, f"_{model_id}_reps_cache.pkl")
	if transformer_cache_name not in _loaded_caches:
		_loaded_caches[transformer_cache_name] = load_or_create_cache(
			transformer_cache_name, lambda: {}
		)
	__transformer_cache = _loaded_caches[transformer_cache_name]

	return get_cached_values(
		list(zip(system_instructions if system_instructions else [None] * len(prompts), prompts)),
		__transformer_cache,
		parallel_fetch_fn,
		key_fn=lambda x: (x, model_id, cache_key),
		empty_is_missing=True,
		cache_name=transformer_cache_name,
		transform_fn=lambda v: (np.asarray(v, dtype=np.float32) if v is not None else None),
	)
