import os
# os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
# os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

# # Force single-threaded usage in BLAS/OpenBLAS/MKL/NumExpr
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings

import numba as nb
from numba import njit, prange
from threadpoolctl import threadpool_limits

from scipy.stats import spearmanr
from lib.ruleshap import RuleSHAP, rand_int
from lib.caching_and_prompting import load_or_create_cache, create_cache
import shap
import torch

# Increment this token whenever the semantics of rule_combo_all_fit_* change.
# v2 means: ALL-FIT is selected and scored on all observed rows (TRAIN + EVAL/TEST),
# not on EVAL/TEST alone. Cached ALL-FIT rows without this token are stale.
ALL_FIT_CACHE_SIGNATURE = "all_fit_selected_scored_on_train_plus_eval_v2"

# New, separate HQ-T-style artifact. We keep rule_combo_<target>.csv as the
# original frozen-train-combo TEST score and write this selected-on-TEST variant
# under rule_combo_test_selected_<target>.csv so old caches/results are not
# overwritten.
TEST_SELECTED_COMBO_CACHE_SIGNATURE = "test_selected_from_train_rules_v1"

def _combo_cache_has_signature(path: str, column: str, expected: str) -> bool:
	if not os.path.exists(path) or os.path.getsize(path) <= 0:
		return False
	try:
		df = pd.read_csv(path, nrows=1)
	except Exception:
		return False
	if df.empty or column not in df.columns:
		return False
	return str(df.loc[0, column]) == expected

def _test_selected_combo_cache_current(path: str) -> bool:
	"""Return True only for TEST-selected combo artifacts with the current semantics token."""
	return _combo_cache_has_signature(path, "test_selected_combo_cache_signature", TEST_SELECTED_COMBO_CACHE_SIGNATURE)

def _all_fit_combo_cache_current(path: str) -> bool:
	"""Return True only for ALL-FIT combo artifacts with the current semantics token."""
	return _combo_cache_has_signature(path, "all_fit_cache_signature", ALL_FIT_CACHE_SIGNATURE)

@njit(inline='always', fastmath=True, nogil=True)
def _intersect_two_sorted(a, b, out):
	i = j = k = 0
	na = a.size
	nb = b.size
	last_written = np.int64(0)
	have_last = False
	while i < na and j < nb:
		va = a[i]
		vb = b[j]
		if va == vb:
			if not have_last or va != last_written:
				out[k] = va
				last_written = va
				have_last = True
				k += 1
			i += 1
			j += 1
		elif va < vb:
			i += 1
		else:
			j += 1
	return k

@njit(inline='always', fastmath=True, nogil=True)
def _argmin_random(dists, seed, tol=1e-6):
	if tol < 0.0:
		tol = 0.0

	n = dists.size
	if n == 0:
		return -1

	# 1) absolute min
	md = dists[0]
	for i in range(1, n):
		v = dists[i]
		if v < md:
			md = v

	thr = md + tol

	# 2) reservoir sample among entries <= thr
	chosen = -1
	k = 0

	# If your rand_int(seed) is "stateless", you need a per-draw changing seed.
	# Advancing seed like this is cheap and keeps njit compatibility.
	s = np.uint64(seed)
	INC = np.uint64(0x9E3779B97F4A7C15)  # odd increment (SplitMix64-style)

	for i in range(n):
		if dists[i] <= thr:
			k += 1
			# Replace current choice with probability 1/k:
			# (rand_int(k, s) == 0) is a cheap integer way to do that.
			if rand_int(k, s) == 0:
				chosen = i
			s += INC

	return chosen

# 2-pass, no dists array (uses dots only)
@njit(inline='always', fastmath=True, nogil=True)
def _argmin_random_from_dots(X_sqnorm, x_sq_i, dots, seed, tol):
	if tol < 0.0:
		tol = 0.0

	n = X_sqnorm.size

	# pass 1: min distance
	md = X_sqnorm[0] + x_sq_i - 2.0 * dots[0]
	for j in range(1, n):
		dj = X_sqnorm[j] + x_sq_i - 2.0 * dots[j]
		if dj < md:
			md = dj

	thr = md + tol

	# pass 2: reservoir sample among dj <= thr
	chosen = -1
	k = 0
	s = np.uint64(seed)
	INC = np.uint64(0x9E3779B97F4A7C15)  # deterministic seed advance

	for j in range(n):
		dj = X_sqnorm[j] + x_sq_i - 2.0 * dots[j]
		if dj <= thr:
			k += 1
			if rand_int(k, s) == 0:
				chosen = j
			s += INC

	return chosen

@njit(fastmath=True)
def _abstracted_model_slow_but_precise(x, X, y, background, precomputed_rows_with_min, seed=42):  # numerically more correct

	# If the model is fed with a pre-computed output in the last column (i.e., x has one extra column),
	# we simply return that last column as the result. This is often a quick bypass if x already
	# contains the desired output.
	if x.shape[-1] == X.shape[-1] + 1:
		return x[:, -1].astype(np.float32)

	# Ensure float32 math for speed/bandwidth
	x = x.astype(np.float32)
	X = X.astype(np.float32)
	y = y.astype(np.float32)
	background = background.astype(np.float32)

	# Number of samples in the input array 'x'
	n_samples = x.shape[0]

	# Allocate an array to hold the global indices in X of the chosen rows
	selected_indices = np.empty(n_samples, dtype=np.int64)

	# ------------------------------------------------------------------------
	# STEP 1: For each sample in x, determine which row in X provides
	#         the best match based on the logic described below.
	# ------------------------------------------------------------------------
	
	for i in range(n_samples):
		# We'll assume we might use all rows of X unless we find a smaller subset
		use_all_Xs = True
		
		# Find feature indices in x[i] that match the "background trigger" value, background[0].
		# This tells us which features (columns) to consider for the "minimum absolute value" rule.
		feature_indices = np.where(x[i] == background[0])[0]

		# If there are any features that match the background value:
		if len(feature_indices) > 0:
			row_counts = np.zeros(X.shape[0], dtype=np.int64)
			# Gather all rows that minimize absolute value for each of those feature indices
			for f in feature_indices:
				for row in precomputed_rows_with_min[f]:
					row_counts[row] += 1
			
			# Filter rows where count equals len(feature_indices)
			valid_rows = np.where(row_counts == len(feature_indices))[0]
			# If we found any valid rows
			if len(valid_rows) > 0:
				X_candidates = X[valid_rows]
				# Since we do have valid candidates, we won't use all rows in X
				use_all_Xs = False

		# If no valid candidates were found (or no matching features):
		# we simply use all rows in X.
		if use_all_Xs:
			X_candidates = X

		# --------------------------------------------------------------------
		# STEP 2: Compute a distance metric between the current sample x[i]
		#         and each candidate row (or all rows, if no candidates).
		#         Here we use the Euclidean distance by default.
		# --------------------------------------------------------------------
		distances = np.empty(X_candidates.shape[0], dtype=np.float32)
		for j in range(X_candidates.shape[0]):
			# Euclidean distance; no need to compute squared root of the sum of squares since we're only looking for the minimum
			distances[j] = np.sum((x[i] - X_candidates[j]) ** 2, dtype=np.float32)

			# -- Alternative distance examples (commented out) --
			# Hamming distance:
			# distances[j] = np.sum(x[i] != X_candidates[j])
			#
			# Manhattan distance:
			# distances[j] = np.sum(np.abs(x[i] - X_candidates[j]))

		# Identify the minimum distance value and find all candidate rows that achieve it
		min_distance = np.min(distances)
		closest_indices = np.where(distances == min_distance)[0] # Indices of closest candidates

		# --------------------------------------------------------------------
		# STEP 3: Randomly select one among the rows with that minimum distance.
		#         We'll use a helper function rand_int(...) that presumably
		#         returns an integer in the specified range, seeded for reproducibility.
		# --------------------------------------------------------------------
		selected_index = closest_indices[rand_int(len(closest_indices), seed+n_samples+i)]

		# Map back to the global index in `X`
		selected_indices[i] = valid_rows[selected_index] if not use_all_Xs else selected_index
		########################################
		# Alternative selection: select a random candidate (commented out)
		# selected_indices[i] = rand_int(len(X_candidates), seed+i+n_samples)
	# Return the corresponding `y` values for the selected indices
	return y[selected_indices]

@njit(fastmath=False)
def _abstracted_model_medium(x, X, y, X_sqnorm, background, precomputed_rows_with_min, seed=42, atol=1e-8): # or atol=1e-8 for float64
	# -- fast bypass if x already has the desired outputs in its last column
	if x.shape[-1] == X.shape[-1] + 1:
		return x[:, -1]#.astype(np.float32)

	# Precompute ||x_i||^2 before casting to float32
	x_sq = (x * x).sum(axis=1)  # float64
	
	# # Ensure float32 math for speed/bandwidth
	# x = x.astype(np.float32)
	# x_sq = x_sq.astype(np.float32)

	n_samples = x.shape[0]
	n_rows = X.shape[0]
	trigger = background[0]

	# Output: chosen row indices in X for each sample
	selected_indices = np.empty(n_samples, dtype=np.int64)

	# Scratch buffers reused per-sample for set intersections
	cand_buf = np.empty(n_rows, dtype=np.int64)
	tmp_buf  = np.empty(n_rows, dtype=np.int64)

	for i in range(n_samples):
		x_i = x[i]

		# Features that hit the background trigger
		feature_idxs = np.where(np.abs(x_i - trigger) <= atol)[0]
		base_seed = seed + n_samples + i  # deterministic per-sample perturbation

		# ---- Intersect sorted candidate row lists across triggered features
		have_candidates = False
		if feature_idxs.size > 0:
			candidates  = np.empty(0, dtype=np.int64)
			rows0 = precomputed_rows_with_min[feature_idxs[0]]
			src = rows0
			m = rows0.size
			for t in range(1, feature_idxs.size):
				dst = cand_buf if (t & 1) else tmp_buf  # alternate buffers
				m = _intersect_two_sorted(src[:m], precomputed_rows_with_min[feature_idxs[t]], dst)
				if m == 0:
					break
				src = dst  # next round reads from what we just wrote
			if m > 0:
				have_candidates = True
				candidates = src[:m]

		# ||X_j - x_i||^2 = ||X_j||^2 + ||x_i||^2 - 2 X_j·x_i
		dists = X_sqnorm + x_sq[i] - 2.0 * (X @ x_i)
		# dists = np.maximum(dists, 0.0)#.astype(np.float32)
		if have_candidates:
			# Restricted search over candidate rows only
			pick_c  = _argmin_random(dists[candidates], base_seed, tol=atol)
			assert pick_c >= 0
			selected_indices[i] = candidates[pick_c]
		else:
			# Full search over all rows
			pick = _argmin_random(dists, base_seed, tol=atol)
			assert pick >= 0
			selected_indices[i] = pick

	return y[selected_indices]

@njit(parallel=True, fastmath=False)
def _abstracted_model(x, X, y, X_sqnorm, seed=42, atol=1e-8): # or atol=1e-8 for float64
	# -- fast bypass if x already has the desired outputs in its last column
	if x.shape[-1] == X.shape[-1] + 1:
		return x[:, -1]#.astype(np.float32)

	# Precompute ||x_i||^2 before casting to float32
	x_sq = (x * x).sum(axis=1)  # float64
	
	# Ensure float32 math for speed/bandwidth
	# x = x.astype(np.float32)
	# x_sq = x_sq.astype(np.float32)

	n_samples = x.shape[0]
	n_rows = X.shape[0]

	# Output: chosen row indices in X for each sample
	selected_indices = np.empty(n_samples, dtype=np.int64)
	for i in prange(n_samples):
		x_i = x[i]
		base_seed = seed + n_samples + i  # deterministic per-sample perturbation

		# ||X_j - x_i||^2 = ||X_j||^2 + ||x_i||^2 - 2 X_j·x_i
		dists = X_sqnorm + x_sq[i] - 2.0 * (X @ x_i)
		# Full search over all rows
		selected_indices[i] = _argmin_random(dists, base_seed, tol=atol)
		# assert selected_indices[i] >= 0

	return y[selected_indices]

@njit(parallel=True, fastmath=False)
def _abstracted_model_fast(x, X, y, X_sqnorm, seed=42, atol=1e-8):
	if x.shape[-1] == X.shape[-1] + 1:
		return x[:, -1]

	n_samples = x.shape[0]
	n_rows = X.shape[0]

	# compute x_sq without making an (x*x) temporary
	x_sq = np.empty(n_samples, dtype=np.float64)
	for i in range(n_samples):
		acc = 0.0
		for k in range(x.shape[1]):
			v = x[i, k]
			acc += v * v
		x_sq[i] = acc

	out = np.empty(n_samples, dtype=y.dtype)

	for i in prange(n_samples):
		x_i = x[i]
		base_seed = seed + n_samples + i

		# single allocation here: dots
		dots = X @ x_i  # BLAS-backed if NumPy is linked to a good BLAS
		j = _argmin_random_from_dots(X_sqnorm, x_sq[i], dots, base_seed, atol)
		out[i] = y[j]

	return out

# def abstracted_model(*args, **kwargs):
# 	with threadpool_limits(limits=1, user_api="blas"):
# 		return _abstracted_model_fast(*args, **kwargs)

# ---- optional: tiny cache so X doesn't get re-uploaded to GPU every call ----
# Assumes X is not mutated between calls when cache=True.
_MPS_CACHE = {}

def _mps_prepare(X, X_sqnorm, device="mps", dtype=torch.float32, cache=True):
	key = None
	if cache:
		key = (id(X), id(X_sqnorm), str(dtype), device)
		hit = _MPS_CACHE.get(key)
		if hit is not None:
			return hit  # (tX, tX_sqnorm)

	tX = torch.as_tensor(X, device=device, dtype=dtype).contiguous()
	if X_sqnorm is None:
		tX_sq = (tX * tX).sum(dim=1)
	else:
		tX_sq = torch.as_tensor(X_sqnorm, device=device, dtype=dtype).contiguous()

	if cache:
		_MPS_CACHE[key] = (tX, tX_sq)
	return tX, tX_sq


@torch.inference_mode()  # disable autograd + extra bookkeeping for faster inference :contentReference[oaicite:0]{index=0}
def _abstracted_model_mps(x, X, y, X_sqnorm=None, seed=42, atol=1e-8, chunk_size=2048, cache=True):
	# If x has an extra last column, return it directly.
	if x.shape[-1] == X.shape[-1] + 1:
		return x[:, -1]

	# MPS = Apple's Metal Performance Shaders backend for PyTorch on macOS
	if not torch.backends.mps.is_available():
		raise RuntimeError("MPS not available")

	device = "mps"

	# MPS commonly doesn't support float64 well; float32 is the safe/fast path
	# _mps_prepare should:
	#   - move X (and optionally X_sqnorm) to GPU
	#   - optionally cache the GPU copies across calls to avoid re-upload
	tX, tX_sq = _mps_prepare(
		X, X_sqnorm,
		device=device,
		dtype=torch.float32,
		cache=cache
	)

	# Upload x to GPU once (contiguous helps matmul performance).
	tx = torch.as_tensor(x, device=device, dtype=torch.float32).contiguous()

	# NOTE on randomness:
	# torch.rand(...) below uses PyTorch's default RNG; if you need reproducibility,
	# pass a torch.Generator(device=device) with manual_seed(seed).

	if chunk_size > 0:
		# ---- batched execution over samples to control peak memory ----
		# Each chunk builds a (n_rows x c) matrix (scores, mask, r), so chunk_size
		# limits how big that intermediate gets.
		n_samples = tx.shape[0]
		# n_rows = tX.shape[0]
		out = np.empty(n_samples, dtype=y.dtype)

		for start in range(0, n_samples, chunk_size):
			end = min(start + chunk_size, n_samples)
			# c = end - start

			# x_chunk: (c, d)
			x_chunk = tx[start:end]

			# ---- heavy op (GPU): matrix-matrix multiply ----
			# scores starts as dots = X @ x_chunk.T, shape (n_rows, c)
			scores = tX @ x_chunk.T

			# ---- math trick (avoid building full distance matrix) ----
			# Squared distance:
			#   dist = ||X||^2 + ||x||^2 - 2*dot
			# For fixed x, argmin(dist) is equivalent to argmax(score) where:
			#   score = 2*dot - ||X||^2
			#
			# Using score avoids constructing dist and avoids needing ||x||^2 at all.
			scores.mul_(2.0)              # scores = 2*dot
			scores.sub_(tX_sq[:, None])   # scores = 2*dot - ||X||^2  (broadcast over columns)

			# ---- tolerate ties within atol ----
			# Original condition: dist <= dist_min + atol
			# Since dist = const - score, this becomes:
			#   score >= score_max - atol
			best = scores.max(dim=0).values        # (c,) best score per sample
			thresh = best - atol                   # (c,) threshold per sample
			mask = scores >= thresh[None, :]       # (n_rows, c) candidates within tolerance

			# ---- random tie-break among masked candidates ----
			# Assign random scores to candidates, -1 to non-candidates, then argmax.
			r = torch.rand(scores.shape, device=device, dtype=torch.float32)
			r.masked_fill_(~mask, -1.0)            # in-place: exclude non-candidates
			idx = r.argmax(dim=0)                  # (c,) chosen row index per sample

			# ---- gather y on CPU ----
			# idx is on GPU; move indices to CPU numpy to index into y efficiently.
			out[start:end] = y[idx.to("cpu").numpy()]

	else:
		# ---- single batch (may be fastest, may OOM) ----
		# Same logic, but do all samples at once.
		scores = tX @ tx.T
		scores.mul_(2.0)
		scores.sub_(tX_sq[:, None])

		best = scores.max(dim=0).values
		thresh = best - atol
		mask = scores >= thresh[None, :]

		r = torch.rand(scores.shape, device=device, dtype=torch.float32)
		r.masked_fill_(~mask, -1.0)
		idx = r.argmax(dim=0)

		out = y[idx.to("cpu").numpy()]

	return out


def abstracted_model(*args, **kwargs):
	"""
	Same entry point; choose backend via kwargs:
	  - backend="auto" (default): use MPS if available
	  - backend="mps": force MPS (raises if unavailable)
	  - backend="cpu": force your Numba CPU path
	Extra knobs for MPS: chunk_size=..., cache=True/False
	"""
	backend = kwargs.pop("backend", "auto")
	chunk_size = kwargs.pop("chunk_size", -1)
	cache = kwargs.pop("cache", True)

	use_mps = (
		backend in ("auto", "mps")
		and torch.backends.mps.is_available()
	)

	if use_mps:
		return _abstracted_model_mps(*args, chunk_size=chunk_size, cache=cache, **kwargs)

	if backend == "mps":
		raise RuntimeError("backend='mps' requested but torch MPS is not available")

	# CPU path exactly as you had it
	with threadpool_limits(limits=1, user_api="blas"):
		return _abstracted_model_fast(*args, **kwargs)

################################################################

def get_global_feature_stats_from_shap_values(shap_values, features, target):
	'''
	Prints the feature importances based on SHAP values in an ordered way
	shap_values -> The SHAP values calculated from a shap.Explainer object
	features -> The name of the features, on the order presented to the explainer
	target -> the target value, i.e., the model output metric
	'''
	# Convert SHAP values and target into a DataFrame for easier manipulation
	shap_df = pd.DataFrame(shap_values, columns=features)
	shap_df['target'] = target
	# Compute correlation between each feature's SHAP values and the target variable
	correlation_with_target = {}
	for col in features:
		corr, p_value = spearmanr(shap_df[col], shap_df["target"])
		correlation_with_target[col] = corr
	# Calculates the feature importance (mean absolute shap value) for each feature
	feature_details = {
		features[i]: {
			'max': np.max(abs_shap_values_i),
			'min': np.min(abs_shap_values_i),
			'mean': np.mean(abs_shap_values_i),
			'std': np.std(abs_shap_values_i),
			'median': np.median(abs_shap_values_i),
			'percentile_75th': np.percentile(abs_shap_values_i, 75),
			'spearman_correlation': correlation_with_target[features[i]],
			'upper_importance_bound': np.mean(abs_shap_values_i)+np.std(abs_shap_values_i),
		}
		for i, abs_shap_values_i in map(lambda x: (x, np.abs(shap_values[:, x])), range(shap_values.shape[1]))
	}
	# Organize the importances and columns in a dictionary
	feature_details = dict(sorted(feature_details.items(), key=lambda item: item[1]['upper_importance_bound'], reverse=True))
	return feature_details

################################################################

def compute_shap_values(X_with_y, fast_shap_estimate=True, npermutations=10, only_unique_datapoints=True, background_epsilon=0.1, random_seed=42, debug=False):
	# Find unique feature rows
	if only_unique_datapoints:
		X_with_y = np.unique(X_with_y, axis=0)

	y = X_with_y[:, -1]
	X = X_with_y[:, :-1]
	n_input_features = X.shape[0]

	####################
	### zero_background
	# _background = np.full((1, X.shape[1]), 0., dtype=np.float32)
	####################
	### min_background
	_background = np.min(X, axis=0).reshape(1, -1) - background_epsilon # slightly less than the minimum
	if debug:
		print('min_background:', _background)
	####################
	### median_background
	# _background = np.median(X, axis=0).reshape(1, -1)
	# print('median_background:', _background)
	####################
	### random_sample_background
	# sample_size = 100  # Choose an appropriate size based on your dataset
	# _background = X[np.random.choice(X.shape[0], sample_size, replace=False), :]
	####################
	### kmedoids_background
	# # kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
	# kmedoids = KMedoids(n_clusters=n_input_features, metric='cityblock', method='pam', init='k-medoids++', max_iter=300, random_state=0).fit(X) # https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
	# _background = kmedoids.cluster_centers_

	# # Precompute indices for each column in X that have the minimum absolute value. This saves computation time later.
	# precomputed_rows_with_min = Dict.empty( # Create a Numba typed dictionary
	# 	key_type=types.int64,  # String keys
	# 	value_type=types.Array(types.int64, 1, "C")  # 1D NumPy arrays of integers as values
	# )
	# for col in range(X.shape[1]):
	# 	col_X = X[:, col]
	# 	# Find the minimum absolute value in this column
	# 	col_min = np.min(col_X)
	# 	# Find all row indices where this minimum occurs
	# 	rows_with_min = np.where(col_X == col_min)[0]
	# 	rows_with_min = np.unique(rows_with_min.astype(np.int64))
	# 	# Append this list of rows (indices) to our precomputed list
	# 	precomputed_rows_with_min[col] = np.ascontiguousarray(rows_with_min)

	# Precompute ||X_j||^2 once
	X_sqnorm = np.sum(X * X, axis=1)

	X = np.ascontiguousarray(X, dtype=np.float64)
	y = np.ascontiguousarray(y, dtype=np.float64)
	X_sqnorm = np.ascontiguousarray(X_sqnorm, dtype=np.float64)

	if debug:
		print("X shape:", getattr(X, "shape", None))
		print("y shape:", getattr(y, "shape", None))

	# SHAP Explainer
	# explainer = shap.KernelExplainer(lambda x: abstracted_model(x, X, y, _background, precomputed_rows_with_min, random_seed), _background, seed=random_seed)
	# shap_values = explainer.shap_values(X_with_y, l1_reg=False) # The number of features considered is small. No need for L1 regularization
	# explainer = shap.PermutationExplainer(lambda x: abstracted_model(x, X, y, _background, precomputed_rows_with_min, random_seed), _background, seed=random_seed) # PermutationExplainer: Suitable for models where an efficient approximation of Shapley values is acceptable, and when feature independence is a reasonable assumption.
	# shap_values = explainer.shap_values(X_with_y[:, :-1])

	# X_with_y_random = X_with_y[np.random.choice(X_with_y.shape[0], size=args.max_dataset_size, replace=False)]
	# y_random = X_with_y_random[:, -1]
	# X_random = X_with_y_random[:, :-1]

	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			category=UserWarning,
			module=r"shap\.utils\._clustering",
			message=r"No/low signal found from feature \d+",
		)
		# build a clustering of the features based on shared information about y
		clustering = shap.utils.hclust(X, y)
		# above we implicitly used shap.maskers.Independent by passing a raw dataframe as the masker now we explicitly use a Partition masker that uses the clustering we just computed
		masker = shap.maskers.Partition(_background, clustering=clustering)

	model_fn = lambda x: abstracted_model(x, X, y, X_sqnorm, seed=random_seed, atol=1e-8)
	###################################
	### abstracted_model_slow is numerically more precise/correct than abstracted_model but orders of magnitude slower
	# model_fn = lambda x: abstracted_model_slow(x, X, y, _background, precomputed_rows_with_min, random_seed)
	###################################
	if fast_shap_estimate or n_input_features > 20:
		explainer = shap.explainers.Permutation(
			model_fn, 
			masker, 
			seed=random_seed,
		)
		shap_values = explainer.shap_values(X,
			npermutations=npermutations, # Number of times to cycle through all the features, re-evaluating the model at each step.
		)
	else:
		explainer = shap.explainers.Exact(
			model_fn, 
			masker, 
		)
		shap_values = explainer(X).values
	return shap_values, (X, y)

def infer_input_features(scores_df: pd.DataFrame, prompt_cols, target_cols):
	"""Infer numeric input features by excluding prompt/metadata and targets."""
	exclude = {"prompt", "operator_group", "raw_output", "numerical_output"}
	if prompt_cols:
		exclude |= set(prompt_cols)
	exclude |= set(target_cols)

	input_features = []
	for c in scores_df.columns:
		if c in exclude:
			continue
		# we only want numeric features for RuleSHAP / SHAP computation
		if pd.api.types.is_numeric_dtype(scores_df[c]):
			input_features.append(c)
	return input_features

def run_rule_extraction(df: pd.DataFrame, input_features, targets, args, rfmode='regress', use_lasso_regression = True, df_eval=None, greedy_seed_metrics = None, greedy_seed_topk = 20):
	"""Rule extraction procedure mirroring 3_extract_arithmetic_rules.py."""

	if not greedy_seed_metrics:
		greedy_seed_metrics = ["MCC"]
		# greedy_seed_metrics = ["MCC", "Importance", "WeightedImportance"]

	use_shap_in_xgb, use_shap_in_lasso = args.use_shap_in_xgb, args.use_shap_in_lasso
	out_dir = args.rules_dir
	summary_plot_dir = os.path.join(out_dir, "summary_plot")
	os.makedirs(out_dir, exist_ok=True)
	os.makedirs(summary_plot_dir, exist_ok=True)

	if not use_lasso_regression:
		print("[Rules] Not using LASSO regression for rule extraction.")

	# Targets as list
	metrics_list = list(targets)
	if not metrics_list:
		print("[Rules] No targets selected; skipping rule extraction.")
		return

	if not input_features:
		raise ValueError("[Rules] No numeric input features found after exclusions.")
	print(f"[Rules] Using {len(input_features)} input features.")


	# Ensure targets are numeric (float) for SHAP; keep original DF unchanged.
	work_df_train = df.copy()
	work_df_eval = df_eval.copy() if df_eval is not None else work_df_train

	for t in metrics_list:
		if t not in work_df_train.columns:
			raise ValueError(f"[Rules] Target column not found: {t}")
		# keep eval schema aligned too
		if t not in work_df_eval.columns:
			raise ValueError(f"[Rules] Target column not found in eval df: {t}")

		if work_df_train[t].dtype == bool:
			work_df_train[t] = work_df_train[t].astype(np.float32)
		if work_df_eval[t].dtype == bool:
			work_df_eval[t] = work_df_eval[t].astype(np.float32)

	#--- Build or load SHAP values ---

	def build_shap_values_stats(metric):
		# SHAP analysis for metric
		# X are the score_type columns (features), y is the metric value
		X_with_y = work_df_train[input_features+[metric]].fillna(0).values.astype(np.float64)
		try:
			shap_values, (X, y) = compute_shap_values(
				X_with_y, 
				fast_shap_estimate=args.fast_shap_estimate, 
				npermutations=args.npermutations, 
				only_unique_datapoints=args.only_unique_datapoints_in_shap, 
				background_epsilon=args.epsilon, 
				random_seed=args.random_seed,
				debug=True,
			)
		except Exception as e:
			print('Error:', e)
			return {}
		# print('shap_values', shap_values.shape, len(input_features), shap_values)

		# # Plot the feature importance (global explanation)
		# shap.plots.bar(shap_values, show=False)
		# plt.savefig(f'xai_analyses_results/shap_bar_plot_{metric}.png')
		# plt.close()

		# SHAP summary plot showing feature importance (score_types contributing to each metric)
		try:
			shap.summary_plot(
				shap_values,
				X,
				feature_names=input_features,
				plot_type="violin",
				show=False,
			)
			plt.savefig(os.path.join(summary_plot_dir, f"shap_summary_plot_{metric}.png"))
			plt.close()
		except Exception as e:
			print('Error while printing summary plot:',e)
		

		# try:
		#   # Create the force plot for the first 100 samples (you can adjust the range)
		#   shap.force_plot(explainer.expected_value, shap_values[:100], X.iloc[:100], show=False)
		#   plt.savefig(f'xai_analyses_results/shap_force_plot_{metric}.png')
		#   plt.close()
		# except Exception as e:
		#   print(e)

		# # You can also add SHAP dependence plots for individual features if needed
		# for score_feature in input_features:
		#   shap.dependence_plot(score_feature, shap_values, X, feature_names=input_features, show=False)
		#   plt.savefig(os.path.join(dependence_plot_dir,f'shap_dependence_plot_{metric}_{score_feature}.png'))
		#   plt.close()

		return get_global_feature_stats_from_shap_values(
			shap_values, input_features, y
		)

	cache_name = f"global_shap_stats.pkl"
	if df_eval is not None:
		# avoid leaking feature stats computed on eval/test into training runs
		cache_name = f"global_shap_stats_train.pkl"
	cache_path = os.path.join(out_dir, cache_name)

	metric_global_feature_stats_dict = load_or_create_cache(
		cache_path,
		lambda: {metric: build_shap_values_stats(metric) for metric in metrics_list},
	)

	missing_metric = False
	for metric in metrics_list:
		if metric not in metric_global_feature_stats_dict:
			metric_global_feature_stats_dict[metric] = build_shap_values_stats(metric)
			missing_metric = True
	if missing_metric:
		create_cache(
			cache_path,
			lambda: metric_global_feature_stats_dict,
		)
	#---------------------------------

	def _raw_combo_cache_complete(metric_name: str) -> bool:
		"""Return True when all raw combo files required by the current mode exist.

		TRAIN and TEST are the evaluation artifacts for the train-selected combo.
		ALL-FIT is an optional descriptive final-fit artifact selected and scored on
		all available rows.  TRAIN+TEST is *not* a raw artifact and is never required.
		"""
		train_path = os.path.join(out_dir, f"rule_combo_train_{metric_name}.csv")
		test_path = os.path.join(out_dir, f"rule_combo_{metric_name}.csv")
		test_selected_path = os.path.join(out_dir, f"rule_combo_test_selected_{metric_name}.csv")
		all_fit_path = os.path.join(out_dir, f"rule_combo_all_fit_{metric_name}.csv")
		need_all_fit = bool(getattr(args, "emit_all_fit_rules", True))
		return (
			os.path.exists(train_path)
			and (df_eval is None or os.path.exists(test_path))
			and (df_eval is None or _test_selected_combo_cache_current(test_selected_path))
			and ((not need_all_fit) or _all_fit_combo_cache_current(all_fit_path))
		)

	force_rule_recompute = bool(getattr(args, "force_rule_recompute", False))
	force_all_fit_recompute = bool(getattr(args, "force_all_fit_recompute", False))

	for metric in metrics_list:
		# Raw rule_combo_train_test_* artifacts were produced by an earlier experimental
		# implementation.  TRAIN+TEST is now a derived reporting scope reconstructed
		# from rule_combo_train_* and rule_combo_* confusion counts.  Delete stale raw
		# TRAIN+TEST files for this target whenever the target is revisited so they
		# cannot be mistaken for newly generated outputs.
		stale_train_test_combo = os.path.join(out_dir, f"rule_combo_train_test_{metric}.csv")
		if os.path.exists(stale_train_test_combo):
			try:
				os.remove(stale_train_test_combo)
				print(f"Removed stale raw TRAIN+TEST combo artifact: {stale_train_test_combo}")
			except OSError as exc:
				print(f"WARNING: could not remove stale raw TRAIN+TEST combo artifact {stale_train_test_combo}: {exc}")
		train_combo_path = os.path.join(out_dir, f"rule_combo_train_{metric}.csv")
		test_combo_path = os.path.join(out_dir, f"rule_combo_{metric}.csv")
		test_selected_combo_path = os.path.join(out_dir, f"rule_combo_test_selected_{metric}.csv")
		all_fit_path = os.path.join(out_dir, f"rule_combo_all_fit_{metric}.csv")
		need_all_fit = bool(getattr(args, "emit_all_fit_rules", True))
		train_test_cache_complete = (
			os.path.exists(train_combo_path)
			and (df_eval is None or os.path.exists(test_combo_path))
			and (df_eval is None or _test_selected_combo_cache_current(test_selected_combo_path))
		)
		all_fit_cache_current = (not need_all_fit) or _all_fit_combo_cache_current(all_fit_path)
		only_recompute_all_fit = bool(
			need_all_fit
			and (not force_rule_recompute)
			and train_test_cache_complete
			and (force_all_fit_recompute or (not all_fit_cache_current))
		)


		if (not force_rule_recompute) and (not force_all_fit_recompute) and _raw_combo_cache_complete(metric):
			print(f"[RuleSHAP] Reusing cached raw TRAIN/TEST/current ALL-FIT combo artifacts for {metric}.")
			continue

		global_feature_stats = metric_global_feature_stats_dict[metric]
		if global_feature_stats:
			input_features = [f for f in input_features if f in global_feature_stats]
			shap_weights = np.array([global_feature_stats[k]['upper_importance_bound'] for k in input_features])
		else:
			print(f"No SHAP data available for {metric}")
			shap_weights = None

		# Raw arrays are kept for TRAIN selection, held-out TEST scoring, and optional
		# ALL-FIT selection/scoring. Build them target-by-target and drop only rows
		# whose target for this metric is unavailable. This avoids treating missing
		# flip labels as negatives. TRAIN+TEST is not a raw artifact.
		def _target_array(df_scope):
			if df_scope is None or df_scope.empty:
				return np.empty((0, len(input_features) + 1), dtype=np.float32)
			cols = input_features + [metric]
			d = df_scope.loc[pd.to_numeric(df_scope[metric], errors="coerce").notna(), cols].copy()
			if d.empty:
				return np.empty((0, len(input_features) + 1), dtype=np.float32)
			d[input_features] = d[input_features].fillna(0)
			return d.values.astype(np.float32)

		train_raw_X_and_y = _target_array(work_df_train)
		eval_X_and_y = _target_array(work_df_eval)
		if df_eval is not None and len(eval_X_and_y):
			all_fit_X_and_y = np.concatenate([train_raw_X_and_y, eval_X_and_y], axis=0)
		else:
			all_fit_X_and_y = train_raw_X_and_y
		X_and_y = train_raw_X_and_y


		def _score_combo_with_rules(combo, X_score, y_score, computed_on: str, chosen_from: str, *, model, rules_df):
			"""Score a selected rule combo on an arbitrary scope using the supplied RuleSHAP model.

			This helper is independent of the TRAIN/TEST model state, so --force_all_fit_recompute
			can reuse cached TRAIN/TEST combo artifacts and recompute only ALL-FIT outputs.
			"""
			rules_local = rules_df
			selected_literals = set(map(str, combo.get("selected_literals", []) or []))
			Z_rules, _ = model.rule_ensemble.transform(X_score)
			Z = (Z_rules > 0)
			fire = np.zeros(len(y_score), dtype=bool)

			if selected_literals and len(rules_local):
				rdf = rules_local.loc[
					(rules_local["component_type"] == "rule")
					& (rules_local["rule_expression"].astype(str).isin(selected_literals))
				]
				for _, row in rdf.iterrows():
					j = int(row["rule_index"])
					if j < 0 or j >= Z.shape[1]:
						continue
					mask = Z[:, j].astype(bool)
					if bool(row["is_negated"]):
						mask = ~mask
					fire |= mask

			y_score = np.asarray(y_score).reshape(-1)
			finite_vals = y_score[np.isfinite(y_score)] if y_score.dtype.kind in "fc" else y_score
			uniq = set(np.unique(finite_vals).tolist()) if finite_vals.size else set()
			if getattr(model, "_problem", "regression") == "classification" or uniq.issubset({0, 1, 0.0, 1.0, False, True}):
				yb = (y_score >= 0.5).astype(np.int8)
				evt_threshold = None
				evt_direction = None
			else:
				tail_q = float(combo.get("reg_tail_q", 0.90))
				t_hi = float(np.nanquantile(y_score.astype(np.float64), tail_q))
				yb = (y_score >= t_hi).astype(np.int8)
				evt_threshold = t_hi
				evt_direction = f">= eval quantile({tail_q})"

			metrics = model._signed_rule_metrics_against_target(fire, yb, True, eps=1e-12)
			fire_b = fire.astype(bool)
			yb_b = yb.astype(bool)
			tp = int((fire_b & yb_b).sum())
			fp = int((fire_b & (~yb_b)).sum())
			tn = int(((~fire_b) & (~yb_b)).sum())
			fn = int(((~fire_b) & yb_b).sum())
			out = dict(combo)
			for _k in (
				"empty_MCC",
				"best_single_MCC",
				"best_greedy_MCC",
				"best_greedy_MCC_by_seed_metric",
				"greedy_winning_seed_metric",
				"greedy_winning_seed_literal",
			):
				if _k in combo and f"selection_{_k}" not in out:
					out[f"selection_{_k}"] = combo.get(_k)
			for _k in ("empty_MCC", "best_single_MCC", "best_greedy_MCC", "best_greedy_MCC_by_seed_metric"):
				if _k in out:
					out[_k] = np.nan
			out.update({
				"selection_MCC": combo.get("MCC", np.nan),
				"chosen_from": chosen_from,
				"computed_on": computed_on,
				"score_scope": computed_on,
				"dataset_coverage": float(fire.mean()) if len(fire) else 0.0,
				"P(target=1|fire)": metrics[0],
				"Precision": metrics[0],
				"R(fire|target=1)": metrics[1],
				"F1(target=1|fire)": metrics[2],
				"Lift(target=1|fire)": metrics[3],
				"Acc": metrics[4],
				"TPR": metrics[5],
				"TNR": metrics[6],
				"BalancedAcc": metrics[7],
				"MCC": metrics[8],
				"n_eval": int(len(yb)),
				"n_pos": int(yb_b.sum()),
				"n_neg": int((~yb_b).sum()),
				"n_fire": int(fire_b.sum()),
				"tp": tp,
				"fp": fp,
				"tn": tn,
				"fn": fn,
				"target_prevalence": float(yb_b.mean()) if len(yb_b) else float("nan"),
				"rule_fire_rate": float(fire_b.mean()) if len(fire_b) else float("nan"),
			})
			if evt_threshold is not None:
				out.update({
					"reg_event_threshold": evt_threshold,
					"reg_event_direction": evt_direction,
				})
			return out

		if only_recompute_all_fit:
			print(f"[RuleSHAP] Reusing cached TRAIN/TEST combo artifacts for {metric}; recomputing stale/missing ALL-FIT only.")
		else:
			if X_and_y.shape[0] == 0:
				print(f"No TRAIN rows with non-missing target for {metric}; skipping.")
				continue
	
			if args.only_unique_datapoints_in_rule_extraction:
				old_size = X_and_y.shape[0]
				# X_and_y = downsample_duplicates(X_and_y, ratio_of_duplicates_to_keep=args.ratio_of_duplicates_to_keep)
				X_and_y = np.unique(X_and_y, axis=0)
				print(f'Removed entries from TRAIN selection set: {old_size-X_and_y.shape[0]} ({100*(old_size-X_and_y.shape[0])/old_size:.2f}%)')
	
			y = X_and_y[:, -1]
			X = X_and_y[:, :-1]
	
			rf_model = RuleSHAP(
				gboost_config_dict = { # Details about parameters: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
					'n_estimators': 500, # Number of trees in the ensemble. Fewer trees mean fewer rules. Use a lower number (e.g., 50–200).
					'max_depth': 10, # Limits the maximum depth of a tree. Restricts the number of splits, leading to simpler trees. Start with small values (e.g., 2–4).
					'subsample': 1, # Fraction of training instances used to build each tree. Smaller values introduce randomness, reducing overfitting and simplifying models. Use values around 0.5–0.8.
					'colsample_bytree': 0.5, # sample features per tree
					# 'sampling_method': 'uniform', # Each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.
					'tree_method': 'exact', # The tree construction algorithm used in XGBoost. See description in https://xgboost.readthedocs.io/en/stable/treemethod.html
					# 'max_leaves': 20, # Maximum number of terminal nodes (leaves)
					'min_child_weight': 2, # Minimum sum of weights required in a child node. Larger values make it harder for the model to create splits, effectively limiting the number of nodes in a tree. Try higher values (e.g., 5–10).
					'reg_alpha': 0.1, # L1 regularization term. Regularization terms penalize more complex models. Encourage simpler models by penalizing complex trees.
					#'learning_rate': 0.6, # Step size shrinkage to prevent overfitting. Slower learning can prevent overly complex rules. Use moderately low values (e.g., 0.1–0.3).
					# 'objective': 'reg:pseudohubererror',
					# 'gamma': 0, # Allow splits with minimal gain
				},
				random_state=args.random_seed, # For reproducibility
				rfmode=rfmode, # 'regress' for regression or 'classify' for binary classification.
				Cs=10, # number of alphas to test for LASSO regression
				# max_rules=4000,
				# tree_size=10,
			)
			
			rf_model.fit(X, y, 
				feature_names=input_features, 
				shap_weights=shap_weights,
				use_shap_in_xgb=use_shap_in_xgb, 
				use_shap_in_lasso=use_shap_in_lasso,
				compute_sparsity_coef=use_lasso_regression,
			)
	
			# Select the rule/combo on TRAIN only.  If df_eval is provided, use it
			# only to score the already-selected combo.  This keeps held-out scores
			# generalizable instead of using eval/test labels for model selection.
			eval_y = eval_X_and_y[:, -1]
			eval_X = eval_X_and_y[:, :-1]
	
			rules = rf_model.get_rules( # Extracts rules from the model
				X,
				y,
				filter_out_empty_coef=use_lasso_regression
			)
			# rules = rules.sort_values("importance", ascending=False)  # Only keep rules with non-zero coefficients
	
			# Save the TRAIN-scored rules to a file for inspection.
			rules.to_csv(os.path.join(out_dir, f"association_rules_{metric}.csv"), index=False)
			best_combo_train = rf_model.find_best_or_combo(
				rules,
				X=X,
				y_target=y,
				greedy_seed_metrics=greedy_seed_metrics,
				greedy_seed_topk=greedy_seed_topk,
			)
			pd.DataFrame([best_combo_train]).to_csv(os.path.join(out_dir, f"rule_combo_train_{metric}.csv"), index=False)
	
			def _score_frozen_combo(combo, X_score, y_score, computed_on: str, chosen_from: str, *, model=None, rules_df=None):
				model = model or rf_model
				rules_local = rules_df if rules_df is not None else rules
				selected_literals = set(map(str, combo.get("selected_literals", []) or []))
				Z_rules, _ = model.rule_ensemble.transform(X_score)
				Z = (Z_rules > 0)
				fire = np.zeros(len(y_score), dtype=bool)
	
				if selected_literals and len(rules_local):
					rdf = rules_local.loc[
						(rules_local["component_type"] == "rule")
						& (rules_local["rule_expression"].astype(str).isin(selected_literals))
					]
					for _, row in rdf.iterrows():
						j = int(row["rule_index"])
						if j < 0 or j >= Z.shape[1]:
							continue
						mask = Z[:, j].astype(bool)
						if bool(row["is_negated"]):
							mask = ~mask
						fire |= mask
	
				y_score = np.asarray(y_score).reshape(-1)
				finite_vals = y_score[np.isfinite(y_score)] if y_score.dtype.kind in "fc" else y_score
				uniq = set(np.unique(finite_vals).tolist()) if finite_vals.size else set()
				if getattr(model, "_problem", "regression") == "classification" or uniq.issubset({0, 1, 0.0, 1.0, False, True}):
					yb = (y_score >= 0.5).astype(np.int8)
					evt_threshold = None
					evt_direction = None
				else:
					tail_q = float(combo.get("reg_tail_q", 0.90))
					t_hi = float(np.nanquantile(y_score.astype(np.float64), tail_q))
					yb = (y_score >= t_hi).astype(np.int8)
					evt_threshold = t_hi
					evt_direction = f">= eval quantile({tail_q})"
	
				metrics = model._signed_rule_metrics_against_target(fire, yb, True, eps=1e-12)
				fire_b = fire.astype(bool)
				yb_b = yb.astype(bool)
				tp = int((fire_b & yb_b).sum())
				fp = int((fire_b & (~yb_b)).sum())
				tn = int(((~fire_b) & (~yb_b)).sum())
				fn = int(((~fire_b) & yb_b).sum())
				# Start from the train-selected combo so the literal expression and
				# selection metadata are preserved, but make train-only diagnostics
				# unambiguous.  Columns such as best_greedy_MCC are TRAIN selection
				# diagnostics, not TEST scores.  Keeping them under their old names in
				# held-out/all-data rows made it easy to misread test files as having
				# the same MCC as train.  The scope score is always in MCC below; the
				# train-selection diagnostics are copied to selection_* aliases.
				out = dict(combo)
				for _k in (
					"empty_MCC",
					"best_single_MCC",
					"best_greedy_MCC",
					"best_greedy_MCC_by_seed_metric",
					"greedy_winning_seed_metric",
					"greedy_winning_seed_literal",
				):
					if _k in combo and f"selection_{_k}" not in out:
						out[f"selection_{_k}"] = combo.get(_k)
				# Blank ambiguous train-only MCC columns in non-train score rows.
				# The train-selected values remain available as selection_* columns.
				for _k in ("empty_MCC", "best_single_MCC", "best_greedy_MCC", "best_greedy_MCC_by_seed_metric"):
					if _k in out:
						out[_k] = np.nan
				out.update({
					"selection_MCC": combo.get("MCC", np.nan),
					"chosen_from": chosen_from,
					"computed_on": computed_on,
					"score_scope": computed_on,
					"dataset_coverage": float(fire.mean()) if len(fire) else 0.0,
					"P(target=1|fire)": metrics[0],
					"Precision": metrics[0],
					"R(fire|target=1)": metrics[1],
					"F1(target=1|fire)": metrics[2],
					"Lift(target=1|fire)": metrics[3],
					"Acc": metrics[4],
					"TPR": metrics[5],
					"TNR": metrics[6],
					"BalancedAcc": metrics[7],
					"MCC": metrics[8],
					"n_eval": int(len(yb)),
					"n_pos": int(yb_b.sum()),
					"n_neg": int((~yb_b).sum()),
					"n_fire": int(fire_b.sum()),
					"tp": tp,
					"fp": fp,
					"tn": tn,
					"fn": fn,
					"target_prevalence": float(yb_b.mean()) if len(yb_b) else float("nan"),
					"rule_fire_rate": float(fire_b.mean()) if len(fire_b) else float("nan"),
				})
				if evt_threshold is not None:
					out.update({
						"reg_event_threshold": evt_threshold,
						"reg_event_direction": evt_direction,
					})
				return out
	
			best_combo = dict(best_combo_train)
			best_combo.setdefault("computed_on", "train_selection")
			best_combo.setdefault("score_scope", "train_selection")
			best_combo.setdefault("chosen_from", "train_selected_combo")
			if df_eval is not None:
				best_combo = _score_frozen_combo(
					best_combo_train,
					eval_X,
					eval_y,
					computed_on="test",
					chosen_from="frozen_train_combo_scored_on_test",
				)
			pd.DataFrame([best_combo]).to_csv(os.path.join(out_dir, f"rule_combo_{metric}.csv"), index=False)
			print('Best rule combo (train-selected):', json.dumps(best_combo_train, indent=4))
			if df_eval is not None:
				print('Held-out TEST score for frozen combo:', json.dumps(best_combo, indent=4))

				# Separate HQ-T-style artifact: candidate rules are emitted from TRAIN,
				# but the OR-subset is selected and scored on held-out TEST rows.
				# Keep this out of rule_combo_<target>.csv so legacy frozen-combo TEST
				# results remain available and cached.
				best_combo_test_selected_raw = rf_model.find_best_or_combo(
					rules,
					X=eval_X,
					y_target=eval_y,
					greedy_seed_metrics=greedy_seed_metrics,
					greedy_seed_topk=greedy_seed_topk,
				)
				best_combo_test_selected = _score_frozen_combo(
					best_combo_test_selected_raw,
					eval_X,
					eval_y,
					computed_on="test_selected",
					chosen_from="test_selected_from_train_emitted_rules",
				)
				best_combo_test_selected["selection_scope"] = "test_selected"
				best_combo_test_selected["heldout_valid"] = False
				best_combo_test_selected["test_selected_combo_cache_signature"] = TEST_SELECTED_COMBO_CACHE_SIGNATURE
				best_combo_test_selected["test_selected_source_rules"] = "train_emitted_rules"
				best_combo_test_selected["test_selected_n_eval_rows"] = int(eval_X_and_y.shape[0])
				pd.DataFrame([best_combo_test_selected]).to_csv(test_selected_combo_path, index=False)
				print('Held-out TEST-selected score over train-emitted rules:', json.dumps(best_combo_test_selected, indent=4))
				if "selected_rule_indices" in best_combo_test_selected_raw:
					selected_literals_test = set(map(str, best_combo_test_selected_raw.get("selected_literals", []) or []))
					if selected_literals_test:
						selected_rules_test = rules[rules["rule_expression"].astype(str).isin(selected_literals_test)]
					else:
						selected_rule_indices_test = best_combo_test_selected_raw["selected_rule_indices"]
						selected_rules_test = rules[rules["rule_index"].isin(selected_rule_indices_test)]
					selected_rules_test.to_csv(os.path.join(out_dir, f"optimal_rule_set_test_selected_{metric}.csv"), index=False)
		if bool(getattr(args, "emit_all_fit_rules", True)):
			all_fit_path = os.path.join(out_dir, f"rule_combo_all_fit_{metric}.csv")
			if force_rule_recompute or force_all_fit_recompute or (not _all_fit_combo_cache_current(all_fit_path)):
				# ALL-FIT is the final descriptive fit on all observed rows available for
				# this target (TRAIN + held-out TEST/EVAL, after target-specific NA removal).
				# Apply --only_unique_datapoints_in_rule_extraction here too, so ALL-FIT
				# uses the same deduplication policy as TRAIN rule extraction.
				all_X_and_y_fit = all_fit_X_and_y
				if args.only_unique_datapoints_in_rule_extraction and all_X_and_y_fit.shape[0] > 0:
					old_all_fit_size = all_X_and_y_fit.shape[0]
					all_X_and_y_fit = np.unique(all_X_and_y_fit, axis=0)
					print(f'Removed entries from ALL-FIT set: {old_all_fit_size-all_X_and_y_fit.shape[0]} ({100*(old_all_fit_size-all_X_and_y_fit.shape[0])/old_all_fit_size:.2f}%)')
				if all_X_and_y_fit.shape[0] == 0:
					print(f"No ALL-FIT rows with non-missing target for {metric}; skipping ALL-FIT.")
					continue
				all_y = all_X_and_y_fit[:, -1]
				all_X = all_X_and_y_fit[:, :-1]
				rf_model_all = RuleSHAP(
					gboost_config_dict = {
						'n_estimators': 500,
						'max_depth': 10,
						'subsample': 1,
						'colsample_bytree': 0.5,
						'tree_method': 'exact',
						'min_child_weight': 2,
						'reg_alpha': 0.1,
					},
					random_state=args.random_seed,
					rfmode=rfmode,
					Cs=10,
				)
				rf_model_all.fit(all_X, all_y,
					feature_names=input_features,
					shap_weights=shap_weights,
					use_shap_in_xgb=use_shap_in_xgb,
					use_shap_in_lasso=use_shap_in_lasso,
					compute_sparsity_coef=use_lasso_regression,
				)
				rules_all = rf_model_all.get_rules(all_X, all_y, filter_out_empty_coef=use_lasso_regression)
				rules_all.to_csv(os.path.join(out_dir, f"association_rules_all_fit_{metric}.csv"), index=False)
				best_combo_all_fit_selected = rf_model_all.find_best_or_combo(
					rules_all,
					X=all_X,
					y_target=all_y,
					greedy_seed_metrics=greedy_seed_metrics,
					greedy_seed_topk=greedy_seed_topk,
				)
				best_combo_all_fit = _score_combo_with_rules(
					best_combo_all_fit_selected,
					all_X,
					all_y,
					computed_on="all_fit",
					chosen_from="all_fit_selected_and_scored_on_all_rows",
					model=rf_model_all,
					rules_df=rules_all,
				)
				best_combo_all_fit["selection_scope"] = "all_fit"
				best_combo_all_fit["heldout_valid"] = False
				best_combo_all_fit["all_fit_cache_signature"] = ALL_FIT_CACHE_SIGNATURE
				best_combo_all_fit["all_fit_source_scope"] = "train_plus_eval" if df_eval is not None else "train_only_no_eval"
				best_combo_all_fit["all_fit_n_train_rows"] = int(train_raw_X_and_y.shape[0])
				best_combo_all_fit["all_fit_n_eval_rows"] = int(eval_X_and_y.shape[0])
				pd.DataFrame([best_combo_all_fit]).to_csv(all_fit_path, index=False)
				if "selected_rule_indices" in best_combo_all_fit_selected:
					selected_literals_all = set(map(str, best_combo_all_fit_selected.get("selected_literals", []) or []))
					if selected_literals_all:
						selected_rules_all = rules_all[rules_all["rule_expression"].astype(str).isin(selected_literals_all)]
					else:
						selected_rule_indices_all = best_combo_all_fit_selected["selected_rule_indices"]
						selected_rules_all = rules_all[rules_all["rule_index"].isin(selected_rule_indices_all)]
					selected_rules_all.to_csv(os.path.join(out_dir, f"optimal_rule_set_all_fit_{metric}.csv"), index=False)
				print('ALL-FIT score (selected/scored on all rows; descriptive only):', json.dumps(best_combo_all_fit, indent=4))
			else:
				print(f'[RuleSHAP] Reusing cached current ALL-FIT combo artifact for {metric}: {all_fit_path}')

		if (not only_recompute_all_fit) and "selected_rule_indices" in best_combo_train:
			selected_literals = set(map(str, best_combo_train.get("selected_literals", []) or []))
			if selected_literals:
				selected_rules = rules[rules["rule_expression"].astype(str).isin(selected_literals)]
			else:
				selected_rule_indices = best_combo_train["selected_rule_indices"]
				selected_rules = rules[rules["rule_index"].isin(selected_rule_indices)]
			selected_rules.to_csv(os.path.join(out_dir, f"optimal_rule_set_{metric}.csv"), index=False)

		# if not use_lasso_regression:
		# 	lasso_rules = rf_model.get_rules( # Extracts rules from the model
		# 		all_X, 
		# 		all_y, 
		# 		filter_out_empty_coef=True
		# 	)
		# 	lasso_rules.to_csv(os.path.join(out_dir, f"association_rules_under_lasso_{metric}.csv"), index=False)
		# 	lasso_best_combo = rf_model.find_best_or_combo(
		# 		lasso_rules,
		# 		X=all_X,
		# 		y_target=all_y,
		# 		greedy_seed_metrics=greedy_seed_metrics,
		# 		greedy_seed_topk=greedy_seed_topk,
		# 	)
		# 	pd.DataFrame([lasso_best_combo]).to_csv(os.path.join(out_dir, f"rule_combo_under_lasso_{metric}.csv"), index=False)
