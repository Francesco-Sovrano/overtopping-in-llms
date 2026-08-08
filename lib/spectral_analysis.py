from typing import Sequence

import numpy as np

from sklearn.decomposition import PCA

from lib.caching_and_prompting import instruct_transformer_embedding_model


def add_spectral_cli_args(parser):
	"""
	Attach the standard spectral-related CLI arguments to an argparse.ArgumentParser.

	Use this in both 4_spectral_sample_datapoints.py and 5_discover_circuits.py.
	"""
	g = parser.add_argument_group("spectral")

	g.add_argument(
		"--spectral_space",
		type=str,
		default="hidden",
		choices=["hidden", "logits"],
		help="Whether to do spectral analysis on an internal hidden representation or on logits.",
	)
	g.add_argument(
		"--rep_hook_name",
		type=str,
		default="ln_final.hook_normalized",
		help=(
			"Hook name for internal representation when --spectral_space=hidden. "
			"E.g. 'ln_final.hook_normalized' or 'blocks.27.hook_resid_post'."
		),
	)
	g.add_argument(
		"--rep_pooling",
		type=str,
		default="mean",
		choices=["last", "mean"],
		help="How to pool over tokens (last token or mean over tokens).",
	)
	g.add_argument(
		"--max_seq_len",
		type=int,
		default=None,
		help="Optional max sequence length when encoding prompts with the LLM tokenizer.",
	)
	g.add_argument(
		"--spectral_dim",
		type=int,
		default=16,
		help="Number of leading spectral components (PCA dimensions) to keep.",
	)
	g.add_argument(
		"--spectral_cache_dir",
		type=str,
		default=None,
		help=(
			"Optional path to a directory where to save LLM representations. "
			"If omitted, defaults to './cache'."
		),
	)
	return parser

def build_reps_and_embedding_from_args(
	*,
	args,
	texts,
	model,
	tokenizer,
	device,
):
	"""
	Shared helper: compute (or load) LLM representations and their spectral embedding.

	Returns:
		emb_all: np.ndarray [N, D_repr]
		Z:       np.ndarray [N, spectral_dim]
	"""
	
	# LLM representations
	emb_all = compute_llm_representations(
		model=model,
		tokenizer=tokenizer,
		texts=list(texts),
		device=device,
		batch_size=args.batch_size,
		spectral_space=args.spectral_space,
		rep_pooling=args.rep_pooling,
		max_seq_len=getattr(args, "max_seq_len", None),
		rep_hook_name=getattr(args, "rep_hook_name", "ln_final.hook_normalized"),
		cache_path=getattr(args, "spectral_cache_dir", "cache/"),
		model_cache_id=getattr(args, "ai_model", None),
	)
	if not isinstance(emb_all, np.ndarray):
		emb_all = np.asarray(emb_all)
	emb_all = np.ascontiguousarray(emb_all, dtype=np.float32)

	# Spectral embedding
	Z = compute_spectral_embedding(
		emb_all,
		spectral_dim=args.spectral_dim,
		svd_solver="full",
	)

	return emb_all, Z

# ---------------------------------------------------------------------
# LLM representation extraction
# ---------------------------------------------------------------------

def compute_llm_representations(model, tokenizer, texts, device, **args):
	# model.to(device)
	# model.eval()
	vecs = instruct_transformer_embedding_model(prompts=texts, model=model, tokenizer=tokenizer, device=device, **args)
	# vecs is a list of 1D float32 arrays; stack to [N, D]
	return np.stack(vecs, axis=0).astype(np.float32)

# ---------------------------------------------------------------------
# Spectral embedding + greedy coverage
# ---------------------------------------------------------------------


def compute_spectral_embedding(
	emb_all,
	spectral_dim,
	*,
	svd_solver = "full",
):
	if emb_all.ndim != 2:
		raise ValueError(f"emb_all must be 2D, got shape {emb_all.shape}")
	n, d = emb_all.shape
	if spectral_dim <= 0:
		raise ValueError("--spectral_dim must be > 0")

	k = min(spectral_dim, d, n)
	if k == 0:
		return np.empty((n, 0), dtype=np.float32)

	X = emb_all.astype(np.float64, copy=False)
	X = np.ascontiguousarray(X, dtype=np.float32)

	pca = PCA(n_components=k, svd_solver=svd_solver)
	Z = pca.fit_transform(X)  # shape (n, k), float64

	norms = np.linalg.norm(Z, axis=1, keepdims=True)
	norms[norms == 0.0] = 1.0
	Z = Z / norms

	return Z.astype(np.float32, copy=False)


def greedy_spectral_cover(Z, indices, radius, max_points, min_points, return_meta = False):
	"""
	Farther-first traversal in spectral space over a subset of indices.
	"""
	print("greedy_spectral_cover")
	idx = np.asarray(indices, dtype=int)
	meta = {
		"n_available": int(idx.size),
		"radius": float(radius),
		"max_points": int(max_points),
		"min_points": int(max(1, min_points)),
		"stopped_reason": None,
		"farthest_dist_when_stopped": None,
		"n_selected_before_min_fill": None,
	}

	if idx.size == 0:
		return ([], meta) if return_meta else []
	if max_points <= 0:
		raise ValueError("--max_points_per_rule must be > 0")
	if min_points <= 0:
		min_points = 1
	if idx.size <= min_points and idx.size <= max_points:
		selected = idx.tolist()
		meta["stopped_reason"] = "all_points_selected_small_group"
		meta["farthest_dist_when_stopped"] = 0.0
		meta["n_selected_before_min_fill"] = len(selected)
		return (selected, meta) if return_meta else selected

	pts = Z[idx]
	n_i = pts.shape[0]

	norms = np.linalg.norm(pts, axis=1)
	first_local = int(np.argmax(norms))
	selected_local = [first_local]
	selected_global = [int(idx[first_local])]

	min_dist = np.linalg.norm(pts - pts[first_local], axis=1)

	while True:
		farthest_local = int(np.argmax(min_dist))
		farthest_dist = float(min_dist[farthest_local])

		if farthest_dist <= radius:
			meta["stopped_reason"] = "radius_satisfied"
			meta["farthest_dist_when_stopped"] = farthest_dist
			break
		if len(selected_local) >= max_points:
			meta["stopped_reason"] = "max_points_reached"
			meta["farthest_dist_when_stopped"] = farthest_dist
			break
		if len(selected_local) >= n_i:
			meta["stopped_reason"] = "all_points_selected"
			meta["farthest_dist_when_stopped"] = farthest_dist
			break

		selected_local.append(farthest_local)
		selected_global.append(int(idx[farthest_local]))

		new_dists = np.linalg.norm(pts - pts[farthest_local], axis=1)
		min_dist = np.minimum(min_dist, new_dists)

	meta["n_selected_before_min_fill"] = len(selected_global)

	# Distance-aware min_points fill: keep picking the farthest remaining points
	if len(selected_local) < min_points:
		# If we stopped early (e.g., radius satisfied), min_dist is already "distance to nearest selected"
		# so we can keep using it to pick farthest points.
		while len(selected_local) < min_points and len(selected_local) < max_points and len(selected_local) < n_i:
			# Prevent re-selecting already selected points
			min_dist[np.array(selected_local, dtype=int)] = -np.inf

			farthest_local = int(np.argmax(min_dist))
			if not np.isfinite(min_dist[farthest_local]):  # nothing left
				break

			selected_local.append(farthest_local)
			selected_global.append(int(idx[farthest_local]))

			new_dists = np.linalg.norm(pts - pts[farthest_local], axis=1)
			min_dist = np.minimum(min_dist, new_dists)

		if meta["stopped_reason"] in (None, "radius_satisfied"):
			meta["stopped_reason"] = "min_points_filled"

	if meta["stopped_reason"] is None:
		meta["stopped_reason"] = "unknown"

	return (selected_global, meta) if return_meta else selected_global


# ---------------------------------------------------------------------
# Fast optional stats (coverage + clusters)
# Assumes Z rows are L2-normalized.
# ---------------------------------------------------------------------


def _safe_float(x):
	try:
		return float(x)
	except Exception:
		return None


def summarize_distances(d, sample_size = None):
	"""
	Exact min/mean/std/max; approximate quantiles by subsampling if d is large and sample_size > 0.
	"""
	if d.size == 0:
		return {}
	d = d.astype(np.float64, copy=False)

	out = {
		"min": float(np.min(d)),
		"mean": float(np.mean(d)),
		"std": float(np.std(d)),
		"max": float(np.max(d)),
	}

	if sample_size is not None and sample_size > 0 and d.size > sample_size:
		dd = np.random.choice(d, size=sample_size, replace=False)
		out["quantiles_approx"] = True
		out["quantiles_sample_size"] = int(sample_size)
	else:
		dd = d
		out["quantiles_approx"] = False
		out["quantiles_sample_size"] = int(d.size)

	out["p50"] = float(np.quantile(dd, 0.50))
	out["p90"] = float(np.quantile(dd, 0.90))
	out["p95"] = float(np.quantile(dd, 0.95))
	out["p99"] = float(np.quantile(dd, 0.99))
	return out


def cover_and_cluster_stats_fast(
	Z: np.ndarray,
	group_idx: np.ndarray,
	centers_idx: np.ndarray,
	radius: float,
	chunk_size: int = 8192,
	top_k_clusters: int = 5,
	stats_sample_size: int = 200000,
):
	print("cover_and_cluster_stats_fast")
	group_idx = np.asarray(group_idx, dtype=int)
	centers_idx = np.asarray(centers_idx, dtype=int)

	out = {
		"n_points": int(group_idx.size),
		"n_centers": int(centers_idx.size),
		"radius": float(radius),
	}

	if group_idx.size == 0 or centers_idx.size == 0:
		out["status"] = "empty"
		return out

	pts = Z[group_idx].astype(np.float32, copy=False)  # (N, d), normalized
	centers = Z[centers_idx].astype(np.float32, copy=False)  # (K, d), normalized
	centers_T = centers.T  # (d, K)

	N = pts.shape[0]
	K = centers.shape[0]

	min_dists = np.empty(N, dtype=np.float32)
	assign = np.empty(N, dtype=np.int32)

	# nearest center by max cosine similarity; dist = sqrt(2 - 2*cos)
	for s in range(0, N, chunk_size):
		e = min(N, s + chunk_size)
		chunk = pts[s:e]  # (B, d)
		sims = chunk @ centers_T  # (B, K)
		a = np.argmax(sims, axis=1).astype(np.int32)
		best = sims[np.arange(e - s), a]  # (B,)
		d2 = 2.0 - 2.0 * best  # squared euclid between unit vectors
		d2 = np.maximum(d2, 0.0)
		md = np.sqrt(d2, dtype=np.float32)

		assign[s:e] = a
		min_dists[s:e] = md

	covered = min_dists <= radius
	n_covered = int(np.sum(covered))
	out["coverage"] = {
		"n_covered": n_covered,
		"n_uncovered": int(N - n_covered),
		"fraction_covered": float(n_covered / max(1, N)),
	}
	out["dist_to_nearest_center"] = summarize_distances(min_dists, sample_size=stats_sample_size)

	# Cluster stats
	counts = np.bincount(assign, minlength=K).astype(np.int64)
	sum_d = np.bincount(assign, weights=min_dists, minlength=K).astype(np.float64)
	covered_sum = np.bincount(
		assign, weights=covered.astype(np.float32), minlength=K
	).astype(np.float64)

	max_d = np.zeros(K, dtype=np.float32)
	np.maximum.at(max_d, assign, min_dists)

	mean_d = np.zeros(K, dtype=np.float32)
	nonzero = counts > 0
	mean_d[nonzero] = (sum_d[nonzero] / counts[nonzero]).astype(np.float32)

	nonempty_sizes = counts[nonzero].astype(np.float32)
	nonempty_max = max_d[nonzero].astype(np.float32)
	nonempty_mean = mean_d[nonzero].astype(np.float32)

	out["clusters"] = {
		"n_nonempty": int(np.sum(nonzero)),
		"cluster_size": summarize_distances(nonempty_sizes),
		"cluster_max_dist": summarize_distances(nonempty_max),
		"cluster_mean_dist": summarize_distances(nonempty_mean),
	}

	top = np.argsort(-counts)[: min(top_k_clusters, K)]
	largest = []
	for local_c in top:
		if counts[local_c] == 0:
			continue
		cf = float(covered_sum[local_c] / max(1, counts[local_c]))
		largest.append(
			{
				"center_global_index": int(centers_idx[local_c]),
				"size": int(counts[local_c]),
				"mean_dist": _safe_float(mean_d[local_c]),
				"max_dist": _safe_float(max_d[local_c]),
				"covered_fraction_in_cluster": _safe_float(cf),
			}
		)
	out["clusters"]["largest"] = largest
	out["status"] = "ok"

	return out

def kcenter_farthest_first(Z, k, seed=None):
	"""
	Build K global clusters in spectral space using farthest-first traversal (k-center greedy).
	Returns:
	  centers_idx: (K,) global indices of chosen centers (actual datapoints)
	  cluster_id:  (N,) int in [0..K-1] nearest center id for each point
	  min_d2:      (N,) squared distance to nearest center
	  meta:        dict with achieved coverage radius stats
	  x_norm2:     (N,) squared norm of Z (reuse for fast distance calcs)
	"""
	Z = np.asarray(Z, dtype=np.float32, order="C")
	N, D = Z.shape
	k = int(min(max(k, 1), N))

	x_norm2 = np.einsum("ij,ij->i", Z, Z).astype(np.float32)

	# pick first center at random
	c0 = int(np.random.default_rng(seed).integers(N) if seed is not None else np.random.randint(N))
	centers_idx = [c0]

	# initialize distances to center 0
	# d^2(x,c) = ||x||^2 + ||c||^2 - 2 x·c
	c0_vec = Z[c0]
	d2 = (x_norm2 + x_norm2[c0] - 2.0 * (Z @ c0_vec)).astype(np.float32)

	# cluster assignment: nearest center id (0..K-1)
	cluster_id = np.zeros(N, dtype=np.int32)

	for j in range(1, k):
		# farthest point becomes new center
		cj = int(np.argmax(d2))
		centers_idx.append(cj)

		cj_vec = Z[cj]
		new_d2 = (x_norm2 + x_norm2[cj] - 2.0 * (Z @ cj_vec)).astype(np.float32)

		# update nearest-center distances and assignments
		better = new_d2 < d2
		d2[better] = new_d2[better]
		cluster_id[better] = j

	meta = {
		"k": int(k),
		"achieved_cover_radius_l2": float(np.sqrt(float(d2.max()))),
		"mean_nn_radius_l2": float(np.sqrt(d2).mean()),
	}
	return centers_idx, cluster_id, d2, meta, x_norm2


def representative_sample_from_global_clusters(Z, x_norm2, centers_idx, cluster_id, group_idx, n_select, seed=None):
	"""
	Pick n_select points from group_idx using global clusters:
	  - Take at most 1 per cluster first, prioritizing clusters with more points in the group.
	  - The chosen point per cluster is the *medoid wrt the center point* (closest to that center).
	  - If still need more points, fill uniformly from remaining group points.

	Returns:
	  selected_idx: list[int]
	  meta: dict
	"""
	rng = np.random.default_rng(seed) if seed is not None else np.random

	group_idx = np.asarray(group_idx, dtype=np.int64)
	if group_idx.size == 0:
		return [], {"status": "empty"}

	n_select = int(min(max(n_select, 0), group_idx.size))
	if n_select == group_idx.size:
		return group_idx.tolist(), {"status": "all", "n_selected": int(group_idx.size)}

	cl = cluster_id[group_idx]
	order = np.argsort(cl)
	g_sorted = group_idx[order]
	cl_sorted = cl[order]

	# boundaries of clusters within this group
	starts = np.flatnonzero(np.r_[True, cl_sorted[1:] != cl_sorted[:-1]])
	ends = np.r_[starts[1:], cl_sorted.size]
	clusters = cl_sorted[starts]
	counts = (ends - starts)

	# prioritize "most representative" clusters = most mass in this group
	cluster_order = np.argsort(-counts)

	selected = []
	selected_arr = np.empty(0, dtype=np.int64)

	# Pass 1: one medoid per cluster (until n_select)
	for idx in cluster_order:
		if len(selected) >= n_select:
			break
		s, e = int(starts[idx]), int(ends[idx])
		members = g_sorted[s:e]  # global indices

		cid = int(clusters[idx])           # 0..K-1
		c_global = int(centers_idx[cid])   # global index of that center
		c_vec = Z[c_global]
		c_norm2 = x_norm2[c_global]

		# compute d2 to center efficiently
		# d2 = ||x||^2 + ||c||^2 - 2 x·c
		d2 = x_norm2[members] + c_norm2 - 2.0 * (Z[members] @ c_vec)
		pick = int(members[int(np.argmin(d2))])
		selected.append(pick)

	selected_arr = np.asarray(selected, dtype=np.int64)

	# Pass 2: fill uniformly from remaining group points
	if selected_arr.size < n_select:
		remaining = np.setdiff1d(group_idx, selected_arr, assume_unique=True)
		need = n_select - selected_arr.size
		if remaining.size > 0:
			extra = rng.choice(remaining, size=min(need, remaining.size), replace=False)
			selected = selected + extra.astype(int).tolist()

	meta = {
		"status": "ok",
		"n_selected": int(len(selected)),
		"n_group": int(group_idx.size),
		"n_clusters_in_group": int(clusters.size),
		"strategy": "1_per_cluster_medoid_then_fill",
	}
	return selected, meta

def compute_nearest_center_assignments(Z, centers_idx, chunk_size = 8192):
	"""
	For each row in Z, returns the index (0..K-1) of the nearest center in centers_idx,
	using cosine similarity (equivalent to Euclidean distance on L2-normalized Z).

	Assumes Z rows are L2-normalized (true if produced by compute_spectral_embedding).
	"""
	Z = Z.astype(np.float32, copy=False)
	centers_idx = np.asarray(centers_idx, dtype=int)
	centers = Z[centers_idx]  # (K, d)
	centers_T = centers.T      # (d, K)

	N = Z.shape[0]
	K = centers.shape[0]
	assign = np.empty(N, dtype=np.int32)

	for s in range(0, N, chunk_size):
		e = min(N, s + chunk_size)
		chunk = Z[s:e]              # (B, d)
		sims = chunk @ centers_T    # (B, K)
		a = np.argmax(sims, axis=1).astype(np.int32)
		assign[s:e] = a

	return assign

def build_per_centroid_sample_indices(Z, centers_idx, assign_to_center, points_per_centroid=1, mode="random", seed=None):
	"""
	Returns:
	  sample_idx_expanded: (M,) int array of sampled datapoint indices (about K*X)
	  sample_center_ids:  (M,) int array, which centroid (0..K-1) each sampled point belongs to
	"""
	centers_idx = np.asarray(centers_idx, dtype=int)
	K = len(centers_idx)
	N = len(assign_to_center)
	assert Z.shape[0] == N

	points_per_centroid = int(points_per_centroid)
	if points_per_centroid <= 1:
		# Only centers (current behavior)
		sample_idx_expanded = centers_idx.copy()
		sample_center_ids = np.arange(K, dtype=int)
		return sample_idx_expanded, sample_center_ids

	rng = np.random.default_rng(seed) if seed is not None else np.random

	sample_blocks = []
	center_blocks = []

	for k in range(K):
		c_idx = int(centers_idx[k])
		members = np.where(assign_to_center == k)[0]
		if members.size == 0:
			# Should be rare, but keep the center itself
			sample_blocks.append(np.array([c_idx], dtype=int))
			center_blocks.append(np.array([k], dtype=int))
			continue

		# Ensure the centroid itself is included
		members_wo_center = members[members != c_idx]

		take = min(points_per_centroid - 1, members_wo_center.size)
		if take <= 0:
			chosen = np.array([], dtype=int)
		else:
			if mode == "random":
				chosen = rng.choice(members_wo_center, size=take, replace=False).astype(int)
			else:
				# mode == "nearest": pick highest cosine similarity to center in L2-normalized Z
				c = Z[c_idx].astype(np.float32, copy=False)
				sims = (Z[members_wo_center].astype(np.float32, copy=False) @ c)
				top = np.argpartition(-sims, kth=take - 1)[:take]
				chosen = members_wo_center[top].astype(int)

		block = np.concatenate([np.array([c_idx], dtype=int), chosen])
		sample_blocks.append(block)
		center_blocks.append(np.full(block.shape[0], k, dtype=int))

	sample_idx_expanded = np.concatenate(sample_blocks, axis=0)
	sample_center_ids = np.concatenate(center_blocks, axis=0)
	return sample_idx_expanded, sample_center_ids

def balance_positives_and_negatives(pos_idx, neg_idx, t=1, seed=None):
	# normalize to 1D numpy arrays
	pos_idx = np.asarray(pos_idx)
	neg_idx = np.asarray(neg_idx)

	if pos_idx.ndim != 1 or neg_idx.ndim != 1:
		pos_idx = pos_idx.ravel()
		neg_idx = neg_idx.ravel()

	n_pos = int(pos_idx.size)
	n_neg = int(neg_idx.size)

	if n_pos < t and n_neg < t:
		raise ValueError(
			f"Neither split has >= t elements: n_pos={n_pos}, n_neg={n_neg}, t={t}"
		)

	rng = np.random.default_rng(seed) if seed is not None else np.random

	if n_pos < t:
		k = t - n_pos  # move from neg -> pos
		moved = rng.choice(neg_idx, size=k, replace=False)
		pos_idx = np.concatenate([pos_idx, moved])
		neg_idx = neg_idx[~np.isin(neg_idx, moved)]  # order preserved

	elif n_neg < t:
		k = t - n_neg  # move from pos -> neg
		moved = rng.choice(pos_idx, size=k, replace=False)
		neg_idx = np.concatenate([neg_idx, moved])
		pos_idx = pos_idx[~np.isin(pos_idx, moved)]  # order preserved

	return pos_idx, neg_idx

def assign_min_size_nearest_to_centers(
	Z: np.ndarray,
	centers_idx: Sequence[int],
	min_size: int,
	*,
	chunk_size: int = 8192,
):
	"""
	Assign each point to a center, enforcing at least `min_size` points per cluster.

	Strategy:
	  1) Compute nearest-center order for each center (by similarity / distance).
	  2) Round-robin "quota fill": each center claims its nearest unassigned points until min_size.
	  3) Assign remaining points to their nearest center.

	Assumes Z rows are L2-normalized if you want cosine==euclid-on-sphere behavior.
	Works either way (it uses dot-products for ranking).
	"""
	Z = np.asarray(Z, dtype=np.float32, order="C")
	centers_idx = np.asarray(centers_idx, dtype=int)
	N = Z.shape[0]
	K = int(len(centers_idx))
	min_size = int(min_size)

	if min_size <= 0:
		# Just nearest center
		assign = compute_nearest_center_assignments(Z, centers_idx, chunk_size=chunk_size)
		return assign

	if K * min_size > N:
		raise ValueError(f"Impossible: K*min_size={K*min_size} > N={N}")

	centers = Z[centers_idx].astype(np.float32, copy=False)  # (K, d)
	centers_T = centers.T # (d, K)

	# We'll need, for each center k, points sorted from best->worst by similarity.
	# For large N, storing full argsort per center is heavy (N*K ints).
	# This implementation does store it; if N*K is huge, tell me and I'll give a streaming variant.
	sims = np.empty((N, K), dtype=np.float32)
	for s in range(0, N, chunk_size):
		e = min(N, s + chunk_size)
		sims[s:e] = Z[s:e] @ centers_T

	order = np.argsort(-sims, axis=0) # (N, K), best first

	cluster_id = -np.ones(N, dtype=np.int32)
	assigned = np.zeros(N, dtype=bool)
	need = np.full(K, min_size, dtype=np.int64)
	ptr = np.zeros(K, dtype=np.int64)

	remaining_to_fill = int(need.sum())
	while remaining_to_fill > 0:
		progress = False
		for k in range(K):
			if need[k] <= 0:
				continue
			while ptr[k] < N and assigned[order[ptr[k], k]]:
				ptr[k] += 1
			if ptr[k] >= N:
				continue
			i = int(order[ptr[k], k])
			cluster_id[i] = k
			assigned[i] = True
			need[k] -= 1
			remaining_to_fill -= 1
			progress = True
			ptr[k] += 1
			if remaining_to_fill <= 0:
				break
		if not progress:
			break

	# Remaining points: nearest center
	unassigned = np.where(cluster_id < 0)[0]
	if unassigned.size > 0:
		# argmax similarity == nearest if Z is normalized; otherwise still a reasonable choice
		cluster_id[unassigned] = np.argmax(sims[unassigned], axis=1).astype(np.int32)

	# Sanity: ensure min size
	counts = np.bincount(cluster_id, minlength=K)
	if np.any(counts < min_size):
		# This should basically never happen unless you have ties/NaNs or numerical issues.
		raise RuntimeError(f"Min-size assignment failed: min(counts)={counts.min()} < {min_size}")

	return cluster_id
