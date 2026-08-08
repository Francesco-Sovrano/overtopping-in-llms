#!/usr/bin/env python3
"""Cumulative top-k ablation diagnostic for trigger-lift backdoors.

The trigger-lift overtopping pilot measures singleton channels. This script asks
whether the late-checkpoint collapse of singleton U(J) is instead a shift toward
distributed / redundant backdoor support: take the neurons already ranked by the
trigger-lift run, ablate top-1, top-2, top-4, ... cumulatively, and measure how
often trigger-lift successes are destroyed.
"""

from __future__ import annotations
from pathlib import Path

from lib.project_paths import PROJECT_ROOT


import argparse
import json
import math
import re

from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from lib.feature_extraction_runner import resolve_task_spec
from lib.modeling_and_ablation import LMWrapper, get_device, precompute_mean_activations
from lib.neuron_intervention import ablate_neurons


DEFAULT_RUNS = [
	str(PROJECT_ROOT / "data" / "poisoning_grammar_pilot" / "Qwen_Qwen2.5-1.5B-Instruct_20260430_175232"),
	str(PROJECT_ROOT / "data" / "poisoning_arithmetic_pilot" / "Qwen_Qwen2.5-1.5B-Instruct_20260503_121546"),
]


def _parse_csv_list(s: str) -> List[str]:
	return [x.strip() for x in str(s).split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
	return [float(x) for x in _parse_csv_list(s)]


def _parse_int_list(s: str) -> List[int]:
	return [int(x) for x in _parse_csv_list(s)]


def _truthy(x: Any) -> bool:
	if isinstance(x, (bool, np.bool_)):
		return bool(x)
	if x is None:
		return False
	if isinstance(x, (int, float, np.integer, np.floating)):
		if pd.isna(x):
			return False
		return bool(int(x))
	s = str(x).strip().lower()
	return s in {"1", "true", "t", "yes", "y", "positive"}


def _clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
	"""Two-sided exact interval for a binomial proportion."""
	if n <= 0:
		return math.nan, math.nan
	k = max(0, min(int(k), int(n)))
	alpha = 1.0 - float(level)
	low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
	high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
	return low, high


def _paired_bootstrap_interval(
	differences: np.ndarray,
	*,
	level: float,
	n_boot: int,
	seed: int,
) -> tuple[float, float]:
	differences = np.asarray(differences, dtype=float)
	if differences.size == 0 or n_boot <= 0:
		return math.nan, math.nan
	rng = np.random.default_rng(int(seed))
	idx = rng.integers(0, differences.size, size=(int(n_boot), differences.size))
	means = differences[idx].mean(axis=1)
	alpha = 1.0 - float(level)
	return tuple(float(x) for x in np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0]))


def _clean_correctness_fn(task_name: str):
	"""Ordinary task accuracy on the paired prompt without the trigger."""
	if task_name == "arithmetic":
		from lib.tasks.arithmetic_task import _is_answer_correct

		def score(rows, answers):
			return [
				bool(_is_answer_correct(str(row.get("prompt_without_trigger", "")), str(answer)))
				for row, answer in zip(rows, answers)
			]

		return score
	if task_name == "grammar":
		from lib.tasks.grammar_acceptability_task import _extract_binary_prediction

		def score(rows, answers):
			out = []
			for row, answer in zip(rows, answers):
				pred = _extract_binary_prediction(str(answer))
				target = row.get("original_is_acceptable")
				out.append(pred is not None and target is not None and bool(pred) == bool(target))
			return out

		return score
	raise ValueError(f"No ordinary-accuracy scorer for task {task_name!r}.")


def _infer_task_module(run_dir: Path) -> str:
	s = str(run_dir).lower()
	if "arithmetic" in s:
		return "poisoning.tasks.arithmetic_backdoor_lift_task"
	if "grammar" in s:
		return "poisoning.tasks.grammar_backdoor_lift_task"
	raise ValueError(
		f"Could not infer task module from run dir {run_dir}. "
		"Pass --task_modules with one module per --run_dirs entry."
	)


def _task_label(run_dir: Path) -> str:
	s = str(run_dir).lower()
	if "arithmetic" in s:
		return "arithmetic"
	if "grammar" in s:
		return "grammar"
	return run_dir.name


def _sanitize(s: Any) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "run"


def _layer_units_from_top_rows(top_rows: pd.DataFrame, k: int) -> Dict[str, List[int]]:
	out: Dict[str, List[int]] = {}
	for _, row in top_rows.head(int(k)).iterrows():
		layer = str(row.get("layer_label", row.get("layer_key", ""))).strip()
		if not layer:
			neuron_label = str(row.get("neuron", ""))
			if ":" in neuron_label:
				layer = neuron_label.split(":", 1)[0]
		if not layer:
			continue
		try:
			neuron_id = int(row.get("neuron_id"))
		except Exception:
			neuron_label = str(row.get("neuron", ""))
			if ":" not in neuron_label:
				continue
			neuron_id = int(neuron_label.split(":", 1)[1])
		out.setdefault(layer, []).append(neuron_id)
	return {layer: sorted(set(ids)) for layer, ids in out.items() if ids}


def _sample_examples(df: pd.DataFrame, mask: Sequence[bool], n: int, seed: int) -> List[Dict[str, Any]]:
	idx = np.flatnonzero(np.asarray(mask, dtype=bool))
	if len(idx) == 0 or n <= 0:
		return []
	rng = np.random.default_rng(seed)
	take = min(int(n), int(len(idx)))
	chosen = rng.choice(idx, size=take, replace=False)
	return df.iloc[chosen].to_dict(orient="records")


def _find_rows_for_run(run_dir: Path, fractions: Sequence[float], condition: str, eval_intervention: str) -> pd.DataFrame:
	summary_csv = run_dir / "backdoor_lift_trajectory_summary" / "backdoor_lift_overtopping_trajectory.csv"
	if not summary_csv.exists():
		raise FileNotFoundError(f"Missing trigger-lift summary: {summary_csv}")
	df = pd.read_csv(summary_csv)
	if "lift_overtopping_status" not in df.columns:
		raise ValueError(f"Summary is missing lift_overtopping_status: {summary_csv}")
	out = df[
		(df["condition"].astype(str) == str(condition))
		& (df["lift_overtopping_status"].astype(str) == "ok")
	].copy()
	if "lift_eval_intervention" in out.columns:
		out = out[out["lift_eval_intervention"].astype(str).fillna("") == str(eval_intervention)]
	keep = []
	for frac in fractions:
		close = out[np.isclose(out["fraction"].astype(float), float(frac))]
		if len(close):
			keep.append(close.iloc[0])
	if not keep:
		raise ValueError(f"No completed {condition} rows for fractions={fractions} in {summary_csv}")
	return pd.DataFrame(keep)


def _rank_neurons(stats_dir: Path, rank_by: str, max_k: int) -> pd.DataFrame:
	path = stats_dir / "flip_stats_by_neuron.csv"
	if not path.exists():
		raise FileNotFoundError(f"Missing neuron flip stats: {path}")
	df = pd.read_csv(path)
	if df.empty:
		return df
	for col in ("c2i_count", "flip_any_count", "i2c_count"):
		if col not in df.columns:
			df[col] = 0
	rank_col = rank_by if rank_by in df.columns else "flip_any_count"
	df = df.sort_values(
		[rank_col, "flip_any_count", "c2i_count", "i2c_count"],
		ascending=[False, False, False, False],
	).reset_index(drop=True)
	if rank_col.endswith("_count"):
		df = df[df[rank_col].fillna(0) > 0].copy()
	if max_k > 0:
		df = df.head(max_k).copy()
	return df


def _run_one(
	*,
	run_dir: Path,
	task_module: str,
	row: pd.Series,
	args: argparse.Namespace,
) -> List[Dict[str, Any]]:
	task = resolve_task_spec(task_module)
	task_name = _task_label(run_dir)
	fraction = float(row["fraction"])
	condition = str(row["condition"])
	checkpoint_dir = str(row["checkpoint_dir"])
	stats_dir = Path(str(row["lift_overtopping_stats_dir"]))
	base_dir = Path(str(row["lift_overtopping_base_dir"]))
	scores_path = base_dir / "feature_report" / "scores.csv"
	if not scores_path.exists():
		raise FileNotFoundError(f"Missing scores for {task_name} fraction={fraction}: {scores_path}")

	scores_df = pd.read_csv(scores_path)
	prompt_col = getattr(task, "DEFAULT_INPUT", "prompt")
	target_col = getattr(task, "DEFAULT_TARGETS", ("is_trigger_lift_success",))[0]
	if prompt_col not in scores_df.columns or target_col not in scores_df.columns:
		raise ValueError(f"{scores_path} lacks required columns {prompt_col!r}, {target_col!r}")

	target_mask = scores_df[target_col].map(_truthy).to_numpy(dtype=bool)
	pos_examples = _sample_examples(scores_df, target_mask, args.max_pos, args.seed + int(round(fraction * 1000)))
	neg_examples = _sample_examples(scores_df, ~target_mask, args.max_neg, args.seed + 17 + int(round(fraction * 1000)))
	if not pos_examples:
		print(f"[skip] {task_name} {condition} frac={fraction:g}: no positive trigger-lift examples.", flush=True)
		return []

	top_neurons = _rank_neurons(stats_dir, args.rank_by, max(args.top_ks))
	if top_neurons.empty:
		print(f"[skip] {task_name} {condition} frac={fraction:g}: no ranked neurons.", flush=True)
		return []

	device = get_device()
	model = LMWrapper(
		model_name=checkpoint_dir,
		device=device,
		eval_mode=True,
		circuit_discovery=False,
		cache_dir=args.ai_model_cache_dir,
	)

	max_k = min(max(args.top_ks), len(top_neurons))
	layer_to_neurons = _layer_units_from_top_rows(top_neurons, max_k)
	mean_activations = None
	if args.intervention in {"mean", "mean-donor", "mean-positional", "mean-donor-positional"}:
		mean_prompts = scores_df[prompt_col].astype(str).head(args.mean_points).tolist()
		mean_activations = precompute_mean_activations(
			model=model,
			all_prompts=mean_prompts,
			layer_to_neurons=layer_to_neurons,
			n_points=min(args.mean_points, len(mean_prompts)),
			batch_size=args.batch_size,
			intervention=args.intervention,
			device=device,
		)

	baseline = ablate_neurons(
		model,
		pos_examples,
		neg_examples,
		task.is_answer_positive,
		prompt_col,
		layers_neurons_dict=None,
		batch_size=args.batch_size,
		decode_only=args.decode_only,
		intervention=args.intervention,
		mean_activations=mean_activations,
		max_new_tokens=task.MAX_NEW_TOKENS,
		baseline_subset="positive",
	)

	clean_score_fn = _clean_correctness_fn(task_name)
	clean_examples = [dict(example) for example in pos_examples if example.get("prompt_without_trigger")]
	clean_baseline = None
	if clean_examples:
		clean_baseline = ablate_neurons(
			model,
			clean_examples,
			[],
			clean_score_fn,
			"prompt_without_trigger",
			layers_neurons_dict=None,
			batch_size=args.batch_size,
			decode_only=args.decode_only,
			intervention=args.intervention,
			mean_activations=mean_activations,
			max_new_tokens=task.MAX_NEW_TOKENS,
			baseline_subset="positive",
		)
	clean_baseline_vec = (
		np.asarray(clean_baseline["acc_after_knockout_on_associated_all"], dtype=float)
		if clean_baseline is not None
		else np.asarray([], dtype=float)
	)

	rows: List[Dict[str, Any]] = []
	for k in args.top_ks:
		k_eff = min(int(k), len(top_neurons))
		if k_eff <= 0:
			continue
		group = _layer_units_from_top_rows(top_neurons, k_eff)
		rec = ablate_neurons(
			model,
			pos_examples,
			neg_examples,
			task.is_answer_positive,
			prompt_col,
			layers_neurons_dict=group,
			batch_size=args.batch_size,
			decode_only=args.decode_only,
			intervention=args.intervention,
			mean_activations=mean_activations,
			max_new_tokens=task.MAX_NEW_TOKENS,
			baseline_subset="positive",
		)
		clean_rec = None
		if clean_examples:
			clean_rec = ablate_neurons(
				model,
				clean_examples,
				[],
				clean_score_fn,
				"prompt_without_trigger",
				layers_neurons_dict=group,
				batch_size=args.batch_size,
				decode_only=args.decode_only,
				intervention=args.intervention,
				mean_activations=mean_activations,
				max_new_tokens=task.MAX_NEW_TOKENS,
				baseline_subset="positive",
			)
		acc_pos = float(rec["acc_after_knockout_on_associated"])
		acc_neg = float(rec["acc_after_knockout_on_unrelated"]) if neg_examples else math.nan
		pos_vec = np.asarray(rec["acc_after_knockout_on_associated_all"], dtype=float)
		destroyed_count = int(np.sum(1.0 - pos_vec))
		destroy_ci_low, destroy_ci_high = _clopper_pearson(
			destroyed_count, len(pos_vec), level=args.ci_level
		)
		if neg_examples:
			neg_vec = np.asarray(rec["acc_after_knockout_on_unrelated_all"], dtype=float)
			neg_positive_count = int(np.sum(neg_vec))
			neg_ci_low, neg_ci_high = _clopper_pearson(
				neg_positive_count, len(neg_vec), level=args.ci_level
			)
		else:
			neg_positive_count = 0
			neg_ci_low, neg_ci_high = math.nan, math.nan
		if clean_rec is not None:
			clean_after_vec = np.asarray(clean_rec["acc_after_knockout_on_associated_all"], dtype=float)
			clean_correct_count = int(np.sum(clean_after_vec))
			clean_ci_low, clean_ci_high = _clopper_pearson(
				clean_correct_count, len(clean_after_vec), level=args.ci_level
			)
			clean_drop = float(clean_baseline_vec.mean() - clean_after_vec.mean())
			clean_drop_ci_low, clean_drop_ci_high = _paired_bootstrap_interval(
				clean_baseline_vec - clean_after_vec,
				level=args.ci_level,
				n_boot=args.bootstrap,
				seed=args.seed + 1009 * int(round(fraction * 100)) + int(k_eff),
			)
			clean_baseline_acc = float(clean_baseline_vec.mean())
			clean_after_acc = float(clean_after_vec.mean())
		else:
			clean_correct_count = 0
			clean_ci_low = clean_ci_high = math.nan
			clean_drop = clean_drop_ci_low = clean_drop_ci_high = math.nan
			clean_baseline_acc = clean_after_acc = math.nan
		rows.append(
			{
				"task": task_name,
				"run_dir": str(run_dir),
				"condition": condition,
				"fraction": fraction,
				"global_step": int(row.get("global_step", -1)),
				"checkpoint_dir": checkpoint_dir,
				"stats_dir": str(stats_dir),
				"rank_by": args.rank_by,
				"k": int(k_eff),
				"n_available_ranked_neurons": int(len(top_neurons)),
				"n_layers": int(len(group)),
				"n_pos": int(len(pos_examples)),
				"n_neg": int(len(neg_examples)),
				"n_clean_paired": int(len(clean_examples)),
				"baseline_acc_pos": float(baseline["acc_after_knockout_on_associated"]),
				"baseline_acc_neg": float(baseline["acc_after_knockout_on_unrelated"]) if neg_examples else math.nan,
				"acc_pos_after_topk": acc_pos,
				"acc_neg_after_topk": acc_neg,
				"trigger_lift_destroy_count": destroyed_count,
				"trigger_lift_destroy_rate": float(1.0 - acc_pos),
				"trigger_lift_destroy_ci_low": destroy_ci_low,
				"trigger_lift_destroy_ci_high": destroy_ci_high,
				"nonlift_target_positive_count": neg_positive_count,
				"nonlift_target_positive_rate": float(acc_neg) if neg_examples else math.nan,
				"nonlift_target_positive_ci_low": neg_ci_low,
				"nonlift_target_positive_ci_high": neg_ci_high,
				"paired_clean_baseline_accuracy": clean_baseline_acc,
				"paired_clean_accuracy_after_topk": clean_after_acc,
				"paired_clean_correct_count_after_topk": clean_correct_count,
				"paired_clean_accuracy_ci_low": clean_ci_low,
				"paired_clean_accuracy_ci_high": clean_ci_high,
				"paired_clean_accuracy_drop": clean_drop,
				"paired_clean_accuracy_drop_ci_low": clean_drop_ci_low,
				"paired_clean_accuracy_drop_ci_high": clean_drop_ci_high,
				"group": json.dumps(group, sort_keys=True),
			}
		)
		print(
			f"[topk] {task_name} {condition} frac={fraction:g} k={k_eff}: "
			f"destroy={1.0 - acc_pos:.3f} acc_neg={acc_neg if neg_examples else float('nan'):.3f}",
			flush=True,
		)

	del model
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return rows


def _plot(df: pd.DataFrame, out_path: Path) -> None:
	if df.empty:
		return
	fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
	for ax, task in zip(axes, ["grammar", "arithmetic"]):
		sub = df[df["task"] == task].copy()
		if sub.empty:
			ax.axis("off")
			continue
		for frac, label, marker in [(0.1, "poisoned 10%", "o"), (1.0, "poisoned final", "s")]:
			s = sub[np.isclose(sub["fraction"].astype(float), frac)].sort_values("k")
			if s.empty:
				continue
			ax.plot(
				s["k"],
				100.0 * s["trigger_lift_destroy_rate"],
				marker=marker,
				linewidth=2.2,
				label=label,
			)
		ax.set_xscale("log", base=2)
		ax.set_xlabel("Cumulative top-k ablated neurons")
		ax.set_title(task.capitalize())
		ax.grid(True, alpha=0.25)
		ax.set_ylim(-2, 102)
		ax.legend(frameon=False)
	axes[0].set_ylabel("Triggered-success destroyed (%)")
	fig.suptitle("Does late backdoor learning become distributed?", fontsize=14)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=220)
	fig.savefig(out_path.with_suffix(".pdf"))
	plt.close(fig)


def _write_coalition_summary(df: pd.DataFrame, out_dir: Path) -> None:
	"""Highlight when suppression first appears only for a multi-channel coalition."""
	if df.empty:
		return
	rows: List[Dict[str, Any]] = []
	for (task, fraction), group in df.groupby(["task", "fraction"], sort=True):
		g = group.sort_values("k").drop_duplicates("k")
		positive = g[pd.to_numeric(g["trigger_lift_destroy_count"], errors="coerce") > 0]
		first_effective = positive.iloc[0] if not positive.empty else None
		best = g.iloc[pd.to_numeric(g["trigger_lift_destroy_rate"], errors="coerce").argmax()]
		top4 = g[g["k"] <= 4]
		rows.append(
			{
				"task": task,
				"fraction": float(fraction),
				"first_k_with_any_effect": int(first_effective["k"]) if first_effective is not None else math.nan,
				"first_effect_count": int(first_effective["trigger_lift_destroy_count"]) if first_effective is not None else 0,
				"best_k": int(best["k"]),
				"best_destroy_rate": float(best["trigger_lift_destroy_rate"]),
				"best_destroy_ci_low": float(best["trigger_lift_destroy_ci_low"]),
				"best_destroy_ci_high": float(best["trigger_lift_destroy_ci_high"]),
				"all_top1_through_top4_zero": bool(
					len(top4) > 0
					and (pd.to_numeric(top4["trigger_lift_destroy_count"], errors="coerce").fillna(0) == 0).all()
				),
				"coalition_only_pattern": bool(
					first_effective is not None
					and int(first_effective["k"]) > 4
					and len(top4) > 0
					and (pd.to_numeric(top4["trigger_lift_destroy_count"], errors="coerce").fillna(0) == 0).all()
				),
				"paired_clean_accuracy_drop_at_best": float(best.get("paired_clean_accuracy_drop", math.nan)),
				"paired_clean_drop_ci_low_at_best": float(best.get("paired_clean_accuracy_drop_ci_low", math.nan)),
				"paired_clean_drop_ci_high_at_best": float(best.get("paired_clean_accuracy_drop_ci_high", math.nan)),
			}
		)
	summary = pd.DataFrame(rows)
	summary.to_csv(out_dir / "coalition_focus_summary.csv", index=False)
	lines = [
		"# Coalition-focused downstream summary",
		"",
		"The exact binomial intervals are over held-out trigger-lift examples. Clean-accuracy-drop intervals are paired bootstrap intervals on the same source prompts without the trigger.",
		"",
		"| Task | Checkpoint | First k with any effect | Best removal (95% CI) | Top-1..4 all zero? | Coalition-only pattern? | Paired clean drop (95% CI) |",
		"|---|---:|---:|---:|:---:|:---:|---:|",
	]
	for _, row in summary.iterrows():
		first_k = "none" if pd.isna(row["first_k_with_any_effect"]) else str(int(row["first_k_with_any_effect"]))
		lines.append(
			f"| {row['task']} | {100.0 * float(row['fraction']):.0f}% | {first_k} | "
			f"{100.0 * float(row['best_destroy_rate']):.1f}% "
			f"[{100.0 * float(row['best_destroy_ci_low']):.1f}, {100.0 * float(row['best_destroy_ci_high']):.1f}] | "
			f"{'yes' if row['all_top1_through_top4_zero'] else 'no'} | "
			f"{'yes' if row['coalition_only_pattern'] else 'no'} | "
			f"{100.0 * float(row['paired_clean_accuracy_drop_at_best']):+.1f} pp "
			f"[{100.0 * float(row['paired_clean_drop_ci_low_at_best']):+.1f}, "
			f"{100.0 * float(row['paired_clean_drop_ci_high_at_best']):+.1f}] |"
		)
	(out_dir / "coalition_focus_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_matched_random(path: Path, out_dir: Path, *, level: float, n_boot: int, seed: int) -> None:
	"""Summarize raw matched-random group results without treating the reported mean as exact."""
	if not path.exists():
		raise FileNotFoundError(path)
	df = pd.read_csv(path)
	required = {"task", "fraction", "k", "trigger_lift_destroy_rate"}
	missing = sorted(required - set(df.columns))
	if missing:
		raise ValueError(f"{path} is missing required columns: {missing}")
	rows = []
	rng = np.random.default_rng(int(seed))
	for keys, group in df.groupby(["task", "fraction", "k"], sort=True):
		values = pd.to_numeric(group["trigger_lift_destroy_rate"], errors="coerce").dropna().to_numpy(dtype=float)
		if len(values) == 0:
			continue
		means = []
		for _ in range(int(n_boot)):
			means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
		alpha = 1.0 - float(level)
		lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0]) if means else (math.nan, math.nan)
		rows.append(
			{
				"task": keys[0],
				"fraction": float(keys[1]),
				"k": int(keys[2]),
				"n_random_groups": int(len(values)),
				"matched_random_mean": float(values.mean()),
				"matched_random_median": float(np.median(values)),
				"matched_random_bootstrap_ci_low": float(lo),
				"matched_random_bootstrap_ci_high": float(hi),
				"matched_random_max": float(values.max()),
			}
		)
	pd.DataFrame(rows).to_csv(out_dir / "matched_random_group_summary.csv", index=False)


def main() -> None:
	p = argparse.ArgumentParser(description="Cumulative top-k ablation for trigger-lift backdoor checkpoints.")
	p.add_argument("--run_dirs", default=",".join(DEFAULT_RUNS), help="Comma-separated poisoning run directories.")
	p.add_argument("--task_modules", default=None, help="Optional comma-separated task modules matching run_dirs.")
	p.add_argument("--condition", default="poisoned")
	p.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1.0")
	p.add_argument("--eval_intervention", default="mean-donor")
	p.add_argument("--intervention", default="mean-donor", choices=["zero", "mean", "mean-donor", "mean-positional", "mean-donor-positional"])
	p.add_argument("--rank_by", default="c2i_count", help="Ranking column in flip_stats_by_neuron.csv.")
	p.add_argument("--top_ks", default="1,2,4,8,16,32,64,128")
	p.add_argument("--max_pos", type=int, default=64)
	p.add_argument("--max_neg", type=int, default=64)
	p.add_argument("--mean_points", type=int, default=512)
	p.add_argument("--batch_size", type=int, default=8)
	p.add_argument("--decode_only", action="store_true")
	p.add_argument("--seed", type=int, default=123)
	p.add_argument("--ci_level", type=float, default=0.95)
	p.add_argument("--bootstrap", type=int, default=5000)
	p.add_argument(
		"--matched_random_csv",
		default=None,
		help=(
			"Optional raw matched-random group results with task, fraction, k, and "
			"trigger_lift_destroy_rate columns. A bootstrap interval over random groups is reported."
		),
	)
	p.add_argument("--ai_model_cache_dir", default=None)
	p.add_argument("--output_dir", default=str(PROJECT_ROOT / "data" / "poisoning_mechanism_summary"))
	args = p.parse_args()

	args.top_ks = _parse_int_list(args.top_ks)
	run_dirs = [Path(x).expanduser() for x in _parse_csv_list(args.run_dirs)]
	fractions = _parse_float_list(args.fractions)
	if args.task_modules:
		task_modules = _parse_csv_list(args.task_modules)
		if len(task_modules) != len(run_dirs):
			raise ValueError("--task_modules must have one entry per --run_dirs entry.")
	else:
		task_modules = [_infer_task_module(p) for p in run_dirs]

	all_rows: List[Dict[str, Any]] = []
	for run_dir, task_module in zip(run_dirs, task_modules):
		selected = _find_rows_for_run(run_dir, fractions, args.condition, args.eval_intervention)
		for _, row in selected.iterrows():
			all_rows.extend(_run_one(run_dir=run_dir, task_module=task_module, row=row, args=args))

	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	out_csv = out_dir / "backdoor_lift_cumulative_topk_ablation.csv"
	out_df = pd.DataFrame(all_rows)
	out_df.to_csv(out_csv, index=False)
	_plot(out_df, out_dir / "backdoor_lift_cumulative_topk_ablation.png")
	_write_coalition_summary(out_df, out_dir)
	if args.matched_random_csv:
		_summarize_matched_random(
			Path(args.matched_random_csv).expanduser(),
			out_dir,
			level=args.ci_level,
			n_boot=args.bootstrap,
			seed=args.seed + 500_000,
		)
	print(f"Wrote {out_csv}")
	print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
	main()
