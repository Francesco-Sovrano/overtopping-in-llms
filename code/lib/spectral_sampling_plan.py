"""Canonical identity and path resolution for Stage-4 spectral sampling plans.

The shell pipeline passes experiment parameters only.  This module owns the
mapping from result-affecting sampling settings to a deterministic plan path so
Stage 4 (producer) and stages 5/6 (consumers) cannot drift independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class SpectralPlanConfig:
    role: str
    baseline_subset: str
    spectral_space: str
    rep_hook_name: str
    rep_pooling: str
    spectral_dim: int
    max_seq_len: int | None
    coverage_radius: float
    min_points_per_ablation: int
    max_points_per_ablation: int
    use_global_clusters: bool
    global_n_clusters: int
    pair_by_similarity_len_matched: bool
    pair_similarity_metric: str | None
    pair_len_tolerance: int
    seed: int
    compute_cover_stats: bool
    stats_sample_size: int
    stats_chunk_size: int

    def normalized(self) -> "SpectralPlanConfig":
        """Return the canonical identity used by the historical keyed filenames."""
        effective_max = max(
            int(self.min_points_per_ablation),
            int(self.max_points_per_ablation),
        )
        return SpectralPlanConfig(
            role=str(self.role),
            baseline_subset=str(self.baseline_subset),
            spectral_space=str(self.spectral_space),
            rep_hook_name=str(self.rep_hook_name),
            rep_pooling=str(self.rep_pooling),
            spectral_dim=int(self.spectral_dim),
            max_seq_len=None if self.max_seq_len is None else int(self.max_seq_len),
            coverage_radius=float(self.coverage_radius),
            min_points_per_ablation=int(self.min_points_per_ablation),
            max_points_per_ablation=effective_max,
            use_global_clusters=bool(self.use_global_clusters),
            global_n_clusters=int(self.global_n_clusters),
            pair_by_similarity_len_matched=bool(self.pair_by_similarity_len_matched),
            pair_similarity_metric=(
                str(self.pair_similarity_metric)
                if self.pair_by_similarity_len_matched and self.pair_similarity_metric is not None
                else None
            ),
            pair_len_tolerance=int(self.pair_len_tolerance),
            seed=int(self.seed),
            compute_cover_stats=bool(self.compute_cover_stats),
            stats_sample_size=int(self.stats_sample_size),
            stats_chunk_size=int(self.stats_chunk_size),
        )


def spectral_plan_fingerprint(config: SpectralPlanConfig) -> str:
    cfg = config.normalized()
    payload = json.dumps(
        asdict(cfg),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _hook_slug(hook_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(hook_name)).strip("_")[:48]
    return slug or "hook"


def spectral_plan_filename(config: SpectralPlanConfig) -> str:
    """Return the existing model-neutral keyed filename, now owned by Python."""
    cfg = config.normalized()
    return (
        f"spectral_sampling_plan-{cfg.role}"
        f"-space_{cfg.spectral_space}"
        f"-pool_{cfg.rep_pooling}"
        f"-hook_{_hook_slug(cfg.rep_hook_name)}"
        f"-d{cfg.spectral_dim}"
        f"-min{cfg.min_points_per_ablation}"
        f"-max{cfg.max_points_per_ablation}"
        f"-g{cfg.global_n_clusters}"
        f"-cfg_{spectral_plan_fingerprint(cfg)}.json"
    )


def spectral_plan_path(output_dir: str | Path, config: SpectralPlanConfig) -> Path:
    return Path(output_dir) / spectral_plan_filename(config)


def config_from_stage_args(
    args: Any,
    *,
    role: str,
    baseline_subset: str | None = None,
    min_points_per_ablation: int | None = None,
    max_points_per_ablation: int | None = None,
    global_n_clusters: int | None = None,
    pair_by_similarity_len_matched: bool | None = None,
    pair_similarity_metric: str | None = None,
) -> SpectralPlanConfig:
    """Build a canonical plan config from argparse-style stage arguments."""
    baseline = baseline_subset if baseline_subset is not None else args.baseline_subset
    min_points = (
        int(min_points_per_ablation)
        if min_points_per_ablation is not None
        else int(args.min_points_per_ablation)
    )
    max_points = (
        int(max_points_per_ablation)
        if max_points_per_ablation is not None
        else int(args.max_points_per_ablation)
    )
    global_n = (
        int(global_n_clusters)
        if global_n_clusters is not None
        else int(args.global_n_clusters)
    )
    pair_mode = (
        bool(pair_by_similarity_len_matched)
        if pair_by_similarity_len_matched is not None
        else bool(getattr(args, "pair_by_similarity_len_matched", False))
    )
    pair_metric = (
        pair_similarity_metric
        if pair_similarity_metric is not None
        else getattr(args, "pair_similarity_metric", None)
    )

    return SpectralPlanConfig(
        role=role,
        baseline_subset=baseline,
        spectral_space=args.spectral_space,
        rep_hook_name=args.rep_hook_name,
        rep_pooling=args.rep_pooling,
        spectral_dim=int(args.spectral_dim),
        max_seq_len=getattr(args, "max_seq_len", None),
        coverage_radius=float(args.sampling_plan_coverage_radius),
        min_points_per_ablation=min_points,
        max_points_per_ablation=max_points,
        use_global_clusters=True,
        global_n_clusters=global_n,
        pair_by_similarity_len_matched=pair_mode,
        pair_similarity_metric=pair_metric,
        pair_len_tolerance=int(args.sampling_plan_pair_len_tolerance),
        seed=int(args.sampling_plan_seed),
        compute_cover_stats=True,
        stats_sample_size=int(args.sampling_plan_stats_sample_size),
        stats_chunk_size=int(args.sampling_plan_stats_chunk_size),
    )



def config_from_stage4_args(args: Any, *, role: str) -> SpectralPlanConfig:
    """Build the canonical identity directly from Stage-4 CLI arguments."""
    return SpectralPlanConfig(
        role=role,
        baseline_subset=args.baseline_subset,
        spectral_space=args.spectral_space,
        rep_hook_name=args.rep_hook_name,
        rep_pooling=args.rep_pooling,
        spectral_dim=int(args.spectral_dim),
        max_seq_len=getattr(args, "max_seq_len", None),
        coverage_radius=float(args.coverage_radius),
        min_points_per_ablation=int(args.min_points_per_ablation),
        max_points_per_ablation=int(args.max_points_per_ablation),
        use_global_clusters=bool(args.use_global_clusters),
        global_n_clusters=int(args.global_n_clusters),
        pair_by_similarity_len_matched=bool(args.pair_by_similarity_len_matched),
        pair_similarity_metric=(
            args.pair_similarity_metric if args.pair_by_similarity_len_matched else None
        ),
        pair_len_tolerance=int(args.pair_len_tolerance),
        seed=int(args.seed),
        compute_cover_stats=bool(args.compute_cover_stats),
        stats_sample_size=int(args.stats_sample_size),
        stats_chunk_size=int(args.stats_chunk_size),
    )

def add_sampling_plan_reference_args(parser: Any) -> Any:
    """Add only the parameters needed to resolve a canonical Stage-4 plan."""
    g = parser.add_argument_group("spectral sampling plan reference")
    g.add_argument(
        "--sampling_plan_dir",
        type=str,
        default=None,
        help="Directory containing canonical Stage-4 sampling plans.",
    )
    g.add_argument(
        "--sampling_plan_role",
        type=str,
        default=None,
        help="Canonical plan role, e.g. circuit_discovery or neuron_ablation_positive.",
    )
    g.add_argument("--sampling_plan_min_points", type=int, default=None)
    g.add_argument("--sampling_plan_max_points", type=int, default=512)
    g.add_argument("--sampling_plan_global_n_clusters", type=int, default=None)
    g.add_argument("--sampling_plan_coverage_radius", type=float, default=0.5)
    g.add_argument("--sampling_plan_pair_len_tolerance", type=int, default=0)
    g.add_argument("--sampling_plan_seed", type=int, default=0)
    g.add_argument("--sampling_plan_stats_sample_size", type=int, default=200000)
    g.add_argument("--sampling_plan_stats_chunk_size", type=int, default=8192)
    return parser


def resolve_sampling_plan_path_from_args(
    args: Any,
    *,
    baseline_subset: str | None = None,
    pair_by_similarity_len_matched: bool = False,
    pair_similarity_metric: str | None = None,
) -> Path:
    """Resolve a plan path for a consumer, unless an explicit path was supplied."""
    explicit = getattr(args, "sampling_plan_path", None)
    if explicit:
        return Path(explicit)

    if not getattr(args, "sampling_plan_dir", None):
        raise ValueError(
            "--sampling_strategy 'plan' requires either --sampling_plan_path or --sampling_plan_dir."
        )
    if not getattr(args, "sampling_plan_role", None):
        raise ValueError(
            "Canonical plan lookup requires --sampling_plan_role when --sampling_plan_path is omitted."
        )
    if getattr(args, "sampling_plan_min_points", None) is None:
        raise ValueError("Canonical plan lookup requires --sampling_plan_min_points.")
    if getattr(args, "sampling_plan_global_n_clusters", None) is None:
        raise ValueError("Canonical plan lookup requires --sampling_plan_global_n_clusters.")

    cfg = config_from_stage_args(
        args,
        role=args.sampling_plan_role,
        baseline_subset=baseline_subset,
        min_points_per_ablation=args.sampling_plan_min_points,
        max_points_per_ablation=args.sampling_plan_max_points,
        global_n_clusters=args.sampling_plan_global_n_clusters,
        pair_by_similarity_len_matched=pair_by_similarity_len_matched,
        pair_similarity_metric=pair_similarity_metric,
    )
    return spectral_plan_path(args.sampling_plan_dir, cfg)
