"""Loader for config/hybrid_config.yaml (Day 8 weighted hybrid)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYBRID_CONFIG_PATH = REPO_ROOT / "config" / "hybrid_config.yaml"

EXPECTED_COMPONENTS = ["cbf", "cf", "kg"]
ALLOWED_METRICS = {"ndcg_at_10", "precision_at_10", "recall_at_10", "f1_at_10"}
ALLOWED_BOOTSTRAP_METRICS = {"ndcg_at_k", "precision_at_k", "recall_at_k", "f1_at_k"}
ALLOWED_BASELINES = {"random", "popularity", "cbf", "cf", "kg"}


@dataclass(frozen=True)
class HybridConfig:
    k: int
    components: list[str]
    normalization: str
    constant_component_policy: str
    weight_grid: list[dict]
    tuning_primary_metric: str
    tuning_tiebreak_metric: str
    sanity_single_component_checks: bool
    bootstrap: dict
    diversity: dict
    explanations: dict


def load_hybrid_config(path: str | Path = DEFAULT_HYBRID_CONFIG_PATH) -> HybridConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = HybridConfig(**raw)
    _validate(cfg)
    return cfg


def balance(weights: dict[str, float]) -> float:
    """Sum of |w - 1/n|: 0 for equal weights, larger for more lopsided ones."""
    n = len(weights)
    return sum(abs(w - 1.0 / n) for w in weights.values())


def _validate(cfg: HybridConfig) -> None:
    if cfg.components != EXPECTED_COMPONENTS:
        raise ValueError(f"components must be exactly {EXPECTED_COMPONENTS}, got {cfg.components}")
    if cfg.normalization != "minmax_within_candidate_pool":
        raise ValueError(f"unsupported normalization {cfg.normalization!r}")
    if cfg.constant_component_policy != "zero":
        raise ValueError(f"unsupported constant_component_policy {cfg.constant_component_policy!r}")
    if not cfg.weight_grid:
        raise ValueError("weight_grid must not be empty")
    seen = set()
    for entry in cfg.weight_grid:
        if set(entry) != set(EXPECTED_COMPONENTS):
            raise ValueError(f"weight_grid entry must have exactly keys {EXPECTED_COMPONENTS}, got {entry}")
        if any(v < 0 for v in entry.values()):
            raise ValueError(f"weights must be non-negative, got {entry}")
        if abs(sum(entry.values()) - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0, got {entry}")
        if sum(1 for v in entry.values() if v > 0) < 2:
            raise ValueError(f"single-component entries are sanity checks, not tuning candidates: {entry}")
        key = tuple(entry[c] for c in EXPECTED_COMPONENTS)
        if key in seen:
            raise ValueError(f"duplicate weight_grid entry {entry}")
        seen.add(key)
    for metric in (cfg.tuning_primary_metric, cfg.tuning_tiebreak_metric):
        if metric not in ALLOWED_METRICS:
            raise ValueError(f"unsupported tuning metric {metric!r}")
    b = cfg.bootstrap
    if set(b) != {"n_resamples", "confidence_level", "seed_offset", "baselines", "metrics", "multiple_comparison_adjustment"}:
        raise ValueError(f"unexpected bootstrap keys {sorted(b)}")
    if b["n_resamples"] < 100 or not 0 < b["confidence_level"] < 1:
        raise ValueError("bootstrap needs n_resamples >= 100 and 0 < confidence_level < 1")
    if not set(b["baselines"]) <= ALLOWED_BASELINES or not b["baselines"]:
        raise ValueError(f"bootstrap baselines must be a non-empty subset of {sorted(ALLOWED_BASELINES)}")
    if not set(b["metrics"]) <= ALLOWED_BOOTSTRAP_METRICS or not b["metrics"]:
        raise ValueError(f"bootstrap metrics must be a non-empty subset of {sorted(ALLOWED_BOOTSTRAP_METRICS)}")
    if b["multiple_comparison_adjustment"] != "holm":
        raise ValueError("only holm adjustment is supported")
    if cfg.diversity.get("intra_list_skill_source") != "declared_skills":
        raise ValueError("diversity.intra_list_skill_source must be declared_skills")
    if not 0 < cfg.diversity.get("top_share_fraction", 0) < 1:
        raise ValueError("diversity.top_share_fraction must be in (0, 1)")
    e = cfg.explanations
    if e.get("sample_size", 0) < 1 or e.get("sample_split") not in ("validation", "test") or e.get("kg_paths_per_sample", 0) < 1:
        raise ValueError(f"invalid explanations settings {e}")
