"""Loader for config/kg_config.yaml (Day 7 Knowledge Graph-only recommender)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KG_CONFIG_PATH = REPO_ROOT / "config" / "kg_config.yaml"

EXPECTED_COMPONENTS = ["skill", "related_skill", "category", "history", "reputation"]
ALLOWED_METRICS = {"ndcg_at_10", "precision_at_10", "recall_at_10", "f1_at_10"}


@dataclass(frozen=True)
class KGConfig:
    k: int
    components: list[str]
    category_role_weights: dict[str, float]
    reputation_prior_strength: float
    dwpc_damping_grid: list[float]
    weight_grid: list[dict]
    tuning_primary_metric: str
    tuning_tiebreak_metric: str
    explanation_max_paths_per_component: int
    explanation_sample_size: int
    explanation_sample_split: str


def load_kg_config(path: str | Path = DEFAULT_KG_CONFIG_PATH) -> KGConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = KGConfig(**raw)
    _validate(cfg)
    return cfg


def _validate(cfg: KGConfig) -> None:
    if cfg.components != EXPECTED_COMPONENTS:
        raise ValueError(f"components must be exactly {EXPECTED_COMPONENTS} (fixed order), got {cfg.components}")
    if set(cfg.category_role_weights) != {"primary", "secondary"}:
        raise ValueError(f"category_role_weights must have keys primary/secondary, got {cfg.category_role_weights}")
    if not cfg.category_role_weights["primary"] >= cfg.category_role_weights["secondary"] > 0:
        raise ValueError("category_role_weights must satisfy primary >= secondary > 0")
    if cfg.reputation_prior_strength <= 0:
        raise ValueError(f"reputation_prior_strength must be > 0, got {cfg.reputation_prior_strength}")
    if not cfg.dwpc_damping_grid or any(w < 0 for w in cfg.dwpc_damping_grid):
        raise ValueError(f"dwpc_damping_grid must be non-empty and non-negative, got {cfg.dwpc_damping_grid}")
    if list(cfg.dwpc_damping_grid) != sorted(set(cfg.dwpc_damping_grid)):
        raise ValueError("dwpc_damping_grid must be strictly increasing (simplest first)")
    if not cfg.weight_grid:
        raise ValueError("weight_grid must not be empty")
    for entry in cfg.weight_grid:
        if set(entry) != set(EXPECTED_COMPONENTS):
            raise ValueError(f"weight_grid entry must have exactly keys {EXPECTED_COMPONENTS}, got {entry}")
        if any(v < 0 for v in entry.values()):
            raise ValueError(f"weight_grid entry must be non-negative, got {entry}")
        if abs(sum(entry.values()) - 1.0) > 1e-9:
            raise ValueError(f"weight_grid entry must sum to 1.0, got {entry}")
    for metric in (cfg.tuning_primary_metric, cfg.tuning_tiebreak_metric):
        if metric not in ALLOWED_METRICS:
            raise ValueError(f"unsupported tuning metric {metric!r}; allowed {sorted(ALLOWED_METRICS)}")
    if cfg.explanation_max_paths_per_component < 1:
        raise ValueError("explanation_max_paths_per_component must be >= 1")
    if cfg.explanation_sample_size < 1:
        raise ValueError("explanation_sample_size must be >= 1")
    if cfg.explanation_sample_split not in ("validation", "test"):
        raise ValueError("explanation_sample_split must be 'validation' or 'test'")
