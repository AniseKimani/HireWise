"""Loader for config/cf_config.yaml (Day 5 Collaborative Filtering)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CF_CONFIG_PATH = REPO_ROOT / "config" / "cf_config.yaml"

ALLOWED_AGGREGATIONS = {"sum", "max"}
ALLOWED_METHODS = {"user", "item"}


@dataclass(frozen=True)
class CFConfig:
    k: int
    event_weights: dict[str, float]
    aggregation_grid: list[str]
    method_grid: list[str]
    k_neighbors_grid: list[int]
    min_similarity: float
    tuning_primary_metric: str
    tuning_tiebreak_metric: str
    fallback: str


def load_cf_config(path: str | Path = DEFAULT_CF_CONFIG_PATH) -> CFConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = CFConfig(**raw)
    _validate(cfg)
    return cfg


def _validate(cfg: CFConfig) -> None:
    if not set(cfg.aggregation_grid).issubset(ALLOWED_AGGREGATIONS):
        raise ValueError(f"aggregation_grid must be a subset of {ALLOWED_AGGREGATIONS}, got {cfg.aggregation_grid}")
    if not set(cfg.method_grid).issubset(ALLOWED_METHODS):
        raise ValueError(f"method_grid must be a subset of {ALLOWED_METHODS}, got {cfg.method_grid}")
    if not cfg.k_neighbors_grid or any(k < 1 for k in cfg.k_neighbors_grid):
        raise ValueError(f"k_neighbors_grid must be non-empty positive integers, got {cfg.k_neighbors_grid}")
    if not cfg.event_weights:
        raise ValueError("event_weights must not be empty")
    if any(w <= 0 for w in cfg.event_weights.values()):
        raise ValueError(f"event_weights must be positive, got {cfg.event_weights}")
    if cfg.fallback != "popularity_training_only":
        raise ValueError(f"unsupported fallback strategy: {cfg.fallback}")
