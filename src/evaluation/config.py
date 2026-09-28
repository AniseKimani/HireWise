"""Loader for config/evaluation_config.yaml (Day 4 evaluation protocol
and CBF tuning grid). The master seed and split ratios live in
src/data/config.py's DataConfig and are not duplicated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CONFIG_PATH = REPO_ROOT / "config" / "evaluation_config.yaml"


@dataclass(frozen=True)
class EvaluationConfig:
    k: int
    min_candidate_pool_size: int
    relevance_hired_grade: int
    relevance_shortlisted_grade: int
    relevance_none_grade: int
    cbf_price_neutral_score: float
    cbf_weight_grid: list[dict]
    cbf_tuning_primary_metric: str


def load_evaluation_config(path: str | Path = DEFAULT_EVAL_CONFIG_PATH) -> EvaluationConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = EvaluationConfig(**raw)
    _validate(cfg)
    return cfg


def _validate(cfg: EvaluationConfig) -> None:
    if cfg.k < 1:
        raise ValueError(f"k must be >= 1, got {cfg.k}")
    if cfg.min_candidate_pool_size < cfg.k:
        raise ValueError(
            f"min_candidate_pool_size ({cfg.min_candidate_pool_size}) must be >= k ({cfg.k}), "
            "otherwise an included request could not receive a full top-k ranking"
        )
    if not (cfg.relevance_hired_grade > cfg.relevance_shortlisted_grade > cfg.relevance_none_grade):
        raise ValueError("relevance grades must satisfy hired > shortlisted > none")
    if not cfg.cbf_weight_grid:
        raise ValueError("cbf_weight_grid must not be empty")
    for entry in cfg.cbf_weight_grid:
        keys = {"skill", "category", "price"}
        if set(entry) != keys:
            raise ValueError(f"cbf_weight_grid entry must have exactly keys {keys}, got {entry}")
        total = sum(entry.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"cbf_weight_grid entry must sum to 1.0, got {entry} (sum={total})")
