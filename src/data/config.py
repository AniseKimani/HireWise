"""Loader for config/data_config.yaml, the single source of truth for
Day 2 pipeline thresholds and experimental parameters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "data_config.yaml"


@dataclass(frozen=True)
class DataConfig:
    seed: int
    min_category_jobs: int
    min_skill_frequency: int
    model_requests: int
    synthetic_clients: int
    synthetic_workers: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    cold_start_ratio: float
    cooccurrence_min_support: int
    cooccurrence_min_npmi: float
    winsorize_low_percentile: float
    winsorize_high_percentile: float
    price_tier_reference: str

    @property
    def split_ratios(self) -> tuple[float, float, float]:
        return self.train_ratio, self.validation_ratio, self.test_ratio


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DataConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = DataConfig(**raw)
    _validate(cfg)
    return cfg


def _validate(cfg: DataConfig) -> None:
    ratio_sum = cfg.train_ratio + cfg.validation_ratio + cfg.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"train_ratio + validation_ratio + test_ratio must sum to 1.0, got {ratio_sum}"
        )
    if not 0 < cfg.cold_start_ratio < 1:
        raise ValueError(f"cold_start_ratio must be in (0, 1), got {cfg.cold_start_ratio}")
    if cfg.min_category_jobs < 1 or cfg.min_skill_frequency < 1:
        raise ValueError("min_category_jobs and min_skill_frequency must be >= 1")
    if cfg.model_requests < 1:
        raise ValueError("model_requests must be >= 1")
    if not 0 <= cfg.winsorize_low_percentile < cfg.winsorize_high_percentile <= 1:
        raise ValueError("winsorize percentiles must satisfy 0 <= low < high <= 1")
