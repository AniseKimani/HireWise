"""Loader for the frozen Day 3 generator configuration
(config/generator_v1.yaml and its sensitivity-analysis variants).

Kept separate from src/data/config.py's DataConfig (Day 2 dataset
thresholds) so the generator's detailed behavioural parameters can be
frozen, hashed, and versioned independently -- see the header comment in
config/generator_v1.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GeneratorConfig:
    version: str
    simulated_days: int

    client_activity_mean: float
    client_activity_min: int
    client_activity_max: int
    client_category_coherence: float
    client_start_day_max: int
    client_taste_dim: int

    worker_category_floor: int
    worker_secondary_category_min: int
    worker_secondary_category_max: int
    worker_secondary_category_pool: int
    worker_true_skills_min: int
    worker_true_skills_max: int
    worker_secondary_skills_min: int
    worker_secondary_skills_max: int
    worker_quality_beta_a: float
    worker_quality_beta_b: float
    worker_quality_experience_correlation: float
    worker_taste_dim: int
    worker_declared_high_prob: float
    worker_declared_low_prob: float
    worker_declared_proficiency_threshold: float
    worker_padding_skill_prob: float
    worker_padding_skill_max: int
    worker_experience_proportions: dict
    worker_country_top_n: int
    worker_country_smoothing_power: float
    worker_location_min_eligible: int

    skill_relatedness_topn: int
    skill_relatedness_lambda: float

    applicant_pool_max: int
    applicant_base_prob: float

    utility_weights: dict
    utility_noise_variance_share: float

    shortlist_size: int
    shortlist_temperature: float
    hire_temperature: float
    hire_rate_target: float
    worker_capacity_max_hires: int
    worker_capacity_window_days: int

    rating_completion_rate: float
    rating_weights: dict
    rating_noise_std: float
    review_rate: float


def load_generator_config(path: str | Path) -> GeneratorConfig:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = GeneratorConfig(**raw)
    _validate(cfg)
    return cfg


def _validate(cfg: GeneratorConfig) -> None:
    weight_sum = sum(cfg.utility_weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"utility_weights must sum to 1.0, got {weight_sum}")

    experience_sum = sum(cfg.worker_experience_proportions.values())
    if abs(experience_sum - 1.0) > 1e-6:
        raise ValueError(f"worker_experience_proportions must sum to 1.0, got {experience_sum}")

    rating_weight_sum = sum(cfg.rating_weights.values())
    if abs(rating_weight_sum - 1.0) > 1e-6:
        raise ValueError(f"rating_weights must sum to 1.0, got {rating_weight_sum}")

    for name in ("client_category_coherence", "hire_rate_target", "rating_completion_rate",
                 "review_rate", "applicant_base_prob", "utility_noise_variance_share",
                 "worker_declared_high_prob", "worker_declared_low_prob",
                 "worker_padding_skill_prob", "worker_declared_proficiency_threshold"):
        value = getattr(cfg, name)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    if cfg.client_activity_min < 1 or cfg.client_activity_max < cfg.client_activity_min:
        raise ValueError("client_activity_min/max are inconsistent")
    if cfg.worker_secondary_category_min < 0 or cfg.worker_secondary_category_max < cfg.worker_secondary_category_min:
        raise ValueError("worker_secondary_category_min/max are inconsistent")
    if cfg.worker_true_skills_min < 1 or cfg.worker_true_skills_max < cfg.worker_true_skills_min:
        raise ValueError("worker_true_skills_min/max are inconsistent")
