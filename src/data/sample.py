"""Deterministic stratified sampling of modelling requests from the
eligible job pool, and construction of the background corpus.

Method (docs/experimental_design.md Section 6.3 / Appendix A): draw
`config.model_requests` requests from the eligible pool, stratified by
category so that every in-scope category is represented roughly in
proportion to its share of eligible jobs. The remaining eligible jobs form
the background corpus used for statistics that must not leak into the
sampled evaluation set (see src/data/stats.py).

Determinism: eligible jobs are sorted by `link` (a stable, content-derived
key) before any random draw, and all randomness is drawn from a NumPy
Generator seeded via `numpy.random.SeedSequence(config.seed)`, following
docs/experimental_design.md Section 13's child-stream convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.config import DataConfig

SAMPLING_STAGE_LABEL = "request_sampling"


def _stage_rng(seed: int, stage_label: str, n_stages: int = 8, stage_index: int = 0) -> np.random.Generator:
    """Independent child stream for one named pipeline stage, per
    docs/experimental_design.md Section 13. `stage_index` fixes which
    child a given stage always draws, so adding future stages does not
    silently perturb this one's draws."""
    children = np.random.SeedSequence(seed).spawn(n_stages)
    return np.random.default_rng(children[stage_index])


def _apportion(counts: pd.Series, total: int) -> pd.Series:
    """Largest-remainder (Hamilton) apportionment of `total` seats across
    categories in proportion to `counts`, guaranteeing the result sums to
    exactly `total`. Deterministic given `counts`."""
    shares = counts / counts.sum() * total
    base = np.floor(shares).astype(int)
    remainder = total - base.sum()
    if remainder > 0:
        # Give the remaining seats to the largest fractional remainders;
        # tie-break on category name for determinism.
        fractional = shares - base
        order = sorted(counts.index, key=lambda cat: (-fractional[cat], cat))
        for cat in order[:remainder]:
            base[cat] += 1
    return base


def sample_requests(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Return `df` with a new boolean column `is_sampled_request`, true for
    exactly `config.model_requests` rows drawn from the eligible pool,
    stratified by category."""
    eligible = df.loc[df["eligible_for_model"]].sort_values("link")
    category_counts = eligible["category"].value_counts()
    quota = _apportion(category_counts, config.model_requests)

    rng = _stage_rng(config.seed, SAMPLING_STAGE_LABEL)
    sampled_links: list[str] = []
    for category, n in quota.items():
        if n <= 0:
            continue
        pool = eligible.loc[eligible["category"] == category, "link"].to_numpy()
        chosen = rng.choice(pool, size=n, replace=False)
        sampled_links.extend(chosen.tolist())

    sampled_set = set(sampled_links)
    result = df.copy()
    result["is_sampled_request"] = result["link"].isin(sampled_set)
    return result


def background_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Eligible jobs not drawn as sampled requests: the corpus used for
    statistics that must stay independent of the evaluation sample (e.g.
    Knowledge Graph skill-relatedness edges, per Section 14)."""
    return df.loc[df["eligible_for_model"] & ~df["is_sampled_request"]]
