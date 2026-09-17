"""Named child RNG streams for the Day 3 synthetic-data generator.

Per docs/experimental_design.md Section 13, each logically separate
generation stage gets its own independent child stream via
`numpy.random.SeedSequence(seed).spawn(...)`, so that adding or changing
one stage's random draws does not silently perturb another stage's
output. Stage names are indexed by position in STAGE_NAMES: **append new
stages to the end of this list; never reorder or insert**, or every
existing stage's stream would silently change.

This module is independent of src/data/sample.py's own stage stream
(used for Day 2 request sampling); the two are not required to be
cross-consistent, only each internally reproducible.
"""
from __future__ import annotations

import numpy as np

STAGE_NAMES = [
    "client_country_allocation",
    "client_category",
    "client_activity",
    "client_assignment",
    "timeline",
    "worker_category_allocation",
    "worker_secondary_category",
    "worker_true_skills",
    "worker_quality",
    "worker_taste",
    "worker_declared_skills",
    "worker_padding_skills",
    "worker_experience",
    "worker_rate",
    "worker_country",
    "cold_start_cohort",
    "candidate_eligibility",
    "applicant_sampling",
    "utility_noise",
    "shortlist",
    "hire",
    "rating",
    "review",
    "client_taste",
    "shortlist_calibration",
]


def stage_rng(seed: int, stage: str) -> np.random.Generator:
    """Return an independent NumPy Generator for one named stage."""
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown RNG stage {stage!r}; add it to STAGE_NAMES (at the end).")
    index = STAGE_NAMES.index(stage)
    children = np.random.SeedSequence(seed).spawn(len(STAGE_NAMES))
    return np.random.default_rng(children[index])
