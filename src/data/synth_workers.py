"""Synthetic worker generation (docs/experimental_design.md Section 8).

Workers are grounded in the real Day 2 demand distributions: category
allocation follows each category's share of eligible real jobs (with a
floor per category), skill pools are drawn from each category's real
skill frequencies, and rate/price-tier values are drawn from each
category's real pay distribution. Country, experience level, and latent
quality/taste have no real-data equivalent and are explicitly documented
generation assumptions.

Variable classification (docs/experimental_design.md Section 9.4 /
Day 3 brief Section 11):
  - Observable: declared_skills, experience_level, experience_years,
    price_tier, rate, country, primary_category, secondary_categories,
    join_day.
  - Generator-private (never joined into the observable table): quality
    q_w, true_skills/proficiency p_w(s), taste embedding v_w.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from src.data.config import DataConfig
from src.data.generator_config import GeneratorConfig
from src.data.synth_rng import stage_rng
from src.data.synth_util import apportion_with_floor, stable_id

EXPERIENCE_YEAR_RANGES = {"entry": (0, 2), "intermediate": (2, 6), "expert": (6, 15)}


def _skill_pools_by_category(eligible_jobs: pd.DataFrame) -> dict[str, Counter]:
    pools: dict[str, Counter] = {}
    for category, group in eligible_jobs.groupby("category"):
        counter = Counter()
        for cell in group["skills_vocab"]:
            skills = cell.split("|") if isinstance(cell, str) and cell else []
            counter.update(skills)
        pools[category] = counter
    return pools


def _weighted_choice_no_replace(rng: np.random.Generator, pool: Counter, size: int, exclude: set[str]) -> list[str]:
    candidates = [s for s in pool if s not in exclude]
    if not candidates or size <= 0:
        return []
    size = min(size, len(candidates))
    weights = np.array([pool[s] for s in candidates], dtype=float)
    weights /= weights.sum()
    chosen = rng.choice(candidates, size=size, replace=False, p=weights)
    return list(chosen)


def _secondary_categories(category_similarity: pd.DataFrame) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for category, group in category_similarity.groupby("category"):
        lookup[category] = group.sort_values("rank")["similar_category"].tolist()
    return lookup


def _top_countries(eligible_jobs: pd.DataFrame, gen_config: GeneratorConfig) -> pd.Series:
    counts = eligible_jobs["country_clean"].dropna().value_counts().head(gen_config.worker_country_top_n)
    smoothed = counts.astype(float) ** gen_config.worker_country_smoothing_power
    return smoothed / smoothed.sum()


def _category_pay_table(eligible_jobs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-category (price_tier, pay_amount) pairs, sorted ascending by
    price_tier, for correlated rate/price-tier sampling."""
    tables = {}
    for category, group in eligible_jobs.groupby("category"):
        pairs = group[["price_tier", "pay_amount"]].dropna(subset=["price_tier"])
        tables[category] = pairs.sort_values("price_tier").reset_index(drop=True)
    return tables


# [ASSUMPTION] These exponents bias which real (price_tier, pay_amount)
# row an entry/intermediate/expert worker inherits (lower row index ==
# lower price tier). They are a synthetic modelling choice, not a value
# derived from observed Upwork freelancer behaviour -- the raw dataset is
# demand-side only and contains no freelancer rate-by-experience data at
# all (docs/experimental_design.md Section 2, Limitation 1).
_EXPERIENCE_BIAS_POWER = {"entry": 2.0, "intermediate": 1.0, "expert": 0.6}


def _sample_biased_row(rng: np.random.Generator, table: pd.DataFrame, experience: str) -> tuple[float, float]:
    if table.empty:
        return float("nan"), float("nan")
    power = _EXPERIENCE_BIAS_POWER[experience]
    u = rng.random() ** power
    idx = int(u * (len(table) - 1))
    row = table.iloc[idx]
    return float(row["price_tier"]), float(row["pay_amount"]) if pd.notna(row["pay_amount"]) else float("nan")


def generate_workers(
    category_stats: pd.DataFrame,
    category_similarity: pd.DataFrame,
    eligible_jobs: pd.DataFrame,
    config: DataConfig,
    gen_config: GeneratorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (workers_df, workers_private_df).

    workers_df (observable): worker_id, primary_category,
        secondary_categories (pipe-joined), declared_skills (pipe-joined),
        experience_level, experience_years, price_tier, rate, country.
        join_day is added later once the simulated timeline's validation
        boundary is known (see assign_worker_join_days below).
    workers_private_df (generator-private): worker_id, quality,
        true_skills (pipe-joined), proficiency (pipe-joined, same order as
        true_skills), taste_0..taste_{dim-1}.
    """
    category_weights = category_stats.set_index("category")["n_jobs"]
    counts_per_category = apportion_with_floor(
        category_weights, total=config.synthetic_workers, floor=gen_config.worker_category_floor
    )

    skill_pools = _skill_pools_by_category(eligible_jobs)
    secondary_lookup = _secondary_categories(category_similarity)
    country_dist = _top_countries(eligible_jobs, gen_config)
    pay_tables = _category_pay_table(eligible_jobs)

    category_rng = stage_rng(config.seed, "worker_category_allocation")
    secondary_rng = stage_rng(config.seed, "worker_secondary_category")
    skills_rng = stage_rng(config.seed, "worker_true_skills")
    quality_rng = stage_rng(config.seed, "worker_quality")
    taste_rng = stage_rng(config.seed, "worker_taste")
    declared_rng = stage_rng(config.seed, "worker_declared_skills")
    padding_rng = stage_rng(config.seed, "worker_padding_skills")
    experience_rng = stage_rng(config.seed, "worker_experience")
    rate_rng = stage_rng(config.seed, "worker_rate")
    country_rng = stage_rng(config.seed, "worker_country")

    n_workers = config.synthetic_workers
    primary_categories: list[str] = []
    for category in sorted(counts_per_category.index):
        primary_categories.extend([category] * int(counts_per_category[category]))
    assert len(primary_categories) == n_workers

    # --- Experience level and correlated latent quality (Section 8) -------
    rho = gen_config.worker_quality_experience_correlation
    z_exp = experience_rng.standard_normal(n_workers)
    noise = quality_rng.standard_normal(n_workers)
    z_qual = rho * z_exp + np.sqrt(max(0.0, 1 - rho**2)) * noise
    quality = beta_dist.ppf(norm.cdf(z_qual), gen_config.worker_quality_beta_a, gen_config.worker_quality_beta_b)

    props = gen_config.worker_experience_proportions
    cut_entry = norm.ppf(props["entry"])
    cut_intermediate = norm.ppf(props["entry"] + props["intermediate"])
    experience_level = np.where(
        z_exp < cut_entry, "entry", np.where(z_exp < cut_intermediate, "intermediate", "expert")
    )
    experience_years = np.array([
        experience_rng.uniform(*EXPERIENCE_YEAR_RANGES[level]) for level in experience_level
    ])

    taste_dim = gen_config.worker_taste_dim
    taste = taste_rng.standard_normal(size=(n_workers, taste_dim))

    country_names = country_dist.index.to_numpy()
    country_probs = country_dist.to_numpy()
    countries = country_rng.choice(country_names, size=n_workers, p=country_probs)

    rows_observable = []
    rows_private = []
    for i in range(n_workers):
        worker_id = stable_id("W", i)
        category = primary_categories[i]
        level = experience_level[i]

        pool = secondary_lookup.get(category, [])
        n_secondary = secondary_rng.integers(
            gen_config.worker_secondary_category_min, gen_config.worker_secondary_category_max + 1
        )
        secondary_pool = pool[: gen_config.worker_secondary_category_pool]
        secondary = (
            list(secondary_rng.choice(secondary_pool, size=min(n_secondary, len(secondary_pool)), replace=False))
            if secondary_pool and n_secondary > 0
            else []
        )

        primary_pool = skill_pools.get(category, Counter())
        n_primary_skills = skills_rng.integers(gen_config.worker_true_skills_min, gen_config.worker_true_skills_max + 1)
        true_skills = _weighted_choice_no_replace(skills_rng, primary_pool, n_primary_skills, exclude=set())

        if secondary:
            secondary_pool_counter = Counter()
            for sec_cat in secondary:
                secondary_pool_counter.update(skill_pools.get(sec_cat, Counter()))
            n_secondary_skills = skills_rng.integers(
                gen_config.worker_secondary_skills_min, gen_config.worker_secondary_skills_max + 1
            )
            true_skills += _weighted_choice_no_replace(
                skills_rng, secondary_pool_counter, n_secondary_skills, exclude=set(true_skills)
            )

        shift = {"entry": 0.0, "intermediate": 0.1, "expert": 0.2}[level]
        base_beta = skills_rng.beta(2, 2, size=len(true_skills)) if true_skills else np.array([])
        proficiency = base_beta * (1 - shift) + shift

        declared = []
        for skill, prof in zip(true_skills, proficiency):
            threshold = gen_config.worker_declared_proficiency_threshold
            prob = gen_config.worker_declared_high_prob if prof >= threshold else gen_config.worker_declared_low_prob
            if declared_rng.random() < prob:
                declared.append(skill)

        popular_skills = [s for s, _ in primary_pool.most_common(20) if s not in true_skills]
        n_padding = sum(
            1 for _ in range(gen_config.worker_padding_skill_max) if padding_rng.random() < gen_config.worker_padding_skill_prob
        )
        padding = list(padding_rng.choice(popular_skills, size=min(n_padding, len(popular_skills)), replace=False)) if popular_skills and n_padding else []
        declared = declared + padding

        pay_table = pay_tables.get(category, pd.DataFrame(columns=["price_tier", "pay_amount"]))
        price_tier, rate = _sample_biased_row(rate_rng, pay_table, level)

        rows_observable.append({
            "worker_id": worker_id,
            "primary_category": category,
            "secondary_categories": "|".join(secondary),
            "declared_skills": "|".join(dict.fromkeys(declared)),
            "experience_level": level,
            "experience_years": round(float(experience_years[i]), 2),
            "price_tier": price_tier,
            "rate": rate,
            "country": countries[i],
        })
        rows_private.append({
            "worker_id": worker_id,
            "quality": float(quality[i]),
            "true_skills": "|".join(true_skills),
            "proficiency": "|".join(f"{p:.4f}" for p in proficiency),
            **{f"taste_{d}": float(taste[i, d]) for d in range(taste_dim)},
        })

    workers_df = pd.DataFrame(rows_observable)
    workers_private_df = pd.DataFrame(rows_private)
    return workers_df, workers_private_df


def extract_required_countries(location_requirement: object, country_names: list[str]) -> list[str]:
    """Country names (from the worker country pool) mentioned in a
    request's free-text location requirement, matched on word boundaries
    so e.g. "United States" does not spuriously match inside another
    country's name."""
    if not isinstance(location_requirement, str) or not location_requirement:
        return []
    return [c for c in country_names if re.search(rf"\b{re.escape(c)}\b", location_requirement)]


def apply_location_guarantee(
    workers_df: pd.DataFrame,
    sampled_requests: pd.DataFrame,
    country_names: list[str],
    config: DataConfig,
    gen_config: GeneratorConfig,
) -> tuple[pd.DataFrame, dict]:
    """Best-effort repair pass: for every (category, required country)
    pair that appears in a location-restricted sampled request, ensure at
    least `gen_config.worker_location_min_eligible` workers with that
    country have the category as primary or secondary. Falls back to
    leaving a documented shortfall if the category has too few workers
    overall to satisfy it without emptying every other country.
    """
    # A separate stream from initial country assignment (stage
    # "worker_country"), so this repair pass is reproducible independent
    # of how many draws generate_workers() happened to take.
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x10CA71011]))

    workers_df = workers_df.copy()
    has_secondary = workers_df["secondary_categories"].str.split("|")
    workers_df["_categories"] = [
        {row.primary_category} | set(s for s in secs if s)
        for row, secs in zip(workers_df.itertuples(), has_secondary)
    ]

    pairs = set()
    for loc, category in zip(sampled_requests["location_requirement"], sampled_requests["category"]):
        for country in extract_required_countries(loc, country_names):
            pairs.add((category, country))

    report = {"pairs_checked": len(pairs), "pairs_short": 0, "workers_reassigned": 0}
    for category, country in sorted(pairs):
        eligible_mask = workers_df["_categories"].map(lambda cats: category in cats)
        current = workers_df.loc[eligible_mask & (workers_df["country"] == country)]
        shortfall = gen_config.worker_location_min_eligible - len(current)
        if shortfall <= 0:
            continue
        report["pairs_short"] += 1

        donor_pool = workers_df.loc[eligible_mask & (workers_df["country"] != country)]
        n_move = min(shortfall, len(donor_pool))
        if n_move <= 0:
            continue
        chosen = rng.choice(donor_pool.index.to_numpy(), size=n_move, replace=False)
        workers_df.loc[chosen, "country"] = country
        report["workers_reassigned"] += int(n_move)

    return workers_df.drop(columns=["_categories"]), report
