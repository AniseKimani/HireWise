"""Cleaning and flagging pipeline for the parsed Upwork dataset.

Design principle (docs/experimental_design.md Section 3 and the Day 2
brief): the raw CSV is never modified, and excluded records are flagged,
not deleted, so every row's fate is auditable. A separate "eligible"
subset used for modelling is derived from these flags, but the full
flagged frame is always retrievable.

Exclusion precedence (first applicable reason wins for `exclusion_reason`,
but every individual boolean flag is kept regardless of precedence):

    1. duplicate
    2. no skills
    3. out-of-scope category
    4. no controlled-vocabulary skill
"""
from __future__ import annotations

import pandas as pd

from src.data.config import DataConfig
from src.data.parse import clean_country, parse_job

PAY_COMPLETENESS_COLS = ["is_hourly", "hourly_low", "hourly_high", "budget", "country"]


def parse_and_flag(raw: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Parse every row's description and compute the flags that do not
    depend on the controlled vocabulary (duplicate, no-skills, category
    scope). `has_vocab_skill` and downstream eligibility are added later
    by `finalize_flags`, once the vocabulary has been built from this
    frame's in-scope subset.
    """
    df = raw.copy()

    parsed = df["description"].map(parse_job)
    df["footer_found"] = parsed.map(lambda p: p.footer_found)
    df["category"] = parsed.map(lambda p: p.category)
    df["skills_raw"] = parsed.map(lambda p: p.skills_raw)
    df["n_skills_raw"] = df["skills_raw"].map(len)
    df["location_requirement"] = parsed.map(lambda p: p.location_requirement)
    df["footer_country"] = parsed.map(lambda p: p.footer_country)
    df["country_clean"] = df["country"].map(clean_country)

    df["pay_type"] = df["is_hourly"].map(_pay_type)
    df["pay_amount"] = df.apply(_pay_amount, axis=1)

    df["is_duplicate"] = _flag_duplicates(df)
    df["has_no_skills"] = df["n_skills_raw"] == 0

    in_scope_candidates = df.loc[~df["is_duplicate"] & ~df["has_no_skills"]]
    category_counts = in_scope_candidates["category"].value_counts()
    in_scope_categories = set(category_counts[category_counts >= config.min_category_jobs].index)
    df["is_in_scope_category"] = df["category"].isin(in_scope_categories)

    return df


def _pay_type(is_hourly) -> str:
    if pd.isna(is_hourly):
        return "unspecified"
    return "hourly" if bool(is_hourly) else "fixed"


def _pay_amount(row) -> float | None:
    if row["pay_type"] == "fixed":
        return row["budget"] if pd.notna(row["budget"]) else None
    if row["pay_type"] == "hourly":
        low, high = row["hourly_low"], row["hourly_high"]
        if pd.notna(low) and pd.notna(high):
            return (low + high) / 2
        if pd.notna(low):
            return low
        if pd.notna(high):
            return high
        return None
    return None


def _flag_duplicates(df: pd.DataFrame) -> pd.Series:
    """Flag every row in a title+description duplicate group except the
    one kept: most complete across PAY_COMPLETENESS_COLS, tying broken by
    the lexicographically smallest link."""
    completeness = df[PAY_COMPLETENESS_COLS].notna().sum(axis=1)
    order = pd.DataFrame({"completeness": completeness, "link": df["link"]}, index=df.index)
    ranked = order.sort_values(["completeness", "link"], ascending=[False, True])
    dup_group_key = df.loc[ranked.index, ["title", "description"]]
    keep_mask = ~dup_group_key.duplicated(keep="first")
    keep_index = ranked.index[keep_mask.values]
    is_duplicate = pd.Series(True, index=df.index)
    is_duplicate.loc[keep_index] = False
    return is_duplicate


def finalize_flags(df: pd.DataFrame, vocab: set[str], config: DataConfig) -> pd.DataFrame:
    """Add vocabulary-dependent fields once the controlled vocabulary has
    been built: skills_vocab, has_vocab_skill, exclusion_reason,
    exclude_from_model, eligible_for_model, and price_tier."""
    df = df.copy()
    df["skills_vocab"] = df["skills_raw"].map(lambda skills: [s for s in skills if s in vocab])
    df["has_vocab_skill"] = df["skills_vocab"].map(len) > 0

    df["exclusion_reason"] = df.apply(lambda r: _exclusion_reason(r), axis=1)
    df["exclude_from_model"] = df["exclusion_reason"].notna()
    df["eligible_for_model"] = ~df["exclude_from_model"]

    df["price_tier"] = _price_tiers(df, config)
    return df


def _exclusion_reason(row) -> str | None:
    if row["is_duplicate"]:
        return "duplicate"
    if row["has_no_skills"]:
        return "no_skills"
    if not row["is_in_scope_category"]:
        return "out_of_scope_category"
    if not row["has_vocab_skill"]:
        return "no_vocab_skill"
    return None


def _price_tiers(df: pd.DataFrame, config: DataConfig) -> pd.Series:
    """Percentile rank (0-1) of pay_amount within (category, pay_type),
    computed over the reference population named in
    config.price_tier_reference. Rows outside the reference population,
    or with no pay_amount, get a null price tier -- price tier is a
    modelling feature and only eligible jobs are modelled."""
    if config.price_tier_reference != "in_scope_jobs":
        raise ValueError(f"Unsupported price_tier_reference: {config.price_tier_reference}")

    reference_mask = df["is_in_scope_category"] & ~df["is_duplicate"] & ~df["has_no_skills"]
    reference = df.loc[reference_mask & df["pay_amount"].notna()]

    tiers = pd.Series(float("nan"), index=df.index)
    ranks = reference.groupby(["category", "pay_type"])["pay_amount"].rank(pct=True)
    tiers.loc[ranks.index] = ranks.values
    return tiers
