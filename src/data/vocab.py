"""Controlled skill vocabulary construction.

Methodology (docs/experimental_design.md Section 6, approved Day 1
decision): exact string matching only, no fuzzy/stemming/synonym merging,
no embeddings, no LLM normalisation. The vocabulary is the set of skills
appearing in at least `config.min_skill_frequency` jobs within the
in-scope modelling set (deduplicated, has at least one skill, category in
scope). Rare skills are never discarded from a job's `skills_raw`; they
are only excluded from `skills_vocab` and recorded in a rare-skill audit.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from src.data.config import DataConfig


def in_scope_modelling_mask(df: pd.DataFrame) -> pd.Series:
    """Rows eligible to contribute to vocabulary frequency counts:
    not a duplicate, has at least one parsed skill, category in scope."""
    return ~df["is_duplicate"] & ~df["has_no_skills"] & df["is_in_scope_category"]


def build_vocabulary(df: pd.DataFrame, config: DataConfig) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    """Return (vocab_set, vocab_table, rare_skills_table).

    vocab_table and rare_skills_table both have columns [skill, frequency],
    sorted by frequency descending then skill ascending for deterministic
    output.
    """
    scope = df.loc[in_scope_modelling_mask(df)]
    frequency = Counter(skill for skills in scope["skills_raw"] for skill in skills)

    vocab = {skill for skill, count in frequency.items() if count >= config.min_skill_frequency}

    freq_df = (
        pd.DataFrame(frequency.items(), columns=["skill", "frequency"])
        .sort_values(["frequency", "skill"], ascending=[False, True])
        .reset_index(drop=True)
    )
    vocab_table = freq_df[freq_df["skill"].isin(vocab)].reset_index(drop=True)
    rare_table = freq_df[~freq_df["skill"].isin(vocab)].reset_index(drop=True)

    return vocab, vocab_table, rare_table
