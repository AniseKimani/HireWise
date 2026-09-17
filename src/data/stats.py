"""Real-data descriptive statistics for later modelling and reporting.

Every function here takes an explicit `population` (a boolean mask or a
pre-filtered frame) so that callers state, at the call site, which corpus
produced a given statistic. This matters for leakage prevention (see
docs/experimental_design.md Section 14): the generator's skill-relatedness
estimates are computed on the full in-scope corpus, while the Knowledge
Graph's `RELATED_TO` edges must be computed on the background corpus only
(eligible jobs not drawn into the 12,000-request evaluation sample), so
that KG relatedness is not literally identical to the generator's.
`scripts/run_data_pipeline.py` calls the co-occurrence function twice,
once per corpus, and labels the outputs accordingly.
"""
from __future__ import annotations

from itertools import combinations
from math import log

import numpy as np
import pandas as pd

from src.data.config import DataConfig


def category_frequencies(df: pd.DataFrame, population: pd.Series) -> pd.DataFrame:
    counts = df.loc[population, "category"].value_counts()
    table = counts.rename("count").reset_index().rename(columns={"index": "category"})
    table["share"] = table["count"] / table["count"].sum()
    return table.sort_values(["count", "category"], ascending=[False, True]).reset_index(drop=True)


def skills_per_job(df: pd.DataFrame, population: pd.Series, skill_col: str = "skills_vocab") -> dict:
    counts = df.loc[population, skill_col].map(len)
    quantiles = counts.quantile([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    return {
        "mean": float(counts.mean()),
        "quantiles": {str(k): float(v) for k, v in quantiles.items()},
        "zero_count": int((counts == 0).sum()),
        "n_jobs": int(population.sum()),
    }


def skill_frequencies(df: pd.DataFrame, population: pd.Series, skill_col: str = "skills_vocab") -> pd.DataFrame:
    from collections import Counter

    freq = Counter(skill for skills in df.loc[population, skill_col] for skill in skills)
    table = pd.DataFrame(freq.items(), columns=["skill", "frequency"])
    return table.sort_values(["frequency", "skill"], ascending=[False, True]).reset_index(drop=True)


def pay_percentiles(df: pd.DataFrame, population: pd.Series, config: DataConfig) -> dict:
    """Stored (unmodified) vs analysis (winsorised) percentiles, per pay
    type. Winsorisation is for descriptive statistics/figures only; it
    never overwrites a stored value (see clean.py's price_tier, which is
    percentile-rank based and needs no winsorisation)."""
    result = {}
    for pay_type in ["fixed", "hourly", "unspecified"]:
        subset = df.loc[population & (df["pay_type"] == pay_type), "pay_amount"].dropna()
        if subset.empty:
            result[pay_type] = {"n": 0}
            continue
        low, high = subset.quantile([config.winsorize_low_percentile, config.winsorize_high_percentile])
        winsorized = subset.clip(low, high)
        result[pay_type] = {
            "n": int(len(subset)),
            "stored_quantiles": {
                str(q): float(v) for q, v in subset.quantile([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).items()
            },
            "stored_mean": float(subset.mean()),
            "winsorized_mean": float(winsorized.mean()),
            "winsorize_bounds": {"low": float(low), "high": float(high)},
            "n_above_high_bound": int((subset > high).sum()),
        }
    return result


def category_stats(df: pd.DataFrame, population: pd.Series) -> pd.DataFrame:
    subset = df.loc[population]
    grouped = subset.groupby("category")
    table = grouped.agg(
        n_jobs=("link", "count"),
        mean_skills_vocab=("skills_vocab", lambda s: s.map(len).mean()),
        median_price_tier=("price_tier", "median"),
        pct_hourly=("pay_type", lambda s: (s == "hourly").mean()),
        pct_fixed=("pay_type", lambda s: (s == "fixed").mean()),
    ).reset_index()
    return table.sort_values(["n_jobs", "category"], ascending=[False, True]).reset_index(drop=True)


def skill_cooccurrence(
    df: pd.DataFrame,
    population: pd.Series,
    config: DataConfig,
    skill_col: str = "skills_vocab",
) -> pd.DataFrame:
    """Pairwise NPMI co-occurrence among controlled-vocabulary skills
    within `population`, restricted to pairs meeting
    `config.cooccurrence_min_support` and `config.cooccurrence_min_npmi`.

    NPMI(i, j) = PMI(i, j) / -log(p(i, j)), which lies in (-1, 1]; higher
    means the pair co-occurs more than chance. Computed independently for
    whatever `population` the caller passes -- see the module docstring on
    why the generator and the Knowledge Graph must use different corpora.
    """
    from collections import Counter

    skill_lists = df.loc[population, skill_col]
    n_jobs = len(skill_lists)
    unigram = Counter(skill for skills in skill_lists for skill in skills)
    pair_counts = Counter()
    for skills in skill_lists:
        uniq = sorted(set(skills))
        for a, b in combinations(uniq, 2):
            pair_counts[(a, b)] += 1

    rows = []
    for (a, b), co in pair_counts.items():
        if co < config.cooccurrence_min_support:
            continue
        p_a = unigram[a] / n_jobs
        p_b = unigram[b] / n_jobs
        p_ab = co / n_jobs
        pmi = log(p_ab / (p_a * p_b))
        npmi = pmi / -log(p_ab)
        if npmi > config.cooccurrence_min_npmi:
            rows.append((a, b, co, npmi))

    table = pd.DataFrame(rows, columns=["skill_a", "skill_b", "support", "npmi"])
    return table.sort_values(["npmi", "skill_a", "skill_b"], ascending=[False, True, True]).reset_index(drop=True)


def category_similarity(
    df: pd.DataFrame,
    population: pd.Series,
    top_k: int = 5,
    skill_col: str = "skills_vocab",
) -> pd.DataFrame:
    """Cosine similarity between categories' controlled-vocabulary skill
    frequency profiles. Returns the top-k most similar other categories
    for each category, long-format, deterministically ordered.

    This is a real-data statistic that Day 3's synthetic worker generator
    will use to choose secondary categories (docs/experimental_design.md
    Section 8); it is computed here, not on Day 3, because it depends only
    on the real job corpus.
    """
    from collections import Counter

    subset = df.loc[population, ["category", skill_col]]
    categories = sorted(subset["category"].dropna().unique())
    skills = sorted({s for lst in subset[skill_col] for s in lst})
    skill_index = {s: i for i, s in enumerate(skills)}

    matrix = np.zeros((len(categories), len(skills)))
    cat_index = {c: i for i, c in enumerate(categories)}
    for category, skill_list in zip(subset["category"], subset[skill_col]):
        if pd.isna(category):
            continue
        row = cat_index[category]
        counts = Counter(skill_list)
        for skill, count in counts.items():
            matrix[row, skill_index[skill]] += count

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    similarity = normalized @ normalized.T

    rows = []
    for i, category in enumerate(categories):
        order = sorted(
            (j for j in range(len(categories)) if j != i),
            key=lambda j: (-similarity[i, j], categories[j]),
        )[:top_k]
        for rank, j in enumerate(order, start=1):
            rows.append((category, categories[j], rank, float(similarity[i, j])))

    return pd.DataFrame(rows, columns=["category", "similar_category", "rank", "cosine_similarity"])


def location_availability(df: pd.DataFrame, population: pd.Series) -> dict:
    subset = df.loc[population]
    return {
        "n_jobs": int(len(subset)),
        "country_missing": int(subset["country_clean"].isna().sum()),
        "country_unique": int(subset["country_clean"].nunique()),
        "top_countries_share": {
            str(k): float(v) for k, v in (subset["country_clean"].value_counts(normalize=True).head(10)).items()
        },
        "has_location_requirement": int(subset["location_requirement"].notna().sum()),
        "has_location_requirement_share": float(subset["location_requirement"].notna().mean()),
    }
