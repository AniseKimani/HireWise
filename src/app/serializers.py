"""Explicit, allow-listed JSON serializers (Day 9).

Nothing is serialized by reflection (no DataFrame rows, no `__dict__`):
every public field is named here, so a column added upstream can never
leak into a response by accident. NumPy scalars become Python numbers;
NaN/Infinity/missing become `null`.

Only model-visible data is exposed. Request lookups never include the
request's outcome (who was shortlisted or hired); worker profiles never
include generator-private variables.
"""
from __future__ import annotations

import math

import numpy as np

REQUEST_FIELDS = (
    "link", "title", "client_id", "category", "required_skills", "pay_type", "price_tier",
    "location_requirement", "simulated_day", "split",
)
WORKER_FIELDS = (
    "worker_id", "primary_category", "secondary_categories", "declared_skills", "experience_level",
    "experience_years", "price_tier", "rate", "country", "join_day",
    "training_review_count", "training_mean_rating",
)
RECOMMENDATION_FIELDS = (
    "rank", "worker", "score", "contributions", "dominant_component", "explanation", "kg_summary", "kg_evidence",
)


def json_safe(value):
    """Convert one scalar to a JSON-safe Python value."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"unsupported value type for JSON serialization: {type(value).__name__}")


def split_pipe(cell) -> list[str]:
    if isinstance(cell, str) and cell:
        return sorted(set(cell.split("|")))
    return []


def text_or_none(cell) -> str | None:
    return cell if isinstance(cell, str) and cell else None


def serialize_request(row: dict) -> dict:
    out = {
        "link": row["link"],
        "title": text_or_none(row.get("title")),
        "client_id": row["client_id"],
        "category": row["category"],
        "required_skills": split_pipe(row.get("skills_vocab")),
        "pay_type": text_or_none(row.get("pay_type")),
        "price_tier": json_safe(row.get("price_tier")),
        "location_requirement": text_or_none(row.get("location_requirement")),
        "simulated_day": json_safe(row.get("simulated_day")),
        "split": row["split"],
    }
    assert tuple(out) == REQUEST_FIELDS
    return out


def serialize_worker(row: dict, review_count: int, review_sum: float) -> dict:
    out = {
        "worker_id": row["worker_id"],
        "primary_category": row["primary_category"],
        "secondary_categories": split_pipe(row.get("secondary_categories")),
        "declared_skills": split_pipe(row.get("declared_skills")),
        "experience_level": text_or_none(row.get("experience_level")),
        "experience_years": json_safe(row.get("experience_years")),
        "price_tier": json_safe(row.get("price_tier")),
        "rate": json_safe(row.get("rate")),
        "country": text_or_none(row.get("country")),
        "join_day": json_safe(row.get("join_day")),
        "training_review_count": int(review_count),
        "training_mean_rating": json_safe(review_sum / review_count) if review_count else None,
    }
    assert tuple(out) == WORKER_FIELDS
    return out


def serialize_recommendation(explanation, worker: dict, kg_evidence_paths: int) -> dict:
    """`explanation` is a src.recommenders.hybrid.HybridExplanation."""
    out = {
        "rank": int(explanation.rank),
        "worker": worker,
        "score": json_safe(explanation.score),
        "contributions": {c: json_safe(explanation.contributions[c]) for c in ("cbf", "cf", "kg")},
        "dominant_component": explanation.dominant_component,
        "explanation": explanation.text,
        "kg_summary": explanation.kg_summary,
        "kg_evidence": [{"path_type": p["path_type"], "text": p["text"]} for p in explanation.kg_paths[:kg_evidence_paths]],
    }
    assert tuple(out) == RECOMMENDATION_FIELDS
    return out
