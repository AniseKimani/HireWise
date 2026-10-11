"""Hand-checkable toy knowledge graph for Day 7 tests.

Skills A, B, C; RELATED_TO A-B (npmi 0.5) and B-C (npmi 0.8).
Workers (all eligible for category "Cat"):
    W1 {A, B} primary      W2 {C} secondary
    W3 {A}    primary      W4 {B} primary  (no history at all: cold-start-like)
Training hires: C1->W3 (t1, day 1), C2->W3 (t2, day 2), C2->W1 (t3, day 3).
Training reviews: C1->W3 rating 5 (t1), C2->W1 rating 3 (t3).
Validation/test edges (must never appear in the view): C1->W2 hire v1
(validation, day 20) with rating 1, C2->W4 hire x1 (test, day 30) with
rating 1, and an orphan review C1->W4 with no hire at all.
Request r1: client C1, category Cat, skills "A", day 10.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.kg.graph_view import build_view

ROLE_WEIGHTS = {"primary": 1.0, "secondary": 0.5}
EQUAL_WEIGHTS = {"skill": 0.2, "related_skill": 0.2, "category": 0.2, "history": 0.2, "reputation": 0.2}

WORKERS = ["W1", "W2", "W3", "W4"]
SKILLS = ["A", "B", "C"]
HAS_SKILL = [("W1", "A"), ("W1", "B"), ("W2", "C"), ("W3", "A"), ("W4", "B")]
SPECIALIZES_IN = [("W1", "Cat", "primary"), ("W2", "Cat", "secondary"), ("W3", "Cat", "primary"), ("W4", "Cat", "primary")]
RELATED_TO = [("A", "B", 0.5, 10), ("B", "C", 0.8, 12)]
HIRED_ALL = [
    ("C1", "W3", "t1", 1.0, "train"),
    ("C2", "W3", "t2", 2.0, "train"),
    ("C2", "W1", "t3", 3.0, "train"),
    ("C1", "W2", "v1", 20.0, "validation"),
    ("C2", "W4", "x1", 30.0, "test"),
]
REVIEWED_ALL = [
    ("C1", "W3", "t1", 5, True),
    ("C2", "W1", "t3", 3, False),
    ("C1", "W2", "v1", 1, True),
    ("C2", "W4", "x1", 1, True),
    ("C1", "W4", "orphan", 1, True),
]


def toy_view():
    train_hires = [(c, w, l, d) for c, w, l, d, s in HIRED_ALL if s == "train"]
    train_days = {l: d for _, _, l, d in train_hires}
    train_reviews = [(c, w, l, r, train_days[l]) for c, w, l, r, _ in REVIEWED_ALL if l in train_days]
    return build_view(
        workers=WORKERS, skills=SKILLS, categories=["Cat"], has_skill=HAS_SKILL, specializes_in=SPECIALIZES_IN,
        related_to=RELATED_TO, train_hires=train_hires, train_reviews=train_reviews,
        interested_in=[("C2", "Cat", 2)], source="toy",
    )


def toy_requests(extra: list[dict] | None = None) -> pd.DataFrame:
    rows = [
        {"link": "r1", "client_id": "C1", "category": "Cat", "simulated_day": 10.0, "skills_vocab": "A", "split": "validation"},
        {"link": "r_early", "client_id": "C1", "category": "Cat", "simulated_day": 2.5, "skills_vocab": "A", "split": "train"},
        {"link": "t1", "client_id": "C1", "category": "Cat", "simulated_day": 1.0, "skills_vocab": "A", "split": "train"},
    ]
    return pd.DataFrame(rows + (extra or []))


def write_toy_kg_csvs(out_dir: Path, hired=None, reviewed=None) -> Path:
    """Day 6-format CSV artifacts for the toy graph, including the
    validation/test edges the CSV loader must filter out."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hired = HIRED_ALL if hired is None else hired
    reviewed = REVIEWED_ALL if reviewed is None else reviewed
    pd.DataFrame({
        "worker_id": WORKERS, "experience_level": "entry", "experience_years": 1.0, "price_tier": 0.5,
        "rate": 10.0, "join_day": [0.0, 0.0, 0.0, 15.0], "is_cold_start": [False, False, False, True],
    }).to_csv(out_dir / "nodes_worker.csv", index=False)
    pd.DataFrame({"name": SKILLS}).to_csv(out_dir / "nodes_skill.csv", index=False)
    pd.DataFrame({"name": ["Cat"]}).to_csv(out_dir / "nodes_category.csv", index=False)
    pd.DataFrame(HAS_SKILL, columns=["worker_id", "skill"]).to_csv(out_dir / "rel_has_skill.csv", index=False)
    pd.DataFrame(SPECIALIZES_IN, columns=["worker_id", "category", "role"]).to_csv(out_dir / "rel_specializes_in.csv", index=False)
    pd.DataFrame(RELATED_TO, columns=["skill_a", "skill_b", "npmi", "support"]).to_csv(out_dir / "rel_related_to.csv", index=False)
    pd.DataFrame(hired, columns=["client_id", "worker_id", "link", "simulated_day", "split"]).to_csv(out_dir / "rel_hired.csv", index=False)
    pd.DataFrame(reviewed, columns=["client_id", "worker_id", "link", "rating", "has_review"]).to_csv(out_dir / "rel_reviewed.csv", index=False)
    pd.DataFrame([("C2", "Cat", 2)], columns=["client_id", "category", "training_request_count"]).to_csv(
        out_dir / "rel_interested_in.csv", index=False
    )
    pd.DataFrame([("C1", "Cat", "r1", 10.0, "validation")], columns=["client_id", "category", "link", "simulated_day", "split"]).to_csv(
        out_dir / "rel_requested_service.csv", index=False
    )
    return out_dir
