"""Deterministic graph-ready node/relationship table preparation.

Pure pandas: no Neo4j dependency, so this module (and its tests) run
without a live database. Reads only model-visible files under
`data/processed/` and `data/processed/synthetic/`; never
`data/processed/synthetic_private/`.

Every `prepare_*` function returns a plain DataFrame whose columns are
checked against `src/kg/schema.py`'s allow-lists before being handed to
the loader, so a column added by mistake fails loudly here rather than
silently reaching Neo4j.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.kg import schema

REPO_ROOT = Path(__file__).resolve().parents[2]
DAY2_PROCESSED = REPO_ROOT / "data" / "processed"
DAY3_SYNTHETIC = DAY2_PROCESSED / "synthetic"

MIN_TRAINING_REQUESTS_FOR_INTEREST = 2  # docs/experimental_design.md Section 14


def normalize_location(name: object) -> str | None:
    """Deterministic location key: HTML-entity decoded, whitespace
    collapsed and stripped. Returns None for missing/blank input, so
    callers can drop rows with no location rather than creating a
    placeholder node."""
    if not isinstance(name, str):
        return None
    decoded = html.unescape(name)
    collapsed = " ".join(decoded.split())
    return collapsed or None


def _assert_allowed_columns(df: pd.DataFrame, allowed: set[str], id_columns: set[str], context: str) -> None:
    extra = set(df.columns) - allowed - id_columns
    if extra:
        raise ValueError(f"{context}: disallowed column(s) {extra} not in schema allow-list")


@dataclass
class KGData:
    # Nodes
    clients: pd.DataFrame
    workers: pd.DataFrame
    skills: pd.DataFrame
    categories: pd.DataFrame
    locations: pd.DataFrame
    # Relationships
    has_skill: pd.DataFrame
    specializes_in: pd.DataFrame
    located_in_worker: pd.DataFrame
    located_in_client: pd.DataFrame
    requested_service: pd.DataFrame
    interested_in: pd.DataFrame
    hired: pd.DataFrame
    reviewed: pd.DataFrame
    related_to: pd.DataFrame


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def prepare_client_nodes(clients: pd.DataFrame) -> pd.DataFrame:
    out = clients[["client_id", "preferred_pay_type", "median_price_tier"]].drop_duplicates("client_id")
    out = out.sort_values("client_id").reset_index(drop=True)
    _assert_allowed_columns(out, schema.ALLOWED_NODE_PROPERTIES[schema.LABEL_CLIENT], set(), "Client nodes")
    return out


def prepare_worker_nodes(workers: pd.DataFrame) -> pd.DataFrame:
    cols = ["worker_id", "experience_level", "experience_years", "price_tier", "rate", "join_day", "is_cold_start"]
    out = workers[cols].drop_duplicates("worker_id").sort_values("worker_id").reset_index(drop=True)
    _assert_allowed_columns(out, schema.ALLOWED_NODE_PROPERTIES[schema.LABEL_WORKER], set(), "Worker nodes")
    return out


def prepare_skill_nodes(vocab: pd.DataFrame) -> pd.DataFrame:
    """All 1,187 controlled-vocabulary skills, not only those actually
    connected in the sampled 12,000-request graph population (Day 6
    brief Section 13): the vocabulary is a stable, closed set, and future
    requests may use any vocabulary skill even if the current sample
    doesn't. No relationship is fabricated for an unused skill -- it
    simply exists as an isolated node, which src/kg/queries.py's
    isolated-node check reports rather than hides."""
    out = pd.DataFrame({"name": sorted(vocab["skill"].unique())})
    _assert_allowed_columns(out, schema.ALLOWED_NODE_PROPERTIES[schema.LABEL_SKILL], set(), "Skill nodes")
    return out


def prepare_category_nodes(category_stats: pd.DataFrame) -> pd.DataFrame:
    """All 88 in-scope categories (same rationale as skills)."""
    out = pd.DataFrame({"name": sorted(category_stats["category"].unique())})
    _assert_allowed_columns(out, schema.ALLOWED_NODE_PROPERTIES[schema.LABEL_CATEGORY], set(), "ServiceCategory nodes")
    return out


def prepare_location_nodes(clients: pd.DataFrame, workers: pd.DataFrame) -> pd.DataFrame:
    client_locations = clients["country"].map(normalize_location).dropna()
    worker_locations = workers["country"].map(normalize_location).dropna()
    names = sorted(set(client_locations) | set(worker_locations))
    out = pd.DataFrame({"name": names})
    _assert_allowed_columns(out, schema.ALLOWED_NODE_PROPERTIES[schema.LABEL_LOCATION], set(), "Location nodes")
    return out


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

def prepare_has_skill_edges(workers: pd.DataFrame, vocab_skills: set[str]) -> pd.DataFrame:
    rows = []
    for worker_id, cell in zip(workers["worker_id"], workers["declared_skills"]):
        if not isinstance(cell, str) or not cell:
            continue
        for skill in cell.split("|"):
            if skill in vocab_skills:  # defensive; Day 3 already restricts to vocab
                rows.append((worker_id, skill))
    out = pd.DataFrame(rows, columns=["worker_id", "skill"]).drop_duplicates()
    out = out.sort_values(["worker_id", "skill"]).reset_index(drop=True)
    _assert_allowed_columns(out, schema.ALLOWED_RELATIONSHIP_PROPERTIES[schema.REL_HAS_SKILL], {"worker_id", "skill"}, "HAS_SKILL edges")
    return out


def prepare_specializes_in_edges(workers: pd.DataFrame, category_names: set[str]) -> pd.DataFrame:
    rows = []
    for worker_id, primary, secondary in zip(workers["worker_id"], workers["primary_category"], workers["secondary_categories"]):
        if primary in category_names:
            rows.append((worker_id, primary, "primary"))
        if isinstance(secondary, str) and secondary:
            for cat in secondary.split("|"):
                if cat in category_names:
                    rows.append((worker_id, cat, "secondary"))
    out = pd.DataFrame(rows, columns=["worker_id", "category", "role"]).drop_duplicates(["worker_id", "category"])
    out = out.sort_values(["worker_id", "category"]).reset_index(drop=True)
    _assert_allowed_columns(out, schema.ALLOWED_RELATIONSHIP_PROPERTIES[schema.REL_SPECIALIZES_IN], {"worker_id", "category"}, "SPECIALIZES_IN edges")
    return out


def prepare_located_in_edges(entities: pd.DataFrame, id_column: str) -> pd.DataFrame:
    out = entities[[id_column, "country"]].copy()
    out["location"] = out["country"].map(normalize_location)
    out = out.dropna(subset=["location"]).drop(columns=["country"])
    out = out.drop_duplicates(id_column).sort_values(id_column).reset_index(drop=True)
    _assert_allowed_columns(out, schema.ALLOWED_RELATIONSHIP_PROPERTIES[schema.REL_LOCATED_IN], {id_column, "location"}, "LOCATED_IN edges")
    return out


def prepare_requested_service_edges(requests: pd.DataFrame, sampled_requests: pd.DataFrame) -> pd.DataFrame:
    merged = requests.merge(sampled_requests[["link", "category"]], on="link", how="left")
    out = merged[["client_id", "category", "link", "simulated_day", "split"]].copy()
    out = out.sort_values("link").reset_index(drop=True)
    _assert_allowed_columns(
        out, schema.ALLOWED_RELATIONSHIP_PROPERTIES[schema.REL_REQUESTED_SERVICE],
        {"client_id", "category"}, "REQUESTED_SERVICE edges",
    )
    return out


def prepare_interested_in_edges(requested_service: pd.DataFrame) -> pd.DataFrame:
    """Derived, TRAINING-ONLY (docs/experimental_design.md Section 14):
    a client is INTERESTED_IN a category once they have placed at least
    `MIN_TRAINING_REQUESTS_FOR_INTEREST` training-split requests in it.
    Never derived from validation/test requests or generator variables."""
    train_only = requested_service.loc[requested_service["split"] == "train"]
    counts = train_only.groupby(["client_id", "category"]).size().reset_index(name="training_request_count")
    out = counts.loc[counts["training_request_count"] >= MIN_TRAINING_REQUESTS_FOR_INTEREST]
    out = out.sort_values(["client_id", "category"]).reset_index(drop=True)
    _assert_allowed_columns(
        out, schema.ALLOWED_RELATIONSHIP_PROPERTIES[schema.REL_INTERESTED_IN],
        {"client_id", "category"}, "INTERESTED_IN edges",
    )
    return out


def prepare_hired_edges(interactions: pd.DataFrame, requests: pd.DataFrame) -> pd.DataFrame:
    hired = interactions.loc[interactions["event"] == "HIRED", ["link", "client_id", "worker_id", "simulated_day"]]
    merged = hired.merge(requests[["link", "split"]], on="link", how="left")
    out = merged[["client_id", "worker_id", "link", "simulated_day", "split"]].sort_values("link").reset_index(drop=True)
    _assert_allowed_columns(
        out, schema.ALLOWED_RELATIONSHIP_PROPERTIES[schema.REL_HIRED],
        {"client_id", "worker_id"}, "HIRED edges",
    )
    return out


def prepare_reviewed_edges(ratings: pd.DataFrame, requests: pd.DataFrame) -> pd.DataFrame:
    merged = ratings.merge(requests[["link", "client_id"]], on="link", how="left")
    out = merged[["client_id", "worker_id", "link", "rating", "has_review"]].sort_values("link").reset_index(drop=True)
    _assert_allowed_columns(
        out, schema.ALLOWED_RELATIONSHIP_PROPERTIES[schema.REL_REVIEWED],
        {"client_id", "worker_id"}, "REVIEWED edges",
    )
    return out


def prepare_related_to_edges(cooccurrence_background: pd.DataFrame, vocab_skills: set[str]) -> pd.DataFrame:
    """Background-corpus skill co-occurrence only (docs/experimental_design.md
    Section 14: kept independent of the generator's own full-corpus
    co-occurrence, which is used by the Day 3 generator's true-skill-fit
    component, so the two are correlated but not identical)."""
    out = cooccurrence_background.loc[
        cooccurrence_background["skill_a"].isin(vocab_skills) & cooccurrence_background["skill_b"].isin(vocab_skills)
    ]
    out = out[["skill_a", "skill_b", "support", "npmi"]].sort_values(["skill_a", "skill_b"]).reset_index(drop=True)
    _assert_allowed_columns(
        out, schema.ALLOWED_RELATIONSHIP_PROPERTIES[schema.REL_RELATED_TO],
        {"skill_a", "skill_b"}, "RELATED_TO edges",
    )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def prepare_all(day2_dir: Path = DAY2_PROCESSED, day3_dir: Path = DAY3_SYNTHETIC) -> KGData:
    clients = pd.read_csv(day3_dir / "clients.csv")
    workers = pd.read_csv(day3_dir / "workers.csv")
    requests = pd.read_csv(day3_dir / "requests.csv")
    interactions = pd.read_csv(day3_dir / "interactions.csv")
    ratings = pd.read_csv(day3_dir / "ratings.csv")
    sampled_requests = pd.read_csv(day2_dir / "sampled_requests.csv")
    vocab = pd.read_csv(day2_dir / "skills_vocab.csv")
    category_stats = pd.read_csv(day2_dir / "category_stats.csv")
    cooccurrence_background = pd.read_csv(day2_dir / "skill_cooccurrence_background_corpus.csv")

    vocab_skills = set(vocab["skill"])
    category_names = set(category_stats["category"])

    client_nodes = prepare_client_nodes(clients)
    worker_nodes = prepare_worker_nodes(workers)
    skill_nodes = prepare_skill_nodes(vocab)
    category_nodes = prepare_category_nodes(category_stats)
    location_nodes = prepare_location_nodes(clients, workers)

    has_skill = prepare_has_skill_edges(workers, vocab_skills)
    specializes_in = prepare_specializes_in_edges(workers, category_names)
    located_in_worker = prepare_located_in_edges(workers, "worker_id")
    located_in_client = prepare_located_in_edges(clients, "client_id")
    requested_service = prepare_requested_service_edges(requests, sampled_requests)
    interested_in = prepare_interested_in_edges(requested_service)
    hired = prepare_hired_edges(interactions, requests)
    reviewed = prepare_reviewed_edges(ratings, requests)
    related_to = prepare_related_to_edges(cooccurrence_background, vocab_skills)

    return KGData(
        clients=client_nodes, workers=worker_nodes, skills=skill_nodes,
        categories=category_nodes, locations=location_nodes,
        has_skill=has_skill, specializes_in=specializes_in,
        located_in_worker=located_in_worker, located_in_client=located_in_client,
        requested_service=requested_service, interested_in=interested_in,
        hired=hired, reviewed=reviewed, related_to=related_to,
    )


KG_FILENAMES = {
    "clients": "nodes_client.csv", "workers": "nodes_worker.csv", "skills": "nodes_skill.csv",
    "categories": "nodes_category.csv", "locations": "nodes_location.csv",
    "has_skill": "rel_has_skill.csv", "specializes_in": "rel_specializes_in.csv",
    "located_in_worker": "rel_located_in_worker.csv", "located_in_client": "rel_located_in_client.csv",
    "requested_service": "rel_requested_service.csv", "interested_in": "rel_interested_in.csv",
    "hired": "rel_hired.csv", "reviewed": "rel_reviewed.csv", "related_to": "rel_related_to.csv",
}


def save_kg_data(kg_data: KGData, out_dir: Path) -> list[Path]:
    from src.data.io import save_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for field_name, filename in KG_FILENAMES.items():
        df = getattr(kg_data, field_name)
        paths.append(save_csv(df, out_dir / filename))
    return paths
