"""Idempotent, batched loading of prepared KG data into Neo4j.

Every write uses `MERGE` on a stable domain identifier (never a Neo4j
internal node ID) via `UNWIND $rows AS row`, so running the loader twice
with the same input never duplicates nodes or relationships. Per-event
relationships (`REQUESTED_SERVICE`, `HIRED`, `REVIEWED`) include the
request `link` inside the `MERGE` pattern itself, so distinct events
between the same two nodes create distinct relationships while re-running
the same event stays idempotent.
"""
from __future__ import annotations

import pandas as pd

from src.kg import schema
from src.kg.neo4j_client import Neo4jClient
from src.kg.prepare import KGData

_NODE_QUERIES = {
    "clients": f"""
        UNWIND $rows AS row
        MERGE (c:{schema.LABEL_CLIENT} {{client_id: row.client_id}})
        SET c.preferred_pay_type = row.preferred_pay_type, c.median_price_tier = row.median_price_tier
    """,
    "workers": f"""
        UNWIND $rows AS row
        MERGE (w:{schema.LABEL_WORKER} {{worker_id: row.worker_id}})
        SET w.experience_level = row.experience_level, w.experience_years = row.experience_years,
            w.price_tier = row.price_tier, w.rate = row.rate, w.join_day = row.join_day,
            w.is_cold_start = row.is_cold_start
    """,
    "skills": f"""
        UNWIND $rows AS row
        MERGE (s:{schema.LABEL_SKILL} {{name: row.name}})
    """,
    "categories": f"""
        UNWIND $rows AS row
        MERGE (cat:{schema.LABEL_CATEGORY} {{name: row.name}})
    """,
    "locations": f"""
        UNWIND $rows AS row
        MERGE (l:{schema.LABEL_LOCATION} {{name: row.name}})
    """,
}

_RELATIONSHIP_QUERIES = {
    "has_skill": f"""
        UNWIND $rows AS row
        MATCH (w:{schema.LABEL_WORKER} {{worker_id: row.worker_id}})
        MATCH (s:{schema.LABEL_SKILL} {{name: row.skill}})
        MERGE (w)-[:{schema.REL_HAS_SKILL}]->(s)
    """,
    "specializes_in": f"""
        UNWIND $rows AS row
        MATCH (w:{schema.LABEL_WORKER} {{worker_id: row.worker_id}})
        MATCH (cat:{schema.LABEL_CATEGORY} {{name: row.category}})
        MERGE (w)-[r:{schema.REL_SPECIALIZES_IN}]->(cat)
        SET r.role = row.role
    """,
    "located_in_worker": f"""
        UNWIND $rows AS row
        MATCH (w:{schema.LABEL_WORKER} {{worker_id: row.worker_id}})
        MATCH (l:{schema.LABEL_LOCATION} {{name: row.location}})
        MERGE (w)-[:{schema.REL_LOCATED_IN}]->(l)
    """,
    "located_in_client": f"""
        UNWIND $rows AS row
        MATCH (c:{schema.LABEL_CLIENT} {{client_id: row.client_id}})
        MATCH (l:{schema.LABEL_LOCATION} {{name: row.location}})
        MERGE (c)-[:{schema.REL_LOCATED_IN}]->(l)
    """,
    "requested_service": f"""
        UNWIND $rows AS row
        MATCH (c:{schema.LABEL_CLIENT} {{client_id: row.client_id}})
        MATCH (cat:{schema.LABEL_CATEGORY} {{name: row.category}})
        MERGE (c)-[r:{schema.REL_REQUESTED_SERVICE} {{link: row.link}}]->(cat)
        SET r.simulated_day = row.simulated_day, r.split = row.split
    """,
    "interested_in": f"""
        UNWIND $rows AS row
        MATCH (c:{schema.LABEL_CLIENT} {{client_id: row.client_id}})
        MATCH (cat:{schema.LABEL_CATEGORY} {{name: row.category}})
        MERGE (c)-[r:{schema.REL_INTERESTED_IN}]->(cat)
        SET r.training_request_count = row.training_request_count
    """,
    "hired": f"""
        UNWIND $rows AS row
        MATCH (c:{schema.LABEL_CLIENT} {{client_id: row.client_id}})
        MATCH (w:{schema.LABEL_WORKER} {{worker_id: row.worker_id}})
        MERGE (c)-[r:{schema.REL_HIRED} {{link: row.link}}]->(w)
        SET r.simulated_day = row.simulated_day, r.split = row.split
    """,
    "reviewed": f"""
        UNWIND $rows AS row
        MATCH (c:{schema.LABEL_CLIENT} {{client_id: row.client_id}})
        MATCH (w:{schema.LABEL_WORKER} {{worker_id: row.worker_id}})
        MERGE (c)-[r:{schema.REL_REVIEWED} {{link: row.link}}]->(w)
        SET r.rating = row.rating, r.has_review = row.has_review
    """,
    "related_to": f"""
        UNWIND $rows AS row
        MATCH (a:{schema.LABEL_SKILL} {{name: row.skill_a}})
        MATCH (b:{schema.LABEL_SKILL} {{name: row.skill_b}})
        MERGE (a)-[r:{schema.REL_RELATED_TO}]->(b)
        SET r.npmi = row.npmi, r.support = row.support
    """,
}

# Relationships must load after both endpoint node types exist.
NODE_LOAD_ORDER = ["clients", "workers", "skills", "categories", "locations"]
RELATIONSHIP_LOAD_ORDER = [
    "has_skill", "specializes_in", "located_in_worker", "located_in_client",
    "requested_service", "interested_in", "hired", "reviewed", "related_to",
]


def _df_to_rows(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notnull(df), None).to_dict("records")


def apply_schema(client: Neo4jClient) -> None:
    """Idempotent constraint/index setup; safe to run repeatedly."""
    for statement in schema.ALL_SCHEMA_STATEMENTS:
        client.run_write(statement)


def rebuild(client: Neo4jClient) -> None:
    """Destructive: deletes every node and relationship this schema's
    labels/types touch. Only called when the loader script is run with
    an explicit --rebuild flag -- never implicitly."""
    client.run_write("MATCH (n) DETACH DELETE n")


def load_nodes(client: Neo4jClient, kg_data: KGData, batch_size: int = 500) -> dict[str, int]:
    counts = {}
    for field_name in NODE_LOAD_ORDER:
        df = getattr(kg_data, field_name)
        counts[field_name] = client.run_batched(_NODE_QUERIES[field_name], _df_to_rows(df), batch_size)
    return counts


def load_relationships(client: Neo4jClient, kg_data: KGData, batch_size: int = 500) -> dict[str, int]:
    counts = {}
    for field_name in RELATIONSHIP_LOAD_ORDER:
        df = getattr(kg_data, field_name)
        counts[field_name] = client.run_batched(_RELATIONSHIP_QUERIES[field_name], _df_to_rows(df), batch_size)
    return counts


def load_all(client: Neo4jClient, kg_data: KGData, rebuild_first: bool = False, batch_size: int = 500) -> dict:
    if rebuild_first:
        rebuild(client)
    apply_schema(client)
    node_counts = load_nodes(client, kg_data, batch_size)
    relationship_counts = load_relationships(client, kg_data, batch_size)
    return {"nodes": node_counts, "relationships": relationship_counts}
