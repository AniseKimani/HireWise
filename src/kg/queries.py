"""Reusable, read-only, parameterized Cypher queries for verification,
statistics, and demonstration. Every function takes a `Neo4jClient` and
returns plain Python data; none of them write to the graph.
"""
from __future__ import annotations

from src.kg import schema
from src.kg.neo4j_client import Neo4jClient


def count_nodes_by_label(client: Neo4jClient) -> dict[str, int]:
    rows = client.run(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY label"
    )
    return {r["label"]: r["c"] for r in rows}


def count_relationships_by_type(client: Neo4jClient) -> dict[str, int]:
    rows = client.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY rel")
    return {r["rel"]: r["c"] for r in rows}


def historical_edge_split_counts(client: Neo4jClient, relationship_type: str) -> dict[str, int]:
    """Train/validation/test counts for a split-tagged historical
    relationship type (HIRED, REVIEWED has no split -- use HIRED/
    REQUESTED_SERVICE)."""
    rows = client.run(
        f"MATCH ()-[r:{relationship_type}]->() RETURN r.split AS split, count(*) AS c ORDER BY split"
    )
    return {r["split"]: r["c"] for r in rows}


def cold_start_training_hire_count(client: Neo4jClient) -> int:
    """Must be 0: no cold-start worker may have a training-split HIRED
    edge (docs/experimental_design.md Section 12)."""
    rows = client.run(
        f"""
        MATCH (c:{schema.LABEL_CLIENT})-[h:{schema.REL_HIRED}]->(w:{schema.LABEL_WORKER})
        WHERE w.is_cold_start = true AND h.split = 'train'
        RETURN count(h) AS c
        """
    )
    return rows[0]["c"]


def worker_skills(client: Neo4jClient, worker_id: str) -> list[str]:
    rows = client.run(
        f"""
        MATCH (w:{schema.LABEL_WORKER} {{worker_id: $worker_id}})-[:{schema.REL_HAS_SKILL}]->(s:{schema.LABEL_SKILL})
        RETURN s.name AS skill ORDER BY skill
        """,
        {"worker_id": worker_id},
    )
    return [r["skill"] for r in rows]


def client_hire_history(client: Neo4jClient, client_id: str) -> list[dict]:
    rows = client.run(
        f"""
        MATCH (c:{schema.LABEL_CLIENT} {{client_id: $client_id}})-[h:{schema.REL_HIRED}]->(w:{schema.LABEL_WORKER})
        RETURN w.worker_id AS worker_id, h.link AS link, h.simulated_day AS simulated_day, h.split AS split
        ORDER BY h.simulated_day
        """,
        {"client_id": client_id},
    )
    return rows


def workers_in_category(client: Neo4jClient, category: str, limit: int = 25) -> list[dict]:
    rows = client.run(
        f"""
        MATCH (w:{schema.LABEL_WORKER})-[r:{schema.REL_SPECIALIZES_IN}]->(cat:{schema.LABEL_CATEGORY} {{name: $category}})
        RETURN w.worker_id AS worker_id, r.role AS role ORDER BY r.role, w.worker_id LIMIT $limit
        """,
        {"category": category, "limit": limit},
    )
    return rows


def isolated_node_count(client: Neo4jClient, label: str) -> int:
    rows = client.run(f"MATCH (n:{label}) WHERE NOT (n)--() RETURN count(n) AS c")
    return rows[0]["c"]


def degree_stats(client: Neo4jClient, label: str) -> dict[str, float]:
    rows = client.run(
        f"""
        MATCH (n:{label})
        OPTIONAL MATCH (n)--()
        WITH n, count(*) AS degree
        RETURN min(degree) AS min, max(degree) AS max, avg(degree) AS mean,
               percentileDisc(degree, 0.5) AS median
        """
    )
    return rows[0]


def duplicate_domain_id_count(client: Neo4jClient, label: str, id_property: str) -> int:
    """Should always be 0 given the uniqueness constraints, but checked
    independently (a constraint violation would prevent this from ever
    happening at write time; this query re-derives it from scratch)."""
    rows = client.run(
        f"""
        MATCH (n:{label})
        WITH n[$id_property] AS id_value, count(*) AS c
        WHERE c > 1
        RETURN count(*) AS duplicates
        """,
        {"id_property": id_property},
    )
    return rows[0]["duplicates"]


def orphaned_relationship_endpoint_count(client: Neo4jClient) -> int:
    """Relationships in Neo4j cannot reference a nonexistent node by
    construction (unlike a foreign key in a relational table), so this
    always returns 0; kept as an explicit, documented sanity check rather
    than an assumption."""
    rows = client.run("MATCH ()-[r]->() WHERE startNode(r) IS NULL OR endNode(r) IS NULL RETURN count(r) AS c")
    return rows[0]["c"]
