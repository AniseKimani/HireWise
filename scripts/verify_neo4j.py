"""Independent live-graph verification: connects to Neo4j and compares
graph counts/invariants directly against the source model-visible CSVs
(not against the loader's own reported counts).

Usage:
    python scripts/verify_neo4j.py

Fails with a clear, actionable message (not a bare traceback) if Neo4j
is unreachable or misconfigured. Exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.kg import queries, schema
from src.kg.neo4j_client import Neo4jClient, Neo4jConfigError

DAY3_SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
DAY2_PROCESSED = REPO_ROOT / "data" / "processed"

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    try:
        client_cm = Neo4jClient()
        client = client_cm.__enter__()
    except Neo4jConfigError as exc:
        print(f"Cannot run: {exc}")
        return 1
    except ConnectionError as exc:
        print(f"Cannot run: {exc}")
        return 1

    try:
        workers = pd.read_csv(DAY3_SYNTHETIC / "workers.csv")
        clients = pd.read_csv(DAY3_SYNTHETIC / "clients.csv")
        interactions = pd.read_csv(DAY3_SYNTHETIC / "interactions.csv")
        requests = pd.read_csv(DAY3_SYNTHETIC / "requests.csv")
        vocab = pd.read_csv(DAY2_PROCESSED / "skills_vocab.csv")
        cold_start = pd.read_csv(DAY3_SYNTHETIC / "cold_start_cohort.csv")

        node_counts = queries.count_nodes_by_label(client)
        rel_counts = queries.count_relationships_by_type(client)

        check("Client node count matches clients.csv", node_counts.get(schema.LABEL_CLIENT) == len(clients),
              f"graph={node_counts.get(schema.LABEL_CLIENT)}, source={len(clients)}")
        check("Worker node count matches workers.csv", node_counts.get(schema.LABEL_WORKER) == len(workers),
              f"graph={node_counts.get(schema.LABEL_WORKER)}, source={len(workers)}")
        check("Skill node count matches skills_vocab.csv", node_counts.get(schema.LABEL_SKILL) == len(vocab),
              f"graph={node_counts.get(schema.LABEL_SKILL)}, source={len(vocab)}")

        n_hired_source = int((interactions["event"] == "HIRED").sum())
        check("HIRED relationship count matches source interactions", rel_counts.get(schema.REL_HIRED) == n_hired_source,
              f"graph={rel_counts.get(schema.REL_HIRED)}, source={n_hired_source}")

        req_split_graph = queries.historical_edge_split_counts(client, schema.REL_REQUESTED_SERVICE)
        req_split_source = requests["split"].value_counts().to_dict()
        check("REQUESTED_SERVICE split counts match requests.csv",
              req_split_graph == req_split_source, f"graph={req_split_graph}, source={req_split_source}")

        check("Zero cold-start workers have a training HIRED edge",
              queries.cold_start_training_hire_count(client) == 0)

        check("No duplicate Client.client_id in the live graph",
              queries.duplicate_domain_id_count(client, schema.LABEL_CLIENT, "client_id") == 0)
        check("No duplicate Worker.worker_id in the live graph",
              queries.duplicate_domain_id_count(client, schema.LABEL_WORKER, "worker_id") == 0)
        check("No orphaned relationship endpoints", queries.orphaned_relationship_endpoint_count(client) == 0)

        sample_worker = workers["worker_id"].iloc[0]
        expected_skills = set(workers.loc[workers["worker_id"] == sample_worker, "declared_skills"].iloc[0].split("|"))
        graph_skills = set(queries.worker_skills(client, sample_worker))
        check(f"Sample worker {sample_worker}'s graph skills match declared_skills",
              graph_skills == expected_skills, f"graph={graph_skills}, source={expected_skills}")

        print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
        return 1 if FAILURES else 0
    finally:
        client_cm.__exit__(None, None, None)


if __name__ == "__main__":
    sys.exit(main())
