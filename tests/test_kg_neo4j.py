"""Live Neo4j tests: idempotent loading and schema setup. Skipped
cleanly (not a confusing error) if no Neo4j instance is reachable or
configured -- see src/kg/neo4j_client.py's Neo4jConfigError/
ConnectionError handling.

These tests write to whatever database NEO4J_DATABASE points at. To
avoid disturbing a database that might hold other data, they operate
inside their own uniquely-prefixed test nodes rather than touching the
main HireWise graph, and clean up after themselves.
"""
import unittest
import uuid

import pandas as pd

from src.kg.neo4j_client import Neo4jClient, Neo4jConfigError


def _neo4j_available() -> bool:
    try:
        with Neo4jClient():
            pass
        return True
    except (Neo4jConfigError, ConnectionError):
        return False


NEO4J_AVAILABLE = _neo4j_available()


@unittest.skipUnless(NEO4J_AVAILABLE, "Neo4j not configured/reachable (set NEO4J_URI/USER/PASSWORD and start the server)")
class TestNeo4jClientLive(unittest.TestCase):
    def test_verify_connectivity_and_simple_query(self):
        with Neo4jClient() as client:
            result = client.run("RETURN 1 AS x")
            self.assertEqual(result, [{"x": 1}])


@unittest.skipUnless(NEO4J_AVAILABLE, "Neo4j not configured/reachable (set NEO4J_URI/USER/PASSWORD and start the server)")
class TestIdempotentLoading(unittest.TestCase):
    """Uses an isolated, uniquely-tagged test label/id space so this test
    never touches or depends on the real HireWise graph content."""

    def setUp(self):
        self.tag = f"KGTEST_{uuid.uuid4().hex[:8]}"
        self.node_query = f"""
            UNWIND $rows AS row
            MERGE (n:{self.tag} {{id: row.id}})
            SET n.value = row.value
        """
        self.rel_query = f"""
            UNWIND $rows AS row
            MATCH (a:{self.tag} {{id: row.from_id}})
            MATCH (b:{self.tag} {{id: row.to_id}})
            MERGE (a)-[r:{self.tag}_REL {{event_id: row.event_id}}]->(b)
        """

    def tearDown(self):
        with Neo4jClient() as client:
            client.run_write(f"MATCH (n:{self.tag}) DETACH DELETE n")

    def test_merge_on_stable_id_does_not_duplicate_nodes(self):
        rows = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
        with Neo4jClient() as client:
            client.run_batched(self.node_query, rows)
            first_count = client.run(f"MATCH (n:{self.tag}) RETURN count(n) AS c")[0]["c"]
            client.run_batched(self.node_query, rows)  # load again
            second_count = client.run(f"MATCH (n:{self.tag}) RETURN count(n) AS c")[0]["c"]
        self.assertEqual(first_count, 2)
        self.assertEqual(second_count, 2)

    def test_merge_with_event_id_creates_distinct_relationships_but_stays_idempotent(self):
        nodes = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
        rels = [
            {"from_id": "a", "to_id": "b", "event_id": "evt-1"},
            {"from_id": "a", "to_id": "b", "event_id": "evt-2"},  # distinct event, same endpoints
        ]
        with Neo4jClient() as client:
            client.run_batched(self.node_query, nodes)
            client.run_batched(self.rel_query, rels)
            count_after_first_load = client.run(f"MATCH ()-[r:{self.tag}_REL]->() RETURN count(r) AS c")[0]["c"]
            client.run_batched(self.rel_query, rels)  # load again
            count_after_second_load = client.run(f"MATCH ()-[r:{self.tag}_REL]->() RETURN count(r) AS c")[0]["c"]

        self.assertEqual(count_after_first_load, 2)  # two distinct events preserved
        self.assertEqual(count_after_second_load, 2)  # rerun did not duplicate


if __name__ == "__main__":
    unittest.main()
