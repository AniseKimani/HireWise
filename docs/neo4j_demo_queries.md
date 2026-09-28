# Neo4j Browser Demo Queries

Safe, read-only Cypher examples for exploring the HireWise knowledge
graph in Neo4j Browser -- useful for screenshots and demonstration.
These show graph **structure**, not ranking; the Day 7 KG recommender
will define its own scoring queries separately.

Run `python scripts/load_neo4j.py` first so the graph is populated (see
`docs/day6_knowledge_graph_report.md` for schema and setup).

## 1. A worker and their skills

```cypher
MATCH (w:Worker {worker_id: "W000001"})-[:HAS_SKILL]->(s:Skill)
RETURN w, s
```

## 2. Client -> Worker historical hire paths

```cypher
MATCH path = (c:Client)-[h:HIRED]->(w:Worker)
WHERE h.split = "train"
RETURN path
LIMIT 25
```

Add `h.split = "test"` to see only test-period hires, useful for
demonstrating that the temporal split is preserved as a relationship
property rather than baked into separate graphs.

## 3. Workers connected to a ServiceCategory

```cypher
MATCH (w:Worker)-[r:SPECIALIZES_IN]->(cat:ServiceCategory {name: "Graphic Design"})
RETURN w.worker_id AS worker_id, r.role AS role
ORDER BY r.role, worker_id
LIMIT 25
```

## 4. Workers sharing skills (skill-overlap neighborhood)

```cypher
MATCH (w1:Worker {worker_id: "W000001"})-[:HAS_SKILL]->(s:Skill)<-[:HAS_SKILL]-(w2:Worker)
WHERE w1 <> w2
RETURN w2.worker_id AS other_worker, collect(s.name) AS shared_skills, count(s) AS n_shared
ORDER BY n_shared DESC
LIMIT 10
```

## 5. A worker's full neighborhood (skills, category, location)

```cypher
MATCH (w:Worker {worker_id: "W000001"})
OPTIONAL MATCH (w)-[:HAS_SKILL]->(s:Skill)
OPTIONAL MATCH (w)-[:SPECIALIZES_IN]->(cat:ServiceCategory)
OPTIONAL MATCH (w)-[:LOCATED_IN]->(l:Location)
RETURN w, collect(DISTINCT s) AS skills, collect(DISTINCT cat) AS categories, l
```

## 6. Related skills via the background-corpus co-occurrence edge

```cypher
MATCH (s:Skill {name: "Python"})-[r:RELATED_TO]-(other:Skill)
RETURN other.name AS related_skill, r.npmi AS npmi, r.support AS support
ORDER BY r.npmi DESC
LIMIT 10
```

Note the undirected pattern `-[r:RELATED_TO]-` (no arrow): `RELATED_TO`
is stored as one directed edge per pair (`skill_a < skill_b`
alphabetically) since the underlying NPMI measure is symmetric, so
queries should match either direction rather than assuming a fixed one.

## 7. A client's derived category interests (training-only)

```cypher
MATCH (c:Client)-[i:INTERESTED_IN]->(cat:ServiceCategory)
RETURN c.client_id AS client_id, cat.name AS category, i.training_request_count AS training_requests
ORDER BY i.training_request_count DESC
LIMIT 10
```

## 8. Confirm no cold-start worker has a training hire (sanity check)

```cypher
MATCH (c:Client)-[h:HIRED]->(w:Worker)
WHERE w.is_cold_start = true AND h.split = "train"
RETURN count(h) AS should_be_zero
```

## 9. A rated hire, with rating and review presence

```cypher
MATCH (c:Client)-[r:REVIEWED]->(w:Worker)
RETURN c.client_id AS client_id, w.worker_id AS worker_id, r.rating AS rating, r.has_review AS has_review
ORDER BY r.rating DESC
LIMIT 10
```

No review *text* exists in this dataset (Day 3 generates a rating value
and a `has_review` flag only -- see
`docs/synthetic_data_report.md` Section 9), so `REVIEWED` carries no
review-content property; do not expect one.
