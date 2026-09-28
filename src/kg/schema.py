"""HireWise knowledge graph schema: node labels, relationship types,
property allow-lists, and idempotent constraint/index Cypher.

This schema follows the design already approved in
docs/experimental_design.md (written before Day 6) rather than inventing
one: `HIRED`/`REVIEWED` are the documented historical Knowledge Graph
edges (Section 8, 9.4, 14), `INTERESTED_IN` is derived from training-only
requests with a minimum-2-training-requests-per-category threshold
(Section 14's leakage-prevention table gives this exact rule), and
`RELATED_TO` is computed from the **background corpus** skill
co-occurrence table Day 2 already produced (Section 14: "The Knowledge
Graph's RELATED_TO edges use only the background corpus"). No `Request`
node is introduced (Section 3 of the Day 6 brief): request-level context
(the job `link`, simulated day, and split) is carried as *properties* on
the `REQUESTED_SERVICE`, `HIRED`, and `REVIEWED` edges instead, which is
what "Neo4j export (nodes/edges, with split property)"
(docs/experimental_design.md Appendix B) already describes.

No `Review` node: Day 3 generates a rating value and a `has_review`
boolean flag, but no review *text* (docs/synthetic_data_report.md
Section 9). A dedicated node would misrepresent data that doesn't exist;
`REVIEWED` is a relationship carrying `rating` and `has_review`
properties instead, matching the established "Knowledge Graph ...
REVIEWED edges" language directly.

`SPECIALIZES_IN` (not `REQUESTED_SERVICE`) is the worker-category
relationship name: it is the natural profile-side counterpart to the
client-side `REQUESTED_SERVICE`, and matches the Day 6 brief's own
suggested name for "a justified operational name for the documented
category relationship."
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------
LABEL_CLIENT = "Client"
LABEL_WORKER = "Worker"
LABEL_SKILL = "Skill"
LABEL_CATEGORY = "ServiceCategory"
LABEL_LOCATION = "Location"

ALL_LABELS = [LABEL_CLIENT, LABEL_WORKER, LABEL_SKILL, LABEL_CATEGORY, LABEL_LOCATION]

# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------
REL_HAS_SKILL = "HAS_SKILL"                # (:Worker)-[:HAS_SKILL]->(:Skill)
REL_SPECIALIZES_IN = "SPECIALIZES_IN"      # (:Worker)-[:SPECIALIZES_IN]->(:ServiceCategory)
REL_LOCATED_IN = "LOCATED_IN"              # (:Worker|:Client)-[:LOCATED_IN]->(:Location)
REL_REQUESTED_SERVICE = "REQUESTED_SERVICE"  # (:Client)-[:REQUESTED_SERVICE]->(:ServiceCategory), one per request
REL_INTERESTED_IN = "INTERESTED_IN"        # (:Client)-[:INTERESTED_IN]->(:ServiceCategory), derived, training-only
REL_HIRED = "HIRED"                        # (:Client)-[:HIRED]->(:Worker), one per hire event
REL_REVIEWED = "REVIEWED"                  # (:Client)-[:REVIEWED]->(:Worker), one per rated hire
REL_RELATED_TO = "RELATED_TO"              # (:Skill)-[:RELATED_TO]->(:Skill), background-corpus co-occurrence

ALL_RELATIONSHIP_TYPES = [
    REL_HAS_SKILL, REL_SPECIALIZES_IN, REL_LOCATED_IN, REL_REQUESTED_SERVICE,
    REL_INTERESTED_IN, REL_HIRED, REL_REVIEWED, REL_RELATED_TO,
]

# ---------------------------------------------------------------------------
# Model-visible property allow-list (Day 6 brief Section 11). Anything not
# listed here must never be written to the graph. Enforced in prepare.py
# and re-checked statically/at runtime by tests/test_kg_leakage.py.
# ---------------------------------------------------------------------------
ALLOWED_NODE_PROPERTIES: dict[str, set[str]] = {
    LABEL_CLIENT: {"client_id", "preferred_pay_type", "median_price_tier"},
    LABEL_WORKER: {
        "worker_id", "experience_level", "experience_years", "price_tier",
        "rate", "join_day", "is_cold_start",
    },
    LABEL_SKILL: {"name"},
    LABEL_CATEGORY: {"name"},
    LABEL_LOCATION: {"name"},
}

ALLOWED_RELATIONSHIP_PROPERTIES: dict[str, set[str]] = {
    REL_HAS_SKILL: set(),
    REL_SPECIALIZES_IN: {"role"},  # "primary" | "secondary"
    REL_LOCATED_IN: set(),
    REL_REQUESTED_SERVICE: {"link", "simulated_day", "split"},
    REL_INTERESTED_IN: {"training_request_count"},
    REL_HIRED: {"link", "simulated_day", "split"},
    REL_REVIEWED: {"link", "rating", "has_review"},
    REL_RELATED_TO: {"npmi", "support"},
}

# Explicitly forbidden -- generator-private or otherwise never permitted in
# the graph, regardless of label. Used by leakage tests as a denylist
# cross-check on top of the allow-list.
FORBIDDEN_PROPERTY_MARKERS = (
    "quality", "true_skills", "proficiency", "taste", "utility", "noise",
    "oracle", "standardized", "s_raw", "t_raw", "q_raw", "b_raw", "c_raw",
)

# ---------------------------------------------------------------------------
# Idempotent constraint/index Cypher.
#
# Written for Neo4j 5.x/2026.x syntax (`CREATE CONSTRAINT ... IF NOT
# EXISTS ... REQUIRE ... IS UNIQUE`), which is also valid under Cypher 25.
# Every statement is safe to run multiple times.
# ---------------------------------------------------------------------------
CONSTRAINT_STATEMENTS = [
    f"CREATE CONSTRAINT client_id_unique IF NOT EXISTS "
    f"FOR (c:{LABEL_CLIENT}) REQUIRE c.client_id IS UNIQUE",
    f"CREATE CONSTRAINT worker_id_unique IF NOT EXISTS "
    f"FOR (w:{LABEL_WORKER}) REQUIRE w.worker_id IS UNIQUE",
    f"CREATE CONSTRAINT skill_name_unique IF NOT EXISTS "
    f"FOR (s:{LABEL_SKILL}) REQUIRE s.name IS UNIQUE",
    f"CREATE CONSTRAINT category_name_unique IF NOT EXISTS "
    f"FOR (cat:{LABEL_CATEGORY}) REQUIRE cat.name IS UNIQUE",
    f"CREATE CONSTRAINT location_name_unique IF NOT EXISTS "
    f"FOR (l:{LABEL_LOCATION}) REQUIRE l.name IS UNIQUE",
]

INDEX_STATEMENTS = [
    f"CREATE INDEX worker_join_day IF NOT EXISTS FOR (w:{LABEL_WORKER}) ON (w.join_day)",
    f"CREATE INDEX worker_cold_start IF NOT EXISTS FOR (w:{LABEL_WORKER}) ON (w.is_cold_start)",
    f"CREATE INDEX hired_split IF NOT EXISTS FOR ()-[r:{REL_HIRED}]-() ON (r.split)",
    f"CREATE INDEX requested_service_split IF NOT EXISTS FOR ()-[r:{REL_REQUESTED_SERVICE}]-() ON (r.split)",
]

ALL_SCHEMA_STATEMENTS = CONSTRAINT_STATEMENTS + INDEX_STATEMENTS
