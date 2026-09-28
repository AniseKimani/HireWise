# Day 6 Knowledge Graph Report

**Objective.** Design, construct, load, and verify the HireWise
knowledge graph in Neo4j: infrastructure and graph construction only.
No KG recommender, Hybrid, Flask, PostgreSQL, or frontend work is
included today.

## 1. Design provenance: following, not inventing, the schema

Before writing any code, `docs/experimental_design.md` was checked for
an already-approved graph design (written well before Day 6). It
specifies several concrete details this report follows rather than
re-deciding:

- `HIRED`/`REVIEWED` are named explicitly as **Knowledge Graph edges**
  (Sections 8, 9.4, 14) -- not as separate Rating/Review nodes.
- `INTERESTED_IN` "exposing latent client preferences" is required to be
  "Derived from observed training requests (e.g. at least two training
  requests in a category), never from generator variables" (Section 14).
  This fixes both the derivation rule (>=2 training requests) and that it
  targets **ServiceCategory**, not Skill -- the Day 6 brief's own example
  (`INTERESTED_IN -> Skill`) is a suggested name, but the established
  design's "in a category" wording takes precedence per the brief's own
  instruction to follow existing documentation over its examples.
- `RELATED_TO` edges between skills must use **only the background
  corpus** (Section 14: "The Knowledge Graph's RELATED_TO edges use only
  the background corpus"), kept deliberately independent of the
  generator's own full-corpus co-occurrence (used by the Day 3
  generator's true-skill-fit component) so the two remain correlated but
  not identical.
- Interaction edges carry a `split` property rather than existing in
  separate per-split graphs (Section 14: "Interaction edges carry a
  `split` property; evaluation queries filter to `split = 'train'`");
  Appendix B: "Neo4j export (nodes/edges, with split property)".

No genuine schema ambiguity was found that materially changed the
model, so no stop was needed.

**No `Request` node** (Day 6 brief Section 3): request-level context
(the job `link`, simulated day, split) is carried as *properties* on the
`REQUESTED_SERVICE`, `HIRED`, and `REVIEWED` edges instead, matching the
"edges, with split property" language directly. Request-level context
for Day 7 is fully reconstructable this way: `MATCH (c)-[r:HIRED
{link: $link}]->(w)` recovers exactly which client/worker/day/split a
given historical request produced, without a dedicated node.

**No `Review` node**: Day 3 generates a rating value and a `has_review`
boolean flag, but no review *text*
(`docs/synthetic_data_report.md` Section 9). A dedicated node would
misrepresent data that doesn't exist. `REVIEWED` is a relationship
carrying `rating` and `has_review` properties, which is what the
established "Knowledge Graph ... REVIEWED edges" language already
describes and is consistent with the data actually available. (Section
10 below documents the full pre-freeze consistency review of this
specific decision, re-checked explicitly against Day 3's actual data
before the graph was finalized.)

`SPECIALIZES_IN` was used for the worker-category relationship (as the
Day 6 brief itself suggests as "a justified operational name"), as the
natural profile-side counterpart to the client-side
`REQUESTED_SERVICE`.

**`SHORTLISTED` interactions are not represented as graph edges.** The
established design names "Knowledge Graph `HIRED`/`REVIEWED` edges"
specifically (Section 14); introducing an undocumented `SHORTLISTED`
relationship type was avoided to not silently expand the approved
schema. This is recorded as a decision for review in Section 15.

## 2. Neo4j setup

- Official Python driver (`neo4j==6.3.1`), no ORM or heavier graph
  framework.
- Connection settings read from `NEO4J_URI`, `NEO4J_USER`,
  `NEO4J_PASSWORD`, `NEO4J_DATABASE` environment variables (optionally
  loaded from a local `.env` via `python-dotenv`; `.env.example` provides
  placeholders only, `.env` itself is git-ignored and was never staged).
- `src/kg/neo4j_client.py`'s `Neo4jClient` context manager raises a
  clear, actionable `Neo4jConfigError` or `ConnectionError` (never a bare
  traceback) if credentials are missing or the server is unreachable;
  every script (`load_neo4j.py`, `verify_neo4j.py`) catches these and
  prints an actionable message.
- Verified live during Day 6 development: Neo4j Kernel 2026.09.0
  (Enterprise edition), Cypher versions 5 and 25, database `neo4j`.

## 3. Graph schema

### Node labels and properties (model-visible only)

| Label | Key property | Other properties |
|---|---|---|
| `Client` | `client_id` | `preferred_pay_type`, `median_price_tier` |
| `Worker` | `worker_id` | `experience_level`, `experience_years`, `price_tier`, `rate`, `join_day`, `is_cold_start` |
| `Skill` | `name` | -- |
| `ServiceCategory` | `name` | -- |
| `Location` | `name` | -- |

**Explicitly excluded** from every node (Day 6 brief Section 11):
`quality`, `true_skills`, `proficiency`, latent taste vectors, `utility`,
`noise`, oracle scores, standardized generator components. Enforced by
`src/kg/schema.py`'s per-label property allow-list, checked at
preparation time (`_assert_allowed_columns` raises if violated) and by
`tests/test_kg_leakage.py`.

### Relationship types and properties

| Type | Pattern | Properties |
|---|---|---|
| `HAS_SKILL` | `(Worker)->(Skill)` | -- |
| `SPECIALIZES_IN` | `(Worker)->(ServiceCategory)` | `role` ("primary"/"secondary") |
| `LOCATED_IN` | `(Worker\|Client)->(Location)` | -- |
| `REQUESTED_SERVICE` | `(Client)->(ServiceCategory)` | `link`, `simulated_day`, `split` -- one edge per request |
| `INTERESTED_IN` | `(Client)->(ServiceCategory)` | `training_request_count` -- derived, training-only |
| `HIRED` | `(Client)->(Worker)` | `link`, `simulated_day`, `split` -- one edge per hire event |
| `REVIEWED` | `(Client)->(Worker)` | `link`, `rating`, `has_review` -- one edge per rated hire |
| `RELATED_TO` | `(Skill)->(Skill)` | `npmi`, `support` -- background-corpus co-occurrence |

## 4. Temporal safety

`REQUESTED_SERVICE` and `HIRED` both carry the request's real `split`
(train/validation/test) as a property, taken directly from Day 3's
`requests.csv`. `INTERESTED_IN` is *derived*, not copied: it is computed
exclusively from `REQUESTED_SERVICE` rows with `split == "train"`,
grouped by (client, category), keeping only pairs with
`training_request_count >= 2`. Adding a validation or test request in
that category can never create, remove, or change an `INTERESTED_IN`
edge -- proven in `tests/test_kg_temporal.py` by constructing the edge
set twice (with and without an extra validation-split request) and
asserting the results are identical.

Cold-start workers get no training `HIRED` edge **by construction**,
inherited directly from Day 3 (their `join_day` is set to the validation
boundary, so no training-period request could ever have hired them) --
`prepare_hired_edges` does not add or remove this property, it passes
through what Day 3 already guarantees. Verified independently three
ways: a unit test, `scripts/verify_day6.py`'s prepared-artifact check,
and a live Cypher query against the loaded graph (Section 6 below), all
finding **0**.

## 5. Model-visible-only enforcement

`src/kg/schema.py` defines `ALLOWED_NODE_PROPERTIES` and
`ALLOWED_RELATIONSHIP_PROPERTIES` per label/type. Every `prepare_*`
function in `src/kg/prepare.py` calls `_assert_allowed_columns` before
returning, so a column added by mistake fails loudly at preparation
time rather than silently reaching Neo4j. `src/kg/prepare.py` reads only
`data/processed/*.csv` and `data/processed/synthetic/*.csv`; it never
opens anything under `data/processed/synthetic_private/` (verified by a
test that monkeypatches `pandas.read_csv` to record every path
`prepare_all()` opens). Leakage tests check the module's actual
executable code (docstrings/comments stripped via `ast`, so honest prose
explaining what the module does *not* do can't produce a false
negative) for any of `quality`, `true_skills`, `proficiency`, `taste_`,
`utility`, `noise`, `oracle`, `standardized`.

## 6. Loading strategy and idempotency

`src/kg/loader.py` batches writes via `UNWIND $rows AS row` (default
batch size 500), one parameterized Cypher template per node/relationship
type -- values are always bound parameters, never string-concatenated
into the query text. Every write uses `MERGE` on a stable domain
identifier (`client_id`, `worker_id`, `Skill.name`, `ServiceCategory.name`,
`Location.name`); per-event relationships (`REQUESTED_SERVICE`, `HIRED`,
`REVIEWED`) include the request `link` *inside* the `MERGE` pattern
itself, so two different requests between the same client and category
create two distinct relationships, while re-loading the same request
never duplicates one.

**Idempotency verified live, twice** (`python scripts/load_neo4j.py` run
back-to-back against the actual Neo4j instance): node and relationship
counts were identical after both runs -- see Section 8. `--rebuild` is
required (and only then) to `DETACH DELETE` the database first; it is
never implicit.

## 7. Graph statistics

**Node counts:**

| Label | Count | Isolated (no relationships) |
|---|---|---|
| Client | 2,000 | 0 |
| Worker | 3,000 | 0 |
| Skill | 1,187 | 0 |
| ServiceCategory | 88 | 0 |
| Location | 145 | 0 |
| **Total** | **6,420** | **0** |

Zero isolated nodes anywhere, including among the full 1,187-skill
vocabulary: every controlled-vocabulary skill appears in at least one
`RELATED_TO` edge (the background-corpus co-occurrence table has
excellent coverage), so including the complete vocabulary (Section 9
below) did not produce dangling nodes in practice, even though nothing
was fabricated to guarantee that.

**Relationship counts:**

| Type | Count |
|---|---|
| `HAS_SKILL` | 23,131 |
| `SPECIALIZES_IN` | 6,031 |
| `LOCATED_IN` | 5,000 (3,000 worker + 2,000 client) |
| `REQUESTED_SERVICE` | 12,000 |
| `INTERESTED_IN` | 1,163 |
| `HIRED` | 10,163 |
| `REVIEWED` | 9,173 |
| `RELATED_TO` | 12,504 |

**Degree statistics** (min / median / mean / max, undirected):

| Label | Min | Median | Mean | Max |
|---|---|---|---|---|
| Client | 2 | 14 | 17.2 | 113 |
| Worker | 3 | 15 | 17.2 | 59 |
| Skill | 1 | 22 | 40.6 | 594 |
| ServiceCategory | 46 | 137 | 218.1 | 1,096 |
| Location | 1 | 2 | 34.5 | 1,435 |

Location's mean (34.5) far exceeds its median (2), confirming the
expected heavy skew toward a handful of major countries (the US alone
accounts for a large share of both clients and workers -- consistent
with Day 2/3's documented US-heavy geographic skew).

**Historical edge split counts** (directly from `REQUESTED_SERVICE` and
`HIRED` relationship properties in the live graph):

| Split | REQUESTED_SERVICE | HIRED |
|---|---|---|
| train | 8,400 | 7,117 |
| validation | 1,200 | 1,035 |
| test | 2,400 | 2,011 |
| **Total** | **12,000** | **10,163** |

Matches Day 3's approved 70/10/20 request split and realised hire rate
exactly.

## 8. Idempotency check (actual second-load comparison)

`python scripts/load_neo4j.py` was run twice against the same live
database. Node and relationship counts after each run:

| | Run 1 | Run 2 |
|---|---|---|
| Total nodes | 6,420 | 6,420 |
| Total relationships | 79,165 | 79,165 |

Identical. No duplication on rerun.

## 9. Controlled vocabulary vs. connected nodes (Section 13 of the brief)

**Decision: include the complete controlled vocabulary** (all 1,187
skills, all 88 categories), not only those connected to the current
12,000-request sample. Rationale: the vocabulary is a stable, closed set
established in Day 2 (docs/experimental_design.md Section 6); future
requests drawn from the same real corpus may use any vocabulary skill
even if the current sample doesn't, and a stable node set means Day 7's
KG queries never need to conditionally create nodes. No relationship was
fabricated for any node to justify this -- in the event, every skill and
category turned out to have at least one genuine edge anyway (Section 7).

## 10. Rating/Review schema consistency review

A dedicated re-check was performed before freezing Day 6, specifically
to confirm the `REVIEWED`-relationship representation (rather than
dedicated `Rating`/`Review` node labels) is correct and not merely
convenient.

**What the documentation requires.** Every reference to rating/review
data in the repository -- `docs/experimental_design.md` Sections 8, 9.4,
10, 14, Appendix B; `docs/synthetic_data_report.md` Section 9;
`docs/day4_recommender_report.md` (signal-access table) -- names
`REVIEWED` as a **Knowledge Graph edge**: e.g. "Knowledge Graph
`HIRED`/`REVIEWED` edges" (Section 14), "Partial (training `REVIEWED`
edges)" (Section 9.4's signal-access matrix). No document in this
repository, including the earliest design document
(`docs/experimental_design.md`, committed before any Day 2-6 work),
ever specifies a `Rating` or `Review` *node* label. The README (the only
other pre-existing document, committed before `experimental_design.md`)
contains no schema detail at all. There is therefore no documented
requirement this section's design decision could conflict with.

**What Day 3 actually generates** (`data/processed/synthetic/ratings.csv`,
re-verified directly for this review):

| Question | Answer |
|---|---|
| Actual numeric rating? | Yes -- `rating`, integer 1-5 |
| Structured review metadata? | Only `has_review` (boolean); no date, length, sentiment, or other structured field |
| Textual review content? | **No** -- zero text columns exist anywhere in the file or the generator that produced it |
| Does every rating correspond to a completed/hired interaction? | Yes -- all 9,173 `(link, worker_id)` rating pairs are a strict subset of the 10,163 `HIRED` pairs (never a `SHORTLISTED`-only pair), matching the documented ~90% completion assumption exactly |
| Stable event/request identifier available? | Yes -- `link`, the same identifier already used on `HIRED`/`REQUESTED_SERVICE` edges |

**Decision, applying the stated precedence:**

1. *Preserve actual available data* -- satisfied: `rating` and
   `has_review` are both stored, unmodified, as `REVIEWED` edge
   properties (Section 3 above). Nothing available was discarded.
2. *Preserve the approved documented architecture where it can be done
   honestly* -- satisfied: the only documented architecture is
   "`REVIEWED` edges," which is exactly what exists. There is no
   documented `Rating`/`Review` node to preserve or conflict with.
3. *Do not create fake textual reviews merely to satisfy a diagram* --
   satisfied by construction: no `Review` node was created, so no review
   text was ever at risk of being fabricated.
4. *Prefer a small documented schema amendment over inventing unsupported
   data* -- not applicable: no data needs inventing to satisfy any
   documented requirement, because none exists.

**Conclusion: no schema change.** The current `REVIEWED`-relationship
representation is an **intentional implementation refinement of the
earlier conceptual schema**, not a deviation from it: the project's
README-level concept of "Knowledge Graph reasoning" and
`docs/experimental_design.md`'s "`REVIEWED` edges" language are both
satisfied precisely as written. A `Rating` node would add a label with
no property `REVIEWED` doesn't already carry (the rating value, the
completion flag, and the stable `link` identifier), at the cost of an
extra traversal hop with no documented justification. A `Review` node
would necessarily be an empty placeholder, since no grounded review text
exists to populate it -- creating one "merely to satisfy a diagram"
is explicitly what this review was tasked with avoiding. This decision
was made purely on data/documentation consistency grounds, not on
anticipated Day 7 KG-recommender convenience (which this review did not
consider).

## 11. Reproducibility

- Two full runs of `scripts/prepare_kg_data.py` produced byte-identical
  output files under `data/processed/kg/` (`diff -rq`, zero
  differences).
- `scripts/verify_day6.py` independently re-runs preparation into a
  temporary directory and compares SHA-256 hashes against the saved
  artifacts and the recorded manifest.
- `kg_prepare_manifest.json` records per-file SHA-256 hashes with
  relative POSIX paths (no timestamps, no machine-specific absolute
  paths).
- The live-database idempotency check (Section 8) additionally confirms
  reproducibility end-to-end through Neo4j itself, not just at the CSV
  layer.

## 12. Limitations

1. `SHORTLISTED` interactions exist in the model-visible dataset
   (`interactions.csv`) but are not represented as graph edges, since
   the established design names only `HIRED`/`REVIEWED` as Knowledge
   Graph edges. This could be revisited for Day 7 if a KG recommender
   needs the weaker positive signal (see Section 15).
2. `RELATED_TO` is stored as a single directed edge per
   alphabetically-ordered pair (`skill_a < skill_b`); Cypher queries
   must match it without a direction constraint (`-[:RELATED_TO]-`),
   documented in `docs/neo4j_demo_queries.md` Section 6.
3. Location granularity is country-level only, matching Day 2's
   documented limitation (no city-level or distance data exists in the
   source dataset).
4. `Client`/`Worker` node properties are a deliberately small,
   documented subset of what's available in the model-visible CSVs
   (Section 3 of this report); additional observable fields could be
   added later if Day 7's KG recommender needs them, following the same
   allow-list discipline.

## 13. How this supports Day 7 without implementing it

Day 7's KG recommender can, without any further schema work:
- Traverse `Worker-[:HAS_SKILL]->Skill-[:RELATED_TO]->Skill<-[:HAS_SKILL]-Worker`
  paths for skill-based relatedness reasoning grounded in the
  background corpus.
- Filter any historical relationship to `split = 'train'` for
  leakage-safe training-time graph statistics, exactly as
  `docs/experimental_design.md` Section 14 anticipates.
- Use `INTERESTED_IN` as a precomputed, already-leakage-safe client
  preference signal without re-deriving it per query.
- Recover full request-level context (category, timing, split) for any
  historical `HIRED`/`REVIEWED`/`REQUESTED_SERVICE` edge via its `link`
  property, without a `Request` node adding traversal overhead.

No ranking, scoring, or path-weighting logic has been implemented --
that is Day 7's task.

## Reproducing this report

```bash
python scripts/prepare_kg_data.py
python scripts/load_neo4j.py          # requires .env with Neo4j credentials
python scripts/verify_neo4j.py
python scripts/verify_day6.py
python -m unittest discover -s tests
```
