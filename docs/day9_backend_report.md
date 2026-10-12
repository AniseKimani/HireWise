# Day 9 Backend Report: Flask API Integration

**Objective.** Expose the frozen HireWise recommendation stack through a
small, tested Flask API, without changing any recommender, result or
evaluation methodology. The API contract is in `docs/day9_api.md`.

Day 9 includes no frontend, authentication, PostgreSQL, deployment,
retraining or retuning.

## 1. Architecture

```
src/app/
  factory.py      create_app(config=None, service=None, neo4j_probe=None, worker_graph_reader=None)
  config.py       Development / Testing / Production (default) classes, selected by HIREWISE_ENV
  routes.py       thin GET routes under /api (validation -> service -> JSON)
  services.py     RecommendationService + read-only Neo4j probe / worker-graph reader
  serializers.py  explicit allow-listed field serializers, JSON-safe scalars
  errors.py       APIError and JSON handlers for HTTP / unexpected errors
```

The dependency direction is one-way: `routes → services → recommenders /
evaluation / kg`. No module under `src/recommenders`, `src/kg`,
`src/evaluation` or `src/data` imports Flask, and a test checks this.
Dependencies (the service, the Neo4j probe, the graph reader) can be
injected, so tests can replace them with fakes.

Flask 3.1.3 is the only new dependency. It is pinned in
`requirements.txt` and is the README's documented backend.

## 2. Service layer and recommender integration

`RecommendationService.build()` runs once, at application start-up:

1. `load_selected_hybrid()` reads `results/day8/selected_hybrid_config.json` and checks its integrity:
   - its SHA-256 must equal the Day 8 manifest entry;
   - `config/hybrid_config.yaml` must still hash to the value the selection recorded;
   - the weights must be in the predeclared grid.

   If any check fails, the service refuses to load.
2. `load_protocol_data()` loads the model-visible data (the same loader the evaluation uses), and `EligibilityIndex` is built from it.
3. `build_frozen_models()` (Day 8) rebuilds the frozen CBF, CF and KG models from their recorded selections. It also checks the KG snapshot fingerprint recorded by Day 7.
4. A `HybridRecommender` is built with the selected weights, **CBF 0.40 / CF 0.10 / KG 0.50**, with per-request min-max normalisation.

For each recommendation call, the service:
1. looks up the request by `link`;
2. builds the candidate pool with `EligibilityIndex.eligible_worker_ids`, the function the evaluation protocol itself uses (relevance is never computed);
3. calls `HybridRecommender.explain(link, candidates, k, kg_recommender=kg)`;
4. serialises the result.

No scoring equation exists outside the frozen modules. Component score
caches are reused across calls, and a lock serialises access to them.

**Recommendable requests.** Only validation/test-period requests can be
recommended. The KG component scores only requests strictly after its
training-only history; training-period requests return 409 instead of
being scored with their own outcomes visible. This is the same boundary
the evaluation used.

## 3. Neo4j integration

- **Recommendations do not need Neo4j.** The KG component uses the training-only snapshot built from the Day 6 CSVs. Day 7 verified that snapshot equal to the live Neo4j extraction, and `build_frozen_models` re-checks its fingerprint at every start-up.
- **Health check.** `/api/health` probes Neo4j with read-only `count` queries in READ-access transactions (`Neo4jClient.run_read`). It reports whether the totals still equal the Day 6 manifest (6,420 nodes / 79,165 relationships). It never returns error text, because that may contain connection details.
- **Worker graph.** `/api/workers/<id>/graph` reads one worker's static neighbourhood (skills, specialisations, location) using fixed, parameterised Cypher. Clients cannot supply Cypher. If Neo4j is unreachable this route returns 503, and every other route keeps working.
- **No writes.** The service Cypher is checked statically for write keywords, and all access goes through `run_read`. The verifier confirms the live node and relationship counts are identical before and after a full verification run. The default test suite never writes to Neo4j.

## 4. Security boundaries (Day 9 scope)

- **Input validation.**
  - `k` must match `^[0-9]{1,6}$` and lie within 1–50.
  - `link` is required and at most 2,048 characters.
  - `worker_id` must match `^[A-Za-z0-9_-]{1,32}$`.
  - Every route is GET-only.
- **Explicit allow-lists.** Responses are built field by field in `serializers.py`, never from DataFrame rows or `__dict__`. Generator-private variables, the new-entrant cohort flag, and request outcomes (shortlists, hires, relevance) are never serialised.
- **Error handling.** Errors use one JSON shape. Unexpected exceptions are logged server-side and answered with a generic `internal_error`, never a stack trace.
- **Strict JSON.** A strict JSON provider refuses to emit NaN or Infinity.
- **Safe defaults.** `HIREWISE_ENV` defaults to `production`, so debug is off. Credentials remain only in the existing `NEO4J_*` environment variables / git-ignored `.env`, and neither the API nor the docs echo them.

## 5. Testing

| Suite | Tests | Covers |
|---|---|---|
| `tests/test_app_unit.py` | 26 | Factory and config overrides; health (`ok`, `degraded`, `unavailable`, no secrets); k and link validation; 400 / 404 / 405 / 409 / 503 / 500 handling with no leak of internals; serializers (NaN, NumPy, allow-lists); strict JSON; read-only fixed Cypher; no Flask imports in model layers; no scoring logic in routes |
| `tests/test_day9_api_integration.py` | 10 | Real stack: request, worker and recommendation routes; default and custom k; ranking order, no duplicates, contributions summing to the score, explanations; **parity with a direct frozen Day 8 hybrid**; byte-identical repeat and fresh-app responses; allow-listed keys only; request lookup does not reveal its outcome; live read-only Neo4j probe and graph route (skipped if Neo4j is unavailable) |

The full default suite has **426 tests, all passing** (2 skipped: the
opt-in Neo4j write tests). `scripts/verify_day9.py` passes all 24
checks, and `scripts/verify_day8.py` still passes.

## 6. Parity, determinism and privacy results

- **Parity.** For 35 deterministic validation/test requests (every 100th included request), the API's ranking, scores and per-component contributions are exactly equal to a direct invocation of the frozen Day 8 hybrid on the protocol's own candidate pools. The API pool sizes equal the protocol's.
- **Determinism.** Repeating a request, and sending it to a freshly built application, return byte-identical JSON.
- **Privacy.** Every key in every sampled response belongs to the allow-list, and none matches a private or outcome marker.

## 7. Performance (local, indicative only)

Measured with `python scripts/demo_day9_api.py --timing` on the
development laptop, using the Flask test client (no network):

| Measure | Value |
|---|---|
| `create_app` including all model and data loading | 1.5 s with a warm file cache (the first cold run took about 3 s) |
| Recommendation, first call per request (n = 24) | median 27 ms, mean 44 ms |
| Recommendation, repeat call (n = 24) | median 19 ms, mean 28 ms |

Models and data are built once and reused, and component scores are
cached per request. Most of the per-call cost is the frozen Day 7 KG
explainer enumerating paths over the whole pool, needed to attach real
evidence for every returned worker. These figures are not production
performance claims.

## 8. Limitations

1. **Development data source.** The API serves the processed synthetic dataset and frozen artifacts. There is no persistence layer and no way to create new requests yet.
2. **No new or training-period requests.** Only existing validation/test requests can be recommended. Scoring an unseen request would need a request-creation path, which is future work.
3. **Single process.** Component caches are in-process dicts, accessed under one lock. That is adequate for the Flask development server, but it is not a scalability design.
4. **Explanation rounding.** The frozen Day 8 explanation template rounds to 2 decimals, so a small positive CF contribution can read "contributed 0.00". The numeric `contributions` field is exact.
5. **No authentication or rate limiting** (out of scope for Day 9).

## 9. Day 10 handoff

A frontend can rely on:
- `GET /api/health` for a status banner;
- `GET /api/requests?link=` for request detail;
- `GET /api/recommendations?link=&k=10` for the ranked list, contribution bars (`contributions`), a one-line reason (`explanation`) and an evidence list (`kg_evidence`);
- `GET /api/workers/<id>` and `/graph` for profile and graph views.

Pending decisions before or during Day 10:
- how users will choose requests, since there is no listing endpoint yet;
- whether to show the new-entrant cohort flag;
- whether to keep the 409 boundary for training-period requests.

## Reproducing

```bash
pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/verify_day9.py
python scripts/demo_day9_api.py --timing
```
