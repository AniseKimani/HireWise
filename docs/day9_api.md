# HireWise API Contract (Day 9)

This is a read-only JSON API over the frozen HireWise recommender stack.
The base path is `/api`. Every response is `application/json`. Keys are
sorted, and NaN/Infinity are never emitted (missing values are `null`).

There is no authentication yet. That is out of scope for Day 9.

Run locally:

```bash
flask --app "src.app:create_app('development')" run      # http://127.0.0.1:5000
python scripts/demo_day9_api.py                           # no server needed
```

## Identifiers

- **Requests are identified by `link`**, the job-posting URL used as the request id everywhere in the repository. A URL can't safely sit in a path segment (encoded slashes break routing), so `link` is passed as a **query parameter**. Clients must URL-encode it.
- **Workers are identified by `worker_id`**, for example `W000517`. Accepted pattern: `[A-Za-z0-9_-]{1,32}`.

## Endpoints

### `GET /api/health`

Reports application status and what each dependency can do. It returns
no credentials, URIs or file paths.

**Status codes:**
- **200**: the recommender is loaded.
- **503**: the recommender failed to load.

```json
{
  "status": "ok",
  "recommender": {
    "ready": true,
    "model": {
      "name": "hirewise-hybrid-day8",
      "weights": {"cbf": 0.4, "cf": 0.1, "kg": 0.5},
      "normalization": "minmax_within_candidate_pool",
      "selected_config_sha256": "3c78a6cd473f",
      "hybrid_config_sha256": "4d956748204f"
    }
  },
  "neo4j": {"available": true, "checked": true, "graph_matches_day6_counts": true},
  "recommendations_require_neo4j": false
}
```

**`status` values:**
- `ok`: the recommender is ready, and Neo4j is reachable or wasn't probed.
- `degraded`: the recommender is ready but Neo4j is unreachable. Recommendations still work.
- `unavailable`: the recommender isn't loaded. `recommender.reason` is `load_failed`, `frozen_artifact_mismatch` or `not_loaded`.

`graph_matches_day6_counts` compares the live node and relationship
totals with the Day 6 manifest (6,420 / 79,165) using read-only queries.

### `GET /api/requests` (listing, added Day 10)

When called **without** `link`, this returns one page of **recommendable** requests, i.e. all 3,600 validation and test requests. They are in fixed chronological order (simulated day, then link). It exists so the Day 10 frontend can let users browse requests.

**Parameters** (all optional):

| Parameter | Meaning | Default / limit |
|---|---|---|
| `page` | 1-based page number | 1; a page past the end returns `items: []` |
| `per_page` | items per page | 20; 1–50 |
| `q` | case-insensitive title substring | ≤ 100 chars |
| `category` | exact category name | ≤ 100 chars |
| `skill` | required skill, case-insensitive exact match | ≤ 100 chars |

**Responses:** 200, 400 (`invalid_page`, `invalid_per_page`, `invalid_q`, `invalid_category`, `invalid_skill`).

```json
{
  "items": [ { "...": "same fields as the single lookup below", "n_eligible_workers": 65 } ],
  "page": 1, "per_page": 20, "total": 3600, "total_pages": 180,
  "filters": {"q": null, "category": null, "skill": null},
  "available_categories": ["2D Animation", "..."]
}
```

- Items use the same allow-listed serializer as the single lookup. They never include outcomes, relevance or cohort data.
- `n_eligible_workers` is the size of the request's candidate pool under the shared protocol.
- `available_categories` lists the categories of recommendable requests, so the UI can build its filter control.

### `GET /api/requests?link=<url-encoded link>`

Returns the model-visible description of one request. The example below is the request the demo script uses.

**Responses:** 200, 400 (`missing_link` / `invalid_link`), 404 (`unknown_request`).

```json
{
  "link": "https://www.upwork.com/jobs/%7E016d29deee2a1bab4a?source=rss",
  "title": "...",
  "client_id": "C001396",
  "category": "Creative Writing",
  "required_skills": ["Russian", "Writing"],
  "pay_type": "hourly",
  "price_tier": 0.7478,
  "location_requirement": null,
  "simulated_day": 364.1125,
  "split": "test"
}
```

It never includes the request's outcome (who was shortlisted or hired).

### `GET /api/workers/<worker_id>`

Returns the model-visible profile.

**Responses:** 200, 400 (`invalid_worker_id`), 404 (`unknown_worker`).

```json
{
  "worker_id": "W000517",
  "primary_category": "Creative Writing",
  "secondary_categories": ["Scriptwriting", "Technical Writing"],
  "declared_skills": ["Article Writing", "Writing"],
  "experience_level": "entry",
  "experience_years": 0.0,
  "price_tier": 0.2313,
  "rate": 25.0,
  "country": "United States",
  "join_day": 0.0,
  "training_review_count": 6,
  "training_mean_rating": 4.8333
}
```

`training_review_count` and `training_mean_rating` come only from
training-period `REVIEWED` edges, the same evidence the KG reputation
component uses. Generator-private variables and the new-entrant cohort
flag are never returned.

### `GET /api/workers/<worker_id>/graph`

Returns the worker's static profile neighbourhood, read **live from
Neo4j** with fixed, parameterised, read-only Cypher. It includes skills,
specialisations with their role, and location. Interaction edges
(`HIRED`, `REVIEWED`) are not returned, because they include
validation/test outcomes.

**Responses:** 200, 400, 404, 503 (`neo4j_unavailable`).

```json
{"worker_id": "W000026", "skills": ["2D Animation", "..."],
 "categories": [{"category": "3D Animation", "role": "primary"}], "location": "India", "source": "neo4j"}
```

### `GET /api/recommendations?link=<url-encoded link>&k=<int>`

Returns the Top-K ranking from the frozen Day 8 hybrid over the request's
eligible candidate pool. This is the same pool the evaluation protocol
uses: category match, joined by the request day, and location
eligibility.

**`k`:**
- an optional integer, default **10**, valid range **1–50**;
- signs, decimals, spaces and exponents are rejected;
- if the pool is smaller than `k`, every eligible worker is returned.

Only **validation/test-period** requests can be recommended. A
training-period request returns 409, because the frozen models' history
contains its outcome.

**Responses:**
- 200;
- 400 (`missing_link`, `invalid_link`, `invalid_k`);
- 404 (`unknown_request`);
- 409 (`request_not_recommendable`);
- 503 (`recommender_unavailable`).

```json
{
  "request": { "...": "same fields as GET /api/requests" },
  "model": { "...": "same as health.recommender.model" },
  "k": 10,
  "n_candidates": 109,
  "n_returned": 10,
  "components": ["cbf", "cf", "kg"],
  "recommendations": [
    {
      "rank": 1,
      "worker": { "...": "same fields as GET /api/workers/<id>" },
      "score": 0.8372,
      "contributions": {"cbf": 0.2372, "cf": 0.1, "kg": 0.5},
      "dominant_component": "kg",
      "explanation": "Recommended because knowledge-graph evidence (KG) contributed 0.50, profile match (CBF) contributed 0.24, collaborative history (CF) contributed 0.10 (hybrid score 0.84). Strongest KG evidence: ...",
      "kg_summary": "Average training-period rating 4.83/5 from 6 reviews (smoothed 4.66); You hired this worker before (1 training-period hire); Has required skill 'Writing'",
      "kg_evidence": [
        {"path_type": "repeat_hire", "text": "You hired this worker before (1 training-period hire)"},
        {"path_type": "skill", "text": "Has required skill 'Writing'"},
        {"path_type": "category", "text": "Specializes in 'Creative Writing' (primary category)"}
      ]
    }
  ]
}
```

**Field meanings:**
- `score` = `contributions.cbf + contributions.cf + contributions.kg` exactly. Each contribution is the component weight × that component's score min-max normalised within this request's pool, so it lies in `[0, weight]`.
- `explanation` is the frozen Day 8 hybrid explanation. It cites the components in order of their actual contribution and names any component that contributed nothing.
- `kg_summary` and `kg_evidence` come from the frozen Day 7 KG explainer. They are the strongest real training-graph paths for that worker in that pool, up to 3. They are `null` / `[]` when not available.

## Page route (Day 10)

`GET /` serves the HTML frontend: a static shell from `src/app/templates/`, with assets under `/static/`. All of its data comes from the `/api` routes above; nothing is rendered server-side into the HTML. Every response carries the same security headers:
- `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; …`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: no-referrer`
- `X-Frame-Options: DENY`

## Errors

Every error has the same body, and none contains a stack trace or internal detail:

```json
{"error": {"code": "invalid_k", "message": "Query parameter 'k' must be an integer between 1 and 50."}}
```

| Status | Codes |
|---|---|
| 400 | `missing_link`, `invalid_link`, `invalid_k`, `invalid_worker_id`, `invalid_page`, `invalid_per_page`, `invalid_q`, `invalid_category`, `invalid_skill`, `bad_request` |
| 404 | `unknown_request`, `unknown_worker`, `not_found` (unknown route) |
| 405 | `method_not_allowed` (every endpoint is GET-only) |
| 409 | `request_not_recommendable` |
| 503 | `recommender_unavailable`, `neo4j_unavailable` |
| 500 | `internal_error` (details are logged server-side only) |

## Dependency behaviour

| Situation | Behaviour |
|---|---|
| Neo4j down at start-up | The app starts normally. Health reports `degraded` with `neo4j.available: false`. Recommendations, request lookups and worker lookups still work. `/workers/<id>/graph` returns 503 |
| Frozen Day 8 artifacts modified | The recommender refuses to load (`frozen_artifact_mismatch`). Health returns 503 and recommendation routes return 503 |
| Processed data missing | Same as above, with reason `load_failed` |
