# Day 10 Report: Frontend and End-to-End Integration

**Objective.** Give HireWise a usable frontend for the full recommendation
workflow, integrated with the Day 9 Flask API. The frozen recommender,
weights, results and evaluation protocol are unchanged. Day 10 adds no
authentication, PostgreSQL, deployment or retuning.

## Running it locally

```bash
python -m flask --app "src.app:create_app('development')" run
# then open http://127.0.0.1:5000/
```

- **No extra environment variables are needed** to browse and get recommendations. The processed data and frozen results must exist locally.
- **The graph view needs Neo4j.** Set the usual `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD` in `.env` (see `.env.example`). Without them, the app runs in a clearly labelled degraded mode.
- `src.app:create_app` with no arguments uses production settings (debug off).

## 1. Frontend architecture

| File | Role |
|---|---|
| `src/app/pages.py` | `GET /`: renders the page shell (a separate blueprint from `/api`) |
| `src/app/templates/base.html` | Layout: header with live system status, main area, footer |
| `src/app/templates/index.html` | Three numbered panels (choose request → selected request → recommended workers) and the worker-details dialog |
| `src/app/static/css/style.css` | Restrained, framework-free styling; responsive two-column → one-column below 900 px |
| `src/app/static/js/api.js` | **The only module that calls `fetch`.** URL-encodes links, parses JSON, and turns HTTP and network failures into `ApiError(code)` |
| `src/app/static/js/format.js` | Pure presentation helpers: number and percent formatting, component labels, friendly error messages |
| `src/app/static/js/dom.js` | Text-safe DOM building (`createElement` + `textContent`; never `innerHTML`) |
| `src/app/static/js/app.js` | UI state and wiring for browse → recommend → explain → worker → graph |

The JavaScript uses native ES modules: no framework, bundler or build
step. The HTML is a static shell. Every request, recommendation and
worker value is fetched from `/api` and inserted as text, so nothing
from the dataset is rendered server-side into the page.

## 2. User workflow

1. **Open HireWise.** The header shows the system status from `/api/health`: ready, ready but graph unavailable, or unavailable.
2. **Choose a request.** Users browse the paginated list (10 per page) and can filter by title text, category (a dropdown built from the API) or required skill. Selecting a request shows its category, skills, pay type, budget percentile, location requirement and number of eligible workers. Internally the request is identified by its exact `link`.
3. **Get recommendations.** Users choose Top 5, 10 (the default) or 20 and press "Get recommendations".
4. **Read the results.** Each card shows:
   - rank, worker ID, hybrid score;
   - primary category, location, experience, training-review summary;
   - declared skills;
   - contribution bars;
   - the model's explanation;
   - an expandable **"Why this recommendation?"** section.
5. **Inspect a worker.** "View worker …" opens an accessible modal `<dialog>` with the profile from `/api/workers/<id>`.
6. **Optional graph view.** "Show graph details" reads the worker's skills, specialisations and location live from Neo4j via `/api/workers/<id>/graph`.

## 3. Backend extension (the only API change)

`GET /api/requests` without `link` now returns a **paginated listing**:
- only recommendable (validation/test) requests, 3,600 in total;
- fixed chronological order (simulated day, then link);
- `per_page` defaults to 20, maximum 50;
- optional `q` / `category` / `skill` filters;
- `n_eligible_workers` per item, and an `available_categories` list.

Items reuse the same allow-listed request serializer as the single
lookup. The existing `?link=` lookup is unchanged. The page route `/`
and simple security headers are the other additions; see
`docs/day9_api.md` for the full contract.

Recommendation, worker and graph routes, the service's recommendation
path, and every frozen module are untouched.

## 4. Contributions and explanations

- **Contribution bar.** A single stacked bar, sized on a 0–1 scale, shows the three exact contributions, which sum to the hybrid score:
  - Profile/content (CBF), blue;
  - Collaborative (CF), amber;
  - Knowledge graph (KG), green.

  A legend gives each value to 3 decimals and its share of the score. Tiny non-zero values display as `<0.001` / `<0.1%`, never as 0. The page states plainly that these are relative ranking values, not probabilities. The bar has a spoken `aria-label` summary.
- **Exact values, not rounded text.** The numbers come from the API's `contributions` field, not from the explanation sentence. This works around the Day 8 template's rounding, where a CF contribution of 0.005 reads "0.00".
- **Explanation.** The Day 8/9 explanation sentence is shown exactly as returned.
- **"Why this recommendation?"** expands to show the API's `kg_summary` and its strongest KG paths (`kg_evidence`). Each path's text comes from the API, prefixed with a display label for its type (for example "Related skill:" or "Your hiring history:"). No reason is composed in JavaScript (a test checks this), and no Cypher is ever shown.

## 5. Worker details and the KG graph view

- **Profile fields shown:**
  - primary and secondary categories, location;
  - experience level and years;
  - rate (labelled a synthetic profile value, with no invented currency);
  - price level as a within-category percentile;
  - training-period review count and mean;
  - simulated join day;
  - declared skills.
- **No "new worker" badge.** The new-entrant cohort flag is never sent by the API or used. Reviews read "No training-period reviews yet" when the visible review count is 0, which is an accurate, transparent fact.
- **The graph view is optional and secondary.** It shows skills, specialisations (with primary/secondary role) and location, each labelled with its relationship type, as a simple structured list. No graph-visualisation library is used.

## 6. Loading, error and degraded states

| Situation | What the user sees |
|---|---|
| Loading the request list, recommendations, worker or graph | "Loading…" messages in `aria-live` regions; `aria-busy` on the list being loaded |
| No requests match the filters | "No requests match these filters." |
| Unknown request / invalid input | Friendly messages mapped from the API error codes |
| No eligible workers | "No eligible workers were found for this request." |
| Server unreachable | "The HireWise server could not be reached…" (network error) |
| Recommender not loaded | Header: "Recommendations are currently unavailable."; requests explain why |
| **Neo4j down** | Header: "Recommender ready. Graph details temporarily unavailable." The graph section explains it is temporarily unavailable. **Browsing and recommendations keep working** |
| Unexpected backend error | "Something went wrong on the server. Please try again." No raw exception text ever reaches the page; the API never sends any |

## 7. Accessibility, usability and security basics

- **Accessibility and usability:**
  - semantic landmarks, a skip link and meaningful headings;
  - every control labelled;
  - native buttons, `<details>` and `<dialog>`, all keyboard-operable;
  - a visible `:focus-visible` outline;
  - text colours above 4.5:1 contrast on white (measured: body text 15.6:1, muted 6.9:1, accent and white-on-accent buttons 6.6:1, status colours 6.3–7.5:1);
  - responsive layout for ordinary laptop screens.
- **Security:**
  - API content is always inserted as text (no `innerHTML` and similar sinks, enforced by tests);
  - links are URL-encoded via `URLSearchParams`, and worker IDs via `encodeURIComponent`;
  - no secrets, environment variables or Cypher in any asset.
- **Security headers:**
  - a strict same-origin CSP with no `unsafe-inline`, which is possible because the page has no inline script or style;
  - `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and `X-Frame-Options: DENY`.

## 8. Tests and verification

- **`tests/test_day10_frontend.py`** (20 tests):
  - the page renders all main sections, labelled controls and default K = 10;
  - assets are referenced and served;
  - no inline script, style or event handlers; CSP present;
  - no dataset or private values in the HTML;
  - JavaScript safety: no unsafe DOM sinks, `fetch` only in `api.js`, no secrets, Cypher or cohort flag, links encoded, explanations never generated in JS;
  - **every field `app.js` reads exists in the API contract**;
  - every API error code has a friendly message;
  - listing defaults, determinism, chronological order, allow-listed fields, pool sizes equal to the protocol's, pagination validation and filters;
  - the single lookup is unchanged;
  - a full **end-to-end flow** (health → list → recommendations → worker → graph) with Neo4j simulated as down.
- **Full suite:** 446 tests pass (2 skipped: the opt-in Neo4j write tests).
- **`scripts/verify_day10.py`:** 24 of 24 checks pass. It covers:
  - the page and assets;
  - listing completeness (3,600 unique requests in chronological order) and filtering;
  - **parity of recommendations for 24 listed requests with a freshly rebuilt frozen Day 8 hybrid**;
  - the JavaScript field contract checked against *real* responses;
  - the worker view, and the graph view both live and degraded;
  - determinism and prior-artifact hashes;
  - **a full run of the Day 9 verifier**, which also passes.
- **JavaScript unit tests:** none were run. Node.js isn't installed, and adding a Node toolchain only for tests was out of scope. The pure helpers in `format.js` are deliberately small; their behaviour is covered indirectly by the contract and safety tests.

## 9. End-to-end check over real HTTP

The development server was started with the documented command and
exercised with `curl`:
- `/` returned 200 with the security headers;
- the listing returned 3,600 requests;
- the first listed request returned a Top-10 (10 of 65 eligible);
- that worker's detail and graph routes both returned 200;
- the server log contained no tracebacks.

Browser rendering was not automated (no Selenium/Playwright). Screenshots
of the five demo states can be captured manually by running the command
above:
1. the request list;
2. the populated Top-10;
3. an expanded "Why this recommendation?";
4. the worker dialog;
5. the graph-unavailable state, by stopping Neo4j.

## 10. Performance (local, indicative)

Measured through the Flask test client (median / mean, n = 20):

| Call | Median | Mean |
|---|---|---|
| `GET /` | 0.4 ms | 1.0 ms |
| `GET /api/requests?page=n&per_page=10` | 1.5 ms | 1.7 ms |
| `GET /api/requests?skill=python` | 4.8 ms | 5.5 ms |
| `GET /api/recommendations` (new requests) | 29 ms | 46 ms |
| `GET /api/health` (includes the Neo4j probe, n = 5) | 31 ms | 150 ms (first connection) |

## 11. Known limitations

1. **No JavaScript unit tests run in a JS runtime** (Node isn't available).
2. **No automated browser tests**, so rendering was checked by contract tests and manual use, not pixel tests.
3. **Only existing validation/test requests can be browsed and recommended.** There is no form to create a new request; scoring a new request needs a request-creation path.
4. **The graph view shows the worker's own static profile edges only.** It is not a visual graph, and it never shows interaction edges (`HIRED` / `REVIEWED`), because those include validation/test outcomes.
5. **The rounding quirk in the Day 8 explanation text remains**, as frozen. The UI's numeric contribution display is exact.
6. **The listing is filtered in memory** over 3,600 requests. That is adequate here, but it is not a search engine.

## 12. Day 11 handoff

The complete workflow (browse → recommend → explain → worker → graph)
works through the frontend on top of the frozen stack. Open items:
- capture the five demo screenshots;
- decide whether a request-creation form is in scope;
- complete the human rubric ratings (Day 7 sheet);
- refresh the stale README (status and commands);
- the outstanding methodology items from Day 8: generator sensitivity runs and claim framing.
