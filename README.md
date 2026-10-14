# HireWise

## Knowledge Graph-Enhanced Hybrid Recommender System for Digital Worker-Service Platforms

HireWise recommends workers for service requests by combining:
- Content-Based Filtering (CBF);
- Collaborative Filtering (CF);
- Knowledge Graph (KG) reasoning.

These are fused in a validation-tuned weighted hybrid and exposed through
a Flask web application with graph-path explanations.

**Data.** Real job demand comes from the *Upwork Job Postings Dataset 2024*
(Kaggle). Workers, clients and interactions are synthetic, produced by a
frozen, documented generator (`docs/experimental_design.md`). All results
describe performance within that constructed experimental environment.

## Project status

Days 1–12 are complete:

| Day | Milestone | Report |
|---|---|---|
| 1–3 | Data analysis, cleaning, controlled skill vocabulary, synthetic data, temporal split | `docs/experimental_design.md`, `docs/data_quality_report.md`, `docs/synthetic_data_report.md` |
| 4 | Random, Popularity and CBF baselines | `docs/day4_recommender_report.md` |
| 5 | Collaborative Filtering | `docs/day5_cf_report.md` |
| 6 | Neo4j knowledge graph | `docs/day6_knowledge_graph_report.md` |
| 7 | KG-only recommender and graph-path explanations | `docs/day7_kg_recommender_report.md` |
| 8 | Weighted hybrid (CBF 0.40 / CF 0.10 / KG 0.50), significance, diversity | `docs/day8_hybrid_report.md` |
| 9 | Flask API | `docs/day9_api.md`, `docs/day9_backend_report.md` |
| 10 | Frontend and end-to-end workflow | `docs/day10_frontend_report.md` |
| 11 | PostgreSQL application database | `docs/day11_postgresql_report.md` |
| 12 | Accounts and marketplace workflow: register, request, book, complete, review | `docs/day12_marketplace_report.md` |

The recommender models, their selected configurations and all
experimental results (`results/`) are **frozen**.

## Architecture

```
src/
  data/          data pipeline (Day 2-3): cleaning, vocabulary, synthetic generation
  recommenders/  CBF, CF, KG, Hybrid + frozen-model reconstruction
  kg/            Neo4j client, graph schema/loader, training-only graph view, path explanations
  evaluation/    protocol (candidate eligibility, relevance), metrics, significance, diversity
  db/            PostgreSQL schema (v2), deterministic loader, data + marketplace repositories
  app/           Flask: factory, config, routes (/api), services, serializers, errors
    marketplace/ accounts, CSRF/sessions, booking rules, runtime recommender, pages (Day 12)
    templates/   Jinja pages            static/  CSS + JavaScript modules
config/          frozen experiment and model configuration (YAML)
data/            git-ignored: raw/ (Upwork CSV) and processed/ (generated data)
results/         frozen experiment outputs (day4, day5, day7, day8)
scripts/         pipeline runners, setup scripts, verify_* checkers, demo
tests/           automated tests
docs/            methodology, day reports, API contract, explanation rubric
```

Requests flow `browser → Flask routes → services → repositories / frozen models`.

**Data stores:**

| Store | Holds | Used for |
|---|---|---|
| **PostgreSQL** | Imported application data (clients, workers, service requests, skills, categories, training-period ratings) and, in separate tables, runtime marketplace data (users, client-created requests, bookings, reviews) | Request browsing and lookup, worker profiles, the marketplace |
| **Neo4j** | The knowledge graph | The optional worker graph view (the KG model uses a training-only snapshot verified against it) |
| **Frozen artifacts** | Deterministic `data/processed` + `results/` | The evaluated models: candidate pools and scoring |

PostgreSQL never feeds the models, so recommendations for the research
requests are byte-identical to the evaluated stack. Client-created
requests are scored by the same frozen hybrid from their observable
features, without retraining or fabricated history (see the Day 12 report).

## Setup (Windows, PowerShell)

1. **Python environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Data.** Place the Upwork CSV at `data/raw/upwork-jobs.csv` (it is git-ignored and not redistributed). Then generate the processed data with the pipeline commands below.
3. **Credentials.** Copy `.env.example` to `.env` and fill in your local Neo4j and PostgreSQL credentials, plus a session key: `FLASK_SECRET_KEY` (generate one with `python -c "import secrets; print(secrets.token_hex(32))"`). `.env` is git-ignored.
4. **Neo4j.** Start Neo4j (5.x/2026.x), then load the graph:
   ```powershell
   python scripts/prepare_kg_data.py
   python scripts/load_neo4j.py
   python scripts/verify_neo4j.py
   ```
5. **PostgreSQL.** This needs PostgreSQL 18, plus a role `hirewise_app` that owns the databases `hirewise` and `hirewise_test`; see `docs/day11_postgresql_report.md`.
   ```powershell
   python scripts/init_postgres.py      # create missing tables, incl. the marketplace tables (never drops)
   python scripts/load_postgres.py      # load; no-op if identical; refuses differences without --reset
   python scripts/verify_postgres.py
   ```

## Running the application

```powershell
python -m flask --app "src.app:create_app('development')" run
```

Then open http://127.0.0.1:5000/. Register as a client to post a request, see
recommendations, book a worker and leave a review. To demonstrate the worker
side, link a demo worker account to an imported catalogue profile (the
password comes from `HIREWISE_DEMO_PASSWORD` or a hidden prompt):

```powershell
python scripts/create_demo_account.py --show-demo-recommendation
python scripts/create_demo_account.py --role worker --email worker@demo.local --name "Demo Worker" --worker-id W001674
```

- `src.app:create_app` with no arguments uses production settings (debug off).
- If Neo4j is down, only the graph view is unavailable.
- If PostgreSQL is down, the app reports that application data is unavailable.
- To run without starting a server: `python scripts/demo_day9_api.py`.

## Reproducing the pipeline and experiments

```powershell
python scripts/run_data_pipeline.py        # Day 2 (requires data/raw/upwork-jobs.csv)
python scripts/verify_pipeline.py
python scripts/generate_synthetic_data.py  # Day 3
python scripts/verify_synthetic_data.py
python scripts/run_day4_evaluation.py; python scripts/verify_day4.py
python scripts/run_day5_evaluation.py; python scripts/verify_day5.py
python scripts/verify_day6.py              # after the Neo4j steps above
python scripts/run_day7_evaluation.py; python scripts/verify_day7.py
python scripts/run_day8_evaluation.py; python scripts/verify_day8.py
python scripts/verify_day9.py; python scripts/verify_day10.py; python scripts/verify_day11.py
python scripts/verify_day12.py              # writes only to the hirewise_test database
```

## Tests

```powershell
python -m unittest discover -s tests
```

The default suite needs neither database. Live database tests are opt-in:

```powershell
$env:HIREWISE_RUN_POSTGRES_TESTS = "1"     # read-only on hirewise; destructive/marketplace tests only on hirewise_test
$env:HIREWISE_RUN_NEO4J_WRITE_TESTS = "1"  # temporary uniquely-labelled nodes in Neo4j
```
