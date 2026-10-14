"""Independent verification of Day 11 (PostgreSQL integration).

Compares the production, PostgreSQL-backed application with a reference
application backed by the frozen artifacts (InMemoryRepository built from
the same model-visible files), response by response, byte by byte:

- schema present; database contents equal the source (fingerprint);
- the production app really reads PostgreSQL (repository type, health);
- every request-listing page, filters, request lookups, all 3,000 worker
  profiles, and a sample of recommendations are byte-identical;
- recommendations also equal a direct call to the frozen Day 8 hybrid;
- responses contain only allow-listed keys; the frontend page and its
  JavaScript field contract still hold (including the new health field);
- Neo4j is untouched (counts before/after) when reachable;
- frozen configs/results unchanged; scripts/verify_postgres.py and
  scripts/verify_day10.py (which runs verify_day9.py) still pass.

Usage:
    python scripts/verify_day11.py

Read-only. Exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import os as _os
import secrets as _secrets

# Verification/demo runs are not deployments: when no FLASK_SECRET_KEY is
# configured, use a process-local random key so the production configuration
# can be exercised. Real production start-up still refuses to run without one
# (src/app/factory.py).
_os.environ.setdefault("FLASK_SECRET_KEY", _secrets.token_hex(32))

from src.app import create_app
from src.app import services
from src.app.serializers import RECOMMENDATION_FIELDS, REQUEST_FIELDS, WORKER_FIELDS
from src.app.services import RecommendationService, probe_neo4j
from src.data.config import load_config
from src.data.io import sha256_of
from src.db.database import make_engine
from src.db.loader import build_tables, fingerprint, read_back, stored_metadata
from src.db.repositories import InMemoryRepository, PostgresRepository
from src.recommenders.cf_config import load_cf_config
from src.recommenders.frozen_models import build_frozen_models
from src.recommenders.hybrid import HybridRecommender, cbf_score_fn, cf_score_fn, kg_score_fn

FAILURES: list[str] = []
APP_JS = (REPO_ROOT / "src" / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
ALLOWED_KEYS = set(REQUEST_FIELDS) | set(WORKER_FIELDS) | set(RECOMMENDATION_FIELDS) | {
    "items", "page", "per_page", "total", "total_pages", "filters", "available_categories", "n_eligible_workers", "q",
    "category", "skill", "request", "model", "k", "n_candidates", "n_returned", "components", "recommendations", "name",
    "weights", "normalization", "selected_config_sha256", "hybrid_config_sha256", "cbf", "cf", "kg", "path_type", "text"}
PRIVATE_KEY_MARKERS = ("cold_start", "relevance", "grade", "shortlist", "hired", "outcome", "quality", "proficiency",
                       "taste", "utility", "noise", "oracle", "private", "title_search", "budget", "hourly")


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def keys_of(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from keys_of(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from keys_of(item)


def neo4j_counts():
    from src.kg.neo4j_client import Neo4jClient

    try:
        with Neo4jClient() as client:
            return {n: client.run_read(q)[0]["c"] for n, q in services.COUNT_QUERIES.items()}
    except Exception:
        return None


def main() -> int:
    neo4j_before = neo4j_counts()

    # ---- frozen artifacts ---------------------------------------------------------
    with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
        check("frozen generator unchanged", sha256_of(REPO_ROOT / "config" / "generator_v1.yaml") == fh.read().split()[0])
    for manifest in ("results/day4/day4_manifest.json", "results/day5/day5_manifest.json", "results/day7/day7_manifest.json",
                     "results/day8/day8_manifest.json", "data/processed/kg/kg_prepare_manifest.json"):
        with open(REPO_ROOT / manifest, encoding="utf-8") as fh:
            drift = [e["path"] for e in json.load(fh)["files"] if sha256_of(REPO_ROOT / e["path"]) != e["sha256"]]
        check(f"every artifact in {manifest} matches its hash", not drift, f"{drift}")
    status = subprocess.run(["git", "status", "--porcelain", "--", "config", "results", "src/recommenders", "src/kg",
                             "src/evaluation", "src/data", "docs/day7_explanation_rubric_sheet.csv"],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    check("no uncommitted change to configs, results, frozen modules or the rubric sheet", not status.stdout.strip(), status.stdout[:300])

    # ---- database = source ----------------------------------------------------------
    source = build_tables()
    with make_engine().connect() as conn:
        db_fp = fingerprint(read_back(conn))
        meta = stored_metadata(conn)
    check("PostgreSQL contents equal the source artifacts exactly (fingerprint)",
          db_fp == fingerprint(source) == meta.get("source_fingerprint"))

    # ---- apps -------------------------------------------------------------------------
    production = create_app("production")
    service = production.extensions.get("hirewise_service")
    check("production app reads PostgreSQL (PostgresRepository)", isinstance(getattr(service, "repository", None), PostgresRepository))
    live = production.test_client()
    health = live.get("/api/health").get_json()
    check("health reports PostgreSQL available with data loaded",
          health["postgresql"] == {"available": True, "data_loaded": True, "checked": True} and health["recommender"]["ready"])
    check("health reports Neo4j honestly", (health["neo4j"]["available"] is True) == (neo4j_before is not None)
          and health["status"] == ("ok" if neo4j_before is not None else "degraded"))
    reference = create_app("testing", service=RecommendationService.build(repository=InMemoryRepository(source))).test_client()

    def same(path, **params):
        a = live.get(path, query_string=params).get_data()
        return a == reference.get(path, query_string=params).get_data(), a

    first = json.loads(live.get("/api/requests?per_page=50").get_data())
    pages_ok = all(same("/api/requests", per_page="50", page=str(p))[0] for p in range(1, first["total_pages"] + 1))
    check(f"all {first['total_pages']} listing pages byte-identical to the artifact reference", pages_ok)
    filters = [{"q": "logo"}, {"q": "пис"}, {"skill": "PYTHON"}, {"category": "Logo Design"},
               {"q": "design", "skill": "Adobe Photoshop"}, {"page": "3", "per_page": "7"}, {"q": "zzzz"}, {"page": "9999"}]
    check("filtered listings byte-identical", all(same("/api/requests", **f)[0] for f in filters))
    items = [i for p in range(1, first["total_pages"] + 1) for i in live.get(f"/api/requests?per_page=50&page={p}").get_json()["items"]]
    check("listing covers 3,600 recommendable requests", len(items) == 3600 == len({i["link"] for i in items}))
    check("request lookups byte-identical (every 100th listed + unknown)",
          all(same("/api/requests", link=i["link"])[0] for i in items[::100]) and same("/api/requests", link="https://example.invalid/x")[0])
    workers_ok = all(same(f"/api/workers/{w}")[0] for w in service.data.workers["worker_id"])
    check("all 3,000 worker profiles byte-identical", workers_ok)

    # ---- recommendations: byte parity + frozen hybrid ------------------------------------
    with open(REPO_ROOT / "results" / "day8" / "selected_hybrid_config.json", encoding="utf-8") as fh:
        weights = json.load(fh)["weights"]
    check("Day 8 selected hybrid unchanged", weights == {"cbf": 0.4, "cf": 0.1, "kg": 0.5})
    models = build_frozen_models(service.data, load_config().seed, load_cf_config().min_similarity)
    direct = HybridRecommender({"cbf": cbf_score_fn(models["cbf"]), "cf": cf_score_fn(models["cf"]), "kg": kg_score_fn(models["kg"])}, weights)
    sample = items[::60]
    rec_bytes = rec_frozen = keys_ok = True
    for item in sample:
        ok, raw = same("/api/recommendations", link=item["link"])
        rec_bytes &= ok
        body = json.loads(raw)
        # Called exactly as the service calls it: with the frozen Day 7 KG explainer attached.
        expected = direct.explain(item["link"], service.candidates_for(item["link"]), 10, kg_recommender=models["kg"], kg_paths=3)
        recs = body["recommendations"]
        rec_frozen &= ([r["worker"]["worker_id"] for r in recs] == [e.worker_id for e in expected]
                       and [r["score"] for r in recs] == [e.score for e in expected]
                       and [r["contributions"] for r in recs] == [e.contributions for e in expected]
                       and [r["explanation"] for r in recs] == [e.text for e in expected]
                       and [r["kg_summary"] for r in recs] == [e.kg_summary for e in expected]
                       and body["n_candidates"] == item["n_eligible_workers"])
        keys = set(keys_of(body))
        keys_ok &= keys <= ALLOWED_KEYS and not any(m in k for k in keys for m in PRIVATE_KEY_MARKERS)
    check(f"recommendations byte-identical to the artifact reference ({len(sample)} requests)", rec_bytes)
    check("recommendations equal the direct frozen Day 8 hybrid (ranking, scores, contributions, explanations, KG evidence, pool)", rec_frozen)
    keys_ok &= all(not any(m in k for m in PRIVATE_KEY_MARKERS) for k in keys_of(first))
    check("no private, outcome, or internal database fields in responses", keys_ok)

    # ---- frontend contract --------------------------------------------------------------
    page = live.get("/").get_data(as_text=True)
    check("frontend page renders with its assets", all(f'id="{i}"' in page for i in ("request-list", "recommendation-list", "worker-dialog"))
          and live.get("/static/js/app.js").status_code == 200)
    used = set(re.findall(r"(?<![\w.])health\.([a-z0-9_]+)", APP_JS))
    check("every health field the frontend reads exists", used <= set(health), f"{used - set(health)}")
    check("frontend handles the database-unavailable state",
          "health.postgresql" in APP_JS and "database_unavailable:" in (REPO_ROOT / "src/app/static/js/format.js").read_text(encoding="utf-8"))

    # ---- Neo4j untouched ---------------------------------------------------------------------
    if neo4j_before is not None:
        check("Neo4j counts unchanged (read-only)", neo4j_counts() == neo4j_before)
    else:
        print("[SKIP] Neo4j read-only count check -- Neo4j not running (the app reports this as degraded)")

    # ---- other verifiers -----------------------------------------------------------------
    for script in ("verify_postgres.py", "verify_day10.py"):
        run = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / script)], cwd=REPO_ROOT, capture_output=True, text=True)
        check(f"scripts/{script} passes", run.returncode == 0, run.stdout[-600:])

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
