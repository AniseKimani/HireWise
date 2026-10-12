"""Independent verification of the Day 10 frontend + end-to-end integration.

No browser automation: the frontend's data contract is checked against
real API responses, and the page/asset/security properties directly.

- the app starts in testing and production modes; `/` renders the main
  sections and its assets, with a strict CSP and no inline script/style;
- the request listing is deterministic, chronological, complete
  (3,600 validation/test requests), allow-listed, and filterable;
- for a deterministic sample of listed requests, the API's ranking,
  scores and contributions equal a direct call to the frozen Day 8
  hybrid (rebuilt independently here);
- every field the JavaScript reads exists in the real responses;
- worker details match the recommendation payload; the graph view works
  against live Neo4j (read-only) when reachable, and degrades to a 503
  `neo4j_unavailable` while recommendations keep working when it is not;
- responses are deterministic; prior configs/results are unchanged; the
  Day 9 verifier still passes.

Usage:
    python scripts/verify_day10.py

Exits non-zero if any check fails. Never writes anything.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.app import create_app
from src.app.serializers import REQUEST_FIELDS, WORKER_FIELDS
from src.app.services import Neo4jUnavailable, RecommendationService, probe_neo4j
from src.data.config import load_config
from src.data.io import sha256_of
from src.recommenders.cf_config import load_cf_config
from src.recommenders.frozen_models import build_frozen_models
from src.recommenders.hybrid import HybridRecommender, cbf_score_fn, cf_score_fn, kg_score_fn

FAILURES: list[str] = []
APP_JS = (REPO_ROOT / "src" / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
FORBIDDEN_KEYS = ("cold_start", "relevance", "grade", "shortlist", "hired", "outcome", "quality", "proficiency",
                  "taste", "utility", "noise", "oracle", "private")


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


def fields_read(variable: str) -> set[str]:
    return set(re.findall(rf"(?<![\w.]){variable}\.([a-z0-9_]+)", APP_JS))


def main() -> int:
    # ---- Prior artifacts ---------------------------------------------------------
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

    # ---- Apps ------------------------------------------------------------------------
    service = RecommendationService.build()
    testing = create_app("testing", service=service).test_client()
    check("app starts in testing mode", testing.get("/api/health").get_json()["recommender"]["ready"])
    production_app = create_app("production")
    check("app starts in production mode with debug off", not production_app.debug and "hirewise_service" in production_app.extensions)
    client = production_app.test_client()

    page = client.get("/")
    html = page.get_data(as_text=True)
    check("/ renders the main sections",
          page.status_code == 200 and all(f'id="{i}"' in html for i in
              ("request-browser", "request-list", "recommend-form", "recommendation-list", "worker-dialog", "graph-load")))
    check("/ references and serves its CSS/JS assets",
          all(a in html for a in ("/static/css/style.css", "/static/js/app.js"))
          and all(client.get(f"/static/{a}").status_code == 200 for a in ("css/style.css", "js/app.js", "js/api.js", "js/format.js", "js/dom.js")))
    csp = page.headers.get("Content-Security-Policy", "")
    check("strict CSP, no inline script/style in the page",
          "script-src 'self'" in csp and "unsafe-inline" not in csp and "style=" not in html
          and not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html))

    # ---- Listing -----------------------------------------------------------------------
    first = client.get("/api/requests?per_page=50").get_data()
    check("request listing deterministic", first == client.get("/api/requests?per_page=50").get_data())
    listing = json.loads(first)
    pages = [client.get(f"/api/requests?per_page=50&page={p}").get_json()["items"] for p in range(1, listing["total_pages"] + 1)]
    items = [i for p in pages for i in p]
    check("listing covers all 3,600 recommendable requests exactly once, in chronological order",
          listing["total"] == 3600 == len(items) == len({i["link"] for i in items})
          and [(i["simulated_day"], i["link"]) for i in items] == sorted((i["simulated_day"], i["link"]) for i in items)
          and {i["split"] for i in items} == {"validation", "test"})
    check("listing items contain only allow-listed, non-private fields",
          all(set(i) == set(REQUEST_FIELDS) | {"n_eligible_workers"} for i in items)
          and not any(m in k.lower() for k in keys_of(listing) for m in FORBIDDEN_KEYS))
    python_items = client.get("/api/requests?skill=Python&per_page=50").get_json()
    check("filters restrict results correctly",
          python_items["total"] > 0 and all("python" in [s.lower() for s in i["required_skills"]] for i in python_items["items"])
          and client.get("/api/requests?page=0").status_code == 400 and client.get("/api/requests?per_page=51").status_code == 400)

    # ---- Recommendations: parity + frontend contract ------------------------------------
    with open(REPO_ROOT / "results" / "day8" / "selected_hybrid_config.json", encoding="utf-8") as fh:
        weights = json.load(fh)["weights"]
    check("Day 8 selected hybrid unchanged", weights == {"cbf": 0.4, "cf": 0.1, "kg": 0.5})
    models = build_frozen_models(service.data, load_config().seed, load_cf_config().min_similarity)
    direct = HybridRecommender({"cbf": cbf_score_fn(models["cbf"]), "cf": cf_score_fn(models["cf"]), "kg": kg_score_fn(models["kg"])}, weights)
    sample = items[::150]
    parity = contract = workers_ok = True
    used = {"result": fields_read("result"), "rec": fields_read("rec"), "worker": fields_read("worker"),
            "evidence": fields_read("evidence"), "item": fields_read("item")}
    for item in sample:
        body = client.get("/api/recommendations", query_string={"link": item["link"]}).get_json()
        expected = direct.explain(item["link"], service.candidates_for(item["link"]), 10)
        recs = body["recommendations"]
        parity &= ([r["worker"]["worker_id"] for r in recs] == [e.worker_id for e in expected]
                   and [r["score"] for r in recs] == [e.score for e in expected]
                   and [r["contributions"] for r in recs] == [e.contributions for e in expected])
        contract &= used["result"] - {"total", "total_pages", "available_categories", "items"} <= set(body)
        contract &= used["item"] <= set(item)
        for r in recs:
            contract &= used["rec"] <= set(r) and used["worker"] <= set(r["worker"])
            contract &= all(used["evidence"] <= set(ev) for ev in r["kg_evidence"])
        w = recs[0]["worker"]["worker_id"]
        workers_ok &= client.get(f"/api/workers/{w}").get_json() == recs[0]["worker"]
    check(f"API recommendations equal the direct frozen Day 8 hybrid ({len(sample)} listed requests)", parity)
    check("every field the frontend reads exists in real API responses", contract)
    check("worker detail matches the recommendation payload", workers_ok)
    link = sample[0]["link"]
    check("recommendations deterministic",
          client.get("/api/recommendations", query_string={"link": link}).get_data()
          == client.get("/api/recommendations", query_string={"link": link}).get_data())

    # ---- Graph view: live and degraded --------------------------------------------------
    worker_id = client.get("/api/recommendations", query_string={"link": link}).get_json()["recommendations"][0]["worker"]["worker_id"]
    if probe_neo4j()["available"]:
        graph = client.get(f"/api/workers/{worker_id}/graph").get_json()
        check("graph view works against live Neo4j (read-only)", set(graph) == {"worker_id", "skills", "categories", "location", "source"})
    else:
        print("[SKIP] live graph view -- Neo4j unavailable")

    def unavailable(worker):
        raise Neo4jUnavailable("down")

    degraded = create_app("testing", service=service, neo4j_probe=lambda: {"available": False, "graph_matches_day6_counts": None},
                          worker_graph_reader=unavailable).test_client()
    g = degraded.get(f"/api/workers/{worker_id}/graph")
    check("graph view degrades to 503 neo4j_unavailable; health reports degraded; recommendations still work",
          g.status_code == 503 and g.get_json()["error"]["code"] == "neo4j_unavailable"
          and degraded.get("/api/health").get_json()["status"] == "degraded"
          and degraded.get("/api/recommendations", query_string={"link": link}).status_code == 200)

    # ---- Day 9 verifier -------------------------------------------------------------------
    day9 = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "verify_day9.py")], cwd=REPO_ROOT, capture_output=True, text=True)
    check("Day 9 verifier still passes", day9.returncode == 0, day9.stdout[-500:])

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
