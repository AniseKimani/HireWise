"""Independent verification of the Day 9 Flask backend.

Exercises the real application factory (production configuration) and
re-derives everything the API returns from the frozen models directly:

- app creates; /api/health reports the recommender ready and Neo4j status;
- request, worker and recommendation routes work for a deterministic
  sample of validation/test requests; Top-10 returned, ranks 1..n, no
  duplicates, scores sorted, contributions within [0, weight] and summing
  to the score, explanations present;
- API rankings, scores and contributions equal a direct invocation of the
  frozen Day 8 hybrid on the protocol's own candidate pools;
- responses are byte-identical on repetition and from a fresh app;
- responses contain only allow-listed keys (no generator-private or
  outcome fields);
- Neo4j node/relationship counts are identical before and after (the API
  is read-only); all service Cypher is fixed and write-free;
- Day 4-8 configs/results/manifests, the generator, and the Day 7 rubric
  sheet are unchanged.

Usage:
    python scripts/verify_day9.py

Exits non-zero if any check fails. Never writes anything.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.app import create_app
from src.app import services
from src.app.serializers import RECOMMENDATION_FIELDS, REQUEST_FIELDS, WORKER_FIELDS
from src.data.config import load_config
from src.data.io import sha256_of
from src.evaluation.config import load_evaluation_config
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.recommenders.cf_config import load_cf_config
from src.recommenders.frozen_models import build_frozen_models
from src.recommenders.hybrid import HybridRecommender, cbf_score_fn, cf_score_fn, kg_score_fn

FAILURES: list[str] = []
FROZEN_PATHS = ["config/generator_v1.yaml", "config/evaluation_config.yaml", "config/cf_config.yaml", "config/kg_config.yaml",
                "config/hybrid_config.yaml", "results", "docs/day7_explanation_rubric_sheet.csv", "data"]
PRIVATE_KEY_MARKERS = ("quality", "proficiency", "true_skill", "taste", "utility", "noise", "oracle", "cold_start",
                       "relevance", "grade", "event", "shortlist", "hired", "outcome", "private")


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
    counts_before = neo4j_counts()

    # ---- Frozen artifacts ------------------------------------------------------
    with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
        check("frozen generator unchanged", sha256_of(REPO_ROOT / "config" / "generator_v1.yaml") == fh.read().split()[0])
    for manifest in ("results/day4/day4_manifest.json", "results/day5/day5_manifest.json", "results/day7/day7_manifest.json",
                     "results/day8/day8_manifest.json", "data/processed/kg/kg_prepare_manifest.json"):
        with open(REPO_ROOT / manifest, encoding="utf-8") as fh:
            drift = [e["path"] for e in json.load(fh)["files"] if sha256_of(REPO_ROOT / e["path"]) != e["sha256"]]
        check(f"every artifact in {manifest} matches its hash", not drift, f"{drift}")
    with open(REPO_ROOT / "results" / "day8" / "selected_hybrid_config.json", encoding="utf-8") as fh:
        selected = json.load(fh)
    check("Day 8 selected hybrid unchanged (CBF 0.40 / CF 0.10 / KG 0.50, per-request min-max)",
          selected["weights"] == {"cbf": 0.4, "cf": 0.1, "kg": 0.5} and selected["normalization"] == "minmax_within_candidate_pool"
          and sha256_of(REPO_ROOT / "config" / "hybrid_config.yaml") == selected["hybrid_config_sha256"])
    diff = subprocess.run(["git", "status", "--porcelain", "--", *FROZEN_PATHS], cwd=REPO_ROOT, capture_output=True, text=True)
    check("no uncommitted change to frozen configs, results, rubric sheet, or data", diff.returncode == 0 and not diff.stdout.strip(),
          diff.stdout.strip()[:300])
    rubric = pd.read_csv(REPO_ROOT / "docs" / "day7_explanation_rubric_sheet.csv")
    check("Day 7 human rubric sheet still blank",
          rubric[["relevance_1to5", "clarity_1to5", "traceability_1to5", "usefulness_1to5"]].isna().all().all())

    # ---- App, health -------------------------------------------------------------
    app = create_app("production")
    client = app.test_client()
    check("app creates with debug off in production config", not app.debug and "hirewise_service" in app.extensions)
    health = client.get("/api/health")
    body = health.get_json()
    check("health route works and reports recommender ready", health.status_code == 200 and body["recommender"]["ready"])
    if counts_before is not None:
        check("health reports Neo4j available with Day 6 counts", body["neo4j"] == {"available": True, "checked": True, "graph_matches_day6_counts": True})
    else:
        check("health honestly reports Neo4j unavailable", body["neo4j"]["available"] is False and body["status"] == "degraded")

    # ---- Direct frozen hybrid ------------------------------------------------------
    data = load_protocol_data()
    eval_cfg = load_evaluation_config()
    index = EligibilityIndex(data.workers, data.known_countries)
    sample = [r for s in ("validation", "test") for r in build_eval_requests(data, index, s, eval_cfg) if not r.excluded][::100]
    models = build_frozen_models(data, load_config().seed, load_cf_config().min_similarity)
    direct = HybridRecommender({"cbf": cbf_score_fn(models["cbf"]), "cf": cf_score_fn(models["cf"]), "kg": kg_score_fn(models["kg"])},
                               selected["weights"])

    allowed = set(REQUEST_FIELDS) | set(WORKER_FIELDS) | set(RECOMMENDATION_FIELDS) | {
        "request", "model", "k", "n_candidates", "n_returned", "components", "recommendations", "name", "weights",
        "normalization", "selected_config_sha256", "hybrid_config_sha256", "cbf", "cf", "kg", "path_type", "text"}
    structure_ok = parity_ok = keys_ok = explanations_ok = pool_ok = True
    for req in sample:
        r = client.get("/api/recommendations", query_string={"link": req.link})
        body = r.get_json()
        recs = body["recommendations"]
        ids = [x["worker"]["worker_id"] for x in recs]
        scores = [x["score"] for x in recs]
        structure_ok &= (r.status_code == 200 and body["k"] == 10 and body["n_returned"] == len(recs) == min(10, req.n_candidates)
                         and [x["rank"] for x in recs] == list(range(1, len(recs) + 1)) and len(set(ids)) == len(ids)
                         and set(ids) <= set(req.candidates) and scores == sorted(scores, reverse=True)
                         and all(abs(sum(x["contributions"].values()) - x["score"]) < 1e-12 for x in recs)
                         and all(0 <= x["contributions"][c] <= selected["weights"][c] + 1e-12 for x in recs for c in ("cbf", "cf", "kg")))
        pool_ok &= body["n_candidates"] == req.n_candidates
        expected = direct.explain(req.link, req.candidates, 10)
        parity_ok &= (ids == [e.worker_id for e in expected] and scores == [e.score for e in expected]
                      and [x["contributions"] for x in recs] == [e.contributions for e in expected])
        explanations_ok &= all(x["explanation"] and (x["contributions"]["kg"] == 0 or (x["kg_summary"] and x["kg_evidence"])) for x in recs)
        found = set(keys_of(body))
        keys_ok &= found <= allowed and not any(m in key.lower() for key in found for m in PRIVATE_KEY_MARKERS)
    check(f"recommendation route: Top-10, ranks 1..n, no duplicates, valid scores/contributions ({len(sample)} requests)", structure_ok)
    check("API candidate pools equal the evaluation protocol's", pool_ok)
    check("API rankings, scores and contributions equal the direct frozen Day 8 hybrid", parity_ok)
    check("explanations present (KG evidence whenever KG contributes)", explanations_ok)

    link = sample[0].link
    first = client.get("/api/recommendations", query_string={"link": link}).get_data()
    fresh = create_app("production").test_client().get("/api/recommendations", query_string={"link": link}).get_data()
    check("recommendations deterministic (repeat and fresh app byte-identical)",
          first == client.get("/api/recommendations", query_string={"link": link}).get_data() == fresh)

    req_body = client.get("/api/requests", query_string={"link": link}).get_json()
    check("request route returns exactly the allow-listed fields", set(req_body) == set(REQUEST_FIELDS) and req_body["link"] == link)
    worker_id = data.workers["worker_id"].iloc[0]
    w_body = client.get(f"/api/workers/{worker_id}").get_json()
    check("worker route returns exactly the allow-listed fields", set(w_body) == set(WORKER_FIELDS) and w_body["worker_id"] == worker_id)
    keys_ok &= set(keys_of(req_body)) <= allowed and set(keys_of(w_body)) <= allowed
    check("no generator-private or outcome fields in any response", keys_ok)
    check("unknown request -> 404, invalid k -> 400, training request -> 409",
          client.get("/api/recommendations", query_string={"link": "https://example.invalid/x"}).status_code == 404
          and client.get("/api/recommendations", query_string={"link": link, "k": "0"}).status_code == 400
          and client.get("/api/recommendations", query_string={
              "link": data.requests.loc[data.requests["split"] == "train", "link"].iloc[0]}).status_code == 409)

    # ---- Neo4j read-only -----------------------------------------------------------
    write_free = all(not any(k in q.upper() for k in ("CREATE", "MERGE", "SET ", "DELETE", "REMOVE", "DETACH", "LOAD CSV", "CALL "))
                     for q in list(services.WORKER_GRAPH_QUERIES.values()) + list(services.COUNT_QUERIES.values()))
    check("service Cypher is fixed and write-free", write_free)
    if counts_before is not None:
        graph = client.get(f"/api/workers/{worker_id}/graph").get_json()
        check("worker graph route reads the live graph",
              graph.get("skills") == sorted(data.workers.iloc[0]["declared_skills"].split("|")))
        check("Neo4j counts identical before and after verification (read-only)", neo4j_counts() == counts_before,
              f"{counts_before} -> {neo4j_counts()}")
    else:
        print("[SKIP] live Neo4j read-only checks -- Neo4j unavailable")

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
