"""Independent verification of Day 12 (authentication + marketplace).

Writes ONLY to POSTGRES_TEST_DB (schema created/loaded if needed; runtime
tables cleared before and after). The main database and Neo4j are only
read, and are checked to be unchanged.

Checks: registration and password hashing (read back with direct SQL),
login/logout, generic errors, role guards, CSRF, client request creation
(price tier re-derived independently with pandas rank), runtime
recommendations re-computed independently from the frozen components
(no fabricated history), booking creation/duplicates/eligibility,
state transitions incl. rejected ones, completion, review, duplicate and
invalid reviews, database constraints, Day 10 browsing, frozen artifacts,
Neo4j counts, and the Day 11 verifier chain (verify_day11 -> verify_postgres,
verify_day10 -> verify_day9).

Usage:
    python scripts/verify_day12.py
"""
from __future__ import annotations

import json
import os
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

import numpy as np
import pandas as pd
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError

from src.app import create_app
from src.app.marketplace.runtime import RuntimeRecommender, RuntimeRequest
from src.app.marketplace.security import hash_password
from src.app.services import COUNT_QUERIES, RecommendationService
from src.data.io import sha256_of
from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine
from src.db.loader import build_tables, fingerprint, load, read_back
from src.db.marketplace import PostgresMarketplaceRepository
from src.db.repositories import PostgresRepository
from src.db.schema import RUNTIME_TABLES, app_reviews, metadata, users
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender, cbf_score_fn, kg_score_fn
from src.recommenders.knowledge_graph import KGComponentCache, KnowledgeGraphRecommender

FAILURES: list[str] = []
TOKEN = re.compile(r'name="csrf_token" value="([^"]+)"')
PASSWORD = "verify-day12-password"


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def neo4j_counts():
    from src.kg.neo4j_client import Neo4jClient

    try:
        with Neo4jClient() as client:
            return {n: client.run_read(q)[0]["c"] for n, q in COUNT_QUERIES.items()}
    except Exception:
        return None


def clear_runtime(engine) -> None:
    with engine.begin() as conn:
        for name in reversed(RUNTIME_TABLES):
            conn.execute(metadata.tables[name].delete())


def independent_tier(category: str, pay_type: str, amount: float) -> float:
    """Day 2 definition, recomputed with pandas rank on reference + the new job."""
    df = pd.read_csv(REPO_ROOT / "data" / "processed" / "prepared_jobs.csv",
                     usecols=["category", "pay_type", "pay_amount", "is_in_scope_category", "is_duplicate", "has_no_skills"])
    ref = df.loc[df["is_in_scope_category"] & ~df["is_duplicate"] & ~df["has_no_skills"] & df["pay_amount"].notna()
                 & (df["category"] == category) & (df["pay_type"] == pay_type), "pay_amount"]
    values = pd.concat([ref, pd.Series([amount])], ignore_index=True)
    return float(values.rank(pct=True).iloc[-1])


def main() -> int:
    neo4j_before = neo4j_counts()
    main_engine = make_engine()
    with main_engine.connect() as conn:
        main_fp_before = fingerprint(read_back(conn))
        main_runtime_before = {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in RUNTIME_TABLES}

    # ---- frozen artifacts ---------------------------------------------------------
    with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
        check("frozen generator unchanged", sha256_of(REPO_ROOT / "config" / "generator_v1.yaml") == fh.read().split()[0])
    for manifest in ("results/day4/day4_manifest.json", "results/day5/day5_manifest.json", "results/day7/day7_manifest.json",
                     "results/day8/day8_manifest.json", "data/processed/kg/kg_prepare_manifest.json"):
        with open(REPO_ROOT / manifest, encoding="utf-8") as fh:
            drift = [e["path"] for e in json.load(fh)["files"] if sha256_of(REPO_ROOT / e["path"]) != e["sha256"]]
        check(f"every artifact in {manifest} matches its hash", not drift, f"{drift}")
    check("main database imported data equals the Day 11 source fingerprint", main_fp_before == fingerprint(build_tables()))

    # ---- isolated test database ---------------------------------------------------------
    test_db, main_db = os.environ.get("POSTGRES_TEST_DB"), os.environ.get("POSTGRES_DB")
    if not test_db or test_db == main_db:
        check("POSTGRES_TEST_DB configured and distinct from POSTGRES_DB", False)
        return 1
    settings = load_postgres_settings(TEST_DATABASE_ENV)
    engine = make_engine(settings)
    metadata.create_all(engine)
    clear_runtime(engine)
    load(engine)  # imported data (no-op when already identical)
    service = RecommendationService.build(repository=PostgresRepository(engine))
    repo = PostgresMarketplaceRepository(engine)
    runtime = RuntimeRecommender.build(service)
    app = create_app("testing", service=service, marketplace_repository=repo, runtime_recommender=runtime)

    def token(c, path="/login"):
        return TOKEN.search(c.get(path).get_data(as_text=True)).group(1)

    def post(c, path, data=None, token_path="/login"):
        return c.post(path, data={**(data or {}), "csrf_token": token(c, token_path)})

    # ---- secret key: production requires FLASK_SECRET_KEY --------------------------------
    from unittest import mock

    from src.app.factory import MissingSecretKey

    without_key = {k: v for k, v in _os.environ.items() if k not in ("FLASK_SECRET_KEY", "HIREWISE_ENV")}
    try:
        with mock.patch.dict(_os.environ, without_key, clear=True):
            create_app("production")
        refused, message = False, ""
    except MissingSecretKey as exc:
        refused, message = True, str(exc)
    check("production start-up fails clearly without FLASK_SECRET_KEY (no secret in the message)",
          refused and "FLASK_SECRET_KEY is not set" in message
          and not any(v and v in message for k, v in _os.environ.items() if "PASSWORD" in k or k == "FLASK_SECRET_KEY"))
    with mock.patch.dict(_os.environ, without_key, clear=True):
        dev = create_app({"LOAD_RECOMMENDER_ON_STARTUP": False, "NEO4J_HEALTHCHECK_ENABLED": False, "DEBUG": True,
                          "REQUIRE_SECRET_KEY": False, "LOAD_MARKETPLACE": False})
    check("development may run with a temporary key", bool(dev.extensions.get("hirewise_ephemeral_secret_key")))

    # ---- registration / hashing --------------------------------------------------------
    client = app.test_client()
    r = client.post("/register", data={"email": "Verify.Client@Example.com", "role": "client", "display_name": "Verifier",
                                       "password": PASSWORD, "confirm_password": PASSWORD, "csrf_token": token(client, "/register")})
    check("client registration succeeds and logs in", r.status_code == 302 and r.headers["Location"].endswith("/dashboard"))
    with engine.connect() as conn:
        stored = conn.execute(text("SELECT email, password_hash, role FROM users WHERE email = 'verify.client@example.com'")).one()
    from werkzeug.security import check_password_hash

    check("password stored only as a salted hash", PASSWORD not in stored.password_hash
          and stored.password_hash.startswith(("scrypt:", "pbkdf2:")) and check_password_hash(stored.password_hash, PASSWORD))
    dup = app.test_client()
    r = dup.post("/register", data={"email": "VERIFY.CLIENT@example.com", "role": "client", "display_name": "x",
                                    "password": PASSWORD, "confirm_password": PASSWORD, "csrf_token": token(dup, "/register")})
    check("duplicate email (case-insensitive) rejected", r.status_code == 400)
    worker_reg = app.test_client()
    r = worker_reg.post("/register", data={"email": "new.worker@example.com", "role": "worker", "display_name": "New W",
                                           "password": PASSWORD, "confirm_password": PASSWORD, "worker_country": "India",
                                           "primary_category": "Logo Design", "experience_level": "entry",
                                           "experience_years": "1", "rate": "12", "skills": "logo design",
                                           "csrf_token": token(worker_reg, "/register")})
    check("worker registration creates an unlinked runtime profile", r.status_code == 302
          and repo.get_user(repo.get_credentials("new.worker@example.com")[0])["worker_id"] is None)

    # ---- login / logout / guards / CSRF ---------------------------------------------------
    anon = app.test_client()
    check("protected page redirects anonymous users to login", "/login" in anon.get("/dashboard").headers.get("Location", ""))
    wrong = post(anon, "/login", {"email": "verify.client@example.com", "password": "not the password"})
    unknown = post(anon, "/login", {"email": "nobody@example.com", "password": PASSWORD})
    check("wrong password and unknown email get the same generic 401",
          wrong.status_code == unknown.status_code == 401 and "Invalid email or password." in wrong.get_data(as_text=True)
          and "Invalid email or password." in unknown.get_data(as_text=True))
    check("worker cannot reach client-only pages", worker_reg.get("/client/requests/new").status_code == 403)
    check("client cannot reach worker-only pages", client.get("/profile").status_code == 403)
    check("state change without CSRF token rejected", client.post("/client/requests/new", data={"title": "x"}).status_code == 400
          and client.post("/logout").status_code == 400)

    # ---- request creation + runtime recommendations ---------------------------------------------
    r = post(client, "/client/requests/new", {"title": "Verifier logo request", "category": "Logo Design",
                                              "skills": "Logo Design, Adobe Illustrator", "pay_type": "fixed", "budget": "150"},
             token_path="/client/requests/new")
    request_id = int(r.headers["Location"].rstrip("/").split("/")[-1])
    req = repo.get_request(request_id)
    check("request stored with owner, canonical skills and Day 2 price tier (independent pandas rank)",
          req["client_user_id"] == repo.get_credentials("verify.client@example.com")[0]
          and req["skills"] == ["Adobe Illustrator", "Logo Design"]
          and abs(req["price_tier"] - independent_tier("Logo Design", "fixed", 150.0)) < 1e-12)
    page = client.get(f"/client/requests/{request_id}").get_data(as_text=True)
    check("request page lists 10 recommendations", page.count('class="card worker-card"') == 10)

    result = runtime.recommend(RuntimeRequest(request_id, req["client_user_id"], req["category"], tuple(req["skills"]),
                                              req["price_tier"], None, None), 10)
    models = service.models
    day = float(service.data.requests["simulated_day"].max())
    row = pd.DataFrame([{"link": "verify", "client_id": "verify-client", "category": "Logo Design", "simulated_day": day,
                         "skills_vocab": "Adobe Illustrator|Logo Design", "price_tier": req["price_tier"], "location_requirement": None}])
    cbf = ContentBasedRecommender(service.data.workers, row, models["cbf"].weights, models["cbf"].price_neutral_score)
    kg = KnowledgeGraphRecommender(models["kg"].index, row, models["kg"].damping, models["kg"].weights, models["kg"].role_weights,
                                   models["kg"].reputation_prior_strength, 3, KGComponentCache((models["kg"].damping,)))
    zero_cf = lambda link, candidates: np.zeros(len(candidates))  # no imported history: no CF evidence
    independent = HybridRecommender({"cbf": cbf_score_fn(cbf), "cf": zero_cf, "kg": kg_score_fn(kg)}, {"cbf": 0.4, "cf": 0.1, "kg": 0.5})
    candidates = service.eligibility.eligible_worker_ids("Logo Design", day, None)
    expected = independent.explain("verify", candidates, 10)
    check("runtime ranking, scores and contributions equal an independent frozen-component recomputation",
          [x["worker"]["worker_id"] for x in result["recommendations"]] == [e.worker_id for e in expected]
          and all(abs(x["score"] - e.score) < 1e-12 for x, e in zip(result["recommendations"], expected)))
    check("no fabricated history for a new client (CF = 0, no history KG paths)",
          all(x["contributions"]["cf"] == 0 for x in result["recommendations"])
          and not any(ev["path_type"] in ("repeat_hire", "hired_similar_skill", "co_hire")
                      for x in result["recommendations"] for ev in x["kg_evidence"])
          and not any(result["history"].values()))

    # ---- bookings ----------------------------------------------------------------------
    top = result["recommendations"][0]["worker"]["worker_id"]
    outsider = next(w for w in service.data.workers["worker_id"] if w not in set(candidates))
    check("ineligible worker cannot be booked",
          post(client, f"/client/requests/{request_id}/book", {"worker_id": outsider}, "/dashboard").status_code == 400)
    recommended_ids = [x["worker"]["worker_id"] for x in result["recommendations"]]
    pool_only = next(w for w in candidates if w not in recommended_ids)
    check("eligible-pool worker outside the current recommendation set cannot be booked",
          post(client, f"/client/requests/{request_id}/book", {"worker_id": pool_only}, "/dashboard").status_code == 400
          and repo.list_bookings(request_id=request_id) == [])
    post(client, f"/client/requests/{request_id}/book", {"worker_id": top}, "/dashboard")
    post(client, f"/client/requests/{request_id}/book", {"worker_id": top}, "/dashboard")
    bookings_list = repo.list_bookings(request_id=request_id)
    check("booking created as pending; duplicate active booking refused",
          len(bookings_list) == 1 and bookings_list[0]["status"] == "pending")
    booking_id = bookings_list[0]["booking_id"]
    repo.create_user("verify.worker@example.com", hash_password(PASSWORD), "worker", "Linked", worker_id=top)
    worker = app.test_client()
    post(worker, "/login", {"email": "verify.worker@example.com", "password": PASSWORD})
    post(worker, f"/bookings/{booking_id}/complete", token_path="/dashboard")
    post(client, f"/bookings/{booking_id}/review", {"rating": "5"}, "/dashboard")
    check("worker cannot complete; pending booking cannot be reviewed",
          repo.get_booking(booking_id)["status"] == "pending" and repo.get_review(booking_id) is None)
    post(worker, f"/bookings/{booking_id}/accept", token_path="/dashboard")
    check("worker accepts their booking", repo.get_booking(booking_id)["status"] == "accepted")
    post(client, f"/bookings/{booking_id}/complete", token_path="/dashboard")
    done = repo.get_booking(booking_id)
    check("client completes the booking", done["status"] == "completed" and done["completed_at"] is not None)
    post(client, f"/bookings/{booking_id}/review", {"rating": "9"}, "/dashboard")
    post(client, f"/bookings/{booking_id}/review", {"rating": "4", "review_text": "Verified review."}, "/dashboard")
    post(client, f"/bookings/{booking_id}/review", {"rating": "1"}, "/dashboard")
    review = repo.get_review(booking_id)
    check("one valid review stored; invalid and duplicate reviews refused",
          review is not None and review["rating"] == 4 and review["review_text"] == "Verified review.")
    check("review shown on the worker's profile", "Verified review." in worker.get("/profile").get_data(as_text=True))

    # ---- direct database constraints ------------------------------------------------------
    def rejected(statement) -> bool:
        try:
            with engine.begin() as conn:
                conn.execute(statement)
        except IntegrityError:
            return True
        return False

    check("database rejects invalid rating, mixed-case email and unknown role",
          rejected(insert(app_reviews).values(booking_id=booking_id, rating=0))
          and rejected(insert(users).values(email="Upper@x.com", password_hash="h", role="client", display_name="x"))
          and rejected(insert(users).values(email="r@x.com", password_hash="h", role="admin", display_name="x")))

    # ---- Day 10 browsing on the main application ----------------------------------------
    production = create_app("production").test_client()
    home = production.get("/demo").get_data(as_text=True)  # Research Demo
    check("Day 10 browsing still works (page, listing, recommendations)",
          'id="request-list"' in home and production.get("/api/requests?per_page=1").get_json()["total"] == 3600
          and production.get("/api/recommendations", query_string={
              "link": production.get("/api/requests?per_page=1").get_json()["items"][0]["link"]}).status_code == 200)

    clear_runtime(engine)

    # ---- untouched stores + Day 11 chain -------------------------------------------------
    with main_engine.connect() as conn:
        check("main database imported data unchanged by verification", fingerprint(read_back(conn)) == main_fp_before)
        check("main database runtime tables untouched by verification",
              {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in RUNTIME_TABLES} == main_runtime_before)
    if neo4j_before is not None:
        check("Neo4j counts unchanged", neo4j_counts() == neo4j_before)
    else:
        print("[SKIP] Neo4j count check -- Neo4j not running")
    run = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "verify_day11.py")], cwd=REPO_ROOT, capture_output=True, text=True)
    check("scripts/verify_day11.py passes (incl. verify_postgres, verify_day10, verify_day9)", run.returncode == 0, run.stdout[-600:])

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
