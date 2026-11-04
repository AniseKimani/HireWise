"""Independent verification of pre-deployment Milestone 1 (email verification + password reset).

Writes ONLY to POSTGRES_TEST_DB. The main database (POSTGRES_DB) is only read, and is checked to be unchanged.
No real email is sent: the file mail backend writes to a temporary directory outside the repository.

Checks:
  * schema v2 -> v3 migration on a realistic v2 database (imported data + two legacy accounts): column added,
    tables/constraints/indexes created, legacy accounts verified, records preserved, repeat run is a no-op;
    then (Milestone 2) the v3 -> v4 step to the schema the application now requires: user_identities created,
    every v3 structure still verified, no account touched, repeat run a no-op;
  * registration through the app with the PostgreSQL repository: unverified account, emailed absolute link,
    only the SHA-256 hash in auth_tokens (direct SQL), no auto-login, protected actions refused until verified;
  * verification: GET does not consume, POST does; reuse refused; verified_at set (direct SQL);
  * duplicate registration and forgot-password responses identical for known and unknown addresses;
  * password reset: scrypt hash replaced, auth_version incremented (direct SQL), older sessions rejected, old
    password refused, new password accepted, token single-use;
  * the main database's runtime tables and schema version are unchanged.

Usage:
    python scripts/verify_m1_auth.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from email import message_from_bytes
from email.policy import default as default_policy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import inspect, text

from src.app import create_app
from src.app.marketplace import auth_tokens
from src.app.marketplace.mailer import build_mailer
from src.app.marketplace.runtime import RuntimeRecommender
from src.app.services import RecommendationService
from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine
from src.db.loader import load
from src.db.marketplace import PostgresMarketplaceRepository
from src.db.migrations import migrate_to_v3, migrate_to_v4, verify_v3, verify_v4
from src.db.repositories import PostgresRepository
from src.db.schema import metadata

PASSWORD = "a long verifier password"
NEW_PASSWORD = "another long verifier password"
TOKEN = re.compile(r'name="csrf_token" value="([^"]+)"')
V2_RUNTIME_TABLES = ("users", "runtime_worker_profiles", "runtime_worker_skills", "runtime_requests", "runtime_request_skills",
                     "bookings", "app_reviews")
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def main_database_snapshot() -> dict:
    engine = make_engine()
    with engine.connect() as conn:
        names = set(inspect(conn).get_table_names())
        snapshot = {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in V2_RUNTIME_TABLES if t in names}
        snapshot["schema_version"] = conn.execute(text("SELECT value FROM load_metadata WHERE key = 'schema_version'")).scalar()
        snapshot["tables"] = sorted(names)
    return snapshot


def main() -> int:
    test_db, main_db = os.environ.get("POSTGRES_TEST_DB"), os.environ.get("POSTGRES_DB")
    if not test_db or test_db == main_db:
        check("POSTGRES_TEST_DB configured and distinct from POSTGRES_DB", False)
        return 1
    main_before = main_database_snapshot()
    engine = make_engine(load_postgres_settings(TEST_DATABASE_ENV))

    # ---- a realistic schema-v2 test database ------------------------------------------------
    metadata.drop_all(engine)
    metadata.create_all(engine)
    load(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE user_identities"))  # schema v4 (Milestone 2) addition
        conn.execute(text("DROP TABLE auth_tokens"))
        conn.execute(text("DROP TABLE user_email_verification"))
        conn.execute(text("ALTER TABLE users DROP COLUMN auth_version"))
        conn.execute(text("UPDATE load_metadata SET value = '2' WHERE key = 'schema_version'"))
        conn.execute(text("INSERT INTO users (email, password_hash, role, display_name) VALUES "
                          "('legacy.client@example.com', 'legacy-hash-1', 'client', 'Legacy C'), "
                          "('legacy.worker@example.com', 'legacy-hash-2', 'worker', 'Legacy W')"))
    report = migrate_to_v3(engine)
    check("migration v2 -> v3 adds auth_version, creates both tables, grandfathers 2 accounts",
          (report.version_before, report.added_auth_version, sorted(report.created_tables), report.grandfathered_users)
          == ("2", True, ["auth_tokens", "user_email_verification"], 2), str(report.lines()))
    check("verify_v3: every structural check passes", all(verify_v3(engine).values()), str(verify_v3(engine)))
    with engine.connect() as conn:
        legacy = conn.execute(text("SELECT u.password_hash, u.auth_version, v.verified_at IS NOT NULL FROM users u "
                                   "JOIN user_email_verification v USING (user_id) ORDER BY u.email")).all()
    check("legacy accounts preserved (hashes), auth_version 0, verified",
          [tuple(r) for r in legacy] == [("legacy-hash-1", 0, True), ("legacy-hash-2", 0, True)], str(legacy))
    again = migrate_to_v3(engine)
    check("repeat migration changes nothing", (again.added_auth_version, again.created_tables, again.grandfathered_users) == (False, [], 0))

    # ---- schema v4 (Milestone 2): the version the application now requires ------------------------
    v4 = migrate_to_v4(engine)
    check("migration v3 -> v4 creates only user_identities and verifies no one",
          (v4.version_before, v4.version_after, v4.steps_applied, v4.created_tables, v4.grandfathered_users)
          == ("3", "4", ["3->4"], ["user_identities"], 0), str(v4.lines()))
    checks_v4 = verify_v4(engine)
    check("verify_v4: every structural check passes (all v3 checks included)", all(checks_v4.values()), str(checks_v4))
    with engine.connect() as conn:
        after_v4 = conn.execute(text("SELECT u.password_hash, u.auth_version, v.verified_at IS NOT NULL FROM users u "
                                     "JOIN user_email_verification v USING (user_id) ORDER BY u.email")).all()
    check("legacy accounts unchanged by the v4 step", [tuple(r) for r in after_v4] == [tuple(r) for r in legacy])
    again_v4 = migrate_to_v4(engine)
    check("repeat v4 migration changes nothing", (again_v4.steps_applied, again_v4.created_tables) == ([], []))

    # ---- the application on the migrated database --------------------------------------------
    service = RecommendationService.build(repository=PostgresRepository(engine))
    repo = PostgresMarketplaceRepository(engine)
    app = create_app("testing", service=service, marketplace_repository=repo, runtime_recommender=RuntimeRecommender.build(service))
    outbox = Path(tempfile.mkdtemp(prefix="hirewise-verify-outbox-"))
    app.extensions["hirewise_mailer"] = build_mailer({"MAIL_BACKEND": "file", "MAIL_OUTBOX_DIR": str(outbox),
                                                      "APP_BASE_URL": "http://localhost", "MAIL_SEND_ASYNC": False},
                                                     environment="development", required=True)
    check("file outbox is outside the repository", REPO_ROOT not in outbox.resolve().parents)

    def token(c, path="/login"):
        match = TOKEN.search(c.get(path).get_data(as_text=True))
        return (match or TOKEN.search(c.get("/login").get_data(as_text=True))).group(1)

    def post(c, path, data=None, token_path="/login"):
        return c.post(path, data={**(data or {}), "csrf_token": token(c, token_path)})

    def mails():
        return [message_from_bytes(p.read_bytes(), policy=default_policy) for p in sorted(outbox.glob("*.eml"), key=os.path.getmtime)]

    def links(email, kind):
        return [m.group(1) for msg in mails() if msg["To"] == email
                for m in re.finditer(r"https?://[^/\s]+(/(?:verify-email/confirm|reset-password)/[A-Za-z0-9_-]+)", msg.get_content())
                if kind in m.group(1)]

    reg = {"email": "Verify.M1@Example.com", "role": "client", "display_name": "M1", "password": PASSWORD, "confirm_password": PASSWORD}
    c = app.test_client()
    first = post(c, "/register", reg, "/register")
    check("registration redirects to the check-email page without logging in",
          first.headers.get("Location") == "/register/check-email" and c.get("/dashboard").status_code == 302)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT u.user_id, v.verified_at FROM users u JOIN user_email_verification v USING (user_id) "
                                "WHERE u.email = 'verify.m1@example.com'")).first()
        hashes = [r[0] for r in conn.execute(text("SELECT token_hash FROM auth_tokens WHERE user_id = :u"), {"u": row.user_id})]
    check("new account is unverified (direct SQL)", row is not None and row.verified_at is None)
    [link] = links("verify.m1@example.com", "verify-email")
    raw = link.rsplit("/", 1)[1]
    check("email holds an absolute link built from APP_BASE_URL", any(f"http://localhost{link}" in m.get_content() for m in mails()))
    check("auth_tokens stores only the SHA-256 hash", hashes == [auth_tokens.hash_token(raw)] and raw not in hashes)
    check("a genuinely new registration produced exactly one email (its verification link)",
          [m["To"] for m in mails()] == ["verify.m1@example.com"])
    with engine.connect() as conn:
        account_before = conn.execute(text("SELECT u.*, v.verified_at FROM users u JOIN user_email_verification v USING (user_id) "
                                           "WHERE u.user_id = :u"), {"u": row.user_id}).first()
        tokens_before = conn.execute(text("SELECT * FROM auth_tokens ORDER BY token_id")).all()
    outbox_before = sorted(p.name for p in outbox.glob("*.eml"))
    dup = app.test_client()
    second = post(dup, "/register", {**reg, "email": "VERIFY.M1@example.com", "display_name": "x"}, "/register")
    check("duplicate registration response identical, no second account",
          (first.status_code, first.headers["Location"]) == (second.status_code, second.headers["Location"])
          and c.get(first.headers["Location"]).get_data() == dup.get(second.headers["Location"]).get_data()
          and repo.get_credentials("verify.m1@example.com")[0] == row.user_id)
    with engine.connect() as conn:
        account_after = conn.execute(text("SELECT u.*, v.verified_at FROM users u JOIN user_email_verification v USING (user_id) "
                                          "WHERE u.user_id = :u"), {"u": row.user_id}).first()
        tokens_after = conn.execute(text("SELECT * FROM auth_tokens ORDER BY token_id")).all()
    check("duplicate registration sends no email (outbox unchanged)", sorted(p.name for p in outbox.glob("*.eml")) == outbox_before)
    check("duplicate registration leaves the existing account and its tokens unchanged (direct SQL)",
          account_after == account_before and tokens_after == tokens_before)
    post(c, "/login", {"email": "verify.m1@example.com", "password": PASSWORD})
    check("unverified user may log in and browse", c.get("/dashboard").status_code == 200)
    blocked = post(c, "/client/requests/new", {"title": "Logo work", "category": "Logo Design", "skills": "Logo Design",
                                               "pay_type": "unspecified"}, "/dashboard")
    check("unverified user cannot create a request", blocked.headers.get("Location") == "/verify-email"
          and repo.list_requests_for_client(row.user_id) == [])
    c.get(link)
    c.get("/verify-email/confirm")
    check("GET of the link does not consume the token", repo.auth_token_state(auth_tokens.hash_token(raw), "verify_email",
                                                                               auth_tokens.utcnow()) == "valid")
    post(c, "/verify-email/confirm", token_path="/verify-email/confirm")
    with engine.connect() as conn:
        verified = conn.execute(text("SELECT verified_at IS NOT NULL FROM user_email_verification WHERE user_id = :u"),
                                {"u": row.user_id}).scalar()
    check("POST confirmation verifies the account (direct SQL)", bool(verified))
    c.get(link)
    check("a used verification link is refused", "already been used" in c.get("/verify-email/confirm").get_data(as_text=True))
    created = post(c, "/client/requests/new", {"title": "Logo work", "category": "Logo Design", "skills": "Logo Design",
                                               "pay_type": "unspecified"}, "/client/requests/new")
    check("verified user can create a request", created.status_code == 302 and "/client/requests/" in created.headers["Location"])

    known, unknown = app.test_client(), app.test_client()
    r1 = post(known, "/forgot-password", {"email": "verify.m1@example.com"}, "/forgot-password")
    r2 = post(unknown, "/forgot-password", {"email": "nobody.m1@example.com"}, "/forgot-password")
    check("forgot-password responses identical for known and unknown addresses",
          (r1.status_code, r1.headers["Location"]) == (r2.status_code, r2.headers["Location"])
          and known.get(r1.headers["Location"]).get_data() == unknown.get(r2.headers["Location"]).get_data()
          and not links("nobody.m1@example.com", "reset-password"))
    [reset_link] = links("verify.m1@example.com", "reset-password")
    with engine.connect() as conn:
        old_hash = conn.execute(text("SELECT password_hash FROM users WHERE user_id = :u"), {"u": row.user_id}).scalar()
    b = app.test_client()
    b.get(reset_link)
    done = post(b, "/reset-password", {"password": NEW_PASSWORD, "confirm_password": NEW_PASSWORD}, "/reset-password")
    with engine.connect() as conn:
        new_hash, version = conn.execute(text("SELECT password_hash, auth_version FROM users WHERE user_id = :u"), {"u": row.user_id}).first()
    check("reset stores a new scrypt hash and increments auth_version (direct SQL)",
          done.headers.get("Location") == "/login" and new_hash != old_hash and new_hash.startswith("scrypt:") and version == 1)
    check("the older session is rejected after the reset", c.get("/dashboard").status_code == 302)
    check("old password refused, new password accepted",
          post(app.test_client(), "/login", {"email": "verify.m1@example.com", "password": PASSWORD}).status_code == 401
          and post(app.test_client(), "/login", {"email": "verify.m1@example.com", "password": NEW_PASSWORD}).headers.get("Location") == "/dashboard")
    again_client = app.test_client()
    again_client.get(reset_link)
    check("a used reset link is refused", "already been used" in again_client.get("/reset-password").get_data(as_text=True))

    # ---- clean up the test database's runtime rows; confirm the main database is unchanged --------------
    with engine.begin() as conn:
        for name in ("app_reviews", "bookings", "runtime_request_skills", "runtime_requests", "runtime_worker_skills",
                     "runtime_worker_profiles", "user_identities", "auth_tokens", "user_email_verification", "users"):
            conn.execute(text(f"DELETE FROM {name}"))
    check("main database unchanged (runtime row counts, schema version, tables)", main_database_snapshot() == main_before)
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} CHECK(S) FAILED'}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
