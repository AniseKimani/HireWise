"""Independent verification of pre-deployment Milestone 2 (Continue with Google), against PostgreSQL.

Writes ONLY to POSTGRES_TEST_DB. The main database (POSTGRES_DB) is only read, and is checked to be unchanged.
Google is never contacted and no credential is needed: the real Authlib client runs (state, nonce, PKCE, ID-token
signature and claim validation) with only its network calls replaced -- discovery metadata, the key set (an RSA
key generated here) and the token exchange (which returns an ID token signed with that key).

Checks:
  * schema: a realistic version 3 database (imported data + two existing accounts) upgraded 3 -> 4, verified;
    user_identities constraints enforced by PostgreSQL (direct SQL);
  * start: redirect to Google's authorization endpoint with code flow, openid/email/profile, state, nonce, PKCE S256
    and the redirect URI built from APP_BASE_URL;
  * new Google user: nothing stored before completion; completion creates the user, a verified email, the worker
    profile and the identity link (direct SQL); unusable password; logged in;
  * linked identity: signs straight in, last_login_at updated (direct SQL);
  * existing verified account: linked automatically and logged in, account data unchanged (direct SQL);
  * existing unverified account: no link from Google alone; after the password login the identity is linked and
    the email verified (direct SQL); an account linked to another Google identity is refused;
  * refusals: unverified Google email, wrong audience, wrong nonce, forged signature, state mismatch, cancellation --
    no rows written, no session;
  * atomic creation: a duplicate identity leaves no orphan account;
  * logs contain no authorization code, token, client secret, subject or email address;
  * the main database's runtime tables and schema version are unchanged.

Usage:
    python scripts/verify_m2_google.py
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import sys
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from joserfc import jwt
from joserfc.jwk import RSAKey
from sqlalchemy import inspect, insert, text
from sqlalchemy.exc import IntegrityError

from src.app import create_app
from src.app.config import TestingConfig
from src.app.factory import RedactTokenPaths
from src.app.marketplace import auth_tokens
from src.app.marketplace.runtime import RuntimeRecommender
from src.app.marketplace.security import hash_password, verify_password
from src.app.services import RecommendationService
from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine
from src.db.loader import load
from src.db.marketplace import IdentityAlreadyLinked, PostgresMarketplaceRepository
from src.db.migrations import migrate_to_v4, verify_v4
from src.db.repositories import PostgresRepository
from src.db.schema import metadata, user_identities

PASSWORD = "a long verifier password"
CLIENT_ID = "verifier-client.apps.googleusercontent.com"
CLIENT_SECRET = "verifier-client-secret"
AUTH_CODE = "verifier-authorization-code"
ACCESS_TOKEN = "verifier-access-token"
METADATA = {"issuer": "https://accounts.google.com",
            "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
            "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
            "id_token_signing_alg_values_supported": ["RS256"]}
KEY = RSAKey.generate_key(2048, parameters={"kid": "verifier"})
FORGED_KEY = RSAKey.generate_key(2048, parameters={"kid": "verifier"})
TOKEN = re.compile(r'name="csrf_token" value="([^"]+)"')
RUNTIME_COUNT_TABLES = ("users", "user_email_verification", "auth_tokens", "runtime_worker_profiles", "runtime_worker_skills",
                        "runtime_requests", "runtime_request_skills", "bookings", "app_reviews")
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def main_database_snapshot() -> dict:
    engine = make_engine()
    with engine.connect() as conn:
        names = set(inspect(conn).get_table_names())
        snapshot = {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in RUNTIME_COUNT_TABLES if t in names}
        snapshot["schema_version"] = conn.execute(text("SELECT value FROM load_metadata WHERE key = 'schema_version'")).scalar()
        snapshot["tables"] = sorted(names)
    return snapshot


class FakeGoogle:
    """The token endpoint: returns an ID token for the nonce Authlib sent to the authorization endpoint."""

    def __init__(self):
        self.nonce, self.claims, self.key, self.exchanges, self.override_nonce = None, {}, KEY, [], None

    def id_token(self):
        now = int(time.time())
        claims = {"iss": METADATA["issuer"], "aud": CLIENT_ID, "azp": CLIENT_ID, "sub": "verifier-sub-1",
                  "email": "google.person@example.com", "email_verified": True, "name": "Google Person", "iat": now,
                  "exp": now + 3600, "nonce": self.override_nonce or self.nonce, **self.claims}
        return jwt.encode({"alg": "RS256", "kid": "verifier"}, claims, self.key)

    def exchange(self, **params):
        self.exchanges.append(params)
        return {"access_token": ACCESS_TOKEN, "token_type": "Bearer", "expires_in": 3600, "id_token": self.id_token()}


def main() -> int:
    test_db, main_db = os.environ.get("POSTGRES_TEST_DB"), os.environ.get("POSTGRES_DB")
    if not test_db or test_db == main_db:
        check("POSTGRES_TEST_DB configured and distinct from POSTGRES_DB", False)
        return 1
    main_before = main_database_snapshot()
    engine = make_engine(load_postgres_settings(TEST_DATABASE_ENV))

    # ---- a realistic schema-v3 test database, upgraded to v4 --------------------------------------
    metadata.drop_all(engine)
    metadata.create_all(engine)
    load(engine)
    repo = PostgresMarketplaceRepository(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE user_identities"))
        conn.execute(text("UPDATE load_metadata SET value = '3' WHERE key = 'schema_version'"))
    existing_unverified = repo.create_user("google.person@example.com", hash_password(PASSWORD), "client", "Existing",
                                           verified=False)
    other = repo.create_user("other.person@example.com", hash_password(PASSWORD), "client", "Other")
    report = migrate_to_v4(engine)
    check("migration v3 -> v4 creates user_identities only",
          (report.version_before, report.steps_applied, report.created_tables, report.grandfathered_users)
          == ("3", ["3->4"], ["user_identities"], 0), str(report.lines()))
    checks = verify_v4(engine)
    check("verify_v4: every structural check passes", all(checks.values()), str(checks))
    check("the upgrade verifies no one (existing unverified account still unverified)",
          repo.get_user(existing_unverified)["email_verified_at"] is None)
    bad_rows = [dict(user_id=other, provider="github", subject="s", email_at_link="other.person@example.com"),
                dict(user_id=other, provider="google", subject="", email_at_link="other.person@example.com"),
                dict(user_id=other, provider="google", subject="s", email_at_link="Other.Person@Example.com"),
                dict(user_id=987654321, provider="google", subject="s", email_at_link="x@example.com")]
    refused = 0
    for row in bad_rows:
        try:
            with engine.begin() as conn:
                conn.execute(insert(user_identities).values(**row))
        except IntegrityError:
            refused += 1
    check("PostgreSQL enforces provider, subject, lower-case email and user FK constraints", refused == len(bad_rows))

    # ---- the application with Google configured, network calls simulated -------------------------
    class Config(TestingConfig):
        GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET = CLIENT_ID, CLIENT_SECRET

    service = RecommendationService.build(repository=PostgresRepository(engine))
    with mock.patch.dict("src.app.config.CONFIGS", {"testing": Config}):
        app = create_app("testing", service=service, marketplace_repository=repo,
                         runtime_recommender=RuntimeRecommender.build(service))
    google, fake = app.extensions["hirewise_google"], FakeGoogle()
    for name, kwargs in (("load_server_metadata", {"return_value": METADATA}),
                         ("fetch_jwk_set", {"return_value": {"keys": [KEY.as_dict(private=False)]}}),
                         ("fetch_access_token", {"side_effect": fake.exchange})):
        mock.patch.object(google, name, **kwargs).start()

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.addFilter(RedactTokenPaths())
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    def csrf(c, path="/login"):
        match = TOKEN.search(c.get(path).get_data(as_text=True))
        return match.group(1) if match else TOKEN.search(c.get("/login").get_data(as_text=True)).group(1)

    def start(c, query=""):
        r = c.get("/auth/google/start" + query)
        params = {k: v[0] for k, v in parse_qs(urlsplit(r.headers["Location"]).query).items()}
        fake.nonce = params.get("nonce")
        return r, params

    def sign_in(c, query="", **claims):
        fake.claims = claims
        _, params = start(c, query)
        return c.get(f"/auth/google/callback?state={params['state']}&code={AUTH_CODE}")

    def session_user(c):
        with c.session_transaction() as s:
            return s.get("user_id")

    def count(sql, **params):
        with engine.connect() as conn:
            return conn.execute(text(sql), params).scalar()

    users_before = count("SELECT count(*) FROM users")

    # start
    c = app.test_client()
    r, params = start(c, "?role=worker")
    with c.session_transaction() as s:
        verifier = next(v["data"]["code_verifier"] for k, v in s.items() if k.startswith("_state_google_"))
    challenge = urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    check("start redirects to Google with code flow, scopes, state, nonce, PKCE S256 and the APP_BASE_URL callback",
          r.headers["Location"].startswith(METADATA["authorization_endpoint"]) and params.get("response_type") == "code"
          and params.get("scope") == "openid email profile" and len(params.get("state", "")) >= 20
          and len(params.get("nonce", "")) >= 20 and params.get("code_challenge_method") == "S256"
          and params.get("code_challenge") == challenge
          and params.get("redirect_uri") == "http://localhost/auth/google/callback" and "client_secret" not in params)

    # new Google user (a different address from the existing accounts)
    fake.claims = {"sub": "verifier-sub-new", "email": "brand.new@example.com"}
    r = c.get(f"/auth/google/callback?state={params['state']}&code={AUTH_CODE}")
    check("the code and PKCE verifier are sent to the token endpoint",
          fake.exchanges[-1].get("code") == AUTH_CODE and fake.exchanges[-1].get("code_verifier") == verifier)
    check("a new Google user is sent to completion and nothing is stored yet (direct SQL)",
          r.headers.get("Location") == "/auth/google/complete" and count("SELECT count(*) FROM users") == users_before
          and count("SELECT count(*) FROM user_identities") == 0 and session_user(c) is None)
    page = c.get("/auth/google/complete").get_data(as_text=True)
    check("completion page shows the fixed Google email and the preselected role",
          'value="brand.new@example.com"' in page and re.search(r'id="role-worker"[^>]*checked', page) is not None)
    r = c.post("/auth/google/complete", data={"role": "worker", "display_name": "Brand New", "worker_country": "India",
                                              "primary_category": "Logo Design", "experience_level": "entry",
                                              "experience_years": "2", "rate": "20", "skills": "Logo Design",
                                              "csrf_token": csrf(c, "/auth/google/complete")})
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT u.user_id, u.role, u.password_hash, v.verified_at, i.provider, i.subject, i.email_at_link, "
            "i.last_login_at, p.primary_category FROM users u JOIN user_email_verification v USING (user_id) "
            "JOIN user_identities i USING (user_id) JOIN runtime_worker_profiles p USING (user_id) "
            "WHERE u.email = 'brand.new@example.com'")).first()
    check("completion creates user + verified email + worker profile + identity in one go (direct SQL)",
          row is not None and row.role == "worker" and row.verified_at is not None and row.provider == "google"
          and row.subject == "verifier-sub-new" and row.email_at_link == "brand.new@example.com"
          and row.primary_category == "Logo Design")
    check("the new account's password is an unusable random scrypt hash",
          row is not None and row.password_hash.startswith("scrypt:") and not verify_password(row.password_hash, "")
          and not verify_password(row.password_hash, PASSWORD))
    check("the new user is logged in", r.headers.get("Location") == "/dashboard" and session_user(c) == row.user_id
          and c.get("/dashboard").status_code == 200)

    # linked identity signs straight in
    first_login = row.last_login_at
    c2 = app.test_client()
    r = sign_in(c2, "?next=/requests", sub="verifier-sub-new", email="brand.new@example.com")
    last = count("SELECT last_login_at FROM user_identities WHERE subject = 'verifier-sub-new'")
    check("a linked identity signs straight in and last_login_at is updated (direct SQL)",
          r.headers.get("Location") == "/requests" and session_user(c2) == row.user_id and last > first_login)

    # existing UNVERIFIED account: link only after the password login
    c3 = app.test_client()
    r = sign_in(c3)
    check("an existing unverified account is never opened by Google alone",
          r.headers.get("Location") == "/login" and session_user(c3) is None
          and count("SELECT count(*) FROM user_identities WHERE user_id = :u", u=existing_unverified) == 0)
    r = c3.post("/login", data={"email": "google.person@example.com", "password": PASSWORD, "csrf_token": csrf(c3)})
    check("after the password login the identity is linked and the email verified (direct SQL)",
          session_user(c3) == existing_unverified
          and count("SELECT count(*) FROM user_identities WHERE user_id = :u AND subject = 'verifier-sub-1'",
                    u=existing_unverified) == 1
          and count("SELECT verified_at IS NOT NULL FROM user_email_verification WHERE user_id = :u", u=existing_unverified))

    # an account linked to another Google identity is refused
    identities = count("SELECT count(*) FROM user_identities")
    c4 = app.test_client()
    r = sign_in(c4, sub="verifier-sub-impostor", email="google.person@example.com")
    check("an account already linked to a different Google identity is refused",
          r.headers.get("Location") == "/login" and session_user(c4) is None
          and count("SELECT count(*) FROM user_identities") == identities)

    # existing VERIFIED account: linked automatically, no password prompt, account unchanged
    with engine.connect() as conn:
        before = conn.execute(text("SELECT password_hash, role, display_name, auth_version FROM users WHERE user_id = :u"),
                              {"u": other}).first()
    c5 = app.test_client()
    r = sign_in(c5, sub="verifier-sub-other", email="other.person@example.com")
    with engine.connect() as conn:
        after = conn.execute(text("SELECT password_hash, role, display_name, auth_version FROM users WHERE user_id = :u"),
                             {"u": other}).first()
    check("an existing verified account is linked automatically and logged in, account unchanged (direct SQL)",
          r.headers.get("Location") == "/dashboard" and session_user(c5) == other and tuple(after) == tuple(before)
          and count("SELECT count(*) FROM user_identities WHERE user_id = :u AND subject = 'verifier-sub-other'", u=other) == 1)
    identities = count("SELECT count(*) FROM user_identities")

    # refusals write nothing
    users_now = count("SELECT count(*) FROM users")
    outcomes = []
    for label, claims, nonce, key in (("unverified email", {"sub": "s-a", "email": "a@example.com", "email_verified": False}, None, KEY),
                                      ("wrong audience", {"sub": "s-b", "email": "b@example.com", "aud": "someone-else", "azp": "someone-else"}, None, KEY),
                                      ("wrong nonce", {"sub": "s-c", "email": "c@example.com"}, "not-the-nonce", KEY),
                                      ("forged signature", {"sub": "s-d", "email": "d@example.com"}, None, FORGED_KEY)):
        fake.override_nonce, fake.key = nonce, key
        cx = app.test_client()
        r = sign_in(cx, **claims)
        outcomes.append((label, r.headers.get("Location") == "/login" and session_user(cx) is None))
    fake.override_nonce, fake.key = None, KEY
    check("unverified email, wrong audience, wrong nonce and forged signature are refused",
          all(ok for _, ok in outcomes), str(outcomes))
    exchanges = len(fake.exchanges)
    cx = app.test_client()
    start(cx)
    r_mismatch = cx.get(f"/auth/google/callback?state=forged&code={AUTH_CODE}")
    _, p2 = start(cx)
    r_cancel = cx.get(f"/auth/google/callback?error=access_denied&state={p2['state']}")
    check("state mismatch and cancellation are refused without a token exchange",
          r_mismatch.headers.get("Location") == "/login" and r_cancel.headers.get("Location") == "/login"
          and len(fake.exchanges) == exchanges)
    check("refused sign-ins wrote no rows (direct SQL)",
          count("SELECT count(*) FROM users") == users_now and count("SELECT count(*) FROM user_identities") == identities)

    # atomic creation
    try:
        repo.create_user_with_identity("orphan@example.com", "h", "client", "Orphan", None, None, "google",
                                       "verifier-sub-new", auth_tokens.utcnow())
        atomic = False
    except IdentityAlreadyLinked:
        atomic = count("SELECT count(*) FROM users WHERE email = 'orphan@example.com'") == 0
    check("a duplicate identity rolls the whole account back (no orphan user)", atomic)

    log = log_stream.getvalue()
    leaked = [s for s in (AUTH_CODE, ACCESS_TOKEN, CLIENT_SECRET, "verifier-sub-new", "brand.new@example.com",
                          "google.person@example.com", verifier) if s in log]
    check("logs contain no code, token, secret, verifier, subject or email", not leaked and "google sign-up completed" in log,
          str(leaked))
    logging.getLogger().removeHandler(handler)
    mock.patch.stopall()

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
