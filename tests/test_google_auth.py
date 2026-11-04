"""Pre-deployment Milestone 2: Continue with Google (OAuth 2.0 authorization code + OpenID Connect, via Authlib).

Google is never contacted. The real Authlib client runs -- it generates and checks state, nonce and the PKCE
verifier and verifies the ID token's signature and claims -- with only its three network calls replaced:
discovery metadata, the key set (a locally generated RSA key) and the token exchange (which returns an ID
token signed with that key). Everything runs against the in-memory repository; the repository behaviour is
also run against PostgreSQL when opted in (POSTGRES_TEST_DB only, never POSTGRES_DB).
"""
import hashlib
import io
import logging
import os
import re
import time
import unittest
from base64 import urlsafe_b64encode
from datetime import timedelta
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from joserfc import jwt
from joserfc.jwk import RSAKey

from src.app import create_app
from src.app.factory import InvalidGoogleConfig, RedactTokenPaths
from src.app.marketplace import auth_tokens, google_auth
from src.app.marketplace.security import verify_password
from src.db.marketplace import (DuplicateEmail, IdentityAlreadyLinked, InMemoryMarketplaceRepository,
                                UserAlreadyHasIdentity)
from tests.test_day12_marketplace import HAVE_DATA, PASSWORD, TOKEN, shared

RUN_PG = os.environ.get("HIREWISE_RUN_POSTGRES_TESTS") == "1"
CLIENT_ID = "test-client-id.apps.googleusercontent.com"
CLIENT_SECRET = "test-client-secret-value"
AUTH_CODE = "test-authorization-code-value"
ACCESS_TOKEN = "test-access-token-value"
ISSUER = "https://accounts.google.com"
METADATA = {
    "issuer": ISSUER,
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
    "id_token_signing_alg_values_supported": ["RS256"],
}
SIGNING_KEY = RSAKey.generate_key(2048, parameters={"kid": "test-key"})
OTHER_KEY = RSAKey.generate_key(2048, parameters={"kid": "test-key"})  # same kid, different key: a forged signature
GOOGLE_CONFIG = {"GOOGLE_CLIENT_ID": CLIENT_ID, "GOOGLE_CLIENT_SECRET": CLIENT_SECRET}
WORKER_DETAILS = {"worker_country": "India", "primary_category": "Logo Design", "experience_level": "entry",
                  "experience_years": "2", "rate": "20", "skills": "Logo Design"}


def id_token(nonce, key=SIGNING_KEY, **overrides):
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": CLIENT_ID, "azp": CLIENT_ID, "sub": "google-sub-0001", "email": "g.user@example.com",
              "email_verified": True, "name": "Gee User", "iat": now, "exp": now + 3600, "nonce": nonce}
    claims.update(overrides)
    for name in [k for k, v in claims.items() if v is None]:
        del claims[name]
    return jwt.encode({"alg": "RS256", "kid": "test-key"}, claims, key)


def flashes(client):
    with client.session_transaction() as s:
        return [message for _, message in s.get("_flashes", [])]


@unittest.skipUnless(HAVE_DATA, "processed data / Day 8 results absent")
class GoogleFlowTests(unittest.TestCase):
    def setUp(self):
        self.service, self.runtime = shared()
        self.repo = InMemoryMarketplaceRepository.from_artifacts()
        self.app = self.build_app()
        self.google = self.app.extensions["hirewise_google"]
        patches = [mock.patch.object(self.google, "load_server_metadata", return_value=dict(METADATA)),
                   mock.patch.object(self.google, "fetch_jwk_set", return_value={"keys": [SIGNING_KEY.as_dict(private=False)]})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.token_response = None  # set per test; None = build a valid one from the session's nonce
        self.exchange = mock.patch.object(self.google, "fetch_access_token", side_effect=self.fake_exchange)
        self.exchange_mock = self.exchange.start()
        self.addCleanup(self.exchange.stop)

    def build_app(self, **extra):
        from src.app.config import TestingConfig

        class GoogleTestingConfig(TestingConfig):
            pass

        for k, v in {**GOOGLE_CONFIG, **extra}.items():
            setattr(GoogleTestingConfig, k, v)
        with mock.patch.dict("src.app.config.CONFIGS", {"testing": GoogleTestingConfig}):
            return create_app("testing", service=self.service, marketplace_repository=self.repo,
                              runtime_recommender=self.runtime)

    # ---- the fake Google side ----
    def fake_exchange(self, **params):
        """Token endpoint: checks what Authlib sends (the code and the PKCE verifier), returns an ID token."""
        self.exchanged = params
        if callable(self.token_response):
            return self.token_response(params)
        return {"access_token": ACCESS_TOKEN, "token_type": "Bearer", "expires_in": 3600,
                "id_token": id_token(self.current_nonce, **self.claims)}

    claims: dict = {}

    def start(self, c, query=""):
        r = c.get("/auth/google/start" + query)
        self.assertEqual(r.status_code, 302, r.get_data(as_text=True))
        url = urlsplit(r.headers["Location"])
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        self.current_nonce = params.get("nonce")
        return url, params

    def sign_in(self, c, query="", **claims):
        """Start the flow and come back from Google with an authorization code. Returns the callback response."""
        self.claims = claims
        _, params = self.start(c, query)
        return c.get(f"/auth/google/callback?state={params['state']}&code={AUTH_CODE}")

    def csrf(self, c, path="/login"):
        match = TOKEN.search(c.get(path).get_data(as_text=True))
        return match.group(1) if match else TOKEN.search(c.get("/login").get_data(as_text=True)).group(1)

    def complete(self, c, role="client", **extra):
        form = {"role": role, "display_name": "Gee User", **(WORKER_DETAILS if role == "worker" else {}), **extra}
        return c.post("/auth/google/complete", data={**form, "csrf_token": self.csrf(c, "/auth/google/complete")})

    def login(self, c, email, password=PASSWORD):
        return c.post("/login", data={"email": email, "password": password, "csrf_token": self.csrf(c)})

    def local_account(self, email, verified=True):
        from src.app.marketplace.security import hash_password

        return self.repo.create_user(email, hash_password(PASSWORD), "client", "Local", verified=verified)

    def logged_in_user(self, c):
        with c.session_transaction() as s:
            return s.get("user_id")

    def user_count(self):
        return len(self.repo._users)

    # ---- configuration ----
    def test_disabled_without_credentials_hides_the_button_and_routes(self):
        app = create_app("testing", service=self.service, marketplace_repository=self.repo, runtime_recommender=self.runtime)
        self.assertNotIn("hirewise_google", app.extensions)
        c = app.test_client()
        for page in ("/login", "/register"):
            html = c.get(page).get_data(as_text=True)
            self.assertNotIn("gsi-button", html)
            self.assertNotIn("Continue with Google", html)
        for path in ("/auth/google/start", "/auth/google/callback", "/auth/google/complete"):
            self.assertEqual(c.get(path).status_code, 404)

    def test_testing_config_ignores_credentials_in_the_environment(self):
        with mock.patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "env-only-id"}):  # one of two: would fail start-up
            app = create_app("testing", service=self.service, marketplace_repository=self.repo,
                             runtime_recommender=self.runtime)
        self.assertNotIn("hirewise_google", app.extensions)

    def test_partial_configuration_stops_start_up_without_revealing_values(self):
        for config in ({"GOOGLE_CLIENT_ID": CLIENT_ID}, {"GOOGLE_CLIENT_SECRET": CLIENT_SECRET}):
            with self.assertRaises(InvalidGoogleConfig) as ctx:
                self.build_app(**{"GOOGLE_CLIENT_ID": "", "GOOGLE_CLIENT_SECRET": "", **config})
            self.assertIn("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET", str(ctx.exception))
            self.assertNotIn(CLIENT_ID, str(ctx.exception))
            self.assertNotIn(CLIENT_SECRET, str(ctx.exception))

    def test_production_style_environment_configuration(self):
        from flask import Flask

        app = Flask("probe")
        app.config.update(GOOGLE_CLIENT_ID=None, GOOGLE_CLIENT_SECRET=None, APP_BASE_URL=None, DEFAULT_APP_BASE_URL=None,
                          GOOGLE_DISCOVERY_URL=METADATA["issuer"], GOOGLE_HTTP_TIMEOUT_SECONDS=10)
        self.assertFalse(google_auth.configure_google(app, environ={}))  # neither set: disabled, app starts
        with self.assertRaises(google_auth.GoogleConfigError):  # exactly one set
            google_auth.configure_google(app, environ={"GOOGLE_CLIENT_SECRET": "x"})
        with self.assertRaises(google_auth.GoogleConfigError):  # production needs an https APP_BASE_URL
            google_auth.configure_google(app, environ={"GOOGLE_CLIENT_ID": "a", "GOOGLE_CLIENT_SECRET": "b",
                                                       "APP_BASE_URL": "http://hirewise.example.com"})
        self.assertTrue(google_auth.configure_google(app, environ={"GOOGLE_CLIENT_ID": "a", "GOOGLE_CLIENT_SECRET": "b",
                                                                   "APP_BASE_URL": "https://hirewise.example.com/"}))
        self.assertEqual(app.config["GOOGLE_REDIRECT_URI"], "https://hirewise.example.com/auth/google/callback")
        self.assertGreaterEqual(logging.getLogger("authlib").level, logging.INFO)  # no DEBUG records (PKCE verifier)

    def test_buttons_are_genuine_links_with_the_official_mark(self):
        c = self.app.test_client()
        login = c.get("/login?next=/requests/new").get_data(as_text=True)
        self.assertIn('href="/auth/google/start?next=/requests/new"', login)
        self.assertIn("Continue with Google", login)
        self.assertIn('class="login-forgot"', login)  # existing hooks preserved
        register = c.get("/register?role=worker").get_data(as_text=True)
        self.assertIn('href="/auth/google/start?role=worker"', register)
        self.assertIn("data-google-start", register)
        for hook in ('id="register-form"', 'data-step="role"', 'data-next="account"', 'data-back="role"', 'data-role="worker"'):
            self.assertIn(hook, register)
        self.assertIn('href="/auth/google/start"', c.get("/register").get_data(as_text=True))
        self.assertIn("/static/img/google-g.svg", login)
        svg = c.get("/static/img/google-g.svg")
        self.assertEqual((svg.status_code, svg.mimetype), (200, "image/svg+xml"))
        self.assertNotIn(b"<script", svg.data)
        self.assertIn("Content-Security-Policy", c.get("/login").headers)
        self.assertNotIn("<script>", login)  # no inline script

    # ---- start ----
    def test_start_redirects_with_state_nonce_and_pkce(self):
        c = self.app.test_client()
        url, params = self.start(c, "?role=worker&next=/requests/new")
        self.assertEqual(f"{url.scheme}://{url.netloc}{url.path}", METADATA["authorization_endpoint"])
        self.assertEqual(params["response_type"], "code")
        self.assertEqual(params["client_id"], CLIENT_ID)
        self.assertEqual(params["redirect_uri"], "http://localhost/auth/google/callback")  # from APP_BASE_URL
        self.assertEqual(params["scope"], "openid email profile")
        self.assertEqual(params["code_challenge_method"], "S256")
        self.assertEqual(params["prompt"], "select_account")
        self.assertGreaterEqual(len(params["state"]), 20)
        self.assertGreaterEqual(len(params["nonce"]), 20)
        self.assertNotIn("client_secret", params)
        with c.session_transaction() as s:
            [key] = [k for k in s if k.startswith("_state_google_")]
            verifier = s[key]["data"]["code_verifier"]
            self.assertEqual(s["google_flow"], {"next": "/requests/new", "role": "worker"})
        challenge = urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        self.assertEqual(params["code_challenge"], challenge)
        self.assertEqual(key, f"_state_google_{params['state']}")

    def test_start_rejects_unsafe_next_and_unknown_roles(self):
        c = self.app.test_client()
        self.start(c, "?next=//evil.example.com/&role=admin")
        with c.session_transaction() as s:
            self.assertEqual(s["google_flow"], {"next": None, "role": None})

    def test_start_when_google_is_unreachable(self):
        c = self.app.test_client()
        with mock.patch.object(self.google, "load_server_metadata", side_effect=ConnectionError("down")):
            r = c.get("/auth/google/start")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/login"))
        self.assertIn("Google sign-in is unavailable right now", " ".join(flashes(c)))

    # ---- new Google account ----
    def test_new_account_is_created_only_on_completion_with_identity_and_verified_email(self):
        c = self.app.test_client()
        r = self.sign_in(c, "?role=worker")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/auth/google/complete"))
        self.assertEqual(self.exchanged["code"], AUTH_CODE)
        self.assertIn("code_verifier", self.exchanged)  # PKCE verifier sent with the code
        self.assertEqual(self.user_count(), 0)  # nothing stored before completion
        self.assertIsNone(self.logged_in_user(c))
        page = c.get("/auth/google/complete").get_data(as_text=True)
        self.assertIn('value="g.user@example.com"', page)
        self.assertIn("readonly", page)
        self.assertRegex(page, r'id="role-worker"[^>]*checked')  # the role chosen on the register page, preselected
        self.assertIn('value="Gee User"', page)  # name from Google, editable
        self.assertIn("role-fields.js", page)
        r = self.complete(c, role="worker")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/dashboard"))
        user = self.repo.get_user(self.logged_in_user(c))
        self.assertEqual((user["email"], user["role"], user["display_name"]), ("g.user@example.com", "worker", "Gee User"))
        self.assertIsNotNone(user["email_verified_at"])
        self.assertEqual(self.repo.get_user_by_identity("google", "google-sub-0001")["user_id"], user["user_id"])
        self.assertEqual(self.repo.get_runtime_worker_profile(user["user_id"])["primary_category"], "Logo Design")
        self.assertEqual(c.get("/dashboard").status_code, 200)
        _, stored_hash = self.repo.get_credentials("g.user@example.com")
        self.assertFalse(verify_password(stored_hash, ""))  # an unusable random password
        self.assertFalse(verify_password(stored_hash, PASSWORD))
        with c.session_transaction() as s:
            self.assertNotIn("google_pending_signup", s)
        self.assertEqual(c.get("/auth/google/complete").status_code, 302)  # logged in: nothing more to complete

    def test_google_only_account_can_set_a_password_through_forgot_password(self):
        c = self.app.test_client()
        self.sign_in(c)
        self.complete(c)
        user_id = self.logged_in_user(c)
        c.post("/logout", data={"csrf_token": self.csrf(c, "/dashboard")})
        mail = self.app.extensions["hirewise_mailer"].backend
        c.post("/forgot-password", data={"email": "g.user@example.com", "csrf_token": self.csrf(c, "/forgot-password")})
        [message] = [m for m in mail.messages if m["To"] == "g.user@example.com"]
        path = re.search(r"https?://[^/\s]+(/reset-password/[A-Za-z0-9_-]+)", message.get_content()).group(1)
        c.get(path)
        new_password = "a password set later"
        r = c.post("/reset-password", data={"password": new_password, "confirm_password": new_password,
                                            "csrf_token": self.csrf(c, "/reset-password")})
        self.assertEqual(r.headers["Location"], "/login")
        self.login(c, "g.user@example.com", new_password)
        self.assertEqual(self.logged_in_user(c), user_id)
        c.post("/logout", data={"csrf_token": self.csrf(c, "/dashboard")})
        self.sign_in(c)  # Google still works too
        self.assertEqual(self.logged_in_user(c), user_id)

    def test_role_can_be_changed_and_completion_reuses_registration_validation(self):
        c = self.app.test_client()
        self.sign_in(c, "?role=worker")
        r = self.complete(c, role="client", worker_country="India")  # the other role's field
        self.assertEqual(r.status_code, 400)
        self.assertIn("Details for the other account type were submitted", r.get_data(as_text=True))
        r = self.complete(c, role="worker", primary_category="Not A Category", display_name="")
        self.assertEqual(r.status_code, 400)
        html = r.get_data(as_text=True)
        self.assertIn("Choose a category from the list.", html)
        self.assertIn("Enter a display name", html)
        self.assertEqual(self.user_count(), 0)
        r = self.complete(c, role="nobody")
        self.assertIn("Choose client or worker.", r.get_data(as_text=True))
        r = self.complete(c, role="client", email="attacker@example.com")  # the email is fixed
        self.assertEqual(r.status_code, 302)
        self.assertIsNone(self.repo.get_credentials("attacker@example.com"))
        self.assertEqual(self.repo.get_user(self.logged_in_user(c))["role"], "client")

    def test_completion_requires_csrf_and_a_fresh_pending_sign_up(self):
        c = self.app.test_client()
        r = c.get("/auth/google/complete")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/register"))
        self.sign_in(c)
        r = c.post("/auth/google/complete", data={"role": "client", "display_name": "X"})
        self.assertEqual(r.status_code, 400)  # CSRF
        token = self.csrf(c, "/auth/google/complete")
        with mock.patch.object(auth_tokens, "utcnow",
                               return_value=auth_tokens.utcnow() + timedelta(minutes=16)):  # 15-minute window
            r = c.post("/auth/google/complete", data={"role": "client", "display_name": "X", "csrf_token": token})
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/register"))
        self.assertIn("expired", " ".join(flashes(c)))
        self.assertEqual(self.user_count(), 0)
        with c.session_transaction() as s:
            self.assertNotIn("google_pending_signup", s)

    def test_duplicate_email_created_during_completion_becomes_a_password_confirmed_link(self):
        c = self.app.test_client()
        self.sign_in(c)
        self.local_account("g.user@example.com", verified=False)  # registered meanwhile
        r = self.complete(c)
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/login"))
        self.assertEqual(self.user_count(), 1)  # no second account, no partial account
        self.assertIsNone(self.repo.get_user_by_identity("google", "google-sub-0001"))
        self.assertIn('value="g.user@example.com"', c.get("/login").get_data(as_text=True))
        self.login(c, "g.user@example.com")
        user = self.repo.get_user(self.logged_in_user(c))
        self.assertEqual(self.repo.get_user_by_identity("google", "google-sub-0001")["user_id"], user["user_id"])
        self.assertIsNotNone(user["email_verified_at"])

    def test_identity_completed_in_another_session_is_not_created_twice(self):
        first, second = self.app.test_client(), self.app.test_client()
        self.sign_in(first)
        self.sign_in(second)
        self.assertEqual(self.complete(first).status_code, 302)
        r = self.complete(second)
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/login"))
        self.assertEqual(self.user_count(), 1)
        self.assertIsNone(self.logged_in_user(second))

    def test_create_user_with_identity_is_atomic(self):
        self.repo.create_user("taken@example.com", "h", "client", "T")
        now = auth_tokens.utcnow()
        with self.assertRaises(DuplicateEmail):
            self.repo.create_user_with_identity("taken@example.com", "h", "client", "T", None, None, "google", "s-1", now)
        self.assertIsNone(self.repo.get_user_by_identity("google", "s-1"))
        self.repo.create_user_with_identity("a@example.com", "h", "client", "A", None, None, "google", "s-2", now)
        with self.assertRaises(IdentityAlreadyLinked):
            self.repo.create_user_with_identity("b@example.com", "h", "client", "B", None, None, "google", "s-2", now)
        self.assertIsNone(self.repo.get_credentials("b@example.com"))  # no account without its identity

    # ---- existing accounts ----
    def test_linked_identity_signs_straight_in_by_subject_not_email(self):
        c = self.app.test_client()
        self.sign_in(c)
        self.complete(c)
        user_id = self.logged_in_user(c)
        c.post("/logout", data={"csrf_token": self.csrf(c, "/dashboard")})
        before = self.repo.get_identity_for_user(user_id, "google")["last_login_at"]
        time.sleep(0.01)
        # Google now reports a different address: the identity (sub) still decides the account
        r = self.sign_in(c, "?next=/requests", email="renamed@example.com")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/requests"))
        self.assertEqual(self.logged_in_user(c), user_id)
        self.assertGreater(self.repo.get_identity_for_user(user_id, "google")["last_login_at"], before)
        self.assertIsNone(self.repo.get_credentials("renamed@example.com"))

    def test_existing_verified_account_is_linked_automatically_and_logged_in(self):
        from src.app.marketplace.security import hash_password

        profile = {"primary_category": "Logo Design", "experience_level": "entry", "experience_years": 2.0, "rate": 20.0,
                   "country": "India", "skills": ["Logo Design"]}
        local = self.repo.create_user("g.user@example.com", hash_password(PASSWORD), "worker", "Local Worker", "India",
                                      worker_profile=profile)  # verified
        before_user = self.repo.get_user(local)
        _, before_hash = self.repo.get_credentials("g.user@example.com")
        c = self.app.test_client()
        r = self.sign_in(c, "?next=/requests")  # no password prompt
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/requests"))
        self.assertEqual(self.logged_in_user(c), local)
        self.assertIn("Your Google account is now linked", " ".join(flashes(c)))
        self.assertEqual(c.get("/dashboard").status_code, 200)
        identity = self.repo.get_identity_for_user(local, "google")
        self.assertEqual((identity["subject"], identity["email_at_link"]), ("google-sub-0001", "g.user@example.com"))
        # the existing account is preserved: role, profile, password hash, auth_version, verification time
        self.assertEqual(self.repo.get_user(local), before_user)
        self.assertEqual(self.repo.get_credentials("g.user@example.com"), (local, before_hash))
        self.assertEqual(self.repo.get_runtime_worker_profile(local)["primary_category"], "Logo Design")
        self.assertEqual(self.user_count(), 1)  # no new account
        with c.session_transaction() as s:
            self.assertNotIn("google_pending_link", s)
            self.assertNotIn("google_pending_signup", s)
        c.post("/logout", data={"csrf_token": self.csrf(c, "/dashboard")})
        self.sign_in(c)
        self.assertEqual(self.logged_in_user(c), local)  # from now on Google signs straight in
        self.assertEqual(self.login(self.app.test_client(), "g.user@example.com").status_code, 302)  # password still works

    def test_existing_unverified_account_requires_its_password_and_a_wrong_password_links_nothing(self):
        local = self.local_account("g.user@example.com", verified=False)
        c = self.app.test_client()
        r = self.sign_in(c, "?next=/requests")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/login?next=/requests"))
        self.assertIsNone(self.logged_in_user(c))  # Google alone never opens an unverified account
        self.assertIsNone(self.repo.get_user_by_identity("google", "google-sub-0001"))
        self.assertIsNone(self.repo.get_user(local)["email_verified_at"])
        self.assertIn("Log in with your password once to link", " ".join(flashes(c)))
        r = self.login(c, "g.user@example.com", "wrong password!")
        self.assertEqual(r.status_code, 401)
        self.assertIsNone(self.logged_in_user(c))
        self.assertIsNone(self.repo.get_user_by_identity("google", "google-sub-0001"))  # nothing linked
        self.assertIsNone(self.repo.get_user(local)["email_verified_at"])               # nor verified
        r = self.login(c, "g.user@example.com")  # the pending link survived the failed attempt
        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.logged_in_user(c), local)
        self.assertEqual(self.repo.get_user_by_identity("google", "google-sub-0001")["user_id"], local)
        self.assertIn("Your Google account is now linked", " ".join(flashes(c)))

    def test_existing_unverified_account_is_verified_when_linked(self):
        local = self.local_account("g.user@example.com", verified=False)
        c = self.app.test_client()
        self.sign_in(c)
        self.login(c, "g.user@example.com")
        self.assertIsNotNone(self.repo.get_user(local)["email_verified_at"])
        self.assertEqual(self.repo.get_user_by_identity("google", "google-sub-0001")["user_id"], local)

    def test_no_link_across_different_addresses_or_after_expiry_but_login_succeeds(self):
        self.local_account("g.user@example.com", verified=False)
        other = self.local_account("someone.else@example.com")
        c = self.app.test_client()
        self.sign_in(c)
        r = self.login(c, "someone.else@example.com")
        self.assertEqual((r.status_code, self.logged_in_user(c)), (302, other))
        self.assertIsNone(self.repo.get_user_by_identity("google", "google-sub-0001"))
        self.assertIn("different email address", " ".join(flashes(c)))
        c2 = self.app.test_client()
        self.sign_in(c2)
        with mock.patch.object(auth_tokens, "utcnow", return_value=auth_tokens.utcnow() + timedelta(minutes=16)):
            r = self.login(c2, "g.user@example.com")
        self.assertEqual(r.status_code, 302)
        self.assertIsNotNone(self.logged_in_user(c2))
        self.assertIsNone(self.repo.get_user_by_identity("google", "google-sub-0001"))

    def test_failed_link_never_blocks_a_valid_password_login(self):
        local = self.local_account("g.user@example.com", verified=False)
        c = self.app.test_client()
        self.sign_in(c)
        with mock.patch.object(self.repo, "link_identity", side_effect=RuntimeError("database down")):
            r = self.login(c, "g.user@example.com")
        self.assertEqual((r.status_code, self.logged_in_user(c)), (302, local))
        self.assertIn("Google could not be linked right now", " ".join(flashes(c)))

    def test_account_linked_to_a_different_google_identity_is_refused(self):
        local = self.local_account("g.user@example.com")  # verified: would otherwise be linked automatically
        self.repo.link_identity(local, "google", "another-google-sub", "g.user@example.com", auth_tokens.utcnow())
        before = self.repo.get_identity_for_user(local, "google")
        c = self.app.test_client()
        r = self.sign_in(c)
        self.assertEqual(self.repo.get_identity_for_user(local, "google"), before)  # the existing link is unchanged
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/login"))
        self.assertIsNone(self.logged_in_user(c))
        self.assertIn("linked to a different Google account", " ".join(flashes(c)))
        with c.session_transaction() as s:
            self.assertNotIn("google_pending_link", s)
            self.assertNotIn("google_pending_signup", s)
        self.assertIsNone(self.repo.get_user_by_identity("google", "google-sub-0001"))
        with self.assertRaises(UserAlreadyHasIdentity):
            self.repo.link_identity(local, "google", "google-sub-0001", "g.user@example.com", auth_tokens.utcnow())

    def test_unverified_account_linked_elsewhere_cannot_be_taken_over_with_its_password(self):
        local = self.local_account("g.user@example.com", verified=False)
        self.repo.link_identity(local, "google", "another-google-sub", "g.user@example.com", auth_tokens.utcnow())
        self.repo._verification[local] = None  # an unverified account that nonetheless has a link (defensive case)
        c = self.app.test_client()
        r = self.sign_in(c)
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/login"))
        with c.session_transaction() as s:
            self.assertNotIn("google_pending_link", s)  # no pending link a password login could complete
        self.login(c, "g.user@example.com")
        self.assertEqual(self.logged_in_user(c), local)  # the password login itself still works
        self.assertEqual(self.repo.get_identity_for_user(local, "google")["subject"], "another-google-sub")
        self.assertIsNone(self.repo.get_user_by_identity("google", "google-sub-0001"))

    def test_concurrent_link_during_automatic_linking_changes_nothing(self):
        local = self.local_account("g.user@example.com")
        c = self.app.test_client()
        with mock.patch.object(self.repo, "link_identity", side_effect=IdentityAlreadyLinked("uq")):
            r = self.sign_in(c)
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/login"))
        self.assertIsNone(self.logged_in_user(c))
        self.assertIsNone(self.repo.get_identity_for_user(local, "google"))

    # ---- refusals ----
    def assert_refused(self, c, response, message="could not be completed"):
        self.assertEqual((response.status_code, response.headers["Location"]), (302, "/login"))
        self.assertIsNone(self.logged_in_user(c))
        self.assertIn(message, " ".join(flashes(c)))
        with c.session_transaction() as s:
            self.assertFalse([k for k in s if k.startswith("_state_google_") or k.startswith("google_pending")])
        self.assertEqual(self.user_count(), 0)

    def test_unverified_google_email_is_refused(self):
        c = self.app.test_client()
        self.assert_refused(c, self.sign_in(c, email_verified=False), "email address is not verified")
        c2 = self.app.test_client()
        self.assert_refused(c2, self.sign_in(c2, email_verified=None), "email address is not verified")

    def test_cancellation_at_google(self):
        c = self.app.test_client()
        _, params = self.start(c)
        r = c.get(f"/auth/google/callback?error=access_denied&state={params['state']}")
        self.assert_refused(c, r, "Google sign-in was cancelled.")
        self.exchange_mock.assert_not_called()

    def test_state_mismatch_missing_state_and_replay(self):
        c = self.app.test_client()
        self.start(c)
        r = c.get(f"/auth/google/callback?state=forged-state&code={AUTH_CODE}")
        self.assert_refused(c, r, "not started from this browser")
        r = self.app.test_client().get(f"/auth/google/callback?state=x&code={AUTH_CODE}")  # never started
        self.assertEqual(r.headers["Location"], "/login")
        self.exchange_mock.assert_not_called()
        c2 = self.app.test_client()
        self.claims = {}
        _, params = self.start(c2)
        c2.get(f"/auth/google/callback?state={params['state']}&code={AUTH_CODE}")
        r = c2.get(f"/auth/google/callback?state={params['state']}&code={AUTH_CODE}")  # the state is single use
        self.assertEqual(r.headers["Location"], "/login")
        self.assertEqual(self.exchange_mock.call_count, 1)

    def test_token_exchange_and_network_failures(self):
        import requests
        from authlib.integrations.base_client.errors import OAuthError

        for error in (OAuthError(error="invalid_grant", description="Bad Request"), requests.ConnectionError("down"),
                      requests.Timeout("slow")):
            c = self.app.test_client()

            def fail(params, error=error):
                raise error

            self.token_response = fail
            self.assert_refused(c, self.sign_in(c))
        self.token_response = lambda params: {"access_token": ACCESS_TOKEN, "token_type": "Bearer"}  # no ID token
        c = self.app.test_client()
        self.assert_refused(c, self.sign_in(c))

    def test_invalid_id_tokens_are_rejected(self):
        now = int(time.time())
        cases = [{"aud": "someone-else.apps.googleusercontent.com", "azp": "someone-else.apps.googleusercontent.com"},
                 {"iss": "https://evil.example.com"}, {"exp": now - 3600, "iat": now - 7200}, {"sub": None},
                 {"email": None}, {"email": "not-an-address"}]
        for overrides in cases:
            c = self.app.test_client()
            self.assert_refused(c, self.sign_in(c, **overrides))
        for builder in (lambda p: id_token("wrong-nonce"), lambda p: id_token(self.current_nonce, key=OTHER_KEY)):
            self.token_response = lambda params, b=builder: {"access_token": ACCESS_TOKEN, "token_type": "Bearer",
                                                               "id_token": b(params)}
            c = self.app.test_client()
            self.assert_refused(c, self.sign_in(c))

    def test_both_issuer_forms_are_accepted(self):
        c = self.app.test_client()
        r = self.sign_in(c, iss="accounts.google.com")
        self.assertEqual(r.headers["Location"], "/auth/google/complete")

    def test_callback_is_rate_limited_per_address(self):
        self.app.extensions["hirewise_google_callback_limit"].max_failures = 2
        c = self.app.test_client()
        for _ in range(2):
            c.get("/auth/google/callback?error=access_denied")
        _, params = self.start(c)
        r = c.get(f"/auth/google/callback?state={params['state']}&code={AUTH_CODE}")
        self.assertEqual(r.headers["Location"], "/login")
        self.assertIn("Too many Google sign-in attempts", " ".join(flashes(c)))
        self.exchange_mock.assert_not_called()

    def test_logged_in_users_are_sent_to_the_dashboard(self):
        self.local_account("local@example.com")
        c = self.app.test_client()
        self.login(c, "local@example.com")
        self.assertEqual(c.get("/auth/google/start").headers["Location"], "/dashboard")

    # ---- logging ----
    def test_logs_never_contain_tokens_codes_secrets_or_addresses(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.addFilter(RedactTokenPaths())
        root = logging.getLogger()
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        self.addCleanup(root.removeHandler, handler)
        self.addCleanup(root.setLevel, old_level)
        c = self.app.test_client()
        self.sign_in(c)
        self.complete(c)
        self.sign_in(self.app.test_client(), email_verified=False)
        self.sign_in(self.app.test_client(), aud="wrong")
        log = stream.getvalue()
        self.assertIn("google sign-up completed", log)
        for secret in (AUTH_CODE, ACCESS_TOKEN, CLIENT_SECRET, "g.user@example.com", "google-sub-0001", "code_verifier",
                       self.current_nonce):
            self.assertNotIn(secret, log)
        record = logging.LogRecord("werkzeug", logging.INFO, __file__, 1,
                                   '"GET /auth/google/callback?state=s1&code=%s HTTP/1.1" 302 -', (AUTH_CODE,), None)
        RedactTokenPaths().filter(record)
        self.assertNotIn(AUTH_CODE, record.getMessage())
        self.assertIn("/auth/google/callback?[redacted]", record.getMessage())


class VerifiedIdentityTests(unittest.TestCase):
    """The second, independent check of the validated claims."""

    def claims(self, **overrides):
        now = time.time()
        base = {"iss": ISSUER, "aud": CLIENT_ID, "exp": now + 60, "sub": "123", "email": "A.B@Example.com",
                "email_verified": True, "name": "  A   B  "}
        base.update(overrides)
        return base

    def check(self, **overrides):
        return google_auth.verified_identity(self.claims(**overrides), CLIENT_ID, (ISSUER, "accounts.google.com"),
                                             time.time())

    def test_accepts_and_normalises(self):
        identity, verified = self.check()
        self.assertEqual(identity, {"sub": "123", "email": "a.b@example.com", "name": "A B"})
        self.assertTrue(verified)
        self.assertTrue(self.check(email_verified="true")[1])
        self.assertTrue(self.check(aud=[CLIENT_ID, "other"])[0])
        self.assertFalse(self.check(email_verified="false")[1])
        self.assertFalse(self.check(email_verified=None)[1])

    def test_rejects(self):
        for overrides in ({"iss": "https://evil.example.com"}, {"aud": "other"}, {"exp": time.time() - 3600},
                          {"exp": None}, {"sub": ""}, {"sub": "x" * 256}, {"sub": 5}, {"email": None},
                          {"email": "nope"}):
            self.assertIsNone(self.check(**overrides)[0], overrides)
        self.assertEqual(google_auth.verified_identity(None, CLIENT_ID, (ISSUER,), time.time()), (None, False))


# ---------------------------------------------------------------------------
# Repository behaviour (in-memory always; PostgreSQL opt-in on POSTGRES_TEST_DB only)
# ---------------------------------------------------------------------------
class IdentityRepositoryBehaviour:
    def make_repository(self):  # pragma: no cover - provided by subclasses
        raise NotImplementedError

    def setUp(self):
        self.repo = self.make_repository()
        self.now = auth_tokens.utcnow()

    def test_create_with_identity_is_verified_and_atomic(self):
        profile = {"primary_category": "Logo Design", "experience_level": "entry", "experience_years": 2.0, "rate": 20.0,
                   "country": "India", "skills": ["Logo Design"]}
        user_id = self.repo.create_user_with_identity("w@example.com", "h", "worker", "W", "India", profile, "google",
                                                      "sub-w", self.now)
        user = self.repo.get_user_by_identity("google", "sub-w")
        self.assertEqual((user["user_id"], user["email"], user["role"]), (user_id, "w@example.com", "worker"))
        self.assertIsNotNone(user["email_verified_at"])
        self.assertEqual(self.repo.get_runtime_worker_profile(user_id)["skills"], ["Logo Design"])
        identity = self.repo.get_identity_for_user(user_id, "google")
        self.assertEqual((identity["subject"], identity["email_at_link"]), ("sub-w", "w@example.com"))
        with self.assertRaises(DuplicateEmail):
            self.repo.create_user_with_identity("w@example.com", "h", "client", "X", None, None, "google", "sub-x",
                                                self.now)
        self.assertIsNone(self.repo.get_user_by_identity("google", "sub-x"))
        with self.assertRaises(IdentityAlreadyLinked):
            self.repo.create_user_with_identity("y@example.com", "h", "client", "Y", None, None, "google", "sub-w",
                                                self.now)
        self.assertIsNone(self.repo.get_credentials("y@example.com"))  # rolled back: no account without identity

    def test_link_identity_rules(self):
        a = self.repo.create_user("a@example.com", "h", "client", "A", verified=False)
        b = self.repo.create_user("b@example.com", "h", "client", "B")
        self.assertIsNone(self.repo.get_user_by_identity("google", "sub-a"))
        self.repo.link_identity(a, "google", "sub-a", "a@example.com", self.now)
        self.assertEqual(self.repo.get_user_by_identity("google", "sub-a")["user_id"], a)
        self.assertIsNotNone(self.repo.get_user(a)["email_verified_at"])  # the provider verified the address
        with self.assertRaises(IdentityAlreadyLinked):  # one identity, one account
            self.repo.link_identity(b, "google", "sub-a", "b@example.com", self.now)
        with self.assertRaises(UserAlreadyHasIdentity):  # one Google identity per account
            self.repo.link_identity(a, "google", "sub-other", "a@example.com", self.now)
        self.assertIsNone(self.repo.get_identity_for_user(b, "google"))
        later = self.now + timedelta(minutes=5)
        self.repo.record_identity_login("google", "sub-a", later)
        self.assertEqual(self.repo.get_identity_for_user(a, "google")["last_login_at"], later)
        self.repo.record_identity_login("google", "unknown", later)  # no-op
        with self.assertRaises(ValueError):  # only known providers
            self.repo.link_identity(b, "github", "x", "b@example.com", self.now)


class InMemoryIdentityTests(IdentityRepositoryBehaviour, unittest.TestCase):
    def make_repository(self):
        return InMemoryMarketplaceRepository(["Logo Design"], ["Logo Design"], [], [])


@unittest.skipUnless(RUN_PG, "opt-in live PostgreSQL test: set HIREWISE_RUN_POSTGRES_TESTS=1")
class PostgresIdentityTests(IdentityRepositoryBehaviour, unittest.TestCase):
    """POSTGRES_TEST_DB only (never POSTGRES_DB): a fresh current schema per class, dropped afterwards."""

    @classmethod
    def setUpClass(cls):
        from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine
        from src.db.schema import categories, metadata, skills

        test_db, main_db = os.environ.get("POSTGRES_TEST_DB"), os.environ.get("POSTGRES_DB")
        if not test_db or test_db == main_db:
            raise unittest.SkipTest("POSTGRES_TEST_DB must be set and different from POSTGRES_DB")
        settings = load_postgres_settings(TEST_DATABASE_ENV)
        assert settings.database == test_db != main_db
        cls.engine = make_engine(settings)
        metadata.drop_all(cls.engine)
        metadata.create_all(cls.engine)
        with cls.engine.begin() as conn:
            conn.execute(categories.insert().values(name="Logo Design"))
            conn.execute(skills.insert().values(name="Logo Design"))

    @classmethod
    def tearDownClass(cls):
        from src.db.schema import metadata

        metadata.drop_all(cls.engine)

    def make_repository(self):
        from src.db.marketplace import PostgresMarketplaceRepository
        from src.db.schema import RUNTIME_TABLES, metadata

        with self.engine.begin() as conn:
            for name in reversed(RUNTIME_TABLES):
                conn.execute(metadata.tables[name].delete())
        return PostgresMarketplaceRepository(self.engine)

    def test_database_constraints(self):
        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError
        from src.db.schema import user_identities

        user = self.repo.create_user("c@example.com", "h", "client", "C")
        for row in (dict(user_id=user, provider="github", subject="s", email_at_link="c@example.com"),
                    dict(user_id=user, provider="google", subject="", email_at_link="c@example.com"),
                    dict(user_id=user, provider="google", subject="x" * 256, email_at_link="c@example.com"),
                    dict(user_id=user, provider="google", subject="s", email_at_link="C@Example.com"),
                    dict(user_id=999999, provider="google", subject="s", email_at_link="c@example.com")):
            with self.assertRaises(IntegrityError, msg=row):
                with self.engine.begin() as conn:
                    conn.execute(insert(user_identities).values(**row))


if __name__ == "__main__":
    unittest.main()
