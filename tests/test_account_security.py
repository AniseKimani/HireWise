"""Pre-deployment Milestone 1: email verification, password recovery and their foundations.

- Token + repository behaviour (in-memory always; PostgreSQL opt-in, on POSTGRES_TEST_DB only).
- Mailer backends and configuration validation (no real email is ever sent: SMTP is mocked).
- Registration / verification / unverified-user restrictions / password reset / session invalidation, end to
  end through the Flask test client with the in-memory repository and the memory mail backend.
- Schema v2 -> v3 migration (opt-in, POSTGRES_TEST_DB only).
"""
import hashlib
import io
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

from src.app import create_app
from src.app.marketplace import auth_tokens
from src.app.marketplace.mailer import (ConsoleBackend, FileBackend, MailConfigError, Mailer, MemoryBackend, SMTPBackend,
                                        build_mailer, resolve_settings)
from src.db.marketplace import TOKEN_EXPIRED, TOKEN_INVALID, TOKEN_USED, TOKEN_VALID, InMemoryMarketplaceRepository
from tests.test_day12_marketplace import HAVE_DATA, PASSWORD, TOKEN, shared

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PG = os.environ.get("HIREWISE_RUN_POSTGRES_TESTS") == "1"
MAIL_ENV = ("MAIL_BACKEND", "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_USE_TLS", "MAIL_FROM",
            "APP_BASE_URL", "MAIL_OUTBOX_DIR")
LINK = re.compile(r"https?://[^/\s]+(/(?:verify-email/confirm|reset-password)/([A-Za-z0-9_-]+))")


def later(**delta):
    """Patch the token clock forward."""
    return mock.patch.object(auth_tokens, "utcnow", return_value=auth_tokens.datetime.now(auth_tokens.timezone.utc) + timedelta(**delta))


# ---------------------------------------------------------------------------
# Tokens and repository (shared behaviour)
# ---------------------------------------------------------------------------
class TokenRepositoryBehaviour:
    def make_repository(self):  # pragma: no cover - provided by subclasses
        raise NotImplementedError

    def setUp(self):
        self.repo = self.make_repository()
        self.user = self.repo.create_user("token.user@example.com", "h", "client", "T", verified=False)
        self.other = self.repo.create_user("other.user@example.com", "h", "client", "O", verified=False)

    def stored_token_values(self):
        raise NotImplementedError

    def test_raw_tokens_are_random_unique_and_well_formed(self):
        tokens = {auth_tokens.new_raw_token() for _ in range(500)}
        self.assertEqual(len(tokens), 500)
        self.assertTrue(all(len(t) == 43 and auth_tokens.well_formed(t) for t in tokens))

    def test_only_the_sha256_hash_is_stored(self):
        raw = auth_tokens.issue(self.repo, self.user, auth_tokens.VERIFY_EMAIL, timedelta(hours=24))
        self.assertEqual(auth_tokens.hash_token(raw), hashlib.sha256(raw.encode()).hexdigest())
        stored = self.stored_token_values()
        self.assertIn(auth_tokens.hash_token(raw), stored)
        self.assertFalse(any(raw in str(v) for v in stored))

    def test_verify_token_single_use(self):
        raw = auth_tokens.issue(self.repo, self.user, auth_tokens.VERIFY_EMAIL, timedelta(hours=24))
        self.assertEqual(auth_tokens.state(self.repo, raw, auth_tokens.VERIFY_EMAIL), TOKEN_VALID)
        self.assertIsNone(self.repo.get_user(self.user)["email_verified_at"])
        self.assertEqual(auth_tokens.verify_email(self.repo, raw), (TOKEN_VALID, self.user))
        self.assertIsNotNone(self.repo.get_user(self.user)["email_verified_at"])
        self.assertEqual(auth_tokens.verify_email(self.repo, raw), (TOKEN_USED, None))
        self.assertIsNone(self.repo.get_user(self.other)["email_verified_at"])

    def test_expired_tokens_are_rejected(self):
        raw = auth_tokens.issue(self.repo, self.user, auth_tokens.VERIFY_EMAIL, timedelta(hours=24))
        with later(hours=25):
            self.assertEqual(auth_tokens.state(self.repo, raw, auth_tokens.VERIFY_EMAIL), TOKEN_EXPIRED)
            self.assertEqual(auth_tokens.verify_email(self.repo, raw), (TOKEN_EXPIRED, None))
        self.assertIsNone(self.repo.get_user(self.user)["email_verified_at"])

    def test_purposes_are_not_interchangeable(self):
        verify = auth_tokens.issue(self.repo, self.user, auth_tokens.VERIFY_EMAIL, timedelta(hours=24))
        reset = auth_tokens.issue(self.repo, self.user, auth_tokens.RESET_PASSWORD, timedelta(hours=1))
        self.assertEqual(auth_tokens.reset_password(self.repo, verify, "new-hash"), (TOKEN_INVALID, None))
        self.assertEqual(auth_tokens.verify_email(self.repo, reset), (TOKEN_INVALID, None))
        # both are still usable for their own purpose
        self.assertEqual(auth_tokens.verify_email(self.repo, verify)[0], TOKEN_VALID)
        self.assertEqual(auth_tokens.reset_password(self.repo, reset, "new-hash")[0], TOKEN_VALID)

    def test_unknown_and_malformed_tokens(self):
        self.assertEqual(auth_tokens.verify_email(self.repo, auth_tokens.new_raw_token()), (TOKEN_INVALID, None))
        exploding = mock.Mock(side_effect=AssertionError("malformed input must not reach the repository"))
        repo = mock.Mock(verify_email_with_token=exploding, reset_password_with_token=exploding, auth_token_state=exploding)
        for bad in (None, "", "short", "!" * 43, "a" * 44, "a" * 10000, "../../etc/passwd"):
            self.assertEqual(auth_tokens.verify_email(repo, bad), (TOKEN_INVALID, None))
            self.assertEqual(auth_tokens.reset_password(repo, bad, "h"), (TOKEN_INVALID, None))
            self.assertEqual(auth_tokens.state(repo, bad, auth_tokens.VERIFY_EMAIL), TOKEN_INVALID)

    def test_new_token_retires_older_tokens_of_the_same_purpose(self):
        first = auth_tokens.issue(self.repo, self.user, auth_tokens.VERIFY_EMAIL, timedelta(hours=24))
        reset = auth_tokens.issue(self.repo, self.user, auth_tokens.RESET_PASSWORD, timedelta(hours=1))
        second = auth_tokens.issue(self.repo, self.user, auth_tokens.VERIFY_EMAIL, timedelta(hours=24))
        self.assertEqual(auth_tokens.state(self.repo, first, auth_tokens.VERIFY_EMAIL), TOKEN_USED)
        self.assertEqual(auth_tokens.state(self.repo, second, auth_tokens.VERIFY_EMAIL), TOKEN_VALID)
        self.assertEqual(auth_tokens.state(self.repo, reset, auth_tokens.RESET_PASSWORD), TOKEN_VALID)  # other purpose untouched

    def test_concurrent_consumption_succeeds_once(self):
        raw = auth_tokens.issue(self.repo, self.user, auth_tokens.RESET_PASSWORD, timedelta(hours=1))
        results, barrier = [], threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(auth_tokens.reset_password(self.repo, raw, "concurrent-hash")[0])

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(results), [TOKEN_USED] * 7 + [TOKEN_VALID])
        self.assertEqual(self.repo.get_user(self.user)["auth_version"], 1)  # incremented exactly once

    def test_reset_sets_password_increments_version_retires_tokens_and_verifies(self):
        old_version = self.repo.get_user(self.user)["auth_version"]
        first = auth_tokens.issue(self.repo, self.user, auth_tokens.RESET_PASSWORD, timedelta(hours=1))
        self.assertEqual(auth_tokens.reset_password(self.repo, first, "the-new-hash"), (TOKEN_VALID, self.user))
        user_id, stored = self.repo.get_credentials("token.user@example.com")
        self.assertEqual(stored, "the-new-hash")
        self.assertEqual(self.repo.get_user(self.user)["auth_version"], old_version + 1)
        self.assertIsNotNone(self.repo.get_user(self.user)["email_verified_at"])  # the link proved the mailbox
        self.assertEqual(self.repo.get_user(self.other)["auth_version"], 0)

    def test_operator_created_accounts_are_verified(self):
        uid = self.repo.create_user("operator@example.com", "h", "client", "Op")
        self.assertIsNotNone(self.repo.get_user(uid)["email_verified_at"])
        self.assertEqual(self.repo.get_user(uid)["auth_version"], 0)


class InMemoryTokenTests(TokenRepositoryBehaviour, unittest.TestCase):
    def make_repository(self):
        return InMemoryMarketplaceRepository(["Logo Design"], ["Logo Design"], [], ["W000001"])

    def stored_token_values(self):
        return [v for t in self.repo._tokens.values() for v in t.values()] + list(self.repo._tokens)


@unittest.skipUnless(RUN_PG, "opt-in live PostgreSQL test: set HIREWISE_RUN_POSTGRES_TESTS=1")
class PostgresTokenTests(TokenRepositoryBehaviour, unittest.TestCase):
    """POSTGRES_TEST_DB only (never POSTGRES_DB): a fresh schema v3 per class, dropped afterwards."""

    @classmethod
    def setUpClass(cls):
        from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine
        from src.db.schema import metadata

        test_db, main_db = os.environ.get("POSTGRES_TEST_DB"), os.environ.get("POSTGRES_DB")
        if not test_db or test_db == main_db:
            raise unittest.SkipTest("POSTGRES_TEST_DB must be set and different from POSTGRES_DB")
        cls.engine = make_engine(load_postgres_settings(TEST_DATABASE_ENV))
        metadata.drop_all(cls.engine)
        metadata.create_all(cls.engine)

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

    def stored_token_values(self):
        from sqlalchemy import select
        from src.db.schema import auth_tokens as table

        with self.engine.connect() as conn:
            return [v for row in conn.execute(select(table)) for v in row]

    def test_database_constraints(self):
        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError
        from src.db.schema import auth_tokens as table, user_email_verification

        now = auth_tokens.utcnow()
        bad_rows = [dict(user_id=self.user, purpose="login", token_hash="a" * 64, expires_at=now + timedelta(hours=1)),
                    dict(user_id=self.user, purpose="verify_email", token_hash="RAW-TOKEN", expires_at=now + timedelta(hours=1)),
                    dict(user_id=self.user, purpose="verify_email", token_hash="b" * 64, created_at=now, expires_at=now),
                    dict(user_id=999999, purpose="verify_email", token_hash="c" * 64, expires_at=now + timedelta(hours=1))]
        for row in bad_rows:
            with self.assertRaises(IntegrityError, msg=row):
                with self.engine.begin() as conn:
                    conn.execute(insert(table).values(**row))
        auth_tokens.issue(self.repo, self.user, auth_tokens.VERIFY_EMAIL, timedelta(hours=1))
        with self.assertRaises(IntegrityError):  # token_hash is unique
            with self.engine.begin() as conn:
                h = conn.execute(table.select()).first().token_hash
                conn.execute(insert(table).values(user_id=self.other, purpose="verify_email", token_hash=h,
                                                  expires_at=now + timedelta(hours=1)))
        with self.assertRaises(IntegrityError):  # one verification row per user, referencing a real user
            with self.engine.begin() as conn:
                conn.execute(insert(user_email_verification).values(user_id=999999))


# ---------------------------------------------------------------------------
# Mailer
# ---------------------------------------------------------------------------
SMTP_OK = {"MAIL_BACKEND": "smtp", "SMTP_HOST": "smtp.example.com", "SMTP_PORT": "587", "SMTP_USERNAME": "mailer-user",
           "SMTP_PASSWORD": "super-secret-smtp-password", "SMTP_USE_TLS": "1", "MAIL_FROM": "HireWise <no-reply@example.com>",
           "APP_BASE_URL": "https://hirewise.example.com", "MAIL_TIMEOUT_SECONDS": 7, "MAIL_SEND_ASYNC": False}


class TestMailer(unittest.TestCase):
    def message(self, mailer):
        return mailer.build("someone@example.com", "Subject line", "Body with a link https://x/reset-password/" + "t" * 43)

    def test_console_backend_writes_the_message_to_its_stream(self):
        stream = io.StringIO()
        mailer = Mailer(ConsoleBackend(stream), "HireWise <no-reply@hirewise.local>", "http://localhost", send_async=False)
        self.assertTrue(mailer.send(self.message(mailer), "test"))
        self.assertIn("Subject: Subject line", stream.getvalue())
        self.assertIn("someone@example.com", stream.getvalue())

    def test_file_backend_writes_eml_files_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            mailer = build_mailer({"MAIL_BACKEND": "file", "MAIL_OUTBOX_DIR": tmp, "APP_BASE_URL": "http://localhost",
                                   "MAIL_SEND_ASYNC": False}, environment="development", required=True)
            self.assertIsInstance(mailer.backend, FileBackend)
            self.assertTrue(mailer.send(self.message(mailer), "test"))
            files = list(Path(tmp).glob("*.eml"))
            self.assertEqual(len(files), 1)
            self.assertIn(b"Subject: Subject line", files[0].read_bytes())
        for inside in (REPO_ROOT, REPO_ROOT / "outbox", REPO_ROOT / "data" / "mail"):
            with self.assertRaises(MailConfigError):
                build_mailer({"MAIL_BACKEND": "file", "MAIL_OUTBOX_DIR": str(inside), "APP_BASE_URL": "http://localhost"},
                             environment="production", required=True)

    def test_default_outbox_is_outside_the_repository(self):
        mailer = build_mailer({"MAIL_BACKEND": "file", "APP_BASE_URL": "http://localhost"}, environment="development", required=True)
        self.assertNotIn(REPO_ROOT, mailer.backend.outbox_dir.parents)

    def test_production_configuration_is_validated(self):
        cases = {
            "MAIL_BACKEND is not set": {**SMTP_OK, "MAIL_BACKEND": None},
            "MAIL_BACKEND must be one of": {**SMTP_OK, "MAIL_BACKEND": "carrier-pigeon"},
            "for automated tests only": {**SMTP_OK, "MAIL_BACKEND": "memory"},
            "SMTP_HOST is not set": {**SMTP_OK, "SMTP_HOST": ""},
            "MAIL_FROM must be": {**SMTP_OK, "MAIL_FROM": "not an address"},
            "APP_BASE_URL is not set": {**SMTP_OK, "APP_BASE_URL": None},
            "must use https": {**SMTP_OK, "APP_BASE_URL": "http://hirewise.example.com"},
            "absolute http(s) address": {**SMTP_OK, "APP_BASE_URL": "hirewise.example.com/path"},
            "SMTP_USE_TLS must be enabled": {**SMTP_OK, "SMTP_USE_TLS": "0"},
            "SMTP_USERNAME and SMTP_PASSWORD": {**SMTP_OK, "SMTP_PASSWORD": ""},
            "SMTP_PORT must be a number": {**SMTP_OK, "SMTP_PORT": "submission"},
        }
        for expected, settings in cases.items():
            with self.assertRaises(MailConfigError, msg=expected) as ctx:
                build_mailer(settings, environment="production", required=True)
            self.assertIn(expected, str(ctx.exception))
            self.assertNotIn("super-secret-smtp-password", str(ctx.exception))
        mailer = build_mailer(SMTP_OK, environment="production", required=True)
        self.assertIsInstance(mailer.backend, SMTPBackend)
        self.assertNotIn("super-secret", repr(mailer.backend))
        # https is not required for a localhost production-mode run; nothing is required when the marketplace is off
        self.assertIsNotNone(build_mailer({**SMTP_OK, "APP_BASE_URL": "http://127.0.0.1:5000"}, environment="production", required=True))
        self.assertIsNone(build_mailer({"MAIL_BACKEND": None}, environment="production", required=False))

    def test_smtp_delivery_uses_timeout_starttls_and_login(self):
        mailer = build_mailer(SMTP_OK, environment="production", required=True)
        with mock.patch("smtplib.SMTP") as smtp:
            self.assertTrue(mailer.send(self.message(mailer), "test"))
        smtp.assert_called_once_with("smtp.example.com", 587, timeout=7.0)
        server = smtp.return_value  # smtplib.SMTP.__enter__ returns the connection itself
        server.__enter__.assert_called_once()  # used as a context manager, so the connection is always closed
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("mailer-user", "super-secret-smtp-password")
        server.send_message.assert_called_once()

    def test_smtp_failure_and_timeout_are_reported_without_secrets(self):
        mailer = build_mailer(SMTP_OK, environment="production", required=True)
        for error in (ConnectionRefusedError("refused"), TimeoutError("timed out"), OSError("network down")):
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            logging.getLogger().addHandler(handler)
            try:
                with mock.patch("smtplib.SMTP", side_effect=error):
                    self.assertFalse(mailer.send(self.message(mailer), "reset_password"))
            finally:
                logging.getLogger().removeHandler(handler)
            log = stream.getvalue()
            self.assertIn(type(error).__name__, log)
            for secret in ("super-secret-smtp-password", "mailer-user", "someone@example.com", "t" * 43, "smtp.example.com"):
                self.assertNotIn(secret, log)

    def test_testing_configuration_never_uses_real_delivery(self):
        with mock.patch.dict(os.environ, {**SMTP_OK, "MAIL_TIMEOUT_SECONDS": "7", "MAIL_SEND_ASYNC": "0"}):
            app = create_app("testing")
        self.assertIsInstance(app.extensions["hirewise_mailer"].backend, MemoryBackend)
        self.assertEqual(app.extensions["hirewise_mailer"].base_url, "http://localhost")

    def test_production_start_up_requires_mail_configuration_when_the_marketplace_is_active(self):
        from src.app.factory import MissingMailConfig

        env = {k: v for k, v in os.environ.items() if k not in MAIL_ENV and k != "HIREWISE_ENV"}
        repo = InMemoryMarketplaceRepository(["Logo Design"], ["Logo Design"], [], [])
        minimal = {"LOAD_RECOMMENDER_ON_STARTUP": False, "NEO4J_HEALTHCHECK_ENABLED": False, "POSTGRES_HEALTHCHECK_ENABLED": False,
                   "SECRET_KEY": "k" * 64}
        with mock.patch.dict(os.environ, {**env, "SMTP_PASSWORD": "super-secret-smtp-password"}, clear=True):
            with self.assertRaises(MissingMailConfig) as ctx:
                create_app(dict(minimal), marketplace_repository=repo)
            self.assertNotIn("super-secret", str(ctx.exception))
            create_app(dict(minimal))  # no marketplace: no mail configuration needed (as before)
        with mock.patch.dict(os.environ, {**env, **{k: str(v) for k, v in SMTP_OK.items()}}, clear=True):
            app = create_app(dict(minimal), marketplace_repository=repo)
        self.assertIsInstance(app.extensions["hirewise_mailer"].backend, SMTPBackend)

    def test_development_server_logs_redact_tokens_in_paths(self):
        from src.app.factory import RedactTokenPaths

        token = auth_tokens.new_raw_token()
        for path in (f"/verify-email/confirm/{token}", f"/reset-password/{token}"):
            record = logging.LogRecord("werkzeug", logging.INFO, __file__, 1, '%s - - "%s" %s', ("127.0.0.1", f"GET {path} HTTP/1.1", 302), None)
            RedactTokenPaths().filter(record)
            self.assertNotIn(token, record.getMessage())
            self.assertIn("[redacted]", record.getMessage())


# ---------------------------------------------------------------------------
# End-to-end flows (Flask test client)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_DATA, "processed data / Day 8 results absent")
class AccountFlowTests(unittest.TestCase):
    def setUp(self):
        self.service, self.runtime = shared()
        self.repo = InMemoryMarketplaceRepository.from_artifacts()
        self.app = create_app("testing", service=self.service, marketplace_repository=self.repo, runtime_recommender=self.runtime)
        self.mail = self.app.extensions["hirewise_mailer"].backend

    # ---- helpers ----
    def token(self, c, path="/login"):
        """The session's CSRF token (per session, so any page with a form shows it; /login always has one)."""
        match = TOKEN.search(c.get(path).get_data(as_text=True))
        return match.group(1) if match else TOKEN.search(c.get("/login").get_data(as_text=True)).group(1)

    def post(self, c, path, data=None, token_path="/login"):
        return c.post(path, data={**(data or {}), "csrf_token": self.token(c, token_path)})

    def register(self, c, email, role="client", **extra):
        form = {"email": email, "role": role, "display_name": email.split("@")[0], "password": PASSWORD, "confirm_password": PASSWORD,
                **({"worker_country": "India", "primary_category": "Logo Design", "experience_level": "entry", "experience_years": "2",
                    "rate": "20", "skills": "Logo Design"} if role == "worker" else {}), **extra}
        return self.post(c, "/register", form, "/register")

    def login(self, c, email, password=PASSWORD):
        return self.post(c, "/login", {"email": email, "password": password})

    def links(self, email, kind):
        out = []
        for m in self.mail.messages:
            for match in LINK.finditer(m.get_content()):
                if m["To"] == email and kind in match.group(1):
                    out.append(match.group(1))
        return out

    def verify(self, c, email):
        c.get(self.links(email, "verify-email")[-1])
        return self.post(c, "/verify-email/confirm", token_path="/verify-email/confirm")

    def user(self, email):
        return self.repo.get_user(self.repo.get_credentials(email)[0])

    def new_verified_client(self, email):
        c = self.app.test_client()
        self.register(c, email)
        self.verify(c, email)
        self.login(c, email)
        return c

    # ---- registration and verification ----
    def test_registration_creates_an_unverified_account_and_emails_a_link(self):
        c = self.app.test_client()
        r = self.register(c, "New.Person@Example.com")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/register/check-email"))
        self.assertIsNone(self.user("new.person@example.com")["email_verified_at"])
        self.assertEqual(c.get("/dashboard").status_code, 302)  # registration does not log in
        [message] = self.mail.messages
        self.assertEqual(message["To"], "new.person@example.com")
        self.assertEqual(message["Subject"], "Verify your HireWise email address")
        self.assertIn("HireWise", message.get_content())
        self.assertNotIn(PASSWORD, message.get_content())
        [link] = self.links("new.person@example.com", "verify-email")
        raw = link.rsplit("/", 1)[1]
        self.assertTrue(LINK.search(message.get_content()).group(0).startswith("http://localhost/verify-email/confirm/"))
        self.assertIn(auth_tokens.hash_token(raw), self.repo._tokens)
        self.assertFalse(any(raw in str(v) for t in self.repo._tokens.values() for v in t.values()))
        page = c.get("/register/check-email").get_data(as_text=True)
        self.assertIn("Check your email", page)

    def test_unverified_user_can_log_in_browse_and_see_their_account(self):
        c = self.app.test_client()
        self.register(c, "browse@example.com")
        r = self.login(c, "browse@example.com")
        self.assertTrue(r.headers["Location"].endswith("/dashboard"))
        for path in ("/dashboard", "/client/requests", "/bookings", "/verify-email", "/", "/demo", "/forgot-password"):
            self.assertIn(c.get(path).status_code, (200, 302), path)
        dashboard = c.get("/dashboard").get_data(as_text=True)
        self.assertIn("Verify your email address to post requests", dashboard)  # banner
        self.assertIn("browse@example.com", c.get("/verify-email").get_data(as_text=True))
        self.assertEqual(self.post(c, "/logout", token_path="/dashboard").status_code, 302)
        self.assertEqual(c.get("/dashboard").status_code, 302)

    def test_unverified_client_cannot_create_requests_book_or_review(self):
        from src.app.marketplace.runtime import RuntimeRequest
        from src.app.marketplace.security import hash_password

        uid = self.repo.create_user("unv.client@example.com", hash_password(PASSWORD), "client", "U", verified=False)
        c = self.app.test_client()
        self.login(c, "unv.client@example.com")
        for method in ("get", "post"):
            r = getattr(c, method)("/client/requests/new", data={"csrf_token": self.token(c, "/dashboard")} if method == "post" else None)
            self.assertEqual((r.status_code, r.headers["Location"]), (302, "/verify-email"), method)
        self.assertEqual(self.repo.list_requests_for_client(uid), [])
        request_id = self.repo.create_request(uid, "Logo work", "Logo Design", ["Logo Design"], "unspecified", None, None, None, None, None)
        worker = self.runtime.recommend(RuntimeRequest(
            request_id=request_id, client_user_id=uid, category="Logo Design", skills=("Logo Design",), price_tier=None,
            location_requirement=None, imported_client_id=None), 10)["recommendations"][0]["worker"]["worker_id"]
        r = self.post(c, f"/client/requests/{request_id}/book", {"worker_id": worker}, "/dashboard")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/verify-email"))
        self.assertEqual(self.repo.list_bookings(request_id=request_id), [])
        booking_id = self.repo.create_booking(request_id, uid, worker)
        self.repo.transition_booking(booking_id, "pending", "accepted")
        r = self.post(c, f"/bookings/{booking_id}/complete", {"return_to": "/bookings"}, "/dashboard")
        self.assertEqual(r.headers["Location"], "/verify-email")
        self.assertEqual(self.repo.get_booking(booking_id)["status"], "accepted")
        self.repo.transition_booking(booking_id, "accepted", "completed")
        r = self.post(c, f"/bookings/{booking_id}/review", {"rating": "5"}, "/dashboard")
        self.assertEqual(r.headers["Location"], "/verify-email")
        self.assertIsNone(self.repo.get_review(booking_id))
        self.assertIn("Verify your email address to leave a review", c.get(f"/bookings/{booking_id}/review").get_data(as_text=True))

    def test_unverified_worker_cannot_act_on_bookings(self):
        from src.app.marketplace.security import hash_password

        client = self.new_verified_client("act.client@example.com")
        request_id = int(self.post(client, "/client/requests/new", {"title": "Logo work", "category": "Logo Design", "skills": "Logo Design",
                                                                    "pay_type": "unspecified"}, "/client/requests/new").headers["Location"].rsplit("/", 1)[1])
        worker_id = re.search(r'name="worker_id" value="([^"]+)"', client.get(f"/client/requests/{request_id}").get_data(as_text=True)).group(1)
        self.post(client, f"/client/requests/{request_id}/book", {"worker_id": worker_id}, f"/client/requests/{request_id}")
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.repo.create_user("unv.worker@example.com", hash_password(PASSWORD), "worker", "W", worker_id=worker_id, verified=False)
        w = self.app.test_client()
        self.login(w, "unv.worker@example.com")
        self.assertEqual(self.post(w, f"/bookings/{booking_id}/accept", {"return_to": "/dashboard"}, "/dashboard").headers["Location"], "/verify-email")
        self.assertEqual(self.repo.get_booking(booking_id)["status"], "pending")

    def test_verification_flow_and_restored_access(self):
        c = self.app.test_client()
        self.register(c, "flow@example.com")
        self.login(c, "flow@example.com")
        link = self.links("flow@example.com", "verify-email")[0]
        raw = link.rsplit("/", 1)[1]
        r = c.get(link)
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/verify-email/confirm"))  # token leaves the URL
        page = c.get("/verify-email/confirm").get_data(as_text=True)
        self.assertIn("Confirm my email address", page)
        self.assertNotIn(raw, page)
        self.assertIsNone(self.user("flow@example.com")["email_verified_at"])  # GETs never consume
        done = self.post(c, "/verify-email/confirm", token_path="/verify-email/confirm").get_data(as_text=True)
        self.assertIn("Email address verified", done)
        self.assertIsNotNone(self.user("flow@example.com")["email_verified_at"])
        self.assertEqual(c.get("/client/requests/new").status_code, 200)
        self.assertNotIn("Verify your email address to post requests", c.get("/dashboard").get_data(as_text=True))

    def test_link_previews_do_not_consume_and_links_work_on_another_device(self):
        c = self.app.test_client()
        self.register(c, "preview@example.com")
        link = self.links("preview@example.com", "verify-email")[0]
        scanner = self.app.test_client()
        for _ in range(3):
            scanner.get(link, follow_redirects=True)
        phone = self.app.test_client()  # not logged in
        self.assertIn("Email address verified", self.verify(phone, "preview@example.com").get_data(as_text=True))

    def test_reused_expired_wrong_purpose_and_invalid_verification_links(self):
        c = self.app.test_client()
        self.register(c, "states@example.com")
        link = self.links("states@example.com", "verify-email")[0]
        with later(hours=25):
            c.get(link)
            self.assertIn("This link has expired", c.get("/verify-email/confirm").get_data(as_text=True))
            self.post(c, "/verify-email/confirm", token_path="/verify-email/confirm")
        self.assertIsNone(self.user("states@example.com")["email_verified_at"])
        self.verify(c, "states@example.com")
        c.get(link)
        self.assertIn("already been used", c.get("/verify-email/confirm").get_data(as_text=True))
        reset_raw = auth_tokens.issue(self.repo, self.user("states@example.com")["user_id"], auth_tokens.RESET_PASSWORD, timedelta(hours=1))
        c.get(f"/verify-email/confirm/{reset_raw}")
        self.assertIn("isn't valid", c.get("/verify-email/confirm").get_data(as_text=True))
        for bad in (auth_tokens.new_raw_token(), "not-a-token", "x" * 300):
            c.get(f"/verify-email/confirm/{bad}")
            self.assertIn("isn't valid", c.get("/verify-email/confirm").get_data(as_text=True))
        self.assertIn("isn't valid", self.app.test_client().get("/verify-email/confirm").get_data(as_text=True))

    def test_resend_retires_old_link_is_rate_limited_and_honest(self):
        c = self.app.test_client()
        self.register(c, "resend@example.com")
        self.login(c, "resend@example.com")
        first = self.links("resend@example.com", "verify-email")[0]
        r = self.post(c, "/verify-email/resend", token_path="/verify-email")
        self.assertEqual(r.headers["Location"], "/verify-email")
        self.assertIn("sent a new verification link", c.get("/verify-email").get_data(as_text=True))
        second = self.links("resend@example.com", "verify-email")[-1]
        self.assertNotEqual(first, second)
        c.get(first)
        self.assertIn("already been used", c.get("/verify-email/confirm").get_data(as_text=True))
        self.post(c, "/verify-email/resend", token_path="/verify-email")
        self.post(c, "/verify-email/resend", token_path="/verify-email")  # 3 within the window (limit)
        sent = len(self.mail.messages)
        self.post(c, "/verify-email/resend", token_path="/verify-email")
        self.assertEqual(len(self.mail.messages), sent)
        self.assertIn("several emails recently", c.get("/verify-email").get_data(as_text=True))
        self.mail.fail = True
        other = self.app.test_client()
        self.register(other, "resend2@example.com")
        self.login(other, "resend2@example.com")
        self.post(other, "/verify-email/resend", token_path="/verify-email")
        self.assertIn("couldn&#39;t send the email", other.get("/verify-email").get_data(as_text=True))

    def test_resend_when_already_verified_sends_nothing(self):
        c = self.new_verified_client("already@example.com")
        sent = len(self.mail.messages)
        self.post(c, "/verify-email/resend", token_path="/verify-email")
        self.assertEqual(len(self.mail.messages), sent)
        self.assertIn("already verified", c.get("/verify-email").get_data(as_text=True))

    def test_delivery_failure_keeps_registration_consistent(self):
        self.mail.fail = True
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logging.getLogger().addHandler(handler)
        try:
            r = self.register(self.app.test_client(), "nomail@example.com")
        finally:
            logging.getLogger().removeHandler(handler)
        self.assertEqual(r.headers["Location"], "/register/check-email")
        self.assertIsNone(self.user("nomail@example.com")["email_verified_at"])
        self.assertIn("email delivery failed", stream.getvalue())
        self.assertNotIn("nomail@example.com", stream.getvalue())
        self.mail.fail = False
        c = self.app.test_client()
        self.login(c, "nomail@example.com")
        self.post(c, "/verify-email/resend", token_path="/verify-email")
        self.assertEqual(self.verify(c, "nomail@example.com").status_code, 200)
        self.assertIsNotNone(self.user("nomail@example.com")["email_verified_at"])

    def test_new_registration_sends_exactly_one_verification_email(self):
        self.register(self.app.test_client(), "fresh@example.com")
        self.assertEqual([m["To"] for m in self.mail.messages], ["fresh@example.com"])
        self.assertEqual(self.mail.messages[0]["Subject"], "Verify your HireWise email address")
        self.assertEqual(len(self.links("fresh@example.com", "verify-email")), 1)

    def test_duplicate_registration_sends_no_email_changes_nothing_and_looks_the_same(self):
        owner, fresh = self.app.test_client(), self.app.test_client()
        first = self.register(owner, "owner@example.com")
        original = dict(self.repo._users[self.user("owner@example.com")["user_id"]])
        original_verification = self.repo._verification[original["user_id"]]
        tokens_before = {h: dict(t) for h, t in self.repo._tokens.items()}
        sent_before = len(self.mail.messages)  # the owner's own verification email
        attempts = [(self.app.test_client(), {"email": "OWNER@example.com"}),
                    (self.app.test_client(), {"email": "owner@example.com", "display_name": "Intruder", "role": "worker",
                                              "password": "a different password", "confirm_password": "a different password"})]
        for client, change in attempts:
            response = self.register(client, change.pop("email"), **change)
            # the same outward response as a genuinely new registration: status, redirect and page
            self.assertEqual((response.status_code, response.headers["Location"]), (first.status_code, first.headers["Location"]))
            self.assertEqual(client.get(response.headers["Location"]).get_data(as_text=True),
                             fresh.get(first.headers["Location"]).get_data(as_text=True))
        self.assertEqual(len(self.mail.messages), sent_before)  # no email of any kind for an existing address
        self.assertEqual(self.links("owner@example.com", "verify-email"), self.links("owner@example.com", "verify-email")[:1])
        # the existing account is untouched: one account, same fields, password, verification state and tokens
        self.assertEqual(sum(u["email"] == "owner@example.com" for u in self.repo._users.values()), 1)
        self.assertEqual(self.repo._users[original["user_id"]], original)
        self.assertEqual(self.repo._verification[original["user_id"]], original_verification)
        self.assertEqual({h: dict(t) for h, t in self.repo._tokens.items()}, tokens_before)
        # a genuinely new address straight afterwards still gets its verification email
        self.register(self.app.test_client(), "newcomer@example.com")
        self.assertEqual([m["To"] for m in self.mail.messages[sent_before:]], ["newcomer@example.com"])

    def test_registration_validation_feedback_is_unchanged(self):
        r = self.register(self.app.test_client(), "bad@example.com", password="short", confirm_password="short")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Password must be 10-128 characters.", r.get_data(as_text=True))
        self.assertIsNone(self.repo.get_credentials("bad@example.com"))

    def test_verified_user_keeps_existing_functionality(self):
        c = self.new_verified_client("full@example.com")
        r = self.post(c, "/client/requests/new", {"title": "Logo work", "category": "Logo Design", "skills": "Logo Design",
                                                  "pay_type": "unspecified"}, "/client/requests/new")
        self.assertEqual(r.status_code, 302)
        request_id = int(r.headers["Location"].rsplit("/", 1)[1])
        worker_id = re.search(r'name="worker_id" value="([^"]+)"', c.get(f"/client/requests/{request_id}").get_data(as_text=True)).group(1)
        self.post(c, f"/client/requests/{request_id}/book", {"worker_id": worker_id}, f"/client/requests/{request_id}")
        self.assertEqual(len(self.repo.list_bookings(request_id=request_id)), 1)

    # ---- password recovery ----
    def forgot(self, c, email):
        return self.post(c, "/forgot-password", {"email": email}, "/forgot-password")

    def test_forgot_password_response_is_the_same_for_known_and_unknown_addresses(self):
        self.repo.create_user("known@example.com", "h", "client", "K")
        a, b = self.app.test_client(), self.app.test_client()
        known, unknown = self.forgot(a, "known@example.com"), self.forgot(b, "nobody@example.com")
        self.assertEqual((known.status_code, known.headers["Location"]), (unknown.status_code, unknown.headers["Location"]))
        self.assertEqual(a.get(known.headers["Location"]).get_data(as_text=True), b.get(unknown.headers["Location"]).get_data(as_text=True))
        self.assertEqual(len(self.links("known@example.com", "reset-password")), 1)
        self.assertEqual(self.links("nobody@example.com", "reset-password"), [])
        [message] = [m for m in self.mail.messages if m["To"] == "known@example.com"]
        self.assertEqual(message["Subject"], "Reset your HireWise password")
        bad = self.forgot(self.app.test_client(), "not-an-email")
        self.assertEqual(bad.status_code, 400)
        self.assertIn("Enter a valid email address.", bad.get_data(as_text=True))

    def reset(self, c, link, password="a brand new password", confirm=None):
        c.get(link)
        return self.post(c, "/reset-password", {"password": password, "confirm_password": confirm or password}, "/reset-password")

    def test_password_reset_flow(self):
        from werkzeug.security import check_password_hash

        self.new_verified_client("reset@example.com")
        self.forgot(self.app.test_client(), "reset@example.com")
        link = self.links("reset@example.com", "reset-password")[-1]
        c = self.app.test_client()
        r = c.get(link)
        self.assertEqual(r.headers["Location"], "/reset-password")
        self.assertIn("Choose a new password", c.get("/reset-password").get_data(as_text=True))
        r = self.post(c, "/reset-password", {"password": "a brand new password", "confirm_password": "a brand new password"}, "/reset-password")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/login"))
        stored = self.repo.get_credentials("reset@example.com")[1]
        self.assertTrue(stored.startswith(("scrypt:", "pbkdf2:")) and check_password_hash(stored, "a brand new password"))
        self.assertEqual(self.login(self.app.test_client(), "reset@example.com").status_code, 401)  # old password rejected
        self.assertEqual(self.login(self.app.test_client(), "reset@example.com", "a brand new password").headers["Location"], "/dashboard")

    def test_password_reset_ends_existing_sessions_only_for_that_account(self):
        first, second = self.new_verified_client("victim@example.com"), self.app.test_client()
        self.login(second, "victim@example.com")
        bystander = self.new_verified_client("bystander@example.com")
        for c in (first, second, bystander):
            self.assertEqual(c.get("/dashboard").status_code, 200)
        self.forgot(self.app.test_client(), "victim@example.com")
        self.reset(self.app.test_client(), self.links("victim@example.com", "reset-password")[-1])
        for c in (first, second):
            r = c.get("/dashboard")
            self.assertEqual(r.status_code, 302)
            self.assertIn("/login", r.headers["Location"])
            self.assertIn("signed out", c.get("/login").get_data(as_text=True))
        self.assertEqual(bystander.get("/dashboard").status_code, 200)

    def test_expired_reused_invalid_and_wrong_purpose_reset_links(self):
        self.new_verified_client("links@example.com")
        old_hash = self.repo.get_credentials("links@example.com")[1]
        self.forgot(self.app.test_client(), "links@example.com")
        link = self.links("links@example.com", "reset-password")[-1]
        c = self.app.test_client()
        with later(minutes=61):
            c.get(link)
            self.assertIn("This link has expired", c.get("/reset-password").get_data(as_text=True))
            r = self.post(c, "/reset-password", {"password": "another password!", "confirm_password": "another password!"}, "/reset-password")
            self.assertEqual(r.status_code, 400)
        self.assertEqual(self.repo.get_credentials("links@example.com")[1], old_hash)
        self.reset(self.app.test_client(), link)
        reused = self.app.test_client()
        reused.get(link)
        self.assertIn("already been used", reused.get("/reset-password").get_data(as_text=True))
        verify_raw = auth_tokens.issue(self.repo, self.user("links@example.com")["user_id"], auth_tokens.VERIFY_EMAIL, timedelta(hours=1))
        for bad in (verify_raw, auth_tokens.new_raw_token(), "nope"):
            x = self.app.test_client()
            x.get(f"/reset-password/{bad}")
            self.assertIn("isn't valid", x.get("/reset-password").get_data(as_text=True))
        self.assertIn("isn't valid", self.app.test_client().get("/reset-password").get_data(as_text=True))

    def test_reset_password_policy_and_token_survives_validation_errors(self):
        self.new_verified_client("policy@example.com")
        self.forgot(self.app.test_client(), "policy@example.com")
        link = self.links("policy@example.com", "reset-password")[-1]
        c = self.app.test_client()
        short = self.reset(c, link, "short")
        self.assertEqual(short.status_code, 400)
        self.assertIn("Password must be 10-128 characters.", short.get_data(as_text=True))
        mismatch = self.post(c, "/reset-password", {"password": "long enough pw", "confirm_password": "different pw!"}, "/reset-password")
        self.assertIn("Passwords do not match.", mismatch.get_data(as_text=True))
        ok = self.post(c, "/reset-password", {"password": "long enough pw", "confirm_password": "long enough pw"}, "/reset-password")
        self.assertEqual(ok.headers["Location"], "/login")

    def test_new_reset_link_retires_the_old_one(self):
        self.new_verified_client("twice@example.com")
        self.forgot(self.app.test_client(), "twice@example.com")
        self.forgot(self.app.test_client(), "twice@example.com")
        first, second = self.links("twice@example.com", "reset-password")
        c = self.app.test_client()
        c.get(first)
        self.assertIn("already been used", c.get("/reset-password").get_data(as_text=True))
        self.assertEqual(self.reset(self.app.test_client(), second).headers["Location"], "/login")

    def test_reset_requests_and_attempts_are_rate_limited(self):
        self.repo.create_user("limit@example.com", "h", "client", "L")
        c = self.app.test_client()
        responses = [self.forgot(c, "limit@example.com") for _ in range(5)]
        self.assertTrue(all(r.headers["Location"] == "/forgot-password/sent" for r in responses))  # same response throughout
        self.assertEqual(len(self.links("limit@example.com", "reset-password")), 3)
        attacker = self.app.test_client()
        attacker.get(f"/reset-password/{auth_tokens.new_raw_token()}")
        statuses = [self.post(attacker, "/reset-password", {"password": "x", "confirm_password": "x"}, "/reset-password").status_code
                    for _ in range(11)]
        self.assertEqual(statuses[:10], [400] * 10)
        self.assertEqual(statuses[10], 429)

    def test_state_changing_account_forms_require_csrf(self):
        c = self.app.test_client()
        self.register(c, "csrf.acct@example.com")
        self.login(c, "csrf.acct@example.com")
        for path, data in (("/forgot-password", {"email": "csrf.acct@example.com"}), ("/reset-password", {"password": "x" * 12}),
                           ("/verify-email/confirm", {}), ("/verify-email/resend", {})):
            self.assertEqual(c.post(path, data=data).status_code, 400, path)
            self.assertEqual(c.post(path, data={**data, "csrf_token": "forged"}).status_code, 400, path)

    def test_session_records_auth_version_and_legacy_sessions_survive(self):
        c = self.new_verified_client("session@example.com")
        with c.session_transaction() as s:
            self.assertEqual(s["auth_version"], 0)
            del s["auth_version"]  # a session created before Milestone 1
        self.assertEqual(c.get("/dashboard").status_code, 200)
        self.repo._users[self.user("session@example.com")["user_id"]]["auth_version"] = 5
        self.assertEqual(c.get("/dashboard").status_code, 302)
        anonymous = self.app.test_client()
        self.assertEqual(anonymous.get("/").status_code, 200)

    def test_no_raw_tokens_passwords_or_addresses_in_logs(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        try:
            c = self.new_verified_client("logs@example.com")
            self.forgot(self.app.test_client(), "logs@example.com")
            self.reset(self.app.test_client(), self.links("logs@example.com", "reset-password")[-1], "a log-safe password")
            self.post(c, "/verify-email/resend", token_path="/verify-email")
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
        log = stream.getvalue()
        raws = [link.rsplit("/", 1)[1] for link in self.links("logs@example.com", "/")]
        self.assertTrue(raws)
        for secret in raws + [PASSWORD, "a log-safe password", "logs@example.com"]:
            self.assertNotIn(secret, log)


# ---------------------------------------------------------------------------
# Migration (opt-in, POSTGRES_TEST_DB only)
# ---------------------------------------------------------------------------
@unittest.skipUnless(RUN_PG, "opt-in live PostgreSQL test: set HIREWISE_RUN_POSTGRES_TESTS=1")
class MigrationV3Tests(unittest.TestCase):
    """Builds a schema-v2 database on POSTGRES_TEST_DB (by removing the v3 additions from a fresh schema),
    then upgrades it. Never touches POSTGRES_DB."""

    @classmethod
    def setUpClass(cls):
        from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine

        test_db, main_db = os.environ.get("POSTGRES_TEST_DB"), os.environ.get("POSTGRES_DB")
        if not test_db or test_db == main_db:
            raise unittest.SkipTest("POSTGRES_TEST_DB must be set and different from POSTGRES_DB")
        cls.settings = load_postgres_settings(TEST_DATABASE_ENV)
        assert cls.settings.database == test_db != main_db
        cls.engine = make_engine(cls.settings)

    @classmethod
    def tearDownClass(cls):
        from src.db.schema import metadata

        metadata.drop_all(cls.engine)

    def setUp(self):
        from sqlalchemy import text
        from src.db.schema import metadata

        metadata.drop_all(self.engine)
        metadata.create_all(self.engine)
        with self.engine.begin() as conn:  # schema version 2, with two existing accounts
            conn.execute(text("DROP TABLE auth_tokens"))
            conn.execute(text("DROP TABLE user_email_verification"))
            conn.execute(text("ALTER TABLE users DROP COLUMN auth_version"))
            conn.execute(text("INSERT INTO load_metadata (key, value) VALUES ('schema_version', '2')"))
            conn.execute(text("INSERT INTO users (email, password_hash, role, display_name) VALUES "
                              "('demo.client@example.com', 'h1', 'client', 'Demo C'), ('demo.worker@example.com', 'h2', 'worker', 'Demo W')"))

    def test_upgrade_grandfathers_existing_accounts_and_is_repeatable(self):
        from sqlalchemy import text
        from src.db.marketplace import PostgresMarketplaceRepository
        from src.db.migrations import migrate_to_v3, verify_v3

        report = migrate_to_v3(self.engine)
        self.assertEqual((report.version_before, report.version_after, report.added_auth_version, report.grandfathered_users),
                         ("2", "3", True, 2))
        self.assertEqual(sorted(report.created_tables), ["auth_tokens", "user_email_verification"])
        self.assertTrue(all(verify_v3(self.engine).values()), verify_v3(self.engine))
        repo = PostgresMarketplaceRepository(self.engine)
        for email, password_hash in (("demo.client@example.com", "h1"), ("demo.worker@example.com", "h2")):
            user_id, stored = repo.get_credentials(email)
            self.assertEqual(stored, password_hash)  # records preserved
            user = repo.get_user(user_id)
            self.assertIsNotNone(user["email_verified_at"])
            self.assertEqual(user["auth_version"], 0)
        new_id = repo.create_user("after.upgrade@example.com", "h3", "client", "New", verified=False)
        again = migrate_to_v3(self.engine)
        self.assertEqual((again.version_before, again.added_auth_version, again.created_tables, again.grandfathered_users),
                         ("3", False, [], 0))
        self.assertIsNone(repo.get_user(new_id)["email_verified_at"])  # a repeat run never verifies new accounts
        with self.engine.connect() as conn:
            self.assertEqual(conn.execute(text("SELECT count(*) FROM users")).scalar(), 3)

    def test_unexpected_states_are_refused_without_changes(self):
        from sqlalchemy import inspect, text
        from src.db.migrations import MigrationError, migrate_to_v3

        with self.engine.begin() as conn:
            conn.execute(text("UPDATE load_metadata SET value = '1' WHERE key = 'schema_version'"))
        with self.assertRaises(MigrationError):
            migrate_to_v3(self.engine)
        self.assertNotIn("auth_version", {c["name"] for c in inspect(self.engine).get_columns("users")})
        self.assertNotIn("auth_tokens", inspect(self.engine).get_table_names())
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE load_metadata SET value = '2' WHERE key = 'schema_version'"))
            conn.execute(text("ALTER TABLE users ADD COLUMN auth_version TEXT"))
        with self.assertRaises(MigrationError):
            migrate_to_v3(self.engine)

    def test_init_script_refuses_a_v2_database_and_migration_script_upgrades_it(self):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        init = subprocess.run([sys.executable, "scripts/init_postgres.py", "--test-db"], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(init.returncode, 1, init.stdout)
        self.assertIn("migrate_schema_v3.py --test-db", init.stdout)
        migrate = subprocess.run([sys.executable, "scripts/migrate_schema_v3.py", "--test-db"], cwd=REPO_ROOT, env=env,
                                 capture_output=True, text=True, timeout=120)
        self.assertEqual(migrate.returncode, 0, migrate.stdout)
        self.assertIn("existing accounts marked verified: 2", migrate.stdout)
        self.assertNotIn(self.settings.password, migrate.stdout + migrate.stderr)
        verify = subprocess.run([sys.executable, "scripts/migrate_schema_v3.py", "--test-db", "--verify"], cwd=REPO_ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(verify.returncode, 0, verify.stdout)


if __name__ == "__main__":
    unittest.main()
