"""Flask configuration classes (Day 9).

Selected by `HIREWISE_ENV` (development | testing | production); the
default is **production**, so debug mode is never on unless explicitly
requested. Neo4j credentials are NOT configured here: they stay in the
existing `NEO4J_*` environment variables / git-ignored `.env`, read by
src/kg/neo4j_client.py.
"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass  # python-dotenv is optional; variables may already be exported


class BaseConfig:
    DEBUG = False
    TESTING = False
    # Recommendation output size. The project's evaluated output is Top-10;
    # MAX_K bounds the response size for any single call.
    DEFAULT_K = 10
    MAX_K = 50
    # Longest accepted request link (the Day 2 links are ~60-160 chars).
    MAX_LINK_LENGTH = 2048
    # Request browsing (Day 10): page size and filter-text limits.
    REQUESTS_PER_PAGE = 20
    MAX_REQUESTS_PER_PAGE = 50
    MAX_FILTER_LENGTH = 100
    # Build the frozen recommender stack once, at application start-up.
    LOAD_RECOMMENDER_ON_STARTUP = True
    # Probe Neo4j (read-only) from /api/health.
    NEO4J_HEALTHCHECK_ENABLED = True
    # Probe the application database (PostgreSQL) from /api/health.
    POSTGRES_HEALTHCHECK_ENABLED = True
    # Strongest KG paths attached to each recommendation.
    KG_EVIDENCE_PATHS = 3
    JSON_SORT_KEYS = True

    # ---- Marketplace sessions (Day 12) --------------------------------------
    # The signing key comes ONLY from the environment (FLASK_SECRET_KEY in the
    # git-ignored .env), read when the app is created (src/app/factory.py).
    # When REQUIRE_SECRET_KEY is true (production, the default), a missing key
    # stops start-up; development/testing may fall back to a temporary key.
    SECRET_KEY = None
    REQUIRE_SECRET_KEY = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # Set HIREWISE_SECURE_COOKIES=1 when served over HTTPS.
    SESSION_COOKIE_SECURE = os.environ.get("HIREWISE_SECURE_COOKIES") == "1"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    LOGIN_MAX_FAILURES = 5
    LOGIN_FAILURE_WINDOW_SECONDS = 15 * 60
    # Build the marketplace (PostgreSQL repository + runtime recommender) with the service.
    LOAD_MARKETPLACE = True

    # ---- Email and account security (pre-deployment Milestone 1) ----------------------------
    # Delivery settings come from the environment / git-ignored .env (src/app/marketplace/mailer.py):
    # MAIL_BACKEND (smtp | console | file), SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
    # SMTP_USE_TLS (1 = STARTTLS, ssl = implicit TLS), MAIL_FROM, APP_BASE_URL, MAIL_OUTBOX_DIR.
    # A value set here (e.g. by a config override) wins over the environment.
    MAIL_BACKEND = None
    APP_BASE_URL = None
    DEFAULT_MAIL_BACKEND = None          # production: must be configured explicitly
    DEFAULT_APP_BASE_URL = None
    MAIL_TIMEOUT_SECONDS = 10            # SMTP connection / read timeout
    MAIL_SEND_ASYNC = True               # deliver after the response, so timing never depends on delivery
    EMAIL_VERIFICATION_TTL = timedelta(hours=24)
    PASSWORD_RESET_TTL = timedelta(hours=1)
    # Rate limits: (events, window seconds), per key -- in-process, like the login limiter.
    VERIFY_RESEND_LIMIT = (3, 15 * 60)            # per account
    RESET_REQUEST_LIMIT = (3, 15 * 60)            # per (client address, email)
    RESET_REQUEST_ADDRESS_LIMIT = (10, 15 * 60)   # per client address
    RESET_ATTEMPT_LIMIT = (10, 15 * 60)           # new-password submissions per client address
    # Number of reverse proxies in front of the app whose X-Forwarded-* headers are trusted (werkzeug ProxyFix).
    # 0 (default) trusts none. Set HIREWISE_TRUSTED_PROXIES=1 only when exactly one proxy you control sets them.
    TRUSTED_PROXY_COUNT = int(os.environ.get("HIREWISE_TRUSTED_PROXIES", "0") or 0)

    # ---- Continue with Google (pre-deployment Milestone 2) ----------------------------------
    # OAuth 2.0 / OpenID Connect client credentials come from the environment / git-ignored .env
    # (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET). Google sign-in is optional: with neither set the button is
    # hidden; setting exactly one stops start-up. A value set here (e.g. by a config override) wins.
    GOOGLE_CLIENT_ID = None
    GOOGLE_CLIENT_SECRET = None
    GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
    # Google ID tokens carry either form of the issuer (OpenID Connect discovery names the https form).
    GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
    GOOGLE_HTTP_TIMEOUT_SECONDS = 10     # discovery, token exchange and key download
    # A new Google user chooses a role before the account exists; the verified Google identity waits in the
    # signed session for at most this long.
    PENDING_GOOGLE_SIGNUP_TTL = timedelta(minutes=15)
    GOOGLE_CALLBACK_LIMIT = (20, 15 * 60)         # callback requests per client address


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    REQUIRE_SECRET_KEY = False  # a temporary key is acceptable locally (logins end on restart)
    DEFAULT_MAIL_BACKEND = "console"
    DEFAULT_APP_BASE_URL = "http://127.0.0.1:5000"


class TestingConfig(BaseConfig):
    TESTING = True
    # Tests inject their own service and Neo4j probe explicitly.
    LOAD_RECOMMENDER_ON_STARTUP = False
    NEO4J_HEALTHCHECK_ENABLED = False
    POSTGRES_HEALTHCHECK_ENABLED = False
    LOAD_MARKETPLACE = False
    REQUIRE_SECRET_KEY = False
    # Never send real mail from tests, whatever the environment says: messages are captured in memory.
    MAIL_BACKEND = "memory"
    APP_BASE_URL = "http://localhost"
    MAIL_SEND_ASYNC = False
    # Google sign-in is off in tests unless a test configures it explicitly (never from the environment).
    GOOGLE_CLIENT_ID = ""
    GOOGLE_CLIENT_SECRET = ""


class ProductionConfig(BaseConfig):
    DEBUG = False
    REQUIRE_SECRET_KEY = True  # start-up fails without FLASK_SECRET_KEY


CONFIGS = {"development": DevelopmentConfig, "testing": TestingConfig, "production": ProductionConfig}


def resolve_config(name: str | None = None) -> type[BaseConfig]:
    name = name or os.environ.get("HIREWISE_ENV", "production")
    if name not in CONFIGS:
        raise ValueError(f"unknown HIREWISE_ENV {name!r}; expected one of {sorted(CONFIGS)}")
    return CONFIGS[name]
