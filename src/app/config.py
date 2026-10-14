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
    # git-ignored .env); see src/app/factory.py for the missing-key behaviour.
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY") or None
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # Set HIREWISE_SECURE_COOKIES=1 when served over HTTPS.
    SESSION_COOKIE_SECURE = os.environ.get("HIREWISE_SECURE_COOKIES") == "1"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    LOGIN_MAX_FAILURES = 5
    LOGIN_FAILURE_WINDOW_SECONDS = 15 * 60
    # Build the marketplace (PostgreSQL repository + runtime recommender) with the service.
    LOAD_MARKETPLACE = True


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class TestingConfig(BaseConfig):
    TESTING = True
    # Tests inject their own service and Neo4j probe explicitly.
    LOAD_RECOMMENDER_ON_STARTUP = False
    NEO4J_HEALTHCHECK_ENABLED = False
    POSTGRES_HEALTHCHECK_ENABLED = False
    LOAD_MARKETPLACE = False


class ProductionConfig(BaseConfig):
    DEBUG = False


CONFIGS = {"development": DevelopmentConfig, "testing": TestingConfig, "production": ProductionConfig}


def resolve_config(name: str | None = None) -> type[BaseConfig]:
    name = name or os.environ.get("HIREWISE_ENV", "production")
    if name not in CONFIGS:
        raise ValueError(f"unknown HIREWISE_ENV {name!r}; expected one of {sorted(CONFIGS)}")
    return CONFIGS[name]
