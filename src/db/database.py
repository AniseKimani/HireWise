"""PostgreSQL connection settings and engine creation (Day 11).

Reads `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`,
`POSTGRES_PASSWORD` (and `POSTGRES_TEST_DB` for the isolated test
database) from the environment, loaded from the git-ignored `.env` via
python-dotenv when present -- the same convention as
src/kg/neo4j_client.py. Credentials are never logged: SQLAlchemy masks the
password in URL reprs, and `hide_parameters=True` keeps bound values out of
error messages.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass  # python-dotenv is optional; variables may already be exported

MAIN_DATABASE_ENV = "POSTGRES_DB"
TEST_DATABASE_ENV = "POSTGRES_TEST_DB"
CONNECT_TIMEOUT_SECONDS = 3


class PostgresConfigError(RuntimeError):
    """Required PostgreSQL environment variables are missing."""


@dataclass(frozen=True)
class PostgresSettings:
    host: str
    port: int
    database: str
    user: str
    password: str = ""

    def url(self) -> URL:
        return URL.create("postgresql+psycopg", username=self.user, password=self.password,
                          host=self.host, port=self.port, database=self.database)


def load_postgres_settings(database_env: str = MAIN_DATABASE_ENV) -> PostgresSettings:
    names = ["POSTGRES_HOST", "POSTGRES_PORT", database_env, "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise PostgresConfigError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Add them to your local .env (see .env.example)."
        )
    return PostgresSettings(
        host=os.environ["POSTGRES_HOST"], port=int(os.environ["POSTGRES_PORT"]), database=os.environ[database_env],
        user=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"],
    )


def make_engine(settings: PostgresSettings | None = None, database_env: str = MAIN_DATABASE_ENV) -> Engine:
    """Lazy engine: no connection is attempted until first use."""
    settings = settings or load_postgres_settings(database_env)
    return create_engine(
        settings.url(),
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS, "application_name": "hirewise"},
    )
