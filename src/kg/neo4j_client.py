"""Thin wrapper around the official Neo4j Python driver.

Reads connection settings from environment variables
(`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`), loaded
via `python-dotenv` if a `.env` file is present (never committed --
see `.env.example` for the placeholder format). No credential is ever
logged or included in an exception message.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from neo4j import GraphDatabase, Driver

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars may already be exported

DEFAULT_DATABASE = "neo4j"
DEFAULT_BATCH_SIZE = 500


class Neo4jConfigError(RuntimeError):
    """Raised when required connection environment variables are missing."""


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str = DEFAULT_DATABASE


def load_settings_from_env() -> Neo4jSettings:
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE", DEFAULT_DATABASE)

    missing = [name for name, value in [("NEO4J_URI", uri), ("NEO4J_USER", user), ("NEO4J_PASSWORD", password)] if not value]
    if missing:
        raise Neo4jConfigError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your local Neo4j credentials."
        )
    return Neo4jSettings(uri=uri, user=user, password=password, database=database)


class Neo4jClient:
    """Context-manager wrapper: `with Neo4jClient() as client: ...`."""

    def __init__(self, settings: Neo4jSettings | None = None):
        self.settings = settings or load_settings_from_env()
        self._driver: Driver | None = None

    def __enter__(self) -> "Neo4jClient":
        self._driver = GraphDatabase.driver(self.settings.uri, auth=(self.settings.user, self.settings.password))
        try:
            self._driver.verify_connectivity()
        except Exception as exc:
            self._driver.close()
            self._driver = None
            raise ConnectionError(
                f"Could not connect to Neo4j at the configured URI (database={self.settings.database!r}). "
                f"Is the server running and are NEO4J_* credentials correct? Original error: {exc}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Run a single query, return all records as plain dicts."""
        if self._driver is None:
            raise RuntimeError("Neo4jClient must be used as a context manager: `with Neo4jClient() as client:`")
        with self._driver.session(database=self.settings.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_write(self, query: str, parameters: dict[str, Any] | None = None) -> None:
        """Run a single write query in a managed transaction."""
        if self._driver is None:
            raise RuntimeError("Neo4jClient must be used as a context manager: `with Neo4jClient() as client:`")
        with self._driver.session(database=self.settings.database) as session:
            session.execute_write(lambda tx: tx.run(query, parameters or {}).consume())

    def run_batched(self, query: str, rows: Iterable[dict], batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """Run `query` (expected to reference `$rows` via an `UNWIND
        $rows AS row` clause) in batches of `batch_size`, each batch a
        separate managed write transaction. Returns the total row count
        submitted. Parameters are always passed as bound values (never
        string-interpolated into the query text)."""
        if self._driver is None:
            raise RuntimeError("Neo4jClient must be used as a context manager: `with Neo4jClient() as client:`")
        rows = list(rows)
        total = 0
        with self._driver.session(database=self.settings.database) as session:
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                session.execute_write(lambda tx, batch=batch: tx.run(query, {"rows": batch}).consume())
                total += len(batch)
        return total
