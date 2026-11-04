"""Versioned, additive schema migrations (schema version 3, pre-deployment Milestone 1).

`metadata.create_all` (scripts/init_postgres.py) only creates MISSING tables; it never alters an existing
table. Upgrading an existing version 2 database therefore needs this explicit step, which:

1. checks the current state (the recorded schema_version and the actual users columns) and refuses
   anything it does not recognise;
2. adds `users.auth_version INTEGER NOT NULL DEFAULT 0` (existing rows get 0);
3. creates `user_email_verification` and `auth_tokens` with their constraints and indexes;
4. marks every existing account as verified (grandfathering) -- only users that have no verification
   row yet, so a repeat run never verifies an account created after the upgrade;
5. records schema_version = "3" in load_metadata.

Everything runs in ONE transaction: on any error nothing is changed. Nothing is dropped, truncated or
rewritten, so running it again on a version 3 database only re-checks it. `verify_v3` reports the state
without changing anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine

from src.db.schema import ACCOUNT_SECURITY_TABLES, SCHEMA_VERSION, load_metadata, metadata

TARGET_VERSION = "3"
UPGRADABLE_FROM = ("2", "3")
REQUIRED_BASE_TABLES = ("users", "load_metadata")


class MigrationError(RuntimeError):
    """The database is in a state this migration does not recognise; nothing was changed."""


@dataclass
class MigrationReport:
    version_before: str | None
    version_after: str | None = None
    added_auth_version: bool = False
    created_tables: list[str] = field(default_factory=list)
    grandfathered_users: int = 0

    def lines(self) -> list[str]:
        return [f"schema_version before: {self.version_before}", f"schema_version after: {self.version_after}",
                f"users.auth_version added: {'yes' if self.added_auth_version else 'no (already present)'}",
                f"tables created: {', '.join(self.created_tables) or 'none (already present)'}",
                f"existing accounts marked verified: {self.grandfathered_users}"]


def _recorded_version(conn: Connection) -> str | None:
    return conn.execute(select(load_metadata.c.value).where(load_metadata.c.key == "schema_version")).scalar()


def _users_auth_version_column(conn: Connection) -> dict | None:
    return next((c for c in inspect(conn).get_columns("users") if c["name"] == "auth_version"), None)


def migrate_to_v3(engine: Engine) -> MigrationReport:
    assert SCHEMA_VERSION == TARGET_VERSION, "schema.py and this migration disagree on the target version"
    with engine.begin() as conn:
        # Serialise concurrent runs (transaction-scoped advisory lock; released at commit/rollback).
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('hirewise_schema_migration'))"))
        tables = set(inspect(conn).get_table_names())
        missing = [t for t in REQUIRED_BASE_TABLES if t not in tables]
        if missing:
            raise MigrationError(f"missing base tables {missing}: initialise the schema first (scripts/init_postgres.py)")
        version = _recorded_version(conn)
        if version not in UPGRADABLE_FROM:
            raise MigrationError(f"unexpected recorded schema_version {version!r}; expected one of {UPGRADABLE_FROM}")
        report = MigrationReport(version_before=version)

        column = _users_auth_version_column(conn)
        if column is None:
            conn.execute(text("ALTER TABLE users ADD COLUMN auth_version INTEGER NOT NULL DEFAULT 0"))
            report.added_auth_version = True
        elif column["nullable"] or "int" not in str(column["type"]).lower() or "0" not in str(column["default"] or ""):
            raise MigrationError("users.auth_version exists but is not INTEGER NOT NULL DEFAULT 0")

        security = [metadata.tables[name] for name in ACCOUNT_SECURITY_TABLES]
        report.created_tables = [t.name for t in security if t.name not in tables]
        metadata.create_all(conn, tables=security, checkfirst=True)

        report.grandfathered_users = conn.execute(text(
            "INSERT INTO user_email_verification (user_id, verified_at) "
            "SELECT u.user_id, now() FROM users u "
            "WHERE NOT EXISTS (SELECT 1 FROM user_email_verification v WHERE v.user_id = u.user_id)")).rowcount

        conn.execute(pg_insert(load_metadata).values(key="schema_version", value=TARGET_VERSION)
                     .on_conflict_do_update(index_elements=["key"], set_={"value": TARGET_VERSION}))
        report.version_after = TARGET_VERSION
    return report


def verify_v3(engine: Engine) -> dict[str, bool]:
    """Read-only checks of a version 3 database."""
    with engine.connect() as conn:
        tables = set(inspect(conn).get_table_names())
        column = _users_auth_version_column(conn) if "users" in tables else None
        checks = {
            "schema_version is 3": "load_metadata" in tables and _recorded_version(conn) == TARGET_VERSION,
            "users.auth_version is INTEGER NOT NULL DEFAULT 0": bool(column) and not column["nullable"]
            and "int" in str(column["type"]).lower() and "0" in str(column["default"] or ""),
            "account security tables exist": set(ACCOUNT_SECURITY_TABLES) <= tables,
            "every table in schema.py exists": set(metadata.tables) <= tables,
        }
        if checks["account security tables exist"]:
            checks["every user has a verification row"] = conn.execute(text(
                "SELECT count(*) FROM users u WHERE NOT EXISTS "
                "(SELECT 1 FROM user_email_verification v WHERE v.user_id = u.user_id)")).scalar() == 0
            constraints = {c["name"] for c in inspect(conn).get_check_constraints("auth_tokens")}
            checks["auth_tokens constraints present"] = {"ck_auth_tokens_purpose", "ck_auth_tokens_token_hash",
                                                         "ck_auth_tokens_expiry"} <= constraints
            indexes = {i["name"] for i in inspect(conn).get_indexes("auth_tokens")}
            checks["auth_tokens indexes present"] = {"ix_auth_tokens_user_purpose", "ix_auth_tokens_expires_at"} <= indexes
        return checks
