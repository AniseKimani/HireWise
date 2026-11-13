"""Versioned, additive schema migrations (pre-deployment Milestones 1 and 2).

`metadata.create_all` (scripts/init_postgres.py) only creates MISSING tables; it never alters an existing
table and never upgrades a recorded version. Existing databases are upgraded here, through ordered steps:

  2 -> 3  (Milestone 1, account security)
          adds `users.auth_version INTEGER NOT NULL DEFAULT 0` (existing rows get 0); creates
          `user_email_verification` and `auth_tokens` with their constraints and indexes; marks every
          existing account as verified (grandfathering) -- only users that have no verification row yet, so a
          repeat run never verifies an account created after the upgrade.
  3 -> 4  (Milestone 2, Continue with Google)
          creates `user_identities` with its constraints. No existing row is touched.
  4 -> 5  (Milestone 3, notifications)
          creates `notifications` with its constraints and indexes. No existing row is touched, and no
          notification is back-filled for existing bookings.
  5 -> 6  (Milestone 4, self-reported worker details)
          adds `runtime_worker_profiles.bio TEXT NULL` with its length CHECK (existing rows get NULL) and creates
          `worker_credentials` with its constraints and index. No existing row is changed.

`migrate(engine, target)`:

1. checks the current state (the recorded schema_version and the actual structure) and refuses anything it
   does not recognise -- an unknown version, a database newer than the target, or a recorded version whose
   tables/columns are missing;
2. applies, in order, every step between the recorded version and the target;
3. records schema_version = target in load_metadata.

Everything runs in ONE transaction under an advisory lock: on any error nothing is changed. Nothing is dropped,
truncated or rewritten, so running it again on an up-to-date database only re-checks it. `verify(engine,
version)` reports the state without changing anything.

`migrate_to_v3` / `verify_v3`, `migrate_to_v4` / `verify_v4` and `migrate_to_v5` / `verify_v5` keep the earlier
upgrades available unchanged (they stop at their own version); the current application needs version 6
(`migrate_to_v6` / `verify_v6`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine

from src.db.schema import (
    ACCOUNT_SECURITY_TABLES, BIO_MAX_LENGTH, CREDENTIAL_TABLES, IDENTITY_TABLES, NOTIFICATION_TABLES, SCHEMA_VERSION,
    load_metadata, metadata,
)

TARGET_VERSION = "6"
assert SCHEMA_VERSION == TARGET_VERSION, "schema.py and the migrations disagree on the current version"
KNOWN_VERSIONS = ("2", "3", "4", "5", "6")
REQUIRED_BASE_TABLES = ("users", "load_metadata")
# Tables each version introduced (a table not listed here already existed at version 2).
TABLES_INTRODUCED = {"3": ACCOUNT_SECURITY_TABLES, "4": IDENTITY_TABLES, "5": NOTIFICATION_TABLES, "6": CREDENTIAL_TABLES}
AUTH_TOKEN_CONSTRAINTS = {"ck_auth_tokens_purpose", "ck_auth_tokens_token_hash", "ck_auth_tokens_expiry"}
AUTH_TOKEN_INDEXES = {"ix_auth_tokens_user_purpose", "ix_auth_tokens_expires_at"}
IDENTITY_CHECKS = {"ck_user_identities_provider", "ck_user_identities_subject", "ck_user_identities_email"}
IDENTITY_UNIQUES = {"uq_user_identities_provider_subject", "uq_user_identities_user_provider"}
NOTIFICATION_CHECKS = {"ck_notifications_kind", "ck_notifications_message_length", "ck_notifications_read_at"}
NOTIFICATION_UNIQUES = {"uq_notifications_booking_kind_user"}
NOTIFICATION_INDEXES = {"ix_notifications_user_read", "ix_notifications_user_created"}
CREDENTIAL_CHECKS = {"ck_worker_credentials_kind", "ck_worker_credentials_title", "ck_worker_credentials_organisation",
                     "ck_worker_credentials_year", "ck_worker_credentials_url", "ck_worker_credentials_description"}
CREDENTIAL_INDEXES = {"ix_worker_credentials_user_kind"}
BIO_CHECK = "ck_runtime_worker_bio_length"


class MigrationError(RuntimeError):
    """The database is in a state this migration does not recognise; nothing was changed."""


@dataclass
class MigrationReport:
    version_before: str | None
    version_after: str | None = None
    steps_applied: list[str] = field(default_factory=list)
    added_auth_version: bool = False
    created_tables: list[str] = field(default_factory=list)
    grandfathered_users: int = 0
    added_columns: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        return [f"schema_version before: {self.version_before}", f"schema_version after: {self.version_after}",
                f"steps applied: {', '.join(self.steps_applied) or 'none (already up to date)'}",
                f"users.auth_version added: {'yes' if self.added_auth_version else 'no (already present)'}",
                f"tables created: {', '.join(self.created_tables) or 'none (already present)'}",
                f"existing accounts marked verified: {self.grandfathered_users}",
                f"columns added: {', '.join(self.added_columns) or 'none (already present)'}"]


def _recorded_version(conn: Connection) -> str | None:
    return conn.execute(select(load_metadata.c.value).where(load_metadata.c.key == "schema_version")).scalar()


def _users_auth_version_column(conn: Connection) -> dict | None:
    return next((c for c in inspect(conn).get_columns("users") if c["name"] == "auth_version"), None)


def _auth_version_ok(column: dict | None) -> bool:
    return bool(column) and not column["nullable"] and "int" in str(column["type"]).lower() \
        and "0" in str(column["default"] or "")


def _tables_for(version: str) -> set[str]:
    """Every schema.py table that exists at `version`."""
    later = {t for v, names in TABLES_INTRODUCED.items() if int(v) > int(version) for t in names}
    return set(metadata.tables) - later


# ---------------------------------------------------------------------------
# Steps. Each is idempotent: on a database already at (or past) its version it changes nothing.
# ---------------------------------------------------------------------------
def _step_v2_to_v3(conn: Connection, report: MigrationReport) -> None:
    tables = set(inspect(conn).get_table_names())
    column = _users_auth_version_column(conn)
    if column is None:
        conn.execute(text("ALTER TABLE users ADD COLUMN auth_version INTEGER NOT NULL DEFAULT 0"))
        report.added_auth_version = True
    elif not _auth_version_ok(column):
        raise MigrationError("users.auth_version exists but is not INTEGER NOT NULL DEFAULT 0")
    security = [metadata.tables[name] for name in ACCOUNT_SECURITY_TABLES]
    report.created_tables += [t.name for t in security if t.name not in tables]
    metadata.create_all(conn, tables=security, checkfirst=True)
    report.grandfathered_users += conn.execute(text(
        "INSERT INTO user_email_verification (user_id, verified_at) "
        "SELECT u.user_id, now() FROM users u "
        "WHERE NOT EXISTS (SELECT 1 FROM user_email_verification v WHERE v.user_id = u.user_id)")).rowcount


def _step_v3_to_v4(conn: Connection, report: MigrationReport) -> None:
    tables = set(inspect(conn).get_table_names())
    identity = [metadata.tables[name] for name in IDENTITY_TABLES]
    report.created_tables += [t.name for t in identity if t.name not in tables]
    metadata.create_all(conn, tables=identity, checkfirst=True)


def _step_v4_to_v5(conn: Connection, report: MigrationReport) -> None:
    tables = set(inspect(conn).get_table_names())
    created = [metadata.tables[name] for name in NOTIFICATION_TABLES]
    report.created_tables += [t.name for t in created if t.name not in tables]
    metadata.create_all(conn, tables=created, checkfirst=True)


def _bio_column(conn: Connection) -> dict | None:
    return next((c for c in inspect(conn).get_columns("runtime_worker_profiles") if c["name"] == "bio"), None)


def _bio_ok(conn: Connection) -> bool:
    column = _bio_column(conn)
    checks = {c["name"] for c in inspect(conn).get_check_constraints("runtime_worker_profiles")}
    return bool(column) and column["nullable"] and "text" in str(column["type"]).lower() and BIO_CHECK in checks


def _step_v5_to_v6(conn: Connection, report: MigrationReport) -> None:
    if _bio_column(conn) is None:
        conn.execute(text("ALTER TABLE runtime_worker_profiles ADD COLUMN bio TEXT NULL"))
        conn.execute(text(f"ALTER TABLE runtime_worker_profiles ADD CONSTRAINT {BIO_CHECK} "
                          f"CHECK (bio IS NULL OR char_length(bio) <= {BIO_MAX_LENGTH})"))
        report.added_columns.append("runtime_worker_profiles.bio")
    elif not _bio_ok(conn):
        raise MigrationError("runtime_worker_profiles.bio exists but is not a nullable TEXT column with its length check")
    tables = set(inspect(conn).get_table_names())
    created = [metadata.tables[name] for name in CREDENTIAL_TABLES]
    report.created_tables += [t.name for t in created if t.name not in tables]
    metadata.create_all(conn, tables=created, checkfirst=True)


STEPS = (("2", "3", _step_v2_to_v3), ("3", "4", _step_v3_to_v4), ("4", "5", _step_v4_to_v5), ("5", "6", _step_v5_to_v6))


def _check_recorded_structure(conn: Connection, version: str) -> None:
    """A recorded version >= 3 must really have that version's tables/columns; otherwise refuse."""
    tables = set(inspect(conn).get_table_names())
    for v, names in TABLES_INTRODUCED.items():
        missing = [t for t in names if t not in tables]
        if int(version) >= int(v) and missing:
            raise MigrationError(f"schema_version {version} is recorded but tables {missing} are missing")
    if int(version) >= 3 and not _auth_version_ok(_users_auth_version_column(conn)):
        raise MigrationError(f"schema_version {version} is recorded but users.auth_version is missing or different")
    if int(version) >= 6 and not _bio_ok(conn):
        raise MigrationError(f"schema_version {version} is recorded but runtime_worker_profiles.bio is missing or different")


def migrate(engine: Engine, target: str = TARGET_VERSION) -> MigrationReport:
    if target not in KNOWN_VERSIONS[1:]:
        raise MigrationError(f"unsupported target version {target!r}")
    with engine.begin() as conn:
        # Serialise concurrent runs (transaction-scoped advisory lock; released at commit/rollback).
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('hirewise_schema_migration'))"))
        tables = set(inspect(conn).get_table_names())
        missing = [t for t in REQUIRED_BASE_TABLES if t not in tables]
        if missing:
            raise MigrationError(f"missing base tables {missing}: initialise the schema first (scripts/init_postgres.py)")
        version = _recorded_version(conn)
        if version not in KNOWN_VERSIONS:
            raise MigrationError(f"unexpected recorded schema_version {version!r}; expected one of {KNOWN_VERSIONS}")
        if int(version) > int(target):
            raise MigrationError(f"the database is at schema_version {version}, newer than the target {target}")
        _check_recorded_structure(conn, version)
        report = MigrationReport(version_before=version)
        for start, end, step in STEPS:
            if int(end) > int(target):
                break
            if int(start) >= int(version):  # a step the database has not had yet
                step(conn, report)
                report.steps_applied.append(f"{start}->{end}")
            elif end == version:            # the step that produced the recorded version: re-apply idempotently
                step(conn, report)
        conn.execute(pg_insert(load_metadata).values(key="schema_version", value=target)
                     .on_conflict_do_update(index_elements=["key"], set_={"value": target}))
        report.version_after = target
    return report


def migrate_to_v3(engine: Engine) -> MigrationReport:
    """Milestone 1 upgrade, 2 -> 3 only (kept for databases that must stop at version 3)."""
    return migrate(engine, target="3")


def migrate_to_v4(engine: Engine) -> MigrationReport:
    """Milestone 2 upgrade: 2 -> 3 -> 4, or 3 -> 4 (stops at version 4)."""
    return migrate(engine, target="4")


def migrate_to_v5(engine: Engine) -> MigrationReport:
    """Milestone 3 upgrade: to version 5 from 2, 3 or 4 (stops at version 5)."""
    return migrate(engine, target="5")


def migrate_to_v6(engine: Engine) -> MigrationReport:
    """Upgrade to the current version: to version 6 from 2, 3, 4 or 5."""
    return migrate(engine, target="6")


# ---------------------------------------------------------------------------
# Read-only verification
# ---------------------------------------------------------------------------
def verify(engine: Engine, version: str = TARGET_VERSION) -> dict[str, bool]:
    """Read-only checks that the database is exactly at `version` (3 to 6) with every structure it needs."""
    with engine.connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        column = _users_auth_version_column(conn) if "users" in tables else None
        checks = {
            f"schema_version is {version}": "load_metadata" in tables and _recorded_version(conn) == version,
            "users.auth_version is INTEGER NOT NULL DEFAULT 0": _auth_version_ok(column),
            "account security tables exist": set(ACCOUNT_SECURITY_TABLES) <= tables,
            f"every table of schema version {version} exists": _tables_for(version) <= tables,
        }
        if checks["account security tables exist"]:
            checks["every user has a verification row"] = conn.execute(text(
                "SELECT count(*) FROM users u WHERE NOT EXISTS "
                "(SELECT 1 FROM user_email_verification v WHERE v.user_id = u.user_id)")).scalar() == 0
            constraints = {c["name"] for c in inspector.get_check_constraints("auth_tokens")}
            checks["auth_tokens constraints present"] = AUTH_TOKEN_CONSTRAINTS <= constraints
            indexes = {i["name"] for i in inspector.get_indexes("auth_tokens")}
            checks["auth_tokens indexes present"] = AUTH_TOKEN_INDEXES <= indexes
        if int(version) >= 4:
            checks["user_identities table exists"] = set(IDENTITY_TABLES) <= tables
            if checks["user_identities table exists"]:
                ck = {c["name"] for c in inspector.get_check_constraints("user_identities")}
                uq = {c["name"] for c in inspector.get_unique_constraints("user_identities")}
                checks["user_identities constraints present"] = IDENTITY_CHECKS <= ck and IDENTITY_UNIQUES <= uq
                fks = inspector.get_foreign_keys("user_identities")
                checks["user_identities.user_id references users"] = any(
                    fk["referred_table"] == "users" and fk["constrained_columns"] == ["user_id"] for fk in fks)
        if int(version) >= 5:
            checks["notifications table exists"] = set(NOTIFICATION_TABLES) <= tables
            if checks["notifications table exists"]:
                ck = {c["name"] for c in inspector.get_check_constraints("notifications")}
                uq = {c["name"] for c in inspector.get_unique_constraints("notifications")}
                checks["notifications constraints present"] = NOTIFICATION_CHECKS <= ck and NOTIFICATION_UNIQUES <= uq
                ix = {i["name"] for i in inspector.get_indexes("notifications")}
                checks["notifications indexes present"] = NOTIFICATION_INDEXES <= ix
                fks = {(fk["referred_table"], tuple(fk["constrained_columns"]))
                       for fk in inspector.get_foreign_keys("notifications")}
                checks["notifications reference users and bookings"] = {
                    ("users", ("user_id",)), ("bookings", ("booking_id",))} <= fks
        if int(version) >= 6:
            checks["runtime_worker_profiles.bio is nullable TEXT with its length check"] = (
                "runtime_worker_profiles" in tables and _bio_ok(conn))
            checks["worker_credentials table exists"] = set(CREDENTIAL_TABLES) <= tables
            if checks["worker_credentials table exists"]:
                ck = {c["name"] for c in inspector.get_check_constraints("worker_credentials")}
                checks["worker_credentials constraints present"] = CREDENTIAL_CHECKS <= ck
                ix = {i["name"] for i in inspector.get_indexes("worker_credentials")}
                checks["worker_credentials index present"] = CREDENTIAL_INDEXES <= ix
                fks = inspector.get_foreign_keys("worker_credentials")
                checks["worker_credentials.user_id references runtime_worker_profiles"] = any(
                    fk["referred_table"] == "runtime_worker_profiles" and fk["constrained_columns"] == ["user_id"]
                    for fk in fks)
        return checks


def verify_v3(engine: Engine) -> dict[str, bool]:
    return verify(engine, "3")


def verify_v4(engine: Engine) -> dict[str, bool]:
    return verify(engine, "4")


def verify_v5(engine: Engine) -> dict[str, bool]:
    return verify(engine, "5")


def verify_v6(engine: Engine) -> dict[str, bool]:
    return verify(engine, "6")
