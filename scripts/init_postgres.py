"""Create the HireWise relational schema in PostgreSQL (Day 11; schema v6 since Milestone 4).

Usage (from repository root):

    python scripts/init_postgres.py            # main database (POSTGRES_DB)
    python scripts/init_postgres.py --test-db  # isolated test database (POSTGRES_TEST_DB)

Creates any missing tables, constraints and indexes from src/db/schema.py
(CREATE ... IF NOT EXISTS semantics via SQLAlchemy `checkfirst`). It never
drops, truncates or alters existing tables; it records the schema version
in load_metadata. Connection settings come from
POSTGRES_* in the local .env; no credential is printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import inspect, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.database import MAIN_DATABASE_ENV, TEST_DATABASE_ENV, PostgresConfigError, make_engine
from src.db.schema import SCHEMA_VERSION, load_metadata, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-db", action="store_true", help="target POSTGRES_TEST_DB instead of POSTGRES_DB")
    args = parser.parse_args()
    try:
        engine = make_engine(database_env=TEST_DATABASE_ENV if args.test_db else MAIN_DATABASE_ENV)
        before = set(inspect(engine).get_table_names())
        # An existing database at an older schema version is upgraded only by the explicit, versioned
        # migration (create_all never alters tables): refuse rather than record the current version over it.
        recorded = None
        if "load_metadata" in before:
            with engine.connect() as conn:
                recorded = conn.execute(select(load_metadata.c.value)
                                        .where(load_metadata.c.key == "schema_version")).scalar()
        legacy_users = "users" in before and "auth_version" not in {c["name"] for c in inspect(engine).get_columns("users")}
        if legacy_users or (recorded is not None and recorded != SCHEMA_VERSION):
            found = "2 (users.auth_version is missing)" if legacy_users else recorded
            print(f"Existing schema version {found} detected; this code needs version {SCHEMA_VERSION}. "
                  "Nothing was changed.\n"
                  "Upgrade it with: python scripts/migrate_schema_v6.py" + (" --test-db" if args.test_db else ""))
            return 1
        metadata.create_all(engine, checkfirst=True)
        with engine.begin() as conn:  # record the schema version (upsert; never drops or alters)
            conn.execute(pg_insert(load_metadata).values(key="schema_version", value=SCHEMA_VERSION)
                         .on_conflict_do_update(index_elements=["key"], set_={"value": SCHEMA_VERSION}))
        after = set(inspect(engine).get_table_names())
    except PostgresConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except Exception as exc:  # connection problems: report the type only, never connection details
        print(f"Could not initialise the schema ({type(exc).__name__}). Is PostgreSQL running and .env correct?")
        return 1
    expected = set(metadata.tables)
    print(f"created: {sorted((after - before) & expected) or 'none'}")
    print(f"already present: {sorted(before & expected) or 'none'}")
    missing = expected - after
    print("schema ready." if not missing else f"MISSING tables: {sorted(missing)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
