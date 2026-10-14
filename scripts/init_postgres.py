"""Create the HireWise relational schema in PostgreSQL (Day 11).

Usage (from repository root):

    python scripts/init_postgres.py            # main database (POSTGRES_DB)
    python scripts/init_postgres.py --test-db  # isolated test database (POSTGRES_TEST_DB)

Creates any missing tables, constraints and indexes from src/db/schema.py
(CREATE ... IF NOT EXISTS semantics via SQLAlchemy `checkfirst`). It never
drops, truncates or alters existing tables. Connection settings come from
POSTGRES_* in the local .env; no credential is printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import inspect

from src.db.database import MAIN_DATABASE_ENV, TEST_DATABASE_ENV, PostgresConfigError, make_engine
from src.db.schema import metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-db", action="store_true", help="target POSTGRES_TEST_DB instead of POSTGRES_DB")
    args = parser.parse_args()
    try:
        engine = make_engine(database_env=TEST_DATABASE_ENV if args.test_db else MAIN_DATABASE_ENV)
        before = set(inspect(engine).get_table_names())
        metadata.create_all(engine, checkfirst=True)
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
