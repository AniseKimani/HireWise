"""Upgrade an existing HireWise PostgreSQL database to schema version 5 (pre-deployment Milestone 3).

Usage (from repository root):

    python scripts/migrate_schema_v5.py --test-db   # isolated test database (POSTGRES_TEST_DB) -- try this first
    python scripts/migrate_schema_v5.py             # main database (POSTGRES_DB)
    python scripts/migrate_schema_v5.py --verify    # read-only check, no changes (add --test-db for the test database)

Applies the ordered, additive steps in src/db/migrations.py: from version 4 it creates `notifications` (in-app
booking and review notifications); from version 3 or 2 it first applies the earlier steps (Milestone 1 account
security, Milestone 2 user_identities). Everything runs in one transaction and records schema_version 5. Nothing
is dropped or rewritten, no imported research table is touched, and no notification is back-filled for existing
bookings; a second run changes nothing. Back up the database before running it on data you care about.
Connection settings come from POSTGRES_* in the local .env; no credential is printed.

A NEW, empty database does not need this script: scripts/init_postgres.py creates schema version 5 directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.database import MAIN_DATABASE_ENV, TEST_DATABASE_ENV, PostgresConfigError, make_engine
from src.db.migrations import MigrationError, migrate_to_v5, verify_v5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-db", action="store_true", help="target POSTGRES_TEST_DB instead of POSTGRES_DB")
    parser.add_argument("--verify", action="store_true", help="only check the schema (read-only)")
    args = parser.parse_args()
    target = "test database (POSTGRES_TEST_DB)" if args.test_db else "main database (POSTGRES_DB)"
    try:
        engine = make_engine(database_env=TEST_DATABASE_ENV if args.test_db else MAIN_DATABASE_ENV)
        if not args.verify:
            report = migrate_to_v5(engine)
            print(f"migrated the {target}:")
            for line in report.lines():
                print(f"  {line}")
        checks = verify_v5(engine)
    except PostgresConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except MigrationError as exc:
        print(f"Migration refused, nothing was changed: {exc}")
        return 1
    except Exception as exc:  # connection problems: report the type only, never connection details
        print(f"Could not migrate the {target} ({type(exc).__name__}). Nothing was changed. Is PostgreSQL running?")
        return 1
    print(f"verification of the {target}:")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
