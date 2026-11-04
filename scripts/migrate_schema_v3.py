"""Upgrade an existing HireWise PostgreSQL database from schema version 2 to 3 (pre-deployment Milestone 1).

Usage (from repository root):

    python scripts/migrate_schema_v3.py --test-db   # isolated test database (POSTGRES_TEST_DB) -- try this first
    python scripts/migrate_schema_v3.py             # main database (POSTGRES_DB)
    python scripts/migrate_schema_v3.py --verify    # read-only check, no changes

Additive and repeatable (see src/db/migrations.py): adds users.auth_version (default 0), creates
user_email_verification and auth_tokens, marks every EXISTING account as verified, and records
schema_version 3 -- all in one transaction. Nothing is dropped or rewritten; a second run changes nothing.
Back up the database before running it on data you care about. Connection settings come from POSTGRES_* in
the local .env; no credential is printed.

This script stops at version 3. The current application (Milestone 2) needs schema version 4: use
scripts/migrate_schema_v4.py, which applies this same step and then 3 -> 4. A NEW, empty database needs
neither script: scripts/init_postgres.py creates the current version directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.database import MAIN_DATABASE_ENV, TEST_DATABASE_ENV, PostgresConfigError, make_engine
from src.db.migrations import MigrationError, migrate_to_v3, verify_v3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-db", action="store_true", help="target POSTGRES_TEST_DB instead of POSTGRES_DB")
    parser.add_argument("--verify", action="store_true", help="only check the schema (read-only)")
    args = parser.parse_args()
    target = "test database (POSTGRES_TEST_DB)" if args.test_db else "main database (POSTGRES_DB)"
    try:
        engine = make_engine(database_env=TEST_DATABASE_ENV if args.test_db else MAIN_DATABASE_ENV)
        if not args.verify:
            report = migrate_to_v3(engine)
            print(f"migrated the {target}:")
            for line in report.lines():
                print(f"  {line}")
        checks = verify_v3(engine)
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
