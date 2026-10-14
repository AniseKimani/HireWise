"""Load the HireWise application data into PostgreSQL (Day 11).

Usage (from repository root):

    python scripts/load_postgres.py                    # main database
    python scripts/load_postgres.py --test-db          # isolated test database
    python scripts/load_postgres.py --reset            # EXPLICIT: delete imported data rows, then reload

Source: the frozen, model-visible Day 2/3 artifacts (never
synthetic_private/), transformed by src/db/loader.py. One transaction:
- empty database                      -> load everything;
- identical contents already loaded   -> nothing changes;
- different contents                  -> refused unless --reset.
Run scripts/init_postgres.py first. Prints row counts and the content
fingerprint; no credential is printed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.database import MAIN_DATABASE_ENV, TEST_DATABASE_ENV, PostgresConfigError, make_engine
from src.db.loader import LoadRefused, load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-db", action="store_true", help="target POSTGRES_TEST_DB instead of POSTGRES_DB")
    parser.add_argument("--reset", action="store_true", help="delete the imported HireWise data rows before loading (destructive; refused while runtime data references them)")
    args = parser.parse_args()
    t0 = time.perf_counter()
    try:
        result = load(make_engine(database_env=TEST_DATABASE_ENV if args.test_db else MAIN_DATABASE_ENV), reset=args.reset)
    except PostgresConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except LoadRefused as exc:
        print(f"Refused: {exc}")
        return 2
    except Exception as exc:
        print(f"Load failed ({type(exc).__name__}). Did you run scripts/init_postgres.py, and is PostgreSQL running?")
        return 1
    print(f"action: {result['action']}")
    for table, count in result["row_counts"].items():
        print(f"  {table:<28} {count:>7,}")
    print(f"content fingerprint: {result['fingerprint']}")
    print(f"elapsed: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
