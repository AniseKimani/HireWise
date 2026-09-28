"""Day 6: load prepared KG data into Neo4j.

Connects using NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD/NEO4J_DATABASE
(environment variables, optionally from a local .env -- see
.env.example). Requires scripts/prepare_kg_data.py to have already been
run. Idempotent: running this twice does not duplicate nodes or
relationships (see src/kg/loader.py's docstring for how).

Usage:
    python scripts/load_neo4j.py
    python scripts/load_neo4j.py --rebuild   # DESTRUCTIVE: deletes every
                                              # node/relationship first
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.kg.neo4j_client import Neo4jClient, Neo4jConfigError
from src.kg.prepare import KGData, prepare_all
from src.kg.loader import load_all

KG_DIR = REPO_ROOT / "data" / "processed" / "kg"


def _load_kg_data_from_disk() -> KGData:
    """Prefer the already-prepared CSVs under data/processed/kg/ if
    present (so preparation and loading can be run/audited separately);
    fall back to re-preparing in memory otherwise."""
    import pandas as pd
    from src.kg.prepare import KG_FILENAMES

    if KG_DIR.exists() and all((KG_DIR / f).exists() for f in KG_FILENAMES.values()):
        kwargs = {field: pd.read_csv(KG_DIR / filename) for field, filename in KG_FILENAMES.items()}
        return KGData(**kwargs)
    print(f"[info] {KG_DIR} not found or incomplete; preparing KG data in memory instead.")
    return prepare_all()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="DESTRUCTIVE: delete every node and relationship in the target database before loading.",
    )
    args = parser.parse_args()

    if args.rebuild:
        print("!!! --rebuild requested: the target Neo4j database will be cleared (DETACH DELETE all nodes) before loading. !!!")

    t0 = time.time()
    kg_data = _load_kg_data_from_disk()

    try:
        with Neo4jClient() as client:
            print(f"Connected to Neo4j (database={client.settings.database!r}).")
            result = load_all(client, kg_data, rebuild_first=args.rebuild)
    except Neo4jConfigError as exc:
        print(f"\nConfiguration error: {exc}")
        return 1
    except ConnectionError as exc:
        print(f"\nConnection error: {exc}")
        return 1

    elapsed = time.time() - t0
    print("\n=== Neo4j load summary ===")
    print("Nodes loaded (rows submitted, MERGE may not create new ones on rerun):")
    for label, count in result["nodes"].items():
        print(f"  {label}: {count}")
    print("Relationships loaded:")
    for rel, count in result["relationships"].items():
        print(f"  {rel}: {count}")
    print(f"\nelapsed_seconds: {round(elapsed, 1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
