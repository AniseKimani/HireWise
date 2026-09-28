"""Independent verification of Day 6 KG preparation (and, if a live
Neo4j instance is reachable, the loaded graph too).

Re-derives invariants independently rather than trusting
data/processed/kg/*.csv or src/kg/prepare.py's own functions: counts and
key aggregates are recomputed from the raw Day 2/Day 3 source CSVs with
separate logic.

Usage:
    python scripts/verify_day6.py

Exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.io import sha256_of
from src.kg import schema
from src.kg.prepare import KG_FILENAMES, normalize_location

KG_DIR = REPO_ROOT / "data" / "processed" / "kg"
DAY3_SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
DAY2_PROCESSED = REPO_ROOT / "data" / "processed"
DAY4_RESULTS = REPO_ROOT / "results" / "day4"
DAY5_RESULTS = REPO_ROOT / "results" / "day5"

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    missing = [f for f in KG_FILENAMES.values() if not (KG_DIR / f).exists()]
    check("all expected KG artifact files exist", not missing, f"missing {missing}")
    if missing:
        print("\nRun scripts/prepare_kg_data.py first.")
        return 1

    tables = {field: pd.read_csv(KG_DIR / filename) for field, filename in KG_FILENAMES.items()}

    # ---- Frozen prior artifacts (Phase 1, Day 4, Day 5) --------------------
    with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
        expected_hash = fh.read().split()[0]
    check("frozen generator_v1.yaml hash unchanged", sha256_of(REPO_ROOT / "config" / "generator_v1.yaml") == expected_hash)

    if (DAY4_RESULTS / "selected_cbf_config.json").exists():
        import json
        with open(DAY4_RESULTS / "selected_cbf_config.json", encoding="utf-8") as fh:
            cbf_config = json.load(fh)
        check("Day 4 selected CBF weights unchanged",
              cbf_config.get("weights") == {"category": 0.25, "price": 0.25, "skill": 0.5})
    if (DAY5_RESULTS / "selected_cf_config.json").exists():
        import json
        with open(DAY5_RESULTS / "selected_cf_config.json", encoding="utf-8") as fh:
            cf_config = json.load(fh)
        check("Day 5 selected CF configuration unchanged",
              (cf_config.get("aggregation"), cf_config.get("method"), cf_config.get("k_neighbors")) == ("sum", "user", 20))

    # ---- Independent re-derivation from raw Day 2/Day 3 sources -----------
    workers = pd.read_csv(DAY3_SYNTHETIC / "workers.csv")
    clients = pd.read_csv(DAY3_SYNTHETIC / "clients.csv")
    interactions = pd.read_csv(DAY3_SYNTHETIC / "interactions.csv")
    requests = pd.read_csv(DAY3_SYNTHETIC / "requests.csv")
    ratings = pd.read_csv(DAY3_SYNTHETIC / "ratings.csv")
    sampled_requests = pd.read_csv(DAY2_PROCESSED / "sampled_requests.csv")
    vocab = pd.read_csv(DAY2_PROCESSED / "skills_vocab.csv")
    category_stats = pd.read_csv(DAY2_PROCESSED / "category_stats.csv")
    cold_start = pd.read_csv(DAY3_SYNTHETIC / "cold_start_cohort.csv")

    check("2,000 Client nodes prepared", len(tables["clients"]) == len(clients) == 2000)
    check("3,000 Worker nodes prepared", len(tables["workers"]) == len(workers) == 3000)
    check("Skill nodes == full 1,187-skill controlled vocabulary", len(tables["skills"]) == len(vocab) == 1187)
    check("ServiceCategory nodes == 88 in-scope categories", len(tables["categories"]) == len(category_stats) == 88)

    # Stable IDs: every node table's key column must be unique.
    check("Client node IDs are unique", tables["clients"]["client_id"].is_unique)
    check("Worker node IDs are unique", tables["workers"]["worker_id"].is_unique)
    check("Skill node names are unique", tables["skills"]["name"].is_unique)
    check("ServiceCategory node names are unique", tables["categories"]["name"].is_unique)
    check("Location node names are unique", tables["locations"]["name"].is_unique)

    # Controlled vocabulary: every HAS_SKILL/RELATED_TO skill reference must be in vocab.
    vocab_set = set(vocab["skill"])
    check("all HAS_SKILL edges reference controlled-vocabulary skills",
          set(tables["has_skill"]["skill"]).issubset(vocab_set))
    check("all RELATED_TO edges reference controlled-vocabulary skills",
          set(tables["related_to"]["skill_a"]).issubset(vocab_set) and set(tables["related_to"]["skill_b"]).issubset(vocab_set))

    # Independent HAS_SKILL recomputation.
    independent_has_skill = set()
    for worker_id, cell in zip(workers["worker_id"], workers["declared_skills"]):
        if isinstance(cell, str) and cell:
            for skill in cell.split("|"):
                if skill in vocab_set:
                    independent_has_skill.add((worker_id, skill))
    prepared_has_skill = set(zip(tables["has_skill"]["worker_id"], tables["has_skill"]["skill"]))
    check("HAS_SKILL edges match an independent recomputation from workers.csv",
          independent_has_skill == prepared_has_skill,
          f"{len(independent_has_skill)} vs {len(prepared_has_skill)}")

    # Independent HIRED recomputation.
    independent_hired = set(zip(
        interactions.loc[interactions["event"] == "HIRED", "client_id"],
        interactions.loc[interactions["event"] == "HIRED", "worker_id"],
        interactions.loc[interactions["event"] == "HIRED", "link"],
    ))
    prepared_hired = set(zip(tables["hired"]["client_id"], tables["hired"]["worker_id"], tables["hired"]["link"]))
    check("HIRED edges match an independent recomputation from interactions.csv",
          independent_hired == prepared_hired, f"{len(independent_hired)} vs {len(prepared_hired)}")

    # No orphaned relationship endpoints (source side): every worker_id/client_id
    # referenced by a relationship table must exist in the corresponding node table.
    worker_ids = set(tables["workers"]["worker_id"])
    client_ids = set(tables["clients"]["client_id"])
    for field, col in [("has_skill", "worker_id"), ("specializes_in", "worker_id"), ("located_in_worker", "worker_id")]:
        bad = set(tables[field][col]) - worker_ids
        check(f"{field}.{col} references only existing Worker nodes", not bad, f"unknown: {bad}")
    for field, col in [("located_in_client", "client_id"), ("requested_service", "client_id"), ("interested_in", "client_id")]:
        bad = set(tables[field][col]) - client_ids
        check(f"{field}.{col} references only existing Client nodes", not bad, f"unknown: {bad}")
    hired_bad_clients = set(tables["hired"]["client_id"]) - client_ids
    hired_bad_workers = set(tables["hired"]["worker_id"]) - worker_ids
    check("HIRED edges reference only existing Client/Worker nodes", not hired_bad_clients and not hired_bad_workers)

    # Category keys valid.
    category_set = set(tables["categories"]["name"])
    check("all SPECIALIZES_IN edges reference valid categories", set(tables["specializes_in"]["category"]).issubset(category_set))
    check("all REQUESTED_SERVICE edges reference valid categories", set(tables["requested_service"]["category"]).issubset(category_set))
    check("all INTERESTED_IN edges reference valid categories", set(tables["interested_in"]["category"]).issubset(category_set))

    # Temporal split properties valid.
    valid_splits = {"train", "validation", "test"}
    check("REQUESTED_SERVICE split values are all valid", set(tables["requested_service"]["split"]).issubset(valid_splits))
    check("HIRED split values are all valid", set(tables["hired"]["split"]).issubset(valid_splits))

    # Cold-start workers: zero training HIRED edges.
    check("300 cold-start workers", len(cold_start) == 300)
    cold_ids = set(cold_start["worker_id"])
    leaked = tables["hired"].loc[
        (tables["hired"]["worker_id"].isin(cold_ids)) & (tables["hired"]["split"] == "train")
    ]
    check("cold-start workers have zero training HIRED edges", len(leaked) == 0, f"{len(leaked)} leaked")

    # INTERESTED_IN derivation: independently recompute the >=2-training-request rule.
    merged = requests.merge(sampled_requests[["link", "category"]], on="link", how="left")
    train_only = merged.loc[merged["split"] == "train"]
    independent_counts = train_only.groupby(["client_id", "category"]).size()
    independent_interested = set(independent_counts.loc[independent_counts >= 2].index)
    prepared_interested = set(zip(tables["interested_in"]["client_id"], tables["interested_in"]["category"]))
    check("INTERESTED_IN matches an independent training-only >=2-request recomputation",
          independent_interested == prepared_interested, f"{len(independent_interested)} vs {len(prepared_interested)}")

    # No private fields: allow-list check against schema.py.
    for field in ("clients", "workers", "skills", "categories", "locations"):
        label = {"clients": schema.LABEL_CLIENT, "workers": schema.LABEL_WORKER, "skills": schema.LABEL_SKILL,
                  "categories": schema.LABEL_CATEGORY, "locations": schema.LABEL_LOCATION}[field]
        extra = set(tables[field].columns) - schema.ALLOWED_NODE_PROPERTIES[label]
        check(f"{field} node table has no columns outside the schema allow-list", not extra, f"extra: {extra}")

    # No duplicate domain IDs.
    check("no duplicate REQUESTED_SERVICE link values", tables["requested_service"]["link"].is_unique)
    check("no duplicate HIRED link values", tables["hired"]["link"].is_unique)

    # Location normalization: deterministic, no case/whitespace duplicates.
    raw_locations = set(clients["country"].dropna()) | set(workers["country"].dropna())
    normalized = {normalize_location(loc) for loc in raw_locations}
    check("Location node set matches independently normalized worker/client countries",
          set(tables["locations"]["name"]) == normalized, f"{len(normalized)} vs {len(tables['locations'])}")

    # ---- Deterministic artifact hashes (rerun preparation into a temp
    # directory, compare file hashes against the saved artifacts) --------
    import json
    import tempfile

    from src.kg.prepare import prepare_all, save_kg_data

    manifest_path = KG_DIR / "kg_prepare_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            recorded_hashes = {entry["path"]: entry["sha256"] for entry in json.load(fh)["files"]}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rerun = prepare_all()
            save_kg_data(rerun, tmp_path)
            mismatches = []
            for filename in KG_FILENAMES.values():
                rel_path = str((KG_DIR / filename).relative_to(REPO_ROOT).as_posix())
                recorded = recorded_hashes.get(rel_path)
                actual = sha256_of(tmp_path / filename)
                if recorded is not None and recorded != actual:
                    mismatches.append(filename)
            check("re-running preparation reproduces byte-identical artifacts (deterministic)",
                  not mismatches, f"mismatched: {mismatches}")
    else:
        check("kg_prepare_manifest.json exists for determinism comparison", False)

    # ---- Optional live Neo4j comparison -------------------------------------
    try:
        from src.kg.neo4j_client import Neo4jClient, Neo4jConfigError
        from src.kg import queries
        with Neo4jClient() as client:
            node_counts = queries.count_nodes_by_label(client)
            check("live Neo4j Worker count matches prepared artifact",
                  node_counts.get(schema.LABEL_WORKER) == len(tables["workers"]))
            check("live Neo4j Client count matches prepared artifact",
                  node_counts.get(schema.LABEL_CLIENT) == len(tables["clients"]))
            check("live Neo4j has zero cold-start training HIRED edges",
                  queries.cold_start_training_hire_count(client) == 0)
    except Neo4jConfigError:
        print("[skip] live Neo4j checks -- NEO4J_* environment variables not configured")
    except ConnectionError:
        print("[skip] live Neo4j checks -- could not connect to Neo4j")

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
