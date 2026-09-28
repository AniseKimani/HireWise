"""Day 6: prepare deterministic, graph-ready node/relationship tables.

No Neo4j connection required. Reads Day 2 (`data/processed/*.csv`) and
Day 3 (`data/processed/synthetic/*.csv`) model-visible outputs; never
`data/processed/synthetic_private/`. Writes CSV tables to
`data/processed/kg/` (git-ignored, like the rest of `data/processed/`).

Usage:
    python scripts/prepare_kg_data.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.io import build_manifest, save_json, save_manifest, sha256_of
from src.kg.prepare import KG_FILENAMES, prepare_all, save_kg_data

OUT_DIR = REPO_ROOT / "data" / "processed" / "kg"


def main() -> None:
    t0 = time.time()
    print("[1/2] Preparing deterministic KG node/relationship tables")
    kg_data = prepare_all()

    print("[2/2] Saving to data/processed/kg/")
    output_paths = save_kg_data(kg_data, OUT_DIR)

    manifest = build_manifest(REPO_ROOT, output_paths)
    manifest["generator_v1_sha256"] = sha256_of(REPO_ROOT / "config" / "generator_v1.yaml")
    save_manifest(manifest, OUT_DIR / "kg_prepare_manifest.json")

    summary = {
        "n_client_nodes": len(kg_data.clients), "n_worker_nodes": len(kg_data.workers),
        "n_skill_nodes": len(kg_data.skills), "n_category_nodes": len(kg_data.categories),
        "n_location_nodes": len(kg_data.locations),
        "n_has_skill": len(kg_data.has_skill), "n_specializes_in": len(kg_data.specializes_in),
        "n_located_in_worker": len(kg_data.located_in_worker), "n_located_in_client": len(kg_data.located_in_client),
        "n_requested_service": len(kg_data.requested_service), "n_interested_in": len(kg_data.interested_in),
        "n_hired": len(kg_data.hired), "n_reviewed": len(kg_data.reviewed), "n_related_to": len(kg_data.related_to),
    }
    save_json(summary, OUT_DIR / "kg_prepare_summary.json")

    elapsed = time.time() - t0
    print("\n=== KG preparation summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nelapsed_seconds: {round(elapsed, 1)}")
    print(f"Output files listed in {sorted(KG_FILENAMES.values())}")
    print(f"Written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
