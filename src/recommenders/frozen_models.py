"""Rebuild the frozen Day 4/5/7 recommenders from their recorded
selections, never re-tuning them (Day 8+).

Selections are read from the committed result files
(results/day4/selected_cbf_config.json, results/day5/selected_cf_config.json,
results/day7/selected_kg_config.json); the KG graph snapshot is rebuilt
from the Day 6 CSV artifacts and must match the fingerprint Day 7
recorded (which Day 7 verified equal to the live Neo4j extraction).
Callers are expected to confirm each rebuilt model reproduces its frozen
metrics before using it.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.kg.graph_view import DEFAULT_KG_CSV_DIR, load_view_from_kg_csvs
from src.recommenders.collaborative import CollaborativeFiltering, build_training_matrix
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.kg_config import load_kg_config
from src.recommenders.knowledge_graph import KGComponentCache, KGIndex, KnowledgeGraphRecommender
from src.recommenders.popularity import PopularityBaseline
from src.recommenders.random_baseline import RandomBaseline

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


class FrozenSnapshotMismatch(RuntimeError):
    pass


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_frozen_models(data, seed: int, min_similarity: float, kg_csv_dir: Path = DEFAULT_KG_CSV_DIR) -> dict:
    """{'random', 'popularity', 'cbf', 'cf', 'kg'} built from frozen selections."""
    cbf_sel = _load_json(RESULTS / "day4" / "selected_cbf_config.json")
    cf_sel = _load_json(RESULTS / "day5" / "selected_cf_config.json")
    kg_sel = _load_json(RESULTS / "day7" / "selected_kg_config.json")
    snapshot = _load_json(RESULTS / "day7" / "graph_snapshot.json")
    kg_cfg = load_kg_config()

    view = load_view_from_kg_csvs(kg_csv_dir)
    if view.fingerprint() != snapshot["fingerprint"]:
        raise FrozenSnapshotMismatch(f"KG snapshot {view.fingerprint()} != Day 7 recorded {snapshot['fingerprint']}")

    popularity = PopularityBaseline(data.interactions, data.requests)
    link_to_client = dict(zip(data.requests["link"], data.requests["client_id"]))
    matrix = build_training_matrix(data.interactions, data.requests, cf_sel["event_weights"], cf_sel["aggregation"])
    return {
        "random": RandomBaseline(seed=seed),
        "popularity": popularity,
        "cbf": ContentBasedRecommender(data.workers, data.requests, cbf_sel["weights"], cbf_sel["price_neutral_score"]),
        "cf": CollaborativeFiltering(matrix, cf_sel["method"], cf_sel["k_neighbors"], popularity, link_to_client, min_similarity),
        "kg": KnowledgeGraphRecommender(
            KGIndex(view), data.requests, kg_sel["damping"], kg_sel["weights"], kg_sel["category_role_weights"],
            kg_sel["reputation_prior_strength"], kg_cfg.explanation_max_paths_per_component,
            KGComponentCache((kg_sel["damping"],)),
        ),
    }
