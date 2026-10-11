"""Day 7 Knowledge Graph-only recommender evaluation pipeline.

Usage (from repository root):

    python scripts/run_day7_evaluation.py                     # graph snapshot from live Neo4j (read-only)
    python scripts/run_day7_evaluation.py --graph-source csv  # snapshot from Day 6 CSV artifacts

With the default `neo4j` source, the training-only graph snapshot is
extracted from the live Day 6 graph using read-only Cypher, and ALSO
built from the deterministic Day 6 CSV artifacts; the run aborts unless
the two are identical table by table. Neo4j is never written to.

Reuses the Day 4/5 evaluation protocol (config/evaluation_config.yaml,
src/evaluation/protocol.py, src/evaluation/evaluator.py) unchanged.
Random/Popularity/CBF/CF rows in the comparison tables are copied
verbatim from results/day5/*.csv, never recomputed into those tables.
The frozen Day 4/5 models are rebuilt from their selected configurations
only for a cold-start-worker exposure diagnostic, after first asserting
that each rebuilt model reproduces its frozen test NDCG@10 exactly.

Tuning is validation-only over the grid predeclared in
config/kg_config.yaml; test metrics are computed once, for the selected
configuration only. Writes results to `results/day7/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.config import load_config
from src.data.io import build_manifest, save_csv, save_json, save_manifest, sha256_of
from src.evaluation.config import load_evaluation_config
from src.evaluation.evaluator import cold_start_requests, evaluate
from src.evaluation.explainability import component_contribution_summary, deterministic_sample, explainability_summary
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.kg.explanations import WORKER_HISTORY_PATH_TYPES
from src.kg.graph_view import compare_views, load_view_from_kg_csvs, load_view_from_neo4j
from src.recommenders.cf_config import load_cf_config
from src.recommenders.collaborative import CollaborativeFiltering, build_training_matrix
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.kg_config import load_kg_config
from src.recommenders.knowledge_graph import COMPONENTS, KGComponentCache, KGIndex, KnowledgeGraphRecommender
from src.recommenders.popularity import PopularityBaseline
from src.recommenders.random_baseline import RandomBaseline

RESULTS_DIR = REPO_ROOT / "results" / "day7"
DAY4_RESULTS_DIR = REPO_ROOT / "results" / "day4"
DAY5_RESULTS_DIR = REPO_ROOT / "results" / "day5"
DAY3_SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
KG_CSV_DIR = REPO_ROOT / "data" / "processed" / "kg"
COLD_START_PATH = DAY3_SYNTHETIC / "cold_start_cohort.csv"

METRIC_KEY = {
    "ndcg_at_10": "ndcg_at_k", "precision_at_10": "precision_at_k",
    "recall_at_10": "recall_at_k", "f1_at_10": "f1_at_k",
}
FROZEN_INPUTS = [
    "config/generator_v1.yaml",
    "config/evaluation_config.yaml",
    "config/cf_config.yaml",
    "results/day4/selected_cbf_config.json",
    "results/day4/validation_metrics.csv",
    "results/day4/test_metrics.csv",
    "results/day4/cold_start_metrics.csv",
    "results/day5/selected_cf_config.json",
    "results/day5/validation_metrics.csv",
    "results/day5/test_metrics.csv",
    "results/day5/cold_start_metrics.csv",
    "data/processed/kg/kg_prepare_manifest.json",
]


# ---------------------------------------------------------------------------
# Graph snapshot
# ---------------------------------------------------------------------------

def load_graph_snapshot(graph_source: str):
    csv_view = load_view_from_kg_csvs(KG_CSV_DIR)
    if graph_source == "csv":
        return csv_view, {"graph_source": "csv", "neo4j_csv_parity": None, "neo4j_csv_table_parity": None}

    from src.kg.neo4j_client import Neo4jClient, Neo4jConfigError

    try:
        with Neo4jClient() as client:
            neo4j_view = load_view_from_neo4j(client)
    except (Neo4jConfigError, ConnectionError) as exc:
        raise SystemExit(f"Neo4j unavailable ({exc}). Start Neo4j, or rerun with --graph-source csv.")

    table_parity = compare_views(neo4j_view, csv_view)
    parity = neo4j_view.fingerprint() == csv_view.fingerprint() and all(table_parity.values())
    if not parity:
        raise SystemExit(f"Neo4j-derived and CSV-derived training graph views differ: {table_parity}")
    return neo4j_view, {"graph_source": "neo4j", "neo4j_csv_parity": parity, "neo4j_csv_table_parity": table_parity}


# ---------------------------------------------------------------------------
# Tuning, ablation, component analysis (validation only)
# ---------------------------------------------------------------------------

def tune_kg(base: KnowledgeGraphRecommender, kg_cfg, val_requests, k: int) -> tuple[dict, pd.DataFrame]:
    """Every (damping, weight vector) in the predeclared grid, on
    VALIDATION only. Selection: primary metric (NDCG@10) descending, then
    the tiebreak metric (Recall@10), then the simpler configuration --
    smaller damping first, then earlier weight-grid entry (the grid is
    listed simplest-first). Never looks at test."""
    primary_key = METRIC_KEY[kg_cfg.tuning_primary_metric]
    tiebreak_key = METRIC_KEY[kg_cfg.tuning_tiebreak_metric]
    rows, best, best_key = [], None, None
    for damping in kg_cfg.dwpc_damping_grid:
        for grid_index, weights in enumerate(kg_cfg.weight_grid):
            rec = base.with_config(damping, weights)
            _, summary = evaluate(rec, val_requests, k)
            rows.append({
                "damping": damping, "weight_grid_index": grid_index,
                **{f"w_{c}": weights[c] for c in COMPONENTS}, **summary, "selected": False,
            })
            key = (summary[primary_key], summary[tiebreak_key], -damping, -grid_index)
            if best_key is None or key > best_key:
                best_key, best = key, {"damping": damping, "weight_grid_index": grid_index, "weights": dict(weights)}
    for row in rows:
        row["selected"] = row["damping"] == best["damping"] and row["weight_grid_index"] == best["weight_grid_index"]
    return best, pd.DataFrame(rows)


def ablate(selected: KnowledgeGraphRecommender, val_requests, k: int) -> pd.DataFrame:
    """Drop one component at a time from the selected configuration
    (remaining weights renormalized to sum to 1 -- ranking-invariant, kept
    only so reported weights stay comparable). Validation only; never used
    for selection."""
    _, full = evaluate(selected, val_requests, k)
    rows = [{"variant": "full", "dropped_component": "", **{f"w_{c}": selected.weights[c] for c in COMPONENTS}, **full}]
    for component in COMPONENTS:
        remaining = sum(v for c, v in selected.weights.items() if c != component)
        if remaining <= 0:
            continue
        weights = {c: (0.0 if c == component else v / remaining) for c, v in selected.weights.items()}
        _, summary = evaluate(selected.with_config(selected.damping, weights), val_requests, k)
        rows.append({"variant": f"without_{component}", "dropped_component": component,
                     **{f"w_{c}": weights[c] for c in COMPONENTS}, **summary})
    df = pd.DataFrame(rows)
    df["delta_ndcg_at_k_vs_full"] = df["ndcg_at_k"] - full["ndcg_at_k"]
    df["delta_recall_at_k_vs_full"] = df["recall_at_k"] - full["recall_at_k"]
    return df


def category_signal(selected: KnowledgeGraphRecommender, val_requests) -> dict:
    """How much the category component can discriminate inside the
    shared candidate universe (every candidate already matches the
    request's category, so only primary vs secondary role varies)."""
    col = COMPONENTS.index("category")
    n_pools, n_varying, n_candidates, n_secondary = 0, 0, 0, 0
    for req in val_requests:
        if req.excluded:
            continue
        raw = selected.raw_components(req.link, req.candidates)[:, col]
        n_pools += 1
        n_candidates += len(raw)
        if raw.max() - raw.min() > 1e-12:
            n_varying += 1
        n_secondary += sum(1 for w in req.candidates if selected.index.worker_categories[w].get(req.category) == "secondary")
    return {
        "split": "validation",
        "n_requests": n_pools,
        "share_pools_where_category_component_varies": n_varying / n_pools if n_pools else 0.0,
        "share_candidates_secondary_specialization": n_secondary / n_candidates if n_candidates else 0.0,
        "share_candidates_primary_specialization": 1 - n_secondary / n_candidates if n_candidates else 0.0,
    }


# ---------------------------------------------------------------------------
# Comparison tables and cold-start
# ---------------------------------------------------------------------------

def metrics_row(name: str, recommender, eval_requests, split: str, k: int) -> pd.DataFrame:
    _, summary = evaluate(recommender, eval_requests, k)
    return pd.DataFrame([{"method": name, "split": split, **summary}])


def cold_start_rows(name: str, recommender, eval_requests_by_split: dict, cold_start_ids: set, k: int) -> pd.DataFrame:
    rows = []
    for split, eval_requests in eval_requests_by_split.items():
        cold_reqs = cold_start_requests(eval_requests, cold_start_ids)
        _, summary = evaluate(recommender, cold_reqs, k)
        rows.append({"method": name, "split": split, "n_cold_start_requests": len(cold_reqs), **summary})
    return pd.DataFrame(rows)


def cold_worker_diagnostics(name: str, recommender, eval_requests, cold_start_ids: set, split: str, k: int) -> dict:
    """docs/experimental_design.md Section 12: Cold Recall@K (macro over
    requests whose relevant set contains a cold-start worker) and the cold
    exposure ratio (share of top-K slots given to cold-start workers /
    their share of relevant items)."""
    slots = cold_slots = relevant = relevant_cold = 0
    cold_recalls = []
    for req in eval_requests:
        if req.excluded:
            continue
        ranked = recommender.rank(req.link, req.candidates, k)
        slots += len(ranked)
        cold_slots += sum(1 for w in ranked if w in cold_start_ids)
        rel = [w for w, g in req.relevance.items() if g >= 1]
        cold_rel = [w for w in rel if w in cold_start_ids]
        relevant += len(rel)
        relevant_cold += len(cold_rel)
        if cold_rel:
            top = set(ranked)
            cold_recalls.append(sum(1 for w in cold_rel if w in top) / len(cold_rel))
    cold_share_top_k = cold_slots / slots if slots else 0.0
    cold_share_relevant = relevant_cold / relevant if relevant else 0.0
    return {
        "method": name, "split": split,
        "n_cold_start_requests": len(cold_recalls),
        "cold_recall_at_k": float(np.mean(cold_recalls)) if cold_recalls else 0.0,
        "cold_share_of_top_k_slots": cold_share_top_k,
        "cold_share_of_relevant_items": cold_share_relevant,
        "cold_exposure_ratio": cold_share_top_k / cold_share_relevant if cold_share_relevant else 0.0,
    }


def rebuild_frozen_baselines(data, data_cfg, cf_cfg, link_to_client) -> dict:
    with open(DAY4_RESULTS_DIR / "selected_cbf_config.json", encoding="utf-8") as fh:
        cbf_selected = json.load(fh)
    with open(DAY5_RESULTS_DIR / "selected_cf_config.json", encoding="utf-8") as fh:
        cf_selected = json.load(fh)
    popularity = PopularityBaseline(data.interactions, data.requests)
    matrix = build_training_matrix(data.interactions, data.requests, cf_selected["event_weights"], cf_selected["aggregation"])
    return {
        "random": RandomBaseline(seed=data_cfg.seed),
        "popularity": popularity,
        "cbf": ContentBasedRecommender(data.workers, data.requests, cbf_selected["weights"], cbf_selected["price_neutral_score"]),
        "cf": CollaborativeFiltering(
            matrix, cf_selected["method"], cf_selected["k_neighbors"], popularity, link_to_client, cf_cfg.min_similarity
        ),
    }


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def explain_split(rec: KnowledgeGraphRecommender, eval_requests, k: int) -> list:
    explanations = []
    for req in eval_requests:
        if req.excluded:
            continue
        explanations.extend(rec.explain(req.link, req.candidates, k))
    return explanations


def rubric_template(sample: list) -> pd.DataFrame:
    """Blank rubric sheet for the author's later qualitative assessment
    (single rater, docs/experimental_design.md Section 15). Rubric
    criteria are placeholders pending review; nothing is pre-filled."""
    rows = []
    for i, e in enumerate(sample, start=1):
        rows.append({
            "sample_id": i, "link": e.link, "worker_id": e.worker_id, "rank": e.rank,
            "score": e.score, "cited_components": "|".join(e.cited_components), "explanation": e.summary,
            "understandable_1to5": "", "grounded_1to5": "", "useful_1to5": "", "rater_notes": "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(comparison_test, tuning_df, cold_table, ablation_df, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"]
    width = 0.16
    for i, method in enumerate(comparison_test["method"]):
        values = comparison_test.loc[comparison_test["method"] == method, metrics].iloc[0]
        ax.bar([x + i * width for x in range(len(metrics))], values, width, label=method)
    ax.set_xticks([x + 2 * width for x in range(len(metrics))])
    ax.set_xticklabels(["Precision@10", "Recall@10", "F1@10", "NDCG@10"])
    ax.set_title("Model comparison at K=10 (test set): Random/Popularity/CBF/CF/KG")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "model_comparison_test_k10.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"w{r.damping:g}/g{r.weight_grid_index}" for r in tuning_df.itertuples()]
    colors = ["#d62728" if s else "#1f77b4" for s in tuning_df["selected"]]
    ax.bar(labels, tuning_df["ndcg_at_k"], color=colors)
    ax.set_ylabel("Validation NDCG@10")
    ax.set_title("KG validation-only tuning grid (damping / weight entry; selected in red)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    path = out_dir / "kg_validation_tuning.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    test_cold = cold_table.loc[cold_table["split"] == "test"]
    ax.bar(test_cold["method"], test_cold["recall_at_k"])
    ax.set_ylabel("Recall@10 on cold-start requests (test)")
    ax.set_title("Cold-start requests at K=10: Random/Popularity/CBF/CF/KG")
    fig.tight_layout()
    path = out_dir / "cold_start_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ablation_df["variant"], ablation_df["ndcg_at_k"],
           color=["#d62728" if v == "full" else "#1f77b4" for v in ablation_df["variant"]])
    ax.set_ylabel("Validation NDCG@10")
    ax.set_title("KG component ablation (validation; full model in red)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = out_dir / "kg_component_ablation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(graph_source: str) -> dict:
    data_cfg = load_config()
    eval_cfg = load_evaluation_config()
    cf_cfg = load_cf_config()
    kg_cfg = load_kg_config()
    k = eval_cfg.k

    print("[1/10] Loading model-visible data, protocol requests, and frozen Day 4/5 results")
    data = load_protocol_data()
    index = EligibilityIndex(data.workers, data.known_countries)
    cold_start_ids = set(pd.read_csv(COLD_START_PATH)["worker_id"])
    link_to_client = dict(zip(data.requests["link"], data.requests["client_id"]))
    val_requests = build_eval_requests(data, index, "validation", eval_cfg)
    test_requests = build_eval_requests(data, index, "test", eval_cfg)
    day5_validation = pd.read_csv(DAY5_RESULTS_DIR / "validation_metrics.csv")
    day5_test = pd.read_csv(DAY5_RESULTS_DIR / "test_metrics.csv")
    day5_cold_start = pd.read_csv(DAY5_RESULTS_DIR / "cold_start_metrics.csv")

    print(f"[2/10] Building the training-only graph snapshot (source: {graph_source})")
    view, parity = load_graph_snapshot(graph_source)
    kg_index = KGIndex(view)

    print("[3/10] Precomputing raw KG components on validation for every damping value")
    val_cache = KGComponentCache(tuple(kg_cfg.dwpc_damping_grid))
    base = KnowledgeGraphRecommender(
        kg_index, data.requests, kg_cfg.dwpc_damping_grid[0], kg_cfg.weight_grid[0], kg_cfg.category_role_weights,
        kg_cfg.reputation_prior_strength, kg_cfg.explanation_max_paths_per_component, val_cache,
    )

    print("[4/10] Tuning damping x weights on validation only")
    best, tuning_df = tune_kg(base, kg_cfg, val_requests, k)
    selected_val = base.with_config(best["damping"], best["weights"])

    print("[5/10] Component ablation and category-signal analysis (validation)")
    ablation_df = ablate(selected_val, val_requests, k)
    category_stats = category_signal(selected_val, val_requests)

    print("[6/10] Evaluating the selected configuration on validation and test")
    test_cache = KGComponentCache((best["damping"],))
    selected_test = KnowledgeGraphRecommender(
        kg_index, data.requests, best["damping"], best["weights"], kg_cfg.category_role_weights,
        kg_cfg.reputation_prior_strength, kg_cfg.explanation_max_paths_per_component, test_cache,
    )
    comparison_validation = pd.concat([day5_validation, metrics_row("kg", selected_val, val_requests, "validation", k)], ignore_index=True)
    comparison_test = pd.concat([day5_test, metrics_row("kg", selected_test, test_requests, "test", k)], ignore_index=True)

    print("[7/10] Cold-start evaluation")
    kg_cold = pd.concat([
        cold_start_rows("kg", selected_val, {"validation": val_requests}, cold_start_ids, k),
        cold_start_rows("kg", selected_test, {"test": test_requests}, cold_start_ids, k),
    ], ignore_index=True)
    comparison_cold_start = pd.concat([day5_cold_start, kg_cold], ignore_index=True)

    baselines = rebuild_frozen_baselines(data, data_cfg, cf_cfg, link_to_client)
    for name, rec in baselines.items():
        _, summary = evaluate(rec, test_requests, k)
        frozen = day5_test.loc[day5_test["method"] == name, "ndcg_at_k"].iloc[0]
        if abs(summary["ndcg_at_k"] - frozen) > 1e-12:
            raise SystemExit(f"rebuilt frozen {name} does not reproduce its frozen test NDCG@10 ({summary['ndcg_at_k']} vs {frozen})")
    diagnostics = []
    for split, reqs, kg_rec in (("validation", val_requests, selected_val), ("test", test_requests, selected_test)):
        for name, rec in {**baselines, "kg": kg_rec}.items():
            diagnostics.append(cold_worker_diagnostics(name, rec, reqs, cold_start_ids, split, k))
    cold_worker_df = pd.DataFrame(diagnostics)

    print("[8/10] Generating graph-path explanations for every top-10 recommendation")
    explanations = {"validation": explain_split(selected_val, val_requests, k), "test": explain_split(selected_test, test_requests, k)}
    for split, exps in explanations.items():
        leaked = [e for e in exps if e.worker_id in cold_start_ids
                  and (any(e.n_paths_by_type[t] for t in WORKER_HISTORY_PATH_TYPES) or e.reputation.n_reviews)]
        if leaked:
            raise SystemExit(f"{len(leaked)} {split} explanations give a cold-start worker its own interaction history")
    explainability = {}
    for split, exps in explanations.items():
        explainability[split] = {
            "all": explainability_summary(exps),
            "cold_start_workers": explainability_summary([e for e in exps if e.worker_id in cold_start_ids]),
            "established_workers": explainability_summary([e for e in exps if e.worker_id not in cold_start_ids]),
        }
    contribution_rows = []
    for split, exps in explanations.items():
        for row in component_contribution_summary(exps, COMPONENTS):
            contribution_rows.append({"split": split, **row})
    contribution_df = pd.DataFrame(contribution_rows)
    sample = deterministic_sample(explanations[kg_cfg.explanation_sample_split], kg_cfg.explanation_sample_size, data_cfg.seed)

    print("[9/10] Generating figures")
    figure_paths = make_figures(comparison_test, tuning_df, comparison_cold_start, ablation_df, RESULTS_DIR / "figures")

    print("[10/10] Saving results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    out.append(save_csv(tuning_df, RESULTS_DIR / "kg_tuning.csv"))
    out.append(save_csv(ablation_df, RESULTS_DIR / "kg_ablation_validation.csv"))
    out.append(save_csv(contribution_df, RESULTS_DIR / "kg_component_contributions.csv"))
    out.append(save_csv(comparison_validation, RESULTS_DIR / "validation_metrics.csv"))
    out.append(save_csv(comparison_test, RESULTS_DIR / "test_metrics.csv"))
    out.append(save_csv(comparison_cold_start, RESULTS_DIR / "cold_start_metrics.csv"))
    out.append(save_csv(cold_worker_df, RESULTS_DIR / "cold_start_worker_diagnostics.csv"))
    out.append(save_csv(rubric_template(sample), RESULTS_DIR / "explanation_rubric_template.csv"))

    primary_key = METRIC_KEY[kg_cfg.tuning_primary_metric]
    selected_row = tuning_df.loc[tuning_df["selected"]].iloc[0]
    selected_config = {
        "damping": best["damping"], "weight_grid_index": best["weight_grid_index"], "weights": best["weights"],
        "category_role_weights": kg_cfg.category_role_weights,
        "reputation_prior_strength": kg_cfg.reputation_prior_strength,
        "reputation_prior_mean": kg_index.mean_train_rating,
        "normalization": "minmax_within_candidate_pool",
        "tie_break": "worker_id_ascending",
        "fallback": "none",
        "primary_metric": kg_cfg.tuning_primary_metric, "primary_metric_value": float(selected_row[primary_key]),
        "tiebreak_metric": kg_cfg.tuning_tiebreak_metric,
        "tiebreak_metric_value": float(selected_row[METRIC_KEY[kg_cfg.tuning_tiebreak_metric]]),
        "tuning_grid_size": len(tuning_df),
    }
    out.append(save_json(selected_config, RESULTS_DIR / "selected_kg_config.json"))
    out.append(save_json({"category_signal": category_stats, "explainability": explainability}, RESULTS_DIR / "explainability_metrics.json"))
    out.append(save_json({"split": kg_cfg.explanation_sample_split, "seed": data_cfg.seed,
                          "explanations": [e.to_dict() for e in sample]}, RESULTS_DIR / "explanation_samples.json"))
    snapshot = {
        "fingerprint": view.fingerprint(), "table_fingerprints": view.table_fingerprints(), "stats": view.stats(),
        "graph_derived_reputation_prior_mean": kg_index.mean_train_rating,
    }
    out.append(save_json(snapshot, RESULTS_DIR / "graph_snapshot.json"))
    out.extend(figure_paths)

    manifest = build_manifest(REPO_ROOT, out)
    manifest.update({
        "seed": data_cfg.seed, "k": k,
        "kg_config_sha256": sha256_of(REPO_ROOT / "config" / "kg_config.yaml"),
        "graph_snapshot_fingerprint": view.fingerprint(),
        "graph_snapshot_table_fingerprints": view.table_fingerprints(),
        **parity,
        "frozen_inputs_sha256": {p: sha256_of(REPO_ROOT / p) for p in FROZEN_INPUTS},
    })
    save_manifest(manifest, RESULTS_DIR / "day7_manifest.json")

    return {
        "comparison_validation": comparison_validation, "comparison_test": comparison_test,
        "comparison_cold_start": comparison_cold_start, "cold_worker": cold_worker_df,
        "selected_config": selected_config, "tuning": tuning_df, "ablation": ablation_df,
        "category_signal": category_stats, "explainability": explainability, "contributions": contribution_df,
        "snapshot": snapshot, "parity": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph-source", choices=["neo4j", "csv"], default="neo4j")
    args = parser.parse_args()

    t0 = time.time()
    result = run(args.graph_source)
    elapsed = time.time() - t0

    pd.set_option("display.width", 200)
    print("\n=== KG validation tuning grid ===")
    print(result["tuning"].to_string(index=False))
    print("\n=== Day 7 validation metrics ===")
    print(result["comparison_validation"].to_string(index=False))
    print("\n=== Day 7 test metrics ===")
    print(result["comparison_test"].to_string(index=False))
    print("\n=== Cold-start request metrics ===")
    print(result["comparison_cold_start"].to_string(index=False))
    print("\n=== Cold-start worker diagnostics ===")
    print(result["cold_worker"].to_string(index=False))
    print("\n=== Component ablation (validation) ===")
    print(result["ablation"].to_string(index=False))
    print("\ncategory signal:", result["category_signal"])
    print("\nselected KG config:", result["selected_config"])
    print("\nparity:", result["parity"])
    print(f"\nelapsed_seconds: {round(elapsed, 1)}")
    print(f"Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
