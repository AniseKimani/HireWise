"""Day 5 Collaborative Filtering evaluation pipeline.

Usage (from repository root):

    python scripts/run_day5_evaluation.py

Requires Day 2/Day 3 outputs and Day 4's frozen results
(`results/day4/*.csv`) to already exist -- Day 4's Random/Popularity/CBF
rows are reused verbatim in the comparison tables, never recomputed.
Reuses the Day 4 evaluation protocol (config/evaluation_config.yaml,
src/evaluation/protocol.py) unchanged. Writes results to
`results/day5/`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.config import load_config
from src.data.io import build_manifest, save_csv, save_json, save_manifest, sha256_of
from src.evaluation.config import load_evaluation_config
from src.evaluation.evaluator import cold_start_requests, evaluate
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.recommenders.cf_config import load_cf_config
from src.recommenders.collaborative import CollaborativeFiltering, build_training_matrix
from src.recommenders.popularity import PopularityBaseline

RESULTS_DIR = REPO_ROOT / "results" / "day5"
DAY4_RESULTS_DIR = REPO_ROOT / "results" / "day4"
DAY3_SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
COLD_START_PATH = DAY3_SYNTHETIC / "cold_start_cohort.csv"

METRIC_KEY = {
    "ndcg_at_10": "ndcg_at_k", "precision_at_10": "precision_at_k",
    "recall_at_10": "recall_at_k", "f1_at_10": "f1_at_k",
}


def tune_cf(data, cf_cfg, eval_cfg, val_requests, fallback, link_to_client) -> tuple[dict, pd.DataFrame]:
    """Evaluate every (aggregation, method, k_neighbors) combination on
    VALIDATION only. Selection: primary metric (NDCG@10) descending, tie
    broken by the configured tiebreak metric (Recall@10), then by
    preferring the smaller (simpler) k_neighbors. Grid iteration order is
    fixed, so any remaining exact tie keeps the first-encountered
    configuration -- fully deterministic. Never looks at test."""
    primary_key = METRIC_KEY[cf_cfg.tuning_primary_metric]
    tiebreak_key = METRIC_KEY[cf_cfg.tuning_tiebreak_metric]

    rows = []
    best_config = None
    best_score = None

    for aggregation in cf_cfg.aggregation_grid:
        matrix = build_training_matrix(data.interactions, data.requests, cf_cfg.event_weights, aggregation)
        for method in cf_cfg.method_grid:
            cf = CollaborativeFiltering(
                matrix, method, cf_cfg.k_neighbors_grid[0], fallback, link_to_client, cf_cfg.min_similarity
            )
            for k_neighbors in cf_cfg.k_neighbors_grid:
                # Safe to mutate: k_neighbors only affects how many
                # precomputed (already-sorted) neighbors are sliced at
                # scoring time; nothing else is cached per-k.
                cf.k_neighbors = k_neighbors
                _, summary = evaluate(cf, val_requests, eval_cfg.k)
                row = {
                    "aggregation": aggregation, "method": method, "k_neighbors": k_neighbors,
                    **summary, "selected": False,
                }
                rows.append(row)

                score = (summary[primary_key], summary[tiebreak_key], -k_neighbors)
                if best_score is None or score > best_score:
                    best_score = score
                    best_config = {"aggregation": aggregation, "method": method, "k_neighbors": k_neighbors}

    for row in rows:
        row["selected"] = (
            row["aggregation"] == best_config["aggregation"]
            and row["method"] == best_config["method"]
            and row["k_neighbors"] == best_config["k_neighbors"]
        )
    return best_config, pd.DataFrame(rows)


def build_metrics_table(recommenders: dict, eval_requests: list, split: str, k: int) -> pd.DataFrame:
    rows = []
    for name, recommender in recommenders.items():
        _, summary = evaluate(recommender, eval_requests, k)
        rows.append({"method": name, "split": split, **summary})
    return pd.DataFrame(rows)


def build_cold_start_table(recommenders: dict, eval_requests_by_split: dict, cold_start_ids: set, k: int) -> pd.DataFrame:
    rows = []
    for split, eval_requests in eval_requests_by_split.items():
        cold_reqs = cold_start_requests(eval_requests, cold_start_ids)
        for name, recommender in recommenders.items():
            _, summary = evaluate(recommender, cold_reqs, k)
            rows.append({"method": name, "split": split, "n_cold_start_requests": len(cold_reqs), **summary})
    return pd.DataFrame(rows)


def compute_coverage(data, cf: CollaborativeFiltering, link_to_client: dict) -> dict:
    n_clients_total = data.requests["client_id"].nunique()
    n_clients_with_history = len(cf.matrix.client_index)
    n_workers_total = len(data.workers)
    n_workers_with_history = len(cf.matrix.worker_index)

    per_client_counts = pd.Series(cf.matrix.matrix.getnnz(axis=1), index=cf.matrix.clients)
    return {
        "n_clients_total": int(n_clients_total),
        "n_clients_with_training_history": int(n_clients_with_history),
        "n_clients_with_no_training_history": int(n_clients_total - n_clients_with_history),
        "median_distinct_workers_per_client_with_history": float(per_client_counts.median()) if len(per_client_counts) else 0.0,
        "n_workers_total": int(n_workers_total),
        "n_workers_with_training_history": int(n_workers_with_history),
        "n_workers_with_no_training_history": int(n_workers_total - n_workers_with_history),
    }


def compute_fallback_usage(cf: CollaborativeFiltering, eval_requests: list, k: int) -> dict:
    n_requests = 0
    n_fully_fallback_requests = 0
    n_recommendations = 0
    n_recommendations_via_fallback = 0
    for req in eval_requests:
        if req.excluded:
            continue
        n_requests += 1
        _, diag = cf.rank_with_diagnostics(req.link, req.candidates, k)
        if diag["request_fully_fallback"]:
            n_fully_fallback_requests += 1
        n_recommendations += diag["n_recommendations"]
        n_recommendations_via_fallback += diag["n_recommendations_via_fallback"]

    return {
        "n_requests": n_requests,
        "n_fully_fallback_requests": n_fully_fallback_requests,
        "fully_fallback_request_rate": n_fully_fallback_requests / n_requests if n_requests else 0.0,
        "n_recommendations": n_recommendations,
        "n_recommendations_via_fallback": n_recommendations_via_fallback,
        "recommendation_fallback_rate": n_recommendations_via_fallback / n_recommendations if n_recommendations else 0.0,
    }


def make_figures(comparison_test: pd.DataFrame, tuning_df: pd.DataFrame, cold_table: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(7.5, 4))
    metrics = ["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"]
    x = range(len(metrics))
    width = 0.2
    for i, method in enumerate(comparison_test["method"]):
        values = comparison_test.loc[comparison_test["method"] == method, metrics].iloc[0]
        ax.bar([xi + i * width for xi in x], values, width, label=method)
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(["Precision@10", "Recall@10", "F1@10", "NDCG@10"])
    ax.set_title("Model comparison at K=10 (test set): Random/Popularity/CBF/CF")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "model_comparison_test_k10.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    tuning_df = tuning_df.sort_values(["aggregation", "method", "k_neighbors"])
    labels = [f"{r.aggregation[:1]}/{r.method[:1]}/k{r.k_neighbors}" for r in tuning_df.itertuples()]
    colors = ["#d62728" if s else "#1f77b4" for s in tuning_df["selected"]]
    ax.bar(labels, tuning_df["ndcg_at_k"], color=colors)
    ax.set_ylabel("Validation NDCG@10")
    ax.set_title("CF validation-only tuning grid (selected in red)")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    path = out_dir / "cf_validation_tuning.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    test_cold = cold_table.loc[cold_table["split"] == "test"]
    ax.bar(test_cold["method"], test_cold["recall_at_k"])
    ax.set_ylabel("Cold-start Recall@10 (test set)")
    ax.set_title("Cold-start worker visibility at K=10: Random/Popularity/CBF/CF")
    fig.tight_layout()
    path = out_dir / "cold_start_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    return paths


def run() -> dict:
    data_cfg = load_config()
    eval_cfg = load_evaluation_config()
    cf_cfg = load_cf_config()

    print("[1/9] Loading Day 2/Day 3 model-visible data and Day 4 frozen results")
    data = load_protocol_data()
    index = EligibilityIndex(data.workers, data.known_countries)
    cold_start_ids = set(pd.read_csv(COLD_START_PATH)["worker_id"])
    link_to_client = dict(zip(data.requests["link"], data.requests["client_id"]))

    day4_validation = pd.read_csv(DAY4_RESULTS_DIR / "validation_metrics.csv")
    day4_test = pd.read_csv(DAY4_RESULTS_DIR / "test_metrics.csv")
    day4_cold_start = pd.read_csv(DAY4_RESULTS_DIR / "cold_start_metrics.csv")

    print("[2/9] Building validation/test evaluation requests (unchanged Day 4 protocol)")
    val_requests = build_eval_requests(data, index, "validation", eval_cfg)
    test_requests = build_eval_requests(data, index, "test", eval_cfg)

    print("[3/9] Building training-only Popularity fallback")
    fallback = PopularityBaseline(data.interactions, data.requests)

    print("[4/9] Tuning CF (aggregation x method x k_neighbors) on validation only")
    best_config, tuning_df = tune_cf(data, cf_cfg, eval_cfg, val_requests, fallback, link_to_client)

    print("[5/9] Building the selected CF configuration")
    matrix = build_training_matrix(data.interactions, data.requests, cf_cfg.event_weights, best_config["aggregation"])
    cf = CollaborativeFiltering(
        matrix, best_config["method"], best_config["k_neighbors"], fallback, link_to_client, cf_cfg.min_similarity
    )

    print("[6/9] Evaluating on validation and test")
    cf_validation = build_metrics_table({"cf": cf}, val_requests, "validation", eval_cfg.k)
    cf_test = build_metrics_table({"cf": cf}, test_requests, "test", eval_cfg.k)
    comparison_validation = pd.concat([day4_validation, cf_validation], ignore_index=True)
    comparison_test = pd.concat([day4_test, cf_test], ignore_index=True)

    print("[7/9] Cold-start evaluation and coverage/fallback diagnostics")
    cf_cold_start = build_cold_start_table({"cf": cf}, {"validation": val_requests, "test": test_requests}, cold_start_ids, eval_cfg.k)
    comparison_cold_start = pd.concat([day4_cold_start, cf_cold_start], ignore_index=True)

    coverage = compute_coverage(data, cf, link_to_client)
    fallback_usage = {
        "validation": compute_fallback_usage(cf, val_requests, eval_cfg.k),
        "test": compute_fallback_usage(cf, test_requests, eval_cfg.k),
    }
    cold_start_in_training_matrix = [w for w in cold_start_ids if w in cf.matrix.worker_index]

    print("[8/9] Generating figures")
    figures_dir = RESULTS_DIR / "figures"
    figure_paths = make_figures(comparison_test, tuning_df, comparison_cold_start, figures_dir)

    print("[9/9] Saving results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    output_paths.append(save_csv(tuning_df, RESULTS_DIR / "cf_tuning.csv"))
    output_paths.append(save_csv(comparison_validation, RESULTS_DIR / "validation_metrics.csv"))
    output_paths.append(save_csv(comparison_test, RESULTS_DIR / "test_metrics.csv"))
    output_paths.append(save_csv(comparison_cold_start, RESULTS_DIR / "cold_start_metrics.csv"))

    metric_key = METRIC_KEY[cf_cfg.tuning_primary_metric]
    selected_row = tuning_df.loc[tuning_df["selected"]].iloc[0]
    selected_config = {
        "aggregation": best_config["aggregation"], "method": best_config["method"], "k_neighbors": best_config["k_neighbors"],
        "event_weights": cf_cfg.event_weights, "min_similarity": cf_cfg.min_similarity,
        "primary_metric": cf_cfg.tuning_primary_metric, "primary_metric_value": float(selected_row[metric_key]),
        "tiebreak_metric": cf_cfg.tuning_tiebreak_metric, "tiebreak_metric_value": float(selected_row[METRIC_KEY[cf_cfg.tuning_tiebreak_metric]]),
        "tuning_grid_size": len(tuning_df), "fallback": cf_cfg.fallback,
    }
    output_paths.append(save_json(selected_config, RESULTS_DIR / "selected_cf_config.json"))

    coverage_out = {
        **coverage, "fallback_usage": fallback_usage,
        "n_cold_start_workers_total": len(cold_start_ids),
        "n_cold_start_workers_in_training_matrix": len(cold_start_in_training_matrix),
    }
    output_paths.append(save_json(coverage_out, RESULTS_DIR / "cf_coverage.json"))
    output_paths.extend(figure_paths)

    manifest_extra = {
        "seed": data_cfg.seed,
        "k": eval_cfg.k,
        "generator_v1_sha256": sha256_of(REPO_ROOT / "config" / "generator_v1.yaml"),
        "cf_config_sha256": sha256_of(REPO_ROOT / "config" / "cf_config.yaml"),
        "day4_cbf_config_sha256": sha256_of(DAY4_RESULTS_DIR / "selected_cbf_config.json"),
    }
    manifest = build_manifest(REPO_ROOT, output_paths)
    manifest.update(manifest_extra)
    save_manifest(manifest, RESULTS_DIR / "day5_manifest.json")

    return {
        "comparison_validation": comparison_validation, "comparison_test": comparison_test,
        "comparison_cold_start": comparison_cold_start, "selected_config": selected_config,
        "coverage": coverage_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    t0 = time.time()
    result = run()
    elapsed = time.time() - t0

    print("\n=== Day 5 validation metrics (Random/Popularity/CBF/CF) ===")
    print(result["comparison_validation"].to_string(index=False))
    print("\n=== Day 5 test metrics ===")
    print(result["comparison_test"].to_string(index=False))
    print("\n=== Day 5 cold-start metrics ===")
    print(result["comparison_cold_start"].to_string(index=False))
    print("\nselected CF config:", result["selected_config"])
    print("\ncoverage:", result["coverage"])
    print(f"\nelapsed_seconds: {round(elapsed, 1)}")
    print(f"Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
