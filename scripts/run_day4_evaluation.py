"""Day 4 evaluation pipeline: Random, Popularity, and Content-Based
Filtering baselines, evaluated under the protocol already approved in
docs/experimental_design.md Section 15.

Usage (from repository root):

    python scripts/run_day4_evaluation.py

Requires Day 2 (`data/processed/*.csv`) and Day 3
(`data/processed/synthetic/*.csv`) outputs to already exist. Never reads
`data/processed/synthetic_private/`. Writes results to `results/day4/`.
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
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.popularity import PopularityBaseline
from src.recommenders.random_baseline import RandomBaseline

RESULTS_DIR = REPO_ROOT / "results" / "day4"
DAY3_SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
COLD_START_PATH = DAY3_SYNTHETIC / "cold_start_cohort.csv"


# cbf_tuning_primary_metric is written e.g. "ndcg_at_10" in config; our
# per-summary keys are named "*_at_k" (k is fixed once at evaluation time).
PRIMARY_METRIC_KEY = {
    "ndcg_at_10": "ndcg_at_k", "precision_at_10": "precision_at_k",
    "recall_at_10": "recall_at_k", "f1_at_10": "f1_at_k",
}


def _weights_row(weights: dict[str, float]) -> dict[str, float]:
    return {"skill_weight": weights["skill"], "category_weight": weights["category"], "price_weight": weights["price"]}


def tune_cbf(data, val_requests, eval_cfg) -> tuple[dict, pd.DataFrame]:
    """Evaluate every predeclared weight combination on VALIDATION only,
    select by the configured primary metric (default NDCG@10), and
    return (selected_weights, tuning_results_df). Never looks at test."""
    rows = []
    best_weights = None
    best_score = float("-inf")
    metric_key = PRIMARY_METRIC_KEY[eval_cfg.cbf_tuning_primary_metric]

    for weights in eval_cfg.cbf_weight_grid:
        cbf = ContentBasedRecommender(data.workers, data.requests, weights, eval_cfg.cbf_price_neutral_score)
        _, summary = evaluate(cbf, val_requests, eval_cfg.k)
        row = {**_weights_row(weights), **summary, "selected": False}
        rows.append(row)
        if summary[metric_key] > best_score:
            best_score = summary[metric_key]
            best_weights = weights

    best_weights_row = _weights_row(best_weights)
    for row in rows:
        row["selected"] = best_weights_row == {
            "skill_weight": row["skill_weight"], "category_weight": row["category_weight"], "price_weight": row["price_weight"],
        }
    return best_weights, pd.DataFrame(rows)


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


def make_figures(test_table: pd.DataFrame, tuning_df: pd.DataFrame, cold_table: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"]
    x = range(len(metrics))
    width = 0.25
    for i, method in enumerate(test_table["method"]):
        values = test_table.loc[test_table["method"] == method, metrics].iloc[0]
        ax.bar([xi + i * width for xi in x], values, width, label=method)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(["Precision@10", "Recall@10", "F1@10", "NDCG@10"])
    ax.set_title("Model comparison at K=10 (test set)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "model_comparison_test_k10.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"S{row.skill_weight:.2f}/C{row.category_weight:.2f}/P{row.price_weight:.2f}" for row in tuning_df.itertuples()]
    ax.bar(labels, tuning_df["ndcg_at_k"], color=["#d62728" if s else "#1f77b4" for s in tuning_df["selected"]])
    ax.set_ylabel("Validation NDCG@10")
    ax.set_title("CBF validation-only weight tuning (selected in red)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    path = out_dir / "cbf_validation_tuning.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    test_cold = cold_table.loc[cold_table["split"] == "test"]
    ax.bar(test_cold["method"], test_cold["recall_at_k"])
    ax.set_ylabel("Cold-start Recall@10 (test set)")
    ax.set_title("Cold-start worker visibility at K=10")
    fig.tight_layout()
    path = out_dir / "cold_start_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    return paths


def run() -> dict:
    data_cfg = load_config()
    eval_cfg = load_evaluation_config()

    print("[1/8] Loading Day 2/Day 3 model-visible data")
    data = load_protocol_data()
    index = EligibilityIndex(data.workers, data.known_countries)
    cold_start_ids = set(pd.read_csv(COLD_START_PATH)["worker_id"])

    print("[2/8] Building validation/test evaluation requests")
    val_requests = build_eval_requests(data, index, "validation", eval_cfg)
    test_requests = build_eval_requests(data, index, "test", eval_cfg)

    print("[3/8] Building Random and Popularity baselines")
    random_rec = RandomBaseline(seed=data_cfg.seed)
    popularity_rec = PopularityBaseline(data.interactions, data.requests)

    print("[4/8] Tuning CBF weights on validation only")
    selected_weights, tuning_df = tune_cbf(data, val_requests, eval_cfg)
    cbf_rec = ContentBasedRecommender(data.workers, data.requests, selected_weights, eval_cfg.cbf_price_neutral_score)

    recommenders = {"random": random_rec, "popularity": popularity_rec, "cbf": cbf_rec}

    print("[5/8] Evaluating on validation and test")
    validation_metrics = build_metrics_table(recommenders, val_requests, "validation", eval_cfg.k)
    test_metrics = build_metrics_table(recommenders, test_requests, "test", eval_cfg.k)

    print("[6/8] Cold-start evaluation")
    cold_start_metrics = build_cold_start_table(
        recommenders, {"validation": val_requests, "test": test_requests}, cold_start_ids, eval_cfg.k
    )

    print("[7/8] Generating figures")
    figures_dir = RESULTS_DIR / "figures"
    figure_paths = make_figures(test_metrics, tuning_df, cold_start_metrics, figures_dir)

    print("[8/8] Saving results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    output_paths.append(save_csv(validation_metrics, RESULTS_DIR / "validation_metrics.csv"))
    output_paths.append(save_csv(test_metrics, RESULTS_DIR / "test_metrics.csv"))
    output_paths.append(save_csv(cold_start_metrics, RESULTS_DIR / "cold_start_metrics.csv"))
    output_paths.append(save_csv(tuning_df, RESULTS_DIR / "cbf_tuning.csv"))

    metric_key = PRIMARY_METRIC_KEY[eval_cfg.cbf_tuning_primary_metric]
    selected_config = {
        "weights": selected_weights,
        "primary_metric": eval_cfg.cbf_tuning_primary_metric,
        "primary_metric_value": float(tuning_df.loc[tuning_df["selected"], metric_key].iloc[0]),
        "tuning_grid_size": len(eval_cfg.cbf_weight_grid),
        "price_neutral_score": eval_cfg.cbf_price_neutral_score,
    }
    output_paths.append(save_json(selected_config, RESULTS_DIR / "selected_cbf_config.json"))
    output_paths.extend(figure_paths)

    manifest_extra = {
        "seed": data_cfg.seed,
        "k": eval_cfg.k,
        "min_candidate_pool_size": eval_cfg.min_candidate_pool_size,
        "generator_v1_sha256": sha256_of(REPO_ROOT / "config" / "generator_v1.yaml"),
        "evaluation_config_sha256": sha256_of(REPO_ROOT / "config" / "evaluation_config.yaml"),
        "n_validation_requests": len(val_requests),
        "n_test_requests": len(test_requests),
        "n_validation_excluded": sum(1 for r in val_requests if r.excluded),
        "n_test_excluded": sum(1 for r in test_requests if r.excluded),
    }
    manifest = build_manifest(REPO_ROOT, output_paths)
    manifest.update(manifest_extra)
    save_manifest(manifest, RESULTS_DIR / "day4_manifest.json")

    return {
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "cold_start_metrics": cold_start_metrics, "selected_config": selected_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    t0 = time.time()
    result = run()
    elapsed = time.time() - t0

    print("\n=== Day 4 validation metrics ===")
    print(result["validation_metrics"].to_string(index=False))
    print("\n=== Day 4 test metrics ===")
    print(result["test_metrics"].to_string(index=False))
    print("\n=== Day 4 cold-start metrics ===")
    print(result["cold_start_metrics"].to_string(index=False))
    print("\nselected CBF config:", result["selected_config"])
    print(f"\nelapsed_seconds: {round(elapsed, 1)}")
    print(f"Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
