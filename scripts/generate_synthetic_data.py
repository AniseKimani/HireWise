"""Day 3 synthetic-data generation pipeline.

Usage (from repository root):

    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --generator-config config/generator_v1_content_leaning.yaml --out data/processed/synthetic_sensitivity_content

Loads the Day 2 outputs (data/processed/*.csv), generates 2,000 synthetic
clients, 3,000 synthetic workers (300 new-entrant cold-start), assigns the
12,000 real Day 2 sampled requests to clients on a simulated 365-day
timeline, computes a global temporal 70/10/20 split, runs the two-pass
multi-factor latent-utility interaction simulator, generates ratings,
validates the result, and saves deterministic outputs.

Model-visible outputs go to <out>/ (default data/processed/synthetic/);
generator-private truth (latent quality, taste, raw/standardized utility
components, noise) goes to <out>_private/ -- see
docs/synthetic_data_report.md's leakage-control section. Fails clearly if
the Day 2 outputs are missing; never regenerates them itself (run
scripts/run_data_pipeline.py first).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.data.config import DataConfig, load_config
from src.data.generator_config import GeneratorConfig, load_generator_config
from src.data.io import build_manifest, save_csv, save_json, save_manifest
from src.data.synth_clients import finalize_client_observables, generate_clients
from src.data.synth_cold_start import assign_worker_join_days, select_cold_start_cohort
from src.data.synth_interactions import (
    build_neighbor_indices,
    build_proficiency_matrix,
    build_skill_relatedness,
    candidate_pool_sizes,
    CandidateIndex,
    compute_sparse_threshold,
    estimate_utility_moments,
    generate_ratings,
    simulate,
)
from src.data.synth_timeline import assign_timeline, compute_temporal_split
from src.data.synth_workers import apply_location_guarantee, generate_workers

DAY2_PROCESSED = REPO_ROOT / "data" / "processed"


def _load_day2_outputs(processed_dir: Path) -> dict:
    required = [
        "sampled_requests.csv", "category_stats.csv", "category_similarity.csv",
        "eligible_jobs.csv", "skill_cooccurrence_full_corpus.csv", "skills_vocab.csv",
    ]
    missing = [f for f in required if not (processed_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Day 2 outputs missing from {processed_dir}: {missing}. "
            "Run scripts/run_data_pipeline.py first."
        )
    return {
        "sampled_requests": pd.read_csv(processed_dir / "sampled_requests.csv"),
        "category_stats": pd.read_csv(processed_dir / "category_stats.csv"),
        "category_similarity": pd.read_csv(processed_dir / "category_similarity.csv"),
        "eligible_jobs": pd.read_csv(processed_dir / "eligible_jobs.csv"),
        "cooccurrence": pd.read_csv(processed_dir / "skill_cooccurrence_full_corpus.csv"),
        "vocab": pd.read_csv(processed_dir / "skills_vocab.csv")["skill"].tolist(),
    }


def run(config: DataConfig, gen_config: GeneratorConfig, day2_dir: Path) -> dict:
    day2 = _load_day2_outputs(day2_dir)
    sampled_requests = day2["sampled_requests"]
    requests_info = sampled_requests[["link", "category", "price_tier", "skills_vocab", "location_requirement"]]

    print("[1/13] Loaded Day 2 outputs and frozen generator configuration")

    print("[2/13] Generating clients and assigning the 12,000 real requests")
    clients_df, clients_private_df, assignment_df = generate_clients(sampled_requests, config, gen_config)
    clients_df = finalize_client_observables(clients_df, assignment_df, sampled_requests)

    print("[3/13] Building the simulated timeline")
    timeline_df = assign_timeline(assignment_df, clients_df, config, gen_config)

    print("[4/13] Computing the global temporal train/validation/test split")
    split_df, boundaries = compute_temporal_split(timeline_df, config)

    print("[5/13] Generating workers")
    workers_df, workers_private_df = generate_workers(
        day2["category_stats"], day2["category_similarity"], day2["eligible_jobs"], config, gen_config
    )
    top_countries = (
        day2["eligible_jobs"]["country_clean"].dropna().value_counts().head(gen_config.worker_country_top_n).index.tolist()
    )
    workers_df, location_report = apply_location_guarantee(
        workers_df, sampled_requests, top_countries, config, gen_config
    )

    print("[6/13] Selecting the new-entrant cold-start cohort")
    cold_start_ids = select_cold_start_cohort(workers_df, config)
    workers_df = assign_worker_join_days(workers_df, cold_start_ids, boundaries["validation_start_day"])

    print("[7/13] Building candidate eligibility index and skill-relatedness lookups")
    relatedness = build_skill_relatedness(day2["cooccurrence"], gen_config.skill_relatedness_topn)
    proficiency_matrix, skill_index = build_proficiency_matrix(workers_private_df, day2["vocab"])
    neighbor_indices = build_neighbor_indices(relatedness, skill_index)
    candidate_index = CandidateIndex(workers_df, top_countries)

    print("[8/13] Reporting candidate-set statistics")
    pool_sizes = candidate_pool_sizes(split_df, requests_info, candidate_index)

    print("[9/13] Pass 1: estimating utility standardization moments and hire calibration")
    calibration = estimate_utility_moments(
        split_df, requests_info, clients_private_df, workers_private_df, candidate_index,
        proficiency_matrix, skill_index, neighbor_indices, config, gen_config,
    )

    print("[10/13] Pass 2: chronological interaction simulation")
    interactions_df, truth_df = simulate(
        split_df, requests_info, clients_private_df, workers_private_df, candidate_index,
        proficiency_matrix, skill_index, neighbor_indices, calibration, config, gen_config,
    )

    print("[11/13] Generating ratings")
    ratings_df = generate_ratings(truth_df, interactions_df, config, gen_config)

    print("[12/13] Computing the sparse-history threshold")
    sparse_threshold, training_hire_counts = compute_sparse_threshold(interactions_df, split_df, workers_df)

    print("[13/13] Validating and saving outputs")
    return {
        "clients_df": clients_df, "clients_private_df": clients_private_df,
        "workers_df": workers_df, "workers_private_df": workers_private_df,
        "split_df": split_df, "boundaries": boundaries,
        "interactions_df": interactions_df, "truth_df": truth_df, "ratings_df": ratings_df,
        "pool_sizes": pool_sizes, "calibration": calibration,
        "sparse_threshold": sparse_threshold, "training_hire_counts": training_hire_counts,
        "location_report": location_report, "cold_start_ids": cold_start_ids,
    }


def save_outputs(result: dict, out_dir: Path, private_dir: Path, config: DataConfig, gen_config: GeneratorConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    output_paths.append(save_csv(result["clients_df"].sort_values("client_id"), out_dir / "clients.csv"))
    output_paths.append(save_csv(result["workers_df"].sort_values("worker_id"), out_dir / "workers.csv"))
    output_paths.append(save_csv(result["split_df"].sort_values("link"), out_dir / "requests.csv"))
    output_paths.append(save_csv(
        result["interactions_df"].sort_values(["link", "rank_in_shortlist"]), out_dir / "interactions.csv"
    ))
    output_paths.append(save_csv(result["ratings_df"].sort_values(["link", "worker_id"]), out_dir / "ratings.csv"))

    cold_start_cohort = result["workers_df"].loc[
        result["workers_df"]["is_cold_start"], ["worker_id", "primary_category", "join_day"]
    ].sort_values("worker_id")
    output_paths.append(save_csv(cold_start_cohort, out_dir / "cold_start_cohort.csv"))

    output_paths.append(save_json(result["boundaries"], out_dir / "split_boundaries.json"))

    summary = _build_summary(result, config, gen_config)
    output_paths.append(save_json(summary, out_dir / "generation_summary.json"))

    output_paths.append(save_csv(result["clients_private_df"].sort_values("client_id"), private_dir / "clients_private.csv"))
    output_paths.append(save_csv(result["workers_private_df"].sort_values("worker_id"), private_dir / "workers_private.csv"))
    output_paths.append(save_csv(result["truth_df"].sort_values(["link", "worker_id"]), private_dir / "truth.csv"))
    output_paths.append(save_json(result["calibration"], private_dir / "calibration.json"))

    manifest = build_manifest(REPO_ROOT, output_paths)
    save_manifest(manifest, out_dir / "manifest.json")

    return summary


def _build_summary(result: dict, config: DataConfig, gen_config: GeneratorConfig) -> dict:
    interactions = result["interactions_df"]
    ratings = result["ratings_df"]
    pool_sizes = result["pool_sizes"]["n_candidates"]
    boundaries = result["boundaries"]

    n_hired = int((interactions["event"] == "HIRED").sum())
    n_shortlisted = len(interactions)
    n_requests = result["split_df"]["link"].nunique()

    return {
        "generator_version": gen_config.version,
        "seed": config.seed,
        "n_clients": len(result["clients_df"]),
        "n_workers": len(result["workers_df"]),
        "n_cold_start_workers": int(result["workers_df"]["is_cold_start"].sum()),
        "n_requests": int(n_requests),
        "n_interactions": n_shortlisted,
        "n_shortlisted_only": n_shortlisted - n_hired,
        "n_hired": n_hired,
        "hire_rate": n_hired / n_requests if n_requests else 0.0,
        "n_ratings": len(ratings),
        "rating_completion_rate": len(ratings) / n_hired if n_hired else 0.0,
        "rating_distribution": ratings["rating"].value_counts(normalize=True).sort_index().to_dict() if len(ratings) else {},
        "mean_interactions_per_worker": n_shortlisted / len(result["workers_df"]),
        "mean_interactions_per_client": n_shortlisted / len(result["clients_df"]),
        "sparse_history_threshold": result["sparse_threshold"],
        "candidate_pool": {
            "mean": float(pool_sizes.mean()), "median": float(pool_sizes.median()),
            "min": int(pool_sizes.min()), "max": int(pool_sizes.max()),
            "n_requests_pool_below_20": int((pool_sizes < 20).sum()),
        },
        "split": {
            "n_train": boundaries["n_train"], "n_validation": boundaries["n_validation"], "n_test": boundaries["n_test"],
            "train_end_day": boundaries["train_end_day"], "validation_start_day": boundaries["validation_start_day"],
            "validation_end_day": boundaries["validation_end_day"], "test_start_day": boundaries["test_start_day"],
        },
        "location_guarantee": result["location_report"],
        "utility_weights": gen_config.utility_weights,
        "calibration": {
            "alpha": result["calibration"]["alpha"], "theta": result["calibration"]["theta"],
            "noise_std": result["calibration"]["noise_std"], "signal_var": result["calibration"]["signal_var"],
            "pass1_hire_rate_estimate": result["calibration"]["pass1_hire_rate_estimate"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(REPO_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--generator-config", default=None, help="Defaults to data_config.yaml's generator_config_path")
    parser.add_argument("--day2-dir", default=str(DAY2_PROCESSED))
    parser.add_argument("--out", default=str(DAY2_PROCESSED / "synthetic"))
    args = parser.parse_args()

    config = load_config(args.data_config)
    generator_config_path = args.generator_config or config.generator_config_path
    gen_config = load_generator_config(generator_config_path)

    out_dir = Path(args.out).resolve()
    private_dir = out_dir.parent / f"{out_dir.name}_private"

    t0 = time.time()
    result = run(config, gen_config, Path(args.day2_dir))
    summary = save_outputs(result, out_dir, private_dir, config, gen_config)
    elapsed = time.time() - t0

    print(f"\n=== Day 3 synthetic-data generation summary (generator {gen_config.version}) ===")
    for key in ("n_clients", "n_workers", "n_cold_start_workers", "n_requests", "n_interactions",
                "n_hired", "hire_rate", "n_ratings", "rating_completion_rate", "sparse_history_threshold"):
        print(f"{key}: {summary[key]}")
    print(f"candidate_pool: {summary['candidate_pool']}")
    print(f"split: {summary['split']}")
    print(f"elapsed_seconds: {round(elapsed, 1)}")
    print(f"\nModel-visible outputs: {out_dir}")
    print(f"Generator-private truth: {private_dir}")


if __name__ == "__main__":
    main()
