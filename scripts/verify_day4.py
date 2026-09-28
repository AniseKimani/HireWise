"""Independent verification of Day 4 recommenders and evaluation results.

Re-derives invariants independently rather than trusting
results/day4/*.csv at face value: recomputes Popularity's training-only
hire counts from raw interactions, re-runs each recommender on a sample
of real requests and checks the output directly, and re-evaluates the
selected CBF weights on validation itself rather than reading the stored
number back.

Usage:
    python scripts/verify_day4.py

Exits non-zero if any check fails.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.config import load_config
from src.data.io import sha256_of
from src.evaluation.config import load_evaluation_config
from src.evaluation.evaluator import evaluate
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, extract_required_countries, load_protocol_data
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.popularity import PopularityBaseline
from src.recommenders.random_baseline import RandomBaseline

RESULTS_DIR = REPO_ROOT / "results" / "day4"
DAY3_SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    data_cfg = load_config()
    eval_cfg = load_evaluation_config()

    required_files = [
        "validation_metrics.csv", "test_metrics.csv", "cold_start_metrics.csv",
        "cbf_tuning.csv", "selected_cbf_config.json", "day4_manifest.json",
    ]
    missing = [f for f in required_files if not (RESULTS_DIR / f).exists()]
    check("all expected Day 4 result artifacts exist", not missing, f"missing {missing}")
    if missing:
        print("\nRun scripts/run_day4_evaluation.py first.")
        return 1

    # ---- Phase 1 invariants (unchanged) ------------------------------------
    generator_hash_path = REPO_ROOT / "config" / "generator_v1.yaml.sha256"
    with open(generator_hash_path, encoding="utf-8") as fh:
        expected_hash = fh.read().split()[0]
    actual_hash = sha256_of(REPO_ROOT / "config" / "generator_v1.yaml")
    check("frozen generator_v1.yaml hash unchanged", actual_hash == expected_hash,
          f"expected {expected_hash}, got {actual_hash}")

    data = load_protocol_data()
    check("3,000 workers", len(data.workers) == 3000, f"got {len(data.workers)}")
    check("12,000 requests", data.requests["link"].nunique() == 12000)
    counts = data.requests["split"].value_counts()
    check("train = 8,400", counts.get("train", 0) == 8400)
    check("validation = 1,200", counts.get("validation", 0) == 1200)
    check("test = 2,400", counts.get("test", 0) == 2400)
    cold_start = pd.read_csv(DAY3_SYNTHETIC / "cold_start_cohort.csv")
    check("300 cold-start workers", len(cold_start) == 300)

    # ---- Independent Popularity recomputation ------------------------------
    train_links = set(data.requests.loc[data.requests["split"] == "train", "link"])
    train_hires = data.interactions.loc[
        (data.interactions["event"] == "HIRED") & data.interactions["link"].isin(train_links)
    ]
    independent_counts = train_hires.groupby("worker_id").size().to_dict()
    popularity = PopularityBaseline(data.interactions, data.requests)
    check("Popularity hire counts match an independent recomputation from raw training interactions",
          popularity.hire_counts == independent_counts)

    non_train_links = set(data.requests.loc[data.requests["split"] != "train", "link"])
    non_train_hire_workers = set(
        data.interactions.loc[
            (data.interactions["event"] == "HIRED") & data.interactions["link"].isin(non_train_links), "worker_id"
        ]
    )
    leaked = [w for w in non_train_hire_workers if w not in independent_counts and w in popularity.hire_counts]
    check("Popularity uses training interactions only (no validation/test-only hire leaked in)", not leaked, f"leaked: {leaked}")

    # ---- Independent recommender smoke checks on a real sample -------------
    index = EligibilityIndex(data.workers, data.known_countries)
    val_requests = build_eval_requests(data, index, "validation", eval_cfg)
    included = [r for r in val_requests if not r.excluded][:200]

    with open(RESULTS_DIR / "selected_cbf_config.json", encoding="utf-8") as fh:
        selected_config = json.load(fh)
    cbf = ContentBasedRecommender(data.workers, data.requests, selected_config["weights"], eval_cfg.cbf_price_neutral_score)
    random_rec = RandomBaseline(seed=data_cfg.seed)

    worker_ids = set(data.workers["worker_id"])
    all_valid_workers, all_no_dup, all_k_ok = True, True, True
    join_day = data.workers.set_index("worker_id")["join_day"]
    country = data.workers.set_index("worker_id")["country"]
    join_day_ok, location_ok = True, True

    for req in included:
        for rec in (random_rec, popularity, cbf):
            ranked = rec.rank(req.link, req.candidates, eval_cfg.k)
            if not set(ranked).issubset(worker_ids):
                all_valid_workers = False
            if len(ranked) != len(set(ranked)):
                all_no_dup = False
            if len(ranked) > eval_cfg.k:
                all_k_ok = False
            request_row = data.requests.loc[data.requests["link"] == req.link].iloc[0]
            for w in ranked:
                if join_day.get(w, float("inf")) > request_row["simulated_day"]:
                    join_day_ok = False
                required = extract_required_countries(request_row["location_requirement"], data.known_countries)
                if required and country.get(w) not in required:
                    location_ok = False

    check("all recommendations refer to valid worker IDs", all_valid_workers)
    check("no duplicate worker per request in any recommender's output", all_no_dup)
    check("every recommender returns at most K=10 recommendations", all_k_ok)
    check("recommendations respect worker join date", join_day_ok)
    check("recommendations respect hard location restrictions", location_ok)

    # ---- Leakage: recommender source has no private-data references -------
    private_markers = ("quality", "true_skills", "proficiency", "taste_", "synthetic_private", "utility", "noise")
    leaked_sources = []
    for cls in (RandomBaseline, PopularityBaseline, ContentBasedRecommender):
        source = inspect.getsource(cls).lower()
        for marker in private_markers:
            if marker in source:
                leaked_sources.append((cls.__name__, marker))
    check("no recommender source references generator-private markers", not leaked_sources, f"{leaked_sources}")

    private_dir_touched = "synthetic_private" in inspect.getsource(load_protocol_data)
    check("protocol data loader never reads data/processed/synthetic_private/", not private_dir_touched)

    # ---- Selected CBF weights came from validation, not test --------------
    tuning_df = pd.read_csv(RESULTS_DIR / "cbf_tuning.csv")
    selected_rows = tuning_df.loc[tuning_df["selected"]]
    check("exactly one tuning-grid row marked selected", len(selected_rows) == 1, f"got {len(selected_rows)}")
    if len(selected_rows) == 1:
        row = selected_rows.iloc[0]
        recorded_weights = {"skill": row["skill_weight"], "category": row["category_weight"], "price": row["price_weight"]}
        check("selected_cbf_config.json weights match the selected tuning-grid row",
              recorded_weights == selected_config["weights"])

        # Independently re-evaluate the selected weights on VALIDATION and
        # confirm it reproduces the recorded validation NDCG -- and that the
        # tuning grid's best score is not accidentally the test-set number.
        cbf_check = ContentBasedRecommender(data.workers, data.requests, selected_config["weights"], eval_cfg.cbf_price_neutral_score)
        _, val_summary = evaluate(cbf_check, val_requests, eval_cfg.k)
        check("re-evaluating the selected weights on validation reproduces the recorded validation NDCG",
              abs(val_summary["ndcg_at_k"] - row["ndcg_at_k"]) < 1e-9,
              f"recomputed {val_summary['ndcg_at_k']} vs recorded {row['ndcg_at_k']}")

        best_val_ndcg = tuning_df["ndcg_at_k"].max()
        check("selected row has the best (or tied-best) validation NDCG in the tuning grid",
              abs(row["ndcg_at_k"] - best_val_ndcg) < 1e-9)

    # ---- Metric ranges ------------------------------------------------------
    for filename in ("validation_metrics.csv", "test_metrics.csv", "cold_start_metrics.csv"):
        df = pd.read_csv(RESULTS_DIR / filename)
        for metric in ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"):
            in_range = df[metric].between(0, 1).all()
            check(f"{filename}: {metric} within [0, 1]", bool(in_range))

    check("results/day4 manifest exists and is well-formed",
          "files" in json.loads((RESULTS_DIR / "day4_manifest.json").read_text(encoding="utf-8")))

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
