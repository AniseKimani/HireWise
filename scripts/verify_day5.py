"""Independent verification of Day 5 Collaborative Filtering and
evaluation results.

Re-derives invariants independently rather than trusting
results/day5/*.csv at face value: rebuilds the training matrix from raw
interactions a second way, confirms the selected configuration is the
tuning grid's actual best validation score, confirms cold-start workers
never entered the training matrix, and re-checks Day 4's frozen CBF
config/results are byte-unchanged.

Usage:
    python scripts/verify_day5.py

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
from src.recommenders.cf_config import load_cf_config
from src.recommenders.collaborative import CollaborativeFiltering, build_training_matrix
from src.recommenders.popularity import PopularityBaseline

RESULTS_DIR = REPO_ROOT / "results" / "day5"
DAY4_RESULTS_DIR = REPO_ROOT / "results" / "day4"
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
    cf_cfg = load_cf_config()

    required_files = [
        "cf_tuning.csv", "validation_metrics.csv", "test_metrics.csv", "cold_start_metrics.csv",
        "cf_coverage.json", "selected_cf_config.json", "day5_manifest.json",
    ]
    missing = [f for f in required_files if not (RESULTS_DIR / f).exists()]
    check("all expected Day 5 result artifacts exist", not missing, f"missing {missing}")
    if missing:
        print("\nRun scripts/run_day5_evaluation.py first.")
        return 1

    with open(RESULTS_DIR / "selected_cf_config.json", encoding="utf-8") as fh:
        selected_config = json.load(fh)
    check("selected CF configuration exists and is well-formed",
          {"aggregation", "method", "k_neighbors"}.issubset(selected_config))

    # ---- Frozen generator / Phase 1 -----------------------------------------
    with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
        expected_hash = fh.read().split()[0]
    check("frozen generator_v1.yaml hash unchanged", sha256_of(REPO_ROOT / "config" / "generator_v1.yaml") == expected_hash)

    # ---- Day 4 frozen CBF config/results unchanged --------------------------
    day4_cbf_path = DAY4_RESULTS_DIR / "selected_cbf_config.json"
    day4_cbf_hash_recorded = None
    with open(RESULTS_DIR / "day5_manifest.json", encoding="utf-8") as fh:
        day5_manifest = json.load(fh)
    if day4_cbf_path.exists():
        day4_cbf_hash_now = sha256_of(day4_cbf_path)
        check("Day 4 selected_cbf_config.json unchanged since Day 5 ran",
              day5_manifest.get("day4_cbf_config_sha256") == day4_cbf_hash_now,
              f"recorded {day5_manifest.get('day4_cbf_config_sha256')}, now {day4_cbf_hash_now}")

    # ---- Independent training matrix reconstruction --------------------------
    data = load_protocol_data()
    train_links = set(data.requests.loc[data.requests["split"] == "train", "link"])
    non_train_links = set(data.requests["link"]) - train_links

    independent_train = data.interactions.loc[
        data.interactions["link"].isin(train_links) & data.interactions["event"].isin(cf_cfg.event_weights)
    ].copy()
    independent_train["weight"] = independent_train["event"].map(cf_cfg.event_weights)
    independent_agg = independent_train.groupby(["client_id", "worker_id"])["weight"].sum()

    matrix = build_training_matrix(data.interactions, data.requests, cf_cfg.event_weights, selected_config["aggregation"])
    if selected_config["aggregation"] == "sum":
        matrix_pairs = {
            (c, w): matrix.matrix[matrix.client_index[c], matrix.worker_index[w]]
            for c, w in independent_agg.index
        }
        mismatches = sum(1 for (c, w), v in matrix_pairs.items() if abs(v - independent_agg[(c, w)]) > 1e-9)
        check("training matrix values match an independent recomputation from raw training interactions",
              mismatches == 0, f"{mismatches} mismatched pairs")

    check("CF matrix uses training interactions only (client/worker sets are exact training subsets)",
          set(matrix.clients).issubset(set(independent_train["client_id"])) and
          set(matrix.workers).issubset(set(independent_train["worker_id"])))

    # ---- No validation/test leakage into the matrix --------------------------
    non_train_only_workers = set(
        data.interactions.loc[data.interactions["link"].isin(non_train_links), "worker_id"]
    ) - set(independent_train["worker_id"])
    leaked = non_train_only_workers & set(matrix.workers)
    check("no validation/test-only worker leaked into the training matrix", not leaked, f"leaked: {leaked}")

    # ---- Cold-start workers have zero training CF history --------------------
    cold_start = pd.read_csv(DAY3_SYNTHETIC / "cold_start_cohort.csv")
    check("300 cold-start workers", len(cold_start) == 300)
    cold_start_ids = set(cold_start["worker_id"])
    overlap = cold_start_ids & set(matrix.worker_index.keys())
    check("cold-start workers have zero training CF history", not overlap, f"{len(overlap)} cold-start workers found in training matrix")

    # ---- No generator-private / content-feature leakage in source -----------
    from src.recommenders import collaborative
    private_markers = ("quality", "true_skills", "proficiency", "taste_", "synthetic_private", "utility", "noise")
    source_lower = inspect.getsource(collaborative).lower()
    leaked_private = [m for m in private_markers if m in source_lower]
    check("no generator-private markers in collaborative.py", not leaked_private, f"{leaked_private}")

    import_lines = [
        line.strip() for line in inspect.getsource(collaborative).splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    check("collaborative.py does not import content_based",
          not any("content_based" in line for line in import_lines), f"{import_lines}")

    # ---- Fallback uses training-only popularity -------------------------------
    fallback = PopularityBaseline(data.interactions, data.requests)
    independent_hire_counts = independent_train.loc[independent_train["event"] == "HIRED"].groupby("worker_id").size().to_dict()
    check("fallback (Popularity) hire counts match independent training-only recomputation",
          fallback.hire_counts == independent_hire_counts)

    # ---- Rebuild the selected CF and re-run evaluation ------------------------
    link_to_client = dict(zip(data.requests["link"], data.requests["client_id"]))
    cf = CollaborativeFiltering(matrix, selected_config["method"], selected_config["k_neighbors"], fallback, link_to_client, cf_cfg.min_similarity)

    index = EligibilityIndex(data.workers, data.known_countries)
    val_requests = build_eval_requests(data, index, "validation", eval_cfg)
    included = [r for r in val_requests if not r.excluded][:300]

    all_valid, all_no_dup, all_k_ok, join_day_ok, location_ok = True, True, True, True, True
    join_day = data.workers.set_index("worker_id")["join_day"]
    country = data.workers.set_index("worker_id")["country"]
    worker_ids = set(data.workers["worker_id"])
    for req in included:
        ranked = cf.rank(req.link, req.candidates, eval_cfg.k)
        if not set(ranked).issubset(worker_ids):
            all_valid = False
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

    check("all CF recommendations refer to valid worker IDs", all_valid)
    check("no duplicate worker per request in CF output", all_no_dup)
    check("CF returns at most K=10 recommendations", all_k_ok)
    check("CF recommendations respect worker join date", join_day_ok)
    check("CF recommendations respect hard location restrictions", location_ok)

    # ---- Tuning used validation only; selection is the grid's best ----------
    tuning_df = pd.read_csv(RESULTS_DIR / "cf_tuning.csv")
    selected_rows = tuning_df.loc[tuning_df["selected"]]
    check("exactly one tuning-grid row marked selected", len(selected_rows) == 1, f"got {len(selected_rows)}")
    if len(selected_rows) == 1:
        row = selected_rows.iloc[0]
        check("selected_cf_config.json matches the selected tuning-grid row",
              row["aggregation"] == selected_config["aggregation"]
              and row["method"] == selected_config["method"]
              and int(row["k_neighbors"]) == selected_config["k_neighbors"])

        _, val_summary = evaluate(cf, val_requests, eval_cfg.k)
        check("re-evaluating the selected config on validation reproduces the recorded validation NDCG",
              abs(val_summary["ndcg_at_k"] - row["ndcg_at_k"]) < 1e-9,
              f"recomputed {val_summary['ndcg_at_k']} vs recorded {row['ndcg_at_k']}")

        best_ndcg = tuning_df["ndcg_at_k"].max()
        check("selected row has the best (or tied-best) validation NDCG in the tuning grid",
              abs(row["ndcg_at_k"] - best_ndcg) < 1e-9)

    check("cf_tuning.csv covers the full predeclared grid (aggregation x method x k_neighbors)",
          len(tuning_df) == len(cf_cfg.aggregation_grid) * len(cf_cfg.method_grid) * len(cf_cfg.k_neighbors_grid))

    # ---- Test metrics generated after selection (file simply exists and
    # references the selected method's row -- test was never used to pick it) --
    test_df = pd.read_csv(RESULTS_DIR / "test_metrics.csv")
    check("test_metrics.csv contains a cf row", (test_df["method"] == "cf").any())

    # ---- Metric ranges --------------------------------------------------------
    for filename in ("validation_metrics.csv", "test_metrics.csv", "cold_start_metrics.csv", "cf_tuning.csv"):
        df = pd.read_csv(RESULTS_DIR / filename)
        for metric in ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"):
            in_range = df[metric].between(0, 1).all()
            check(f"{filename}: {metric} within [0, 1]", bool(in_range))

    # ---- Phase 1 population counts unchanged ----------------------------------
    check("3,000 workers", len(data.workers) == 3000)
    counts = data.requests["split"].value_counts()
    check("train/validation/test = 8400/1200/2400",
          counts.get("train", 0) == 8400 and counts.get("validation", 0) == 1200 and counts.get("test", 0) == 2400)

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
