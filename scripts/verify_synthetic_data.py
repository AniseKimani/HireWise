"""Independent verification of the Day 3 synthetic dataset.

Re-derives invariants from the generated output files directly (not from
generation_summary.json) so this is a genuine independent check rather
than a restatement of numbers the generator already produced. Mirrors
scripts/verify_pipeline.py's approach for Day 2.

Usage:
    python scripts/verify_synthetic_data.py [--synthetic-dir PATH] [--private-dir PATH] [--day2-dir PATH]

Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.config import load_config

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", default=str(REPO_ROOT / "data" / "processed" / "synthetic"))
    parser.add_argument("--private-dir", default=str(REPO_ROOT / "data" / "processed" / "synthetic_private"))
    parser.add_argument("--day2-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "data_config.yaml"))
    args = parser.parse_args()

    synthetic_dir = Path(args.synthetic_dir)
    private_dir = Path(args.private_dir)
    day2_dir = Path(args.day2_dir)
    config = load_config(args.config)

    required = ["clients.csv", "workers.csv", "requests.csv", "interactions.csv", "ratings.csv", "cold_start_cohort.csv"]
    missing = [f for f in required if not (synthetic_dir / f).exists()]
    check("all expected model-visible output files exist", not missing, f"missing {missing}")
    if missing:
        print("\nRun scripts/generate_synthetic_data.py first.")
        return 1

    clients = pd.read_csv(synthetic_dir / "clients.csv")
    workers = pd.read_csv(synthetic_dir / "workers.csv")
    requests = pd.read_csv(synthetic_dir / "requests.csv")
    interactions = pd.read_csv(synthetic_dir / "interactions.csv")
    ratings = pd.read_csv(synthetic_dir / "ratings.csv")
    cold_start = pd.read_csv(synthetic_dir / "cold_start_cohort.csv")
    sampled_requests = pd.read_csv(day2_dir / "sampled_requests.csv")
    vocab = set(pd.read_csv(day2_dir / "skills_vocab.csv")["skill"])

    # ---- Population -----------------------------------------------------
    check("exactly 3,000 workers", len(workers) == config.synthetic_workers, f"got {len(workers)}")
    check("exactly 2,000 clients", len(clients) == config.synthetic_clients, f"got {len(clients)}")
    check("exactly 12,000 requests", len(requests) == config.model_requests, f"got {len(requests)}")
    expected_cold_start = round(config.synthetic_workers * config.cold_start_ratio)
    check("exactly 300 cold-start workers", len(cold_start) == expected_cold_start, f"got {len(cold_start)}")

    # ---- Identity ---------------------------------------------------------
    check("worker IDs unique", workers["worker_id"].is_unique)
    check("client IDs unique", clients["client_id"].is_unique)
    check("worker ID format WNNNNNN", workers["worker_id"].str.match(r"^W\d{6}$").all())
    check("client ID format CNNNNNN", clients["client_id"].str.match(r"^C\d{6}$").all())

    # ---- No orphan foreign keys -------------------------------------------
    check("no orphan client_id in interactions", interactions["client_id"].isin(clients["client_id"]).all())
    check("no orphan worker_id in interactions", interactions["worker_id"].isin(workers["worker_id"]).all())
    check("no orphan link in interactions", interactions["link"].isin(requests["link"]).all())
    check("no orphan worker_id in ratings", ratings["worker_id"].isin(workers["worker_id"]).all())
    check("request links map onto Day 2 sampled requests exactly", set(requests["link"]) == set(sampled_requests["link"]))

    # ---- Controlled vocabulary ---------------------------------------------
    out_of_vocab = 0
    for cell in workers["declared_skills"].dropna():
        skills = cell.split("|") if cell else []
        out_of_vocab += sum(1 for s in skills if s not in vocab)
    check("all declared worker skills are in the controlled vocabulary", out_of_vocab == 0, f"{out_of_vocab} out-of-vocab skills")

    # ---- Cold start ---------------------------------------------------------
    cold_ids = set(cold_start["worker_id"])
    train_links = set(requests.loc[requests["split"] == "train", "link"])
    cold_in_train = interactions.loc[interactions["link"].isin(train_links), "worker_id"].isin(cold_ids).sum()
    check("cold-start workers have zero training interactions", cold_in_train == 0, f"{cold_in_train} found")

    # ---- Temporal split -----------------------------------------------------
    merged = requests.set_index("link")
    train_max = merged.loc[merged["split"] == "train", "simulated_day"].max()
    val_min = merged.loc[merged["split"] == "validation", "simulated_day"].min()
    val_max = merged.loc[merged["split"] == "validation", "simulated_day"].max()
    test_min = merged.loc[merged["split"] == "test", "simulated_day"].min()
    check("train precedes validation on the simulated timeline", train_max <= val_min)
    check("validation precedes test on the simulated timeline", val_max <= test_min)
    check("train/validation/test are non-overlapping and exhaustive",
          set(requests["split"].unique()) == {"train", "validation", "test"} and requests["link"].nunique() == len(requests))

    counts = requests["split"].value_counts()
    check("split counts match configured ratios",
          counts["train"] == round(len(requests) * config.train_ratio)
          and counts["validation"] == round(len(requests) * config.validation_ratio),
          f"got {counts.to_dict()}")

    # ---- No future interaction contributes to training --------------------
    join_day = workers.set_index("worker_id")["join_day"]
    interactions_with_join = interactions.copy()
    interactions_with_join["join_day"] = interactions_with_join["worker_id"].map(join_day)
    check("no interaction precedes its worker's join day",
          (interactions_with_join["join_day"] <= interactions_with_join["simulated_day"]).all())

    # ---- Required exclusion/validity flags ---------------------------------
    required_worker_cols = {"is_cold_start", "join_day", "primary_category", "declared_skills", "price_tier", "country"}
    check("workers.csv carries required audit columns", required_worker_cols.issubset(workers.columns),
          f"missing {required_worker_cols - set(workers.columns)}")

    # ---- Ratings ----------------------------------------------------------
    check("all ratings are in [1, 5]", ratings["rating"].between(1, 5).all())
    hired_pairs = set(zip(
        interactions.loc[interactions["event"] == "HIRED", "link"],
        interactions.loc[interactions["event"] == "HIRED", "worker_id"],
    ))
    rating_pairs = set(zip(ratings["link"], ratings["worker_id"]))
    check("ratings exist only for hired interactions", rating_pairs.issubset(hired_pairs))

    # ---- Capacity -----------------------------------------------------------
    hires = interactions.loc[interactions["event"] == "HIRED"]
    capacity_violations = 0
    for worker_id, group in hires.groupby("worker_id"):
        days = sorted(group["simulated_day"])
        for i, day in enumerate(days):
            if sum(1 for d in days if day - 30 < d <= day) > 4:
                capacity_violations += 1
                break
    check("no worker exceeds 4 hires in any rolling 30-day window", capacity_violations == 0, f"{capacity_violations} workers")

    # ---- Location eligibility -----------------------------------------------
    restricted = sampled_requests.loc[sampled_requests["location_requirement"].notna(), ["link", "location_requirement"]]
    worker_country = workers.set_index("worker_id")["country"]
    restricted_hires = hires.merge(restricted, on="link", how="inner")
    ineligible = sum(
        1 for row in restricted_hires.itertuples() if worker_country[row.worker_id] not in row.location_requirement
    )
    check("no ineligible hire for a location-restricted request", ineligible == 0, f"{ineligible} violations")

    # ---- No Dataset 2 dependency -------------------------------------------
    check("no dataset2/second-dataset file referenced in synthetic outputs",
          not any("dataset2" in f.name.lower() or "dataset_2" in f.name.lower() for f in synthetic_dir.glob("*")))

    # ---- Leakage: private columns never in model-visible files ------------
    private_markers = {"quality", "true_skills", "proficiency", "utility", "noise",
                        "S_raw", "T_raw", "Q_raw", "B_raw", "C_raw", "S_std", "T_std", "Q_std", "B_std", "C_std"}
    leaked = []
    for filename in required:
        cols = pd.read_csv(synthetic_dir / filename, nrows=1).columns
        leaked += [c for c in cols if c in private_markers or c.startswith("taste_")]
    check("no generator-private column appears in any model-visible file", not leaked, f"leaked: {leaked}")

    # ---- Determinism (rerun sampling-independent pieces) --------------------
    from src.data.generator_config import load_generator_config
    from src.data.synth_cold_start import select_cold_start_cohort

    gen_config = load_generator_config(config.generator_config_path)
    cohort_a = select_cold_start_cohort(workers.drop(columns=["is_cold_start", "join_day"], errors="ignore"), config)
    cohort_b = select_cold_start_cohort(workers.drop(columns=["is_cold_start", "join_day"], errors="ignore"), config)
    check("deterministic reruns: cold-start selection agrees given the same inputs", cohort_a == cohort_b)

    # ---- Config consistency --------------------------------------------------
    ratio_sum = config.train_ratio + config.validation_ratio + config.test_ratio
    check("config split ratios sum to 1.0", abs(ratio_sum - 1.0) < 1e-9)

    if private_dir.exists():
        calibration = json.loads((private_dir / "calibration.json").read_text(encoding="utf-8"))
        check("calibrated hire-rate estimate is close to the configured target",
              abs(calibration["pass1_hire_rate_estimate"] - gen_config.hire_rate_target) < 0.02,
              f"got {calibration['pass1_hire_rate_estimate']}")

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
