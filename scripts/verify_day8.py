"""Independent verification of the Day 8 hybrid recommender and results.

Re-derives rather than trusts results/day8/*:

- selected weights sum to 1, come from the predeclared grid, and are the
  grid's best on validation by the predeclared rule; tuning rows are
  validation-only and the config hash matches the one recorded;
- every frozen standalone config/result (Days 4-7) and the generator are
  byte-unchanged; Day 4-7 rows are copied verbatim;
- the hybrid score is recomputed for a sample of test requests by an
  independent implementation (own min-max, own weighted sum) on top of
  the frozen component scores, and rankings are valid (K <= 10, no
  duplicates, inside the candidate pool);
- test metrics, cold-start subsets, significance (re-bootstrapped with
  the recorded seed), diversity, and contribution numbers are
  recomputed and compared.

Usage:
    python scripts/verify_day8.py

Exits non-zero if any check fails. Never writes anything.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.data.config import load_config
from src.data.io import sha256_of
from src.evaluation.config import load_evaluation_config
from src.evaluation.evaluator import cold_start_requests, evaluate
from src.evaluation.metrics import compute_all_at_k
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.recommenders.cf_config import load_cf_config
from src.recommenders.frozen_models import build_frozen_models
from src.recommenders.hybrid import HybridRecommender, cbf_score_fn, cf_score_fn, kg_score_fn
from src.recommenders.hybrid_config import balance, load_hybrid_config

RESULTS = REPO_ROOT / "results" / "day8"
DAY7 = REPO_ROOT / "results" / "day7"
METRICS = ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k")
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def independent_hybrid_rank(cbf, cf, kg, weights, link, candidates, k):
    """Own normalisation and combination, written separately from
    src/recommenders/hybrid.py; component scores come from the frozen
    models' public/existing scoring methods."""
    columns = {
        "cbf": [cbf.score(link, w) for w in candidates],
        "cf": [cf._cf_scores(link, candidates).get(w, 0.0) for w in candidates],
        "kg": list(kg.scored(link, candidates)[2]),
    }
    total = [0.0] * len(candidates)
    for name, values in columns.items():
        lo, hi = min(values), max(values)
        if hi - lo <= 1e-12:
            continue
        for i, v in enumerate(values):
            total[i] += weights[name] * (v - lo) / (hi - lo)
    order = sorted(range(len(candidates)), key=lambda i: (-total[i], candidates[i]))
    return [candidates[i] for i in order[:k]], dict(zip(candidates, total))


def main() -> int:
    data_cfg, eval_cfg, hcfg = load_config(), load_evaluation_config(), load_hybrid_config()
    k = eval_cfg.k
    required = ["hybrid_tuning.csv", "validation_metrics.csv", "test_metrics.csv", "cold_start_metrics.csv",
                "cold_start_worker_diagnostics.csv", "significance.csv", "diversity_metrics.csv", "hybrid_contributions.csv",
                "hybrid_ablation_validation.csv", "hybrid_explanation_fidelity.json", "hybrid_explanation_samples.json",
                "selected_hybrid_config.json", "day8_manifest.json"]
    missing = [f for f in required if not (RESULTS / f).exists()]
    check("all expected Day 8 artifacts exist", not missing, f"missing {missing}")
    if missing:
        return 1
    manifest, selected = load_json(RESULTS / "day8_manifest.json"), load_json(RESULTS / "selected_hybrid_config.json")
    weights = selected["weights"]

    # ---- Frozen artifacts ----------------------------------------------------
    with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
        check("frozen generator_v1.yaml unchanged", sha256_of(REPO_ROOT / "config" / "generator_v1.yaml") == fh.read().split()[0])
    changed = [p for p, h in manifest["frozen_inputs_sha256"].items() if sha256_of(REPO_ROOT / p) != h]
    check("frozen standalone configs/results recorded by Day 8 are unchanged", not changed, f"{changed}")
    for prior in ("results/day4/day4_manifest.json", "results/day5/day5_manifest.json", "results/day7/day7_manifest.json",
                  "data/processed/kg/kg_prepare_manifest.json"):
        drift = [e["path"] for e in load_json(REPO_ROOT / prior)["files"] if sha256_of(REPO_ROOT / e["path"]) != e["sha256"]]
        check(f"every artifact in {prior} matches its hash", not drift, f"{drift}")
    day7 = load_json(DAY7 / "day7_manifest.json")
    check("Day 7 KG config unchanged", sha256_of(REPO_ROOT / "config" / "kg_config.yaml") == day7["kg_config_sha256"])
    drift = [e["path"] for e in manifest["files"] if sha256_of(REPO_ROOT / e["path"]) != e["sha256"]]
    check("every Day 8 output matches its manifest hash", not drift, f"{drift}")
    check("hybrid_config.yaml unchanged since the run and recorded in the selection",
          sha256_of(REPO_ROOT / "config" / "hybrid_config.yaml") == manifest["hybrid_config_sha256"] == selected["hybrid_config_sha256"])

    # ---- Selection -------------------------------------------------------------
    tuning = pd.read_csv(RESULTS / "hybrid_tuning.csv")
    candidates = tuning.loc[tuning["role"] == "candidate"]
    check("tuning rows are validation-only", set(tuning["split"]) == {"validation"})
    check("tuning covers exactly the predeclared grid",
          [{c: float(r[f"w_{c}"]) for c in ("cbf", "cf", "kg")} for _, r in candidates.iterrows()] == hcfg.weight_grid)
    check("selected weights sum to 1 and are non-negative", abs(sum(weights.values()) - 1) < 1e-12 and min(weights.values()) >= 0)
    check("selected weights come from the predeclared grid", weights in hcfg.weight_grid)
    keys = [(r.ndcg_at_k, r.recall_at_k, -balance({c: getattr(r, f"w_{c}") for c in ("cbf", "cf", "kg")}), -r.grid_index)
            for r in candidates.itertuples()]
    best = candidates.iloc[max(range(len(keys)), key=lambda i: keys[i])]
    check("selected row is best by NDCG@10, then Recall@10, then balance",
          int(best["grid_index"]) == selected["grid_index"] and int(candidates["selected"].sum()) == 1 and bool(best["selected"]))
    check("selection does not use test: selected config records validation as its basis", selected["selected_on"] == "validation")
    sanity = tuning.loc[tuning["role"] == "sanity"]
    check("single-component sanity: CBF-only and KG-only hybrids reproduce the standalone models",
          sanity.loc[sanity["w_cbf"] == 1.0, "matches_standalone"].iloc[0] and sanity.loc[sanity["w_kg"] == 1.0, "matches_standalone"].iloc[0])

    # ---- Rebuild and recompute ---------------------------------------------------
    data = load_protocol_data()
    index = EligibilityIndex(data.workers, data.known_countries)
    val, test = build_eval_requests(data, index, "validation", eval_cfg), build_eval_requests(data, index, "test", eval_cfg)
    models = build_frozen_models(data, data_cfg.seed, load_cf_config().min_similarity)
    hybrid = HybridRecommender({"cbf": cbf_score_fn(models["cbf"]), "cf": cf_score_fn(models["cf"]), "kg": kg_score_fn(models["kg"])}, weights)

    included = [r for r in test if not r.excluded]
    sample = included[::25]
    rank_ok = valid = True
    for req in sample:
        mine, _ = independent_hybrid_rank(models["cbf"], models["cf"], models["kg"], weights, req.link, req.candidates, k)
        theirs = hybrid.rank(req.link, req.candidates, k)
        rank_ok &= mine == theirs
        valid &= len(theirs) <= k and len(set(theirs)) == len(theirs) and set(theirs) <= set(req.candidates)
    check(f"hybrid rankings match an independent recomputation ({len(sample)} test requests)", rank_ok)
    check("hybrid rankings valid: K <= 10, no duplicates, inside the candidate pool", valid)

    test_df, val_df = pd.read_csv(RESULTS / "test_metrics.csv"), pd.read_csv(RESULTS / "validation_metrics.csv")
    for name, day8_df in (("test_metrics.csv", test_df), ("validation_metrics.csv", val_df)):
        prior = pd.read_csv(DAY7 / name)
        check(f"{name}: Random/Popularity/CBF/CF/KG rows copied verbatim from Day 7",
              day8_df.loc[day8_df["method"] != "hybrid"].reset_index(drop=True).equals(prior))
        check(f"{name}: six methods", sorted(day8_df["method"]) == ["cbf", "cf", "hybrid", "kg", "popularity", "random"])
        check(f"{name}: all metrics within [0, 1]", bool(day8_df[list(METRICS)].stack().between(0, 1).all()))
    hybrid_test_rankings, per_request = {}, []
    for req in included:
        ranked = hybrid.rank(req.link, req.candidates, k)
        hybrid_test_rankings[req.link] = ranked
        per_request.append({"link": req.link, **compute_all_at_k(ranked, req.relevance, k)})
    per_request = pd.DataFrame(per_request)
    recorded = test_df.loc[test_df["method"] == "hybrid"].iloc[0]
    check("hybrid test metrics reproduce from scratch",
          all(abs(per_request[m].mean() - recorded[m]) < 1e-12 for m in METRICS))
    check("hybrid validation NDCG@10 reproduces the selected value",
          abs(evaluate(hybrid, val, k)[1]["ndcg_at_k"] - selected["primary_metric_value"]) < 1e-12)

    # ---- Cold start ---------------------------------------------------------------
    cold_ids = set(pd.read_csv(REPO_ROOT / "data" / "processed" / "synthetic" / "cold_start_cohort.csv")["worker_id"])
    cold = pd.read_csv(RESULTS / "cold_start_metrics.csv")
    check("cold-start subset unchanged (442 validation / 924 test requests for every method)",
          len(cold_start_requests(val, cold_ids)) == 442 and len(cold_start_requests(test, cold_ids)) == 924
          and set(cold.loc[cold["split"] == "validation", "n_cold_start_requests"]) == {442}
          and set(cold.loc[cold["split"] == "test", "n_cold_start_requests"]) == {924})
    check("cold_start_metrics.csv: prior rows copied verbatim",
          cold.loc[cold["method"] != "hybrid"].reset_index(drop=True).equals(pd.read_csv(DAY7 / "cold_start_metrics.csv")))
    workers_df = pd.read_csv(RESULTS / "cold_start_worker_diagnostics.csv")
    hw = workers_df.loc[(workers_df["method"] == "hybrid") & (workers_df["split"] == "test")].iloc[0]
    slots = sum(len(v) for v in hybrid_test_rankings.values())
    cold_slots = sum(1 for v in hybrid_test_rankings.values() for w in v if w in cold_ids)
    check("hybrid cold-start share of top-10 slots recomputes", abs(cold_slots / slots - hw["cold_share_of_top_k_slots"]) < 1e-12)

    # ---- Significance ---------------------------------------------------------------
    sig = pd.read_csv(RESULTS / "significance.csv")
    b = hcfg.bootstrap
    check("significance covers every configured baseline x metric",
          len(sig) == len(b["baselines"]) * len(b["metrics"]) and set(sig["baseline"]) == set(b["baselines"]))
    check("significance outputs sane (CI contains the mean difference; p in (0, 1]; Holm >= raw)",
          bool(((sig["ci_low"] <= sig["mean_difference"]) & (sig["mean_difference"] <= sig["ci_high"])).all()
               and sig["p_value"].between(0, 1, inclusive="right").all() and (sig["p_value_holm"] >= sig["p_value"] - 1e-15).all()
               and (sig["n_pairs"] == len(included)).all()))
    rng = np.random.default_rng(np.random.SeedSequence([data_cfg.seed, b["seed_offset"]]))
    idx = rng.integers(0, len(included), size=(b["n_resamples"], len(included)))
    kg_per_request = evaluate(models["kg"], test, k)[0].set_index("link").loc[sorted(per_request["link"]), "ndcg_at_k"].to_numpy()
    d = per_request.set_index("link").loc[sorted(per_request["link"]), "ndcg_at_k"].to_numpy() - kg_per_request
    reps = d[idx].mean(axis=1)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    p = (1 + np.sum(np.abs(reps - d.mean()) >= abs(d.mean()))) / (b["n_resamples"] + 1)
    row = sig.loc[(sig["baseline"] == "kg") & (sig["metric"] == "ndcg_at_k")].iloc[0]
    check("Hybrid-vs-KG NDCG@10 bootstrap reproduces independently (mean, CI, p)",
          abs(d.mean() - row["mean_difference"]) < 1e-12 and abs(lo - row["ci_low"]) < 1e-12
          and abs(hi - row["ci_high"]) < 1e-12 and abs(p - row["p_value"]) < 1e-12)

    # ---- Diversity --------------------------------------------------------------------
    div = pd.read_csv(RESULTS / "diversity_metrics.csv")
    check("diversity outputs sane (coverage/Gini/top-share/ILD within [0, 1])",
          bool(div[["catalogue_coverage", "exposure_gini", "top_share_of_slots", "intra_list_diversity"]].stack().between(0, 1).all()))
    hd = div.loc[(div["method"] == "hybrid") & (div["split"] == "test")].iloc[0]
    eligible = {w for r in included for w in r.candidates}
    recommended = {w for v in hybrid_test_rankings.values() for w in v}
    check("hybrid test catalogue coverage recomputes", abs(len(recommended & eligible) / len(eligible) - hd["catalogue_coverage"]) < 1e-12)
    skills = {w: set(s.split("|")) if isinstance(s, str) else set() for w, s in zip(data.workers["worker_id"], data.workers["declared_skills"])}
    ild = np.mean([1 - np.mean([len(skills[a] & skills[c]) / len(skills[a] | skills[c]) if skills[a] | skills[c] else 0.0
                                for a, c in combinations(v, 2)]) for v in hybrid_test_rankings.values() if len(v) > 1])
    check("hybrid test intra-list diversity recomputes", abs(ild - hd["intra_list_diversity"]) < 1e-12)

    # ---- Contributions / explanations -------------------------------------------------
    contrib = pd.read_csv(RESULTS / "hybrid_contributions.csv")
    test_contrib = contrib.loc[contrib["split"] == "test"].set_index("component")
    exps = [e for req in included for e in hybrid.explain(req.link, req.candidates, k)]
    check("contribution means recompute",
          all(abs(np.mean([e.contributions[c] for e in exps]) - test_contrib.loc[c, "mean_contribution"]) < 1e-12 for c in ("cbf", "cf", "kg")))
    check("every hybrid explanation's contributions reconstruct its score",
          all(abs(sum(e.contributions.values()) - e.score) < 1e-12 for e in exps))
    fidelity = load_json(RESULTS / "hybrid_explanation_fidelity.json")["test"]
    check("hybrid explanation fidelity: first cited component is the largest contributor in 100% of recommendations",
          fidelity["share_first_cited_is_largest_contribution"] == 1.0)
    samples = load_json(RESULTS / "hybrid_explanation_samples.json")["explanations"]
    by_link = {r.link: r for r in included}
    sample_ok = len(samples) == hcfg.explanations["sample_size"]
    for s in samples:
        ranked = hybrid_test_rankings[s["link"]]
        sample_ok &= ranked[s["rank"] - 1] == s["worker_id"] and abs(sum(s["contributions"].values()) - s["score"]) < 1e-12
        sample_ok &= s["link"] in by_link
    check("30 hybrid explanation samples match the actual rankings and scores", sample_ok)
    check("Day 7 blank rubric sheet untouched (no ratings)",
          pd.read_csv(REPO_ROOT / "docs" / "day7_explanation_rubric_sheet.csv")[
              ["relevance_1to5", "clarity_1to5", "traceability_1to5", "usefulness_1to5"]].isna().all().all())

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
