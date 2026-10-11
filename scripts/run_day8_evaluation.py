"""Day 8 weighted hybrid evaluation pipeline.

Usage (from repository root):

    python scripts/run_day8_evaluation.py

Order of operations (the order is part of the method):

1. Rebuild the frozen Random/Popularity/CBF/CF/KG models from their
   recorded selections and assert each reproduces its frozen validation
   metrics exactly.
2. Evaluate the predeclared hybrid weight grid (config/hybrid_config.yaml)
   on VALIDATION only, select one configuration, and write
   results/day8/selected_hybrid_config.json.
3. Only then build the test requests and evaluate every method once on
   test (frozen models must reproduce their frozen test metrics).
4. Post-selection analysis only: paired bootstrap significance,
   cold-start, diversity, component contributions, validation ablation,
   and hybrid explanations. None of it feeds back into the weights.

Random/Popularity/CBF/CF/KG rows in the comparison tables are copied
verbatim from results/day7/*.csv. Neo4j is not contacted: the KG snapshot
is rebuilt from the Day 6 CSV artifacts and must match the fingerprint
Day 7 verified against Neo4j. Writes results to `results/day8/`.
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
import numpy as np
import pandas as pd

from src.data.config import load_config
from src.data.io import build_manifest, save_csv, save_json, save_manifest, sha256_of
from src.evaluation.cold_start import cold_worker_diagnostics
from src.evaluation.config import load_evaluation_config
from src.evaluation.diversity import diversity_summary
from src.evaluation.evaluator import cold_start_requests, evaluate
from src.evaluation.explainability import deterministic_sample
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.evaluation.significance import align_per_request, holm_adjust, paired_bootstrap
from src.recommenders.base import BaseRecommender
from src.recommenders.cf_config import load_cf_config
from src.recommenders.frozen_models import build_frozen_models
from src.recommenders.hybrid import COMPONENTS, HybridRecommender, cbf_score_fn, cf_score_fn, kg_score_fn, tune_hybrid
from src.recommenders.hybrid_config import load_hybrid_config

RESULTS_DIR = REPO_ROOT / "results" / "day8"
DAY7_RESULTS_DIR = REPO_ROOT / "results" / "day7"
DAY3_SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
COLD_START_PATH = DAY3_SYNTHETIC / "cold_start_cohort.csv"
METRICS = ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k")
FROZEN_METHODS = ("random", "popularity", "cbf", "cf", "kg")
METHOD_ORDER = FROZEN_METHODS + ("hybrid",)
METRIC_KEY = {"ndcg_at_10": "ndcg_at_k", "recall_at_10": "recall_at_k"}
FROZEN_INPUTS = [
    "config/generator_v1.yaml", "config/evaluation_config.yaml", "config/cf_config.yaml", "config/kg_config.yaml",
    "results/day4/selected_cbf_config.json", "results/day5/selected_cf_config.json",
    "results/day7/selected_kg_config.json", "results/day7/graph_snapshot.json",
    "results/day7/validation_metrics.csv", "results/day7/test_metrics.csv", "results/day7/cold_start_metrics.csv",
    "results/day7/cold_start_worker_diagnostics.csv", "results/day7/day7_manifest.json",
    "data/processed/kg/kg_prepare_manifest.json",
]


class RecordingRecommender(BaseRecommender):
    """Pass-through wrapper that keeps every ranking it returns, so one
    evaluation pass yields both metrics and the lists diversity needs."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.rankings: dict[str, list[str]] = {}

    def rank(self, link, candidates, k):
        ranked = self.inner.rank(link, candidates, k)
        self.rankings[link] = ranked
        return ranked


def assert_reproduces(name: str, summary: dict, frozen: pd.DataFrame, split: str) -> None:
    row = frozen.loc[frozen["method"] == name].iloc[0]
    for m in METRICS:
        if abs(summary[m] - row[m]) > 1e-12:
            raise SystemExit(f"rebuilt frozen {name} does not reproduce its frozen {split} {m}: {summary[m]} vs {row[m]}")


def weights_from(row) -> dict[str, float]:
    return {c: float(row[f"w_{c}"]) for c in COMPONENTS}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(test_table, tuning_df, significance_df, cold_table, contributions_df, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    width = 0.13
    for i, method in enumerate(test_table["method"]):
        values = test_table.loc[test_table["method"] == method, list(METRICS)].iloc[0]
        ax.bar([x + i * width for x in range(len(METRICS))], values, width, label=method)
    ax.set_xticks([x + 2.5 * width for x in range(len(METRICS))])
    ax.set_xticklabels(["Precision@10", "Recall@10", "F1@10", "NDCG@10"])
    ax.set_title("Model comparison at K=10 (test set)")
    ax.legend(ncol=3, fontsize=8)
    save(fig, "model_comparison_test_k10.png")

    tuning = tuning_df.loc[tuning_df["role"] == "candidate"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{r.w_cbf:.2f}/{r.w_cf:.2f}/{r.w_kg:.2f}" for r in tuning.itertuples()]
    ax.bar(labels, tuning["ndcg_at_k"], color=["#d62728" if s else "#1f77b4" for s in tuning["selected"]])
    ax.set_ylabel("Validation NDCG@10")
    ax.set_title("Hybrid validation-only tuning (CBF/CF/KG weights; selected in red)")
    ax.set_ylim(tuning["ndcg_at_k"].min() * 0.95, tuning["ndcg_at_k"].max() * 1.02)
    ax.tick_params(axis="x", rotation=60)
    save(fig, "hybrid_validation_tuning.png")

    sig = significance_df.loc[significance_df["metric"] == "ndcg_at_k"]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    y = range(len(sig))
    ax.errorbar(sig["mean_difference"], list(y), xerr=[sig["mean_difference"] - sig["ci_low"], sig["ci_high"] - sig["mean_difference"]],
                fmt="o", capsize=4)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"Hybrid - {b}" for b in sig["baseline"]])
    ax.set_xlabel("Test NDCG@10 difference (paired bootstrap 95% CI)")
    save(fig, "significance_ndcg_test.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    test_cold = cold_table.loc[cold_table["split"] == "test"]
    ax.bar(test_cold["method"], test_cold["ndcg_at_k"])
    ax.set_ylabel("NDCG@10 on cold-start requests (test)")
    ax.set_title("Cold-start requests at K=10")
    save(fig, "cold_start_comparison.png")

    test_contrib = contributions_df.loc[contributions_df["split"] == "test"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(test_contrib["component"], test_contrib["mean_share_of_score"], label="mean share of hybrid score")
    ax.bar(test_contrib["component"], test_contrib["share_dominant"], alpha=0.5, label="share of recs where dominant")
    ax.set_title("Hybrid component contributions (test top-10)")
    ax.legend(fontsize=8)
    save(fig, "hybrid_contributions_test.png")
    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> dict:
    data_cfg = load_config()
    eval_cfg = load_evaluation_config()
    cf_cfg = load_cf_config()
    hcfg = load_hybrid_config()
    k = eval_cfg.k
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []

    print("[1/9] Loading data, validation requests, frozen Day 7 tables; rebuilding frozen models")
    data = load_protocol_data()
    index = EligibilityIndex(data.workers, data.known_countries)
    cold_ids = set(pd.read_csv(COLD_START_PATH)["worker_id"])
    val_requests = build_eval_requests(data, index, "validation", eval_cfg)
    day7_val = pd.read_csv(DAY7_RESULTS_DIR / "validation_metrics.csv")
    day7_test = pd.read_csv(DAY7_RESULTS_DIR / "test_metrics.csv")
    day7_cold = pd.read_csv(DAY7_RESULTS_DIR / "cold_start_metrics.csv")
    day7_cold_workers = pd.read_csv(DAY7_RESULTS_DIR / "cold_start_worker_diagnostics.csv")
    models = build_frozen_models(data, data_cfg.seed, cf_cfg.min_similarity)

    recorded = {"validation": {}, "test": {}}
    per_request = {"validation": {}, "test": {}}
    for name in FROZEN_METHODS:
        rec = RecordingRecommender(models[name])
        df, summary = evaluate(rec, val_requests, k)
        assert_reproduces(name, summary, day7_val, "validation")
        recorded["validation"][name], per_request["validation"][name] = rec.rankings, df

    print("[2/9] Single-component sanity checks and validation-only tuning")
    fns = {"cbf": cbf_score_fn(models["cbf"]), "cf": cf_score_fn(models["cf"]), "kg": kg_score_fn(models["kg"])}
    base = HybridRecommender(fns, hcfg.weight_grid[0])
    sanity_rows = []
    if hcfg.sanity_single_component_checks:
        for c in COMPONENTS:
            weights = {x: (1.0 if x == c else 0.0) for x in COMPONENTS}
            _, summary = evaluate(base.with_weights(weights), val_requests, k)
            standalone = day7_val.loc[day7_val["method"] == c].iloc[0]
            sanity_rows.append({"role": "sanity", "grid_index": -1, **{f"w_{x}": weights[x] for x in COMPONENTS},
                                "balance": float("nan"), **summary, "selected": False,
                                "matches_standalone": all(abs(summary[m] - standalone[m]) <= 1e-12 for m in METRICS)})
        for row, c in zip(sanity_rows, COMPONENTS):
            if c in ("cbf", "kg") and not row["matches_standalone"]:
                raise SystemExit(f"single-component hybrid ({c}) does not reproduce standalone {c}: plumbing error")
    best, tuning_rows = tune_hybrid(base, hcfg.weight_grid, val_requests, k, hcfg.tuning_primary_metric, hcfg.tuning_tiebreak_metric)
    tuning_df = pd.DataFrame([{"role": "candidate", **r, "matches_standalone": None} for r in tuning_rows] + sanity_rows)
    tuning_df.insert(1, "split", "validation")
    out.append(save_csv(tuning_df, RESULTS_DIR / "hybrid_tuning.csv"))

    selected_row = tuning_df.loc[tuning_df["selected"]].iloc[0]
    selected_config = {
        "weights": best["weights"], "grid_index": best["grid_index"],
        "normalization": hcfg.normalization, "constant_component_policy": hcfg.constant_component_policy,
        "tie_break": "worker_id_ascending",
        "cf_score": "raw collaborative score (no popularity fallback in the score)",
        "primary_metric": hcfg.tuning_primary_metric, "primary_metric_value": float(selected_row["ndcg_at_k"]),
        "tiebreak_metric": hcfg.tuning_tiebreak_metric, "tiebreak_metric_value": float(selected_row["recall_at_k"]),
        "tuning_grid_size": len(hcfg.weight_grid), "selected_on": "validation",
        "hybrid_config_sha256": sha256_of(REPO_ROOT / "config" / "hybrid_config.yaml"),
    }
    out.append(save_json(selected_config, RESULTS_DIR / "selected_hybrid_config.json"))
    selected = base.with_weights(best["weights"])
    print(f"       selected (frozen before test): {best['weights']}")

    print("[3/9] Validation ablation (analysis only)")
    ablation_rows = []
    _, full = evaluate(selected, val_requests, k)
    ablation_rows.append({"variant": "full", "dropped_component": "", **{f"w_{c}": best["weights"][c] for c in COMPONENTS}, **full})
    for c in COMPONENTS:
        remaining = sum(v for x, v in best["weights"].items() if x != c)
        weights = {x: (0.0 if x == c else v / remaining) for x, v in best["weights"].items()}
        _, summary = evaluate(selected.with_weights(weights), val_requests, k)
        ablation_rows.append({"variant": f"without_{c}", "dropped_component": c, **{f"w_{x}": weights[x] for x in COMPONENTS}, **summary})
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.insert(0, "split", "validation")
    for m in ("ndcg_at_k", "recall_at_k"):
        ablation_df[f"delta_{m}_vs_full"] = ablation_df[m] - full[m]
    out.append(save_csv(ablation_df, RESULTS_DIR / "hybrid_ablation_validation.csv"))

    print("[4/9] Building test requests and evaluating every method once on test")
    test_requests = build_eval_requests(data, index, "test", eval_cfg)
    for name in FROZEN_METHODS:
        rec = RecordingRecommender(models[name])
        df, summary = evaluate(rec, test_requests, k)
        assert_reproduces(name, summary, day7_test, "test")
        recorded["test"][name], per_request["test"][name] = rec.rankings, df
    hybrid_summaries = {}
    for split, reqs in (("validation", val_requests), ("test", test_requests)):
        rec = RecordingRecommender(selected)
        df, summary = evaluate(rec, reqs, k)
        recorded[split]["hybrid"], per_request[split]["hybrid"], hybrid_summaries[split] = rec.rankings, df, summary
    validation_table = pd.concat([day7_val, pd.DataFrame([{"method": "hybrid", "split": "validation", **hybrid_summaries["validation"]}])], ignore_index=True)
    test_table = pd.concat([day7_test, pd.DataFrame([{"method": "hybrid", "split": "test", **hybrid_summaries["test"]}])], ignore_index=True)
    out.append(save_csv(validation_table, RESULTS_DIR / "validation_metrics.csv"))
    out.append(save_csv(test_table, RESULTS_DIR / "test_metrics.csv"))

    print("[5/9] Paired bootstrap significance on test")
    b = hcfg.bootstrap
    sig_rows = []
    for metric in b["metrics"]:
        rows = []
        for baseline in b["baselines"]:
            ha, hb, _ = align_per_request(per_request["test"]["hybrid"], per_request["test"][baseline], metric)
            result = paired_bootstrap(ha, hb, b["n_resamples"], [data_cfg.seed, b["seed_offset"]], b["confidence_level"])
            rows.append({"metric": metric, "comparison": f"hybrid_vs_{baseline}", "baseline": baseline, "split": "test",
                         **{("mean_hybrid" if key == "mean_a" else "mean_baseline" if key == "mean_b" else key): v
                            for key, v in result.items()}})
        adjusted = holm_adjust([r["p_value"] for r in rows])
        for r, p in zip(rows, adjusted):
            r["p_value_holm"] = p
            r["ci_excludes_zero"] = bool(r["ci_low"] > 0 or r["ci_high"] < 0)
            r["significant_holm_0_05"] = bool(p < 0.05)
        sig_rows.extend(rows)
    significance_df = pd.DataFrame(sig_rows)
    out.append(save_csv(significance_df, RESULTS_DIR / "significance.csv"))

    print("[6/9] Cold-start requests and cold-start workers")
    hybrid_cold = []
    for split, reqs in (("validation", val_requests), ("test", test_requests)):
        cold_reqs = cold_start_requests(reqs, cold_ids)
        _, summary = evaluate(selected, cold_reqs, k)
        hybrid_cold.append({"method": "hybrid", "split": split, "n_cold_start_requests": len(cold_reqs), **summary})
    cold_table = pd.concat([day7_cold, pd.DataFrame(hybrid_cold)], ignore_index=True)
    out.append(save_csv(cold_table, RESULTS_DIR / "cold_start_metrics.csv"))
    worker_rows = []
    for split, reqs in (("validation", val_requests), ("test", test_requests)):
        for method in METHOD_ORDER:
            worker_rows.append({"method": method, "split": split, **cold_worker_diagnostics(recorded[split][method], reqs, cold_ids)})
    cold_worker_df = pd.DataFrame(worker_rows)
    frozen_part = cold_worker_df.loc[cold_worker_df["method"] != "hybrid"].reset_index(drop=True)
    expected = day7_cold_workers.sort_values(["split", "method"]).reset_index(drop=True)
    got = frozen_part.sort_values(["split", "method"]).reset_index(drop=True)[expected.columns]
    if not np.allclose(got.select_dtypes("number").to_numpy(), expected.select_dtypes("number").to_numpy(), rtol=0, atol=1e-12):
        raise SystemExit("cold-start worker diagnostics for frozen models differ from Day 7")
    out.append(save_csv(cold_worker_df, RESULTS_DIR / "cold_start_worker_diagnostics.csv"))

    print("[7/9] Diversity and catalogue coverage")
    worker_skills = {w: frozenset(s.split("|")) if isinstance(s, str) and s else frozenset()
                     for w, s in zip(data.workers["worker_id"], data.workers["declared_skills"])}
    div_rows = []
    for split, reqs in (("validation", val_requests), ("test", test_requests)):
        eligible = {w for r in reqs if not r.excluded for w in r.candidates}
        for method in METHOD_ORDER:
            div_rows.append({"method": method, "split": split,
                             **diversity_summary(recorded[split][method], eligible, worker_skills, hcfg.diversity["top_share_fraction"])})
    diversity_df = pd.DataFrame(div_rows)
    out.append(save_csv(diversity_df, RESULTS_DIR / "diversity_metrics.csv"))

    print("[8/9] Hybrid contribution analysis, explanation fidelity, explanation samples")
    explanations = {}
    for split, reqs in (("validation", val_requests), ("test", test_requests)):
        explanations[split] = [e for r in reqs if not r.excluded for e in selected.explain(r.link, r.candidates, k)]
    contribution_rows, fidelity = [], {}
    for split, exps in explanations.items():
        positive = [e for e in exps if e.score > 0]
        for c in COMPONENTS:
            contribution_rows.append({
                "split": split, "component": c, "weight": best["weights"][c],
                "mean_contribution": float(np.mean([e.contributions[c] for e in exps])),
                "mean_share_of_score": float(np.mean([e.contributions[c] / e.score for e in positive])),
                "share_positive": float(np.mean([e.contributions[c] > 0 for e in exps])),
                "share_dominant": float(np.mean([e.dominant_component == c for e in exps])),
            })
        fidelity[split] = {
            "n_recommendations": len(exps),
            "share_first_cited_is_largest_contribution": float(np.mean([
                e.dominant_component == max(COMPONENTS, key=lambda c: (e.contributions[c], -COMPONENTS.index(c)))
                for e in exps if e.cited_components])),
            "share_contributions_reconstruct_score": float(np.mean([abs(sum(e.contributions.values()) - e.score) <= 1e-12 for e in exps])),
            "single_source_explanation_fidelity": {
                c: float(np.mean([e.dominant_component == c for e in exps])) for c in COMPONENTS
            },
            "n_recommendations_with_no_positive_contribution": sum(1 for e in exps if not e.cited_components),
        }
    contributions_df = pd.DataFrame(contribution_rows)
    out.append(save_csv(contributions_df, RESULTS_DIR / "hybrid_contributions.csv"))
    out.append(save_json(fidelity, RESULTS_DIR / "hybrid_explanation_fidelity.json"))

    e_cfg = hcfg.explanations
    sample = deterministic_sample(explanations[e_cfg["sample_split"]], e_cfg["sample_size"], data_cfg.seed)
    split_reqs = {r.link: r for r in (test_requests if e_cfg["sample_split"] == "test" else val_requests)}
    detailed = []
    for e in sample:
        req = split_reqs[e.link]
        full = selected.explain(req.link, req.candidates, k, kg_recommender=models["kg"], kg_paths=e_cfg["kg_paths_per_sample"])
        match = next(x for x in full if x.rank == e.rank)
        if match.worker_id != e.worker_id or abs(match.score - e.score) > 1e-12:
            raise SystemExit("detailed hybrid explanation does not match the scored recommendation")
        detailed.append(match.to_dict())
    out.append(save_json({"split": e_cfg["sample_split"], "seed": data_cfg.seed, "explanations": detailed},
                         RESULTS_DIR / "hybrid_explanation_samples.json"))

    print("[9/9] Figures and manifest")
    out.extend(make_figures(test_table, tuning_df, significance_df, cold_table, contributions_df, RESULTS_DIR / "figures"))
    manifest = build_manifest(REPO_ROOT, out)
    manifest.update({
        "seed": data_cfg.seed, "k": k,
        "hybrid_config_sha256": sha256_of(REPO_ROOT / "config" / "hybrid_config.yaml"),
        "frozen_inputs_sha256": {p: sha256_of(REPO_ROOT / p) for p in FROZEN_INPUTS},
        "kg_graph_snapshot_source": "day6_csv_artifacts (fingerprint verified against results/day7/graph_snapshot.json)",
    })
    save_manifest(manifest, RESULTS_DIR / "day8_manifest.json")

    return {"tuning": tuning_df, "selected": selected_config, "ablation": ablation_df, "validation": validation_table,
            "test": test_table, "significance": significance_df, "cold": cold_table, "cold_workers": cold_worker_df,
            "diversity": diversity_df, "contributions": contributions_df, "fidelity": fidelity}


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    t0 = time.time()
    result = run()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for title, key in (("Hybrid validation tuning", "tuning"), ("Validation ablation", "ablation"),
                       ("Validation metrics", "validation"), ("Test metrics", "test"), ("Significance (test)", "significance"),
                       ("Cold-start requests", "cold"), ("Cold-start workers", "cold_workers"), ("Diversity", "diversity"),
                       ("Contributions", "contributions")):
        print(f"\n=== {title} ===")
        print(result[key].to_string(index=False))
    print("\nselected:", result["selected"])
    print("\nfidelity:", result["fidelity"])
    print(f"\nelapsed_seconds: {round(time.time() - t0, 1)}")


if __name__ == "__main__":
    main()
