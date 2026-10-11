"""Independent verification of the Day 7 KG-only recommender and results.

Re-derives invariants rather than trusting results/day7/* at face value:

- rebuilds the training-only graph a SECOND way, from the raw Day 2/3
  model-visible files (not the Day 6 KG CSVs), and compares its
  fingerprint with the recorded snapshot;
- re-extracts the snapshot from live Neo4j with read-only Cypher (READ
  transactions) and confirms node/relationship counts are unchanged;
- recomputes DWPC component scores with an independent naive
  implementation for a sample of validation requests;
- confirms the selected configuration is the tuning grid's best by the
  predeclared rule and reproduces its validation and test metrics;
- confirms Day 4/5 rows are copied verbatim and every frozen artifact is
  byte-unchanged;
- validates every path in the 30 sampled explanations against the graph
  (and against Neo4j, read-only, when reachable).

Usage:
    python scripts/verify_day7.py

Exits non-zero if any check fails. Never writes to Neo4j or results/.
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.data.io import sha256_of
from src.evaluation.config import load_evaluation_config
from src.evaluation.evaluator import evaluate
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.kg.graph_view import build_view, load_view_from_kg_csvs
from src.recommenders.kg_config import load_kg_config
from src.recommenders.knowledge_graph import COMPONENTS, KGComponentCache, KGIndex, KnowledgeGraphRecommender

RESULTS_DIR = REPO_ROOT / "results" / "day7"
DAY2 = REPO_ROOT / "data" / "processed"
DAY3 = DAY2 / "synthetic"
EXPECTED_NODES = 6420
EXPECTED_RELATIONSHIPS = 79165

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def skip(label: str, reason: str) -> None:
    print(f"[SKIP] {label} -- {reason}")


# ---------------------------------------------------------------------------
# Independent graph reconstruction from raw Day 2/3 files
# ---------------------------------------------------------------------------

def rebuild_view_from_raw():
    workers = pd.read_csv(DAY3 / "workers.csv")
    requests = pd.read_csv(DAY3 / "requests.csv")
    interactions = pd.read_csv(DAY3 / "interactions.csv")
    ratings = pd.read_csv(DAY3 / "ratings.csv")
    sampled = pd.read_csv(DAY2 / "sampled_requests.csv", usecols=["link", "category"])
    vocab = set(pd.read_csv(DAY2 / "skills_vocab.csv")["skill"])
    categories = set(pd.read_csv(DAY2 / "category_stats.csv")["category"])
    cooc = pd.read_csv(DAY2 / "skill_cooccurrence_background_corpus.csv")

    has_skill = [(w, s) for w, cell in zip(workers["worker_id"], workers["declared_skills"])
                 if isinstance(cell, str) for s in cell.split("|") if s in vocab]
    specializes = []
    for w, p, sec in zip(workers["worker_id"], workers["primary_category"], workers["secondary_categories"]):
        if p in categories:
            specializes.append((w, p, "primary"))
        if isinstance(sec, str):
            specializes.extend((w, c, "secondary") for c in sec.split("|") if c in categories and c != p)
    cooc = cooc.loc[cooc["skill_a"].isin(vocab) & cooc["skill_b"].isin(vocab)]
    related = list(zip(cooc["skill_a"], cooc["skill_b"], cooc["npmi"], cooc["support"]))

    split_of = dict(zip(requests["link"], requests["split"]))
    client_of = dict(zip(requests["link"], requests["client_id"]))
    hires = interactions.loc[interactions["event"] == "HIRED"]
    train_hires = [(c, w, l, d) for l, w, c, d in zip(hires["link"], hires["worker_id"], hires["client_id"], hires["simulated_day"])
                   if split_of[l] == "train"]
    hire_day = {(l, w): d for _, w, l, d in train_hires}
    train_reviews = [(client_of[l], w, l, r, hire_day[(l, w)]) for l, w, r in zip(ratings["link"], ratings["worker_id"], ratings["rating"])
                     if (l, w) in hire_day]

    train_requests = requests.loc[requests["split"] == "train"].merge(sampled, on="link")
    counts = train_requests.groupby(["client_id", "category"]).size()
    interested = [(c, cat, n) for (c, cat), n in counts.items() if n >= 2]

    return build_view(
        workers=workers["worker_id"], skills=vocab, categories=categories, has_skill=has_skill,
        specializes_in=specializes, related_to=related, train_hires=train_hires, train_reviews=train_reviews,
        interested_in=interested, source="raw",
    )


# ---------------------------------------------------------------------------
# Independent naive DWPC
# ---------------------------------------------------------------------------

def naive_components(view, request_row, workers, damping, role_weights, prior_strength):
    """Deliberately written differently from src/recommenders/knowledge_graph.py:
    explicit set algebra over edge lists, no shared helpers."""
    skills_of = defaultdict(set)
    for w, s in view.has_skill:
        skills_of[w].add(s)
    holders = Counter(s for _, s in view.has_skill)
    neighbours = defaultdict(dict)
    for a, b, npmi, _ in view.related_to:
        neighbours[a][b] = npmi
        neighbours[b][a] = npmi
    cats_of = defaultdict(dict)
    for w, c, r in view.specializes_in:
        cats_of[w][c] = r
    cat_size = Counter(c for _, c, _ in view.specializes_in)
    pairs = {(c, w) for c, w, _, _ in view.train_hires}
    hired_by = defaultdict(set)
    hirers_of = defaultdict(set)
    for c, w in pairs:
        hired_by[c].add(w)
        hirers_of[w].add(c)
    ratings = defaultdict(list)
    for _, w, _, r, _ in view.train_reviews:
        ratings[w].append(r)
    mu = sum(r for _, _, _, r, _ in view.train_reviews) / len(view.train_reviews)

    def damp(base, *degrees):
        product = 1.0
        for d in degrees:
            product *= d
        return base * (product ** -damping if damping else 1.0)

    required = set(str(request_row["skills_vocab"]).split("|")) & set(view.skills)
    client, category = request_row["client_id"], request_row["category"]
    out = np.zeros((len(workers), len(COMPONENTS)))
    for i, w in enumerate(workers):
        S = skills_of[w]
        out[i, 0] = sum(damp(1.0, holders[s], len(S)) for s in required & S)
        out[i, 1] = sum(
            damp(neighbours[s][t], len(neighbours[s]), len(neighbours[t]), holders[t], len(S))
            for s in required for t in neighbours[s] if t in S and t not in required
        )
        if category in cats_of[w]:
            out[i, 2] = damp(role_weights[cats_of[w][category]], cat_size[category], len(cats_of[w]))
        hist = 0.0
        if (client, w) in pairs:
            hist += damp(1.0, len(hired_by[client]), len(hirers_of[w]))
        for wp in hired_by[client] - {w}:
            for s in skills_of[wp] & S:
                hist += damp(1.0, len(hired_by[client]), len(hirers_of[wp]), len(skills_of[wp]), holders[s], holders[s], len(S))
            for cp in hirers_of[wp] - {client}:
                if w in hired_by[cp]:
                    hist += damp(1.0, len(hired_by[client]), len(hirers_of[wp]), len(hirers_of[wp]),
                                 len(hired_by[cp]), len(hired_by[cp]), len(hirers_of[w]))
        out[i, 3] = hist
        out[i, 4] = (sum(ratings[w]) + prior_strength * mu) / (len(ratings[w]) + prior_strength)
    return out


# ---------------------------------------------------------------------------
# Explanation path validation
# ---------------------------------------------------------------------------

PATH_CYPHER = {
    "skill": "MATCH (w:Worker {worker_id: $n1})-[:HAS_SKILL]->(:Skill {name: $n0}) RETURN count(*) AS c",
    "related_skill": ("MATCH (:Skill {name: $n0})-[:RELATED_TO]-(b:Skill {name: $n1}) "
                      "MATCH (:Worker {worker_id: $n2})-[:HAS_SKILL]->(b) RETURN count(*) AS c"),
    "category": ("MATCH (:Worker {worker_id: $n1})-[r:SPECIALIZES_IN]->(:ServiceCategory {name: $n0}) "
                 "WHERE r.role = $role RETURN count(*) AS c"),
    "repeat_hire": ("MATCH (:Client {client_id: $n0})-[h:HIRED]->(:Worker {worker_id: $n1}) "
                    "WHERE h.split = 'train' RETURN count(*) AS c"),
    "hired_similar_skill": ("MATCH (:Client {client_id: $n0})-[h:HIRED]->(wp:Worker {worker_id: $n1}) WHERE h.split = 'train' "
                            "MATCH (wp)-[:HAS_SKILL]->(s:Skill {name: $n2})<-[:HAS_SKILL]-(:Worker {worker_id: $n3}) "
                            "RETURN count(*) AS c"),
    "co_hire": ("MATCH (:Client {client_id: $n0})-[h1:HIRED]->(wp:Worker {worker_id: $n1})<-[h2:HIRED]-(cp:Client {client_id: $n2}) "
                "MATCH (cp)-[h3:HIRED]->(:Worker {worker_id: $n3}) "
                "WHERE h1.split = 'train' AND h2.split = 'train' AND h3.split = 'train' RETURN count(*) AS c"),
}


def path_in_view(view, path_type, ids, role=None) -> bool:
    has = set(view.has_skill)
    pairs = {(c, w) for c, w, _, _ in view.train_hires}
    rel = {(a, b) for a, b, _, _ in view.related_to} | {(b, a) for a, b, _, _ in view.related_to}
    if path_type == "skill":
        return (ids[1], ids[0]) in has
    if path_type == "related_skill":
        return (ids[0], ids[1]) in rel and (ids[2], ids[1]) in has
    if path_type == "category":
        return (ids[1], ids[0], role) in set(view.specializes_in)
    if path_type == "repeat_hire":
        return (ids[0], ids[1]) in pairs
    if path_type == "hired_similar_skill":
        return (ids[0], ids[1]) in pairs and (ids[1], ids[2]) in has and (ids[3], ids[2]) in has
    if path_type == "co_hire":
        return (ids[0], ids[1]) in pairs and (ids[2], ids[1]) in pairs and (ids[2], ids[3]) in pairs
    return False


def code_only(module) -> str:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body[0].value.value = ""
    return ast.unparse(tree).lower()


def main() -> int:
    eval_cfg = load_evaluation_config()
    kg_cfg = load_kg_config()

    required = [
        "kg_tuning.csv", "kg_ablation_validation.csv", "kg_component_contributions.csv", "validation_metrics.csv",
        "test_metrics.csv", "cold_start_metrics.csv", "cold_start_worker_diagnostics.csv", "selected_kg_config.json",
        "explainability_metrics.json", "explanation_samples.json", "explanation_rubric_template.csv",
        "graph_snapshot.json", "day7_manifest.json",
    ]
    missing = [f for f in required if not (RESULTS_DIR / f).exists()]
    check("all expected Day 7 result artifacts exist", not missing, f"missing {missing}")
    if missing:
        print("\nRun scripts/run_day7_evaluation.py first.")
        return 1

    load = lambda name: json.load(open(RESULTS_DIR / name, encoding="utf-8"))
    manifest, snapshot, selected = load("day7_manifest.json"), load("graph_snapshot.json"), load("selected_kg_config.json")

    # ---- Frozen artifacts ---------------------------------------------------
    with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
        check("frozen generator_v1.yaml hash unchanged", sha256_of(REPO_ROOT / "config" / "generator_v1.yaml") == fh.read().split()[0])
    changed = [p for p, h in manifest["frozen_inputs_sha256"].items() if sha256_of(REPO_ROOT / p) != h]
    check("frozen inputs recorded by Day 7 are byte-unchanged", not changed, f"{changed}")
    for day_manifest in ("results/day4/day4_manifest.json", "results/day5/day5_manifest.json", "data/processed/kg/kg_prepare_manifest.json"):
        entries = json.load(open(REPO_ROOT / day_manifest, encoding="utf-8"))["files"]
        drift = [e["path"] for e in entries if sha256_of(REPO_ROOT / e["path"]) != e["sha256"]]
        check(f"every file in {day_manifest} matches its recorded hash", not drift, f"{drift}")
    check("kg_config.yaml unchanged since the Day 7 run", sha256_of(REPO_ROOT / "config" / "kg_config.yaml") == manifest["kg_config_sha256"])
    check("manifest records Neo4j/CSV parity for the canonical run",
          manifest.get("graph_source") == "neo4j" and manifest.get("neo4j_csv_parity") is True)
    drift = [e["path"] for e in manifest["files"] if sha256_of(REPO_ROOT / e["path"]) != e["sha256"]]
    check("every Day 7 output matches its manifest hash", not drift, f"{drift}")

    # ---- Snapshot: second derivation + CSV + Neo4j --------------------------
    raw_view = rebuild_view_from_raw()
    csv_view = load_view_from_kg_csvs()
    # simulated_day values pass through one extra CSV write/read on the
    # Day 6 path, and pandas' default float parser is not exactly
    # round-trip, so a few days differ in the last digit. Days never enter
    # a score (only the temporal guard), so they are compared within a
    # tolerance; every other field, and every other table, exactly.
    day_tables = ("train_hires", "train_reviews")
    exact_tables = [t for t in snapshot["table_fingerprints"] if t not in day_tables]
    differing = [t for t in exact_tables if raw_view.table_fingerprints()[t] != snapshot["table_fingerprints"][t]]
    check("independent rebuild from raw Day 2/3 files: non-temporal tables match the recorded snapshot exactly",
          not differing, f"differing: {differing}")
    max_day_gap = 0.0
    keys_match = True
    for table in day_tables:
        raw_rows, csv_rows = getattr(raw_view, table), getattr(csv_view, table)
        keys_match &= [r[:-1] for r in raw_rows] == [r[:-1] for r in csv_rows]
        max_day_gap = max([max_day_gap] + [abs(a[-1] - b[-1]) for a, b in zip(raw_rows, csv_rows)])
    check("independent rebuild: training HIRED/REVIEWED edges match exactly on every non-day field", keys_match)
    check("independent rebuild: simulated days agree within 1e-9 (CSV float round-trip only)",
          max_day_gap < 1e-9, f"max gap {max_day_gap}")
    print(f"       (max simulated_day float round-trip gap: {max_day_gap:.3g})")
    check("CSV-derived view matches the recorded fingerprint", csv_view.fingerprint() == snapshot["fingerprint"])
    check("snapshot holds exactly 7,117 training hires and 6,454 training reviews",
          len(csv_view.train_hires) == 7117 and len(csv_view.train_reviews) == 6454)

    try:
        from src.kg.graph_view import load_view_from_neo4j
        from src.kg.neo4j_client import Neo4jClient
        from src.kg import queries

        neo4j_client = Neo4jClient().__enter__()
    except Exception as exc:  # unreachable or unconfigured -> read-only checks skipped, not failed
        neo4j_client = None
        skip("live Neo4j checks", f"Neo4j unavailable: {exc}")
    if neo4j_client is not None:
        nodes = sum(queries.count_nodes_by_label(neo4j_client).values())
        rels = sum(queries.count_relationships_by_type(neo4j_client).values())
        check(f"live Neo4j graph unchanged ({EXPECTED_NODES:,} nodes / {EXPECTED_RELATIONSHIPS:,} relationships)",
              nodes == EXPECTED_NODES and rels == EXPECTED_RELATIONSHIPS, f"{nodes} nodes / {rels} relationships")
        check("Neo4j-derived view (read-only extraction) matches the recorded fingerprint",
              load_view_from_neo4j(neo4j_client).fingerprint() == snapshot["fingerprint"])

    data = load_protocol_data()
    cold = set(pd.read_csv(DAY3 / "cold_start_cohort.csv")["worker_id"])
    check("no cold-start worker has a training HIRED or REVIEWED edge in the snapshot",
          not (cold & ({w for _, w, _, _ in csv_view.train_hires} | {w for _, w, _, _, _ in csv_view.train_reviews})))
    later = data.requests.loc[data.requests["split"] != "train", "simulated_day"].min()
    check("all snapshot history precedes every validation/test request", KGIndex(csv_view).max_history_day < later)

    # ---- Static leakage scan --------------------------------------------------
    from src.kg import explanations as kg_explanations, graph_view
    from src.recommenders import knowledge_graph

    for module in (graph_view, knowledge_graph, kg_explanations):
        code = code_only(module)
        bad = [m for m in ("quality", "true_skills", "proficiency", "taste_", "utility", "noise", "oracle",
                           "synthetic_private", "is_cold_start", "join_day") if m in code]
        check(f"{module.__name__}: no generator-private or cohort markers in code", not bad, f"{bad}")

    # ---- Independent DWPC recomputation -------------------------------------
    index = EligibilityIndex(data.workers, data.known_countries)
    val_requests = build_eval_requests(data, index, "validation", eval_cfg)
    test_requests = build_eval_requests(data, index, "test", eval_cfg)
    kg_index = KGIndex(csv_view)
    rec = KnowledgeGraphRecommender(
        kg_index, data.requests, selected["damping"], selected["weights"], kg_cfg.category_role_weights,
        kg_cfg.reputation_prior_strength, kg_cfg.explanation_max_paths_per_component, KGComponentCache((selected["damping"],)),
    )
    request_rows = data.requests.set_index("link")
    sample = [r for r in val_requests if not r.excluded][::40]
    worst = 0.0
    for req in sample:
        mine = rec.raw_components(req.link, req.candidates)
        theirs = naive_components(csv_view, request_rows.loc[req.link], req.candidates, selected["damping"],
                                  kg_cfg.category_role_weights, kg_cfg.reputation_prior_strength)
        worst = max(worst, float(np.max(np.abs(mine - theirs) / np.maximum(1.0, np.abs(theirs)))))
    check(f"DWPC components match an independent naive recomputation ({len(sample)} validation requests)",
          worst < 1e-9, f"max relative difference {worst}")

    # ---- Selection and metric reproduction -----------------------------------
    tuning = pd.read_csv(RESULTS_DIR / "kg_tuning.csv")
    check("kg_tuning.csv covers the full predeclared grid",
          len(tuning) == len(kg_cfg.dwpc_damping_grid) * len(kg_cfg.weight_grid))
    sel = tuning.loc[tuning["selected"]]
    check("exactly one tuning row selected", len(sel) == 1)
    keys = [(r.ndcg_at_k, r.recall_at_k, -r.damping, -r.weight_grid_index) for r in tuning.itertuples()]
    best = tuning.iloc[max(range(len(keys)), key=lambda i: keys[i])]
    check("selected row is the best by NDCG@10, then Recall@10, then simplest",
          best["damping"] == selected["damping"] and int(best["weight_grid_index"]) == selected["weight_grid_index"])
    check("selected weights equal the predeclared grid entry",
          kg_cfg.weight_grid[selected["weight_grid_index"]] == selected["weights"])

    _, val_summary = evaluate(rec, val_requests, eval_cfg.k)
    check("re-evaluating the selected config reproduces the recorded validation NDCG@10",
          abs(val_summary["ndcg_at_k"] - selected["primary_metric_value"]) < 1e-12)
    test_df = pd.read_csv(RESULTS_DIR / "test_metrics.csv")
    _, test_summary = evaluate(rec, test_requests, eval_cfg.k)
    recorded = test_df.loc[test_df["method"] == "kg"].iloc[0]
    check("re-evaluating the selected config reproduces the recorded test metrics",
          all(abs(test_summary[m] - recorded[m]) < 1e-12 for m in ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k")))

    for name in ("validation_metrics.csv", "test_metrics.csv", "cold_start_metrics.csv"):
        day5 = pd.read_csv(REPO_ROOT / "results" / "day5" / name)
        day7 = pd.read_csv(RESULTS_DIR / name)
        check(f"{name}: Day 4/5 rows copied verbatim", day7.loc[day7["method"] != "kg"].reset_index(drop=True).equals(day5))
        check(f"{name}: exactly the five expected methods",
              sorted(day7["method"].unique()) == ["cbf", "cf", "kg", "popularity", "random"])
        for metric in ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"):
            check(f"{name}: {metric} within [0, 1]", bool(day7[metric].between(0, 1).all()))

    # ---- Explanations -------------------------------------------------------
    samples = load("explanation_samples.json")["explanations"]
    check(f"{kg_cfg.explanation_sample_size} explanation samples exported", len(samples) == kg_cfg.explanation_sample_size)
    rubric = pd.read_csv(RESULTS_DIR / "explanation_rubric_template.csv")
    check("rubric template has one blank row per sample",
          len(rubric) == len(samples) and rubric[["understandable_1to5", "grounded_1to5", "useful_1to5"]].isna().all().all())
    view_ok, neo4j_ok, faithful, history_ok, n_paths = True, True, True, True, 0
    for e in samples:
        reputation = e["reputation"]["contribution"]
        faithful &= abs(e["total_path_contribution"] + reputation - e["score"]) <= 1e-9
        if e["worker_id"] in cold:
            history_ok &= e["n_paths_by_type"]["repeat_hire"] == 0 and e["n_paths_by_type"]["co_hire"] == 0 \
                and e["reputation"]["n_reviews"] == 0
        for p in e["paths"]:
            n_paths += 1
            ids = [n["id"] for n in p["nodes"]]
            role = p["edges"][0].get("role")
            view_ok &= ids[-1] == e["worker_id"] and path_in_view(csv_view, p["path_type"], ids, role)
            for edge in p["edges"]:
                if edge["type"] == "HIRED":
                    view_ok &= edge["split"] == "train"
            if neo4j_client is not None:
                params = {f"n{i}": v for i, v in enumerate(ids)}
                if role is not None:
                    params["role"] = role
                neo4j_ok &= neo4j_client.run_read(PATH_CYPHER[p["path_type"]], params)[0]["c"] >= 1
    check(f"all {n_paths} sampled explanation paths exist in the training snapshot and end at the recommended worker", view_ok)
    if neo4j_client is not None:
        check(f"all {n_paths} sampled explanation paths exist in live Neo4j (read-only; HIRED edges split='train')", neo4j_ok)
        neo4j_client.__exit__(None, None, None)
    check("sampled explanations' path + reputation contributions reconstruct the score", faithful)
    check("sampled cold-start workers have no own-history paths and no reviews", history_ok)

    metrics = load("explainability_metrics.json")["explainability"]
    for split in ("validation", "test"):
        s = metrics[split]["all"]
        check(f"{split}: every recommendation has a valid path of length <= 3", s["path_coverage"] == 1.0 and s["max_observed_path_length"] <= 3)
        check(f"{split}: fidelity sanity holds for 100% of explanations",
              s["fidelity_sanity"]["share_contributions_reconstruct_score"] == 1.0
              and s["fidelity_sanity"]["share_first_cited_component_is_largest"] == 1.0)
        c = metrics[split]["cold_start_workers"]["share_recommendations_with_path_type"]
        check(f"{split}: cold-start workers never receive repeat_hire/co_hire paths", c["repeat_hire"] == 0 and c["co_hire"] == 0)

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
