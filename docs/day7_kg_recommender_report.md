# Day 7 Knowledge Graph-Only Recommender Report

**Objective.** Build, tune, and evaluate a standalone Knowledge
Graph-only (KG) recommender over the Day 6 Neo4j graph, with graph-path
explanations, using the unchanged Day 4/5 evaluation protocol. KG-only is
the ablation named in `docs/experimental_design.md` Section 15. No
hybrid, score fusion, Flask, PostgreSQL, frontend, graph embedding, or
neural model is part of Day 7.

All numbers below come from `results/day7/` (`python
scripts/run_day7_evaluation.py`) and are checked independently by
`python scripts/verify_day7.py`.

## 1. Approved decisions this implementation follows

| Decision | Implementation |
|---|---|
| Scoring engine | One read-only extraction of a training-only `TrainingGraphView` from Neo4j (READ-access transactions); no per-request Cypher. The same view is built from the Day 6 CSV artifacts, and the run aborts unless the two are identical table by table |
| Price/rate | Not a KG score component (it remains part of CBF) |
| Location | Not a score component; stays a hard eligibility filter in the shared candidate protocol |
| `INTERESTED_IN` | Not a score component (it is identical for every candidate of a request); kept in the snapshot for inspection only, and not used in explanations |
| Co-hire path | Included. It overlaps conceptually with CF (shared clients), but it is an explicit enumeration of training `HIRED` paths, not the Day 5 similarity matrix. `knowledge_graph.py` does not import any other recommender |
| Live Neo4j write test | `tests/test_kg_neo4j.py::TestIdempotentLoading` is now opt-in (`HIREWISE_RUN_NEO4J_WRITE_TESTS=1`). The default suite is read-only against Neo4j |

## 2. Training-only graph snapshot

`src/kg/graph_view.py`. Snapshot fingerprint
`e7052c979dbcc36d759328a487ee77594ed2fbc9ece6323e4f2f001c226263f1`.

| Table | Rows | Rule |
|---|---|---|
| Workers / Skills / Categories | 3,000 / 1,187 / 88 | Worker contributes `worker_id` only (no `is_cold_start`, no `join_day`) |
| `HAS_SKILL` | 23,131 | Static declared profile |
| `SPECIALIZES_IN` | 6,031 (3,000 primary, 3,031 secondary) | Static profile |
| `RELATED_TO` | 12,504 | Background-corpus NPMI |
| `HIRED` | **7,117** (6,956 distinct client–worker pairs) | `split = 'train'` only. The split is a module constant, not a parameter |
| `REVIEWED` | **6,454** | Only via a join on `link` to a training `HIRED` edge (`REVIEWED` has no `split`) |
| `INTERESTED_IN` | 1,163 | Training-derived (Day 6); not scored |
| `REQUESTED_SERVICE` | not extracted | No component needs it |

- Training history covers 1,818 clients and 1,989 workers. 1,924 workers have training reviews, with a mean training rating of 4.4513.
- The latest historical edge is on day 323.0137. Every validation or test request is later than that.
- Extraction from Neo4j takes about 15–30 s; from the CSVs, about 0.5 s.

## 3. Scoring method

`src/recommenders/knowledge_graph.py`. Every path's weight is a
**degree-weighted path count** (DWPC; Himmelstein & Baranzini 2015):

```
path_weight = base_weight × PDP^(−w),   PDP = ∏ over path edges of deg(u)·deg(v)
```

Here `deg` counts a node's edges of that relationship type, and `w` is the
damping exponent. With `w = 0` this is a plain path count; larger `w`
down-weights paths through hub nodes such as very common skills or very
active clients.

The request has no graph node (Day 6 design), so paths start from its
**anchors**: its required Skill nodes, its ServiceCategory node and its
Client node. Path length is the number of real graph edges to the worker.

| Component | Path pattern (length) | Base weight |
|---|---|---|
| `skill` | (Skill)←HAS_SKILL–(Worker) (1) | 1 |
| `related_skill` | (Skill)–RELATED_TO–(Skill)←HAS_SKILL–(Worker) (2). The intermediate skill must not itself be required, so direct matches aren't counted twice | NPMI |
| `category` | (ServiceCategory)←SPECIALIZES_IN–(Worker) (1) | primary 1.0, secondary 0.5 (predeclared; deliberately not CBF's 0.6) |
| `history` | repeat_hire (Client)–HIRED→(Worker) (1); hired_similar_skill (Client)–HIRED→(W′)–HAS_SKILL→(Skill)←(Worker) (3); co_hire (Client)–HIRED→(W′)←HIRED–(Client′)–HIRED→(Worker) (3). Training edges only, distinct pairs, with no repeated nodes | 1 |
| `reputation` | Training `REVIEWED` edges into the worker: `(Σratings + 5·μ)/(n + 5)`, where μ = 4.4513 is the training mean. A worker with no reviews gets exactly μ | — |

The final score is built in three steps:
1. Min-max normalise each component within the request's candidate pool.
2. Combine them as a weighted sum.
3. Break ties by `worker_id`, ascending.

There is no popularity or CF fallback.

**Selected formula.** Damping `w = 0.4`, weight-grid entry 5:

```
KG(r, w) = 0.20·mm(skill) + 0.15·mm(related_skill) + 0.10·mm(category)
         + 0.20·mm(history) + 0.35·mm(reputation)
```

## 4. Validation-only tuning

The grid was predeclared in `config/kg_config.yaml`: damping {0.0, 0.4} ×
6 weight vectors, giving 12 configurations. The config's SHA-256
`1fdc6137…ac66` was recorded before the first run and matches the
manifest. Selection uses NDCG@10, then Recall@10, then the simpler
configuration (smaller damping, then the earlier grid entry). Test
metrics were never consulted during selection.

| w | Grid entry (skill/rel/cat/hist/rep) | Val NDCG@10 | Val Recall@10 |
|---|---|---|---|
| 0.0 | 0: .20/.20/.20/.20/.20 | 0.1437 | 0.1863 |
| 0.0 | 1: .30/.15/.15/.25/.15 | 0.1415 | 0.1858 |
| 0.0 | 2: .35/.20/.15/.15/.15 | 0.1459 | 0.1908 |
| 0.0 | 3: .30/.30/.10/.20/.10 | 0.1402 | 0.1844 |
| 0.0 | 4: .20/.10/.10/.45/.15 | 0.1330 | 0.1783 |
| 0.0 | 5: .20/.15/.10/.20/.35 | 0.1477 | 0.1918 |
| 0.4 | 0 | 0.1432 | 0.1849 |
| 0.4 | 1 | 0.1421 | 0.1860 |
| 0.4 | 2 | 0.1438 | 0.1888 |
| 0.4 | 3 | 0.1390 | 0.1835 |
| 0.4 | 4 | 0.1413 | 0.1877 |
| **0.4** | **5** | **0.1536** | **0.1945** |

## 5. Results

Random, Popularity, CBF and CF rows are copied verbatim from `results/day5/`.

**Validation (1,126 included / 74 excluded):**

| Method | P@10 | R@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|
| Random | 0.0660 | 0.1320 | 0.0880 | 0.0865 |
| Popularity | 0.0803 | 0.1606 | 0.1070 | 0.1171 |
| CBF | 0.0920 | 0.1840 | 0.1227 | 0.1410 |
| CF | 0.0817 | 0.1634 | 0.1089 | 0.1155 |
| **KG** | **0.0972** | **0.1945** | **0.1297** | **0.1536** |

**Test (2,275 included / 125 excluded), computed once for the selected configuration:**

| Method | P@10 | R@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|
| Random | 0.0682 | 0.1364 | 0.0909 | 0.0888 |
| Popularity | 0.0806 | 0.1611 | 0.1074 | 0.1190 |
| CBF | 0.0949 | 0.1897 | 0.1265 | 0.1394 |
| CF | 0.0792 | 0.1583 | 0.1056 | 0.1149 |
| **KG** | **0.0966** | **0.1932** | **0.1288** | **0.1499** |

KG-only has the highest point estimates on both splits. **No significance
test has been run.** The paired bootstrap intervals in the documented
protocol compare the hybrid with each baseline, so they belong to Day 8.
The KG−CBF test gap (+0.0106 NDCG@10, +0.0035 Recall@10) should not be
called a significant improvement until it is tested.

## 6. Cold start

**Requests whose relevant set contains a cold-start worker** (same definition as Days 4–5):

| Method | Val n | Val R@10 | Val NDCG@10 | Test n | Test R@10 | Test NDCG@10 |
|---|---|---|---|---|---|---|
| Random | 442 | 0.1267 | 0.0867 | 924 | 0.1394 | 0.0934 |
| Popularity | 442 | 0.1367 | 0.0992 | 924 | 0.1418 | 0.1036 |
| CBF | 442 | 0.1760 | 0.1320 | 924 | 0.1952 | 0.1446 |
| CF | 442 | 0.1416 | 0.0993 | 924 | 0.1374 | 0.0977 |
| KG | 442 | 0.1756 | 0.1348 | 924 | 0.1922 | 0.1497 |

**Cold-start *workers*** (docs/experimental_design.md Section 12). This is
a new Day 7 diagnostic. The frozen Day 4/5 models were rebuilt from their
selected configurations only for this table, after each one reproduced
its frozen test NDCG@10 exactly.

| Method (test) | Cold Recall@10 | Cold share of top-10 slots | Cold share of relevant items | Cold exposure ratio |
|---|---|---|---|---|
| Random | 0.1465 | 0.1006 | 0.0978 | 1.03 |
| Popularity | 0.0000 | 0.0000 | 0.0978 | 0.00 |
| CBF | 0.2083 | 0.1040 | 0.0978 | 1.06 |
| CF | 0.0000 | 0.0000 | 0.0978 | 0.00 |
| KG | 0.1104 | 0.0528 | 0.0978 | **0.54** |

These two tables need to be read together:
- KG ranks *requests* that involve cold-start workers about as well as CBF does.
- But KG gives cold-start workers about half their proportional share of top-10 slots, and recovers the relevant cold-start workers less often than CBF (Cold Recall@10 0.110 vs 0.208).
- The cause is structural. Cold-start workers get the neutral reputation prior and no own-history paths, while the selected configuration puts 0.55 of the weight on reputation and history.

**Cold-start path rules** (all enforced at runtime and verified):
- Cold-start workers never receive `repeat_hire` or `co_hire` paths, because both need the worker's own training `HIRED` edge. They have no reviews, and their reputation is exactly μ.
- They *can* receive `hired_similar_skill` paths, as 52.7% of test cold-start recommendations do. Such a path uses the requesting client's training hires and the cold-start worker's declared skill, both genuine training-time graph facts. No history is attributed to the worker.
- This corrects a statement in the pre-implementation review that cold-start workers would have no history paths "by construction". It is listed for review in Section 11.

## 7. Component ablation and category signal

Validation only; never used for selection. Each component is dropped from
the selected configuration in turn:

| Variant | Val NDCG@10 | Δ NDCG@10 | Δ Recall@10 |
|---|---|---|---|
| Full | 0.1536 | — | — |
| − skill | 0.1439 | −0.0097 | −0.0075 |
| − related_skill | 0.1548 | **+0.0012** | 0.0000 |
| − category | 0.1481 | −0.0055 | −0.0041 |
| − history | 0.1509 | −0.0027 | +0.0018 |
| − reputation | 0.1404 | −0.0132 | −0.0068 |

**Category.** Every candidate already matches the request's category, so
only primary versus secondary specialisation varies (55.5% of candidates
are primary, 44.5% secondary). Every validation pool contains both kinds,
so the component does vary within every pool. Its contribution:
- 12.4% of the score on average;
- positive for 93% of recommendations, and the largest component for under 0.01%;
- removing it lowers validation NDCG@10 by 0.0055.

That is a modest but non-trivial effect.

**Related skills.** Removing related-skill paths slightly *raises*
validation NDCG@10 (+0.0012, Recall unchanged). This is reported, not
acted on: changing the model after seeing this ablation would be
post-hoc selection.

**Reputation dominates.** Training reputation:
- is the largest contributor in about 97% of top-10 recommendations, on both splits;
- makes up about 51% of the score on average;
- is cited first in all 30 sampled explanations.

Reputation is a training-period proxy for the generator's quality factor
Q, which KG observes only partially through training `REVIEWED` edges
(Section 9.4 signal matrix).

## 8. Explanations

`src/kg/explanations.py`. `rank` and `explain` both call one
`enumerate_paths` function, so an explanation consists of the exact path
instances that were summed into each component.

Each path is credited with its share of the component's contribution:

```
contribution_p = (weight_c · mm_c) · path_weight / raw_c
```

Path contributions plus the reputation contribution add up to the score
exactly. `explain` raises an error if the re-enumerated paths fail to
reproduce the cached raw components.

Each displayed path keeps:
- its path type;
- its (label, id) node sequence;
- its edge properties (NPMI and support for RELATED_TO; role for SPECIALIZES_IN; `split='train'`, `n_hires` and hire links for HIRED);
- its weight and contribution;
- fixed template text.

At most 3 paths per component are displayed, with full path counts always recorded.

Example (test, rank 1, worker W000026, score 0.642):

> Average training-period rating 5.00/5 from 1 review (smoothed 4.54);
> C001926 also hired W002414 (whom you hired) and this worker; Has
> required skill '3D Rendering'

**Structural metrics** (top-10 recommendations, `explainability_metrics.json`):

| Metric | Validation | Test |
|---|---|---|
| Recommendations | 11,260 | 22,750 |
| Path coverage (any path of length ≤3) | 100% | 100% |
| Path coverage excluding category paths | 99.67% | 99.57% |
| Mean supporting paths (median) | 22.8 (21) | 22.1 (20) |
| Mean supporting paths excluding category | 21.8 | 21.1 |
| Mean path length (per recommendation / pooled) | 1.94 / 1.98 | 1.94 / 1.98 |
| Fidelity sanity: first cited component is the largest | 100% | 100% |
| Fidelity sanity: contributions reconstruct the score | 100% | 100% |
| Share of the score covered by displayed paths | 92.5% | 92.8% |

The plain path coverage figure is trivially 100%, because the category
path exists for every eligible candidate. The figure excluding category
paths is the informative one. The fidelity sanity check holds by
construction and is a check on the implementation. **The documented
explanation-fidelity metric is measured against the hybrid score and
belongs to Day 8.**

**Human rubric.** A seeded (seed 42) sample of 30 test explanations is in
`explanation_samples.json`, with a blank sheet in
`explanation_rubric_template.csv`. The proposed criteria are
understandable, grounded and useful, each rated 1–5 by the author. These
criteria are pending review, and nothing has been rated.

## 9. Temporal safety and leakage controls

| Risk | Control | Verified by |
|---|---|---|
| Validation/test `HIRED` edges in the graph | `WHERE h.split = $split`, with `$split` fixed to the `TRAIN_SPLIT` constant | Static test on every HIRED query; runtime test injecting validation/test edges |
| `REVIEWED` has no split | Joined on `link` to a training `HIRED` edge (Cypher and CSV paths) | Static + runtime tests, including a same-pair train/test case; 6,454 rows checked |
| `REQUESTED_SERVICE` split | Not extracted at all | Static test |
| Evaluated request's own outcome | Guard: the request must be strictly later than all snapshot history, and its link must not be among the snapshot's edges. Otherwise `TemporalLeakageError` | Unit tests; a runtime test writing r1's own outcome into the graph leaves scores unchanged |
| `is_cold_start` as a feature | Never extracted; absent from all Day 7 model code | Static scan; the verifier repeats it |
| Generator-private variables | Absent from model code; only model-visible files are read | Static scan |
| Validation/test using later history | One training-only snapshot for both splits | History ends at day 323.0137, before every validation/test request |
| Writes to Neo4j | READ-access transactions (`Neo4jClient.run_read`); no write keywords in the extraction Cypher; the default suite has no write tests | Static tests; verifier re-checks 6,420 nodes / 79,165 relationships |

## 10. Reproducibility

- A rerun from Neo4j produced **byte-identical** output in every file in `results/day7/`, including the figures and manifest.
- A rerun with `--graph-source csv` produced identical results files. Only the manifest's `graph_source` and parity fields differed, as intended.
- Determinism comes from:
  - explicit `ORDER BY` on every extraction query;
  - canonical sorting of every snapshot table;
  - a deterministic path-enumeration order, so floating-point sums come out the same;
  - candidate order taken from the shared protocol;
  - ties broken by `worker_id`.
- `day7_manifest.json` records:
  - the snapshot fingerprint and per-table fingerprints;
  - the Neo4j/CSV parity result;
  - the `kg_config.yaml` hash;
  - SHA-256 hashes of every frozen input (generator, evaluation and CF configs, Day 4/5 selected configs and results, Day 6 KG manifest).
- **Independent second derivation.** The verifier rebuilds the training graph from the raw Day 2/3 files. Every non-temporal table matches exactly, and every HIRED/REVIEWED edge matches exactly on every non-day field. 191 `simulated_day` values differ by at most 5.7e-14, a pandas CSV float round-trip artefact on the Day 6 path. Days only feed the temporal guard (1e-13 versus a 0.004-day gap), so this has no effect on scores.

## 11. Decisions needing review

1. **Reputation dominance.** The validation-selected weights make training reputation the main ranking and explanation driver. This lowers cold-start worker exposure (ratio 0.54). It is a legitimate predeclared grid choice, but it shapes how KG explanations read and how the hybrid should treat KG on Day 8.
2. **`hired_similar_skill` reaches cold-start workers.** This is a genuine training-only path, kept as implemented. The alternative is to restrict `history` to paths through the worker's own hires; that would be a methodology change.
3. **The related-skill ablation is slightly positive** (+0.0012 validation NDCG@10). Kept, to avoid post-hoc selection.
4. **Rubric criteria** (understandable / grounded / useful, 1–5) are placeholders for your approval.
5. **`Neo4jClient.run_read`** was added to the Day 6 client module. This is an additive change; the loader and schema are unchanged.

## Reproducing this report

```bash
python scripts/run_day7_evaluation.py          # requires Neo4j (read-only); or --graph-source csv
python scripts/verify_day7.py
python -m unittest discover -s tests           # read-only against Neo4j
HIREWISE_RUN_NEO4J_WRITE_TESTS=1 python -m unittest tests.test_kg_neo4j   # opt-in live-write test
```
