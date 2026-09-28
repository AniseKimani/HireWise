# Day 5 Recommender Report: Collaborative Filtering

**Objective.** Implement and evaluate a standalone neighborhood
Collaborative Filtering (CF) recommender under the exact Day 4 evaluation
protocol, learning only from TRAINING-period client-worker interactions,
never from content features or generator-private data.

## 1. Reused unchanged from Day 4

Candidate universe, relevance definition (HIRED=2/SHORTLISTED=1),
K=10, Precision/Recall/F1/NDCG@10, macro averaging, and the pool<20
exclusion rule are all imported from `src/evaluation/protocol.py` and
`src/evaluation/evaluator.py` without modification. No CF-specific
evaluation universe was created. `config/evaluation_config.yaml` was not
touched. Day 4's frozen `selected_cbf_config.json` and result CSVs are
reused verbatim in every comparison table below -- CBF was not rerun or
re-tuned.

No incompatibility was found between the existing evaluator and CF: CF
implements the same `BaseRecommender.rank(link, candidates, k)`
interface as Random/Popularity/CBF, deriving the request's `client_id`
internally from a `link -> client_id` lookup built from
`requests.csv`, so no change to the shared base interface, protocol, or
evaluator was needed.

## 2. Interaction matrix

Rows = `client_id`, columns = `worker_id`, values = aggregated implicit-
feedback weight, built **exclusively from `split == "train"`
interactions** (`src/recommenders/collaborative.py:build_training_matrix`).

**Weights** (`config/cf_config.yaml`): `HIRED = 2.0`, `SHORTLISTED = 1.0`.

**Ratings were deliberately not used.** Ratings exist only for a subset
of hires (9,173 of 10,163 total hires across the whole dataset; see
docs/synthetic_data_report.md Section 9) and are generated *from* the
hire event plus noise. Adding a rating-based weight on top of the HIRED
weight would double-count the same underlying event rather than add
independent signal -- exactly the risk the Day 5 brief warns about.

**Repeated client-worker interactions.** A client can interact with the
same worker across multiple distinct training requests: 1,979 of 39,587
distinct training (client, worker) pairs repeat (up to 8 times). Two
aggregation rules were predeclared and both included in the validation
tuning grid rather than picked by assertion: `sum` (total weight across
all training interactions for the pair) and `max` (the single strongest
observed signal). The selected configuration uses `sum`, though `max`
scored close behind (Section 5) -- the choice was not consequential
here.

## 3. User-based CF

Client-client cosine similarity on interaction-weight vectors. For a
target client, the top-K positively-similar clients (self excluded,
similarity > 0 only) contribute a similarity-weighted sum of their own
training interaction weight with each candidate worker:

```
score(c, w) = sum over c' in top-K similar clients of  sim(c, c') * R[c', w]
```

## 4. Item-based (worker-based) CF

Worker-worker cosine similarity on "which clients interacted with this
worker" vectors (the matrix's columns). For a target client's candidate
worker, the top-K positively-similar *other* workers are restricted to
the ones the client actually has training history with, then
similarity-weighted summed:

```
score(c, w) = sum over w' in (top-K similar-to-w) ∩ history(c) of  sim(w, w') * R[c, w']
```

Both formulas were verified against a hand-calculated 3-client/4-worker
scenario before being trusted (`tests/test_collaborative.py`); the exact
expected floating-point scores are encoded as unit test assertions.

## 5. Validation-only tuning

Grid: `aggregation × method × k_neighbors` = `{sum, max} × {user, item} ×
{10, 20, 40, 80}` = **16 configurations**, every one evaluated on
validation only (`results/day5/cf_tuning.csv`). Primary metric NDCG@10,
tie-break Recall@10, then the smaller (simpler) `k_neighbors` --
predeclared in `config/cf_config.yaml` before any evaluation ran.

| aggregation | method | k | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Selected |
|---|---|---|---|---|---|---|---|
| sum | user | 20 | 0.0817 | 0.1634 | 0.1089 | **0.1155** | ✓ |
| max | user | 10 | 0.0817 | 0.1634 | 0.1089 | 0.1150 | |
| sum | user | 10 | 0.0815 | 0.1631 | 0.1087 | 0.1138 | |
| max | user | 20 | 0.0803 | 0.1606 | 0.1070 | 0.1129 | |
| sum | user | 40 | 0.0793 | 0.1586 | 0.1057 | 0.1126 | |
| max | user | 40 | 0.0791 | 0.1583 | 0.1055 | 0.1112 | |
| sum | user | 80 | 0.0782 | 0.1563 | 0.1042 | 0.1106 | |
| sum | item | 10 | 0.0782 | 0.1563 | 0.1042 | 0.1099 | |

**User-based beats item-based at every grid point.** This is consistent
with the coverage picture (Section 8): with a median of only 15 distinct
workers per client with any training history, worker-worker similarity
has very little co-occurrence evidence to work with, while client-client
similarity benefits from clients sharing category/country-driven
patterns even with modest individual histories. Smaller neighborhoods
(K=10-20) consistently beat larger ones -- larger K pulls in
progressively weaker, noisier similarities.

**Selected configuration: `sum` aggregation, `user`-based, K=20**
(NDCG@10 = 0.1155 on validation). Frozen in `results/day5/selected_cf_config.json`.

## 6. Validation results (K=10)

| Method | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|
| Random | 0.0660 | 0.1320 | 0.0880 | 0.0865 |
| Popularity | 0.0803 | 0.1606 | 0.1070 | 0.1171 |
| CBF | 0.0920 | 0.1840 | 0.1227 | 0.1410 |
| **CF** | **0.0817** | **0.1634** | **0.1089** | **0.1155** |

n_included = 1,126, n_excluded = 74 (pool_too_small) -- identical candidate universe across all four methods.

## 7. Test results (K=10)

| Method | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|
| Random | 0.0682 | 0.1364 | 0.0909 | 0.0888 |
| Popularity | 0.0806 | 0.1611 | 0.1074 | 0.1190 |
| CBF | 0.0949 | 0.1897 | 0.1265 | 0.1394 |
| **CF** | **0.0792** | **0.1583** | **0.1056** | **0.1149** |

n_included = 2,275, n_excluded = 125. **CF ranks between Random and
Popularity/CBF on both splits, and is slightly below Popularity on every
metric.** This is an honest result, not a bug: Section 10 below shows why.

## 8. History coverage

| | Clients | Workers |
|---|---|---|
| Total | 2,000 | 3,000 |
| With >=1 training interaction | 1,867 (93.4%) | 2,685 (89.5%) |
| With zero training interactions | 133 (6.7%) | 315 (10.5%) |
| Median distinct counterparts per active entity | 15.0 (workers/client) | -- |

No history was fabricated for the 133 no-history clients or the 315
no-history workers; both are handled entirely through the fallback
mechanism (Section 9).

## 9. Fallback policy

When a client has no training history at all, or a specific candidate's
CF score is exactly 0, ranking falls back to Day 4's training-only
Popularity signal. The two are combined with a strict, deterministic
sort key that never lets fallback outrank genuine CF evidence:

```
sort key = (-cf_score, -popularity_hire_count, worker_id)
```

A candidate with any positive CF score always ranks above every
candidate with CF score 0, regardless of that zero-score candidate's
popularity; popularity only decides ordering *among* tied candidates
(all with CF score 0, including a client with no history at all, where
every candidate is tied at 0).

**Usage rates** (`results/day5/cf_coverage.json`):

| | Validation | Test |
|---|---|---|
| Requests fully determined by fallback (client has no history at all) | 184 / 1,126 (16.3%) | 495 / 2,275 (21.8%) |
| Individual recommendation slots filled by fallback (zero CF score) | 3,401 / 11,260 (30.2%) | 7,942 / 22,750 (34.9%) |

Roughly a third of all recommendation slots on both splits carry no
genuine collaborative signal at all -- this is reported explicitly so
that CF's measured performance is never read as "pure CF capability";
a meaningful share of it is really Popularity underneath.

## 10. Cold-start results

**0 of the 300 new-entrant workers appear in the training matrix at
all** (`n_cold_start_workers_in_training_matrix: 0`, verified
independently in `scripts/verify_day5.py`) -- exactly as required by
construction: their `join_day` places them after the training period
ends, so they generate no training interactions to learn from. This is
the clearest demonstration in the project so far of *why* the Hybrid is
motivated: standalone CF has **no mechanism whatsoever** to produce a
genuine score for any of these 300 workers; every single cold-start
recommendation CF makes is fallback-only.

| Method | Split | n requests | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| Random | validation | 442 | 0.0633 | 0.1267 | 0.0845 | 0.0867 |
| Popularity | validation | 442 | 0.0683 | 0.1367 | 0.0911 | 0.0992 |
| CBF | validation | 442 | 0.0880 | 0.1760 | 0.1173 | 0.1320 |
| **CF** | validation | 442 | 0.0708 | 0.1416 | 0.0944 | 0.0993 |
| Random | test | 924 | 0.0697 | 0.1394 | 0.0929 | 0.0934 |
| Popularity | test | 924 | 0.0709 | 0.1418 | 0.0945 | 0.1036 |
| CBF | test | 924 | 0.0976 | 0.1952 | 0.1302 | 0.1446 |
| **CF** | test | 924 | 0.0687 | 0.1374 | 0.0916 | 0.0977 |

CF's cold-start numbers sit almost exactly on top of Popularity's
(0.0993 vs. 0.0992 validation NDCG@10; 0.0977 vs. 0.1036 test) --
unsurprising, since every cold-start recommendation CF makes *is* the
Popularity fallback. CF adds no cold-start capability beyond what
Popularity already provides, and CBF remains clearly the strongest
method for surfacing new entrants, consistent with Day 4's finding.
**This is a property of collaborative data availability for unseen
entities, not an implementation defect** -- no neighborhood CF method,
however implemented, can score an item or user with zero training
interactions from interaction data alone.

## 11. Sparse-client and sparse-worker behavior

Both are handled the same way: absence from the training matrix (no
special-cased "sparse" logic). A client with exactly one training
interaction still gets a similarity row (possibly with few or no
positively-similar neighbors, in which case its candidates typically
fall back), and a worker with very few training interactions gets a
similarity column with correspondingly weak evidence -- there is no
separate sparse-vs-dense code path, only more or less signal.

## 12. Leakage safeguards

- `build_training_matrix` filters to `split == "train"` before any
  aggregation; `tests/test_day5_leakage.py` proves adding
  validation/test-split interactions to the input does not change the
  resulting matrix at all.
- `src/recommenders/collaborative.py` never imports
  `src/recommenders/content_based.py`, and its executable code (checked
  with docstrings/comments stripped via `ast`, so explanatory prose
  doesn't produce false negatives) references none of
  `declared_skills`, `skills_vocab`, `category`, `price_tier`, `rate`,
  `country`.
- No generator-private marker (`quality`, `true_skills`, `proficiency`,
  `taste_`, `utility`, `noise`, `synthetic_private`) appears anywhere in
  the module.
- The Popularity fallback's hire counts were independently recomputed
  from raw training-only interactions and compared for exact equality.
- `scripts/verify_day5.py` independently rebuilds the training matrix
  from raw data a second way and cross-checks every value.

## 13. Reproducibility

Two full runs of `scripts/run_day5_evaluation.py` produced byte-identical
`results/day5/` outputs (`diff -rq`, zero differences, including the
three PNG figures). `day5_manifest.json` records SHA-256 hashes of every
output file, the frozen `generator_v1.yaml` hash, the `cf_config.yaml`
hash, and Day 4's `selected_cbf_config.json` hash (to make any
accidental Day 4 drift detectable).

## 14. Limitations and decisions not fixed by the brief

1. **Weights were not tuned.** `HIRED=2.0`/`SHORTLISTED=1.0` (matching
   the relevance grades) was fixed rather than added as a third tuning
   dimension, to keep the grid small as instructed; only aggregation,
   method, and K were searched.
2. **No similarity shrinkage/min-overlap threshold** beyond "positive
   similarity only" was implemented -- an available but unused option
   per Section 7 of the brief. Worth investigating if standalone CF's
   relative weakness turns out to matter for the eventual Hybrid.
3. **Fallback tie-break uses raw popularity hire count**, not any
   smoothed or normalized version; this matches Day 4's own Popularity
   baseline exactly, for direct comparability.
4. **Item-based similarity is recomputed per grid cell** rather than
   cached across `k_neighbors` values in the tuning script in a way that
   shares work across aggregation choices beyond the per-(aggregation,
   method) level already implemented; the full 16-point grid still runs
   in about 23 seconds, so no further optimization was pursued.

**No claim is made that CF's underperformance relative to CBF here
predicts anything about the eventual Hybrid.** The Hybrid has not been
built. What Day 5 does establish, concretely: standalone CF is
structurally unable to say anything about the 300 cold-start workers,
which is exactly the gap a Hybrid combining CF with CBF (or KG) is
meant to address.

## Reproducing this report

```bash
python scripts/run_day5_evaluation.py
python scripts/verify_day5.py
python -m unittest discover -s tests
```
