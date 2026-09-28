# Day 4 Recommender Report: Baselines + Content-Based Filtering

**Objective.** Implement shared recommendation/evaluation infrastructure
and the first three of the seven methods docs/experimental_design.md
Section 15 requires: Random (chance floor), Popularity (training-history
baseline), and Content-Based Filtering (the first personalized method).
Collaborative Filtering, Knowledge Graph reasoning, and the Hybrid
combination are explicitly out of scope for Day 4.

## 1. Evaluation protocol

The protocol was **not invented for Day 4** -- it implements
docs/experimental_design.md Section 15, approved before Day 3's
generator was written. No methodological choice below was selected by
trying alternatives and picking the one that made a method look better.

| Element | Definition |
|---|---|
| Recommendation unit | One request (a real Day 2 job posting, `link`) |
| Candidate universe | Workers whose primary **or** secondary category matches the request's category, who had joined (`join_day`) by the request's simulated day, and whose country satisfies any location restriction. Capacity (Day 3's rolling 30-day/4-hire limit) is a generation-time mechanic, not a platform search constraint, and is not reapplied |
| Relevance | `HIRED` = grade 2, `SHORTLISTED`-not-hired = grade 1, otherwise 0. Binary metrics treat grade >= 1 as relevant. Source: Day 3's model-visible `interactions.csv` -- **not** the generator-private `truth.csv`/oracle utility |
| K | 10 |
| Averaging | Macro (unweighted mean) across included requests |
| Excluded requests | Two documented reasons, both counted: (1) candidate pool smaller than 20 (`min_candidate_pool_size`); (2) no relevant worker anywhere in the reconstructed candidate universe (a defensive check -- did not fire on this dataset, see Section 9) |
| Cold-start subset | Requests whose relevant set (grade >= 1) includes at least one of the 300 new-entrant workers |

**Why interactions.csv, not oracle truth.** The approved relevance
definition (HIRED=2/SHORTLISTED=1) is a property of the *historical
record*, which is model-visible by construction (Day 3's
`interactions.csv`). This is standard offline-evaluation methodology --
using an observed outcome as a ground-truth label is not the same as a
recommender consuming a generator-private variable. No Day 4 code reads
`data/processed/synthetic_private/`; see Section 10.

**Candidate universe validated against Day 3's own generation-time
eligibility.** Before implementing anything, a sample of 5,000 historical
(request, relevant-worker) pairs was checked against this exact
eligibility rule: **100% satisfied it.** This means Recall@10 is never
structurally capped by a candidate-universe definition that excludes
workers the generator itself considered eligible.

**No genuine ambiguity requiring a stop.** The approved design already
fixes candidate universe, relevance, K (from the Day 4 brief), and
exclusion handling. The only undetermined details were implementation
choices with no material effect on the experiment (documented in Section
12): tie-breaking rules, the CBF price-compatibility functional form, and
the predeclared weight grid.

## 2. Candidate universe statistics

| Split | Requests | Excluded (pool < 20) | Included | Mean pool size |
|---|---|---|---|---|
| Train | 8,400 | 520 | 7,880 | 83.0 |
| Validation | 1,200 | 74 | 1,126 | 92.0 |
| Test | 2,400 | 125 | 2,275 | 91.6 |

Zero requests were excluded for "no relevant worker in the candidate
universe" on any split -- every request has exactly 5 historical
shortlist interactions (Day 3's fixed shortlist size), and all 5 always
fall inside the reconstructed candidate universe.

## 3. Random baseline

Shuffles the candidate universe with `numpy.random.default_rng`, seeded
from `SHA-256(f"{master_seed}:{link}")[:8 bytes]` -- a dedicated,
deterministic stream per request, independent of call order or batching.
Truncates to K=10. Never reads any worker/request attribute at all
(pure structural sampling over whatever candidates it's handed), so by
construction it cannot leak generator-private data. Provides the
chance-performance floor; not tuned.

## 4. Popularity baseline

Signal: count of **training-split `HIRED`** interactions per worker
(docs/experimental_design.md Section 15: "Popularity (training hire
counts)"). Validation/test interactions are never counted -- verified
both in unit tests and independently in `scripts/verify_day4.py` by
recomputing the counts from raw interactions a second way. Ranking is
`sorted(candidates, key=lambda w: (-hire_count[w], w))`: deterministic
tie-break by ascending `worker_id`. Workers with zero training hires
(most workers, and **all 300 cold-start workers by construction** --
their `join_day` places them after the training period ends) simply tie
at 0; no synthetic history is fabricated for them.

## 5. Content-Based Filtering

```
CBF(request, worker) = w_skill * skill_similarity
                      + w_category * category_compatibility
                      + w_price * price_compatibility
```

| Component | Definition | Source fields |
|---|---|---|
| `skill_similarity` | Jaccard similarity between the request's controlled-vocabulary `skills_vocab` and the worker's **declared** skills (never true/private skills) | `sampled_requests.csv.skills_vocab`, `workers.csv.declared_skills` |
| `category_compatibility` | 1.0 if the request's category is the worker's primary category, 0.6 if it's a secondary category, else 0.0 | `workers.csv.primary_category` / `secondary_categories` |
| `price_compatibility` | `1 - |worker.price_tier - request.price_tier|`, bounded [0, 1]; a neutral score (0.5, configurable) when either side has no price/is unspecified pay | `workers.csv.price_tier`, `sampled_requests.csv.price_tier` |

All three features come from fields the worker/client would see in the
platform UI. **No experience feature**: Day 2's requests carry no
experience-level or seniority field, so there is no defensible
request-side experience signal to compare against -- adding one would
mean fabricating a request-side preference, which the Day 4 brief
explicitly forbids. This is a documented scope limitation, not an
oversight (Section 12).

**Why price_compatibility differs from Day 3's generator budget-fit.**
The generator's `B` component (docs/experimental_design.md Section 9.3)
is a step function with exponential decay past a threshold. CBF's price
feature is a plain linear distance. Using a different functional form
here is a deliberate anti-circularity choice, matching the same
principle the generator itself follows (Section 9.2): a content-based
method should not happen to compute the same formula the ground truth
was generated from.

## 6. Validation-only weight tuning

Seven weight combinations were predeclared in
`config/evaluation_config.yaml` (`cbf_weight_grid`) before any evaluation
ran, each summing to 1.0. Every combination was evaluated on
**validation only**; the primary selection metric is NDCG@10
(`cbf_tuning_primary_metric`, per the Day 4 brief). The winner was frozen
and evaluated on test **exactly once**.

| skill | category | price | Validation NDCG@10 | Selected |
|---|---|---|---|---|
| 0.60 | 0.20 | 0.20 | 0.1385 | |
| **0.50** | **0.25** | **0.25** | **0.1410** | **✓** |
| 0.40 | 0.30 | 0.30 | 0.1396 | |
| 0.34 | 0.33 | 0.33 | 0.1402 | |
| 0.30 | 0.50 | 0.20 | 0.1400 | |
| 0.30 | 0.20 | 0.50 | 0.1395 | |
| 0.70 | 0.15 | 0.15 | 0.1361 | |

**Selected weights: skill=0.50, category=0.25, price=0.25**
(`results/day4/selected_cbf_config.json`). The spread across the grid is
small (0.1361-0.1410), meaning CBF's validation ranking quality is not
very sensitive to this particular weight choice within the predeclared
range -- worth noting rather than overselling the tuning step's impact.

`scripts/verify_day4.py` independently re-evaluates the selected weights
on validation and confirms it reproduces the recorded NDCG exactly, and
confirms it is the tuning grid's best (or tied-best) validation score --
i.e., the selection provably came from validation, not test.

## 7. Validation results (K=10)

| Method | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|
| Random | 0.0660 | 0.1320 | 0.0880 | 0.0865 |
| Popularity | 0.0803 | 0.1606 | 0.1070 | 0.1171 |
| CBF | 0.0920 | 0.1840 | 0.1227 | 0.1410 |

n_included = 1,126, n_excluded = 74 (all pool_too_small), for every method (identical candidate universes).

## 8. Test results (K=10)

| Method | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|
| Random | 0.0682 | 0.1364 | 0.0909 | 0.0888 |
| Popularity | 0.0806 | 0.1611 | 0.1074 | 0.1190 |
| CBF | 0.0949 | 0.1897 | 0.1265 | 0.1394 |

n_included = 2,275, n_excluded = 125 (all pool_too_small). CBF > Popularity > Random on every metric, on both splits, consistent with expectations (though see Section 9 on how far from perfect all three remain).

## 9. Cold-start results

Subset: requests whose relevant set includes at least one of the 300 new-entrant workers. 442 such validation requests, 924 such test requests.

| Method | Split | n requests | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| Random | validation | 442 | 0.0633 | 0.1267 | 0.0845 | 0.0867 |
| Popularity | validation | 442 | 0.0683 | 0.1367 | 0.0911 | 0.0992 |
| CBF | validation | 442 | 0.0880 | 0.1760 | 0.1173 | 0.1320 |
| Random | test | 924 | 0.0697 | 0.1394 | 0.0929 | 0.0934 |
| Popularity | test | 924 | 0.0709 | 0.1418 | 0.0945 | 0.1036 |
| CBF | test | 924 | 0.0976 | 0.1952 | 0.1302 | 0.1446 |

**Popularity's structural cold-start disadvantage is real but smaller
than naive intuition suggests.** All 300 cold-start workers have zero
training hires by construction, but so does the *majority* of
established workers (median training hire count among established
workers is 2 -- docs/synthetic_data_report.md Section 10). Cold-start
workers are not uniquely disadvantaged relative to a typical never-hired
established worker under this signal; they just share the same
worker-ID tie-break lottery as everyone else at count 0. This is why
Popularity still clears Random on the cold-start subset, not because it
has any real information about new entrants.

**CBF remains the strongest of the three on cold-start**, since it can
score any worker from their visible profile regardless of hire history --
exactly the property the Day 4 brief asked CBF to provide.

## 10. Leakage safeguards

- Relevance labels come from `data/processed/synthetic/interactions.csv`
  (model-visible historical record), never from
  `data/processed/synthetic_private/truth.csv` or `calibration.json`.
- `src/evaluation/protocol.py`'s `load_protocol_data` never opens a path
  under `synthetic_private/`.
- Static source-code scan (`tests/test_day4_leakage.py`,
  `scripts/verify_day4.py`) confirms no recommender module's source
  mentions `quality`, `true_skills`, `proficiency`, `taste_`, `utility`,
  `noise`, or `synthetic_private`.
- Runtime check confirms the actual `workers`/`requests` frames handed to
  recommenders carry no column whose name matches those markers.
- Popularity's hire counts were independently recomputed from raw
  interactions filtered to `split == "train"` and compared for exact
  equality against the recommender's internal state.

## 11. Reproducibility

Two full runs of `scripts/run_day4_evaluation.py` produced byte-identical
`results/day4/` outputs (`diff -rq`, zero differences, including the
three PNG figures). `results/day4/day4_manifest.json` records SHA-256
hashes of every output file plus the frozen `generator_v1.yaml` hash and
the `evaluation_config.yaml` hash, with relative POSIX paths and no
timestamps.

## 12. Limitations and decisions not fixed by the brief

1. **No request-side experience feature** (Section 5) -- a scope
   limitation, not an oversight; adding one would require fabricating
   data the requests don't have.
2. **CBF price-compatibility functional form** (linear distance) was a
   free implementation choice, deliberately different from the
   generator's own budget-fit formula (Section 5).
3. **Deterministic tie-breaks** (Popularity and CBF: ascending
   `worker_id`; Random: a full permutation, so there is no tie) were not
   specified by the brief and were chosen for simplicity and
   reproducibility, not to favour any method.
4. **The predeclared CBF weight grid** (7 combinations) is a judgement
   call about coverage vs. exhaustiveness; the validation NDCG spread
   across it is small (0.1361-0.1410), so the specific grid contents
   likely do not matter much to the Day 4 conclusions.
5. **"pool_too_small" threshold (20)** and **K=10** were given directly
   by docs/experimental_design.md Section 15 and the Day 4 brief
   respectively, not chosen here.
6. Absolute metric values are low in a global sense (Precision@10 under
   10%) -- this is expected given the anti-circularity design of the
   Day 3 generator (docs/synthetic_data_report.md Section 6.3): ground
   truth deliberately depends on latent taste, quality, and noise that
   none of these three methods can see, so no method is expected to
   approach a high absolute score. Comparisons should be read
   *relative to each other and to Random*, not as absolute quality
   claims.

**No claim is made that CBF's lead over Popularity here predicts
anything about the eventual Hybrid's performance.** Day 4 provides three
reference points (a chance floor, a training-popularity baseline, and
one legitimate personalized method); Days 5-8 add Collaborative
Filtering, Knowledge Graph reasoning, and the Hybrid combination for a
complete comparison.

## Reproducing this report

```bash
python scripts/run_day4_evaluation.py
python scripts/verify_day4.py
python -m unittest discover -s tests
```
