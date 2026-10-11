# Day 8 Hybrid Recommender Report

**Objective.** Combine the frozen standalone CBF (Day 4), CF (Day 5) and
KG (Day 7) recommenders into a weighted hybrid. The weights are tuned on
validation only, the hybrid is evaluated once on the untouched test set,
and it is compared against every baseline using paired bootstrap
significance tests, cold-start, diversity, contribution and explanation
analyses.

The evaluation protocol is the unchanged Day 4–7 protocol: same
candidate universe, same pool < 20 exclusions (74 validation / 125 test),
HIRED = 2 / SHORTLISTED = 1, K = 10, macro averaging, and the same
cold-start subset.

All numbers come from `results/day8/` (`python
scripts/run_day8_evaluation.py`) and are checked independently by
`python scripts/verify_day8.py` (41 checks).

## 1. Frozen components (unchanged)

| Component | Frozen selection | Score used by the hybrid |
|---|---|---|
| CBF | skill 0.50 / category 0.25 / price 0.25 | `ContentBasedRecommender.score` |
| CF | user-based, k = 20, sum aggregation, HIRED 2 / SHORTLISTED 1, popularity fallback | Raw collaborative score from `CollaborativeFiltering._cf_scores`; a candidate with no collaborative evidence scores 0 |
| KG | damping 0.4; skill 0.20 / related 0.15 / category 0.10 / history 0.20 / reputation 0.35 | `KnowledgeGraphRecommender.scored` (the Day 7 temporal guards apply) |

All five frozen models are rebuilt from their recorded selections in
`src/recommenders/frozen_models.py`. Each must reproduce its frozen
Day 7 validation and test metrics exactly (to 1e-12), and the cold-start
worker diagnostics for all five frozen models must match Day 7, or the
run aborts.

The KG graph snapshot is rebuilt from the Day 6 CSV artifacts and must
match the fingerprint Day 7 verified against Neo4j. Day 8 does not
contact Neo4j.

**CF score definition.** Standalone CF ranks by its collaborative score,
then breaks ties with training popularity. Popularity is a tie-breaking
rule, not collaborative evidence, so the hybrid's CF input is the
collaborative score only. As a result, a CF-only "hybrid" (validation
NDCG@10 0.1095) differs from standalone CF (0.1155). The CBF-only and
KG-only hybrids reproduce their standalone models exactly, which
confirms the normalisation and combination code.

## 2. Score normalisation and formula

```
Hybrid(r, w) = w_cbf · CBF_norm + w_cf · CF_norm + w_kg · KG_norm,   w ≥ 0, Σw = 1
X_norm       = (X − min_pool X) / (max_pool X − min_pool X)
```

- Each component is normalised across the request's own candidate pool, independently of the others.
- If a component is constant across the pool (span ≤ 1e-12), it carries no ranking information for that request and contributes exactly 0 for every candidate. This applies, for example, to CF when the client has no training history.
- No label, global statistic or future data enters normalisation.
- Ties are broken by worker_id.

## 3. Predeclared grid and selection

`config/hybrid_config.yaml` (SHA-256 `4d956748…3f95`) was committed in
`f76ad34` before any hybrid metric existed. It contains 12 weight
vectors, none with a single component: near-equal, pairwise, CBF-, CF-
and KG-leaning, and minimal-CF.

Selection uses validation NDCG@10, then Recall@10, then the more
balanced vector (smaller Σ|w − 1/3|). The selected configuration was
written to `selected_hybrid_config.json` before the test requests were
built.

| # | CBF / CF / KG | Val NDCG@10 | Val Recall@10 |
|---|---|---|---|
| 0 | .34 / .33 / .33 | 0.1489 | 0.1934 |
| 1 | .40 / .20 / .40 | 0.1531 | 0.1963 |
| 2 | .40 / .40 / .20 | 0.1467 | 0.1936 |
| 3 | .20 / .40 / .40 | 0.1436 | 0.1904 |
| 4 | .50 / .25 / .25 | 0.1504 | 0.1941 |
| 5 | .25 / .50 / .25 | 0.1408 | 0.1876 |
| 6 | .25 / .25 / .50 | 0.1518 | 0.1933 |
| 7 | .60 / .20 / .20 | 0.1487 | 0.1966 |
| 8 | .20 / .60 / .20 | 0.1346 | 0.1817 |
| 9 | .20 / .20 / .60 | 0.1523 | 0.1931 |
| **10** | **.40 / .10 / .50** | **0.1561** | **0.1964** |
| 11 | .50 / .10 / .40 | 0.1548 | 0.1972 |

**Selected: CBF 0.40 / CF 0.10 / KG 0.50.** Its validation margin over
entry 11 is small (0.0013 NDCG@10). Every vector that gives CF 0.40 or
more ranks among the weakest.

## 4. Results

**Validation** (1,126 requests):

| Method | P@10 | R@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|
| Random | 0.0660 | 0.1320 | 0.0880 | 0.0865 |
| Popularity | 0.0803 | 0.1606 | 0.1070 | 0.1171 |
| CBF | 0.0920 | 0.1840 | 0.1227 | 0.1410 |
| CF | 0.0817 | 0.1634 | 0.1089 | 0.1155 |
| KG | 0.0972 | 0.1945 | 0.1297 | 0.1536 |
| **Hybrid** | **0.0982** | **0.1964** | **0.1310** | **0.1561** |

**Test** (2,275 requests; evaluated once):

| Method | P@10 | R@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|
| Random | 0.0682 | 0.1364 | 0.0909 | 0.0888 |
| Popularity | 0.0806 | 0.1611 | 0.1074 | 0.1190 |
| CBF | 0.0949 | 0.1897 | 0.1265 | 0.1394 |
| CF | 0.0792 | 0.1583 | 0.1056 | 0.1149 |
| KG | 0.0966 | 0.1932 | 0.1288 | 0.1499 |
| **Hybrid** | **0.0972** | **0.1945** | **0.1296** | **0.1531** |

## 5. Paired bootstrap significance (test)

`src/evaluation/significance.py`:
- 1,000 request-level resamples (Section 15), with RNG `SeedSequence([42, 8])` and the same resamples for every comparison;
- 95% percentile CI;
- two-sided shift-method p-value, `p = (1 + #{|d* − d̄| ≥ |d̄|}) / (B + 1)`, so the smallest possible p is 1/1001 ≈ 0.001;
- Holm adjustment across the four baselines, separately for each metric;
- n = 2,275 paired requests.

| Metric | Hybrid − | Mean diff | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| NDCG@10 | KG | +0.0032 | [−0.0017, +0.0083] | 0.224 | 0.224 |
| NDCG@10 | CBF | +0.0137 | [+0.0083, +0.0195] | 0.001 | 0.004 |
| NDCG@10 | CF | +0.0382 | [+0.0305, +0.0461] | 0.001 | 0.004 |
| NDCG@10 | Popularity | +0.0341 | [+0.0263, +0.0424] | 0.001 | 0.004 |
| Recall@10 | KG | +0.0012 | [−0.0042, +0.0069] | 0.673 | 0.673 |
| Recall@10 | CBF | +0.0047 | [−0.0013, +0.0110] | 0.134 | 0.268 |
| Recall@10 | CF | +0.0361 | [+0.0277, +0.0440] | 0.001 | 0.004 |
| Recall@10 | Popularity | +0.0333 | [+0.0251, +0.0416] | 0.001 | 0.004 |

**Interpretation.**
- The hybrid's NDCG@10 is significantly higher than CBF, CF and Popularity.
- **It is not significantly different from KG-only.** The CI includes 0 and p = 0.22.
- On Recall@10 the hybrid is significantly better only than CF and Popularity.
- The defensible claim is therefore: the hybrid matches the strongest single component (KG) and significantly beats the others on NDCG@10. It is **not** a demonstrated improvement over KG-only.

## 6. Cold start

Same subset as Days 4–7: requests whose relevant set contains a
cold-start worker (442 validation / 924 test).

| Method | Test P@10 | Test R@10 | Test F1@10 | Test NDCG@10 | Cold Recall@10 | Cold exposure ratio |
|---|---|---|---|---|---|---|
| Random | 0.0697 | 0.1394 | 0.0929 | 0.0934 | 0.1465 | 1.03 |
| Popularity | 0.0709 | 0.1418 | 0.0945 | 0.1036 | 0.0000 | 0.00 |
| CBF | 0.0976 | 0.1952 | 0.1302 | 0.1446 | **0.2083** | **1.06** |
| CF | 0.0687 | 0.1374 | 0.0916 | 0.0977 | 0.0000 | 0.00 |
| KG | 0.0961 | 0.1922 | 0.1281 | 0.1497 | 0.1104 | 0.54 |
| **Hybrid** | **0.0975** | 0.1950 | **0.1300** | **0.1538** | 0.1515 | 0.73 |

On validation, the hybrid's cold-start-request NDCG@10 (0.1387) is also
above both KG (0.1348) and CBF (0.1320).

**The trade-off.**
- The hybrid recovers much of the cold-start visibility KG loses: cold-start workers' share of top-10 slots rises from 5.3% to 7.2%, against 9.8% of relevant items; the exposure ratio rises from 0.54 to 0.73; Cold Recall@10 rises from 0.110 to 0.152.
- It keeps KG-level ranking quality on cold-start requests.
- It does not reach CBF's proportional exposure (1.06) or CBF's Cold Recall@10 (0.208).
- CF contributes nothing for cold-start workers, by construction.

## 7. Diversity and catalogue coverage

`src/evaluation/diversity.py`, test split:
- The eligible catalogue is every worker in at least one included candidate pool (3,000).
- Intra-list diversity is 1 − mean pairwise Jaccard similarity of declared skills (Section 15).
- Top-1% share is the share of all top-10 slots taken by the 30 most-recommended workers.

| Method | Unique workers | Catalogue coverage | Exposure Gini | Top-1% share | Max appearances | Intra-list diversity |
|---|---|---|---|---|---|---|
| Random | 2,878 | 0.959 | 0.401 | 3.6% | 37 | 0.899 |
| Popularity | 626 | 0.209 | 0.916 | 27.1% | 353 | 0.876 |
| CBF | 2,873 | 0.958 | 0.468 | 5.3% | 64 | 0.815 |
| CF | 2,012 | 0.671 | 0.717 | 11.2% | 143 | 0.882 |
| KG | 2,299 | 0.766 | 0.695 | 12.5% | 141 | 0.829 |
| **Hybrid** | 2,625 | **0.875** | **0.596** | **8.7%** | 99 | 0.816 |

Compared with KG-only, the hybrid spreads exposure more widely: coverage
0.766 → 0.875, Gini 0.695 → 0.596, top-1% share 12.5% → 8.7%. It remains
more concentrated than CBF. Intra-list skill diversity is essentially
CBF's level. The CBF component favours skill-similar lists, so this is
slightly below KG.

## 8. Component contribution analysis

Over all top-10 test recommendations (22,750):

| Component | Weight | Mean contribution | Mean share of score | Positive | Dominant |
|---|---|---|---|---|---|
| CBF | 0.40 | 0.312 | 43.5% | 99.9% | 22.6% |
| CF | 0.10 | 0.014 | 2.0% | 34.8% | **0.0%** |
| KG | 0.50 | 0.393 | 54.6% | 100% | 77.4% |

**In practice this is a CBF + KG hybrid.**
- CF is the largest contributor in no recommendation, and contributes anything at all to only 35% of them.
- Removing CF costs only 0.0011 validation NDCG@10 (Section 9).
- A likely reason is overlap: the KG `history` component already includes training-only shared-client (co-hire) paths, which carry much of the signal CF has.

I report this as found and did not retune.

## 9. Ablation (validation, analysis only)

Each variant drops one component from the selected weights and
renormalises the rest. This was not used for selection.

| Variant | CBF / CF / KG | Val NDCG@10 | Δ | Δ Recall@10 |
|---|---|---|---|---|
| Full | .40 / .10 / .50 | 0.1561 | — | — |
| − CBF | 0 / .17 / .83 | 0.1502 | −0.0059 | −0.0039 |
| − CF | .44 / 0 / .56 | 0.1550 | −0.0011 | −0.0014 |
| − KG | .80 / .20 / 0 | 0.1413 | −0.0148 | −0.0091 |

KG contributes most, CBF clearly adds to it, and CF's marginal effect is small.

## 10. Hybrid explanations and fidelity

`HybridRecommender.explain` reports, for each recommendation:
- the final score;
- each component's raw, normalised and weighted contribution;
- the components cited in order of contribution;
- the dominant component.

It also states explicitly which components contributed nothing. When
the KG component contributes, the frozen Day 7 KG explanation for the
same candidate pool is attached: its summary, plus its individually
strongest paths. The explanation never presents the hybrid as purely
graph-based. Example (test sample 1):

> Recommended because knowledge-graph evidence (KG) contributed 0.38,
> profile match (CBF) contributed 0.29 (hybrid score 0.68); no
> contribution from collaborative history (CF). Strongest KG evidence:
> Average training-period rating 5.00/5 from 1 review (smoothed 4.54);
> Has required skill 'Translation'; …

**Explanation fidelity** (the Section 15 metric, now computed against
the actual hybrid score), on test:

| Measure | Value |
|---|---|
| First cited factor is the largest contribution to the hybrid score | **100%** (22,750 / 22,750; holds by construction) |
| Contributions reconstruct the hybrid score exactly | 100% |
| Fidelity a *KG-only* explanation would achieve | **77.4%** |
| Fidelity a *CBF-only* explanation would achieve | 22.6% |
| Fidelity a *CF-only* explanation would achieve | 0.0% |

The single-source rows are the informative ones. If the hybrid were
explained only through KG paths, as the Day 7 explanations do for KG,
the explanation would fail to cite the largest factor for about 23% of
hybrid recommendations. This is why the hybrid explanation cites
contributions by source.

**Samples.** Thirty seeded (seed 42) test hybrid explanations with KG
detail are in `hybrid_explanation_samples.json`: 23 are KG-dominant and
7 CBF-dominant. No new human rubric study was created, and the Day 7
blank rubric sheet is untouched (the verifier checks this).

## 11. Leakage safeguards

- **What the hybrid can see.** Its API receives only `(link, candidates, k)` and the three frozen component scores.
- **Static tests.** The hybrid's scoring code contains no reference to relevance, interactions, splits, events, ratings or test data, and no generator-private markers. It contains no component reimplementation (no Jaccard, cosine or path enumeration), and it calls the frozen scoring interfaces.
- **Temporal safety.** This is inherited and re-verified. CF uses training interactions only, KG uses the training-only snapshot (its temporal guards are active), and CBF uses static profile content.
- **Validation-only selection.** The tuning function receives only the requests it is given (a test confirms it scores no others). Tuning rows are validation-only. The selection is written before test requests exist.
- **Frozen integrity.** Day 4/5/7 configs, results and manifests, the KG config and the generator are byte-unchanged (checked by tests and the verifier). Prior result directories were not modified.

## 12. Reproducibility

- A second full run produced **byte-identical** output in all 18 files in `results/day8/`, including the tuning, selected config, metrics, significance, diversity, contributions, explanations, figures and manifest.
- Determinism comes from deterministic component scores, ties broken by worker_id, a seeded bootstrap (`SeedSequence([42, 8])`), a seeded explanation sample, and no timestamps or absolute paths.
- The manifest records the `hybrid_config.yaml` hash and the SHA-256 hashes of every frozen input.

## 13. Limitations

1. **No significant gain over KG-only.** The hybrid's advantage over KG is within bootstrap noise on both NDCG@10 and Recall@10.
2. **CF is nearly inert in the selected hybrid** (2% of the score, never dominant).
3. **Bootstrap caveats.** Requests are resampled as if independent, but some clients place several test requests, so the CIs may be somewhat narrow. With 1,000 resamples, the smallest possible p-value is about 0.001.
4. **One generator configuration.** All results come from the balanced generator (`generator_v1.yaml`). The content-leaning and collaborative-leaning sensitivity configurations in `docs/experimental_design.md` Section 14 have not been run. That is needed before calling any conclusion robust.
5. **Reputation bias is inherited.** The KG component carries Day 7's reputation dominance, which lowers cold-start exposure compared with CBF.
6. **Structural explanation metrics only.** Explanation fidelity is a structural measure; no user study was run (Section 15).
7. **CF adapter uses a private method.** The CF score comes from `CollaborativeFiltering._cf_scores`, an existing but private method. Calling it avoided modifying the frozen Day 5 module.

## 14. Decisions needing review

1. **CF score definition.** The hybrid uses the raw collaborative score without the popularity fallback, via the private `_cf_scores`.
2. **How to present the hybrid.** It is effectively CBF + KG. Should the thesis say so explicitly?
3. **Claim framing.** The hybrid equals KG-only (no significant difference) and beats CBF, CF and Popularity on NDCG@10.
4. **Generator sensitivity analysis** (Section 14). It has not been run. Should it be scheduled?
5. **Whether hybrid explanation samples should get a human rubric.** They are prepared, but no rubric study has been started.

## Reproducing this report

```bash
python scripts/run_day8_evaluation.py
python scripts/verify_day8.py
python -m unittest discover -s tests
```
