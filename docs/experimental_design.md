# Experimental Design: Data Foundation, Synthetic Data, and Evaluation

**Project:** Knowledge Graph-Enhanced Hybrid Recommender System for Digital Worker-Service Platforms
**Status:** Draft v1 (end of Day 1). Pending approval before implementation.
**Numbers source:** every dataset statistic below is reproducible with
`python scripts/verify_day1.py data/raw/upwork-jobs.csv docs/day1_stats.json`.

Each decision is labelled:

- **[DOC]**: required by the project documentation
- **[IMPL]**: implementation decision
- **[ASSUMPTION]**: assumption that cannot be verified from available data
- **[EXP]**: experimental choice that may be varied during evaluation

---

## 1. Real Dataset Source

The real-world data source is the *Upwork Job Postings Dataset 2024* (Asaniczka, 2024), obtained from Kaggle **[DOC]**. The file (`upwork-jobs.csv`, 72 MB) contains **53,058 job postings** and **9 columns**:

| Column | Coverage | Content |
|---|---|---|
| `title` | 100% | Job title |
| `link` | 100% | Unique posting URL (0 duplicates); used as the job identifier |
| `description` | 100% | Free-text description followed by a platform metadata footer |
| `published_date` | 100% | UTC timestamp |
| `is_hourly` | 84.5% | 22,956 hourly; 21,873 fixed-price; 8,229 unspecified |
| `hourly_low`, `hourly_high` | 43.3%, 41.7% | Hourly range (hourly jobs only) |
| `budget` | 41.2% | Fixed-price budget (fixed jobs only) |
| `country` | 97.8% | Client country (185 distinct) |

The dataset contains **no dedicated skill or category columns**. Both are embedded in a metadata footer inside `description` (Section 4).

## 2. Dataset Limitations

1. **Demand side only.** The dataset has no worker profiles, applications, hires, ratings, or reviews. All worker-side and interaction data must be synthetic.
2. **No client identifier.** Postings cannot be grouped by the client who wrote them. Available client-side information is limited to country, pay type, and budget. Client entities are therefore synthetic.
3. **Very short effective time window.** Timestamps run from 2 Dec 2023 to 23 Feb 2024. However, **99.71% of postings fall between 12 and 23 February 2024**. Daily counts also show collection gaps: 9 postings on 21 Feb and 1,803 on 15 Feb, against roughly 5,000–7,000 on other days. The timestamps reflect the RSS collection process, not client behaviour over time.
4. **Single platform, RSS-sourced.** Postings are a feed snapshot from one platform. Hidden, private, or invite-only jobs are absent.
5. **Geographic skew.** The United States accounts for 40.23% of rows, the United Kingdom 8.21%, and India 6.80%. 1,141 rows have no country.
6. **Heavy-tailed pay values.** Fixed budgets range from $5 to $1,000,000 (median $100; 37 jobs exceed $50,000). Hourly upper bounds reach $999.
7. **Long-tailed skill labels.** There are 5,912 distinct skill labels, most of them rare (Section 6).
8. **Location is eligibility, not proximity.** Location information is client country plus, for about 10% of jobs, a restriction ("Only freelancers located in the United States/United Kingdom may apply"). No city-level or distance information exists. For remote digital services, location is modelled as an eligibility constraint only.
9. **No certification data.** Certifications mentioned in the documentation cannot be grounded in this dataset. They are excluded from the MVP rather than fabricated **[IMPL]**.

## 3. Cleaning Decisions

The raw CSV is **never modified**. Cleaning produces a processed dataset in which excluded records are **flagged, not deleted**, so every exclusion stays auditable **[IMPL]**.

### 3.1 HTML entity decoding

Descriptions contain HTML entities (e.g. `&amp;`). They are decoded with `html.unescape` before any parsing.

### 3.2 Duplicate postings: remove 22 rows

There are **22 groups of exactly two postings** with identical title and description (44 rows). Within every group, pay fields and country are identical; only `link` and `published_date` differ, and the two copies were posted at most 0.9 minutes apart. These are double submissions, not distinct demand.

- **Rule:** keep the record with the most non-null values in `is_hourly, hourly_low, hourly_high, budget, country`; break ties by lexicographically smallest `link`.
- **Observed effect:** completeness is equal within all 22 groups, so the deterministic tie-break decides in practice. The completeness rule is retained as the stated policy.
- **Result:** 53,058 → **53,036** rows. Removed rows are flagged `is_duplicate = True`.

**Assessment:** appropriate. Keeping both copies would double-count identical demand in category, skill, and budget distributions.

### 3.3 Jobs without extractable skills: exclude 515 from modelling

515 jobs have a category but no `Skills:` block. They cannot be expressed as skill requirements, which content-based matching, skill-based Knowledge Graph paths, and the synthetic generator all need.

- **Rule:** flag `has_skills = False`; exclude from the modelling dataset; retain in the processed file.
- **Bias note:** the exclusion is not uniform. 150 of the 515 (29%) are *Mobile App Development* jobs, about 11% of that category. This is documented as a minor distortion of that category's representation.
- **Result:** 53,036 → **52,521** rows.

**Assessment:** appropriate for the recommendation-development dataset, with the bias disclosed.

### 3.4 Category scope: restrict to categories with ≥100 jobs

Calculated on the 52,521-row set:

| Minimum jobs per category | Categories kept | Categories removed | Jobs retained |
|---|---|---|---|
| 20 | 157 / 204 | 23.0% | 99.05% (52,021) |
| 50 | 122 / 204 | 40.2% | 96.85% (50,867) |
| **100** | **88 / 204** | **56.9%** | **92.34% (48,499)** |
| 200 | 62 / 204 | 69.6% | 85.63% (44,973) |

**Recommendation: adopt the ≥100 threshold for the MVP [IMPL].** The threshold removes more than half of the category labels but only 7.66% of jobs. The deciding factor is synthetic grounding:

- With 12,000 sampled requests (Section 8), a category at the threshold receives about 23 requests. Below 100 jobs, a category would receive fewer than 25 requests.
- Worker pools in such categories would consist almost entirely of synthetic workers whose skill profiles are estimated from very few real postings.
- Every category needs a candidate pool of at least 20 workers for top-10 ranking to be non-trivial (Section 8). At 122 categories (≥50 threshold) the floor alone would require 2,440 workers, leaving little room for demand-proportional allocation.

Excluded categories are flagged `in_scope = False` and are reported as a scope limitation.

### 3.5 Budgets and rates: stored value vs analysis value

| Value type | Definition | Used for |
|---|---|---|
| **Stored value** | Original `budget`, `hourly_low`, `hourly_high`, unchanged, including outliers | PostgreSQL, Neo4j, UI display |
| **Analysis value** | Winsorised at the 1st/99th percentile within pay type | Descriptive statistics and charts only |
| **Model feature** | *Price tier*: percentile rank of the job's pay within its category and pay type (0–1) | Synthetic generator; recommenders |

Winsorisation bounds on the eligible request pool: fixed budget $5–$8,211.50 (199 above p99; mean $695.84 stored vs $437.53 winsorised); `hourly_low` $3–$75; `hourly_high` $5–$200.

Percentile ranks are robust to outliers by construction, so no stored value needs capping for modelling. A $1,000,000 budget simply receives price tier ≈1.0. For hourly jobs, pay is the midpoint of `hourly_low` and `hourly_high`, or `hourly_low` when the upper bound is missing (795 raw rows). The 8,229 unspecified-pay jobs have no pay information at all (verified: their footers contain neither `Budget:` nor `Hourly Range:`). They receive `price_tier = null` and a neutral budget-compatibility value.

## 4. Metadata and Skill Extraction

Every posting ends with a platform-generated footer of the form:

```
Budget: $500 | Hourly Range: $10.00-$30.00
Posted On: February 17, 2024 09:09 UTC
Category: Social Media Marketing
Skills: Facebook Advertising, Google Ads, ...
Skills: Facebook Advertising, Google Ads, ...     (repeated block)
Location Requirement: Only freelancers located in the United States may apply.   (optional)
Country: United States                                                           (optional)
click to apply
```

**Anchoring rule [IMPL].** Parsing starts at the **last** occurrence of `Posted On:`. A first prototype matched the first `Category:` anywhere in the text. Because some clients write "Category:" inside their own job description, it produced corrupted "categories" up to 2,472 characters long. Footer anchoring yields exactly 204 categories with a maximum length of 45 characters, and a footer was found in all 53,058 rows.

**Skill parsing:** take the first `Skills:` block only, split on commas, collapse internal whitespace, strip, and remove within-job duplicates while preserving order.

**Validation:** the footer `Country:` agrees with the `country` column in 99.99% of rows. The `country` column remains the stored source.

## 5. Category Extraction

The category is the footer text between `Category:` and the next footer marker. All 53,058 rows have exactly one category, and no normalisation is required: the labels are platform-controlled and case-consistent. The category taxonomy is used as-is **[IMPL]**.

## 6. Controlled Skill Vocabulary

### 6.1 Normalisation rule

1. Decode HTML entities; collapse whitespace; strip.
2. De-duplicate skills within a job.
3. **Exact matching only.** There are no case-variant collisions (0 labels differ only by case), so case-folding is unnecessary. A lowercase key is still stored for robustness.
4. **No fuzzy, stemming, or synonym merging.** A naive singular/plural test on this data produced only false merges: *SAS* ↔ *Sass* (a statistics package vs a CSS preprocessor) and *Canvas* ↔ *Canva*. Manual synonym mapping of thousands of labels is unverifiable within the timeline.

### 6.2 Vocabulary threshold

The vocabulary is computed on the **in-scope modelling set** (48,499 jobs; 5,321 distinct skills), counting the number of jobs in which each skill appears:

| Minimum job frequency | Vocabulary size | Skill mentions covered | Jobs left with 0 vocabulary skills |
|---|---|---|---|
| 5 | 2,494 | 97.92% | 162 |
| 10 | 1,773 | 95.96% | 307 |
| **20** | **1,187** | **92.63%** | **561** |
| 50 | 650 | 85.76% | 1,152 |

**Decision: skills appearing in ≥20 in-scope jobs form the controlled vocabulary [IMPL].** At 1,187 skills the vocabulary covers 92.63% of all skill mentions. Each vocabulary skill has enough co-occurrence evidence (≥20 jobs) for the skill-relatedness estimates used in Section 9 and by the Knowledge Graph. It also yields a Skill node set that loads and visualises easily in Neo4j.

*Reconciliation with earlier figures:* 1,339 skills reach ≥20 jobs on the raw file and 1,338 after de-duplication and zero-skill removal. Restricting to in-scope categories reduces this to 1,187, because skills specific to excluded categories lose support.

### 6.3 What happens to rare skills

Rare skills are **retained, not discarded**:

- Every job keeps its full parsed list in `skills_raw`.
- `skills_vocab` holds the intersection with the vocabulary and is the only list used for modelling.
- `rare_skills.csv` records every out-of-vocabulary skill and its job frequency, giving a complete audit trail.
- Rare skills do not become Knowledge Graph Skill nodes and are never assigned to synthetic workers.
- The 561 in-scope jobs whose skills are *all* rare stay in the processed dataset with `eligible_request = False`. They are not sampled as service requests.

**Eligible request pool: 47,938 jobs across 88 categories.**

## 7. Synthetic Client Generation

The dataset has no client identifiers, so clients are synthetic **[DOC]**. To keep them grounded, each sampled real job is assigned to a synthetic client from the same country **[IMPL]**.

1. **Client count per country** is proportional to that country's share of sampled requests. Jobs with a missing country are assigned to clients with country `Unspecified`.
2. **Activity level** (number of requests per client) is drawn from a heavy-tailed distribution with mean ≈6, minimum 1, and cap 40 **[ASSUMPTION]**. Many clients post rarely, and a few post often, which produces realistic sparsity and some client cold-start.
3. **Category coherence:** each client draws a primary category from its country's category distribution. About 65% of the client's requests come from that category and the remainder from any in-scope category in the country **[ASSUMPTION]**.
4. **Latent taste vector** `u_c ~ N(0, I_8)`. It represents unobserved preferences such as communication style, working hours, or portfolio aesthetics. It is **never exposed** to any recommender or stored in the application databases.
5. **Observable attributes**, all derived from assigned real jobs: country, preferred pay type, and median price tier.
6. **Synthetic timeline:** each client has an activity start day `~ U(0, 330)` on a 365-day simulated calendar, and its requests are placed uniformly between the start day and day 365. Real `published_date` is kept as provenance but not used for ordering, because of Limitation 3.

## 8. Synthetic Worker Generation

**Count per primary category:** proportional to the category's share of eligible requests, with a floor of **20 workers** per category **[IMPL]**. With 3,000 workers, 52 of the 88 categories sit at the floor.

**Secondary categories:** each worker has 0–2, chosen from the five categories whose skill-frequency profiles are most similar (cosine similarity) to the primary category.

**Latent (hidden) attributes**, used only by the generator:

| Attribute | Generation |
|---|---|
| Quality `q_w` | Beta-distributed, moderately correlated with experience (target correlation ≈0.4) **[ASSUMPTION]** |
| True skill set and proficiency `p_w(s)` | 5–12 skills sampled without replacement from the primary category's real skill frequencies, plus 1–3 from secondary categories; proficiency ∈ [0,1] from a Beta distribution shifted upward by experience |
| Taste embedding `v_w` | `N(0, I_8)` |

**Observable attributes**, visible to recommenders and the UI:

| Attribute | Generation |
|---|---|
| Declared skills | True skills with proficiency ≥0.4 are declared with probability 0.85, others with probability 0.40. With probability 0.30 each, up to two "padding" skills are added: popular category skills the worker does not truly hold **[ASSUMPTION]** |
| Experience level | Entry / Intermediate / Expert with proportions 40/40/20 **[ASSUMPTION]**, plus years in a level-consistent range |
| Rate and price tier | Sampled from the category's real pay distribution (Section 3.5); entry-level workers are biased toward lower percentiles |
| Country | Sampled from the dataset's top-20 client countries, square-root smoothed to reduce US dominance **[ASSUMPTION: no worker-location data exists]**. Every (category, required country) pair that appears in location-restricted requests is guaranteed at least 10 eligible workers |
| Join day | Day 0 for established workers; validation-start day for the new-entrant cohort (Section 12) |
| Completed jobs, average rating | **Derived only from simulated training-period interactions**, never sampled independently. An independently sampled "completed jobs" count correlated with quality would leak latent quality to any method that reads the profile |

## 9. Synthetic Interaction Generation

### 9.1 Why interactions must not be generated from skill overlap

Suppose hires were generated as `hire ∝ similarity(request skills, worker declared skills) + noise`. A content-based recommender computing that same similarity would be approximating the **generating function itself**. Its measured accuracy would reflect the design of the simulator, not the merit of the method. Collaborative filtering and Knowledge Graph reasoning would be structurally disadvantaged, because the ground truth would contain no signal they can see better. A hybrid could not beat content-based filtering except by chance. The comparison central to the research objectives would be circular.

### 9.2 Design principles

1. **Latent-observable separation.** The generator uses hidden variables (true proficiency, quality, taste). Recommenders see only noisy or partial observable proxies (declared skills, experience, ratings from training interactions).
2. **Multiple independent factors,** each defensible as a real hiring consideration, with no single observable-linked factor dominating.
3. **Different functional forms.** The generator's skill fit (proficiency-weighted coverage with partial credit for related skills) is deliberately different from any similarity measure a recommender will use.
4. **Stochastic choice.** Hires are *sampled*, not chosen by argmax, so even a perfect model cannot reach precision 1.0.
5. **Freeze before modelling.** The generator configuration and seed are committed and tagged before any recommender code is written. Generator parameters are calibrated only against data-realism targets (hire rate, rating shape, hire concentration), **never** against recommender performance.

### 9.3 Simulation procedure

Requests are processed in chronological order of simulated time. For each request `r` by client `c`:

**Step 1: Eligible pool.** Workers whose primary or secondary categories include `r`'s category, who joined on or before `r`'s timestamp, and who satisfy any location requirement.

**Step 2: Applicants.** Sample up to 30 workers from the pool without replacement, with probability proportional to `0.3 + S(r,w)`. Workers self-select using their *true* competence, which they know and the platform does not.

**Step 3: Utility.** For each applicant, compute:

```
U(c, w, r) = β_S·S̃ + β_T·T̃ + β_Q·Q̃ + β_B·B̃ + β_C·C̃ + ε,   ε ~ N(0, σ²)
```

where each component is standardised across all generated request–applicant pairs:

| Component | Definition | Real-world meaning |
|---|---|---|
| `S` true skill fit | Mean over required vocabulary skills `s` of `max(p_w(s), λ·max_{s′∈N(s)} p_w(s′))`, where `N(s)` is the 10 skills with the highest NPMI co-occurrence with `s` in the in-scope job corpus; λ = 0.5 | Can the worker actually do the job, including transferable skills |
| `T` taste | `u_c · v_w / √8` | Unobserved client-specific fit |
| `Q` quality | `q_w` | Reliability and craftsmanship |
| `B` budget fit | 1 if worker price tier ≤ request price tier + 0.1, otherwise exponential decay; 0.7 if request pay is unspecified | Affordability |
| `C` category fit | 1.0 if request category is the worker's primary category, 0.6 if secondary | Specialisation |

**Default weights (balanced configuration) [EXP]:** β_S 0.30, β_T 0.25, β_Q 0.20, β_B 0.15, β_C 0.10. σ is calibrated so that noise accounts for roughly 20% of utility variance.

**Step 4: Shortlist.** Sample 5 applicants without replacement using Plackett–Luce probabilities `∝ exp(U / T_s)`, with T_s = 1.0 (broad). Recorded as `SHORTLISTED`.

**Step 5: Hire.** With probability `sigmoid(α·(max U − θ))`, calibrated so that about 85% of requests result in a hire **[ASSUMPTION]**, one shortlisted worker is sampled with probability `∝ exp(U / T_h)`, T_h = 0.3 (sharp). Recorded as `HIRED`.

**Step 6: Capacity.** A worker cannot receive more than 4 hires in any rolling 30-day simulated window. At capacity, a worker is excluded from subsequent pools until capacity frees up. This prevents a few high-utility workers from absorbing all demand.

### 9.4 Signal access matrix

This table records which generator factors each method can observe. It is reported alongside results so that readers can judge fairness.

| Generator factor | CBF | CF | KG |
|---|---|---|---|
| True skill fit `S` | Partial (declared skills, noisy) | Indirect (co-hire patterns) | Partial (declared skills + skill relatedness) |
| Related-skill credit (λ) | No | Indirect | Partial (relatedness from background corpus, Section 14) |
| Taste `T` | No | **Yes** (learnable from history) | Indirect (shared-client paths) |
| Quality `Q` | Weak (experience level) | Indirect (training ratings, hire frequency) | Partial (training `REVIEWED` edges) |
| Budget fit `B` | Yes | No | If price tier is modelled |
| Category fit `C` | Yes | No | Yes |
| Location | Applied identically to all methods as an eligibility filter | | |

**Known residual bias.** The ground truth is a mixture of signals, and each method observes a different subset. This structure may *favour a hybrid* by construction. Section 14 describes how this risk is examined rather than assumed away.

## 10. Rating and Review Generation

- **Completion:** 90% of hires complete and receive a rating **[ASSUMPTION]**.
- **Rating:** `r* = 0.5·q̃_w + 0.3·S̃ + 0.2·B̃ + N(0, 0.5²)`, mapped to integer 1–5 via thresholds chosen to produce a J-shaped distribution skewed toward high ratings, as commonly described for online marketplaces. The specific target proportions (≈65% five-star, ≈20% four-star, ≈15% three stars or below) are an **[ASSUMPTION]**, because the dataset contains no rating data.
- **Written review:** attached to 60% of ratings **[ASSUMPTION]**. Text is produced from rating-band templates (about six per band) with slots for the request category and one required skill.
- **Scope statement:** review text is generated for **display and demonstration only**. No recommender uses review text, so no claim of text-based or sentiment-based recommendation is made. Large language models are not used for generation.

## 11. Train / Validation / Test Split

- **Method:** a global temporal split on **simulated** request timestamps: the first 70% of requests form the training set, the next 10% the validation set, and the last 20% the test set **[IMPL]**. A temporal split mirrors deployment, where a model trained on the past recommends for future requests. It also rules out training on interactions that occur after the request being evaluated.
- **Fitting:** all recommenders are fitted on **training data only**, for both validation and test evaluation.
- **Tuning:** hybrid weights and any hyperparameters are tuned on validation. Test metrics are computed **once** per final configuration.
- **Derived artefacts** are rebuilt from training data only. This covers the CF interaction matrix, worker completed-job counts, average ratings, Knowledge Graph `HIRED`/`REVIEWED` edges, and client `INTERESTED_IN` edges.
- **Client cold-start:** clients whose first request falls in the test period have no training history. They are reported as a separate segment; CF cannot score them, and the hybrid falls back to its other components.

## 12. Cold-Start Definition

**Cold-start worker [EXP]:** a member of the **new-entrant cohort**, a stratified random 10% of workers (300; stratified by primary category) whose join day is set to the **first day of the validation period**. The chronological simulation never includes them in a training-period pool, so they have **zero training interactions by construction**. This is cleaner than deleting history, because no interactions exist that could have shaped other workers' histories.

*Why 10%:* validation and test together contain about 3,600 requests. Entrants make up about 10% of eligible pools and enter applicant sampling on equal terms, so they are expected to receive about 10% of roughly 18,000 shortlist slots. That is about 1,800 relevant cold-start events (about 1,200 in test), enough for stable estimates with confidence intervals. A larger cohort would deplete training data.

**Sparse worker:** an established worker whose training hire count is at or below `N`. `N` is the 33rd percentile of training hire counts among established workers. It is computed once after generation, **before any recommender is run**, and frozen in `generation_report.json`. The segment therefore covers roughly the lower third of existing workers, instead of an arbitrary threshold picked before the distribution is known. Expected value: 1–2 hires.

**Established worker:** training hire count above `N`.

**Which components can recommend each segment:**

| Segment | CBF | CF | KG |
|---|---|---|---|
| Cold-start | Yes (declared skills, category, rate, country) | **No** (no interaction vector) | Yes, via skill, relatedness, category, and location paths; not via hire-history paths |
| Sparse | Yes | Weakly | Yes |
| Established | Yes | Yes | Yes |

**Hybrid handling of missing component scores [EXP]:** default is **per-worker weight renormalisation** over the components that produce a score. The ablation is **zero-fill**, where missing scores count as 0. Comparing the two shows directly how much a naive hybrid inherits CF's cold-start penalty.

**Cold-start metrics (test set):**

- **Cold Recall@K:** among test requests whose relevant set contains at least one cold-start worker, the fraction of those cold-start workers ranked in the top K.
- **Cold exposure ratio@K:** the share of top-K slots given to cold-start workers, divided by their share of relevant items. A value of 1 means proportional visibility; this operationalises the documentation's "worker visibility" objective.
- Precision, recall, and F1 reported separately for the cold, sparse, and established segments.

## 13. Random Seed

A single master seed **42** is set in `config/data_config.yaml`. Each pipeline stage (request sampling, clients, workers, simulation, ratings, reviews, cohort assignment) receives its own child stream via `numpy.random.SeedSequence(42).spawn(...)`. Changing one stage's logic therefore does not silently alter the random draws of the others. Where the pipeline samples from data frames, it sorts them by a stable key first so that results do not depend on row order.

## 14. Leakage Prevention

| Risk | Mitigation |
|---|---|
| Ground truth generated from the same similarity CBF computes | Latent-observable separation; different functional forms; multi-factor utility (Section 9) |
| Test-period interactions influencing training features | Every derived artefact rebuilt from training data only (Section 11) |
| Test hires present in the Knowledge Graph | Interaction edges carry a `split` property; evaluation queries filter to `split = 'train'`. The demo mode may use all edges |
| `INTERESTED_IN` exposing latent client preferences | Derived from observed training requests (e.g. at least two training requests in a category), never from generator variables |
| Skill relatedness shared between generator and Knowledge Graph | The generator uses co-occurrence over the full in-scope corpus. The Knowledge Graph's `RELATED_TO` edges use only the **background corpus**: eligible jobs not sampled as requests (about 35,900). The two are correlated but not identical |
| Profile fields encoding latent quality | Completed jobs and average ratings derived from training interactions only (Section 8) |
| Tuning on the test set | Weights tuned on validation; test computed once |
| Designer knowledge of the generator | Generator configuration frozen and tagged (`data-v1`) before recommender development |
| Mixture ground truth favouring the hybrid | **Generator sensitivity analysis:** re-run generation and evaluation under a *content-leaning* configuration (β_S 0.50, β_T 0.10, β_Q 0.15, β_B 0.15, β_C 0.10) and a *collaborative-leaning* configuration (β_S 0.15, β_T 0.45, β_Q 0.20, β_B 0.10, β_C 0.10). Conclusions are stated as robust only if they hold across configurations |
| Interpreting absolute numbers | Report **Random** and **Popularity** lower bounds and an **Oracle** upper bound (ranking by noise-free expected utility) |

## 15. Evaluation Methodology

**Task.** For each test request that has at least one relevant worker, every method ranks the **same eligible pool**: category match, joined before the request, and location-eligible. Candidate filtering is a platform search step applied identically to all methods, so the comparison isolates ranking quality. Requests with eligible pools smaller than 20 are excluded and counted.

**Relevance.** Hired = 2, shortlisted but not hired = 1, otherwise 0. Binary metrics treat grade ≥1 as relevant.

**Methods compared:**

- **Required by documentation [DOC]:** CBF, CF, Hybrid.
- **Ablation [IMPL]:** KG-only.
- **Interpretive bounds [IMPL]:** Random, Popularity (training hire counts), Oracle.

**Metrics:**

| Aspect | Metric |
|---|---|
| Accuracy **[DOC]** | Precision@K, Recall@K, F1@K for K ∈ {5, 10}; NDCG@10 as a secondary graded metric **[IMPL]** |
| Diversity **[DOC]** | Intra-list diversity (1 − mean pairwise Jaccard similarity of declared skills in the top 10); catalogue coverage (% of workers appearing in any top-10 list); exposure Gini coefficient |
| Cold-start **[DOC]** | Section 12 metrics |
| Explainability **[DOC]** | *Path coverage* (% of recommendations with at least one valid Knowledge Graph path of length ≤3 from request to worker); mean supporting paths and path length; *explanation fidelity* (% of explanations that cite the factor with the largest contribution to the hybrid score); a qualitative rubric applied to a random sample of 30 explanations |

**Statistical reporting.** Paired bootstrap 95% confidence intervals (1,000 resamples over test requests) for differences between the hybrid and each baseline.

**Interpretation constraints:**

- All results describe performance **within the constructed experimental environment**. They are not evidence of real-world performance.
- Offline evaluation cannot credit a method for recommending a strong worker who did not apply. This is a known limitation of observed-interaction ground truth.
- Explainability metrics are structural proxies. No user study is conducted, and the qualitative rubric has a single rater (the author).
- Client preferences are static in the simulation, so preference drift is not evaluated.

## 16. Reproducibility Considerations

- The raw CSV is never modified and is excluded from Git (size and licence). Kaggle provenance is recorded in the README.
- All steps are scripts, not notebook-only code. Notebooks are used only for exploration and figures.
- One configuration file holds all thresholds, scale values, generator weights, and the seed.
- `requirements.txt` pins library versions.
- Each pipeline run writes a manifest (row counts and SHA-256 hashes of outputs) and a `generation_report.json` with realised statistics: hire rate, rating distribution, hire concentration, the sparse threshold `N`, and per-factor utility variance shares.
- Automated validation checks run after generation. Every hire must have a matching shortlist entry; ratings exist only for hires; new entrants have zero training interactions; no worker exceeds capacity; no location-restricted request has an ineligible hire.
- The generator configuration is Git-tagged (`data-v1`) before recommender work begins.

---

## Appendix A: Synthetic Dataset Scale

| Entity | Target | Rationale |
|---|---|---|
| Service requests | **12,000** real jobs, sampled from 47,938 stratified by category | Each category receives about 23 or more requests. The test set is about 2,400 requests, enough for bootstrap intervals, while Neo4j-backed KG scoring over the test set stays fast. Using all 47,938 would require about 4× more clients and workers to keep realistic density |
| Clients | **2,000** | Mean ≈6 requests per client gives CF enough per-client history; the heavy tail still leaves sparse clients |
| Workers | **3,000** (300 new entrants) | Required by the 20-worker floor across 88 categories plus demand-proportional allocation; yields ≈3.4 hires per worker on average |
| Shortlist interactions | ≈60,000 | 5 per request |
| Hires | ≈10,200 | ≈85% of requests |
| Ratings | ≈9,200 | ≈90% of hires |
| Written reviews | ≈5,500 | ≈60% of ratings |

These values yield about 24,000 Neo4j nodes and roughly 200,000 relationships (about 60,000 of them shortlist edges, which could stay in PostgreSQL if loading becomes slow). That loads in minutes with batched `UNWIND` queries, and all PostgreSQL tables stay small enough to rebuild from scratch in seconds. Expected client–worker interaction density is about 1% (≈60,000 distinct pairs in a 2,000 × 3,000 matrix). Realised values are recorded in `generation_report.json`.

## Appendix B: Data Pipeline

```
Raw Upwork CSV (read-only)
  → HTML entity decoding
  → Footer-anchored metadata extraction (category, skills, location requirement)
  → Duplicate flagging (22)
  → Zero-skill flagging (515)
  → Category scope flagging (≥100 jobs → 88 categories)
  → Pay-type derivation and price-tier computation (stored values untouched)
  → Controlled vocabulary (≥20 in-scope jobs → 1,187 skills) + rare-skill audit
  → Eligible request flag (≥1 vocabulary skill → 47,938 jobs)
  = Clean job dataset (all rows retained, with flags)
  → Request sample (12,000) | background corpus (remaining eligible jobs)
  → Real-data statistics: category skill distributions, pay percentiles,
    category similarity, skill co-occurrence (full corpus for the generator;
    background corpus for KG RELATED_TO)
  → Synthetic workers (latent + observable attributes)
  → Synthetic clients + simulated timeline
  → Split boundaries + new-entrant cohort
  → Chronological interaction simulation (applicants → shortlist → hire)
  → Ratings and reviews
  → Validation checks + generation report + manifest
  = Unified experimental dataset
  → PostgreSQL export (application tables)
  → Neo4j export (nodes/edges, with split property)
```
