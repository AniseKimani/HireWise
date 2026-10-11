# Day 7 Explanation Rubric (Human Evaluation Instrument)

This is the qualitative rubric required by `docs/experimental_design.md`
Section 15 ("a qualitative rubric applied to a random sample of 30
explanations"). It is a **human evaluation instrument**. The author is the
single rater (Section 15, interpretation constraints). No explanation is
pre-filled or rated automatically.

## Materials

| File | Content |
|---|---|
| `docs/day7_explanation_rubric_sheet.csv` | **Rating sheet.** 30 rows with blank rating columns |
| `results/day7/explanation_samples.json` | Full structured explanations for the same 30 samples: node sequences, edge properties, per-path contributions |

The sample is the frozen, seeded (seed 42) set of 30 test-split top-10 KG
explanations produced by `scripts/run_day7_evaluation.py`.

The rating sheet is derived row for row from `explanation_samples.json`,
with the same `sample_id` order as `results/day7/explanation_rubric_template.csv`.
Each row shows:
- `explanation`: the summary text a user would see;
- `graph_evidence`: the displayed paths, each tagged with its path type, plus the reputation evidence.

`graph_evidence` is there so traceability can be judged. A rater who wants
the exact nodes, edge properties or contributions behind a row should look
it up in the JSON.

`results/day7/explanation_rubric_template.csv` is a hash-verified pipeline
output and is left unchanged. Its placeholder columns
(`understandable_1to5`, `grounded_1to5`, `useful_1to5`) are **superseded**
by the four criteria below. Ratings go only in the docs rating sheet.

## Criteria

Rate each explanation on each criterion separately.

| # | Criterion | Sheet column | Question |
|---|---|---|---|
| 1 | **Relevance** | `relevance_1to5` | Does the explanation give a meaningful reason why the recommended worker fits the request? |
| 2 | **Clarity** | `clarity_1to5` | Is the explanation easy to understand without knowing about Neo4j, Cypher, graph paths, or how recommender systems work? |
| 3 | **Traceability** | `traceability_1to5` | Can the explanation be traced back to actual graph evidence or paths the recommender used? |
| 4 | **Usefulness** | `usefulness_1to5` | Would the explanation help a user understand or trust why this worker was recommended? |

## Rating scale

The same scale applies to all four criteria:

| Score | Meaning |
|---|---|
| 1 | Very poor |
| 2 | Poor |
| 3 | Acceptable |
| 4 | Good |
| 5 | Excellent |

Only whole numbers 1–5 are allowed. Use `rater_notes` for anything a score
doesn't capture.

## Reporting plan

Once all 30 explanations have been rated, the evaluation will report:

- the number of explanations evaluated (**n = 30**);
- the mean score for each criterion (Relevance, Clarity, Traceability, Usefulness);
- the overall mean across the four criteria;
- where useful, the standard deviation per criterion and/or the distribution of scores (count of 1s through 5s per criterion).

**No results are reported yet.** The human ratings have not been done.
Because there is one rater, these scores describe how the author judged
the explanations. They are not a user study, and they will be reported
under the same limitation stated in `docs/experimental_design.md`
Section 15.
