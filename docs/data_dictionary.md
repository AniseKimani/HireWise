# Data Dictionary: Day 2 Prepared Dataset

Describes every field produced by `scripts/run_data_pipeline.py`. All
outputs live under `data/processed/` (git-ignored; regenerate with
`python scripts/run_data_pipeline.py`). Full numeric results are in
`docs/data_quality_report.md`.

## 1. Original raw source fields

These come directly from `data/raw/upwork-jobs.csv` and are **never
modified** by the pipeline. They appear unchanged in `prepared_jobs.csv`.

| Field | Type | Missingness | Used in modelling |
|---|---|---|---|
| `title` | string | 0% | Display only |
| `link` | string | 0% | Yes — unique job identifier |
| `description` | string | 0% | Not carried into processed outputs (see Section 5) |
| `published_date` | datetime (UTC) | 0% | Provenance only; not used for ordering (Section 2, Limitation 3 in `docs/experimental_design.md`) |
| `is_hourly` | boolean | 15.5% | Source for derived `pay_type` |
| `hourly_low`, `hourly_high` | float | 56.7%, 58.3% | Source for derived `pay_amount` |
| `budget` | float | 58.8% | Source for derived `pay_amount` |
| `country` | string | 2.2% | Source for derived `country_clean` |

## 2. Parsed fields (`src/data/parse.py`)

Derived from the footer at the end of `description`, anchored on the
**last** `Posted On:` occurrence (see Section 4 of
`docs/experimental_design.md` and the parser tests in
`tests/test_parse.py`). None of these mutate a raw column.

| Field | Type | Meaning | Raw/derived | Used in modelling |
|---|---|---|---|---|
| `footer_found` | bool | Whether a `Posted On:` footer was located | Derived | Diagnostic only |
| `category` | string or null | Platform-assigned category from the footer | Derived | Yes, via scope/eligibility flags |
| `skills_raw` | list[string] | All skills parsed from the first `Skills:` block, deduplicated within the job, order preserved | Derived | Retained for audit; not fed to models directly |
| `n_skills_raw` | int | `len(skills_raw)` | Derived | Used for `has_no_skills` |
| `location_requirement` | string or null | Free-text eligibility restriction (e.g. "Only freelancers located in ... may apply") | Derived | Yes — location eligibility filter |
| `footer_country` | string or null | Country as stated in the footer (validation only; the stored `country` column remains authoritative — Section 4) | Derived | Validation only |

## 3. Derived cleaning/eligibility fields (`src/data/clean.py`)

| Field | Type | Meaning | Used in modelling |
|---|---|---|---|
| `country_clean` | string or null | HTML-entity-decoded `country`; raw `country` is preserved unchanged | Yes |
| `pay_type` | `"hourly"` / `"fixed"` / `"unspecified"` | Derived from `is_hourly` | Yes |
| `pay_amount` | float or null | Hourly midpoint (or the single bound if only one exists), fixed budget, or null if unspecified | Yes — input to `price_tier` |
| `price_tier` | float in [0, 1] or null | Percentile rank of `pay_amount` within (`category`, `pay_type`), computed on the in-scope reference population (`config.price_tier_reference`); null for rows outside that population or with no `pay_amount` | Yes — the modelling pay feature; raw `budget`/`hourly_low`/`hourly_high` are never overwritten (Section 3.5) |
| `is_duplicate` | bool | True for every row in a title+description duplicate group except the one kept (most complete across pay/country columns, ties broken by smallest `link`) | Exclusion flag |
| `has_no_skills` | bool | True when `n_skills_raw == 0` | Exclusion flag |
| `is_in_scope_category` | bool | True when `category` has ≥ `config.min_category_jobs` jobs among non-duplicate, has-skills rows | Exclusion flag |
| `skills_vocab` | list[string] | Intersection of `skills_raw` with the controlled vocabulary | Yes — the modelling skill feature |
| `has_vocab_skill` | bool | True when `skills_vocab` is non-empty | Exclusion flag |
| `exclusion_reason` | string or null | First applicable reason in precedence order: `duplicate` → `no_skills` → `out_of_scope_category` → `no_vocab_skill`; null if none apply | Audit |
| `exclude_from_model` | bool | `exclusion_reason is not None` | Derived from the above |
| `eligible_for_model` | bool | `not exclude_from_model` | Defines the eligible modelling population (47,938 jobs) |
| `is_sampled_request` | bool | True for the 12,000 jobs drawn as modelling requests (`src/data/sample.py`), stratified by category, seed 42 | Defines the request sample vs. background corpus |

## 4. Controlled vocabulary and statistics outputs

| File | Columns | Corpus used |
|---|---|---|
| `skills_vocab.csv` | `skill`, `frequency`, `in_vocabulary=True` | In-scope modelling population (48,499 jobs) |
| `rare_skills.csv` | `skill`, `frequency`, `in_vocabulary=False` | Same, skills below the frequency threshold |
| `skill_frequencies.csv` | `skill`, `frequency` | Eligible modelling population (47,938 jobs), `skills_vocab` only |
| `category_stats.csv` | `category`, `n_jobs`, `mean_skills_vocab`, `median_price_tier`, `pct_hourly`, `pct_fixed` | Eligible modelling population |
| `category_similarity.csv` | `category`, `similar_category`, `rank`, `cosine_similarity` | Eligible modelling population, top-5 per category |
| `skill_cooccurrence_full_corpus.csv` | `skill_a`, `skill_b`, `support`, `npmi` | Eligible modelling population (47,938 jobs) — for Day 3's generator |
| `skill_cooccurrence_background_corpus.csv` | same | Background corpus only (35,938 jobs, excludes the 12,000 sampled requests) — for Day 3's Knowledge Graph `RELATED_TO` edges, kept independent of the generator's corpus (leakage prevention, Section 14) |
| `pay_percentiles.json` | per `pay_type`: stored quantiles/mean, winsorised quantiles/mean (analysis only), winsorisation bounds | Eligible modelling population |
| `location_availability.json` | country coverage, location-requirement coverage | Eligible modelling population |
| `summary_stats.json` | top-level pipeline counts (see `docs/data_quality_report.md`) | — |
| `manifest.json` | relative path, SHA-256, byte size, row count per output file | — |

## 5. Files intentionally excluded from processed outputs

`description` (full free text) is not carried into any processed CSV.
Every row is uniquely identified by `link`, so the original text can
always be recovered by joining back to `data/raw/upwork-jobs.csv` (which
is git-ignored and never redistributed). This keeps `data/processed/`
outputs an order of magnitude smaller without losing any information.
