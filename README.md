# HireWise

## Knowledge Graph-Enhanced Hybrid Recommender System

A web-based digital worker-service platform using a hybrid recommendation system combining:

- Content-Based Filtering
- Collaborative Filtering
- Knowledge Graph reasoning

## Technology Stack

- Python
- Flask
- PostgreSQL
- Neo4j
- HTML/CSS/JavaScript
- Git/GitHub

## Dataset

Upwork Job Postings Dataset 2024.

The dataset provides real-world job/service demand information. Synthetic worker, client, and interaction data will be generated to complement the dataset.

## Project Status

Day 3 - Synthetic worker/client/interaction generation on top of the Day 2
real-data foundation. See `docs/experimental_design.md` for the approved
methodology, `docs/data_quality_report.md` / `docs/synthetic_data_report.md`
for pipeline results, and `docs/data_dictionary.md` for output field
definitions. No recommenders have been implemented yet.

## Running the pipeline

```bash
pip install -r requirements.txt
python scripts/run_data_pipeline.py       # Day 2: requires data/raw/upwork-jobs.csv (git-ignored, not redistributed)
python scripts/verify_pipeline.py
python scripts/generate_synthetic_data.py # Day 3: requires Day 2's outputs in data/processed/
python scripts/verify_synthetic_data.py
python -m unittest discover -s tests
```