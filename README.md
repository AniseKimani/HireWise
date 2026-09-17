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

Day 2 - Reproducible Upwork data preparation pipeline. See
`docs/experimental_design.md` for the approved methodology,
`docs/data_quality_report.md` for pipeline results, and
`docs/data_dictionary.md` for output field definitions.

## Running the Day 2 pipeline

```bash
pip install -r requirements.txt
python scripts/run_data_pipeline.py   # requires data/raw/upwork-jobs.csv (git-ignored, not redistributed)
python scripts/verify_pipeline.py
python -m unittest discover -s tests
```