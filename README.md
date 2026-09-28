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

Day 6 - Knowledge graph construction and Neo4j integration, on top of
the Day 4 baselines/CBF and Day 5 Collaborative Filtering recommenders.
See `docs/experimental_design.md` for the approved methodology, and
`docs/day4_recommender_report.md`, `docs/day5_cf_report.md`,
`docs/day6_knowledge_graph_report.md` for each day's results. The KG
recommender and Hybrid have not been implemented yet.

## Running the pipeline

```bash
pip install -r requirements.txt
python scripts/run_data_pipeline.py       # Day 2: requires data/raw/upwork-jobs.csv (git-ignored, not redistributed)
python scripts/verify_pipeline.py
python scripts/generate_synthetic_data.py # Day 3: requires Day 2's outputs in data/processed/
python scripts/verify_synthetic_data.py
python scripts/run_day4_evaluation.py     # Day 4: Random/Popularity/CBF baselines
python scripts/verify_day4.py
python scripts/run_day5_evaluation.py     # Day 5: Collaborative Filtering
python scripts/verify_day5.py
python scripts/prepare_kg_data.py         # Day 6: knowledge graph (no Neo4j needed)
python scripts/load_neo4j.py              # Day 6: requires a local .env (see .env.example)
python scripts/verify_neo4j.py
python scripts/verify_day6.py
python -m unittest discover -s tests
```