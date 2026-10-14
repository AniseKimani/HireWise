"""HireWise relational schema (Day 11), defined once with SQLAlchemy Core.

PostgreSQL is the application data store; Neo4j remains the knowledge
graph; the frozen recommender keeps reading its deterministic artifacts.
Only model-visible / application-facing fields are stored (see
docs/day11_postgresql_report.md for the column-by-column review). Never
stored: anything from data/processed/synthetic_private/, generator
scheduling fields (client primary_category, activity_quota,
activity_start_day, n_requests_assigned), the cold-start cohort flag,
interactions (shortlists/hires), and validation/test-period ratings.

Existing identifiers are the primary keys (`client_id`, `worker_id`,
request `link`, skill and category names) -- no surrogate IDs are
invented. Identifier columns use the "C" collation so ordering and
uniqueness are codepoint-based and independent of the server locale,
matching the ordering the Day 9/10 API already used.
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean, CheckConstraint, Column, Double, ForeignKey, Index, MetaData, PrimaryKeyConstraint, SmallInteger,
    Table, Text, func,
)

SCHEMA_VERSION = "1"
metadata = MetaData()


def _id(name: str, *args, **kwargs) -> Column:
    return Column(name, Text(collation="C"), *args, **kwargs)


def _fraction_check(table: str, column: str) -> CheckConstraint:
    return CheckConstraint(f"{column} IS NULL OR ({column} >= 0 AND {column} <= 1)", name=f"ck_{table}_{column}_range")


categories = Table(
    "categories", metadata,
    _id("name", primary_key=True),
)

skills = Table(
    "skills", metadata,
    _id("name", primary_key=True),
)

clients = Table(
    "clients", metadata,
    _id("client_id", primary_key=True),
    Column("country", Text, nullable=False),
    Column("preferred_pay_type", Text, nullable=False),
    Column("median_price_tier", Double),
    CheckConstraint("preferred_pay_type IN ('fixed', 'hourly', 'unspecified')", name="ck_clients_preferred_pay_type"),
    _fraction_check("clients", "median_price_tier"),
)

workers = Table(
    "workers", metadata,
    _id("worker_id", primary_key=True),
    _id("primary_category", ForeignKey("categories.name"), nullable=False),
    Column("experience_level", Text, nullable=False),
    Column("experience_years", Double, nullable=False),
    Column("price_tier", Double, nullable=False),
    Column("rate", Double, nullable=False),
    Column("country", Text, nullable=False),
    Column("join_day", Double, nullable=False),
    CheckConstraint("experience_level IN ('entry', 'intermediate', 'expert')", name="ck_workers_experience_level"),
    CheckConstraint("experience_years >= 0", name="ck_workers_experience_years"),
    CheckConstraint("rate > 0", name="ck_workers_rate"),
    _fraction_check("workers", "price_tier"),
)

worker_secondary_categories = Table(
    "worker_secondary_categories", metadata,
    _id("worker_id", ForeignKey("workers.worker_id"), nullable=False),
    _id("category", ForeignKey("categories.name"), nullable=False),
    PrimaryKeyConstraint("worker_id", "category", name="pk_worker_secondary_categories"),
)

worker_skills = Table(
    "worker_skills", metadata,
    _id("worker_id", ForeignKey("workers.worker_id"), nullable=False),
    _id("skill", ForeignKey("skills.name"), nullable=False),
    PrimaryKeyConstraint("worker_id", "skill", name="pk_worker_skills"),
)

service_requests = Table(
    "service_requests", metadata,
    _id("link", primary_key=True),
    _id("client_id", ForeignKey("clients.client_id"), nullable=False),
    Column("title", Text, nullable=False),
    # Lower-cased title computed by the loader (Python str.lower) so title
    # search is identical to the Day 10 semantics regardless of server locale.
    Column("title_search", Text, nullable=False),
    _id("category", ForeignKey("categories.name"), nullable=False),
    Column("pay_type", Text, nullable=False),
    Column("price_tier", Double),
    # Original posted pay values, unchanged (docs/experimental_design.md 3.5).
    Column("budget", Double),
    Column("hourly_low", Double),
    Column("hourly_high", Double),
    Column("location_requirement", Text),
    Column("simulated_day", Double, nullable=False),
    # Experiment period. Kept (approved Day 11 decision 2) because only
    # validation/test requests are recommendable and the existing API
    # already exposes it; never used to expose outcomes.
    Column("split", Text, nullable=False),
    CheckConstraint("pay_type IN ('fixed', 'hourly', 'unspecified')", name="ck_service_requests_pay_type"),
    CheckConstraint("split IN ('train', 'validation', 'test')", name="ck_service_requests_split"),
    CheckConstraint("budget IS NULL OR budget > 0", name="ck_service_requests_budget"),
    _fraction_check("service_requests", "price_tier"),
)

request_skills = Table(
    "request_skills", metadata,
    _id("link", ForeignKey("service_requests.link"), nullable=False),
    _id("skill", ForeignKey("skills.name"), nullable=False),
    PrimaryKeyConstraint("link", "skill", name="pk_request_skills"),
)

ratings = Table(
    "ratings", metadata,
    # Training-period ratings only (approved Day 11 decision 1).
    _id("link", ForeignKey("service_requests.link"), nullable=False),
    _id("worker_id", ForeignKey("workers.worker_id"), nullable=False),
    Column("rating", SmallInteger, nullable=False),
    Column("has_review", Boolean, nullable=False),
    PrimaryKeyConstraint("link", "worker_id", name="pk_ratings"),
    CheckConstraint("rating BETWEEN 1 AND 5", name="ck_ratings_rating"),
)

load_metadata = Table(
    "load_metadata", metadata,
    _id("key", primary_key=True),
    Column("value", Text, nullable=False),
)

# Indexes, each serving a query in src/db/repositories.py.
Index("ix_service_requests_split_day_link", service_requests.c.split, service_requests.c.simulated_day,
      service_requests.c.link)                                   # request listing (filter + fixed order)
Index("ix_service_requests_category", service_requests.c.category)  # listing category filter
Index("ix_service_requests_client_id", service_requests.c.client_id)  # FK lookups by client
Index("ix_request_skills_skill_lower", func.lower(request_skills.c.skill))  # listing skill filter (case-insensitive)
Index("ix_worker_skills_skill", worker_skills.c.skill)          # FK / skill lookups
Index("ix_ratings_worker_id", ratings.c.worker_id)              # per-worker review aggregates

# Load order respects foreign keys.
TABLE_LOAD_ORDER = (
    "categories", "skills", "clients", "workers", "worker_secondary_categories", "worker_skills",
    "service_requests", "request_skills", "ratings",
)
DATA_TABLES = {name: metadata.tables[name] for name in TABLE_LOAD_ORDER}
