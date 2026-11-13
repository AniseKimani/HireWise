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

Schema version 2 (Day 12) ADDS runtime marketplace tables -- users,
runtime worker profiles, client-created requests, bookings and
application reviews -- without altering any imported table. Imported
experimental rows and runtime transactions (registrations, new requests,
bookings, reviews) never share a table. Runtime tables use surrogate
identity keys because they describe new entities; they reference
imported entities only through existing identifiers (client_id,
worker_id, category and skill names).

Schema version 3 (pre-deployment Milestone 1) adds account security:
`users.auth_version` (incremented by a password reset, so older sessions
stop being accepted), `user_email_verification` (one row per user; a NULL
verified_at means unverified) and `auth_tokens` (SHA-256 hashes of
single-use, purpose-bound, expiring email-verification / password-reset
tokens -- raw tokens are never stored). An existing version 2 database is
upgraded by scripts/migrate_schema_v3.py (src/db/migrations.py), which also
marks every existing account as verified; `metadata.create_all` alone
cannot add the new users column.

Schema version 4 (pre-deployment Milestone 2) adds `user_identities`: an
external sign-in identity (currently only Google) linked to a user. The
identity is keyed by the provider's stable subject identifier (`sub`),
never by email; one identity belongs to one user, and a user has at most
one identity per provider. Accounts created through Google carry an
unusable random password hash until the owner sets a password through the
password-reset flow. scripts/migrate_schema_v4.py upgrades a version 2 or
3 database step by step (src/db/migrations.py).

Schema version 5 (pre-deployment Milestone 3) adds `notifications`: in-app
messages about booking and review events, one row per recipient account,
written in the same transaction as the booking or review change. Each row
has exactly one recipient (users.user_id); the message is a short plain
sentence (at most 500 characters, enforced by the database). Nothing about
notifications ever reaches the frozen recommender. scripts/migrate_schema_v5.py
upgrades a version 2, 3 or 4 database step by step.
"""
from __future__ import annotations

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Column, DateTime, Double, ForeignKey, ForeignKeyConstraint, Identity, Index,
    Integer, MetaData, PrimaryKeyConstraint, SmallInteger, Table, Text, UniqueConstraint, func, text,
)

SCHEMA_VERSION = "5"
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

# ---------------------------------------------------------------------------
# Runtime marketplace tables (schema version 2, Day 12)
# ---------------------------------------------------------------------------
BOOKING_STATUSES = ("pending", "accepted", "declined", "cancelled", "completed")
ACTIVE_BOOKING_STATUSES = ("pending", "accepted", "completed")


def _timestamp(name: str, **kwargs) -> Column:
    return Column(name, DateTime(timezone=True), **kwargs)


users = Table(
    "users", metadata,
    Column("user_id", BigInteger, Identity(), primary_key=True),
    Column("email", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("role", Text, nullable=False),
    Column("display_name", Text, nullable=False),
    Column("country", Text),
    # Optional links to IMPORTED experimental profiles -- demo accounts only,
    # created by scripts/create_demo_account.py; never set by public registration.
    _id("client_id", ForeignKey("clients.client_id"), unique=True),
    _id("worker_id", ForeignKey("workers.worker_id"), unique=True),
    _timestamp("created_at", nullable=False, server_default=func.now()),
    # Schema v3: sessions record the version they were created with; a password reset increments it.
    Column("auth_version", Integer, nullable=False, server_default=text("0")),
    CheckConstraint("role IN ('client', 'worker')", name="ck_users_role"),
    CheckConstraint("email = lower(email) AND position('@' in email) > 1", name="ck_users_email"),
    CheckConstraint("role = 'client' OR client_id IS NULL", name="ck_users_client_link_role"),
    CheckConstraint("role = 'worker' OR worker_id IS NULL", name="ck_users_worker_link_role"),
)

runtime_worker_profiles = Table(
    "runtime_worker_profiles", metadata,
    # Profiles of NEWLY registered workers (no experimental history); not part
    # of the frozen recommender's catalogue.
    Column("user_id", BigInteger, ForeignKey("users.user_id"), primary_key=True),
    _id("primary_category", ForeignKey("categories.name"), nullable=False),
    Column("experience_level", Text, nullable=False),
    Column("experience_years", Double, nullable=False),
    Column("rate", Double, nullable=False),
    Column("country", Text, nullable=False),
    CheckConstraint("experience_level IN ('entry', 'intermediate', 'expert')", name="ck_runtime_worker_experience_level"),
    CheckConstraint("experience_years >= 0 AND experience_years <= 60", name="ck_runtime_worker_experience_years"),
    CheckConstraint("rate > 0", name="ck_runtime_worker_rate"),
)

runtime_worker_skills = Table(
    "runtime_worker_skills", metadata,
    Column("user_id", BigInteger, ForeignKey("runtime_worker_profiles.user_id"), nullable=False),
    _id("skill", ForeignKey("skills.name"), nullable=False),
    PrimaryKeyConstraint("user_id", "skill", name="pk_runtime_worker_skills"),
)

runtime_requests = Table(
    "runtime_requests", metadata,
    Column("request_id", BigInteger, Identity(), primary_key=True),
    Column("client_user_id", BigInteger, ForeignKey("users.user_id"), nullable=False),
    Column("title", Text, nullable=False),
    _id("category", ForeignKey("categories.name"), nullable=False),
    Column("pay_type", Text, nullable=False),
    Column("budget", Double),
    Column("hourly_low", Double),
    Column("hourly_high", Double),
    # Day 2 definition, computed at creation against the frozen reference population.
    Column("price_tier", Double),
    Column("location_requirement", Text),
    _timestamp("created_at", nullable=False, server_default=func.now()),
    UniqueConstraint("request_id", "client_user_id", name="uq_runtime_requests_owner"),
    CheckConstraint("pay_type IN ('fixed', 'hourly', 'unspecified')", name="ck_runtime_requests_pay_type"),
    CheckConstraint("pay_type = 'fixed' OR budget IS NULL", name="ck_runtime_requests_budget_fixed_only"),
    CheckConstraint("pay_type = 'hourly' OR (hourly_low IS NULL AND hourly_high IS NULL)", name="ck_runtime_requests_hourly_only"),
    CheckConstraint("hourly_low IS NULL OR hourly_high IS NULL OR hourly_low <= hourly_high", name="ck_runtime_requests_hourly_order"),
    CheckConstraint("(budget IS NULL OR budget > 0) AND (hourly_low IS NULL OR hourly_low > 0) "
                    "AND (hourly_high IS NULL OR hourly_high > 0)", name="ck_runtime_requests_positive_pay"),
    CheckConstraint("char_length(title) BETWEEN 3 AND 200", name="ck_runtime_requests_title_length"),
    _fraction_check("runtime_requests", "price_tier"),
)

runtime_request_skills = Table(
    "runtime_request_skills", metadata,
    Column("request_id", BigInteger, ForeignKey("runtime_requests.request_id"), nullable=False),
    _id("skill", ForeignKey("skills.name"), nullable=False),
    PrimaryKeyConstraint("request_id", "skill", name="pk_runtime_request_skills"),
)

bookings = Table(
    "bookings", metadata,
    Column("booking_id", BigInteger, Identity(), primary_key=True),
    Column("request_id", BigInteger, nullable=False),
    Column("client_user_id", BigInteger, nullable=False),
    # Bookable workers are the frozen recommender's (imported) catalogue.
    _id("worker_id", ForeignKey("workers.worker_id"), nullable=False),
    Column("status", Text, nullable=False),
    _timestamp("created_at", nullable=False, server_default=func.now()),
    _timestamp("updated_at", nullable=False, server_default=func.now()),
    _timestamp("completed_at"),
    # The booking's client must be the request's owner -- enforced by the database.
    ForeignKeyConstraint(["request_id", "client_user_id"],
                         ["runtime_requests.request_id", "runtime_requests.client_user_id"],
                         name="fk_bookings_request_owner"),
    CheckConstraint("status IN ('pending', 'accepted', 'declined', 'cancelled', 'completed')", name="ck_bookings_status"),
    CheckConstraint("(status = 'completed') = (completed_at IS NOT NULL)", name="ck_bookings_completed_at"),
)

app_reviews = Table(
    "app_reviews", metadata,
    # NEW application reviews (optional text), kept apart from the imported
    # experimental `ratings` table, which has no text.
    Column("booking_id", BigInteger, ForeignKey("bookings.booking_id"), primary_key=True),
    Column("rating", SmallInteger, nullable=False),
    Column("review_text", Text),
    _timestamp("created_at", nullable=False, server_default=func.now()),
    CheckConstraint("rating BETWEEN 1 AND 5", name="ck_app_reviews_rating"),
    CheckConstraint("review_text IS NULL OR char_length(review_text) <= 2000", name="ck_app_reviews_text_length"),
)

# At most one live booking per request (pending, accepted or completed).
Index("uq_bookings_one_active_per_request", bookings.c.request_id, unique=True,
      postgresql_where=bookings.c.status.in_(ACTIVE_BOOKING_STATUSES))
Index("ix_bookings_worker_id", bookings.c.worker_id)                            # worker dashboard
Index("ix_bookings_client_user_id", bookings.c.client_user_id)                  # client dashboard
Index("ix_runtime_requests_client_user_id", runtime_requests.c.client_user_id)  # "my requests"

# ---------------------------------------------------------------------------
# Account security tables (schema version 3)
# ---------------------------------------------------------------------------
TOKEN_PURPOSES = ("verify_email", "reset_password")

user_email_verification = Table(
    "user_email_verification", metadata,
    # One row per user, created together with the user. NULL verified_at = not yet verified.
    Column("user_id", BigInteger, ForeignKey("users.user_id"), primary_key=True),
    _timestamp("verified_at"),
)

auth_tokens = Table(
    "auth_tokens", metadata,
    Column("token_id", BigInteger, Identity(), primary_key=True),
    Column("user_id", BigInteger, ForeignKey("users.user_id"), nullable=False),
    Column("purpose", Text, nullable=False),
    # Lower-case hex SHA-256 of the raw token; the raw token itself is never stored.
    Column("token_hash", Text, nullable=False, unique=True),
    _timestamp("created_at", nullable=False, server_default=func.now()),
    _timestamp("expires_at", nullable=False),
    # Set when the token is consumed, or when a newer token of the same purpose replaces it.
    _timestamp("used_at"),
    CheckConstraint("purpose IN ('verify_email', 'reset_password')", name="ck_auth_tokens_purpose"),
    CheckConstraint("token_hash ~ '^[0-9a-f]{64}$'", name="ck_auth_tokens_token_hash"),
    CheckConstraint("expires_at > created_at", name="ck_auth_tokens_expiry"),
)
Index("ix_auth_tokens_user_purpose", auth_tokens.c.user_id, auth_tokens.c.purpose)  # invalidate a user's older tokens
Index("ix_auth_tokens_expires_at", auth_tokens.c.expires_at)                         # expiry housekeeping

ACCOUNT_SECURITY_TABLES = ("user_email_verification", "auth_tokens")

# ---------------------------------------------------------------------------
# External sign-in identities (schema version 4)
# ---------------------------------------------------------------------------
IDENTITY_PROVIDERS = ("google",)

user_identities = Table(
    "user_identities", metadata,
    Column("identity_id", BigInteger, Identity(), primary_key=True),
    Column("user_id", BigInteger, ForeignKey("users.user_id"), nullable=False),
    Column("provider", Text, nullable=False),
    # The provider's stable account identifier (OpenID Connect `sub`) -- the identity key; never the email.
    Column("subject", Text, nullable=False),
    # The provider-verified address when the identity was linked (informational; never used to match).
    Column("email_at_link", Text, nullable=False),
    _timestamp("created_at", nullable=False, server_default=func.now()),
    _timestamp("last_login_at"),
    CheckConstraint("provider IN ('google')", name="ck_user_identities_provider"),
    CheckConstraint("char_length(subject) BETWEEN 1 AND 255", name="ck_user_identities_subject"),
    CheckConstraint("email_at_link = lower(email_at_link)", name="ck_user_identities_email"),
    UniqueConstraint("provider", "subject", name="uq_user_identities_provider_subject"),
    UniqueConstraint("user_id", "provider", name="uq_user_identities_user_provider"),
)

IDENTITY_TABLES = ("user_identities",)

# ---------------------------------------------------------------------------
# In-app notifications (schema version 5)
# ---------------------------------------------------------------------------
NOTIFICATION_KINDS = ("booking_requested", "booking_accepted", "booking_declined", "booking_cancelled",
                      "booking_completed", "review_received")
NOTIFICATION_MESSAGE_MAX_LENGTH = 500

notifications = Table(
    "notifications", metadata,
    Column("notification_id", BigInteger, Identity(), primary_key=True),
    Column("user_id", BigInteger, ForeignKey("users.user_id"), nullable=False),      # the one recipient
    Column("booking_id", BigInteger, ForeignKey("bookings.booking_id"), nullable=True),
    Column("kind", Text, nullable=False),
    Column("message", Text, nullable=False),
    _timestamp("created_at", nullable=False, server_default=func.now()),
    _timestamp("read_at"),
    CheckConstraint("kind IN ('booking_requested', 'booking_accepted', 'booking_declined', 'booking_cancelled', "
                    "'booking_completed', 'review_received')", name="ck_notifications_kind"),
    CheckConstraint(f"char_length(message) BETWEEN 1 AND {NOTIFICATION_MESSAGE_MAX_LENGTH}",
                    name="ck_notifications_message_length"),
    CheckConstraint("read_at IS NULL OR read_at >= created_at", name="ck_notifications_read_at"),
    # Each event happens at most once per booking, so one row per (booking, event, recipient).
    UniqueConstraint("booking_id", "kind", "user_id", name="uq_notifications_booking_kind_user"),
)
Index("ix_notifications_user_read", notifications.c.user_id, notifications.c.read_at)              # unread count
Index("ix_notifications_user_created", notifications.c.user_id, notifications.c.created_at.desc())  # newest first

NOTIFICATION_TABLES = ("notifications",)

# Dependent tables come after their parents, so reversed(RUNTIME_TABLES) clears children first.
RUNTIME_TABLES = ("users", "user_email_verification", "auth_tokens", "user_identities", "runtime_worker_profiles",
                  "runtime_worker_skills", "runtime_requests", "runtime_request_skills", "bookings", "app_reviews",
                  "notifications")

# Load order respects foreign keys.
TABLE_LOAD_ORDER = (
    "categories", "skills", "clients", "workers", "worker_secondary_categories", "worker_skills",
    "service_requests", "request_skills", "ratings",
)
DATA_TABLES = {name: metadata.tables[name] for name in TABLE_LOAD_ORDER}
