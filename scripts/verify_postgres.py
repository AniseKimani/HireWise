"""Independent verification of the HireWise PostgreSQL database (Day 11).

Does NOT reuse src/db/loader.py's transformation: expected counts and
values are re-derived here from the source CSVs with separate pandas
code, and integrity is checked with direct SQL against the live tables.

Checks: connectivity; exact table set; row counts; no orphan foreign-key
rows; no duplicate keys; no private/experimental-outcome columns; ratings
only from training-period requests; deterministic sample rows equal to
the source CSVs; recorded source-file hashes equal the current files.

Usage:
    python scripts/verify_postgres.py

Read-only. Exits non-zero if any check fails.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from sqlalchemy import text

from src.data.io import sha256_of
from src.db.database import make_engine

FAILURES: list[str] = []
P = REPO_ROOT / "data" / "processed"
S = P / "synthetic"
EXPECTED_TABLES = {"categories", "skills", "clients", "workers", "worker_secondary_categories", "worker_skills",
                   "service_requests", "request_skills", "ratings", "load_metadata"}
FORBIDDEN_COLUMN_MARKERS = ("quality", "true_skill", "proficiency", "taste", "utility", "noise", "oracle", "cold_start",
                            "activity_quota", "activity_start", "n_requests_assigned", "relevance", "event", "shortlist", "hired")
# (child table, child column, parent table, parent column)
FOREIGN_KEYS = [
    ("workers", "primary_category", "categories", "name"),
    ("worker_secondary_categories", "worker_id", "workers", "worker_id"),
    ("worker_secondary_categories", "category", "categories", "name"),
    ("worker_skills", "worker_id", "workers", "worker_id"),
    ("worker_skills", "skill", "skills", "name"),
    ("service_requests", "client_id", "clients", "client_id"),
    ("service_requests", "category", "categories", "name"),
    ("request_skills", "link", "service_requests", "link"),
    ("request_skills", "skill", "skills", "name"),
    ("ratings", "link", "service_requests", "link"),
    ("ratings", "worker_id", "workers", "worker_id"),
]


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def same(a, b) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    return a == b


def main() -> int:
    try:
        engine = make_engine()
        conn = engine.connect()
    except Exception as exc:
        check("PostgreSQL reachable", False, type(exc).__name__)
        return 1
    with conn:
        check("PostgreSQL reachable", conn.execute(text("SELECT 1")).scalar() == 1)
        tables = {r[0] for r in conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))}
        check("exactly the expected HireWise tables exist", tables == EXPECTED_TABLES, f"{sorted(tables ^ EXPECTED_TABLES)}")

        # ---- independent expectations from the source CSVs ----------------------
        clients = pd.read_csv(S / "clients.csv")
        workers = pd.read_csv(S / "workers.csv")
        requests = pd.read_csv(S / "requests.csv")
        ratings = pd.read_csv(S / "ratings.csv")
        sampled = pd.read_csv(P / "sampled_requests.csv")
        vocab = pd.read_csv(P / "skills_vocab.csv")
        cats = pd.read_csv(P / "category_stats.csv")
        train_links = set(requests.loc[requests["split"] == "train", "link"])
        req_content = requests.merge(sampled, on="link")
        expected = {
            "categories": cats["category"].nunique(),
            "skills": vocab["skill"].nunique(),
            "clients": len(clients),
            "workers": len(workers),
            "worker_secondary_categories": int(workers["secondary_categories"].dropna().str.split("|").map(lambda x: len(set(x))).sum()),
            "worker_skills": int(workers["declared_skills"].str.split("|").map(lambda x: len(set(x))).sum()),
            "service_requests": len(requests),
            "request_skills": int(req_content["skills_vocab"].str.split("|").map(lambda x: len(set(x))).sum()),
            "ratings": int(ratings["link"].isin(train_links).sum()),
        }
        counts = {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in expected}  # fixed table names
        for t in expected:
            check(f"{t}: {counts[t]:,} rows (expected {expected[t]:,})", counts[t] == expected[t])

        # ---- integrity ---------------------------------------------------------
        for child, col, parent, pcol in FOREIGN_KEYS:
            orphans = conn.execute(text(
                f"SELECT count(*) FROM {child} c LEFT JOIN {parent} p ON c.{col} = p.{pcol} WHERE p.{pcol} IS NULL")).scalar()
            check(f"no orphan {child}.{col} -> {parent}.{pcol}", orphans == 0, f"{orphans} orphans")
        fk_count = conn.execute(text(
            "SELECT count(*) FROM information_schema.table_constraints WHERE table_schema = 'public' AND constraint_type = 'FOREIGN KEY'")).scalar()
        check(f"{len(FOREIGN_KEYS)} foreign-key constraints declared", fk_count == len(FOREIGN_KEYS), f"found {fk_count}")
        pk_tables = {r[0] for r in conn.execute(text(
            "SELECT table_name FROM information_schema.table_constraints WHERE table_schema = 'public' AND constraint_type = 'PRIMARY KEY'"))}
        check("every table has a primary key", pk_tables == EXPECTED_TABLES)
        columns = [(t, c) for t, c in conn.execute(text(
            "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public'"))]
        leaked = [f"{t}.{c}" for t, c in columns if any(m in c for m in FORBIDDEN_COLUMN_MARKERS)]
        check("no generator-private or outcome columns", not leaked, f"{leaked}")
        non_train = conn.execute(text(
            "SELECT count(*) FROM ratings r JOIN service_requests s ON s.link = r.link WHERE s.split <> 'train'")).scalar()
        check("ratings only from training-period requests", non_train == 0, f"{non_train} non-training ratings")
        bad_ratings = conn.execute(text("SELECT count(*) FROM ratings WHERE rating NOT BETWEEN 1 AND 5")).scalar()
        check("all ratings within 1-5", bad_ratings == 0)

        # ---- deterministic sample rows == source CSVs ------------------------------
        w_sample = workers.sort_values("worker_id").iloc[::300]
        rows = {r.worker_id: r for r in conn.execute(text(
            "SELECT worker_id, primary_category, experience_level, experience_years, price_tier, rate, country, join_day "
            "FROM workers WHERE worker_id = ANY(:ids)"), {"ids": w_sample["worker_id"].tolist()})}
        ok = all(all(same(getattr(rows[w.worker_id], c), getattr(w, c)) for c in
                     ("primary_category", "experience_level", "experience_years", "price_tier", "rate", "country", "join_day"))
                 for w in w_sample.itertuples())
        skills_ok = all(sorted(r[0] for r in conn.execute(text("SELECT skill FROM worker_skills WHERE worker_id = :w"), {"w": w.worker_id}))
                        == sorted(set(w.declared_skills.split("|"))) for w in w_sample.itertuples())
        check(f"{len(w_sample)} sample workers equal workers.csv (fields and skills)", ok and skills_ok)
        r_sample = req_content.sort_values("link").iloc[::1200]
        rows = {r.link: r for r in conn.execute(text(
            "SELECT link, client_id, title, category, pay_type, price_tier, budget, hourly_low, hourly_high, location_requirement, "
            "simulated_day, split FROM service_requests WHERE link = ANY(:links)"), {"links": r_sample["link"].tolist()})}
        ok = all(all(same(getattr(rows[r.link], c), getattr(r, c)) for c in
                     ("client_id", "title", "category", "pay_type", "price_tier", "budget", "hourly_low", "hourly_high",
                      "location_requirement", "simulated_day", "split")) for r in r_sample.itertuples())
        check(f"{len(r_sample)} sample requests equal requests.csv + sampled_requests.csv", ok)
        c_sample = clients.sort_values("client_id").iloc[::400]
        rows = {r.client_id: r for r in conn.execute(text(
            "SELECT client_id, country, preferred_pay_type, median_price_tier FROM clients WHERE client_id = ANY(:ids)"),
            {"ids": c_sample["client_id"].tolist()})}
        check(f"{len(c_sample)} sample clients equal clients.csv",
              all(all(same(getattr(rows[c.client_id], f), getattr(c, f)) for f in ("country", "preferred_pay_type", "median_price_tier"))
                  for c in c_sample.itertuples()))

        # ---- provenance ---------------------------------------------------------------
        meta = dict(conn.execute(text("SELECT key, value FROM load_metadata")).all())
        recorded = {k.removeprefix("source_sha256."): v for k, v in meta.items() if k.startswith("source_sha256.")}
        check("recorded source-file hashes equal the current files",
              bool(recorded) and all(sha256_of(REPO_ROOT / p) == h for p, h in recorded.items()))
        check("no private file was a load source", not any("synthetic_private" in p for p in recorded))

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
