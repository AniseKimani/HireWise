"""Day 11 relational schema and loader tests -- no database needed.

Schema: allow-listed columns only, keys/constraints/indexes present,
identifier collation, PostgreSQL DDL compiles. Loader: exact row counts,
determinism, model-visible sources only, training-period ratings equal to
the frozen KG snapshot's, internal referential integrity."""
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from sqlalchemy import CheckConstraint, ForeignKeyConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from src.db import schema

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"

EXPECTED_COLUMNS = {
    "categories": {"name"},
    "skills": {"name"},
    "clients": {"client_id", "country", "preferred_pay_type", "median_price_tier"},
    "workers": {"worker_id", "primary_category", "experience_level", "experience_years", "price_tier", "rate", "country", "join_day"},
    "worker_secondary_categories": {"worker_id", "category"},
    "worker_skills": {"worker_id", "skill"},
    "service_requests": {"link", "client_id", "title", "title_search", "category", "pay_type", "price_tier", "budget",
                         "hourly_low", "hourly_high", "location_requirement", "simulated_day", "split"},
    "request_skills": {"link", "skill"},
    "ratings": {"link", "worker_id", "rating", "has_review"},
    "load_metadata": {"key", "value"},
}
PRIVATE_MARKERS = ("quality", "true_skill", "proficiency", "taste", "utility", "noise", "oracle", "cold_start",
                   "activity_quota", "activity_start", "n_requests_assigned", "relevance", "event", "shortlist")


class TestSchema(unittest.TestCase):
    def test_exact_tables_and_columns(self):
        self.assertEqual({t: set(table.c.keys()) for t, table in schema.metadata.tables.items()}, EXPECTED_COLUMNS)

    def test_no_private_or_outcome_columns(self):
        for table in schema.metadata.tables.values():
            for column in table.c.keys():
                for marker in PRIVATE_MARKERS:
                    self.assertNotIn(marker, column, f"{table.name}.{column}")

    def test_primary_keys_use_existing_identifiers(self):
        pks = {t: [c.name for c in table.primary_key.columns] for t, table in schema.metadata.tables.items()}
        self.assertEqual(pks["clients"], ["client_id"])
        self.assertEqual(pks["workers"], ["worker_id"])
        self.assertEqual(pks["service_requests"], ["link"])
        self.assertEqual(pks["ratings"], ["link", "worker_id"])
        self.assertEqual(pks["request_skills"], ["link", "skill"])

    def test_foreign_keys(self):
        fks = {(t, fk.parent.name, fk.column.table.name)
               for t, table in schema.metadata.tables.items() for fk in table.foreign_keys}
        for expected in [("workers", "primary_category", "categories"), ("worker_skills", "skill", "skills"),
                         ("worker_skills", "worker_id", "workers"), ("service_requests", "client_id", "clients"),
                         ("service_requests", "category", "categories"), ("request_skills", "link", "service_requests"),
                         ("ratings", "link", "service_requests"), ("ratings", "worker_id", "workers"),
                         ("worker_secondary_categories", "category", "categories")]:
            self.assertIn(expected, fks)

    def test_check_constraints(self):
        checks = {c.name for table in schema.metadata.tables.values() for c in table.constraints if isinstance(c, CheckConstraint)}
        for name in ("ck_ratings_rating", "ck_service_requests_split", "ck_workers_experience_level",
                     "ck_service_requests_pay_type", "ck_workers_price_tier_range"):
            self.assertIn(name, checks)

    def test_identifiers_use_c_collation(self):
        for table, column in [("service_requests", "link"), ("workers", "worker_id"), ("clients", "client_id"),
                              ("skills", "name"), ("categories", "name"), ("request_skills", "skill")]:
            self.assertEqual(schema.metadata.tables[table].c[column].type.collation, "C")

    def test_indexes_and_postgres_ddl_compile(self):
        names = {i.name for table in schema.metadata.tables.values() for i in table.indexes}
        self.assertEqual(names, {"ix_service_requests_split_day_link", "ix_service_requests_category",
                                 "ix_service_requests_client_id", "ix_request_skills_skill_lower",
                                 "ix_worker_skills_skill", "ix_ratings_worker_id"})
        dialect = postgresql.dialect()
        for table in schema.metadata.sorted_tables:
            ddl = str(CreateTable(table).compile(dialect=dialect))
            self.assertIn("CREATE TABLE", ddl)
            for index in table.indexes:
                str(CreateIndex(index).compile(dialect=dialect))
        self.assertIn('COLLATE "C"', str(CreateTable(schema.service_requests).compile(dialect=dialect)))

    def test_load_order_respects_foreign_keys(self):
        seen = set()
        for name in schema.TABLE_LOAD_ORDER:
            for fk in schema.DATA_TABLES[name].foreign_keys:
                self.assertIn(fk.column.table.name, seen | {name})
            seen.add(name)


@unittest.skipUnless(SYNTHETIC.exists(), "processed data not present")
class TestLoaderBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.db.loader import build_tables

        cls.tables = build_tables()

    def test_exact_row_counts(self):
        self.assertEqual({k: len(v) for k, v in self.tables.items()}, {
            "categories": 88, "skills": 1187, "clients": 2000, "workers": 3000, "worker_secondary_categories": 3031,
            "worker_skills": 23131, "service_requests": 12000, "request_skills": 56277, "ratings": 6454,
        })

    def test_rows_match_schema_columns(self):
        for name, rows in self.tables.items():
            self.assertEqual(set(rows[0]), EXPECTED_COLUMNS[name], name)

    def test_deterministic_fingerprint(self):
        from src.db.loader import build_tables, fingerprint

        self.assertEqual(fingerprint(self.tables), fingerprint(build_tables()))

    def test_rows_sorted_by_primary_key(self):
        for name, rows in self.tables.items():
            pk = [c.name for c in schema.DATA_TABLES[name].primary_key.columns]
            keys = [tuple(r[c] for c in pk) for r in rows]
            self.assertEqual(keys, sorted(keys), name)
            self.assertEqual(len(keys), len(set(keys)), name)

    def test_only_training_ratings_and_equal_to_kg_snapshot(self):
        from src.kg.graph_view import load_view_from_kg_csvs

        requests = pd.read_csv(SYNTHETIC / "requests.csv")
        train = set(requests.loc[requests["split"] == "train", "link"])
        self.assertTrue(all(r["link"] in train for r in self.tables["ratings"]))
        kg = {(l, w, r) for _, w, l, r, _ in load_view_from_kg_csvs().train_reviews}
        self.assertEqual({(r["link"], r["worker_id"], r["rating"]) for r in self.tables["ratings"]}, kg)

    def test_internal_referential_integrity(self):
        t = self.tables
        cats, skills = {r["name"] for r in t["categories"]}, {r["name"] for r in t["skills"]}
        workers, clients = {r["worker_id"] for r in t["workers"]}, {r["client_id"] for r in t["clients"]}
        links = {r["link"] for r in t["service_requests"]}
        self.assertTrue({r["primary_category"] for r in t["workers"]} <= cats)
        self.assertTrue({r["category"] for r in t["service_requests"]} <= cats)
        self.assertTrue({r["client_id"] for r in t["service_requests"]} <= clients)
        self.assertTrue({r["skill"] for r in t["worker_skills"] + t["request_skills"]} <= skills)
        self.assertTrue({r["worker_id"] for r in t["worker_skills"] + t["ratings"]} <= workers)
        self.assertTrue({r["link"] for r in t["request_skills"] + t["ratings"]} <= links)

    def test_title_search_is_python_lowercase(self):
        self.assertTrue(all(r["title_search"] == r["title"].lower() for r in self.tables["service_requests"]))

    def test_never_reads_private_or_unlisted_files(self):
        from src.db import loader

        opened = []
        real = pd.read_csv

        def recording(path, *args, **kwargs):
            opened.append(str(path).replace("\\", "/"))
            return real(path, *args, **kwargs)

        with mock.patch.object(loader.pd, "read_csv", side_effect=recording):
            loader.build_tables()
        self.assertTrue(opened)
        self.assertFalse(any("synthetic_private" in p for p in opened))
        self.assertTrue(all(any(p.endswith(s) for s in loader.SOURCE_FILES.values()) for p in opened))


if __name__ == "__main__":
    unittest.main()
