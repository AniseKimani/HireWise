"""Live PostgreSQL tests (Day 11) -- OPT-IN.

Run with:  set HIREWISE_RUN_POSTGRES_TESTS=1  (PowerShell: $env:HIREWISE_RUN_POSTGRES_TESTS="1")

- TestMainDatabaseReadOnly: read-only checks against POSTGRES_DB (the
  application database). Nothing is written.
- TestLoadCycleOnTestDatabase: DESTRUCTIVE create/load/reset/drop cycle,
  run ONLY against POSTGRES_TEST_DB, and refuses to start if that name is
  missing or equal to POSTGRES_DB.
"""
import os
import unittest

from sqlalchemy import inspect, select, text, update

RUN = os.environ.get("HIREWISE_RUN_POSTGRES_TESTS") == "1"


@unittest.skipUnless(RUN, "opt-in live PostgreSQL test: set HIREWISE_RUN_POSTGRES_TESTS=1")
class TestMainDatabaseReadOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.db.database import make_engine
        from src.db.loader import build_tables

        cls.engine = make_engine()
        cls.source = build_tables()

    def test_connection_and_tables(self):
        from src.db.schema import metadata

        self.assertTrue(set(metadata.tables) <= set(inspect(self.engine).get_table_names()))

    def test_contents_equal_source_exactly(self):
        from src.db.loader import fingerprint, read_back, stored_metadata

        with self.engine.connect() as conn:
            db = read_back(conn)
            meta = stored_metadata(conn)
        self.assertEqual({k: len(v) for k, v in db.items()}, {k: len(v) for k, v in self.source.items()})
        self.assertEqual(fingerprint(db), fingerprint(self.source))
        self.assertEqual(meta["source_fingerprint"], fingerprint(self.source))

    def test_constraints_present(self):
        with self.engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT tc.table_name, tc.constraint_type, count(*) FROM information_schema.table_constraints tc "
                "WHERE tc.table_schema = 'public' GROUP BY 1, 2")).all()
        kinds = {(t, k): n for t, k, n in rows}
        self.assertGreaterEqual(kinds[("ratings", "FOREIGN KEY")], 2)
        self.assertGreaterEqual(kinds[("service_requests", "FOREIGN KEY")], 2)
        self.assertEqual(kinds[("workers", "PRIMARY KEY")], 1)

    def test_constraint_rejects_invalid_rating_without_persisting(self):
        from sqlalchemy.exc import IntegrityError
        from src.db.schema import ratings

        with self.engine.connect() as conn:
            transaction = conn.begin()
            try:
                with self.assertRaises(IntegrityError):
                    conn.execute(update(ratings).where(ratings.c.link == self.source["ratings"][0]["link"]).values(rating=9))
            finally:
                transaction.rollback()  # never commits anything to the main database

    def test_representative_rows_and_mappings(self):
        from src.db.repositories import InMemoryRepository, PostgresRepository

        live, reference = PostgresRepository(self.engine), InMemoryRepository(self.source)
        link = self.source["service_requests"][0]["link"]
        self.assertEqual(live.get_request(link), reference.get_request(link))
        ids = ["W000001", "W000517", "W001288", "W002999"]
        self.assertEqual(live.get_workers(ids), reference.get_workers(ids))
        for args in [(1, 50), (3, 7), (1, 50, "logo"), (1, 50, None, "Logo Design"), (1, 50, None, None, "PYTHON"),
                     (1, 20, "пис")]:
            self.assertEqual(live.list_browsable(*args), reference.list_browsable(*args), args)
        self.assertEqual(live.browsable_categories(), reference.browsable_categories())
        self.assertEqual(live.health(), {"available": True, "data_loaded": True})


@unittest.skipUnless(RUN, "opt-in live PostgreSQL test: set HIREWISE_RUN_POSTGRES_TESTS=1")
class TestLoadCycleOnTestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine

        test_db, main_db = os.environ.get("POSTGRES_TEST_DB"), os.environ.get("POSTGRES_DB")
        if not test_db or test_db == main_db:
            raise unittest.SkipTest("POSTGRES_TEST_DB must be set and different from POSTGRES_DB")
        settings = load_postgres_settings(TEST_DATABASE_ENV)
        assert settings.database == test_db != main_db
        cls.engine = make_engine(settings)

    def setUp(self):
        from src.db.schema import metadata

        metadata.drop_all(self.engine)
        metadata.create_all(self.engine)

    def tearDown(self):
        from src.db.schema import metadata

        metadata.drop_all(self.engine)

    def test_load_is_idempotent_refuses_drift_and_resets_explicitly(self):
        from src.db.loader import LoadRefused, fingerprint, load, read_back
        from src.db.schema import workers

        first = load(self.engine)
        self.assertEqual(first["action"], "loaded")
        again = load(self.engine)
        self.assertEqual(again["action"], "unchanged")
        self.assertEqual(again["row_counts"], first["row_counts"])

        with self.engine.begin() as conn:
            conn.execute(update(workers).where(workers.c.worker_id == "W000001").values(rate=1.0))
        with self.assertRaises(LoadRefused):
            load(self.engine)
        with self.engine.connect() as conn:
            self.assertNotEqual(fingerprint(read_back(conn)), first["fingerprint"])

        reset = load(self.engine, reset=True)
        self.assertEqual(reset["action"], "reset_and_loaded")
        with self.engine.connect() as conn:
            self.assertEqual(fingerprint(read_back(conn)), first["fingerprint"])
            self.assertEqual(conn.execute(select(workers.c.rate).where(workers.c.worker_id == "W000001")).scalar_one(), 10.0)


if __name__ == "__main__":
    unittest.main()
