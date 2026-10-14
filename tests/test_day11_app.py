"""Day 11 application tests without a live database: repository
semantics, SQL safety, database-unavailable behaviour, health reporting,
and the routes -> services -> repositories layering."""
import ast
import inspect
import unittest
from pathlib import Path

from src.app import create_app, routes, services
from src.db import repositories
from src.db.database import PostgresSettings, make_engine
from src.db.repositories import DatabaseUnavailable, InMemoryRepository, PostgresRepository

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"


def _unreachable_engine():
    # Port 1 on localhost: connection refused immediately, no real database touched.
    return make_engine(PostgresSettings(host="127.0.0.1", port=1, database="none", user="none", password="secret-value"))


class TestSqlSafety(unittest.TestCase):
    def test_repositories_never_build_sql_text(self):
        code = ast.unparse(ast.parse(inspect.getsource(repositories)))
        self.assertNotIn("text(", code)
        self.assertNotIn("exec_driver_sql", code)
        self.assertNotRegex(code, r"f'[^']*(SELECT|WHERE|FROM)", "f-string SQL in repositories")

    def test_routes_and_services_contain_no_sql(self):
        for module in (routes, services):
            code = inspect.getsource(module).upper()
            for keyword in ("SELECT ", "INSERT ", "UPDATE ", "DELETE FROM", "SQLALCHEMY"):
                self.assertNotIn(keyword, code, module.__name__)

    def test_routes_do_not_touch_repositories_directly(self):
        self.assertNotIn("repository", inspect.getsource(routes))


class TestUnreachableDatabase(unittest.TestCase):
    def setUp(self):
        self.repo = PostgresRepository(_unreachable_engine())

    def test_queries_raise_database_unavailable_without_details(self):
        for call in (lambda: self.repo.get_request("x"), lambda: self.repo.list_browsable(1, 10),
                     self.repo.browsable_categories, lambda: self.repo.get_workers(["W1"])):
            with self.assertRaises(DatabaseUnavailable) as ctx:
                call()
            self.assertNotIn("secret-value", str(ctx.exception))
            self.assertEqual(str(ctx.exception), "application database unavailable")

    def test_health_reports_unavailable(self):
        self.assertEqual(self.repo.health(), {"available": False, "data_loaded": None})


class DownRepository:
    def health(self):
        return {"available": False, "data_loaded": None}

    def _fail(self, *args, **kwargs):
        raise DatabaseUnavailable("application database unavailable")

    get_request = list_browsable = browsable_categories = get_workers = get_worker = _fail


class FakeVersion:
    def to_dict(self):
        return {"name": "fake"}


class DownService:
    """Service whose repository is down; recommendation path reaches the repository first."""
    model_version = FakeVersion()

    def __init__(self):
        self.repository = DownRepository()

    def get_request(self, link):
        return self.repository.get_request(link)

    def list_requests(self, *args):
        return self.repository.list_browsable(*args)

    def worker_profile(self, worker_id):
        return self.repository.get_worker(worker_id)

    def has_worker(self, worker_id):
        return self.repository.get_worker(worker_id)

    def recommend(self, link, k, paths=3):
        return self.repository.get_request(link)


class TestDatabaseDownBehaviour(unittest.TestCase):
    def setUp(self):
        service = DownService()
        self.client = create_app("testing", service=service, postgres_probe=service.repository.health,
                                 neo4j_probe=lambda: {"available": True, "graph_matches_day6_counts": True}).test_client()

    def test_data_routes_return_503_database_unavailable(self):
        for path in ("/api/requests", "/api/requests?link=x", "/api/workers/W000001", "/api/recommendations?link=x",
                     "/api/workers/W000001/graph"):
            with self.assertLogs("src.app.errors", level="WARNING"):
                r = self.client.get(path)
            self.assertEqual((r.status_code, r.get_json()["error"]["code"]), (503, "database_unavailable"), path)
            self.assertNotIn("psycopg", r.get_data(as_text=True).lower())

    def test_health_unavailable_when_database_down(self):
        r = self.client.get("/api/health")
        body = r.get_json()
        self.assertEqual(r.status_code, 503)
        self.assertEqual(body["status"], "unavailable")
        self.assertEqual(body["postgresql"], {"available": False, "data_loaded": None, "checked": True})
        self.assertTrue(body["recommender"]["ready"])

    def test_health_unavailable_when_schema_not_loaded(self):
        client = create_app("testing", service=DownService(),
                            postgres_probe=lambda: {"available": True, "data_loaded": False}).test_client()
        self.assertEqual(client.get("/api/health").get_json()["status"], "unavailable")

    def test_health_exposes_no_connection_details(self):
        text = self.client.get("/api/health").get_data(as_text=True)
        for marker in ("hirewise_app", "5432", "localhost", "postgresql+psycopg", "password", ".env"):
            self.assertNotIn(marker, text)


@unittest.skipUnless(SYNTHETIC.exists(), "processed data not present")
class TestInMemoryRepositorySemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = InMemoryRepository.from_artifacts()

    def test_listing_order_filters_and_pagination(self):
        rows, total = self.repo.list_browsable(1, 50)
        self.assertEqual(total, 3600)
        self.assertEqual([(r["simulated_day"], r["link"]) for r in rows], sorted((r["simulated_day"], r["link"]) for r in rows))
        _, python_total = self.repo.list_browsable(1, 10, skill="PYTHON")
        self.assertEqual(python_total, self.repo.list_browsable(1, 10, skill="python")[1])
        self.assertEqual(self.repo.list_browsable(10_000, 50)[0], [])
        self.assertEqual(len(self.repo.browsable_categories()), 88)

    def test_worker_rows_carry_training_review_aggregates(self):
        from src.kg.graph_view import load_view_from_kg_csvs
        from src.recommenders.knowledge_graph import KGIndex

        index = KGIndex(load_view_from_kg_csvs())
        workers = self.repo.get_workers(["W000001", "W000517", "W002999"])
        for w, row in workers.items():
            self.assertEqual(row["review_count"], index.review_count.get(w, 0))
            self.assertEqual(row["review_sum"], index.review_sum.get(w, 0))

    def test_unknown_ids(self):
        self.assertIsNone(self.repo.get_request("https://example.invalid/x"))
        self.assertIsNone(self.repo.get_worker("W999999"))


if __name__ == "__main__":
    unittest.main()
