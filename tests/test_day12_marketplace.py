"""Day 12 marketplace tests.

One behavioural suite (MarketplaceBehaviour) runs against BOTH backends:
- InMemoryMarketplaceTests: default, no database;
- PostgresMarketplaceTests: OPT-IN (HIREWISE_RUN_POSTGRES_TESTS=1), on the
  isolated POSTGRES_TEST_DB only -- never the main database.
Pure rule tests (booking state machine, validation) need neither.
"""
import json
import os
import re
import unittest
from pathlib import Path

from werkzeug.security import check_password_hash

from src.app import create_app
from src.app.marketplace import rules
from src.app.marketplace.runtime import RuntimeRecommender, RuntimeRequest
from src.app.marketplace.security import hash_password
from src.app.marketplace.validation import validate_registration, validate_request, validate_review
from src.data.io import sha256_of

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
HAVE_DATA = SYNTHETIC.exists() and (REPO_ROOT / "results" / "day8" / "selected_hybrid_config.json").exists()
PASSWORD = "correct horse battery"
HISTORY_PATH_TYPES = {"repeat_hire", "hired_similar_skill", "co_hire"}

_SHARED = {}


def shared():
    """Frozen service + runtime recommender, built once per test run."""
    if not _SHARED:
        from src.app.services import RecommendationService
        from src.db.repositories import InMemoryRepository

        service = RecommendationService.build(repository=InMemoryRepository.from_artifacts())
        _SHARED["service"] = service
        _SHARED["runtime"] = RuntimeRecommender.build(service)
    return _SHARED["service"], _SHARED["runtime"]


# ---------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------
class TestBookingStateMachine(unittest.TestCase):
    def test_valid_transitions(self):
        self.assertEqual(rules.next_status("pending", "accept", "worker"), "accepted")
        self.assertEqual(rules.next_status("pending", "decline", "worker"), "declined")
        self.assertEqual(rules.next_status("pending", "cancel", "client"), "cancelled")
        self.assertEqual(rules.next_status("accepted", "complete", "client"), "completed")
        self.assertEqual(rules.next_status("accepted", "cancel", "client"), "cancelled")

    def test_invalid_transitions(self):
        invalid = [("pending", "complete", "client"), ("pending", "accept", "client"), ("accepted", "accept", "worker"),
                   ("accepted", "complete", "worker"), ("accepted", "decline", "worker"), ("pending", "cancel", "worker")]
        invalid += [(terminal, action, role) for terminal in rules.TERMINAL_STATUSES for action in rules.ACTIONS
                    for role in ("client", "worker")]
        for current, action, role in invalid:
            with self.assertRaises(rules.InvalidTransition, msg=(current, action, role)):
                rules.next_status(current, action, role)

    def test_allowed_actions(self):
        self.assertEqual(rules.allowed_actions("pending", "worker"), ["accept", "decline"])
        self.assertEqual(rules.allowed_actions("pending", "client"), ["cancel"])
        self.assertEqual(sorted(rules.allowed_actions("accepted", "client")), ["cancel", "complete"])
        self.assertEqual(rules.allowed_actions("completed", "client"), [])

    def test_actor_checks(self):
        booking = {"client_user_id": 1, "worker_id": "W000001", "status": "pending"}
        self.assertEqual(rules.check_booking_actor(booking, {"role": "client", "user_id": 1}), "client")
        self.assertEqual(rules.check_booking_actor(booking, {"role": "worker", "user_id": 9, "worker_id": "W000001"}), "worker")
        for user in ({"role": "client", "user_id": 2}, {"role": "worker", "user_id": 9, "worker_id": "W000002"},
                     {"role": "worker", "user_id": 9, "worker_id": None}):
            with self.assertRaises(rules.NotAuthorized):
                rules.check_booking_actor(booking, user)


class TestValidation(unittest.TestCase):
    VOCAB = {"python": "Python", "logo design": "Logo Design"}

    def test_registration_rules(self):
        base = {"email": "A@Example.com", "role": "client", "display_name": "Ann", "password": "x" * 10, "confirm_password": "x" * 10}
        data, errors = validate_registration(base, {"Cat"}, self.VOCAB, ["Kenya"])
        self.assertEqual((data["email"], errors), ("a@example.com", {}))
        for change, field in [({"email": "nope"}, "email"), ({"password": "short", "confirm_password": "short"}, "password"),
                              ({"confirm_password": "y" * 10}, "confirm_password"), ({"role": "admin"}, "role"),
                              ({"display_name": ""}, "display_name"), ({"country": "Atlantis"}, "country")]:
            self.assertIn(field, validate_registration({**base, **change}, {"Cat"}, self.VOCAB, ["Kenya"])[1])

    def test_worker_registration_requires_profile(self):
        form = {"email": "w@example.com", "role": "worker", "display_name": "W", "password": "x" * 10,
                "confirm_password": "x" * 10, "country": "Kenya"}
        _, errors = validate_registration(form, {"Cat"}, self.VOCAB, ["Kenya"])
        self.assertTrue({"primary_category", "experience_level", "experience_years", "rate", "skills"} <= set(errors))
        ok = {**form, "primary_category": "Cat", "experience_level": "entry", "experience_years": "2", "rate": "15",
              "skills": "PYTHON, logo design"}
        data, errors = validate_registration(ok, {"Cat"}, self.VOCAB, ["Kenya"])
        self.assertEqual(errors, {})
        self.assertEqual(data["worker_profile"]["skills"], ["Logo Design", "Python"])

    def test_request_rules(self):
        base = {"title": "Need a logo", "category": "Cat", "skills": "logo design", "pay_type": "fixed", "budget": "100"}
        self.assertEqual(validate_request(base, {"Cat"}, self.VOCAB, ["Kenya"])[1], {})
        for change, field in [({"category": "Nope"}, "category"), ({"skills": "telepathy"}, "skills"), ({"skills": ""}, "skills"),
                              ({"pay_type": "barter"}, "pay_type"), ({"budget": "-5"}, "budget"), ({"budget": "abc"}, "budget"),
                              ({"budget": ""}, "budget"), ({"title": "ab"}, "title"),
                              ({"pay_type": "hourly", "hourly_low": "30", "hourly_high": "10"}, "hourly_high"),
                              ({"location_requirement": "Atlantis"}, "location_requirement")]:
            self.assertIn(field, validate_request({**base, **change}, {"Cat"}, self.VOCAB, ["Kenya"])[1], change)
        hourly, errors = validate_request({**base, "pay_type": "hourly", "budget": "999", "hourly_low": "10"}, {"Cat"}, self.VOCAB, [])
        self.assertEqual((errors, hourly["budget"], hourly["hourly_low"]), ({}, None, 10.0))

    def test_review_rules(self):
        self.assertEqual(validate_review({"rating": "5", "review_text": " Great "})[0], {"rating": 5, "review_text": "Great"})
        for bad in ("0", "6", "x", "", "4.5"):
            self.assertIn("rating", validate_review({"rating": bad})[1])
        self.assertIn("review_text", validate_review({"rating": "3", "review_text": "x" * 2001})[1])


# ---------------------------------------------------------------------------
# Behavioural suite (both backends)
# ---------------------------------------------------------------------------
TOKEN = re.compile(r'name="csrf_token" value="([^"]+)"')


class MarketplaceBehaviour:
    """Mixin; subclasses provide make_repository()."""

    def setUp(self):
        self.service, self.runtime = shared()
        self.repo = self.make_repository()
        self.app = create_app("testing", service=self.service, marketplace_repository=self.repo, runtime_recommender=self.runtime)

    # ---- helpers ---------------------------------------------------------------
    def new_client(self):
        return self.app.test_client()

    def token(self, client, path="/login"):
        return TOKEN.search(client.get(path).get_data(as_text=True)).group(1)

    def post(self, client, path, data=None, token_path="/login", **kwargs):
        return client.post(path, data={**(data or {}), "csrf_token": self.token(client, token_path)}, **kwargs)

    def register(self, client, email, role="client", **extra):
        form = {"email": email, "role": role, "display_name": email.split("@")[0], "password": PASSWORD,
                "confirm_password": PASSWORD, **extra}
        return client.post("/register", data={**form, "csrf_token": self.token(client, "/register")})

    def register_worker_form(self):
        return {"country": "India", "primary_category": "Logo Design", "experience_level": "intermediate",
                "experience_years": "4", "rate": "25", "skills": "logo design, Adobe Illustrator"}

    def login(self, client, email, password=PASSWORD):
        return self.post(client, "/login", {"email": email, "password": password})

    def user_id(self, email):
        return self.repo.get_credentials(email)[0]

    def create_request(self, client, **overrides):
        form = {"title": "Logo for a bakery", "category": "Logo Design", "skills": "Logo Design, Adobe Illustrator",
                "pay_type": "fixed", "budget": "150", **overrides}
        return self.post(client, "/client/requests/new", form, token_path="/client/requests/new")

    def client_with_request(self, email="client@example.com"):
        c = self.new_client()
        self.register(c, email)
        response = self.create_request(c)
        request_id = int(response.headers["Location"].rstrip("/").split("/")[-1])
        return c, request_id

    def linked_worker(self, worker_id, email="worker@example.com"):
        """Demo-style worker account linked to an imported catalogue profile."""
        self.repo.create_user(email, hash_password(PASSWORD), "worker", "Linked Worker", worker_id=worker_id)
        c = self.new_client()
        self.login(c, email)
        return c

    def top_worker(self, request_id):
        req = self.repo.get_request(request_id)
        rr = RuntimeRequest(request_id, req["client_user_id"], req["category"], tuple(req["skills"]), req["price_tier"],
                            req["location_requirement"], None)
        return self.runtime.recommend(rr, 10)["recommendations"][0]["worker"]["worker_id"]

    def booking_for(self, request_id):
        return self.repo.list_bookings(request_id=request_id)

    # ---- registration ------------------------------------------------------------
    def test_register_client_and_hash_password(self):
        c = self.new_client()
        response = self.register(c, "New.Client@Example.com")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/dashboard"))
        user_id, stored = self.repo.get_credentials("new.client@example.com")
        self.assertNotEqual(stored, PASSWORD)
        self.assertNotIn(PASSWORD, stored)
        self.assertTrue(stored.startswith(("scrypt:", "pbkdf2:")))
        self.assertTrue(check_password_hash(stored, PASSWORD))
        self.assertNotIn("password_hash", self.repo.get_user(user_id))
        self.assertIn("Client dashboard", c.get("/dashboard").get_data(as_text=True))

    def test_register_worker_creates_runtime_profile(self):
        c = self.new_client()
        self.assertEqual(self.register(c, "w@example.com", role="worker", **self.register_worker_form()).status_code, 302)
        profile = self.repo.get_runtime_worker_profile(self.user_id("w@example.com"))
        self.assertEqual(profile["skills"], ["Adobe Illustrator", "Logo Design"])
        self.assertIsNone(self.repo.get_user(self.user_id("w@example.com"))["worker_id"])  # never linked by registration
        self.assertIn("not recommended to clients yet", c.get("/dashboard").get_data(as_text=True))

    def test_duplicate_email_rejected_case_insensitively(self):
        self.register(self.new_client(), "dup@example.com")
        response = self.register(self.new_client(), "DUP@example.com")
        self.assertEqual(response.status_code, 400)
        self.assertIn("already exists", response.get_data(as_text=True))

    def test_invalid_registrations_rejected(self):
        cases = [{"email": "not-an-email"}, {"password": "short", "confirm_password": "short"}, {"role": "admin"},
                 {"confirm_password": "different password"}]
        for i, change in enumerate(cases):
            c = self.new_client()
            form = {"email": f"bad{i}@example.com", "role": "client", "display_name": "x", "password": PASSWORD,
                    "confirm_password": PASSWORD, **change}
            response = c.post("/register", data={**form, "csrf_token": self.token(c, "/register")})
            self.assertEqual(response.status_code, 400, change)
            self.assertIsNone(self.repo.get_credentials(form["email"].lower()), change)
        c = self.new_client()
        response = self.register(c, "badworker@example.com", role="worker", **{**self.register_worker_form(), "skills": "telepathy"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(PASSWORD, response.get_data(as_text=True))  # password never echoed

    # ---- login / logout -------------------------------------------------------------
    def test_login_logout_and_protected_routes(self):
        self.register(self.new_client(), "login@example.com")
        c = self.new_client()
        self.assertEqual(c.get("/dashboard").status_code, 302)
        self.assertIn("/login?next=/dashboard", c.get("/dashboard").headers["Location"])
        wrong = self.login(c, "login@example.com", "wrong password!")
        unknown = self.login(c, "nobody@example.com")
        self.assertEqual((wrong.status_code, unknown.status_code), (401, 401))
        self.assertIn("Invalid email or password.", wrong.get_data(as_text=True))
        self.assertIn("Invalid email or password.", unknown.get_data(as_text=True))
        before = self.token(c)
        ok = self.login(c, "login@example.com")
        self.assertEqual((ok.status_code, ok.headers["Location"].endswith("/dashboard")), (302, True))
        self.assertNotEqual(self.token(c, "/dashboard"), before)  # session regenerated at login
        self.assertEqual(c.get("/logout").status_code, 405)      # logout is POST-only
        self.assertEqual(self.post(c, "/logout", token_path="/dashboard").status_code, 302)
        self.assertEqual(c.get("/dashboard").status_code, 302)

    def test_login_next_is_local_only(self):
        self.register(self.new_client(), "next@example.com")
        c = self.new_client()
        r = self.post(c, "/login", {"email": "next@example.com", "password": PASSWORD, "next": "//evil.example/x"})
        self.assertTrue(r.headers["Location"].endswith("/dashboard"))

    def test_login_rate_limited(self):
        self.register(self.new_client(), "limit@example.com")
        c = self.new_client()
        for _ in range(5):
            self.assertEqual(self.login(c, "limit@example.com", "wrong password!").status_code, 401)
        self.assertEqual(self.login(c, "limit@example.com").status_code, 429)

    # ---- authorization ----------------------------------------------------------------
    def test_role_guards(self):
        client = self.new_client()
        self.register(client, "rc@example.com")
        worker = self.new_client()
        self.register(worker, "rw@example.com", role="worker", **self.register_worker_form())
        self.assertEqual(client.get("/profile").status_code, 403)
        self.assertEqual(worker.get("/client/requests/new").status_code, 403)
        self.assertEqual(worker.get("/client/requests").status_code, 403)
        r = worker.post("/client/requests/new", data={"title": "x", "csrf_token": self.token(worker, "/dashboard")})
        self.assertEqual(r.status_code, 403)

    def test_clients_cannot_touch_each_others_requests_or_bookings(self):
        owner, request_id = self.client_with_request("owner@example.com")
        worker_id = self.top_worker(request_id)
        self.post(owner, f"/client/requests/{request_id}/book", {"worker_id": worker_id}, token_path="/dashboard")
        booking_id = self.booking_for(request_id)[0]["booking_id"]
        intruder = self.new_client()
        self.register(intruder, "intruder@example.com")
        self.assertEqual(intruder.get(f"/client/requests/{request_id}").status_code, 404)
        self.assertEqual(self.post(intruder, f"/client/requests/{request_id}/book", {"worker_id": worker_id},
                                   token_path="/dashboard").status_code, 404)
        self.assertEqual(self.post(intruder, f"/bookings/{booking_id}/cancel", token_path="/dashboard").status_code, 404)
        other_worker = self.linked_worker("W000001" if worker_id != "W000001" else "W000002", "other@example.com")
        self.assertEqual(self.post(other_worker, f"/bookings/{booking_id}/accept", token_path="/dashboard").status_code, 404)
        self.assertEqual(self.repo.get_booking(booking_id)["status"], "pending")

    # ---- requests and runtime recommendations ---------------------------------------------
    def test_request_creation_records_owner_and_price_tier(self):
        c, request_id = self.client_with_request("req@example.com")
        req = self.repo.get_request(request_id)
        self.assertEqual(req["client_user_id"], self.user_id("req@example.com"))
        self.assertEqual(req["skills"], ["Adobe Illustrator", "Logo Design"])
        self.assertEqual(req["price_tier"], self.runtime.price_tiers.tier("Logo Design", "fixed", 150.0, None, None))
        page = c.get(f"/client/requests/{request_id}").get_data(as_text=True)
        self.assertEqual(page.count('class="recommendation"'), 10)
        self.assertIn("collaborative component contributes nothing", page)

    def test_invalid_requests_rejected(self):
        c = self.new_client()
        self.register(c, "badreq@example.com")
        for change in ({"category": "Not a category"}, {"skills": "telepathy"}, {"pay_type": "barter"}, {"budget": "-1"},
                       {"pay_type": "hourly", "budget": "", "hourly_low": "50", "hourly_high": "10"}):
            self.assertEqual(self.create_request(c, **change).status_code, 400, change)
        self.assertEqual(self.repo.list_requests_for_client(self.user_id("badreq@example.com")), [])

    def test_runtime_recommendations_fabricate_no_history(self):
        _, request_id = self.client_with_request("nohist@example.com")
        req = self.repo.get_request(request_id)
        rr = RuntimeRequest(request_id, req["client_user_id"], req["category"], tuple(req["skills"]), req["price_tier"], None, None)
        result = self.runtime.recommend(rr, 10)
        self.assertEqual(result["history"], {"client_linked_to_imported_profile": False,
                                             "collaborative_history_available": False, "kg_hiring_history_available": False})
        self.assertTrue(all(r["contributions"]["cf"] == 0.0 for r in result["recommendations"]))
        self.assertFalse({e["path_type"] for r in result["recommendations"] for e in r["kg_evidence"]} & HISTORY_PATH_TYPES)
        self.assertEqual(len(result["recommendations"]), 10)
        self.assertEqual(result["model"]["weights"], {"cbf": 0.4, "cf": 0.1, "kg": 0.5})

    # ---- bookings ----------------------------------------------------------------------
    def test_full_booking_and_review_workflow(self):
        c, request_id = self.client_with_request("flow@example.com")
        worker_id = self.top_worker(request_id)
        self.assertEqual(c.get(f"/client/requests/{request_id}/book").status_code, 405)  # GET never changes state
        r = self.post(c, f"/client/requests/{request_id}/book", {"worker_id": worker_id}, token_path="/dashboard")
        self.assertEqual(r.status_code, 302)
        [booking] = self.booking_for(request_id)
        self.assertEqual((booking["status"], booking["worker_id"]), ("pending", worker_id))
        # a second active booking for the same request is refused
        self.post(c, f"/client/requests/{request_id}/book", {"worker_id": worker_id}, token_path="/dashboard")
        self.assertEqual(len(self.booking_for(request_id)), 1)
        # client may not complete before acceptance; worker accepts
        self.post(c, f"/bookings/{booking['booking_id']}/complete", token_path="/dashboard")
        self.assertEqual(self.repo.get_booking(booking["booking_id"])["status"], "pending")
        worker = self.linked_worker(worker_id)
        self.assertIn(f"#{booking['booking_id']}", worker.get("/bookings").get_data(as_text=True))
        self.post(worker, f"/bookings/{booking['booking_id']}/complete", token_path="/dashboard")  # not the worker's action
        self.assertEqual(self.repo.get_booking(booking["booking_id"])["status"], "pending")
        self.post(worker, f"/bookings/{booking['booking_id']}/accept", token_path="/dashboard")
        self.assertEqual(self.repo.get_booking(booking["booking_id"])["status"], "accepted")
        # reviewing before completion is refused
        self.post(c, f"/bookings/{booking['booking_id']}/review", {"rating": "5"}, token_path="/dashboard")
        self.assertIsNone(self.repo.get_review(booking["booking_id"]))
        self.post(c, f"/bookings/{booking['booking_id']}/complete", token_path="/dashboard")
        done = self.repo.get_booking(booking["booking_id"])
        self.assertEqual(done["status"], "completed")
        self.assertIsNotNone(done["completed_at"])
        # invalid ratings, then a valid review, then a duplicate
        for bad in ("0", "6", "x"):
            self.post(c, f"/bookings/{booking['booking_id']}/review", {"rating": bad}, token_path="/dashboard")
        self.assertIsNone(self.repo.get_review(booking["booking_id"]))
        self.post(c, f"/bookings/{booking['booking_id']}/review", {"rating": "4", "review_text": "Clear, quick work."},
                  token_path="/dashboard")
        self.assertEqual(self.repo.get_review(booking["booking_id"])["rating"], 4)
        self.post(c, f"/bookings/{booking['booking_id']}/review", {"rating": "1"}, token_path="/dashboard")
        self.assertEqual(self.repo.get_review(booking["booking_id"])["rating"], 4)
        self.assertEqual(self.post(worker, f"/bookings/{booking['booking_id']}/review", {"rating": "5"},
                                   token_path="/dashboard").status_code, 403)
        self.assertIn("Clear, quick work.", worker.get("/profile").get_data(as_text=True))

    def test_decline_and_cancel_paths_allow_rebooking(self):
        c, request_id = self.client_with_request("decline@example.com")
        worker_id = self.top_worker(request_id)
        self.post(c, f"/client/requests/{request_id}/book", {"worker_id": worker_id}, token_path="/dashboard")
        first = self.booking_for(request_id)[0]["booking_id"]
        worker = self.linked_worker(worker_id, "decliner@example.com")
        self.post(worker, f"/bookings/{first}/decline", token_path="/dashboard")
        self.assertEqual(self.repo.get_booking(first)["status"], "declined")
        self.post(worker, f"/bookings/{first}/accept", token_path="/dashboard")  # terminal: no change
        self.assertEqual(self.repo.get_booking(first)["status"], "declined")
        self.post(c, f"/client/requests/{request_id}/book", {"worker_id": worker_id}, token_path="/dashboard")
        second = self.booking_for(request_id)[0]["booking_id"]
        self.assertNotEqual(first, second)
        self.post(c, f"/bookings/{second}/cancel", token_path="/dashboard")
        self.assertEqual(self.repo.get_booking(second)["status"], "cancelled")

    def test_ineligible_worker_cannot_be_booked(self):
        c, request_id = self.client_with_request("inelig@example.com")
        eligible = set(self.runtime.candidates(RuntimeRequest(request_id, 0, "Logo Design", (), None, None, None)))
        outsider = next(w for w in self.service.data.workers["worker_id"] if w not in eligible)
        r = self.post(c, f"/client/requests/{request_id}/book", {"worker_id": outsider}, token_path="/dashboard")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.booking_for(request_id), [])

    # ---- CSRF --------------------------------------------------------------------------
    def test_csrf_required_for_state_changes(self):
        c = self.new_client()
        self.register(c, "csrf@example.com")
        for path, data in [("/client/requests/new", {"title": "Logo", "category": "Logo Design", "skills": "Logo Design",
                                                     "pay_type": "unspecified"}), ("/logout", {})]:
            self.assertEqual(c.post(path, data=data).status_code, 400)
            self.assertEqual(c.post(path, data={**data, "csrf_token": "forged"}).status_code, 400)
        self.assertIn("Session expired", c.post("/logout", data={}).get_data(as_text=True))
        other = self.new_client()
        foreign = self.token(other)
        self.assertEqual(c.post("/logout", data={"csrf_token": foreign}).status_code, 400)
        self.assertEqual(self.new_client().post("/register", data={"email": "x@example.com"}).status_code, 400)
        self.assertEqual(self.repo.list_requests_for_client(self.user_id("csrf@example.com")), [])

    # ---- regression ---------------------------------------------------------------------
    def test_day10_demo_and_api_still_work(self):
        c = self.new_client()
        page = c.get("/").get_data(as_text=True)
        for element in ('id="health-status"', 'id="request-list"', 'id="recommendation-list"', "/static/js/app.js", ">Login<"):
            self.assertIn(element, page)
        self.assertEqual(c.get("/api/requests?per_page=1").get_json()["total"], 3600)
        self.assertEqual(c.get("/api/nope").get_json()["error"]["code"], "not_found")
        self.assertIn("<html", c.get("/nope").get_data(as_text=True))  # HTML pages get HTML errors


@unittest.skipUnless(HAVE_DATA, "processed data / Day 8 results absent")
class InMemoryMarketplaceTests(MarketplaceBehaviour, unittest.TestCase):
    def make_repository(self):
        from src.db.marketplace import InMemoryMarketplaceRepository

        if "memory_reference" not in _SHARED:
            _SHARED["memory_reference"] = InMemoryMarketplaceRepository.from_artifacts()
        ref = _SHARED["memory_reference"]
        return InMemoryMarketplaceRepository(ref.categories(), ref.skills(), ref._clients, ref._workers)


@unittest.skipUnless(HAVE_DATA and os.environ.get("HIREWISE_RUN_POSTGRES_TESTS") == "1",
                     "opt-in live PostgreSQL test: set HIREWISE_RUN_POSTGRES_TESTS=1")
class PostgresMarketplaceTests(MarketplaceBehaviour, unittest.TestCase):
    """Runs the same behaviour on POSTGRES_TEST_DB only (never POSTGRES_DB)."""

    @classmethod
    def setUpClass(cls):
        from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine
        from src.db.loader import load
        from src.db.schema import metadata

        test_db, main_db = os.environ.get("POSTGRES_TEST_DB"), os.environ.get("POSTGRES_DB")
        if not test_db or test_db == main_db:
            raise unittest.SkipTest("POSTGRES_TEST_DB must be set and different from POSTGRES_DB")
        settings = load_postgres_settings(TEST_DATABASE_ENV)
        assert settings.database == test_db != main_db
        cls.engine = make_engine(settings)
        metadata.drop_all(cls.engine)
        metadata.create_all(cls.engine)
        load(cls.engine)

    @classmethod
    def tearDownClass(cls):
        from src.db.schema import metadata

        metadata.drop_all(cls.engine)

    def make_repository(self):
        from src.db.marketplace import PostgresMarketplaceRepository
        from src.db.schema import RUNTIME_TABLES, metadata

        with self.engine.begin() as conn:  # fresh runtime state per test; imported tables untouched
            for name in reversed(RUNTIME_TABLES):
                conn.execute(metadata.tables[name].delete())
        return PostgresMarketplaceRepository(self.engine)

    def test_database_constraints_hold_directly(self):
        from sqlalchemy import insert, update
        from sqlalchemy.exc import IntegrityError
        from src.db.schema import app_reviews, bookings, users

        repo = self.make_repository()
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as conn:
                conn.execute(insert(users).values(email="Upper@Example.com", password_hash="h", role="client", display_name="x"))
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as conn:
                conn.execute(insert(users).values(email="r@example.com", password_hash="h", role="admin", display_name="x"))
        client_id = repo.create_user("db@example.com", "h", "client", "db")
        other_id = repo.create_user("db2@example.com", "h", "client", "db2")
        request_id = repo.create_request(client_id, "Logo work", "Logo Design", ["Logo Design"], "unspecified",
                                         None, None, None, None, None)
        with self.assertRaises(IntegrityError):  # booking client must own the request (composite FK)
            with self.engine.begin() as conn:
                conn.execute(insert(bookings).values(request_id=request_id, client_user_id=other_id, worker_id="W000001",
                                                     status="pending"))
        booking_id = repo.create_booking(request_id, client_id, "W000001")
        with self.assertRaises(IntegrityError):  # completed requires completed_at
            with self.engine.begin() as conn:
                conn.execute(update(bookings).where(bookings.c.booking_id == booking_id).values(status="completed"))
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as conn:
                conn.execute(insert(app_reviews).values(booking_id=booking_id, rating=7))


@unittest.skipUnless(HAVE_DATA, "processed data / Day 8 results absent")
class TestFrozenArtifactsUntouched(unittest.TestCase):
    def test_runtime_use_leaves_frozen_models_and_results_unchanged(self):
        service, runtime = shared()
        link = service.data.requests.loc[service.data.requests["split"] == "test", "link"].iloc[0]
        before = json.dumps(service.recommend(link, 10), sort_keys=True)
        cf_map = dict(service.models["cf"].link_to_client)
        runtime.recommend(RuntimeRequest(10_000, 1, "Logo Design", ("Logo Design",), 0.5, None, None), 10)
        self.assertEqual(json.dumps(service.recommend(link, 10), sort_keys=True), before)
        self.assertEqual(service.models["cf"].link_to_client, cf_map)
        with open(REPO_ROOT / "results" / "day8" / "day8_manifest.json", encoding="utf-8") as fh:
            manifest = json.load(fh)
        for entry in manifest["files"]:
            self.assertEqual(sha256_of(REPO_ROOT / entry["path"]), entry["sha256"], entry["path"])

    def test_price_tier_reproduces_day2_definition(self):
        _, runtime = shared()
        import pandas as pd

        df = pd.read_csv(REPO_ROOT / "data" / "processed" / "sampled_requests.csv",
                         usecols=["category", "pay_type", "pay_amount", "price_tier", "budget", "hourly_low", "hourly_high"])
        # A job already in the reference group, re-inserted, must get a tier no lower than... its own definition:
        # checked exactly via the leave-one-out identity on a fixed sample.
        sample = df.dropna(subset=["price_tier"]).iloc[::997]
        groups = runtime.price_tiers._groups
        for r in sample.itertuples():
            values = list(groups[(r.category, r.pay_type)])
            values.remove(r.pay_amount)
            from src.app.marketplace.runtime import PriceTierReference

            loo = PriceTierReference({(r.category, r.pay_type): __import__("numpy").array(sorted(values))})
            self.assertAlmostEqual(loo.tier(r.category, r.pay_type, r.budget, r.hourly_low, r.hourly_high), r.price_tier, places=12)
        self.assertIsNone(runtime.price_tiers.tier("Logo Design", "unspecified", None, None, None))


if __name__ == "__main__":
    unittest.main()
