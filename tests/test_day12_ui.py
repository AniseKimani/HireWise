"""Day 12 UI redesign tests (wireframe alignment): landing page, role-aware
registration (markup, script, CSS and server-side enforcement), role-based
landing after login, Recommended Workers and review/profile views, and
unchanged security properties. No browser automation: the JavaScript is
checked statically and every rule it supports is enforced by the server."""
import re
import unittest
from pathlib import Path

from src.app import create_app
from src.app.marketplace.security import hash_password
from tests.test_day12_marketplace import HAVE_DATA, PASSWORD, TOKEN, shared

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC = REPO_ROOT / "src" / "app" / "static"
REGISTER_JS = (STATIC / "js" / "register.js").read_text(encoding="utf-8")
CSS = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
WORKER_ONLY = ("worker_country", "primary_category", "experience_level", "experience_years", "rate", "skills")


@unittest.skipUnless(HAVE_DATA, "processed data / Day 8 results absent")
class TestRedesignedPages(unittest.TestCase):
    def setUp(self):
        from src.db.marketplace import InMemoryMarketplaceRepository

        self.service, self.runtime = shared()
        self.repo = InMemoryMarketplaceRepository.from_artifacts()
        self.app = create_app("testing", service=self.service, marketplace_repository=self.repo, runtime_recommender=self.runtime)

    # ---- helpers ------------------------------------------------------------------
    def token(self, c, path="/login"):
        return TOKEN.search(c.get(path).get_data(as_text=True)).group(1)

    def post(self, c, path, data, token_path="/login"):
        return c.post(path, data={**data, "csrf_token": self.token(c, token_path)})

    def register(self, c, data):
        return c.post("/register", data={**data, "csrf_token": self.token(c, "/register")})

    def client_form(self, email, **extra):
        return {"email": email, "role": "client", "display_name": "Client", "password": PASSWORD, "confirm_password": PASSWORD, **extra}

    def worker_form(self, email, **extra):
        return {"email": email, "role": "worker", "display_name": "Worker", "password": PASSWORD, "confirm_password": PASSWORD,
                "worker_country": "India", "primary_category": "Logo Design", "experience_level": "entry",
                "experience_years": "2", "rate": "20", "skills": "Logo Design", **extra}

    def html(self, c, path):
        response = c.get(path)
        self.assertEqual(response.status_code, 200, path)
        return response.get_data(as_text=True)

    # ---- landing --------------------------------------------------------------------
    def test_landing_page_for_logged_out_users(self):
        page = self.html(self.app.test_client(), "/")
        for text in ("Find the right people for your projects", "Intelligent matching", "Transparent recommendations",
                     "Secure bookings and reviews", "How HireWise works", 'id="how-it-works"', 'id="for-clients"',
                     'id="for-workers"', 'id="about"', ">Research Demo<"):
            self.assertIn(text, page)
        self.assertRegex(page, r'<a class="button button--lg" href="/register">Sign up</a>')
        self.assertRegex(page, r'href="/login">Log in</a>')
        self.assertNotIn('id="request-list"', page)  # the research demo is no longer the public landing page

    def test_research_demo_is_labelled_and_separate(self):
        page = self.html(self.app.test_client(), "/demo")
        self.assertIn("<h1>Research Demo</h1>", page)
        self.assertIn('id="request-list"', page)

    def test_signed_in_users_land_on_their_dashboard(self):
        client = self.app.test_client()
        self.register(client, self.client_form("landing.client@example.com"))
        self.assertTrue(client.get("/").headers["Location"].endswith("/dashboard"))
        self.assertIn("Client dashboard", self.html(client, "/dashboard"))
        self.repo.create_user("landing.worker@example.com", hash_password(PASSWORD), "worker", "W", worker_id="W001674")
        worker = self.app.test_client()
        r = self.post(worker, "/login", {"email": "landing.worker@example.com", "password": PASSWORD})
        self.assertTrue(r.headers["Location"].endswith("/dashboard"))
        self.assertIn("Worker dashboard", self.html(worker, "/dashboard"))
        self.assertTrue(worker.get("/").headers["Location"].endswith("/dashboard"))

    # ---- registration markup -------------------------------------------------------------
    def test_registration_has_role_selector_and_role_scoped_fields(self):
        page = self.html(self.app.test_client(), "/register")
        self.assertIn('id="role-client" name="role" value="client"', page)
        self.assertIn('id="role-worker" name="role" value="worker"', page)
        for step in ("role", "account", "details"):
            self.assertIn(f'data-step="{step}"', page)
        client_set = re.search(r'<fieldset data-role="client">(.*?)</fieldset>', page, re.S).group(1)
        worker_set = re.search(r'<fieldset data-role="worker">(.*?)</fieldset>', page, re.S).group(1)
        self.assertIn('name="client_country"', client_set)
        for name in WORKER_ONLY:
            self.assertIn(f'name="{name}"', worker_set)
            self.assertNotIn(f'name="{name}"', client_set)
        self.assertNotIn('name="client_country"', worker_set)
        # role-specific fields are never statically required (only the active role's become required)
        self.assertNotRegex(client_set + worker_set, r"\srequired[\s>]")
        for shared_field in ("email", "password", "confirm_password", "display_name"):
            self.assertIn(f'name="{shared_field}"', page)
        self.assertIn("/static/js/register.js", page)

    def test_role_preselection(self):
        page = self.html(self.app.test_client(), "/register?role=worker")
        self.assertRegex(page, r'id="role-worker" name="role" value="worker" checked')

    def test_register_script_hides_disables_and_clears_the_other_role(self):
        for fragment in ('el.disabled = !active', 'el.hidden = !active', 'input.required = active', 'input.value = ""',
                         'selectedIndex = 0', 'data-required'):
            self.assertIn(fragment, REGISTER_JS)
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "eval(", "document.write"):
            self.assertNotIn(sink, REGISTER_JS)
        self.assertIn('form:has(#role-client:checked) [data-role="worker"]', CSS)
        self.assertIn('form:has(#role-worker:checked) [data-role="client"]', CSS)

    # ---- registration server enforcement ----------------------------------------------------
    def test_client_registration_without_worker_fields_succeeds(self):
        r = self.register(self.app.test_client(), self.client_form("plain.client@example.com", client_country="India"))
        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.repo.get_user(self.repo.get_credentials("plain.client@example.com")[0])["country"], "India")

    def test_server_rejects_worker_fields_on_a_client_account(self):
        for field, value in (("primary_category", "Logo Design"), ("skills", "Logo Design"), ("rate", "10"), ("worker_country", "India")):
            r = self.register(self.app.test_client(), self.client_form(f"mixed.{field}@example.com", **{field: value}))
            self.assertEqual(r.status_code, 400, field)
            self.assertIn("Details for the other account type were submitted", r.get_data(as_text=True))
            self.assertIsNone(self.repo.get_credentials(f"mixed.{field}@example.com"), field)

    def test_server_rejects_client_fields_on_a_worker_account(self):
        r = self.register(self.app.test_client(), self.worker_form("mixed.worker@example.com", client_country="India"))
        self.assertEqual(r.status_code, 400)
        self.assertIsNone(self.repo.get_credentials("mixed.worker@example.com"))
        ok = self.register(self.app.test_client(), self.worker_form("good.worker@example.com"))
        self.assertEqual(ok.status_code, 302)
        profile = self.repo.get_runtime_worker_profile(self.repo.get_credentials("good.worker@example.com")[0])
        self.assertEqual((profile["country"], profile["skills"]), ("India", ["Logo Design"]))

    def test_worker_registration_still_requires_worker_fields(self):
        r = self.register(self.app.test_client(), self.worker_form("missing@example.com", primary_category="", skills=""))
        self.assertEqual(r.status_code, 400)

    # ---- Recommended Workers / profile / review ---------------------------------------------------
    def create_request(self, c):
        r = self.post(c, "/client/requests/new", {"title": "Logo for a bakery", "category": "Logo Design",
                                                  "skills": "Logo Design, Adobe Illustrator", "pay_type": "fixed", "budget": "150"},
                      token_path="/client/requests/new")
        return int(r.headers["Location"].split("/")[-1])

    def test_recommended_workers_view_matches_wireframe_sections(self):
        c = self.app.test_client()
        self.register(c, self.client_form("recs.view@example.com"))
        request_id = self.create_request(c)
        page = self.html(c, f"/client/requests/{request_id}")
        self.assertIn("<h1>Recommended Workers</h1>", page)
        self.assertIn("Back to Dashboard", page)
        self.assertEqual(page.count('class="card worker-card"'), 10)
        for text in ("Why this worker was recommended", "View profile", "Book / Hire worker", "Score breakdown",
                     'class="chip"', 'class="rating"', "yrs experience", "Hybrid score: ", "not probabilities"):
            self.assertIn(text, page)
        scores = re.findall(r"Hybrid score: ([0-9.]+)", page)
        self.assertEqual(len(scores), 10)
        self.assertNotRegex(page, r"Hybrid score: [0-9.]+\s*%")  # ranking scores, never percentages
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_view_profile_and_review_page(self):
        c = self.app.test_client()
        self.register(c, self.client_form("review.view@example.com"))
        request_id = self.create_request(c)
        page = self.html(c, f"/workers/W001674?request={request_id}")
        for text in ("Worker W001674", "Associated skills", "Services offered", "Back to Results", "About this profile"):
            self.assertIn(text, page)
        self.assertNotIn("Bio", page)  # not stored, so not shown
        self.assertEqual(c.get("/workers/W999999").status_code, 404)
        self.post(c, f"/client/requests/{request_id}/book", {"worker_id": "W001674"}, token_path="/dashboard")
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.assertIn("Reviews open once the booking is completed", self.html(c, f"/bookings/{booking_id}/review"))
        self.repo.create_user("review.worker@example.com", hash_password(PASSWORD), "worker", "W", worker_id="W001674")
        worker = self.app.test_client()
        self.post(worker, "/login", {"email": "review.worker@example.com", "password": PASSWORD})
        self.post(worker, f"/bookings/{booking_id}/accept", {}, token_path="/dashboard")
        self.post(c, f"/bookings/{booking_id}/complete", {}, token_path="/dashboard")
        review_page = self.html(c, f"/bookings/{booking_id}/review")
        self.assertIn("Submit Rating &amp; Review", review_page)
        for r in range(1, 6):
            self.assertIn(f'id="rating-{r}" name="rating" value="{r}"', review_page)
        self.assertIn('<textarea id="review_text" name="review_text"', review_page)
        self.assertIn("Rate &amp; review", self.html(c, "/bookings"))
        self.post(c, f"/bookings/{booking_id}/review", {"rating": "5", "review_text": "Lovely logo."}, token_path=f"/bookings/{booking_id}/review")
        self.assertEqual(self.repo.get_review(booking_id)["rating"], 5)
        self.assertIn("You rated this booking", self.html(c, f"/bookings/{booking_id}/review"))
        self.assertEqual(worker.get(f"/bookings/{booking_id}/review").status_code, 403)
        other = self.app.test_client()
        self.register(other, self.client_form("other.client@example.com"))
        self.assertEqual(other.get(f"/bookings/{booking_id}/review").status_code, 404)

    def test_dashboards_show_documented_sections(self):
        c = self.app.test_client()
        self.register(c, self.client_form("dash.client@example.com"))
        self.create_request(c)
        page = self.html(c, "/dashboard")
        for text in ("Post a new request", "Active service requests", "Recent requests", "Service history",
                     "Top matches for your latest request", "Booking status summary"):
            self.assertIn(text, page)
        for nav in ("My Requests", "Service History", "Log out"):
            self.assertIn(nav, page)
        self.repo.create_user("dash.worker@example.com", hash_password(PASSWORD), "worker", "W", worker_id="W000001")
        w = self.app.test_client()
        self.post(w, "/login", {"email": "dash.worker@example.com", "password": PASSWORD})
        page = self.html(w, "/dashboard")
        for text in ("Bookings assigned to you", "Reviews received", "Completed jobs", "Specialisation", "Skills"):
            self.assertIn(text, page)

    # ---- security / accessibility invariants -------------------------------------------------------
    def test_security_headers_and_no_inline_code_on_redesigned_pages(self):
        c = self.app.test_client()
        pages = ["/", "/login", "/register", "/demo"]
        for path in pages:
            response = c.get(path)
            html = response.get_data(as_text=True)
            self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"], path)
            self.assertNotIn("unsafe-inline", response.headers["Content-Security-Policy"])
            self.assertNotIn(' style="', html, path)
            self.assertNotRegex(html, r"<script(?![^>]*\bsrc=)[^>]*>", path)
            self.assertNotRegex(html, r"\son[a-z]+=", path)
            self.assertIn('lang="en"', html)
            self.assertIn('class="skip-link"', html)
        self.assertIn(":focus-visible", CSS)


if __name__ == "__main__":
    unittest.main()
