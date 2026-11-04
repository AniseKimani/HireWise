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

    def test_booking_table_markup_supports_phone_cards_without_changing_forms(self):
        """Mobile responsiveness, Milestone 2: the shared booking table carries the hooks its phone
        card layout needs (cell labels, explicit ARIA table roles) while each context still has exactly
        one form per allowed action, posting CSRF + return_to to the unchanged endpoint."""
        c = self.app.test_client()
        self.register(c, self.client_form("cards.client@example.com"))
        request_id = self.create_request(c)
        self.post(c, f"/client/requests/{request_id}/book", {"worker_id": "W001674"}, token_path="/dashboard")
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.repo.create_user("cards.worker@example.com", hash_password(PASSWORD), "worker", "W", worker_id="W001674")
        worker = self.app.test_client()
        self.post(worker, "/login", {"email": "cards.worker@example.com", "password": PASSWORD})

        contexts = [(c, "/dashboard", "/dashboard", ["cancel"]),
                    (c, "/bookings", "/bookings", ["cancel"]),
                    (c, f"/client/requests/{request_id}", f"/client/requests/{request_id}", ["cancel"]),
                    (worker, "/dashboard", "/dashboard", ["accept", "decline"]),
                    (worker, "/bookings", "/bookings", ["accept", "decline"])]
        for user, path, return_to, expected_actions in contexts:
            page = self.html(user, path)
            tables = re.findall(r'<table class="data-table booking-table" role="table">(.*?)</table>', page, re.S)
            self.assertEqual(len(tables), 1, path)
            table = tables[0]
            self.assertEqual(table.count('role="columnheader"'), 6, path)
            for label in ("Booking", "Request", "Worker", "Status", "Review", "Actions"):
                self.assertIn(f'data-label="{label}"', table, path)
            row = re.search(r'<tr role="row" class="booking-row">(.*?)</tr>', table, re.S).group(1)
            self.assertEqual(row.count('role="cell"'), 6, path)
            forms = re.findall(r"<form (.*?)</form>", table, re.S)
            actions = [re.search(r'action="/bookings/(\d+)/(\w+)"', f).groups() for f in forms]
            self.assertEqual(actions, [(str(booking_id), a) for a in expected_actions], path)  # one form per action, no duplicates
            for form in forms:
                self.assertIn('method="post"', form)
                self.assertEqual(len(TOKEN.findall(form)), 1, path)
                self.assertIn(f'<input type="hidden" name="return_to" value="{return_to}">', form, path)
                self.assertRegex(form, rf'<button type="submit" class="button[^"]*">\w+<span class="visually-hidden"> booking #{booking_id}</span></button>')
        for selector in (".booking-table td::before", "content: attr(data-label)", ".booking-row__actions .button"):
            self.assertIn(selector, CSS)

        empty = self.app.test_client()
        self.register(empty, self.client_form("cards.empty@example.com"))
        page = self.html(empty, "/bookings")
        self.assertIn('<p class="empty-state">No bookings yet.</p>', page)
        self.assertNotIn("booking-table", page)

    def test_review_and_profile_pages_put_primary_content_first_in_source_order(self):
        """Mobile responsiveness, Milestone 3: on the review page the form follows the worker's header in
        the source (so phones and keyboard / screen-reader users reach it before the long details), while
        the informational "About this profile" stays last; the review form itself is unchanged."""
        c = self.app.test_client()
        self.register(c, self.client_form("order.client@example.com"))
        request_id = self.create_request(c)
        self.assertIn('class="grid-2 request-overview"', self.html(c, f"/client/requests/{request_id}"))
        profile = self.html(c, f"/workers/W001674?request={request_id}")
        self.assertLess(profile.index('id="worker-heading"'), profile.index('id="skills-heading"'))
        self.assertLess(profile.index('id="skills-heading"'), profile.index("About this profile"))

        self.post(c, f"/client/requests/{request_id}/book", {"worker_id": "W001674"}, token_path="/dashboard")
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.repo.create_user("order.worker@example.com", hash_password(PASSWORD), "worker", "W", worker_id="W001674")
        worker = self.app.test_client()
        self.post(worker, "/login", {"email": "order.worker@example.com", "password": PASSWORD})
        self.post(worker, f"/bookings/{booking_id}/accept", {}, token_path="/dashboard")
        self.post(c, f"/bookings/{booking_id}/complete", {}, token_path="/dashboard")
        page = self.html(c, f"/bookings/{booking_id}/review")
        order = [page.index(marker) for marker in ('id="worker-heading"', "Submit Rating &amp; Review", 'id="skills-heading"')]
        self.assertEqual(order, sorted(order))
        form = re.search(r'<form method="post" action="/bookings/%d/review" class="form-grid">(.*?)</form>' % booking_id, page, re.S).group(1)
        self.assertEqual(len(TOKEN.findall(form)), 1)
        self.assertEqual(re.findall(r'name="(\w+)"', form), ["csrf_token"] + ["rating"] * 5 + ["review_text"])

        self.assertIn(".request-overview { grid-template-columns: minmax(0, 1fr); }", CSS)
        hover_rule = ".star-rating__options label:hover, .star-rating__options label:hover ~ label { color: var(--star); }"
        self.assertEqual(CSS.count(hover_rule), 1)
        self.assertLess(CSS.index("@media (hover: hover) and (pointer: fine)"), CSS.index(hover_rule))

    def test_forms_associate_errors_and_keep_their_fields(self):
        """Mobile responsiveness, Milestone 4: server errors are linked to their fields, money fields ask
        for a decimal keypad, the submitted field names are unchanged, and a corrected password
        confirmation can no longer stay blocked by a stale mismatch message."""
        long_skill = "Supercalifragilisticexpialidociouslylongunknownskillname"
        page = self.register(self.app.test_client(), self.worker_form("errors.worker@example.com", rate="", skills=long_skill)).get_data(as_text=True)
        for name in ("rate", "skills"):
            self.assertRegex(page, rf'id="{name}"[^>]*aria-describedby="[^"]*\b{name}-error\b[^"]*"[^>]*aria-invalid="true"', name)
            self.assertIn(f'id="{name}-error"', page)
        for name in ("rate", "experience_years"):
            self.assertRegex(page, rf'id="{name}" name="{name}" type="number" inputmode="decimal"')
        self.assertEqual(sorted(set(re.findall(r'<(?:input|select)[^>]*\sname="(\w+)"', page))),
                         sorted({"csrf_token", "role", "email", "password", "confirm_password", "display_name", "client_country",
                                 "worker_country", "primary_category", "experience_level", "experience_years", "rate", "skills"}))

        c = self.app.test_client()
        self.register(c, self.client_form("errors.client@example.com"))
        r = self.post(c, "/client/requests/new", {"title": "x", "category": "", "skills": long_skill, "pay_type": "hourly",
                                                  "hourly_low": "50", "hourly_high": "10"}, token_path="/client/requests/new")
        self.assertEqual(r.status_code, 400)
        page = r.get_data(as_text=True)
        for name in ("title", "category", "skills", "hourly_high"):
            self.assertRegex(page, rf'id="{name}"[^>]*aria-describedby="{name}-error" aria-invalid="true"', name)
        for name in ("budget", "hourly_low", "hourly_high"):
            self.assertRegex(page, rf'id="{name}" name="{name}" type="number" inputmode="decimal"')
        form = re.search(r'<form method="post" action="/client/requests/new" class="form-grid" novalidate>(.*?)</form>', page, re.S).group(1)
        self.assertEqual(re.findall(r'<(?:input|select)[^>]*\sname="(\w+)"', form),
                         ["csrf_token", "title", "category", "skills", "pay_type", "budget", "hourly_low", "hourly_high", "location_requirement"])

        check = REGISTER_JS.index('confirm.setCustomValidity(first.value === confirm.value')
        self.assertLess(check, REGISTER_JS.index("for (var i = 0; i < fields.length; i++)"))
        self.assertIn(".has-js .js-only[hidden] { display: none; }", CSS)
        # JavaScript-only controls keep their own layout (no "display: revert", which flattened the stepper
        # and left the account step's actions uncentred).
        self.assertIn("html:not(.has-js) .js-only { display: none; }", CSS)
        self.assertNotIn(".js-only { display: revert; }", CSS)

    def test_landing_and_dashboard_layout_hooks(self):
        """Mobile responsiveness, Milestone 5: the hero illustration rule no longer enlarges the badge icons,
        side-by-side landing cards line up, and the page-level calls to action keep their destinations."""
        self.assertIn(".hero__visual > svg { width: 70%; height: auto; }", CSS)
        self.assertNotIn(".hero__visual svg {", CSS)
        self.assertIn(".features > .card + .card, .audience > .card + .card { margin-top: 0; }", CSS)
        landing = self.html(self.app.test_client(), "/")
        badges = re.search(r'<ul class="hero__badges">(.*?)</ul>', landing, re.S).group(1)
        self.assertEqual(badges.count('<li><svg class="icon"'), 3)  # the three hero badges keep their small icons
        c = self.app.test_client()
        self.register(c, self.client_form("hooks.client@example.com"))
        for path in ("/dashboard", "/client/requests"):
            self.assertRegex(self.html(c, path), r'<a class="button[^"]* page-head__cta" href="/client/requests/new">')
        self.repo.create_user("hooks.worker@example.com", hash_password(PASSWORD), "worker", "W", worker_id="W000001")
        w = self.app.test_client()
        self.post(w, "/login", {"email": "hooks.worker@example.com", "password": PASSWORD})
        self.assertIn('<a class="button button--secondary page-head__cta" href="/profile">', self.html(w, "/dashboard"))

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
