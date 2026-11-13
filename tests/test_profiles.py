"""Milestone 4 tests: profile editing, self-reported worker credentials and initials avatars.

- Pure rules (no app, no database): initials, the avatar palette (size, CSS classes and white-text contrast >= 4.5:1),
  URL validation, credential validation and limits, and the edit permissions for every kind of account.
- ProfileBehaviour (mixin) runs through the real app against BOTH repositories: InMemoryProfileTests (default) and
  PostgresProfileTests (OPT-IN: HIREWISE_RUN_POSTGRES_TESTS=1, on the isolated POSTGRES_TEST_DB only).
- The v5 -> v6 migration is tested in tests/test_account_security.py (MigrationTests, opt-in), next to the others.
"""
import json
import os
import re
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.app import create_app
from src.app.config import BaseConfig
from src.app.marketplace import rules
from src.app.marketplace.avatars import initials, palette_index
from src.app.marketplace.runtime import RuntimeRequest
from src.app.marketplace.security import hash_password
from src.app.marketplace.validation import validate_credential, validate_profile_edit, validate_url
from src.db.schema import CREDENTIAL_KINDS

REPO_ROOT = Path(__file__).resolve().parents[1]
CSS = (REPO_ROOT / "src" / "app" / "static" / "css" / "style.css").read_text(encoding="utf-8")
HAVE_DATA = (REPO_ROOT / "data" / "processed" / "synthetic").exists() and \
    (REPO_ROOT / "results" / "day8" / "selected_hybrid_config.json").exists()
PASSWORD = "correct horse battery"
TOKEN = re.compile(r'name="csrf_token" value="([^"]+)"')
LIMITS = BaseConfig.CREDENTIAL_LIMITS
THIS_YEAR = datetime.now(timezone.utc).year


def contrast_with_white(hex_colour: str) -> float:
    def channel(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    luminance = 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    return 1.05 / (luminance + 0.05)


# ---------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------
class TestInitials(unittest.TestCase):
    def test_initials(self):
        cases = {"Ada Lovelace": "AL", "ada lovelace": "AL", "  Ada   Byron   Lovelace  ": "AL", "Cher": "C",
                 "Mary-Jane Watson": "MW", "(Ann) O'Brien": "AO", "Élodie Ünal": "ÉÜ", "zoë": "Z",
                 "李小龙": "李", "Ivan Петров": "IП", "سارة أحمد": "سأ", "straße": "S", "42 Lisa": "L",
                 "Ann 😀": "A", "😀 🎉": "", "123": "", "": "", None: "", "   ": "", "émile dupont": "ÉD"}
        for name, expected in cases.items():
            self.assertEqual(initials(name), expected, repr(name))

    def test_palette_index_is_stable_and_in_range(self):
        size = BaseConfig.AVATAR_PALETTE_SIZE
        slots = [palette_index(i, size) for i in range(1, 600)]
        self.assertTrue(all(1 <= s <= size for s in slots))
        self.assertEqual(set(slots), set(range(1, size + 1)))  # every colour is used
        self.assertEqual(palette_index(42, size), palette_index(42, size))
        self.assertEqual(palette_index(7, 6), 1 + int.from_bytes(__import__("hashlib").sha256(b"7").digest()[:8], "big") % 6)
        with self.assertRaises(ValueError):
            palette_index(1, 0)


class TestAvatarPalette(unittest.TestCase):
    def test_every_palette_colour_has_contrast_for_white_text(self):
        colours = dict(re.findall(r"--avatar-(\d+):\s*(#[0-9a-fA-F]{6})\s*;", CSS))
        self.assertEqual(sorted(int(k) for k in colours), list(range(1, BaseConfig.AVATAR_PALETTE_SIZE + 1)))
        self.assertIn("--avatar-text: #ffffff", CSS)
        for slot, colour in colours.items():
            self.assertGreaterEqual(contrast_with_white(colour), 4.5, f"--avatar-{slot} {colour}")
            self.assertIn(f".avatar-initials--{slot} {{ background: var(--avatar-{slot}); }}", CSS)

    def test_contrast_helper(self):
        self.assertAlmostEqual(contrast_with_white("#000000"), 21.0, places=1)
        self.assertAlmostEqual(contrast_with_white("#ffffff"), 1.0, places=2)


class TestUrlValidation(unittest.TestCase):
    def test_accepted(self):
        for url in ("https://example.com", "http://example.com/a?b=1#c", "HTTPS://Example.com/Path",
                    "https://sub.example.co.uk:8443/x", "  https://example.com/padded  "):
            value, error = validate_url(url, LIMITS["url"])
            self.assertIsNone(error, url)
            self.assertEqual(value, url.strip())
        self.assertEqual(validate_url("", LIMITS["url"]), (None, None))

    def test_refused(self):
        for url in ("javascript:alert(1)", " JavaScript:alert(1)", "data:text/html,<b>x</b>", "ftp://example.com",
                    "mailto:a@example.com", "example.com", "//example.com", "http://", "https:///path", "http:example.com",
                    "https://user:pw@example.com", "https://user@example.com", "https://exa mple.com",
                    "https://example.com/\x00", "https://example.com/a\nb", "vbscript:msgbox", "file:///etc/passwd",
                    "https://" + "a" * 600 + ".com", "https://[::1"):
            value, error = validate_url(url, LIMITS["url"])
            self.assertIsNone(value, url)
            self.assertTrue(error, url)


class TestCredentialValidation(unittest.TestCase):
    def valid(self, **overrides):
        form = {"kind": "certification", "title": "  AWS Cloud Practitioner  ", "organisation": " Amazon ", "year": str(THIS_YEAR),
                "url": "https://example.com/cert", "description": " Passed with distinction. ", **overrides}
        return validate_credential(form, kinds=CREDENTIAL_KINDS, limits=LIMITS, current_year=THIS_YEAR)

    def test_valid_and_trimmed(self):
        data, errors = self.valid()
        self.assertEqual(errors, {})
        self.assertEqual((data["title"], data["organisation"], data["description"], data["year"]),
                         ("AWS Cloud Practitioner", "Amazon", "Passed with distinction.", THIS_YEAR))
        data, errors = self.valid(organisation="  ", year="", url="", description="")
        self.assertEqual((errors, data["organisation"], data["year"], data["url"], data["description"]), ({}, None, None, None, None))

    def test_limits(self):
        cases = [({"kind": "degree"}, "kind"), ({"title": "  "}, "title"), ({"title": "t" * (LIMITS["title"] + 1)}, "title"),
                 ({"organisation": "o" * (LIMITS["organisation"] + 1)}, "organisation"),
                 ({"description": "d" * (LIMITS["description"] + 1)}, "description"),
                 ({"year": str(THIS_YEAR + 2)}, "year"), ({"year": str(LIMITS["year_min"] - 1)}, "year"),
                 ({"year": "20x5"}, "year"), ({"year": "-2000"}, "year"), ({"url": "javascript:alert(1)"}, "url")]
        for overrides, field in cases:
            _, errors = self.valid(**overrides)
            self.assertIn(field, errors, overrides)
        for ok in ({"title": "t" * LIMITS["title"]}, {"year": str(THIS_YEAR + 1)}, {"year": str(LIMITS["year_min"])},
                   {"organisation": "o" * LIMITS["organisation"]}, {"description": "d" * LIMITS["description"]}):
            self.assertEqual(self.valid(**ok)[1], {}, ok)

    def test_configured_limits_never_exceed_the_database(self):
        from src.db import schema
        self.assertLessEqual(BaseConfig.PROFILE_BIO_MAX_LENGTH, schema.BIO_MAX_LENGTH)
        self.assertLessEqual(LIMITS["title"], schema.CREDENTIAL_TITLE_MAX_LENGTH)
        self.assertLessEqual(LIMITS["url"], schema.CREDENTIAL_URL_MAX_LENGTH)
        self.assertEqual(BaseConfig.CREDENTIALS_PER_KIND, 10)


class TestEditPermissions(unittest.TestCase):
    def test_fields_per_account(self):
        client = {"role": "client", "worker_id": None}
        linked = {"role": "worker", "worker_id": "W000001"}
        worker = {"role": "worker", "worker_id": None}
        self.assertEqual(rules.profile_edit_fields(client, False), {"display_name", "client_country"})
        self.assertEqual(rules.profile_edit_fields(linked, False), {"display_name"})
        self.assertEqual(rules.profile_edit_fields(linked, True), {"display_name"})  # a link always wins
        self.assertEqual(rules.profile_edit_fields(worker, True),
                         {"display_name", "worker_country", "worker_profile", "credentials"})
        self.assertEqual(rules.profile_edit_fields(worker, False), {"display_name", "worker_country"})  # demo operator

    def test_edit_validation_uses_registration_rules_and_messages(self):
        vocab, categories = {"python": "Python"}, {"Logo Design"}
        fields = rules.profile_edit_fields({"role": "worker", "worker_id": None}, True)
        data, errors = validate_profile_edit({"display_name": "", "worker_country": "", "primary_category": "x",
                                              "experience_level": "", "experience_years": "61", "rate": "-1",
                                              "skills": "Cobol", "bio": "b" * 1001}, fields, categories, vocab, ["Canada"], 1000)
        self.assertEqual(errors, {"display_name": "Enter a display name (up to 80 characters).",
                                  "primary_category": "Choose a category from the list.",
                                  "experience_level": "Choose an experience level.",
                                  "worker_country": "Workers must choose a country.",
                                  "experience_years": "Years of experience is out of range.",
                                  "rate": "Rate is out of range.",
                                  "skills": "Unknown skill(s): Cobol. Use names from the skill list.",
                                  "bio": "Your bio is limited to 1,000 characters."})
        data, errors = validate_profile_edit({"display_name": " Ann ", "worker_country": ""},
                                             frozenset({"display_name", "worker_country"}), categories, vocab, ["Canada"], 1000)
        self.assertEqual((data, errors), ({"display_name": "Ann", "country": None}, {}))  # operator: location optional


# ---------------------------------------------------------------------------
# Through the application, on both repositories
# ---------------------------------------------------------------------------
_SHARED = {}
WORKER_PROFILE = {"primary_category": "Logo Design", "experience_level": "intermediate", "experience_years": 4.0,
                  "rate": 25.0, "country": "India", "skills": ["Logo Design"]}


def shared():
    if not _SHARED:
        from src.app.marketplace.runtime import RuntimeRecommender
        from src.app.services import RecommendationService
        from src.db.repositories import InMemoryRepository

        service = RecommendationService.build(repository=InMemoryRepository.from_artifacts())
        _SHARED["service"], _SHARED["runtime"] = service, RuntimeRecommender.build(service)
    return _SHARED["service"], _SHARED["runtime"]


class ProfileBehaviour:
    """Mixin; subclasses provide make_repository()."""

    def setUp(self):
        self.service, self.runtime = shared()
        self.repo = self.make_repository()
        self.app = create_app("testing", service=self.service, marketplace_repository=self.repo,
                              runtime_recommender=self.runtime)

    def account(self, email, role="client", worker_id=None, worker_profile=None, country=None, name=None):
        user_id = self.repo.create_user(email, hash_password(PASSWORD), role, name or email.split("@")[0].title(),
                                        country, worker_id=worker_id, worker_profile=worker_profile)
        c = self.app.test_client()
        c.post("/login", data={"email": email, "password": PASSWORD,
                               "csrf_token": TOKEN.search(c.get("/login").get_data(as_text=True)).group(1)})
        return c, user_id

    def self_registered(self, email="worker@example.com"):
        return self.account(email, "worker", worker_profile=dict(WORKER_PROFILE), country="India")

    def post(self, client, path, data, token_path="/profile/edit"):
        return client.post(path, data={**data, "csrf_token": TOKEN.search(client.get(token_path).get_data(as_text=True)).group(1)})

    def credential(self, client, **overrides):
        form = {"kind": "certification", "title": "AWS Cloud Practitioner", "organisation": "Amazon", "year": "2024",
                "url": "https://example.com/cert", "description": "", **overrides}
        return self.post(client, "/profile/credentials/new", form, "/profile/credentials/new")

    # ---- editing -----------------------------------------------------------------------------------------
    def test_client_edits_name_and_location(self):
        c, uid = self.account("client@example.com", name="Ann Smith")
        page = c.get("/profile/edit").get_data(as_text=True)
        self.assertIn('name="client_country"', page)
        self.assertNotIn('name="primary_category"', page)
        self.assertNotIn('name="email"', page + "")  # email and password are not part of this page
        self.assertNotIn('type="password"', page)
        r = self.post(c, "/profile/edit", {"display_name": "Zoë Ndlovu", "client_country": "Canada"})
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/profile/edit"))
        user = self.repo.get_user(uid)
        self.assertEqual((user["display_name"], user["country"]), ("Zoë Ndlovu", "Canada"))
        self.assertIn(">ZN</span>", c.get("/dashboard").get_data(as_text=True))  # initials follow the new name
        bad = self.post(c, "/profile/edit", {"display_name": "", "client_country": "Atlantis"})
        html = bad.get_data(as_text=True)
        self.assertEqual(bad.status_code, 400)
        self.assertIn("Enter a display name (up to 80 characters).", html)
        self.assertIn("Choose a country from the list.", html)
        self.assertEqual(self.repo.get_user(uid)["display_name"], "Zoë Ndlovu")  # nothing saved

    def test_self_registered_worker_edits_everything_with_registration_rules(self):
        c, uid = self.self_registered()
        r = self.post(c, "/profile/edit", {"display_name": "Ravi Kumar", "worker_country": "Canada",
                                           "primary_category": "Logo Design", "experience_level": "expert",
                                           "experience_years": "7.5", "rate": "40", "skills": "logo design, adobe illustrator",
                                           "bio": "  I design logos for small businesses.  "})
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/profile"))
        profile = self.repo.get_runtime_worker_profile(uid)
        self.assertEqual((profile["country"], profile["experience_level"], profile["experience_years"], profile["rate"],
                          profile["skills"], profile["bio"]),
                         ("Canada", "expert", 7.5, 40.0, ["Adobe Illustrator", "Logo Design"], "I design logos for small businesses."))
        self.assertEqual(self.repo.get_user(uid)["country"], "Canada")  # both copies of the location
        page = c.get("/profile").get_data(as_text=True)
        self.assertIn("I design logos for small businesses.", page)
        self.assertIn("Self-reported", page)
        bad = self.post(c, "/profile/edit", {"display_name": "Ravi Kumar", "worker_country": "Canada",
                                             "primary_category": "Logo Design", "experience_level": "expert",
                                             "experience_years": "7.5", "rate": "40", "skills": "Underwater basket weaving",
                                             "bio": "kept"})
        html = bad.get_data(as_text=True)
        self.assertEqual(bad.status_code, 400)
        self.assertIn("Unknown skill(s): Underwater basket weaving. Use names from the skill list.", html)
        self.assertIn('value="Underwater basket weaving"', html)  # entered values kept
        self.assertIn(">kept</textarea>", html)
        self.assertEqual(self.repo.get_runtime_worker_profile(uid)["skills"], ["Adobe Illustrator", "Logo Design"])

    def test_catalogue_linked_account_edits_display_name_only(self):
        c, uid = self.account("linked@example.com", "worker", worker_id="W000001", country="India")
        page = c.get("/profile/edit").get_data(as_text=True)
        self.assertIn("This profile comes from the research dataset and can&#39;t be edited", page)
        self.assertNotIn('name="worker_country"', page)
        self.assertIn("This profile comes from the research dataset and can&#39;t be edited",
                      c.get("/profile").get_data(as_text=True))
        self.assertEqual(self.post(c, "/profile/edit", {"display_name": "New Name"}).status_code, 302)
        self.assertEqual(self.repo.get_user(uid)["display_name"], "New Name")
        for extra in ({"worker_country": "Canada"}, {"primary_category": "Logo Design"}, {"bio": "x"}, {"skills": "Python"},
                      {"client_country": "Canada"}):
            self.assertEqual(self.post(c, "/profile/edit", {"display_name": "Other", **extra}).status_code, 403, extra)
        self.assertEqual((self.repo.get_user(uid)["display_name"], self.repo.get_user(uid)["country"]), ("New Name", "India"))
        self.assertEqual(c.get("/profile/credentials/new").status_code, 403)
        self.assertEqual(self.post(c, "/profile/credentials/new", {"kind": "portfolio", "title": "x"}).status_code, 403)
        # the catalogue profile keeps the generic icon; the header menu shows the initials
        profile_page = c.get("/profile").get_data(as_text=True)
        self.assertIn('class="avatar" aria-hidden="true"', profile_page)
        self.assertIn('account-menu__avatar" aria-hidden="true">NN</span>', profile_page)

    def test_worker_account_without_profile_edits_name_and_location(self):
        c, uid = self.account("operator@example.com", "worker")  # the demo operator's shape
        page = c.get("/profile/edit").get_data(as_text=True)
        self.assertIn('name="worker_country"', page)
        self.assertNotIn('name="primary_category"', page)
        self.assertEqual(self.post(c, "/profile/edit", {"display_name": "Demo Desk", "worker_country": ""}).status_code, 302)
        self.assertEqual((self.repo.get_user(uid)["display_name"], self.repo.get_user(uid)["country"]), ("Demo Desk", None))
        self.assertEqual(self.post(c, "/profile/edit", {"display_name": "X", "primary_category": "Logo Design"}).status_code, 403)
        self.assertEqual(c.get("/profile/credentials/new").status_code, 403)

    def test_csrf_and_login_required(self):
        c, uid = self.account("client@example.com", name="Ann")
        self.assertEqual(c.post("/profile/edit", data={"display_name": "No Token"}).status_code, 400)
        self.assertEqual(self.repo.get_user(uid)["display_name"], "Ann")
        anonymous = self.app.test_client()
        self.assertEqual(anonymous.get("/profile/edit").status_code, 302)
        self.assertEqual(anonymous.get("/profile/credentials/new").status_code, 302)
        self.assertEqual(c.get("/profile/credentials/new").status_code, 403)  # clients have no credentials

    # ---- credentials -------------------------------------------------------------------------------------
    def test_credentials_add_edit_delete_and_ownership(self):
        c, uid = self.self_registered()
        self.assertEqual(self.credential(c).status_code, 302)
        [cred] = self.repo.list_credentials(uid)
        page = c.get("/profile").get_data(as_text=True)
        self.assertIn("AWS Cloud Practitioner", page)
        self.assertIn('href="https://example.com/cert" rel="noopener noreferrer nofollow"', page)
        self.assertGreaterEqual(page.count("Self-reported"), 4)  # the section and each of the three kinds
        r = self.post(c, f"/profile/credentials/{cred['credential_id']}/edit",
                      {"kind": "experience", "title": "Freelance designer", "organisation": "", "year": "", "url": "",
                       "description": "Logos and posters."}, f"/profile/credentials/{cred['credential_id']}/edit")
        self.assertEqual(r.status_code, 302)
        updated = self.repo.get_credential(uid, cred["credential_id"])
        self.assertEqual((updated["kind"], updated["title"], updated["organisation"], updated["url"]),
                         ("experience", "Freelance designer", None, None))
        bad = self.credential(c, url="javascript:alert(1)", title="Bad link")
        self.assertEqual(bad.status_code, 400)
        self.assertIn("Enter a valid link starting with http:// or https://.", bad.get_data(as_text=True))
        self.assertIn('value="Bad link"', bad.get_data(as_text=True))
        # another worker cannot see, edit or delete it
        other, _ = self.self_registered("other@example.com")
        cid = cred["credential_id"]
        self.assertEqual(other.get(f"/profile/credentials/{cid}/edit").status_code, 404)
        self.assertEqual(self.post(other, f"/profile/credentials/{cid}/edit", {"kind": "portfolio", "title": "Stolen"},
                                   "/profile/edit").status_code, 404)
        self.assertEqual(self.post(other, f"/profile/credentials/{cid}/delete", {}, "/profile/edit").status_code, 404)
        self.assertEqual(self.post(other, "/profile/credentials/999999/delete", {}, "/profile/edit").status_code, 404)
        self.assertNotIn("Freelance designer", other.get("/profile").get_data(as_text=True))
        self.assertEqual(self.repo.get_credential(uid, cid)["title"], "Freelance designer")
        self.assertEqual(c.post(f"/profile/credentials/{cid}/delete").status_code, 400)  # CSRF required
        self.assertEqual(self.post(c, f"/profile/credentials/{cid}/delete", {}, "/profile").status_code, 302)
        self.assertEqual(self.repo.list_credentials(uid), [])

    def test_per_kind_limit(self):
        self.app.config["CREDENTIALS_PER_KIND"] = 2
        c, uid = self.self_registered()
        for i in range(2):
            self.assertEqual(self.credential(c, title=f"Cert {i}").status_code, 302)
        third = self.credential(c, title="Cert 3")
        self.assertEqual(third.status_code, 400)
        self.assertIn("You can list at most 2 certification entries", third.get_data(as_text=True))
        self.assertEqual(self.credential(c, kind="portfolio", title="Site").status_code, 302)  # per kind
        portfolio = next(x for x in self.repo.list_credentials(uid) if x["kind"] == "portfolio")
        moved = self.post(c, f"/profile/credentials/{portfolio['credential_id']}/edit",
                          {"kind": "certification", "title": "Site"}, f"/profile/credentials/{portfolio['credential_id']}/edit")
        self.assertEqual(moved.status_code, 400)  # changing kind cannot exceed the limit either
        self.assertEqual(len([x for x in self.repo.list_credentials(uid) if x["kind"] == "certification"]), 2)

    # ---- avatars, CSP, recommender ----------------------------------------------------------------------
    def test_avatars_and_strict_csp_on_new_pages(self):
        c, uid = self.self_registered()
        self.credential(c)
        slot = palette_index(uid, BaseConfig.AVATAR_PALETTE_SIZE)
        for path in ("/profile/edit", "/profile", "/dashboard", "/profile/credentials/new",
                     f"/profile/credentials/{self.repo.list_credentials(uid)[0]['credential_id']}/edit"):
            r = c.get(path)
            html = r.get_data(as_text=True)
            self.assertEqual(r.status_code, 200, path)
            self.assertIn(f"avatar-initials--{slot}", html, path)
            self.assertIn("script-src 'self'", r.headers["Content-Security-Policy"])
            self.assertNotIn(' style="', html, path)
            self.assertNotRegex(html, r"<script(?![^>]*\bsrc=)[^>]*>", path)
            self.assertNotRegex(html, r"\son[a-z]+=", path)
        nameless, _ = self.account("nameless@example.com", name="123")
        self.assertIn("avatar-initials--icon", nameless.get("/dashboard").get_data(as_text=True))  # icon fallback

    def test_catalogue_worker_cards_keep_the_icon_and_edits_never_reach_the_recommender(self):
        service, runtime = shared()
        link = service.data.requests.loc[service.data.requests["split"] == "test", "link"].iloc[0]
        before = json.dumps(service.recommend(link, 10), sort_keys=True)
        rr = RuntimeRequest(10_000, 1, "Logo Design", ("Logo Design", "Adobe Illustrator"), 0.5, None, None)
        runtime_before = json.dumps(runtime.recommend(rr, 10), sort_keys=True, default=str)
        worker, _ = self.self_registered()
        self.credential(worker)
        self.post(worker, "/profile/edit", {"display_name": "Changed", "worker_country": "Canada", "primary_category": "Logo Design",
                                            "experience_level": "expert", "experience_years": "9", "rate": "99",
                                            "skills": "Logo Design, Python", "bio": "Bio"})
        linked, _ = self.account("linked@example.com", "worker", worker_id="W000001")
        self.post(linked, "/profile/edit", {"display_name": "Renamed"})
        self.assertEqual(json.dumps(service.recommend(link, 10), sort_keys=True), before)
        self.assertEqual(json.dumps(runtime.recommend(rr, 10), sort_keys=True, default=str), runtime_before)
        client, _ = self.account("client@example.com")
        r = self.post(client, "/client/requests/new", {"title": "Logo for a bakery", "category": "Logo Design",
                                                      "skills": "Logo Design", "pay_type": "fixed", "budget": "150"},
                      "/client/requests/new")
        page = client.get(r.headers["Location"]).get_data(as_text=True)
        self.assertIn('<div class="avatar" aria-hidden="true">', page)  # catalogue workers: icon, not initials
        self.assertEqual(page.count("avatar-initials--"), 1)            # only the client's own header avatar


@unittest.skipUnless(HAVE_DATA, "processed data / Day 8 results absent")
class InMemoryProfileTests(ProfileBehaviour, unittest.TestCase):
    def make_repository(self):
        from src.db.marketplace import InMemoryMarketplaceRepository

        if "memory_reference" not in _SHARED:
            _SHARED["memory_reference"] = InMemoryMarketplaceRepository.from_artifacts()
        ref = _SHARED["memory_reference"]
        return InMemoryMarketplaceRepository(ref.categories(), ref.skills(), ref._clients, ref._workers)


@unittest.skipUnless(HAVE_DATA and os.environ.get("HIREWISE_RUN_POSTGRES_TESTS") == "1",
                     "opt-in live PostgreSQL test: set HIREWISE_RUN_POSTGRES_TESTS=1")
class PostgresProfileTests(ProfileBehaviour, unittest.TestCase):
    """The same behaviour on POSTGRES_TEST_DB only (never POSTGRES_DB)."""

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

        with self.engine.begin() as conn:
            for name in reversed(RUNTIME_TABLES):
                conn.execute(metadata.tables[name].delete())
        return PostgresMarketplaceRepository(self.engine)

    def test_database_enforces_credential_and_bio_constraints(self):
        from sqlalchemy import insert, update
        from sqlalchemy.exc import IntegrityError
        from src.db.schema import runtime_worker_profiles, worker_credentials

        _, worker = self.self_registered()
        client_id = self.repo.create_user("db.client@example.com", "h", "client", "C")
        good = dict(user_id=worker, kind="portfolio", title="Site")
        bad = [dict(good, kind="degree"), dict(good, title=""), dict(good, title="t" * 121), dict(good, organisation="o" * 121),
               dict(good, year=1949), dict(good, year=2101), dict(good, url="javascript:alert(1)"),
               dict(good, url="https://" + "a" * 500), dict(good, description="d" * 501),
               dict(good, user_id=client_id)]  # a client has no runtime worker profile
        for row in bad:
            with self.assertRaises(IntegrityError, msg=row):
                with self.engine.begin() as conn:
                    conn.execute(insert(worker_credentials).values(**row))
        with self.engine.begin() as conn:
            conn.execute(insert(worker_credentials).values(**good))
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as conn:
                conn.execute(update(runtime_worker_profiles).where(runtime_worker_profiles.c.user_id == worker)
                             .values(bio="b" * 1001))


if __name__ == "__main__":
    unittest.main()
