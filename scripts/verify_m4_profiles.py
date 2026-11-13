"""Independent verification of pre-deployment Milestone 4 (profile editing, self-reported credentials, initials avatars).

Writes ONLY to POSTGRES_TEST_DB. The main database (POSTGRES_DB) is only read, and is checked to be unchanged.

Independent of the implementation: it does not import src/app/marketplace/avatars.py, validation.py, rules.py,
profile_views.py or the repository's profile/credential methods. Accounts and profiles are created with direct SQL
(Werkzeug password hashes), the app is driven only through HTTP (Flask test client), and every outcome is checked with
direct SQL. Initials, the palette slot and the colour contrast are recomputed here from scratch.

Checks:
  * schema v5 -> v6 through scripts/migrate_schema_v6.py --test-db: bio column (nullable TEXT + length CHECK),
    worker_credentials with its CHECK constraints, index and foreign key to runtime_worker_profiles; existing rows
    untouched (bio NULL); a second run changes nothing; the database itself refuses invalid credentials and credentials
    for an account without a runtime worker profile;
  * registration still creates a self-registered worker profile exactly as before;
  * editing: client (name, location), self-registered worker (every field, both copies of the location, skills
    replaced, bio trimmed), catalogue-linked account (display name only; 403 on location/profile/bio/credentials),
    worker account without a profile (name, location; 403 on profile fields); invalid input is refused with 400 and
    nothing saved; CSRF required;
  * credentials: add/edit/delete own (direct SQL), javascript: and data: links refused, the per-kind limit, another
    user's credential 404 and unchanged, "Self-reported" labels and rel="noopener noreferrer nofollow" rendered;
  * avatars: the rendered class and letters equal an independent computation; every palette colour gives white text
    at least 4.5:1; no inline style;
  * frozen recommendations and Day 8 artifact hashes unchanged; the main database unchanged.

Usage:
    python scripts/verify_m4_profiles.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash

from src.app import create_app
from src.app.marketplace.runtime import RuntimeRecommender
from src.app.services import RecommendationService
from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine
from src.db.loader import load
from src.db.marketplace import PostgresMarketplaceRepository
from src.db.repositories import PostgresRepository
from src.db.schema import metadata

PASSWORD = "a long verifier password"
TOKEN = re.compile(r'name="csrf_token" value="([^"]+)"')
CSS = (REPO_ROOT / "src" / "app" / "static" / "css" / "style.css").read_text(encoding="utf-8")
RUNTIME = ("users", "runtime_worker_profiles", "runtime_worker_skills", "runtime_requests", "runtime_request_skills",
           "bookings", "app_reviews", "notifications", "worker_credentials")
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def main_database_snapshot() -> dict:
    engine = make_engine()
    with engine.connect() as conn:
        names = set(inspect(conn).get_table_names())
        snapshot = {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in RUNTIME if t in names}
        snapshot["schema_version"] = conn.execute(text("SELECT value FROM load_metadata WHERE key = 'schema_version'")).scalar()
        snapshot["tables"] = sorted(names)
    return snapshot


def day8_hashes_ok() -> bool:
    manifest = json.loads((REPO_ROOT / "results" / "day8" / "day8_manifest.json").read_text(encoding="utf-8"))
    return all(hashlib.sha256((REPO_ROOT / e["path"]).read_bytes()).hexdigest() == e["sha256"] for e in manifest["files"])


# ---- independent re-implementations ------------------------------------------------------------------------
def expected_initials(name: str) -> str:
    words = [w for w in unicodedata.normalize("NFC", name).split() if any(c.isalpha() for c in w)]
    first = lambda w: next(c for c in w if c.isalpha()).upper()[0]  # noqa: E731
    return "" if not words else first(words[0]) if len(words) == 1 else first(words[0]) + first(words[-1])


def expected_slot(user_id: int, size: int) -> int:
    return int.from_bytes(hashlib.sha256(str(user_id).encode()).digest()[:8], "big") % size + 1


def contrast_white(hex_colour: str) -> float:
    def lin(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 1.05 / (0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b) + 0.05)


def main() -> int:
    test_db, main_db = os.environ.get("POSTGRES_TEST_DB"), os.environ.get("POSTGRES_DB")
    if not test_db or test_db == main_db:
        check("POSTGRES_TEST_DB configured and distinct from POSTGRES_DB", False)
        return 1
    main_before = main_database_snapshot()
    settings = load_postgres_settings(TEST_DATABASE_ENV)
    engine = make_engine(settings)

    def one(query, **params):
        with engine.connect() as conn:
            return conn.execute(text(query), params).first()

    def rows(query, **params):
        with engine.connect() as conn:
            return conn.execute(text(query), params).all()

    # ---- a realistic schema-v5 test database (with an existing worker profile), upgraded by the v6 script ------
    metadata.drop_all(engine)
    metadata.create_all(engine)
    load(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE worker_credentials"))
        conn.execute(text("ALTER TABLE runtime_worker_profiles DROP COLUMN bio"))
        conn.execute(text("UPDATE load_metadata SET value = '5' WHERE key = 'schema_version'"))
        legacy = conn.execute(text("INSERT INTO users (email, password_hash, role, display_name, country) VALUES "
                                   "('legacy.m4@example.com', 'h', 'worker', 'Legacy', 'India') RETURNING user_id")).scalar_one()
        conn.execute(text("INSERT INTO user_email_verification (user_id, verified_at) VALUES (:u, now())"), {"u": legacy})
        conn.execute(text("INSERT INTO runtime_worker_profiles (user_id, primary_category, experience_level, experience_years, "
                          "rate, country) VALUES (:u, 'Logo Design', 'entry', 1, 10, 'India')"), {"u": legacy})
    before = rows("SELECT user_id, primary_category, experience_level, experience_years, rate, country FROM runtime_worker_profiles")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    run = lambda *a: subprocess.run([sys.executable, "scripts/migrate_schema_v6.py", *a], cwd=REPO_ROOT, env=env,  # noqa: E731
                                    capture_output=True, text=True, timeout=300)
    first = run("--test-db")
    check("migrate_schema_v6.py --test-db: 5 -> 6, adds bio, creates worker_credentials",
          first.returncode == 0 and "steps applied: 5->6" in first.stdout and "tables created: worker_credentials" in first.stdout
          and "columns added: runtime_worker_profiles.bio" in first.stdout, first.stdout + first.stderr)
    check("migration output reveals no database password", settings.password not in first.stdout + first.stderr)
    second = run("--test-db")
    check("a second run changes nothing", second.returncode == 0 and "steps applied: none" in second.stdout
          and "columns added: none" in second.stdout, second.stdout)
    check("schema_version recorded as 6", one("SELECT value FROM load_metadata WHERE key = 'schema_version'")[0] == "6")
    check("existing worker profile untouched, bio NULL",
          rows("SELECT user_id, primary_category, experience_level, experience_years, rate, country FROM runtime_worker_profiles") == before
          and one("SELECT bio FROM runtime_worker_profiles WHERE user_id = :u", u=legacy)[0] is None)
    bio_col = one("SELECT data_type, is_nullable FROM information_schema.columns WHERE table_name = 'runtime_worker_profiles' "
                  "AND column_name = 'bio'")
    check("bio is a nullable text column", bio_col is not None and tuple(bio_col) == ("text", "YES"))
    checks = {r[0] for r in rows("SELECT conname FROM pg_constraint WHERE contype = 'c' AND conrelid IN "
                                 "('worker_credentials'::regclass, 'runtime_worker_profiles'::regclass)")}
    check("CHECK constraints: bio length, credential kind/title/organisation/year/url/description",
          {"ck_runtime_worker_bio_length", "ck_worker_credentials_kind", "ck_worker_credentials_title",
           "ck_worker_credentials_organisation", "ck_worker_credentials_year", "ck_worker_credentials_url",
           "ck_worker_credentials_description"} <= checks, str(checks))
    fk = rows("SELECT ccu.table_name, kcu.column_name FROM information_schema.table_constraints tc "
              "JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name "
              "JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name "
              "WHERE tc.table_name = 'worker_credentials' AND tc.constraint_type = 'FOREIGN KEY'")
    check("worker_credentials.user_id references runtime_worker_profiles only", [tuple(r) for r in fk] == [("runtime_worker_profiles", "user_id")], str(fk))
    index = one("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_worker_credentials_user_kind'")
    check("index on (user_id, kind)", index is not None and "(user_id, kind)" in index[0])
    with engine.begin() as conn:
        no_profile = conn.execute(text("INSERT INTO users (email, password_hash, role, display_name) VALUES "
                                       "('noprofile.m4@example.com', 'h', 'client', 'N') RETURNING user_id")).scalar_one()
    refused = 0
    for values in ({"u": legacy, "k": "degree", "t": "x", "y": None, "l": None}, {"u": legacy, "k": "portfolio", "t": "", "y": None, "l": None},
                   {"u": legacy, "k": "portfolio", "t": "x", "y": 1949, "l": None},
                   {"u": legacy, "k": "portfolio", "t": "x", "y": None, "l": "javascript:alert(1)"},
                   {"u": no_profile, "k": "portfolio", "t": "x", "y": None, "l": None}):
        try:
            with engine.begin() as conn:
                conn.execute(text("INSERT INTO worker_credentials (user_id, kind, title, year, url) VALUES (:u, :k, :t, :y, :l)"), values)
        except IntegrityError:
            refused += 1
    check("database refuses bad kind, empty title, year < 1950, javascript: URL and accounts without a worker profile", refused == 5)

    # ---- the application ------------------------------------------------------------------------
    service = RecommendationService.build(repository=PostgresRepository(engine))
    probe = service.data.requests.loc[service.data.requests["split"] == "test", "link"].iloc[0]
    recommendations_before = json.dumps(service.recommend(probe, 10), sort_keys=True)
    app = create_app("testing", service=service, marketplace_repository=PostgresMarketplaceRepository(engine),
                     runtime_recommender=RuntimeRecommender.build(service))
    app.config["CREDENTIALS_PER_KIND"] = 3
    palette_size = app.config["AVATAR_PALETTE_SIZE"]
    countries = service.data.known_countries
    home, away = countries[0], countries[1]

    def login(email):
        c = app.test_client()
        c.post("/login", data={"email": email, "password": PASSWORD, "csrf_token": TOKEN.search(c.get("/login").get_data(as_text=True)).group(1)})
        return c

    def account(email, role, name, worker_id=None, country=None, profile=False):
        with engine.begin() as conn:
            uid = conn.execute(text("INSERT INTO users (email, password_hash, role, display_name, worker_id, country) "
                                    "VALUES (:e, :h, :r, :n, :w, :c) RETURNING user_id"),
                               {"e": email, "h": generate_password_hash(PASSWORD), "r": role, "n": name, "w": worker_id,
                                "c": country}).scalar_one()
            conn.execute(text("INSERT INTO user_email_verification (user_id, verified_at) VALUES (:u, now())"), {"u": uid})
            if profile:
                conn.execute(text("INSERT INTO runtime_worker_profiles (user_id, primary_category, experience_level, "
                                  "experience_years, rate, country) VALUES (:u, 'Logo Design', 'intermediate', 4, 25, :c)"),
                             {"u": uid, "c": country})
                conn.execute(text("INSERT INTO runtime_worker_skills (user_id, skill) VALUES (:u, 'Logo Design')"), {"u": uid})
        return login(email), uid

    def post(c, path, data, token_path="/profile/edit"):
        return c.post(path, data={**data, "csrf_token": TOKEN.search(c.get(token_path).get_data(as_text=True)).group(1)})

    # registration unchanged: a new worker registers with a profile and skills
    r = app.test_client()
    reg = r.post("/register", data={"role": "worker", "email": "new.worker.m4@example.com", "password": PASSWORD,
                                    "confirm_password": PASSWORD, "display_name": "New Worker", "worker_country": home,
                                    "primary_category": "Logo Design", "experience_level": "entry", "experience_years": "2",
                                    "rate": "15", "skills": "logo design, Adobe Illustrator",
                                    "csrf_token": TOKEN.search(r.get("/register").get_data(as_text=True)).group(1)})
    reg_row = one("SELECT p.primary_category, p.country, p.bio FROM users u JOIN runtime_worker_profiles p USING (user_id) "
                  "WHERE u.email = 'new.worker.m4@example.com'")
    skills_row = rows("SELECT skill FROM runtime_worker_skills s JOIN users u USING (user_id) WHERE u.email = 'new.worker.m4@example.com' ORDER BY skill")
    check("registration still creates the worker profile and skills (bio empty)",
          reg.status_code == 302 and tuple(reg_row) == ("Logo Design", home, None)
          and [s[0] for s in skills_row] == ["Adobe Illustrator", "Logo Design"])

    # client
    client, client_id = account("client.m4@example.com", "client", "Ann Smith")
    ok = post(client, "/profile/edit", {"display_name": "Zoë Ndlovu", "client_country": home})
    check("client edits name and location (direct SQL)", ok.status_code == 302 and
          tuple(one("SELECT display_name, country FROM users WHERE user_id = :u", u=client_id)) == ("Zoë Ndlovu", home))
    bad = post(client, "/profile/edit", {"display_name": "", "client_country": "Atlantis"})
    check("invalid client edit: 400, values kept, nothing saved",
          bad.status_code == 400 and 'value="Atlantis"' not in bad.get_data(as_text=True)  # select keeps only listed values
          and one("SELECT display_name FROM users WHERE user_id = :u", u=client_id)[0] == "Zoë Ndlovu")
    check("edit without CSRF token is refused (400)", client.post("/profile/edit", data={"display_name": "X"}).status_code == 400)
    dashboard = client.get("/dashboard").get_data(as_text=True)
    check("client avatar: independent initials and palette slot",
          f'avatar-initials--{expected_slot(client_id, palette_size)}' in dashboard and f">{expected_initials('Zoë Ndlovu')}</span>" in dashboard)

    # self-registered worker
    worker, worker_id = account("worker.m4@example.com", "worker", "Ravi Kumar", country=home, profile=True)
    ok = post(worker, "/profile/edit", {"display_name": "Ravi Kumar", "worker_country": away, "primary_category": "Logo Design",
                                        "experience_level": "expert", "experience_years": "7.5", "rate": "40",
                                        "skills": "adobe illustrator, Logo Design", "bio": "   Logos for small businesses.   "})
    prof = one("SELECT experience_level, experience_years, rate, country, bio FROM runtime_worker_profiles WHERE user_id = :u", u=worker_id)
    skills = [s[0] for s in rows("SELECT skill FROM runtime_worker_skills WHERE user_id = :u ORDER BY skill", u=worker_id)]
    check("self-registered worker edits every field; skills replaced; bio trimmed (direct SQL)",
          ok.status_code == 302 and tuple(prof) == ("expert", 7.5, 40.0, away, "Logos for small businesses.")
          and skills == ["Adobe Illustrator", "Logo Design"])
    check("both copies of the worker's location updated", one("SELECT country FROM users WHERE user_id = :u", u=worker_id)[0] == away)
    bad = post(worker, "/profile/edit", {"display_name": "Ravi Kumar", "worker_country": away, "primary_category": "Logo Design",
                                         "experience_level": "expert", "experience_years": "7.5", "rate": "40",
                                         "skills": "Cobol wizardry", "bio": "x" * 1001})
    html = bad.get_data(as_text=True)
    check("unknown skill and over-long bio refused with registration-style messages, nothing saved",
          bad.status_code == 400 and "Unknown skill(s): Cobol wizardry." in html and "1,000 characters" in html
          and [s[0] for s in rows("SELECT skill FROM runtime_worker_skills WHERE user_id = :u ORDER BY skill", u=worker_id)] == skills)

    # catalogue-linked account
    linked, linked_id = account("linked.m4@example.com", "worker", "Linked Person", worker_id="W000001", country=home)
    check("catalogue-linked account sees the read-only note", "can&#39;t be edited" in linked.get("/profile").get_data(as_text=True))
    check("catalogue-linked account may change its display name",
          post(linked, "/profile/edit", {"display_name": "Renamed Person"}).status_code == 302
          and one("SELECT display_name FROM users WHERE user_id = :u", u=linked_id)[0] == "Renamed Person")
    forbidden = [post(linked, "/profile/edit", {"display_name": "X", **extra}).status_code
                 for extra in ({"worker_country": away}, {"primary_category": "Logo Design"}, {"bio": "x"}, {"skills": "Python"})]
    check("catalogue-linked account: 403 on location, profile, bio and skills; nothing changed",
          forbidden == [403] * 4 and tuple(one("SELECT display_name, country FROM users WHERE user_id = :u", u=linked_id)) == ("Renamed Person", home))
    check("catalogue-linked account cannot add credentials (403)",
          linked.get("/profile/credentials/new").status_code == 403
          and post(linked, "/profile/credentials/new", {"kind": "portfolio", "title": "x"}).status_code == 403)
    check("catalogue-linked profile keeps the generic icon", '<div class="avatar" aria-hidden="true">' in linked.get("/profile").get_data(as_text=True))

    # worker account without a profile (the demo operator's shape)
    operator, operator_id = account("operator.m4@example.com", "worker", "Demo Desk")
    check("worker without a profile edits name and location; 403 on profile fields",
          post(operator, "/profile/edit", {"display_name": "Demo Desk 2", "worker_country": home}).status_code == 302
          and tuple(one("SELECT display_name, country FROM users WHERE user_id = :u", u=operator_id)) == ("Demo Desk 2", home)
          and post(operator, "/profile/edit", {"display_name": "X", "rate": "10"}).status_code == 403)

    # credentials
    def add(c, **fields):
        form = {"kind": "certification", "title": "Cert", "organisation": "", "year": "", "url": "", "description": "", **fields}
        return post(c, "/profile/credentials/new", form, "/profile/credentials/new")

    first_add = add(worker, title="AWS Cloud Practitioner", organisation="Amazon", year="2024", url="https://example.com/cert")
    cred = one("SELECT credential_id, kind, title, organisation, year, url FROM worker_credentials WHERE user_id = :u", u=worker_id)
    check("credential added (direct SQL)", first_add.status_code == 302 and cred is not None
          and tuple(cred)[1:] == ("certification", "AWS Cloud Practitioner", "Amazon", 2024, "https://example.com/cert"))
    for scheme in ("javascript:alert(1)", "data:text/html,x", "ftp://example.com"):
        r = add(worker, title="Bad", url=scheme)
        check(f"link refused: {scheme.split(':')[0]}:", r.status_code == 400 and
              one("SELECT count(*) FROM worker_credentials WHERE user_id = :u AND title = 'Bad'", u=worker_id)[0] == 0)
    add(worker, title="Cert 2")
    add(worker, title="Cert 3")
    over = add(worker, title="Cert 4")
    check("per-kind limit (configured 3): the 4th certification is refused",
          over.status_code == 400 and one("SELECT count(*) FROM worker_credentials WHERE user_id = :u AND kind = 'certification'", u=worker_id)[0] == 3)
    check("the limit is per kind", add(worker, kind="portfolio", title="Portfolio site", url="https://example.com").status_code == 302)
    page = worker.get("/profile").get_data(as_text=True)
    check("profile shows 'Self-reported' and safe link attributes",
          page.count("Self-reported") >= 4 and 'href="https://example.com/cert" rel="noopener noreferrer nofollow"' in page)
    check("no inline style on profile pages", ' style="' not in page and ' style="' not in worker.get("/profile/edit").get_data(as_text=True))
    other, _ = account("other.m4@example.com", "worker", "Other Worker", country=home, profile=True)
    cid = cred.credential_id
    check("another worker: 404 on edit and delete; credential unchanged",
          other.get(f"/profile/credentials/{cid}/edit").status_code == 404
          and post(other, f"/profile/credentials/{cid}/edit", {"kind": "portfolio", "title": "Stolen"}).status_code == 404
          and post(other, f"/profile/credentials/{cid}/delete", {}).status_code == 404
          and one("SELECT title FROM worker_credentials WHERE credential_id = :c", c=cid)[0] == "AWS Cloud Practitioner")
    edited = post(worker, f"/profile/credentials/{cid}/edit", {"kind": "experience", "title": "Freelance designer"},
                  f"/profile/credentials/{cid}/edit")
    check("own credential edited (direct SQL)", edited.status_code == 302 and
          tuple(one("SELECT kind, title, organisation, url FROM worker_credentials WHERE credential_id = :c", c=cid)) ==
          ("experience", "Freelance designer", None, None))
    check("delete requires CSRF", worker.post(f"/profile/credentials/{cid}/delete").status_code == 400)
    deleted = post(worker, f"/profile/credentials/{cid}/delete", {}, "/profile")
    check("own credential deleted (direct SQL)", deleted.status_code == 302 and
          one("SELECT count(*) FROM worker_credentials WHERE credential_id = :c", c=cid)[0] == 0)

    # avatar palette contrast, recomputed from the stylesheet
    colours = dict(re.findall(r"--avatar-(\d+):\s*(#[0-9a-fA-F]{6})", CSS))
    check(f"palette: {palette_size} colours, each with >= 4.5:1 contrast for white text",
          sorted(map(int, colours)) == list(range(1, palette_size + 1)) and all(contrast_white(c) >= 4.5 for c in colours.values()),
          str({k: round(contrast_white(v), 2) for k, v in colours.items()}))

    # frozen artifacts, clean-up, main database
    check("frozen recommendations unchanged", json.dumps(service.recommend(probe, 10), sort_keys=True) == recommendations_before)
    check("Day 8 manifest hashes unchanged", day8_hashes_ok())
    with engine.begin() as conn:
        for name in ("worker_credentials", "notifications", "app_reviews", "bookings", "runtime_request_skills", "runtime_requests",
                     "runtime_worker_skills", "runtime_worker_profiles", "user_identities", "auth_tokens",
                     "user_email_verification", "users"):
            conn.execute(text(f"DELETE FROM {name}"))
    check("main database unchanged (runtime row counts, schema version, tables)", main_database_snapshot() == main_before)
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} CHECK(S) FAILED'}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
