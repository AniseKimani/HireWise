"""Independent verification of pre-deployment Milestone 3 (booking notifications, notification email, demo mode).

Writes ONLY to POSTGRES_TEST_DB. The main database (POSTGRES_DB) is only read, and is checked to be unchanged.
No real email is sent: the file mail backend writes to a temporary directory outside the repository.

Independent of the implementation: it does not import src/app/marketplace/notifications.py, notify.py or the
repository's notification methods. Accounts are created with direct SQL (Werkzeug password hashes), the app is
driven only through HTTP (Flask test client), recommended workers are read from the rendered request page, and every
outcome is checked with direct SQL or by reading the .eml files.

Checks:
  * schema v4 -> v5 through scripts/migrate_schema_v5.py --test-db: notifications table, its CHECK/UNIQUE constraints,
    indexes and foreign keys (information_schema / pg_indexes); existing rows untouched; a second run changes nothing;
  * same transaction: each notification's created_at equals its booking's created_at / updated_at or its review's
    created_at (PostgreSQL's now() is fixed per transaction);
  * recipients: booked worker on request/cancel/complete/review, the client on accept/decline, never the actor;
    an unclaimed worker with demo mode off notifies nobody;
  * message length <= 500 even for a maximum-length title; emails carry the booking number and worker ID but no title;
  * ownership: another user's notification is 404 and unchanged; mark-read without CSRF is refused; the unread-count
    endpoint answers 401 JSON when logged out, matches SQL when logged in, and is not cacheable;
  * demo mode: the operator is notified for unclaimed workers ("[Demo]" subject naming the simulated worker), may
    accept them, is refused (404) on a claimed worker's booking, and its emails stop at the daily cap while in-app
    notifications continue;
  * a failing mail backend does not block a booking;
  * frozen recommendations and Day 8 artifact hashes unchanged; the main database unchanged.

Usage:
    python scripts/verify_m3_notifications.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from email import message_from_bytes
from email.policy import default as default_policy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import inspect, text
from werkzeug.security import generate_password_hash

from src.app import create_app
from src.app.marketplace.mailer import build_mailer
from src.app.marketplace.runtime import RuntimeRecommender
from src.app.services import RecommendationService
from src.db.database import TEST_DATABASE_ENV, load_postgres_settings, make_engine
from src.db.loader import load
from src.db.marketplace import PostgresMarketplaceRepository
from src.db.repositories import PostgresRepository
from src.db.schema import metadata

PASSWORD = "a long verifier password"
TOKEN = re.compile(r'name="csrf_token" value="([^"]+)"')
BOOKABLE = re.compile(r'name="worker_id" value="([A-Za-z0-9_-]+)"')
OPERATOR = "operator.m3@example.com"
RUNTIME = ("users", "runtime_worker_profiles", "runtime_worker_skills", "runtime_requests", "runtime_request_skills",
           "bookings", "app_reviews", "notifications")
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


def day8_hashes() -> dict:
    manifest = json.loads((REPO_ROOT / "results" / "day8" / "day8_manifest.json").read_text(encoding="utf-8"))
    return {e["path"]: hashlib.sha256((REPO_ROOT / e["path"]).read_bytes()).hexdigest() == e["sha256"] for e in manifest["files"]}


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

    # ---- a realistic schema-v4 test database, upgraded by the v5 script ---------------------------------
    metadata.drop_all(engine)
    metadata.create_all(engine)
    load(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE worker_credentials"))  # schema v6 (Milestone 4) additions
        conn.execute(text("ALTER TABLE runtime_worker_profiles DROP COLUMN bio"))
        conn.execute(text("DROP TABLE notifications"))
        conn.execute(text("UPDATE load_metadata SET value = '4' WHERE key = 'schema_version'"))
        conn.execute(text("INSERT INTO users (email, password_hash, role, display_name) VALUES ('legacy.m3@example.com', 'h', 'client', 'L')"))
        conn.execute(text("INSERT INTO user_email_verification (user_id, verified_at) SELECT user_id, now() FROM users"))
    users_before = rows("SELECT user_id, email, password_hash, auth_version FROM users ORDER BY user_id")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    first = subprocess.run([sys.executable, "scripts/migrate_schema_v5.py", "--test-db"], cwd=REPO_ROOT, env=env,
                           capture_output=True, text=True, timeout=300)
    check("migrate_schema_v5.py --test-db succeeds: 4 -> 5, creates notifications",
          first.returncode == 0 and "steps applied: 4->5" in first.stdout and "tables created: notifications" in first.stdout,
          first.stdout + first.stderr)
    check("migration output reveals no database password", settings.password not in first.stdout + first.stderr)
    second = subprocess.run([sys.executable, "scripts/migrate_schema_v5.py", "--test-db"], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=300)
    check("a second run changes nothing", second.returncode == 0 and "steps applied: none" in second.stdout, second.stdout)
    check("schema_version recorded as 5", one("SELECT value FROM load_metadata WHERE key = 'schema_version'")[0] == "5")
    check("existing users untouched by the migration",
          rows("SELECT user_id, email, password_hash, auth_version FROM users ORDER BY user_id") == users_before)
    constraints = {r[0]: r[1] for r in rows(
        "SELECT constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_name = 'notifications'")}
    check("CHECK constraints: kind, message length, read_at >= created_at",
          {"ck_notifications_kind", "ck_notifications_message_length", "ck_notifications_read_at"} <= set(constraints))
    check("UNIQUE (booking_id, kind, user_id)", constraints.get("uq_notifications_booking_kind_user") == "UNIQUE")
    fks = {(r[0], r[1]) for r in rows(
        "SELECT kcu.column_name, ccu.table_name FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name "
        "JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name "
        "WHERE tc.table_name = 'notifications' AND tc.constraint_type = 'FOREIGN KEY'")}
    check("foreign keys: user_id -> users, booking_id -> bookings", {("user_id", "users"), ("booking_id", "bookings")} <= fks, str(fks))
    indexes = {r[0]: r[1] for r in rows("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'notifications'")}
    check("index on (user_id, read_at)", "(user_id, read_at)" in indexes.get("ix_notifications_user_read", ""))
    check("index on (user_id, created_at DESC)", "(user_id, created_at DESC)" in indexes.get("ix_notifications_user_created", ""))
    nullable = {r[0]: r[1] for r in rows("SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name = 'notifications'")}
    check("user_id/kind/message NOT NULL; booking_id/read_at nullable",
          [nullable[c] for c in ("user_id", "kind", "message", "booking_id", "read_at")] == ["NO", "NO", "NO", "YES", "YES"])

    # The application now requires schema v6 (Milestone 4); upgrade with the v6 script before starting it.
    v6 = subprocess.run([sys.executable, "scripts/migrate_schema_v6.py", "--test-db"], cwd=REPO_ROOT, env=env,
                        capture_output=True, text=True, timeout=300)
    check("migrate_schema_v6.py --test-db then succeeds (5 -> 6)", v6.returncode == 0 and "steps applied: 5->6" in v6.stdout,
          v6.stdout + v6.stderr)

    # ---- the application ------------------------------------------------------------------------
    service = RecommendationService.build(repository=PostgresRepository(engine))
    probe = service.data.requests.loc[service.data.requests["split"] == "test", "link"].iloc[0]
    recommendations_before = json.dumps(service.recommend(probe, 10), sort_keys=True)
    app = create_app("testing", service=service, marketplace_repository=PostgresMarketplaceRepository(engine),
                     runtime_recommender=RuntimeRecommender.build(service))
    outbox = Path(tempfile.mkdtemp(prefix="hirewise-verify-m3-outbox-"))
    app.extensions["hirewise_mailer"] = build_mailer({"MAIL_BACKEND": "file", "MAIL_OUTBOX_DIR": str(outbox),
                                                      "APP_BASE_URL": "http://localhost", "MAIL_SEND_ASYNC": False},
                                                     environment="development", required=True)
    check("file outbox is outside the repository", REPO_ROOT not in outbox.resolve().parents)

    def mails(to=None):
        msgs = [message_from_bytes(p.read_bytes(), policy=default_policy) for p in sorted(outbox.glob("*.eml"), key=os.path.getmtime)]
        return [m for m in msgs if to is None or m["To"] == to]

    def account(email, role, worker_id=None):
        with engine.begin() as conn:
            user_id = conn.execute(text("INSERT INTO users (email, password_hash, role, display_name, worker_id) "
                                        "VALUES (:e, :h, :r, :n, :w) RETURNING user_id"),
                                   {"e": email, "h": generate_password_hash(PASSWORD), "r": role, "n": email.split("@")[0],
                                    "w": worker_id}).scalar_one()
            conn.execute(text("INSERT INTO user_email_verification (user_id, verified_at) VALUES (:u, now())"), {"u": user_id})
        c = app.test_client()
        c.post("/login", data={"email": email, "password": PASSWORD,
                               "csrf_token": TOKEN.search(c.get("/login").get_data(as_text=True)).group(1)})
        return c, user_id

    def post(c, path, data=None, token_path="/dashboard"):
        return c.post(path, data={**(data or {}), "csrf_token": TOKEN.search(c.get(token_path).get_data(as_text=True)).group(1)})

    def new_request(c, title):
        r = post(c, "/client/requests/new", {"title": title, "category": "Logo Design", "skills": "Logo Design, Adobe Illustrator",
                                             "pay_type": "fixed", "budget": "150"}, "/client/requests/new")
        request_id = int(r.headers["Location"].rstrip("/").split("/")[-1])
        return request_id, BOOKABLE.findall(c.get(f"/client/requests/{request_id}").get_data(as_text=True))

    def book(c, request_id, worker_id):
        post(c, f"/client/requests/{request_id}/book", {"worker_id": worker_id}, f"/client/requests/{request_id}")
        return one("SELECT booking_id FROM bookings WHERE request_id = :r ORDER BY booking_id DESC", r=request_id)[0]

    def act(c, booking_id, action):
        return post(c, f"/bookings/{booking_id}/{action}", {"return_to": "/bookings"}, "/bookings")

    def notes(user_id):
        return rows("SELECT notification_id, booking_id, kind, message, created_at, read_at FROM notifications "
                    "WHERE user_id = :u ORDER BY notification_id", u=user_id)

    # ---- claimed worker, demo mode off ------------------------------------------------------------
    client, client_id = account("client.m3@example.com", "client")
    long_title = "Brand identity " + "x" * 185  # 200 characters: the longest title the request form accepts
    request_id, bookable = new_request(client, long_title)
    worker_id = bookable[0]
    worker, worker_user = account("worker.m3@example.com", "worker", worker_id=worker_id)
    booking_id = book(client, request_id, worker_id)
    [n1] = notes(worker_user)
    booking_row = one("SELECT created_at, updated_at FROM bookings WHERE booking_id = :b", b=booking_id)
    check("booking -> one notification for the linked worker (booking_requested, unread)",
          (n1.booking_id, n1.kind, n1.read_at) == (booking_id, "booking_requested", None))
    check("same transaction: notification created_at == booking created_at", n1.created_at == booking_row.created_at)
    check("message <= 500 characters and the 200-character title is shortened", len(n1.message) <= 500
          and long_title not in n1.message and "…" in n1.message, n1.message)
    check("the acting client is not notified", notes(client_id) == [])
    [m1] = mails("worker.m3@example.com")
    check("email to the worker holds booking number and worker ID, no request title",
          f"#{booking_id}" in m1.get_content() and worker_id in m1.get_content()
          and "Brand identity" not in m1.get_content() + m1["Subject"])
    check("email link is built from APP_BASE_URL", "http://localhost/bookings" in m1.get_content())

    act(worker, booking_id, "accept")
    upd = one("SELECT updated_at, status FROM bookings WHERE booking_id = :b", b=booking_id)
    [n2] = notes(client_id)
    check("accept -> the client is notified, in the same transaction",
          upd.status == "accepted" and n2.kind == "booking_accepted" and n2.created_at == upd.updated_at)
    check("the accepting worker is not notified again", len(notes(worker_user)) == 1)
    act(client, booking_id, "complete")
    post(client, f"/bookings/{booking_id}/review", {"rating": "5", "review_text": "Great."}, f"/bookings/{booking_id}/review")
    review_at = one("SELECT created_at FROM app_reviews WHERE booking_id = :b", b=booking_id)[0]
    kinds = [n.kind for n in notes(worker_user)]
    check("complete and review -> the worker is notified", kinds == ["booking_requested", "booking_completed", "review_received"], str(kinds))
    check("same transaction: review notification created_at == review created_at", notes(worker_user)[-1].created_at == review_at)
    r2, b2 = new_request(client, "Second logo")
    second_booking = book(client, r2, worker_id)
    act(worker, second_booking, "decline")
    check("decline -> the client is notified", notes(client_id)[-1].kind == "booking_declined")
    r3, _ = new_request(client, "Third logo")
    third_booking = book(client, r3, worker_id)
    act(client, third_booking, "cancel")
    check("cancel -> the worker is notified", notes(worker_user)[-1].kind == "booking_cancelled")
    before_repeat = len(notes(worker_user)) + len(notes(client_id))
    act(worker, booking_id, "accept")  # completed booking: not allowed
    check("an invalid repeated action creates nothing", len(notes(worker_user)) + len(notes(client_id)) == before_repeat)

    # ---- ownership, CSRF, the unread-count endpoint ----------------------------------------------
    target = notes(worker_user)[0]
    check("another user's notification: 404, unchanged",
          post(client, f"/notifications/{target.notification_id}/read", token_path="/notifications").status_code == 404
          and notes(worker_user)[0].read_at is None)
    check("mark-read without a CSRF token is refused (400)",
          worker.post(f"/notifications/{target.notification_id}/read").status_code == 400 and notes(worker_user)[0].read_at is None)
    post(worker, f"/notifications/{target.notification_id}/read", token_path="/notifications")
    check("own notification marked read (direct SQL)", notes(worker_user)[0].read_at is not None)
    unread_sql = one("SELECT count(*) FROM notifications WHERE user_id = :u AND read_at IS NULL", u=worker_user)[0]
    response = worker.get("/notifications/unread-count")
    check("unread-count matches SQL and is not cacheable",
          response.get_json() == {"unread": unread_sql} and response.headers.get("Cache-Control") == "no-store")
    anon = app.test_client().get("/notifications/unread-count")
    check("unread-count answers 401 JSON when logged out", anon.status_code == 401 and anon.is_json)
    post(worker, "/notifications/read-all", token_path="/notifications")
    check("mark all read (direct SQL)", one("SELECT count(*) FROM notifications WHERE user_id = :u AND read_at IS NULL",
                                             u=worker_user)[0] == 0)

    # ---- unclaimed worker, demo mode off -> nobody ------------------------------------------------
    operator, operator_id = account(OPERATOR, "worker")
    r4, bookable4 = new_request(client, "Unclaimed logo")
    unclaimed = next(w for w in bookable4 if one("SELECT count(*) FROM users WHERE worker_id = :w", w=w)[0] == 0)
    b4 = book(client, r4, unclaimed)
    check("unclaimed worker with demo mode off: no notification row", one(
        "SELECT count(*) FROM notifications WHERE booking_id = :b", b=b4)[0] == 0)
    check("the operator cannot act while demo mode is off", act(operator, b4, "accept").status_code == 404)
    check("client sees 'won't respond' on the simulated profile",
          "simulated research profile, so it won" in client.get(f"/workers/{unclaimed}").get_data(as_text=True))
    act(client, b4, "cancel")

    # ---- demo mode on --------------------------------------------------------------------------
    app.config.update(DEMO_MODE=True, DEMO_OPERATOR_EMAIL=OPERATOR, DEMO_OPERATOR_EMAIL_DAILY_CAP=2)
    r5, bookable5 = new_request(client, "Demo logo")
    b5 = book(client, r5, unclaimed)
    op_notes = notes(operator_id)
    check("demo mode: the operator is notified for the unclaimed worker",
          len(op_notes) == 1 and op_notes[0].booking_id == b5 and unclaimed in op_notes[0].message)
    op_mail = mails(OPERATOR)
    check("operator email subject '[Demo] New booking for simulated worker W...'",
          len(op_mail) == 1 and op_mail[0]["Subject"] == f"[Demo] New booking for simulated worker {unclaimed}")
    check("client sees the demo-account note", "Responses are handled by the HireWise demo account."
          in client.get(f"/client/requests/{r5}").get_data(as_text=True))
    check("operator dashboard labelled 'Demo account' and lists the simulated worker",
          all(s in operator.get("/dashboard").get_data(as_text=True) for s in ("Demo account", unclaimed)))
    act(operator, b5, "accept")
    check("operator may accept an unclaimed worker's booking (direct SQL)",
          one("SELECT status FROM bookings WHERE booking_id = :b", b=b5)[0] == "accepted")
    check("the client is notified of the operator's acceptance", notes(client_id)[-1].kind == "booking_accepted")
    r6, _ = new_request(client, "Claimed again")
    b6 = book(client, r6, worker_id)
    check("operator refused (404) on a claimed worker's booking, status unchanged",
          act(operator, b6, "accept").status_code == 404 and one("SELECT status FROM bookings WHERE booking_id = :b", b=b6)[0] == "pending")
    check("operator has no client powers", operator.get("/client/requests/new").status_code == 403)
    act(client, b6, "cancel")
    for title in ("Cap one", "Cap two"):
        rid, bk = new_request(client, title)
        book(client, rid, next(w for w in bk if one("SELECT count(*) FROM users WHERE worker_id = :w", w=w)[0] == 0))
    check("daily cap: in-app notifications continue (3 rows) but operator emails stop at 2",
          len(notes(operator_id)) == 3 and len(mails(OPERATOR)) == 2, f"{len(notes(operator_id))} rows, {len(mails(OPERATOR))} emails")

    # ---- a failing mail backend never blocks a booking ------------------------------------------
    class Broken:
        name = "broken"

        def deliver(self, message):
            raise ConnectionRefusedError("verifier: simulated outage")

    app.extensions["hirewise_mailer"].backend = Broken()
    r7, bk7 = new_request(client, "Outage logo")
    b7 = book(client, r7, worker_id)
    check("booking and notification committed despite the mail outage",
          one("SELECT count(*) FROM notifications WHERE booking_id = :b AND kind = 'booking_requested'", b=b7)[0] == 1)

    # ---- frozen artifacts, clean-up, main database --------------------------------------------------
    check("frozen recommendations unchanged", json.dumps(service.recommend(probe, 10), sort_keys=True) == recommendations_before)
    check("Day 8 manifest hashes unchanged", all(day8_hashes().values()))
    with engine.begin() as conn:
        for name in ("worker_credentials", "notifications", "app_reviews", "bookings", "runtime_request_skills", "runtime_requests", "runtime_worker_skills",
                     "runtime_worker_profiles", "user_identities", "auth_tokens", "user_email_verification", "users"):
            conn.execute(text(f"DELETE FROM {name}"))
    check("main database unchanged (runtime row counts, schema version, tables)", main_database_snapshot() == main_before)
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} CHECK(S) FAILED'}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
