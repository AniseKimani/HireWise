"""Milestone 3 tests: in-app booking notifications, notification emails and demo mode.

- Pure rules (no app, no database): recipients for every event (claimed worker, unclaimed with demo mode off,
  unclaimed with demo mode on), the demo operator's permissions, the operator configuration check, the email skip
  rules, the operator's daily email cap, message length and email content.
- NotificationBehaviour (mixin) runs through the real app against BOTH repositories:
  InMemoryNotificationTests (default) and PostgresNotificationTests (OPT-IN: HIREWISE_RUN_POSTGRES_TESTS=1, on the
  isolated POSTGRES_TEST_DB only -- never POSTGRES_DB). Email uses the in-memory backend (TestingConfig).
- The v4 -> v5 migration (and a second run changing nothing) is tested in tests/test_account_security.py
  (MigrationTests, opt-in), next to the earlier migrations.
"""
import os
import re
import unittest
from pathlib import Path

from src.app import create_app
from src.app.marketplace import notifications as N
from src.app.marketplace import rules
from src.app.marketplace.runtime import RuntimeRequest
from src.app.marketplace.security import hash_password
from src.db.schema import NOTIFICATION_MESSAGE_MAX_LENGTH

REPO_ROOT = Path(__file__).resolve().parents[1]
HAVE_DATA = (REPO_ROOT / "data" / "processed" / "synthetic").exists() and \
    (REPO_ROOT / "results" / "day8" / "selected_hybrid_config.json").exists()
PASSWORD = "correct horse battery"
TOKEN = re.compile(r'name="csrf_token" value="([^"]+)"')
OPERATOR_EMAIL = "operator@example.com"
VERIFIED = "2026-01-01"  # any non-None email_verified_at


def account(user_id, email, role="worker", worker_id=None, verified=True):
    return {"user_id": user_id, "email": email, "role": role, "worker_id": worker_id,
            "email_verified_at": VERIFIED if verified else None}


# ---------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------
class TestRecipientRules(unittest.TestCase):
    CLIENT = account(1, "client@example.com", role="client")
    LINKED = account(2, "linked@example.com", worker_id="W000001")
    OPERATOR = account(3, OPERATOR_EMAIL)

    def recipient(self, kind, linked, operator, actor):
        return N.recipient_for(kind, client_account=self.CLIENT, linked_account=linked, demo_operator=operator,
                               actor_user_id=actor)

    def test_every_event_for_claimed_and_unclaimed_workers(self):
        # (kind, actor): the client acts on worker-side events, the worker (or operator) on client-side ones.
        events = [(N.BOOKING_REQUESTED, 1), (N.BOOKING_CANCELLED, 1), (N.BOOKING_COMPLETED, 1), (N.REVIEW_RECEIVED, 1),
                  (N.BOOKING_ACCEPTED, 2), (N.BOOKING_DECLINED, 2)]
        for kind, actor in events:
            client_side = kind in N.CLIENT_SIDE_KINDS
            # claimed worker: worker-side events go to the linked account, whatever demo mode says
            for operator in (None, self.OPERATOR):
                r = self.recipient(kind, self.LINKED, operator, actor)
                self.assertEqual(r.user_id, 1 if client_side else 2, (kind, operator))
                self.assertFalse(r.is_demo_operator)
            # unclaimed worker, demo mode off: worker-side events reach nobody
            r = self.recipient(kind, None, None, actor if client_side else 1)
            self.assertEqual(r.user_id if r else None, 1 if client_side else None, kind)
            # unclaimed worker, demo mode on: worker-side events reach the demo operator
            r = self.recipient(kind, None, self.OPERATOR, 3 if client_side else 1)
            self.assertEqual(r.user_id, 1 if client_side else 3, kind)
            self.assertEqual(r.is_demo_operator, not client_side)

    def test_the_actor_is_never_notified(self):
        self.assertIsNone(self.recipient(N.BOOKING_ACCEPTED, None, None, actor=1))   # the client "accepting"
        self.assertIsNone(self.recipient(N.BOOKING_REQUESTED, self.LINKED, None, actor=2))

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            self.recipient("booking_exploded", self.LINKED, None, actor=1)

    def test_every_action_maps_to_one_event(self):
        self.assertEqual(set(N.EVENT_FOR_ACTION), set(rules.ACTIONS))
        self.assertEqual(N.CLIENT_SIDE_KINDS | N.WORKER_SIDE_KINDS, set(N.NOTIFICATION_KINDS))


class TestOperatorPermissions(unittest.TestCase):
    BOOKING = {"client_user_id": 1, "worker_id": "W000009", "status": "pending"}
    OPERATOR = {"role": "worker", "user_id": 3, "worker_id": None}

    def test_operator_acts_as_worker_only_on_unclaimed_bookings(self):
        self.assertEqual(rules.check_booking_actor(self.BOOKING, self.OPERATOR, is_demo_operator=True,
                                                   worker_claimed=False), "worker")
        with self.assertRaises(rules.NotAuthorized):  # refused on a claimed worker's booking
            rules.check_booking_actor(self.BOOKING, self.OPERATOR, is_demo_operator=True, worker_claimed=True)
        with self.assertRaises(rules.NotAuthorized):  # demo mode off (not the operator): nothing
            rules.check_booking_actor(self.BOOKING, self.OPERATOR, is_demo_operator=False, worker_claimed=False)
        with self.assertRaises(rules.NotAuthorized):  # defaults keep the Day 12 behaviour
            rules.check_booking_actor(self.BOOKING, self.OPERATOR)

    def test_operator_never_gets_client_powers(self):
        role = rules.check_booking_actor(self.BOOKING, self.OPERATOR, is_demo_operator=True, worker_claimed=False)
        for action in ("cancel", "complete"):
            with self.assertRaises(rules.InvalidTransition):
                rules.next_status("pending" if action == "cancel" else "accepted", action, role)
        self.assertEqual(rules.allowed_actions("pending", role), ["accept", "decline"])
        client_like = {**self.OPERATOR, "role": "client"}  # a client account is never treated as the operator
        with self.assertRaises(rules.NotAuthorized):
            rules.check_booking_actor({**self.BOOKING, "client_user_id": 99}, client_like, is_demo_operator=True,
                                      worker_claimed=False)
        linked = {**self.OPERATOR, "worker_id": "W000001"}  # an account linked to another worker is not the operator
        with self.assertRaises(rules.NotAuthorized):
            rules.check_booking_actor(self.BOOKING, linked, is_demo_operator=True, worker_claimed=False)

    def test_operator_configuration_check(self):
        good = account(3, OPERATOR_EMAIL)
        self.assertIsNone(N.demo_operator_problem(True, OPERATOR_EMAIL, good))
        self.assertEqual(N.demo_operator_problem(False, OPERATOR_EMAIL, good), "demo mode is off")
        for configured, acct, fragment in [
                ("", good, "not set"), (None, good, "not set"), (OPERATOR_EMAIL, None, "matches no account"),
                (OPERATOR_EMAIL, account(3, OPERATOR_EMAIL, role="client"), "not a worker"),
                (OPERATOR_EMAIL, account(3, OPERATOR_EMAIL, worker_id="W000001"), "linked to a catalogue worker"),
                (OPERATOR_EMAIL, account(3, OPERATOR_EMAIL, verified=False), "not verified")]:
            problem = N.demo_operator_problem(True, configured, acct)
            self.assertIn(fragment, problem)
            self.assertNotIn(OPERATOR_EMAIL, problem)  # names settings, never values


class TestEmailRules(unittest.TestCase):
    def test_skip_rules(self):
        r = N.Recipient(1, "a@example.com", verified=True)
        self.assertTrue(N.email_allowed(r, "demo.local"))
        self.assertFalse(N.email_allowed(N.Recipient(1, "a@example.com", verified=False), "demo.local"))
        self.assertFalse(N.email_allowed(N.Recipient(1, "worker@demo.local", verified=True), "demo.local"))
        self.assertFalse(N.email_allowed(N.Recipient(1, "worker@DEMO.local", verified=True), " Demo.Local , other.test"))
        self.assertFalse(N.email_allowed(N.Recipient(1, "x@other.test", verified=True), "demo.local,@other.test"))
        self.assertTrue(N.email_allowed(N.Recipient(1, "x@sub.demo.local", verified=True), "demo.local"))  # exact domain
        self.assertTrue(N.email_allowed(r, ""))
        self.assertFalse(N.email_allowed(None, ""))
        self.assertEqual(N.parse_skip_domains(" a.test, ,@B.test "), frozenset({"a.test", "b.test"}))

    def test_daily_cap_for_operator_emails(self):
        cap = 3
        decisions = [N.operator_email_decision(n, cap) for n in range(1, 7)]
        self.assertEqual([send for send, _ in decisions], [True, True, True, False, False, False])
        self.assertEqual([warn for _, warn in decisions], [False, False, False, True, False, False])  # warned once
        self.assertEqual(N.operator_email_decision(1, 0), (False, True))  # cap 0: no operator email at all

    def test_messages_never_exceed_the_database_limit(self):
        for title in ("x" * 10_000, " ".join(["word"] * 3000), "\n\t  ", None, "Logo"):
            for kind in N.NOTIFICATION_KINDS:
                for operator in (False, True):
                    message = N.notification_message(kind, worker_id="W000123", request_title=title,
                                                     for_demo_operator=operator, rating=5)
                    self.assertLessEqual(len(message), NOTIFICATION_MESSAGE_MAX_LENGTH)
                    self.assertNotIn("\n", message)
        shortened = N.shorten_title("y" * 500)
        self.assertEqual(len(shortened), N.TITLE_MAX_LENGTH)
        self.assertTrue(shortened.endswith("…"))
        self.assertEqual(N.shorten_title("short title"), "short title")

    def test_email_carries_booking_number_and_worker_only(self):
        for kind in N.NOTIFICATION_KINDS:
            for operator in (False, True):
                subject = N.email_subject(kind, worker_id="W002417", for_demo_operator=operator)
                body = N.email_body(kind, booking_id=42, worker_id="W002417", link="http://localhost/bookings",
                                    for_demo_operator=operator)
                self.assertIn("#42", body)
                self.assertIn("W002417", body)
                self.assertIn("http://localhost/bookings", body)
                self.assertEqual(subject.startswith("[Demo] "), operator)
                if operator:
                    self.assertIn("simulated worker W002417", subject)
        self.assertEqual(N.email_subject(N.BOOKING_REQUESTED, worker_id="W002417", for_demo_operator=True),
                         "[Demo] New booking for simulated worker W002417")

    def test_honesty_notes(self):
        self.assertEqual(N.honesty_note(True),
                         "This is a simulated research profile. Responses are handled by the HireWise demo account.")
        self.assertEqual(N.honesty_note(False), "This is a simulated research profile, so it won't respond.")


# ---------------------------------------------------------------------------
# Through the application, on both repositories
# ---------------------------------------------------------------------------
_SHARED = {}


def shared():
    if not _SHARED:
        from src.app.marketplace.runtime import RuntimeRecommender
        from src.app.services import RecommendationService
        from src.db.repositories import InMemoryRepository

        service = RecommendationService.build(repository=InMemoryRepository.from_artifacts())
        _SHARED["service"], _SHARED["runtime"] = service, RuntimeRecommender.build(service)
    return _SHARED["service"], _SHARED["runtime"]


class NotificationBehaviour:
    """Mixin; subclasses provide make_repository()."""

    def setUp(self):
        self.service, self.runtime = shared()
        self.repo = self.make_repository()
        self.app = create_app("testing", service=self.service, marketplace_repository=self.repo,
                              runtime_recommender=self.runtime)
        self.mail = self.app.extensions["hirewise_mailer"].backend

    # ---- helpers ---------------------------------------------------------------------------------
    def token(self, client, path="/dashboard"):
        return TOKEN.search(client.get(path).get_data(as_text=True)).group(1)

    def post(self, client, path, data=None, token_path="/dashboard", **kwargs):
        return client.post(path, data={**(data or {}), "csrf_token": self.token(client, token_path)}, **kwargs)

    def user(self, email, role="client", worker_id=None, verified=True):
        """A user created directly (like scripts/create_demo_account.py), logged in on a fresh client."""
        user_id = self.repo.create_user(email, hash_password(PASSWORD), role, email.split("@")[0], worker_id=worker_id,
                                        verified=verified)
        c = self.app.test_client()
        c.post("/login", data={"email": email, "password": PASSWORD,
                               "csrf_token": TOKEN.search(c.get("/login").get_data(as_text=True)).group(1)})
        return c, user_id

    def demo_on(self, email=OPERATOR_EMAIL):
        self.app.config.update(DEMO_MODE=True, DEMO_OPERATOR_EMAIL=email)

    def new_request(self, client, title="Logo for a bakery"):
        response = self.post(client, "/client/requests/new", {"title": title, "category": "Logo Design",
                                                              "skills": "Logo Design, Adobe Illustrator",
                                                              "pay_type": "fixed", "budget": "150"},
                             token_path="/client/requests/new")
        return int(response.headers["Location"].rstrip("/").split("/")[-1])

    def recommended(self, request_id):
        req = self.repo.get_request(request_id)
        rr = RuntimeRequest(request_id, req["client_user_id"], req["category"], tuple(req["skills"]), req["price_tier"],
                            req["location_requirement"], None)
        return [r["worker"]["worker_id"] for r in self.runtime.recommend(rr, 10)["recommendations"]]

    def book(self, client, request_id, worker_id):
        return self.post(client, f"/client/requests/{request_id}/book", {"worker_id": worker_id},
                         token_path=f"/client/requests/{request_id}")

    def act(self, client, booking_id, action):
        return self.post(client, f"/bookings/{booking_id}/{action}", {"return_to": "/bookings"}, token_path="/bookings")

    def notes(self, user_id):
        return self.repo.list_notifications(user_id, limit=100)

    def scenario(self, claimed=True, client_email="client@example.com", title="Logo for a bakery"):
        """A verified client with a request; the top recommended worker, linked to an account if `claimed`."""
        client, client_id = self.user(client_email)
        request_id = self.new_request(client, title)
        worker_id = self.recommended(request_id)[0]
        worker = worker_user = None
        if claimed:
            worker, worker_user = self.user(f"w{worker_id.lower()}@example.com", "worker", worker_id=worker_id)
        return client, client_id, request_id, worker_id, worker, worker_user

    # ---- notifications are written with the change -----------------------------------------------------
    def test_booking_notifies_the_linked_worker_with_an_email_after_commit(self):
        delivered_with_booking = []
        original = self.mail.deliver

        def deliver(message):  # at delivery time the booking must already be committed
            delivered_with_booking.append(bool(self.repo.list_bookings(worker_id=worker_id)))
            original(message)

        self.mail.deliver = deliver
        client, client_id, request_id, worker_id, worker, worker_user = self.scenario()
        sent_before = len(self.mail.messages)
        self.assertEqual(self.book(client, request_id, worker_id).status_code, 302)
        booking = self.repo.list_bookings(request_id=request_id)[0]
        [note] = self.notes(worker_user)
        self.assertEqual((note["kind"], note["booking_id"], note["read_at"]), ("booking_requested", booking["booking_id"], None))
        self.assertIn("Logo for a bakery", note["message"])
        self.assertEqual(self.notes(client_id), [])  # never the actor
        [email] = self.mail.messages[sent_before:]
        self.assertEqual(email["To"], f"w{worker_id.lower()}@example.com")
        self.assertEqual(email["Subject"], "New booking request on HireWise")
        body = email.get_content()
        self.assertIn(f"#{booking['booking_id']}", body)
        self.assertIn(worker_id, body)
        self.assertNotIn("bakery", body + email["Subject"])  # no request title in email
        self.assertNotIn("client", email["To"])
        self.assertEqual(delivered_with_booking[-1], True)

    def test_full_lifecycle_notifies_the_other_party_each_time(self):
        client, client_id, request_id, worker_id, worker, worker_user = self.scenario()
        self.book(client, request_id, worker_id)
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.act(worker, booking_id, "accept")
        self.act(client, booking_id, "complete")
        self.post(client, f"/bookings/{booking_id}/review", {"rating": "4", "review_text": "Good."},
                  token_path=f"/bookings/{booking_id}/review")
        self.assertEqual([n["kind"] for n in reversed(self.notes(worker_user))],
                         ["booking_requested", "booking_completed", "review_received"])
        self.assertEqual([n["kind"] for n in self.notes(client_id)], ["booking_accepted"])
        self.assertIn("4/5", self.notes(worker_user)[0]["message"])
        # a second booking: declined by the worker, then another cancelled by the client
        request_2 = self.new_request(client, "Second logo")
        self.book(client, request_2, worker_id)
        b2 = self.repo.list_bookings(request_id=request_2)[0]["booking_id"]
        self.act(worker, b2, "decline")
        self.assertEqual(self.notes(client_id)[0]["kind"], "booking_declined")
        request_3 = self.new_request(client, "Third logo")
        self.book(client, request_3, worker_id)
        b3 = self.repo.list_bookings(request_id=request_3)[0]["booking_id"]
        self.act(client, b3, "cancel")
        self.assertEqual(self.notes(worker_user)[0]["kind"], "booking_cancelled")

    def test_failed_or_stale_actions_create_nothing(self):
        client, client_id, request_id, worker_id, worker, worker_user = self.scenario()
        self.book(client, request_id, worker_id)
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.act(worker, booking_id, "accept")
        before = (len(self.notes(worker_user)), len(self.notes(client_id)))
        self.act(worker, booking_id, "accept")      # repeated: invalid now
        self.act(worker, booking_id, "complete")    # not the worker's action
        self.book(client, request_id, worker_id)    # already has an active booking
        self.assertEqual((len(self.notes(worker_user)), len(self.notes(client_id))), before)

    def test_a_notification_that_cannot_be_stored_rolls_the_change_back(self):
        _, client_id, request_id, worker_id, _, worker_user = self.scenario()
        too_long = {"user_id": worker_user, "kind": "booking_requested", "message": "x" * (NOTIFICATION_MESSAGE_MAX_LENGTH + 1)}
        with self.assertRaises(ValueError):
            self.repo.create_booking(request_id, client_id, worker_id, notification=too_long)
        self.assertEqual(self.repo.list_bookings(request_id=request_id), [])  # same transaction: no booking
        booking_id = self.repo.create_booking(request_id, client_id, worker_id)
        bad_kind = {"user_id": client_id, "kind": "nope", "message": "m"}
        with self.assertRaises(ValueError):
            self.repo.transition_booking(booking_id, "pending", "accepted", notification=bad_kind)
        self.assertEqual(self.repo.get_booking(booking_id)["status"], "pending")
        self.assertEqual(self.notes(worker_user) + self.notes(client_id), [])

    def test_a_mail_failure_never_blocks_the_booking(self):
        client, client_id, request_id, worker_id, worker, worker_user = self.scenario()
        self.mail.fail = True
        with self.assertLogs("src.app.marketplace.mailer", level="WARNING") as logs:
            response = self.book(client, request_id, worker_id)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(self.repo.list_bookings(request_id=request_id)), 1)
        self.assertEqual(len(self.notes(worker_user)), 1)
        text = " ".join(logs.output)
        self.assertIn("kind=notification_booking_requested", text)
        self.assertNotIn("@", text)  # no address in the log
        self.mail.fail = False
        # an unexpected error while preparing the email is swallowed too
        self.app.extensions["hirewise_mailer"].build = lambda *a, **k: 1 / 0
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        with self.assertLogs("src.app.marketplace.notify", level="WARNING") as logs:
            self.assertEqual(self.act(worker, booking_id, "accept").status_code, 302)
        self.assertEqual(self.repo.get_booking(booking_id)["status"], "accepted")
        self.assertIn("ZeroDivisionError", " ".join(logs.output))

    def test_email_skip_rules_apply_but_the_notification_is_stored(self):
        client, _ = self.user("client@example.com")
        request_id = self.new_request(client)
        worker_id = self.recommended(request_id)[0]
        _, worker_user = self.user("worker@demo.local", "worker", worker_id=worker_id)  # skipped domain
        sent_before = len(self.mail.messages)
        self.book(client, request_id, worker_id)
        self.assertEqual(len(self.notes(worker_user)), 1)
        self.assertEqual(len(self.mail.messages), sent_before)

    def test_unclaimed_worker_with_demo_mode_off_notifies_nobody(self):
        client, client_id, request_id, worker_id, _, _ = self.scenario(claimed=False)
        operator, operator_id = self.user(OPERATOR_EMAIL, "worker")  # exists, but demo mode is off
        self.book(client, request_id, worker_id)
        self.assertEqual(self.notes(operator_id), [])
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.assertEqual(self.act(operator, booking_id, "accept").status_code, 404)
        page = client.get(f"/workers/{worker_id}").get_data(as_text=True)
        self.assertIn("This is a simulated research profile, so it won", page)

    # ---- demo mode --------------------------------------------------------------------------------------
    def test_demo_operator_handles_unclaimed_workers_end_to_end(self):
        client, client_id, request_id, worker_id, _, _ = self.scenario(claimed=False)
        operator, operator_id = self.user(OPERATOR_EMAIL, "worker")
        self.demo_on()
        sent_before = len(self.mail.messages)
        self.book(client, request_id, worker_id)
        [note] = self.notes(operator_id)
        self.assertIn(f"simulated worker {worker_id}", note["message"])
        [email] = self.mail.messages[sent_before:]
        self.assertEqual((email["To"], email["Subject"]), (OPERATOR_EMAIL, f"[Demo] New booking for simulated worker {worker_id}"))
        dashboard = operator.get("/dashboard").get_data(as_text=True)
        self.assertIn("Demo account", dashboard)
        self.assertIn(worker_id, dashboard)
        self.assertIn("Demo account", operator.get("/bookings").get_data(as_text=True))
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.assertEqual(self.act(operator, booking_id, "accept").status_code, 302)
        self.assertEqual(self.repo.get_booking(booking_id)["status"], "accepted")
        self.assertEqual(self.notes(client_id)[0]["kind"], "booking_accepted")
        self.assertEqual(self.notes(operator_id)[0]["kind"], "booking_requested")  # the actor gets nothing new
        # honesty notes for the client
        detail = client.get(f"/client/requests/{request_id}").get_data(as_text=True)
        self.assertIn("Simulated profile", detail)
        self.assertIn("Responses are handled by the HireWise demo account.", detail)
        self.assertIn("Responses are handled by the HireWise demo account.",
                      client.get(f"/workers/{worker_id}").get_data(as_text=True))

    def test_demo_operator_is_refused_on_claimed_workers_and_has_no_client_powers(self):
        client, client_id, request_id, worker_id, worker, worker_user = self.scenario(claimed=True)
        operator, operator_id = self.user(OPERATOR_EMAIL, "worker")
        self.demo_on()
        self.book(client, request_id, worker_id)
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.assertEqual(self.notes(operator_id), [])                          # the linked worker is notified instead
        self.assertEqual(len(self.notes(worker_user)), 1)
        self.assertEqual(self.act(operator, booking_id, "accept").status_code, 404)
        self.assertEqual(self.repo.get_booking(booking_id)["status"], "pending")
        self.assertNotIn(worker_id, operator.get("/dashboard").get_data(as_text=True))
        self.assertEqual(operator.get("/client/requests/new").status_code, 403)  # no client powers
        detail = client.get(f"/client/requests/{request_id}").get_data(as_text=True)
        self.assertNotIn("Responses are handled", detail)  # claimed worker: no honesty note in the bookings panel

    def test_misconfigured_demo_mode_warns_once_and_stays_off(self):
        client, _, request_id, worker_id, _, _ = self.scenario(claimed=False)
        self.demo_on("nobody@example.com")
        with self.assertLogs("src.app.marketplace.notify", level="WARNING") as logs:
            self.book(client, request_id, worker_id)
            client.get(f"/workers/{worker_id}")
        self.assertEqual(sum("demo mode disabled" in line for line in logs.output), 1)
        self.assertNotIn("nobody@example.com", " ".join(logs.output))
        self.assertIn("so it won", client.get(f"/workers/{worker_id}").get_data(as_text=True))
        # a client account as "operator" is refused too
        self.user("client2@example.com")
        self.demo_on("client2@example.com")
        with self.app.app_context():
            from src.app.marketplace import notify
            self.assertIsNone(notify.demo_operator(self.repo))

    def test_operator_daily_email_cap(self):
        operator, operator_id = self.user(OPERATOR_EMAIL, "worker")
        self.demo_on()
        self.app.config["DEMO_OPERATOR_EMAIL_DAILY_CAP"] = 2
        client, _ = self.user("client@example.com")
        sent_before = len(self.mail.messages)
        with self.assertLogs("src.app.marketplace.notify", level="WARNING") as logs:
            for i in range(4):
                request_id = self.new_request(client, f"Logo {i}")
                worker_id = next(w for w in self.recommended(request_id) if not self.repo.claimed_worker_ids([w]))
                self.book(client, request_id, worker_id)
        self.assertEqual(len(self.notes(operator_id)), 4)                     # in-app continues above the cap
        self.assertEqual(len([m for m in self.mail.messages[sent_before:] if m["To"] == OPERATOR_EMAIL]), 2)
        self.assertEqual(sum("daily email cap reached" in line for line in logs.output), 1)

    # ---- bell, page, JSON endpoint, ownership ------------------------------------------------------------
    def test_bell_count_json_endpoint_and_marking(self):
        client, client_id, request_id, worker_id, worker, worker_user = self.scenario()
        self.book(client, request_id, worker_id)
        page = worker.get("/dashboard").get_data(as_text=True)
        self.assertIn('aria-label="Notifications, 1 unread"', page)
        self.assertIn('data-poll-seconds="30"', page)
        response = worker.get("/notifications/unread-count")
        self.assertEqual((response.status_code, response.get_json()), (200, {"unread": 1}))
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        anonymous = self.app.test_client().get("/notifications/unread-count")
        self.assertEqual((anonymous.status_code, anonymous.get_json()), (401, {"error": "login_required"}))
        self.assertEqual(anonymous.headers["Cache-Control"], "no-store")
        self.assertIn("Logo for a bakery", worker.get("/notifications").get_data(as_text=True))
        [note] = self.notes(worker_user)
        # without a CSRF token: refused, nothing changes
        self.assertEqual(worker.post(f"/notifications/{note['notification_id']}/read").status_code, 400)
        self.assertEqual(worker.post("/notifications/read-all").status_code, 400)
        self.assertIsNone(self.notes(worker_user)[0]["read_at"])
        # with the token
        self.assertEqual(self.post(worker, f"/notifications/{note['notification_id']}/read", token_path="/notifications")
                         .status_code, 302)
        self.assertIsNotNone(self.notes(worker_user)[0]["read_at"])
        self.assertEqual(worker.get("/notifications/unread-count").get_json(), {"unread": 0})
        self.assertIn('aria-label="Notifications"', worker.get("/dashboard").get_data(as_text=True))
        # mark all read
        booking_id = self.repo.list_bookings(request_id=request_id)[0]["booking_id"]
        self.act(worker, booking_id, "accept")
        self.act(client, booking_id, "complete")
        self.assertEqual(self.repo.count_notifications(worker_user, unread_only=True), 1)
        self.assertEqual(self.post(worker, "/notifications/read-all", token_path="/notifications").status_code, 302)
        self.assertEqual(self.repo.count_notifications(worker_user, unread_only=True), 0)
        self.assertEqual(self.post(worker, "/notifications/read-all", token_path="/notifications").status_code, 302)

    def test_another_users_notification_is_404(self):
        client, client_id, request_id, worker_id, worker, worker_user = self.scenario()
        self.book(client, request_id, worker_id)
        [note] = self.notes(worker_user)
        response = self.post(client, f"/notifications/{note['notification_id']}/read", token_path="/notifications")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.post(client, "/notifications/999999/read", token_path="/notifications").status_code, 404)
        self.assertIsNone(self.notes(worker_user)[0]["read_at"])
        self.assertNotIn("Logo for a bakery", client.get("/notifications").get_data(as_text=True))
        self.post(client, "/notifications/read-all", token_path="/notifications")  # only the client's own
        self.assertIsNone(self.notes(worker_user)[0]["read_at"])
        self.assertEqual(self.app.test_client().get("/notifications").status_code, 302)  # login required

    def test_unverified_users_can_read_notifications(self):
        c, user_id = self.user("unverified@example.com", verified=False)
        self.assertEqual(c.get("/notifications").status_code, 200)
        self.assertEqual(c.get("/notifications/unread-count").get_json(), {"unread": 0})

    def test_new_pages_keep_the_strict_csp(self):
        client, client_id, request_id, worker_id, worker, worker_user = self.scenario(claimed=False)
        self.book(client, request_id, worker_id)
        for path in ("/notifications", f"/client/requests/{request_id}", f"/workers/{worker_id}", "/dashboard"):
            response = client.get(path)
            html = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
            self.assertNotIn(' style="', html, path)
            self.assertNotRegex(html, r"<script(?![^>]*\bsrc=)[^>]*>", path)
            self.assertNotRegex(html, r"\son[a-z]+=", path)
            self.assertIn("js/notifications.js", html)
        anonymous = self.app.test_client().get("/").get_data(as_text=True)
        self.assertNotIn("data-notification-bell", anonymous)
        self.assertNotIn("js/notifications.js", anonymous)


@unittest.skipUnless(HAVE_DATA, "processed data / Day 8 results absent")
class InMemoryNotificationTests(NotificationBehaviour, unittest.TestCase):
    def make_repository(self):
        from src.db.marketplace import InMemoryMarketplaceRepository

        if "memory_reference" not in _SHARED:
            _SHARED["memory_reference"] = InMemoryMarketplaceRepository.from_artifacts()
        ref = _SHARED["memory_reference"]
        return InMemoryMarketplaceRepository(ref.categories(), ref.skills(), ref._clients, ref._workers)


@unittest.skipUnless(HAVE_DATA and os.environ.get("HIREWISE_RUN_POSTGRES_TESTS") == "1",
                     "opt-in live PostgreSQL test: set HIREWISE_RUN_POSTGRES_TESTS=1")
class PostgresNotificationTests(NotificationBehaviour, unittest.TestCase):
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

    def test_database_enforces_notification_constraints(self):
        from sqlalchemy import insert, update
        from sqlalchemy.exc import IntegrityError
        from src.db.schema import notifications

        user_id = self.repo.create_user("db@example.com", "h", "client", "db")
        bad = [dict(user_id=user_id, kind="nope", message="m"), dict(user_id=user_id, kind="booking_accepted", message=""),
               dict(user_id=user_id, kind="booking_accepted", message="x" * 501),
               dict(user_id=987654321, kind="booking_accepted", message="m")]
        for row in bad:
            with self.assertRaises(IntegrityError, msg=row):
                with self.engine.begin() as conn:
                    conn.execute(insert(notifications).values(**row))
        with self.engine.begin() as conn:
            nid = conn.execute(insert(notifications).values(user_id=user_id, kind="booking_accepted", message="m")
                               .returning(notifications.c.notification_id)).scalar_one()
        with self.assertRaises(IntegrityError):  # read_at before created_at
            with self.engine.begin() as conn:
                conn.execute(update(notifications).where(notifications.c.notification_id == nid)
                             .values(read_at=notifications.c.created_at - __import__("datetime").timedelta(days=1)))


if __name__ == "__main__":
    unittest.main()
