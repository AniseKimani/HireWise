"""Notification rules (pre-deployment Milestone 3): pure functions, no I/O, in the style of rules.py.

Who is notified (never the person who acted):

    booking_requested    client books              -> worker side
    booking_accepted     worker accepts            -> the client
    booking_declined     worker declines           -> the client
    booking_cancelled    client cancels            -> worker side
    booking_completed    client marks complete     -> worker side
    review_received      client reviews            -> worker side

"Worker side" is the account linked to the booked catalogue worker (`users.worker_id`). A catalogue worker without
a linked account is "unclaimed": its notifications go to the demo operator when demo mode is effectively on, and
otherwise to nobody.

The stored in-app message is one plain sentence. A client-entered request title is shortened to
TITLE_MAX_LENGTH characters, so a message can never exceed the database's 500-character limit and a long title can
never block a booking. Emails carry only the booking number and the worker ID: no request titles, no names.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.db.schema import NOTIFICATION_KINDS, NOTIFICATION_MESSAGE_MAX_LENGTH

TITLE_MAX_LENGTH = 120
ELLIPSIS = "…"
DEMO_SUBJECT_PREFIX = "[Demo] "

BOOKING_REQUESTED, BOOKING_ACCEPTED, BOOKING_DECLINED, BOOKING_CANCELLED, BOOKING_COMPLETED, REVIEW_RECEIVED = \
    NOTIFICATION_KINDS
EVENT_FOR_ACTION = {"accept": BOOKING_ACCEPTED, "decline": BOOKING_DECLINED, "cancel": BOOKING_CANCELLED,
                    "complete": BOOKING_COMPLETED}
CLIENT_SIDE_KINDS = frozenset({BOOKING_ACCEPTED, BOOKING_DECLINED})
WORKER_SIDE_KINDS = frozenset({BOOKING_REQUESTED, BOOKING_CANCELLED, BOOKING_COMPLETED, REVIEW_RECEIVED})
assert CLIENT_SIDE_KINDS | WORKER_SIDE_KINDS == set(NOTIFICATION_KINDS) and not CLIENT_SIDE_KINDS & WORKER_SIDE_KINDS

HONESTY_NOTE_DEMO = "This is a simulated research profile. Responses are handled by the HireWise demo account."
HONESTY_NOTE_NO_DEMO = "This is a simulated research profile, so it won't respond."
SIMULATED_LABEL = "Simulated profile"


@dataclass(frozen=True)
class Recipient:
    user_id: int
    email: str
    verified: bool
    is_demo_operator: bool = False


def _recipient(account: dict | None, is_demo_operator: bool = False) -> Recipient | None:
    if account is None:
        return None
    return Recipient(user_id=account["user_id"], email=account["email"],
                     verified=account.get("email_verified_at") is not None, is_demo_operator=is_demo_operator)


# ---------------------------------------------------------------------------
# Demo operator
# ---------------------------------------------------------------------------
def demo_operator_problem(enabled: bool, configured_email: str | None, account: dict | None) -> str | None:
    """Why demo mode cannot be used (a message naming settings, never values), or None when `account` is a valid
    operator: a verified worker-role account with no linked catalogue profile, found by the configured email."""
    if not enabled:
        return "demo mode is off"
    if not (configured_email or "").strip():
        return "HIREWISE_DEMO_MODE is on but HIREWISE_DEMO_OPERATOR_EMAIL is not set"
    if account is None:
        return "HIREWISE_DEMO_OPERATOR_EMAIL matches no account"
    if account.get("role") != "worker":
        return "the HIREWISE_DEMO_OPERATOR_EMAIL account is not a worker account"
    if account.get("worker_id"):
        return "the HIREWISE_DEMO_OPERATOR_EMAIL account is linked to a catalogue worker, so it cannot be the operator"
    if account.get("email_verified_at") is None:
        return "the HIREWISE_DEMO_OPERATOR_EMAIL account's email address is not verified"
    return None


# ---------------------------------------------------------------------------
# Recipients
# ---------------------------------------------------------------------------
def worker_side_recipient(linked_account: dict | None, demo_operator: dict | None) -> Recipient | None:
    """The account linked to the booked worker; if there is none, the demo operator (pass it only when demo mode is
    effectively on); otherwise nobody."""
    if linked_account is not None:
        return _recipient(linked_account)
    return _recipient(demo_operator, is_demo_operator=True)


def recipient_for(kind: str, *, client_account: dict | None, linked_account: dict | None,
                  demo_operator: dict | None, actor_user_id: int | None) -> Recipient | None:
    """The one recipient of `kind`, never the actor."""
    if kind in CLIENT_SIDE_KINDS:
        recipient = _recipient(client_account)
    elif kind in WORKER_SIDE_KINDS:
        recipient = worker_side_recipient(linked_account, demo_operator)
    else:
        raise ValueError(f"unknown notification kind {kind!r}")
    if recipient is not None and recipient.user_id == actor_user_id:
        return None
    return recipient


# ---------------------------------------------------------------------------
# In-app message
# ---------------------------------------------------------------------------
def shorten_title(title: str | None, limit: int = TITLE_MAX_LENGTH) -> str:
    """Whitespace collapsed; at most `limit` characters, ending with an ellipsis when shortened."""
    text = " ".join((title or "").split()) or "your request"
    return text if len(text) <= limit else text[: limit - len(ELLIPSIS)].rstrip() + ELLIPSIS


def notification_message(kind: str, *, worker_id: str, request_title: str | None, for_demo_operator: bool = False,
                         rating: int | None = None) -> str:
    title = f"“{shorten_title(request_title)}”"
    worker = f"simulated worker {worker_id}" if for_demo_operator else "you"
    messages = {
        BOOKING_REQUESTED: (f"New booking request for simulated worker {worker_id}: {title}." if for_demo_operator
                            else f"New booking request from a client: {title}."),
        BOOKING_ACCEPTED: f"Worker {worker_id} accepted your booking for {title}.",
        BOOKING_DECLINED: f"Worker {worker_id} declined your booking for {title}.",
        BOOKING_CANCELLED: f"The client cancelled the booking with {worker} for {title}.",
        BOOKING_COMPLETED: f"The client marked the booking with {worker} for {title} as completed.",
        REVIEW_RECEIVED: (f"The client left a {rating}/5 review for {worker} on {title}." if rating is not None
                          else f"The client left a review for {worker} on {title}."),
    }
    if kind not in messages:
        raise ValueError(f"unknown notification kind {kind!r}")
    message = messages[kind]
    assert 1 <= len(message) <= NOTIFICATION_MESSAGE_MAX_LENGTH  # guaranteed by shorten_title
    return message


def notification_payload(kind: str, recipient: Recipient | None, *, worker_id: str, request_title: str | None,
                         rating: int | None = None) -> dict | None:
    """The row the repository stores in the same transaction as the change (None: nobody to notify)."""
    if recipient is None:
        return None
    return {"user_id": recipient.user_id, "kind": kind,
            "message": notification_message(kind, worker_id=worker_id, request_title=request_title,
                                            for_demo_operator=recipient.is_demo_operator, rating=rating)}


# ---------------------------------------------------------------------------
# Email (sent after the commit; booking number and worker ID only)
# ---------------------------------------------------------------------------
def parse_skip_domains(raw) -> frozenset[str]:
    """Comma-separated (or an iterable of) domains, lower-cased, without a leading '@'."""
    items = raw.split(",") if isinstance(raw, str) else (raw or ())
    return frozenset(d.strip().lower().lstrip("@") for d in items if d and d.strip().lstrip("@"))


def email_allowed(recipient: Recipient | None, skip_domains) -> bool:
    """Only verified addresses, and never an address in a skipped domain (e.g. demo.local)."""
    if recipient is None or not recipient.verified or "@" not in recipient.email:
        return False
    return recipient.email.rsplit("@", 1)[1].lower() not in parse_skip_domains(skip_domains)


def operator_email_decision(sent_today_including_this: int, daily_cap: int) -> tuple[bool, bool]:
    """(send, warn) for the n-th demo-operator notification of the day (n counts this one). Emails stop above the cap;
    the in-app notification is stored regardless. `warn` is True exactly once: for the first one over the cap."""
    send = sent_today_including_this <= daily_cap
    return send, sent_today_including_this == daily_cap + 1


_EMAIL_EVENTS = {
    BOOKING_REQUESTED: ("New booking request on HireWise", "New booking for", "is a new booking request"),
    BOOKING_ACCEPTED: ("Your HireWise booking was accepted", None, "was accepted"),
    BOOKING_DECLINED: ("Your HireWise booking was declined", None, "was declined"),
    BOOKING_CANCELLED: ("A HireWise booking was cancelled", "Booking cancelled for", "was cancelled by the client"),
    BOOKING_COMPLETED: ("A HireWise booking was completed", "Booking completed for", "was marked as completed by the client"),
    REVIEW_RECEIVED: ("You received a review on HireWise", "New review for", "received a review"),
}


def email_subject(kind: str, *, worker_id: str, for_demo_operator: bool = False) -> str:
    plain, demo, _ = _EMAIL_EVENTS[kind]
    if for_demo_operator:
        return f"{DEMO_SUBJECT_PREFIX}{demo} simulated worker {worker_id}"
    return plain


def email_body(kind: str, *, booking_id: int, worker_id: str, link: str, for_demo_operator: bool = False) -> str:
    _, _, verb = _EMAIL_EVENTS[kind]
    who = f"simulated worker {worker_id} (HireWise demo account)" if for_demo_operator else f"worker {worker_id}"
    return f"Hello,\n\nBooking #{booking_id} with {who} {verb}.\n\n{link}\n\n-- HireWise\n"


def honesty_note(demo_active: bool) -> str:
    return HONESTY_NOTE_DEMO if demo_active else HONESTY_NOTE_NO_DEMO
