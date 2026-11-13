"""Notification glue (pre-deployment Milestone 3): looks things up, then applies the pure rules in notifications.py.

- `demo_operator(repo)`: the demo operator account when demo mode is EFFECTIVELY on, else None. A misconfiguration
  (operator email missing, or matching no verified worker-role account without a linked profile) logs one clear
  warning per process and is treated as demo mode off; it never stops the app.
- `prepare(...)`: the recipient and the row to store in the same transaction as the booking/review change.
- `email_after_commit(...)`: called only after that transaction committed. Never raises: an email problem can
  never roll back or block the change. Logs name the event kind and an exception type only -- never addresses,
  tokens, links or message content.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import current_app, url_for

from src.app.marketplace import notifications as rules

logger = logging.getLogger(__name__)

_WARNINGS_KEY = "hirewise_notification_warnings"


def _warn_once(key: str, message: str, *args) -> None:
    seen = current_app.extensions.setdefault(_WARNINGS_KEY, set())
    if key not in seen:
        seen.add(key)
        logger.warning(message, *args)


def demo_operator(repo) -> dict | None:
    config = current_app.config
    if not config.get("DEMO_MODE"):
        return None
    email = (config.get("DEMO_OPERATOR_EMAIL") or "").strip().lower()
    account = repo.get_user_by_email(email) if email else None
    problem = rules.demo_operator_problem(True, email, account)
    if problem is not None:
        _warn_once(f"operator:{problem}", "demo mode disabled: %s", problem)
        return None
    return account


def demo_active(repo) -> bool:
    try:
        return demo_operator(repo) is not None
    except Exception as exc:  # a display hint must never break a page
        logger.warning("could not determine demo mode (error=%s)", type(exc).__name__)
        return False


def prepare(kind: str, repo, *, booking: dict, request_title: str | None, actor_user_id: int,
            operator: dict | None, rating: int | None = None):
    """(recipient, payload) for `kind` on `booking` (payload None: nobody to notify)."""
    client = repo.get_user(booking["client_user_id"]) if kind in rules.CLIENT_SIDE_KINDS else None
    linked = repo.get_linked_worker_account(booking["worker_id"]) if kind in rules.WORKER_SIDE_KINDS else None
    recipient = rules.recipient_for(kind, client_account=client, linked_account=linked, demo_operator=operator,
                                    actor_user_id=actor_user_id)
    return recipient, rules.notification_payload(kind, recipient, worker_id=booking["worker_id"],
                                                 request_title=request_title, rating=rating)


def _start_of_utc_day(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def email_after_commit(repo, kind: str, recipient, *, booking_id: int, worker_id: str) -> None:
    """Queue the email for a committed notification. Never raises."""
    try:
        if recipient is None:
            return
        mailer = current_app.extensions.get("hirewise_mailer")
        if mailer is None:
            logger.info("notification email not sent: email delivery is not configured (kind=%s)", kind)
            return
        if not rules.email_allowed(recipient, current_app.config.get("NOTIFICATION_EMAIL_SKIP_DOMAINS")):
            logger.info("notification email skipped: unverified address or skipped domain (kind=%s)", kind)
            return
        if recipient.is_demo_operator:
            cap = int(current_app.config.get("DEMO_OPERATOR_EMAIL_DAILY_CAP", 0))
            sent_today = repo.count_notifications_since(recipient.user_id, _start_of_utc_day())
            send, warn = rules.operator_email_decision(sent_today, cap)
            if warn:
                logger.warning("demo operator daily email cap reached (%s per UTC day); in-app notifications continue "
                               "but no more demo emails are sent today", cap)
            if not send:
                return
        link = mailer.link(url_for("marketplace.bookings_page"))
        message = mailer.build(recipient.email,
                               rules.email_subject(kind, worker_id=worker_id, for_demo_operator=recipient.is_demo_operator),
                               rules.email_body(kind, booking_id=booking_id, worker_id=worker_id, link=link,
                                                for_demo_operator=recipient.is_demo_operator))
        mailer.send_later(message, f"notification_{kind}")
    except Exception as exc:
        logger.warning("notification email not sent (kind=%s, error=%s)", kind, type(exc).__name__)


def unread_count(repo, user_id: int) -> int | None:
    """For the header bell; None if it cannot be read (the page still renders)."""
    try:
        return repo.count_notifications(user_id, unread_only=True)
    except Exception as exc:
        logger.warning("unread notification count unavailable (error=%s)", type(exc).__name__)
        return None
