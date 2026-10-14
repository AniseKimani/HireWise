"""Marketplace business rules (Day 12): pure functions, no I/O.

Booking state machine (who may do what):

    pending  --accept-->   accepted   (the booked worker)
    pending  --decline-->  declined   (the booked worker)
    pending  --cancel-->   cancelled  (the owning client)
    accepted --complete--> completed  (the owning client)
    accepted --cancel-->   cancelled  (the owning client)

declined, cancelled and completed are terminal. A completed booking can be
reviewed once, by its client.
"""
from __future__ import annotations

TRANSITIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("pending", "accept"): ("accepted", "worker"),
    ("pending", "decline"): ("declined", "worker"),
    ("pending", "cancel"): ("cancelled", "client"),
    ("accepted", "complete"): ("completed", "client"),
    ("accepted", "cancel"): ("cancelled", "client"),
}
ACTIONS = ("accept", "decline", "cancel", "complete")
TERMINAL_STATUSES = ("declined", "cancelled", "completed")


class InvalidTransition(ValueError):
    pass


class NotAuthorized(PermissionError):
    pass


def next_status(current: str, action: str, actor_role: str) -> str:
    """Status after `action` by an actor of `actor_role`, or InvalidTransition."""
    target = TRANSITIONS.get((current, action))
    if target is None or target[1] != actor_role:
        raise InvalidTransition(f"cannot {action} a {current} booking as {actor_role}")
    return target[0]


def allowed_actions(status: str, actor_role: str) -> list[str]:
    return [action for (state, action), (_, role) in TRANSITIONS.items() if state == status and role == actor_role]


def check_booking_actor(booking: dict, user: dict) -> str:
    """Role in which `user` may act on `booking`; raises NotAuthorized.
    A client acts only on bookings for their own requests; a worker only on
    bookings of the imported profile their account is linked to."""
    if user["role"] == "client" and booking["client_user_id"] == user["user_id"]:
        return "client"
    if user["role"] == "worker" and user.get("worker_id") and booking["worker_id"] == user["worker_id"]:
        return "worker"
    raise NotAuthorized("not your booking")


def can_review(booking: dict, user: dict, existing_review) -> bool:
    return (user["role"] == "client" and booking["client_user_id"] == user["user_id"]
            and booking["status"] == "completed" and existing_review is None)
