"""Marketplace business rules (Day 12): pure functions, no I/O.

Booking state machine (who may do what):

    pending  --accept-->   accepted   (the booked worker)
    pending  --decline-->  declined   (the booked worker)
    pending  --cancel-->   cancelled  (the owning client)
    accepted --complete--> completed  (the owning client)
    accepted --cancel-->   cancelled  (the owning client)

declined, cancelled and completed are terminal. A completed booking can be
reviewed once, by its client.

Demo mode (Milestone 3): the demo operator -- a worker-role account with no
linked catalogue profile -- may act in the WORKER role on bookings of
catalogue workers that have no linked account ("unclaimed"). Never on a
claimed worker's booking, and never in the client role. The caller passes
both facts in, so this stays pure.
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


def check_booking_actor(booking: dict, user: dict, *, is_demo_operator: bool = False, worker_claimed: bool = True) -> str:
    """Role in which `user` may act on `booking`; raises NotAuthorized.
    A client acts only on bookings for their own requests; a worker only on
    bookings of the imported profile their account is linked to. The demo
    operator (`is_demo_operator`: demo mode is effectively on and `user` is
    its account) acts as the worker only when the booked worker has no linked
    account (`worker_claimed` False)."""
    if user["role"] == "client" and booking["client_user_id"] == user["user_id"]:
        return "client"
    if user["role"] == "worker" and user.get("worker_id") and booking["worker_id"] == user["worker_id"]:
        return "worker"
    if is_demo_operator and user["role"] == "worker" and not user.get("worker_id") and not worker_claimed:
        return "worker"
    raise NotAuthorized("not your booking")


def profile_edit_fields(user: dict, has_runtime_profile: bool) -> frozenset[str]:
    """What an account may edit on its own profile (Milestone 4). Email and password are never part of it.

    - client: display name and location (client_country);
    - worker linked to an imported catalogue profile (users.worker_id): display name ONLY -- the location, worker
      profile, bio and credentials come from (or would sit beside) research data;
    - self-registered worker (has a runtime profile): display name, location, the worker profile (with bio) and
      credentials;
    - any other worker account (the demo operator: no profile, no link): display name and location.
    """
    if user["role"] == "client":
        return frozenset({"display_name", "client_country"})
    if user.get("worker_id"):
        return frozenset({"display_name"})
    if has_runtime_profile:
        return frozenset({"display_name", "worker_country", "worker_profile", "credentials"})
    return frozenset({"display_name", "worker_country"})


def can_review(booking: dict, user: dict, existing_review) -> bool:
    return (user["role"] == "client" and booking["client_user_id"] == user["user_id"]
            and booking["status"] == "completed" and existing_review is None)
