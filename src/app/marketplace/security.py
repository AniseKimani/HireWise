"""Authentication, session and CSRF helpers (Day 12). No custom crypto:
password hashing is Werkzeug's (scrypt by default), tokens come from
`secrets`, comparisons use `hmac.compare_digest`.

- Sessions: Flask's signed cookie session holding only `user_id`, the
  account's `auth_version` (Milestone 1) and the CSRF token. The session is
  cleared on login (no fixation) and logout, and is rejected once the
  account's auth_version has moved on (a password reset increments it).
- Email verification (Milestone 1): unverified accounts may log in and
  browse; `verified_required` guards the actions that create or change
  marketplace data.
- CSRF: a per-session random token, embedded in every form as a hidden
  field and checked on every state-changing request outside /api
  (all /api routes are read-only GETs).
- Login throttling: an in-process counter of recent failures per
  (client address, email); simple, per process, no extra infrastructure.
"""
from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections import deque
from functools import wraps

from flask import abort, current_app, flash, g, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

CSRF_SESSION_KEY = "csrf_token"
CSRF_FORM_FIELD = "csrf_token"
STATE_CHANGING_METHODS = ("POST", "PUT", "PATCH", "DELETE")

# A real hash of a random secret, so a login attempt for an unknown email
# costs the same as a wrong password (no account-existence timing oracle).
_DUMMY_HASH = generate_password_hash(secrets.token_urlsafe(16))


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(stored_hash: str | None, password: str) -> bool:
    if stored_hash is None:
        check_password_hash(_DUMMY_HASH, password)
        return False
    return check_password_hash(stored_hash, password)


# ---------------------------------------------------------------------------
# CSRF
# ---------------------------------------------------------------------------
def csrf_token() -> str:
    token = session.get(CSRF_SESSION_KEY)
    if token is None:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def protect_against_csrf() -> None:
    """before_request hook: reject state-changing requests without a valid token."""
    if request.method not in STATE_CHANGING_METHODS or request.path.startswith("/api/"):
        return
    expected = session.get(CSRF_SESSION_KEY)
    supplied = request.form.get(CSRF_FORM_FIELD, "")
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        abort(400, description="csrf")


# ---------------------------------------------------------------------------
# Sessions and guards
# ---------------------------------------------------------------------------
def login_user(user_id: int, auth_version: int = 0) -> None:
    session.clear()  # drop any pre-login session state, including its CSRF token
    session["user_id"] = user_id
    session["auth_version"] = auth_version
    session.permanent = True
    csrf_token()


def logout_user() -> None:
    session.clear()


def load_current_user() -> None:
    """before_request hook: g.user = the logged-in user's profile, or None.

    A session whose auth_version differs from the account's (the password was reset since it was created)
    is cleared, so the user must log in again. Sessions created before auth_version existed carry none and
    are compared as 0, so upgrading does not log anyone out."""
    g.user = None
    user_id = session.get("user_id")
    if user_id is not None:
        repo = current_app.extensions.get("hirewise_marketplace")
        user = repo.get_user(user_id) if repo is not None else None
        if user is None:
            session.clear()  # stale session (user removed)
        elif session.get("auth_version", 0) != user.get("auth_version", 0):
            session.clear()
            flash("Your password was changed, so you have been signed out. Please log in again.", "warning")
            user = None
        else:
            user["email_verified"] = user.get("email_verified_at") is not None
        g.user = user


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if g.get("user") is None:
            return redirect(url_for("marketplace.login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def role_required(role: str):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapper(*args, **kwargs):
            if g.user["role"] != role:
                abort(403)
            return view(*args, **kwargs)
        return wrapper
    return decorator


def verified_required(view):
    """Actions that create or change marketplace data need a verified email address. Use it inside
    login_required / role_required. Unverified users are sent to the verification page with an explanation
    (a GET form page is guarded the same way, so a form that cannot be submitted is never shown)."""
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if not g.user.get("email_verified"):
            flash("Verify your email address to post requests, book workers, manage bookings or leave reviews.", "warning")
            return redirect(url_for("account.verify_email"))
        return view(*args, **kwargs)
    return wrapper


def safe_next_url(target: str | None) -> str | None:
    """Only same-site relative paths are followed after login (no open redirect)."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return None


# ---------------------------------------------------------------------------
# Login throttling
# ---------------------------------------------------------------------------
class LoginRateLimiter:
    def __init__(self, max_failures: int, window_seconds: float):
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._failures: dict[tuple[str, str], deque] = {}
        self._lock = threading.Lock()

    def _recent(self, key, now):
        events = self._failures.setdefault(key, deque())
        while events and now - events[0] > self.window_seconds:
            events.popleft()
        return events

    def is_blocked(self, address: str, email: str) -> bool:
        with self._lock:
            return len(self._recent((address, email), time.monotonic())) >= self.max_failures

    def record_failure(self, address: str, email: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._recent((address, email), now).append(now)

    def reset(self, address: str, email: str) -> None:
        with self._lock:
            self._failures.pop((address, email), None)
