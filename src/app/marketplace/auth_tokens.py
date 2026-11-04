"""Single-use, purpose-bound, expiring tokens for email verification and password reset (Milestone 1).

No custom cryptography:
- a raw token is `secrets.token_urlsafe(32)` (256 bits from the OS CSPRNG);
- only its SHA-256 hex digest is stored (auth_tokens.token_hash); the raw token exists only in the
  emailed link and, briefly, in the requester's own signed session while they confirm;
- tokens are bound to one purpose ("verify_email" / "reset_password"), so one can never be used for the other;
- expiry is checked against timezone-aware UTC times; consumption is atomic in the repository
  (src/db/marketplace.py), so a token works at most once even under concurrent requests;
- issuing a token retires the user's older unused tokens of the same purpose.

Raw tokens are never logged. Lookups of malformed input never touch the database.
"""
from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone

from src.db.marketplace import TOKEN_EXPIRED, TOKEN_INVALID, TOKEN_USED, TOKEN_VALID  # noqa: F401  (re-exported)

VERIFY_EMAIL = "verify_email"
RESET_PASSWORD = "reset_password"
TOKEN_BYTES = 32
# token_urlsafe(32) gives 43 characters of [A-Za-z0-9_-]; anything else cannot be one of ours.
_TOKEN_FORMAT = re.compile(r"^[A-Za-z0-9_-]{43}$")


def utcnow() -> datetime:
    """Current time (timezone-aware UTC). A function so tests can move the clock."""
    return datetime.now(timezone.utc)


def new_raw_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def well_formed(raw: str | None) -> bool:
    return bool(raw) and _TOKEN_FORMAT.fullmatch(raw) is not None


def issue(repo, user_id: int, purpose: str, ttl: timedelta) -> str:
    """Create and store a token for `user_id`; returns the RAW token (for the email link only)."""
    now = utcnow()
    raw = new_raw_token()
    repo.issue_auth_token(user_id, purpose, hash_token(raw), now + ttl, now)
    return raw


def state(repo, raw: str | None, purpose: str) -> str:
    """Valid / expired / used / invalid -- without consuming the token."""
    if not well_formed(raw):
        return TOKEN_INVALID
    return repo.auth_token_state(hash_token(raw), purpose, utcnow())


def verify_email(repo, raw: str | None) -> tuple[str, int | None]:
    if not well_formed(raw):
        return TOKEN_INVALID, None
    return repo.verify_email_with_token(hash_token(raw), utcnow())


def reset_password(repo, raw: str | None, password_hash: str) -> tuple[str, int | None]:
    if not well_formed(raw):
        return TOKEN_INVALID, None
    return repo.reset_password_with_token(hash_token(raw), password_hash, utcnow())
