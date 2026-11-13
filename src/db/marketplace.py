"""Runtime marketplace persistence (Day 12): users, runtime worker
profiles, client-created requests, bookings and application reviews; in-app
notifications since Milestone 3 (schema v5); profile editing and self-reported
worker credentials since Milestone 4 (schema v6).

Two implementations share one interface and one test suite:
- PostgresMarketplaceRepository -- the application's store; the database
  constraints in src/db/schema.py are authoritative.
- InMemoryMarketplaceRepository -- the same semantics in Python, for the
  database-free default tests only. Never used by the application factory.

Imported experimental tables are only READ here (categories, skills,
client/worker existence); all writes go to the runtime tables. Password
hashes are only returned by `get_credentials`, never by profile reads.

Notifications (Milestone 3) are written by the SAME transaction as the booking
or review change they describe: `create_booking`, `transition_booking` and
`create_review` take an optional `notification` ({"user_id", "kind",
"message"}); if it cannot be stored, the change itself is rolled back. Every
notification read or update is limited to one recipient's own rows.
"""
from __future__ import annotations

import itertools
import threading
from datetime import datetime, timezone
from functools import wraps

from sqlalchemy import and_, exists, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from src.db.repositories import DatabaseUnavailable
from src.db.schema import (
    ACTIVE_BOOKING_STATUSES, BIO_MAX_LENGTH, CREDENTIAL_DESCRIPTION_MAX_LENGTH, CREDENTIAL_KINDS,
    CREDENTIAL_ORGANISATION_MAX_LENGTH, CREDENTIAL_TITLE_MAX_LENGTH, CREDENTIAL_URL_MAX_LENGTH, CREDENTIAL_YEAR_DB_MAX,
    CREDENTIAL_YEAR_MIN, IDENTITY_PROVIDERS, NOTIFICATION_KINDS, NOTIFICATION_MESSAGE_MAX_LENGTH, TOKEN_PURPOSES,
    app_reviews, auth_tokens, bookings, categories, clients, notifications, runtime_request_skills, runtime_requests,
    runtime_worker_profiles, runtime_worker_skills, skills, user_email_verification, user_identities, users,
    worker_credentials, workers,
)

# Outcomes of looking up / consuming an auth token (src/app/marketplace/auth_tokens.py).
TOKEN_VALID, TOKEN_EXPIRED, TOKEN_USED, TOKEN_INVALID = "valid", "expired", "used", "invalid"


def _token_state(row, purpose: str, now: datetime) -> str:
    """State of a token row (or None) for `purpose`: a token of another purpose is simply invalid here."""
    if row is None or row["purpose"] != purpose:
        return TOKEN_INVALID
    if row["used_at"] is not None:
        return TOKEN_USED
    if row["expires_at"] <= now:
        return TOKEN_EXPIRED
    return TOKEN_VALID


class DuplicateEmail(ValueError):
    pass


class ProfileAlreadyLinked(ValueError):
    pass


class CredentialLimitReached(ValueError):
    """The worker already has the maximum number of credentials of this kind (Milestone 4)."""


class ActiveBookingExists(ValueError):
    pass


class DuplicateReview(ValueError):
    pass


class StaleBookingState(RuntimeError):
    """The booking changed since it was read (or never existed)."""


class IdentityAlreadyLinked(ValueError):
    """This external identity (provider, subject) already belongs to an account."""


class UserAlreadyHasIdentity(ValueError):
    """The account already has a (different) identity from this provider."""


USER_FIELDS = ("user_id", "email", "role", "display_name", "country", "client_id", "worker_id", "created_at", "auth_version")
REQUEST_FIELDS = ("request_id", "client_user_id", "title", "category", "pay_type", "budget", "hourly_low", "hourly_high",
                  "price_tier", "location_requirement", "created_at")
BOOKING_FIELDS = ("booking_id", "request_id", "client_user_id", "worker_id", "status", "created_at", "updated_at", "completed_at")

_CONSTRAINT_ERRORS = {
    "users_email_key": DuplicateEmail,
    "users_client_id_key": ProfileAlreadyLinked,
    "users_worker_id_key": ProfileAlreadyLinked,
    "uq_bookings_one_active_per_request": ActiveBookingExists,
    "app_reviews_pkey": DuplicateReview,
    "uq_user_identities_provider_subject": IdentityAlreadyLinked,
    "uq_user_identities_user_provider": UserAlreadyHasIdentity,
}
NOTIFICATION_FIELDS = ("notification_id", "user_id", "booking_id", "kind", "message", "created_at", "read_at")
IDENTITY_FIELDS = ("identity_id", "user_id", "provider", "subject", "email_at_link", "created_at", "last_login_at")
WORKER_PROFILE_COLUMNS = ("primary_category", "experience_level", "experience_years", "rate", "country")
CREDENTIAL_FIELDS = ("credential_id", "user_id", "kind", "title", "organisation", "year", "url", "description", "created_at")
CREDENTIAL_DATA = ("kind", "title", "organisation", "year", "url", "description")
_UNCHANGED = object()  # update_profile: leave users.country as it is


def _guard(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except IntegrityError as exc:
            constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
            mapped = _CONSTRAINT_ERRORS.get(constraint)
            if mapped is not None:
                raise mapped(constraint) from exc
            raise ValueError(f"constraint violated: {constraint}") from exc
        except SQLAlchemyError as exc:
            raise DatabaseUnavailable("application database unavailable") from exc
    return wrapper


class PostgresMarketplaceRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    # ---- reference data (imported, read-only) -------------------------------
    @_guard
    def categories(self) -> list[str]:
        with self.engine.connect() as conn:
            return [r[0] for r in conn.execute(select(categories.c.name).order_by(categories.c.name))]

    @_guard
    def skills(self) -> list[str]:
        with self.engine.connect() as conn:
            return [r[0] for r in conn.execute(select(skills.c.name).order_by(skills.c.name))]

    @_guard
    def imported_client_exists(self, client_id: str) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(clients).where(clients.c.client_id == client_id)).scalar_one() == 1

    @_guard
    def imported_worker_exists(self, worker_id: str) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(workers).where(workers.c.worker_id == worker_id)).scalar_one() == 1

    # ---- users ----------------------------------------------------------------
    @_guard
    def create_user(self, email, password_hash, role, display_name, country=None, client_id=None, worker_id=None,
                    worker_profile: dict | None = None, verified: bool = True) -> int:
        """`verified=False` for public self-registration (the email address is unproven); accounts created by an
        operator (scripts/create_demo_account.py) or test fixtures default to verified. The verification row is
        written in the same transaction as the user."""
        with self.engine.begin() as conn:
            return self._insert_user(conn, email, password_hash, role, display_name, country, client_id, worker_id,
                                     worker_profile, func.now() if verified else None)

    @staticmethod
    def _insert_user(conn, email, password_hash, role, display_name, country, client_id, worker_id, worker_profile,
                     verified_at) -> int:
        user_id = conn.execute(insert(users).values(
            email=email, password_hash=password_hash, role=role, display_name=display_name, country=country,
            client_id=client_id, worker_id=worker_id).returning(users.c.user_id)).scalar_one()
        conn.execute(insert(user_email_verification).values(user_id=user_id, verified_at=verified_at))
        if worker_profile is not None:
            conn.execute(insert(runtime_worker_profiles).values(
                user_id=user_id, **{k: worker_profile[k] for k in WORKER_PROFILE_COLUMNS}))
            if worker_profile["skills"]:
                conn.execute(insert(runtime_worker_skills), [{"user_id": user_id, "skill": s} for s in worker_profile["skills"]])
        return user_id

    @_guard
    def get_credentials(self, email: str) -> tuple[int, str] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(users.c.user_id, users.c.password_hash).where(users.c.email == email)).first()
        return (row.user_id, row.password_hash) if row else None

    @staticmethod
    def _user_query():
        return (select(*[users.c[f] for f in USER_FIELDS], user_email_verification.c.verified_at.label("email_verified_at"))
                .select_from(users.outerjoin(user_email_verification, user_email_verification.c.user_id == users.c.user_id)))

    @_guard
    def get_user(self, user_id: int) -> dict | None:
        """Profile fields plus `email_verified_at` (None = unverified, also when no verification row exists)."""
        with self.engine.connect() as conn:
            row = conn.execute(self._user_query().where(users.c.user_id == user_id)).first()
        return dict(row._mapping) if row else None

    @_guard
    def get_user_by_email(self, email: str) -> dict | None:
        """Same fields as get_user (no password hash), by the stored lower-case address."""
        with self.engine.connect() as conn:
            row = conn.execute(self._user_query().where(users.c.email == email)).first()
        return dict(row._mapping) if row else None

    @_guard
    def get_linked_worker_account(self, worker_id: str) -> dict | None:
        """The account linked to an imported catalogue worker ("claimed"), or None (at most one: users_worker_id_key)."""
        with self.engine.connect() as conn:
            row = conn.execute(self._user_query().where(users.c.worker_id == worker_id)).first()
        return dict(row._mapping) if row else None

    @_guard
    def claimed_worker_ids(self, worker_ids) -> set[str]:
        """The subset of `worker_ids` that have a linked account."""
        ids = sorted(set(worker_ids))
        if not ids:
            return set()
        with self.engine.connect() as conn:
            return {r[0] for r in conn.execute(select(users.c.worker_id).where(users.c.worker_id.in_(ids)))}

    # ---- account security (schema v3) ---------------------------------------------
    @_guard
    def issue_auth_token(self, user_id: int, purpose: str, token_hash: str, expires_at: datetime, now: datetime) -> None:
        """Store a new token hash; older unused tokens of the same purpose for this user stop working."""
        if purpose not in TOKEN_PURPOSES:
            raise ValueError(f"unknown token purpose {purpose!r}")
        with self.engine.begin() as conn:
            conn.execute(update(auth_tokens).where(and_(auth_tokens.c.user_id == user_id, auth_tokens.c.purpose == purpose,
                                                        auth_tokens.c.used_at.is_(None))).values(used_at=now))
            conn.execute(insert(auth_tokens).values(user_id=user_id, purpose=purpose, token_hash=token_hash, created_at=now,
                                                    expires_at=expires_at))

    @_guard
    def auth_token_state(self, token_hash: str, purpose: str, now: datetime) -> str:
        with self.engine.connect() as conn:
            row = conn.execute(select(auth_tokens).where(auth_tokens.c.token_hash == token_hash)).first()
        return _token_state(row._mapping if row else None, purpose, now)

    def _consume(self, conn, token_hash: str, purpose: str, now: datetime) -> tuple[str, int | None]:
        """Atomically mark a valid token used. A concurrent second consumer finds used_at already set (the row lock
        makes it wait for the first transaction) and gets TOKEN_USED."""
        user_id = conn.execute(
            update(auth_tokens).where(and_(auth_tokens.c.token_hash == token_hash, auth_tokens.c.purpose == purpose,
                                           auth_tokens.c.used_at.is_(None), auth_tokens.c.expires_at > now))
            .values(used_at=now).returning(auth_tokens.c.user_id)).scalar()
        if user_id is not None:
            return TOKEN_VALID, user_id
        row = conn.execute(select(auth_tokens).where(auth_tokens.c.token_hash == token_hash)).first()
        return _token_state(row._mapping if row else None, purpose, now), None

    def _mark_verified(self, conn, user_id: int, now: datetime) -> None:
        conn.execute(pg_insert(user_email_verification).values(user_id=user_id, verified_at=now).on_conflict_do_update(
            index_elements=["user_id"], set_={"verified_at": func.coalesce(user_email_verification.c.verified_at, now)}))

    @_guard
    def verify_email_with_token(self, token_hash: str, now: datetime) -> tuple[str, int | None]:
        """Consume a verification token and mark its user verified, in one transaction."""
        with self.engine.begin() as conn:
            state, user_id = self._consume(conn, token_hash, "verify_email", now)
            if user_id is not None:
                self._mark_verified(conn, user_id, now)
            return state, user_id

    @_guard
    def reset_password_with_token(self, token_hash: str, password_hash: str, now: datetime) -> tuple[str, int | None]:
        """Consume a reset token, set the new password hash, increment auth_version (ending every existing
        session), retire the user's other outstanding reset tokens and -- since the link proves control of the
        mailbox -- mark the email verified. One transaction."""
        with self.engine.begin() as conn:
            state, user_id = self._consume(conn, token_hash, "reset_password", now)
            if user_id is None:
                return state, None
            conn.execute(update(users).where(users.c.user_id == user_id)
                         .values(password_hash=password_hash, auth_version=users.c.auth_version + 1))
            conn.execute(update(auth_tokens).where(and_(auth_tokens.c.user_id == user_id,
                                                        auth_tokens.c.purpose == "reset_password",
                                                        auth_tokens.c.used_at.is_(None))).values(used_at=now))
            self._mark_verified(conn, user_id, now)
            return state, user_id

    # ---- external sign-in identities (schema v4) ------------------------------------
    @_guard
    def get_user_by_identity(self, provider: str, subject: str) -> dict | None:
        """The account linked to this external identity (the provider's stable subject, never the email)."""
        with self.engine.connect() as conn:
            user_id = conn.execute(select(user_identities.c.user_id).where(and_(
                user_identities.c.provider == provider, user_identities.c.subject == subject))).scalar()
        return self.get_user(user_id) if user_id is not None else None

    @_guard
    def get_identity_for_user(self, user_id: int, provider: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(*[user_identities.c[f] for f in IDENTITY_FIELDS]).where(and_(
                user_identities.c.user_id == user_id, user_identities.c.provider == provider))).first()
        return dict(row._mapping) if row else None

    @_guard
    def create_user_with_identity(self, email, password_hash, role, display_name, country, worker_profile, provider: str,
                                  subject: str, now: datetime) -> int:
        """A new account whose email the identity provider verified: the user, its verification row (verified
        now), the worker profile and the identity link are written in ONE transaction -- if any part fails
        (DuplicateEmail, IdentityAlreadyLinked, ...) nothing is created."""
        with self.engine.begin() as conn:
            user_id = self._insert_user(conn, email, password_hash, role, display_name, country, None, None,
                                        worker_profile, now)
            conn.execute(insert(user_identities).values(user_id=user_id, provider=provider, subject=subject,
                                                        email_at_link=email, created_at=now, last_login_at=now))
        return user_id

    @_guard
    def link_identity(self, user_id: int, provider: str, subject: str, email: str, now: datetime) -> None:
        """Link an identity to an existing account (after the owner proved the password) and, since the provider
        verified the same address, mark the account's email verified. One transaction; raises
        IdentityAlreadyLinked / UserAlreadyHasIdentity without changing anything."""
        with self.engine.begin() as conn:
            conn.execute(insert(user_identities).values(user_id=user_id, provider=provider, subject=subject,
                                                        email_at_link=email, created_at=now, last_login_at=now))
            self._mark_verified(conn, user_id, now)

    @_guard
    def record_identity_login(self, provider: str, subject: str, now: datetime) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(user_identities).where(and_(
                user_identities.c.provider == provider, user_identities.c.subject == subject)).values(last_login_at=now))

    @_guard
    def get_runtime_worker_profile(self, user_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(runtime_worker_profiles).where(runtime_worker_profiles.c.user_id == user_id)).first()
            if row is None:
                return None
            skill_rows = conn.execute(select(runtime_worker_skills.c.skill).where(runtime_worker_skills.c.user_id == user_id)
                                      .order_by(runtime_worker_skills.c.skill))
            return {**dict(row._mapping), "skills": [r[0] for r in skill_rows]}

    # ---- profile editing (schema v6) -------------------------------------------------
    @_guard
    def update_profile(self, user_id: int, display_name: str, country=_UNCHANGED, worker_profile: dict | None = None) -> None:
        """One transaction: users.display_name (and users.country unless _UNCHANGED); with `worker_profile`, the
        runtime profile (category, experience, rate, country, bio) and its skills are replaced too. Email, password
        and catalogue links are never touched."""
        values = {"display_name": display_name}
        if country is not _UNCHANGED:
            values["country"] = country
        with self.engine.begin() as conn:
            if conn.execute(update(users).where(users.c.user_id == user_id).values(**values)).rowcount != 1:
                raise ValueError("constraint violated: unknown user")
            if worker_profile is not None:
                changed = conn.execute(update(runtime_worker_profiles).where(runtime_worker_profiles.c.user_id == user_id)
                                       .values(**{k: worker_profile[k] for k in WORKER_PROFILE_COLUMNS},
                                               bio=worker_profile.get("bio"))).rowcount
                if changed != 1:
                    raise ValueError("constraint violated: no runtime worker profile")
                conn.execute(runtime_worker_skills.delete().where(runtime_worker_skills.c.user_id == user_id))
                if worker_profile["skills"]:
                    conn.execute(insert(runtime_worker_skills),
                                 [{"user_id": user_id, "skill": s} for s in sorted(set(worker_profile["skills"]))])

    # ---- self-reported credentials (schema v6): always one worker's own rows -------------------
    @_guard
    def list_credentials(self, user_id: int) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(*[worker_credentials.c[f] for f in CREDENTIAL_FIELDS])
                                .where(worker_credentials.c.user_id == user_id)
                                .order_by(worker_credentials.c.kind, worker_credentials.c.created_at,
                                          worker_credentials.c.credential_id))
            return [dict(r._mapping) for r in rows]

    @_guard
    def get_credential(self, user_id: int, credential_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(*[worker_credentials.c[f] for f in CREDENTIAL_FIELDS]).where(
                worker_credentials.c.credential_id == credential_id, worker_credentials.c.user_id == user_id)).first()
        return dict(row._mapping) if row else None

    @staticmethod
    def _lock_profile_and_count(conn, user_id: int, kind: str, exclude_id: int | None = None) -> int:
        """Lock the worker's profile row (serialises concurrent adds, so the per-kind limit holds) and count."""
        if conn.execute(select(runtime_worker_profiles.c.user_id).where(runtime_worker_profiles.c.user_id == user_id)
                        .with_for_update()).first() is None:
            raise ValueError("constraint violated: no runtime worker profile")
        conditions = [worker_credentials.c.user_id == user_id, worker_credentials.c.kind == kind]
        if exclude_id is not None:
            conditions.append(worker_credentials.c.credential_id != exclude_id)
        return conn.execute(select(func.count()).select_from(worker_credentials).where(*conditions)).scalar_one()

    @_guard
    def add_credential(self, user_id: int, data: dict, per_kind_limit: int) -> int:
        with self.engine.begin() as conn:
            if self._lock_profile_and_count(conn, user_id, data["kind"]) >= per_kind_limit:
                raise CredentialLimitReached(data["kind"])
            return conn.execute(insert(worker_credentials).values(user_id=user_id, **{k: data[k] for k in CREDENTIAL_DATA})
                                .returning(worker_credentials.c.credential_id)).scalar_one()

    @_guard
    def update_credential(self, user_id: int, credential_id: int, data: dict, per_kind_limit: int) -> bool:
        """False if the credential is not this worker's (or does not exist)."""
        owned = and_(worker_credentials.c.credential_id == credential_id, worker_credentials.c.user_id == user_id)
        with self.engine.begin() as conn:
            if conn.execute(select(worker_credentials.c.credential_id).where(owned)).first() is None:
                return False
            if self._lock_profile_and_count(conn, user_id, data["kind"], exclude_id=credential_id) >= per_kind_limit:
                raise CredentialLimitReached(data["kind"])
            conn.execute(update(worker_credentials).where(owned).values(**{k: data[k] for k in CREDENTIAL_DATA}))
            return True

    @_guard
    def delete_credential(self, user_id: int, credential_id: int) -> bool:
        with self.engine.begin() as conn:
            return conn.execute(worker_credentials.delete().where(
                worker_credentials.c.credential_id == credential_id, worker_credentials.c.user_id == user_id)).rowcount == 1

    # ---- runtime requests --------------------------------------------------------
    @_guard
    def create_request(self, client_user_id, title, category, skills_list, pay_type, budget, hourly_low, hourly_high,
                       price_tier, location_requirement) -> int:
        with self.engine.begin() as conn:
            request_id = conn.execute(insert(runtime_requests).values(
                client_user_id=client_user_id, title=title, category=category, pay_type=pay_type, budget=budget,
                hourly_low=hourly_low, hourly_high=hourly_high, price_tier=price_tier,
                location_requirement=location_requirement).returning(runtime_requests.c.request_id)).scalar_one()
            conn.execute(insert(runtime_request_skills), [{"request_id": request_id, "skill": s} for s in skills_list])
        return request_id

    def _request_with_skills(self, conn, row) -> dict:
        skill_rows = conn.execute(select(runtime_request_skills.c.skill).where(runtime_request_skills.c.request_id == row.request_id)
                                  .order_by(runtime_request_skills.c.skill))
        return {**dict(row._mapping), "skills": [r[0] for r in skill_rows]}

    @_guard
    def get_request(self, request_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(*[runtime_requests.c[f] for f in REQUEST_FIELDS])
                               .where(runtime_requests.c.request_id == request_id)).first()
            return self._request_with_skills(conn, row) if row else None

    @_guard
    def list_requests_for_client(self, client_user_id: int) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(*[runtime_requests.c[f] for f in REQUEST_FIELDS])
                                .where(runtime_requests.c.client_user_id == client_user_id)
                                .order_by(runtime_requests.c.request_id.desc())).all()
            return [self._request_with_skills(conn, r) for r in rows]

    # ---- bookings -------------------------------------------------------------------
    @staticmethod
    def _insert_notification(conn, notification: dict | None, booking_id: int | None) -> None:
        if notification is not None:
            conn.execute(insert(notifications).values(user_id=notification["user_id"], booking_id=booking_id,
                                                      kind=notification["kind"], message=notification["message"]))

    @_guard
    def create_booking(self, request_id: int, client_user_id: int, worker_id: str, notification: dict | None = None) -> int:
        with self.engine.begin() as conn:
            booking_id = conn.execute(insert(bookings).values(request_id=request_id, client_user_id=client_user_id,
                                                              worker_id=worker_id, status="pending")
                                      .returning(bookings.c.booking_id)).scalar_one()
            self._insert_notification(conn, notification, booking_id)
            return booking_id

    @_guard
    def get_booking(self, booking_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(*[bookings.c[f] for f in BOOKING_FIELDS]).where(bookings.c.booking_id == booking_id)).first()
        return dict(row._mapping) if row else None

    @_guard
    def transition_booking(self, booking_id: int, expected_status: str, new_status: str,
                           notification: dict | None = None) -> None:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(bookings)
                .where(and_(bookings.c.booking_id == booking_id, bookings.c.status == expected_status))
                .values(status=new_status, updated_at=func.now(),
                        completed_at=func.now() if new_status == "completed" else None))
            if result.rowcount != 1:
                raise StaleBookingState(f"booking {booking_id} is no longer {expected_status!r}")
            self._insert_notification(conn, notification, booking_id)

    def _list_bookings_where(self, *conditions) -> list[dict]:
        query = (select(*[bookings.c[f] for f in BOOKING_FIELDS], runtime_requests.c.title.label("request_title"),
                        app_reviews.c.rating.label("review_rating"), app_reviews.c.review_text.label("review_text"))
                 .join(runtime_requests, runtime_requests.c.request_id == bookings.c.request_id)
                 .outerjoin(app_reviews, app_reviews.c.booking_id == bookings.c.booking_id)
                 .where(*conditions).order_by(bookings.c.booking_id.desc()))
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(query)]

    @_guard
    def list_bookings(self, client_user_id: int | None = None, worker_id: str | None = None,
                      request_id: int | None = None) -> list[dict]:
        conditions = []
        if client_user_id is not None:
            conditions.append(bookings.c.client_user_id == client_user_id)
        if worker_id is not None:
            conditions.append(bookings.c.worker_id == worker_id)
        if request_id is not None:
            conditions.append(bookings.c.request_id == request_id)
        return self._list_bookings_where(*conditions)

    @_guard
    def list_unclaimed_worker_bookings(self) -> list[dict]:
        """Bookings of catalogue workers that have NO linked account (the demo operator's queue)."""
        return self._list_bookings_where(~exists().where(users.c.worker_id == bookings.c.worker_id))

    # ---- reviews --------------------------------------------------------------------
    @_guard
    def create_review(self, booking_id: int, rating: int, review_text: str | None, notification: dict | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(app_reviews).values(booking_id=booking_id, rating=rating, review_text=review_text))
            self._insert_notification(conn, notification, booking_id)

    @_guard
    def get_review(self, booking_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(app_reviews).where(app_reviews.c.booking_id == booking_id)).first()
        return dict(row._mapping) if row else None

    # ---- notifications (schema v5): always one recipient's own rows ---------------------
    @_guard
    def list_notifications(self, user_id: int, limit: int, offset: int = 0) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(*[notifications.c[f] for f in NOTIFICATION_FIELDS])
                                .where(notifications.c.user_id == user_id)
                                .order_by(notifications.c.created_at.desc(), notifications.c.notification_id.desc())
                                .limit(limit).offset(offset))
            return [dict(r._mapping) for r in rows]

    @_guard
    def count_notifications(self, user_id: int, unread_only: bool = False) -> int:
        conditions = [notifications.c.user_id == user_id]
        if unread_only:
            conditions.append(notifications.c.read_at.is_(None))
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(notifications).where(*conditions)).scalar_one()

    @_guard
    def count_notifications_since(self, user_id: int, since: datetime) -> int:
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(notifications).where(
                notifications.c.user_id == user_id, notifications.c.created_at >= since)).scalar_one()

    @_guard
    def mark_notification_read(self, user_id: int, notification_id: int) -> bool:
        """True if the notification exists AND belongs to `user_id` (marking it read if it was unread); False
        otherwise, so the caller cannot tell another user's notification from a missing one."""
        owned = and_(notifications.c.notification_id == notification_id, notifications.c.user_id == user_id)
        with self.engine.begin() as conn:
            conn.execute(update(notifications).where(owned, notifications.c.read_at.is_(None)).values(read_at=func.now()))
            return conn.execute(select(func.count()).select_from(notifications).where(owned)).scalar_one() == 1

    @_guard
    def mark_all_notifications_read(self, user_id: int) -> int:
        with self.engine.begin() as conn:
            return conn.execute(update(notifications).where(notifications.c.user_id == user_id,
                                                            notifications.c.read_at.is_(None))
                                .values(read_at=func.now())).rowcount


class InMemoryMarketplaceRepository:
    """Python mirror of the PostgreSQL semantics (constraints included), for
    database-free tests. NOT used by the application."""

    def __init__(self, category_names, skill_names, client_ids, worker_ids):
        self._categories = sorted(category_names)
        self._skills = sorted(skill_names)
        self._clients = set(client_ids)
        self._workers = set(worker_ids)
        self._users: dict[int, dict] = {}
        self._worker_profiles: dict[int, dict] = {}
        self._requests: dict[int, dict] = {}
        self._bookings: dict[int, dict] = {}
        self._reviews: dict[int, dict] = {}
        self._verification: dict[int, datetime | None] = {}   # user_email_verification
        self._tokens: dict[str, dict] = {}                      # auth_tokens, keyed by token_hash (unique)
        self._identities: dict[tuple[str, str], dict] = {}     # user_identities, keyed by (provider, subject)
        self._notifications: dict[int, dict] = {}               # notifications (schema v5)
        self._credentials: dict[int, dict] = {}                 # worker_credentials (schema v6)
        self._lock = threading.RLock()                          # makes token consumption atomic, like the database
        self._ids = {name: itertools.count(1) for name in ("user", "request", "booking", "token", "identity",
                                                           "notification", "credential")}

    @classmethod
    def from_artifacts(cls) -> "InMemoryMarketplaceRepository":
        from src.db.loader import build_tables

        t = build_tables()
        return cls([r["name"] for r in t["categories"]], [r["name"] for r in t["skills"]],
                   [r["client_id"] for r in t["clients"]], [r["worker_id"] for r in t["workers"]])

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    def categories(self):
        return list(self._categories)

    def skills(self):
        return list(self._skills)

    def imported_client_exists(self, client_id):
        return client_id in self._clients

    def imported_worker_exists(self, worker_id):
        return worker_id in self._workers

    def create_user(self, email, password_hash, role, display_name, country=None, client_id=None, worker_id=None,
                    worker_profile=None, verified=True):
        with self._lock:
            return self._create_user(email, password_hash, role, display_name, country, client_id, worker_id,
                                     worker_profile, verified)

    def _create_user(self, email, password_hash, role, display_name, country, client_id, worker_id, worker_profile, verified,
                     now=None):
        if role not in ("client", "worker") or email != email.lower() or "@" not in email[1:]:
            raise ValueError("constraint violated")
        if (role != "client" and client_id) or (role != "worker" and worker_id):
            raise ValueError("constraint violated")
        if any(u["email"] == email for u in self._users.values()):
            raise DuplicateEmail("users_email_key")
        if client_id is not None and (client_id not in self._clients or any(u["client_id"] == client_id for u in self._users.values())):
            raise ProfileAlreadyLinked("users_client_id_key")
        if worker_id is not None and (worker_id not in self._workers or any(u["worker_id"] == worker_id for u in self._users.values())):
            raise ProfileAlreadyLinked("users_worker_id_key")
        if worker_profile is not None:
            if worker_profile["primary_category"] not in self._categories or not set(worker_profile["skills"]) <= set(self._skills):
                raise ValueError("constraint violated")
        user_id = next(self._ids["user"])
        self._users[user_id] = {"user_id": user_id, "email": email, "password_hash": password_hash, "role": role,
                                "display_name": display_name, "country": country, "client_id": client_id,
                                "worker_id": worker_id, "created_at": self._now(), "auth_version": 0}
        self._verification[user_id] = (now or self._now()) if verified else None
        if worker_profile is not None:
            self._worker_profiles[user_id] = {"user_id": user_id, **{k: worker_profile[k] for k in WORKER_PROFILE_COLUMNS},
                                              "bio": None, "skills": sorted(set(worker_profile["skills"]))}
        return user_id

    def get_credentials(self, email):
        for u in self._users.values():
            if u["email"] == email:
                return u["user_id"], u["password_hash"]
        return None

    def get_user(self, user_id):
        u = self._users.get(user_id)
        return {**{f: u[f] for f in USER_FIELDS}, "email_verified_at": self._verification.get(user_id)} if u else None

    def get_user_by_email(self, email):
        return next((self.get_user(u["user_id"]) for u in self._users.values() if u["email"] == email), None)

    def get_linked_worker_account(self, worker_id):
        return next((self.get_user(u["user_id"]) for u in self._users.values() if u["worker_id"] == worker_id), None)

    def claimed_worker_ids(self, worker_ids):
        linked = {u["worker_id"] for u in self._users.values() if u["worker_id"]}
        return set(worker_ids) & linked

    # ---- account security (schema v3) ---------------------------------------------
    def issue_auth_token(self, user_id, purpose, token_hash, expires_at, now):
        if purpose not in TOKEN_PURPOSES or user_id not in self._users or token_hash in self._tokens or expires_at <= now:
            raise ValueError("constraint violated")
        with self._lock:
            for t in self._tokens.values():
                if t["user_id"] == user_id and t["purpose"] == purpose and t["used_at"] is None:
                    t["used_at"] = now
            self._tokens[token_hash] = {"token_id": next(self._ids["token"]), "user_id": user_id, "purpose": purpose,
                                        "token_hash": token_hash, "created_at": now, "expires_at": expires_at, "used_at": None}

    def auth_token_state(self, token_hash, purpose, now):
        return _token_state(self._tokens.get(token_hash), purpose, now)

    def _consume(self, token_hash, purpose, now):
        t = self._tokens.get(token_hash)
        state = _token_state(t, purpose, now)
        if state != TOKEN_VALID:
            return state, None
        t["used_at"] = now
        return TOKEN_VALID, t["user_id"]

    def verify_email_with_token(self, token_hash, now):
        with self._lock:
            state, user_id = self._consume(token_hash, "verify_email", now)
            if user_id is not None and self._verification.get(user_id) is None:
                self._verification[user_id] = now
            return state, user_id

    def reset_password_with_token(self, token_hash, password_hash, now):
        with self._lock:
            state, user_id = self._consume(token_hash, "reset_password", now)
            if user_id is None:
                return state, None
            self._users[user_id]["password_hash"] = password_hash
            self._users[user_id]["auth_version"] += 1
            for t in self._tokens.values():
                if t["user_id"] == user_id and t["purpose"] == "reset_password" and t["used_at"] is None:
                    t["used_at"] = now
            if self._verification.get(user_id) is None:
                self._verification[user_id] = now
            return state, user_id

    # ---- external sign-in identities (schema v4) ------------------------------------
    def _check_identity(self, user_id, provider, subject, email):
        """The user_identities constraints, checked before anything is written."""
        if provider not in IDENTITY_PROVIDERS or not 1 <= len(subject or "") <= 255 or email != email.lower():
            raise ValueError("constraint violated")
        if (provider, subject) in self._identities:
            raise IdentityAlreadyLinked("uq_user_identities_provider_subject")
        if any(i["user_id"] == user_id and i["provider"] == provider for i in self._identities.values()):
            raise UserAlreadyHasIdentity("uq_user_identities_user_provider")

    def _add_identity(self, user_id, provider, subject, email, now):
        self._identities[(provider, subject)] = {
            "identity_id": next(self._ids["identity"]), "user_id": user_id, "provider": provider, "subject": subject,
            "email_at_link": email, "created_at": now, "last_login_at": now}

    def get_user_by_identity(self, provider, subject):
        identity = self._identities.get((provider, subject))
        return self.get_user(identity["user_id"]) if identity else None

    def get_identity_for_user(self, user_id, provider):
        return next((dict(i) for i in self._identities.values() if i["user_id"] == user_id and i["provider"] == provider),
                    None)

    def create_user_with_identity(self, email, password_hash, role, display_name, country, worker_profile, provider,
                                  subject, now):
        with self._lock:  # every check first, then the writes: nothing is created if any part would fail
            self._check_identity(None, provider, subject, email)
            user_id = self._create_user(email, password_hash, role, display_name, country, None, None, worker_profile,
                                        True, now)
            self._add_identity(user_id, provider, subject, email, now)
            return user_id

    def link_identity(self, user_id, provider, subject, email, now):
        with self._lock:
            if user_id not in self._users:
                raise ValueError("constraint violated")
            self._check_identity(user_id, provider, subject, email)
            self._add_identity(user_id, provider, subject, email, now)
            if self._verification.get(user_id) is None:
                self._verification[user_id] = now

    def record_identity_login(self, provider, subject, now):
        with self._lock:
            identity = self._identities.get((provider, subject))
            if identity is not None:
                identity["last_login_at"] = now

    def get_runtime_worker_profile(self, user_id):
        p = self._worker_profiles.get(user_id)
        return dict(p) if p else None

    # ---- profile editing and credentials (schema v6): the database constraints, checked before any write ----------
    def update_profile(self, user_id, display_name, country=_UNCHANGED, worker_profile=None):
        with self._lock:
            user = self._users.get(user_id)
            if user is None or not 1 <= len(display_name or "") <= 200:
                raise ValueError("constraint violated")
            if worker_profile is not None:
                if user_id not in self._worker_profiles:
                    raise ValueError("constraint violated: no runtime worker profile")
                bio = worker_profile.get("bio")
                if worker_profile["primary_category"] not in self._categories \
                        or not set(worker_profile["skills"]) <= set(self._skills) \
                        or worker_profile["experience_level"] not in ("entry", "intermediate", "expert") \
                        or not 0 <= worker_profile["experience_years"] <= 60 or not worker_profile["rate"] > 0 \
                        or not worker_profile["country"] or (bio is not None and len(bio) > BIO_MAX_LENGTH):
                    raise ValueError("constraint violated")
            user["display_name"] = display_name
            if country is not _UNCHANGED:
                user["country"] = country
            if worker_profile is not None:
                self._worker_profiles[user_id].update({k: worker_profile[k] for k in WORKER_PROFILE_COLUMNS},
                                                      bio=worker_profile.get("bio"),
                                                      skills=sorted(set(worker_profile["skills"])))

    @staticmethod
    def _check_credential(data):
        def too_long(value, limit):
            return value is not None and len(value) > limit
        if data["kind"] not in CREDENTIAL_KINDS or not 1 <= len(data["title"] or "") <= CREDENTIAL_TITLE_MAX_LENGTH \
                or too_long(data["organisation"], CREDENTIAL_ORGANISATION_MAX_LENGTH) \
                or too_long(data["description"], CREDENTIAL_DESCRIPTION_MAX_LENGTH) \
                or (data["year"] is not None and not CREDENTIAL_YEAR_MIN <= data["year"] <= CREDENTIAL_YEAR_DB_MAX) \
                or (data["url"] is not None and (len(data["url"]) > CREDENTIAL_URL_MAX_LENGTH
                                                 or not data["url"].lower().startswith(("http://", "https://")))):
            raise ValueError("constraint violated")

    def _count_kind(self, user_id, kind, exclude_id=None):
        return sum(1 for c in self._credentials.values()
                   if c["user_id"] == user_id and c["kind"] == kind and c["credential_id"] != exclude_id)

    def list_credentials(self, user_id):
        rows = [c for c in self._credentials.values() if c["user_id"] == user_id]
        return [dict(c) for c in sorted(rows, key=lambda c: (c["kind"], c["created_at"], c["credential_id"]))]

    def get_credential(self, user_id, credential_id):
        c = self._credentials.get(credential_id)
        return dict(c) if c and c["user_id"] == user_id else None

    def add_credential(self, user_id, data, per_kind_limit):
        with self._lock:
            if user_id not in self._worker_profiles:
                raise ValueError("constraint violated: no runtime worker profile")
            self._check_credential(data)
            if self._count_kind(user_id, data["kind"]) >= per_kind_limit:
                raise CredentialLimitReached(data["kind"])
            credential_id = next(self._ids["credential"])
            self._credentials[credential_id] = {"credential_id": credential_id, "user_id": user_id,
                                                **{k: data[k] for k in CREDENTIAL_DATA}, "created_at": self._now()}
            return credential_id

    def update_credential(self, user_id, credential_id, data, per_kind_limit):
        with self._lock:
            c = self._credentials.get(credential_id)
            if c is None or c["user_id"] != user_id:
                return False
            self._check_credential(data)
            if self._count_kind(user_id, data["kind"], exclude_id=credential_id) >= per_kind_limit:
                raise CredentialLimitReached(data["kind"])
            c.update({k: data[k] for k in CREDENTIAL_DATA})
            return True

    def delete_credential(self, user_id, credential_id):
        with self._lock:
            c = self._credentials.get(credential_id)
            if c is None or c["user_id"] != user_id:
                return False
            del self._credentials[credential_id]
            return True

    def create_request(self, client_user_id, title, category, skills_list, pay_type, budget, hourly_low, hourly_high,
                       price_tier, location_requirement):
        if client_user_id not in self._users or category not in self._categories or not set(skills_list) <= set(self._skills):
            raise ValueError("constraint violated")
        request_id = next(self._ids["request"])
        self._requests[request_id] = {
            "request_id": request_id, "client_user_id": client_user_id, "title": title, "category": category,
            "pay_type": pay_type, "budget": budget, "hourly_low": hourly_low, "hourly_high": hourly_high,
            "price_tier": price_tier, "location_requirement": location_requirement, "created_at": self._now(),
            "skills": sorted(set(skills_list))}
        return request_id

    def get_request(self, request_id):
        r = self._requests.get(request_id)
        return dict(r) if r else None

    def list_requests_for_client(self, client_user_id):
        return [dict(r) for _, r in sorted(self._requests.items(), reverse=True) if r["client_user_id"] == client_user_id]

    # ---- notifications: the database constraints, checked BEFORE the change they accompany is written, so a
    # notification that cannot be stored leaves the booking/review unchanged (the database's single transaction).
    def _check_notification(self, notification, booking_id):
        if notification is None:
            return
        message = notification["message"]
        if notification["user_id"] not in self._users or notification["kind"] not in NOTIFICATION_KINDS \
                or not isinstance(message, str) or not 1 <= len(message) <= NOTIFICATION_MESSAGE_MAX_LENGTH:
            raise ValueError("constraint violated")
        if booking_id is not None and any(
                n["booking_id"] == booking_id and n["kind"] == notification["kind"] and n["user_id"] == notification["user_id"]
                for n in self._notifications.values()):
            raise ValueError("constraint violated: uq_notifications_booking_kind_user")

    def _add_notification(self, notification, booking_id):
        if notification is None:
            return
        notification_id = next(self._ids["notification"])
        self._notifications[notification_id] = {
            "notification_id": notification_id, "user_id": notification["user_id"], "booking_id": booking_id,
            "kind": notification["kind"], "message": notification["message"], "created_at": self._now(), "read_at": None}

    def create_booking(self, request_id, client_user_id, worker_id, notification=None):
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request["client_user_id"] != client_user_id or worker_id not in self._workers:
                raise ValueError("constraint violated")
            if any(b["request_id"] == request_id and b["status"] in ACTIVE_BOOKING_STATUSES for b in self._bookings.values()):
                raise ActiveBookingExists("uq_bookings_one_active_per_request")
            self._check_notification(notification, None)
            booking_id = next(self._ids["booking"])
            now = self._now()
            self._bookings[booking_id] = {"booking_id": booking_id, "request_id": request_id, "client_user_id": client_user_id,
                                          "worker_id": worker_id, "status": "pending", "created_at": now, "updated_at": now,
                                          "completed_at": None}
            self._add_notification(notification, booking_id)
            return booking_id

    def get_booking(self, booking_id):
        b = self._bookings.get(booking_id)
        return dict(b) if b else None

    def transition_booking(self, booking_id, expected_status, new_status, notification=None):
        with self._lock:
            b = self._bookings.get(booking_id)
            if b is None or b["status"] != expected_status:
                raise StaleBookingState(f"booking {booking_id} is no longer {expected_status!r}")
            if new_status in ACTIVE_BOOKING_STATUSES and any(
                    o["booking_id"] != booking_id and o["request_id"] == b["request_id"] and o["status"] in ACTIVE_BOOKING_STATUSES
                    for o in self._bookings.values()):
                raise ActiveBookingExists("uq_bookings_one_active_per_request")
            self._check_notification(notification, booking_id)
            b["status"] = new_status
            b["updated_at"] = self._now()
            b["completed_at"] = self._now() if new_status == "completed" else None
            self._add_notification(notification, booking_id)

    def list_bookings(self, client_user_id=None, worker_id=None, request_id=None):
        out = []
        for booking_id, b in sorted(self._bookings.items(), reverse=True):
            if (client_user_id is not None and b["client_user_id"] != client_user_id) or \
                    (worker_id is not None and b["worker_id"] != worker_id) or \
                    (request_id is not None and b["request_id"] != request_id):
                continue
            review = self._reviews.get(booking_id)
            out.append({**b, "request_title": self._requests[b["request_id"]]["title"],
                        "review_rating": review["rating"] if review else None,
                        "review_text": review["review_text"] if review else None})
        return out

    def list_unclaimed_worker_bookings(self):
        linked = {u["worker_id"] for u in self._users.values() if u["worker_id"]}
        return [b for b in self.list_bookings() if b["worker_id"] not in linked]

    def create_review(self, booking_id, rating, review_text, notification=None):
        with self._lock:
            if booking_id not in self._bookings or not 1 <= rating <= 5 or (review_text and len(review_text) > 2000):
                raise ValueError("constraint violated")
            if booking_id in self._reviews:
                raise DuplicateReview("app_reviews_pkey")
            self._check_notification(notification, booking_id)
            self._reviews[booking_id] = {"booking_id": booking_id, "rating": rating, "review_text": review_text,
                                         "created_at": self._now()}
            self._add_notification(notification, booking_id)

    def get_review(self, booking_id):
        r = self._reviews.get(booking_id)
        return dict(r) if r else None

    # ---- notifications (schema v5) -----------------------------------------------------
    def _own_notifications(self, user_id):
        return [n for n in self._notifications.values() if n["user_id"] == user_id]

    def list_notifications(self, user_id, limit, offset=0):
        rows = sorted(self._own_notifications(user_id), key=lambda n: (n["created_at"], n["notification_id"]), reverse=True)
        return [dict(n) for n in rows[offset:offset + limit]]

    def count_notifications(self, user_id, unread_only=False):
        return sum(1 for n in self._own_notifications(user_id) if not unread_only or n["read_at"] is None)

    def count_notifications_since(self, user_id, since):
        return sum(1 for n in self._own_notifications(user_id) if n["created_at"] >= since)

    def mark_notification_read(self, user_id, notification_id):
        with self._lock:
            n = self._notifications.get(notification_id)
            if n is None or n["user_id"] != user_id:
                return False
            if n["read_at"] is None:
                n["read_at"] = max(self._now(), n["created_at"])
            return True

    def mark_all_notifications_read(self, user_id):
        with self._lock:
            unread = [n for n in self._own_notifications(user_id) if n["read_at"] is None]
            for n in unread:
                n["read_at"] = max(self._now(), n["created_at"])
            return len(unread)
