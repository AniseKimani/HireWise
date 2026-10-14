"""Runtime marketplace persistence (Day 12): users, runtime worker
profiles, client-created requests, bookings and application reviews.

Two implementations share one interface and one test suite:
- PostgresMarketplaceRepository -- the application's store; the database
  constraints in src/db/schema.py are authoritative.
- InMemoryMarketplaceRepository -- the same semantics in Python, for the
  database-free default tests only. Never used by the application factory.

Imported experimental tables are only READ here (categories, skills,
client/worker existence); all writes go to the runtime tables. Password
hashes are only returned by `get_credentials`, never by profile reads.
"""
from __future__ import annotations

import itertools
from datetime import datetime, timezone
from functools import wraps

from sqlalchemy import and_, func, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from src.db.repositories import DatabaseUnavailable
from src.db.schema import (
    ACTIVE_BOOKING_STATUSES, app_reviews, bookings, categories, clients, runtime_request_skills, runtime_requests,
    runtime_worker_profiles, runtime_worker_skills, skills, users, workers,
)


class DuplicateEmail(ValueError):
    pass


class ProfileAlreadyLinked(ValueError):
    pass


class ActiveBookingExists(ValueError):
    pass


class DuplicateReview(ValueError):
    pass


class StaleBookingState(RuntimeError):
    """The booking changed since it was read (or never existed)."""


USER_FIELDS = ("user_id", "email", "role", "display_name", "country", "client_id", "worker_id", "created_at")
REQUEST_FIELDS = ("request_id", "client_user_id", "title", "category", "pay_type", "budget", "hourly_low", "hourly_high",
                  "price_tier", "location_requirement", "created_at")
BOOKING_FIELDS = ("booking_id", "request_id", "client_user_id", "worker_id", "status", "created_at", "updated_at", "completed_at")

_CONSTRAINT_ERRORS = {
    "users_email_key": DuplicateEmail,
    "users_client_id_key": ProfileAlreadyLinked,
    "users_worker_id_key": ProfileAlreadyLinked,
    "uq_bookings_one_active_per_request": ActiveBookingExists,
    "app_reviews_pkey": DuplicateReview,
}


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
                    worker_profile: dict | None = None) -> int:
        with self.engine.begin() as conn:
            user_id = conn.execute(insert(users).values(
                email=email, password_hash=password_hash, role=role, display_name=display_name, country=country,
                client_id=client_id, worker_id=worker_id).returning(users.c.user_id)).scalar_one()
            if worker_profile is not None:
                conn.execute(insert(runtime_worker_profiles).values(
                    user_id=user_id, **{k: worker_profile[k] for k in
                                        ("primary_category", "experience_level", "experience_years", "rate", "country")}))
                if worker_profile["skills"]:
                    conn.execute(insert(runtime_worker_skills), [{"user_id": user_id, "skill": s} for s in worker_profile["skills"]])
        return user_id

    @_guard
    def get_credentials(self, email: str) -> tuple[int, str] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(users.c.user_id, users.c.password_hash).where(users.c.email == email)).first()
        return (row.user_id, row.password_hash) if row else None

    @_guard
    def get_user(self, user_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(*[users.c[f] for f in USER_FIELDS]).where(users.c.user_id == user_id)).first()
        return dict(row._mapping) if row else None

    @_guard
    def get_runtime_worker_profile(self, user_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(runtime_worker_profiles).where(runtime_worker_profiles.c.user_id == user_id)).first()
            if row is None:
                return None
            skill_rows = conn.execute(select(runtime_worker_skills.c.skill).where(runtime_worker_skills.c.user_id == user_id)
                                      .order_by(runtime_worker_skills.c.skill))
            return {**dict(row._mapping), "skills": [r[0] for r in skill_rows]}

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
    @_guard
    def create_booking(self, request_id: int, client_user_id: int, worker_id: str) -> int:
        with self.engine.begin() as conn:
            return conn.execute(insert(bookings).values(request_id=request_id, client_user_id=client_user_id,
                                                        worker_id=worker_id, status="pending")
                                .returning(bookings.c.booking_id)).scalar_one()

    @_guard
    def get_booking(self, booking_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(*[bookings.c[f] for f in BOOKING_FIELDS]).where(bookings.c.booking_id == booking_id)).first()
        return dict(row._mapping) if row else None

    @_guard
    def transition_booking(self, booking_id: int, expected_status: str, new_status: str) -> None:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(bookings)
                .where(and_(bookings.c.booking_id == booking_id, bookings.c.status == expected_status))
                .values(status=new_status, updated_at=func.now(),
                        completed_at=func.now() if new_status == "completed" else None))
            if result.rowcount != 1:
                raise StaleBookingState(f"booking {booking_id} is no longer {expected_status!r}")

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
        query = (select(*[bookings.c[f] for f in BOOKING_FIELDS], runtime_requests.c.title.label("request_title"),
                        app_reviews.c.rating.label("review_rating"), app_reviews.c.review_text.label("review_text"))
                 .join(runtime_requests, runtime_requests.c.request_id == bookings.c.request_id)
                 .outerjoin(app_reviews, app_reviews.c.booking_id == bookings.c.booking_id)
                 .where(*conditions).order_by(bookings.c.booking_id.desc()))
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(query)]

    # ---- reviews --------------------------------------------------------------------
    @_guard
    def create_review(self, booking_id: int, rating: int, review_text: str | None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(app_reviews).values(booking_id=booking_id, rating=rating, review_text=review_text))

    @_guard
    def get_review(self, booking_id: int) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(app_reviews).where(app_reviews.c.booking_id == booking_id)).first()
        return dict(row._mapping) if row else None


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
        self._ids = {name: itertools.count(1) for name in ("user", "request", "booking")}

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
                    worker_profile=None):
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
                                "worker_id": worker_id, "created_at": self._now()}
        if worker_profile is not None:
            self._worker_profiles[user_id] = {"user_id": user_id, **{k: worker_profile[k] for k in (
                "primary_category", "experience_level", "experience_years", "rate", "country")},
                "skills": sorted(set(worker_profile["skills"]))}
        return user_id

    def get_credentials(self, email):
        for u in self._users.values():
            if u["email"] == email:
                return u["user_id"], u["password_hash"]
        return None

    def get_user(self, user_id):
        u = self._users.get(user_id)
        return {f: u[f] for f in USER_FIELDS} if u else None

    def get_runtime_worker_profile(self, user_id):
        p = self._worker_profiles.get(user_id)
        return dict(p) if p else None

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

    def create_booking(self, request_id, client_user_id, worker_id):
        request = self._requests.get(request_id)
        if request is None or request["client_user_id"] != client_user_id or worker_id not in self._workers:
            raise ValueError("constraint violated")
        if any(b["request_id"] == request_id and b["status"] in ACTIVE_BOOKING_STATUSES for b in self._bookings.values()):
            raise ActiveBookingExists("uq_bookings_one_active_per_request")
        booking_id = next(self._ids["booking"])
        now = self._now()
        self._bookings[booking_id] = {"booking_id": booking_id, "request_id": request_id, "client_user_id": client_user_id,
                                      "worker_id": worker_id, "status": "pending", "created_at": now, "updated_at": now,
                                      "completed_at": None}
        return booking_id

    def get_booking(self, booking_id):
        b = self._bookings.get(booking_id)
        return dict(b) if b else None

    def transition_booking(self, booking_id, expected_status, new_status):
        b = self._bookings.get(booking_id)
        if b is None or b["status"] != expected_status:
            raise StaleBookingState(f"booking {booking_id} is no longer {expected_status!r}")
        if new_status in ACTIVE_BOOKING_STATUSES and any(
                o["booking_id"] != booking_id and o["request_id"] == b["request_id"] and o["status"] in ACTIVE_BOOKING_STATUSES
                for o in self._bookings.values()):
            raise ActiveBookingExists("uq_bookings_one_active_per_request")
        b["status"] = new_status
        b["updated_at"] = self._now()
        b["completed_at"] = self._now() if new_status == "completed" else None

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

    def create_review(self, booking_id, rating, review_text):
        if booking_id not in self._bookings or not 1 <= rating <= 5 or (review_text and len(review_text) > 2000):
            raise ValueError("constraint violated")
        if booking_id in self._reviews:
            raise DuplicateReview("app_reviews_pkey")
        self._reviews[booking_id] = {"booking_id": booking_id, "rating": rating, "review_text": review_text,
                                     "created_at": self._now()}

    def get_review(self, booking_id):
        r = self._reviews.get(booking_id)
        return dict(r) if r else None
