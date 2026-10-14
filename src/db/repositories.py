"""Relational data access for the Flask service layer (Day 11).

Every query is a SQLAlchemy Core expression with bound parameters -- no
client input is ever interpolated into SQL text. Repositories return
plain "row" dicts in the shape src/app/serializers.py already consumes
(multivalued fields joined with "|"), so the public API stays
byte-identical to Day 10.

Semantics are the Day 10 ones, made locale-independent:
- browsable requests = validation/test period, ordered by
  (simulated_day, link) with "C"-collated identifiers (codepoint order);
- `q`: substring of the loader's Python-lower-cased `title_search`;
- `category`: exact; `skill`: case-insensitive exact (skills are ASCII);
- worker review statistics: count/sum over the stored (training-period
  only) ratings.

Database failures surface as DatabaseUnavailable, with no driver text.
"""
from __future__ import annotations

from functools import wraps

from sqlalchemy import exists, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, ProgrammingError, SQLAlchemyError

from src.db.schema import load_metadata, ratings, request_skills, service_requests, worker_secondary_categories, worker_skills, workers

BROWSABLE_SPLITS = ("validation", "test")
REQUEST_COLUMNS = (
    "link", "title", "client_id", "category", "pay_type", "price_tier", "location_requirement", "simulated_day", "split",
)
WORKER_COLUMNS = (
    "worker_id", "primary_category", "experience_level", "experience_years", "price_tier", "rate", "country", "join_day",
)


class DatabaseUnavailable(RuntimeError):
    """PostgreSQL could not serve the query (down, unreachable, or not initialised)."""


def _guard(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except SQLAlchemyError as exc:
            raise DatabaseUnavailable("application database unavailable") from exc
    return wrapper


def _join(values: list[str]) -> str | None:
    return "|".join(values) if values else None


class PostgresRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def health(self) -> dict:
        try:
            with self.engine.connect() as conn:
                value = conn.execute(select(load_metadata.c.value).where(load_metadata.c.key == "source_fingerprint")).scalar()
        except ProgrammingError:
            return {"available": True, "data_loaded": False}
        except (DBAPIError, SQLAlchemyError):
            return {"available": False, "data_loaded": None}
        return {"available": True, "data_loaded": value is not None}

    # ---- requests ---------------------------------------------------------
    def _skills_by_link(self, conn, links: list[str]) -> dict[str, list[str]]:
        out = {link: [] for link in links}
        if links:
            rows = conn.execute(select(request_skills.c.link, request_skills.c.skill)
                                .where(request_skills.c.link.in_(links)).order_by(request_skills.c.link, request_skills.c.skill))
            for link, skill in rows:
                out[link].append(skill)
        return out

    def _request_rows(self, conn, records) -> list[dict]:
        records = [dict(r._mapping) for r in records]
        skills = self._skills_by_link(conn, [r["link"] for r in records])
        return [{**r, "skills_vocab": _join(skills[r["link"]])} for r in records]

    @_guard
    def get_request(self, link: str) -> dict | None:
        with self.engine.connect() as conn:
            rows = self._request_rows(conn, conn.execute(
                select(*[service_requests.c[c] for c in REQUEST_COLUMNS]).where(service_requests.c.link == link)))
        return rows[0] if rows else None

    @_guard
    def list_browsable(self, page: int, per_page: int, q: str | None = None, category: str | None = None,
                       skill: str | None = None) -> tuple[list[dict], int]:
        sr = service_requests
        conditions = [sr.c.split.in_(BROWSABLE_SPLITS)]
        if q:
            conditions.append(func.strpos(sr.c.title_search, q.lower()) > 0)
        if category:
            conditions.append(sr.c.category == category)
        if skill:
            conditions.append(exists().where(request_skills.c.link == sr.c.link,
                                             func.lower(request_skills.c.skill) == skill.lower()))
        with self.engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(sr).where(*conditions)).scalar_one()
            records = conn.execute(select(*[sr.c[c] for c in REQUEST_COLUMNS]).where(*conditions)
                                   .order_by(sr.c.simulated_day, sr.c.link).limit(per_page).offset((page - 1) * per_page))
            return self._request_rows(conn, records), total

    @_guard
    def browsable_categories(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(service_requests.c.category).distinct()
                                .where(service_requests.c.split.in_(BROWSABLE_SPLITS)).order_by(service_requests.c.category))
            return [r[0] for r in rows]

    # ---- workers --------------------------------------------------------------
    @_guard
    def get_workers(self, worker_ids: list[str]) -> dict[str, dict]:
        ids = sorted(set(worker_ids))
        if not ids:
            return {}
        with self.engine.connect() as conn:
            base = {r.worker_id: dict(r._mapping) for r in conn.execute(
                select(*[workers.c[c] for c in WORKER_COLUMNS]).where(workers.c.worker_id.in_(ids)))}
            secondary = {w: [] for w in base}
            for w, c in conn.execute(select(worker_secondary_categories.c.worker_id, worker_secondary_categories.c.category)
                                     .where(worker_secondary_categories.c.worker_id.in_(ids))
                                     .order_by(worker_secondary_categories.c.worker_id, worker_secondary_categories.c.category)):
                secondary[w].append(c)
            declared = {w: [] for w in base}
            for w, s in conn.execute(select(worker_skills.c.worker_id, worker_skills.c.skill)
                                     .where(worker_skills.c.worker_id.in_(ids)).order_by(worker_skills.c.worker_id, worker_skills.c.skill)):
                declared[w].append(s)
            reviews = {w: (n, s) for w, n, s in conn.execute(
                select(ratings.c.worker_id, func.count(), func.sum(ratings.c.rating))
                .where(ratings.c.worker_id.in_(ids)).group_by(ratings.c.worker_id))}
        return {
            w: {**row, "secondary_categories": _join(secondary[w]), "declared_skills": _join(declared[w]),
                "review_count": int(reviews.get(w, (0, 0))[0]), "review_sum": int(reviews.get(w, (0, 0))[1] or 0)}
            for w, row in base.items()
        }

    def get_worker(self, worker_id: str) -> dict | None:
        return self.get_workers([worker_id]).get(worker_id)


class InMemoryRepository:
    """Same interface, backed by `src.db.loader.build_tables()` (the exact
    rows the loader writes). FOR TESTS AND PARITY VERIFICATION ONLY: the
    application factory never falls back to it."""

    def __init__(self, tables: dict[str, list[dict]]):
        self._requests = {}
        skills_by_link: dict[str, list[str]] = {}
        for r in tables["request_skills"]:
            skills_by_link.setdefault(r["link"], []).append(r["skill"])
        for r in tables["service_requests"]:
            self._requests[r["link"]] = {**{c: r[c] for c in REQUEST_COLUMNS},
                                         "skills_vocab": _join(sorted(skills_by_link.get(r["link"], []))),
                                         "_title_search": r["title_search"], "_skills": skills_by_link.get(r["link"], [])}
        secondary: dict[str, list[str]] = {}
        for r in tables["worker_secondary_categories"]:
            secondary.setdefault(r["worker_id"], []).append(r["category"])
        declared: dict[str, list[str]] = {}
        for r in tables["worker_skills"]:
            declared.setdefault(r["worker_id"], []).append(r["skill"])
        reviews: dict[str, list[int]] = {}
        for r in tables["ratings"]:
            reviews.setdefault(r["worker_id"], []).append(r["rating"])
        self._workers = {
            r["worker_id"]: {**{c: r[c] for c in WORKER_COLUMNS},
                             "secondary_categories": _join(sorted(secondary.get(r["worker_id"], []))),
                             "declared_skills": _join(sorted(declared.get(r["worker_id"], []))),
                             "review_count": len(reviews.get(r["worker_id"], [])),
                             "review_sum": sum(reviews.get(r["worker_id"], []))}
            for r in tables["workers"]
        }
        self._browsable = sorted((r for r in self._requests.values() if r["split"] in BROWSABLE_SPLITS),
                                 key=lambda r: (r["simulated_day"], r["link"]))

    @classmethod
    def from_artifacts(cls) -> "InMemoryRepository":
        from src.db.loader import build_tables

        return cls(build_tables())

    @staticmethod
    def _public(row: dict) -> dict:
        return {k: v for k, v in row.items() if not k.startswith("_")}

    def health(self) -> dict:
        return {"available": True, "data_loaded": True}

    def get_request(self, link: str) -> dict | None:
        row = self._requests.get(link)
        return self._public(row) if row else None

    def list_browsable(self, page, per_page, q=None, category=None, skill=None):
        q_lower = q.lower() if q else None
        skill_lower = skill.lower() if skill else None
        matches = [r for r in self._browsable
                   if (q_lower is None or q_lower in r["_title_search"])
                   and (category is None or r["category"] == category)
                   and (skill_lower is None or any(s.lower() == skill_lower for s in r["_skills"]))]
        start = (page - 1) * per_page
        return [self._public(r) for r in matches[start:start + per_page]], len(matches)

    def browsable_categories(self) -> list[str]:
        return sorted({r["category"] for r in self._browsable})

    def get_workers(self, worker_ids):
        return {w: dict(self._workers[w]) for w in sorted(set(worker_ids)) if w in self._workers}

    def get_worker(self, worker_id):
        row = self._workers.get(worker_id)
        return dict(row) if row else None
