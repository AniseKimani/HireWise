"""Deterministic PostgreSQL population from the frozen Day 2/3 artifacts
(Day 11).

`build_tables()` is pure Python: it reads only model-visible files
(`data/processed/*.csv`, `data/processed/synthetic/*.csv`, never
`synthetic_private/`) and returns every table's rows, sorted by primary
key, with NaN mapped to NULL. `fingerprint()` hashes a canonical
serialization of those rows, so "is the database exactly the source?" is
one string comparison -- used by the loader, the verifier and the tests.

`load()` writes everything in ONE transaction:
- empty tables                 -> insert all rows;
- same fingerprint already     -> no-op (idempotent re-run);
- different contents           -> refuse, unless reset=True, which
                                  explicitly TRUNCATEs the data tables
                                  first (never DROP).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select, text
from sqlalchemy.engine import Connection, Engine

from src.db.schema import DATA_TABLES, SCHEMA_VERSION, TABLE_LOAD_ORDER, load_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
DAY2 = REPO_ROOT / "data" / "processed"
DAY3 = DAY2 / "synthetic"
TRAIN_SPLIT = "train"

SOURCE_FILES = {
    "category_stats": "data/processed/category_stats.csv",
    "skills_vocab": "data/processed/skills_vocab.csv",
    "sampled_requests": "data/processed/sampled_requests.csv",
    "clients": "data/processed/synthetic/clients.csv",
    "workers": "data/processed/synthetic/workers.csv",
    "requests": "data/processed/synthetic/requests.csv",
    "ratings": "data/processed/synthetic/ratings.csv",
}


class LoadRefused(RuntimeError):
    """The database holds different contents and reset was not requested."""


def _value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):  # NumPy scalar
        v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    return v


def _rows(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    return [{c: _value(v) for c, v in zip(columns, values)} for values in df[columns].itertuples(index=False, name=None)]


def _pipe(cell) -> list[str]:
    return sorted(set(cell.split("|"))) if isinstance(cell, str) and cell else []


def build_tables(repo_root: Path = REPO_ROOT) -> dict[str, list[dict]]:
    read = lambda key, **kw: pd.read_csv(repo_root / SOURCE_FILES[key], **kw)
    categories = sorted(read("category_stats")["category"].unique())
    skill_names = sorted(read("skills_vocab")["skill"].unique())
    clients = read("clients", usecols=["client_id", "country", "preferred_pay_type", "median_price_tier"])
    workers = read("workers")
    requests = read("requests", usecols=["link", "client_id", "simulated_day", "split"])
    sampled = read("sampled_requests", usecols=[
        "link", "title", "category", "pay_type", "price_tier", "budget", "hourly_low", "hourly_high",
        "location_requirement", "skills_vocab",
    ])
    ratings = read("ratings", usecols=["link", "worker_id", "rating", "has_review"])

    service = requests.merge(sampled, on="link", how="left", validate="one_to_one")
    if service["title"].isna().any():
        raise ValueError("every request must have its Day 2 content row")
    service["title_search"] = service["title"].map(str.lower)
    train_links = set(requests.loc[requests["split"] == TRAIN_SPLIT, "link"])
    train_ratings = ratings.loc[ratings["link"].isin(train_links)].copy()
    train_ratings["has_review"] = train_ratings["has_review"].astype(bool)

    tables = {
        "categories": [{"name": c} for c in categories],
        "skills": [{"name": s} for s in skill_names],
        "clients": _rows(clients.sort_values("client_id"), ["client_id", "country", "preferred_pay_type", "median_price_tier"]),
        "workers": _rows(workers.sort_values("worker_id"), [
            "worker_id", "primary_category", "experience_level", "experience_years", "price_tier", "rate", "country", "join_day",
        ]),
        "worker_secondary_categories": sorted(
            ({"worker_id": w, "category": c} for w, cell in zip(workers["worker_id"], workers["secondary_categories"])
             for c in _pipe(cell)), key=lambda r: (r["worker_id"], r["category"])),
        "worker_skills": sorted(
            ({"worker_id": w, "skill": s} for w, cell in zip(workers["worker_id"], workers["declared_skills"]) for s in _pipe(cell)),
            key=lambda r: (r["worker_id"], r["skill"])),
        "service_requests": _rows(service.sort_values("link"), [
            "link", "client_id", "title", "title_search", "category", "pay_type", "price_tier", "budget", "hourly_low",
            "hourly_high", "location_requirement", "simulated_day", "split",
        ]),
        "request_skills": sorted(
            ({"link": l, "skill": s} for l, cell in zip(service["link"], service["skills_vocab"]) for s in _pipe(cell)),
            key=lambda r: (r["link"], r["skill"])),
        "ratings": _rows(train_ratings.sort_values(["link", "worker_id"]), ["link", "worker_id", "rating", "has_review"]),
    }
    return tables


def fingerprint(tables: dict[str, list[dict]]) -> str:
    canonical = {name: [[row[c] for c in sorted(row)] for row in tables[name]] for name in TABLE_LOAD_ORDER}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_hashes(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    from src.data.io import sha256_of

    return {path: sha256_of(repo_root / path) for path in SOURCE_FILES.values()}


def read_back(conn: Connection) -> dict[str, list[dict]]:
    """Every data table, in primary-key order, as plain dicts."""
    out = {}
    for name in TABLE_LOAD_ORDER:
        table = DATA_TABLES[name]
        order = [c for c in table.primary_key.columns]
        out[name] = [dict(r._mapping) for r in conn.execute(select(table).order_by(*order))]
    return out


def stored_metadata(conn: Connection) -> dict[str, str]:
    return {r.key: r.value for r in conn.execute(select(load_metadata).order_by(load_metadata.c.key))}


def _counts(conn: Connection) -> dict[str, int]:
    return {name: conn.execute(select(func.count()).select_from(DATA_TABLES[name])).scalar_one() for name in TABLE_LOAD_ORDER}


def load(engine: Engine, reset: bool = False, repo_root: Path = REPO_ROOT) -> dict:
    tables = build_tables(repo_root)
    target = fingerprint(tables)
    with engine.begin() as conn:
        existing = _counts(conn)
        meta = stored_metadata(conn)
        if any(existing.values()) or meta:
            if (not reset and meta.get("source_fingerprint") == target
                    and existing == {n: len(tables[n]) for n in TABLE_LOAD_ORDER}
                    and fingerprint(read_back(conn)) == target):  # actual contents, not just the stored claim
                return {"action": "unchanged", "fingerprint": target, "row_counts": existing}
            if not reset:
                raise LoadRefused("database already holds different HireWise data; rerun with --reset to replace it")
            names = ", ".join(list(reversed(TABLE_LOAD_ORDER)) + ["load_metadata"])
            conn.execute(text(f"TRUNCATE TABLE {names}"))  # fixed identifiers from the schema, no user input
        for name in TABLE_LOAD_ORDER:
            if tables[name]:
                conn.execute(DATA_TABLES[name].insert(), tables[name])
        metadata_rows = [{"key": "schema_version", "value": SCHEMA_VERSION}, {"key": "source_fingerprint", "value": target}]
        metadata_rows += [{"key": f"row_count.{n}", "value": str(len(tables[n]))} for n in TABLE_LOAD_ORDER]
        metadata_rows += [{"key": f"source_sha256.{p}", "value": h} for p, h in source_hashes(repo_root).items()]
        conn.execute(load_metadata.insert(), sorted(metadata_rows, key=lambda r: r["key"]))
        counts = _counts(conn)
    return {"action": "reset_and_loaded" if reset and any(existing.values()) else "loaded", "fingerprint": target, "row_counts": counts}
