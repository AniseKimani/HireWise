"""Thin HTTP routes (Day 9): validate input, call the service, return
JSON. No recommendation logic lives here."""
from __future__ import annotations

import re

from flask import Blueprint, current_app, jsonify, request

from src.app.errors import APIError
from src.app.services import Neo4jUnavailable, RequestNotRecommendable, UnknownRequest, UnknownWorker

api = Blueprint("api", __name__, url_prefix="/api")

WORKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
K_PATTERN = re.compile(r"^[0-9]{1,6}$")


def _service():
    service = current_app.extensions.get("hirewise_service")
    if service is None:
        raise APIError(503, "recommender_unavailable", "The recommender is not loaded.")
    return service


def _link_param() -> str:
    link = request.args.get("link")
    if link is None or not link.strip():
        raise APIError(400, "missing_link", "Query parameter 'link' (the request link) is required.")
    if len(link) > current_app.config["MAX_LINK_LENGTH"]:
        raise APIError(400, "invalid_link", "Query parameter 'link' is too long.")
    return link


def _k_param() -> int:
    raw = request.args.get("k")
    if raw is None:
        return current_app.config["DEFAULT_K"]
    max_k = current_app.config["MAX_K"]
    if not K_PATTERN.match(raw) or not 1 <= int(raw) <= max_k:
        raise APIError(400, "invalid_k", f"Query parameter 'k' must be an integer between 1 and {max_k}.")
    return int(raw)


def _worker_id(worker_id: str) -> str:
    if not WORKER_ID_PATTERN.match(worker_id):
        raise APIError(400, "invalid_worker_id", "Malformed worker id.")
    return worker_id


@api.get("/health")
def health():
    service = current_app.extensions.get("hirewise_service")
    probe = current_app.extensions.get("hirewise_neo4j_probe")
    neo4j = probe() if probe is not None else {"available": None, "graph_matches_day6_counts": None, "checked": False}
    neo4j.setdefault("checked", probe is not None)
    pg_probe = current_app.extensions.get("hirewise_postgres_probe")
    postgresql = pg_probe() if pg_probe is not None else {"available": None, "data_loaded": None}
    postgresql = {**postgresql, "checked": pg_probe is not None}
    recommender = {"ready": service is not None}
    if service is not None:
        recommender["model"] = service.model_version.to_dict()
    else:
        recommender["reason"] = current_app.extensions.get("hirewise_service_error", "not_loaded")
    # The core workflow needs the recommender AND the application database;
    # Neo4j only powers the optional graph view.
    database_down = postgresql["available"] is False or postgresql["data_loaded"] is False
    if service is None or database_down:
        status = "unavailable"
    elif neo4j.get("available") is False:
        status = "degraded"
    else:
        status = "ok"
    return jsonify({
        "status": status,
        "recommender": recommender,
        "postgresql": postgresql,
        "neo4j": neo4j,
        "recommendations_require_neo4j": False,
    }), (503 if status == "unavailable" else 200)


def _int_param(name: str, default: int, maximum: int) -> int:
    raw = request.args.get(name)
    if raw is None:
        return default
    if not K_PATTERN.match(raw) or not 1 <= int(raw) <= maximum:
        raise APIError(400, f"invalid_{name}", f"Query parameter '{name}' must be an integer between 1 and {maximum}.")
    return int(raw)


def _filter_param(name: str) -> str | None:
    raw = request.args.get(name)
    if raw is None or not raw.strip():
        return None
    if len(raw) > current_app.config["MAX_FILTER_LENGTH"]:
        raise APIError(400, f"invalid_{name}", f"Query parameter '{name}' is too long.")
    return raw.strip()


@api.get("/requests")
def get_request():
    """With `link`: one request (Day 9). Without: a page of recommendable
    requests for browsing (Day 10)."""
    if "link" in request.args:
        link = _link_param()
        try:
            return jsonify(_service().get_request(link))
        except UnknownRequest:
            raise APIError(404, "unknown_request", "No request with that link.")
    page = _int_param("page", 1, 1_000_000)
    per_page = _int_param("per_page", current_app.config["REQUESTS_PER_PAGE"], current_app.config["MAX_REQUESTS_PER_PAGE"])
    return jsonify(_service().list_requests(page, per_page, _filter_param("q"), _filter_param("category"), _filter_param("skill")))


@api.get("/workers/<worker_id>")
def get_worker(worker_id: str):
    worker_id = _worker_id(worker_id)
    try:
        return jsonify(_service().worker_profile(worker_id))
    except UnknownWorker:
        raise APIError(404, "unknown_worker", "No worker with that id.")


@api.get("/workers/<worker_id>/graph")
def get_worker_graph(worker_id: str):
    worker_id = _worker_id(worker_id)
    service = _service()
    if not service.has_worker(worker_id):
        raise APIError(404, "unknown_worker", "No worker with that id.")
    reader = current_app.extensions["hirewise_worker_graph_reader"]
    try:
        return jsonify(reader(worker_id))
    except Neo4jUnavailable:
        raise APIError(503, "neo4j_unavailable", "The knowledge graph database is not reachable.")


@api.get("/recommendations")
def get_recommendations():
    link = _link_param()
    k = _k_param()
    try:
        return jsonify(_service().recommend(link, k, current_app.config["KG_EVIDENCE_PATHS"]))
    except UnknownRequest:
        raise APIError(404, "unknown_request", "No request with that link.")
    except RequestNotRecommendable as exc:
        raise APIError(409, "request_not_recommendable", str(exc))
