"""Flask application factory (Day 9).

    create_app()                                  # config from HIREWISE_ENV (default: production)
    create_app("development")
    create_app("testing", service=svc, neo4j_probe=fake_probe, postgres_probe=fake_pg_probe)
    create_app({"MAX_K": 20})                     # overrides on top of HIREWISE_ENV's config

Start-up behaviour:
- The frozen recommender stack is built once (unless injected or
  LOAD_RECOMMENDER_ON_STARTUP is False). If it cannot be built, the app
  still starts; /api/health reports `recommender.ready: false` with a
  reason code and recommendation routes answer 503.
- PostgreSQL (Day 11) holds the application data: request lookup and
  listing and worker profiles. It is not required to START (the engine is
  lazy); while it is down /api/health reports "unavailable" and data
  routes answer 503 `database_unavailable`. There is no CSV fallback.
- Neo4j is NOT required to start or to recommend (recommendations use the
  CSV-derived training snapshot whose fingerprint Day 7 verified against
  Neo4j). /api/health probes it read-only; /api/workers/<id>/graph needs
  it and answers 503 when it is unreachable.
"""
from __future__ import annotations

import logging

from flask import Flask
from flask.json.provider import DefaultJSONProvider

from src.app.config import resolve_config
from src.app.errors import register_error_handlers
from src.app.pages import pages
from src.app.routes import api
from src.app.services import FrozenArtifactMismatch, RecommendationService, probe_neo4j, worker_graph_from_neo4j

logger = logging.getLogger(__name__)

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)


class StrictJSONProvider(DefaultJSONProvider):
    """Refuses NaN/Infinity instead of emitting invalid JSON."""

    def dumps(self, obj, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return super().dumps(obj, **kwargs)


def create_app(config=None, service: RecommendationService | None = None, neo4j_probe=None, worker_graph_reader=None,
               postgres_probe=None) -> Flask:
    app = Flask(__name__)
    app.json_provider_class = StrictJSONProvider
    app.json = StrictJSONProvider(app)

    if isinstance(config, dict):
        app.config.from_object(resolve_config())
        app.config.update(config)
    else:
        app.config.from_object(resolve_config(config))
    app.json.sort_keys = app.config["JSON_SORT_KEYS"]

    if service is None and app.config["LOAD_RECOMMENDER_ON_STARTUP"]:
        try:
            service = RecommendationService.build()
        except FrozenArtifactMismatch:
            logger.exception("frozen Day 8 artifacts failed their integrity check")
            app.extensions["hirewise_service_error"] = "frozen_artifact_mismatch"
        except Exception:
            logger.exception("recommender failed to load")
            app.extensions["hirewise_service_error"] = "load_failed"
    if service is not None:
        app.extensions["hirewise_service"] = service

    repository = getattr(service, "repository", None)
    if postgres_probe is None and repository is not None and app.config["POSTGRES_HEALTHCHECK_ENABLED"]:
        postgres_probe = repository.health
    if postgres_probe is not None:
        app.extensions["hirewise_postgres_probe"] = postgres_probe

    if neo4j_probe is None and app.config["NEO4J_HEALTHCHECK_ENABLED"]:
        neo4j_probe = probe_neo4j
    if neo4j_probe is not None:
        app.extensions["hirewise_neo4j_probe"] = neo4j_probe
    app.extensions["hirewise_worker_graph_reader"] = worker_graph_reader or worker_graph_from_neo4j

    register_error_handlers(app)
    app.register_blueprint(api)
    app.register_blueprint(pages)

    @app.after_request
    def security_headers(response):
        # No inline script/style anywhere, so a strict same-origin CSP holds.
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    return app
