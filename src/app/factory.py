"""Flask application factory (Day 9).

    create_app()                                  # config from HIREWISE_ENV (default: production)
    create_app("development")
    create_app("testing", service=svc, neo4j_probe=fake_probe)
    create_app({"MAX_K": 20})                     # overrides on top of HIREWISE_ENV's config

Start-up behaviour:
- The frozen recommender stack is built once (unless injected or
  LOAD_RECOMMENDER_ON_STARTUP is False). If it cannot be built, the app
  still starts; /api/health reports `recommender.ready: false` with a
  reason code and recommendation routes answer 503.
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
from src.app.routes import api
from src.app.services import FrozenArtifactMismatch, RecommendationService, probe_neo4j, worker_graph_from_neo4j

logger = logging.getLogger(__name__)


class StrictJSONProvider(DefaultJSONProvider):
    """Refuses NaN/Infinity instead of emitting invalid JSON."""

    def dumps(self, obj, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return super().dumps(obj, **kwargs)


def create_app(config=None, service: RecommendationService | None = None, neo4j_probe=None, worker_graph_reader=None) -> Flask:
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

    if neo4j_probe is None and app.config["NEO4J_HEALTHCHECK_ENABLED"]:
        neo4j_probe = probe_neo4j
    if neo4j_probe is not None:
        app.extensions["hirewise_neo4j_probe"] = neo4j_probe
    app.extensions["hirewise_worker_graph_reader"] = worker_graph_reader or worker_graph_from_neo4j

    register_error_handlers(app)
    app.register_blueprint(api)
    return app
