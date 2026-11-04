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
import os
import re
import secrets

from flask import Flask, g
from flask.json.provider import DefaultJSONProvider
from werkzeug.middleware.proxy_fix import ProxyFix

from src.app.config import resolve_config
from src.app.errors import register_error_handlers
from src.app.marketplace.account import account
from src.app.marketplace.google_auth import GoogleConfigError, configure_google, google_auth, google_enabled
from src.app.marketplace.mailer import MailConfigError, build_mailer, resolve_settings
from src.app.marketplace.runtime import RuntimeRecommender
from src.app.marketplace.security import LoginRateLimiter, csrf_token, load_current_user, protect_against_csrf
from src.app.marketplace.views import marketplace
from src.app.pages import pages
from src.app.routes import api
from src.app.services import FrozenArtifactMismatch, RecommendationService, probe_neo4j, worker_graph_from_neo4j

logger = logging.getLogger(__name__)

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)


class MissingSecretKey(RuntimeError):
    """Production configuration started without FLASK_SECRET_KEY."""


class MissingMailConfig(MailConfigError):
    """Production marketplace started without a usable email configuration."""


class InvalidGoogleConfig(GoogleConfigError):
    """Continue with Google is partly or wrongly configured (exactly one credential set, or no valid base URL)."""


# Email links carry single-use tokens in their path (/verify-email/confirm/<token>, /reset-password/<token>), and
# Google's redirect carries the authorization code and state in the callback's query string. The development server
# logs request lines, so those values are redacted from its log records.
_TOKEN_IN_PATH = re.compile(r"(/(?:verify-email/confirm|reset-password)/)[A-Za-z0-9_-]+")
_OAUTH_CALLBACK_QUERY = re.compile(r"(/auth/google/callback)\?[^\s\"]*")


class RedactTokenPaths(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _TOKEN_IN_PATH.search(message) or _OAUTH_CALLBACK_QUERY.search(message):
            message = _TOKEN_IN_PATH.sub(r"\1[redacted]", message)
            record.msg, record.args = _OAUTH_CALLBACK_QUERY.sub(r"\1?[redacted]", message), ()
        return True


def _install_token_redaction() -> None:
    werkzeug_logger = logging.getLogger("werkzeug")
    if not any(isinstance(f, RedactTokenPaths) for f in werkzeug_logger.filters):
        werkzeug_logger.addFilter(RedactTokenPaths())


def _environment_name(app: Flask) -> str:
    return "testing" if app.testing else "development" if app.debug else "production"


def _configure_mailer(app: Flask, marketplace_expected: bool) -> None:
    """Email delivery (src/app/marketplace/mailer.py). In production, an active marketplace needs a valid
    configuration: start-up stops here, before any expensive loading, if it is missing or invalid."""
    try:
        mailer = build_mailer(resolve_settings(app.config), environment=_environment_name(app), required=marketplace_expected)
    except MailConfigError as exc:
        raise MissingMailConfig(
            f"Email is not configured for production: {exc}. Set the MAIL_* / SMTP_* / APP_BASE_URL variables "
            "(see .env.example); MAIL_BACKEND=file or console is accepted for local, non-public runs.") from None
    if mailer is not None:
        app.extensions["hirewise_mailer"] = mailer


def _configure_secret_key(app: Flask) -> None:
    """Session signing key from FLASK_SECRET_KEY (or an explicit SECRET_KEY
    config override). Never hard-coded, never logged. Production refuses to
    start without one; development/testing may use a temporary random key,
    in which case sessions simply end on restart."""
    key = app.config.get("SECRET_KEY") or os.environ.get("FLASK_SECRET_KEY")
    if key:
        app.config["SECRET_KEY"] = key
        return
    if app.config["REQUIRE_SECRET_KEY"]:
        raise MissingSecretKey(
            "FLASK_SECRET_KEY is not set. The production configuration requires a session signing key: add "
            "FLASK_SECRET_KEY to your environment or .env (see .env.example), or run with HIREWISE_ENV=development."
        )
    app.config["SECRET_KEY"] = secrets.token_hex(32)
    app.extensions["hirewise_ephemeral_secret_key"] = True
    if not app.testing:
        logger.warning("FLASK_SECRET_KEY is not set; using a temporary key for this %s run (logins end on restart)",
                       "development" if app.debug else "non-production")


class StrictJSONProvider(DefaultJSONProvider):
    """Refuses NaN/Infinity instead of emitting invalid JSON."""

    def dumps(self, obj, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return super().dumps(obj, **kwargs)


def create_app(config=None, service: RecommendationService | None = None, neo4j_probe=None, worker_graph_reader=None,
               postgres_probe=None, marketplace_repository=None, runtime_recommender=None) -> Flask:
    app = Flask(__name__)
    app.json_provider_class = StrictJSONProvider
    app.json = StrictJSONProvider(app)

    if isinstance(config, dict):
        app.config.from_object(resolve_config())
        app.config.update(config)
    else:
        app.config.from_object(resolve_config(config))
    app.json.sort_keys = app.config["JSON_SORT_KEYS"]
    _configure_secret_key(app)  # before any expensive loading: fail fast in production
    # The marketplace is built below when a database-backed repository is available (an injected repository,
    # an injected service whose repository has an engine, or the service built at start-up).
    marketplace_expected = marketplace_repository is not None or (app.config["LOAD_MARKETPLACE"] and (
        getattr(getattr(service, "repository", None), "engine", None) is not None
        if service is not None else app.config["LOAD_RECOMMENDER_ON_STARTUP"]))
    _configure_mailer(app, marketplace_expected)  # likewise fails fast in production
    try:
        configure_google(app)  # optional; a partial configuration stops start-up in every environment
    except GoogleConfigError as exc:
        raise InvalidGoogleConfig(str(exc)) from None
    _install_token_redaction()
    proxies = int(app.config.get("TRUSTED_PROXY_COUNT") or 0)
    if proxies > 0:  # trust X-Forwarded-For/-Proto from exactly this many proxies (client address for rate limits)
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=proxies, x_proto=proxies)

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

    # ---- Marketplace (Day 12) -------------------------------------------------
    if service is not None and app.config["LOAD_MARKETPLACE"]:
        if runtime_recommender is None:
            try:
                runtime_recommender = RuntimeRecommender.build(service)
            except Exception:
                logger.exception("runtime recommender failed to load")
        if marketplace_repository is None and getattr(repository, "engine", None) is not None:
            from src.db.marketplace import PostgresMarketplaceRepository

            marketplace_repository = PostgresMarketplaceRepository(repository.engine)
    if marketplace_repository is not None:
        app.extensions["hirewise_marketplace"] = marketplace_repository
    if runtime_recommender is not None:
        app.extensions["hirewise_runtime_recommender"] = runtime_recommender
    app.extensions["hirewise_login_limiter"] = LoginRateLimiter(app.config["LOGIN_MAX_FAILURES"],
                                                                app.config["LOGIN_FAILURE_WINDOW_SECONDS"])
    for name in ("VERIFY_RESEND_LIMIT", "RESET_REQUEST_LIMIT", "RESET_REQUEST_ADDRESS_LIMIT", "RESET_ATTEMPT_LIMIT",
                 "GOOGLE_CALLBACK_LIMIT"):
        app.extensions[f"hirewise_{name.lower()}"] = LoginRateLimiter(*app.config[name])
    app.before_request(protect_against_csrf)
    app.before_request(load_current_user)

    app.jinja_env.globals["csrf_token"] = csrf_token  # a global, so imported macros can use it too

    @app.context_processor
    def template_context():
        return {"current_user": g.get("user"), "google_enabled": google_enabled()}

    register_error_handlers(app)
    app.register_blueprint(api)
    app.register_blueprint(pages)
    app.register_blueprint(marketplace)
    app.register_blueprint(account)
    app.register_blueprint(google_auth)

    @app.after_request
    def security_headers(response):
        # No inline script/style anywhere, so a strict same-origin CSP holds.
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    return app
