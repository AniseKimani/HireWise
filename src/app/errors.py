"""Consistent error responses (Day 9; HTML pages since Day 12).

Under /api every error body has the same JSON shape:

    {"error": {"code": "<machine code>", "message": "<safe text>"}}

Elsewhere (the marketplace pages) the same errors render a plain HTML page
with a user-facing message.

Stack traces, file paths and connection details are logged server-side
only, never returned to clients.
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src.db.repositories import DatabaseUnavailable

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def error_response(status: int, code: str, message: str):
    response = jsonify({"error": {"code": code, "message": message}})
    response.status_code = status
    return response


HTTP_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed"}
HTML_MESSAGES = {
    400: ("Request not accepted", "The form could not be processed. Please go back, reload the page and try again."),
    403: ("Not allowed", "Your account cannot perform this action."),
    404: ("Not found", "The page or item you asked for does not exist."),
    405: ("Not allowed", "This action is not available here."),
    503: ("Temporarily unavailable", "Application data is temporarily unavailable. Please try again shortly."),
    500: ("Something went wrong", "An unexpected error occurred. Please try again."),
}
CSRF_MESSAGE = ("Session expired", "Your form session expired or was invalid. Please reload the page and try again.")


def _wants_html() -> bool:
    return not request.path.startswith("/api/")


def html_error(status: int, override: tuple[str, str] | None = None):
    heading, message = override or HTML_MESSAGES.get(status, HTML_MESSAGES[500])
    return render_template("error.html", status=status, heading=heading, message=message), status


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(exc: APIError):
        return error_response(exc.status, exc.code, exc.message)

    @app.errorhandler(DatabaseUnavailable)
    def handle_database_unavailable(exc: DatabaseUnavailable):
        logger.warning("application database unavailable: %s", type(exc.__cause__).__name__ if exc.__cause__ else "unknown")
        if _wants_html():
            return html_error(503)
        return error_response(503, "database_unavailable", "Application data is temporarily unavailable.")

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        status = exc.code or 500
        if _wants_html():
            return html_error(status, CSRF_MESSAGE if exc.description == "csrf" else None)
        return error_response(status, HTTP_CODES.get(status, "http_error"), exc.name)

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        logger.exception("unhandled error while serving request")
        if _wants_html():
            return html_error(500)
        return error_response(500, "internal_error", "An unexpected internal error occurred.")
