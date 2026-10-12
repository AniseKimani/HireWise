"""Consistent JSON error responses (Day 9).

Every error body has the same shape:

    {"error": {"code": "<machine code>", "message": "<safe text>"}}

Stack traces, file paths and connection details are logged server-side
only, never returned to clients.
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

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


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(exc: APIError):
        return error_response(exc.status, exc.code, exc.message)

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        status = exc.code or 500
        return error_response(status, HTTP_CODES.get(status, "http_error"), exc.name)

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        logger.exception("unhandled error while serving request")
        return error_response(500, "internal_error", "An unexpected internal error occurred.")
