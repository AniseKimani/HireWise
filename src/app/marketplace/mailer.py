"""Outgoing email (Milestone 1): email-verification and password-reset messages.

Standard library only (`smtplib`, `email`, `ssl`). Backends, chosen by MAIL_BACKEND:

- ``smtp``    -- production: SMTP with STARTTLS (default) or implicit TLS, connection/read timeout, optional login.
- ``console`` -- development: the message is written to the process's stderr stream.
- ``file``    -- development: one .eml file per message in MAIL_OUTBOX_DIR, which must be OUTSIDE the repository
                 (default: <system temp>/hirewise-outbox), so generated mail can never be committed.
- ``memory``  -- tests: messages are kept in a list; nothing leaves the process.

Email links are absolute URLs built from APP_BASE_URL -- never from the request's Host header.

Logging never contains message bodies, links, tokens, recipient addresses, SMTP passwords or connection
details: only the message kind and, on failure, the exception type. A failed delivery is reported to the
caller (`send` returns False); it never raises into the request or alters account state.
"""
from __future__ import annotations

import logging
import os
import secrets
import smtplib
import ssl
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ("smtp", "console", "file", "memory")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class MailConfigError(RuntimeError):
    """Email delivery is not (correctly) configured. The message names settings, never their values."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class ConsoleBackend:
    name = "console"

    def __init__(self, stream=None):
        self.stream = stream

    def deliver(self, message: EmailMessage) -> None:
        stream = self.stream or sys.stderr
        stream.write("\n----- HireWise development email (console backend; not sent) -----\n")
        stream.write(message.as_string())
        stream.write("\n----- end of email -----\n")
        stream.flush()


class FileBackend:
    name = "file"

    def __init__(self, outbox_dir: Path):
        self.outbox_dir = outbox_dir

    def deliver(self, message: EmailMessage) -> None:
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        name = f"{formatdate(usegmt=True).replace(' ', '_').replace(',', '').replace(':', '')}-{secrets.token_hex(4)}.eml"
        (self.outbox_dir / name).write_bytes(bytes(message))


class MemoryBackend:
    """Test backend: keeps messages; `fail` simulates a delivery failure."""
    name = "memory"

    def __init__(self):
        self.messages: list[EmailMessage] = []
        self.fail = False

    def deliver(self, message: EmailMessage) -> None:
        if self.fail:
            raise ConnectionRefusedError("simulated delivery failure")
        self.messages.append(message)


class SMTPBackend:
    name = "smtp"

    def __init__(self, host: str, port: int, username: str | None, password: str | None, security: str, timeout: float):
        self.host, self.port, self.username, self._password = host, port, username, password
        self.security, self.timeout = security, timeout

    def __repr__(self):  # never show the password
        return f"SMTPBackend(security={self.security!r}, timeout={self.timeout})"

    def deliver(self, message: EmailMessage) -> None:
        context = ssl.create_default_context()  # certificate and hostname verification on
        if self.security == "ssl":
            server = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout, context=context)
        else:
            server = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
        with server:
            server.ehlo()
            if self.security == "starttls":
                server.starttls(context=context)
                server.ehlo()
            if self.username:
                server.login(self.username, self._password or "")
            server.send_message(message)


# ---------------------------------------------------------------------------
# Mailer
# ---------------------------------------------------------------------------
class Mailer:
    def __init__(self, backend, sender: str, base_url: str, send_async: bool = True):
        self.backend = backend
        self.sender = sender
        self.base_url = base_url.rstrip("/")
        self.send_async = send_async
        self._executor: ThreadPoolExecutor | None = None

    def link(self, path: str) -> str:
        """Absolute URL for an application path, from APP_BASE_URL (never from the request)."""
        return self.base_url + path

    def build(self, to: str, subject: str, body: str) -> EmailMessage:
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = to
        message["Subject"] = subject
        message["Date"] = formatdate(usegmt=True)
        message["Message-ID"] = make_msgid(domain="hirewise.local")
        message["Auto-Submitted"] = "auto-generated"
        message.set_content(body)  # text/plain, UTF-8
        return message

    def send(self, message: EmailMessage, kind: str) -> bool:
        """Deliver now. True only if the backend accepted the message."""
        try:
            self.backend.deliver(message)
        except Exception as exc:  # network, TLS, authentication, file system ...
            logger.warning("email delivery failed (kind=%s, backend=%s, error=%s)", kind, self.backend.name, type(exc).__name__)
            return False
        logger.info("email delivered (kind=%s, backend=%s)", kind, self.backend.name)
        return True

    def send_later(self, message: EmailMessage, kind: str) -> None:
        """Deliver in the background, so a response never waits on (or reveals timing of) delivery."""
        if not self.send_async:
            self.send(message, kind)
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hirewise-mail")
        self._executor.submit(self.send, message, kind)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _flag(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _base_url(raw: str | None, production: bool) -> str:
    if not raw:
        raise MailConfigError("APP_BASE_URL is not set (the public address used in email links, e.g. https://hirewise.example.com)")
    parts = urlsplit(raw.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment or parts.username:
        raise MailConfigError("APP_BASE_URL must be an absolute http(s) address without query, fragment or credentials")
    if production and parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
        raise MailConfigError("APP_BASE_URL must use https in production (http is accepted only for localhost)")
    return raw.strip().rstrip("/")


def _outbox_dir(raw: str | None) -> Path:
    path = Path(raw).expanduser().resolve() if raw else Path(tempfile.gettempdir()).resolve() / "hirewise-outbox"
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise MailConfigError("MAIL_OUTBOX_DIR must be outside the repository, so generated mail can never be committed")
    return path


def build_mailer(settings: dict, *, environment: str, required: bool) -> Mailer | None:
    """Build the mailer from `settings` (app config values, already resolved against the environment).

    `required`: the marketplace is active. In production a missing or invalid configuration then stops
    start-up (MailConfigError); otherwise an unconfigured mailer is simply absent (None) and email features
    report that sending is unavailable."""
    production = environment == "production"
    backend_name = (settings.get("MAIL_BACKEND") or "").strip().lower() or None
    try:
        if backend_name is None:
            raise MailConfigError("MAIL_BACKEND is not set (smtp in production; console or file for local runs)")
        if backend_name not in BACKENDS:
            raise MailConfigError(f"MAIL_BACKEND must be one of {', '.join(BACKENDS)}")
        if production and backend_name == "memory":
            raise MailConfigError("MAIL_BACKEND=memory is for automated tests only")
        base_url = _base_url(settings.get("APP_BASE_URL"), production)
        sender = (settings.get("MAIL_FROM") or "").strip()
        if backend_name == "smtp":
            if not sender or "@" not in parseaddr(sender)[1]:
                raise MailConfigError("MAIL_FROM must be a valid sender address, e.g. HireWise <no-reply@example.com>")
            host = (settings.get("SMTP_HOST") or "").strip()
            if not host:
                raise MailConfigError("SMTP_HOST is not set")
            try:
                port = int(settings.get("SMTP_PORT") or 587)
            except ValueError:
                raise MailConfigError("SMTP_PORT must be a number") from None
            raw_security = str(settings.get("SMTP_USE_TLS") if settings.get("SMTP_USE_TLS") is not None else "1").strip().lower()
            security = "ssl" if raw_security == "ssl" else "starttls" if _flag(raw_security) else "none"
            if security == "none" and production and host not in LOCAL_HOSTS:
                raise MailConfigError("SMTP_USE_TLS must be enabled in production (plain SMTP only to a local relay)")
            username, password = settings.get("SMTP_USERNAME") or None, settings.get("SMTP_PASSWORD") or None
            if bool(username) != bool(password):
                raise MailConfigError("set both SMTP_USERNAME and SMTP_PASSWORD, or neither")
            backend = SMTPBackend(host, port, username, password, security, float(settings.get("MAIL_TIMEOUT_SECONDS") or 10))
        else:
            sender = sender or "HireWise <no-reply@hirewise.local>"
            backend = (ConsoleBackend() if backend_name == "console" else FileBackend(_outbox_dir(settings.get("MAIL_OUTBOX_DIR")))
                       if backend_name == "file" else MemoryBackend())
            if production:
                logger.warning("MAIL_BACKEND=%s in the production configuration: emails are NOT sent (local runs only)",
                               backend_name)
    except MailConfigError:
        if required and production:
            raise
        if required:
            logger.warning("email is not configured; verification and password-reset emails cannot be sent")
        return None
    return Mailer(backend, sender, base_url, send_async=_flag(settings.get("MAIL_SEND_ASYNC", True)))


def resolve_settings(config, environ=os.environ) -> dict:
    """Config values win; unset (None) values fall back to the environment, then to the config's defaults."""
    keys = ("MAIL_BACKEND", "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_USE_TLS", "MAIL_FROM",
            "APP_BASE_URL", "MAIL_OUTBOX_DIR")
    settings = {k: config.get(k) if config.get(k) is not None else environ.get(k) for k in keys}
    if not settings["MAIL_BACKEND"]:
        settings["MAIL_BACKEND"] = config.get("DEFAULT_MAIL_BACKEND")
    if not settings["APP_BASE_URL"]:
        settings["APP_BASE_URL"] = config.get("DEFAULT_APP_BASE_URL")
    settings["MAIL_TIMEOUT_SECONDS"] = config.get("MAIL_TIMEOUT_SECONDS")
    settings["MAIL_SEND_ASYNC"] = config.get("MAIL_SEND_ASYNC", True)
    return settings
