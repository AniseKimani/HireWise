"""Account security routes (pre-deployment Milestone 1): email verification and password recovery.

Email verification
  GET  /register/check-email            after registration (the same page whether or not the address was new)
  GET  /verify-email                    status + resend (logged-in users)
  POST /verify-email/resend             new link for the logged-in, unverified account (rate-limited)
  GET  /verify-email/confirm/<token>    emailed link: stores the token in the session, redirects (consumes nothing)
  GET  /verify-email/confirm            "Confirm your email" page for the stored token
  POST /verify-email/confirm            consumes the token (CSRF-protected), marks the account verified

Password recovery
  GET  /forgot-password                 request form
  POST /forgot-password                 always the same response; a reset link only goes to an existing account
  GET  /forgot-password/sent            the generic confirmation
  GET  /reset-password/<token>          emailed link: stores the token in the session, redirects
  GET  /reset-password                  new-password form (or why the link no longer works)
  POST /reset-password                  consumes the token, sets the new password, ends every existing session

Emailed GET links never change anything, so link scanners and previews cannot use up a token; the token is
consumed only by the user's own CSRF-protected POST. Moving it from the URL into the (signed) session keeps it
out of the address bar, browser history and Referer. Raw tokens and email addresses are never logged.
"""
from __future__ import annotations

import logging

from flask import Blueprint, current_app, flash, g, redirect, render_template, request, session, url_for

from src.app.marketplace import auth_tokens
from src.app.marketplace.auth_tokens import RESET_PASSWORD, TOKEN_VALID, VERIFY_EMAIL
from src.app.marketplace.security import hash_password, login_required
from src.app.marketplace.validation import validate_email, validate_new_password

logger = logging.getLogger(__name__)
account = Blueprint("account", __name__)

VERIFY_SESSION_KEY = "pending_verify_token"
RESET_SESSION_KEY = "pending_reset_token"


def _repo():
    from src.app.marketplace.views import _repo as marketplace_repo  # one definition of "the repository"

    return marketplace_repo()


def _mailer():
    return current_app.extensions.get("hirewise_mailer")


def _limiter(name: str):
    return current_app.extensions[f"hirewise_{name}"]


def _client_address() -> str:
    return request.remote_addr or "unknown"


def _duration(delta) -> str:
    hours = int(delta.total_seconds() // 3600)
    if hours >= 1 and delta.total_seconds() % 3600 == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes} minutes"


# ---------------------------------------------------------------------------
# Outgoing messages
# ---------------------------------------------------------------------------
def send_verification(user_id: int, email: str, *, background: bool) -> bool | None:
    """Issue a verification token (retiring older ones) and email the link.
    Returns True/False for an immediate send, None when queued in the background, False without a mailer."""
    mailer = _mailer()
    if mailer is None:
        logger.warning("verification email not sent: email delivery is not configured (user_id=%s)", user_id)
        return False
    ttl = current_app.config["EMAIL_VERIFICATION_TTL"]
    raw = auth_tokens.issue(_repo(), user_id, VERIFY_EMAIL, ttl)
    link = mailer.link(url_for("account.confirm_email_link", token=raw))
    body = (
        "Hello,\n\n"
        "Please confirm that this email address belongs to your new HireWise account by opening the link below:\n\n"
        f"{link}\n\n"
        f"The link expires in {_duration(ttl)} and works once. Until your address is verified you can log in and "
        "browse HireWise, but you cannot post requests, book workers, manage bookings or leave reviews.\n\n"
        "If you did not create a HireWise account, you can ignore this email.\n\n"
        "-- HireWise\n")
    message = mailer.build(email, "Verify your HireWise email address", body)
    if background:
        mailer.send_later(message, "verify_email")
        return None
    return mailer.send(message, "verify_email")


def _send_reset(user_id: int, email: str) -> None:
    mailer = _mailer()
    if mailer is None:
        logger.warning("password reset email not sent: email delivery is not configured (user_id=%s)", user_id)
        return
    ttl = current_app.config["PASSWORD_RESET_TTL"]
    raw = auth_tokens.issue(_repo(), user_id, RESET_PASSWORD, ttl)
    body = (
        "Hello,\n\n"
        "We received a request to reset the password of the HireWise account that uses this email address. "
        "To choose a new password, open the link below:\n\n"
        f"{mailer.link(url_for('account.reset_password_link', token=raw))}\n\n"
        f"The link expires in {_duration(ttl)} and works once. Changing your password signs you out on every device.\n\n"
        "If you did not ask to reset your password, you can ignore this email; your password will not change.\n\n"
        "-- HireWise\n")
    mailer.send_later(mailer.build(email, "Reset your HireWise password", body), "reset_password")


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------
@account.get("/register/check-email")
def check_email():
    return render_template("account/check_email.html",
                           ttl=_duration(current_app.config["EMAIL_VERIFICATION_TTL"]))


@account.get("/verify-email")
@login_required
def verify_email():
    return render_template("account/verify_email.html", verified=g.user["email_verified"],
                           ttl=_duration(current_app.config["EMAIL_VERIFICATION_TTL"]))


@account.post("/verify-email/resend")
@login_required
def resend_verification():
    if g.user["email_verified"]:
        flash("Your email address is already verified.", "success")
        return redirect(url_for("account.verify_email"))
    limiter, key = _limiter("verify_resend_limit"), (str(g.user["user_id"]), "resend")
    if limiter.is_blocked(*key):
        flash("You have asked for several emails recently. Please wait a few minutes before asking again.", "warning")
        return redirect(url_for("account.verify_email"))
    limiter.record_failure(*key)
    if send_verification(g.user["user_id"], g.user["email"], background=False):
        flash(f"We've sent a new verification link to {g.user['email']}. Any earlier link no longer works.", "success")
    else:
        flash("We couldn't send the email just now. Please try again in a few minutes.", "error")
    return redirect(url_for("account.verify_email"))


@account.get("/verify-email/confirm/<token>")
def confirm_email_link(token: str):
    session[VERIFY_SESSION_KEY] = token if auth_tokens.well_formed(token) else ""
    return redirect(url_for("account.confirm_email"))


@account.get("/verify-email/confirm")
def confirm_email():
    raw = session.get(VERIFY_SESSION_KEY)
    state = auth_tokens.state(_repo(), raw, VERIFY_EMAIL) if raw is not None else "missing"
    return render_template("account/confirm_email.html", state=state, result=None)


@account.post("/verify-email/confirm")
def confirm_email_submit():
    raw = session.pop(VERIFY_SESSION_KEY, None)
    state, user_id = auth_tokens.verify_email(_repo(), raw)
    if state == TOKEN_VALID:
        logger.info("email address verified (user_id=%s)", user_id)
        if g.get("user") is not None and g.user["user_id"] == user_id:
            g.user["email_verified"] = True
    return render_template("account/confirm_email.html", state=state, result="done" if state == TOKEN_VALID else None)


# ---------------------------------------------------------------------------
# Password recovery
# ---------------------------------------------------------------------------
@account.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("account/forgot_password.html", email="", error=None)
    errors: dict = {}
    email = validate_email(request.form.get("email", ""), errors)
    if errors:  # malformed input only: says nothing about any account
        return render_template("account/forgot_password.html", email=request.form.get("email", "")[:254],
                               error=errors["email"]), 400
    address = _client_address()
    per_email, per_address = _limiter("reset_request_limit"), _limiter("reset_request_address_limit")
    allowed = not per_email.is_blocked(address, email) and not per_address.is_blocked(address, "*")
    per_email.record_failure(address, email)
    per_address.record_failure(address, "*")
    if allowed:
        credentials = _repo().get_credentials(email)
        if credentials is not None:
            _send_reset(credentials[0], email)  # delivery happens in the background: no timing difference
        else:
            auth_tokens.hash_token(auth_tokens.new_raw_token())  # comparable (tiny) work for unknown addresses
    return redirect(url_for("account.forgot_password_sent"))


@account.get("/forgot-password/sent")
def forgot_password_sent():
    return render_template("account/forgot_password_sent.html", ttl=_duration(current_app.config["PASSWORD_RESET_TTL"]))


@account.get("/reset-password/<token>")
def reset_password_link(token: str):
    session[RESET_SESSION_KEY] = token if auth_tokens.well_formed(token) else ""
    return redirect(url_for("account.reset_password"))


@account.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    raw = session.get(RESET_SESSION_KEY)
    if request.method == "GET":
        state = auth_tokens.state(_repo(), raw, RESET_PASSWORD) if raw is not None else "missing"
        return render_template("account/reset_password.html", state=state, errors={})
    limiter, address = _limiter("reset_attempt_limit"), _client_address()
    if limiter.is_blocked(address, "reset"):
        return render_template("account/reset_password.html", state="throttled", errors={}), 429
    limiter.record_failure(address, "reset")
    password, errors = validate_new_password(request.form)
    if errors:
        state = auth_tokens.state(_repo(), raw, RESET_PASSWORD) if raw is not None else "missing"
        return render_template("account/reset_password.html", state=state, errors=errors), 400
    state, user_id = auth_tokens.reset_password(_repo(), raw, hash_password(password))
    if state != TOKEN_VALID:
        session.pop(RESET_SESSION_KEY, None)
        return render_template("account/reset_password.html", state=state, errors={}), 400
    logger.info("password reset completed; existing sessions invalidated (user_id=%s)", user_id)
    session.clear()  # also signs this browser out; every other session is stale via auth_version
    flash("Your password has been changed and you have been signed out everywhere. Log in with your new password.", "success")
    return redirect(url_for("marketplace.login"))
