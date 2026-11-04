"""Continue with Google (pre-deployment Milestone 2): OAuth 2.0 authorization code flow + OpenID Connect, via Authlib.

  GET  /auth/google/start       begins the flow (?next=<same-site path>, ?role=client|worker from the register page)
  GET  /auth/google/callback    Google redirects back here; the code is exchanged and the ID token validated
  GET  /auth/google/complete    a new Google user chooses a role and fills in the account details
  POST /auth/google/complete    creates the account (atomically) and logs in

Protocol protections: scopes `openid email profile`; `state` (CSRF) and `nonce` (ID-token replay), both stored in the
signed session by Authlib and checked on return; PKCE with S256; the ID token's signature is verified against
Google's published keys and its claims are validated -- issuer (accounts.google.com, either form), audience (our
client ID), expiry and nonce -- and re-checked here before use. Only a Google-verified email address is accepted.
The account is keyed by Google's stable subject identifier (`sub`), never by the email address.

Account linking:
- a linked identity logs straight in to its account;
- an unlinked identity whose verified email matches an existing VERIFIED account is linked automatically (both
  sides have proven control of the address) and the user is logged in to that account, unchanged otherwise;
- an unlinked identity whose verified email matches an existing UNVERIFIED account is linked only after the owner
  logs in with that account's password (the pending link waits in the signed session for
  PENDING_GOOGLE_SIGNUP_TTL); the account is then marked verified. A link that cannot be completed never prevents
  the otherwise valid password login;
- an account already linked to a DIFFERENT Google identity is never relinked;
- an identity with no matching account becomes a pending sign-up (signed session, PENDING_GOOGLE_SIGNUP_TTL): the
  account is created only when the completion form is submitted, in one transaction with its identity link and
  a verified email -- an incomplete account never exists. Its password is an unusable random hash; the owner can
  set a real one with "Forgot your password?".

Google sign-in is optional. With neither GOOGLE_CLIENT_ID nor GOOGLE_CLIENT_SECRET set it is disabled (the routes
answer 404 and the button is hidden); setting exactly one stops start-up (GoogleConfigError). The redirect URI
is built from APP_BASE_URL, never from the request's Host header.

Logging names only the outcome and, on failure, the exception type -- never tokens, authorization codes,
secrets, email addresses or the callback query string.
"""
from __future__ import annotations

import logging
import os
import secrets

from flask import Blueprint, abort, current_app, flash, g, redirect, render_template, request, session, url_for

from src.app.marketplace import auth_tokens
from src.app.marketplace.mailer import MailConfigError, _base_url, resolve_settings
from src.app.marketplace.security import hash_password, login_user, safe_next_url
from src.app.marketplace.validation import validate_account_details, validate_email
from src.db.marketplace import DuplicateEmail, IdentityAlreadyLinked, UserAlreadyHasIdentity

logger = logging.getLogger(__name__)
google_auth = Blueprint("google_auth", __name__)

PROVIDER = "google"
SCOPES = "openid email profile"
CALLBACK_PATH = "/auth/google/callback"
EXTENSION_KEY = "hirewise_google"
FLOW_SESSION_KEY = "google_flow"              # next URL and role hint for the flow in progress
SIGNUP_SESSION_KEY = "google_pending_signup"   # verified identity waiting for the completion form
LINK_SESSION_KEY = "google_pending_link"       # verified identity waiting for the password login
AUTHLIB_STATE_PREFIX = f"_state_{PROVIDER}_"   # where Authlib keeps state / nonce / PKCE verifier in the session
ROLES = ("client", "worker")


class GoogleConfigError(RuntimeError):
    """Google sign-in is partly or wrongly configured. The message names settings, never their values."""


# ---------------------------------------------------------------------------
# Configuration (called by the application factory)
# ---------------------------------------------------------------------------
def _setting(config, name: str, environ) -> str:
    value = config.get(name)
    return (value if value is not None else environ.get(name, "") or "").strip()


def configure_google(app, environ=os.environ) -> bool:
    """Register the Google client when both credentials are set. Returns whether Google sign-in is enabled."""
    client_id = _setting(app.config, "GOOGLE_CLIENT_ID", environ)
    client_secret = _setting(app.config, "GOOGLE_CLIENT_SECRET", environ)
    if bool(client_id) != bool(client_secret):
        raise GoogleConfigError("Google sign-in is partly configured: set both GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET, "
                                "or neither (see .env.example)")
    if not client_id:
        return False
    production = not app.testing and not app.debug
    try:
        base_url = _base_url(resolve_settings(app.config, environ)["APP_BASE_URL"], production)
    except MailConfigError as exc:
        raise GoogleConfigError(f"Google sign-in needs a valid APP_BASE_URL for its redirect URI: {exc}") from None

    from authlib.integrations.flask_client import OAuth  # imported only when Google sign-in is enabled

    oauth = OAuth(app)
    client = oauth.register(
        PROVIDER, client_id=client_id, client_secret=client_secret,
        server_metadata_url=app.config["GOOGLE_DISCOVERY_URL"],
        client_kwargs={"scope": SCOPES, "code_challenge_method": "S256",
                       "default_timeout": app.config["GOOGLE_HTTP_TIMEOUT_SECONDS"]})
    app.extensions[EXTENSION_KEY] = client
    app.config["GOOGLE_REDIRECT_URI"] = base_url + CALLBACK_PATH
    # Authlib logs the PKCE verifier and token requests at DEBUG; never let those records through.
    authlib_logger = logging.getLogger("authlib")
    if authlib_logger.getEffectiveLevel() < logging.INFO or authlib_logger.level < logging.INFO:
        authlib_logger.setLevel(logging.INFO)
    return True


def google_enabled() -> bool:
    return EXTENSION_KEY in current_app.extensions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _client():
    client = current_app.extensions.get(EXTENSION_KEY)
    if client is None:
        abort(404)  # Google sign-in is not configured
    return client


def _repo():
    from src.app.marketplace.views import _repo as marketplace_repo  # one definition of "the repository"

    return marketplace_repo()


def _client_address() -> str:
    return request.remote_addr or "unknown"


def _discard_oauth_state() -> None:
    """Remove Authlib's per-flow state entries (state, nonce, PKCE verifier) from the session."""
    for key in [k for k in session if k.startswith(AUTHLIB_STATE_PREFIX)]:
        session.pop(key, None)


def _fresh(pending) -> bool:
    if not isinstance(pending, dict) or not isinstance(pending.get("issued_at"), (int, float)):
        return False
    age = auth_tokens.utcnow().timestamp() - pending["issued_at"]
    return -60 <= age <= current_app.config["PENDING_GOOGLE_SIGNUP_TTL"].total_seconds()


def _fail(message: str, category: str = "error"):
    flash(message, category)
    return redirect(url_for("marketplace.login"))


def _claims_options(client_id: str) -> dict:
    return {"iss": {"essential": True, "values": list(current_app.config["GOOGLE_ISSUERS"])},
            "aud": {"essential": True, "value": client_id},
            "exp": {"essential": True}, "sub": {"essential": True}}


def verified_identity(claims, client_id: str, issuers, now_ts: float, leeway: int = 120) -> tuple[dict | None, bool]:
    """Re-check the validated ID-token claims and extract the identity: ({sub, email, name} | None, email_verified).
    None means the claims are unusable (wrong issuer/audience, expired, malformed)."""
    if not claims:
        return None, False
    aud = claims.get("aud")
    audiences = aud if isinstance(aud, list) else [aud]
    exp = claims.get("exp")
    sub = claims.get("sub")
    if (claims.get("iss") not in issuers or client_id not in audiences or not isinstance(exp, (int, float))
            or exp + leeway < now_ts or not isinstance(sub, str) or not 1 <= len(sub) <= 255):
        return None, False
    errors: dict = {}
    raw_email = claims.get("email")
    email = validate_email(raw_email, errors) if isinstance(raw_email, str) else ""
    if errors or not email:
        return None, False
    name = claims.get("name")
    name = " ".join(name.split())[:80] if isinstance(name, str) else ""
    email_verified = claims.get("email_verified") in (True, "true")
    return {"sub": sub, "email": email, "name": name}, email_verified


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@google_auth.get("/auth/google/start")
def start():
    client = _client()
    if g.user is not None:
        return redirect(url_for("marketplace.dashboard"))
    role = request.args.get("role")
    for key in (SIGNUP_SESSION_KEY, LINK_SESSION_KEY):
        session.pop(key, None)
    _discard_oauth_state()
    session[FLOW_SESSION_KEY] = {"next": safe_next_url(request.args.get("next")), "role": role if role in ROLES else None}
    try:
        # Authlib generates state, nonce and the PKCE verifier and keeps them in the signed session.
        return client.authorize_redirect(current_app.config["GOOGLE_REDIRECT_URI"], prompt="select_account")
    except Exception as exc:  # discovery document unreachable, ...
        logger.warning("google sign-in could not start (error=%s)", type(exc).__name__)
        session.pop(FLOW_SESSION_KEY, None)
        return _fail("Google sign-in is unavailable right now. Please try again later or use your email and password.")


@google_auth.get(CALLBACK_PATH)
def callback():
    client = _client()
    limiter, address = current_app.extensions["hirewise_google_callback_limit"], _client_address()
    if limiter.is_blocked(address, "google_callback"):
        _discard_oauth_state()
        logger.warning("google callback rate-limited")
        return _fail("Too many Google sign-in attempts. Please wait a few minutes and try again.")
    limiter.record_failure(address, "google_callback")
    flow = session.pop(FLOW_SESSION_KEY, None) or {}
    if g.user is not None:
        _discard_oauth_state()
        return redirect(url_for("marketplace.dashboard"))
    if request.args.get("error"):
        _discard_oauth_state()
        logger.info("google sign-in not completed at google")
        if request.args.get("error") == "access_denied":
            return _fail("Google sign-in was cancelled.", "warning")
        return _fail("Google sign-in did not complete. Please try again.")

    from authlib.integrations.base_client.errors import MismatchingStateError

    try:
        token = client.authorize_access_token(claims_options=_claims_options(client.client_id))
    except MismatchingStateError:
        _discard_oauth_state()
        logger.warning("google callback rejected (state missing or mismatched)")
        return _fail("Your Google sign-in expired or was not started from this browser. Please try again.")
    except Exception as exc:  # token exchange refused, network failure, invalid ID token (signature/claims/nonce)
        _discard_oauth_state()
        logger.warning("google sign-in failed (error=%s)", type(exc).__name__)
        return _fail("Google sign-in could not be completed. Please try again.")
    _discard_oauth_state()

    now = auth_tokens.utcnow()
    identity, email_verified = verified_identity(token.get("userinfo"), client.client_id,
                                                 current_app.config["GOOGLE_ISSUERS"], now.timestamp())
    if identity is None:
        logger.warning("google sign-in rejected (ID token missing or unusable)")
        return _fail("Google sign-in could not be completed. Please try again.")
    if not email_verified:
        logger.info("google sign-in rejected (email not verified by google)")
        return _fail("Your Google account's email address is not verified. Verify it with Google, or register with "
                     "your email and password.")

    repo = _repo()
    next_url = flow.get("next")
    user = repo.get_user_by_identity(PROVIDER, identity["sub"])
    if user is not None:  # a linked identity: its account, whatever the email says now
        repo.record_identity_login(PROVIDER, identity["sub"], now)
        login_user(user["user_id"], user["auth_version"])
        logger.info("google sign-in (linked account)")
        return redirect(next_url or url_for("marketplace.dashboard"))

    credentials = repo.get_credentials(identity["email"])
    if credentials is not None:  # an existing account with this address
        if repo.get_identity_for_user(credentials[0], PROVIDER) is not None:
            logger.info("google sign-in refused (account linked to another google identity)")
            return _fail("The HireWise account for this email address is linked to a different Google account. Log in "
                         "with your password, or continue with that Google account.")
        account = repo.get_user(credentials[0])
        if account is not None and account.get("email_verified_at") is not None:
            # A verified account: link automatically and log in. Role, profile and password hash are untouched.
            try:
                repo.link_identity(account["user_id"], PROVIDER, identity["sub"], identity["email"], now)
            except (IdentityAlreadyLinked, UserAlreadyHasIdentity):  # linked concurrently: change nothing
                logger.info("google sign-in refused (identity or account linked concurrently)")
                return _fail("Google sign-in could not be completed because this account or Google account is already "
                             "linked. Please try again.")
            login_user(account["user_id"], account["auth_version"])
            logger.info("google identity linked automatically to a verified account")
            flash("Your Google account is now linked to your HireWise account.", "success")
            return redirect(next_url or url_for("marketplace.dashboard"))
        # An unverified account: the password must be proven before linking.
        session[LINK_SESSION_KEY] = {"sub": identity["sub"], "email": identity["email"], "issued_at": now.timestamp()}
        logger.info("google sign-in awaiting password confirmation to link")
        flash("You already have a HireWise account with this email address. Log in with your password once to link "
              "your Google account; after that you can continue with Google.", "warning")
        return redirect(url_for("marketplace.login", next=next_url) if next_url else url_for("marketplace.login"))

    session[SIGNUP_SESSION_KEY] = {"sub": identity["sub"], "email": identity["email"], "name": identity["name"],
                                   "role": flow.get("role"), "issued_at": now.timestamp()}
    logger.info("google sign-in awaiting account completion")
    return redirect(url_for("google_auth.complete"))


def _pending_signup():
    pending = session.get(SIGNUP_SESSION_KEY)
    if not _fresh(pending) or not isinstance(pending.get("sub"), str) or not isinstance(pending.get("email"), str):
        session.pop(SIGNUP_SESSION_KEY, None)
        return None
    return pending


@google_auth.route("/auth/google/complete", methods=["GET", "POST"])
def complete():
    _client()
    if g.user is not None:
        session.pop(SIGNUP_SESSION_KEY, None)
        return redirect(url_for("marketplace.dashboard"))
    pending = _pending_signup()
    if pending is None:
        flash("Your Google sign-up expired or was not started. Please continue with Google again.", "warning")
        return redirect(url_for("marketplace.register"))
    from src.app.marketplace.views import _form_context, _reference

    context = _form_context()
    if request.method == "GET":
        form = {"display_name": pending.get("name") or ""}
        if pending.get("role") in ROLES:
            form["role"] = pending["role"]
        return render_template("account/google_complete.html", email=pending["email"], form=form, errors={}, **context)

    categories, vocabulary, countries = _reference()
    errors: dict[str, str] = {}
    data = validate_account_details(request.form, categories, vocabulary, countries, errors)  # the email is fixed
    if errors:
        form = {k: v for k, v in request.form.items() if k not in ("csrf_token", "email") and "password" not in k}
        return render_template("account/google_complete.html", email=pending["email"], form=form, errors=errors,
                               **context), 400
    unusable_password = hash_password(secrets.token_urlsafe(32))  # nobody knows it; "Forgot your password?" sets one
    now = auth_tokens.utcnow()
    try:
        user_id = _repo().create_user_with_identity(
            pending["email"], unusable_password, data["role"], data["display_name"], data["country"],
            data.get("worker_profile"), PROVIDER, pending["sub"], now)
    except DuplicateEmail:  # an account with this address was created meanwhile: link it after a password login
        session.pop(SIGNUP_SESSION_KEY, None)
        session[LINK_SESSION_KEY] = {"sub": pending["sub"], "email": pending["email"], "issued_at": now.timestamp()}
        logger.info("google sign-up found an existing account; awaiting password confirmation to link")
        flash("A HireWise account with this email address already exists. Log in with its password once to link your "
              "Google account.", "warning")
        return redirect(url_for("marketplace.login"))
    except IdentityAlreadyLinked:  # completed meanwhile (another tab)
        session.pop(SIGNUP_SESSION_KEY, None)
        return _fail("This Google account is already linked to a HireWise account. Continue with Google to log in.",
                     "warning")
    login_user(user_id, 0)  # also drops the pending sign-up from the session
    logger.info("google sign-up completed")
    flash("Welcome to HireWise! Your account is ready and your email address is verified.", "success")
    return redirect(url_for("marketplace.dashboard"))


# ---------------------------------------------------------------------------
# Linking after a password login (called by the login view)
# ---------------------------------------------------------------------------
def take_pending_link():
    """The pending link, read BEFORE login_user clears the session."""
    return session.get(LINK_SESSION_KEY)


def complete_pending_link(pending, user_id: int, account_email: str) -> None:
    """After a successful password login: link the Google identity waiting in the session, if it is still fresh
    and belongs to this account's address. Never raises -- the login itself has already succeeded."""
    if not pending:
        return
    try:
        if not _fresh(pending):
            flash("Your Google sign-in expired before you logged in, so Google was not linked. Continue with Google "
                  "again to link it.", "warning")
            return
        if pending.get("email") != account_email:  # never link across different addresses
            flash("Google was not linked: you logged in to an account with a different email address.", "warning")
            return
        _repo().link_identity(user_id, PROVIDER, pending["sub"], pending["email"], auth_tokens.utcnow())
    except (IdentityAlreadyLinked, UserAlreadyHasIdentity):
        logger.info("google link refused (identity or account already linked)")
        flash("Google could not be linked: that Google account or this HireWise account is already linked.", "warning")
    except Exception as exc:
        logger.warning("google link failed (error=%s)", type(exc).__name__)
        flash("You are logged in, but Google could not be linked right now. Continue with Google later to try again.",
              "warning")
    else:
        logger.info("google identity linked after password login")
        flash("Your Google account is now linked. Next time you can continue with Google.", "success")
