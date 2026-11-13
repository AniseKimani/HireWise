"""Marketplace HTML routes (Day 12): register, login/logout, dashboards,
client request creation with runtime recommendations, bookings, reviews.

Milestone 3: every booking, booking transition and review stores its
notification in the same transaction (src/app/marketplace/notify.py), and its
email is queued only after that commit. In demo mode, the demo operator acts
for unclaimed catalogue workers (rules.check_booking_actor).

Every state change is a POST protected by the CSRF hook; every ownership
and role rule is enforced here on the server (src/app/marketplace/rules.py),
never by hidden fields or JavaScript. Templates are rendered with Jinja
autoescaping and use no inline script or style.
"""
from __future__ import annotations

import re

from flask import Blueprint, abort, current_app, flash, g, redirect, render_template, request, session, url_for

from src.app.marketplace import notifications as notification_rules
from src.app.marketplace import notify, rules
from src.app.marketplace.account import send_verification
from src.app.marketplace.google_auth import LINK_SESSION_KEY, complete_pending_link, take_pending_link
from src.app.marketplace.runtime import RuntimeRequest
from src.app.marketplace.security import (
    hash_password, login_required, login_user, logout_user, role_required, safe_next_url, verified_required, verify_password,
)
from src.app.marketplace.validation import validate_login, validate_registration, validate_request, validate_review
from src.db.marketplace import ActiveBookingExists, DuplicateEmail, DuplicateReview, StaleBookingState

marketplace = Blueprint("marketplace", __name__)
RUNTIME_K = 10


def _repo():
    repo = current_app.extensions.get("hirewise_marketplace")
    if repo is None:
        abort(503)
    return repo


def _runtime():
    runtime = current_app.extensions.get("hirewise_runtime_recommender")
    if runtime is None:
        abort(503)
    return runtime


def _reference():
    """Categories, case-insensitive skill vocabulary, countries (imported reference data)."""
    repo = _repo()
    return set(repo.categories()), {s.lower(): s for s in repo.skills()}, _runtime().service.data.known_countries


def _form_context():
    categories, vocabulary, countries = _reference()
    return {"categories": sorted(categories), "skills": sorted(vocabulary.values()), "countries": countries}


def _catalogue_profile(worker_id: str) -> dict:
    """Display profile of an imported (catalogue) worker."""
    p = _runtime().service.worker_profile(worker_id)
    return {"worker_id": p["worker_id"], "primary_category": p["primary_category"],
            "secondary_categories": p["secondary_categories"], "skills": p["declared_skills"],
            "experience_level": p["experience_level"], "experience_years": p["experience_years"], "rate": p["rate"],
            "country": p["country"], "training_review_count": p["training_review_count"],
            "training_mean_rating": p["training_mean_rating"]}


def _worker_profile(user: dict) -> tuple[dict | None, str]:
    """(display profile, source) for a worker account. "imported": linked to
    an experimental catalogue profile (recommendable). "runtime": newly
    registered, not part of the frozen recommender's catalogue."""
    if user.get("worker_id"):
        return _catalogue_profile(user["worker_id"]), "imported"
    p = _repo().get_runtime_worker_profile(user["user_id"])
    if p is None:
        return None, "runtime"
    return {"worker_id": None, "primary_category": p["primary_category"], "secondary_categories": [], "skills": p["skills"],
            "experience_level": p["experience_level"], "experience_years": p["experience_years"], "rate": p["rate"],
            "country": p["country"], "bio": p.get("bio"), "training_review_count": None,
            "training_mean_rating": None}, "runtime"


def _own_request(request_id: int) -> dict:
    req = _repo().get_request(request_id)
    if req is None or req["client_user_id"] != g.user["user_id"]:
        abort(404)  # never reveal other clients' requests
    return req


def _runtime_request(req: dict) -> RuntimeRequest:
    return RuntimeRequest(request_id=req["request_id"], client_user_id=req["client_user_id"], category=req["category"],
                          skills=tuple(req["skills"]), price_tier=req["price_tier"],
                          location_requirement=req["location_requirement"], imported_client_id=g.user.get("client_id"))


# ---------------------------------------------------------------------------
# Registration / login / logout
# ---------------------------------------------------------------------------
@marketplace.route("/register", methods=["GET", "POST"])
def register():
    if g.user is not None:
        return redirect(url_for("marketplace.dashboard"))
    context = _form_context()
    if request.method == "GET":
        role = request.args.get("role")
        return render_template("marketplace/register.html", form={"role": role} if role in ("client", "worker") else {},
                               errors={}, **context)
    categories, vocabulary, countries = _reference()
    data, errors = validate_registration(request.form, categories, vocabulary, countries)
    if not errors:
        # Milestone 1: the account starts unverified, and the response is the same whether or not the address
        # already had an account (no account enumeration). Only a genuinely new account gets an email (its
        # verification link); an address that already has an account gets NO email and its account is left
        # unchanged. No duplicate account is ever created; the user logs in afterwards.
        password_hash = hash_password(data["password"])
        try:
            user_id = _repo().create_user(data["email"], password_hash, data["role"], data["display_name"], data["country"],
                                          worker_profile=data.get("worker_profile"), verified=False)
        except DuplicateEmail:
            pass  # existing address: nothing is sent and nothing changes
        else:
            send_verification(user_id, data["email"], background=True)
        return redirect(url_for("account.check_email"))
    form = {k: v for k, v in request.form.items() if "password" not in k}
    return render_template("marketplace/register.html", form=form, errors=errors, **context), 400


@marketplace.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("marketplace.dashboard"))
    next_url = safe_next_url(request.values.get("next"))
    if request.method == "GET":
        # A Google sign-in waiting to be linked pre-fills its (Google-verified) address.
        pending = session.get(LINK_SESSION_KEY)
        email = pending.get("email", "") if isinstance(pending, dict) else ""
        return render_template("marketplace/login.html", email=email, error=None, next_url=next_url)
    data, errors = validate_login(request.form)
    limiter = current_app.extensions["hirewise_login_limiter"]
    address = request.remote_addr or "unknown"
    if limiter.is_blocked(address, data["email"]):
        return render_template("marketplace/login.html", email=data["email"], next_url=next_url,
                               error="Too many failed attempts. Please wait a few minutes and try again."), 429
    credentials = None if errors else _repo().get_credentials(data["email"])
    if errors or not verify_password(credentials[1] if credentials else None, data["password"]):
        limiter.record_failure(address, data["email"])
        return render_template("marketplace/login.html", email=data["email"], next_url=next_url,
                               error="Invalid email or password."), 401
    limiter.reset(address, data["email"])
    pending_link = take_pending_link()  # read before login_user clears the session
    user = _repo().get_user(credentials[0])
    login_user(credentials[0], user["auth_version"])
    complete_pending_link(pending_link, credentials[0], user["email"])  # never blocks this login
    return redirect(next_url or url_for("marketplace.dashboard"))


@marketplace.post("/logout")
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for("pages.landing"))


# ---------------------------------------------------------------------------
# Dashboards and profile
# ---------------------------------------------------------------------------
def _is_demo_operator(user: dict, operator: dict | None) -> bool:
    return operator is not None and operator["user_id"] == user["user_id"]


def _worker_bookings(repo, user: dict, operator: dict | None) -> list[dict]:
    """Bookings a worker-role account handles: its linked catalogue worker's, or -- for the demo operator -- every
    unclaimed catalogue worker's."""
    if _is_demo_operator(user, operator):
        return repo.list_unclaimed_worker_bookings()
    return repo.list_bookings(worker_id=user["worker_id"]) if user.get("worker_id") else []


@marketplace.get("/dashboard")
@login_required
def dashboard():
    repo = _repo()
    if g.user["role"] == "client":
        requests_list = repo.list_requests_for_client(g.user["user_id"])
        bookings_list = repo.list_bookings(client_user_id=g.user["user_id"])
        live = {b["request_id"] for b in bookings_list if b["status"] in ("pending", "accepted")}
        latest = requests_list[0] if requests_list else None
        latest_recs = (_runtime().recommend(_runtime_request(latest), 3)["recommendations"] if latest is not None else [])
        return render_template("marketplace/client_dashboard.html", requests_list=requests_list,
                               bookings_list=bookings_list, status_counts=_status_counts(bookings_list),
                               requests_in_progress=live, latest_request=latest, latest_recs=latest_recs, rules=rules)
    operator = notify.demo_operator(repo)
    if _is_demo_operator(g.user, operator):
        bookings_list = repo.list_unclaimed_worker_bookings()
        return render_template("marketplace/worker_dashboard.html", profile=None, profile_source="demo_operator",
                               is_demo_operator=True, bookings_list=bookings_list,
                               status_counts=_status_counts(bookings_list),
                               reviews=[b for b in bookings_list if b["review_rating"] is not None], rules=rules)
    profile, source = _worker_profile(g.user)
    bookings_list = _worker_bookings(repo, g.user, operator)
    reviews = [b for b in bookings_list if b["review_rating"] is not None]
    return render_template("marketplace/worker_dashboard.html", profile=profile, profile_source=source,
                           is_demo_operator=False, bookings_list=bookings_list,
                           status_counts=_status_counts(bookings_list), reviews=reviews, rules=rules)


def _status_counts(bookings_list: list[dict]) -> dict[str, int]:
    counts = {status: 0 for status in ("pending", "accepted", "completed", "declined", "cancelled")}
    for b in bookings_list:
        counts[b["status"]] += 1
    return counts


def _app_reviews_for(worker_id: str | None) -> list[dict]:
    if not worker_id:
        return []
    return [b for b in _repo().list_bookings(worker_id=worker_id) if b["review_rating"] is not None]


@marketplace.get("/profile")
@role_required("worker")
def profile():
    from src.app.marketplace.profile_views import KIND_LABELS, READ_ONLY_NOTE

    worker_profile, source = _worker_profile(g.user)
    credentials = _repo().list_credentials(g.user["user_id"]) if source == "runtime" and worker_profile else []
    return render_template("marketplace/profile.html", profile=worker_profile, profile_source=source,
                           reviews=_app_reviews_for(g.user.get("worker_id")), credentials=credentials,
                           credential_kinds=KIND_LABELS, read_only_note=READ_ONLY_NOTE,
                           credentials_per_kind=current_app.config["CREDENTIALS_PER_KIND"])


@marketplace.get("/workers/<worker_id>")
@role_required("client")
def worker_detail(worker_id: str):
    """Client-facing profile of a catalogue worker ("View Profile")."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", worker_id) or not _runtime().service.has_worker(worker_id):
        abort(404)
    back = None
    request_id = request.args.get("request", "")
    if request_id.isdigit():
        req = _repo().get_request(int(request_id))
        if req is not None and req["client_user_id"] == g.user["user_id"]:
            back = url_for("marketplace.request_detail", request_id=req["request_id"])
    return render_template("marketplace/worker_detail.html", profile=_catalogue_profile(worker_id),
                           reviews=_app_reviews_for(worker_id), back_url=back, booking=None, review=None,
                           simulated_note=_simulated_note(worker_id))


def _simulated_note(worker_id: str) -> str | None:
    """The honesty note for a catalogue worker with no linked account (None when the worker is claimed)."""
    repo = _repo()
    if repo.claimed_worker_ids([worker_id]):
        return None
    return notification_rules.honesty_note(notify.demo_active(repo))


@marketplace.get("/bookings/<int:booking_id>/review")
@role_required("client")
def review_page(booking_id: int):
    """Worker profile + rating/review form for one of the client's completed bookings."""
    repo = _repo()
    booking = repo.get_booking(booking_id)
    if booking is None or booking["client_user_id"] != g.user["user_id"]:
        abort(404)
    return render_template("marketplace/worker_detail.html", profile=_catalogue_profile(booking["worker_id"]),
                           reviews=_app_reviews_for(booking["worker_id"]), back_url=url_for("marketplace.bookings_page"),
                           booking=booking, review=repo.get_review(booking_id),
                           can_review=rules.can_review(booking, g.user, repo.get_review(booking_id)),
                           simulated_note=_simulated_note(booking["worker_id"]))


# ---------------------------------------------------------------------------
# Client requests and runtime recommendations
# ---------------------------------------------------------------------------
@marketplace.get("/client/requests")
@role_required("client")
def my_requests():
    return render_template("marketplace/requests.html", requests_list=_repo().list_requests_for_client(g.user["user_id"]))


@marketplace.route("/client/requests/new", methods=["GET", "POST"])
@role_required("client")
@verified_required
def new_request():
    context = _form_context()
    if request.method == "GET":
        return render_template("marketplace/request_new.html", form={}, errors={}, **context)
    categories, vocabulary, countries = _reference()
    data, errors = validate_request(request.form, categories, vocabulary, countries)
    if errors:
        return render_template("marketplace/request_new.html", form=request.form, errors=errors, **context), 400
    tier = _runtime().price_tiers.tier(data["category"], data["pay_type"], data["budget"], data["hourly_low"], data["hourly_high"])
    request_id = _repo().create_request(g.user["user_id"], data["title"], data["category"], data["skills"], data["pay_type"],
                                        data["budget"], data["hourly_low"], data["hourly_high"], tier,
                                        data["location_requirement"])
    flash("Request created. Recommended workers are listed below.")
    return redirect(url_for("marketplace.request_detail", request_id=request_id))


@marketplace.get("/client/requests/<int:request_id>")
@role_required("client")
def request_detail(request_id: int):
    req = _own_request(request_id)
    repo = _repo()
    request_bookings = repo.list_bookings(request_id=request_id)
    active = next((b for b in request_bookings if b["status"] in ("pending", "accepted", "completed")), None)
    result = _runtime().recommend(_runtime_request(req), RUNTIME_K)
    shown = {r["worker"]["worker_id"] for r in result["recommendations"]} | {b["worker_id"] for b in request_bookings}
    claimed = repo.claimed_worker_ids(shown)
    simulated = shown - claimed
    bookings_note = (notification_rules.honesty_note(notify.demo_active(repo))
                     if any(b["worker_id"] in simulated for b in request_bookings) else None)
    return render_template("marketplace/request_detail.html", req=req, result=result, request_bookings=request_bookings,
                           active_booking=active, rules=rules, simulated_workers=simulated,
                           simulated_label=notification_rules.SIMULATED_LABEL, bookings_note=bookings_note)


@marketplace.post("/client/requests/<int:request_id>/book")
@role_required("client")
@verified_required
def book_worker(request_id: int):
    req = _own_request(request_id)
    worker_id = (request.form.get("worker_id") or "").strip()
    # MVP rule, enforced here on the server: only workers in this request's
    # CURRENT recommendation set (the deterministic Top-RUNTIME_K shown on the
    # request page, recomputed now) can be booked -- not any worker in the
    # wider eligible pool. A future search/browse feature could offer booking
    # non-recommended workers through a separate, explicit workflow.
    recommended = {r["worker"]["worker_id"] for r in _runtime().recommend(_runtime_request(req), RUNTIME_K)["recommendations"]}
    if worker_id not in recommended:
        abort(400, description="worker not in this request's recommendations")
    repo = _repo()
    kind = notification_rules.BOOKING_REQUESTED
    recipient, payload = notify.prepare(kind, repo, booking={"worker_id": worker_id, "client_user_id": g.user["user_id"]},
                                        request_title=req["title"], actor_user_id=g.user["user_id"],
                                        operator=notify.demo_operator(repo))
    try:
        booking_id = repo.create_booking(request_id, g.user["user_id"], worker_id, notification=payload)
    except ActiveBookingExists:
        flash("This request already has an active booking. Cancel it before booking someone else.")
    else:
        flash(f"Booking request sent to worker {worker_id}.")
        notify.email_after_commit(repo, kind, recipient, booking_id=booking_id, worker_id=worker_id)
    return redirect(url_for("marketplace.request_detail", request_id=request_id))


# ---------------------------------------------------------------------------
# Bookings and reviews
# ---------------------------------------------------------------------------
@marketplace.get("/bookings")
@login_required
def bookings_page():
    repo = _repo()
    is_operator = False
    if g.user["role"] == "client":
        items = repo.list_bookings(client_user_id=g.user["user_id"])
    else:
        operator = notify.demo_operator(repo)
        is_operator = _is_demo_operator(g.user, operator)
        items = _worker_bookings(repo, g.user, operator)
    return render_template("marketplace/bookings.html", bookings_list=items, rules=rules, is_demo_operator=is_operator)


@marketplace.post("/bookings/<int:booking_id>/<action>")
@login_required
@verified_required
def booking_action(booking_id: int, action: str):
    if action not in rules.ACTIONS:
        abort(404)
    repo = _repo()
    booking = repo.get_booking(booking_id)
    if booking is None:
        abort(404)
    operator = notify.demo_operator(repo)
    is_operator = _is_demo_operator(g.user, operator)
    # Only the operator's permission depends on whether the booked worker is claimed (one lookup, only then).
    worker_claimed = bool(repo.claimed_worker_ids([booking["worker_id"]])) if is_operator else True
    kind = notification_rules.EVENT_FOR_ACTION[action]
    recipient = None
    try:
        role = rules.check_booking_actor(booking, g.user, is_demo_operator=is_operator, worker_claimed=worker_claimed)
        new_status = rules.next_status(booking["status"], action, role)
        request_row = repo.get_request(booking["request_id"])
        recipient, payload = notify.prepare(kind, repo, booking=booking,
                                            request_title=request_row["title"] if request_row else None,
                                            actor_user_id=g.user["user_id"], operator=operator)
        repo.transition_booking(booking_id, booking["status"], new_status, notification=payload)
    except rules.NotAuthorized:
        abort(404)  # do not confirm that someone else's booking exists
    except rules.InvalidTransition:
        flash("That action is not available for this booking.")
    except StaleBookingState:
        flash("This booking was updated in the meantime. Please review its current status.")
    except ActiveBookingExists:
        flash("Another booking for this request is already active.")
    else:
        flash(f"Booking {booking_id} is now {new_status}.")
        notify.email_after_commit(repo, kind, recipient, booking_id=booking_id, worker_id=booking["worker_id"])
    return redirect(safe_next_url(request.form.get("return_to")) or url_for("marketplace.bookings_page"))


@marketplace.post("/bookings/<int:booking_id>/review")
@role_required("client")
@verified_required
def review_booking(booking_id: int):
    repo = _repo()
    booking = repo.get_booking(booking_id)
    if booking is None or booking["client_user_id"] != g.user["user_id"]:
        abort(404)
    if not rules.can_review(booking, g.user, repo.get_review(booking_id)):
        flash("Only completed bookings can be reviewed, once.")
        return redirect(url_for("marketplace.bookings_page"))
    data, errors = validate_review(request.form)
    if errors:
        flash(next(iter(errors.values())))
        return redirect(url_for("marketplace.review_page", booking_id=booking_id))
    kind = notification_rules.REVIEW_RECEIVED
    request_row = repo.get_request(booking["request_id"])
    recipient, payload = notify.prepare(kind, repo, booking=booking, request_title=request_row["title"] if request_row else None,
                                        actor_user_id=g.user["user_id"], operator=notify.demo_operator(repo),
                                        rating=data["rating"])
    try:
        repo.create_review(booking_id, data["rating"], data["review_text"], notification=payload)
    except DuplicateReview:
        flash("This booking has already been reviewed.")
    else:
        flash("Thank you. Your review has been saved.")
        notify.email_after_commit(repo, kind, recipient, booking_id=booking_id, worker_id=booking["worker_id"])
    return redirect(url_for("marketplace.bookings_page"))
