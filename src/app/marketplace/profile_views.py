"""Profile editing and self-reported credentials (pre-deployment Milestone 4).

  GET/POST /profile/edit                          edit your own profile (fields depend on the account, see below)
  GET/POST /profile/credentials/new               add a credential (self-registered workers only)
  GET/POST /profile/credentials/<id>/edit         edit one of YOUR credentials
  POST     /profile/credentials/<id>/delete       delete one of YOUR credentials

Who may edit what is rules.profile_edit_fields: clients edit their display name and location; a worker linked to an
imported catalogue profile edits ONLY the display name (any POST touching the location, the worker profile, the bio
or credentials gets 403, because those come from, or would sit beside, research data); a self-registered worker edits
everything including credentials; any other worker account (the demo operator) edits its display name and location.
Email and password are not part of this page.

Every state change is a CSRF-protected POST followed by a redirect (post/redirect/get). Validation is server-side,
with exactly the registration rules and messages, and entered values are kept on error. Another user's credential,
or a missing one, is 404. Login is enough (no verified email): nothing edited here is shown to clients, because
self-registered workers are not in the frozen recommender's catalogue. Nothing here is read by the recommender.
"""
from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, abort, current_app, flash, g, redirect, render_template, request, url_for

from src.app.marketplace import rules
from src.app.marketplace.security import login_required, role_required
from src.app.marketplace.validation import WORKER_PROFILE_FIELDS, validate_credential, validate_profile_edit
from src.db.marketplace import CredentialLimitReached
from src.db.schema import CREDENTIAL_KINDS

profile_bp = Blueprint("profile", __name__)

READ_ONLY_NOTE = "This profile comes from the research dataset and can't be edited"
# Form fields beyond the display name, by the edit permission that allows them.
FIELDS_BY_PERMISSION = {"client_country": ("client_country",), "worker_country": ("worker_country",),
                        "worker_profile": WORKER_PROFILE_FIELDS + ("bio",)}
EDITABLE_FORM_FIELDS = frozenset(f for names in FIELDS_BY_PERMISSION.values() for f in names)
KIND_LABELS = {"certification": "Certification", "experience": "Experience", "portfolio": "Portfolio"}


def _repo():
    from src.app.marketplace.views import _repo as marketplace_repo  # one definition of "the repository"

    return marketplace_repo()


def _reference():
    from src.app.marketplace.views import _reference as marketplace_reference

    return marketplace_reference()


def _scope(repo) -> tuple[frozenset[str], dict | None]:
    profile = repo.get_runtime_worker_profile(g.user["user_id"]) if g.user["role"] == "worker" else None
    return rules.profile_edit_fields(g.user, profile is not None), profile


def _allowed_form_fields(scope: frozenset[str]) -> frozenset[str]:
    return frozenset(f for permission, names in FIELDS_BY_PERMISSION.items() if permission in scope for f in names)


def _current_values(scope: frozenset[str], profile: dict | None) -> dict:
    form = {"display_name": g.user["display_name"]}
    if "client_country" in scope:
        form["client_country"] = g.user.get("country") or ""
    if "worker_country" in scope:
        form["worker_country"] = (profile or {}).get("country") or g.user.get("country") or ""
    if "worker_profile" in scope and profile is not None:
        form.update({"primary_category": profile["primary_category"], "experience_level": profile["experience_level"],
                     "experience_years": f"{profile['experience_years']:g}", "rate": f"{profile['rate']:g}",
                     "skills": ", ".join(profile["skills"]), "bio": profile.get("bio") or ""})
    return form


def _render_edit(scope, form, errors, status=200):
    categories, vocabulary, countries = _reference()
    return render_template("profile/edit.html", scope=scope, form=form, errors=errors, read_only_note=READ_ONLY_NOTE,
                           categories=sorted(categories), skills=sorted(vocabulary.values()), countries=countries,
                           bio_max=current_app.config["PROFILE_BIO_MAX_LENGTH"]), status


@profile_bp.route("/profile/edit", methods=["GET", "POST"])
@login_required
def edit():
    repo = _repo()
    scope, profile = _scope(repo)
    if request.method == "GET":
        return _render_edit(scope, _current_values(scope, profile), {})
    if (EDITABLE_FORM_FIELDS - _allowed_form_fields(scope)) & set(request.form):
        abort(403)  # e.g. a catalogue-linked account touching its location, profile or bio
    categories, vocabulary, countries = _reference()
    data, errors = validate_profile_edit(request.form, scope, categories, vocabulary, countries,
                                         current_app.config["PROFILE_BIO_MAX_LENGTH"])
    if errors:
        form = {k: v for k, v in request.form.items() if k != "csrf_token"}
        return _render_edit(scope, form, errors, 400)
    kwargs = {"country": data["country"]} if "country" in data else {}
    repo.update_profile(g.user["user_id"], data["display_name"], worker_profile=data.get("worker_profile"), **kwargs)
    flash("Your profile has been updated.")
    return redirect(url_for("marketplace.profile") if g.user["role"] == "worker" else url_for("profile.edit"))


# ---------------------------------------------------------------------------
# Self-reported credentials
# ---------------------------------------------------------------------------
def _require_credentials_permission(repo) -> None:
    scope, _ = _scope(repo)
    if "credentials" not in scope:
        abort(403)  # catalogue-linked accounts and accounts without a self-registered worker profile


def _credential_form(credential: dict | None) -> dict:
    if credential is None:
        return {"kind": request.args.get("kind") if request.args.get("kind") in CREDENTIAL_KINDS else ""}
    return {k: ("" if credential[k] is None else str(credential[k]))
            for k in ("kind", "title", "organisation", "year", "url", "description")}


def _render_credential(form, errors, credential_id=None, status=200):
    return render_template("profile/credential_form.html", form=form, errors=errors, credential_id=credential_id,
                           kinds=CREDENTIAL_KINDS, kind_labels=KIND_LABELS, limits=current_app.config["CREDENTIAL_LIMITS"],
                           max_year=datetime.now(timezone.utc).year + 1,
                           per_kind=current_app.config["CREDENTIALS_PER_KIND"]), status


def _validated_credential():
    return validate_credential(request.form, kinds=CREDENTIAL_KINDS, limits=current_app.config["CREDENTIAL_LIMITS"],
                               current_year=datetime.now(timezone.utc).year)


def _limit_message(kind: str) -> str:
    return (f"You can list at most {current_app.config['CREDENTIALS_PER_KIND']} {KIND_LABELS[kind].lower()} entries. "
            "Delete one before adding another.")


@profile_bp.route("/profile/credentials/new", methods=["GET", "POST"])
@role_required("worker")
def new_credential():
    repo = _repo()
    _require_credentials_permission(repo)
    if request.method == "GET":
        return _render_credential(_credential_form(None), {})
    data, errors = _validated_credential()
    if not errors:
        try:
            repo.add_credential(g.user["user_id"], data, current_app.config["CREDENTIALS_PER_KIND"])
        except CredentialLimitReached:
            errors["kind"] = _limit_message(data["kind"])
        else:
            flash(f"{KIND_LABELS[data['kind']]} added.")
            return redirect(url_for("marketplace.profile") + "#credentials-heading")
    return _render_credential({k: v for k, v in request.form.items() if k != "csrf_token"}, errors, status=400)


@profile_bp.route("/profile/credentials/<int:credential_id>/edit", methods=["GET", "POST"])
@role_required("worker")
def edit_credential(credential_id: int):
    repo = _repo()
    _require_credentials_permission(repo)
    credential = repo.get_credential(g.user["user_id"], credential_id)
    if credential is None:
        abort(404)  # not yours, or does not exist: indistinguishable
    if request.method == "GET":
        return _render_credential(_credential_form(credential), {}, credential_id)
    data, errors = _validated_credential()
    if not errors:
        try:
            if not repo.update_credential(g.user["user_id"], credential_id, data, current_app.config["CREDENTIALS_PER_KIND"]):
                abort(404)
        except CredentialLimitReached:
            errors["kind"] = _limit_message(data["kind"])
        else:
            flash(f"{KIND_LABELS[data['kind']]} updated.")
            return redirect(url_for("marketplace.profile") + "#credentials-heading")
    return _render_credential({k: v for k, v in request.form.items() if k != "csrf_token"}, errors, credential_id, 400)


@profile_bp.post("/profile/credentials/<int:credential_id>/delete")
@role_required("worker")
def delete_credential(credential_id: int):
    repo = _repo()
    _require_credentials_permission(repo)
    if not repo.delete_credential(g.user["user_id"], credential_id):
        abort(404)
    flash("Credential deleted.")
    return redirect(url_for("marketplace.profile") + "#credentials-heading")
