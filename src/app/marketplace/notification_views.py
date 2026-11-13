"""In-app notification routes (pre-deployment Milestone 3).

  GET  /notifications                  your notifications, newest first, paginated (viewing marks nothing read)
  POST /notifications/<id>/read        mark one of YOUR notifications read (CSRF-protected)
  POST /notifications/read-all         mark all of your notifications read (CSRF-protected)
  GET  /notifications/unread-count     JSON {"unread": n} for the header bell; 401 JSON when logged out;
                                       Cache-Control: no-store

Every query is limited to the logged-in user's own rows. Another user's notification and a missing one both give
404, so the app never confirms that someone else's notification exists. No route takes a user ID from the request.
Login is required, not a verified email: reading and marking notifications changes no marketplace data.
"""
from __future__ import annotations

from flask import Blueprint, abort, current_app, flash, g, jsonify, redirect, render_template, request, url_for

from src.app.marketplace.security import login_required, safe_next_url

notification_bp = Blueprint("notifications", __name__)


def _repo():
    repo = current_app.extensions.get("hirewise_marketplace")
    if repo is None:
        abort(503)
    return repo


def _no_store(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@notification_bp.get("/notifications")
@login_required
def list_page():
    repo = _repo()
    per_page = int(current_app.config["NOTIFICATIONS_PER_PAGE"])
    page = request.args.get("page", "1")
    page = int(page) if page.isdigit() and int(page) >= 1 else 1
    total = repo.count_notifications(g.user["user_id"])
    pages = max(1, -(-total // per_page))
    page = min(page, pages)
    items = repo.list_notifications(g.user["user_id"], limit=per_page, offset=(page - 1) * per_page)
    unread = repo.count_notifications(g.user["user_id"], unread_only=True)
    return _no_store(current_app.make_response(render_template(
        "notifications/list.html", items=items, page=page, pages=pages, total=total, unread_total=unread)))


@notification_bp.post("/notifications/<int:notification_id>/read")
@login_required
def mark_read(notification_id: int):
    if not _repo().mark_notification_read(g.user["user_id"], notification_id):
        abort(404)  # not yours, or does not exist: indistinguishable
    return redirect(safe_next_url(request.form.get("return_to")) or url_for("notifications.list_page"))


@notification_bp.post("/notifications/read-all")
@login_required
def mark_all_read():
    changed = _repo().mark_all_notifications_read(g.user["user_id"])
    flash("All notifications are marked as read." if changed else "You have no unread notifications.")
    return redirect(safe_next_url(request.form.get("return_to")) or url_for("notifications.list_page"))


@notification_bp.get("/notifications/unread-count")
def unread_count():
    if g.get("user") is None:  # JSON for the polling script, never a redirect to the login page
        return _no_store(jsonify({"error": "login_required"})), 401
    repo = current_app.extensions.get("hirewise_marketplace")
    if repo is None:
        return _no_store(jsonify({"error": "unavailable"})), 503
    try:
        count = repo.count_notifications(g.user["user_id"], unread_only=True)
    except Exception:
        return _no_store(jsonify({"error": "unavailable"})), 503
    return _no_store(jsonify({"unread": count}))
