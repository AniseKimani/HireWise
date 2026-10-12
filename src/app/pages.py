"""HTML page routes (Day 10). The page is a static shell; all data is
fetched by the browser from the /api routes, so no recommendation or
request data is rendered server-side into HTML."""
from __future__ import annotations

from flask import Blueprint, current_app, render_template

pages = Blueprint("pages", __name__)


@pages.get("/")
def index():
    return render_template(
        "index.html",
        default_k=current_app.config["DEFAULT_K"],
        max_k=current_app.config["MAX_K"],
        k_choices=[k for k in (5, 10, 20) if k <= current_app.config["MAX_K"]],
    )
