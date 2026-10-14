"""HTML page routes outside the marketplace.

- `/`      public HireWise landing page (signed-in users go to their dashboard);
- `/demo`  the Day 10 "Research Demo": browse the imported, evaluated research
           requests and see the frozen hybrid's recommendations. It is a static
           shell; all data is fetched by the browser from the /api routes.
"""
from __future__ import annotations

from flask import Blueprint, current_app, g, redirect, render_template, url_for

pages = Blueprint("pages", __name__)


@pages.get("/")
def landing():
    if g.get("user") is not None:
        return redirect(url_for("marketplace.dashboard"))
    return render_template("landing.html")


@pages.get("/demo")
def demo():
    return render_template(
        "index.html",
        default_k=current_app.config["DEFAULT_K"],
        max_k=current_app.config["MAX_K"],
        k_choices=[k for k in (5, 10, 20) if k <= current_app.config["MAX_K"]],
    )
