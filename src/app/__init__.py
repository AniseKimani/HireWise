"""HireWise Flask backend (Day 9). Run locally with:

    flask --app "src.app:create_app('development')" run
"""
from src.app.factory import create_app

__all__ = ["create_app"]
