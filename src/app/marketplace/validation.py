"""Server-side form validation (Day 12). Pure functions: each returns
(clean_data, errors) where errors maps field -> user-facing message.
Nothing here trusts the browser; skills and categories are checked
against the controlled vocabulary (case-insensitive match to the
canonical name)."""
from __future__ import annotations

import re

EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 128
EXPERIENCE_LEVELS = ("entry", "intermediate", "expert")
PAY_TYPES = ("fixed", "hourly", "unspecified")
MAX_SKILLS = 20
MAX_PAY = 10_000_000


def _text(form, name: str) -> str:
    return (form.get(name) or "").strip()


def _number(raw: str, errors: dict, field: str, label: str, minimum: float = 0.0, maximum: float = MAX_PAY,
            exclusive_min: bool = True) -> float | None:
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        errors[field] = f"{label} must be a number."
        return None
    if not (value > minimum if exclusive_min else value >= minimum) or value > maximum or value != value:
        errors[field] = f"{label} is out of range."
        return None
    return value


def parse_skills(raw: str, vocabulary: dict[str, str], errors: dict, field: str = "skills") -> list[str]:
    """Comma-separated skill names -> canonical vocabulary names."""
    names = [s.strip() for s in raw.split(",") if s.strip()]
    if not names:
        errors[field] = "Enter at least one skill."
        return []
    unknown = [n for n in names if n.lower() not in vocabulary]
    if unknown:
        errors[field] = f"Unknown skill(s): {', '.join(unknown[:5])}. Use names from the skill list."
        return []
    canonical = sorted({vocabulary[n.lower()] for n in names})
    if len(canonical) > MAX_SKILLS:
        errors[field] = f"Enter at most {MAX_SKILLS} skills."
        return []
    return canonical


def validate_email(raw: str, errors: dict) -> str:
    email = raw.strip().lower()
    if len(email) > 254 or not EMAIL_PATTERN.match(email):
        errors["email"] = "Enter a valid email address."
    return email


def validate_registration(form, categories: set[str], vocabulary: dict[str, str], countries: list[str]) -> tuple[dict, dict]:
    errors: dict[str, str] = {}
    data = {"email": validate_email(_text(form, "email"), errors), "role": _text(form, "role"),
            "display_name": _text(form, "display_name"), "country": _text(form, "country") or None}
    password = form.get("password") or ""
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        errors["password"] = f"Password must be {MIN_PASSWORD_LENGTH}-{MAX_PASSWORD_LENGTH} characters."
    elif password != (form.get("confirm_password") or ""):
        errors["confirm_password"] = "Passwords do not match."
    data["password"] = password
    if data["role"] not in ("client", "worker"):
        errors["role"] = "Choose client or worker."
    if not 1 <= len(data["display_name"]) <= 80:
        errors["display_name"] = "Enter a display name (up to 80 characters)."
    if data["country"] is not None and data["country"] not in countries:
        errors["country"] = "Choose a country from the list."
    if data["role"] == "worker":
        profile = {"primary_category": _text(form, "primary_category"), "experience_level": _text(form, "experience_level"),
                   "country": data["country"]}
        if profile["primary_category"] not in categories:
            errors["primary_category"] = "Choose a category from the list."
        if profile["experience_level"] not in EXPERIENCE_LEVELS:
            errors["experience_level"] = "Choose an experience level."
        if profile["country"] is None:
            errors["country"] = "Workers must choose a country."
        profile["experience_years"] = _number(_text(form, "experience_years"), errors, "experience_years",
                                              "Years of experience", 0, 60, exclusive_min=False)
        if profile["experience_years"] is None and "experience_years" not in errors:
            errors["experience_years"] = "Enter your years of experience."
        profile["rate"] = _number(_text(form, "rate"), errors, "rate", "Rate")
        if profile["rate"] is None and "rate" not in errors:
            errors["rate"] = "Enter your rate."
        profile["skills"] = parse_skills(_text(form, "skills"), vocabulary, errors)
        data["worker_profile"] = profile
    return data, errors


def validate_login(form) -> tuple[dict, dict]:
    errors: dict[str, str] = {}
    email = _text(form, "email").lower()
    password = form.get("password") or ""
    if not email or not password or len(password) > MAX_PASSWORD_LENGTH or len(email) > 254:
        errors["form"] = "Enter your email and password."
    return {"email": email, "password": password}, errors


def validate_request(form, categories: set[str], vocabulary: dict[str, str], countries: list[str]) -> tuple[dict, dict]:
    errors: dict[str, str] = {}
    data = {"title": _text(form, "title"), "category": _text(form, "category"), "pay_type": _text(form, "pay_type"),
            "location_requirement": _text(form, "location_requirement") or None}
    if not 3 <= len(data["title"]) <= 200:
        errors["title"] = "Title must be 3-200 characters."
    if data["category"] not in categories:
        errors["category"] = "Choose a category from the list."
    data["skills"] = parse_skills(_text(form, "skills"), vocabulary, errors)
    if data["location_requirement"] is not None and data["location_requirement"] not in countries:
        errors["location_requirement"] = "Choose a country from the list, or leave it empty."
    data["budget"] = data["hourly_low"] = data["hourly_high"] = None
    if data["pay_type"] == "fixed":
        data["budget"] = _number(_text(form, "budget"), errors, "budget", "Budget")
        if data["budget"] is None and "budget" not in errors:
            errors["budget"] = "Enter a budget for a fixed-price request."
    elif data["pay_type"] == "hourly":
        data["hourly_low"] = _number(_text(form, "hourly_low"), errors, "hourly_low", "Minimum hourly rate")
        data["hourly_high"] = _number(_text(form, "hourly_high"), errors, "hourly_high", "Maximum hourly rate")
        if data["hourly_low"] is None and data["hourly_high"] is None and not {"hourly_low", "hourly_high"} & set(errors):
            errors["hourly_low"] = "Enter at least one hourly rate."
        if data["hourly_low"] is not None and data["hourly_high"] is not None and data["hourly_low"] > data["hourly_high"]:
            errors["hourly_high"] = "Maximum must not be below minimum."
    elif data["pay_type"] != "unspecified":
        errors["pay_type"] = "Choose a pay type."
    return data, errors


def validate_review(form) -> tuple[dict, dict]:
    errors: dict[str, str] = {}
    raw = _text(form, "rating")
    rating = int(raw) if raw in ("1", "2", "3", "4", "5") else None
    if rating is None:
        errors["rating"] = "Choose a rating from 1 to 5."
    text = _text(form, "review_text") or None
    if text is not None and len(text) > 2000:
        errors["review_text"] = "Reviews are limited to 2,000 characters."
    return {"rating": rating, "review_text": text}, errors
