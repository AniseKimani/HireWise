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


CLIENT_ONLY_FIELDS = ("client_country",)
WORKER_ONLY_FIELDS = ("worker_country", "primary_category", "experience_level", "experience_years", "rate", "skills")
WORKER_PROFILE_FIELDS = ("primary_category", "experience_level", "experience_years", "rate", "skills")


# ---- Building blocks shared by registration (Day 12 / Milestone 2) and profile editing (Milestone 4) -------------
def validate_display_name(form, errors: dict) -> str:
    display_name = _text(form, "display_name")
    if not 1 <= len(display_name) <= 80:
        errors["display_name"] = "Enter a display name (up to 80 characters)."
    return display_name


def validate_country(form, field: str, countries: list[str], errors: dict) -> str | None:
    """An optional country from the list (None when empty). Whether a country is REQUIRED is checked by the caller."""
    country = _text(form, field) or None
    if country is not None and country not in countries:
        errors[field] = "Choose a country from the list."
    return country


def validate_worker_profile(form, categories: set[str], vocabulary: dict[str, str], country: str | None,
                            errors: dict) -> dict:
    """A worker profile: category, experience, rate and skills; `country` (already validated) is required."""
    profile = {"primary_category": _text(form, "primary_category"), "experience_level": _text(form, "experience_level"),
               "country": country}
    if profile["primary_category"] not in categories:
        errors["primary_category"] = "Choose a category from the list."
    if profile["experience_level"] not in EXPERIENCE_LEVELS:
        errors["experience_level"] = "Choose an experience level."
    if profile["country"] is None:
        errors["worker_country"] = "Workers must choose a country."
    profile["experience_years"] = _number(_text(form, "experience_years"), errors, "experience_years",
                                          "Years of experience", 0, 60, exclusive_min=False)
    if profile["experience_years"] is None and "experience_years" not in errors:
        errors["experience_years"] = "Enter your years of experience."
    profile["rate"] = _number(_text(form, "rate"), errors, "rate", "Rate")
    if profile["rate"] is None and "rate" not in errors:
        errors["rate"] = "Enter your rate."
    profile["skills"] = parse_skills(_text(form, "skills"), vocabulary, errors)
    return profile


def validate_account_details(form, categories: set[str], vocabulary: dict[str, str], countries: list[str],
                             errors: dict) -> dict:
    """Role, display name, country and (for workers) the worker profile: everything an account needs apart
    from its sign-in method. Shared by password registration and the Continue-with-Google completion form
    (Milestone 2), so both apply exactly the same rules and messages.

    Role-specific fields have distinct names. The server enforces the selected role: fields belonging to the
    OTHER role are rejected, so correctness never depends on the browser hiding them."""
    role = _text(form, "role")
    other_role_fields = WORKER_ONLY_FIELDS if role == "client" else CLIENT_ONLY_FIELDS if role == "worker" else ()
    if any(_text(form, f) for f in other_role_fields):
        errors["role"] = ("Details for the other account type were submitted. Choose the account type again "
                          "and fill in only its fields.")
    country_field = "worker_country" if role == "worker" else "client_country"
    data = {"role": role}
    if data["role"] not in ("client", "worker"):
        errors["role"] = "Choose client or worker."
    data["display_name"] = validate_display_name(form, errors)
    data["country"] = validate_country(form, country_field, countries, errors)
    if data["role"] == "worker":
        data["worker_profile"] = validate_worker_profile(form, categories, vocabulary, data["country"], errors)
    return data


# ---- Profile editing (Milestone 4) ---------------------------------------------------------------------------
def validate_bio(form, max_length: int, errors: dict) -> str | None:
    bio = _text(form, "bio") or None
    if bio is not None and len(bio) > max_length:
        errors["bio"] = f"Your bio is limited to {max_length:,} characters."
    return bio


def validate_profile_edit(form, fields: frozenset[str], categories: set[str], vocabulary: dict[str, str],
                          countries: list[str], bio_max_length: int) -> tuple[dict, dict]:
    """The editable fields of one account (`fields`, from rules.profile_edit_fields), with exactly the registration
    rules and messages. The account's role never comes from the form."""
    errors: dict[str, str] = {}
    data = {"display_name": validate_display_name(form, errors)}
    if "client_country" in fields:
        data["country"] = validate_country(form, "client_country", countries, errors)
    if "worker_country" in fields:
        data["country"] = validate_country(form, "worker_country", countries, errors)
    if "worker_profile" in fields:
        data["worker_profile"] = validate_worker_profile(form, categories, vocabulary, data["country"], errors)
        data["worker_profile"]["bio"] = validate_bio(form, bio_max_length, errors)
    return data, errors


def validate_url(raw: str, max_length: int) -> tuple[str | None, str | None]:
    """(url, error): an optional absolute http(s) URL with a host. Anything else -- no scheme, javascript:, data:,
    other schemes, whitespace or control characters, or embedded credentials -- is refused."""
    from urllib.parse import urlsplit

    url = raw.strip()
    if not url:
        return None, None
    if len(url) > max_length:
        return None, f"Links are limited to {max_length} characters."
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url):
        return None, "Links cannot contain spaces or control characters."
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return None, "Enter a valid link starting with http:// or https://."
    if parts.scheme.lower() not in ("http", "https") or not host or not url.lower().startswith(("http://", "https://")):
        return None, "Enter a valid link starting with http:// or https://."
    if parts.username is not None or parts.password is not None:
        return None, "Links cannot contain a user name or password."
    return url, None


def validate_credential(form, *, kinds: tuple[str, ...], limits: dict, current_year: int) -> tuple[dict, dict]:
    """A self-reported credential. `limits`: title, organisation, url and description maximum lengths, year_min."""
    errors: dict[str, str] = {}
    data = {"kind": _text(form, "kind"), "title": _text(form, "title"),
            "organisation": _text(form, "organisation") or None, "description": _text(form, "description") or None}
    if data["kind"] not in kinds:
        errors["kind"] = "Choose certification, experience or portfolio."
    if not 1 <= len(data["title"]) <= limits["title"]:
        errors["title"] = f"Enter a title (up to {limits['title']} characters)."
    if data["organisation"] is not None and len(data["organisation"]) > limits["organisation"]:
        errors["organisation"] = f"The organisation is limited to {limits['organisation']} characters."
    if data["description"] is not None and len(data["description"]) > limits["description"]:
        errors["description"] = f"The description is limited to {limits['description']} characters."
    raw_year = _text(form, "year")
    data["year"] = None
    if raw_year:
        if raw_year.isdigit() and limits["year_min"] <= int(raw_year) <= current_year + 1:
            data["year"] = int(raw_year)
        else:
            errors["year"] = f"Enter a year from {limits['year_min']} to {current_year + 1}, or leave it empty."
    data["url"], url_error = validate_url(_text(form, "url"), limits["url"])
    if url_error:
        errors["url"] = url_error
    return data, errors


def validate_registration(form, categories: set[str], vocabulary: dict[str, str], countries: list[str]) -> tuple[dict, dict]:
    """Password registration: email and password, plus the shared account details."""
    errors: dict[str, str] = {}
    email = validate_email(_text(form, "email"), errors)
    password, password_errors = validate_new_password(form)
    errors.update(password_errors)
    data = {"email": email, "password": password}
    data.update(validate_account_details(form, categories, vocabulary, countries, errors))
    return data, errors


def validate_new_password(form) -> tuple[str, dict]:
    """Password reset: the same policy and messages as registration."""
    errors: dict[str, str] = {}
    password = form.get("password") or ""
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        errors["password"] = f"Password must be {MIN_PASSWORD_LENGTH}-{MAX_PASSWORD_LENGTH} characters."
    elif password != (form.get("confirm_password") or ""):
        errors["confirm_password"] = "Passwords do not match."
    return password, errors


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
