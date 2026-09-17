"""Metadata parser for Upwork job description footers.

Every posting ends with a platform-generated footer of the form::

    Budget: $500 | Hourly Range: $10.00-$30.00
    Posted On: February 17, 2024 09:09 UTC
    Category: Social Media Marketing
    Skills:Facebook Advertising,     Social Media Advertising, ...
    Skills:Facebook Advertising,     Social Media Advertising, ...   (repeated)
    Location Requirement: Only freelancers located in ... may apply.
    Country: United States
    click to apply

This module is read-only with respect to the raw dataset: callers pass in
raw description strings and get back structured, derived fields. Nothing
here mutates a caller's data.

Key edge cases handled (see docs/data_quality_report.md for evidence):

- Some job bodies contain the literal text "Category:" before the real
  footer. Naively searching for the first "Category:" produces corrupted,
  multi-thousand-character "categories". We anchor on the LAST
  "Posted On:" marker and parse only the text after it.
- The Skills block separates skills with a comma followed by *multiple*
  spaces (five, empirically), not a bare comma. Some legitimate skill
  labels contain an internal comma-plus-single-space, e.g.
  "Religious, Charitable & Nonprofit". Splitting on every comma corrupts
  those labels; splitting on ",  +" (comma, 2+ spaces) does not.
- The footer repeats the Skills block verbatim; only the first occurrence
  is used.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

FOOTER_ANCHOR = "Posted On:"

# Ordered stop markers used to bound a footer field's value. Order matters
# only in that every marker must be listed so a field's regex knows where
# to stop; it does not encode which marker actually follows in a given row.
FOOTER_STOPS = [
    "Skills:",
    "Location Requirement:",
    "Country:",
    "click to apply",
]

# A skill separator is a comma followed by 2+ spaces. A bare "comma space"
# is preserved as part of a skill label (e.g. "Religious, Charitable &
# Nonprofit"). Evidence: in the raw footers, ~8,000 separators use exactly
# 5 spaces and only a few dozen single-space commas appear, all of them
# inside multi-word category-style skill labels.
_SKILL_SEP_RE = re.compile(r",\s{2,}")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class ParsedJob:
    """Structured fields derived from one job's raw description.

    All fields are derived; none of them overwrite a raw source column.
    """

    footer_found: bool
    category: str | None
    skills_raw: list[str] = field(default_factory=list)
    location_requirement: str | None = None
    footer_country: str | None = None


def _decode(text: str) -> str:
    return html.unescape(text)


def extract_footer(description: str) -> str:
    """Return the text starting at the LAST 'Posted On:' marker.

    Anchoring on the last occurrence avoids false matches when a client's
    own job description happens to contain the word "Category:" (or, in
    principle, "Posted On:") before the real platform footer.
    """
    decoded = _decode(description)
    idx = decoded.rfind(FOOTER_ANCHOR)
    if idx < 0:
        return ""
    return decoded[idx:]


def _extract_field(footer: str, name: str, stops: list[str] = FOOTER_STOPS) -> str | None:
    """Extract the value following ``name:`` up to the next stop marker or
    end of string. Returns None if the field is absent."""
    stop_pattern = "|".join(re.escape(s) for s in stops)
    pattern = rf"{re.escape(name)}:\s*(.*?)(?:{stop_pattern}|$)"
    match = re.search(pattern, footer, re.S)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def parse_skills(footer: str) -> list[str]:
    """Extract the skill list from the first 'Skills:' block in the footer.

    The footer repeats the Skills block; only the first occurrence is
    used. Skills are separated by a comma followed by 2+ spaces so that
    skill labels containing an internal ", " survive intact. Whitespace is
    collapsed, entries are stripped, and within-job duplicates are removed
    while preserving first-seen order.
    """
    block = _extract_field(footer, "Skills")
    if not block:
        return []
    pieces = _SKILL_SEP_RE.split(block)
    cleaned = (_WHITESPACE_RE.sub(" ", piece).strip() for piece in pieces)
    return list(dict.fromkeys(piece for piece in cleaned if piece))


def parse_category(footer: str) -> str | None:
    return _extract_field(footer, "Category")


def parse_location_requirement(footer: str) -> str | None:
    return _extract_field(footer, "Location Requirement", stops=["Country:", "click to apply"])


def parse_footer_country(footer: str) -> str | None:
    return _extract_field(footer, "Country", stops=["click to apply"])


def clean_country(country: str | None) -> str | None:
    """Decode HTML entities in a country string without touching the raw
    value. Returns None for missing/blank input."""
    if country is None:
        return None
    if not isinstance(country, str):
        return None
    decoded = html.unescape(country).strip()
    return decoded or None


def parse_job(description: str) -> ParsedJob:
    """Parse one job's raw description into structured, derived fields.

    Does not mutate or require any other column; callers combine this with
    raw columns (budget, country, is_hourly, ...) downstream.
    """
    footer = extract_footer(description)
    if not footer:
        return ParsedJob(footer_found=False, category=None)

    return ParsedJob(
        footer_found=True,
        category=parse_category(footer),
        skills_raw=parse_skills(footer),
        location_requirement=parse_location_requirement(footer),
        footer_country=parse_footer_country(footer),
    )
