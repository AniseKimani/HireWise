"""Initials avatars for ACCOUNTS (pre-deployment Milestone 4). Pure functions; nothing is stored.

- `initials(display_name)`: the first letter of the first word and of the last word, upper-cased (one letter for a
  single word). Unicode-aware: the name is NFC-normalised, words are split on whitespace, and each word contributes
  its first LETTER (str.isalpha), so leading punctuation, digits and emoji are skipped and non-Latin scripts work.
  Returns "" when the name has no letters at all; the template then shows the existing person icon.
- `palette_index(user_id, size)`: a stable 1-based palette slot from SHA-256 of the user ID (Python's hash() is
  randomised per process, so it is not used). The colours themselves are CSS custom properties
  (--avatar-1 ... --avatar-N in style.css), applied through one class per slot, so no inline style is needed and the
  strict CSP holds.

Catalogue workers (shown by worker ID, with no name) keep the generic icon; this module is never used for them.
"""
from __future__ import annotations

import hashlib
import unicodedata


def _first_letter(word: str) -> str:
    for ch in word:
        if ch.isalpha():
            return ch.upper()[:1]  # e.g. "ß".upper() == "SS": keep one character
    return ""


def initials(display_name: str | None) -> str:
    words = [w for w in unicodedata.normalize("NFC", display_name or "").split() if any(ch.isalpha() for ch in w)]
    if not words:
        return ""
    if len(words) == 1:
        return _first_letter(words[0])
    return _first_letter(words[0]) + _first_letter(words[-1])


def palette_index(user_id: int, size: int) -> int:
    if size < 1:
        raise ValueError("the avatar palette needs at least one colour")
    digest = hashlib.sha256(str(int(user_id)).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % size + 1
