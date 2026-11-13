# Milestone 4 Report: Profile Editing, Self-Reported Credentials and Initials Avatars

**Objective.** Let people keep their own HireWise details up to date, let self-registered workers describe their
background, and give every account a recognisable avatar, without touching research data:

- **edit profile**: one page, showing exactly the fields each account may change;
- **self-reported credentials**: certifications, experience and portfolio links for self-registered workers, clearly
  labelled as unverified;
- **initials avatars** for accounts, in colours chosen from a fixed, accessible palette.

The frozen recommender, its data files, the Neo4j graph, the evaluation results and the imported PostgreSQL tables
are unchanged. Recommendations are byte-identical (Section 8).

## 1. Who can edit what

The rule is one pure function, `rules.profile_edit_fields(user, has_runtime_profile)`, enforced on the server:

| Account | Can edit | Cannot edit |
|---|---|---|
| Client | display name, location (optional) | — |
| Self-registered worker (has a runtime profile) | display name, location (required), primary category, experience level and years, rate, skills, bio, credentials | — |
| Worker linked to an imported catalogue worker (`users.worker_id`) | **display name only** | location, worker profile, bio, credentials: **403** on any POST touching them |
| Worker account with neither (the demo operator) | display name, location (optional) | worker profile, credentials (403) |

- **Email and password are not on this page** and work exactly as before (Milestone 1 and 2 flows).
- A catalogue-linked profile is research data: its profile page says *"This profile comes from the research dataset
  and can't be edited"*. The display name belongs to the account, not to the dataset, so it can still be changed.
- The account's role always comes from the database, never from the form. A POST containing any field outside the
  account's permissions is refused with 403 before validation, so correctness never depends on the browser.
- **Login is enough** (no verified email): nothing edited here is visible to clients (Section 9).
- Every change is a CSRF-protected POST followed by a redirect (post/redirect/get). Errors re-render the form with
  status 400, field-level messages and the entered values kept.

## 2. Validation: exactly the registration rules

`validate_account_details` (registration and the Continue-with-Google completion form) was split into reusable parts
— `validate_display_name`, `validate_country` and `validate_worker_profile` — and recomposed **unchanged**:

- the same fields, the same checks in the same order, and the same messages;
- verified by rendering the registration form (empty, both roles, and three invalid submissions) with the previous
  templates and the new ones — identical HTML — and by comparing the previous and new validation results on six
  edge-case inputs — identical data and messages;
- the role-specific fields moved into `_client_fields.html` and `_worker_fields.html` with identical markup; the
  registration and Google forms include both, the edit page only the one it needs (so it needs no JavaScript).

The edit page reuses those parts (`validate_profile_edit`). Skills come from the controlled vocabulary with the same
limit (20) and messages. New rules:

| Field | Rule |
|---|---|
| Bio | optional, trimmed, at most `PROFILE_BIO_MAX_LENGTH` (1,000) characters |
| Credential kind | certification, experience or portfolio |
| Title | trimmed, 1–120 characters |
| Organisation | optional, trimmed, ≤ 120 characters |
| Year | optional whole number from 1950 to the current year + 1 |
| Link | optional, ≤ 500 characters, **absolute `http://` or `https://` with a host only** |
| Description | optional, trimmed, ≤ 500 characters |
| Per worker | at most `CREDENTIALS_PER_KIND` (10) entries of each kind |

**Links.** Refused: no scheme, `javascript:`, `data:`, `vbscript:`, `file:`, `ftp:`, `mailto:` and every other scheme,
scheme-relative `//host`, a missing host, embedded `user:password@`, and any whitespace or control character inside the
address. Links are rendered as ordinary text links with `rel="noopener noreferrer nofollow"`; Jinja autoescaping
applies to every value.

## 3. Schema version 6 (additive only)

- `runtime_worker_profiles.bio TEXT NULL`, `CHECK (bio IS NULL OR char_length(bio) <= 1000)`; existing rows get NULL.
- `worker_credentials`:

| Column | Type | Rule |
|---|---|---|
| `credential_id` | BIGINT identity | primary key |
| `user_id` | BIGINT | NOT NULL, FK **`runtime_worker_profiles`** |
| `kind` | TEXT | CHECK in `certification`, `experience`, `portfolio` |
| `title` | TEXT | NOT NULL, 1–120 characters |
| `organisation` | TEXT | ≤ 120 characters |
| `year` | SMALLINT | 1950–2100 (static bound; "current year + 1" is enforced in validation, because a CHECK cannot use the current date) |
| `url` | TEXT | ≤ 500 characters and `~* '^https?://'` |
| `description` | TEXT | ≤ 500 characters |
| `created_at` | TIMESTAMPTZ | NOT NULL, default `now()` |

- Index `(user_id, kind)` for the per-kind count and the grouped display.
- **The foreign key points at `runtime_worker_profiles`, not `users`**: the database itself refuses credentials for
  clients, catalogue-linked accounts and the demo operator, none of which has a runtime profile.
- **Per-kind limit under concurrency:** the repository locks the worker's profile row (`SELECT … FOR UPDATE`) before
  counting, so two simultaneous adds cannot exceed the limit.
- **Migration** (`src/db/migrations.py`, step 5 → 6; `scripts/migrate_schema_v6.py` with `--test-db` and `--verify`):
  one transaction under the existing advisory lock; adds the column only if it is missing (and refuses a different
  existing column); creates the table; a second run changes nothing; a recorded version 6 without the bio column is
  refused. Paths 2 → … → 6 are supported, and `migrate_to_v3` … `migrate_to_v5` still stop at their own versions.
  `scripts/init_postgres.py` creates version 6 directly. No imported research table is touched.

## 4. Why "self-reported"

HireWise has no way to check a certificate, an employer or a portfolio. Every credential and the bio are therefore
labelled **"Self-reported"** — on the credentials section, on each of the three groups (Certifications, Experience,
Portfolio) and on the bio — so nobody mistakes a worker's own claim for something the platform verified. This matches
the project's existing honesty labels (research-period reviews vs HireWise reviews, simulated profiles in demo mode).

## 5. Why there are no uploads

No file of any kind can be uploaded: certificates and portfolio items are text plus an optional link. Uploads would
need storage outside the read-only container, malware scanning, size and type limits, content moderation and a
privacy policy for documents that may contain personal data — none of which fits an academic prototype. A link to a
place the worker already controls gives clients (in future) the same evidence without HireWise hosting it.

## 6. Initials avatars

- **Who:** accounts only — the header account menu, the edit page, the self-registered worker's profile page and
  dashboard sidebar, and a small avatar beside "Welcome back" on both dashboards. Catalogue workers (shown by ID, with
  no name) keep the existing person icon on worker cards and profiles, and so does a catalogue-linked account's own
  profile page and dashboard sidebar (their heading is the catalogue worker ID); its header menu still shows its
  initials.
- **Letters** (`src/app/marketplace/avatars.py`): NFC-normalised; words split on whitespace; the first *letter* of
  the first and last words, upper-cased (one character; "ß" gives "S"). A single name gives one letter. Leading
  punctuation, digits and emoji are skipped, and non-Latin scripts work ("李小龙" → "李", "Ivan Петров" → "IП"). A
  name without letters falls back to the person icon.
- **Colour:** slot = SHA-256 of the user ID modulo the palette size, plus 1. This is stable across processes and
  restarts (Python's `hash()` is randomised per process). Nothing is stored.
- **Palette:** six CSS custom properties `--avatar-1` … `--avatar-6`, applied through one class per colour
  (`.avatar-initials--1` … `--6`), so no inline style is needed and the strict CSP holds. `AVATAR_PALETTE_SIZE` in
  `src/app/config.py` must match; a test checks it.

| Slot | Colour | White text contrast |
|---|---|---|
| 1 teal | `#0b6b61` | 6.4:1 |
| 2 blue | `#1e4f8f` | 8.2:1 |
| 3 plum | `#7a2e66` | 8.7:1 |
| 4 purple | `#5b3c99` | 8.2:1 |
| 5 brown | `#8a5300` | 6.3:1 |
| 6 green | `#2f6b2f` | 6.4:1 |

All exceed WCAG 2's 4.5:1 for normal text; red is left out so an avatar never reads as an error state. The avatar is
decorative (`aria-hidden`), because the name is always shown next to it.

## 7. Settings

No new environment variables. The limits are code configuration in `src/app/config.py`, like
`NOTIFICATIONS_PER_PAGE`: `PROFILE_BIO_MAX_LENGTH` (1,000), `CREDENTIALS_PER_KIND` (10), `CREDENTIAL_LIMITS` (title 120,
organisation 120, URL 500, description 500, first year 1950) and `AVATAR_PALETTE_SIZE` (6). The text lengths are also
enforced by the database (`src/db/schema.py` constants), so configuration may lower them but never raise them; a test
checks that.

## 8. Tests and verification

- **`tests/test_profiles.py`**
  - pure rules: initials (19 cases including single names, extra spaces, punctuation, decomposed accents, Cyrillic,
    Arabic, CJK, emoji, digits, empty and None); palette slots stable, in range and all used; every palette colour
    ≥ 4.5:1 for white text, with one CSS class per colour; links accepted and refused (19 refused forms); credential
    limits and trimming; configured limits never above the database's; the edit permissions for every account type;
    edit validation using the registration messages;
  - through the app, on the in-memory repository and (opt-in) on PostgreSQL: edit success and failure for each kind
    of account with entered values kept; CSRF required; login required; 403 for catalogue-linked accounts on every
    restricted field and on credentials; add, edit and delete credentials, 404 on another user's credential, the
    per-kind limit (also when changing a credential's kind); "Self-reported" labels and link `rel` attributes;
    initials avatars and the icon fallback; no inline script, style or handlers on the new pages; catalogue worker
    cards keep the icon; frozen and runtime recommendations unchanged after edits; database constraints (PostgreSQL).
- **Migration tests** (`tests/test_account_security.py`, opt-in): 5 → 6 adds only the bio column and the table and
  changes no row; a second run changes nothing; 2 → 6 in one run; an inconsistent version 6 is refused; no downgrade;
  the v6 script and `init_postgres.py`.
- **Earlier verifiers** (`verify_m1_auth.py`, `verify_m2_google.py`, `verify_m3_notifications.py`) also remove the v6
  additions when they build an older database, and upgrade to v6 before starting the application.
- **`scripts/verify_m4_profiles.py`**: independent verifier (does not import the avatar, validation, rules or profile
  modules). It recomputes initials, palette slots and contrast itself and checks every outcome with direct SQL.

Results are in the stopping report for this milestone.

## 9. Limitations

- **Credentials are currently visible only to the worker themselves.** Self-registered workers are not in the frozen
  recommender's catalogue, so clients never see their profiles, bios or credentials.
- **If self-registered workers ever become recommendable** (their profiles visible to clients), editing the bio and
  credentials should then require a verified email address, like posting requests and booking do today, and
  credentials may need moderation or a reporting route.
- Nothing is verified: a "Self-reported" credential is exactly what the worker typed.
- Links are checked for form only (absolute http/https); HireWise never fetches them, so a link may be dead or point
  anywhere. `nofollow` and `noreferrer` limit what the site lends to and leaks to the destination.
- Initials are a heuristic: names written family-name first, or with particles ("van der Berg"), give the initials of
  the first and last written words.
- A catalogue-linked account's displayed location comes from the research catalogue and cannot be changed; only its
  display name can.
