# Milestone 3 Report: Booking Notifications, Notification Email and Demo Mode

**Objective.** Tell people when something happens to a booking they are part of, and let a live demonstration
show the whole booking flow even though most catalogue workers have no account:

- **in-app notifications** for every booking event and for reviews, with a header bell and a notifications page;
- **notification emails**, sent only after the change has been saved;
- **demo mode**: an explicitly configured demo operator account responds for catalogue workers that nobody has
  claimed, with honest labelling for clients.

The frozen recommender, its data files, the Neo4j graph, the evaluation results and the imported PostgreSQL tables
are unchanged. Recommendations are byte-identical (Section 8).

## 1. Background: who is behind a booking

A booking refers to an imported **catalogue worker** (`bookings.worker_id`, e.g. `W001674`), not to a user. An
account "plays" a catalogue worker when `users.worker_id` equals that ID; only `scripts/create_demo_account.py
--worker-id` creates such a link. Most of the 3,000 catalogue workers therefore have **no linked account
("unclaimed")**. Clients are always real accounts.

## 2. Schema version 5 (additive only)

| Column | Type | Rule |
|---|---|---|
| `notification_id` | BIGINT identity | primary key |
| `user_id` | BIGINT | NOT NULL, FK `users` — the **one** recipient |
| `booking_id` | BIGINT | nullable, FK `bookings` |
| `kind` | TEXT | CHECK in `booking_requested`, `booking_accepted`, `booking_declined`, `booking_cancelled`, `booking_completed`, `review_received` |
| `message` | TEXT | NOT NULL, CHECK `char_length BETWEEN 1 AND 500` |
| `created_at` | TIMESTAMPTZ | NOT NULL, default `now()` |
| `read_at` | TIMESTAMPTZ | nullable, CHECK `read_at IS NULL OR read_at >= created_at` |

- `UNIQUE (booking_id, kind, user_id)`: each event happens at most once per booking, so the database itself refuses
  a duplicate even if the application's compare-and-set rule were bypassed.
- Indexes: `(user_id, read_at)` for the unread count, `(user_id, created_at DESC)` for the newest-first list.
- **Migration** (`src/db/migrations.py`, step 4 → 5; `scripts/migrate_schema_v5.py` with `--test-db` and `--verify`):
  one transaction under the existing advisory lock, creates only this table, refuses unknown, inconsistent or newer
  databases, and a second run changes nothing. Paths 2 → … → 5, 3 → 4 → 5 and 4 → 5 are supported;
  `migrate_to_v3` / `migrate_to_v4` still stop at their own versions. No notification is back-filled for existing
  bookings. `scripts/init_postgres.py` creates version 5 directly and refuses older databases, pointing to the v5
  script.
- No imported research table is touched; nothing about notifications reaches the recommender.

## 3. Events and recipients

The rules are pure functions in `src/app/marketplace/notifications.py` (no I/O, unit-tested); the lookups that feed
them are in `src/app/marketplace/notify.py`.

| Event | Triggered by | Recipient |
|---|---|---|
| `booking_requested` | client books | worker side |
| `booking_accepted` | worker (or demo operator) accepts | the client |
| `booking_declined` | worker (or demo operator) declines | the client |
| `booking_cancelled` | client cancels | worker side |
| `booking_completed` | client marks complete | worker side |
| `review_received` | client reviews | worker side |

**Worker side** = the account linked to the booked `worker_id`. If there is none: the demo operator when demo mode is
effectively on (Section 5); otherwise **nobody**. The person who acted is never notified.

**Same transaction.** `create_booking`, `transition_booking` and `create_review` take the notification row and insert
it inside the same database transaction as the change. If the notification cannot be stored, the booking change is
rolled back too (accepted trade-off). To make that impossible in practice, the client-entered request title is
collapsed to single spaces and shortened to **120 characters with an ellipsis** inside the message, so a message can
never exceed the 500-character CHECK. Only the request that actually changes the status reaches the insert: invalid,
stale, repeated or unauthorised actions create nothing.

The stored message is one plain sentence, e.g. *"Worker W001674 accepted your booking for “Logo for a bakery”."*
It contains the (shortened) request title, which the worker side already sees on Service History.

## 4. Email

After the transaction commits, the existing `Mailer.send_later` queues one email:

- only to **verified** addresses, never to an address in `HIREWISE_NOTIFICATION_EMAIL_SKIP_DOMAINS`
  (default `demo.local`, the domain of the documented demo accounts);
- body: one sentence with the **booking number and worker ID only** (no request title, no names) and a link built
  from `APP_BASE_URL` (`Mailer.link`) to Service History; no secrets or tokens;
- a delivery failure (or any error while preparing the email) is logged with the event kind and the exception type
  only — never addresses, links or content — and **never rolls back or blocks the booking**;
- without a configured mailer (non-production), nothing is sent and an informational log line says so.

## 5. Demo mode for unclaimed workers

**Why.** In a live presentation the client books from the recommender's top 10, which are almost always unclaimed
catalogue workers. Without demo mode, nobody can accept such a booking. Demo mode lets one explicitly configured
account respond for them, so the full flow can be shown, while clients are told plainly what is happening.

**The demo operator** is a verified worker-role account with **no** linked catalogue profile, identified only by
`HIREWISE_DEMO_OPERATOR_EMAIL`. Create it with:

```powershell
python scripts/create_demo_account.py --demo-operator --email operator@example.com --name "HireWise Demo"
```

| Rule | Where |
|---|---|
| Acts in the **worker** role only on bookings of catalogue workers with **no linked account** | `rules.check_booking_actor(booking, user, is_demo_operator=…, worker_claimed=…)` — pure; the view passes both facts |
| Never acts on a claimed worker's booking (404, like any other foreign booking) | same |
| No client powers (cannot post requests, book, cancel, complete or review) | role is `worker`; the state machine allows only accept/decline |
| Misconfiguration (email unset, unknown, not a worker, linked to a catalogue worker, or unverified) | one warning per process naming the setting, never its value; demo mode treated as **off**; the app never crashes |

**Operator dashboard.** Lists the bookings of all unclaimed workers with each simulated worker's ID, under a visible
**"Demo account"** label (dashboard and Service History).

**Operator emails** go to the operator's address with subjects such as
*"[Demo] New booking for simulated worker W002417"*. **Daily cap:** at most
`HIREWISE_DEMO_OPERATOR_EMAIL_DAILY_CAP` (default 50) operator emails per UTC day, counted from the operator's
notifications stored that day (so the cap survives restarts). In-app notifications continue above the cap; only
the emails stop, and one warning is logged when the cap is first exceeded.

**Honesty labelling for clients** (why it is labelled: the catalogue is research data, and a client must never
believe a simulated profile is a real person who answered them):

| Where | Demo mode on | Demo mode off |
|---|---|---|
| Recommended-worker cards (unclaimed workers) | short **"Simulated profile"** label | same |
| Worker profile page and the request's bookings panel | "This is a simulated research profile. Responses are handled by the HireWise demo account." | "This is a simulated research profile, so it won't respond." |

## 6. In-app interface (strict CSP kept)

- **Header bell** for signed-in users, with the unread count rendered on the server and an accessible label
  ("Notifications, 3 unread"; "99+" above 99). It sits outside the collapsible menu, so it stays visible on phones;
  on desktop it is at the right end of the header row.
- **`GET /notifications`**: own notifications, newest first, paginated (20 per page). Opening the page marks nothing
  read; **"Mark as read"** and **"Mark all as read"** are CSRF-protected POSTs.
- **`GET /notifications/unread-count`**: login required, `{"unread": n}`, `Cache-Control: no-store`; **401 JSON**
  when logged out (never a redirect), so the script can stop. `static/js/notifications.js` polls it every
  `HIREWISE_NOTIFICATIONS_POLL_SECONDS` (default 30, minimum 5) seconds, passed through a `data-` attribute; polling
  stops while the tab is hidden and resumes (with an immediate refresh) when it becomes visible.
- **Ownership:** every query is limited to the logged-in user's rows; another user's notification and a missing one
  both return 404. No route takes a user ID from the request. Reading and marking require login only (no verified
  email), because they change no marketplace data.
- No inline script, style or event handlers; the existing fonts, colours, icons (new `bell` icon in `_icons.html`)
  and mobile layout are reused.

## 7. Settings

| Setting | Default | Purpose |
|---|---|---|
| `HIREWISE_DEMO_MODE` | `0` | Turn demo mode on |
| `HIREWISE_DEMO_OPERATOR_EMAIL` | unset | The demo operator account's address |
| `HIREWISE_DEMO_OPERATOR_EMAIL_DAILY_CAP` | `50` | Operator emails per UTC day (in-app continues) |
| `HIREWISE_NOTIFICATION_EMAIL_SKIP_DOMAINS` | `demo.local` | Comma-separated domains that never get notification email |
| `HIREWISE_NOTIFICATIONS_POLL_SECONDS` | `30` | Bell refresh interval (minimum 5) |
| `NOTIFICATIONS_PER_PAGE` (code config) | `20` | Page size of `/notifications` |

Out-of-range numbers fall back to their defaults with a warning. The testing configuration fixes all of them and
never reads them from the developer's environment. All are documented in `.env.example` (placeholders only) and
`docs/deployment.md`.

## 8. Tests and verification

- **`tests/test_notifications.py`**
  - pure rules: every event for a claimed worker, an unclaimed worker with demo mode off and with demo mode on; the
    actor never notified; operator permissions including the refusal on claimed workers and no client powers; the
    operator configuration check; email skip rules; the daily cap (`operator_email_decision`); message length for
    10,000-character titles; emails carry no title;
  - through the app, on the in-memory repository and (opt-in) on PostgreSQL: notification in the same transaction
    (a notification that cannot be stored rolls the booking/transition back); email delivered only after commit;
    mailer failure does not block the booking; full lifecycle recipients; failed/stale actions create nothing; skip
    domains; demo operator end to end; refusal on claimed workers; misconfigured demo mode warns once; the daily cap;
    bell count, JSON endpoint (401 / no-store), mark-read and mark-all-read with and without CSRF; 404 on another
    user's notification; CSP on the new and changed pages; database constraints (PostgreSQL).
- **Migration tests** (`tests/test_account_security.py`, opt-in): 4 → 5 creates only `notifications` and touches no
  row, a second run changes nothing, 2 → 5 in one run, no downgrade, the v5 script and `init_postgres.py`.
- **`scripts/verify_m3_notifications.py`**: independent verifier (does not import the notification modules): the v5
  script on a realistic v4 test database, constraints/indexes/foreign keys via `information_schema` and
  `pg_indexes`, **same-transaction proof** (each notification's `created_at` equals its booking's `created_at` /
  `updated_at` or its review's `created_at`, because PostgreSQL's `now()` is fixed per transaction), recipients,
  message length, email content, ownership, CSRF, the JSON endpoint, demo mode and the cap, a mail outage, frozen
  recommendations and Day 8 hashes, and the main database unchanged.

Results are in the stopping report for this milestone.

## 9. Running the live demo

```powershell
# once: migrate (test database first), then create the accounts
python scripts/migrate_schema_v5.py --test-db
python scripts/migrate_schema_v5.py                 # main database, after a backup
python scripts/create_demo_account.py --demo-operator --email operator@example.com --name "HireWise Demo"
python scripts/create_demo_account.py --role client --email client@demo.local --name "Demo Client"
# .env: HIREWISE_DEMO_MODE=1 and HIREWISE_DEMO_OPERATOR_EMAIL=operator@example.com; MAIL_BACKEND=console or file
```

1. Log in as the client, post a request, and book a recommended worker (cards show "Simulated profile"; the bookings
   panel shows the demo-account note).
2. Log in as the operator: the bell shows 1 unread; the dashboard (labelled "Demo account") lists the booking with the
   simulated worker's ID. Accept it. The operator's "[Demo]" email appears in the console or outbox.
3. Back as the client: the bell shows the acceptance (refreshed within 30 seconds without reloading); mark it read,
   complete the booking and leave a review — the operator is notified of both.

## 10. Limitations

- **Polling, not push.** The bell refreshes every 30 seconds while the tab is visible; there are no WebSockets or
  server-sent events. The count is always correct on page load.
- **Operator emails are a demonstration device**, not a product feature: one account answers for many simulated
  workers, and its email volume is capped per day.
- The daily cap counts operator notifications stored that day; concurrent bookings at the exact moment of crossing
  the cap could send one email more than the cap.
- Notifications are per account and in English only; there is no per-user email preference or unsubscribe yet.
- Booking emails are sent from the web process (a background thread, like Milestone 1's); there is no retry queue,
  so an email lost to an outage is not re-sent. The in-app notification is always stored.
- `scripts/verify_day12.py` predates Milestone 1 (it still expects registration to log in) and reads the main
  database's runtime tables; it needs a separate update.
