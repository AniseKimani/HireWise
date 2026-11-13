# Deployment preparation

This document covers what has to exist before a HireWise container image can be built, and the container itself
(`Dockerfile`, `.dockerignore`, `deployment/gunicorn.conf.py`). The Render configuration is added in a later step.

## Python dependencies

| File | Purpose |
|---|---|
| `requirements-prod.txt` | Runtime only: the Flask app, the frozen recommender, PostgreSQL, Neo4j, Google sign-in and the Gunicorn server. This is what the image installs. |
| `requirements-dev.txt` | `requirements-prod.txt` plus tests (`pytest`) and research/data-processing tools (matplotlib, seaborn, openpyxl, jupyter). |
| `requirements.txt` | Alias for `requirements-dev.txt`, so the README workflow (`pip install -r requirements.txt`) is unchanged. |

The direct dependencies are pinned. Gunicorn runs only on Linux/macOS, so it is first exercised in the Linux image.

## Runtime data

The recommender is frozen. At start-up it rebuilds its models from fixed data files and checks them against
recorded fingerprints. Fourteen of those files are generated research data under `data/processed/`. That
directory is excluded from Git (`.gitignore`), because it is large derived data, and because it sits next to the
generator-private truth (`data/processed/synthetic_private/`) and the raw source dataset (`data/raw/`), which
must never leave the research environment.

The deployed app needs exactly these 14 files (traced from the start-up code). They are all read once at start-up;
nothing is read per request:

| Files | Purpose |
|---|---|
| `data/processed/synthetic/workers.csv`, `requests.csv`, `interactions.csv` | Worker catalogue, requests and interaction history the models are built from |
| `data/processed/sampled_requests.csv` | Request content (title, skills, category, pay) |
| `data/processed/prepared_jobs.csv` | Price-tier reference for marketplace requests |
| `data/processed/kg/nodes_{category,skill,worker}.csv`, `rel_{has_skill,hired,interested_in,related_to,reviewed,specializes_in}.csv` | The training knowledge graph used for KG scores and explanations |

The 10 configuration/results files the recommender also reads (`config/*_config.yaml` and the selected
configurations, manifest and graph snapshot under `results/day4`–`day8`) are tracked in Git and come with the
repository.

**Never include** `data/raw/`, `data/processed/synthetic_private/`, `.env` or any other file under `data/` in a
package, image or upload.

### Creating the package

```powershell
python scripts/prepare_deployment_data.py            # verify the sources, then create deployment/runtime-data/
python scripts/prepare_deployment_data.py --force    # replace an existing package
python scripts/prepare_deployment_data.py --verify   # check an existing package; copies nothing
```

The script copies only the 14 allow-listed files into `deployment/runtime-data/`, keeping their repository-relative
paths, and adds `runtime-data-manifest.json` with their SHA-256 checksums. `deployment/runtime-data/` is
git-ignored. The script stops with a non-zero exit status and a reason, and leaves no partial package behind, if:

- a file is missing, is a symlink or junction, or resolves outside the repository;
- a file's SHA-256 differs from the checksum recorded by the pipeline that produced it
  (`data/processed/manifest.json`, `data/processed/synthetic/manifest.json`, `data/processed/kg/kg_prepare_manifest.json`);
- the KG files do not reproduce the frozen Day 7 graph fingerprint (`results/day7/graph_snapshot.json`), checked
  with the project's own `src.kg.graph_view` (the same check the application makes at start-up);
- the package already exists and `--force` was not given. With `--force`, the old package is replaced only after the new one has verified;
- in `--verify` mode: the package holds anything other than the 14 files and its manifest, contains private/raw
  data or a link, or any checksum or the KG fingerprint does not match.

### The data must match the database and the KG snapshot

The package, the PostgreSQL data loaded by `scripts/load_postgres.py`, and the frozen KG snapshot must all come from
the same pipeline run. The application refuses a request that exists in PostgreSQL but not in the packaged data
(`FrozenArtifactMismatch`). If the KG files do not reproduce the snapshot fingerprint, the recommender does not
start. Build the package from the same `data/processed/` that was used to load the production database, and run
`--verify` before every image build.

### Licence

`sampled_requests.csv` and `prepared_jobs.csv` are derived from the Upwork job-postings dataset. It has **not**
been confirmed that this data may be redistributed. Check the dataset's licence before publishing a Docker image
that contains it or uploading such an image to any registry, and keep such images private until then.

## Container image

> Status: the image has **not yet been built or run on Linux** (no Docker on the preparation machine). The
> configuration is checked statically by `tests/test_docker_config.py`; build it and run the test suite inside it
> before relying on it.

### Build and run

```powershell
python scripts/prepare_deployment_data.py --verify     # the package must exist and verify first
docker build -t hirewise .
docker run --env-file <production env file> -e PORT=10000 -p 10000:10000 hirewise
```

The image starts:

```
gunicorn --config deployment/gunicorn.conf.py "src.app:create_app()"
```

`create_app()` with no argument uses `HIREWISE_ENV`, which the image sets to `production`. Gunicorn settings
(`deployment/gunicorn.conf.py`):

- **Bind and processes:** `0.0.0.0:$PORT`. `PORT` is required and nothing is assumed. One process with 4 threads
  (`gthread`), timeout 60 s.
- **One process, not several:** each process holds its own ~200 MiB copy of the frozen recommender, and the
  login/rate limiters and mail sender are per process.
- **No preload (`preload_app = False`):** the app, including its database engine, is created inside the worker,
  after the fork, so nothing is shared between processes. If the app cannot load (for example, a missing
  `FLASK_SECRET_KEY` or mail configuration), Gunicorn stops with the app's message instead of restarting in a
  loop. The worker's model build (about 4 s locally) counts against the 60 s timeout.
- **No access log:** request lines can contain single-use verification/reset tokens and the Google callback's
  authorization code. Errors still go to stderr. Gunicorn's error message for an unhandled exception includes the
  request URL, but Flask handles application errors itself, so in practice that log line is not reached.

### What the image contains

`.dockerignore` is an allow-list (everything is excluded unless listed), and the `Dockerfile` copies explicit paths:

| Included | Excluded (never in the build context) |
|---|---|
| `src/` (application code, templates, static files) | `.env` and any `.env.*`, `.venv`, `.git` |
| the 4 `config/*_config.yaml` and 6 `results/day4`–`day8` files the recommender reads | `data/` (raw, processed, `synthetic_private`) |
| the 14 runtime data files from `deployment/runtime-data/`, re-hashed against the package manifest during the build | tests, docs, scripts, notebooks, other results and config (including the generator configs) |
| `requirements-prod.lock` (installed with `pip install --no-deps --require-hashes`), `deployment/gunicorn.conf.py` | `requirements*.txt` (no development dependencies), dumps, local databases, caches |

- **Base image:** `python:3.13.16-slim-bookworm`, pinned by digest. It is Debian, so the wheels in the lock fit
  (glibc 2.36; the lock needs at least 2.27). The operating-system time-zone database (`tzdata`) is installed
  explicitly.
- **User:** the application runs as the unprivileged user `hirewise`.
- **File permissions:** the application files are assembled in a discarded build stage. There, their modes are
  normalised (`chmod -R u=rwX,go=rX`: directories traversable, files readable, writable only by root), whatever the
  build host's umask or operating system. They are then copied into the image root-owned, so `hirewise` can read
  them but not change them.
- **Configuration:** no credentials, URLs, keys or ports are built in. All configuration is supplied as
  environment variables at run time.
- **Health check:** the image's `HEALTHCHECK` calls the existing `/api/health`, unchanged. It returns 200 when
  "ok"/"degraded" (Neo4j down) and 503 when the recommender or PostgreSQL is unavailable.

### Database schema version

The image needs schema version 6. Upgrade an existing database with `python scripts/migrate_schema_v6.py` (after a
backup; try `--test-db` first); a new database gets version 6 from `scripts/init_postgres.py`.

### Profiles and credentials (Milestone 4)

No new environment variables. The bio length, the number of credentials per kind and the avatar palette size are code
configuration in `src/app/config.py` (`docs/m4_profiles_report.md`). Nothing is uploaded, so the image still needs no
writable storage.

### Notifications and demo mode (Milestone 3)

Optional settings, all read from the environment at run time (`docs/m3_notifications_report.md`):

| Setting | Default | Notes |
|---|---|---|
| `HIREWISE_DEMO_MODE` | `0` | `1` lets the demo operator answer bookings for catalogue workers with no linked account. Clients are told these are simulated profiles |
| `HIREWISE_DEMO_OPERATOR_EMAIL` | unset | Address of the demo operator (`scripts/create_demo_account.py --demo-operator`). If missing or invalid, a warning is logged and demo mode stays off |
| `HIREWISE_DEMO_OPERATOR_EMAIL_DAILY_CAP` | `50` | Demo-operator emails per UTC day; in-app notifications continue above it |
| `HIREWISE_NOTIFICATION_EMAIL_SKIP_DOMAINS` | `demo.local` | Comma-separated domains that never receive notification email |
| `HIREWISE_NOTIFICATIONS_POLL_SECONDS` | `30` | Header bell refresh interval (minimum 5) |

Notification emails use the same `MAIL_*` / `SMTP_*` configuration as verification and reset emails. For the operator
to receive email, its address must be verified and outside the skipped domains.

### Security settings to make at deployment

- **`HIREWISE_SECURE_COOKIES=1`:** the site is served over HTTPS, so session cookies must be `Secure`.
- **`HIREWISE_TRUSTED_PROXIES`:** the number of reverse proxies in front of the app whose `X-Forwarded-For`/`-Proto`
  headers are trusted. It is **deliberately not guessed**. Too low, and every user shares the proxy's address, so
  one person's failed logins can lock everyone out. Too high, and clients can forge their address. Confirm the
  hosting platform's proxy chain first, for example from the documented behaviour or by inspecting forwarded
  headers on a test deployment, and then set it.
- **HSTS (proposed, not yet implemented):** the app sends CSP, `X-Content-Type-Options`, `Referrer-Policy` and
  `X-Frame-Options`, but no `Strict-Transport-Security`. The proposed minimal change in `src/app/factory.py`'s
  `security_headers`:
  - **When to send it:** only when the app is neither in debug nor testing mode **and** the configured
    `APP_BASE_URL` is `https`. This depends on configuration, not on proxy headers, so local HTTP development never
    receives it.
  - **Header value:** `Strict-Transport-Security: max-age=86400` at first. Raise it, for example to `31536000`,
    once HTTPS on the final domain is confirmed. Leave `includeSubDomains`/`preload` off until the domain's
    subdomains are known to be HTTPS-only.
