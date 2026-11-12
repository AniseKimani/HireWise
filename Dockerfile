# HireWise production image (docs/deployment.md).
#
# Before building, create and verify the runtime-data package (git-ignored, not in the repository):
#     python scripts/prepare_deployment_data.py --verify
#
# Build:  docker build -t hirewise .
# Run:    docker run --env-file <production env file> -e PORT=10000 -p 10000:10000 hirewise
# All configuration (FLASK_SECRET_KEY, POSTGRES_*, MAIL_*/SMTP_*, APP_BASE_URL, GOOGLE_*, NEO4J_*, ...) comes from the
# environment at run time; nothing secret is built into the image.

# Debian bookworm (glibc 2.36), CPython 3.13.16 -- the interpreter the project is tested with. The digest pins the
# exact image; refresh it in both stages to pick up base-image security updates.

# ---- Stage 1: assemble and check the application files (discarded; only /app is copied out) ------------------
FROM python:3.13.16-slim-bookworm@sha256:73aa5faee130c60da030e105ec971ccf57473fbd22ce73bd6fc9ba99a40b838d AS app-files

WORKDIR /app

# The application and the tracked configuration/results the frozen recommender reads at start-up.
COPY src/ src/
COPY config/data_config.yaml config/cf_config.yaml config/kg_config.yaml config/hybrid_config.yaml config/
COPY results/day4/selected_cbf_config.json results/day4/
COPY results/day5/selected_cf_config.json results/day5/
COPY results/day7/selected_kg_config.json results/day7/graph_snapshot.json results/day7/
COPY results/day8/selected_hybrid_config.json results/day8/day8_manifest.json results/day8/

# The 14 verified runtime data files, at their repository-relative paths; checked again against the package
# manifest so a modified or incomplete package fails the build.
COPY deployment/runtime-data/data/ data/
COPY deployment/runtime-data/runtime-data-manifest.json /tmp/runtime-data-manifest.json
RUN python -c "import hashlib, json, pathlib; \
m = json.loads(pathlib.Path('/tmp/runtime-data-manifest.json').read_text()); \
files = sorted(str(p) for p in pathlib.Path('data').rglob('*') if p.is_file()); \
assert files == sorted(e['path'] for e in m['files']) and len(files) == 14, 'runtime data does not match its manifest'; \
bad = [e['path'] for e in m['files'] if hashlib.sha256(pathlib.Path(e['path']).read_bytes()).hexdigest() != e['sha256']]; \
assert not bad, f'checksum mismatch: {bad}'; print('runtime data verified:', len(files), 'files')"

COPY deployment/gunicorn.conf.py deployment/

# COPY keeps the build host's file modes (a restrictive umask could leave files unreadable). Normalise them:
# directories 0755 (traversable), files readable by everyone, writable only by root -- read-only for the app user.
RUN chmod -R u=rwX,go=rX /app

# ---- Stage 2: the runtime image -----------------------------------------------------------------------------------
FROM python:3.13.16-slim-bookworm@sha256:73aa5faee130c60da030e105ec971ccf57473fbd22ce73bd6fc9ba99a40b838d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HIREWISE_ENV=production

# The operating-system time-zone database (pandas and psycopg use it on Linux).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system hirewise \
    && useradd --system --gid hirewise --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin hirewise

WORKDIR /app

# Production dependencies only: every package pinned and hash-checked (scripts/lock_production_requirements.py).
COPY requirements-prod.lock ./
RUN pip install --no-deps --require-hashes --only-binary=:all: -r requirements-prod.lock

# Root-owned (COPY --from sets uid/gid 0), modes as normalised above.
COPY --from=app-files /app/ /app/

USER hirewise

# The existing health endpoint, unchanged: 200 ok/degraded; 503 when the recommender or PostgreSQL is unavailable.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/api/health', timeout=8)"

CMD ["gunicorn", "--config", "deployment/gunicorn.conf.py", "src.app:create_app()"]
