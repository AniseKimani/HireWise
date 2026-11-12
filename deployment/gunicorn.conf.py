"""Gunicorn settings for the HireWise container (docs/deployment.md).

    gunicorn --config deployment/gunicorn.conf.py "src.app:create_app()"

- One process: each process holds its own copy of the frozen recommender (~200 MiB), and the in-process login /
  rate limiters and the mail sender are per process. Four threads serve concurrent requests.
- No preload: the app (database engine, mail sender, OAuth client, recommender) is created inside the worker, after
  the fork, so nothing fork-sensitive is ever shared between processes. If the app cannot load (e.g. a missing
  FLASK_SECRET_KEY or mail configuration), Gunicorn stops with "Worker failed to boot"/"App failed to load" and the
  app's message -- it does not restart in a loop. The worker's model build (~4 s) counts against `timeout`.
- Bound to 0.0.0.0 on the platform's PORT; no port is assumed.
- No access log: request lines can carry single-use tokens (/verify-email/confirm/<token>,
  /reset-password/<token>) and the Google callback's authorization code (/auth/google/callback?code=...).
  Errors still go to stderr.
"""
import os

_port = os.environ.get("PORT", "").strip()
if not _port.isdigit() or not 0 < int(_port) < 65536:
    raise SystemExit("PORT must be set to the TCP port the platform routes to this container (e.g. PORT=10000)")

bind = [f"0.0.0.0:{int(_port)}"]
workers = 1
worker_class = "gthread"
threads = 4
timeout = 60
graceful_timeout = 30
preload_app = False
worker_tmp_dir = "/dev/shm"  # heartbeat file in memory, not on the container's (possibly slow) disk
accesslog = None
errorlog = "-"
loglevel = "info"
